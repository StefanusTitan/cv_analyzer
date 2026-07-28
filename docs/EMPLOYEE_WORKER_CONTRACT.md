# Employee Worker Contract — `POST /cv/analyze`

This document describes the contract between the **durable `service_employees`
worker** and the **stateless `cv-analyzer`** compute service.

The analyzer is a **synchronous, stateless** compute service. It does **not**:

- download files from MinIO (the worker downloads and uploads the bytes),
- know about `candidate_id`,
- call an employee callback,
- store any database state,
- create analysis/retry/job tables,
- or accept arbitrary source URLs.

The worker is responsible for durability and persistence; this service only
analyzes uploaded bytes and returns a plain-text summary.

## Endpoint

```
POST /cv/analyze
Content-Type: multipart/form-data
```

### Multipart fields

| Field           | Required | Type   | Notes                                            |
|-----------------|----------|--------|--------------------------------------------------|
| `job_posting_id`| yes      | string | UUID of the job posting. Validated by the analyzer. |
| `files`         | yes      | files  | One or more uploaded files. At least one is required. |

The analyzer **downloads nothing**. The worker must upload the file bytes that
it fetched from MinIO as `multipart/form-data` file parts.

### Supported formats

- `application/pdf` (`.pdf`) — detected by `%PDF-` magic bytes, not extension alone.
- DOCX / `application/vnd.openxmlformats-officedocument.wordprocessingml.document`
  (`.docx`) — validated as a safe OOXML zip archive (`[Content_Types].xml` and
  `word/document.xml` present, with bounded entry count, uncompressed size, and
  compression ratio).

Any other type is rejected with `415 unsupported_format`. A `.pdf`/`.docx` whose
content does not match its extension is rejected with `422 invalid_document`.

### Size and content limits (defaults; configurable by the analyzer)

| Limit                        | Default        | Error code on violation         |
|------------------------------|----------------|---------------------------------|
| Per-file size                | 10 MiB         | `file_size_exceeded` (413)      |
| Total upload size            | 20 MiB         | `total_size_exceeded` (413)     |
| Gateway request body limit   | 25 MiB         | `request_size_exceeded` (413)   |
| Maximum number of files      | 3              | `file_count_exceeded` (400)     |
| Maximum PDF pages            | 100            | `page_limit_exceeded` (422)     |
| Max extracted characters     | 100,000        | (truncated, not an error)       |

These are upper bounds; do not send larger payloads.

## Success response

`HTTP 200`

```json
{
  "message": "CV analyzed successfully",
  "result": {
    "job_posting_id": "32a594ac-9e1b-4a9e-a3be-6e6ca87db8ff",
    "job_title": "Backend Engineer",
    "analysis": "non-empty plain-text summary the worker should persist",
    "sources": [],
    "warnings": []
  }
}
```

The worker persists **`result.analysis`** into `service_employees`. It is:

- a single plain-text string,
- stripped of internal citation markers (`[document:0]`, `[github:...]`,
  `[web:...]`, `[job_description]`, `[cv_and_resume]`),
- safe to render and store as-is.

`sources` and `warnings` are supplementary metadata and may be ignored by the
worker.

## Failure response

```json
{
  "message": "...",
  "result": null,
  "errors": ["machine_readable_code"]
}
```

`errors` is a list of machine-readable string codes. Responses never include
exception stacks, upstream bodies, credentials, or extracted CV text.

> Note: framework-level validation errors for a **missing `job_posting_id`
> field** are returned as `422` with a Pydantic-style `errors` list (list of
> objects, not strings). This is a permanent 4xx failure; the worker should
> treat any 4xx it does not recognize as permanent.

## Retryable (5xx) vs permanent (4xx)

Classify by **HTTP status code first**; `errors[0]` provides the stable code.

### Permanent — do not retry (4xx)

| Code                         | Status | Meaning                                                    |
|------------------------------|--------|------------------------------------------------------------|
| `missing_files`              | 400    | No file parts were uploaded.                               |
| `file_count_exceeded`        | 400    | More than the allowed number of files.                     |
| `invalid_job_posting_id`      | 400    | `job_posting_id` is not a UUID.                            |
| `file_size_exceeded`         | 413    | A single file exceeds the per-file limit.                  |
| `total_size_exceeded`        | 413    | Combined upload exceeds the total limit.                   |
| `request_size_exceeded`       | 413    | Request body exceeds the gateway limit.                   |
| `unsupported_format`         | 415    | File type is not PDF or DOCX.                              |
| `invalid_document`           | 422    | File matches extension but content is corrupt/invalid.    |
| `empty_document`             | 422    | No extractable text in any uploaded document.             |
| `page_limit_exceeded`         | 422    | PDF exceeds the maximum page count.                       |
| `extraction_failed`          | 422    | A document could not be extracted.                        |
| `job_posting_not_found`      | 404    | The job posting does not exist.                            |
| *(any other 4xx)*            | 4xx    | Treat as permanent.                                        |

### Retryable — retry with backoff (5xx)

| Code                              | Status | Meaning                                                |
|-----------------------------------|--------|--------------------------------------------------------|
| `job_posting_timeout`             | 504    | Job-posting service timed out.                         |
| `job_posting_unavailable`         | 502    | Job-posting service is unreachable.                   |
| `job_posting_invalid_response`     | 502    | Job-posting service returned a malformed payload.     |
| `llm_timeout`                     | 504    | LLM provider timed out.                                |
| `llm_unavailable`                  | 502    | LLM provider is temporarily unavailable / rate-limited.|
| `llm_provider_error`              | 502    | LLM provider returned an error.                        |
| `llm_empty_response`              | 502    | LLM provider returned an empty response.              |
| `llm_authentication_failed`        | 502    | LLM credentials rejected (operator action needed; 5xx).|
| `internal_server_error`            | 500    | Unhandled analyzer error.                              |
| *(any other 5xx)*                 | 5xx    | Treat as retryable.                                    |

> `llm_authentication_failed` is a 5xx by shape, but operationally it usually
> requires operator intervention rather than worker retries; retries are still
> safe (they will keep failing until the credential is fixed).

## Concurrency, cleanup, and timeouts

- The analyzer is stateless and synchronous: one request → one analysis.
- Per-request state is local; concurrent requests cannot mix content or
  results. Extraction and scraping concurrency are bounded internally.
- Uploaded file handles and the temporary extraction directory are always closed
  / cleaned up — on success, on failure, and on client cancellation.
- Outbound calls have bounded timeouts: job-posting lookup, LLM request, and
  scraper/GitHub enrichment. There is **no internal durable queue** in this
  service; durability lives in `service_employees`.

## What the worker must NOT assume

- The analyzer will not store, update, or callback into `service_employees`.
- The analyzer will not fetch URLs from the CV or from the worker.
- `result.analysis` is the only field the worker must persist; `sources` and
  `warnings` are optional supplementary metadata.