import ipaddress
import re
from collections.abc import Callable
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

_TRACKING_PARAMS = {"fbclid", "gclid", "mc_cid", "mc_eid"}
HTTP_URL_RE = re.compile(r"https?://[^\s<>\"\]\)]+", re.IGNORECASE)
BARE_URL_RE = re.compile(
    r"(?<![\w@./-])(?:www\.)?(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+"
    r"[a-z]{2,63}(?::\d{2,5})?(?:[/?#][^\s<>\"\]\)]*)?",
    re.IGNORECASE,
)


def _matches_host(host: str, domain: str) -> bool:
    return host == domain or host.endswith(f".{domain}")


def normalize_url(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip().rstrip(".,;:!?)]}>`*")
    if len(value) > 2_048:
        return None
    if "://" not in value:
        if not BARE_URL_RE.fullmatch(value):
            return None
        value = f"https://{value}"
    try:
        parsed = urlsplit(value)
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
            return None
        if parsed.username or parsed.password:
            return None
        host = parsed.hostname.lower().rstrip(".")
        port = parsed.port
        if port and not (
            (parsed.scheme.lower() == "http" and port == 80)
            or (parsed.scheme.lower() == "https" and port == 443)
        ):
            host = f"{host}:{port}"
        query = [
            (k, v)
            for k, v in parse_qsl(parsed.query, keep_blank_values=True)
            if k.lower() not in _TRACKING_PARAMS
        ]
        return urlunsplit(
            (parsed.scheme.lower(), host, parsed.path or "/", urlencode(query), "")
        )
    except ValueError:
        return None


def is_github_url(value: str) -> bool:
    try:
        host = (urlsplit(value).hostname or "").lower().rstrip(".")
        return host in {"github.com", "www.github.com", "api.github.com"}
    except ValueError:
        return False


def is_gitlab_url(value: str) -> bool:
    try:
        host = (urlsplit(value).hostname or "").lower().rstrip(".")
        return host in {"gitlab.com", "www.gitlab.com"}
    except ValueError:
        return False


def is_bitbucket_url(value: str) -> bool:
    try:
        host = (urlsplit(value).hostname or "").lower().rstrip(".")
        return host in {"bitbucket.org", "www.bitbucket.org"}
    except ValueError:
        return False


def is_huggingface_url(value: str) -> bool:
    try:
        host = (urlsplit(value).hostname or "").lower().rstrip(".")
        return host in {"huggingface.co", "www.huggingface.co"}
    except ValueError:
        return False


def classify_source(value: str) -> tuple[str, str]:
    try:
        parsed = urlsplit(value)
    except ValueError:
        return "website", "web_page"
    host = (parsed.hostname or "").lower().rstrip(".")
    parts = [part for part in parsed.path.split("/") if part]

    if host in {"github.com", "www.github.com", "api.github.com"}:
        if host == "api.github.com" and parts[:1] == ["users"]:
            return "github", "profile"
        return "github", "repository" if len(parts) >= 2 else "profile"
    if host in {"gitlab.com", "www.gitlab.com"}:
        project_parts = parts[: parts.index("-")] if "-" in parts else parts
        return "gitlab", "repository" if len(project_parts) >= 2 else "profile"
    if host in {"bitbucket.org", "www.bitbucket.org"}:
        return "bitbucket", "repository" if len(parts) >= 2 else "profile"
    if _matches_host(host, "behance.net"):
        return "behance", "project" if "gallery" in parts else "profile"
    if _matches_host(host, "figma.com"):
        return "figma", "prototype" if parts[:1] == ["proto"] else "design"
    if _matches_host(host, "canva.com"):
        return "canva", "design"
    if _matches_host(host, "notion.site") or _matches_host(host, "notion.so"):
        return "notion", "portfolio"
    if host == "docs.google.com":
        kinds = {
            "document": "document",
            "presentation": "presentation",
            "spreadsheets": "spreadsheet",
            "forms": "form",
        }
        return "google_docs", kinds.get(parts[0] if parts else "", "document")
    if host == "drive.google.com":
        return "google_drive", "shared_file"
    if host == "youtu.be":
        return "youtube", "video" if parts else "profile"
    if _matches_host(host, "youtube.com"):
        if parts[:1] in (["watch"], ["shorts"], ["embed"], ["live"]):
            return "youtube", "video"
        if parts[:1] == ["playlist"]:
            return "youtube", "collection"
        return "youtube", "profile"
    if _matches_host(host, "vimeo.com"):
        return "vimeo", "video" if parts and parts[-1].isdigit() else "profile"
    if _matches_host(host, "kaggle.com"):
        return "kaggle", "notebook" if "code" in parts else "profile"
    if host in {"huggingface.co", "www.huggingface.co"} or host.endswith(".hf.space"):
        if parts[:1] == ["datasets"]:
            return "huggingface", "dataset"
        if parts[:1] == ["spaces"] or host.endswith(".hf.space"):
            return "huggingface", "application"
        return "huggingface", "model" if len(parts) >= 2 else "profile"
    if _matches_host(host, "tableau.com"):
        return "tableau", "dashboard"
    if host == "app.powerbi.com":
        return "power_bi", "dashboard"
    if host == "orcid.org":
        return "orcid", "research_profile"
    if _matches_host(host, "credly.com"):
        return "credly", "credential"
    if host == "learn.microsoft.com" and "credentials" in parts:
        return "microsoft_learn", "credential"
    if _matches_host(host, "medium.com"):
        return "medium", "publication"
    if _matches_host(host, "substack.com"):
        return "substack", "publication"
    if _matches_host(host, "linkedin.com"):
        return "linkedin", "profile"
    return "website", "web_page"


def discover_urls(value: str) -> list[str]:
    matches = [(match.start(), match.group(0)) for match in HTTP_URL_RE.finditer(value)]
    matches.extend(
        (match.start(), match.group(0)) for match in BARE_URL_RE.finditer(value)
    )
    return [url for _, url in sorted(matches, key=lambda item: item[0])]


def is_authentication_url(value: str) -> bool:
    try:
        parsed = urlsplit(value)
    except ValueError:
        return True
    host = (parsed.hostname or "").lower().rstrip(".")
    path = parsed.path.lower()
    if host in {
        "accounts.google.com",
        "account.adobe.com",
        "auth.services.adobe.com",
        "login.live.com",
        "login.microsoftonline.com",
    }:
        return True
    if _matches_host(host, "notion.so") or _matches_host(host, "figma.com"):
        return path.startswith(("/login", "/signup"))
    if _matches_host(host, "canva.com"):
        return path.startswith(("/login", "/signup"))
    return _matches_host(host, "linkedin.com") and path.startswith(
        ("/authwall", "/login")
    )


def is_skippable_enrichment_url(value: str) -> bool:
    """Hosts that never yield usable public evidence (auth walls, etc.)."""
    try:
        host = (urlsplit(value).hostname or "").lower().rstrip(".")
    except ValueError:
        return True
    if not host:
        return True
    return host in {"linkedin.com", "www.linkedin.com"} or host.endswith(
        ".linkedin.com"
    )


def is_private_ip(host: str) -> bool:
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
    )


def stable_urls(
    values: list[object],
    limit: int,
    skip: Callable[[str], bool] | None = None,
) -> list[str]:
    """Normalize, dedupe, and cap URLs in discovery order.

    URLs matching ``skip`` are dropped before the cap is applied so hosts
    that can never be enriched do not consume enrichment slots.
    """
    if limit <= 0:
        return []
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        url = normalize_url(value)
        if url and url not in seen:
            if skip is not None and skip(url):
                continue
            result.append(url)
            seen.add(url)
            if len(result) >= limit:
                break
    return result
