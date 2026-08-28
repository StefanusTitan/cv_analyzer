import json
import os
import sys
from typing import Any, ClassVar

from loguru import logger as loguru_logger


class Logger:
    DEFAULT_SENSITIVE_KEYS: ClassVar[set[str]] = {
        "authorization",
        "password",
        "passwd",
        "secret",
        "token",
        "access_token",
        "refresh_token",
        "api_key",
        "apikey",
        "image",
        "image_bytes",
        "video",
        "video_bytes",
        "file_bytes",
    }

    def __init__(self):
        self.masking_enabled = self._get_bool_env("LOG_MASKING_ENABLED", True)
        self.sensitive_keys = self._load_sensitive_keys()
        self.max_value_length = self._get_int_env("LOG_MAX_VALUE_LENGTH", 2000)
        self.max_collection_items = self._get_int_env("LOG_MAX_COLLECTION_ITEMS", 30)
        self.pretty_json = self._get_bool_env("LOG_PRETTY_JSON", False)
        self.file_enabled = self._get_bool_env("LOG_FILE_ENABLED", True)
        self.logger = loguru_logger.patch(self.patching)
        self.logger.remove()
        self.logger.add(
            self._stdout_sink,
            backtrace=False,
            diagnose=False,
        )
        if self.file_enabled:
            os.makedirs("logs", exist_ok=True)
            self.logger.add(
                self._file_sink_path(),
                retention="10 days",
                rotation="00:00",
                backtrace=False,
                diagnose=False,
                format="{extra[output]}",
            )

    _HTTP_EXTRA_KEYS: ClassVar[set[str]] = {
        "method",
        "url",
        "client_ip",
        "query_params",
        "request_payload",
        "response_body",
        "duration_ms",
        "user_id",
    }
    _STRUCTURED_EXTRA_KEYS: ClassVar[set[str]] = {
        "output",
        "request_id",
        "component",
        "event",
        "stage",
        "error_code",
        "error",
        "performance",
        "status_code",
        *_HTTP_EXTRA_KEYS,
    }
    _UNTRUNCATED_EXTRA_KEYS: ClassVar[set[str]] = {
        "llm",
    }

    def serialize(self, record):
        extra = record["extra"]
        subset = {
            "timestamp": record["time"].strftime("%d/%m/%Y %H.%M.%S.%f WIB"),
            "request_id": extra.get("request_id"),
            "level": record["level"].name,
            "component": extra.get("component"),
            "event": extra.get("event"),
            "stage": extra.get("stage"),
            "error_code": extra.get("error_code"),
            "status_code": extra.get("status_code"),
            "message": record["message"],
            "error": self._mask_data(extra.get("error")),
            "performance": self._truncate_data(extra.get("performance")),
        }
        if extra.get("method") or extra.get("url"):
            subset["request"] = {
                "method": extra.get("method"),
                "url": str(extra["url"]) if extra.get("url") else None,
                "client_ip": extra.get("client_ip"),
                "query_params": self._truncate_data(
                    self._mask_data(extra.get("query_params"))
                ),
                "payload": self._truncate_data(
                    self._mask_data(extra.get("request_payload"))
                ),
            }
            subset["response"] = {
                "status": extra.get("status_code"),
                "body": self._truncate_data(self._mask_data(extra.get("response_body"))),
                "duration_ms": extra.get("duration_ms"),
            }
        for key, value in extra.items():
            if key in self._STRUCTURED_EXTRA_KEYS or key in subset:
                continue
            masked = self._mask_data(value)
            subset[key] = (
                masked
                if key in self._UNTRUNCATED_EXTRA_KEYS
                else self._truncate_data(masked)
            )
        indent = 2 if self.pretty_json else None
        return json.dumps(
            self._omit_empty(subset), default=str, ensure_ascii=False, indent=indent
        )

    def _omit_empty(self, value: Any):
        if isinstance(value, dict):
            compacted = {}
            for item_key, item_value in value.items():
                item = self._omit_empty(item_value)
                if item is None or item == {} or item == []:
                    continue
                compacted[item_key] = item
            return compacted
        if isinstance(value, list):
            return [self._omit_empty(item) for item in value]
        if isinstance(value, tuple):
            return [self._omit_empty(item) for item in value]
        return value

    def _mask_data(self, value: Any, key: str | None = None):
        if not self.masking_enabled:
            return value

        if key and self._is_sensitive_key(key):
            return self._mask_value(value)

        if isinstance(value, dict):
            return {
                item_key: self._mask_data(item_value, item_key)
                for item_key, item_value in value.items()
            }

        if isinstance(value, list):
            return [self._mask_data(item) for item in value]

        if isinstance(value, tuple):
            return [self._mask_data(item) for item in value]

        return value

    def _truncate_data(self, value: Any):
        if isinstance(value, dict):
            items = list(value.items())
            truncated = {
                item_key: self._truncate_data(item_value)
                for item_key, item_value in items[: self.max_collection_items]
            }
            if len(items) > self.max_collection_items:
                truncated["__truncated_items__"] = (
                    len(items) - self.max_collection_items
                )
            return truncated

        if isinstance(value, list):
            truncated = [
                self._truncate_data(item) for item in value[: self.max_collection_items]
            ]
            if len(value) > self.max_collection_items:
                truncated.append(
                    f"... truncated {len(value) - self.max_collection_items} items"
                )
            return truncated

        if isinstance(value, tuple):
            return self._truncate_data(list(value))

        if isinstance(value, str):
            if len(value) <= self.max_value_length:
                return value
            return f"{value[: self.max_value_length]}... [truncated {len(value) - self.max_value_length} chars]"

        return value

    def _is_sensitive_key(self, key: str) -> bool:
        normalized = key.lower()
        return any(marker in normalized for marker in self.sensitive_keys)

    def _mask_value(self, value: Any):
        if value is None:
            return None

        if isinstance(value, dict):
            return {item_key: "***masked***" for item_key in value}

        if isinstance(value, list):
            return ["***masked***" for _ in value]

        if isinstance(value, tuple):
            return ["***masked***" for _ in value]

        if isinstance(value, str):
            if len(value) <= 4:
                return "***masked***"
            return f"{value[:2]}***{value[-2:]}"

        return "***masked***"

    def _load_sensitive_keys(self):
        extra_keys = self._split_csv_env("LOG_MASK_FIELDS")
        excluded_keys = self._split_csv_env("LOG_UNMASK_FIELDS")
        return {
            key
            for key in self.DEFAULT_SENSITIVE_KEYS.union(extra_keys)
            if key not in excluded_keys
        }

    def _split_csv_env(self, env_name: str):
        raw_value = os.getenv(env_name, "")
        return {item.strip().lower() for item in raw_value.split(",") if item.strip()}

    def _get_bool_env(self, env_name: str, default: bool):
        raw_value = os.getenv(env_name)
        if raw_value is None:
            return default
        return raw_value.strip().lower() in {"1", "true", "yes", "on"}

    def _get_int_env(self, env_name: str, default: int):
        raw_value = os.getenv(env_name)
        if raw_value is None:
            return default
        try:
            return int(raw_value)
        except ValueError:
            return default

    def patching(self, record):
        record["extra"]["output"] = self.serialize(record)

    def _stdout_sink(self, message):
        sys.stdout.write(message.record["extra"]["output"] + "\n")

    def _file_sink_path(self):
        return "logs/log-{time:YYYY-MM-DD}.log"

    def get_logger(self):
        return self.logger


logger = Logger().get_logger()
