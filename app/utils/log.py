import json
import os
import sys
import uuid
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
        self.logger = loguru_logger.patch(self.patching)
        os.makedirs("logs", exist_ok=True)
        self.logger.remove()
        self.logger.add(
            self._stdout_sink,
            backtrace=False,
            diagnose=False,
        )
        self.logger.add(
            self._file_sink_path(),
            retention="10 days",
            rotation="00:00",
            backtrace=False,
            diagnose=False,
            format="{extra[output]}",
        )

    def serialize(self, record):
        subset = {
            "timestamp": record["time"].strftime("%d/%m/%Y %H.%M.%S.%f WIB"),
            "id": str(uuid.uuid4()),
            "request_id": record["extra"].get("request_id"),
            "level": record["level"].name,
            "message": record["message"],
            "error": self._mask_data(record["extra"].get("error", None)),
            "request": {
                "method": record["extra"].get("method"),
                "url": str(record["extra"].get("url"))
                if record["extra"].get("url")
                else None,
                "client_ip": record["extra"].get("client_ip"),
                "query_params": self._truncate_data(
                    self._mask_data(record["extra"].get("query_params"))
                ),
                "payload": self._truncate_data(
                    self._mask_data(record["extra"].get("request_payload"))
                ),
            },
            "response": {
                "status": record["extra"].get("status_code"),
                "body": self._truncate_data(
                    self._mask_data(record["extra"].get("response_body", None))
                ),
                "duration_ms": record["extra"].get("duration_ms"),
            },
            "performance": self._truncate_data(record["extra"].get("performance")),
            "user": {"id": record["extra"].get("user_id", None)},
        }
        indent = 2 if self.pretty_json else None
        return json.dumps(subset, default=str, ensure_ascii=False, indent=indent)

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
