import ipaddress
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

_TRACKING_PARAMS = {"fbclid", "gclid", "mc_cid", "mc_eid"}


def normalize_url(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip().rstrip(".,;:!?)]}>`*")
    if len(value) > 2_048:
        return None
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


def stable_urls(values: list[object], limit: int) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        url = normalize_url(value)
        if url and url not in seen:
            result.append(url)
            seen.add(url)
            if len(result) >= limit:
                break
    return result
