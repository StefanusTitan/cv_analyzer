from app.utils.urls import (
    is_github_url,
    is_private_ip,
    is_skippable_enrichment_url,
    normalize_url,
    stable_urls,
)


def test_normalize_url_removes_tracking_and_fragments():
    assert (
        normalize_url("HTTPS://Example.com/profile?gclid=x&ok=1#bio")
        == "https://example.com/profile?ok=1"
    )


def test_normalize_url_removes_markdown_fences():
    assert normalize_url("https://github.com/example/repo`") == (
        "https://github.com/example/repo"
    )


def test_stable_urls_ignores_invalid_values_and_preserves_order():
    assert stable_urls(
        ["bad", "https://a.test", {"url": "x"}, "https://a.test/"], 10
    ) == [
        "https://a.test/",
    ]


def test_stable_urls_skipped_hosts_do_not_consume_cap_slots():
    values = [
        "https://www.linkedin.com/in/candidate",
        "https://github.com/user",
        "https://github.com/user/repo",
    ]
    assert stable_urls(values, 1, skip=is_skippable_enrichment_url) == [
        "https://github.com/user",
    ]


def test_private_ip_detection():
    assert is_private_ip("10.0.0.1")
    assert is_private_ip("169.254.169.254")
    assert is_private_ip("::1")
    assert not is_private_ip("8.8.8.8")


def test_github_hostname_is_exact():
    assert is_github_url("https://github.com/user")
    assert not is_github_url("https://github.com.example.test/user")


def test_linkedin_urls_are_skipped_for_enrichment():
    assert is_skippable_enrichment_url("https://www.linkedin.com/in/someone")
    assert is_skippable_enrichment_url("https://linkedin.com/in/someone")
    assert is_skippable_enrichment_url("https://m.linkedin.com/in/someone")
    assert not is_skippable_enrichment_url("https://example.com/cv")
    assert not is_skippable_enrichment_url("https://github.com/user")
