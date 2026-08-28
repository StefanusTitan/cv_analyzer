from app.utils.urls import (
    classify_source,
    discover_urls,
    is_authentication_url,
    is_github_url,
    is_gitlab_url,
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


def test_normalize_url_accepts_bare_public_domains():
    assert normalize_url("portfolio.example.com/candidate") == (
        "https://portfolio.example.com/candidate"
    )
    assert normalize_url("not-a-domain") is None


def test_discover_urls_recovers_bare_cross_role_sources_without_emails():
    text = (
        "Design behance.net/candidate and research orcid.org/0000-0001. "
        "Email candidate@example.com or visit https://gitlab.com/team/project."
    )

    assert discover_urls(text) == [
        "behance.net/candidate",
        "orcid.org/0000-0001.",
        "https://gitlab.com/team/project.",
    ]


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


def test_stable_urls_respects_disabled_enrichment():
    assert stable_urls(["https://github.com/user"], 0) == []


def test_private_ip_detection():
    assert is_private_ip("10.0.0.1")
    assert is_private_ip("169.254.169.254")
    assert is_private_ip("::1")
    assert not is_private_ip("8.8.8.8")


def test_github_hostname_is_exact():
    assert is_github_url("https://github.com/user")
    assert not is_github_url("https://github.com.example.test/user")


def test_gitlab_hostname_is_exact():
    assert is_gitlab_url("https://gitlab.com/user/project")
    assert not is_gitlab_url("https://gitlab.com.example.test/user/project")


def test_cross_role_sources_are_classified_by_platform_and_artifact():
    assert classify_source("https://gitlab.com/team/project") == (
        "gitlab",
        "repository",
    )
    assert classify_source("https://www.behance.net/gallery/1/Project") == (
        "behance",
        "project",
    )
    assert classify_source("https://docs.google.com/presentation/d/abc") == (
        "google_docs",
        "presentation",
    )
    assert classify_source("https://orcid.org/0000-0001") == (
        "orcid",
        "research_profile",
    )
    assert classify_source("https://www.youtube.com/watch?v=abc") == (
        "youtube",
        "video",
    )
    assert classify_source("https://www.youtube.com/@candidate") == (
        "youtube",
        "profile",
    )


def test_linkedin_urls_are_skipped_for_enrichment():
    assert is_skippable_enrichment_url("https://www.linkedin.com/in/someone")
    assert is_skippable_enrichment_url("https://linkedin.com/in/someone")
    assert is_skippable_enrichment_url("https://m.linkedin.com/in/someone")
    assert not is_skippable_enrichment_url("https://example.com/cv")
    assert not is_skippable_enrichment_url("https://github.com/user")


def test_shared_platform_authentication_redirects_are_recognized():
    assert is_authentication_url("https://accounts.google.com/signin")
    assert is_authentication_url("https://www.notion.so/login")
    assert is_authentication_url("https://www.figma.com/signup")
    assert is_authentication_url("https://auth.services.adobe.com/signin")
    assert not is_authentication_url("https://candidate.notion.site/portfolio")
