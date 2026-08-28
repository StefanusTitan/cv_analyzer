from app.services.web_scraper import WebScraper


def test_shared_document_text_preserves_readable_line_structure():
    text = "Heading\n  First   point  \n\nSecond point"

    assert WebScraper._normalize_text(text, True) == (
        "Heading\nFirst point\nSecond point"
    )
    assert WebScraper._normalize_text(text, False) == (
        "Heading First point Second point"
    )


def test_restricted_shared_page_content_is_detected_without_generic_login_text():
    assert WebScraper._has_restricted_content(
        "Google Drive", "You need access. Request access to this file."
    )
    assert not WebScraper._has_restricted_content(
        "Public portfolio", "Sign in to comment, or continue reading publicly."
    )


def test_shared_platforms_use_targeted_content_regions():
    assert "docs-editor-container" in WebScraper._content_selector("google_docs")
    assert '[role="main"]' in WebScraper._content_selector("notion")
    assert WebScraper._content_selector("website") == "main, article, body"
