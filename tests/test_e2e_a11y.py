"""Accessibility assertions for the two main surfaces, via axe-core.

The library home page and the movie detail modal are scanned with axe-core
(bundled by ``axe-playwright-python``). Gating is scoped to **serious + critical**
impact — the violations that actually block or mislead assistive-technology
users. Best-practice hints (heading order, landmark coverage) still surface in
the report but do not fail the build, so the check stays meaningful without
turning into a style linter.

Requires ``pytest-playwright`` + ``axe-playwright-python`` (see
``requirements-dev.txt``) and browsers (``python -m playwright install
--with-deps chromium``). Run with ``pytest -m e2e -o addopts=""``.
"""
import pytest

pytest.importorskip("playwright.sync_api", reason="pytest-playwright is not installed")
pytest.importorskip("axe_playwright_python.sync_playwright", reason="axe-playwright-python is not installed")

from playwright.sync_api import expect  # noqa: E402
from axe_playwright_python.sync_playwright import Axe  # noqa: E402

pytestmark = pytest.mark.e2e

#: Impacts that block or mislead users, as opposed to best-practice hints.
_BLOCKING_IMPACTS = {"serious", "critical"}


def _blocking_violations(page):
    """Run axe and return only the serious/critical violations."""
    result = Axe().run(page)
    return [v for v in result.response["violations"] if v.get("impact") in _BLOCKING_IMPACTS]


def _describe(violations):
    return "\n".join(
        f"[{v.get('impact')}] {v.get('id')}: {v.get('help')} ({len(v.get('nodes', []))} node(s))"
        for v in violations
    )


def test_library_home_has_no_blocking_a11y_violations(page, server_url):
    """The landing page (movie library) must be free of serious/critical issues."""
    page.set_default_timeout(30000)
    page.goto(f"{server_url}/")
    expect(page.locator("#movies-grid .movie-card").first).to_be_visible()

    violations = _blocking_violations(page)
    assert not violations, f"Blocking accessibility violations on the home page:\n{_describe(violations)}"


def test_detail_modal_has_no_blocking_a11y_violations(page, server_url):
    """The movie detail modal must be free of serious/critical issues when open."""
    page.set_default_timeout(30000)
    page.goto(f"{server_url}/")
    expect(page.locator("#movies-grid .movie-card").first).to_be_visible()

    page.locator("#movies-grid .movie-card").first.click()
    expect(page.locator("#detail-title")).to_be_visible()

    violations = _blocking_violations(page)
    assert not violations, (
        f"Blocking accessibility violations with the detail modal open:\n{_describe(violations)}"
    )
