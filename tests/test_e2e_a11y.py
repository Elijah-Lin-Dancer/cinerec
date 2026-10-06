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


def _settle(page):
    """Let entrance animations finish before scanning.

    axe's ``color-contrast`` rule folds an element's *effective* opacity into the
    colours it reports, so a scan that lands mid fade-in measures a blend that
    never exists at rest — that race is what intermittently failed CI (a
    pagination node measured at ~1.9:1 while its section was still fading in,
    against a passing ~6:1 once settled). So we wait for every finite animation
    and transition to drain, and for the web fonts to load, before scanning.

    The three always-on decorative loops (the two aurora drifts and the film
    grain) never finish by design, so they are excluded; GSAP cannot be polled
    via ``globalTimeline.isActive()`` because its ticker keeps the timeline
    active even when idle. The assertion itself is unchanged, so a genuinely
    low-contrast element still fails.
    """
    page.wait_for_function(
        """() => {
            if (document.fonts && document.fonts.status !== 'loaded') return false;
            return !document.getAnimations().some(a => {
                if (a.playState !== 'running') return false;
                const timing = (a.effect && a.effect.getTiming) ? a.effect.getTiming() : null;
                // Skip the infinite decorative loops (aurora-drift, grain-shift).
                return !(timing && timing.iterations === Infinity);
            });
        }""",
        timeout=20000,
    )
    # Two frames so the final paint is committed before axe samples it.
    page.evaluate("() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))")


def _describe(violations):
    """Render violations with their selector targets, so a failure is actionable."""
    lines = []
    for v in violations:
        nodes = v.get("nodes", [])
        targets = [n.get("target") for n in nodes]
        lines.append(f"[{v.get('impact')}] {v.get('id')}: {v.get('help')} ({len(nodes)} node(s))")
        for t, n in zip(targets, nodes):
            lines.append(f"    - {t}  ::  {n.get('failureSummary', '').splitlines()[0] if n.get('failureSummary') else ''}")
    return "\n".join(lines)


def test_library_home_has_no_blocking_a11y_violations(page, server_url):
    """The landing page (movie library) must be free of serious/critical issues."""
    page.set_default_timeout(30000)
    page.goto(f"{server_url}/")
    expect(page.locator("#movies-grid .movie-card").first).to_be_visible()
    _settle(page)

    violations = _blocking_violations(page)
    assert not violations, f"Blocking accessibility violations on the home page:\n{_describe(violations)}"


def test_detail_modal_has_no_blocking_a11y_violations(page, server_url):
    """The movie detail modal must be free of serious/critical issues when open."""
    page.set_default_timeout(30000)
    page.goto(f"{server_url}/")
    expect(page.locator("#movies-grid .movie-card").first).to_be_visible()

    page.locator("#movies-grid .movie-card").first.click()
    expect(page.locator("#detail-title")).to_be_visible()
    _settle(page)

    violations = _blocking_violations(page)
    assert not violations, (
        f"Blocking accessibility violations with the detail modal open:\n{_describe(violations)}"
    )
