"""End-to-end journeys through the real UI against a real server.

Three reviewer journeys are pinned here, each driven through an actual browser
and the actual HTTP API (no request mocking):

1. guest sign-in → the recommendation page renders cards;
2. library filter → a title's detail modal opens and closes;
3. rating a recommended title → it drops out of the refreshed recommendations
   (the rating changed the user's exclusion set, so the served page changed).

The module is marked ``e2e`` and is excluded from the default ``pytest`` run
(see ``pyproject.toml``). Run it explicitly::

    pip install -r requirements-dev.txt
    python -m playwright install --with-deps chromium
    pytest -m e2e -o addopts=""

Unless ``CINEREC_E2E_BASE_URL`` names an already-running server, the shared
``server_url`` fixture (see ``conftest.py``) starts a ``lite`` uvicorn instance on
a free port for the session. ``lite`` is used because it is the free-tier
deployment shape and needs no torch.
"""
import re

import pytest

pytest.importorskip("playwright.sync_api", reason="pytest-playwright is not installed")

from playwright.sync_api import expect  # noqa: E402

pytestmark = pytest.mark.e2e


@pytest.fixture()
def app_page(page, server_url):
    """The SPA, opened and fully initialised (the library has rendered)."""
    page.set_default_timeout(30000)
    page.set_default_navigation_timeout(60000)
    page.goto(f"{server_url}/")
    expect(page.locator("#movies-grid .movie-card").first).to_be_visible()
    return page


def test_guest_signin_renders_recommendations(app_page):
    """Journey 1 — a visitor signs in as a guest and sees recommendations."""
    page = app_page
    page.click(".nav-login-link")
    expect(page.locator("#page-login")).to_be_visible()

    page.click("#btn-guest")

    expect(page.locator("#page-recommend")).to_be_visible()
    expect(page.locator(".rec-card").first).to_be_visible()
    assert page.locator(".rec-card").count() >= 1
    expect(page.locator("#user-badge")).to_be_visible()


def test_library_filter_opens_and_closes_detail_modal(app_page):
    """Journey 2 — filter the library, then open and close a title's details."""
    page = app_page

    # The genre chips are populated asynchronously; wait for a real genre.
    page.wait_for_selector("#genre-chips .chip:nth-child(2)")
    second_genre = page.locator("#genre-chips .chip").nth(1)
    second_genre.click()
    expect(second_genre).to_have_class(re.compile(r"\bactive\b"))

    card = page.locator("#movies-grid .movie-card").first
    expect(card).to_be_visible()
    card.click()

    expect(page.locator("#detail-modal")).to_be_visible()
    expect(page.locator("#detail-title")).not_to_be_empty()
    page.click("#detail-modal .detail-close")
    expect(page.locator("#detail-modal")).to_be_hidden()


def test_rating_refreshes_recommendations_excluding_rated_title(app_page):
    """Journey 3 — rating a recommended title removes it from the next page.

    The rating joins the user's exclusion set, so the refreshed recommendations
    must not serve that title again.
    """
    page = app_page
    page.click(".nav-login-link")
    page.click("#btn-guest")
    expect(page.locator("#page-recommend")).to_be_visible()

    first_rec = page.locator(".rec-card[data-id]").first
    expect(first_rec).to_be_visible()
    rated_id = first_rec.get_attribute("data-id")
    assert rated_id

    first_rec.click()
    expect(page.locator("#detail-modal")).to_be_visible()
    page.click("#detail-modal .detail-actions .btn-primary")  # "Rate"
    expect(page.locator("#rating-modal")).to_be_visible()

    page.click(".star-rating .star[data-value='5']")
    page.click("#btn-submit-rating")
    expect(page.locator("#rating-modal")).to_be_hidden()  # POST completed

    page.click("#btn-refresh-rec")
    expect(page.locator(".rec-card[data-id]").first).to_be_visible()
    expect(page.locator(f".rec-card[data-id='{rated_id}']")).to_have_count(0)
