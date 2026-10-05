/**
 * CineRec — Movie Browsing Page Logic
 *
 * Server-provided strings (title, poster_url, genres) are inserted through the
 * DOM API / textContent only — never innerHTML — and clicks are handled by
 * event delegation, so a hostile title cannot inject markup or script.
 */

const PER_PAGE = 20;
const PAGE_WINDOW = 1; // neighbours shown on each side of the current page

let _genresCache = null;

// --- Data helpers -----------------------------------------------------------

// MovieLens stores genres pipe-separated, but the enriched feed has a few
// comma-joined values ("Action, Mystery, Thriller"), so split on both.
function _splitGenres(raw) {
    return String(raw || '')
        .split(/[|,]/)
        .map(g => g.trim())
        .filter(Boolean);
}

// Titles arrive as "Toy Story (1995)"; the year is rendered as its own chip, so
// drop the duplicate suffix from the displayed name.
function _displayTitle(title) {
    return String(title || '').replace(/\s*\(\d{4}\)\s*$/, '').trim();
}

function _initialOf(title) {
    const clean = _displayTitle(title);
    return clean ? clean.charAt(0).toUpperCase() : '?';
}

// --- Filter chips -----------------------------------------------------------

const YEAR_FILTERS = [
    { key: 'all', from: 0, to: 0 },
    { label: '2020s', from: 2020, to: 2029 },
    { label: '2010s', from: 2010, to: 2019 },
    { label: '2000s', from: 2000, to: 2009 },
    { label: '1990s', from: 1990, to: 1999 },
    { label: '1980s', from: 1980, to: 1989 },
    { key: 'earlier', from: 0, to: 1979 },
];

const SORT_FILTERS = [
    { value: 'id', key: 'movies.sortDefault' },
    { value: 'title', key: 'movies.sortName' },
    { value: 'year', key: 'movies.sortByYear' },
];

function _chip(label, active, dataset) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = `chip${active ? ' active' : ''}`;
    btn.textContent = label;
    Object.assign(btn.dataset, dataset);
    return btn;
}

function _activateChip(container, chip) {
    container.querySelectorAll('.chip').forEach(c => c.classList.remove('active'));
    if (chip) chip.classList.add('active');
}

function _fill(container, chips) {
    const scrollLeft = container.scrollLeft;
    container.textContent = '';
    chips.forEach(c => container.appendChild(c));
    container.scrollLeft = scrollLeft;
}

// Fetch and normalise the genre list once.
async function _loadGenres() {
    if (_genresCache) return _genresCache;
    try {
        const data = await CineRec.api('/api/movies/genres');
        const set = new Set();
        (data.genres || []).forEach(g => _splitGenres(g).forEach(x => set.add(x)));
        _genresCache = Array.from(set).sort((a, b) => a.localeCompare(b));
    } catch (e) {
        _genresCache = [];
    }
    return _genresCache;
}

function _renderGenreChips() {
    const host = document.getElementById('genre-chips');
    const f = CineRec.state.filters;
    const chips = [_chip(CineRec.t('movies.all'), f.genre === '', { genre: '' })];
    (_genresCache || []).forEach(g => {
        chips.push(_chip(CineRec.genreName(g), f.genre === g, { genre: g }));
    });
    _fill(host, chips);
}

function _renderYearChips() {
    const host = document.getElementById('year-chips');
    const f = CineRec.state.filters;
    const chips = YEAR_FILTERS.map(y => {
        const label = y.label || CineRec.t(`movies.${y.key}`);
        const active = f.yearFrom === y.from && f.yearTo === y.to;
        return _chip(label, active, { yearFrom: String(y.from), yearTo: String(y.to) });
    });
    _fill(host, chips);
}

function _renderSortChips() {
    const host = document.getElementById('sort-chips');
    const f = CineRec.state.filters;
    const chips = SORT_FILTERS.map(s =>
        _chip(CineRec.t(s.key), f.sort === s.value, { sort: s.value })
    );
    _fill(host, chips);
}

// Build (or rebuild) the filter chip rows. Idempotent and safe to call again
// after any language change.
async function buildMovieFilters() {
    _renderYearChips();
    _renderSortChips();
    _renderGenreChips();
    await _loadGenres();
    _renderGenreChips();
}

function _bindFilterEvents() {
    const genreHost = document.getElementById('genre-chips');
    const yearHost = document.getElementById('year-chips');
    const sortHost = document.getElementById('sort-chips');
    if (!genreHost || genreHost.dataset.bound) return;
    genreHost.dataset.bound = '1';
    yearHost.dataset.bound = '1';
    sortHost.dataset.bound = '1';

    genreHost.addEventListener('click', (e) => {
        const chip = e.target.closest('.chip');
        if (!chip) return;
        CineRec.state.filters.genre = chip.dataset.genre || '';
        _activateChip(genreHost, chip);
        loadMovies(1);
    });

    yearHost.addEventListener('click', (e) => {
        const chip = e.target.closest('.chip');
        if (!chip) return;
        CineRec.state.filters.yearFrom = Number(chip.dataset.yearFrom) || 0;
        CineRec.state.filters.yearTo = Number(chip.dataset.yearTo) || 0;
        _activateChip(yearHost, chip);
        loadMovies(1);
    });

    sortHost.addEventListener('click', (e) => {
        const chip = e.target.closest('.chip');
        if (!chip) return;
        CineRec.state.filters.sort = chip.dataset.sort || 'id';
        _activateChip(sortHost, chip);
        loadMovies(1);
    });
}

// --- Data loading -----------------------------------------------------------

async function loadMovies(page = 1) {
    const search = document.getElementById('movie-search').value;
    const f = CineRec.state.filters;

    const params = new URLSearchParams({
        page: String(page),
        per_page: String(PER_PAGE),
        search,
        genre: f.genre,
        sort: f.sort,
    });
    if (f.yearFrom) params.set('year_from', String(f.yearFrom));
    if (f.yearTo) params.set('year_to', String(f.yearTo));

    const grid = document.getElementById('movies-grid');
    grid.setAttribute('aria-busy', 'true');

    try {
        const data = await CineRec.api(`/api/movies?${params.toString()}`);
        CineRec.state.movies = data.movies;
        CineRec.state.moviesLoaded = true;
        CineRec.state.moviesTotal = data.total;
        CineRec.state.moviesPages = data.pages;
        CineRec.state.moviePage = data.page;

        renderMovieGrid(data.movies);
        renderPagination(data.page, data.pages);
        renderMovieCount(data.total);
        if (typeof Animations !== 'undefined') Animations.animateMovieCards();
    } catch (err) {
        grid.textContent = '';
        const p = document.createElement('p');
        p.className = 'empty-state';
        p.textContent = CineRec.t('common.error');
        grid.appendChild(p);
    } finally {
        grid.removeAttribute('aria-busy');
    }
}

function renderMovieCount(total) {
    const el = document.getElementById('movies-count');
    if (el) el.textContent = `${Number(total).toLocaleString()} ${CineRec.t('movies.results')}`;
}

// --- Rendering --------------------------------------------------------------

function _moviePlaceholder(title) {
    const ph = document.createElement('div');
    ph.className = 'poster-placeholder';
    const span = document.createElement('span');
    span.className = 'ph-initial';
    span.textContent = _initialOf(title);
    ph.appendChild(span);
    return ph;
}

function _moviePoster(movie) {
    const wrap = document.createElement('div');
    wrap.className = 'movie-poster';
    if (movie.poster_url) {
        const img = document.createElement('img');
        img.src = movie.poster_url;
        img.alt = _displayTitle(movie.title);
        img.loading = 'lazy';
        img.decoding = 'async';
        // Poster hosts often reject hot-linking based on Referer; send none.
        img.referrerPolicy = 'no-referrer';
        img.addEventListener('error', () => {
            wrap.textContent = '';
            wrap.appendChild(_moviePlaceholder(movie.title));
        });
        wrap.appendChild(img);
    } else {
        wrap.appendChild(_moviePlaceholder(movie.title));
    }
    return wrap;
}

function _movieCard(movie) {
    const card = document.createElement('div');
    card.className = 'movie-card tilt-card spotlight-card';
    card.dataset.id = String(movie.id);
    // The whole card is the affordance that opens the detail modal, so expose
    // it to the keyboard as a button as well.
    card.tabIndex = 0;
    card.setAttribute('role', 'button');
    card.setAttribute('aria-label', `${_displayTitle(movie.title)} · ${CineRec.t('detail.viewDetails')}`);

    const poster = _moviePoster(movie);
    const hint = document.createElement('span');
    hint.className = 'movie-card-hint';
    hint.textContent = CineRec.t('detail.viewDetails');
    poster.appendChild(hint);
    card.appendChild(poster);

    const info = document.createElement('div');
    info.className = 'movie-info';

    const title = document.createElement('h3');
    title.className = 'movie-title';
    title.textContent = _displayTitle(movie.title);
    title.title = _displayTitle(movie.title);
    info.appendChild(title);

    const meta = document.createElement('div');
    meta.className = 'movie-meta';
    if (movie.release_year) {
        const year = document.createElement('span');
        year.className = 'movie-year';
        year.textContent = String(movie.release_year);
        meta.appendChild(year);
    }
    const genreNames = _splitGenres(movie.genres).slice(0, 3).map(g => CineRec.genreName(g));
    if (genreNames.length) {
        const genres = document.createElement('span');
        genres.className = 'movie-genres';
        genres.textContent = genreNames.join(' · ');
        genres.title = _splitGenres(movie.genres).map(g => CineRec.genreName(g)).join(' · ');
        meta.appendChild(genres);
    }
    info.appendChild(meta);

    const actions = document.createElement('div');
    actions.className = 'movie-actions';
    const rateBtn = document.createElement('button');
    rateBtn.className = 'btn-sm btn-outline';
    rateBtn.dataset.action = 'rate';
    rateBtn.textContent = CineRec.t('movies.rate');
    actions.appendChild(rateBtn);
    info.appendChild(actions);

    card.appendChild(info);
    return card;
}

function renderMovieGrid(movies) {
    const grid = document.getElementById('movies-grid');
    grid.textContent = '';

    if (!movies.length) {
        const p = document.createElement('p');
        p.className = 'empty-state';
        p.textContent = CineRec.t('movies.noResults');
        grid.appendChild(p);
        return;
    }

    const frag = document.createDocumentFragment();
    movies.forEach(m => frag.appendChild(_movieCard(m)));
    grid.appendChild(frag);

    // Re-init effects for new cards
    if (typeof Effects !== 'undefined') Effects.refresh();
}

// Windowed page list: always shows the first and last page, the current page
// and its neighbours, and an ellipsis where pages are skipped. Replaces the
// previous behaviour of laying out every page number (1..85) in a flat row.
function _pageWindow(current, total) {
    if (total <= 7) {
        return Array.from({ length: total }, (_, i) => i + 1);
    }
    const pages = [1];
    const start = Math.max(2, current - PAGE_WINDOW);
    const end = Math.min(total - 1, current + PAGE_WINDOW);

    if (start > 2) pages.push('…');
    for (let i = start; i <= end; i++) pages.push(i);
    if (end < total - 1) pages.push('…');
    pages.push(total);
    return pages;
}

function _navButton(label, page, disabled, ariaLabel) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'page-btn page-nav';
    btn.textContent = label;
    btn.dataset.page = String(page);
    btn.disabled = disabled;
    if (ariaLabel) btn.setAttribute('aria-label', ariaLabel);
    return btn;
}

function renderPagination(current, total) {
    const pag = document.getElementById('movies-pagination');
    pag.textContent = '';
    if (!total || total <= 1) {
        pag.classList.add('hidden');
        return;
    }
    pag.classList.remove('hidden');

    const frag = document.createDocumentFragment();

    frag.appendChild(_navButton('‹', current - 1, current <= 1, CineRec.t('movies.prev')));

    _pageWindow(current, total).forEach(item => {
        if (item === '…') {
            const gap = document.createElement('span');
            gap.className = 'page-gap';
            gap.textContent = '…';
            frag.appendChild(gap);
            return;
        }
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = `page-btn${item === current ? ' active' : ''}`;
        btn.dataset.page = String(item);
        btn.textContent = String(item);
        if (item === current) btn.setAttribute('aria-current', 'page');
        frag.appendChild(btn);
    });

    frag.appendChild(_navButton('›', current + 1, current >= total, CineRec.t('movies.next')));

    const info = document.createElement('span');
    info.className = 'pagination-info';
    info.textContent = CineRec.t('movies.pageOf', { page: current, pages: total });
    frag.appendChild(info);

    pag.appendChild(frag);
}

// --- Rating modal -----------------------------------------------------------

let ratingMovieId = null;
function openRatingModal(movieId, title) {
    ratingMovieId = movieId;
    document.getElementById('rating-movie-title').textContent = title || '';
    document.getElementById('rating-modal').classList.remove('hidden');
    document.querySelectorAll('.star-rating .star').forEach(s => s.classList.remove('active'));
    if (typeof Effects !== 'undefined') Effects.refresh();
}

function _movieTitleById(movieId) {
    const match = (CineRec.state.movies || []).find(m => Number(m.id) === Number(movieId));
    return match ? _displayTitle(match.title) : '';
}

// --- Wiring -----------------------------------------------------------------

document.addEventListener('DOMContentLoaded', () => {
    _bindFilterEvents();

    // Event delegation: a card click opens the detail modal; the Rate button
    // inside it opens the rating modal instead. Ids come from data-id, so no
    // title or id is ever interpolated into markup.
    document.getElementById('movies-grid').addEventListener('click', (e) => {
        const card = e.target.closest('.movie-card');
        if (!card) return;
        const movieId = Number(card.dataset.id);
        if (e.target.closest('[data-action="rate"]')) {
            openRatingModal(movieId, _movieTitleById(movieId));
            return;
        }
        if (typeof MovieDetail !== 'undefined') MovieDetail.open(movieId);
    });

    // Keyboard equivalent of the card click, so the library is usable without a
    // mouse. Enter/Space on the focused card opens its details.
    document.getElementById('movies-grid').addEventListener('keydown', (e) => {
        if (e.key !== 'Enter' && e.key !== ' ' && e.key !== 'Spacebar') return;
        const card = e.target.closest('.movie-card');
        if (!card || e.target.closest('[data-action="rate"]')) return;
        e.preventDefault();
        if (typeof MovieDetail !== 'undefined') MovieDetail.open(Number(card.dataset.id));
    });

    // Event delegation for pagination buttons.
    document.getElementById('movies-pagination').addEventListener('click', (e) => {
        const btn = e.target.closest('.page-btn');
        if (!btn || btn.disabled) return;
        loadMovies(Number(btn.dataset.page));
    });

    // Close modal
    document.querySelector('.modal-close').addEventListener('click', () => {
        document.getElementById('rating-modal').classList.add('hidden');
    });
    document.getElementById('rating-modal').addEventListener('click', (e) => {
        if (e.target === e.currentTarget) e.currentTarget.classList.add('hidden');
    });

    // Star rating interaction
    const stars = document.querySelectorAll('.star-rating .star');
    let currentRating = 0;
    stars.forEach(star => {
        star.addEventListener('mouseenter', () => {
            const val = parseInt(star.dataset.value);
            stars.forEach(s => s.classList.toggle('active', parseInt(s.dataset.value) <= val));
        });
        star.addEventListener('click', () => {
            const val = parseInt(star.dataset.value);
            currentRating = val;
            stars.forEach(s => s.classList.toggle('active', parseInt(s.dataset.value) <= val));
        });
    });
    document.querySelector('.star-rating').addEventListener('mouseleave', () => {
        stars.forEach(s => s.classList.toggle('active', parseInt(s.dataset.value) <= currentRating));
    });

    // Submit rating
    document.getElementById('btn-submit-rating').addEventListener('click', async () => {
        const activeStar = document.querySelector('.star-rating .star.active:last-of-type');
        if (!activeStar || !ratingMovieId) return;
        const rating = parseInt(activeStar.dataset.value);
        if (!CineRec.state.userId) {
            document.getElementById('rating-modal').classList.add('hidden');
            CineRec.navigateTo('login');
            return;
        }
        try {
            await CineRec.api(`/api/movies/${ratingMovieId}/rate?user_id=${CineRec.state.userId}&rating=${rating}`);
        } catch (e) {}
        document.getElementById('rating-modal').classList.add('hidden');
    });

    // Search with debounce
    let searchTimer;
    document.getElementById('movie-search').addEventListener('input', () => {
        clearTimeout(searchTimer);
        searchTimer = setTimeout(() => loadMovies(1), 300);
    });

    // Re-render dynamic content when the language changes.
    CineRec.onLangChange(() => {
        buildMovieFilters();
        if (CineRec.state.moviesLoaded) {
            renderMovieGrid(CineRec.state.movies);
            renderPagination(CineRec.state.moviePage, CineRec.state.moviesPages);
            renderMovieCount(CineRec.state.moviesTotal);
        }
    });
});