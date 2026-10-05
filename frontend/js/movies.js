/**
 * CineRec — Movie Browsing Page Logic
 *
 * Server-provided strings (title, poster_url, genres) are inserted through the
 * DOM API / textContent only — never innerHTML — and clicks are handled by
 * event delegation, so a hostile title cannot inject markup or script.
 */
async function loadMovies(page = 1) {
    const search = document.getElementById('movie-search').value;
    const genre = document.getElementById('genre-filter').value;
    const sort = document.getElementById('sort-select').value;

    try {
        const data = await CineRec.api(
            `/api/movies?page=${page}&per_page=20&search=${encodeURIComponent(search)}&genre=${encodeURIComponent(genre)}&sort=${sort}`
        );
        CineRec.state.movies = data.movies;
        renderMovieGrid(data.movies);
        renderPagination(data.page, data.pages);
        Animations.animateMovieCards();
    } catch (err) {
        const grid = document.getElementById('movies-grid');
        grid.textContent = '';
        const p = document.createElement('p');
        p.textContent = CineRec.t('common.error');
        grid.appendChild(p);
    }
}

function _moviePlaceholder(title) {
    const ph = document.createElement('div');
    ph.className = 'poster-placeholder';
    const span = document.createElement('span');
    span.textContent = (title && String(title).charAt(0)) || '?';
    ph.appendChild(span);
    return ph;
}

function _moviePoster(movie) {
    const wrap = document.createElement('div');
    wrap.className = 'movie-poster';
    if (movie.poster_url) {
        const img = document.createElement('img');
        img.src = movie.poster_url;
        img.alt = movie.title || '';
        img.loading = 'lazy';
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

    card.appendChild(_moviePoster(movie));

    const info = document.createElement('div');
    info.className = 'movie-info';

    const title = document.createElement('h3');
    title.className = 'movie-title';
    title.textContent = movie.title || '';
    info.appendChild(title);

    const meta = document.createElement('div');
    meta.className = 'movie-meta';
    if (movie.release_year) {
        const year = document.createElement('span');
        year.className = 'movie-year';
        year.textContent = String(movie.release_year);
        meta.appendChild(year);
    }
    if (movie.genres) {
        const genres = document.createElement('span');
        genres.className = 'movie-genres';
        genres.textContent = String(movie.genres).split('|').slice(0, 3).join(' · ');
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

function renderPagination(current, total) {
    const pag = document.getElementById('movies-pagination');
    pag.textContent = '';
    if (total <= 1) return;

    const frag = document.createDocumentFragment();
    for (let i = 1; i <= total; i++) {
        const btn = document.createElement('button');
        btn.className = `page-btn ${i === current ? 'active' : ''}`;
        btn.dataset.page = String(i);
        btn.textContent = String(i);
        frag.appendChild(btn);
    }
    pag.appendChild(frag);
}

// Rating modal
let ratingMovieId = null;
function openRatingModal(movieId, title) {
    ratingMovieId = movieId;
    document.getElementById('rating-movie-title').textContent = title || '';
    document.getElementById('rating-modal').classList.remove('hidden');
    document.querySelectorAll('.star-rating .star').forEach(s => s.classList.remove('active'));
    Effects.refresh();
}

function _movieTitleById(movieId) {
    const match = (CineRec.state.movies || []).find(m => Number(m.id) === Number(movieId));
    return match ? match.title : '';
}

document.addEventListener('DOMContentLoaded', () => {
    // Event delegation: rating is opened from the card's data-id, so no title or
    // id is ever interpolated into markup.
    document.getElementById('movies-grid').addEventListener('click', (e) => {
        const btn = e.target.closest('[data-action="rate"]');
        if (!btn) return;
        const card = btn.closest('.movie-card');
        if (!card) return;
        const movieId = Number(card.dataset.id);
        openRatingModal(movieId, _movieTitleById(movieId));
    });

    // Event delegation for pagination buttons.
    document.getElementById('movies-pagination').addEventListener('click', (e) => {
        const btn = e.target.closest('.page-btn');
        if (!btn) return;
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
        try {
            await CineRec.api(`/api/movies/${ratingMovieId}/rate?user_id=${CineRec.state.userId}&rating=${rating}`);
        } catch(e) {}
        document.getElementById('rating-modal').classList.add('hidden');
    });

    // Search with debounce
    let searchTimer;
    document.getElementById('movie-search').addEventListener('input', () => {
        clearTimeout(searchTimer);
        searchTimer = setTimeout(() => loadMovies(1), 300);
    });

    // Filter changes
    document.getElementById('genre-filter').addEventListener('change', () => loadMovies(1));
    document.getElementById('sort-select').addEventListener('change', () => loadMovies(1));
});