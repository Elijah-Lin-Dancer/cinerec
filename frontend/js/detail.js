/**
 * CineRec — Movie Detail Modal
 *
 * Shows a single title's metadata and synopsis. As everywhere else in the
 * frontend, server-provided strings (title, overview, genres, poster_url) are
 * inserted through the DOM API only — never innerHTML — so a hostile title
 * cannot inject markup.
 *
 * The modal is deliberately self-contained and driven by a movie id, so it can
 * later be hung off a real route (#/movie/:id) without reworking the renderer.
 */
window.MovieDetail = (() => {
    let currentId = null;   // id of the in-flight / last request
    let current = null;     // payload currently rendered (kept for re-render)
    let lastFocus = null;   // element to hand focus back to on close

    function el(tag, cls, text) {
        const node = document.createElement(tag);
        if (cls) node.className = cls;
        if (text != null) node.textContent = text;
        return node;
    }

    // A stored tmdb_id is not guaranteed to be present or to resolve for every
    // row, so an IMDb search link (title + year) is the most stable way to send
    // someone to further information about a title.
    function _imdbUrl(title, year) {
        const name = _displayTitle(title);
        const query = year ? `${name} (${year})` : name;
        return `https://www.imdb.com/find/?q=${encodeURIComponent(query)}`;
    }

    function _posterNode(movie) {
        const wrap = el('div', 'detail-poster');
        const fallback = () => {
            wrap.textContent = '';
            const ph = el('div', 'poster-placeholder');
            ph.appendChild(el('span', 'ph-initial', _initialOf(movie.title)));
            wrap.appendChild(ph);
        };
        if (movie.poster_url) {
            const img = document.createElement('img');
            img.src = movie.poster_url;
            img.alt = _displayTitle(movie.title);
            img.decoding = 'async';
            img.referrerPolicy = 'no-referrer';
            img.addEventListener('error', fallback);
            wrap.appendChild(img);
        } else {
            fallback();
        }
        return wrap;
    }

    function _metaRow(movie) {
        const row = el('div', 'detail-meta');
        if (movie.release_year) {
            row.appendChild(el('span', 'detail-year', String(movie.release_year)));
        }
        if (movie.rating_count > 0) {
            const rating = el('span', 'detail-rating');
            rating.appendChild(el('span', 'detail-rating-star', '\u2605'));
            rating.appendChild(el('span', 'detail-rating-value', Number(movie.avg_rating).toFixed(1)));
            rating.appendChild(el('span', 'detail-rating-count',
                `${Number(movie.rating_count).toLocaleString()} ${CineRec.t('movies.ratings')}`));
            row.appendChild(rating);
        }
        return row;
    }

    function _genreTags(movie) {
        const wrap = el('div', 'detail-genres');
        _splitGenres(movie.genres).forEach(g => {
            wrap.appendChild(el('span', 'genre-tag', CineRec.genreName(g)));
        });
        return wrap;
    }

    function render(movie) {
        const body = document.getElementById('detail-body');
        if (!body) return;
        body.textContent = '';

        body.appendChild(_posterNode(movie));

        const main = el('div', 'detail-main');

        const title = el('h3', 'detail-title', _displayTitle(movie.title));
        title.id = 'detail-title';
        main.appendChild(title);

        main.appendChild(_metaRow(movie));
        main.appendChild(_genreTags(movie));

        const overview = el('p', 'detail-overview');
        if (movie.overview && movie.overview.trim()) {
            overview.textContent = movie.overview.trim();
        } else {
            overview.textContent = CineRec.t('detail.noOverview');
            overview.classList.add('muted');
        }
        main.appendChild(overview);

        const actions = el('div', 'detail-actions');
        const rateBtn = el('button', 'btn-primary btn-sm', CineRec.t('movies.rate'));
        rateBtn.type = 'button';
        rateBtn.addEventListener('click', () => {
            const id = movie.id;
            const name = _displayTitle(movie.title);
            close();
            if (typeof openRatingModal === 'function') openRatingModal(id, name);
        });
        actions.appendChild(rateBtn);

        const imdb = el('a', 'btn-outline btn-sm', CineRec.t('detail.imdb'));
        imdb.href = _imdbUrl(movie.title, movie.release_year);
        imdb.target = '_blank';
        imdb.rel = 'noopener noreferrer';
        actions.appendChild(imdb);

        main.appendChild(actions);
        body.appendChild(main);
    }

    function _showMessage(text) {
        const body = document.getElementById('detail-body');
        if (!body) return;
        body.textContent = '';
        body.appendChild(el('p', 'detail-message', text));
    }

    async function open(movieId) {
        const modal = document.getElementById('detail-modal');
        if (!modal) return;
        const id = Number(movieId);

        lastFocus = document.activeElement;
        currentId = id;
        current = null;
        modal.classList.remove('hidden');
        _showMessage(CineRec.t('common.loading'));

        const closeBtn = modal.querySelector('.modal-close');
        if (closeBtn) closeBtn.focus();

        try {
            const movie = await CineRec.api(`/api/movies/${id}`);
            if (currentId !== id) return; // a newer request superseded this one
            current = movie;
            render(movie);
        } catch (e) {
            if (currentId !== id) return;
            _showMessage(CineRec.t('common.error'));
        }
    }

    function close() {
        const modal = document.getElementById('detail-modal');
        if (!modal || modal.classList.contains('hidden')) return;
        modal.classList.add('hidden');
        currentId = null;
        current = null;
        if (lastFocus && typeof lastFocus.focus === 'function') lastFocus.focus();
        lastFocus = null;
    }

    document.addEventListener('DOMContentLoaded', () => {
        const modal = document.getElementById('detail-modal');
        if (!modal) return;

        modal.querySelectorAll('.modal-close').forEach(btn => {
            btn.addEventListener('click', close);
        });
        // Clicking the dimmed backdrop (but not the panel) closes.
        modal.addEventListener('click', (e) => {
            if (e.target === modal) close();
        });
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && !modal.classList.contains('hidden')) close();
        });

        // Re-draw an open modal in the newly selected language.
        CineRec.onLangChange(() => {
            if (current) render(current);
        });
    });

    return { open, close };
})();