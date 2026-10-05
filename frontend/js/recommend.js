/**
 * CineRec — Recommendations Page Logic
 *
 * All server-provided strings are rendered via textContent / DOM APIs (never
 * innerHTML) so fields like title, poster_url and explanation text cannot
 * inject markup.
 */
let _lastRecAlgo = 'SVD';
let _lastRecFallback = false;

async function loadRecommendations() {
    if (!CineRec.state.userId) {
        CineRec.navigateTo('login');
        return;
    }

    const algo = CineRec.state.currentAlgo;
    const recList = document.getElementById('rec-list');
    recList.innerHTML = '<div class="loading-spinner"></div>';

    try {
        const data = await CineRec.api(
            `/api/recommend?algorithm=${encodeURIComponent(algo)}&top_k=10`
        );
        CineRec.state.recommendations = data.recommendations;
        _lastRecAlgo = algo;
        _lastRecFallback = !!data.fallback;
        renderRecommendations(data.recommendations, algo, data.fallback);
        if (typeof Animations !== 'undefined') Animations.animateRecCards();
    } catch (err) {
        const status = err && err.message ? err.message : '';
        recList.textContent = '';
        const p = document.createElement('p');
        if (status.includes('401') || status.includes('403')) {
            p.textContent = CineRec.t('rec.sessionExpired');
            CineRec.navigateTo('login');
            return;
        }
        p.textContent = status.includes('503') ? CineRec.t('rec.algoError') : CineRec.t('common.error');
        recList.appendChild(p);
    }
}

function _posterNode(rec) {
    const wrap = document.createElement('div');
    wrap.className = 'rec-poster';

    const placeholder = () => {
        const ph = document.createElement('div');
        ph.className = 'poster-placeholder';
        const span = document.createElement('span');
        span.className = 'ph-initial';
        span.textContent = _initialOf(rec.title);
        ph.appendChild(span);
        return ph;
    };

    if (rec.poster_url) {
        const img = document.createElement('img');
        img.src = rec.poster_url;
        img.alt = _displayTitle(rec.title);
        img.loading = 'lazy';
        img.decoding = 'async';
        img.referrerPolicy = 'no-referrer';
        img.addEventListener('error', () => {
            wrap.textContent = '';
            wrap.appendChild(placeholder());
        });
        wrap.appendChild(img);
    } else {
        wrap.appendChild(placeholder());
    }
    return wrap;
}

function _reasonNode(reason) {
    const tag = document.createElement('div');
    tag.className = `reason-tag ${reason.type || ''}`;

    const icon = document.createElement('span');
    icon.className = 'reason-icon';
    icon.textContent = reason.type === 'collaborative' ? '👥'
        : reason.type === 'content' ? '🎬'
        : reason.type === 'genre_match' ? '🏷️' : '✨';

    const text = document.createElement('span');
    text.className = 'reason-text';
    text.textContent = (CineRec.state.lang === 'zh'
        ? (reason.reason_zh || reason.reason)
        : (reason.reason_en || reason.reason)) || '';

    tag.appendChild(icon);
    tag.appendChild(text);
    return tag;
}

function renderRecommendations(recs, algo, isFallback) {
    const recList = document.getElementById('rec-list');
    recList.textContent = '';

    if (!recs || !recs.length) {
        const p = document.createElement('p');
        p.className = 'empty-state';
        p.textContent = CineRec.t('rec.noRec');
        recList.appendChild(p);
        return;
    }

    if (isFallback) {
        const note = document.createElement('p');
        note.className = 'rec-note empty-state';
        note.textContent = CineRec.t('rec.fallbackNote');
        recList.appendChild(note);
    }

    recs.forEach((rec, idx) => {
        const card = document.createElement('div');
        card.className = 'rec-card spotlight-card glass-card';
        card.style.setProperty('--delay', `${idx * 0.05}s`);

        const rank = document.createElement('div');
        rank.className = 'rec-rank';
        rank.textContent = String(idx + 1);
        card.appendChild(rank);

        card.appendChild(_posterNode(rec));

        const info = document.createElement('div');
        info.className = 'rec-info';

        const title = document.createElement('h3');
        title.className = 'rec-title';
        title.textContent = _displayTitle(rec.title);
        title.title = _displayTitle(rec.title);
        info.appendChild(title);

        const meta = document.createElement('div');
        meta.className = 'rec-meta';
        if (rec.release_year) {
            const year = document.createElement('span');
            year.textContent = String(rec.release_year);
            meta.appendChild(year);
        }
        const genreNames = _splitGenres(rec.genres).slice(0, 3).map(g => CineRec.genreName(g));
        if (genreNames.length) {
            const genres = document.createElement('span');
            genres.className = 'rec-genres';
            genres.textContent = genreNames.join(' · ');
            meta.appendChild(genres);
        }
        info.appendChild(meta);

        const score = document.createElement('div');
        score.className = 'rec-score';
        const scoreLabel = document.createElement('span');
        scoreLabel.className = 'score-label';
        scoreLabel.textContent = `${CineRec.t('rec.score')}:`;
        const scoreValue = document.createElement('span');
        scoreValue.className = 'score-value';
        scoreValue.dataset.count = rec.score;
        scoreValue.textContent = Number(rec.score).toFixed(4);
        score.appendChild(scoreLabel);
        score.appendChild(scoreValue);
        info.appendChild(score);

        const reasons = document.createElement('div');
        reasons.className = 'rec-reasons';
        (rec.reasons || []).forEach(r => reasons.appendChild(_reasonNode(r)));
        info.appendChild(reasons);

        card.appendChild(info);
        recList.appendChild(card);
    });

    document.querySelectorAll('.score-value[data-count]').forEach(el => {
        if (typeof Animations !== 'undefined') Animations.countUp(el, parseFloat(el.dataset.count));
        else el.textContent = parseFloat(el.dataset.count).toFixed(4);
    });

    if (typeof Effects !== 'undefined') Effects.refresh();
}

document.addEventListener('DOMContentLoaded', () => {
    // Algorithm switcher
    document.querySelectorAll('.algo-chip').forEach(chip => {
        chip.addEventListener('click', () => {
            document.querySelectorAll('.algo-chip').forEach(c => c.classList.remove('active'));
            chip.classList.add('active');
            CineRec.state.currentAlgo = chip.dataset.algo;
            loadRecommendations();
        });
    });

    // Refresh button
    document.getElementById('btn-refresh-rec').addEventListener('click', loadRecommendations);

    // Redraw cards (titles fall back to the original, genres/UI translate).
    CineRec.onLangChange(() => {
        if (CineRec.state.recommendations && CineRec.state.recommendations.length) {
            renderRecommendations(CineRec.state.recommendations, _lastRecAlgo, _lastRecFallback);
        }
    });
});