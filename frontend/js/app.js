/**
 * CineRec — SPA Router, State Management, i18n, and Theme
 */
const CineRec = (() => {
    // State
    const state = {
        // The movie library is the public home page; the login form is reached
        // explicitly from the nav link.
        currentPage: 'movies',
        userId: null,
        username: null,
        token: null,
        lang: localStorage.getItem('cinerec-lang') || 'zh',
        theme: localStorage.getItem('cinerec-theme') || 'dark',
        apiBase: window.location.origin,
        movies: [],
        moviesLoaded: false,
        moviesTotal: 0,
        moviesPages: 1,
        recommendations: [],
        currentAlgo: 'SVD',
        moviePage: 1,
        // Active movie-library filters, owned here so any module can read them.
        filters: { genre: '', yearFrom: 0, yearTo: 0, sort: 'id' },
    };

    // i18n
    let i18nData = { en: {}, zh: {} };
    const langListeners = [];

    async function loadI18n() {
        try {
            const [enRes, zhRes] = await Promise.all([
                fetch('/assets/i18n/en.json'),
                fetch('/assets/i18n/zh.json')
            ]);
            i18nData.en = await enRes.json();
            i18nData.zh = await zhRes.json();
        } catch (e) {
            console.warn('Failed to load i18n:', e);
        }
    }

    function t(key, vars) {
        const keys = key.split('.');
        let val = i18nData[state.lang];
        for (const k of keys) {
            val = val?.[k];
        }
        if (typeof val !== 'string') {
            console.warn(`[i18n] Missing key: ${key}`);
            return key;
        }
        if (vars) {
            for (const k of Object.keys(vars)) {
                val = val.replace(`{${k}}`, vars[k]);
            }
        }
        return val;
    }

    // Translate a single MovieLens genre label; unknown genres pass through.
    function genreName(genre) {
        const map = i18nData[state.lang]?.genres;
        return (map && map[genre]) || genre;
    }

    function applyI18n() {
        document.documentElement.lang = state.lang;
        document.querySelectorAll('[data-i18n]').forEach(el => {
            el.textContent = t(el.dataset.i18n);
        });
        document.querySelectorAll('[data-i18n-placeholder]').forEach(el => {
            el.placeholder = t(el.dataset.i18nPlaceholder);
        });
        const label = document.getElementById('lang-label');
        if (label) label.textContent = state.lang.toUpperCase();
    }

    // Modules register here so their dynamically rendered content can be
    // redrawn when the language changes (applyI18n only touches static markup).
    function onLangChange(fn) {
        if (typeof fn === 'function') langListeners.push(fn);
    }

    function toggleLang() {
        state.lang = state.lang === 'en' ? 'zh' : 'en';
        applyI18n();
        localStorage.setItem('cinerec-lang', state.lang);
        langListeners.forEach(fn => {
            try { fn(); } catch (e) { console.error('Lang listener failed:', e); }
        });
    }

    // Theme
    function applyTheme() {
        document.documentElement.setAttribute('data-theme', state.theme);
        const icon = document.getElementById('theme-icon');
        if (icon) icon.innerHTML = state.theme === 'light' ? '&#9789;' : '&#9788;';
    }

    function toggleTheme() {
        state.theme = state.theme === 'dark' ? 'light' : 'dark';
        applyTheme();
        localStorage.setItem('cinerec-theme', state.theme);
    }

    // Navigation
    function navigateTo(page) {
        state.currentPage = page;
        document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
        document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
        const pageEl = document.getElementById(`page-${page}`);
        const linkEl = document.querySelector(`.nav-link[data-page="${page}"]`);
        if (pageEl) pageEl.classList.add('active');
        if (linkEl) linkEl.classList.add('active');

        // Hide login link in nav when on login page, show when not
        const loginLink = document.querySelector('.nav-login-link');
        const userBadge = document.getElementById('user-badge');
        if (loginLink) loginLink.style.display = (page === 'login' || state.userId) ? 'none' : '';
        if (userBadge && state.userId) userBadge.style.display = '';

        // Trigger page-specific loading
        if (page === 'recommend') loadRecommendations();
        if (page === 'dashboard') loadDashboard();
        if (page === 'movies' && !state.moviesLoaded) loadMovies(state.moviePage);

        // Scroll to top
        window.scrollTo(0, 0);
    }

    // Auth state
    function setUser(userId, username, token) {
        state.userId = userId;
        state.username = username;
        state.token = token || state.token || null;
        localStorage.setItem('cinerec-user', JSON.stringify({ userId, username, token: state.token }));
        const badge = document.getElementById('user-badge');
        const nameEl = document.getElementById('user-name');
        const loginLink = document.querySelector('.nav-login-link');
        badge.classList.remove('hidden');
        badge.style.display = '';
        nameEl.textContent = username;
        if (loginLink) loginLink.style.display = 'none';
    }

    // API helper — attaches the session token when present.
    async function api(path, options = {}) {
        const url = `${state.apiBase}${path}`;
        const headers = { ...(options.headers || {}) };
        if (state.token) headers['Authorization'] = `Bearer ${state.token}`;
        try {
            const res = await fetch(url, { ...options, headers });
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            return await res.json();
        } catch (e) {
            console.error(`API Error [${path}]:`, e);
            throw e;
        }
    }

    // One-time UI wiring: navigation, language, theme, mobile menu.
    let uiBound = false;
    function bindUI() {
        if (uiBound) return;
        uiBound = true;

        document.querySelectorAll('.nav-link').forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                navigateTo(link.dataset.page);
            });
        });

        // Lang toggle
        document.getElementById('lang-toggle').addEventListener('click', toggleLang);

        // Theme toggle
        document.getElementById('theme-toggle').addEventListener('click', toggleTheme);

        // Build the movie-library filter chips (defined in movies.js).
        if (typeof buildMovieFilters === 'function') buildMovieFilters();

        // Hamburger menu (mobile)
        const hamburger = document.getElementById('hamburger');
        if (hamburger) {
            hamburger.addEventListener('click', () => {
                hamburger.classList.toggle('active');
                document.querySelector('.nav-links').classList.toggle('open');
            });
            // Close menu when a nav link is clicked
            document.querySelectorAll('.nav-link').forEach(link => {
                link.addEventListener('click', () => {
                    hamburger.classList.remove('active');
                    document.querySelector('.nav-links').classList.remove('open');
                });
            });
        }
    }

    // Init
    async function init() {
        // Check CDN dependencies
        const deps = ['gsap', 'Lenis', 'echarts'];
        const missing = deps.filter(d => typeof window[d] === 'undefined' && d !== 'Lenis');
        if (missing.length > 0) {
            console.warn('Missing CDN dependencies:', missing.join(', '));
        }

        applyTheme();
        await loadI18n();
        applyI18n();

        // Bind interactions before anything else can fail.
        bindUI();

        if (typeof Animations !== 'undefined' && Animations.initDecryptedText) {
            Animations.initDecryptedText();
        }

        // Restore a saved session, but keep the library as the landing page.
        const savedUser = localStorage.getItem('cinerec-user');
        if (savedUser) {
            try {
                const { userId, username, token } = JSON.parse(savedUser);
                setUser(userId, username, token);
            } catch (e) {
                localStorage.removeItem('cinerec-user');
            }
        }

        navigateTo('movies');
    }

    return { state, t, genreName, navigateTo, setUser, api, init, applyI18n, onLangChange };
})();

document.addEventListener('DOMContentLoaded', () => CineRec.init());