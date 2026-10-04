/**
 * CineRec — SPA Router, State Management, i18n, and Theme
 */
const CineRec = (() => {
    // State
    const state = {
        currentPage: 'login',
        userId: null,
        username: null,
        token: null,
        lang: 'zh',
        theme: localStorage.getItem('cinerec-theme') || 'dark',
        apiBase: window.location.origin,
        movies: [],
        recommendations: [],
        currentAlgo: 'SVD',
        moviePage: 1,
    };

    // i18n
    let i18nData = { en: {}, zh: {} };

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

    function t(key) {
        const keys = key.split('.');
        let val = i18nData[state.lang];
        for (const k of keys) {
            val = val?.[k];
        }
        if (!val) {
            console.warn(`[i18n] Missing key: ${key}`);
            return key;
        }
        return val;
    }

    function applyI18n() {
        document.documentElement.lang = state.lang;
        document.querySelectorAll('[data-i18n]').forEach(el => {
            el.textContent = t(el.dataset.i18n);
        });
        document.querySelectorAll('[data-i18n-placeholder]').forEach(el => {
            el.placeholder = t(el.dataset.i18nPlaceholder);
        });
        document.getElementById('lang-label').textContent = state.lang.toUpperCase();
    }

    function toggleLang() {
        state.lang = state.lang === 'en' ? 'zh' : 'en';
        applyI18n();
        localStorage.setItem('cinerec-lang', state.lang);
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

        // Trigger page-specific animations
        if (page === 'recommend') loadRecommendations();
        if (page === 'dashboard') loadDashboard();
        if (page === 'movies' && state.movies.length === 0) loadMovies();

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

    // Init
    async function init() {
        // Check CDN dependencies
        const deps = ['gsap', 'Lenis', 'tsParticles', 'echarts'];
        const missing = deps.filter(d => typeof window[d] === 'undefined' && d !== 'Lenis');
        if (missing.length > 0) {
            console.warn('Missing CDN dependencies:', missing.join(', '));
        }

        // Restore user session
        const savedUser = localStorage.getItem('cinerec-user');
        if (savedUser) {
            try {
                const { userId, username, token } = JSON.parse(savedUser);
                setUser(userId, username, token);
                const savedLang = localStorage.getItem('cinerec-lang');
                if (savedLang) state.lang = savedLang;
                applyTheme();
                await loadI18n();
                applyI18n();
                if (typeof Animations !== 'undefined' && Animations.initDecryptedText) {
                    Animations.initDecryptedText();
                }
                navigateTo('recommend');
                return;
            } catch(e) {}
        }

        // Restore saved lang
        const savedLang = localStorage.getItem('cinerec-lang');
        if (savedLang) state.lang = savedLang;

        applyTheme();
        await loadI18n();
        applyI18n();
        if (typeof Animations !== 'undefined' && Animations.initDecryptedText) {
            Animations.initDecryptedText();
        }
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

        // Load genre filter
        try {
            const genresData = await api('/api/movies/genres');
            const select = document.getElementById('genre-filter');
            genresData.genres.forEach(g => {
                const opt = document.createElement('option');
                opt.value = g;
                opt.textContent = g;
                select.appendChild(opt);
            });
        } catch (e) {
            console.warn('Could not load genres');
        }

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

    return { state, t, navigateTo, setUser, api, init, applyI18n };
})();

document.addEventListener('DOMContentLoaded', () => CineRec.init());
