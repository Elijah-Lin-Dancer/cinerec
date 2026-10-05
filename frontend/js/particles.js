/**
 * CineRec — Canvas star field.
 *
 * Self-contained replacement for the tsParticles background: the CDN bundle
 * (v3.7.0) mounted a canvas but never sized or drew it, so the layer stayed
 * empty. This version has no third-party dependency, resizes with the
 * viewport, and drives one requestAnimationFrame loop.
 */
const Particles = (() => {
    const PALETTE = ['#d4a843', '#e8c36a', '#4a9eff', '#a78bfa', '#ffffff'];
    const LINK_DISTANCE = 120;
    const CURSOR_DISTANCE = 170;
    // Cap the backing-store scale: full-screen canvas at DPR 2 means a large
    // clear+draw every frame, which is the bulk of the animation cost on
    // low-power machines.
    const MAX_DPR = 1.5;

    let canvas = null;
    let ctx = null;
    let stars = [];
    let width = 0;
    let height = 0;
    let running = false;
    const pointer = { x: -9999, y: -9999, active: false };

    const rand = (min, max) => Math.random() * (max - min) + min;

    function seed() {
        const density = Math.round((width * height) / 14000);
        const count = Math.max(40, Math.min(90, density));
        stars = Array.from({ length: count }, () => ({
            x: rand(0, width),
            y: rand(0, height),
            r: rand(0.6, 2.4),
            a: rand(0.15, 0.7),
            tw: rand(0.4, 1.6),
            phase: rand(0, Math.PI * 2),
            vx: rand(-0.18, 0.18),
            vy: rand(-0.18, 0.18),
            color: PALETTE[(Math.random() * PALETTE.length) | 0],
        }));
    }

    function resize() {
        const rect = canvas.getBoundingClientRect();
        width = Math.max(1, rect.width);
        height = Math.max(1, rect.height);
        const dpr = Math.min(window.devicePixelRatio || 1, MAX_DPR);
        canvas.width = Math.round(width * dpr);
        canvas.height = Math.round(height * dpr);
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        seed();
    }

    function paint(time) {
        ctx.clearRect(0, 0, width, height);

        // Constellation links between nearby stars.
        ctx.lineWidth = 0.6;
        for (let i = 0; i < stars.length; i++) {
            const s = stars[i];
            for (let j = i + 1; j < stars.length; j++) {
                const o = stars[j];
                const dx = s.x - o.x;
                const dy = s.y - o.y;
                const d2 = dx * dx + dy * dy;
                if (d2 < LINK_DISTANCE * LINK_DISTANCE) {
                    const d = Math.sqrt(d2);
                    ctx.strokeStyle = `rgba(138,180,255,${(0.10 * (1 - d / LINK_DISTANCE)).toFixed(3)})`;
                    ctx.beginPath();
                    ctx.moveTo(s.x, s.y);
                    ctx.lineTo(o.x, o.y);
                    ctx.stroke();
                }
            }
        }

        // Cursor reveals the nearby stars it can "grab".
        if (pointer.active) {
            ctx.lineWidth = 0.8;
            for (const s of stars) {
                const dx = s.x - pointer.x;
                const dy = s.y - pointer.y;
                const d2 = dx * dx + dy * dy;
                if (d2 < CURSOR_DISTANCE * CURSOR_DISTANCE) {
                    const d = Math.sqrt(d2);
                    ctx.strokeStyle = `rgba(212,168,67,${(0.35 * (1 - d / CURSOR_DISTANCE)).toFixed(3)})`;
                    ctx.beginPath();
                    ctx.moveTo(s.x, s.y);
                    ctx.lineTo(pointer.x, pointer.y);
                    ctx.stroke();
                }
            }
        }

        // Stars drift slowly and twinkle.
        const t = time * 0.001;
        for (const s of stars) {
            const twinkle = 0.6 + 0.4 * Math.sin(t * s.tw + s.phase);
            ctx.globalAlpha = s.a * twinkle;
            ctx.fillStyle = s.color;
            ctx.beginPath();
            ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
            ctx.fill();

            s.x += s.vx;
            s.y += s.vy;
            if (s.x < -5) s.x = width + 5;
            else if (s.x > width + 5) s.x = -5;
            if (s.y < -5) s.y = height + 5;
            else if (s.y > height + 5) s.y = -5;
        }
        ctx.globalAlpha = 1;
    }

    function loop(time) {
        if (!running) return;
        paint(time);
        requestAnimationFrame(loop);
    }

    function paintOnce() {
        ctx.clearRect(0, 0, width, height);
        for (const s of stars) {
            ctx.globalAlpha = s.a;
            ctx.fillStyle = s.color;
            ctx.beginPath();
            ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
            ctx.fill();
        }
        ctx.globalAlpha = 1;
    }

    function init() {
        const host = document.getElementById('particles-bg');
        if (!host) return;
        if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;

        canvas = document.createElement('canvas');
        canvas.setAttribute('aria-hidden', 'true');
        host.appendChild(canvas);
        ctx = canvas.getContext('2d');
        if (!ctx) return;

        resize();
        window.addEventListener('resize', () => {
            resize();
            paintOnce();
        }, { passive: true });
        window.addEventListener('mousemove', (e) => {
            pointer.x = e.clientX;
            pointer.y = e.clientY;
            pointer.active = true;
        }, { passive: true });
        window.addEventListener('mouseout', () => { pointer.active = false; }, { passive: true });

        // Stop the loop entirely while the tab is hidden instead of relying on
        // the browser to throttle it.
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                running = false;
            } else if (!running) {
                running = true;
                requestAnimationFrame(loop);
            }
        });

        running = true;
        requestAnimationFrame(loop);
    }

    return { init };
})();

document.addEventListener('DOMContentLoaded', () => Particles.init());