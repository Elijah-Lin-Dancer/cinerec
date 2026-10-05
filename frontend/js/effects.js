/**
 * CineRec — Interaction Effects
 * Magnetic Button, 3D Tilt Cards, Spotlight Cards (event delegation)
 *
 * Performance: a single rAF-throttled pointer handler drives all three effects.
 * The previous version attached three separate document-level `mousemove`
 * listeners, each doing layout reads (getBoundingClientRect) on every mouse
 * event — the main source of the scroll/hover jank.
 */
const Effects = (() => {
    let _delegatedInitialized = false;

    function initDelegatedEffects() {
        if (_delegatedInitialized) return;
        _delegatedInitialized = true;

        const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
        const coarsePointer = window.matchMedia('(pointer: coarse)').matches;

        let pendingEvent = null;
        let frameScheduled = false;

        function applyEffects() {
            frameScheduled = false;
            const e = pendingEvent;
            if (!e || !e.target || !e.target.closest) return;

            const x = e.clientX;
            const y = e.clientY;

            // Magnetic buttons
            const btn = e.target.closest('.magnetic-btn');
            if (btn && !reduceMotion) {
                const rect = btn.getBoundingClientRect();
                btn.style.transform = `translate(${(x - rect.left - rect.width / 2) * 0.3}px, ${(y - rect.top - rect.height / 2) * 0.3}px)`;
            }

            // Tilt cards
            const tilt = e.target.closest('.tilt-card');
            if (tilt && !reduceMotion) {
                const rect = tilt.getBoundingClientRect();
                const rx = (x - rect.left) / rect.width;
                const ry = (y - rect.top) / rect.height;
                tilt.style.transform = `perspective(600px) rotateY(${(rx - 0.5) * 10}deg) rotateX(${-(ry - 0.5) * 10}deg)`;
            }

            // Spotlight cards
            const spot = e.target.closest('.movie-card, .rec-card');
            if (spot) {
                const rect = spot.getBoundingClientRect();
                spot.style.setProperty('--mouse-x', (x - rect.left) + 'px');
                spot.style.setProperty('--mouse-y', (y - rect.top) + 'px');
            }
        }

        if (!coarsePointer && !reduceMotion) {
            document.addEventListener('mousemove', (e) => {
                pendingEvent = e;
                if (!frameScheduled) {
                    frameScheduled = true;
                    requestAnimationFrame(applyEffects);
                }
            }, { passive: true });

            // Reset a card's transform once the pointer truly leaves it.
            document.addEventListener('mouseout', (e) => {
                const t = e.target;
                if (!t || !t.closest) return;
                const btn = t.closest('.magnetic-btn');
                if (btn && !btn.contains(e.relatedTarget)) btn.style.transform = '';
                const tilt = t.closest('.tilt-card');
                if (tilt && !tilt.contains(e.relatedTarget)) tilt.style.transform = '';
            }, true);
        }

        // Click spark
        if (!reduceMotion) {
            document.addEventListener('click', (e) => {
                const spark = document.createElement('div');
                spark.className = 'spark';
                spark.style.left = e.clientX + 'px';
                spark.style.top = e.clientY + 'px';

                for (let i = 0; i < 8; i++) {
                    const particle = document.createElement('div');
                    particle.className = 'spark-particle';
                    const angle = (Math.PI * 2 * i) / 8 + (Math.random() - 0.5) * 0.5;
                    const distance = 20 + Math.random() * 30;
                    particle.style.setProperty('--dx', Math.cos(angle) * distance + 'px');
                    particle.style.setProperty('--dy', Math.sin(angle) * distance + 'px');
                    spark.appendChild(particle);
                }

                document.body.appendChild(spark);
                setTimeout(() => spark.remove(), 700);
            });
        }
    }

    function init() {
        initDelegatedEffects();
    }

    // Re-init for dynamically added elements.
    // With event delegation, this is a no-op after the first call.
    function refresh() {
        if (!_delegatedInitialized) {
            initDelegatedEffects();
        }
    }

    return { init, refresh };
})();

document.addEventListener('DOMContentLoaded', () => Effects.init());