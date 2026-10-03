/**
 * CineRec — Authentication Page Logic
 */
document.addEventListener('DOMContentLoaded', () => {
    // Tab switching
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.login-form').forEach(f => f.classList.remove('active'));
            btn.classList.add('active');
            document.getElementById(`form-${btn.dataset.tab}`).classList.add('active');
        });
    });

    // Sign in
    document.getElementById('form-signin').addEventListener('submit', async (e) => {
        e.preventDefault();
        const username = document.getElementById('login-username').value.trim();
        const password = document.getElementById('login-password').value.trim();
        if (!username || !password) return;

        try {
            const data = await CineRec.api('/api/auth/login', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ username, password })
            });
            CineRec.setUser(data.user_id, data.username);
            CineRec.navigateTo('recommend');
        } catch (err) {
            alert(CineRec.t('login.loginFailed'));
        }
    });

    // Sign up
    document.getElementById('form-signup').addEventListener('submit', async (e) => {
        e.preventDefault();
        const username = document.getElementById('reg-username').value.trim();
        const password = document.getElementById('reg-password').value.trim();
        if (!username || !password) return;

        try {
            const data = await CineRec.api('/api/auth/register', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ username, password })
            });
            CineRec.setUser(data.user_id, data.username);
            CineRec.navigateTo('recommend');
        } catch (err) {
            alert(CineRec.t('login.registerFailed'));
        }
    });

    // Guest login
    document.getElementById('btn-guest').addEventListener('click', async () => {
        try {
            const data = await CineRec.api('/api/auth/guest');
            CineRec.setUser(data.user_id, data.username);
            CineRec.navigateTo('recommend');
        } catch (err) {
            alert(CineRec.t('login.guestFailed'));
        }
    });
});
