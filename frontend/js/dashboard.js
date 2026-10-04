/**
 * CineRec — Dashboard / Model Comparison Page Logic
 */
let evalChartInstance = null;
let ablationChartInstance = null;
let dashboardResizeBound = false;

function animateNumber(el, target, duration = 800) {
    const start = 0;
    const startTime = performance.now();

    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        // easeOutExpo
        const eased = progress === 1 ? 1 : 1 - Math.pow(2, -10 * progress);
        const current = start + (target - start) * eased;
        el.textContent = current.toFixed(4);
        if (progress < 1) requestAnimationFrame(update);
    }
    requestAnimationFrame(update);
}

function handleDashboardResize() {
    if (evalChartInstance) evalChartInstance.resize();
    if (ablationChartInstance) ablationChartInstance.resize();
}

async function loadDashboard() {
    try {
        const [results, ablation] = await Promise.all([
            CineRec.api('/api/eval/results'),
            CineRec.api('/api/eval/ablation')
        ]);
        renderEvalTable(results);
        renderEvalChart(results);
        renderAblationChart(ablation);

        if (!dashboardResizeBound) {
            window.addEventListener('resize', handleDashboardResize);
            dashboardResizeBound = true;
        }
    } catch (err) {
        console.error('Failed to load dashboard data:', err);
    }
}

function renderEvalTable(results) {
    const container = document.getElementById('eval-table');
    const models = ['UserCF', 'ItemCF', 'SVD', 'NeuMF', 'LightGCN', 'MultiModalNCF'];

    if (!results || results.available === false) {
        container.textContent = results && results.note
            ? results.note
            : 'Evaluation results are unavailable.';
        return;
    }

    let html = `<table class="data-table">
        <thead><tr>
            <th>${CineRec.t('dash.model')}</th>
            <th>${CineRec.t('dash.hr')}</th><th>${CineRec.t('dash.ndcg')}</th><th>${CineRec.t('dash.recall')}</th>
            <th>${CineRec.t('dash.time')}</th>
        </tr></thead><tbody>`;

    models.forEach(model => {
        const data = results[model] || {};
        const isActive = model === 'MultiModalNCF';
        html += `<tr class="${isActive ? 'highlight-row' : ''}">
            <td><span class="model-name gradient-text ${isActive ? '' : ''}">${model}</span></td>
            <td class="metric-cell" data-count="${data['HR@10'] || 0}">${(data['HR@10'] || 0).toFixed(4)}</td>
            <td class="metric-cell" data-count="${data['NDCG@10'] || 0}">${(data['NDCG@10'] || 0).toFixed(4)}</td>
            <td class="metric-cell" data-count="${data['Recall@10'] || 0}">${(data['Recall@10'] || 0).toFixed(4)}</td>
            <td>${data.train_time || '-'}s</td>
        </tr>`;
    });

    html += '</tbody></table>';
    container.innerHTML = html;

    // CountUp for metric cells
    document.querySelectorAll('.metric-cell[data-count]').forEach(el => {
        animateNumber(el, parseFloat(el.dataset.count));
    });
}

function renderEvalChart(results) {
    if (typeof echarts === 'undefined') return;
    if (!results || results.available === false) return;

    if (evalChartInstance) { evalChartInstance.dispose(); evalChartInstance = null; }

    const chartTheme = CineRec.state.theme === 'light' ? null : 'dark';
    const axisLabelColor = CineRec.state.theme === 'light' ? '#6b7280' : '#8888a0';
    const splitLineColor = CineRec.state.theme === 'light' ? '#e5e7eb' : 'rgba(255,255,255,0.06)';
    const tooltipBg = CineRec.state.theme === 'light' ? '#fff' : '#1a1a2e';
    const tooltipColor = CineRec.state.theme === 'light' ? '#1a1a2e' : '#e8e8ed';

    evalChartInstance = echarts.init(document.getElementById('eval-chart'), chartTheme);

    const models = ['UserCF', 'ItemCF', 'SVD', 'NeuMF', 'LightGCN', 'MultiModalNCF'];

    evalChartInstance.setOption({
        backgroundColor: 'transparent',
        tooltip: { trigger: 'axis', backgroundColor: tooltipBg, borderColor: '#d4a843', textStyle: { color: tooltipColor } },
        legend: { data: [CineRec.t('dash.hr'), CineRec.t('dash.ndcg'), CineRec.t('dash.recall')], textStyle: { color: axisLabelColor }, top: 5 },
        grid: { left: '3%', right: '4%', bottom: '3%', top: '15%', containLabel: true },
        xAxis: { type: 'category', data: models, axisLabel: { color: axisLabelColor }, axisLine: { lineStyle: { color: splitLineColor } } },
        yAxis: { type: 'value', axisLabel: { color: axisLabelColor, formatter: v => v.toFixed(2) }, splitLine: { lineStyle: { color: splitLineColor } } },
        series: [
            {
                name: CineRec.t('dash.hr'), type: 'bar', barGap: '10%',
                data: models.map(m => results[m]?.['HR@10'] || 0),
                itemStyle: { color: '#d4a843', borderRadius: [4, 4, 0, 0] },
                animationDuration: 1500, animationEasing: 'cubicOut'
            },
            {
                name: CineRec.t('dash.ndcg'), type: 'bar',
                data: models.map(m => results[m]?.['NDCG@10'] || 0),
                itemStyle: { color: '#22d3ee', borderRadius: [4, 4, 0, 0] },
                animationDuration: 1500, animationDelay: 200, animationEasing: 'cubicOut'
            },
            {
                name: CineRec.t('dash.recall'), type: 'bar',
                data: models.map(m => results[m]?.['Recall@10'] || 0),
                itemStyle: { color: '#4ade80', borderRadius: [4, 4, 0, 0] },
                animationDuration: 1500, animationDelay: 400, animationEasing: 'cubicOut'
            }
        ]
    });
}

function renderAblationChart(ablation) {
    if (typeof echarts === 'undefined') return;
    if (!ablation || ablation.available === false) return;

    if (ablationChartInstance) { ablationChartInstance.dispose(); ablationChartInstance = null; }

    const chartTheme = CineRec.state.theme === 'light' ? null : 'dark';
    const axisLabelColor = CineRec.state.theme === 'light' ? '#6b7280' : '#8888a0';
    const splitLineColor = CineRec.state.theme === 'light' ? '#e5e7eb' : 'rgba(255,255,255,0.06)';
    const tooltipBg = CineRec.state.theme === 'light' ? '#fff' : '#1a1a2e';
    const tooltipColor = CineRec.state.theme === 'light' ? '#1a1a2e' : '#e8e8ed';

    ablationChartInstance = echarts.init(document.getElementById('ablation-chart'), chartTheme);

    const variants = Object.keys(ablation).filter(k => !k.startsWith('_'));
    const hrData = variants.map(v => ablation[v]?.['HR@10'] || 0);
    const ndcgData = variants.map(v => ablation[v]?.['NDCG@10'] || 0);

    ablationChartInstance.setOption({
        backgroundColor: 'transparent',
        tooltip: { trigger: 'axis', backgroundColor: tooltipBg, borderColor: '#d4a843', textStyle: { color: tooltipColor } },
        legend: { data: [CineRec.t('dash.hr'), CineRec.t('dash.ndcg')], textStyle: { color: axisLabelColor }, top: 5 },
        grid: { left: '3%', right: '4%', bottom: '3%', top: '15%', containLabel: true },
        xAxis: { type: 'category', data: variants, axisLabel: { color: axisLabelColor, rotate: 15 }, axisLine: { lineStyle: { color: splitLineColor } } },
        yAxis: { type: 'value', min: 0, axisLabel: { color: axisLabelColor, formatter: v => v.toFixed(2) }, splitLine: { lineStyle: { color: splitLineColor } } },
        series: [
            {
                name: CineRec.t('dash.hr'), type: 'bar',
                data: hrData,
                itemStyle: { color: '#d4a843', borderRadius: [4, 4, 0, 0] },
                animationDuration: 1500, animationEasing: 'cubicOut'
            },
            {
                name: CineRec.t('dash.ndcg'), type: 'bar',
                data: ndcgData,
                itemStyle: { color: '#a78bfa', borderRadius: [4, 4, 0, 0] },
                animationDuration: 1500, animationDelay: 200, animationEasing: 'cubicOut'
            }
        ]
    });
}
