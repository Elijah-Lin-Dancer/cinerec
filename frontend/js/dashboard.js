/**
 * CineRec — Dashboard / Model Comparison Page Logic
 *
 * Reads the measured evaluation artefacts through the /api/eval endpoints.
 * The multiseed payloads (mean ± std over seeds 42–46) are preferred; the
 * single-seed fallback keeps the page working on older checkouts.
 */
let evalChartInstance = null;
let ablationChartInstance = null;
let crossChartInstance = null;
let coldChartInstance = null;
let dashboardResizeBound = false;

/** Servable ladder + the offline SOTA baselines, in chart order. */
const EVAL_MODELS = ['UserCF', 'ItemCF', 'SVD', 'BPRMF', 'NeuMF', 'LightGCN', 'VBPR', 'MultiModalNCF'];
const OFFLINE_BASELINES = ['BPRMF', 'VBPR'];

/** Read a metric as [mean, std] from either the multiseed or flat payloads. */
function mv(obj, metric) {
    const v = (obj || {})[metric];
    if (v && typeof v === 'object') return [v.mean || 0, v.std || 0];
    return [v || 0, 0];
}

/** "0.3650 ± 0.0077" — the spread is omitted for deterministic models (std 0). */
function fmtMS(obj, metric) {
    const [m, s] = mv(obj, metric);
    return s > 0 ? `${m.toFixed(4)} ± ${s.toFixed(4)}` : m.toFixed(4);
}

function animateNumber(el, target, finalText, duration = 800) {
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
        else if (finalText) el.textContent = finalText; // restore "mean ± std"
    }
    requestAnimationFrame(update);
}

function handleDashboardResize() {
    [evalChartInstance, ablationChartInstance, crossChartInstance, coldChartInstance]
        .forEach(c => { if (c) c.resize(); });
}

/** ECharts option fragments shared by the dark-cinema theme. */
function chartBase() {
    const light = CineRec.state.theme === 'light';
    return {
        theme: light ? null : 'dark',
        axisLabelColor: light ? '#6b7280' : '#8888a0',
        splitLineColor: light ? '#e5e7eb' : 'rgba(255,255,255,0.06)',
        tooltipBg: light ? '#fff' : '#1a1a2e',
        tooltipColor: light ? '#1a1a2e' : '#e8e8ed'
    };
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
    } catch (err) {
        console.error('Failed to load dashboard data:', err);
    }
    // Secondary studies load independently — a failure must not blank the page.
    CineRec.api('/api/eval/cross').then(renderCrossChart).catch(e => console.warn('cross:', e));
    CineRec.api('/api/eval/coldstart').then(renderColdChart).catch(e => console.warn('coldstart:', e));

    if (!dashboardResizeBound) {
        window.addEventListener('resize', handleDashboardResize);
        dashboardResizeBound = true;
    }
}

function renderEvalTable(results) {
    const container = document.getElementById('eval-table');
    if (!results || results.available === false) {
        container.textContent = results && results.note
            ? results.note
            : 'Evaluation results are unavailable.';
        return;
    }
    const models = EVAL_MODELS.filter(m => m in results);

    let html = `<table class="data-table">
        <thead><tr>
            <th>${CineRec.t('dash.model')}</th>
            <th>${CineRec.t('dash.hr')}</th><th>${CineRec.t('dash.ndcg')}</th><th>${CineRec.t('dash.recall')}</th>
            <th>${CineRec.t('dash.time')}</th>
        </tr></thead><tbody>`;

    models.forEach(model => {
        const data = results[model] || {};
        const isActive = model === 'MultiModalNCF';
        const tag = OFFLINE_BASELINES.includes(model)
            ? ` <span class="offline-tag">${CineRec.t('dash.baseline')}</span>` : '';
        const time = mv(data, 'train_time')[0];
        html += `<tr class="${isActive ? 'highlight-row' : ''}">
            <td><span class="model-name gradient-text ${isActive ? '' : ''}">${model}</span>${tag}</td>
            <td class="metric-cell" data-count="${mv(data, 'HR@10')[0]}" data-ms="${fmtMS(data, 'HR@10')}">${fmtMS(data, 'HR@10')}</td>
            <td class="metric-cell" data-count="${mv(data, 'NDCG@10')[0]}" data-ms="${fmtMS(data, 'NDCG@10')}">${fmtMS(data, 'NDCG@10')}</td>
            <td class="metric-cell" data-count="${mv(data, 'Recall@10')[0]}" data-ms="${fmtMS(data, 'Recall@10')}">${fmtMS(data, 'Recall@10')}</td>
            <td>${time ? time.toFixed(2) : '-'}s</td>
        </tr>`;
    });

    html += '</tbody></table>';
    if (results.multiseed) {
        html += `<p class="dash-note">${CineRec.t('dash.seedsNote')}</p>`;
    }
    container.innerHTML = html;

    // CountUp for metric cells — the animated number gives way to "mean ± std".
    document.querySelectorAll('.metric-cell[data-count]').forEach(el => {
        animateNumber(el, parseFloat(el.dataset.count), el.dataset.ms);
    });
}

/** i18n keys for the metric names used in chart legends. */
const METRIC_I18N = { 'HR@10': 'hr', 'NDCG@10': 'ndcg', 'Recall@10': 'recall' };

function renderEvalChart(results) {
    if (typeof echarts === 'undefined') return;
    if (!results || results.available === false) return;

    if (evalChartInstance) { evalChartInstance.dispose(); evalChartInstance = null; }

    const base = chartBase();
    evalChartInstance = echarts.init(document.getElementById('eval-chart'), base.theme);

    const models = EVAL_MODELS.filter(m => m in results);
    const metrics = ['HR@10', 'NDCG@10', 'Recall@10'];
    const colors = ['#d4a843', '#22d3ee', '#4ade80'];

    evalChartInstance.setOption({
        backgroundColor: 'transparent',
        tooltip: {
            trigger: 'axis', backgroundColor: base.tooltipBg, borderColor: '#d4a843',
            textStyle: { color: base.tooltipColor },
            formatter: params => params.map(p => {
                const [m, s] = mv(results[p.axisValue], p.seriesName);
                return `${p.marker}${p.seriesName}: ${m.toFixed(4)}${s > 0 ? ` ± ${s.toFixed(4)}` : ''}`;
            }).join('<br/>')
        },
        legend: { data: metrics.map(k => CineRec.t('dash.' + METRIC_I18N[k])),
                  textStyle: { color: base.axisLabelColor }, top: 5 },
        grid: { left: '3%', right: '4%', bottom: '3%', top: '15%', containLabel: true },
        xAxis: { type: 'category', data: models,
                 axisLabel: { color: base.axisLabelColor, rotate: 20, interval: 0, fontSize: 10 },
                 axisLine: { lineStyle: { color: base.splitLineColor } } },
        yAxis: { type: 'value', axisLabel: { color: base.axisLabelColor, formatter: v => v.toFixed(2) }, splitLine: { lineStyle: { color: base.splitLineColor } } },
        series: metrics.map((metric, i) => ({
            name: metric, type: 'bar', barGap: '10%',
            data: models.map(m => mv(results[m], metric)[0]),
            itemStyle: { color: colors[i], borderRadius: [4, 4, 0, 0] },
            animationDuration: 1500, animationDelay: i * 200, animationEasing: 'cubicOut'
        }))
    });
}

function renderAblationChart(ablation) {
    if (typeof echarts === 'undefined') return;
    if (!ablation || ablation.available === false) return;

    if (ablationChartInstance) { ablationChartInstance.dispose(); ablationChartInstance = null; }

    const base = chartBase();
    ablationChartInstance = echarts.init(document.getElementById('ablation-chart'), base.theme);

    const variants = Object.keys(ablation).filter(k => !k.startsWith('_') && typeof ablation[k] === 'object');
    const metrics = ['HR@10', 'NDCG@10'];

    ablationChartInstance.setOption({
        backgroundColor: 'transparent',
        tooltip: {
            trigger: 'axis', backgroundColor: base.tooltipBg, borderColor: '#d4a843',
            textStyle: { color: base.tooltipColor },
            formatter: params => params.map(p => {
                const [m, s] = mv(ablation[p.axisValue], p.seriesName);
                return `${p.marker}${p.seriesName}: ${m.toFixed(4)}${s > 0 ? ` ± ${s.toFixed(4)}` : ''}`;
            }).join('<br/>')
        },
        legend: { data: metrics.map(k => CineRec.t('dash.' + METRIC_I18N[k])),
                  textStyle: { color: base.axisLabelColor }, top: 5 },
        grid: { left: '3%', right: '4%', bottom: '3%', top: '15%', containLabel: true },
        xAxis: { type: 'category', data: variants, axisLabel: { color: base.axisLabelColor, rotate: 15 }, axisLine: { lineStyle: { color: base.splitLineColor } } },
        yAxis: { type: 'value', min: 0, axisLabel: { color: base.axisLabelColor, formatter: v => v.toFixed(2) }, splitLine: { lineStyle: { color: base.splitLineColor } } },
        series: metrics.map((metric, i) => ({
            name: metric, type: 'bar',
            data: variants.map(v => mv(ablation[v], metric)[0]),
            itemStyle: { color: i === 0 ? '#d4a843' : '#a78bfa', borderRadius: [4, 4, 0, 0] },
            animationDuration: 1500, animationDelay: i * 200, animationEasing: 'cubicOut'
        }))
    });
}

/**
 * RQ5: the same protocol re-run on Amazon Video_Games — side-by-side HR@10.
 * The ~2.8x larger catalogue depresses absolute values on Amazon; read the
 * orderings, not the levels.
 */
function renderCrossChart(cross) {
    if (typeof echarts === 'undefined') return;
    if (!cross || cross.available === false) return;
    const ml = cross.movielens && cross.movielens.eval;
    const amz = cross.amazon && cross.amazon.eval;
    if (!ml || ml.available === false || !amz || amz.available === false) return;

    if (crossChartInstance) { crossChartInstance.dispose(); crossChartInstance = null; }

    const base = chartBase();
    crossChartInstance = echarts.init(document.getElementById('cross-chart'), base.theme);

    const models = EVAL_MODELS.filter(m => m in ml && m in amz);
    const mlName = CineRec.t('dash.ml');
    const amzName = CineRec.t('dash.amz');

    crossChartInstance.setOption({
        backgroundColor: 'transparent',
        tooltip: {
            trigger: 'axis', backgroundColor: base.tooltipBg, borderColor: '#d4a843',
            textStyle: { color: base.tooltipColor },
            formatter: params => {
                const src = params[0].seriesIndex === 0 ? ml : amz;
                return `${params[0].axisValue}<br/>` + params.map(p => {
                    const [m, s] = mv(src[p.axisValue], 'HR@10');
                    return `${p.marker}${p.seriesName}: ${m.toFixed(4)}${s > 0 ? ` ± ${s.toFixed(4)}` : ''}`;
                }).join('<br/>');
            }
        },
        legend: { data: [mlName, amzName], textStyle: { color: base.axisLabelColor }, top: 5 },
        grid: { left: '3%', right: '4%', bottom: '3%', top: '15%', containLabel: true },
        xAxis: { type: 'category', data: models, axisLabel: { color: base.axisLabelColor, rotate: 15 }, axisLine: { lineStyle: { color: base.splitLineColor } } },
        yAxis: { type: 'value', axisLabel: { color: base.axisLabelColor, formatter: v => v.toFixed(2) }, splitLine: { lineStyle: { color: base.splitLineColor } } },
        series: [
            {
                name: mlName, type: 'bar',
                data: models.map(m => mv(ml[m], 'HR@10')[0]),
                itemStyle: { color: '#d4a843', borderRadius: [4, 4, 0, 0] },
                animationDuration: 1500, animationEasing: 'cubicOut'
            },
            {
                name: amzName, type: 'bar',
                data: models.map(m => mv(amz[m], 'HR@10')[0]),
                itemStyle: { color: '#22d3ee', borderRadius: [4, 4, 0, 0] },
                animationDuration: 1500, animationDelay: 200, animationEasing: 'cubicOut'
            }
        ]
    });
}

/** Cold-start study: the content tower vs the content-blind control vs random. */
function renderColdChart(cold) {
    if (typeof echarts === 'undefined') return;
    if (!cold || cold.available === false) return;
    const metrics = cold.metrics || cold;
    const opts = ['MultiModalNCF (content)', 'NeuMF (content-blind)', 'Random'].filter(k => k in metrics);
    if (!opts.length) return;

    if (coldChartInstance) { coldChartInstance.dispose(); coldChartInstance = null; }

    const base = chartBase();
    coldChartInstance = echarts.init(document.getElementById('cold-chart'), base.theme);

    coldChartInstance.setOption({
        backgroundColor: 'transparent',
        tooltip: {
            trigger: 'axis', backgroundColor: base.tooltipBg, borderColor: '#d4a843',
            textStyle: { color: base.tooltipColor },
            formatter: params => params.map(p => {
                const [m, s] = mv(metrics[p.axisValue], 'HR@10');
                return `${p.marker}${p.seriesName}: ${m.toFixed(4)}${s > 0 ? ` ± ${s.toFixed(4)}` : ''}`;
            }).join('<br/>')
        },
        grid: { left: '3%', right: '4%', bottom: '3%', top: '12%', containLabel: true },
        xAxis: { type: 'category', data: opts, axisLabel: { color: base.axisLabelColor, rotate: 15, width: 150, overflow: 'truncate' }, axisLine: { lineStyle: { color: base.splitLineColor } } },
        yAxis: { type: 'value', axisLabel: { color: base.axisLabelColor, formatter: v => v.toFixed(2) }, splitLine: { lineStyle: { color: base.splitLineColor } } },
        series: [{
            name: 'HR@10', type: 'bar',
            data: opts.map(k => mv(metrics[k], 'HR@10')[0]),
            itemStyle: {
                borderRadius: [4, 4, 0, 0],
                color: p => (opts[p.dataIndex] === 'Random' ? '#8b8ba6' : '#d4a843')
            },
            animationDuration: 1500, animationEasing: 'cubicOut'
        }]
    });
}

// Table headers and chart legends are built at render time, so redraw the
// dashboard when the language changes while it is the active page.
document.addEventListener('DOMContentLoaded', () => {
    CineRec.onLangChange(() => {
        if (CineRec.state.currentPage === 'dashboard') loadDashboard();
    });
});
