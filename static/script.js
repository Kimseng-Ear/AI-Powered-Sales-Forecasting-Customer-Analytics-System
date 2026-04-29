/**
 * AI-Powered Sales Forecasting & Customer Analytics System
 * Main Frontend JavaScript
 */

/* ════════════════════════════════════════════════════════════
   GLOBAL CONFIG
   ════════════════════════════════════════════════════════════ */
const CHART_COLORS = {
  primary   : '#6C63FF',
  secondary : '#FF6584',
  accent    : '#43B89C',
  warning   : '#FDCB6E',
  info      : '#4A90D9',
  danger    : '#E17055',
  purple    : '#A29BFE',
  teal      : '#00CEC9',
};

const PALETTE = Object.values(CHART_COLORS);

Chart.defaults.color = '#a0a8bb';
Chart.defaults.borderColor = 'rgba(255,255,255,0.06)';
Chart.defaults.font.family = "'Inter', 'Segoe UI', system-ui, sans-serif";

/* ════════════════════════════════════════════════════════════
   UTILITY HELPERS
   ════════════════════════════════════════════════════════════ */
const $ = (sel, ctx = document) => ctx.querySelector(sel);
const $$ = (sel, ctx = document) => [...ctx.querySelectorAll(sel)];

function fmtCurrency(val, decimals = 0) {
  if (val === null || val === undefined || isNaN(val)) return '$0';
  return '$' + Number(val).toLocaleString('en-US', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
}

function fmtNumber(val) {
  if (val === null || val === undefined || isNaN(val)) return '0';
  return Number(val).toLocaleString('en-US');
}

function fmtPercent(val, decimals = 1) {
  return Number(val).toFixed(decimals) + '%';
}

/* Animated counter */
function animateCounter(el, target, duration = 1200, prefix = '', suffix = '') {
  const start     = 0;
  const startTime = performance.now();
  const isDecimal = String(target).includes('.');

  function update(now) {
    const elapsed  = now - startTime;
    const progress = Math.min(elapsed / duration, 1);
    const eased    = 1 - Math.pow(1 - progress, 3); // ease-out cubic
    const current  = start + (target - start) * eased;
    el.textContent = prefix + (isDecimal
      ? current.toFixed(2)
      : Math.floor(current).toLocaleString()) + suffix;
    if (progress < 1) requestAnimationFrame(update);
  }
  requestAnimationFrame(update);
}

/* Show / hide loading overlay */
function showLoading() {
  const overlay = $('#loadingOverlay');
  if (overlay) overlay.style.display = 'flex';
}

function hideLoading() {
  const overlay = $('#loadingOverlay');
  if (overlay) overlay.style.display = 'none';
}

/* Toast notification */
function showToast(message, type = 'success') {
  let container = $('#toastContainer');
  if (!container) {
    container = document.createElement('div');
    container.id = 'toastContainer';
    container.style.cssText = `
      position:fixed; top:20px; right:20px; z-index:9999;
      display:flex; flex-direction:column; gap:8px;`;
    document.body.appendChild(container);
  }

  const colors = {
    success: '#43B89C',
    error  : '#FF6584',
    info   : '#4A90D9',
    warning: '#FDCB6E',
  };

  const toast = document.createElement('div');
  toast.style.cssText = `
    background:rgba(15,52,96,0.95); border:1px solid ${colors[type]||colors.info};
    border-left:4px solid ${colors[type]||colors.info};
    border-radius:10px; padding:12px 18px;
    color:#fff; font-size:0.875rem; font-weight:500;
    max-width:320px; backdrop-filter:blur(10px);
    animation: fadeInUp 0.3s ease;
    box-shadow: 0 8px 32px rgba(0,0,0,0.35);`;
  toast.textContent = message;
  container.appendChild(toast);
  setTimeout(() => toast.remove(), 4000);
}

/* ════════════════════════════════════════════════════════════
   CHART BUILDERS (Chart.js wrappers)
   ════════════════════════════════════════════════════════════ */
function makeLineChart(canvasId, labels, datasets, options = {}) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;
  return new Chart(ctx, {
    type: 'line',
    data: { labels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: 'top' },
        tooltip: { mode: 'index', intersect: false },
        ...options.plugins,
      },
      scales: {
        x: { grid: { color: 'rgba(255,255,255,0.04)' } },
        y: { grid: { color: 'rgba(255,255,255,0.06)' },
             ticks: { callback: v => fmtCurrency(v) } },
        ...options.scales,
      },
      interaction: { mode: 'index', intersect: false },
      ...options,
    },
  });
}

function makeBarChart(canvasId, labels, data, colors, options = {}) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;
  return new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: options.label || 'Value',
        data,
        backgroundColor: Array.isArray(colors) ? colors : colors,
        borderRadius: 6,
        borderSkipped: false,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: ctx => ' ' + fmtCurrency(ctx.raw) } },
        ...options.plugins,
      },
      scales: {
        x: { grid: { display: false } },
        y: { grid: { color: 'rgba(255,255,255,0.06)' },
             ticks: { callback: v => fmtCurrency(v) } },
        ...options.scales,
      },
      ...options,
    },
  });
}

function makeDoughnutChart(canvasId, labels, data, colors) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;
  return new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels,
      datasets: [{ data, backgroundColor: colors, borderWidth: 2,
                   borderColor: 'rgba(26,26,46,0.8)', hoverOffset: 8 }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: '68%',
      plugins: {
        legend: {
          position: 'right',
          labels: { boxWidth: 12, padding: 16, font: { size: 12 } },
        },
        tooltip: {
          callbacks: {
            label: ctx => ` ${ctx.label}: ${fmtNumber(ctx.raw)} (${
              fmtPercent(ctx.raw / ctx.dataset.data.reduce((a, b) => a + b, 0) * 100)
            })`,
          },
        },
      },
    },
  });
}

function makeHorizontalBarChart(canvasId, labels, data, colors) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;
  return new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{ data, backgroundColor: colors, borderRadius: 4, borderSkipped: false }],
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: ctx => ' ' + fmtCurrency(ctx.raw) } },
      },
      scales: {
        x: { grid: { color: 'rgba(255,255,255,0.06)' },
             ticks: { callback: v => fmtCurrency(v) } },
        y: { grid: { display: false } },
      },
    },
  });
}

/* ════════════════════════════════════════════════════════════
   DASHBOARD PAGE
   ════════════════════════════════════════════════════════════ */
function initDashboard() {
  if (!document.getElementById('dashboardPage')) return;

  // Animate KPI counters
  $$('[data-counter]').forEach(el => {
    const val  = parseFloat(el.dataset.counter) || 0;
    const pre  = el.dataset.prefix  || '';
    const suf  = el.dataset.suffix  || '';
    animateCounter(el, val, 1400, pre, suf);
  });

  // Monthly Trend Chart
  try {
    const monthlyEl = document.getElementById('monthlyTrendChart');
    if (monthlyEl && window.MONTHLY_DATA) {
      const labels   = MONTHLY_DATA.map(d => d.YearMonth);
      const revenues = MONTHLY_DATA.map(d => d.Revenue);
      makeLineChart('monthlyTrendChart', labels, [{
        label          : 'Monthly Revenue',
        data           : revenues,
        borderColor    : CHART_COLORS.primary,
        backgroundColor: 'rgba(108,99,255,0.12)',
        fill           : true,
        tension        : 0.4,
        pointRadius    : 3,
        pointHoverRadius: 6,
      }]);
    }
  } catch(e) { console.warn('Monthly chart error:', e); }

  // Category Revenue Pie
  try {
    if (window.CAT_DATA) {
      const labels = CAT_DATA.map(d => d.Category);
      const values = CAT_DATA.map(d => d.Revenue);
      makeDoughnutChart('categoryChart', labels, values, PALETTE);
    }
  } catch(e) {}

  // Top Products
  try {
    if (window.TOP_PRODUCTS) {
      const labels = TOP_PRODUCTS.map(d => d.Product);
      const values = TOP_PRODUCTS.map(d => d.Revenue);
      makeHorizontalBarChart('topProductsChart', labels, values,
        PALETTE.slice(0, labels.length));
    }
  } catch(e) {}

  // Payment Methods
  try {
    if (window.PAYMENT_DATA) {
      const labels = PAYMENT_DATA.map(d => d.Method);
      const values = PAYMENT_DATA.map(d => d.Count);
      makeDoughnutChart('paymentChart', labels, values, PALETTE);
    }
  } catch(e) {}

  // RFM Segments
  try {
    if (window.RFM_DATA) {
      const labels = RFM_DATA.map(d => d.Segment);
      const values = RFM_DATA.map(d => d.Count);
      makeDoughnutChart('rfmChart', labels, values, PALETTE);
    }
  } catch(e) {}

  // Churn Chart
  try {
    if (window.CHURN_DATA) {
      const labels = CHURN_DATA.map(d => d.Risk);
      const values = CHURN_DATA.map(d => d.Count);
      const churnColors = ['#43B89C', '#FDCB6E', '#F7C59F', '#FF6584'];
      makeDoughnutChart('churnChart', labels, values, churnColors);
    }
  } catch(e) {}

  // Date range filter
  const filterBtn = document.getElementById('applyFilters');
  if (filterBtn) {
    filterBtn.addEventListener('click', () => {
      showToast('Filter functionality applied', 'info');
    });
  }
}

/* ════════════════════════════════════════════════════════════
   PREDICTION PAGE
   ════════════════════════════════════════════════════════════ */
function initPrediction() {
  if (!document.getElementById('predictionPage')) return;

  const form        = document.getElementById('predictionForm');
  const resultDiv   = document.getElementById('predictionResult');
  const demandForm  = document.getElementById('demandForm');
  const demandResult= document.getElementById('demandResult');

  // Sales Prediction
  if (form) {
    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      const btn = form.querySelector('[type=submit]');
      btn.disabled = true;
      btn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Predicting…';

      try {
        const payload = {
          quantity       : parseFloat($('#qty', form).value),
          unit_price     : parseFloat($('#unitPrice', form).value),
          product        : $('#product', form).value,
          category       : $('#category', form).value,
          payment_method : $('#paymentMethod', form).value,
          review_rating  : parseFloat($('#rating', form).value || 3),
          month          : parseInt($('#month', form).value),
          year           : parseInt($('#year', form).value),
          weekday        : parseInt($('#weekday', form).value || 0),
          purchase_frequency : parseFloat($('#freq', form).value || 5),
          average_order_value: parseFloat($('#avgOrder', form).value || 0),
        };

        const res  = await fetch('/predict-sales', {
          method : 'POST',
          headers: { 'Content-Type': 'application/json' },
          body   : JSON.stringify(payload),
        });
        const data = await res.json();

        if (data.status === 'success') {
          resultDiv.style.display = 'block';
          resultDiv.classList.add('animate-fade-up');

          const revEl = document.getElementById('predRevenue');
          if (revEl) animateCounter(revEl, data.predicted_revenue, 1200, '$');

          const hvEl = document.getElementById('predHighValue');
          if (hvEl) {
            hvEl.textContent = data.is_high_value ? '⭐ High-Value Transaction' : 'Standard Transaction';
            hvEl.className   = 'badge ' + (data.is_high_value ? 'badge-success' : 'badge-info') +
                               ' badge-custom fs-6 p-2';
          }

          const probEl = document.getElementById('predProbBar');
          if (probEl) probEl.style.width = data.high_value_probability + '%';

          const probText = document.getElementById('predProbText');
          if (probText) probText.textContent = fmtPercent(data.high_value_probability) + ' high-value probability';

          showToast('Prediction completed successfully!', 'success');
        } else {
          showToast('Prediction error: ' + data.message, 'error');
        }
      } catch(err) {
        showToast('Network error: ' + err.message, 'error');
      } finally {
        btn.disabled = false;
        btn.innerHTML = 'Predict Revenue';
      }
    });
  }

  // Demand Prediction
  if (demandForm) {
    demandForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      const btn = demandForm.querySelector('[type=submit]');
      btn.disabled = true;
      btn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Predicting…';

      try {
        const payload = {
          product : $('#demandProduct', demandForm).value,
          month   : parseInt($('#demandMonth', demandForm).value),
          year    : parseInt($('#demandYear', demandForm).value),
        };

        const res  = await fetch('/predict-demand', {
          method : 'POST',
          headers: { 'Content-Type': 'application/json' },
          body   : JSON.stringify(payload),
        });
        const data = await res.json();

        if (data.status === 'success') {
          demandResult.style.display = 'block';
          demandResult.classList.add('animate-fade-up');

          const qtyEl = document.getElementById('demandQty');
          if (qtyEl) animateCounter(qtyEl, data.predicted_quantity, 1000);

          const revEl2 = document.getElementById('demandRevenue');
          if (revEl2) animateCounter(revEl2, data.predicted_revenue, 1200, '$');

          // Draw mini history chart
          if (data.monthly_avg_history && data.monthly_avg_history.length) {
            const months = ['Jan','Feb','Mar','Apr','May','Jun',
                            'Jul','Aug','Sep','Oct','Nov','Dec'];
            makeBarChart('demandHistoryChart',
              data.monthly_avg_history.map(d => months[(d.Month||1) - 1]),
              data.monthly_avg_history.map(d => d.Quantity),
              PALETTE[0], { label: 'Avg Quantity' });
          }

          showToast('Demand forecast ready!', 'success');
        } else {
          showToast('Error: ' + data.message, 'error');
        }
      } catch(err) {
        showToast('Network error: ' + err.message, 'error');
      } finally {
        btn.disabled = false;
        btn.innerHTML = 'Forecast Demand';
      }
    });
  }

  // Unit price auto-compute average order
  const qtyEl   = document.getElementById('qty');
  const priceEl = document.getElementById('unitPrice');
  const avgEl   = document.getElementById('avgOrder');
  function updateAvg() {
    if (qtyEl && priceEl && avgEl) {
      const val = (parseFloat(qtyEl.value)||0) * (parseFloat(priceEl.value)||0);
      avgEl.value = val.toFixed(2);
    }
  }
  if (qtyEl)   qtyEl.addEventListener('input', updateAvg);
  if (priceEl) priceEl.addEventListener('input', updateAvg);
}

/* ════════════════════════════════════════════════════════════
   CUSTOMER SEARCH (dashboard / analytics page)
   ════════════════════════════════════════════════════════════ */
let custSearchTimer;
function initCustomerSearch() {
  const searchInput = document.getElementById('customerSearch');
  const tableBody   = document.getElementById('customerTableBody');
  if (!searchInput || !tableBody) return;

  async function loadCustomers(query = '') {
    try {
      const url = `/customer-insights?search=${encodeURIComponent(query)}&limit=15`;
      const res = await fetch(url);
      const data = await res.json();
      if (data.status === 'success') renderCustomerTable(data.data);
    } catch(e) { console.error(e); }
  }

  function renderCustomerTable(customers) {
    tableBody.innerHTML = customers.map((c, i) => `
      <tr>
        <td><span class="text-muted-custom">${i+1}</span></td>
        <td><strong>${c.CustomerName || c.CustomerID}</strong><br>
            <small class="text-muted-custom">${c.CustomerID}</small></td>
        <td><strong class="text-primary-custom">${fmtCurrency(c.TotalRevenue)}</strong></td>
        <td>${fmtNumber(c.TotalOrders)}</td>
        <td>${fmtCurrency(c.AvgOrderValue)}</td>
        <td>${renderStars(c.AvgRating)}</td>
        <td><small>${c.LastPurchase ? c.LastPurchase.substring(0,10) : 'N/A'}</small></td>
        <td>${getValueBadge(c.TotalRevenue)}</td>
      </tr>`).join('');
  }

  function renderStars(rating) {
    const r = Math.round(rating || 0);
    return '⭐'.repeat(Math.min(r, 5)) + '<small class="text-muted-custom ms-1">' +
           (rating ? rating.toFixed(1) : '–') + '</small>';
  }

  function getValueBadge(revenue) {
    if (revenue > 5000) return '<span class="badge-custom badge-primary">VIP</span>';
    if (revenue > 2000) return '<span class="badge-custom badge-success">High</span>';
    if (revenue > 500)  return '<span class="badge-custom badge-warning">Mid</span>';
    return '<span class="badge-custom badge-info">Low</span>';
  }

  searchInput.addEventListener('input', () => {
    clearTimeout(custSearchTimer);
    custSearchTimer = setTimeout(() => loadCustomers(searchInput.value), 350);
  });

  loadCustomers();
}

/* ════════════════════════════════════════════════════════════
   MODEL METRICS TABLE
   ════════════════════════════════════════════════════════════ */
function initModelMetrics() {
  const container = document.getElementById('modelMetricsContainer');
  if (!container || !window.MODEL_RESULTS) return;

  const reg = MODEL_RESULTS.regression || {};
  const rows = Object.entries(reg).map(([name, metrics]) => {
    const r2Class = metrics.R2 >= 0.9 ? 'metric-excellent'
                  : metrics.R2 >= 0.7 ? 'metric-good'
                  : metrics.R2 >= 0.5 ? 'metric-fair' : 'metric-poor';
    const isBest  = name === MODEL_RESULTS.best_regression_model;
    return `<tr ${isBest ? 'class="table-active"' : ''}>
      <td>${isBest ? '🏆 ' : ''}<strong>${name}</strong></td>
      <td>${fmtCurrency(metrics.MAE, 2)}</td>
      <td>${fmtCurrency(metrics.RMSE, 2)}</td>
      <td><span class="metric-pill ${r2Class}">${metrics.R2.toFixed(4)}</span></td>
    </tr>`;
  }).join('');

  container.innerHTML = `
    <div class="table-responsive">
      <table class="table table-dark-custom">
        <thead><tr>
          <th>Model</th><th>MAE</th><th>RMSE</th><th>R² Score</th>
        </tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>`;
}

/* ════════════════════════════════════════════════════════════
   INDEX PAGE PARTICLES (decorative)
   ════════════════════════════════════════════════════════════ */
function initParticles() {
  const canvas = document.getElementById('particleCanvas');
  if (!canvas) return;
  const ctx    = canvas.getContext('2d');
  canvas.width  = canvas.offsetWidth;
  canvas.height = canvas.offsetHeight;

  const particles = Array.from({ length: 50 }, () => ({
    x : Math.random() * canvas.width,
    y : Math.random() * canvas.height,
    r : Math.random() * 2 + 0.5,
    vx: (Math.random() - 0.5) * 0.4,
    vy: (Math.random() - 0.5) * 0.4,
    a : Math.random() * 0.5 + 0.1,
  }));

  function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    particles.forEach(p => {
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(108,99,255,${p.a})`;
      ctx.fill();
      p.x += p.vx;
      p.y += p.vy;
      if (p.x < 0 || p.x > canvas.width)  p.vx *= -1;
      if (p.y < 0 || p.y > canvas.height) p.vy *= -1;
    });
    requestAnimationFrame(draw);
  }
  draw();
}

/* ════════════════════════════════════════════════════════════
   MAIN INIT
   ════════════════════════════════════════════════════════════ */
document.addEventListener('DOMContentLoaded', () => {
  // Mark active nav link
  const path = window.location.pathname;
  $$('.nav-link').forEach(link => {
    if (link.getAttribute('href') === path) link.classList.add('active');
  });

  initDashboard();
  initPrediction();
  initCustomerSearch();
  initModelMetrics();
  initParticles();

  // Smooth scroll for anchor links
  $$('a[href^="#"]').forEach(a => {
    a.addEventListener('click', e => {
      const target = document.querySelector(a.getAttribute('href'));
      if (target) {
        e.preventDefault();
        target.scrollIntoView({ behavior: 'smooth' });
      }
    });
  });
});
