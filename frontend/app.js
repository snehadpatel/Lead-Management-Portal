/* ═══════════════════════════════════════════════════════════════
   LUME AI — Core Application Logic
   ═══════════════════════════════════════════════════════════════ */

const API_BASE = (window.location.protocol.startsWith('http') && window.location.port !== '5500' && window.location.port !== '3000') 
  ? window.location.origin 
  : 'http://localhost:8000';

// ── State ──
const state = {
  role: localStorage.getItem('lume_role') || null,
  persona: localStorage.getItem('lume_persona') || null,
  userName: localStorage.getItem('lume_name') || 'Demo User',
  surveyStep: 0,
  surveyAnswers: {},
  apiOnline: false,
};

// ── Survey Questions ──
const SURVEY = [
  { id: 'ProfManage', question: 'How important is professional management of funds to you?', options: {'Very Low':1,'Low':2,'Moderate':3,'High':4,'Extremely High':5} },
  { id: 'Diversification', question: 'To what extent do you prefer spreading investments across asset classes?', options: {'Rarely':1,'Seldom':2,'Sometimes':3,'Usually':4,'Always':5} },
  { id: 'Affordability', question: 'How comfortable are you with required investment amounts?', options: {'Uncomfortable':1,'Slightly':2,'Neutral':3,'Comfortable':4,'Very Comfortable':5} },
  { id: 'Liquidity', question: 'How important is it to withdraw your money quickly?', options: {'Not Important':1,'Slightly':2,'Neutral':3,'Important':4,'Critical':5} },
  { id: 'Growth', question: 'How much do you prioritize aggressive capital appreciation over safety?', options: {'Safety First':1,'Mainly Safety':2,'Balanced':3,'Mainly Growth':4,'Max Growth':5} },
  { id: 'Trustworthiness', question: 'How much does brand reputation influence your choice?', options: {'Not at all':1,'A little':2,'Neutral':3,'Significantly':4,'Everything':5} },
  { id: 'Technology', question: 'How important are digital features and AI-driven insights?', options: {'Prefer Manual':1,'Minimal Tech':2,'Neutral':3,'Tech Oriented':4,'Tech First':5} },
];

const PERSONAS = {
  growth:       { name: '🚀 Growth Seeker', color: '#10B981', risk: 'High', funds: ['Mid Cap','Small Cap','Sectoral'] },
  conservative: { name: '🛡️ Wealth Preserver', color: '#2563EB', risk: 'Low', funds: ['Liquid','Short Duration','Corporate Bond'] },
  balanced:     { name: '⚖️ Balanced Allocator', color: '#F59E0B', risk: 'Medium', funds: ['Balanced Advantage','Aggressive Hybrid'] },
  passive:      { name: '📊 Index Tracker', color: '#8B5CF6', risk: 'Medium', funds: ['Index Funds','ETFs','FoF'] },
};

// ── API Client ──
async function apiCall(endpoint, options = {}) {
  try {
    const res = await fetch(`${API_BASE}${endpoint}`, {
      headers: { 'Content-Type': 'application/json' },
      ...options,
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } catch (e) {
    console.warn(`API ${endpoint} failed:`, e.message);
    return null;
  }
}

async function checkApiHealth() {
  const data = await apiCall('/health');
  state.apiOnline = !!data;
  const pulse = document.getElementById('apiPulse');
  const statusEl = document.getElementById('apiStatus');
  if (pulse) {
    pulse.className = state.apiOnline ? 'pulse pulse-live' : 'pulse pulse-offline';
  }
  if (statusEl) {
    const label = statusEl.querySelector('.status-item');
    if (label) label.innerHTML = `<span class="pulse ${state.apiOnline ? 'pulse-live' : 'pulse-offline'}" id="apiPulse"></span> API ${state.apiOnline ? 'Online' : 'Offline'}`;
  }
  return data;
}

async function predictLead(leadData) {
  return apiCall('/predict', { method: 'POST', body: JSON.stringify({ task: 'lead_scoring', lead: leadData }) });
}

async function predictCluster(behaviorData) {
  return apiCall('/predict', { method: 'POST', body: JSON.stringify({ task: 'investor_cluster', investor_behavior: behaviorData }) });
}

async function predictSentiment(text) {
  return apiCall('/predict', { method: 'POST', body: JSON.stringify({ task: 'sentiment', text }) });
}

async function getAnalytics() {
  return apiCall('/analytics');
}

async function getRiskAnalysis(persona, holdings) {
  return apiCall('/risk/analyze', { method: 'POST', body: JSON.stringify({ persona, holdings }) });
}

async function getMarketSnapshot() {
  return apiCall('/risk/market-snapshot');
}

async function getDashboardOverview() {
  return apiCall('/dashboard/overview');
}

async function getDashboardLeads(limit = 20) {
  return apiCall(`/dashboard/leads?limit=${limit}`);
}

async function getModelInfo(modelName) {
  return apiCall(`/model/${modelName}/info`);
}

async function getInsights() {
  return apiCall('/insights');
}

// ── Role Selection ──
function selectRole(role) {
  state.role = role;
  localStorage.setItem('lume_role', role);
  const saved = localStorage.getItem('lume_persona');
  if (saved) {
    state.persona = saved;
    window.location.href = role === 'distributor' ? 'distributor.html' : 'investor.html';
  } else {
    openSurvey();
  }
}

// ── Survey Engine ──
function openSurvey() {
  const modal = document.getElementById('surveyModal');
  if (modal) { modal.classList.remove('hidden'); modal.style.display = 'flex'; }
  state.surveyStep = 0;
  state.surveyAnswers = {};
  renderSurveyStep();
}

function renderSurveyStep() {
  const container = document.getElementById('surveyContent');
  if (!container) return;

  if (state.surveyStep >= SURVEY.length) {
    finishSurvey();
    return;
  }

  const q = SURVEY[state.surveyStep];
  const progress = ((state.surveyStep + 1) / SURVEY.length * 100).toFixed(0);
  const roleLabel = state.role === 'distributor' ? '👔 Distributor' : '💼 Investor';

  let html = `
    <div class="text-center mb-24">
      <span class="badge badge-blue">${roleLabel} Persona Assessment</span>
    </div>
    <div class="progress-bar mb-16"><div class="progress-fill" style="width:${progress}%"></div></div>
    <p class="text-xs text-muted mb-16">Question ${state.surveyStep + 1} of ${SURVEY.length}</p>
    <h3 class="mb-24">${q.question}</h3>
  `;
  for (const [label, value] of Object.entries(q.options)) {
    html += `<button class="survey-option" onclick="answerSurvey('${q.id}', ${value})">${label}</button>`;
  }
  container.innerHTML = html;
}

function answerSurvey(id, value) {
  state.surveyAnswers[id] = value;
  state.surveyStep++;
  renderSurveyStep();
}

async function finishSurvey() {
  const container = document.getElementById('surveyContent');
  if (container) container.innerHTML = '<div class="text-center"><h3>🔄 Analyzing your profile...</h3><p class="text-muted mt-8">Running K-Means clustering model</p></div>';

  // Try API first
  let persona = null;
  if (state.apiOnline) {
    const result = await predictCluster(state.surveyAnswers);
    if (result && result.prediction) {
      const cMap = { 0: 'conservative', 1: 'balanced', 2: 'passive', 3: 'growth' };
      persona = cMap[result.prediction.cluster_id] || 'balanced';
    }
  }

  // Fallback
  if (!persona) {
    const vals = Object.values(state.surveyAnswers);
    const avg = vals.reduce((a, b) => a + b, 0) / vals.length;
    if (avg >= 4) persona = 'growth';
    else if (avg >= 3) persona = 'balanced';
    else if (avg >= 2) persona = 'passive';
    else persona = 'conservative';
  }

  state.persona = persona;
  localStorage.setItem('lume_persona', persona);

  if (container) {
    const p = PERSONAS[persona];
    container.innerHTML = `
      <div class="text-center">
        <h2 style="font-size:3rem;">${p.name}</h2>
        <p class="text-muted mt-8">Risk Tolerance: <strong style="color:${p.color}">${p.risk}</strong></p>
        <p class="text-muted mt-8">Recommended: ${p.funds.join(', ')}</p>
        <button class="btn btn-primary btn-lg mt-32" onclick="goToDashboard()">Enter Dashboard →</button>
      </div>`;
  }
}

function goToDashboard() {
  window.location.href = state.role === 'distributor' ? 'distributor.html' : 'investor.html';
}

function logout() {
  localStorage.removeItem('lume_role');
  localStorage.removeItem('lume_persona');
  localStorage.removeItem('lume_name');
  window.location.href = 'index.html';
}

// ── Tab System ──
function switchTab(tabGroup, tabId) {
  document.querySelectorAll(`[data-tab-group="${tabGroup}"] .tab`).forEach(t => t.classList.remove('active'));
  document.querySelectorAll(`[data-tab-content="${tabGroup}"]`).forEach(c => c.classList.remove('active'));
  document.querySelector(`[data-tab-group="${tabGroup}"] [data-tab="${tabId}"]`)?.classList.add('active');
  document.getElementById(tabId)?.classList.add('active');
}

// ── Formatting Helpers ──
function formatCurrency(n) {
  if (n >= 10000000) return `₹${(n / 10000000).toFixed(2)} Cr`;
  if (n >= 100000) return `₹${(n / 100000).toFixed(2)} L`;
  return `₹${n.toLocaleString('en-IN')}`;
}

function formatPercent(n) { return `${n >= 0 ? '+' : ''}${n.toFixed(2)}%`; }

function getScoreColor(score) {
  if (score >= 85) return 'var(--emerald)';
  if (score >= 65) return 'var(--amber)';
  return 'var(--text-muted)';
}

function getScoreTier(score) {
  if (score >= 85) return { label: '🔥 HOT', class: 'badge-red' };
  if (score >= 65) return { label: '🌡️ WARM', class: 'badge-amber' };
  return { label: '❄️ COLD', class: 'badge-blue' };
}

// ── Sample Data Generators ──
function generateSampleLeads(n) {
  const firstNames = ['Rajesh','Priya','Amit','Sneha','Vikram','Ananya','Rahul','Divya','Karan','Meera','Arjun','Kavita','Sanjay','Nisha','Rohan'];
  const lastNames = ['Sharma','Patel','Kumar','Gupta','Singh','Verma','Reddy','Iyer','Joshi','Shah','Mehta','Agarwal','Chopra','Malhotra','Rao'];
  const sources = ['Organic Search','Google Ads','Referral','Direct Traffic','Social Media','Seminar','Walk-in'];
  const cities = ['Mumbai','Delhi','Bangalore','Hyderabad','Chennai','Pune','Kolkata','Ahmedabad','Jaipur','Lucknow'];
  const occupations = ['Working Professional','Business Owner','Student','Retired','Housewife'];
  const leads = [];
  for (let i = 0; i < n; i++) {
    const score = Math.random() * 0.6 + 0.35;
    leads.push({
      id: `L-${1000 + i}`,
      name: `${firstNames[i % firstNames.length]} ${lastNames[i % lastNames.length]}`,
      source: sources[Math.floor(Math.random() * sources.length)],
      city: cities[Math.floor(Math.random() * cities.length)],
      occupation: occupations[Math.floor(Math.random() * occupations.length)],
      score: Math.round(score * 100),
      phone: `+91-9${Math.random().toString().slice(2,11)}`,
      investment: `₹${Math.floor(Math.random() * 48 + 2)}L`,
    });
  }
  return leads.sort((a, b) => b.score - a.score);
}

function generateSampleFunds() {
  return [
    { id:'F001', name:'SBI Magnum MidCap Fund', category:'Mid Cap', risk:'high', r1:38.5, r3:22.1, r5:18.7, aum:12500, expense:1.2, rating:5, personas:['growth'] },
    { id:'F002', name:'HDFC Liquid Fund', category:'Liquid', risk:'low', r1:6.8, r3:6.2, r5:5.9, aum:45000, expense:0.2, rating:4, personas:['conservative'] },
    { id:'F003', name:'UTI Nifty 50 Index Fund', category:'Index', risk:'medium', r1:12.5, r3:14.8, r5:13.2, aum:18000, expense:0.1, rating:5, personas:['passive'] },
    { id:'F004', name:'Edelweiss Balanced Advantage', category:'Hybrid', risk:'medium', r1:18.5, r3:15.2, r5:14.1, aum:9800, expense:0.8, rating:4, personas:['balanced'] },
    { id:'F005', name:'Nippon India Small Cap', category:'Small Cap', risk:'high', r1:42.3, r3:28.5, r5:22.8, aum:31000, expense:1.0, rating:5, personas:['growth'] },
    { id:'F006', name:'ICICI Pru Short Term', category:'Short Duration', risk:'low', r1:7.2, r3:7.8, r5:7.1, aum:22000, expense:0.4, rating:4, personas:['conservative'] },
    { id:'F007', name:'Motilal Oswal Nifty Next 50', category:'Index', risk:'medium', r1:25.1, r3:18.3, r5:16.5, aum:8500, expense:0.3, rating:4, personas:['passive'] },
    { id:'F008', name:'Kotak Flexicap Fund', category:'Flexicap', risk:'medium', r1:22.8, r3:17.5, r5:15.9, aum:42000, expense:0.6, rating:5, personas:['balanced','growth'] },
    { id:'F009', name:'Axis Banking & PSU Fund', category:'Debt', risk:'low', r1:7.5, r3:7.1, r5:6.8, aum:18500, expense:0.3, rating:4, personas:['conservative'] },
    { id:'F010', name:'Tata Digital India Fund', category:'Sectoral', risk:'high', r1:35.2, r3:25.7, r5:21.3, aum:7200, expense:0.9, rating:4, personas:['growth'] },
  ];
}

// ── Chart Helpers ──
const chartDefaults = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: { legend: { labels: { color: '#64748B', font: { family: 'Inter' } } } },
  scales: {
    x: { ticks: { color: '#64748B' }, grid: { color: 'rgba(0,0,0,0.05)' } },
    y: { ticks: { color: '#64748B' }, grid: { color: 'rgba(0,0,0,0.05)' } },
  },
};

function createLineChart(canvasId, labels, datasets, opts = {}) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return null;
  return new Chart(ctx, {
    type: 'line',
    data: { labels, datasets },
    options: { ...chartDefaults, ...opts },
  });
}

function createBarChart(canvasId, labels, data, colors, opts = {}) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return null;
  return new Chart(ctx, {
    type: 'bar',
    data: { labels, datasets: [{ data, backgroundColor: colors, borderRadius: 6, barThickness: 32 }] },
    options: { ...chartDefaults, plugins: { ...chartDefaults.plugins, legend: { display: false } }, ...opts },
  });
}

function createDoughnutChart(canvasId, labels, data, colors, opts = {}) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return null;
  return new Chart(ctx, {
    type: 'doughnut',
    data: { labels, datasets: [{ data, backgroundColor: colors, borderWidth: 0 }] },
    options: { responsive: true, maintainAspectRatio: false, cutout: '70%',
      plugins: { legend: { position: 'bottom', labels: { color: '#64748B', font: { family: 'Inter' }, padding: 16 } } }, ...opts },
  });
}

// ── AI Copilot Helper Functions ──
async function validateRecommendation(clientId, fundName, category, riskLevel) {
  return apiCall('/distributors/validate-rec', {
    method: 'POST',
    body: JSON.stringify({ client_id: clientId, fund_name: fundName, category, risk_level: riskLevel })
  });
}

async function getClientInsights() {
  return apiCall('/distributors/client-insights');
}

async function askAdvisor(query, persona) {
  return apiCall('/advisor/query', {
    method: 'POST',
    body: JSON.stringify({ query, persona })
  });
}

async function askBuddyChat(message, history = [], distributorId = null, leadId = null) {
  return apiCall('/buddy/chat', {
    method: 'POST',
    body: JSON.stringify({ message, history, distributor_id: distributorId, lead_id: leadId })
  });
}

// ── SIP Calculator ──
function calculateSIP(monthly, rate, years) {
  const months = years * 12;
  const r = rate / 100 / 12;
  const fv = monthly * (((1 + r) ** months - 1) / r) * (1 + r);
  return { futureValue: Math.round(fv), invested: monthly * months, gains: Math.round(fv) - monthly * months };
}

// ── Init ──
document.addEventListener('DOMContentLoaded', () => {
  checkApiHealth();
  setInterval(checkApiHealth, 30000);
});
