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
  isDemo: localStorage.getItem('lume_is_demo') === '1',
  surveyStep: 0,
  surveyAnswers: {},
  apiOnline: false,
  streamConnected: false,
  latestRealtimePacket: null,
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
    const token = localStorage.getItem('lume_auth_token');
    const headers = { 'Content-Type': 'application/json' };
    if (token) {
      headers['Authorization'] = `Bearer ${token}`;
    }
    const res = await fetch(`${API_BASE}${endpoint}`, {
      ...options,
      headers: {
        ...headers,
        ...(options.headers || {})
      },
    });
    if (!res.ok) {
      if (res.status === 401) {
        console.warn('Unauthorized request - logging out');
        logout();
      }
      throw new Error(`HTTP ${res.status}`);
    }
    return await res.json();
  } catch (e) {
    console.warn(`API ${endpoint} failed:`, e.message);
    return null;
  }
}

async function checkApiHealth() {
  const data = await apiCall('/health');
  state.apiOnline = !!data;
  if (data && data.models_loaded) {
    state.modelsLoaded = data.models_loaded;
  }
  const pulse = document.getElementById('apiPulse');
  const statusEl = document.getElementById('apiStatus');
  if (pulse) {
    pulse.className = state.apiOnline ? 'pulse pulse-live' : 'pulse pulse-offline';
  }
  if (statusEl) {
    const label = statusEl.querySelector('.status-item');
    if (label) {
      const wsBadge = state.streamConnected
        ? ` · <span style="color:var(--emerald);font-weight:700;">LIVE STREAM</span>`
        : '';
      label.innerHTML = `<span class="pulse ${state.apiOnline ? 'pulse-live' : 'pulse-offline'}" id="apiPulse"></span> API ${state.apiOnline ? 'Online' : 'Offline'}${wsBadge}`;
    }
  }
  return data;
}

async function getRealtimePulse(advance = false) {
  return apiCall(`/api/realtime/pulse?advance=${advance ? 'true' : 'false'}`);
}

async function getBigDataLiveStats() {
  return apiCall('/api/bigdata/live-stats');
}

async function searchFundsSemantic(query, topK = 12) {
  return apiCall(`/search/funds?query=${encodeURIComponent(query)}&top_k=${topK}`);
}

async function predictLeadDual(leadData) {
  return apiCall('/api/dl-lab/predict-lead', { method: 'POST', body: JSON.stringify(leadData) });
}

async function analyzeFinancialNLP(text) {
  return apiCall('/api/nlp/financial-sentiment', { method: 'POST', body: JSON.stringify({ text }) });
}

async function getNLPBenchmark() {
  return apiCall('/api/nlp/benchmark');
}

async function classifyVisionPattern(schemeCode = '100119') {
  return apiCall('/api/vision/pattern', { method: 'POST', body: JSON.stringify({ scheme_code: String(schemeCode) }) });
}

async function fetchFundFullAI(schemeCode, days = 90) {
  return apiCall(`/api/fund/${encodeURIComponent(schemeCode)}/full-ai?days=${days}`);
}

async function fetchBuddyBriefing(distributorId = 'demo_distributor') {
  return apiCall(`/buddy/briefing?distributor_id=${encodeURIComponent(distributorId)}`);
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

async function getLiveAlerts() {
  return apiCall('/alerts/live');
}

async function getMarketImpact() {
  return apiCall('/market/impact/latest');
}

async function getDriftStatus() {
  return apiCall('/monitoring/drift/status');
}

async function triggerRetrain() {
  return apiCall('/monitoring/retrain', { method: 'POST' });
}

async function getFundCatalog(persona = null, topK = 100, query = null) {
  const params = new URLSearchParams({ top_k: String(topK) });
  if (persona) params.set('persona', persona);
  if (query) params.set('query', query);
  return apiCall(`/funds/catalog?${params.toString()}`);
}

async function getPersonaRecommendations(persona, topK = 5, investmentHorizon = null) {
  const params = new URLSearchParams({ persona, top_k: String(topK) });
  if (investmentHorizon) params.set('investment_horizon', investmentHorizon);
  return apiCall(`/funds/recommendations?${params.toString()}`);
}

async function saveUserProfile(userId, profile) {
  return apiCall(`/user/profile?user_id=${encodeURIComponent(userId)}`, {
    method: 'POST',
    body: JSON.stringify(profile),
  });
}

// ── Portfolio API ──
async function fetchPortfolio() {
  return apiCall('/portfolio');
}

async function addToPortfolio(schemeCode, schemeName, category, amount, buyDate = null) {
  const body = { scheme_code: schemeCode, scheme_name: schemeName, category, amount };
  if (buyDate) body.buy_date = buyDate;
  return apiCall('/portfolio/add', { method: 'POST', body: JSON.stringify(body) });
}

async function removeHolding(holdingId) {
  return apiCall(`/portfolio/${holdingId}`, { method: 'DELETE' });
}

async function sellHolding(holdingId, units = null, amount = null, sellAll = false) {
  const body = { holding_id: holdingId, sell_all: sellAll };
  if (units !== null) body.units = parseFloat(units);
  if (amount !== null) body.amount = parseFloat(amount);
  return apiCall('/portfolio/sell', { method: 'POST', body: JSON.stringify(body) });
}

async function fetchPortfolioSummary() {
  return apiCall('/portfolio/summary');
}

async function fetchLiveNav(schemeCode) {
  return apiCall(`/nav/${schemeCode}`);
}

async function fetchNavHistory(schemeCode, days = 365) {
  return apiCall(`/nav/${schemeCode}/history?days=${days}`);
}

async function fetchPortfolioAnalytics(horizon = '1y') {
  return apiCall(`/portfolio/analytics?horizon=${horizon}`);
}

async function fetchLiveNews(maxArticles = 12) {
  return apiCall(`/news/live?max_articles=${maxArticles}`);
}

async function fetchBenchmarkHistory(days = 30) {
  return apiCall(`/benchmark/history?days=${days}`);
}

function renderAlertItems(container, title, items, accent = 'var(--red)') {
  if (!container) return;
  const list = Array.isArray(items) ? items : [];
  if (!list.length) {
    container.innerHTML = `<div class="card card-left-amber" style="padding:12px 14px; border-left-color:${accent};"><h4 style="margin:0;">${title}</h4><p class="text-sm text-muted mt-8">No active signals right now.</p></div>`;
    return;
  }
  container.innerHTML = `<h4 style="margin-bottom:12px;">${title}</h4>` + list.map(item => {
    const text = typeof item === 'string' ? item : (item.message || item.alert || item.text || JSON.stringify(item));
    return `<div class="card card-left-red mt-8" style="padding:10px 14px; border-left-color:${accent};"><p class="text-sm">${text}</p></div>`;
  }).join('');
}

async function refreshLiveOpsWidgets() {
  const [alerts, impact, drift] = await Promise.all([
    getLiveAlerts(),
    getMarketImpact(),
    getDriftStatus(),
  ]);

  const alertsTargets = ['homeLiveAlerts', 'distLiveAlerts', 'invLiveAlerts'];
  alertsTargets.forEach(id => {
    const el = document.getElementById(id);
    if (el && alerts) renderAlertItems(el, 'Live Alerts', alerts.alerts || []);
  });

  const impactTargets = ['homeMarketImpact', 'distMarketImpact', 'invMarketImpact'];
  impactTargets.forEach(id => {
    const el = document.getElementById(id);
    if (!el || !impact) return;
    const sectors = impact.affected_sectors || [];
    el.innerHTML = `
      <h4 style="margin-bottom:12px;">Market Impact</h4>
      <div class="badge badge-${impact.severity === 'critical' || impact.severity === 'high' ? 'red' : impact.severity === 'positive' ? 'emerald' : 'amber'}">${String(impact.severity || 'moderate').toUpperCase()}</div>
      <p class="text-sm mt-12"><strong>Confidence:</strong> ${(impact.confidence || 0).toFixed ? impact.confidence.toFixed(2) : impact.confidence}</p>
      <p class="text-sm text-muted mt-8"><strong>Affected sectors:</strong> ${sectors.length ? sectors.join(', ') : 'None'}</p>
    `;
  });

  const driftTargets = ['homeDriftStatus', 'distDriftStatus', 'invDriftStatus'];
  driftTargets.forEach(id => {
    const el = document.getElementById(id);
    if (!el || !drift) return;
    const color = drift.drift_detected ? 'var(--red)' : drift.retrain_in_progress ? 'var(--amber)' : 'var(--emerald)';
    el.innerHTML = `
      <h4 style="margin-bottom:12px;">Retraining Monitor</h4>
      <div class="badge" style="background:${color}20;color:${color};">${drift.drift_detected ? 'DRIFT DETECTED' : drift.retrain_in_progress ? 'TRAINING' : 'STABLE'}</div>
      <p class="text-sm mt-12"><strong>Samples:</strong> ${drift.samples_seen || 0}</p>
      <p class="text-sm"><strong>Mean Score:</strong> ${Number(drift.mean_score || 0).toFixed(3)}</p>
      <p class="text-sm"><strong>Negative Ratio:</strong> ${Number(drift.negative_ratio || 0).toFixed(2)}</p>
      <p class="text-xs text-muted mt-8">${drift.last_message || 'idle'}</p>
    `;
  });
}

// ── Role Selection ──
function selectRole(role) {
  localStorage.setItem('lume_onboarding_role', role);
  window.location.href = `signup.html?role=${role}`;
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

  // Sync profile with the backend
  if (state.userName) {
    await saveUserProfile(state.userName, {
      inferred_risk_profile: persona,
      goals: `${persona} mutual fund recommendations`,
      investment_horizon: 'long term'
    });
  }

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
  const role = state.role || localStorage.getItem('lume_role') || 'distributor';
  window.location.href = role === 'distributor' ? 'distributor.html' : 'investor.html';
}

function logout() {
  localStorage.removeItem('lume_role');
  localStorage.removeItem('lume_persona');
  localStorage.removeItem('lume_name');
  localStorage.removeItem('lume_auth_token');
  localStorage.removeItem('lume_is_demo');
  window.location.href = 'index.html';
}

async function loginDemo(role = 'investor', targetRedirect = true) {
  try {
    const cleanRole = role === 'distributor' ? 'distributor' : 'investor';
    const res = await fetch(`${API_BASE}/auth/demo-login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ role: cleanRole }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Demo login failed');

    localStorage.setItem('lume_auth_token', data.access_token);
    localStorage.setItem('lume_role', data.role);
    localStorage.setItem('lume_persona', data.persona || 'balanced');
    localStorage.setItem('lume_name', data.full_name || (cleanRole === 'distributor' ? 'Demo Distributor' : 'Demo Investor'));
    localStorage.setItem('lume_is_demo', '1');

    state.role = data.role;
    state.persona = data.persona || 'balanced';
    state.userName = localStorage.getItem('lume_name');
    state.isDemo = true;

    if (targetRedirect) {
      window.location.href = cleanRole === 'distributor' ? 'distributor.html' : 'investor.html';
    }
    return data;
  } catch (err) {
    console.error('Demo login failed:', err);
    const errBox = document.getElementById('errorBox');
    if (errBox) {
      errBox.textContent = `Demo login failed: ${err.message}. Is FastAPI running on ${API_BASE}?`;
      errBox.classList.remove('hidden');
    } else {
      alert(`Demo login failed (${err.message}). Make sure FastAPI backend is running.`);
    }
    return null;
  }
}

async function switchDemoRole(targetRole = null) {
  const current = state.role || localStorage.getItem('lume_role') || 'investor';
  const nextRole = targetRole || (current === 'investor' ? 'distributor' : 'investor');
  await loginDemo(nextRole, true);
}

// ── Client-Side Authentication Handlers ──
async function handleSignup(e) {
  e.preventDefault();
  const email = document.getElementById('email').value.trim();
  const password = document.getElementById('password').value;
  const confirmPassword = document.getElementById('confirmPassword').value;
  const role = document.getElementById('role').value;
  const errorBox = document.getElementById('errorBox');

  if (password !== confirmPassword) {
    errorBox.textContent = 'Passwords do not match';
    errorBox.classList.remove('hidden');
    return;
  }

  try {
    const res = await fetch(`${API_BASE}/auth/register`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password, role })
    });
    
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Registration failed');

    localStorage.setItem('lume_auth_token', data.access_token);
    localStorage.setItem('lume_role', data.role);
    localStorage.setItem('lume_name', email.split('@')[0]);

    state.role = data.role;
    state.userName = email.split('@')[0];

    // Show survey modal for onboarding matching
    openSurvey();
  } catch (err) {
    errorBox.textContent = err.message;
    errorBox.classList.remove('hidden');
  }
}

async function handleLogin(e) {
  e.preventDefault();
  const email = document.getElementById('email').value.trim();
  const password = document.getElementById('password').value;
  const errorBox = document.getElementById('errorBox');

  try {
    const res = await fetch(`${API_BASE}/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password })
    });
    
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Login failed');

    localStorage.setItem('lume_auth_token', data.access_token);
    localStorage.setItem('lume_role', data.role);
    localStorage.setItem('lume_name', email.split('@')[0]);

    state.role = data.role;
    state.userName = email.split('@')[0];

    // Fetch profile from me endpoint
    const meRes = await fetch(`${API_BASE}/auth/me`, {
      headers: { 'Authorization': `Bearer ${data.access_token}` }
    });
    if (meRes.ok) {
      const meData = await meRes.json();
      if (meData.persona) {
        localStorage.setItem('lume_persona', meData.persona);
        state.persona = meData.persona;
        window.location.href = data.role === 'distributor' ? 'distributor.html' : 'investor.html';
        return;
      }
    }

    openSurvey();
  } catch (err) {
    errorBox.textContent = err.message;
    errorBox.classList.remove('hidden');
  }
}

function checkAuthRedirect() {
  const token = localStorage.getItem('lume_auth_token');
  if (!token) {
    window.location.href = 'login.html';
  }
}

async function verifyAuthAndSyncProfile() {
  const token = localStorage.getItem('lume_auth_token');
  if (!token) {
    window.location.href = 'login.html';
    return null;
  }
  
  try {
    const res = await fetch(`${API_BASE}/auth/me`, {
      headers: { 'Authorization': `Bearer ${token}` }
    });
    if (!res.ok) {
      logout();
      return null;
    }
    const user = await res.json();
    
    // Sync state
    state.userName = user.email.split('@')[0];
    localStorage.setItem('lume_name', state.userName);
    if (user.persona) {
      state.persona = user.persona;
      localStorage.setItem('lume_persona', user.persona);
    }
    state.role = user.role;
    localStorage.setItem('lume_role', user.role);
    
    const pathname = window.location.pathname;
    if (pathname.endsWith('distributor.html') && user.role !== 'distributor') {
      window.location.href = 'investor.html';
      return null;
    }
    if (pathname.endsWith('investor.html') && user.role !== 'investor') {
      window.location.href = 'distributor.html';
      return null;
    }
    
    // Update welcome title
    const welcomeTitle = document.getElementById('welcomeTitle');
    if (welcomeTitle) {
      const displayName = user.full_name || (state.userName.charAt(0).toUpperCase() + state.userName.slice(1));
      const demoTag = user.is_demo ? ' <span class="badge badge-amber" style="font-size:0.65rem;vertical-align:middle;">DEMO LIVE</span>' : '';
      welcomeTitle.innerHTML = `Welcome, ${displayName}${demoTag}`;
    }

    // Update persona label
    const personaLabel = document.getElementById('personaLabel');
    if (personaLabel) {
      const p = PERSONAS[state.persona || 'balanced'] || PERSONAS.balanced;
      const isDist = pathname.endsWith('distributor.html');
      personaLabel.innerHTML = isDist
        ? `Persona: <strong style="color:${p.color}">${p.name}</strong> · Risk: ${p.risk}`
        : `Persona: <strong style="color:${p.color}">${p.name}</strong> · Risk Tolerance: ${p.risk}`;
    }

    return user;
  } catch (err) {
    console.warn('Error in verifyAuthAndSyncProfile:', err);
    // If API is down, use local storage fallback
    const welcomeTitle = document.getElementById('welcomeTitle');
    if (welcomeTitle && state.userName) {
      const displayName = state.userName.charAt(0).toUpperCase() + state.userName.slice(1);
      welcomeTitle.textContent = `Welcome, ${displayName}`;
    }
    return null;
  }
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

async function askAdvisor(query, persona, dashboard_context = null) {
  return apiCall('/advisor/query', {
    method: 'POST',
    body: JSON.stringify({ query, persona, dashboard_context })
  });
}

async function askBuddyChat(message, history = [], distributorId = null, leadId = null, dashboard_context = null) {
  return apiCall('/buddy/chat', {
    method: 'POST',
    body: JSON.stringify({ message, history, distributor_id: distributorId, lead_id: leadId, dashboard_context })
  });
}

// ── SIP Calculator ──
function calculateSIP(monthly, rate, years) {
  const months = years * 12;
  const r = rate / 100 / 12;
  const fv = monthly * (((1 + r) ** months - 1) / r) * (1 + r);
  return { futureValue: Math.round(fv), invested: monthly * months, gains: Math.round(fv) - monthly * months };
}

// ── Real-Time WebSocket Stream Client ──
let _realtimeWs = null;
let _realtimePollTimer = null;
let _wsRetryMs = 2000;

function applyRealtimeTickToSharedWidgets(packet) {
  if (!packet) return;
  state.latestRealtimePacket = packet;

  const rtPill = document.getElementById('rtStreamPill');
  if (rtPill) {
    const nifty = packet.market?.nifty_50 ? Number(packet.market.nifty_50).toLocaleString('en-IN', { maximumFractionDigits: 1 }) : '—';
    const vix = packet.market?.vix != null ? Number(packet.market.vix).toFixed(1) : '—';
    const seq = packet.tick_seq || 1;
    rtPill.innerHTML = `<span class="pulse pulse-live"></span> LIVE #${seq} · NIFTY ${nifty} · VIX ${vix}`;
  }

  // Live Drift Status cards across Home / Investor / Distributor
  const drift = packet.drift;
  if (drift) {
    const color = drift.drift_detected ? 'var(--red)' : drift.retrain_in_progress ? 'var(--amber)' : 'var(--emerald)';
    ['homeDriftStatus', 'distDriftStatus', 'invDriftStatus'].forEach(id => {
      const el = document.getElementById(id);
      if (!el) return;
      el.innerHTML = `
        <h4 style="margin-bottom:12px;">Retraining Monitor <span class="text-xs text-muted">(Real-Time)</span></h4>
        <div class="badge" style="background:${color}20;color:${color};">${drift.drift_detected ? 'DRIFT DETECTED' : drift.retrain_in_progress ? 'TRAINING' : 'STABLE'}</div>
        <p class="text-sm mt-12"><strong>Samples:</strong> ${drift.samples_seen || 0}</p>
        <p class="text-sm"><strong>Mean Score:</strong> ${Number(drift.mean_score || 0).toFixed(3)}</p>
        <p class="text-sm"><strong>Negative Ratio:</strong> ${Number(drift.negative_ratio || 0).toFixed(2)}</p>
        <p class="text-xs text-muted mt-8">${drift.last_message || 'streaming telemetry active'}</p>
      `;
    });
  }

  // Live Signal Alerts preview on Home / Distributor
  const sig = packet.latest_signal;
  if (sig) {
    const sentColor = sig.sentiment === 'positive' ? 'var(--emerald)' : sig.sentiment === 'negative' ? 'var(--red)' : 'var(--amber)';
    const tokensHtml = (sig.token_saliency || []).slice(0, 4).map(t =>
      `<span class="badge" style="font-size:0.65rem;padding:2px 6px;background:${t.polarity === 'positive' ? 'rgba(16,185,129,0.12)' : t.polarity === 'negative' ? 'rgba(239,68,68,0.12)' : 'rgba(148,163,184,0.12)'};color:${t.polarity === 'positive' ? 'var(--emerald)' : t.polarity === 'negative' ? 'var(--red)' : 'var(--text-secondary)'};">${t.token}</span>`
    ).join(' ');

    ['homeLiveAlerts', 'distLiveAlerts'].forEach(id => {
      const el = document.getElementById(id);
      if (!el) return;
      el.innerHTML = `
        <div class="flex justify-between items-center mb-8">
          <h4 style="margin:0;">Live Transformer Stream</h4>
          <span class="badge" style="background:${sentColor}20;color:${sentColor};">${String(sig.sentiment || 'neutral').toUpperCase()} ${(Number(sig.confidence || 0.5) * 100).toFixed(0)}%</span>
        </div>
        <p class="text-sm" style="font-weight:600;line-height:1.4;">${sig.headline || ''}</p>
        <div class="flex gap-4 mt-8" style="flex-wrap:wrap;">${tokensHtml}</div>
        <p class="text-xs text-muted mt-8">Event ${sig.event_id || ''} · ${sig.latency_ms ? sig.latency_ms.toFixed(1) + ' ms' : ''} · ${sig.source || ''}</p>
      `;
    });
  }

  // Update Market Impact card from real-time market tick if loaded
  const mkt = packet.market;
  if (mkt) {
    const sev = mkt.vix > 22 ? 'high' : mkt.nifty_change_pct > 0.4 ? 'positive' : 'moderate';
    const sevBadge = sev === 'high' ? 'red' : sev === 'positive' ? 'emerald' : 'amber';
    const sectors = Object.entries(mkt.sector_performance || {})
      .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
      .slice(0, 3)
      .map(([k, v]) => `${k} (${v >= 0 ? '+' : ''}${Number(v).toFixed(2)}%)`);
    ['homeMarketImpact', 'distMarketImpact'].forEach(id => {
      const el = document.getElementById(id);
      if (!el) return;
      el.innerHTML = `
        <h4 style="margin-bottom:12px;">Live Market Pulse Impact</h4>
        <div class="badge badge-${sevBadge}">${String(mkt.market_sentiment || sev).toUpperCase()}</div>
        <p class="text-sm mt-12"><strong>NIFTY 50:</strong> ${Number(mkt.nifty_50 || 0).toLocaleString('en-IN', { maximumFractionDigits: 1 })} (${mkt.nifty_change_pct >= 0 ? '+' : ''}${Number(mkt.nifty_change_pct || 0).toFixed(2)}%)</p>
        <p class="text-sm"><strong>India VIX:</strong> ${Number(mkt.vix || 0).toFixed(2)} · <strong>Models:</strong> ${packet.models_online || 7}/${packet.models_total || 7}</p>
        <p class="text-xs text-muted mt-8"><strong>Key Movers:</strong> ${sectors.join(', ') || 'Broad Market'}</p>
      `;
    });
  }

  window.dispatchEvent(new CustomEvent('lume:realtime-tick', { detail: packet }));
}

function connectRealtimeStream() {
  if (_realtimeWs && (_realtimeWs.readyState === WebSocket.OPEN || _realtimeWs.readyState === WebSocket.CONNECTING)) {
    return;
  }
  try {
    const wsUrl = `${API_BASE.replace(/^http/i, 'ws')}/ws/realtime`;
    _realtimeWs = new WebSocket(wsUrl);

    _realtimeWs.onopen = () => {
      state.streamConnected = true;
      state.apiOnline = true;
      _wsRetryMs = 2000;
      if (_realtimePollTimer) {
        clearInterval(_realtimePollTimer);
        _realtimePollTimer = null;
      }
      checkApiHealth();
    };

    _realtimeWs.onmessage = (event) => {
      try {
        const packet = JSON.parse(event.data);
        if (packet && packet.type !== 'pong') {
          applyRealtimeTickToSharedWidgets(packet);
        }
      } catch (e) {
        console.warn('WebSocket parse error:', e);
      }
    };

    _realtimeWs.onclose = () => {
      state.streamConnected = false;
      checkApiHealth();
      // Start REST polling fallback while retrying WS
      if (!_realtimePollTimer) {
        _realtimePollTimer = setInterval(async () => {
          const snap = await getRealtimePulse(false);
          if (snap) applyRealtimeTickToSharedWidgets(snap);
        }, 6000);
      }
      setTimeout(connectRealtimeStream, _wsRetryMs);
      _wsRetryMs = Math.min(15000, Math.round(_wsRetryMs * 1.5));
    };

    _realtimeWs.onerror = () => {
      state.streamConnected = false;
    };
  } catch (err) {
    console.warn('WebSocket setup error, falling back to HTTP pulse:', err);
    if (!_realtimePollTimer) {
      _realtimePollTimer = setInterval(async () => {
        const snap = await getRealtimePulse(false);
        if (snap) applyRealtimeTickToSharedWidgets(snap);
      }, 5000);
    }
  }
}

// ── Init ──
document.addEventListener('DOMContentLoaded', () => {
  checkApiHealth();
  setInterval(checkApiHealth, 30000);
  refreshLiveOpsWidgets();
  setInterval(refreshLiveOpsWidgets, 30000);
  connectRealtimeStream();

  // Pre-fill signup role if present in query parameters
  if (window.location.pathname.endsWith('signup.html')) {
    const params = new URLSearchParams(window.location.search);
    const r = params.get('role');
    if (r && document.getElementById('role')) {
      document.getElementById('role').value = r;
    }
  }

  // Handle dynamic navigation auth buttons
  const token = localStorage.getItem('lume_auth_token');
  const role = localStorage.getItem('lume_role');
  const navAuth = document.getElementById('navAuthSection');
  if (navAuth) {
    if (token) {
      const dashboardUrl = role === 'distributor' ? 'distributor.html' : 'investor.html';
      const switchTarget = role === 'distributor' ? 'investor' : 'distributor';
      const switchLabel = role === 'distributor' ? '⇄ Demo Investor' : '⇄ Demo Distributor';
      navAuth.innerHTML = `
        <button class="btn btn-sm btn-secondary" onclick="switchDemoRole('${switchTarget}')" title="1-click role switch">${switchLabel}</button>
        <a href="${dashboardUrl}" class="btn btn-sm btn-primary">Dashboard &rarr;</a>
        <button class="btn btn-sm btn-secondary" onclick="logout()">Logout</button>
      `;
    } else {
      navAuth.innerHTML = `
        <button class="btn btn-sm btn-emerald" onclick="loginDemo('investor')">Demo Investor</button>
        <button class="btn btn-sm btn-primary" onclick="loginDemo('distributor')">Demo Distributor</button>
        <a href="login.html" class="btn btn-sm btn-secondary">Log In</a>
      `;
    }
  }

  // Pre-calculate ROI on landing page load if elements exist
  if (document.getElementById('roiLeads')) {
    calculateROI();
  }
});

// ── ROI Calculator Handler ──
function calculateROI() {
  const leads = parseInt(document.getElementById('roiLeads').value) || 0;
  const hours = parseFloat(document.getElementById('roiHours').value) || 0;
  const rate = parseFloat(document.getElementById('roiRate').value) || 0;

  const timeReduction = 0.85; // Assume 85% time reduction with Lume AI
  const hoursSavedPerMonth = Math.round(leads * hours * timeReduction);
  const annualSavings = Math.round(hoursSavedPerMonth * rate * 12);

  document.getElementById('roiSavedHours').textContent = `${hoursSavedPerMonth} hrs`;
  document.getElementById('roiSavedWealth').textContent = `₹${annualSavings.toLocaleString('en-IN')}`;
}
