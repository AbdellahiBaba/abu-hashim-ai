const QalamDashboard = {
  state: {
    currentPage: 'dashboard',
    modelStatus: 'offline',
    chatHistory: [],
    stats: {
      totalRequests: 0,
      avgLatency: 0,
      activeModels: 0,
      bufferSize: 0
    }
  },

  init() {
    this.bindNavigation();
    this.bindChatInput();
    this.bindQuickActions();
    this.bindFileUpload();
    this.loadStats();
    this.checkModelStatus();
    this.startPolling();
  },

  bindNavigation() {
    document.querySelectorAll('.nav-item[data-page]').forEach(item => {
      item.addEventListener('click', (e) => {
        e.preventDefault();
        const page = item.dataset.page;
        this.navigateTo(page);
      });
    });
  },

  navigateTo(page) {
    document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active'));
    const activeNav = document.querySelector(`.nav-item[data-page="${page}"]`);
    if (activeNav) activeNav.classList.add('active');

    document.querySelectorAll('.page-content').forEach(el => {
      el.style.display = 'none';
    });
    const pageEl = document.getElementById(`page-${page}`);
    if (pageEl) pageEl.style.display = 'block';

    this.state.currentPage = page;

    if (page === 'bridge') this.loadBridgeStats();

    const titles = {
      dashboard: 'Dashboard',
      chat: 'Chat Playground',
      bridge: 'QalamAI Bridge',
      training: 'Training',
      evaluation: 'Evaluation',
      settings: 'Settings'
    };
    const titleEl = document.querySelector('.topbar-title h1');
    if (titleEl) titleEl.textContent = titles[page] || 'Dashboard';
  },

  bindChatInput() {
    const input = document.getElementById('chat-input');
    const sendBtn = document.getElementById('chat-send');

    if (input && sendBtn) {
      sendBtn.addEventListener('click', () => this.sendMessage());
      input.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
          e.preventDefault();
          this.sendMessage();
        }
      });
    }
  },

  bindQuickActions() {
    document.querySelectorAll('[data-action]').forEach(btn => {
      btn.addEventListener('click', (e) => {
        e.preventDefault();
        const action = btn.dataset.action;
        this.handleAction(action, btn);
      });
    });
  },

  async handleAction(action, btn) {
    const originalText = btn.textContent;
    btn.textContent = 'Processing...';
    btn.disabled = true;

    const endpoints = {
      'load-model': { url: '/api/status', method: 'GET' },
      'start-training': { url: '/api/update-model', method: 'POST' },
      'run-evaluation': { url: '/api/evaluate', method: 'POST' },
      'process-buffer': { url: '/api/process-buffer', method: 'POST' },
      'build-dataset': { url: '/api/qalam-build-dataset', method: 'POST' }
    };

    const config = endpoints[action];
    if (!config) {
      btn.textContent = originalText;
      btn.disabled = false;
      return;
    }

    try {
      const response = await fetch(config.url, { method: config.method });
      const data = await response.json();

      if (data.status === 'error') {
        this.showNotification(data.message || 'Operation failed', 'error');
      } else {
        this.showNotification(`${originalText}: completed successfully`, 'success');
      }

      this.loadStats();
    } catch (error) {
      this.showNotification('Request failed: ' + error.message, 'error');
    }

    btn.textContent = originalText;
    btn.disabled = false;
  },

  showNotification(message, type) {
    const existing = document.querySelector('.notification');
    if (existing) existing.remove();

    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.style.cssText = `
      position: fixed; top: 20px; right: 20px; padding: 12px 24px;
      border-radius: 8px; z-index: 10000; font-size: 14px;
      color: white; max-width: 400px; box-shadow: 0 4px 12px rgba(0,0,0,0.3);
      background: ${type === 'error' ? '#e74c3c' : '#27ae60'};
    `;
    notification.textContent = message;
    document.body.appendChild(notification);

    setTimeout(() => notification.remove(), 4000);
  },

  async sendMessage() {
    const input = document.getElementById('chat-input');
    const message = input.value.trim();
    if (!message) return;

    this.appendChatMessage('user', message);
    input.value = '';

    const tempId = this.appendChatMessage('assistant', '...');

    try {
      const response = await fetch('/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: message,
          temperature: parseFloat(document.getElementById('temperature')?.value || '0.7'),
          max_new_tokens: parseInt(document.getElementById('max-tokens')?.value || '512')
        })
      });

      if (!response.ok) throw new Error('Generation failed');

      const data = await response.json();
      this.updateChatMessage(tempId, data.generated_text || 'No response received.');
    } catch (error) {
      this.updateChatMessage(tempId, 'Error: Could not generate response. Please check if the model is loaded.');
    }
  },

  appendChatMessage(role, content) {
    const container = document.getElementById('chat-messages');
    if (!container) return null;

    const id = 'msg-' + Date.now();
    const avatar = role === 'assistant' ? 'Q' : 'U';

    const messageHtml = `
      <div class="chat-message ${role}" id="${id}">
        <div class="chat-avatar">${avatar}</div>
        <div class="chat-bubble">${this.escapeHtml(content)}</div>
      </div>
    `;
    container.insertAdjacentHTML('beforeend', messageHtml);
    container.scrollTop = container.scrollHeight;

    return id;
  },

  updateChatMessage(id, content) {
    const msg = document.getElementById(id);
    if (msg) {
      const bubble = msg.querySelector('.chat-bubble');
      if (bubble) bubble.textContent = content;
    }
  },

  escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  },

  async loadStats() {
    try {
      const response = await fetch('/api/stats');
      if (response.ok) {
        const data = await response.json();
        this.updateStats(data);
      }
    } catch (error) {
      console.log('Stats endpoint not available');
    }
  },

  updateStats(data) {
    const mappings = {
      'stat-requests': data.total_requests,
      'stat-latency': data.avg_latency ? data.avg_latency.toFixed(0) + 'ms' : '\u2014',
      'stat-models': data.active_models,
      'stat-buffer': data.buffer_size
    };

    Object.entries(mappings).forEach(([id, value]) => {
      const el = document.getElementById(id);
      if (el && value !== undefined) el.textContent = value;
    });

    const activityTable = document.getElementById('activity-table');
    if (activityTable && data.recent_activity && data.recent_activity.length > 0) {
      activityTable.innerHTML = data.recent_activity.reverse().map(entry => `
        <tr>
          <td>${new Date(entry.time).toLocaleTimeString()}</td>
          <td>${entry.type}</td>
          <td><span class="badge ${entry.status === 'success' ? 'badge-success' : 'badge-danger'}">${entry.status}</span></td>
          <td>${entry.details || '-'}</td>
        </tr>
      `).join('');
    }
  },

  async checkModelStatus() {
    try {
      const response = await fetch('/api/health');
      if (response.ok) {
        const data = await response.json();
        this.setModelStatus(data.model_loaded ? 'online' : 'offline');
      } else {
        this.setModelStatus('offline');
      }
    } catch {
      this.setModelStatus('offline');
    }
  },

  setModelStatus(status) {
    this.state.modelStatus = status;
    const dot = document.querySelector('.status-dot');
    const text = document.querySelector('.status-text');

    if (dot) {
      dot.className = 'status-dot';
      if (status === 'offline') dot.classList.add('offline');
      if (status === 'loading') dot.classList.add('loading');
    }

    if (text) {
      const labels = { online: 'Model Online', offline: 'Model Offline', loading: 'Loading...' };
      text.textContent = labels[status] || status;
    }
  },

  startPolling() {
    setInterval(() => {
      this.checkModelStatus();
      this.loadStats();
    }, 30000);
  },

  updateSliderDisplay(sliderId, displayId) {
    const slider = document.getElementById(sliderId);
    const display = document.getElementById(displayId);
    if (slider && display) {
      display.textContent = slider.value;
    }
  },

  bindFileUpload() {
    const fileInput = document.getElementById('qalam-file-input');
    if (fileInput) {
      fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) this.uploadQalamFile(e.target.files[0]);
      });
    }
    const dropZone = document.getElementById('drop-zone');
    if (dropZone) {
      dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.style.borderColor = '#D4AF37'; });
      dropZone.addEventListener('dragleave', () => { dropZone.style.borderColor = ''; });
      dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = '';
        if (e.dataTransfer.files.length > 0) this.uploadQalamFile(e.dataTransfer.files[0]);
      });
    }
  },

  async uploadQalamFile(file) {
    const statusEl = document.getElementById('import-status');
    const resultEl = document.getElementById('import-result');
    if (statusEl) statusEl.style.display = 'block';
    if (resultEl) resultEl.innerHTML = '<p>Importing ' + this.escapeHtml(file.name) + '...</p>';

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('/api/qalam-import', { method: 'POST', body: formData });
      const data = await response.json();

      if (data.status === 'error') {
        if (resultEl) resultEl.innerHTML = '<p style="color: #e74c3c;">Error: ' + this.escapeHtml(data.message) + '</p>';
      } else {
        const r = data.result || {};
        if (resultEl) resultEl.innerHTML = `
          <p style="color: #27ae60; font-weight: 600;">Import completed successfully!</p>
          <p>Accepted: ${r.accepted || 0} | Rejected: ${r.rejected || 0} | Duplicates: ${r.duplicates || 0}</p>
        `;
        this.loadBridgeStats();
        this.showNotification('Import completed: ' + (r.accepted || 0) + ' records accepted', 'success');
      }
    } catch (error) {
      if (resultEl) resultEl.innerHTML = '<p style="color: #e74c3c;">Upload failed: ' + error.message + '</p>';
    }
  },

  async loadBridgeStats() {
    try {
      const response = await fetch('/api/qalam-stats');
      if (!response.ok) return;
      const data = await response.json();

      const rawEl = document.getElementById('bridge-raw');
      const procEl = document.getElementById('bridge-processed');
      const qualEl = document.getElementById('bridge-quality');
      const catEl = document.getElementById('bridge-categories');

      if (rawEl) rawEl.textContent = data.raw_exports || 0;
      if (procEl) procEl.textContent = data.processed_records || 0;
      if (qualEl) qualEl.textContent = data.avg_quality != null ? data.avg_quality.toFixed(3) : '\u2014';

      if (catEl && data.categories && Object.keys(data.categories).length > 0) {
        catEl.innerHTML = Object.entries(data.categories).map(([cat, count]) =>
          `<div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
            <span style="color: var(--medium-gray);">${cat}</span>
            <span style="font-weight: 500;">${count}</span>
          </div>`
        ).join('');
      }
    } catch (error) {
      console.log('Bridge stats not available');
    }

    try {
      const whRes = await fetch('/api/qalam-webhook-stats');
      if (!whRes.ok) return;
      const wh = await whRes.json();

      const statusEl = document.getElementById('webhook-status');
      const totalEl = document.getElementById('webhook-total');
      const acceptedEl = document.getElementById('webhook-accepted');
      const rejectedEl = document.getElementById('webhook-rejected');
      const lastEl = document.getElementById('webhook-last');

      if (totalEl) totalEl.textContent = wh.total_received || 0;
      if (acceptedEl) acceptedEl.textContent = wh.accepted || 0;
      if (rejectedEl) rejectedEl.textContent = (wh.rejected || 0) + (wh.errors || 0);

      if (statusEl) {
        if (wh.total_received > 0) {
          statusEl.textContent = 'Connected';
          statusEl.style.color = '#27ae60';
        } else {
          statusEl.textContent = 'Waiting';
          statusEl.style.color = 'var(--medium-gray)';
        }
      }

      if (lastEl) {
        if (wh.last_received) {
          const d = new Date(wh.last_received);
          lastEl.textContent = d.toLocaleTimeString();
        } else {
          lastEl.textContent = 'Never';
        }
      }
    } catch (error) {
      console.log('Webhook stats not available');
    }
  }
};

document.addEventListener('DOMContentLoaded', () => {
  QalamDashboard.init();
});
