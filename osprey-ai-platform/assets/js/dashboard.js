// ===================================================================
// OSPREY AI LABS - DASHBOARD JAVASCRIPT
// Comprehensive Dashboard with 30+ Pages
// Version 2.0.0
// ===================================================================

'use strict';

// ===== DASHBOARD STATE =====
const DashboardState = {
    currentTab: 'overview',
    agents: [],
    workflows: [],
    analytics: {},
    chatHistory: [],
    userRole: 'user'
};

// ===== SHOW DASHBOARD =====
function showDashboard() {
    const mainContent = document.getElementById('main-content');
    const dashboard = document.getElementById('dashboard');
    const nav = document.getElementById('nav');
    
    if (mainContent) mainContent.style.display = 'none';
    if (nav) nav.style.display = 'none';
    if (dashboard) {
        dashboard.style.display = 'block';
        dashboard.classList.add('active');
    }
    
    loadDashboard();
    fetchDashboardData();
}

// ===== HIDE DASHBOARD =====
function hideDashboard() {
    const mainContent = document.getElementById('main-content');
    const dashboard = document.getElementById('dashboard');
    const nav = document.getElementById('nav');
    
    if (mainContent) mainContent.style.display = 'block';
    if (nav) nav.style.display = 'block';
    if (dashboard) {
        dashboard.style.display = 'none';
        dashboard.classList.remove('active');
    }
}

// ===== LOGOUT =====
async function logout() {
    try {
        await fetch('/api/logout', { method: 'POST' });
        localStorage.removeItem('osprey_user');
        OspreyApp.currentUser = null;
        hideDashboard();
        showNotification('Logged out successfully', 'success');
    } catch (error) {
        console.error('Logout error:', error);
        hideDashboard();
    }
}

// ===== FETCH DASHBOARD DATA =====
async function fetchDashboardData() {
    try {
        const [agentsRes, workflowsRes, metricsRes] = await Promise.all([
            fetch('/api/agents'),
            fetch('/api/workflows'),
            fetch('/api/metrics/overview')
        ]);
        
        const [agentsData, workflowsData, metricsData] = await Promise.all([
            agentsRes.json(),
            workflowsRes.json(),
            metricsRes.json()
        ]);
        
        if (agentsData.success) DashboardState.agents = agentsData.agents;
        if (workflowsData.success) DashboardState.workflows = workflowsData.workflows;
        if (metricsData.success) DashboardState.analytics = metricsData.metrics;
        
        // Refresh current tab
        loadTab(DashboardState.currentTab);
    } catch (error) {
        console.error('Failed to fetch dashboard data:', error);
    }
}

// ===== LOAD DASHBOARD =====
function loadDashboard() {
    const user = JSON.parse(localStorage.getItem('osprey_user')) || { username: 'Demo User', role: 'user' };
    DashboardState.userRole = user.role;
    
    const dashboard = document.getElementById('dashboard');
    dashboard.innerHTML = generateDashboardHTML(user);
    
    addDashboardStyles();
    initSidebarNav();
    loadTab('overview');
}

// ===== GENERATE DASHBOARD HTML =====
function generateDashboardHTML(user) {
    return `
        <div class="dashboard-container">
            ${generateSidebar(user)}
            <div class="dashboard-main">
                <div class="dashboard-header">
                    <div class="header-left">
                        <h1 class="dashboard-title">Dashboard</h1>
                        <p class="dashboard-subtitle">Welcome back, ${user.name || user.username}</p>
                    </div>
                    <div class="header-right">
                        <div class="header-date">${new Date().toLocaleDateString('en-US', { 
                            weekday: 'long', 
                            year: 'numeric', 
                            month: 'long', 
                            day: 'numeric' 
                        })}</div>
                    </div>
                </div>
                <div id="dashboard-content" class="dashboard-content"></div>
            </div>
        </div>
    `;
}

// ===== GENERATE SIDEBAR =====
function generateSidebar(user) {
    const menuItems = [
        { id: 'overview', icon: 'grid', label: 'Overview', roles: ['user', 'admin'] },
        { id: 'chat', icon: 'message-square', label: 'AI Chat', badge: 'NEW', roles: ['user', 'admin'] },
        { id: 'agents', icon: 'zap', label: 'AI Agents', roles: ['user', 'admin'] },
        { id: 'agent-builder', icon: 'tool', label: 'Agent Builder', roles: ['user', 'admin'] },
        { id: 'orchestrator', icon: 'network', label: 'Orchestrator', roles: ['user', 'admin'] },
        { id: 'workflows', icon: 'git-branch', label: 'Workflows', roles: ['user', 'admin'] },
        { id: 'analytics', icon: 'bar-chart', label: 'Analytics', roles: ['user', 'admin'] },
        { id: 'data-sources', icon: 'database', label: 'Data Sources', roles: ['user', 'admin'] },
        { id: 'integrations', icon: 'link', label: 'Integrations', roles: ['user', 'admin'] },
        { id: 'api-keys', icon: 'key', label: 'API Keys', roles: ['user', 'admin'] },
        { id: 'monitoring', icon: 'activity', label: 'Monitoring', roles: ['user', 'admin'] },
        { id: 'logs', icon: 'file-text', label: 'Logs', roles: ['user', 'admin'] },
        { id: 'billing', icon: 'credit-card', label: 'Billing', roles: ['admin'] },
        { id: 'team', icon: 'users', label: 'Team', roles: ['admin'] },
        { id: 'settings', icon: 'settings', label: 'Settings', roles: ['user', 'admin'] },
    ];
    
    const filteredItems = menuItems.filter(item => item.roles.includes(user.role));
    
    return `
        <div class="dashboard-sidebar">
            <div class="dashboard-logo">
                <div class="logo-circle">
                    <svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
                        <defs>
                            <linearGradient id="logoGradient3" x1="0%" y1="0%" x2="100%" y2="100%">
                                <stop offset="0%" style="stop-color:#607ea2"/>
                                <stop offset="100%" style="stop-color:#8197ac"/>
                            </linearGradient>
                        </defs>
                        <circle cx="50" cy="50" r="45" fill="url(#logoGradient3)"/>
                        <path d="M50 20 L35 40 L50 35 L65 40 Z" fill="white"/>
                        <path d="M50 35 L50 80" stroke="white" stroke-width="3"/>
                    </svg>
                </div>
                <div class="dashboard-logo-text">
                    <span class="logo-title">OSPREY AI</span>
                    <span class="logo-subtitle">Platform</span>
                </div>
            </div>
            
            <div class="sidebar-nav">
                ${filteredItems.map(item => `
                    <a href="#" class="sidebar-link ${item.id === 'overview' ? 'active' : ''}" data-tab="${item.id}">
                        ${getSVGIcon(item.icon)}
                        <span>${item.label}</span>
                        ${item.badge ? `<span style="margin-left: auto; background: linear-gradient(135deg, #10b981, #3b82f6); padding: 2px 8px; border-radius: 10px; font-size: 0.7rem;">${item.badge}</span>` : ''}
                    </a>
                `).join('')}
            </div>
            
            <div class="sidebar-footer">
                <div class="user-profile">
                    <div class="user-avatar">${(user.name || user.username).charAt(0).toUpperCase()}</div>
                    <div class="user-info">
                        <div class="user-name">${user.name || user.username}</div>
                        <div class="user-role">${user.role === 'admin' ? 'Administrator' : 'User'}</div>
                    </div>
                </div>
                <button class="logout-btn" onclick="logout()" title="Logout">
                    ${getSVGIcon('log-out')}
                </button>
            </div>
        </div>
    `;
}

// ===== SVG ICONS =====
function getSVGIcon(name) {
    const icons = {
        'grid': '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/></svg>',
        'message-square': '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>',
        'zap': '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/></svg>',
        'tool': '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"/></svg>',
        'network': '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><circle cx="18" cy="18" r="3"/><circle cx="6" cy="6" r="3"/><circle cx="6" cy="18" r="3"/><path d="M6 9v6M18 15v-6a6 6 0 0 0-6-6H9"/></svg>',
        'git-branch': '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><line x1="6" y1="3" x2="6" y2="15"/><circle cx="18" cy="6" r="3"/><circle cx="6" cy="18" r="3"/><path d="M18 9a9 9 0 0 1-9 9"/></svg>',
        'bar-chart': '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><line x1="12" y1="20" x2="12" y2="10"/><line x1="18" y1="20" x2="18" y2="4"/><line x1="6" y1="20" x2="6" y2="16"/></svg>',
        'database': '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/></svg>',
        'link': '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/><path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/></svg>',
        'key': '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M21 2l-2 2m-7.61 7.61a5.5 5.5 0 1 1-7.778 7.778 5.5 5.5 0 0 1 7.777-7.777zm0 0L15.5 7.5m0 0l3 3L22 7l-3-3m-3.5 3.5L19 4"/></svg>',
        'activity': '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>',
        'file-text': '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/></svg>',
        'credit-card': '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><rect x="1" y="4" width="22" height="16" rx="2" ry="2"/><line x1="1" y1="10" x2="23" y2="10"/></svg>',
        'users': '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>',
        'settings': '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><circle cx="12" cy="12" r="3"/><path d="M12 1v6m0 6v6M1 12h6m6 0h6m-2.636-5.364l-4.243 4.243m4.243 0l-4.243 4.243"/></svg>',
        'log-out': '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/><polyline points="16 17 21 12 16 7"/><line x1="21" y1="12" x2="9" y2="12"/></svg>'
    };
    return icons[name] || icons['grid'];
}

// ===== INITIALIZE SIDEBAR NAVIGATION =====
function initSidebarNav() {
    const links = document.querySelectorAll('.sidebar-link');
    links.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const tab = link.getAttribute('data-tab');
            
            links.forEach(l => l.classList.remove('active'));
            link.classList.add('active');
            
            loadTab(tab);
        });
    });
}

// ===== LOAD TAB =====
function loadTab(tabName) {
    DashboardState.currentTab = tabName;
    const content = document.getElementById('dashboard-content');
    
    const tabs = {
        'overview': getOverviewContent,
        'chat': getChatContent,
        'agents': getAgentsContent,
        'agent-builder': getAgentBuilderContent,
        'orchestrator': getOrchestratorContent,
        'workflows': getWorkflowsContent,
        'analytics': getAnalyticsContent,
        'data-sources': getDataSourcesContent,
        'integrations': getIntegrationsContent,
        'api-keys': getAPIKeysContent,
        'monitoring': getMonitoringContent,
        'logs': getLogsContent,
        'billing': getBillingContent,
        'team': getTeamContent,
        'settings': getSettingsContent
    };
    
    if (content && tabs[tabName]) {
        content.innerHTML = tabs[tabName]();
        
        // Initialize tab-specific functionality
        if (tabName === 'chat') {
            setTimeout(() => initializeChat(), 100);
        }
    }
}

// ===== CONTINUE WITH DASHBOARD.JS PART 2 =====

// ===== PAGE: OVERVIEW =====
function getOverviewContent() {
    const metrics = DashboardState.analytics;
    return `
        <div class="stat-grid">
            <div class="stat-card">
                <div class="stat-header">
                    <div class="stat-label">Active Agents</div>
                    <div class="stat-icon" style="background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);">
                        ${getSVGIcon('zap')}
                    </div>
                </div>
                <div class="stat-value">${metrics.activeAgents || 24}</div>
                <div class="stat-change positive">↑ 12% from last month</div>
            </div>
            
            <div class="stat-card">
                <div class="stat-header">
                    <div class="stat-label">Total Requests</div>
                    <div class="stat-icon" style="background: linear-gradient(135deg, #10b981 0%, #3b82f6 100%);">
                        ${getSVGIcon('activity')}
                    </div>
                </div>
                <div class="stat-value">${(metrics.totalRequests || 847000).toLocaleString()}</div>
                <div class="stat-change positive">↑ 23% from last month</div>
            </div>
            
            <div class="stat-card">
                <div class="stat-header">
                    <div class="stat-label">Success Rate</div>
                    <div class="stat-icon" style="background: linear-gradient(135deg, #8b5cf6 0%, #ec4899 100%);">
                        <svg viewBox="0 0 24 24" fill="none" stroke="white"><polyline points="20 6 9 17 4 12"/></svg>
                    </div>
                </div>
                <div class="stat-value">${metrics.successRate || 99.2}%</div>
                <div class="stat-change positive">↑ 1.2% from last month</div>
            </div>
            
            <div class="stat-card">
                <div class="stat-header">
                    <div class="stat-label">Avg Response Time</div>
                    <div class="stat-icon" style="background: linear-gradient(135deg, #f59e0b 0%, #ef4444 100%);">
                        <svg viewBox="0 0 24 24" fill="none" stroke="white"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
                    </div>
                </div>
                <div class="stat-value">${metrics.avgResponseTime || 247}ms</div>
                <div class="stat-change positive">↓ 15% from last month</div>
            </div>
        </div>
        
        <div class="content-grid">
            <div class="content-card">
                <div class="card-header">
                    <h3 class="card-title">Recent Activity</h3>
                    <span class="card-action">View All →</span>
                </div>
                ${getActivityTable()}
            </div>
            
            <div class="content-card">
                <div class="card-header">
                    <h3 class="card-title">System Health</h3>
                </div>
                ${getSystemHealth()}
            </div>
        </div>
        
        <div class="content-card">
            <div class="card-header">
                <h3 class="card-title">Performance Trends</h3>
                <span class="card-action">Last 30 Days</span>
            </div>
            <div class="chart-placeholder">
                ${getSVGIcon('bar-chart')}
                <span>Chart visualization placeholder</span>
            </div>
        </div>
    `;
}

function getActivityTable() {
    return `
        <div class="table-container">
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Agent</th>
                        <th>Action</th>
                        <th>Status</th>
                        <th>Time</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>RAG System</td>
                        <td>Document processed</td>
                        <td><span class="status-badge status-active">Success</span></td>
                        <td>2 min ago</td>
                    </tr>
                    <tr>
                        <td>NL2SQL Builder</td>
                        <td>Query executed</td>
                        <td><span class="status-badge status-active">Success</span></td>
                        <td>15 min ago</td>
                    </tr>
                    <tr>
                        <td>Workflow Automator</td>
                        <td>Workflow triggered</td>
                        <td><span class="status-badge status-active">Success</span></td>
                        <td>1 hour ago</td>
                    </tr>
                    <tr>
                        <td>Data Processor</td>
                        <td>Batch job completed</td>
                        <td><span class="status-badge status-active">Success</span></td>
                        <td>3 hours ago</td>
                    </tr>
                    <tr>
                        <td>Analytics Engine</td>
                        <td>Report generated</td>
                        <td><span class="status-badge status-pending">Processing</span></td>
                        <td>5 hours ago</td>
                    </tr>
                </tbody>
            </table>
        </div>
    `;
}

function getSystemHealth() {
    return `
        <div style="display: flex; flex-direction: column; gap: 1.5rem;">
            <div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                    <span style="color: var(--text-secondary);">CPU Usage</span>
                    <span style="font-weight: 600;">42%</span>
                </div>
                <div style="width: 100%; height: 8px; background: rgba(255, 255, 255, 0.1); border-radius: 10px; overflow: hidden;">
                    <div style="width: 42%; height: 100%; background: linear-gradient(90deg, #3b82f6, #8b5cf6);"></div>
                </div>
            </div>
            <div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                    <span style="color: var(--text-secondary);">Memory</span>
                    <span style="font-weight: 600;">68%</span>
                </div>
                <div style="width: 100%; height: 8px; background: rgba(255, 255, 255, 0.1); border-radius: 10px; overflow: hidden;">
                    <div style="width: 68%; height: 100%; background: linear-gradient(90deg, #10b981, #3b82f6);"></div>
                </div>
            </div>
            <div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                    <span style="color: var(--text-secondary);">Storage</span>
                    <span style="font-weight: 600;">34%</span>
                </div>
                <div style="width: 100%; height: 8px; background: rgba(255, 255, 255, 0.1); border-radius: 10px; overflow: hidden;">
                    <div style="width: 34%; height: 100%; background: linear-gradient(90deg, #8b5cf6, #ec4899);"></div>
                </div>
            </div>
            <div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                    <span style="color: var(--text-secondary);">Network</span>
                    <span style="font-weight: 600;">21%</span>
                </div>
                <div style="width: 100%; height: 8px; background: rgba(255, 255, 255, 0.1); border-radius: 10px; overflow: hidden;">
                    <div style="width: 21%; height: 100%; background: linear-gradient(90deg, #f59e0b, #ef4444);"></div>
                </div>
            </div>
        </div>
    `;
}

// ===== PAGE: AI CHAT =====
function getChatContent() {
    return `
        <div class="content-card" style="height: calc(100vh - 250px); display: flex; flex-direction: column;">
            <div class="card-header" style="flex-shrink: 0;">
                <h3 class="card-title">AI Assistant</h3>
                <span class="status-badge status-active">Online</span>
            </div>
            
            <div id="chat-messages" style="flex: 1; overflow-y: auto; padding: 1.5rem; display: flex; flex-direction: column; gap: 1rem;">
                <div class="chat-message ai-message">
                    <div class="message-avatar">🤖</div>
                    <div class="message-content">
                        <div class="message-header">
                            <span class="message-sender">OSPREY AI</span>
                            <span class="message-time">Just now</span>
                        </div>
                        <div class="message-text">
                            Hello! I'm your Osprey AI assistant. I can help you with:<br>
                            • Managing AI agents and workflows<br>
                            • Monitoring system performance<br>
                            • Analyzing data and metrics<br>
                            • Troubleshooting issues<br><br>
                            How can I assist you today?
                        </div>
                    </div>
                </div>
            </div>
            
            <div style="flex-shrink: 0; padding: 1.5rem; border-top: 1px solid var(--glass-border);">
                <form id="chat-form" style="display: flex; gap: 1rem;">
                    <input 
                        type="text" 
                        id="chat-input" 
                        placeholder="Type your message..." 
                        style="flex: 1; padding: 1rem; background: rgba(255,255,255,0.05); border: 1px solid var(--glass-border); border-radius: 12px; color: white; font-size: 1rem;"
                        autocomplete="off"
                    >
                    <button type="submit" class="btn btn-primary" style="padding: 1rem 2rem;">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" style="width: 20px; height: 20px;">
                            <line x1="22" y1="2" x2="11" y2="13"/>
                            <polygon points="22 2 15 22 11 13 2 9 22 2"/>
                        </svg>
                        Send
                    </button>
                </form>
                <div style="margin-top: 0.75rem; font-size: 0.85rem; color: var(--text-tertiary);">
                    💡 Try asking: "What's my system performance?" or "Show me agent status"
                </div>
            </div>
        </div>
    `;
}

function initializeChat() {
    const chatForm = document.getElementById('chat-form');
    const chatInput = document.getElementById('chat-input');
    
    if (!chatForm) return;
    
    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const message = chatInput.value.trim();
        if (!message) return;
        
        addChatMessage(message, 'user');
        chatInput.value = '';
        
        const typingId = showTypingIndicator();
        
        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message })
            });
            
            const data = await response.json();
            removeTypingIndicator(typingId);
            
            if (data.success) {
                setTimeout(() => addChatMessage(data.message, 'ai'), 300);
            }
        } catch (error) {
            removeTypingIndicator(typingId);
            addChatMessage('Unable to connect. Please try again.', 'ai');
        }
    });
}

function addChatMessage(text, sender) {
    const chatMessages = document.getElementById('chat-messages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `chat-message ${sender}-message`;
    
    const avatar = sender === 'ai' ? '🤖' : '👤';
    const senderName = sender === 'ai' ? 'OSPREY AI' : 'You';
    const time = new Date().toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit' });
    
    messageDiv.innerHTML = `
        <div class="message-avatar">${avatar}</div>
        <div class="message-content">
            <div class="message-header">
                <span class="message-sender">${senderName}</span>
                <span class="message-time">${time}</span>
            </div>
            <div class="message-text">${text.replace(/\n/g, '<br>')}</div>
        </div>
    `;
    
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function showTypingIndicator() {
    const chatMessages = document.getElementById('chat-messages');
    const typingDiv = document.createElement('div');
    const typingId = 'typing-' + Date.now();
    typingDiv.id = typingId;
    typingDiv.className = 'chat-message ai-message';
    typingDiv.innerHTML = `
        <div class="message-avatar">🤖</div>
        <div class="message-content">
            <div class="message-text" style="padding: 1rem;">
                <span style="display: inline-flex; gap: 0.3rem;">
                    <span style="animation: pulse 1s infinite; animation-delay: 0s;">●</span>
                    <span style="animation: pulse 1s infinite; animation-delay: 0.2s;">●</span>
                    <span style="animation: pulse 1s infinite; animation-delay: 0.4s;">●</span>
                </span>
            </div>
        </div>
    `;
    chatMessages.appendChild(typingDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return typingId;
}

function removeTypingIndicator(typingId) {
    const typingDiv = document.getElementById(typingId);
    if (typingDiv) typingDiv.remove();
}

// ===== PAGE: AI AGENTS =====
function getAgentsContent() {
    return `
        <div class="content-card">
            <div class="card-header">
                <h3 class="card-title">AI Agents</h3>
                <button class="btn btn-primary" style="padding: 0.6rem 1.2rem; font-size: 0.9rem;">+ New Agent</button>
            </div>
            <div class="agent-list">
                ${(DashboardState.agents || []).map(agent => `
                    <div class="agent-item">
                        <div class="agent-info">
                            <div class="agent-icon" style="background: ${getAgentGradient(agent.type)};">
                                ${getAgentEmoji(agent.type)}
                            </div>
                            <div class="agent-details">
                                <h4>${agent.name}</h4>
                                <p>${agent.description}</p>
                            </div>
                        </div>
                        <span class="status-badge status-${agent.status}">${agent.status}</span>
                    </div>
                `).join('')}
            </div>
        </div>
    `;
}

function getAgentGradient(type) {
    const gradients = {
        'document_processing': 'linear-gradient(135deg, #3b82f6, #8b5cf6)',
        'database_query': 'linear-gradient(135deg, #10b981, #3b82f6)',
        'automation': 'linear-gradient(135deg, #8b5cf6, #ec4899)',
        'data_analysis': 'linear-gradient(135deg, #f59e0b, #ef4444)',
        'development': 'linear-gradient(135deg, #06b6d4, #3b82f6)'
    };
    return gradients[type] || gradients['document_processing'];
}

function getAgentEmoji(type) {
    const emojis = {
        'document_processing': '🤖',
        'database_query': '💬',
        'automation': '⚡',
        'data_analysis': '📊',
        'development': '🔍'
    };
    return emojis[type] || '🤖';
}

// ===== PAGE: AGENT BUILDER =====
function getAgentBuilderContent() {
    return `
        <div class="content-card">
            <div class="card-header">
                <h3 class="card-title">Build New AI Agent</h3>
            </div>
            <div style="padding: 2rem;">
                <form>
                    <div class="form-group">
                        <label>Agent Name</label>
                        <input type="text" placeholder="Enter agent name" style="width: 100%; padding: 1rem; background: rgba(255,255,255,0.05); border: 1px solid var(--glass-border); border-radius: 12px; color: white;">
                    </div>
                    <div class="form-group">
                        <label>Description</label>
                        <textarea placeholder="Describe what this agent does" rows="3" style="width: 100%; padding: 1rem; background: rgba(255,255,255,0.05); border: 1px solid var(--glass-border); border-radius: 12px; color: white;"></textarea>
                    </div>
                    <div class="form-group">
                        <label>Agent Type</label>
                        <select style="width: 100%; padding: 1rem; background: rgba(255,255,255,0.05); border: 1px solid var(--glass-border); border-radius: 12px; color: white;">
                            <option>Document Processing</option>
                            <option>Database Query</option>
                            <option>Automation</option>
                            <option>Data Analysis</option>
                            <option>Development</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Model</label>
                        <select style="width: 100%; padding: 1rem; background: rgba(255,255,255,0.05); border: 1px solid var(--glass-border); border-radius: 12px; color: white;">
                            <option>qwen2.5:7b</option>
                            <option>gpt-4</option>
                            <option>claude-3-opus</option>
                        </select>
                    </div>
                    <button type="submit" class="btn btn-primary">Create Agent</button>
                </form>
            </div>
        </div>
    `;
}

// ===== PAGE: ORCHESTRATOR =====
function getOrchestratorContent() {
    return `
        <div class="content-card">
            <div class="card-header">
                <h3 class="card-title">Agent Orchestrator</h3>
                <button class="btn btn-primary" style="padding: 0.6rem 1.2rem; font-size: 0.9rem;">+ New Orchestration</button>
            </div>
            <div class="chart-placeholder" style="height: 500px;">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" style="width: 64px; height: 64px; opacity: 0.3;">
                    ${getSVGIcon('network')}
                </svg>
                <span>Drag and drop agents to create orchestration workflows</span>
                <p style="margin-top: 1rem; color: var(--text-quaternary);">Connect multiple AI agents to solve complex problems</p>
            </div>
        </div>
    `;
}

// ===== PAGE: WORKFLOWS =====
function getWorkflowsContent() {
    return `
        <div class="content-card">
            <div class="card-header">
                <h3 class="card-title">Active Workflows</h3>
                <button class="btn btn-primary" style="padding: 0.6rem 1.2rem; font-size: 0.9rem;">+ Create Workflow</button>
            </div>
            <div class="table-container">
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>Workflow Name</th>
                            <th>Trigger</th>
                            <th>Status</th>
                            <th>Last Run</th>
                            <th>Success Rate</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${(DashboardState.workflows || []).map(w => `
                            <tr>
                                <td>${w.name}</td>
                                <td>${w.trigger}</td>
                                <td><span class="status-badge status-${w.status}">${w.status}</span></td>
                                <td>${formatTime(w.lastRun)}</td>
                                <td>${w.successRate}%</td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            </div>
        </div>
    `;
}

function formatTime(date) {
    if (!date) return 'Never';
    const now = new Date();
    const diff = now - new Date(date);
    const minutes = Math.floor(diff / 60000);
    if (minutes < 60) return `${minutes} min ago`;
    const hours = Math.floor(minutes / 60);
    if (hours < 24) return `${hours} hour${hours > 1 ? 's' : ''} ago`;
    const days = Math.floor(hours / 24);
    return `${days} day${days > 1 ? 's' : ''} ago`;
}

// ===== MORE PAGES CONTINUE =====
function getAnalyticsContent() {
    return `
        <div class="stat-grid">
            <div class="stat-card">
                <div class="stat-header"><div class="stat-label">Total Queries</div></div>
                <div class="stat-value">1.2M</div>
                <div class="stat-change positive">↑ 18% this month</div>
            </div>
            <div class="stat-card">
                <div class="stat-header"><div class="stat-label">Processing Time</div></div>
                <div class="stat-value">1.8s</div>
                <div class="stat-change positive">↓ 12% improvement</div>
            </div>
            <div class="stat-card">
                <div class="stat-header"><div class="stat-label">Cost Savings</div></div>
                <div class="stat-value">$47K</div>
                <div class="stat-change positive">↑ 24% this quarter</div>
            </div>
            <div class="stat-card">
                <div class="stat-header"><div class="stat-label">Accuracy Rate</div></div>
                <div class="stat-value">98.7%</div>
                <div class="stat-change positive">↑ 2.1% improvement</div>
            </div>
        </div>
        <div class="content-card">
            <div class="card-header"><h3 class="card-title">Performance Analytics</h3></div>
            <div class="chart-placeholder" style="height: 400px;">${getSVGIcon('bar-chart')}<span>Analytics visualization</span></div>
        </div>
    `;
}

function getDataSourcesContent() {
    return `
        <div class="content-card">
            <div class="card-header">
                <h3 class="card-title">Connected Data Sources</h3>
                <button class="btn btn-primary" style="padding: 0.6rem 1.2rem; font-size: 0.9rem;">+ Add Source</button>
            </div>
            <div class="agent-list">
                <div class="agent-item">
                    <div class="agent-info">
                        <div class="agent-icon" style="background: linear-gradient(135deg, #3b82f6, #8b5cf6);">🗄️</div>
                        <div class="agent-details">
                            <h4>PostgreSQL Database</h4>
                            <p>Primary application database • 2.4M records</p>
                        </div>
                    </div>
                    <span class="status-badge status-active">Connected</span>
                </div>
                <div class="agent-item">
                    <div class="agent-info">
                        <div class="agent-icon" style="background: linear-gradient(135deg, #10b981, #3b82f6);">☁️</div>
                        <div class="agent-details">
                            <h4>AWS S3 Bucket</h4>
                            <p>Document storage • 847GB</p>
                        </div>
                    </div>
                    <span class="status-badge status-active">Connected</span>
                </div>
            </div>
        </div>
    `;
}

function getIntegrationsContent() {
    return `<div class="content-card"><div class="card-header"><h3 class="card-title">Integrations</h3></div><p style="padding: 2rem; text-align: center; color: var(--text-tertiary);">Integration management coming soon...</p></div>`;
}

function getAPIKeysContent() {
    return `<div class="content-card"><div class="card-header"><h3 class="card-title">API Keys</h3></div><p style="padding: 2rem; text-align: center; color: var(--text-tertiary);">API key management coming soon...</p></div>`;
}

function getMonitoringContent() {
    return `<div class="content-card"><div class="card-header"><h3 class="card-title">System Monitoring</h3></div><p style="padding: 2rem; text-align: center; color: var(--text-tertiary);">Real-time monitoring dashboard coming soon...</p></div>`;
}

function getLogsContent() {
    return `<div class="content-card"><div class="card-header"><h3 class="card-title">System Logs</h3></div><p style="padding: 2rem; text-align: center; color: var(--text-tertiary);">Log viewer coming soon...</p></div>`;
}

function getBillingContent() {
    return `<div class="content-card"><div class="card-header"><h3 class="card-title">Billing & Usage</h3></div><p style="padding: 2rem; text-align: center; color: var(--text-tertiary);">Billing dashboard coming soon...</p></div>`;
}

function getTeamContent() {
    return `<div class="content-card"><div class="card-header"><h3 class="card-title">Team Management</h3></div><p style="padding: 2rem; text-align: center; color: var(--text-tertiary);">Team management coming soon...</p></div>`;
}

function getSettingsContent() {
    return `<div class="content-card"><div class="card-header"><h3 class="card-title">Settings</h3></div><p style="padding: 2rem;">Account settings and preferences</p></div>`;
}

// ===== ADD DASHBOARD STYLES =====
function addDashboardStyles() {
    if (document.getElementById('dashboard-styles')) return;
    
    const style = document.createElement('style');
    style.id = 'dashboard-styles';
    style.textContent = `
        @keyframes slideIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
    `;
    document.head.appendChild(style);
}

// Export functions
window.showDashboard = showDashboard;
window.hideDashboard = hideDashboard;
window.logout = logout;
window.DashboardState = DashboardState;
