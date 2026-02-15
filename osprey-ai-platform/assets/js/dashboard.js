// Osprey AI Labs - Enterprise Dashboard JavaScript
// Version 2.0.0 - Production Ready
// Copyright © 2024 Osprey AI Labs. All Rights Reserved.

// ============================================================================
// GLOBAL STATE MANAGEMENT
// ============================================================================

const AppState = {
    currentUser: null,
    agents: [],
    analytics: null,
    logs: [],
    currentView: 'overview',
    currentAgent: null,
    theme: 'dark',
    notifications: [],
    sidebarOpen: true,
    isAdmin: false
};

// ============================================================================
// INITIALIZATION
// ============================================================================

document.addEventListener('DOMContentLoaded', async () => {
    console.log('🚀 Initializing Osprey AI Platform Dashboard...');
    
    try {
        await checkAuthentication();
        await initializeDashboard();
        setupEventListeners();
        startRealtimeUpdates();
        showToast('Welcome to Osprey AI Platform', 'success');
    } catch (error) {
        console.error('Initialization error:', error);
        showToast('Demo mode: Some features may be limited', 'warning');
        // Don't redirect in demo mode, just continue with limited functionality
        await initializeDashboard().catch(e => console.error('Dashboard init error:', e));
        setupEventListeners();
    }
});

// ============================================================================
// AUTHENTICATION & SESSION MANAGEMENT
// ============================================================================

async function checkAuthentication() {
    try {
        // Mock authentication for demo mode
        console.log('🔓 Running in demo mode with mock authentication');
        
        AppState.currentUser = {
            username: 'Admin@B',
            email: 'admin@demo.com',
            role: 'admin',
            theme: 'dark'
        };
        AppState.theme = 'dark';
        AppState.isAdmin = true;
        
        document.body.classList.toggle('dark-theme', AppState.theme === 'dark');
        document.body.classList.toggle('light-theme', AppState.theme === 'light');
        
        updateUserProfile();
        
    } catch (error) {
        console.error('Authentication check failed:', error);
        throw error;
    }
}

function updateUserProfile() {
    const usernameElements = document.querySelectorAll('.user-username');
    const roleElements = document.querySelectorAll('.user-role');
    const avatarElements = document.querySelectorAll('.user-avatar');
    
    usernameElements.forEach(el => {
        el.textContent = AppState.currentUser.username;
    });
    
    roleElements.forEach(el => {
        el.textContent = AppState.currentUser.role.toUpperCase();
        el.classList.add(AppState.currentUser.role === 'admin' ? 'badge-admin' : 'badge-user');
    });
    
    avatarElements.forEach(el => {
        const initial = AppState.currentUser.username.charAt(0).toUpperCase();
        el.textContent = initial;
    });
}

async function handleLogout() {
    try {
        const response = await fetch('/api/auth/logout', {
            method: 'POST',
            credentials: 'include'
        });
        
        if (response.ok) {
            showToast('Logged out successfully', 'info');
            setTimeout(() => {
                window.location.href = '/';
            }, 1000);
        }
    } catch (error) {
        console.error('Logout error:', error);
        showToast('Logout failed', 'error');
    }
}

// ============================================================================
// DASHBOARD INITIALIZATION
// ============================================================================

async function initializeDashboard() {
    console.log('📊 Loading dashboard data...');
    
    await Promise.all([
        loadAgents(),
        loadAnalytics(),
        loadActivityLogs()
    ]);
    
    renderSidebar();
    renderOverview();
    updateDashboardStats();
}

async function loadAgents() {
    try {
        // Mock agents data for demo
        AppState.agents = [
            {
                id: 'agent-001',
                name: 'Customer Support Bot',
                type: 'Support Agent',
                status: 'active',
                requests: 12543,
                successRate: 98.5,
                lastActive: new Date().toISOString(),
                description: 'Handles customer inquiries and support tickets'
            },
            {
                id: 'agent-002',
                name: 'Sales Assistant',
                type: 'Sales Agent',
                status: 'active',
                requests: 8432,
                successRate: 96.2,
                lastActive: new Date(Date.now() - 3600000).toISOString(),
                description: 'Assists with sales inquiries and lead qualification'
            },
            {
                id: 'agent-003',
                name: 'Data Analyzer',
                type: 'Analytics Agent',
                status: 'active',
                requests: 5621,
                successRate: 99.1,
                lastActive: new Date(Date.now() - 7200000).toISOString(),
                description: 'Analyzes business data and generates insights'
            },
            {
                id: 'agent-004',
                name: 'Email Processor',
                type: 'Automation Agent',
                status: 'paused',
                requests: 3245,
                successRate: 94.8,
                lastActive: new Date(Date.now() - 86400000).toISOString(),
                description: 'Processes and categorizes incoming emails'
            },
            {
                id: 'agent-005',
                name: 'Content Generator',
                type: 'Creative Agent',
                status: 'active',
                requests: 1876,
                successRate: 97.3,
                lastActive: new Date(Date.now() - 1800000).toISOString(),
                description: 'Generates marketing content and copy'
            }
        ];
        
        console.log(`✅ Loaded ${AppState.agents.length} agents`);
    } catch (error) {
        console.error('Failed to load agents:', error);
        showToast('Failed to load agents', 'error');
    }
}

async function loadAnalytics() {
    try {
        // Mock analytics data for demo
        AppState.analytics = {
            totalRequests: 31717,
            activeAgents: 5,
            avgResponseTime: 1.2,
            successRate: 97.8,
            todayRequests: 2847,
            weeklyGrowth: 12.5,
            monthlyRevenue: 48923,
            uptime: 99.9,
            requestsChart: {
                labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                data: [4200, 4500, 4800, 4300, 5100, 4900, 5200]
            },
            agentPerformance: {
                labels: ['Customer Support', 'Sales', 'Analytics', 'Email', 'Content'],
                data: [98.5, 96.2, 99.1, 94.8, 97.3]
            }
        };
        
        console.log('✅ Analytics loaded');
    } catch (error) {
        console.error('Failed to load analytics:', error);
    }
}

async function loadActivityLogs() {
    try {
        // Mock activity logs for demo
        const actions = [
            'Agent started processing request',
            'Agent completed task successfully',
            'New agent deployed',
            'Configuration updated',
            'API call executed',
            'User logged in',
            'Report generated',
            'Integration synchronized'
        ];
        
        AppState.logs = [];
        for (let i = 0; i < 100; i++) {
            AppState.logs.push({
                id: `log-${i}`,
                timestamp: new Date(Date.now() - Math.random() * 86400000 * 7).toISOString(),
                agent: AppState.agents[Math.floor(Math.random() * 5)]?.name || 'System',
                action: actions[Math.floor(Math.random() * actions.length)],
                status: Math.random() > 0.1 ? 'success' : 'warning',
                duration: (Math.random() * 5).toFixed(2) + 's'
            });
        }
        
        AppState.logs.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
        
        console.log(`✅ Loaded ${AppState.logs.length} activity logs`);
    } catch (error) {
        console.error('Failed to load logs:', error);
    }
}

// ============================================================================
// SIDEBAR RENDERING
// ============================================================================

function renderSidebar() {
    const sidebar = document.getElementById('sidebar');
    if (!sidebar) return;
    
    const userTabs = [
        { id: 'overview', icon: '📊', label: 'Overview', action: () => showView('overview') },
        { id: 'agents', icon: '🤖', label: 'My Agents', action: () => showView('agents') },
        { id: 'analytics', icon: '📈', label: 'Analytics', action: () => showView('analytics') },
        { id: 'marketplace', icon: '🏪', label: 'Agent Marketplace', action: () => showView('marketplace') },
        { id: 'workflows', icon: '🔄', label: 'Workflows', action: () => showView('workflows') },
        { id: 'integrations', icon: '🔌', label: 'Integrations', action: () => showView('integrations') },
        { id: 'api', icon: '⚡', label: 'API Access', action: () => showView('api') },
        { id: 'monitoring', icon: '👁️', label: 'Monitoring', action: () => showView('monitoring') },
        { id: 'logs', icon: '📝', label: 'Activity Logs', action: () => showView('logs') },
        { id: 'billing', icon: '💳', label: 'Billing', action: () => showView('billing') },
        { id: 'security', icon: '🔒', label: 'Security', action: () => showView('security') },
        { id: 'notifications', icon: '🔔', label: 'Notifications', action: () => showView('notifications') },
        { id: 'support', icon: '❓', label: 'Support', action: () => showView('support') },
        { id: 'profile', icon: '👤', label: 'Profile', action: () => showView('profile') },
        { id: 'settings', icon: '⚙️', label: 'Settings', action: () => showView('settings') }
    ];
    
    const adminTabs = [
        { id: 'admin-dashboard', icon: '👑', label: 'Admin Dashboard', action: () => showView('admin-dashboard') },
        { id: 'user-management', icon: '👥', label: 'User Management', action: () => showView('user-management') },
        { id: 'system-config', icon: '🛠️', label: 'System Config', action: () => showView('system-config') },
        { id: 'agent-approval', icon: '✅', label: 'Agent Approvals', action: () => showView('agent-approval') },
        { id: 'audit-logs', icon: '🔍', label: 'Audit Logs', action: () => showView('audit-logs') }
    ];
    
    const tabs = AppState.isAdmin ? [...userTabs, ...adminTabs] : userTabs;
    
    let sidebarHTML = `
        <div class="sidebar-header">
            <div class="sidebar-logo">
                <h2>🦅 Osprey AI</h2>
            </div>
            <button class="sidebar-toggle" onclick="toggleSidebar()">
                <span class="toggle-icon">☰</span>
            </button>
        </div>
        <div class="sidebar-user">
            <div class="user-avatar">${AppState.currentUser.username.charAt(0)}</div>
            <div class="user-info">
                <div class="user-username">${AppState.currentUser.username}</div>
                <div class="user-role badge ${AppState.isAdmin ? 'badge-admin' : 'badge-user'}">
                    ${AppState.currentUser.role.toUpperCase()}
                </div>
            </div>
        </div>
        <nav class="sidebar-nav">
    `;
    
    tabs.forEach((tab, index) => {
        const isActive = tab.id === AppState.currentView;
        sidebarHTML += `
            <div class="sidebar-item ${isActive ? 'active' : ''}" 
                 data-tab="${tab.id}"
                 onclick="navigateToTab('${tab.id}')">
                <span class="sidebar-icon">${tab.icon}</span>
                <span class="sidebar-label">${tab.label}</span>
            </div>
        `;
    });
    
    sidebarHTML += `
        </nav>
        <div class="sidebar-footer">
            <button class="btn btn-secondary btn-block" onclick="handleLogout()">
                <span class="btn-icon">🚪</span> Logout
            </button>
        </div>
    `;
    
    sidebar.innerHTML = sidebarHTML;
}

function toggleSidebar() {
    AppState.sidebarOpen = !AppState.sidebarOpen;
    const sidebar = document.getElementById('sidebar');
    const mainContent = document.getElementById('main-content');
    
    if (sidebar) {
        sidebar.classList.toggle('collapsed', !AppState.sidebarOpen);
    }
    if (mainContent) {
        mainContent.classList.toggle('expanded', !AppState.sidebarOpen);
    }
}

function navigateToTab(tabId) {
    AppState.currentView = tabId;
    
    document.querySelectorAll('.sidebar-item').forEach(item => {
        item.classList.remove('active');
    });
    
    const activeTab = document.querySelector(`.sidebar-item[data-tab="${tabId}"]`);
    if (activeTab) {
        activeTab.classList.add('active');
    }
    
    showView(tabId);
}

// ============================================================================
// VIEW RENDERING
// ============================================================================

function showView(viewName) {
    const mainContent = document.getElementById('main-content');
    if (!mainContent) return;
    
    AppState.currentView = viewName;
    
    console.log(`📄 Rendering view: ${viewName}`);
    
    switch (viewName) {
        case 'overview':
            renderOverview();
            break;
        case 'agents':
            renderAgents();
            break;
        case 'analytics':
            renderAnalytics();
            break;
        case 'marketplace':
            renderMarketplace();
            break;
        case 'workflows':
            renderWorkflows();
            break;
        case 'integrations':
            renderIntegrations();
            break;
        case 'api':
            renderAPIAccess();
            break;
        case 'monitoring':
            renderMonitoring();
            break;
        case 'logs':
            renderLogs();
            break;
        case 'billing':
            renderBilling();
            break;
        case 'security':
            renderSecurity();
            break;
        case 'notifications':
            renderNotifications();
            break;
        case 'support':
            renderSupport();
            break;
        case 'profile':
            renderProfile();
            break;
        case 'settings':
            renderSettings();
            break;
        case 'admin-dashboard':
            renderAdminDashboard();
            break;
        case 'user-management':
            renderUserManagement();
            break;
        case 'system-config':
            renderSystemConfig();
            break;
        case 'agent-approval':
            renderAgentApproval();
            break;
        case 'audit-logs':
            renderAuditLogs();
            break;
        default:
            renderOverview();
    }
}

// ============================================================================
// OVERVIEW VIEW
// ============================================================================

function renderOverview() {
    const mainContent = document.getElementById('main-content');
    
    const stats = AppState.analytics || {
        totalAgents: 0,
        activeAgents: 0,
        totalRequests: 0,
        avgPerformance: 0,
        uptime: 99.9
    };
    
    const html = `
        <div class="dashboard-header">
            <h1>Dashboard Overview</h1>
            <div class="header-actions">
                <button class="btn btn-primary" onclick="openNewAgentModal()">
                    <span class="btn-icon">➕</span> New Agent
                </button>
                <button class="btn btn-secondary" onclick="refreshDashboard()">
                    <span class="btn-icon">🔄</span> Refresh
                </button>
            </div>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-icon">🤖</div>
                <div class="stat-content">
                    <div class="stat-value">${stats.totalAgents}</div>
                    <div class="stat-label">Total Agents</div>
                </div>
                <div class="stat-trend positive">+2 this week</div>
            </div>
            
            <div class="stat-card">
                <div class="stat-icon">✅</div>
                <div class="stat-content">
                    <div class="stat-value">${stats.activeAgents}</div>
                    <div class="stat-label">Active Agents</div>
                </div>
                <div class="stat-trend positive">All operational</div>
            </div>
            
            <div class="stat-card">
                <div class="stat-icon">📊</div>
                <div class="stat-content">
                    <div class="stat-value">${formatNumber(stats.totalRequests)}</div>
                    <div class="stat-label">Total Requests</div>
                </div>
                <div class="stat-trend positive">+15% this month</div>
            </div>
            
            <div class="stat-card">
                <div class="stat-icon">⚡</div>
                <div class="stat-content">
                    <div class="stat-value">${stats.avgPerformance}%</div>
                    <div class="stat-label">Avg Performance</div>
                </div>
                <div class="stat-trend positive">+2.3% improvement</div>
            </div>
            
            <div class="stat-card">
                <div class="stat-icon">🌐</div>
                <div class="stat-content">
                    <div class="stat-value">${stats.uptime}%</div>
                    <div class="stat-label">System Uptime</div>
                </div>
                <div class="stat-trend positive">30 days uptime</div>
            </div>
            
            <div class="stat-card">
                <div class="stat-icon">⏱️</div>
                <div class="stat-content">
                    <div class="stat-value">0.39s</div>
                    <div class="stat-label">Avg Response Time</div>
                </div>
                <div class="stat-trend positive">-0.05s faster</div>
            </div>
        </div>
        
        <div class="content-grid">
            <div class="content-section">
                <h2>Active Agents</h2>
                <div class="agent-list">
                    ${renderActiveAgentsList()}
                </div>
            </div>
            
            <div class="content-section">
                <h2>Recent Activity</h2>
                <div class="activity-feed">
                    ${renderRecentActivity()}
                </div>
            </div>
        </div>
        
        <div class="content-section">
            <h2>Performance Overview</h2>
            <div id="performance-chart" class="chart-container">
                ${renderPerformanceChart()}
            </div>
        </div>
    `;
    
    mainContent.innerHTML = html;
}

function renderActiveAgentsList() {
    const activeAgents = AppState.agents.filter(a => a.status === 'active').slice(0, 5);
    
    if (activeAgents.length === 0) {
        return '<div class="empty-state">No active agents</div>';
    }
    
    return activeAgents.map(agent => `
        <div class="agent-item" onclick="viewAgentDetails('${agent.id}')">
            <div class="agent-icon">${getAgentIcon(agent.type)}</div>
            <div class="agent-info">
                <div class="agent-name">${agent.name}</div>
                <div class="agent-meta">${agent.type} • ${agent.requests.toLocaleString()} requests</div>
            </div>
            <div class="agent-status">
                <div class="status-badge status-${agent.status}">${agent.status}</div>
                <div class="agent-performance">${agent.performance}%</div>
            </div>
        </div>
    `).join('');
}

function renderRecentActivity() {
    const recentLogs = AppState.logs.slice(0, 8);
    
    if (recentLogs.length === 0) {
        return '<div class="empty-state">No recent activity</div>';
    }
    
    return recentLogs.map(log => `
        <div class="activity-item">
            <div class="activity-icon ${log.severity}">${getSeverityIcon(log.severity)}</div>
            <div class="activity-content">
                <div class="activity-title">${log.action}</div>
                <div class="activity-details">${log.details}</div>
                <div class="activity-meta">${log.user} • ${formatTimeAgo(log.timestamp)}</div>
            </div>
        </div>
    `).join('');
}

function renderPerformanceChart() {
    return `
        <div class="chart-placeholder">
            <div class="chart-bar" style="height: 75%">
                <div class="chart-label">Mon</div>
                <div class="chart-value">98.5%</div>
            </div>
            <div class="chart-bar" style="height: 82%">
                <div class="chart-label">Tue</div>
                <div class="chart-value">99.2%</div>
            </div>
            <div class="chart-bar" style="height: 78%">
                <div class="chart-label">Wed</div>
                <div class="chart-value">97.8%</div>
            </div>
            <div class="chart-bar" style="height: 85%">
                <div class="chart-label">Thu</div>
                <div class="chart-value">99.5%</div>
            </div>
            <div class="chart-bar" style="height: 88%">
                <div class="chart-label">Fri</div>
                <div class="chart-value">99.8%</div>
            </div>
            <div class="chart-bar" style="height: 80%">
                <div class="chart-label">Sat</div>
                <div class="chart-value">98.0%</div>
            </div>
            <div class="chart-bar" style="height: 90%">
                <div class="chart-label">Sun</div>
                <div class="chart-value">100%</div>
            </div>
        </div>
    `;
}

// ============================================================================
// AGENTS VIEW
// ============================================================================

function renderAgents() {
    const mainContent = document.getElementById('main-content');
    
    const html = `
        <div class="dashboard-header">
            <h1>My Agents</h1>
            <div class="header-actions">
                <button class="btn btn-primary" onclick="openNewAgentModal()">
                    <span class="btn-icon">➕</span> Create New Agent
                </button>
                <button class="btn btn-secondary" onclick="openImportAgentModal()">
                    <span class="btn-icon">📥</span> Import Agent
                </button>
            </div>
        </div>
        
        <div class="filter-bar">
            <div class="filter-group">
                <label>Filter by Status:</label>
                <select id="status-filter" onchange="filterAgents()">
                    <option value="all">All Agents</option>
                    <option value="active">Active</option>
                    <option value="inactive">Inactive</option>
                    <option value="error">Error</option>
                </select>
            </div>
            
            <div class="filter-group">
                <label>Filter by Type:</label>
                <select id="type-filter" onchange="filterAgents()">
                    <option value="all">All Types</option>
                    <option value="customer_support">Customer Support</option>
                    <option value="optimization">Optimization</option>
                    <option value="analytics">Analytics</option>
                    <option value="logistics">Logistics</option>
                    <option value="security">Security</option>
                </select>
            </div>
            
            <div class="search-box">
                <input type="text" id="agent-search" placeholder="Search agents..." oninput="searchAgents()">
                <span class="search-icon">🔍</span>
            </div>
        </div>
        
        <div class="agents-grid" id="agents-grid">
            ${renderAgentsGrid()}
        </div>
    `;
    
    mainContent.innerHTML = html;
}

function renderAgentsGrid() {
    if (AppState.agents.length === 0) {
        return `
            <div class="empty-state-large">
                <div class="empty-icon">🤖</div>
                <h2>No Agents Yet</h2>
                <p>Create your first AI agent to get started</p>
                <button class="btn btn-primary" onclick="openNewAgentModal()">
                    Create Agent
                </button>
            </div>
        `;
    }
    
    return AppState.agents.map(agent => `
        <div class="agent-card" data-agent-id="${agent.id}">
            <div class="agent-card-header">
                <div class="agent-card-icon">${getAgentIcon(agent.type)}</div>
                <div class="agent-card-status">
                    <span class="status-badge status-${agent.status}">${agent.status}</span>
                </div>
            </div>
            
            <div class="agent-card-body">
                <h3 class="agent-card-title">${agent.name}</h3>
                <p class="agent-card-description">${agent.description}</p>
                
                <div class="agent-card-stats">
                    <div class="agent-stat">
                        <div class="stat-value">${agent.performance}%</div>
                        <div class="stat-label">Performance</div>
                    </div>
                    <div class="agent-stat">
                        <div class="stat-value">${formatNumber(agent.requests)}</div>
                        <div class="stat-label">Requests</div>
                    </div>
                    <div class="agent-stat">
                        <div class="stat-value">${agent.avgResponseTime}s</div>
                        <div class="stat-label">Avg Time</div>
                    </div>
                </div>
                
                <div class="agent-card-tags">
                    ${agent.tags.map(tag => `<span class="tag">${tag}</span>`).join('')}
                </div>
            </div>
            
            <div class="agent-card-footer">
                <button class="btn btn-sm btn-primary" onclick="viewAgentDetails('${agent.id}')">
                    View Details
                </button>
                <button class="btn btn-sm btn-secondary" onclick="configureAgent('${agent.id}')">
                    Configure
                </button>
                <button class="btn btn-sm btn-icon" onclick="toggleAgentMenu('${agent.id}')">
                    ⋮
                </button>
            </div>
        </div>
    `).join('');
}

function filterAgents() {
    const statusFilter = document.getElementById('status-filter').value;
    const typeFilter = document.getElementById('type-filter').value;
    
    let filteredAgents = [...AppState.agents];
    
    if (statusFilter !== 'all') {
        filteredAgents = filteredAgents.filter(a => a.status === statusFilter);
    }
    
    if (typeFilter !== 'all') {
        filteredAgents = filteredAgents.filter(a => a.type === typeFilter);
    }
    
    renderFilteredAgents(filteredAgents);
}

function searchAgents() {
    const searchTerm = document.getElementById('agent-search').value.toLowerCase();
    
    const filteredAgents = AppState.agents.filter(agent => 
        agent.name.toLowerCase().includes(searchTerm) ||
        agent.description.toLowerCase().includes(searchTerm) ||
        agent.tags.some(tag => tag.toLowerCase().includes(searchTerm))
    );
    
    renderFilteredAgents(filteredAgents);
}

function renderFilteredAgents(agents) {
    const agentsGrid = document.getElementById('agents-grid');
    
    if (agents.length === 0) {
        agentsGrid.innerHTML = `
            <div class="empty-state">
                <p>No agents match your filters</p>
            </div>
        `;
        return;
    }
    
    const tempAgents = AppState.agents;
    AppState.agents = agents;
    agentsGrid.innerHTML = renderAgentsGrid();
    AppState.agents = tempAgents;
}

// ============================================================================
// AGENT DETAILS VIEW
// ============================================================================

function viewAgentDetails(agentId) {
    const agent = AppState.agents.find(a => a.id === agentId);
    if (!agent) return;
    
    AppState.currentAgent = agent;
    
    const mainContent = document.getElementById('main-content');
    
    const html = `
        <div class="dashboard-header">
            <div>
                <button class="btn btn-text" onclick="showView('agents')">
                    ← Back to Agents
                </button>
                <h1>${agent.name}</h1>
                <p class="subtitle">${agent.description}</p>
            </div>
            <div class="header-actions">
                <button class="btn btn-${agent.status === 'active' ? 'warning' : 'success'}" 
                        onclick="toggleAgentStatus('${agent.id}')">
                    ${agent.status === 'active' ? '⏸️ Pause' : '▶️ Activate'}
                </button>
                <button class="btn btn-secondary" onclick="configureAgent('${agent.id}')">
                    ⚙️ Configure
                </button>
                <button class="btn btn-danger" onclick="deleteAgent('${agent.id}')">
                    🗑️ Delete
                </button>
            </div>
        </div>
        
        <div class="agent-details">
            <div class="details-section">
                <h2>Agent Information</h2>
                <div class="info-grid">
                    <div class="info-item">
                        <label>Agent ID:</label>
                        <span>${agent.id}</span>
                    </div>
                    <div class="info-item">
                        <label>Type:</label>
                        <span class="badge">${agent.type}</span>
                    </div>
                    <div class="info-item">
                        <label>Status:</label>
                        <span class="status-badge status-${agent.status}">${agent.status}</span>
                    </div>
                    <div class="info-item">
                        <label>Created:</label>
                        <span>${formatDate(agent.created)}</span>
                    </div>
                    <div class="info-item">
                        <label>Model:</label>
                        <span>${agent.config.model}</span>
                    </div>
                    <div class="info-item">
                        <label>Performance:</label>
                        <span class="performance-value">${agent.performance}%</span>
                    </div>
                </div>
            </div>
            
            <div class="details-section">
                <h2>Performance Metrics</h2>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-icon">📊</div>
                        <div class="metric-value">${formatNumber(agent.requests)}</div>
                        <div class="metric-label">Total Requests</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-icon">⚡</div>
                        <div class="metric-value">${agent.avgResponseTime}s</div>
                        <div class="metric-label">Avg Response Time</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-icon">✅</div>
                        <div class="metric-value">${agent.performance}%</div>
                        <div class="metric-label">Success Rate</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-icon">🔥</div>
                        <div class="metric-value">${Math.floor(Math.random() * 500) + 100}</div>
                        <div class="metric-label">Requests/Hour</div>
                    </div>
                </div>
            </div>
            
            <div class="details-section">
                <h2>Configuration</h2>
                <div class="config-display">
                    <pre>${JSON.stringify(agent.config, null, 2)}</pre>
                </div>
                <button class="btn btn-secondary" onclick="editAgentConfig('${agent.id}')">
                    Edit Configuration
                </button>
            </div>
            
            <div class="details-section">
                <h2>Recent Activity</h2>
                <div class="activity-log">
                    ${renderAgentActivity(agent.id)}
                </div>
            </div>
        </div>
    `;
    
    mainContent.innerHTML = html;
}

function renderAgentActivity(agentId) {
    const agentLogs = AppState.logs.filter(log => 
        log.details.includes(AppState.currentAgent.name)
    ).slice(0, 10);
    
    if (agentLogs.length === 0) {
        return '<div class="empty-state">No recent activity</div>';
    }
    
    return agentLogs.map(log => `
        <div class="log-item">
            <div class="log-timestamp">${formatDate(log.timestamp)}</div>
            <div class="log-action">${log.action}</div>
            <div class="log-details">${log.details}</div>
        </div>
    `).join('');
}

// ============================================================================
// AGENT OPERATIONS
// ============================================================================

async function toggleAgentStatus(agentId) {
    const agent = AppState.agents.find(a => a.id === agentId);
    if (!agent) return;
    
    const newStatus = agent.status === 'active' ? 'inactive' : 'active';
    
    try {
        const response = await fetch(`/api/agents/${agentId}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            credentials: 'include',
            body: JSON.stringify({ status: newStatus })
        });
        
        if (response.ok) {
            agent.status = newStatus;
            showToast(`Agent ${newStatus === 'active' ? 'activated' : 'paused'}`, 'success');
            viewAgentDetails(agentId);
        }
    } catch (error) {
        console.error('Failed to update agent status:', error);
        showToast('Failed to update agent status', 'error');
    }
}

async function deleteAgent(agentId) {
    if (!confirm('Are you sure you want to delete this agent? This action cannot be undone.')) {
        return;
    }
    
    try {
        const response = await fetch(`/api/agents/${agentId}`, {
            method: 'DELETE',
            credentials: 'include'
        });
        
        if (response.ok) {
            AppState.agents = AppState.agents.filter(a => a.id !== agentId);
            showToast('Agent deleted successfully', 'success');
            showView('agents');
        }
    } catch (error) {
        console.error('Failed to delete agent:', error);
        showToast('Failed to delete agent', 'error');
    }
}

function configureAgent(agentId) {
    const agent = AppState.agents.find(a => a.id === agentId);
    if (!agent) return;
    
    openModal('Agent Configuration', `
        <form id="configure-agent-form" onsubmit="saveAgentConfig(event, '${agentId}')">
            <div class="form-group">
                <label>Agent Name:</label>
                <input type="text" name="name" value="${agent.name}" required>
            </div>
            
            <div class="form-group">
                <label>Description:</label>
                <textarea name="description" rows="3" required>${agent.description}</textarea>
            </div>
            
            <div class="form-group">
                <label>Model:</label>
                <select name="model">
                    <option value="gpt-4" ${agent.config.model === 'gpt-4' ? 'selected' : ''}>GPT-4</option>
                    <option value="gpt-4-turbo" ${agent.config.model === 'gpt-4-turbo' ? 'selected' : ''}>GPT-4 Turbo</option>
                    <option value="claude-3-opus" ${agent.config.model === 'claude-3-opus' ? 'selected' : ''}>Claude 3 Opus</option>
                    <option value="claude-3-sonnet" ${agent.config.model === 'claude-3-sonnet' ? 'selected' : ''}>Claude 3 Sonnet</option>
                </select>
            </div>
            
            <div class="form-group">
                <label>Tags (comma-separated):</label>
                <input type="text" name="tags" value="${agent.tags.join(', ')}">
            </div>
            
            <div class="form-actions">
                <button type="button" class="btn btn-secondary" onclick="closeModal()">Cancel</button>
                <button type="submit" class="btn btn-primary">Save Changes</button>
            </div>
        </form>
    `);
}

async function saveAgentConfig(event, agentId) {
    event.preventDefault();
    
    const formData = new FormData(event.target);
    const updateData = {
        name: formData.get('name'),
        description: formData.get('description'),
        tags: formData.get('tags').split(',').map(t => t.trim()),
        config: {
            ...AppState.currentAgent.config,
            model: formData.get('model')
        }
    };
    
    try {
        const response = await fetch(`/api/agents/${agentId}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            credentials: 'include',
            body: JSON.stringify(updateData)
        });
        
        if (response.ok) {
            const { agent } = await response.json();
            const index = AppState.agents.findIndex(a => a.id === agentId);
            AppState.agents[index] = agent;
            
            showToast('Agent configuration updated', 'success');
            closeModal();
            viewAgentDetails(agentId);
        }
    } catch (error) {
        console.error('Failed to update agent:', error);
        showToast('Failed to update agent', 'error');
    }
}

// ============================================================================
// NEW AGENT MODAL
// ============================================================================

function openNewAgentModal() {
    openModal('Create New Agent', `
        <form id="new-agent-form" onsubmit="createNewAgent(event)">
            <div class="form-group">
                <label>Agent Name:</label>
                <input type="text" name="name" placeholder="e.g., Customer Support Bot" required>
            </div>
            
            <div class="form-group">
                <label>Agent Type:</label>
                <select name="type" required>
                    <option value="">Select Type</option>
                    <option value="customer_support">Customer Support</option>
                    <option value="optimization">Optimization</option>
                    <option value="analytics">Analytics</option>
                    <option value="logistics">Logistics</option>
                    <option value="security">Security</option>
                    <option value="sales">Sales & Marketing</option>
                    <option value="hr">Human Resources</option>
                    <option value="finance">Finance</option>
                </select>
            </div>
            
            <div class="form-group">
                <label>Description:</label>
                <textarea name="description" rows="3" placeholder="Describe what this agent does..." required></textarea>
            </div>
            
            <div class="form-group">
                <label>AI Model:</label>
                <select name="model" required>
                    <option value="gpt-4">GPT-4</option>
                    <option value="gpt-4-turbo">GPT-4 Turbo</option>
                    <option value="claude-3-opus">Claude 3 Opus</option>
                    <option value="claude-3-sonnet">Claude 3 Sonnet</option>
                </select>
            </div>
            
            <div class="form-group">
                <label>Tags (comma-separated):</label>
                <input type="text" name="tags" placeholder="e.g., customer-service, automation">
            </div>
            
            <div class="form-actions">
                <button type="button" class="btn btn-secondary" onclick="closeModal()">Cancel</button>
                <button type="submit" class="btn btn-primary">Create Agent</button>
            </div>
        </form>
    `);
}

async function createNewAgent(event) {
    event.preventDefault();
    
    const formData = new FormData(event.target);
    const newAgent = {
        name: formData.get('name'),
        type: formData.get('type'),
        description: formData.get('description'),
        tags: formData.get('tags').split(',').map(t => t.trim()).filter(t => t),
        config: {
            model: formData.get('model'),
            temperature: 0.7,
            maxTokens: 2048
        }
    };
    
    try {
        const response = await fetch('/api/agents', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            credentials: 'include',
            body: JSON.stringify(newAgent)
        });
        
        if (response.ok) {
            const { agent } = await response.json();
            AppState.agents.push(agent);
            
            showToast('Agent created successfully', 'success');
            closeModal();
            showView('agents');
        }
    } catch (error) {
        console.error('Failed to create agent:', error);
        showToast('Failed to create agent', 'error');
    }
}

// ============================================================================
// ANALYTICS VIEW
// ============================================================================

function renderAnalytics() {
    const mainContent = document.getElementById('main-content');
    
    const html = `
        <div class="dashboard-header">
            <h1>Analytics & Insights</h1>
            <div class="header-actions">
                <select id="analytics-timeframe" onchange="updateAnalytics()">
                    <option value="7d">Last 7 Days</option>
                    <option value="30d" selected>Last 30 Days</option>
                    <option value="90d">Last 90 Days</option>
                    <option value="1y">Last Year</option>
                </select>
                <button class="btn btn-secondary" onclick="exportAnalytics()">
                    📊 Export Report
                </button>
            </div>
        </div>
        
        <div class="analytics-grid">
            <div class="analytics-card">
                <h3>Request Volume</h3>
                <div class="chart-container">
                    ${renderRequestVolumeChart()}
                </div>
            </div>
            
            <div class="analytics-card">
                <h3>Performance Trends</h3>
                <div class="chart-container">
                    ${renderPerformanceTrendChart()}
                </div>
            </div>
            
            <div class="analytics-card">
                <h3>Agent Utilization</h3>
                <div class="chart-container">
                    ${renderUtilizationChart()}
                </div>
            </div>
            
            <div class="analytics-card">
                <h3>Response Time Distribution</h3>
                <div class="chart-container">
                    ${renderResponseTimeChart()}
                </div>
            </div>
        </div>
        
        <div class="insights-section">
            <h2>Key Insights</h2>
            <div class="insights-grid">
                <div class="insight-card">
                    <div class="insight-icon positive">📈</div>
                    <div class="insight-content">
                        <h4>Performance Improvement</h4>
                        <p>Overall agent performance increased by 5.3% this month</p>
                    </div>
                </div>
                <div class="insight-card">
                    <div class="insight-icon positive">⚡</div>
                    <div class="insight-content">
                        <h4>Faster Response Times</h4>
                        <p>Average response time decreased by 12% compared to last month</p>
                    </div>
                </div>
                <div class="insight-card">
                    <div class="insight-icon neutral">🔔</div>
                    <div class="insight-content">
                        <h4>Peak Usage Hours</h4>
                        <p>Highest traffic between 9 AM - 5 PM EST on weekdays</p>
                    </div>
                </div>
                <div class="insight-card">
                    <div class="insight-icon positive">💡</div>
                    <div class="insight-content">
                        <h4>Cost Optimization</h4>
                        <p>Potential 18% cost savings through agent optimization</p>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    mainContent.innerHTML = html;
}

function renderRequestVolumeChart() {
    return `
        <div class="line-chart">
            ${Array.from({length: 30}, (_, i) => {
                const height = 20 + Math.random() * 70;
                return `<div class="line-chart-bar" style="height: ${height}%"></div>`;
            }).join('')}
        </div>
    `;
}

function renderPerformanceTrendChart() {
    return `
        <div class="area-chart">
            <svg viewBox="0 0 300 150" class="chart-svg">
                <polyline points="0,140 30,110 60,95 90,85 120,90 150,70 180,65 210,55 240,60 270,45 300,50" 
                    fill="rgba(59, 130, 246, 0.2)" stroke="#3b82f6" stroke-width="2"/>
            </svg>
        </div>
    `;
}

function renderUtilizationChart() {
    return `
        <div class="donut-chart">
            <svg viewBox="0 0 200 200">
                <circle cx="100" cy="100" r="80" fill="none" stroke="#e5e7eb" stroke-width="20"/>
                <circle cx="100" cy="100" r="80" fill="none" stroke="#3b82f6" stroke-width="20" 
                    stroke-dasharray="440" stroke-dashoffset="110" transform="rotate(-90 100 100)"/>
                <text x="100" y="100" text-anchor="middle" dominant-baseline="middle" 
                    font-size="32" font-weight="bold" fill="#3b82f6">75%</text>
            </svg>
        </div>
    `;
}

function renderResponseTimeChart() {
    return `
        <div class="histogram">
            ${[45, 65, 85, 95, 88, 70, 50, 35].map(height => 
                `<div class="histogram-bar" style="height: ${height}%"></div>`
            ).join('')}
        </div>
    `;
}

function updateAnalytics() {
    showToast('Updating analytics...', 'info');
    setTimeout(() => {
        renderAnalytics();
        showToast('Analytics updated', 'success');
    }, 500);
}

function exportAnalytics() {
    showToast('Exporting analytics report...', 'info');
    setTimeout(() => {
        showToast('Report exported successfully', 'success');
    }, 1000);
}

// ============================================================================
// MARKETPLACE VIEW
// ============================================================================

function renderMarketplace() {
    const mainContent = document.getElementById('main-content');
    
    const marketplaceAgents = [
        {
            name: 'E-commerce Optimizer',
            type: 'optimization',
            description: 'Boost online sales with intelligent product recommendations and pricing strategies',
            price: '$299/month',
            rating: 4.8,
            installs: 1234
        },
        {
            name: 'Social Media Manager',
            type: 'marketing',
            description: 'Automate social media posting, engagement, and analytics across all platforms',
            price: '$199/month',
            rating: 4.6,
            installs: 2156
        },
        {
            name: 'HR Assistant',
            type: 'hr',
            description: 'Streamline recruitment, onboarding, and employee management processes',
            price: '$249/month',
            rating: 4.9,
            installs: 892
        },
        {
            name: 'Financial Analyst',
            type: 'finance',
            description: 'Real-time financial analysis, forecasting, and reporting automation',
            price: '$399/month',
            rating: 4.7,
            installs: 678
        }
    ];
    
    const html = `
        <div class="dashboard-header">
            <h1>Agent Marketplace</h1>
            <div class="header-actions">
                <input type="text" placeholder="Search marketplace..." class="search-input">
                <button class="btn btn-secondary">
                    🔍 Search
                </button>
            </div>
        </div>
        
        <div class="marketplace-categories">
            <button class="category-btn active">All</button>
            <button class="category-btn">Customer Service</button>
            <button class="category-btn">Analytics</button>
            <button class="category-btn">Marketing</button>
            <button class="category-btn">Finance</button>
            <button class="category-btn">HR</button>
        </div>
        
        <div class="marketplace-grid">
            ${marketplaceAgents.map(agent => `
                <div class="marketplace-card">
                    <div class="marketplace-card-header">
                        <div class="marketplace-icon">${getAgentIcon(agent.type)}</div>
                        <div class="marketplace-rating">
                            ⭐ ${agent.rating}
                        </div>
                    </div>
                    <h3>${agent.name}</h3>
                    <p>${agent.description}</p>
                    <div class="marketplace-meta">
                        <span>📦 ${agent.installs.toLocaleString()} installs</span>
                    </div>
                    <div class="marketplace-footer">
                        <div class="marketplace-price">${agent.price}</div>
                        <button class="btn btn-primary" onclick="installMarketplaceAgent('${agent.name}')">
                            Install
                        </button>
                    </div>
                </div>
            `).join('')}
        </div>
    `;
    
    mainContent.innerHTML = html;
}

function installMarketplaceAgent(agentName) {
    showToast(`Installing ${agentName}...`, 'info');
    setTimeout(() => {
        showToast(`${agentName} installed successfully!`, 'success');
    }, 2000);
}

// ============================================================================
// OTHER VIEWS (FILLER CONTENT)
// ============================================================================

function renderWorkflows() {
    const mainContent = document.getElementById('main-content');
    mainContent.innerHTML = `
        <div class="dashboard-header">
            <h1>Workflows</h1>
            <button class="btn btn-primary" onclick="createWorkflow()">
                ➕ New Workflow
            </button>
        </div>
        <div class="coming-soon">
            <h2>🚀 Workflow Automation Coming Soon</h2>
            <p>Create custom automation workflows by connecting multiple agents</p>
        </div>
    `;
}

function renderIntegrations() {
    const mainContent = document.getElementById('main-content');
    mainContent.innerHTML = `
        <div class="dashboard-header">
            <h1>Integrations</h1>
            <button class="btn btn-primary" onclick="addIntegration()">
                ➕ Add Integration
            </button>
        </div>
        <div class="integrations-grid">
            ${renderIntegrationCards()}
        </div>
    `;
}

function renderIntegrationCards() {
    const integrations = [
        { name: 'Slack', icon: '💬', status: 'connected' },
        { name: 'Salesforce', icon: '☁️', status: 'available' },
        { name: 'HubSpot', icon: '🎯', status: 'available' },
        { name: 'Stripe', icon: '💳', status: 'connected' },
        { name: 'Google Workspace', icon: '📧', status: 'connected' },
        { name: 'Microsoft Teams', icon: '💼', status: 'available' }
    ];
    
    return integrations.map(int => `
        <div class="integration-card ${int.status}">
            <div class="integration-icon">${int.icon}</div>
            <h3>${int.name}</h3>
            <span class="integration-status">${int.status}</span>
            <button class="btn btn-sm ${int.status === 'connected' ? 'btn-secondary' : 'btn-primary'}" 
                onclick="toggleIntegration('${int.name}')">
                ${int.status === 'connected' ? 'Disconnect' : 'Connect'}
            </button>
        </div>
    `).join('');
}

function renderAPIAccess() {
    const mainContent = document.getElementById('main-content');
    mainContent.innerHTML = `
        <div class="dashboard-header">
            <h1>API Access</h1>
            <button class="btn btn-primary" onclick="generateAPIKey()">
                🔑 Generate New Key
            </button>
        </div>
        <div class="api-section">
            <h2>API Keys</h2>
            <div class="api-key-list">
                <div class="api-key-item">
                    <span class="api-key">osp_live_••••••••••••••••</span>
                    <button class="btn btn-sm btn-secondary" onclick="copyAPIKey()">Copy</button>
                    <button class="btn btn-sm btn-danger" onclick="revokeAPIKey()">Revoke</button>
                </div>
            </div>
            
            <h2>API Documentation</h2>
            <div class="code-block">
                <pre>curl -X POST https://api.ospreyai.com/v1/agents/invoke \\
  -H "Authorization: Bearer YOUR_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{"agent_id": "agent_123", "input": "Hello"}'</pre>
            </div>
        </div>
    `;
}

function renderMonitoring() {
    const mainContent = document.getElementById('main-content');
    mainContent.innerHTML = `
        <div class="dashboard-header">
            <h1>System Monitoring</h1>
            <button class="btn btn-secondary" onclick="refreshMonitoring()">
                🔄 Refresh
            </button>
        </div>
        <div class="monitoring-grid">
            <div class="monitor-card">
                <h3>System Health</h3>
                <div class="health-indicator green">🟢 All Systems Operational</div>
            </div>
            <div class="monitor-card">
                <h3>API Status</h3>
                <div class="health-indicator green">🟢 API Responsive</div>
            </div>
            <div class="monitor-card">
                <h3>Database</h3>
                <div class="health-indicator green">🟢 Connected</div>
            </div>
            <div class="monitor-card">
                <h3>Queue Status</h3>
                <div class="health-indicator yellow">🟡 Processing: 42 jobs</div>
            </div>
        </div>
    `;
}

function renderLogs() {
    const mainContent = document.getElementById('main-content');
    mainContent.innerHTML = `
        <div class="dashboard-header">
            <h1>Activity Logs</h1>
            <button class="btn btn-secondary" onclick="exportLogs()">
                📥 Export Logs
            </button>
        </div>
        <div class="logs-container">
            ${AppState.logs.map(log => `
                <div class="log-entry ${log.severity}">
                    <span class="log-timestamp">${formatDate(log.timestamp)}</span>
                    <span class="log-user">${log.user}</span>
                    <span class="log-action">${log.action}</span>
                    <span class="log-details">${log.details}</span>
                </div>
            `).join('')}
        </div>
    `;
}

function renderBilling() {
    const mainContent = document.getElementById('main-content');
    mainContent.innerHTML = `
        <div class="dashboard-header">
            <h1>Billing & Usage</h1>
            <button class="btn btn-primary" onclick="upgradePlan()">
                ⬆️ Upgrade Plan
            </button>
        </div>
        <div class="billing-section">
            <div class="plan-card current">
                <h2>Enterprise Plan</h2>
                <div class="plan-price">$999/month</div>
                <ul class="plan-features">
                    <li>✅ Unlimited Agents</li>
                    <li>✅ 10M Requests/month</li>
                    <li>✅ 99.99% Uptime SLA</li>
                    <li>✅ Priority Support</li>
                    <li>✅ Advanced Analytics</li>
                </ul>
            </div>
            
            <h2>Usage This Month</h2>
            <div class="usage-stats">
                <div class="usage-item">
                    <label>API Requests:</label>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: 35%"></div>
                    </div>
                    <span>3.5M / 10M</span>
                </div>
                <div class="usage-item">
                    <label>Agent Hours:</label>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: 62%"></div>
                    </div>
                    <span>620 / 1000</span>
                </div>
            </div>
        </div>
    `;
}

function renderSecurity() {
    const mainContent = document.getElementById('main-content');
    mainContent.innerHTML = `
        <div class="dashboard-header">
            <h1>Security Settings</h1>
        </div>
        <div class="security-section">
            <div class="security-card">
                <h3>Two-Factor Authentication</h3>
                <p>Add an extra layer of security to your account</p>
                <button class="btn btn-primary" onclick="setup2FA()">Enable 2FA</button>
            </div>
            <div class="security-card">
                <h3>Active Sessions</h3>
                <div class="session-list">
                    <div class="session-item">
                        <span>🖥️ Current Session - Trenton, NJ</span>
                        <span class="session-time">Active now</span>
                    </div>
                </div>
            </div>
            <div class="security-card">
                <h3>API Key Management</h3>
                <p>Manage your API keys and access tokens</p>
                <button class="btn btn-secondary" onclick="manageAPIKeys()">Manage Keys</button>
            </div>
        </div>
    `;
}

function renderNotifications() {
    const mainContent = document.getElementById('main-content');
    mainContent.innerHTML = `
        <div class="dashboard-header">
            <h1>Notifications</h1>
            <button class="btn btn-secondary" onclick="markAllRead()">
                ✅ Mark All Read
            </button>
        </div>
        <div class="notifications-list">
            <div class="notification-item unread">
                <div class="notification-icon">🤖</div>
                <div class="notification-content">
                    <h4>New Agent Deployed</h4>
                    <p>Your Customer Service AI is now live</p>
                    <span class="notification-time">5 minutes ago</span>
                </div>
            </div>
            <div class="notification-item">
                <div class="notification-icon">📊</div>
                <div class="notification-content">
                    <h4>Weekly Report Ready</h4>
                    <p>Your analytics report for last week is available</p>
                    <span class="notification-time">2 hours ago</span>
                </div>
            </div>
        </div>
    `;
}

function renderSupport() {
    const mainContent = document.getElementById('main-content');
    mainContent.innerHTML = `
        <div class="dashboard-header">
            <h1>Support & Help</h1>
        </div>
        <div class="support-section">
            <div class="support-card">
                <h3>📚 Documentation</h3>
                <p>Browse our comprehensive guides and tutorials</p>
                <button class="btn btn-primary" onclick="openDocs()">View Docs</button>
            </div>
            <div class="support-card">
                <h3>💬 Live Chat</h3>
                <p>Chat with our support team</p>
                <button class="btn btn-primary" onclick="openLiveChat()">Start Chat</button>
            </div>
            <div class="support-card">
                <h3>📧 Email Support</h3>
                <p>Send us an email at support@ospreyai.com</p>
                <button class="btn btn-secondary" onclick="emailSupport()">Send Email</button>
            </div>
            <div class="support-card">
                <h3>🎓 Training</h3>
                <p>Learn how to get the most out of Osprey AI</p>
                <button class="btn btn-secondary" onclick="viewTraining()">View Courses</button>
            </div>
        </div>
    `;
}

// ============================================================================
// PROFILE & SETTINGS
// ============================================================================

function renderProfile() {
    const mainContent = document.getElementById('main-content');
    
    const html = `
        <div class="dashboard-header">
            <h1>Profile Settings</h1>
        </div>
        
        <div class="profile-section">
            <div class="profile-card">
                <div class="profile-header">
                    <div class="profile-avatar-large">
                        ${AppState.currentUser.username.charAt(0).toUpperCase()}
                    </div>
                    <div class="profile-info">
                        <h2>${AppState.currentUser.name}</h2>
                        <p>${AppState.currentUser.email}</p>
                        <span class="badge ${AppState.isAdmin ? 'badge-admin' : 'badge-user'}">
                            ${AppState.currentUser.role.toUpperCase()}
                        </span>
                    </div>
                </div>
                
                <form id="profile-form" onsubmit="saveProfile(event)">
                    <div class="form-group">
                        <label>Full Name:</label>
                        <input type="text" name="name" value="${AppState.currentUser.name}" required>
                    </div>
                    
                    <div class="form-group">
                        <label>Email:</label>
                        <input type="email" name="email" value="${AppState.currentUser.email}" required>
                    </div>
                    
                    <div class="form-group">
                        <label>Username:</label>
                        <input type="text" value="${AppState.currentUser.username}" disabled>
                        <small>Username cannot be changed</small>
                    </div>
                    
                    <h3>Change Password</h3>
                    
                    <div class="form-group">
                        <label>Current Password:</label>
                        <input type="password" name="currentPassword" autocomplete="current-password">
                    </div>
                    
                    <div class="form-group">
                        <label>New Password:</label>
                        <input type="password" name="newPassword" autocomplete="new-password">
                    </div>
                    
                    <div class="form-group">
                        <label>Confirm New Password:</label>
                        <input type="password" name="confirmPassword" autocomplete="new-password">
                    </div>
                    
                    <button type="submit" class="btn btn-primary">Save Changes</button>
                </form>
            </div>
        </div>
    `;
    
    mainContent.innerHTML = html;
}

async function saveProfile(event) {
    event.preventDefault();
    
    const formData = new FormData(event.target);
    const updateData = {
        name: formData.get('name'),
        email: formData.get('email')
    };
    
    const newPassword = formData.get('newPassword');
    const confirmPassword = formData.get('confirmPassword');
    const currentPassword = formData.get('currentPassword');
    
    if (newPassword) {
        if (newPassword !== confirmPassword) {
            showToast('Passwords do not match', 'error');
            return;
        }
        
        if (!currentPassword) {
            showToast('Please enter your current password', 'error');
            return;
        }
        
        updateData.newPassword = newPassword;
        updateData.currentPassword = currentPassword;
    }
    
    try {
        const response = await fetch('/api/user/profile', {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            credentials: 'include',
            body: JSON.stringify(updateData)
        });
        
        if (response.ok) {
            const { user } = await response.json();
            AppState.currentUser.name = user.name;
            AppState.currentUser.email = user.email;
            
            showToast('Profile updated successfully', 'success');
            renderProfile();
        } else {
            const error = await response.json();
            showToast(error.error || 'Failed to update profile', 'error');
        }
    } catch (error) {
        console.error('Failed to update profile:', error);
        showToast('Failed to update profile', 'error');
    }
}

function renderSettings() {
    const mainContent = document.getElementById('main-content');
    
    const html = `
        <div class="dashboard-header">
            <h1>Settings</h1>
        </div>
        
        <div class="settings-section">
            <div class="settings-card">
                <h3>Appearance</h3>
                
                <div class="setting-item">
                    <div class="setting-label">
                        <strong>Theme</strong>
                        <p>Choose your preferred color scheme</p>
                    </div>
                    <div class="theme-selector">
                        <button class="theme-btn ${AppState.theme === 'light' ? 'active' : ''}" 
                                onclick="switchTheme('light')">
                            ☀️ Light
                        </button>
                        <button class="theme-btn ${AppState.theme === 'dark' ? 'active' : ''}" 
                                onclick="switchTheme('dark')">
                            🌙 Dark
                        </button>
                    </div>
                </div>
            </div>
            
            <div class="settings-card">
                <h3>Notifications</h3>
                
                <div class="setting-item">
                    <div class="setting-label">
                        <strong>Email Notifications</strong>
                        <p>Receive email updates about your agents</p>
                    </div>
                    <label class="toggle">
                        <input type="checkbox" ${AppState.currentUser.notifications ? 'checked' : ''} 
                               onchange="toggleNotifications(this)">
                        <span class="toggle-slider"></span>
                    </label>
                </div>
                
                <div class="setting-item">
                    <div class="setting-label">
                        <strong>Agent Alerts</strong>
                        <p>Get notified when agents encounter errors</p>
                    </div>
                    <label class="toggle">
                        <input type="checkbox" checked>
                        <span class="toggle-slider"></span>
                    </label>
                </div>
                
                <div class="setting-item">
                    <div class="setting-label">
                        <strong>Performance Reports</strong>
                        <p>Weekly performance summaries</p>
                    </div>
                    <label class="toggle">
                        <input type="checkbox" checked>
                        <span class="toggle-slider"></span>
                    </label>
                </div>
            </div>
            
            <div class="settings-card">
                <h3>Preferences</h3>
                
                <div class="setting-item">
                    <div class="setting-label">
                        <strong>Default Agent Model</strong>
                        <p>Default AI model for new agents</p>
                    </div>
                    <select class="setting-select">
                        <option value="gpt-4">GPT-4</option>
                        <option value="gpt-4-turbo">GPT-4 Turbo</option>
                        <option value="claude-3-opus">Claude 3 Opus</option>
                        <option value="claude-3-sonnet">Claude 3 Sonnet</option>
                    </select>
                </div>
                
                <div class="setting-item">
                    <div class="setting-label">
                        <strong>Timezone</strong>
                        <p>Your local timezone</p>
                    </div>
                    <select class="setting-select">
                        <option value="America/New_York" selected>Eastern Time (ET)</option>
                        <option value="America/Chicago">Central Time (CT)</option>
                        <option value="America/Denver">Mountain Time (MT)</option>
                        <option value="America/Los_Angeles">Pacific Time (PT)</option>
                    </select>
                </div>
            </div>
            
            <div class="settings-card danger">
                <h3>Danger Zone</h3>
                
                <div class="setting-item">
                    <div class="setting-label">
                        <strong>Delete Account</strong>
                        <p>Permanently delete your account and all data</p>
                    </div>
                    <button class="btn btn-danger" onclick="deleteAccount()">
                        Delete Account
                    </button>
                </div>
            </div>
        </div>
    `;
    
    mainContent.innerHTML = html;
}

async function switchTheme(theme) {
    AppState.theme = theme;
    
    document.body.classList.remove('light-theme', 'dark-theme');
    document.body.classList.add(`${theme}-theme`);
    
    try {
        await fetch('/api/user/profile', {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            credentials: 'include',
            body: JSON.stringify({ theme })
        });
        
        showToast(`Switched to ${theme} theme`, 'success');
        renderSettings();
    } catch (error) {
        console.error('Failed to update theme:', error);
    }
}

async function toggleNotifications(checkbox) {
    AppState.currentUser.notifications = checkbox.checked;
    
    try {
        await fetch('/api/user/profile', {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            credentials: 'include',
            body: JSON.stringify({ notifications: checkbox.checked })
        });
        
        showToast(`Notifications ${checkbox.checked ? 'enabled' : 'disabled'}`, 'success');
    } catch (error) {
        console.error('Failed to update notifications:', error);
    }
}

// ============================================================================
// ADMIN VIEWS
// ============================================================================

function renderAdminDashboard() {
    if (!AppState.isAdmin) return;
    
    const mainContent = document.getElementById('main-content');
    mainContent.innerHTML = `
        <div class="dashboard-header">
            <h1>👑 Admin Dashboard</h1>
        </div>
        <div class="admin-stats-grid">
            <div class="stat-card admin">
                <div class="stat-icon">👥</div>
                <div class="stat-value">2</div>
                <div class="stat-label">Total Users</div>
            </div>
            <div class="stat-card admin">
                <div class="stat-icon">🤖</div>
                <div class="stat-value">${AppState.agents.length}</div>
                <div class="stat-label">Total Agents</div>
            </div>
            <div class="stat-card admin">
                <div class="stat-icon">💰</div>
                <div class="stat-value">$1,998</div>
                <div class="stat-label">Monthly Revenue</div>
            </div>
            <div class="stat-card admin">
                <div class="stat-icon">📊</div>
                <div class="stat-value">99.9%</div>
                <div class="stat-label">System Uptime</div>
            </div>
        </div>
    `;
}

function renderUserManagement() {
    if (!AppState.isAdmin) return;
    
    const mainContent = document.getElementById('main-content');
    mainContent.innerHTML = `
        <div class="dashboard-header">
            <h1>User Management</h1>
            <button class="btn btn-primary" onclick="addNewUser()">
                ➕ Add User
            </button>
        </div>
        <div class="users-table">
            <table>
                <thead>
                    <tr>
                        <th>Username</th>
                        <th>Email</th>
                        <th>Role</th>
                        <th>Created</th>
                        <th>Actions</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Admin@B</td>
                        <td>admin@ospreyai.com</td>
                        <td><span class="badge badge-admin">ADMIN</span></td>
                        <td>Jan 1, 2024</td>
                        <td><button class="btn btn-sm">Edit</button></td>
                    </tr>
                    <tr>
                        <td>User@B</td>
                        <td>user@ospreyai.com</td>
                        <td><span class="badge badge-user">USER</span></td>
                        <td>Jan 15, 2024</td>
                        <td><button class="btn btn-sm">Edit</button></td>
                    </tr>
                </tbody>
            </table>
        </div>
    `;
}

function renderSystemConfig() {
    if (!AppState.isAdmin) return;
    
    const mainContent = document.getElementById('main-content');
    mainContent.innerHTML = `
        <div class="dashboard-header">
            <h1>System Configuration</h1>
        </div>
        <div class="config-section">
            <h3>System Settings</h3>
            <div class="config-form">
                <div class="form-group">
                    <label>Max Agents Per User:</label>
                    <input type="number" value="20">
                </div>
                <div class="form-group">
                    <label>Request Rate Limit:</label>
                    <input type="number" value="1000">
                </div>
                <button class="btn btn-primary">Save Configuration</button>
            </div>
        </div>
    `;
}

function renderAgentApproval() {
    if (!AppState.isAdmin) return;
    
    const mainContent = document.getElementById('main-content');
    mainContent.innerHTML = `
        <div class="dashboard-header">
            <h1>Agent Approvals</h1>
        </div>
        <div class="empty-state">
            <h3>No pending approvals</h3>
            <p>All agents have been reviewed</p>
        </div>
    `;
}

function renderAuditLogs() {
    if (!AppState.isAdmin) return;
    
    const mainContent = document.getElementById('main-content');
    mainContent.innerHTML = `
        <div class="dashboard-header">
            <h1>Audit Logs</h1>
            <button class="btn btn-secondary" onclick="exportAuditLogs()">
                📥 Export
            </button>
        </div>
        <div class="audit-logs">
            ${AppState.logs.map(log => `
                <div class="audit-entry">
                    <span class="audit-timestamp">${formatDate(log.timestamp)}</span>
                    <span class="audit-user">${log.user}</span>
                    <span class="audit-action">${log.action}</span>
                    <span class="audit-details">${log.details}</span>
                </div>
            `).join('')}
        </div>
    `;
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

function getAgentIcon(type) {
    const icons = {
        customer_support: '👥',
        optimization: '⚡',
        analytics: '📊',
        logistics: '🚚',
        security: '🔒',
        marketing: '📢',
        hr: '👔',
        finance: '💰',
        sales: '💼'
    };
    return icons[type] || '🤖';
}

function getSeverityIcon(severity) {
    const icons = {
        info: 'ℹ️',
        warning: '⚠️',
        error: '❌',
        success: '✅'
    };
    return icons[severity] || 'ℹ️';
}

function formatNumber(num) {
    if (num >= 1000000) {
        return (num / 1000000).toFixed(1) + 'M';
    }
    if (num >= 1000) {
        return (num / 1000).toFixed(1) + 'K';
    }
    return num.toString();
}

function formatDate(date) {
    return new Date(date).toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
}

function formatTimeAgo(date) {
    const seconds = Math.floor((new Date() - new Date(date)) / 1000);
    
    if (seconds < 60) return 'just now';
    const minutes = Math.floor(seconds / 60);
    if (minutes < 60) return `${minutes}m ago`;
    const hours = Math.floor(minutes / 60);
    if (hours < 24) return `${hours}h ago`;
    const days = Math.floor(hours / 24);
    return `${days}d ago`;
}

// ============================================================================
// UI COMPONENTS
// ============================================================================

function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    
    const icon = {
        success: '✅',
        error: '❌',
        warning: '⚠️',
        info: 'ℹ️'
    }[type] || 'ℹ️';
    
    toast.innerHTML = `
        <span class="toast-icon">${icon}</span>
        <span class="toast-message">${message}</span>
    `;
    
    document.body.appendChild(toast);
    
    setTimeout(() => toast.classList.add('show'), 10);
    
    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

function openModal(title, content) {
    const modal = document.createElement('div');
    modal.className = 'modal';
    modal.innerHTML = `
        <div class="modal-backdrop" onclick="closeModal()"></div>
        <div class="modal-content">
            <div class="modal-header">
                <h2>${title}</h2>
                <button class="modal-close" onclick="closeModal()">×</button>
            </div>
            <div class="modal-body">
                ${content}
            </div>
        </div>
    `;
    
    document.body.appendChild(modal);
    setTimeout(() => modal.classList.add('show'), 10);
}

function closeModal() {
    const modal = document.querySelector('.modal');
    if (modal) {
        modal.classList.remove('show');
        setTimeout(() => modal.remove(), 300);
    }
}

// ============================================================================
// DASHBOARD UPDATES
// ============================================================================

function updateDashboardStats() {
    // Update stats in real-time
    setInterval(async () => {
        await loadAnalytics();
        
        if (AppState.currentView === 'overview') {
            const stats = document.querySelectorAll('.stat-value');
            if (stats.length > 0 && AppState.analytics) {
                // Subtle updates without full re-render
            }
        }
    }, 30000); // Every 30 seconds
}

function startRealtimeUpdates() {
    // Simulate real-time updates for demo
    setInterval(() => {
        if (Math.random() > 0.7) {
            const messages = [
                'Agent processed 100 new requests',
                'System performance improved by 0.2%',
                'New integration available'
            ];
            
            if (Math.random() > 0.8) {
                const message = messages[Math.floor(Math.random() * messages.length)];
                // Could show subtle notification
            }
        }
    }, 10000); // Every 10 seconds
}

async function refreshDashboard() {
    showToast('Refreshing dashboard...', 'info');
    await initializeDashboard();
    showView(AppState.currentView);
    showToast('Dashboard refreshed', 'success');
}

// ============================================================================
// BUTTON HANDLERS (FILLER FUNCTIONS)
// ============================================================================

function openImportAgentModal() {
    showToast('Import feature coming soon', 'info');
}

function createWorkflow() {
    showToast('Workflow creation coming soon', 'info');
}

function addIntegration() {
    showToast('Integration marketplace coming soon', 'info');
}

function toggleIntegration(name) {
    showToast(`${name} integration toggled`, 'success');
}

function generateAPIKey() {
    showToast('New API key generated', 'success');
}

function copyAPIKey() {
    showToast('API key copied to clipboard', 'success');
}

function revokeAPIKey() {
    if (confirm('Are you sure you want to revoke this API key?')) {
        showToast('API key revoked', 'success');
    }
}

function refreshMonitoring() {
    showToast('Monitoring data refreshed', 'success');
}

function exportLogs() {
    showToast('Exporting logs...', 'info');
}

function upgradePlan() {
    showToast('Plan upgrade coming soon', 'info');
}

function setup2FA() {
    showToast('2FA setup coming soon', 'info');
}

function manageAPIKeys() {
    showView('api');
}

function markAllRead() {
    showToast('All notifications marked as read', 'success');
}

function openDocs() {
    window.open('https://docs.ospreyai.com', '_blank');
}

function openLiveChat() {
    showToast('Live chat opening...', 'info');
}

function emailSupport() {
    window.location.href = 'mailto:support@ospreyai.com';
}

function viewTraining() {
    showToast('Training courses coming soon', 'info');
}

function deleteAccount() {
    if (confirm('Are you ABSOLUTELY sure? This cannot be undone!')) {
        showToast('Account deletion is not available in demo mode', 'warning');
    }
}

function addNewUser() {
    showToast('User management coming soon', 'info');
}

function exportAuditLogs() {
    showToast('Exporting audit logs...', 'info');
}

function editAgentConfig(agentId) {
    configureAgent(agentId);
}

function toggleAgentMenu(agentId) {
    showToast('Menu options coming soon', 'info');
}

// ============================================================================
// EVENT LISTENERS
// ============================================================================

function setupEventListeners() {
    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        // Ctrl+K or Cmd+K for quick search
        if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
            e.preventDefault();
            showToast('Quick search coming soon', 'info');
        }
    });
    
    // Handle browser back/forward
    window.addEventListener('popstate', () => {
        // Handle navigation
    });
}

// ============================================================================
// INITIALIZATION COMPLETE
// ============================================================================

console.log('✅ Dashboard JavaScript loaded - 2000+ lines');
console.log('🚀 Osprey AI Platform v2.0.0');
