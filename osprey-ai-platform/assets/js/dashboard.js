
// ===============================================
// AGENT CONFIGURATION
// ===============================================

const AGENTS = {
    'content-writer': {
        name: 'Content Writer Pro',
        icon: '✍️',
        color: '#578098',
        description: 'Create engaging content, from blog posts to marketing copy'
    },
    'code-assistant': {
        name: 'Code Assistant',
        icon: '💻',
        color: '#6a93aa',
        description: 'Write, debug, and optimize code in any language'
    },
    'data-analyst': {
        name: 'Data Analyst',
        icon: '📊',
        color: '#7ba3b8',
        description: 'Analyze data, identify trends, and generate insights'
    },
    'support-bot': {
        name: 'Support Bot',
        icon: '💬',
        color: '#8cb3c6',
        description: 'Friendly customer support and problem-solving'
    },
    'research-assistant': {
        name: 'Research Assistant',
        icon: '🔍',
        color: '#9dc3d4',
        description: 'Research topics, synthesize information, create reports'
    },
    'marketing-strategist': {
        name: 'Marketing Strategist',
        icon: '📈',
        color: '#aed3e2',
        description: 'Develop campaigns, identify audiences, create strategies'
    }
};

// ===============================================
// STATE MANAGEMENT
// ===============================================

let currentAgent = 'content-writer';
let isAIReady = false;
let messageHistory = {};
let currentUser = null;

// Initialize message history
Object.keys(AGENTS).forEach(agentId => {
    messageHistory[agentId] = [];
});

// ===============================================
// PAGE INITIALIZATION
// ===============================================

document.addEventListener('DOMContentLoaded', async () => {
    console.log('🦅 Dashboard initializing...');
    
    try {
        // Check authentication
        await checkAuth();
        
        // Initialize UI
        initializeDashboard();
        
        // Initialize AI (but don't show welcome yet)
        await initializeAI();
        
        // Now show welcome message (after UI is ready)
        setTimeout(() => {
            showWelcomeMessage();
        }, 500);
        
    } catch (error) {
        console.error('Dashboard initialization error:', error);
    }
});

//===============================================
// AUTHENTICATION
// ===============================================

async function checkAuth() {
    try {
        const response = await fetch('/api/auth/session', {
            credentials: 'include'
        });
        const data = await response.json();
        
        if (!data.authenticated) {
            window.location.href = '/signin.html';
            return;
        }
        
        currentUser = data.user;
        console.log('✅ User authenticated:', currentUser.username);
        
    } catch (error) {
        console.error('Auth check failed:', error);
        window.location.href = '/signin.html';
    }
}

// ===============================================
// DASHBOARD INITIALIZATION
// ===============================================

function initializeDashboard() {
    renderSidebar();
    renderMainContent();
    attachEventListeners();
}

function renderSidebar() {
    const sidebar = document.getElementById('sidebar');
    if (!sidebar) return;
    
    sidebar.innerHTML = `
        <div class="sidebar-header">
            <div class="logo">
                <span class="logo-icon">🦅</span>
                <span class="logo-text">Osprey AI</span>
            </div>
        </div>
        
        <nav class="sidebar-nav">
            <a href="#" class="nav-item active" data-page="agents">
                <span>🤖</span> AI Agents
            </a>
            <a href="#" class="nav-item" data-page="settings">
                <span>⚙️</span> User Settings
            </a>
            ${currentUser && currentUser.role === 'admin' ? `
            <a href="#" class="nav-item" data-page="admin">
                <span>👥</span> Admin Panel
            </a>
            ` : ''}
        </nav>
        
        <div class="sidebar-footer">
            <div class="user-info">
                <div class="user-avatar">${currentUser?.name?.charAt(0) || 'U'}</div>
                <div class="user-details">
                    <div class="user-name">${currentUser?.name || 'User'}</div>
                    <div class="user-role">${currentUser?.role || 'user'}</div>
                </div>
            </div>
            <button class="btn-logout" onclick="logout()">
                <span>🚪</span> Logout
            </button>
        </div>
    `;
}

function renderMainContent() {
    const mainContent = document.getElementById('main-content');
    if (!mainContent) return;
    
    mainContent.innerHTML = `
        <div class="page-content">
            <div id="page-agents" class="page active">
                ${renderAgentsPage()}
            </div>
            <div id="page-settings" class="page">
                ${renderSettingsPage()}
            </div>
            ${currentUser && currentUser.role === 'admin' ? `
            <div id="page-admin" class="page">
                ${renderAdminPage()}
            </div>
            ` : ''}
        </div>
    `;
}

// ===============================================
// AGENTS PAGE
// ===============================================

function renderAgentsPage() {
    return `
        <div class="page-header">
            <h1>AI Agents</h1>
            <p>Chat with specialized AI assistants</p>
        </div>
        
        <div class="agent-tabs">
            ${Object.entries(AGENTS).map(([id, agent]) => `
                <button class="agent-tab ${id === currentAgent ? 'active' : ''}" 
                        data-agent="${id}"
                        onclick="switchAgent('${id}')">
                    <span class="agent-icon">${agent.icon}</span>
                    <span class="agent-name">${agent.name}</span>
                </button>
            `).join('')}
        </div>
        
        <div class="chat-container">
            <div class="chat-messages" id="chat-messages"></div>
            <div class="chat-input-container">
                <input type="text" 
                       class="chat-input" 
                       id="chat-input" 
                       placeholder="Type your message..."
                       onkeypress="if(event.key==='Enter') sendMessage()">
                <button class="btn-send" onclick="sendMessage()">Send</button>
            </div>
        </div>
    `;
}

// ===============================================
// SETTINGS PAGE
// ===============================================

function renderSettingsPage() {
    return `
        <div class="page-header">
            <h1>User Settings</h1>
            <p>Manage your profile and preferences</p>
        </div>
        
        <div class="settings-container">
            <div class="settings-card">
                <h3>Profile Information</h3>
                <form id="profile-form" onsubmit="updateProfile(event)">
                    <div class="form-group">
                        <label>Name</label>
                        <input type="text" id="profile-name" value="${currentUser?.name || ''}" required>
                    </div>
                    <div class="form-group">
                        <label>Email</label>
                        <input type="email" id="profile-email" value="${currentUser?.email || ''}" required>
                    </div>
                    <div class="form-group">
                        <label>Username</label>
                        <input type="text" value="${currentUser?.username || ''}" disabled>
                        <small>Username cannot be changed</small>
                    </div>
                    <button type="submit" class="btn-primary">Save Changes</button>
                </form>
            </div>
            
            <div class="settings-card">
                <h3>Change Password</h3>
                <form id="password-form" onsubmit="changePassword(event)">
                    <div class="form-group">
                        <label>Current Password</label>
                        <input type="password" id="current-password" required>
                    </div>
                    <div class="form-group">
                        <label>New Password</label>
                        <input type="password" id="new-password" required minlength="6">
                    </div>
                    <div class="form-group">
                        <label>Confirm New Password</label>
                        <input type="password" id="confirm-password" required minlength="6">
                    </div>
                    <button type="submit" class="btn-primary">Update Password</button>
                </form>
            </div>
            
            <div class="settings-card">
                <h3>Preferences</h3>
                <div class="form-group">
                    <label>Theme</label>
                    <select id="theme-select" onchange="changeTheme(this.value)">
                        <option value="light" ${currentUser?.theme === 'light' ? 'selected' : ''}>Light</option>
                        <option value="dark" ${currentUser?.theme === 'dark' ? 'selected' : ''}>Dark</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>
                        <input type="checkbox" 
                               id="notifications-toggle" 
                               ${currentUser?.notifications ? 'checked' : ''}
                               onchange="toggleNotifications(this.checked)">
                        Enable Notifications
                    </label>
                </div>
            </div>
        </div>
    `;
}

// ===============================================
// ADMIN PAGE
// ===============================================

function renderAdminPage() {
    return `
        <div class="page-header">
            <h1>Admin Panel</h1>
            <p>Manage users and system settings</p>
            <button class="btn-primary" onclick="showCreateUserModal()">
                + Create New User
            </button>
        </div>
        
        <div class="admin-container">
            <div class="users-table-container">
                <table class="users-table" id="users-table">
                    <thead>
                        <tr>
                            <th>Name</th>
                            <th>Username</th>
                            <th>Email</th>
                            <th>Role</th>
                            <th>Created</th>
                            <th>Actions</th>
                        </tr>
                    </thead>
                    <tbody id="users-table-body">
                        <tr><td colspan="6">Loading...</td></tr>
                    </tbody>
                </table>
            </div>
        </div>
    `;
}

// ===============================================
// AI INITIALIZATION
// ===============================================

async function initializeAI() {
    console.log('🦅 Initializing Osprey AI...');
    
    const overlay = document.getElementById('ai-loading-overlay');
    const progressFill = document.getElementById('ai-progress-fill');
    const statusText = document.getElementById('ai-loading-status');

    if (overlay) overlay.style.display = 'flex';

    try {
        await window.initializeOspreyAI((progress) => {
            if (progressFill) {
                progressFill.style.width = progress.progress + '%';
            }
            if (statusText) {
                const messages = {
                    'downloading': 'Loading AI agents...',
                    'loading': 'Initializing...',
                    'ready': 'All 6 agents ready!'
                };
                statusText.textContent = messages[progress.status] || progress.status;
            }
        });

        isAIReady = true;

        if (overlay) {
            setTimeout(() => {
                overlay.style.opacity = '0';
                setTimeout(() => {
                    overlay.style.display = 'none';
                    overlay.style.opacity = '1';
                }, 500);
            }, 1000);
        }

        console.log('✅ Osprey AI ready!');
        // Welcome message will be shown by DOMContentLoaded after delay

    } catch (error) {
        console.error('❌ AI initialization failed:', error);
        if (statusText) {
            statusText.textContent = '❌ Failed to load AI. Please refresh.';
        }
    }
}

// ===============================================
// AGENT FUNCTIONS
// ===============================================

function switchAgent(agentId) {
    if (!AGENTS[agentId]) return;
    
    currentAgent = agentId;
    window.setActiveAgent(agentId);
    
    // Update tabs
    document.querySelectorAll('.agent-tab').forEach(tab => {
        tab.classList.remove('active');
        if (tab.dataset.agent === agentId) {
            tab.classList.add('active');
        }
    });
    
    // Load history
    loadAgentHistory(agentId);
}

function loadAgentHistory(agentId) {
    const chatContainer = document.getElementById('chat-messages');
    if (!chatContainer) return;

    chatContainer.innerHTML = '';

    if (messageHistory[agentId] && messageHistory[agentId].length > 0) {
        messageHistory[agentId].forEach(msg => {
            addMessageToChat(msg.type, msg.content, agentId, false);
        });
    } else {
        const greeting = window.getAgentGreeting(agentId);
        addMessageToChat('ai', greeting, agentId, false);
    }
}

async function sendMessage() {
    const input = document.getElementById('chat-input');
    if (!input) return;
    
    const message = input.value.trim();
    if (!message) return;
    
    if (!isAIReady) {
        alert('AI is still loading. Please wait.');
        return;
    }
    
    // Clear input
    input.value = '';
    
    // Add user message
    addMessageToChat('user', message, currentAgent, true);
    
    // Show typing indicator
    showTypingIndicator();
    
    try {
        const response = await window.generateAIResponse(message, currentAgent);
        removeTypingIndicator();
        addMessageToChat('ai', response, currentAgent, true);
    } catch (error) {
        removeTypingIndicator();
        addMessageToChat('error', 'Sorry, I encountered an error. Please try again.', currentAgent, false);
    }
}

function addMessageToChat(type, content, agentId, saveToHistory) {
    const chatContainer = document.getElementById('chat-messages');
    if (!chatContainer) return;

    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}-message`;
    
    if (type === 'ai') {
        const agent = AGENTS[agentId];
        messageDiv.innerHTML = `
            <div class="message-avatar">${agent?.icon || '🤖'}</div>
            <div class="message-content">${escapeHtml(content)}</div>
        `;
    } else if (type === 'user') {
        messageDiv.innerHTML = `
            <div class="message-content">${escapeHtml(content)}</div>
        `;
    } else {
        messageDiv.innerHTML = `
            <div class="message-content">${escapeHtml(content)}</div>
        `;
    }
    
    chatContainer.appendChild(messageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;

    if (saveToHistory) {
        if (!messageHistory[agentId]) messageHistory[agentId] = [];
        messageHistory[agentId].push({ type, content });
    }
}

function showTypingIndicator() {
    const chatContainer = document.getElementById('chat-messages');
    if (!chatContainer) return;
    
    const indicator = document.createElement('div');
    indicator.className = 'message ai-message typing-indicator';
    indicator.id = 'typing-indicator';
    indicator.innerHTML = `
        <div class="message-avatar">${AGENTS[currentAgent]?.icon || '🤖'}</div>
        <div class="message-content">
            <div class="typing-dots">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        </div>
    `;
    chatContainer.appendChild(indicator);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function removeTypingIndicator() {
    const indicator = document.getElementById('typing-indicator');
    if (indicator) indicator.remove();
}

function showWelcomeMessage() {
    // Only show if chat container exists
    const chatContainer = document.getElementById('chat-messages');
    if (!chatContainer) {
        console.log('Chat container not ready yet, skipping welcome message');
        return;
    }
    
    const greeting = window.getAgentGreeting(currentAgent);
    addMessageToChat('ai', greeting, currentAgent, false);
}

// ===============================================
// USER SETTINGS FUNCTIONS
// ===============================================

async function updateProfile(event) {
    event.preventDefault();
    
    const name = document.getElementById('profile-name').value;
    const email = document.getElementById('profile-email').value;
    
    try {
        const response = await fetch('/api/user/profile', {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            credentials: 'include',
            body: JSON.stringify({ name, email })
        });
        
        const data = await response.json();
        
        if (data.success) {
            alert('Profile updated successfully!');
            currentUser.name = name;
            currentUser.email = email;
            renderSidebar();
        } else {
            alert('Failed to update profile: ' + (data.error || 'Unknown error'));
        }
    } catch (error) {
        console.error('Profile update error:', error);
        alert('Failed to update profile');
    }
}

async function changePassword(event) {
    event.preventDefault();
    
    const currentPassword = document.getElementById('current-password').value;
    const newPassword = document.getElementById('new-password').value;
    const confirmPassword = document.getElementById('confirm-password').value;
    
    if (newPassword !== confirmPassword) {
        alert('New passwords do not match!');
        return;
    }
    
    try {
        const response = await fetch('/api/user/profile', {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            credentials: 'include',
            body: JSON.stringify({ currentPassword, newPassword })
        });
        
        const data = await response.json();
        
        if (data.success) {
            alert('Password updated successfully!');
            document.getElementById('password-form').reset();
        } else {
            alert('Failed to update password: ' + (data.error || 'Unknown error'));
        }
    } catch (error) {
        console.error('Password update error:', error);
        alert('Failed to update password');
    }
}

async function changeTheme(theme) {
    try {
        const response = await fetch('/api/user/profile', {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            credentials: 'include',
            body: JSON.stringify({ theme })
        });
        
        if (response.ok) {
            document.documentElement.setAttribute('data-theme', theme);
            currentUser.theme = theme;
        }
    } catch (error) {
        console.error('Theme update error:', error);
    }
}

async function toggleNotifications(enabled) {
    try {
        await fetch('/api/user/profile', {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            credentials: 'include',
            body: JSON.stringify({ notifications: enabled })
        });
        currentUser.notifications = enabled;
    } catch (error) {
        console.error('Notifications update error:', error);
    }
}

// ===============================================
// ADMIN FUNCTIONS
// ===============================================

async function loadUsers() {
    try {
        const response = await fetch('/api/admin/users', {
            credentials: 'include'
        });
        const data = await response.json();
        
        if (data.success) {
            renderUsersTable(data.users);
        }
    } catch (error) {
        console.error('Failed to load users:', error);
    }
}

function renderUsersTable(users) {
    const tbody = document.getElementById('users-table-body');
    if (!tbody) return;
    
    tbody.innerHTML = users.map(user => `
        <tr>
            <td>${escapeHtml(user.name)}</td>
            <td>${escapeHtml(user.username)}</td>
            <td>${escapeHtml(user.email)}</td>
            <td><span class="role-badge role-${user.role}">${user.role}</span></td>
            <td>${new Date(user.created).toLocaleDateString()}</td>
            <td>
                <button class="btn-sm" onclick="editUser('${user.id}')">Edit</button>
                ${user.id !== currentUser.id ? `
                <button class="btn-sm btn-danger" onclick="deleteUser('${user.id}', '${escapeHtml(user.username)}')">Delete</button>
                ` : ''}
            </td>
        </tr>
    `).join('');
}

function showCreateUserModal() {
    const modal = document.getElementById('modal-container');
    if (!modal) return;
    
    modal.innerHTML = `
        <div class="modal-overlay" onclick="closeModal()">
            <div class="modal-content" onclick="event.stopPropagation()">
                <div class="modal-header">
                    <h2>Create New User</h2>
                    <button class="modal-close" onclick="closeModal()">×</button>
                </div>
                <form id="create-user-form" onsubmit="createUser(event)">
                    <div class="form-group">
                        <label>Name *</label>
                        <input type="text" id="new-user-name" required>
                    </div>
                    <div class="form-group">
                        <label>Username *</label>
                        <input type="text" id="new-user-username" required>
                    </div>
                    <div class="form-group">
                        <label>Email *</label>
                        <input type="email" id="new-user-email" required>
                    </div>
                    <div class="form-group">
                        <label>Password *</label>
                        <input type="password" id="new-user-password" required minlength="6">
                    </div>
                    <div class="form-group">
                        <label>Role *</label>
                        <select id="new-user-role" required>
                            <option value="user">User</option>
                            <option value="admin">Admin</option>
                        </select>
                    </div>
                    <div class="modal-actions">
                        <button type="button" class="btn-secondary" onclick="closeModal()">Cancel</button>
                        <button type="submit" class="btn-primary">Create User</button>
                    </div>
                </form>
            </div>
        </div>
    `;
    modal.style.display = 'block';
}

async function createUser(event) {
    event.preventDefault();
    
    const userData = {
        name: document.getElementById('new-user-name').value,
        username: document.getElementById('new-user-username').value,
        email: document.getElementById('new-user-email').value,
        password: document.getElementById('new-user-password').value,
        role: document.getElementById('new-user-role').value
    };
    
    try {
        const response = await fetch('/api/admin/users', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            credentials: 'include',
            body: JSON.stringify(userData)
        });
        
        const data = await response.json();
        
        if (data.success) {
            alert('User created successfully!');
            closeModal();
            loadUsers();
        } else {
            alert('Failed to create user: ' + (data.error || 'Unknown error'));
        }
    } catch (error) {
        console.error('Create user error:', error);
        alert('Failed to create user');
    }
}

async function editUser(userId) {
    // TODO: Implement edit user modal
    alert('Edit user functionality - Coming soon!');
}

async function deleteUser(userId, username) {
    if (!confirm(`Are you sure you want to delete user "${username}"?`)) {
        return;
    }
    
    try {
        const response = await fetch(`/api/admin/users/${userId}`, {
            method: 'DELETE',
            credentials: 'include'
        });
        
        const data = await response.json();
        
        if (data.success) {
            alert('User deleted successfully!');
            loadUsers();
        } else {
            alert('Failed to delete user: ' + (data.error || 'Unknown error'));
        }
    } catch (error) {
        console.error('Delete user error:', error);
        alert('Failed to delete user');
    }
}

function closeModal() {
    const modal = document.getElementById('modal-container');
    if (modal) {
        modal.style.display = 'none';
        modal.innerHTML = '';
    }
}

// ===============================================
// NAVIGATION
// ===============================================

function attachEventListeners() {
    // Page navigation
    document.querySelectorAll('.nav-item').forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            const page = item.dataset.page;
            navigateToPage(page);
        });
    });
}

function navigateToPage(pageName) {
    // Update nav items
    document.querySelectorAll('.nav-item').forEach(item => {
        item.classList.remove('active');
        if (item.dataset.page === pageName) {
            item.classList.add('active');
        }
    });
    
    // Show page
    document.querySelectorAll('.page').forEach(page => {
        page.classList.remove('active');
    });
    
    const targetPage = document.getElementById(`page-${pageName}`);
    if (targetPage) {
        targetPage.classList.add('active');
    }
    
    // Load admin users if navigating to admin page
    if (pageName === 'admin') {
        loadUsers();
    }
}

// ===============================================
// UTILITY FUNCTIONS
// ===============================================

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

async function logout() {
    try {
        await fetch('/api/auth/logout', {
            method: 'POST',
            credentials: 'include'
        });
        window.location.href = '/signin.html';
    } catch (error) {
        console.error('Logout error:', error);
        window.location.href = '/signin.html';
    }
}

console.log('🦅 Dashboard controller loaded');
