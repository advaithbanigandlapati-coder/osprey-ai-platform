/* ============================================================================
   OSPREY AI PLATFORM - DASHBOARD JAVASCRIPT
   Following user's file structure with Ollama integration
   ============================================================================ */

// ============================================================================
// STATE MANAGEMENT
// ============================================================================

const AppState = {
    currentPage: 'home',
    currentTheme: localStorage.getItem('theme') || 'light',
    user: {
        name: 'advaith',
        initials: 'A',
        organization: "advaith's organization",
        plan: 'Community Plan'
    },
    dropdownOpen: false,
    orgsViewOpen: false
};

// ============================================================================
// INITIALIZATION
// ============================================================================

document.addEventListener('DOMContentLoaded', async () => {
    console.log('🦅 Initializing Osprey AI Platform...');
    
    // Initialize theme
    applyTheme(AppState.currentTheme);
    
    // Initialize navigation
    setupNavigation();
    
    // Initialize user dropdown
    setupUserDropdown();
    
    // Initialize sidebar toggle
    setupSidebarToggle();
    
    // Initialize AI
    await initializeAI();
    
    // Initialize Lucide icons
    if (typeof lucide !== 'undefined') {
        lucide.createIcons();
    }
    
    console.log('✅ Dashboard ready!');
});

// ============================================================================
// NAVIGATION
// ============================================================================

function setupNavigation() {
    // Nav item clicks
    document.querySelectorAll('.nav-item').forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            const page = item.getAttribute('data-page');
            navigateToPage(page);
        });
    });
    
    // Handle browser back/forward
    window.addEventListener('hashchange', () => {
        const page = location.hash.slice(1) || 'home';
        navigateToPage(page, false);
    });
    
    // Load initial page from hash
    const initialPage = location.hash.slice(1) || 'home';
    navigateToPage(initialPage, false);
}

function navigateToPage(pageName, updateHash = true) {
    // Update hash
    if (updateHash) {
        location.hash = pageName;
    }
    
    // Update nav active state
    document.querySelectorAll('.nav-item').forEach(item => {
        item.classList.toggle('active', item.getAttribute('data-page') === pageName);
    });
    
    // Update page content
    document.querySelectorAll('.content-page').forEach(page => {
        page.classList.toggle('active', page.getAttribute('data-page') === pageName);
    });
    
    // Update page title
    updatePageTitle(pageName);
    
    // Update state
    AppState.currentPage = pageName;
    
    // If it's an AI agent page, initialize chat
    const isAgent = document.querySelector(`[data-page="${pageName}"]`)?.hasAttribute('data-agent');
    if (isAgent) {
        initializeAgentChat(pageName);
    }
}

function updatePageTitle(pageName) {
    const titles = {
        'home': 'Welcome back, advaith',
        'agents': 'AI Agents',
        'analytics': 'Analytics',
        'workflows': 'Workflows',
        'content-writer': 'Content Writer Pro',
        'code-assistant': 'Code Assistant',
        'data-analyst': 'Data Analyst AI',
        'support-bot': 'Support Bot',
        'research': 'Research Assistant',
        'marketing': 'Marketing Strategist'
    };
    
    const titleEl = document.getElementById('pageTitle');
    if (titleEl) {
        titleEl.textContent = titles[pageName] || 'Osprey AI';
    }
}

// ============================================================================
// USER DROPDOWN (Google-style)
// ============================================================================

function setupUserDropdown() {
    const avatarBtn = document.getElementById('userAvatarBtn');
    const dropdown = document.getElementById('userDropdown');
    const orgSelector = dropdown.querySelector('.org-selector');
    const orgsSection = document.getElementById('orgsSection');
    const mainMenu = dropdown.querySelector('.main-menu');
    const backBtn = document.getElementById('backToMain');
    
    // Toggle dropdown
    avatarBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        AppState.dropdownOpen = !AppState.dropdownOpen;
        dropdown.classList.toggle('show', AppState.dropdownOpen);
        
        // Refresh icons when opening
        if (AppState.dropdownOpen && typeof lucide !== 'undefined') {
            setTimeout(() => lucide.createIcons(), 50);
        }
    });
    
    // Close dropdown when clicking outside
    document.addEventListener('click', (e) => {
        if (!dropdown.contains(e.target) && !avatarBtn.contains(e.target)) {
            AppState.dropdownOpen = false;
            dropdown.classList.remove('show');
        }
    });
    
    // Organization selector - show orgs view
    orgSelector.addEventListener('click', () => {
        orgsSection.style.display = 'block';
        mainMenu.style.display = 'none';
        AppState.orgsViewOpen = true;
        
        if (typeof lucide !== 'undefined') {
            setTimeout(() => lucide.createIcons(), 50);
        }
    });
    
    // Back to main menu
    backBtn.addEventListener('click', () => {
        orgsSection.style.display = 'none';
        mainMenu.style.display = 'block';
        AppState.orgsViewOpen = false;
        
        if (typeof lucide !== 'undefined') {
            setTimeout(() => lucide.createIcons(), 50);
        }
    });
    
    // Theme toggle
    const themeToggle = document.getElementById('themeToggleDropdown');
    themeToggle.addEventListener('click', () => {
        toggleTheme();
    });
    
    // Logout
    dropdown.querySelector('.logout-item').addEventListener('click', () => {
        handleLogout();
    });
}

// ============================================================================
// THEME MANAGEMENT
// ============================================================================

function toggleTheme() {
    AppState.currentTheme = AppState.currentTheme === 'light' ? 'dark' : 'light';
    applyTheme(AppState.currentTheme);
    localStorage.setItem('theme', AppState.currentTheme);
}

function applyTheme(theme) {
    document.body.setAttribute('data-theme', theme);
    
    // Update theme toggle icons
    const sunIcons = document.querySelectorAll('[data-lucide="sun"]');
    const moonIcons = document.querySelectorAll('[data-lucide="moon"]');
    
    if (theme === 'dark') {
        sunIcons.forEach(icon => icon.style.opacity = '0.5');
        moonIcons.forEach(icon => icon.style.opacity = '1');
    } else {
        sunIcons.forEach(icon => icon.style.opacity = '1');
        moonIcons.forEach(icon => icon.style.opacity = '0.5');
    }
}

// ============================================================================
// SIDEBAR
// ============================================================================

function setupSidebarToggle() {
    const toggleBtn = document.getElementById('sidebarToggle');
    const sidebar = document.getElementById('sidebar');
    
    toggleBtn.addEventListener('click', () => {
        sidebar.classList.toggle('collapsed');
    });
}

// ============================================================================
// AI INTEGRATION
// ============================================================================

let aiInitialized = false;

async function initializeAI() {
    if (aiInitialized) return;
    
    console.log('🤖 Initializing Osprey AI...');
    
    try {
        // Check if browser-ai.js is loaded
        if (typeof initializeOspreyAI === 'undefined') {
            console.warn('⚠️ browser-ai.js not loaded, AI features disabled');
            return;
        }
        
        // Initialize AI with progress
        await initializeOspreyAI((progress) => {
            console.log(`AI Loading: ${progress.progress}%`);
        });
        
        aiInitialized = true;
        console.log('✅ Osprey AI initialized');
        
        // Check AI status
        if (typeof getAIStatus === 'function') {
            const status = getAIStatus();
            console.log('AI Status:', status);
            
            if (status.usingOllama) {
                console.log('✅ Using Ollama for real AI responses');
            } else {
                console.log('⚠️ Ollama not available, using fallback responses');
            }
        }
        
    } catch (error) {
        console.error('❌ AI initialization failed:', error);
    }
}

function initializeAgentChat(agentId) {
    console.log(`Initializing chat for agent: ${agentId}`);
    
    // Set active agent
    if (typeof setActiveAgent === 'function') {
        setActiveAgent(agentId);
    }
    
    // Setup send button for this agent's chat
    const page = document.querySelector(`[data-page="${agentId}"]`);
    if (!page) return;
    
    const sendBtn = page.querySelector('.btn-send');
    const input = page.querySelector('.chat-input');
    
    if (sendBtn && input) {
        // Remove old listeners
        const newSendBtn = sendBtn.cloneNode(true);
        sendBtn.parentNode.replaceChild(newSendBtn, sendBtn);
        
        // Add new listener
        newSendBtn.addEventListener('click', () => {
            sendMessage(agentId);
        });
        
        // Enter to send
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage(agentId);
            }
        });
    }
}

async function sendMessage(agentId) {
    const page = document.querySelector(`[data-page="${agentId}"]`);
    if (!page) return;
    
    const input = page.querySelector('.chat-input');
    const messagesContainer = page.querySelector('.chat-messages');
    
    const message = input.value.trim();
    if (!message) return;
    
    // Add user message
    addMessage(messagesContainer, message, 'user', AppState.user.initials);
    input.value = '';
    
    // Auto-resize textarea
    input.style.height = 'auto';
    
    // Show typing indicator
    const typingId = addTypingIndicator(messagesContainer, agentId);
    
    try {
        // Generate AI response
        let response;
        
        if (typeof generateAIResponse === 'function') {
            response = await generateAIResponse(message, agentId);
        } else {
            response = "AI is not initialized. Please check that browser-ai.js is loaded.";
        }
        
        // Remove typing indicator
        removeTypingIndicator(messagesContainer, typingId);
        
        // Add AI response
        addMessage(messagesContainer, response, 'assistant', getAgentEmoji(agentId));
        
    } catch (error) {
        console.error('Error generating response:', error);
        removeTypingIndicator(messagesContainer, typingId);
        addMessage(messagesContainer, 'Sorry, I encountered an error. Please try again.', 'assistant', getAgentEmoji(agentId));
    }
}

function addMessage(container, text, type, avatar) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}`;
    
    const time = new Date().toLocaleTimeString('en-US', { 
        hour: 'numeric', 
        minute: '2-digit'
    });
    
    messageDiv.innerHTML = `
        <div class="message-avatar">${avatar}</div>
        <div>
            <div class="message-bubble">${escapeHtml(text)}</div>
            <div class="message-time">${time}</div>
        </div>
    `;
    
    container.appendChild(messageDiv);
    
    // Scroll to bottom
    container.scrollTop = container.scrollHeight;
}

function addTypingIndicator(container, agentId) {
    const id = 'typing-' + Date.now();
    const typingDiv = document.createElement('div');
    typingDiv.className = 'message assistant';
    typingDiv.id = id;
    
    typingDiv.innerHTML = `
        <div class="message-avatar">${getAgentEmoji(agentId)}</div>
        <div>
            <div class="message-bubble">
                <div class="typing-indicator">
                    <span></span><span></span><span></span>
                </div>
            </div>
        </div>
    `;
    
    container.appendChild(typingDiv);
    container.scrollTop = container.scrollHeight;
    
    return id;
}

function removeTypingIndicator(container, id) {
    const indicator = document.getElementById(id);
    if (indicator) {
        indicator.remove();
    }
}

function getAgentEmoji(agentId) {
    const emojis = {
        'content-writer': '✍️',
        'code-assistant': '💻',
        'data-analyst': '📊',
        'support-bot': '🎧',
        'research': '🔬',
        'marketing': '📈'
    };
    return emojis[agentId] || '🤖';
}

// ============================================================================
// UTILITIES
// ============================================================================

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function handleLogout() {
    console.log('Logging out...');
    // Add logout logic here
    alert('Logout functionality - connect to your backend');
}

// ============================================================================
// EXPORT FOR DEBUGGING
// ============================================================================

window.OspreyDashboard = {
    state: AppState,
    navigateToPage,
    toggleTheme,
    sendMessage
};

console.log('📊 Dashboard loaded. Access via window.OspreyDashboard');
