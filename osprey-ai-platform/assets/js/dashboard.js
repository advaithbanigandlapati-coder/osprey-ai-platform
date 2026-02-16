
// ===============================================
// AGENT CONFIGURATION (6 Specialized Agents)
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

// Initialize message history for each agent
Object.keys(AGENTS).forEach(agentId => {
    messageHistory[agentId] = [];
});

// ===============================================
// AI INITIALIZATION
// ===============================================

/**
 * Initialize Osprey AI on page load
 */
async function initializeAI() {
    console.log('🦅 Initializing Osprey AI...');
    
    const overlay = document.getElementById('ai-loading-overlay');
    const progressFill = document.getElementById('ai-progress-fill');
    const statusText = document.getElementById('ai-loading-status');

    // Show loading overlay
    if (overlay) overlay.style.display = 'flex';

    try {
        // Initialize AI with progress tracking
        await window.initializeOspreyAI((progress) => {
            if (progressFill) {
                progressFill.style.width = progress.progress + '%';
            }
            
            if (statusText) {
                const messages = {
                    'downloading': 'Downloading AI model...',
                    'loading': 'Loading into memory...',
                    'ready': 'All 6 agents ready!'
                };
                statusText.textContent = messages[progress.status] || progress.status;
            }
        });

        isAIReady = true;

        // Hide overlay with animation
        if (overlay) {
            setTimeout(() => {
                overlay.style.opacity = '0';
                setTimeout(() => {
                    overlay.style.display = 'none';
                    overlay.style.opacity = '1';
                }, 500);
            }, 1000);
        }

        console.log('✅ Osprey AI ready! All 6 agents online.');
        
        // Show welcome message
        showWelcomeMessage();

    } catch (error) {
        console.error('❌ AI initialization failed:', error);
        if (statusText) {
            statusText.textContent = '❌ Failed to load AI. Please refresh the page.';
            statusText.style.color = '#dc3545';
        }
    }
}

// ===============================================
// AGENT SWITCHING
// ===============================================

/**
 * Switch to a different agent
 */
function switchAgent(agentId) {
    if (!AGENTS[agentId]) {
        console.error('Unknown agent:', agentId);
        return;
    }

    currentAgent = agentId;
    window.setActiveAgent(agentId);

    // Update UI
    updateActiveTab(agentId);
    loadAgentHistory(agentId);
    
    console.log(`Switched to ${AGENTS[agentId].name}`);
}

/**
 * Update active tab in UI
 */
function updateActiveTab(agentId) {
    // Update tab buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
        if (btn.dataset.tab === agentId || btn.dataset.agent === agentId) {
            btn.classList.add('active');
        }
    });
}

/**
 * Load agent's message history
 */
function loadAgentHistory(agentId) {
    const chatContainer = document.getElementById('chat-messages') || 
                         document.querySelector('.chat-messages');
    
    if (!chatContainer) return;

    // Clear current messages
    chatContainer.innerHTML = '';

    // Load history for this agent
    if (messageHistory[agentId] && messageHistory[agentId].length > 0) {
        messageHistory[agentId].forEach(msg => {
            addMessageToChat(msg.type, msg.content, agentId, false);
        });
    } else {
        // Show greeting for new conversation
        const greeting = window.getAgentGreeting(agentId);
        addMessageToChat('ai', greeting, agentId, false);
    }
}

// ===============================================
// MESSAGE HANDLING
// ===============================================

/**
 * Send a message to the current agent
 */
async function sendMessage(message, agentId = null) {
    if (!message || !message.trim()) return;

    const activeAgent = agentId || currentAgent;
    
    if (!isAIReady) {
        alert('AI is still loading. Please wait a moment.');
        return;
    }

    // Add user message
    addMessageToChat('user', message, activeAgent);

    // Show typing indicator
    const typingId = showTypingIndicator(activeAgent);

    try {
        // Generate AI response
        const response = await window.generateAIResponse(message, activeAgent);

        // Remove typing indicator
        removeTypingIndicator(typingId);

        // Add AI response
        addMessageToChat('ai', response, activeAgent);

    } catch (error) {
        console.error('Error generating response:', error);
        removeTypingIndicator(typingId);
        addMessageToChat('error', 'Sorry, I encountered an error. Please try again.', activeAgent);
    }
}

/**
 * Add message to chat and history
 */
function addMessageToChat(type, message, agentId, saveToHistory = true) {
    const chatContainer = document.getElementById('chat-messages') || 
                         document.querySelector('.chat-messages');
    
    if (!chatContainer) {
        console.error('Chat container not found');
        return;
    }

    // Create message element
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}-message`;
    
    if (type === 'ai') {
        const agent = AGENTS[agentId];
        messageDiv.innerHTML = `
            <div class="message-avatar">${agent ? agent.icon : '🦅'}</div>
            <div class="message-content">${escapeHtml(message)}</div>
        `;
    } else {
        messageDiv.innerHTML = `
            <div class="message-content">${escapeHtml(message)}</div>
        `;
    }

    chatContainer.appendChild(messageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;

    // Save to history
    if (saveToHistory && type !== 'error') {
        if (!messageHistory[agentId]) messageHistory[agentId] = [];
        messageHistory[agentId].push({ type, content: message });
    }
}

/**
 * Show typing indicator
 */
function showTypingIndicator(agentId) {
    const chatContainer = document.getElementById('chat-messages') || 
                         document.querySelector('.chat-messages');
    
    if (!chatContainer) return null;

    const typingId = 'typing-' + Date.now();
    const agent = AGENTS[agentId];
    
    const typingDiv = document.createElement('div');
    typingDiv.id = typingId;
    typingDiv.className = 'message ai-message typing-indicator';
    typingDiv.innerHTML = `
        <div class="message-avatar">${agent ? agent.icon : '🦅'}</div>
        <div class="message-content">
            <div class="typing-dots">
                <span class="typing-dot"></span>
                <span class="typing-dot"></span>
                <span class="typing-dot"></span>
            </div>
        </div>
    `;
    
    chatContainer.appendChild(typingDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
    
    return typingId;
}

/**
 * Remove typing indicator
 */
function removeTypingIndicator(typingId) {
    if (!typingId) return;
    const typingDiv = document.getElementById(typingId);
    if (typingDiv) typingDiv.remove();
}

/**
 * Show welcome message
 */
function showWelcomeMessage() {
    const greeting = window.getAgentGreeting(currentAgent);
    addMessageToChat('ai', greeting, currentAgent, false);
}

/**
 * Escape HTML to prevent XSS
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ===============================================
// UI CONTROLS
// ===============================================

/**
 * Clear current agent's chat history
 */
function clearChat() {
    if (confirm('Clear chat history with this agent?')) {
        messageHistory[currentAgent] = [];
        window.clearAgentHistory(currentAgent);
        loadAgentHistory(currentAgent);
    }
}

/**
 * Download chat history
 */
function downloadChat() {
    const messages = messageHistory[currentAgent] || [];
    const agent = AGENTS[currentAgent];
    
    const chatText = messages.map(msg => 
        `${msg.type === 'user' ? 'You' : agent.name}: ${msg.content}`
    ).join('\n\n');
    
    const blob = new Blob([chatText], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `osprey-chat-${currentAgent}-${Date.now()}.txt`;
    a.click();
    URL.revokeObjectURL(url);
}

// ===============================================
// EVENT LISTENERS
// ===============================================

document.addEventListener('DOMContentLoaded', () => {
    // Initialize AI
    initializeAI();

    // Message form submission
    const messageForm = document.getElementById('message-form') || 
                       document.querySelector('.message-form') ||
                       document.querySelector('form');
    
    const messageInput = document.getElementById('message-input') || 
                        document.querySelector('.message-input') ||
                        document.querySelector('input[type="text"]');

    if (messageForm && messageInput) {
        messageForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const message = messageInput.value.trim();
            if (!message) return;

            // Clear input
            messageInput.value = '';

            // Send message
            await sendMessage(message, currentAgent);
        });

        // Auto-focus input
        messageInput.focus();
    }

    // Agent tab switching
    document.querySelectorAll('.tab-btn, [data-agent]').forEach(btn => {
        btn.addEventListener('click', () => {
            const agentId = btn.dataset.tab || btn.dataset.agent;
            if (agentId && AGENTS[agentId]) {
                switchAgent(agentId);
            }
        });
    });

    // Clear chat button
    const clearBtn = document.getElementById('clear-chat-btn');
    if (clearBtn) {
        clearBtn.addEventListener('click', clearChat);
    }

    // Download chat button
    const downloadBtn = document.getElementById('download-chat-btn');
    if (downloadBtn) {
        downloadBtn.addEventListener('click', downloadChat);
    }

    console.log('🦅 Osprey Dashboard initialized');
});

// ===============================================
// KEYBOARD SHORTCUTS
// ===============================================

document.addEventListener('keydown', (e) => {
    // Ctrl/Cmd + K to focus message input
    if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        const input = document.getElementById('message-input') || 
                     document.querySelector('.message-input');
        if (input) input.focus();
    }

    // Ctrl/Cmd + L to clear chat
    if ((e.ctrlKey || e.metaKey) && e.key === 'l') {
        e.preventDefault();
        clearChat();
    }

    // Number keys 1-6 to switch agents
    if (e.key >= '1' && e.key <= '6') {
        const agentIndex = parseInt(e.key) - 1;
        const agentIds = Object.keys(AGENTS);
        if (agentIds[agentIndex]) {
            switchAgent(agentIds[agentIndex]);
        }
    }
});

// ===============================================
// GLOBAL EXPORTS
// ===============================================

window.sendMessage = sendMessage;
window.switchAgent = switchAgent;
window.clearChat = clearChat;
window.downloadChat = downloadChat;
window.AGENTS = AGENTS;

console.log('🦅 Osprey Dashboard Controller loaded');
