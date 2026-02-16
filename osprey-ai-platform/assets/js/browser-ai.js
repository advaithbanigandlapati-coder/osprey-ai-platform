/**
 * Osprey AI Platform - Browser AI (Server-Powered via Ollama)
 * REPLACE your existing /assets/js/browser-ai.js with THIS FILE
 * 
 * This connects to YOUR server's Ollama AI - users install NOTHING!
 */

console.log('🦅 Loading Osprey AI System (Server-Powered)...');

// ===============================================
// CONFIGURATION
// ===============================================

const AI_CONFIG = {
    apiBase: '',  // Empty = same server (your Render URL)
    timeout: 30000,  // 30 seconds
    retryAttempts: 2
};

// ===============================================
// OSPREY BROWSER AI CLIENT
// ===============================================

class OspreyBrowserAI {
    constructor() {
        this.isReady = false;
        this.isLoading = false;
        this.currentAgent = 'content-writer';
        this.conversationHistory = {};
        this.apiBase = AI_CONFIG.apiBase || window.location.origin;
        
        // Initialize conversation history for each agent
        this.initializeHistory();
    }

    initializeHistory() {
        const agents = [
            'content-writer',
            'code-assistant',
            'data-analyst',
            'support-bot',
            'research-assistant',
            'marketing-strategist'
        ];

        agents.forEach(agentId => {
            this.conversationHistory[agentId] = [];
        });
    }

    async initialize(onProgress) {
        if (this.isReady) return true;
        if (this.isLoading) {
            while (!this.isReady) await new Promise(r => setTimeout(r, 100));
            return true;
        }

        this.isLoading = true;
        console.log('🦅 Initializing Osprey AI...');

        try {
            // Show progress
            if (onProgress) onProgress({ status: 'connecting', progress: 0 });

            // Check if AI backend is available
            const healthCheck = await this.checkHealth();
            
            if (onProgress) onProgress({ status: 'connecting', progress: 50 });

            if (healthCheck.status === 'healthy') {
                console.log('✅ Connected to Ollama AI');
                console.log('📦 Available models:', healthCheck.models);
            } else {
                console.warn('⚠️ AI backend unavailable - using fallback mode');
            }

            // Simulate final loading
            await new Promise(r => setTimeout(r, 500));
            if (onProgress) onProgress({ status: 'ready', progress: 100 });

            this.isReady = true;
            this.isLoading = false;
            console.log('✅ Osprey AI ready! All 6 agents online.');
            return true;

        } catch (error) {
            this.isLoading = false;
            this.isReady = true;  // Continue in fallback mode
            console.warn('⚠️ AI initialization issue:', error.message);
            if (onProgress) onProgress({ status: 'ready', progress: 100 });
            return true;
        }
    }

    async checkHealth() {
        try {
            const response = await fetch(`${this.apiBase}/api/ai/health`, {
                method: 'GET',
                headers: { 'Content-Type': 'application/json' }
            });

            if (response.ok) {
                return await response.json();
            }

            return { status: 'unavailable' };

        } catch (error) {
            console.error('Health check failed:', error);
            return { status: 'error' };
        }
    }

    setAgent(agentId) {
        if (this.conversationHistory.hasOwnProperty(agentId)) {
            this.currentAgent = agentId;
            console.log(`Agent switched to: ${agentId}`);
        }
    }

    async getAgentGreeting(agentId) {
        try {
            const response = await fetch(
                `${this.apiBase}/api/ai/agents/${agentId}/greeting`,
                {
                    method: 'GET',
                    headers: { 'Content-Type': 'application/json' }
                }
            );

            if (response.ok) {
                const data = await response.json();
                return data.greeting;
            }

        } catch (error) {
            console.error('Failed to get greeting:', error);
        }

        // Fallback greetings
        const fallbackGreetings = {
            'content-writer': "Hi! I'm your Content Writer Pro. What would you like to write today?",
            'code-assistant': "Hey! I'm your Code Assistant. What are you working on?",
            'data-analyst': "Hello! I'm your Data Analyst. What data would you like to explore?",
            'support-bot': "Hi there! I'm your Support Bot. How can I assist you?",
            'research-assistant': "Hello! I'm your Research Assistant. What would you like to research?",
            'marketing-strategist': "Hi! I'm your Marketing Strategist. What are your goals?"
        };

        return fallbackGreetings[agentId] || "Hello! How can I help you today?";
    }

    async generateResponse(message, agentId = null) {
        if (!this.isReady) await this.initialize();

        const activeAgent = agentId || this.currentAgent;

        // Initialize history if needed
        if (!this.conversationHistory[activeAgent]) {
            this.conversationHistory[activeAgent] = [];
        }

        try {
            // Call your server's AI endpoint
            const response = await fetch(`${this.apiBase}/api/ai/generate`, {
                method: 'POST',
                headers: { 
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    message: message,
                    agent: activeAgent,
                    history: this.conversationHistory[activeAgent]
                })
            });

            if (!response.ok) {
                throw new Error(`Server returned ${response.status}`);
            }

            const data = await response.json();

            // Handle fallback mode
            if (!data.success && data.fallback) {
                console.warn('Using fallback response');
                return data.fallback;
            }

            // Update conversation history
            this.conversationHistory[activeAgent].push(
                { role: 'user', content: message },
                { role: 'assistant', content: data.response }
            );

            // Keep history manageable (last 10 messages)
            if (this.conversationHistory[activeAgent].length > 20) {
                this.conversationHistory[activeAgent] = 
                    this.conversationHistory[activeAgent].slice(-20);
            }

            return data.response;

        } catch (error) {
            console.error('AI generation error:', error);
            
            // Fallback response
            return `I'm having trouble connecting to the AI server. Please check your connection and try again. (Error: ${error.message})`;
        }
    }

    clearHistory(agentId) {
        if (agentId) {
            this.conversationHistory[agentId] = [];
        } else {
            this.initializeHistory();
        }
    }

    async getAgents() {
        try {
            const response = await fetch(`${this.apiBase}/api/ai/agents`);
            if (response.ok) {
                const data = await response.json();
                return data.agents;
            }
        } catch (error) {
            console.error('Failed to fetch agents:', error);
        }

        // Fallback agent list
        return [
            { id: 'content-writer', name: 'Content Writer Pro' },
            { id: 'code-assistant', name: 'Code Assistant' },
            { id: 'data-analyst', name: 'Data Analyst AI' },
            { id: 'support-bot', name: 'Customer Support Bot' },
            { id: 'research-assistant', name: 'Research Assistant' },
            { id: 'marketing-strategist', name: 'Marketing Strategist' }
        ];
    }

    getStatus() {
        return {
            isReady: this.isReady,
            currentAgent: this.currentAgent,
            backend: 'Ollama (via server)',
            apiBase: this.apiBase
        };
    }
}

// ===============================================
// INITIALIZE AND EXPOSE GLOBALLY
// ===============================================

const ospreyAI = new OspreyBrowserAI();

// Make available globally (same API as before!)
window.ospreyAI = ospreyAI;
window.initializeOspreyAI = (onProgress) => ospreyAI.initialize(onProgress);
window.generateAIResponse = (message, agentId) => ospreyAI.generateResponse(message, agentId);
window.setActiveAgent = (agentId) => ospreyAI.setAgent(agentId);
window.getAgentGreeting = (agentId) => ospreyAI.getAgentGreeting(agentId);
window.getAvailableAgents = () => ospreyAI.getAgents();
window.clearAgentHistory = (agentId) => ospreyAI.clearHistory(agentId);
window.getAIStatus = () => ospreyAI.getStatus();

console.log('✅ Osprey AI System loaded (Server-Powered)');
