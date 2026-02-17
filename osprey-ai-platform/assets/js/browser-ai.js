/**
 * Osprey AI - Browser AI Client (Ollama-Powered)
 * REPLACE: /assets/js/browser-ai.js
 * 
 * This connects YOUR browser to YOUR server's Ollama AI
 * No templates, no bullshit - just works with your existing code
 */

console.log('🦅 Osprey AI loading...');

// ===============================================
// OLLAMA AI CLIENT
// ===============================================

class OspreyBrowserAI {
    constructor() {
        this.isReady = false;
        this.apiBase = window.location.origin;
        this.conversationHistory = {};
    }

    async initialize(onProgress) {
        console.log('Initializing AI...');
        
        if (onProgress) {
            onProgress({ status: 'connecting', progress: 30 });
        }

        // Check Ollama backend
        try {
            const health = await fetch(`${this.apiBase}/api/ai/health`);
            const data = await health.json();
            
            if (data.status === 'healthy') {
                console.log('✅ Ollama connected:', data.models);
            } else {
                console.warn('⚠️ Ollama unavailable - fallback mode');
            }
        } catch (error) {
            console.warn('⚠️ Could not reach AI backend');
        }

        if (onProgress) {
            onProgress({ status: 'ready', progress: 100 });
        }

        this.isReady = true;
        return true;
    }

    async chat(message, agentId = 'content-writer') {
        if (!this.isReady) {
            await this.initialize();
        }

        try {
            const response = await fetch(`${this.apiBase}/api/ai/generate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                credentials: 'include', // Keep your session!
                body: JSON.stringify({
                    message: message,
                    agent: agentId,
                    history: this.conversationHistory[agentId] || []
                })
            });

            if (!response.ok) {
                throw new Error(`Server error: ${response.status}`);
            }

            const data = await response.json();

            // Save to history
            if (!this.conversationHistory[agentId]) {
                this.conversationHistory[agentId] = [];
            }
            this.conversationHistory[agentId].push(
                { role: 'user', content: message },
                { role: 'assistant', content: data.response }
            );

            // Keep last 20 messages only
            if (this.conversationHistory[agentId].length > 20) {
                this.conversationHistory[agentId] = 
                    this.conversationHistory[agentId].slice(-20);
            }

            return data.response || data.fallback || 'No response';

        } catch (error) {
            console.error('AI error:', error);
            return `Error: ${error.message}. Check if Ollama is running.`;
        }
    }

    clearHistory(agentId) {
        if (agentId) {
            this.conversationHistory[agentId] = [];
        } else {
            this.conversationHistory = {};
        }
    }

    getStatus() {
        return {
            isReady: this.isReady,
            backend: 'Ollama',
            apiBase: this.apiBase
        };
    }
}

// ===============================================
// INITIALIZE & EXPOSE
// ===============================================

const ospreyAI = new OspreyBrowserAI();

// Make available globally (same API as your existing code!)
window.ospreyAI = ospreyAI;
window.initializeOspreyAI = (onProgress) => ospreyAI.initialize(onProgress);
window.generateAIResponse = (message, agentId) => ospreyAI.chat(message, agentId);
window.clearAgentHistory = (agentId) => ospreyAI.clearHistory(agentId);
window.getAIStatus = () => ospreyAI.getStatus();

console.log('✅ Osprey AI loaded');
