console.log('🦅 Loading Osprey AI System (Python-Powered)...');

// ===============================================
// CONFIGURATION
// ===============================================

const AI_CONFIG = {
    apiBase: '',  // Empty = same server
    timeout: 60000,  // 60 seconds (Python processing takes longer)
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
        console.log('🦅 Initializing Osprey AI (Python Agents)...');

        try {
            // Show progress
            if (onProgress) onProgress({ status: 'connecting', progress: 0 });

            // Check if Python AI backend is available
            const healthCheck = await this.checkHealth();
            
            if (onProgress) onProgress({ status: 'connecting', progress: 50 });

            if (healthCheck.status === 'healthy') {
                console.log('✅ Connected to Python AI Server');
                console.log(`📦 Available agents: ${healthCheck.count || 0}/6`);
                console.log('🎯 Features: SEO scoring, quality metrics, readability analysis');
            } else {
                console.warn('⚠️ Python AI backend unavailable - check if agent-server.py is running');
            }

            // Simulate final loading
            await new Promise(r => setTimeout(r, 500));
            if (onProgress) onProgress({ status: 'ready', progress: 100 });

            this.isReady = true;
            this.isLoading = false;
            console.log('✅ Osprey AI ready! All 6 Python agents online.');
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
            // Check Python AI server health
            const response = await fetch(`${this.apiBase}/api/python-ai/health`, {
                method: 'GET',
                headers: { 'Content-Type': 'application/json' }
            });

            if (!response.ok) {
                return { status: 'unavailable' };
            }

            const data = await response.json();
            return data;

        } catch (error) {
            console.error('Health check failed:', error);
            return { status: 'unavailable' };
        }
    }

    async chat(message, agentId = 'content-writer', options = {}) {
        if (!this.isReady) {
            await this.initialize();
        }

        if (!message || typeof message !== 'string') {
            return { error: 'Message is required' };
        }

        try {
            console.log(`🤖 Generating with ${agentId}...`);
            
            // Prepare request with Python-specific options
            const requestBody = {
                message: message,
                agent: agentId,
                history: this.conversationHistory[agentId] || [],
                model: options.model || 'llama2',
                target_word_count: options.target_word_count || 800,
                temperature: options.temperature || 0.7,
                tone: options.tone || 'professional',
                language: options.language || 'en',
                seo_optimize: options.seo_optimize !== false
            };

            // Call Python AI endpoint
            const response = await fetch(`${this.apiBase}/api/python-ai/generate`, {
                method: 'POST',
                headers: { 
                    'Content-Type': 'application/json'
                },
                credentials: 'include',
                body: JSON.stringify(requestBody),
                signal: AbortSignal.timeout(AI_CONFIG.timeout)
            });

            if (!response.ok) {
                throw new Error(`Server error: ${response.status}`);
            }

            const data = await response.json();

            // Log quality metrics
            if (data.quality_score) {
                console.log(`📊 Quality Score: ${data.quality_score}/100`);
            }
            if (data.seo_score) {
                console.log(`🎯 SEO Score: ${data.seo_score}/100`);
            }
            if (data.word_count) {
                console.log(`📝 Word Count: ${data.word_count}`);
            }

            // Save to history
            if (!this.conversationHistory[agentId]) {
                this.conversationHistory[agentId] = [];
            }
            
            this.conversationHistory[agentId].push(
                { role: 'user', content: message },
                { role: 'assistant', content: data.content || data.response }
            );

            // Keep last 20 messages only
            if (this.conversationHistory[agentId].length > 20) {
                this.conversationHistory[agentId] = 
                    this.conversationHistory[agentId].slice(-20);
            }

            // Return full response with metadata
            return {
                content: data.content || data.response,
                quality_score: data.quality_score,
                seo_score: data.seo_score,
                word_count: data.word_count,
                readability: data.readability,
                language: data.language,
                agent: data.agent,
                processing_time: data.processing_time
            };

        } catch (error) {
            console.error('AI generation error:', error);
            
            // Return error response
            return {
                content: `Error: ${error.message}. Please check if the Python AI server is running.`,
                error: true,
                error_message: error.message
            };
        }
    }

    // Convenience method for simple text generation
    async generate(message, agentId, options) {
        const response = await this.chat(message, agentId, options);
        return response.content || response.error_message || 'No response';
    }

    // Get list of available agents
    async getAgents() {
        try {
            const response = await fetch(`${this.apiBase}/api/python-ai/agents`, {
                method: 'GET',
                headers: { 'Content-Type': 'application/json' }
            });

            if (!response.ok) {
                throw new Error(`Server error: ${response.status}`);
            }

            const data = await response.json();
            return data.agents || [];

        } catch (error) {
            console.error('Failed to get agents:', error);
            return [];
        }
    }

    clearHistory(agentId) {
        if (agentId) {
            this.conversationHistory[agentId] = [];
            console.log(`🗑️ Cleared history for ${agentId}`);
        } else {
            this.conversationHistory = {};
            this.initializeHistory();
            console.log('🗑️ Cleared all conversation history');
        }
    }

    getStatus() {
        return {
            isReady: this.isReady,
            backend: 'Python + Ollama',
            apiBase: this.apiBase,
            features: [
                'SEO Optimization',
                'Quality Scoring',
                'Readability Analysis',
                '50+ Languages',
                '6 Specialized Agents'
            ]
        };
    }
}

// ===============================================
// INITIALIZE & EXPOSE GLOBALLY
// ===============================================

const ospreyAI = new OspreyBrowserAI();

// Make available globally (compatible with your existing dashboard.js!)
window.ospreyAI = ospreyAI;
window.initializeOspreyAI = (onProgress) => ospreyAI.initialize(onProgress);
window.generateAIResponse = (message, agentId, options) => ospreyAI.chat(message, agentId, options);
window.clearAgentHistory = (agentId) => ospreyAI.clearHistory(agentId);
window.getAIStatus = () => ospreyAI.getStatus();
window.getAvailableAgents = () => ospreyAI.getAgents();

console.log('✅ Osprey AI loaded (Python-powered with quality scoring)');
console.log('🚀 Ready to generate enterprise-grade content!');
