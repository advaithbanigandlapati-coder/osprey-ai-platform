import { pipeline } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.1';

// ===============================================
// AGENT SYSTEM PROMPTS (6 Specialized Agents)
// ===============================================

const AGENT_PROMPTS = {
    'content-writer': {
        name: 'Content Writer Pro',
        systemPrompt: `You are a professional content writer specializing in creating engaging, SEO-optimized content. You excel at:
- Blog posts and articles
- Social media content
- Marketing copy
- Product descriptions
- Email campaigns
Your writing is clear, persuasive, and tailored to the target audience.`,
        greeting: "Hi! I'm your Content Writer Pro. I can help you create engaging content, from blog posts to marketing copy. What would you like to write today?"
    },

    'code-assistant': {
        name: 'Code Assistant',
        systemPrompt: `You are an expert programming assistant who helps with:
- Writing clean, efficient code in multiple languages
- Debugging and troubleshooting
- Code reviews and best practices
- Explaining complex programming concepts
- Providing code examples and documentation
You write production-ready code with proper error handling.`,
        greeting: "Hey! I'm your Code Assistant. I can help you write, debug, and optimize code. What are you working on?"
    },

    'data-analyst': {
        name: 'Data Analyst AI',
        systemPrompt: `You are a skilled data analyst who specializes in:
- Data analysis and interpretation
- Statistical insights
- Trend identification
- Creating data visualizations
- Generating actionable recommendations
You explain complex data in simple, understandable terms.`,
        greeting: "Hello! I'm your Data Analyst. I can help you understand your data, identify trends, and make data-driven decisions. What data would you like to explore?"
    },

    'support-bot': {
        name: 'Customer Support Bot',
        systemPrompt: `You are a friendly, patient customer support agent who:
- Resolves customer issues quickly
- Provides clear, step-by-step solutions
- Maintains a positive, empathetic tone
- Escalates complex issues when needed
- Follows up to ensure satisfaction
You always prioritize customer happiness and satisfaction.`,
        greeting: "Hi there! I'm your Customer Support Bot. I'm here to help resolve any issues you're experiencing. How can I assist you today?"
    },

    'research-assistant': {
        name: 'Research Assistant',
        systemPrompt: `You are a knowledgeable research assistant who helps with:
- Topic research and analysis
- Summarizing complex information
- Finding reliable sources
- Synthesizing multiple perspectives
- Providing well-structured reports
You deliver accurate, well-researched information with proper context.`,
        greeting: "Hello! I'm your Research Assistant. I can help you research topics, analyze information, and synthesize insights. What would you like to research?"
    },

    'marketing-strategist': {
        name: 'Marketing Strategist',
        systemPrompt: `You are a creative marketing strategist who excels at:
- Developing marketing campaigns
- Identifying target audiences
- Creating brand messaging
- Competitive analysis
- ROI optimization strategies
You think strategically and provide actionable marketing plans.`,
        greeting: "Hi! I'm your Marketing Strategist. I can help you plan campaigns, identify audiences, and develop winning marketing strategies. What are your marketing goals?"
    }
};

// ===============================================
// BROWSER AI ENGINE
// ===============================================

class OspreyBrowserAI {
    constructor() {
        this.generator = null;
        this.isReady = false;
        this.isLoading = false;
        this.currentAgent = 'content-writer';
        this.conversationHistory = {};
    }

    /**
     * Initialize the AI model
     */
    async initialize(onProgress) {
        if (this.isReady) return this.generator;
        if (this.isLoading) {
            while (!this.isReady) await new Promise(r => setTimeout(r, 100));
            return this.generator;
        }

        this.isLoading = true;
        console.log('🦅 Loading Osprey AI to browser...');

        try {
            if (onProgress) onProgress({ status: 'downloading', progress: 0 });

            // Load DistilGPT2 model (82MB, cached after first load)
            this.generator = await pipeline('text-generation', 'Xenova/distilgpt2', {
                progress_callback: (data) => {
                    if (onProgress && data.progress) {
                        onProgress({
                            status: data.status || 'downloading',
                            progress: Math.round(data.progress)
                        });
                    }
                }
            });

            this.isReady = true;
            this.isLoading = false;
            if (onProgress) onProgress({ status: 'ready', progress: 100 });
            console.log('✅ Osprey AI ready! All 6 agents online.');

            return this.generator;
        } catch (error) {
            this.isLoading = false;
            console.error('❌ Osprey AI load failed:', error);
            throw error;
        }
    }

    /**
     * Set the active agent
     */
    setAgent(agentId) {
        if (AGENT_PROMPTS[agentId]) {
            this.currentAgent = agentId;
            if (!this.conversationHistory[agentId]) {
                this.conversationHistory[agentId] = [];
            }
        }
    }

    /**
     * Get agent greeting message
     */
    getAgentGreeting(agentId) {
        const agent = AGENT_PROMPTS[agentId] || AGENT_PROMPTS['content-writer'];
        return agent.greeting;
    }

    /**
     * Generate response from current agent
     */
    async generateResponse(message, agentId = null) {
        if (!this.isReady) await this.initialize();

        const activeAgent = agentId || this.currentAgent;
        const agent = AGENT_PROMPTS[activeAgent] || AGENT_PROMPTS['content-writer'];

        // Initialize conversation history for this agent if needed
        if (!this.conversationHistory[activeAgent]) {
            this.conversationHistory[activeAgent] = [];
        }

        // Add user message to history
        this.conversationHistory[activeAgent].push({
            role: 'user',
            content: message
        });

        // Build prompt with agent context
        const systemContext = agent.systemPrompt;
        
        // Get recent conversation (last 3 exchanges to keep context short)
        const recentHistory = this.conversationHistory[activeAgent].slice(-6);
        const conversationText = recentHistory.map(msg => 
            `${msg.role === 'user' ? 'User' : 'Assistant'}: ${msg.content}`
        ).join('\n');

        const fullPrompt = `${systemContext}\n\n${conversationText}\n\nAssistant:`;

        try {
            // Generate response
            const result = await this.generator(fullPrompt, {
                max_new_tokens: 200,
                temperature: 0.8,
                do_sample: true,
                top_k: 50,
                top_p: 0.95,
                repetition_penalty: 1.2,
                pad_token_id: 50256  // GPT-2 EOS token
            });

            // Extract response
            let response = result[0].generated_text;
            response = response.split('Assistant:').pop() || response;
            response = response.split('User:')[0]; // Stop at next user turn
            response = response.trim();

            // Clean up
            response = response.replace(/^[\s\n]+|[\s\n]+$/g, '');
            
            // If response is too short or empty, provide a fallback
            if (response.length < 10) {
                response = "I understand. Could you provide more details so I can assist you better?";
            }

            // Add assistant response to history
            this.conversationHistory[activeAgent].push({
                role: 'assistant',
                content: response
            });

            // Keep history manageable (last 10 messages)
            if (this.conversationHistory[activeAgent].length > 10) {
                this.conversationHistory[activeAgent] = 
                    this.conversationHistory[activeAgent].slice(-10);
            }

            return response;

        } catch (error) {
            console.error('Error generating response:', error);
            throw new Error('Failed to generate response. Please try again.');
        }
    }

    /**
     * Clear conversation history for an agent
     */
    clearHistory(agentId) {
        if (agentId) {
            this.conversationHistory[agentId] = [];
        } else {
            this.conversationHistory = {};
        }
    }

    /**
     * Get all available agents
     */
    getAgents() {
        return Object.keys(AGENT_PROMPTS).map(id => ({
            id,
            name: AGENT_PROMPTS[id].name,
            greeting: AGENT_PROMPTS[id].greeting
        }));
    }
}

// ===============================================
// SINGLETON INSTANCE & GLOBAL FUNCTIONS
// ===============================================

const ospreyAI = new OspreyBrowserAI();

// Make available globally
window.ospreyAI = ospreyAI;

// Convenience functions
window.initializeOspreyAI = (onProgress) => ospreyAI.initialize(onProgress);
window.generateAIResponse = (message, agentId) => ospreyAI.generateResponse(message, agentId);
window.setActiveAgent = (agentId) => ospreyAI.setAgent(agentId);
window.getAgentGreeting = (agentId) => ospreyAI.getAgentGreeting(agentId);
window.getAvailableAgents = () => ospreyAI.getAgents();
window.clearAgentHistory = (agentId) => ospreyAI.clearHistory(agentId);

// Export for module systems
export default ospreyAI;
export {
    ospreyAI,
    AGENT_PROMPTS
};

console.log('🦅 Osprey AI System loaded with 6 specialized agents');
