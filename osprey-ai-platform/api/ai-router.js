/**
 * AI Router - Routes requests to appropriate agents with specialized prompts
 * Location: /api/ai-router.js
 */

const ollamaHandler = require('./ollama-handler');

// Agent configurations with specialized system prompts
const AGENT_CONFIGS = {
    'content-writer': {
        name: 'Content Writer Pro',
        systemPrompt: 'You are a professional content writer. Create engaging, clear, SEO-optimized content for blogs, marketing, and social media. Be creative yet concise.',
        greeting: "Hi! I'm your Content Writer Pro. I can help you create engaging content. What would you like to write today?"
    },
    'code-assistant': {
        name: 'Code Assistant',
        systemPrompt: 'You are an expert programming assistant. Write clean, well-documented code. Debug issues systematically. Explain technical concepts clearly with examples.',
        greeting: "Hey! I'm your Code Assistant. I can help you write, debug, and optimize code. What are you working on?"
    },
    'data-analyst': {
        name: 'Data Analyst AI',
        systemPrompt: 'You are a skilled data analyst. Identify trends, patterns, and insights from data. Provide clear, actionable recommendations backed by analysis.',
        greeting: "Hello! I'm your Data Analyst. I can help you understand data and identify trends. What data would you like to explore?"
    },
    'support-bot': {
        name: 'Customer Support Bot',
        systemPrompt: 'You are a friendly, patient customer support agent. Resolve issues with clear step-by-step solutions. Be empathetic and helpful.',
        greeting: "Hi there! I'm your Customer Support Bot. I'm here to help resolve any issues. How can I assist you today?"
    },
    'research-assistant': {
        name: 'Research Assistant',
        systemPrompt: 'You are a knowledgeable research assistant. Provide well-researched, accurate information from multiple perspectives. Cite sources when relevant.',
        greeting: "Hello! I'm your Research Assistant. I can help you research topics and synthesize insights. What would you like to research?"
    },
    'marketing-strategist': {
        name: 'Marketing Strategist',
        systemPrompt: 'You are a creative marketing strategist. Develop data-driven campaigns, identify target audiences, create compelling messaging, and set measurable goals.',
        greeting: "Hi! I'm your Marketing Strategist. I can help you plan campaigns and develop marketing strategies. What are your goals?"
    }
};

class AIRouter {
    /**
     * Generate response from specific agent
     */
    async generateResponse(agentId, userMessage, conversationHistory = []) {
        // Get agent config or use default
        const agentConfig = AGENT_CONFIGS[agentId] || AGENT_CONFIGS['content-writer'];

        // Build conversation messages
        const messages = [
            { role: 'system', content: agentConfig.systemPrompt }
        ];

        // Add conversation history (last 10 messages to keep context manageable)
        const recentHistory = conversationHistory.slice(-10);
        messages.push(...recentHistory);

        // Add current user message
        messages.push({ role: 'user', content: userMessage });

        // Generate response
        const result = await ollamaHandler.chat(messages, agentId);

        return {
            ...result,
            agent: {
                id: agentId,
                name: agentConfig.name
            }
        };
    }

    /**
     * Get agent greeting message
     */
    getGreeting(agentId) {
        const agentConfig = AGENT_CONFIGS[agentId] || AGENT_CONFIGS['content-writer'];
        return agentConfig.greeting;
    }

    /**
     * Get all available agents
     */
    getAgents() {
        return Object.keys(AGENT_CONFIGS).map(id => ({
            id,
            name: AGENT_CONFIGS[id].name,
            greeting: AGENT_CONFIGS[id].greeting
        }));
    }

    /**
     * Check if agent exists
     */
    isValidAgent(agentId) {
        return AGENT_CONFIGS.hasOwnProperty(agentId);
    }
}

module.exports = new AIRouter();
