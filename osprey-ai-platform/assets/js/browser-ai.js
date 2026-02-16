/**
 * Osprey AI Platform - Browser AI with 6 Specialized Agents
 * Browser-compatible version (no ES6 modules)
 */

console.log('🦅 Loading Osprey AI System...');

// ===============================================
// AGENT SYSTEM PROMPTS (6 Specialized Agents)
// ===============================================

const AGENT_PROMPTS = {
    'content-writer': {
        name: 'Content Writer Pro',
        systemPrompt: `You are a professional content writer. Create engaging, clear content for blogs, marketing, and social media.`,
        greeting: "Hi! I'm your Content Writer Pro. I can help you create engaging content. What would you like to write today?"
    },
    'code-assistant': {
        name: 'Code Assistant',
        systemPrompt: `You are an expert programming assistant. Help write clean code, debug issues, and explain concepts clearly.`,
        greeting: "Hey! I'm your Code Assistant. I can help you write, debug, and optimize code. What are you working on?"
    },
    'data-analyst': {
        name: 'Data Analyst AI',
        systemPrompt: `You are a skilled data analyst. Analyze data, identify trends, and provide clear insights.`,
        greeting: "Hello! I'm your Data Analyst. I can help you understand data and identify trends. What data would you like to explore?"
    },
    'support-bot': {
        name: 'Customer Support Bot',
        systemPrompt: `You are a friendly customer support agent. Resolve issues quickly with clear, step-by-step solutions.`,
        greeting: "Hi there! I'm your Customer Support Bot. I'm here to help resolve any issues. How can I assist you today?"
    },
    'research-assistant': {
        name: 'Research Assistant',
        systemPrompt: `You are a knowledgeable research assistant. Research topics thoroughly and provide well-structured information.`,
        greeting: "Hello! I'm your Research Assistant. I can help you research topics and synthesize insights. What would you like to research?"
    },
    'marketing-strategist': {
        name: 'Marketing Strategist',
        systemPrompt: `You are a creative marketing strategist. Develop campaigns, identify audiences, and create winning strategies.`,
        greeting: "Hi! I'm your Marketing Strategist. I can help you plan campaigns and develop marketing strategies. What are your goals?"
    }
};

// ===============================================
// SIMPLE MOCK AI (No external dependencies)
// ===============================================

class OspreyBrowserAI {
    constructor() {
        this.isReady = false;
        this.isLoading = false;
        this.currentAgent = 'content-writer';
        this.conversationHistory = {};
        
        // Initialize conversation history
        Object.keys(AGENT_PROMPTS).forEach(agentId => {
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
            // Simulate loading with progress
            if (onProgress) onProgress({ status: 'downloading', progress: 0 });
            
            for (let i = 0; i <= 100; i += 10) {
                await new Promise(r => setTimeout(r, 100));
                if (onProgress) {
                    onProgress({ 
                        status: i < 100 ? 'downloading' : 'ready', 
                        progress: i 
                    });
                }
            }

            this.isReady = true;
            this.isLoading = false;
            console.log('✅ Osprey AI ready! All 6 agents online.');
            return true;

        } catch (error) {
            this.isLoading = false;
            console.error('❌ Osprey AI initialization failed:', error);
            throw error;
        }
    }

    setAgent(agentId) {
        if (AGENT_PROMPTS[agentId]) {
            this.currentAgent = agentId;
            if (!this.conversationHistory[agentId]) {
                this.conversationHistory[agentId] = [];
            }
        }
    }

    getAgentGreeting(agentId) {
        const agent = AGENT_PROMPTS[agentId] || AGENT_PROMPTS['content-writer'];
        return agent.greeting;
    }

    async generateResponse(message, agentId = null) {
        if (!this.isReady) await this.initialize();

        const activeAgent = agentId || this.currentAgent;
        const agent = AGENT_PROMPTS[activeAgent] || AGENT_PROMPTS['content-writer'];

        // Initialize conversation history
        if (!this.conversationHistory[activeAgent]) {
            this.conversationHistory[activeAgent] = [];
        }

        // Add user message
        this.conversationHistory[activeAgent].push({
            role: 'user',
            content: message
        });

        // Simulate thinking delay
        await new Promise(r => setTimeout(r, 800 + Math.random() * 1200));

        // Generate contextual response based on agent type and message
        let response = this.generateContextualResponse(message, agent, activeAgent);

        // Add assistant response
        this.conversationHistory[activeAgent].push({
            role: 'assistant',
            content: response
        });

        // Keep history manageable
        if (this.conversationHistory[activeAgent].length > 10) {
            this.conversationHistory[activeAgent] = 
                this.conversationHistory[activeAgent].slice(-10);
        }

        return response;
    }

    generateContextualResponse(message, agent, agentId) {
        const lowerMsg = message.toLowerCase();
        
        // Agent-specific responses
        const responses = {
            'content-writer': {
                keywords: ['write', 'blog', 'article', 'content', 'post', 'copy'],
                responses: [
                    "I'd be happy to help you write that! Here's an approach: Start with a compelling headline, craft an engaging introduction, develop your main points with supporting details, and conclude with a strong call-to-action. Would you like me to draft a specific section?",
                    "Great topic! For best results, I recommend: 1) Research your audience, 2) Create an outline, 3) Write in a conversational tone, 4) Use subheadings for readability, 5) Include relevant examples. What aspect would you like to focus on first?",
                    "I can help with that! Effective content should be clear, engaging, and valuable to your readers. Let's start by defining your target audience and main message. What's the primary goal of this piece?"
                ]
            },
            'code-assistant': {
                keywords: ['code', 'function', 'bug', 'error', 'program', 'debug', 'python', 'javascript'],
                responses: [
                    "I can help you with that! To write clean code, I recommend: 1) Plan your logic first, 2) Use descriptive variable names, 3) Break complex problems into smaller functions, 4) Add comments for clarity, 5) Test incrementally. What language are you working in?",
                    "Let me assist with that coding task. Best practices include proper error handling, input validation, and clear documentation. Would you like me to outline the function structure or help debug a specific issue?",
                    "Good question! For debugging, I suggest: 1) Check your syntax, 2) Verify variable types, 3) Use console.log() or print() statements, 4) Review your logic flow, 5) Test with different inputs. What error are you encountering?"
                ]
            },
            'data-analyst': {
                keywords: ['data', 'analyze', 'trend', 'chart', 'graph', 'statistics', 'metrics'],
                responses: [
                    "Great question about data analysis! I recommend: 1) Clean and validate your data first, 2) Identify key metrics, 3) Look for patterns and outliers, 4) Use appropriate visualization types, 5) Draw actionable insights. What type of data are you working with?",
                    "For effective data analysis, consider: visualizing trends with line charts for time series, bar charts for comparisons, and scatter plots for correlations. What insights are you hoping to uncover?",
                    "I can help analyze that! Key steps: 1) Define your question, 2) Collect relevant data, 3) Clean and prepare it, 4) Apply statistical methods, 5) Visualize findings. What's your analysis goal?"
                ]
            },
            'support-bot': {
                keywords: ['help', 'issue', 'problem', 'fix', 'reset', 'password', 'login'],
                responses: [
                    "I'm here to help! Let me guide you through this step-by-step: 1) First, let's identify the exact issue, 2) Check if there are any error messages, 3) Try the basic troubleshooting steps, 4) If needed, we'll escalate to technical support. Can you describe what's happening?",
                    "I understand this can be frustrating. Here's what we'll do: 1) Verify your account details, 2) Check system status, 3) Try refreshing or restarting, 4) Clear cache if needed. Have you tried any troubleshooting steps yet?",
                    "Let me assist you with that issue. To best help, I need to know: 1) When did this start? 2) What were you doing when it occurred? 3) Have you seen any error messages? Please provide these details."
                ]
            },
            'research-assistant': {
                keywords: ['research', 'study', 'information', 'learn', 'explain', 'history'],
                responses: [
                    "I'd be glad to research that topic! A thorough approach includes: 1) Identifying key concepts and terms, 2) Gathering information from reliable sources, 3) Analyzing different perspectives, 4) Synthesizing findings, 5) Organizing into a coherent summary. What specific aspect interests you most?",
                    "Interesting topic! For comprehensive research, I recommend: examining primary sources, cross-referencing multiple viewpoints, noting historical context, and identifying current developments. Shall I provide an overview or focus on a particular angle?",
                    "Let me help you explore this subject. Effective research involves: defining scope, collecting credible information, evaluating sources, identifying patterns, and presenting clear conclusions. What would you like to know specifically?"
                ]
            },
            'marketing-strategist': {
                keywords: ['marketing', 'campaign', 'strategy', 'audience', 'brand', 'promote'],
                responses: [
                    "Excellent marketing question! A successful strategy includes: 1) Define your target audience, 2) Identify unique value proposition, 3) Choose appropriate channels, 4) Create compelling messaging, 5) Set measurable goals. What product/service are we marketing?",
                    "For an effective campaign, consider: audience segmentation, competitive positioning, channel mix (social, email, content, paid), messaging consistency, and ROI tracking. What's your primary marketing objective?",
                    "Let's build a winning strategy! Key elements: 1) Market analysis, 2) Audience personas, 3) Value proposition, 4) Channel selection, 5) Content calendar, 6) Performance metrics. What's your target market?"
                ]
            }
        };

        // Find matching agent responses
        const agentResponses = responses[agentId] || responses['content-writer'];
        
        // Check if message contains agent keywords
        const hasKeyword = agentResponses.keywords.some(kw => lowerMsg.includes(kw));
        
        if (hasKeyword) {
            // Return random agent-specific response
            return agentResponses.responses[Math.floor(Math.random() * agentResponses.responses.length)];
        }

        // Generic helpful responses
        const genericResponses = [
            "I understand what you're asking. " + agent.systemPrompt.split('.')[0] + ". Could you provide more details so I can give you the most helpful response?",
            "That's an interesting question! Based on my role as " + agent.name + ", I can help you with that. What specific aspect would you like to explore?",
            "I'm here to assist! As your " + agent.name + ", I specialize in helping with these types of tasks. What would you like to focus on first?",
            "Great question! Let me help you with that from my perspective as " + agent.name + ". Can you tell me more about what you're trying to accomplish?"
        ];

        return genericResponses[Math.floor(Math.random() * genericResponses.length)];
    }

    clearHistory(agentId) {
        if (agentId) {
            this.conversationHistory[agentId] = [];
        } else {
            this.conversationHistory = {};
            Object.keys(AGENT_PROMPTS).forEach(id => {
                this.conversationHistory[id] = [];
            });
        }
    }

    getAgents() {
        return Object.keys(AGENT_PROMPTS).map(id => ({
            id,
            name: AGENT_PROMPTS[id].name,
            greeting: AGENT_PROMPTS[id].greeting
        }));
    }
}

// ===============================================
// INITIALIZE AND EXPOSE GLOBALLY
// ===============================================

const ospreyAI = new OspreyBrowserAI();

// Make available globally
window.ospreyAI = ospreyAI;
window.initializeOspreyAI = (onProgress) => ospreyAI.initialize(onProgress);
window.generateAIResponse = (message, agentId) => ospreyAI.generateResponse(message, agentId);
window.setActiveAgent = (agentId) => ospreyAI.setAgent(agentId);
window.getAgentGreeting = (agentId) => ospreyAI.getAgentGreeting(agentId);
window.getAvailableAgents = () => ospreyAI.getAgents();
window.clearAgentHistory = (agentId) => ospreyAI.clearHistory(agentId);

console.log('✅ Osprey AI System loaded with 6 specialized agents');
