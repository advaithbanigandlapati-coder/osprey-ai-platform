/**
 * Ollama Handler - Manages Ollama AI on Render
 * Location: /api/ollama-handler.js
 */

class OllamaHandler {
    constructor() {
        this.baseUrl = process.env.OLLAMA_URL || 'http://localhost:11434';
        this.defaultModel = process.env.OLLAMA_MODEL || 'llama2';
        this.isReady = false;
    }

    /**
     * Check if Ollama is running
     */
    async checkHealth() {
        try {
            const response = await fetch(`${this.baseUrl}/api/tags`);
            if (response.ok) {
                this.isReady = true;
                const data = await response.json();
                console.log('✅ Ollama is running. Models:', data.models?.map(m => m.name));
                return true;
            }
            return false;
        } catch (error) {
            console.error('❌ Ollama not accessible:', error.message);
            return false;
        }
    }

    /**
     * Generate AI response
     */
    async generate(prompt, systemPrompt = '', agentId = 'default') {
        if (!this.isReady) {
            await this.checkHealth();
        }

        try {
            // Build the full prompt with system context
            const fullPrompt = systemPrompt 
                ? `${systemPrompt}\n\nUser: ${prompt}\nAssistant:`
                : prompt;

            const response = await fetch(`${this.baseUrl}/api/generate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    model: this.defaultModel,
                    prompt: fullPrompt,
                    stream: false,
                    options: {
                        temperature: 0.7,
                        top_p: 0.9,
                        num_predict: 1024
                    }
                })
            });

            if (!response.ok) {
                throw new Error(`Ollama returned ${response.status}`);
            }

            const data = await response.json();
            return {
                success: true,
                response: data.response,
                model: this.defaultModel,
                agent: agentId
            };

        } catch (error) {
            console.error('Ollama generation error:', error);
            return {
                success: false,
                error: error.message,
                fallback: this.getFallbackResponse(agentId)
            };
        }
    }

    /**
     * Chat-style generation with conversation history
     */
    async chat(messages, agentId = 'default') {
        if (!this.isReady) {
            await this.checkHealth();
        }

        try {
            const response = await fetch(`${this.baseUrl}/api/chat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    model: this.defaultModel,
                    messages: messages,
                    stream: false,
                    options: {
                        temperature: 0.7,
                        top_p: 0.9
                    }
                })
            });

            if (!response.ok) {
                throw new Error(`Ollama returned ${response.status}`);
            }

            const data = await response.json();
            return {
                success: true,
                response: data.message.content,
                model: this.defaultModel,
                agent: agentId
            };

        } catch (error) {
            console.error('Ollama chat error:', error);
            return {
                success: false,
                error: error.message,
                fallback: this.getFallbackResponse(agentId)
            };
        }
    }

    /**
     * Fallback responses if Ollama fails
     */
    getFallbackResponse(agentId) {
        const fallbacks = {
            'content-writer': "I can help you write content! However, I'm currently experiencing connectivity issues with the AI engine. Please try again in a moment.",
            'code-assistant': "I'm your code assistant! I'm having trouble connecting to the AI engine right now. Please refresh and try again.",
            'data-analyst': "I analyze data and trends! The AI engine is temporarily unavailable. Please try again shortly.",
            'support-bot': "I'm here to help! I'm experiencing technical difficulties connecting to the AI. Please try again in a moment.",
            'research-assistant': "I can research topics for you! The AI engine is currently unavailable. Please refresh the page.",
            'marketing-strategist': "I create marketing strategies! I'm having trouble connecting right now. Please try again soon.",
            'default': "I'm experiencing technical difficulties. Please try again in a moment."
        };

        return fallbacks[agentId] || fallbacks['default'];
    }

    /**
     * Get available models
     */
    async getModels() {
        try {
            const response = await fetch(`${this.baseUrl}/api/tags`);
            if (response.ok) {
                const data = await response.json();
                return data.models || [];
            }
            return [];
        } catch (error) {
            console.error('Error fetching models:', error);
            return [];
        }
    }

    /**
     * Change model
     */
    setModel(modelName) {
        this.defaultModel = modelName;
        console.log(`Model changed to: ${modelName}`);
    }
}

module.exports = new OllamaHandler();
