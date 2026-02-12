const express = require('express');
const bcrypt = require('bcryptjs');
const session = require('express-session');
const path = require('path');
const { v4: uuidv4 } = require('uuid');

const app = express();
const PORT = process.env.PORT || 10000; // Render uses PORT env variable

// Middleware
app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use(express.static(__dirname)); // FIXED: Serve static files from current directory

// Session configuration
app.use(session({
    secret: process.env.SESSION_SECRET || 'osprey-ai-labs-secret-change-in-production',
    resave: false,
    saveUninitialized: false,
    cookie: {
        httpOnly: true,
        secure: false, // Set to true in production with HTTPS
        maxAge: 24 * 60 * 60 * 1000 // 24 hours
    }
}));

// User database (replace with real DB in production)
const users = {
    'admin': {
        username: 'admin',
        password: bcrypt.hashSync('admin123', 10),
        role: 'admin',
        name: 'Admin User',
        email: 'admin@ospreyai.com'
    },
    'demo': {
        username: 'demo',
        password: bcrypt.hashSync('demo123', 10),
        role: 'user',
        name: 'Demo User',
        email: 'demo@ospreyai.com'
    },
    'investor': {
        username: 'investor',
        password: bcrypt.hashSync('investor123', 10),
        role: 'user',
        name: 'Investor Access',
        email: 'investor@ospreyai.com'
    }
};

// AI Agents Registry
const aiAgents = [
    {
        id: 'rag-001',
        name: 'RAG Document Processor',
        type: 'document_processing',
        description: 'Advanced document processing with semantic search and knowledge retrieval',
        status: 'active',
        model: 'qwen2.5:7b',
        tools: ['document_parse', 'vector_search', 'summarize', 'extract_entities'],
        metrics: {
            requests: 15420,
            successRate: 99.8,
            avgResponseTime: 234
        }
    },
    {
        id: 'nl2sql-002',
        name: 'NL2SQL Builder',
        type: 'database_query',
        description: 'Natural language to SQL conversion with query optimization',
        status: 'active',
        model: 'qwen2.5:7b',
        tools: ['schema_analysis', 'query_generation', 'query_validation', 'results_formatting'],
        metrics: {
            requests: 8934,
            successRate: 98.5,
            avgResponseTime: 189
        }
    },
    {
        id: 'workflow-003',
        name: 'Workflow Orchestrator',
        type: 'automation',
        description: 'Multi-agent workflow coordination and task automation',
        status: 'active',
        model: 'qwen2.5:7b',
        tools: ['task_planning', 'agent_coordination', 'error_handling', 'monitoring'],
        metrics: {
            requests: 12567,
            successRate: 99.2,
            avgResponseTime: 456
        }
    },
    {
        id: 'analytics-004',
        name: 'Analytics Engine',
        type: 'data_analysis',
        description: 'Deep insights with predictive analytics and trend analysis',
        status: 'training',
        model: 'qwen2.5:7b',
        tools: ['data_analysis', 'visualization', 'prediction', 'reporting'],
        metrics: {
            requests: 6783,
            successRate: 97.3,
            avgResponseTime: 678
        }
    },
    {
        id: 'code-005',
        name: 'Code Assistant',
        type: 'development',
        description: 'AI-powered code generation, review, and optimization',
        status: 'active',
        model: 'qwen2.5:7b',
        tools: ['code_generation', 'code_review', 'debugging', 'optimization'],
        metrics: {
            requests: 9876,
            successRate: 98.9,
            avgResponseTime: 312
        }
    }
];

// Workflows database
const workflows = [
    {
        id: 'wf-001',
        name: 'Customer Onboarding',
        trigger: 'New User Signup',
        status: 'active',
        lastRun: new Date(Date.now() - 10 * 60 * 1000),
        successRate: 99.8,
        steps: 5
    },
    {
        id: 'wf-002',
        name: 'Data Processing Pipeline',
        trigger: 'Schedule (Hourly)',
        status: 'active',
        lastRun: new Date(Date.now() - 45 * 60 * 1000),
        successRate: 98.2,
        steps: 8
    },
    {
        id: 'wf-003',
        name: 'Report Generation',
        trigger: 'Manual Trigger',
        status: 'running',
        lastRun: new Date(Date.now() - 2 * 60 * 60 * 1000),
        successRate: 100,
        steps: 6
    }
];

// Authentication middleware
const requireAuth = (req, res, next) => {
    if (!req.session.user) {
        return res.status(401).json({ 
            success: false, 
            message: 'Authentication required' 
        });
    }
    next();
};

const requireAdmin = (req, res, next) => {
    if (!req.session.user || req.session.user.role !== 'admin') {
        return res.status(403).json({ 
            success: false, 
            message: 'Admin access required' 
        });
    }
    next();
};

// Routes
app.post('/api/login', async (req, res) => {
    const { username, password } = req.body;
    
    if (!username || !password) {
        return res.status(400).json({ 
            success: false, 
            message: 'Username and password required' 
        });
    }
    
    const user = users[username.toLowerCase()];
    
    if (!user) {
        return res.status(401).json({ 
            success: false, 
            message: 'Invalid credentials' 
        });
    }
    
    const isValidPassword = await bcrypt.compare(password, user.password);
    
    if (!isValidPassword) {
        return res.status(401).json({ 
            success: false, 
            message: 'Invalid credentials' 
        });
    }
    
    req.session.user = {
        username: user.username,
        role: user.role,
        name: user.name,
        email: user.email
    };
    
    res.json({ 
        success: true, 
        user: req.session.user
    });
});

app.post('/api/logout', (req, res) => {
    req.session.destroy((err) => {
        if (err) {
            return res.status(500).json({ 
                success: false, 
                message: 'Logout failed' 
            });
        }
        res.json({ success: true, message: 'Logged out successfully' });
    });
});

app.get('/api/session', (req, res) => {
    if (req.session.user) {
        res.json({ 
            success: true, 
            user: req.session.user 
        });
    } else {
        res.status(401).json({ 
            success: false, 
            message: 'Not authenticated' 
        });
    }
});

// AI Agents endpoints
app.get('/api/agents', requireAuth, (req, res) => {
    res.json({ 
        success: true, 
        agents: aiAgents 
    });
});

app.get('/api/agents/:id', requireAuth, (req, res) => {
    const agent = aiAgents.find(a => a.id === req.params.id);
    if (!agent) {
        return res.status(404).json({ 
            success: false, 
            message: 'Agent not found' 
        });
    }
    res.json({ 
        success: true, 
        agent 
    });
});

// Workflows endpoints
app.get('/api/workflows', requireAuth, (req, res) => {
    res.json({ 
        success: true, 
        workflows 
    });
});

// Chat endpoint with enhanced responses
app.post('/api/chat', requireAuth, (req, res) => {
    const { message, agentId } = req.body;
    
    const responses = {
        'hello': 'Hello! I\'m your Osprey AI assistant. How can I help you optimize your workflows today?',
        'hi': 'Hi there! Ready to streamline your operations with AI?',
        'help': 'I can assist you with:\n• Managing AI agents and workflows\n• Monitoring system performance\n• Analyzing data and generating insights\n• Troubleshooting issues\n• Optimizing processes\n\nWhat would you like to explore?',
        'agents': `You currently have ${aiAgents.filter(a => a.status === 'active').length} active AI agents. Your RAG Document Processor and NL2SQL Builder are performing exceptionally well with 99%+ success rates.`,
        'performance': `Your platform is performing excellently:\n• 99.2% overall success rate\n• 247ms average response time\n• 847K+ requests processed\n• All critical systems operational\n• 24 agents running smoothly`,
        'status': 'All systems are operational. Current load: 42% CPU, 68% Memory. Network performance is optimal.',
        'workflows': `You have ${workflows.length} workflows configured:\n• ${workflows.filter(w => w.status === 'active').length} active\n• ${workflows.filter(w => w.status === 'running').length} currently running\n• Average success rate: ${(workflows.reduce((acc, w) => acc + w.successRate, 0) / workflows.length).toFixed(1)}%`,
        'default': 'I understand you\'re asking about that. As your AI assistant, I\'m here to help you optimize your operations with Osprey AI Labs. Could you be more specific about what you need?'
    };
    
    const lowerMessage = message.toLowerCase();
    let response = responses.default;
    
    // Simple keyword matching
    for (const [key, value] of Object.entries(responses)) {
        if (lowerMessage.includes(key)) {
            response = value;
            break;
        }
    }
    
    // Add agent-specific responses
    if (agentId) {
        const agent = aiAgents.find(a => a.id === agentId);
        if (agent) {
            response = `[${agent.name}] ${response}`;
        }
    }
    
    res.json({ 
        success: true, 
        message: response,
        timestamp: new Date().toISOString(),
        agentId: agentId || 'general'
    });
});

// Dashboard metrics
app.get('/api/metrics/overview', requireAuth, (req, res) => {
    const totalRequests = aiAgents.reduce((sum, agent) => sum + agent.metrics.requests, 0);
    const avgSuccessRate = aiAgents.reduce((sum, agent) => sum + agent.metrics.successRate, 0) / aiAgents.length;
    const avgResponseTime = aiAgents.reduce((sum, agent) => sum + agent.metrics.avgResponseTime, 0) / aiAgents.length;
    
    res.json({
        success: true,
        metrics: {
            activeAgents: aiAgents.filter(a => a.status === 'active').length,
            totalRequests,
            successRate: avgSuccessRate.toFixed(1),
            avgResponseTime: Math.round(avgResponseTime),
            systemHealth: {
                cpu: 42,
                memory: 68,
                storage: 34,
                network: 21
            }
        }
    });
});

// Activity feed
app.get('/api/activity/recent', requireAuth, (req, res) => {
    const activities = [
        {
            agent: 'RAG Document Processor',
            action: 'Document processed',
            status: 'success',
            timestamp: new Date(Date.now() - 2 * 60 * 1000)
        },
        {
            agent: 'NL2SQL Builder',
            action: 'Query executed',
            status: 'success',
            timestamp: new Date(Date.now() - 15 * 60 * 1000)
        },
        {
            agent: 'Workflow Orchestrator',
            action: 'Workflow triggered',
            status: 'success',
            timestamp: new Date(Date.now() - 60 * 60 * 1000)
        },
        {
            agent: 'Analytics Engine',
            action: 'Report generated',
            status: 'processing',
            timestamp: new Date(Date.now() - 5 * 60 * 60 * 1000)
        }
    ];
    
    res.json({
        success: true,
        activities
    });
});

// Serve index.html for root
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'index.html')); // FIXED: Serve from current directory
});

// Error handling
app.use((err, req, res, next) => {
    console.error(err.stack);
    res.status(500).json({
        success: false,
        message: 'Internal server error'
    });
});

// Start server
app.listen(PORT, '0.0.0.0', () => {
    console.log(`
╔═══════════════════════════════════════════════════════╗
║                                                       ║
║   🦅 Osprey AI Labs Platform - Running! 🚀           ║
║                                                       ║
╠═══════════════════════════════════════════════════════╣
║                                                       ║
║   Environment: ${process.env.NODE_ENV || 'development'}                           ║
║   Port: ${PORT}                                        ║
║                                                       ║
║   Demo Credentials:                                   ║
║   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   ║
║   Admin:    admin / admin123                          ║
║   Demo:     demo / demo123                            ║
║   Investor: investor / investor123                    ║
║                                                       ║
║   Ready for investor demo! ✨                         ║
║                                                       ║
╚═══════════════════════════════════════════════════════╝
    `);
});
