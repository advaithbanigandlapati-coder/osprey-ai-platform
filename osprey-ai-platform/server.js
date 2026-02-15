// server.js file
const express = require('express');
const bcrypt = require('bcryptjs');
const session = require('express-session');
const cors = require('cors');
const path = require('path');
const { v4: uuidv4 } = require('uuid');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 10000;

// Middleware
app.use(cors({
    origin: process.env.NODE_ENV === 'production'
        ? ['https://osprey-ai-platform.onrender.com']
        : ['http://localhost:10000'],
    credentials: true
}));
app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use(express.static(__dirname));

// Session configuration
app.use(session({
    secret: process.env.SESSION_SECRET || 'osprey-ai-labs-secret-change-in-production',
    resave: false,
    saveUninitialized: false,
    cookie: {
        httpOnly: true,
        secure: process.env.NODE_ENV === 'production',
        maxAge: 24 * 60 * 60 * 1000 // 24 hours
    }
}));

// ============================================================================
// USER DATABASE
// ============================================================================

const users = [
    {
        id: uuidv4(),
        username: 'Admin@B',
        password: bcrypt.hashSync('Admin123', 10),
        role: 'admin',
        name: 'Admin User',
        email: 'admin@ospreyai.com',
        theme: 'dark',
        notifications: true,
        created: new Date('2024-01-01')
    },
    {
        id: uuidv4(),
        username: 'User@B',
        password: bcrypt.hashSync('User123', 10),
        role: 'user',
        name: 'Standard User',
        email: 'user@ospreyai.com',
        theme: 'dark',
        notifications: true,
        created: new Date('2024-01-15')
    }
];

// ============================================================================
// ACTIVITY LOGS
// ============================================================================

let activityLogs = [
    {
        id: uuidv4(),
        timestamp: new Date(),
        user: 'Admin@B',
        action: 'System Start',
        details: 'Platform initialized',
        severity: 'info'
    }
];

// ============================================================================
// AI AGENTS REGISTRY
// ============================================================================

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

// ============================================================================
// WORKFLOWS DATABASE
// ============================================================================

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

// ============================================================================
// AUTH MIDDLEWARE
// ============================================================================

const requireAuth = (req, res, next) => {
    if (req.session.userId) {
        next();
    } else {
        res.status(401).json({ error: 'Unauthorized' });
    }
};

const requireAdmin = (req, res, next) => {
    if (req.session.userId && req.session.role === 'admin') {
        next();
    } else {
        res.status(403).json({ error: 'Forbidden - Admin access required' });
    }
};

// ============================================================================
// AUTH ROUTES  (dashboard.js calls /api/auth/*)
// ============================================================================

app.post('/api/auth/login', async (req, res) => {
    const { username, password } = req.body;

    if (!username || !password) {
        return res.status(400).json({ error: 'Username and password required' });
    }

    const user = users.find(u => u.username === username);

    if (!user) {
        return res.status(401).json({ error: 'Invalid credentials' });
    }

    const isValidPassword = await bcrypt.compare(password, user.password);

    if (!isValidPassword) {
        return res.status(401).json({ error: 'Invalid credentials' });
    }

    req.session.userId = user.id;
    req.session.username = user.username;
    req.session.role = user.role;

    activityLogs.unshift({
        id: uuidv4(),
        timestamp: new Date(),
        user: user.username,
        action: 'User Login',
        details: `${user.username} logged in`,
        severity: 'info'
    });

    res.json({
        success: true,
        user: {
            id: user.id,
            username: user.username,
            role: user.role,
            email: user.email,
            name: user.name,
            theme: user.theme
        }
    });
});

app.post('/api/auth/logout', (req, res) => {
    const username = req.session.username;
    req.session.destroy((err) => {
        if (err) {
            return res.status(500).json({ error: 'Logout failed' });
        }
        activityLogs.unshift({
            id: uuidv4(),
            timestamp: new Date(),
            user: username,
            action: 'User Logout',
            details: `${username} logged out`,
            severity: 'info'
        });
        res.json({ success: true });
    });
});

app.get('/api/auth/session', (req, res) => {
    if (!req.session.userId) {
        return res.json({ authenticated: false });
    }
    const user = users.find(u => u.id === req.session.userId);
    if (!user) {
        return res.json({ authenticated: false });
    }
    res.json({
        authenticated: true,
        user: {
            id: user.id,
            username: user.username,
            role: user.role,
            email: user.email,
            name: user.name,
            theme: user.theme
        }
    });
});

// ============================================================================
// AGENT ROUTES
// ============================================================================

app.get('/api/agents', requireAuth, (req, res) => {
    res.json({ success: true, agents: aiAgents });
});

app.get('/api/agents/:id', requireAuth, (req, res) => {
    const agent = aiAgents.find(a => a.id === req.params.id);
    if (!agent) {
        return res.status(404).json({ success: false, message: 'Agent not found' });
    }
    res.json({ success: true, agent });
});

// ============================================================================
// ANALYTICS ROUTE  (dashboard.js calls /api/analytics/dashboard)
// ============================================================================

app.get('/api/analytics/dashboard', requireAuth, (req, res) => {
    const totalRequests = aiAgents.reduce((sum, a) => sum + a.metrics.requests, 0);
    const avgPerformance = aiAgents.reduce((sum, a) => sum + a.metrics.successRate, 0) / aiAgents.length;

    res.json({
        totalAgents: aiAgents.length,
        activeAgents: aiAgents.filter(a => a.status === 'active').length,
        totalRequests,
        avgPerformance: avgPerformance.toFixed(1),
        uptime: 99.9,
        recentActivity: activityLogs.slice(0, 10)
    });
});

// ============================================================================
// ACTIVITY LOGS ROUTE  (dashboard.js calls /api/logs)
// ============================================================================

app.get('/api/logs', requireAuth, (req, res) => {
    const limit = parseInt(req.query.limit) || 50;
    res.json({ logs: activityLogs.slice(0, limit) });
});

// ============================================================================
// USER PROFILE ROUTES  (dashboard.js calls /api/user/profile)
// ============================================================================

app.get('/api/user/profile', requireAuth, (req, res) => {
    const user = users.find(u => u.id === req.session.userId);
    res.json({
        id: user.id,
        username: user.username,
        email: user.email,
        name: user.name,
        role: user.role,
        theme: user.theme,
        notifications: user.notifications,
        created: user.created
    });
});

app.put('/api/user/profile', requireAuth, async (req, res) => {
    const idx = users.findIndex(u => u.id === req.session.userId);
    const { name, email, theme, notifications, currentPassword, newPassword } = req.body;

    if (newPassword && currentPassword) {
        const valid = await bcrypt.compare(currentPassword, users[idx].password);
        if (!valid) {
            return res.status(401).json({ error: 'Current password is incorrect' });
        }
        users[idx].password = await bcrypt.hash(newPassword, 10);
    }

    if (name) users[idx].name = name;
    if (email) users[idx].email = email;
    if (theme) users[idx].theme = theme;
    if (notifications !== undefined) users[idx].notifications = notifications;

    activityLogs.unshift({
        id: uuidv4(),
        timestamp: new Date(),
        user: req.session.username,
        action: 'Profile Updated',
        details: 'User updated their profile settings',
        severity: 'info'
    });

    res.json({ success: true });
});

// ============================================================================
// WORKFLOWS ROUTE
// ============================================================================

app.get('/api/workflows', requireAuth, (req, res) => {
    res.json({ success: true, workflows });
});

// ============================================================================
// CHAT ROUTE
// ============================================================================

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

    for (const [key, value] of Object.entries(responses)) {
        if (lowerMessage.includes(key)) {
            response = value;
            break;
        }
    }

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

// ============================================================================
// METRICS ROUTE
// ============================================================================

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

// ============================================================================
// ACTIVITY FEED ROUTE
// ============================================================================

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

    res.json({ success: true, activities });
});

// ============================================================================
// PAGE ROUTES
// ============================================================================

app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'index.html'));
});

app.get('/signin', (req, res) => {
    res.sendFile(path.join(__dirname, 'signin.html'));
});

app.get('/dashboard', (req, res) => {
    res.sendFile(path.join(__dirname, 'dashboard.html'));
});

// ============================================================================
// ERROR HANDLING
// ============================================================================

app.use((err, req, res, next) => {
    console.error(err.stack);
    res.status(500).json({ success: false, message: 'Internal server error' });
});

// ============================================================================
// START SERVER
// ============================================================================

app.listen(PORT, '0.0.0.0', () => {
    console.log(`
╔═══════════════════════════════════════════════════════╗
║                                                       ║
║   🦅 Osprey AI Labs Platform - Running! 🚀           ║
║                                                       ║
╠═══════════════════════════════════════════════════════╣
║                                                       ║
║   Environment: ${(process.env.NODE_ENV || 'development').padEnd(38)}║
║   Port: ${String(PORT).padEnd(45)}║
║                                                       ║
║   Dashboard Credentials:                              ║
║   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   ║
║   Admin:  Admin@B  / Admin123                         ║
║   User:   User@B   / User123                          ║
║                                                       ║
╚═══════════════════════════════════════════════════════╝
    `);
});
