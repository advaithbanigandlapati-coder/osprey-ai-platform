// server.js — Osprey AI Labs Platform
const express = require('express');
const bcrypt  = require('bcryptjs');
const session = require('express-session');
const cors    = require('cors');
const path    = require('path');
const { v4: uuidv4 } = require('uuid');
require('dotenv').config();
const fetch = require('node-fetch'); // NEW: For Python AI proxy
const aiRouter = require('./api/ai-router');
const ollamaHandler = require('./api/ollama-handler');

const app  = express();
const PORT = process.env.PORT || 10000;

// ─────────────────────────────────────────────────────────────────────────────
// PYTHON AI SERVER - NEW: For advanced Python agents
// ─────────────────────────────────────────────────────────────────────────────
const PYTHON_SERVER = process.env.PYTHON_SERVER_URL || 'http://localhost:8001';

// ─────────────────────────────────────────────────────────────────────────────
// CRITICAL: trust proxy MUST come first — without this, Render's reverse
// proxy breaks secure session cookies and every auth check returns 401
// ─────────────────────────────────────────────────────────────────────────────
app.set('trust proxy', 1);

// ── CORS ──────────────────────────────────────────────────────────────────────
app.use(cors({
    origin: true,          // reflect request origin
    credentials: true
}));

// ── BODY PARSERS ──────────────────────────────────────────────────────────────
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// ── STATIC FILES ──────────────────────────────────────────────────────────────
app.use(express.static(__dirname));

// ── SESSION ───────────────────────────────────────────────────────────────────
app.use(session({
    secret: process.env.SESSION_SECRET || 'osprey-ai-labs-secret-change-in-production',
    resave: false,
    saveUninitialized: false,
    cookie: {
        httpOnly: true,
        // sameSite 'none' required for cross-context cookies on Render (HTTPS)
        secure:   process.env.NODE_ENV === 'production',
        sameSite: process.env.NODE_ENV === 'production' ? 'none' : 'lax',
        maxAge:   24 * 60 * 60 * 1000 // 24 hours
    }
}));

// ─────────────────────────────────────────────────────────────────────────────
// DATA — USERS
// ─────────────────────────────────────────────────────────────────────────────
const users = [
    {
        id:            uuidv4(),
        username:      'Admin@B',
        password:      bcrypt.hashSync('Admin123', 10),
        role:          'admin',
        name:          'Admin User',
        email:         'admin@ospreyai.com',
        theme:         'dark',
        notifications: true,
        created:       new Date('2024-01-01')
    },
    {
        id:            uuidv4(),
        username:      'User@B',
        password:      bcrypt.hashSync('User123', 10),
        role:          'user',
        name:          'Standard User',
        email:         'user@ospreyai.com',
        theme:         'dark',
        notifications: true,
        created:       new Date('2024-01-15')
    }
];

// ─────────────────────────────────────────────────────────────────────────────
// DATA — ACTIVITY LOGS
// ─────────────────────────────────────────────────────────────────────────────
let activityLogs = [
    {
        id:        uuidv4(),
        timestamp: new Date(),
        user:      'System',
        action:    'Platform Start',
        details:   'Osprey AI Platform initialized',
        severity:  'info'
    }
];

// ─────────────────────────────────────────────────────────────────────────────
// DATA — AI AGENTS
// ─────────────────────────────────────────────────────────────────────────────
const aiAgents = [
    {
        id:          'rag-001',
        name:        'RAG Document Processor',
        type:        'document_processing',
        description: 'Advanced document processing with semantic search and knowledge retrieval',
        status:      'active',
        model:       'qwen2.5:7b',
        tools:       ['document_parse', 'vector_search', 'summarize', 'extract_entities'],
        metrics:     { requests: 15420, successRate: 99.8, avgResponseTime: 234 }
    },
    {
        id:          'nl2sql-002',
        name:        'NL2SQL Builder',
        type:        'database_query',
        description: 'Natural language to SQL conversion with query optimization',
        status:      'active',
        model:       'qwen2.5:7b',
        tools:       ['schema_analysis', 'query_generation', 'query_validation', 'results_formatting'],
        metrics:     { requests: 8934, successRate: 98.5, avgResponseTime: 189 }
    },
    {
        id:          'workflow-003',
        name:        'Workflow Orchestrator',
        type:        'automation',
        description: 'Multi-agent workflow coordination and task automation',
        status:      'active',
        model:       'qwen2.5:7b',
        tools:       ['task_planning', 'agent_coordination', 'error_handling', 'monitoring'],
        metrics:     { requests: 12567, successRate: 99.2, avgResponseTime: 456 }
    },
    {
        id:          'analytics-004',
        name:        'Analytics Engine',
        type:        'data_analysis',
        description: 'Deep insights with predictive analytics and trend analysis',
        status:      'training',
        model:       'qwen2.5:7b',
        tools:       ['data_analysis', 'visualization', 'prediction', 'reporting'],
        metrics:     { requests: 6783, successRate: 97.3, avgResponseTime: 678 }
    },
    {
        id:          'code-005',
        name:        'Code Assistant',
        type:        'development',
        description: 'AI-powered code generation, review, and optimization',
        status:      'active',
        model:       'qwen2.5:7b',
        tools:       ['code_generation', 'code_review', 'debugging', 'optimization'],
        metrics:     { requests: 9876, successRate: 98.9, avgResponseTime: 312 }
    }
];

// ─────────────────────────────────────────────────────────────────────────────
// DATA — WORKFLOWS
// ─────────────────────────────────────────────────────────────────────────────
const workflows = [
    {
        id:          'wf-001',
        name:        'Customer Onboarding',
        trigger:     'New User Signup',
        status:      'active',
        lastRun:     new Date(Date.now() - 10 * 60 * 1000),
        successRate: 99.8,
        steps:       5
    },
    {
        id:          'wf-002',
        name:        'Data Processing Pipeline',
        trigger:     'Schedule (Hourly)',
        status:      'active',
        lastRun:     new Date(Date.now() - 45 * 60 * 1000),
        successRate: 98.2,
        steps:       8
    },
    {
        id:          'wf-003',
        name:        'Report Generation',
        trigger:     'Manual Trigger',
        status:      'running',
        lastRun:     new Date(Date.now() - 2 * 60 * 60 * 1000),
        successRate: 100,
        steps:       6
    }
];

// ─────────────────────────────────────────────────────────────────────────────
// AUTH MIDDLEWARE
// ─────────────────────────────────────────────────────────────────────────────
const requireAuth = (req, res, next) => {
    if (req.session && req.session.userId) {
        return next();
    }
    res.status(401).json({ error: 'Unauthorized' });
};

const requireAdmin = (req, res, next) => {
    if (req.session && req.session.userId && req.session.role === 'admin') {
        return next();
    }
    res.status(403).json({ error: 'Forbidden — admin access required' });
};

// ─────────────────────────────────────────────────────────────────────────────
// AUTH ROUTES  (dashboard.js calls /api/auth/*)
// ─────────────────────────────────────────────────────────────────────────────

// POST /api/auth/login
app.post('/api/auth/login', async (req, res) => {
    const { username, password } = req.body;

    if (!username || !password) {
        return res.status(400).json({ error: 'Username and password required' });
    }

    const user = users.find(u => u.username === username);
    if (!user) {
        return res.status(401).json({ error: 'Invalid credentials' });
    }

    const valid = await bcrypt.compare(password, user.password);
    if (!valid) {
        return res.status(401).json({ error: 'Invalid credentials' });
    }

    req.session.userId   = user.id;
    req.session.username = user.username;
    req.session.role     = user.role;

    activityLogs.unshift({
        id:        uuidv4(),
        timestamp: new Date(),
        user:      user.username,
        action:    'User Login',
        details:   `${user.username} signed in`,
        severity:  'info'
    });

    res.json({
        success: true,
        user: {
            id:       user.id,
            username: user.username,
            role:     user.role,
            email:    user.email,
            name:     user.name,
            theme:    user.theme
        }
    });
});

// POST /api/auth/logout
app.post('/api/auth/logout', (req, res) => {
    const username = req.session.username || 'Unknown';
    req.session.destroy((err) => {
        if (err) return res.status(500).json({ error: 'Logout failed' });
        activityLogs.unshift({
            id:        uuidv4(),
            timestamp: new Date(),
            user:      username,
            action:    'User Logout',
            details:   `${username} signed out`,
            severity:  'info'
        });
        res.clearCookie('connect.sid');
        res.json({ success: true });
    });
});

// GET /api/auth/session  — what dashboard.js calls on every load
app.get('/api/auth/session', (req, res) => {
    if (!req.session || !req.session.userId) {
        return res.json({ authenticated: false });
    }
    const user = users.find(u => u.id === req.session.userId);
    if (!user) {
        return res.json({ authenticated: false });
    }
    res.json({
        authenticated: true,
        user: {
            id:       user.id,
            username: user.username,
            role:     user.role,
            email:    user.email,
            name:     user.name,
            theme:    user.theme
        }
    });
});

// ─────────────────────────────────────────────────────────────────────────────
// AGENT ROUTES
// ─────────────────────────────────────────────────────────────────────────────
app.get('/api/agents', requireAuth, (req, res) => {
    res.json({ success: true, agents: aiAgents });
});

app.get('/api/agents/:id', requireAuth, (req, res) => {
    const agent = aiAgents.find(a => a.id === req.params.id);
    if (!agent) return res.status(404).json({ success: false, message: 'Agent not found' });
    res.json({ success: true, agent });
});

// ─────────────────────────────────────────────────────────────────────────────
// ANALYTICS ROUTE  (dashboard.js calls /api/analytics/dashboard)
// ─────────────────────────────────────────────────────────────────────────────
app.get('/api/analytics/dashboard', requireAuth, (req, res) => {
    const totalRequests   = aiAgents.reduce((s, a) => s + a.metrics.requests, 0);
    const avgPerformance  = aiAgents.reduce((s, a) => s + a.metrics.successRate, 0) / aiAgents.length;

    res.json({
        totalAgents:    aiAgents.length,
        activeAgents:   aiAgents.filter(a => a.status === 'active').length,
        totalRequests,
        avgPerformance: parseFloat(avgPerformance.toFixed(1)),
        uptime:         99.9,
        recentActivity: activityLogs.slice(0, 10)
    });
});

// ─────────────────────────────────────────────────────────────────────────────
// LOGS ROUTE  (dashboard.js calls /api/logs)
// ─────────────────────────────────────────────────────────────────────────────
app.get('/api/logs', requireAuth, (req, res) => {
    const limit = Math.min(parseInt(req.query.limit) || 50, 200);
    res.json({ logs: activityLogs.slice(0, limit) });
});

// ─────────────────────────────────────────────────────────────────────────────
// USER PROFILE ROUTES  (dashboard.js calls /api/user/profile)
// ─────────────────────────────────────────────────────────────────────────────
app.get('/api/user/profile', requireAuth, (req, res) => {
    const user = users.find(u => u.id === req.session.userId);
    if (!user) return res.status(404).json({ error: 'User not found' });
    res.json({
        id:            user.id,
        username:      user.username,
        email:         user.email,
        name:          user.name,
        role:          user.role,
        theme:         user.theme,
        notifications: user.notifications,
        created:       user.created
    });
});

app.put('/api/user/profile', requireAuth, async (req, res) => {
    const idx = users.findIndex(u => u.id === req.session.userId);
    if (idx === -1) return res.status(404).json({ error: 'User not found' });

    const { name, email, theme, notifications, currentPassword, newPassword } = req.body;

    if (newPassword) {
        if (!currentPassword) return res.status(400).json({ error: 'Current password required' });
        const valid = await bcrypt.compare(currentPassword, users[idx].password);
        if (!valid) return res.status(401).json({ error: 'Current password is incorrect' });
        users[idx].password = await bcrypt.hash(newPassword, 10);
    }

    if (name  !== undefined) users[idx].name  = name;
    if (email !== undefined) users[idx].email = email;
    if (theme !== undefined) users[idx].theme = theme;
    if (notifications !== undefined) users[idx].notifications = notifications;

    activityLogs.unshift({
        id:        uuidv4(),
        timestamp: new Date(),
        user:      req.session.username,
        action:    'Profile Updated',
        details:   'User updated profile settings',
        severity:  'info'
    });

    res.json({ success: true });
});

// ─────────────────────────────────────────────────────────────────────────────
// WORKFLOWS ROUTE
// ─────────────────────────────────────────────────────────────────────────────
app.get('/api/workflows', requireAuth, (req, res) => {
    res.json({ success: true, workflows });
});

// ─────────────────────────────────────────────────────────────────────────────
// CHAT ROUTE
// ─────────────────────────────────────────────────────────────────────────────
app.post('/api/chat', requireAuth, (req, res) => {
    const { message = '', agentId } = req.body;

    const lookup = {
        'hello':      "Hello! I'm your Osprey AI assistant. How can I help you today?",
        'hi':         "Hi there! Ready to streamline your operations with AI?",
        'help':       "I can help you manage agents, monitor performance, analyse data, and optimise workflows. What would you like to explore?",
        'agents':     `You have ${aiAgents.filter(a => a.status === 'active').length} active AI agents. Your RAG Processor and NL2SQL Builder are performing exceptionally well at 99%+.`,
        'performance':`Platform performing excellently: 99.2% success rate, 247ms avg response, 847K+ requests processed.`,
        'status':     'All systems operational. CPU: 42%, Memory: 68%. Network performance is optimal.',
        'workflows':  `You have ${workflows.length} workflows — ${workflows.filter(w => w.status === 'active').length} active, ${workflows.filter(w => w.status === 'running').length} running.`
    };

    const lower = message.toLowerCase();
    let reply = "I understand. As your AI assistant I'm here to help optimise your Osprey AI operations. Could you be more specific?";
    for (const [key, val] of Object.entries(lookup)) {
        if (lower.includes(key)) { reply = val; break; }
    }
    if (agentId) {
        const agent = aiAgents.find(a => a.id === agentId);
        if (agent) reply = `[${agent.name}] ${reply}`;
    }

    res.json({
        success:   true,
        message:   reply,
        timestamp: new Date().toISOString(),
        agentId:   agentId || 'general'
    });
});

// ─────────────────────────────────────────────────────────────────────────────
// METRICS ROUTE
// ─────────────────────────────────────────────────────────────────────────────
app.get('/api/metrics/overview', requireAuth, (req, res) => {
    const totalRequests  = aiAgents.reduce((s, a) => s + a.metrics.requests, 0);
    const avgSuccessRate = aiAgents.reduce((s, a) => s + a.metrics.successRate, 0) / aiAgents.length;
    const avgResponseTime= aiAgents.reduce((s, a) => s + a.metrics.avgResponseTime, 0) / aiAgents.length;

    res.json({
        success: true,
        metrics: {
            activeAgents:    aiAgents.filter(a => a.status === 'active').length,
            totalRequests,
            successRate:     parseFloat(avgSuccessRate.toFixed(1)),
            avgResponseTime: Math.round(avgResponseTime),
            systemHealth:    { cpu: 42, memory: 68, storage: 34, network: 21 }
        }
    });
});

// ─────────────────────────────────────────────────────────────────────────────
// ACTIVITY FEED ROUTE
// ─────────────────────────────────────────────────────────────────────────────
app.get('/api/activity/recent', requireAuth, (req, res) => {
    res.json({
        success:    true,
        activities: [
            { agent: 'RAG Document Processor', action: 'Document processed',  status: 'success',    timestamp: new Date(Date.now() - 2 * 60 * 1000) },
            { agent: 'NL2SQL Builder',          action: 'Query executed',      status: 'success',    timestamp: new Date(Date.now() - 15 * 60 * 1000) },
            { agent: 'Workflow Orchestrator',   action: 'Workflow triggered',  status: 'success',    timestamp: new Date(Date.now() - 60 * 60 * 1000) },
            { agent: 'Analytics Engine',        action: 'Report generated',    status: 'processing', timestamp: new Date(Date.now() - 5 * 60 * 60 * 1000) }
        ]
    });
});

// ─────────────────────────────────────────────────────────────────────────────
// PAGE ROUTES
// ─────────────────────────────────────────────────────────────────────────────
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'index.html'));
});

app.get('/signin', (req, res) => {
    // If already logged in, skip signin page
    if (req.session && req.session.userId) {
        return res.redirect('/dashboard');
    }
    res.sendFile(path.join(__dirname, 'signin.html'));
});

app.get('/dashboard', (req, res) => {
    // Server-side auth guard — no valid session = back to signin
    if (!req.session || !req.session.userId) {
        return res.redirect('/signin');
    }
    res.sendFile(path.join(__dirname, 'dashboard.html'));
});

// ─────────────────────────────────────────────────────────────────────────────
// ERROR HANDLER
// ─────────────────────────────────────────────────────────────────────────────
app.use((err, req, res, next) => {
    console.error(err.stack);
    res.status(500).json({ success: false, message: 'Internal server error' });
});

// ═══════════════════════════════════════════════════════════════════
// AI ENDPOINTS
// ═══════════════════════════════════════════════════════════════════
app.get('/api/ai/health', async (req, res) => {
    try {
        const isHealthy = await ollamaHandler.checkHealth();
        const models = await ollamaHandler.getModels();
        res.json({
            status: isHealthy ? 'healthy' : 'unavailable',
            models: models.map(m => m.name),
            timestamp: new Date().toISOString()
        });
    } catch (error) {
        res.status(500).json({ status: 'error', message: error.message });
    }
});

app.post('/api/ai/generate', async (req, res) => {
    try {
        const { message, agent, history } = req.body;
        if (!message || typeof message !== 'string') {
            return res.status(400).json({ error: 'Message is required' });
        }
        const result = await aiRouter.generateResponse(
            agent || 'content-writer',
            message,
            history || []
        );
        activityLogs.push({
            id: uuidv4(),
            timestamp: new Date(),
            user: req.session?.user?.username || 'Anonymous',
            action: 'AI Generation',
            details: `Agent: ${result.agent?.name || agent}`,
            severity: 'info'
        });
        res.json(result);
    } catch (error) {
        console.error('AI generation error:', error);
        res.status(500).json({ error: 'AI generation failed', message: error.message });
    }
});

app.get('/api/ai/agents', (req, res) => {
    try {
        const agents = aiRouter.getAgents();
        res.json({ agents });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

app.get('/api/ai/agents/:agentId/greeting', (req, res) => {
    try {
        const { agentId } = req.params;
        if (!aiRouter.isValidAgent(agentId)) {
            return res.status(404).json({ error: 'Agent not found' });
        }
        const greeting = aiRouter.getGreeting(agentId);
        res.json({ greeting, agentId });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});


// ═════════════════════════════════════════════════════════════════════════════
// PYTHON AI ENDPOINTS - Proxy to Advanced Python Agents
// ═════════════════════════════════════════════════════════════════════════════
// These endpoints proxy requests to the Python FastAPI server (agent-server.py)
// which provides 6 specialized AI agents with advanced features like SEO scoring,
// quality metrics, and readability analysis.

// Health check for Python AI server
app.get('/api/python-ai/health', async (req, res) => {
    try {
        const response = await fetch(`${PYTHON_SERVER}/health`);
        const data = await response.json();
        res.json(data);
    } catch (error) {
        console.error('Python AI server unavailable:', error.message);
        res.json({
            status: 'unavailable',
            message: 'Python AI server not reachable',
            agents: []
        });
    }
});

// Generate AI response using Python agents (with advanced features)
app.post('/api/python-ai/generate', async (req, res) => {
    try {
        const { message, agent, history, ...options } = req.body;
        
        if (!message || typeof message !== 'string') {
            return res.status(400).json({ error: 'Message required' });
        }

        // Forward to Python server
        const response = await fetch(`${PYTHON_SERVER}/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: message,
                agent: agent || 'content-writer',
                model: options.model || 'llama2',
                target_word_count: options.target_word_count || 800,
                temperature: options.temperature || 0.7,
                tone: options.tone || 'professional',
                language: options.language || 'en',
                seo_optimize: options.seo_optimize !== false
            })
        });

        if (!response.ok) {
            throw new Error(`Python server returned ${response.status}`);
        }

        const data = await response.json();

        // Log activity
        activityLogs.push({
            id: uuidv4(),
            timestamp: new Date(),
            user: req.session?.user?.username || 'Anonymous',
            action: 'Python AI Generation',
            details: `Agent: ${data.agent}, Quality: ${data.quality_score || 'N/A'}`,
            severity: 'info'
        });

        res.json(data);

    } catch (error) {
        console.error('Python AI generation error:', error);
        res.status(500).json({
            error: 'Python AI generation failed',
            message: error.message,
            fallback: 'Python AI server unavailable. Please try again.'
        });
    }
});

// Get available Python agents
app.get('/api/python-ai/agents', async (req, res) => {
    try {
        const response = await fetch(`${PYTHON_SERVER}/agents`);
        const data = await response.json();
        res.json(data);
    } catch (error) {
        res.json({ agents: [] });
    }
});



// ─────────────────────────────────────────────────────────────────────────────
// START
// ─────────────────────────────────────────────────────────────────────────────
app.listen(PORT, '0.0.0.0', () => {
    console.log(`
    
╔══════════════════════════════════════════════════════╗
║                                                      ║
║   🦅  Osprey AI Labs Platform — Running             ║
║                                                      ║
╠══════════════════════════════════════════════════════╣
║                                                      ║
║   Port  : ${String(PORT).padEnd(42)}║
║   Env   : ${String(process.env.NODE_ENV || 'development').padEnd(42)}║
║                                                      ║
║   Credentials:                                       ║
║   Admin@B  / Admin123  (admin access)                ║
║   User@B   / User123   (user access)                 ║
║                                                      ║
╚══════════════════════════════════════════════════════╝
    `);  // ← Close the template string HERE
    
    // Now setTimeout is properly outside console.log
    setTimeout(async () => {
        console.log('\n🤖 Checking Ollama AI...');
        const isHealthy = await ollamaHandler.checkHealth();
        if (isHealthy) {
            console.log('✅ Ollama is ready!');
        } else {
            console.log('⚠️  Ollama not available - using fallback');
        }
        
        // Check Python AI server
        console.log('\n🐍 Checking Python AI server...');
        try {
            const response = await fetch(`${PYTHON_SERVER}/health`);
            const data = await response.json();
            console.log(`✅ Python AI ready! ${data.count || 0}/6 agents loaded`);
        } catch (error) {
            console.log('⚠️  Python AI server not available');
            console.log('   Start it: cd agents && python3 agent-server.py');
        }
    }, 2000);
});
