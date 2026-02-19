// ═══════════════════════════════════════════════════════════════════
// OSPREY AI LABS - UI SHOWCASE SERVER
// Pure aesthetics, mock backend, ready for demo
// ═══════════════════════════════════════════════════════════════════

const express = require('express');
const session = require('express-session');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 10000;

app.set('trust proxy', 1);

app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use(express.static(__dirname));

app.use(session({
    secret: 'osprey-ui-showcase',
    resave: false,
    saveUninitialized: false,
    cookie: {
        httpOnly: true,
        secure: process.env.NODE_ENV === 'production',
        maxAge: 24 * 60 * 60 * 1000
    }
}));

// Mock data
const mockUser = {
    id: '1',
    username: 'demo@osprey.ai',
    name: 'Demo User',
    role: 'admin',
    avatar: 'DU',
    theme: 'light'
};

const mockResponses = {
    'content-writer': 'Here's a beautifully crafted piece of content...',
    'code-assistant': '```javascript\n// Here's your code solution\nfunction example() {\n  return "perfect";\n}\n```',
    'data-analyst': 'Based on the data analysis, here are the key insights...',
    'marketing': 'Here's a compelling marketing strategy that will drive results...',
    'seo': 'SEO-optimized content with perfect keyword density and structure...'
};

// Routes
app.get('/', (req, res) => res.sendFile(path.join(__dirname, 'dashboard.html')));
app.get('/dashboard', (req, res) => res.sendFile(path.join(__dirname, 'dashboard.html')));
app.get('/settings', (req, res) => res.sendFile(path.join(__dirname, 'settings.html')));

// Mock API
app.get('/api/user', (req, res) => res.json({ success: true, user: mockUser }));

app.post('/api/ai/generate', (req, res) => {
    const { message, agent = 'content-writer' } = req.body;
    setTimeout(() => {
        res.json({
            success: true,
            content: mockResponses[agent] || `Mock response for: ${message}`,
            agent,
            model: 'mock-v1',
            tokens: Math.floor(Math.random() * 400) + 150,
            quality_score: Math.floor(Math.random() * 20) + 80
        });
    }, 1200);
});

app.get('/api/stats', (req, res) => {
    res.json({
        success: true,
        stats: {
            requests: 15234,
            agents: 6,
            uptime: '99.9%',
            responseTime: '0.8s'
        }
    });
});

app.listen(PORT, () => {
    console.log(`\n╔═══════════════════════════════════════╗`);
    console.log(`║  🦅 OSPREY AI LABS - UI SHOWCASE     ║`);
    console.log(`╠═══════════════════════════════════════╣`);
    console.log(`║  → http://localhost:${PORT.toString().padEnd(18)}║`);
    console.log(`║  Status: READY ✨                    ║`);
    console.log(`╚═══════════════════════════════════════╝\n`);
});
