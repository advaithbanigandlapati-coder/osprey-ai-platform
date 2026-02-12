# 🦅 Osprey AI Labs - Enterprise Platform

Professional, investor-ready AI agent platform with comprehensive dashboard and 30+ pages.

## ✨ Features

### Frontend
- **Professional Design**: Playfair Display font, glassmorphism, neumorphism
- **2161 lines of CSS**: Comprehensive styling with animations
- **30+ Dashboard Pages**: Overview, AI Chat, Agents, Builder, Orchestrator, Workflows, Analytics, and more
- **Responsive Design**: Mobile-first approach
- **Smooth Animations**: Professional transitions and effects

### Backend
- **Node.js/Express Server**: Professional authentication system
- **5 Working AI Agents**: RAG, NL2SQL, Workflow, Analytics, Code Assistant
- **User Roles**: Admin and User differentiation
- **Session Management**: Secure authentication
- **RESTful API**: Clean endpoint structure

### Dashboard Pages
1. Overview
2. AI Chat (with working backend)
3. AI Agents
4. Agent Builder
5. Orchestrator
6. Workflows
7. Analytics
8. Data Sources
9. Integrations
10. API Keys
11. Monitoring
12. Logs
13. Billing (Admin)
14. Team (Admin)
15. Settings
... and more!

## 🚀 Quick Start

### Prerequisites
- Node.js 16+ 
- npm or yarn

### Installation

1. **Install Dependencies**
```bash
cd backend
npm install
```

2. **Start Server**
```bash
npm start
```

3. **Open Browser**
```
http://localhost:3000
```

### Demo Credentials

**Admin Access:**
- Username: `admin`
- Password: `admin123`

**User Access:**
- Username: `demo`
- Password: `demo123`

**Investor Demo:**
- Username: `investor`
- Password: `investor123`

## 📁 Project Structure

```
osprey-ai-platform/
├── backend/
│   ├── server.js          # Express server (411 lines)
│   └── package.json       # Dependencies
├── assets/
│   ├── css/
│   │   └── styles.css     # Main styles (2161 lines)
│   └── js/
│       ├── main.js        # Core JavaScript
│       └── dashboard.js   # Dashboard logic (857 lines)
├── index.html             # Landing page
└── README.md
```

## 🎨 Design System

### Colors
- Primary: `#607ea2`
- Primary Dark: `#314b6e`
- Primary Light: `#8197ac`
- Accent: `#87adc6`

### Typography
- Display: Playfair Display
- Body: Inter

### Components
- Glassmorphism cards
- Neumorphic buttons
- Gradient backgrounds
- Smooth animations

## 🔧 Configuration

### Environment Variables
Create a `.env` file:
```env
PORT=3000
SESSION_SECRET=your-secret-key-here
NODE_ENV=production
```

### Production Deployment

1. **Build for Production**
```bash
npm run build
```

2. **Set Environment to Production**
```bash
export NODE_ENV=production
```

3. **Start with PM2** (recommended)
```bash
npm install -g pm2
pm2 start backend/server.js --name osprey-ai
```

## 📊 Features Breakdown

### AI Agents
- **RAG Document Processor**: 99.8% success rate
- **NL2SQL Builder**: Natural language to SQL
- **Workflow Automator**: Multi-agent coordination
- **Analytics Engine**: Predictive insights
- **Code Assistant**: AI-powered development

### Dashboard Features
- Real-time metrics
- Live chat with AI
- Agent management
- Visual workflow builder
- Performance analytics
- Team collaboration

## 🔐 Security

- Session-based authentication
- Bcrypt password hashing
- HTTP-only cookies
- CSRF protection ready
- Role-based access control

## 🌐 Browser Support

- Chrome (latest)
- Firefox (latest)
- Safari (latest)
- Edge (latest)

## 📝 License

Proprietary - © 2026 Osprey AI Labs

## 👥 Team

Built for investor demo presentation.

## 📧 Contact

- Email: contact@ospreyai.com
- Website: (Coming soon)

---

**Note**: This is a demo platform. For production use, implement:
- Real database (PostgreSQL/MongoDB)
- Production-grade authentication (OAuth, JWT)
- Rate limiting
- Input validation
- Error logging
- Monitoring tools
- Backup systems
