# ⚡ Render Deployment - 2 Minute Guide

## 🚀 Deploy to Render NOW

### Option 1: One-Click Deploy (Fastest)

1. **Push to GitHub**
```bash
git init
git add .
git commit -m "Osprey AI Platform"
git remote add origin YOUR_GITHUB_URL
git push -u origin main
```

2. **Deploy on Render**
- Go to: https://dashboard.render.com/
- Click: **New + → Web Service**
- Connect: Your GitHub repo
- Click: **Create Web Service**

3. **Done!** 🎉
Your app will be live at: `https://your-app.onrender.com`

---

## 🔑 Important Files (Already Configured)

✅ `server.js` - In root directory
✅ `package.json` - In root directory  
✅ `render.yaml` - Auto-deploy config
✅ All static files in place

---

## 🎯 Login Credentials

After deployment, use these to login:

- **Admin**: `admin` / `admin123`
- **Demo**: `demo` / `demo123`
- **Investor**: `investor` / `investor123`

---

## ⚙️ Optional: Environment Variables

In Render Dashboard → Environment:

```
SESSION_SECRET=your-random-secret-here
NODE_ENV=production
```

Generate random secret:
```bash
node -e "console.log(require('crypto').randomBytes(32).toString('hex'))"
```

---

## ✅ That's It!

Your Osprey AI Labs platform is now:
- 🌐 Live on the internet
- 🔒 Secure with authentication
- 📱 Mobile responsive
- 🎨 Professional design
- 🤖 5 working AI agents
- 📊 30+ dashboard pages

**Ready for your investor demo!** 🦅
