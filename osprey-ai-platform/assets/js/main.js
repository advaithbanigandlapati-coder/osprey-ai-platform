// ===================================================================
// OSPREY AI LABS - MAIN JAVASCRIPT
// Professional Interactions and Animations
// Version 2.0.0
// ===================================================================

'use strict';

// ===== GLOBAL STATE =====
const OspreyApp = {
    currentUser: null,
    isDashboardOpen: false,
    activeTab: 'overview',
    scrollPosition: 0
};

// ===== INITIALIZATION =====
document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
    setupEventListeners();
    setupScrollAnimations();
    checkAuthStatus();
});

function initializeApp() {
    console.log('%c🦅 Osprey AI Labs Platform Initialized', 'color: #607ea2; font-size: 16px; font-weight: bold;');
    
    // Add smooth page transition
    document.body.style.opacity = '0';
    setTimeout(() => {
        document.body.style.transition = 'opacity 0.6s ease';
        document.body.style.opacity = '1';
    }, 100);
    
    // Initialize kinetic cursor
    initKineticCursor();
    
    // Check for saved theme preference
    const savedTheme = localStorage.getItem('osprey_theme');
    if (savedTheme) {
        document.body.setAttribute('data-theme', savedTheme);
    }
}

// ===== NAVIGATION FUNCTIONALITY =====
function setupEventListeners() {
    // Navigation scroll effect
    window.addEventListener('scroll', handleNavScroll);
    
    // Smooth scroll for navigation links
    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', handleNavLinkClick);
    });
    
    // Mobile navigation toggle
    const mobileToggle = document.querySelector('.nav-mobile-toggle');
    if (mobileToggle) {
        mobileToggle.addEventListener('click', toggleMobileNav);
    }
    
    // Close mobile nav when clicking outside
    document.addEventListener('click', handleOutsideClick);
}

function handleNavScroll() {
    const nav = document.getElementById('nav');
    if (!nav) return;
    
    const scrollPosition = window.scrollY;
    
    if (scrollPosition > 50) {
        nav.classList.add('scrolled');
    } else {
        nav.classList.remove('scrolled');
    }
    
    // Update active section
    updateActiveNavLink();
}

function handleNavLinkClick(e) {
    e.preventDefault();
    const targetId = e.target.getAttribute('href');
    
    if (!targetId || !targetId.startsWith('#')) return;
    
    const targetSection = document.querySelector(targetId);
    
    if (targetSection) {
        // Smooth scroll to section
        targetSection.scrollIntoView({ 
            behavior: 'smooth',
            block: 'start'
        });
        
        // Update active link
        document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
        e.target.classList.add('active');
        
        // Close mobile nav if open
        closeMobileNav();
    }
}

function updateActiveNavLink() {
    const sections = document.querySelectorAll('.section');
    const navLinks = document.querySelectorAll('.nav-link');
    
    let current = '';
    
    sections.forEach(section => {
        const sectionTop = section.offsetTop;
        const sectionHeight = section.clientHeight;
        
        if (window.scrollY >= (sectionTop - 200)) {
            current = section.getAttribute('id');
        }
    });
    
    navLinks.forEach(link => {
        link.classList.remove('active');
        if (link.getAttribute('href') === `#${current}`) {
            link.classList.add('active');
        }
    });
}

function toggleMobileNav() {
    const navLinks = document.querySelector('.nav-links');
    const toggle = document.querySelector('.nav-mobile-toggle');
    
    if (navLinks) {
        const isOpen = navLinks.style.display === 'flex';
        navLinks.style.display = isOpen ? 'none' : 'flex';
        
        if (toggle) {
            toggle.classList.toggle('active');
        }
    }
}

function closeMobileNav() {
    const navLinks = document.querySelector('.nav-links');
    const toggle = document.querySelector('.nav-mobile-toggle');
    
    if (navLinks && window.innerWidth <= 968) {
        navLinks.style.display = 'none';
        if (toggle) {
            toggle.classList.remove('active');
        }
    }
}

function handleOutsideClick(e) {
    const navLinks = document.querySelector('.nav-links');
    const toggle = document.querySelector('.nav-mobile-toggle');
    
    if (navLinks && toggle && 
        !navLinks.contains(e.target) && 
        !toggle.contains(e.target) &&
        navLinks.style.display === 'flex') {
        closeMobileNav();
    }
}

// ===== SCROLL ANIMATIONS =====
function setupScrollAnimations() {
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -100px 0px'
    };
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('active');
                
                // Add stagger effect for children
                const children = entry.target.querySelectorAll('.stat-item, .about-card, .service-card, .solution-item');
                children.forEach((child, index) => {
                    setTimeout(() => {
                        child.style.animation = `fadeInUp 0.6s ease-out ${index * 0.1}s both`;
                    }, 0);
                });
            }
        });
    }, observerOptions);
    
    // Observe all reveal elements
    const revealElements = document.querySelectorAll(
        '.about-card, .service-card, .solution-item, .contact-grid, .stat-item'
    );
    
    revealElements.forEach(el => {
        el.classList.add('scroll-reveal');
        observer.observe(el);
    });
}

// ===== KINETIC CURSOR EFFECT =====
function initKineticCursor() {
    let mouseX = 0;
    let mouseY = 0;
    let ballX = 0;
    let ballY = 0;
    const speed = 0.08;
    
    document.addEventListener('mousemove', (e) => {
        mouseX = e.clientX;
        mouseY = e.clientY;
    });
    
    function animateCursor() {
        const distX = mouseX - ballX;
        const distY = mouseY - ballY;
        
        ballX += distX * speed;
        ballY += distY * speed;
        
        // Update gradient orbs position slightly based on mouse
        const orbs = document.querySelectorAll('.gradient-orb');
        orbs.forEach((orb, index) => {
            const factor = (index + 1) * 0.015;
            const currentTransform = orb.style.transform || '';
            
            // Preserve existing transforms (like from animation)
            orb.style.transform = `${currentTransform} translate(${ballX * factor}px, ${ballY * factor}px)`;
        });
        
        requestAnimationFrame(animateCursor);
    }
    
    animateCursor();
}

// ===== LOGIN MODAL =====
function showLogin() {
    const modal = document.getElementById('login-modal');
    if (modal) {
        modal.classList.add('active');
        document.body.style.overflow = 'hidden';
        
        // Focus on username field
        setTimeout(() => {
            const usernameInput = document.getElementById('username');
            if (usernameInput) {
                usernameInput.focus();
            }
        }, 100);
    }
}

function closeLogin() {
    const modal = document.getElementById('login-modal');
    if (modal) {
        modal.classList.remove('active');
        document.body.style.overflow = '';
    }
}

// Close modal on ESC key
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        closeLogin();
    }
});

async function handleLogin(event) {
    event.preventDefault();
    
    const username = document.getElementById('username').value;
    const password = document.getElementById('password').value;
    const submitBtn = event.target.querySelector('button[type="submit"]');
    
    if (!username || !password) {
        showNotification('Please enter username and password', 'error');
        return;
    }
    
    // Disable button and show loading
    submitBtn.disabled = true;
    const originalText = submitBtn.textContent;
    submitBtn.innerHTML = '<span>Signing in...</span>';
    
    try {
        const response = await fetch('/api/login', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ username, password })
        });
        
        const data = await response.json();
        
        if (data.success) {
            // Store user info
            localStorage.setItem('osprey_user', JSON.stringify(data.user));
            OspreyApp.currentUser = data.user;
            
            // Show success notification
            showNotification('Welcome back! Loading dashboard...', 'success');
            
            // Close modal with animation
            setTimeout(() => {
                closeLogin();
                
                // Show dashboard
                setTimeout(() => {
                    showDashboard();
                }, 300);
            }, 800);
        } else {
            showNotification(data.message || 'Login failed. Please try again.', 'error');
            submitBtn.disabled = false;
            submitBtn.textContent = originalText;
        }
    } catch (error) {
        console.error('Login error:', error);
        showNotification('Unable to connect to server. Please check your connection.', 'error');
        submitBtn.disabled = false;
        submitBtn.textContent = originalText;
    }
}

// ===== CONTACT FORM =====
function handleContactForm(event) {
    event.preventDefault();
    
    const formData = new FormData(event.target);
    const data = Object.fromEntries(formData);
    
    // Show loading state
    const submitBtn = event.target.querySelector('button[type="submit"]');
    submitBtn.disabled = true;
    submitBtn.textContent = 'Sending...';
    
    // Simulate API call
    setTimeout(() => {
        showNotification('Thank you for your message! We\'ll get back to you soon.', 'success');
        event.target.reset();
        submitBtn.disabled = false;
        submitBtn.textContent = 'Send Message';
    }, 1500);
}

// ===== NOTIFICATION SYSTEM =====
function showNotification(message, type = 'info') {
    // Remove existing notifications
    const existingNotif = document.querySelector('.notification');
    if (existingNotif) {
        existingNotif.remove();
    }
    
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.innerHTML = `
        <div class="notification-content">
            <div class="notification-icon">
                ${getNotificationIcon(type)}
            </div>
            <div class="notification-message">${message}</div>
            <button class="notification-close" onclick="this.parentElement.parentElement.remove()">×</button>
        </div>
    `;
    
    // Add styles if not already present
    if (!document.getElementById('notification-styles')) {
        const style = document.createElement('style');
        style.id = 'notification-styles';
        style.textContent = `
            .notification {
                position: fixed;
                top: 100px;
                right: 30px;
                z-index: 10000;
                min-width: 320px;
                max-width: 450px;
                background: rgba(26, 30, 46, 0.98);
                backdrop-filter: blur(20px);
                border-radius: 16px;
                padding: 20px;
                box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
                border: 1px solid rgba(255, 255, 255, 0.1);
                animation: slideInRight 0.4s cubic-bezier(0.16, 1, 0.3, 1);
            }
            
            @keyframes slideInRight {
                from {
                    opacity: 0;
                    transform: translateX(100px);
                }
                to {
                    opacity: 1;
                    transform: translateX(0);
                }
            }
            
            .notification-content {
                display: flex;
                align-items: start;
                gap: 15px;
            }
            
            .notification-icon {
                flex-shrink: 0;
                width: 24px;
                height: 24px;
            }
            
            .notification-message {
                flex: 1;
                color: white;
                font-size: 0.95rem;
                line-height: 1.5;
            }
            
            .notification-close {
                flex-shrink: 0;
                width: 28px;
                height: 28px;
                border-radius: 50%;
                border: none;
                background: rgba(255, 255, 255, 0.1);
                color: white;
                cursor: pointer;
                font-size: 1.2rem;
                transition: all 0.2s;
            }
            
            .notification-close:hover {
                background: rgba(255, 255, 255, 0.2);
                transform: rotate(90deg);
            }
            
            .notification-success {
                border-left: 3px solid #10b981;
            }
            
            .notification-error {
                border-left: 3px solid #ef4444;
            }
            
            .notification-warning {
                border-left: 3px solid #f59e0b;
            }
            
            .notification-info {
                border-left: 3px solid #3b82f6;
            }
            
            @media (max-width: 640px) {
                .notification {
                    right: 15px;
                    left: 15px;
                    min-width: auto;
                }
            }
        `;
        document.head.appendChild(style);
    }
    
    document.body.appendChild(notification);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        notification.style.animation = 'slideInRight 0.3s reverse';
        setTimeout(() => {
            notification.remove();
        }, 300);
    }, 5000);
}

function getNotificationIcon(type) {
    const icons = {
        success: '<svg viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg>',
        error: '<svg viewBox="0 0 24 24" fill="none" stroke="#ef4444" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>',
        warning: '<svg viewBox="0 0 24 24" fill="none" stroke="#f59e0b" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
        info: '<svg viewBox="0 0 24 24" fill="none" stroke="#3b82f6" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>'
    };
    return icons[type] || icons.info;
}

// ===== AUTH CHECK =====
async function checkAuthStatus() {
    try {
        const response = await fetch('/api/session');
        const data = await response.json();
        
        if (data.success && data.user) {
            OspreyApp.currentUser = data.user;
            localStorage.setItem('osprey_user', JSON.stringify(data.user));
        }
    } catch (error) {
        console.log('No active session');
    }
}

// ===== UTILITY FUNCTIONS =====
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

function throttle(func, limit) {
    let inThrottle;
    return function(...args) {
        if (!inThrottle) {
            func.apply(this, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}

// Debounced scroll handler
const debouncedScrollHandler = debounce(handleNavScroll, 10);
window.addEventListener('scroll', debouncedScrollHandler);

// ===== PERFORMANCE MONITORING =====
if ('performance' in window) {
    window.addEventListener('load', () => {
        setTimeout(() => {
            const perfData = performance.getEntriesByType('navigation')[0];
            console.log('%c⚡ Performance Metrics', 'color: #10b981; font-weight: bold;');
            console.log('Load Time:', Math.round(perfData.loadEventEnd - perfData.fetchStart), 'ms');
            console.log('DOM Interactive:', Math.round(perfData.domInteractive - perfData.fetchStart), 'ms');
        }, 0);
    });
}

// ===== ACCESSIBILITY =====
document.addEventListener('keydown', (e) => {
    // Tab key navigation enhancement
    if (e.key === 'Tab') {
        document.body.classList.add('keyboard-navigation');
    }
});

document.addEventListener('mousedown', () => {
    document.body.classList.remove('keyboard-navigation');
});

// ===== SERVICE WORKER (PWA Support - Optional) =====
if ('serviceWorker' in navigator && window.location.protocol === 'https:') {
    window.addEventListener('load', () => {
        // Uncomment to enable PWA
        // navigator.serviceWorker.register('/sw.js')
        //     .then(reg => console.log('Service Worker registered'))
        //     .catch(err => console.log('Service Worker registration failed'));
    });
}

// ===== EXPORT FUNCTIONS FOR GLOBAL USE =====
window.showLogin = showLogin;
window.closeLogin = closeLogin;
window.handleLogin = handleLogin;
window.handleContactForm = handleContactForm;
window.showNotification = showNotification;
window.OspreyApp = OspreyApp;

// ===== CONSOLE BRANDING =====
console.log(`
%c
╔═══════════════════════════════════════════════════════╗
║                                                       ║
║   🦅 OSPREY AI LABS                                   ║
║   Enterprise Agent Production Infrastructure          ║
║                                                       ║
║   Version 2.0.0                                       ║
║   © ${new Date().getFullYear()} Osprey AI Labs. All rights reserved.     ║
║                                                       ║
╚═══════════════════════════════════════════════════════╝
`, 'color: #607ea2; font-family: monospace; font-size: 12px;');

console.log('%cInterested in joining our team? Email: careers@ospreyai.com', 'color: #8197ac; font-size: 11px;');
