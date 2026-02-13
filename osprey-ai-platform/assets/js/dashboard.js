// ============================================================================
// OSPREY AI DASHBOARD - JAVASCRIPT
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
    // Initialize icons
    lucide.createIcons();

    // Real-time stats updates
    function updateStats() {
        const stats = {
            agents: Math.floor(Math.random() * 5) + 10,
            apiCalls: (Math.random() * 100 + 800).toFixed(0) + 'K',
            successRate: (99.5 + Math.random() * 0.5).toFixed(1) + '%',
            responseTime: (80 + Math.random() * 20).toFixed(0) + 'ms'
        };

        // Animate stat updates
        document.querySelectorAll('.stat-card-value').forEach((el, index) => {
            const values = Object.values(stats);
            if (el.textContent !== values[index]) {
                el.style.transform = 'scale(1.1)';
                setTimeout(() => {
                    el.textContent = values[index];
                    el.style.transform = 'scale(1)';
                }, 200);
            }
        });
    }

    // Update stats every 5 seconds
    setInterval(updateStats, 5000);

    // Agent status indicators
    function updateAgentStatus() {
        document.querySelectorAll('.agent-status.active .agent-status-dot').forEach(dot => {
            dot.style.animation = 'none';
            setTimeout(() => {
                dot.style.animation = 'pulse 2s ease-in-out infinite';
            }, 10);
        });
    }

    setInterval(updateAgentStatus, 3000);

    // Activity feed - add new items
    const activityFeed = document.querySelector('.activity-feed');
    const activities = [
        {
            icon: 'check',
            bg: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
            title: 'Task completed',
            description: 'Data pipeline processed 50K records',
            time: 'Just now'
        },
        {
            icon: 'alert-circle',
            bg: 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
            title: 'Warning detected',
            description: 'High memory usage on agent-003',
            time: '1 minute ago'
        },
        {
            icon: 'zap',
            bg: 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)',
            title: 'Agent deployed',
            description: 'New content agent is now live',
            time: '5 minutes ago'
        }
    ];

    let activityIndex = 0;
    function addActivity() {
        if (!activityFeed) return;
        
        const activity = activities[activityIndex % activities.length];
        activityIndex++;

        const activityItem = document.createElement('div');
        activityItem.className = 'activity-item';
        activityItem.style.opacity = '0';
        activityItem.style.transform = 'translateY(-10px)';
        
        activityItem.innerHTML = `
            <div style="width: 40px; height: 40px; border-radius: 50%; background: ${activity.bg}; display: flex; align-items: center; justify-content: center; flex-shrink: 0;">
                <i data-lucide="${activity.icon}" style="width: 20px; height: 20px; color: white;"></i>
            </div>
            <div>
                <div style="font-weight: 600; margin-bottom: var(--space-1);">${activity.title}</div>
                <div style="font-size: var(--font-size-sm); color: var(--color-text-tertiary);">${activity.description}</div>
                <div style="font-size: var(--font-size-xs); color: var(--color-text-tertiary); margin-top: 4px;">${activity.time}</div>
            </div>
        `;

        activityFeed.insertBefore(activityItem, activityFeed.firstChild);
        lucide.createIcons();

        // Animate in
        setTimeout(() => {
            activityItem.style.transition = 'all 0.3s ease-out';
            activityItem.style.opacity = '1';
            activityItem.style.transform = 'translateY(0)';
        }, 10);

        // Remove old activities
        const items = activityFeed.querySelectorAll('.activity-item');
        if (items.length > 10) {
            items[items.length - 1].remove();
        }
    }

    // Add new activity every 10 seconds
    setInterval(addActivity, 10000);

    // Chart data animation
    function animateChart() {
        const bars = document.querySelectorAll('[style*="border-radius: var(--radius-md)"]');
        bars.forEach((bar, index) => {
            setTimeout(() => {
                const currentHeight = bar.style.height;
                bar.style.height = '0%';
                bar.style.transition = 'height 0.5s ease-out';
                setTimeout(() => {
                    bar.style.height = currentHeight;
                }, 50);
            }, index * 100);
        });
    }

    // Animate chart on load
    setTimeout(animateChart, 500);

    // Copy API key functionality
    document.querySelectorAll('[data-lucide="copy"]').forEach(btn => {
        btn.closest('button')?.addEventListener('click', () => {
            const row = btn.closest('tr');
            const keyText = row?.querySelector('td:nth-child(2)')?.textContent;
            
            // Simulate copy (in real app, would copy actual key)
            window.OspreyAI.showToast('success', 'Copied', 'API key copied to clipboard');
        });
    });

    // Delete API key functionality
    document.querySelectorAll('[data-lucide="trash-2"]').forEach(btn => {
        btn.closest('button')?.addEventListener('click', () => {
            if (confirm('Are you sure you want to delete this API key? This action cannot be undone.')) {
                const row = btn.closest('tr');
                row.style.opacity = '0';
                row.style.transform = 'translateX(-20px)';
                setTimeout(() => row.remove(), 300);
                window.OspreyAI.showToast('success', 'Deleted', 'API key has been deleted');
            }
        });
    });

    // Simulate real-time log updates
    const logsContainer = document.querySelector('[style*="font-family: var(--font-mono)"]');
    if (logsContainer) {
        const logTypes = [
            { level: 'INFO', color: 'var(--color-success)', border: 'var(--color-success)' },
            { level: 'WARN', color: 'var(--color-warning)', border: 'var(--color-warning)' },
            { level: 'ERROR', color: 'var(--color-error)', border: 'var(--color-error)' }
        ];

        const agents = ['research-agent-001', 'code-agent-002', 'data-agent-003', 'content-agent-004', 'ops-agent-005'];
        const messages = [
            'Query processed successfully',
            'Generated code output',
            'Database connection established',
            'Workflow completed',
            'High memory usage detected',
            'Rate limit exceeded',
            'Health check passed',
            'Task queued for processing'
        ];

        function addLogEntry() {
            const logType = logTypes[Math.floor(Math.random() * logTypes.length)];
            const agent = agents[Math.floor(Math.random() * agents.length)];
            const message = messages[Math.floor(Math.random() * messages.length)];
            const now = new Date();
            const timestamp = now.toISOString().replace('T', ' ').substring(0, 19);

            const logEntry = document.createElement('div');
            logEntry.style.cssText = `padding: var(--space-2); border-left: 2px solid ${logType.border}; padding-left: var(--space-4); margin-bottom: var(--space-2); opacity: 0; transform: translateY(-10px);`;
            
            logEntry.innerHTML = `
                <span style="color: var(--color-text-tertiary);">[${timestamp}]</span>
                <span style="color: ${logType.color}; margin: 0 var(--space-2);">[${logType.level}]</span>
                <span style="color: var(--color-primary);">[${agent}]</span>
                <span style="color: var(--color-text-primary);"> ${message}</span>
            `;

            logsContainer.insertBefore(logEntry, logsContainer.firstChild);

            // Animate in
            setTimeout(() => {
                logEntry.style.transition = 'all 0.3s ease-out';
                logEntry.style.opacity = '1';
                logEntry.style.transform = 'translateY(0)';
            }, 10);

            // Keep only last 20 logs
            const logs = logsContainer.querySelectorAll('div');
            if (logs.length > 20) {
                logs[logs.length - 1].remove();
            }
        }

        // Add log every 2-5 seconds
        setInterval(() => {
            if (Math.random() > 0.3) { // 70% chance
                addLogEntry();
            }
        }, 3000);
    }

    // Progress bar animations
    document.querySelectorAll('.progress-bar-fill').forEach(bar => {
        const width = bar.style.width;
        bar.style.width = '0%';
        setTimeout(() => {
            bar.style.transition = 'width 1s ease-out';
            bar.style.width = width;
        }, 200);
    });

    // Dropdown menu functionality (already handled in main.js, but adding specific dashboard logic)
    document.querySelectorAll('.agent-status').forEach(status => {
        status.addEventListener('click', (e) => {
            e.stopPropagation();
            const menu = document.createElement('div');
            menu.className = 'dropdown-menu';
            menu.style.display = 'block';
            menu.style.position = 'absolute';
            menu.style.top = (e.target.getBoundingClientRect().bottom + 5) + 'px';
            menu.style.left = e.target.getBoundingClientRect().left + 'px';
            menu.innerHTML = `
                <a href="#" class="dropdown-item"><i data-lucide="play"></i> Start</a>
                <a href="#" class="dropdown-item"><i data-lucide="pause"></i> Pause</a>
                <a href="#" class="dropdown-item"><i data-lucide="square"></i> Stop</a>
                <div class="dropdown-divider"></div>
                <a href="#" class="dropdown-item"><i data-lucide="settings"></i> Configure</a>
            `;
            document.body.appendChild(menu);
            lucide.createIcons();

            document.addEventListener('click', () => menu.remove(), { once: true });
        });
    });

    // Agent card interactions
    document.querySelectorAll('.glass-card [data-lucide="more-vertical"]').forEach(btn => {
        btn.closest('button')?.addEventListener('click', (e) => {
            e.preventDefault();
            window.OspreyAI.showToast('info', 'Agent Options', 'Configure, pause, or delete this agent');
        });
    });

    // Settings toggles
    document.querySelectorAll('.toggle input').forEach(input => {
        input.addEventListener('change', (e) => {
            const label = e.target.closest('.settings-row')?.querySelector('.settings-label-title');
            const setting = label?.textContent || 'Setting';
            const status = e.target.checked ? 'enabled' : 'disabled';
            window.OspreyAI.showToast('success', `${setting} ${status}`, '');
        });
    });

    // Tab switching
    document.querySelectorAll('.tab').forEach(tab => {
        tab.addEventListener('click', () => {
            const parent = tab.closest('.tabs');
            parent.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            
            // Simulate chart data change
            setTimeout(animateChart, 100);
        });
    });

    // Refresh stats button (if exists)
    const refreshBtn = document.querySelector('[data-action="refresh-stats"]');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', () => {
            updateStats();
            window.OspreyAI.showToast('success', 'Refreshed', 'Stats updated');
        });
    }

    console.log('🦅 Osprey AI Dashboard initialized');
});
