// Theme toggle functionality - loaded on all pages
(function() {
    const html = document.documentElement;
    const savedTheme = localStorage.getItem('theme') || 'light';
    
    // Apply theme immediately on page load (before page renders)
    html.setAttribute('data-theme', savedTheme);
    
    // Wait for DOM to be ready
    document.addEventListener('DOMContentLoaded', function() {
        // Initialize Lucide icons
        if (typeof lucide !== 'undefined') {
            lucide.createIcons();
        }
        
        const themeToggle = document.getElementById('themeToggle');
        
        function updateTheme(theme) {
            html.setAttribute('data-theme', theme);
            localStorage.setItem('theme', theme);
            
            if (themeToggle) {
                const sunIcon = themeToggle.querySelector('.sun-icon');
                const moonIcon = themeToggle.querySelector('.moon-icon');
                
                if (theme === 'dark') {
                    if (sunIcon) sunIcon.style.display = 'none';
                    if (moonIcon) moonIcon.style.display = 'block';
                } else {
                    if (sunIcon) sunIcon.style.display = 'block';
                    if (moonIcon) moonIcon.style.display = 'none';
                }
            }
        }
        
        // Set up click handler
        if (themeToggle) {
            themeToggle.addEventListener('click', () => {
                const currentTheme = html.getAttribute('data-theme');
                const newTheme = currentTheme === 'light' ? 'dark' : 'light';
                updateTheme(newTheme);
            });
        }
        
        // Apply theme on load
        updateTheme(savedTheme);
    });
})();
