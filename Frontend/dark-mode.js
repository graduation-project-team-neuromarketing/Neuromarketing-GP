document.addEventListener('DOMContentLoaded', () => {
    // --- Dark Mode Logic ---
    const themeToggleBtn = document.getElementById('theme-toggle');
    const themeIcon = document.getElementById('theme-icon');
    const htmlElement = document.documentElement;
    
    // Check for saved theme or system preference
    if (localStorage.theme === 'dark' || (!('theme' in localStorage) && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
        htmlElement.classList.add('dark');
        htmlElement.classList.remove('light');
        if(themeIcon) {
            if (themeIcon.tagName.toLowerCase() === 'i') {
                themeIcon.className = 'ph ph-sun';
            } else {
                themeIcon.textContent = 'light_mode';
            }
        }
    } else {
        htmlElement.classList.remove('dark');
        htmlElement.classList.add('light');
        if(themeIcon) {
            if (themeIcon.tagName.toLowerCase() === 'i') {
                themeIcon.className = 'ph ph-moon';
            } else {
                themeIcon.textContent = 'dark_mode';
            }
        }
    }

    if(themeToggleBtn) {
        themeToggleBtn.addEventListener('click', () => {
            if (htmlElement.classList.contains('dark')) {
                htmlElement.classList.remove('dark');
                htmlElement.classList.add('light');
                localStorage.theme = 'light';
                if(themeIcon) {
                    if (themeIcon.tagName.toLowerCase() === 'i') {
                        themeIcon.className = 'ph ph-moon';
                    } else {
                        themeIcon.textContent = 'dark_mode';
                    }
                }
            } else {
                htmlElement.classList.add('dark');
                htmlElement.classList.remove('light');
                localStorage.theme = 'dark';
                if(themeIcon) {
                    if (themeIcon.tagName.toLowerCase() === 'i') {
                        themeIcon.className = 'ph ph-sun';
                    } else {
                        themeIcon.textContent = 'light_mode';
                    }
                }
            }
        });
    }
});
