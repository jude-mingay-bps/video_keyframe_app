// /static/js/tabs.js

document.addEventListener('DOMContentLoaded', () => {
    const tabLinks = document.querySelectorAll('.tab-link');
    const tabContents = document.querySelectorAll('.tab-content');

    tabLinks.forEach(link => {
        link.addEventListener('click', () => {
            const tabId = link.dataset.tab;

            // Deactivate all tabs and content
            tabLinks.forEach(item => item.classList.remove('active'));
            tabContents.forEach(item => item.classList.remove('active'));

            // Activate the clicked tab and corresponding content
            link.classList.add('active');
            document.getElementById(tabId).classList.add('active');
        });
    });

    // Activate the first tab by default
    if (tabLinks.length > 0) {
        tabLinks[0].click();
    }
});
