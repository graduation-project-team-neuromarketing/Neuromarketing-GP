import os
import glob
import re

html_files = glob.glob('*.html')
for file in html_files:
    with open(file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Identify pages with the profile picture in the header
    if 'alt="User Profile"' in content and 'class="w-full h-full object-cover"' in content:
        # Give the image an ID if it doesn't have one
        if 'id="header-profile-img"' not in content:
            # We look for the image tag specifically
            new_content = re.sub(
                r'<img alt="User Profile" src="([^"]+)" class="([^"]+)"/>',
                r'<img id="header-profile-img" alt="User Profile" src="\1" class="\2"/>',
                content
            )
            
            script = """
<script>
document.addEventListener('DOMContentLoaded', async () => {
    const token = localStorage.getItem('access_token');
    const headerProfileImg = document.getElementById('header-profile-img');
    if (token && headerProfileImg) {
        try {
            const API_BASE = 'http://127.0.0.1:8000';
            const response = await fetch(`${API_BASE}/users/me`, {
                headers: { 'Authorization': `Bearer ${token}` }
            });
            if (response.ok) {
                const userData = await response.json();
                if (userData.profile_pic) {
                    headerProfileImg.src = userData.profile_pic.startsWith('http') ? userData.profile_pic : `${API_BASE}${userData.profile_pic}`;
                }
            }
        } catch (error) {
            console.error('Error fetching profile pic for header:', error);
        }
    }
});
</script>
</body>"""
            
            # Prevent double appending if script already added
            if "header-profile-img" not in content and "Error fetching profile pic for header" not in new_content:
                 new_content = new_content.replace('</body>', script)
            
            with open(file, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f'Updated {file}')
