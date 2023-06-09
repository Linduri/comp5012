"""
Identify what fonts matplotlib can use
"""

import matplotlib.font_manager
fpaths = matplotlib.font_manager.findSystemFonts()

for i in fpaths:
    f = matplotlib.font_manager.get_font(i)
    print(f"{f.family_name}")
