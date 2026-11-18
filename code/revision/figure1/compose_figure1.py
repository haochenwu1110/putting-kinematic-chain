"""Rebuild Figure 1: original illustration + panels (b) and (e) redrawn with v1.1 data + two relabelled waterfall values in (d)."""
import sys
from PIL import Image, ImageDraw, ImageFont
import numpy as np
orig, b_png, e_png, out = sys.argv[1:5]
F = Image.open(orig).convert("RGB")
# (b) right: crop origin (3100, 110); keep the dashed frame at x >= 1136
b = Image.open(b_png).convert("RGB"); F.paste(b.crop((0, 0, 1136, 1100)), (3100, 110))
# (e) scatters: crop origin (2780, 1600); keep the dashed frame at x < 20 and x >= 1445
e = Image.open(e_png).convert("RGB"); F.paste(e.crop((20, 20, 1445, 915)), (2800, 1620))
# (d) waterfall labels: crop origin (1270, 2780)
dr = ImageDraw.Draw(F); X0, Y0 = 1270, 2780
INK, GRID = (27, 27, 27), (228, 227, 223)
REG = "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"
BOLD = "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf"
def relabel(box, text, font_path, grid_rows=()):
    x0, y0, x1, y1 = box
    dr.rectangle((X0 + x0 - 4, Y0 + y0 - 4, X0 + x1 + 4, Y0 + y1 + 4), fill="white")
    for gy in grid_rows:
        dr.line((X0 + x0 - 4, Y0 + gy, X0 + x1 + 4, Y0 + gy), fill=GRID)
    h = y1 - y0 + 1
    for size in range(20, 60):                       # match digit height of the original label
        f = ImageFont.truetype(font_path, size)
        bb = dr.textbbox((0, 0), text, font=f)
        if bb[3] - bb[1] >= h: break
    cx = X0 + (x0 + x1) / 2
    dr.text((cx - (bb[2] + bb[0]) / 2, Y0 + y0 - bb[1]), text, font=f, fill=INK)
    return size
print("size 0.9:", relabel((237, 704, 277, 727), "0.9", REG))
# "Measured" = grip displacement integrated from the middle of the address window to the top, i.e. the sum of all
# bars including the residual (example putt P19, trial 4: terms 76.72 + residual 0.24 = 76.96 mm)
print("size 77.0:", relabel((1132, 152, 1191, 175), "77.0", BOLD, grid_rows=(168, 169, 170)))
F.save(out, dpi=(600, 600))
print("saved", out, F.size)
