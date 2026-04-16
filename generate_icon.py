"""Generate Verbatim.icns app icon programmatically using Pillow."""
import math
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

SIZE = 1024
OUT_PNG = Path("icon_source.png")
ICONSET = Path("Verbatim.iconset")
ASSETS = Path("assets")
ICNS_OUT = ASSETS / "Verbatim.icns"


def make_icon():
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Rounded-rect background: blue-to-indigo gradient (vertical)
    from PIL import ImageFilter

    # Draw gradient pixel by pixel using a helper image
    grad = Image.new("RGBA", (SIZE, SIZE))
    grad_draw = ImageDraw.Draw(grad)
    top_r, top_g, top_b = 0x2D, 0x9C, 0xFF   # cornflower blue
    bot_r, bot_g, bot_b = 0x4F, 0x46, 0xE5   # indigo

    for y in range(SIZE):
        t = y / (SIZE - 1)
        r = int(top_r + t * (bot_r - top_r))
        g = int(top_g + t * (bot_g - top_g))
        b = int(top_b + t * (bot_b - top_b))
        grad_draw.line([(0, y), (SIZE - 1, y)], fill=(r, g, b, 255))

    # Rounded-rect mask (radius ~22% of size)
    radius = int(SIZE * 0.22)
    mask = Image.new("L", (SIZE, SIZE), 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.rounded_rectangle([(0, 0), (SIZE - 1, SIZE - 1)], radius=radius, fill=255)

    # Composite gradient with rounded mask
    img.paste(grad, (0, 0), mask=mask)
    draw = ImageDraw.Draw(img)

    # Draw a closing-quotation-mark " centred on the icon
    # Use a large size and PIL's default font fallback
    quote_char = "\u201c"   # left double quotation mark
    font_size = int(SIZE * 0.62)
    font = None
    # Try a few system fonts likely available on macOS
    for font_path in [
        "/System/Library/Fonts/Supplemental/Georgia.ttf",
        "/Library/Fonts/Georgia.ttf",
        "/System/Library/Fonts/Times.ttc",
        "/System/Library/Fonts/Helvetica.ttc",
    ]:
        try:
            font = ImageFont.truetype(font_path, font_size)
            break
        except Exception:
            pass

    if font is None:
        # Fallback: draw a simple waveform instead
        draw_waveform(draw, SIZE)
    else:
        # Measure and centre
        bbox = draw.textbbox((0, 0), quote_char, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        x = (SIZE - tw) // 2 - bbox[0]
        y = (SIZE - th) // 2 - bbox[1] - int(SIZE * 0.04)
        # Drop shadow
        draw.text((x + 8, y + 8), quote_char, font=font, fill=(0, 0, 0, 60))
        draw.text((x, y), quote_char, font=font, fill=(255, 255, 255, 240))

    img.save(OUT_PNG)
    print(f"Saved {OUT_PNG}")


def draw_waveform(draw, size):
    """Fallback: draw a simple audio waveform."""
    cx = size // 2
    cy = size // 2
    bars = 9
    bar_w = int(size * 0.045)
    gap = int(size * 0.025)
    total_w = bars * bar_w + (bars - 1) * gap
    heights = [0.20, 0.38, 0.55, 0.70, 0.82, 0.70, 0.55, 0.38, 0.20]
    x = cx - total_w // 2
    for i, h in enumerate(heights):
        bh = int(size * h)
        x0 = x + i * (bar_w + gap)
        y0 = cy - bh // 2
        y1 = cy + bh // 2
        r = bar_w // 2
        draw.rounded_rectangle([(x0, y0), (x0 + bar_w, y1)], radius=r, fill=(255, 255, 255, 230))


def build_iconset():
    ICONSET.mkdir(exist_ok=True)
    ASSETS.mkdir(exist_ok=True)

    sizes = [
        ("icon_512x512@2x.png", 1024),
        ("icon_512x512.png", 512),
        ("icon_256x256@2x.png", 512),
        ("icon_256x256.png", 256),
        ("icon_128x128@2x.png", 256),
        ("icon_128x128.png", 128),
        ("icon_32x32@2x.png", 64),
        ("icon_32x32.png", 32),
        ("icon_16x16@2x.png", 32),
        ("icon_16x16.png", 16),
    ]

    src = Image.open(OUT_PNG).convert("RGBA")
    for name, px in sizes:
        resized = src.resize((px, px), Image.LANCZOS)
        resized.save(ICONSET / name)
        print(f"  {name} ({px}x{px})")

    subprocess.run(
        ["iconutil", "-c", "icns", str(ICONSET), "-o", str(ICNS_OUT)],
        check=True,
    )
    print(f"Created {ICNS_OUT}")


if __name__ == "__main__":
    make_icon()
    build_iconset()
    print("Done.")
