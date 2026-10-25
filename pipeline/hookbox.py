#!/usr/bin/env python3
"""The hook as TikTok draws its own text: white rounded boxes, one per line, merged into one shape. Outer corners
are rounded, and where a narrow line sits on a wider one the step gets a concave curve. The font is TikTok Sans,
and emoji are drawn in color from Noto Color Emoji.

Rendered with Pillow at 4x and downscaled, so it stays sharp. Returns a transparent PNG for ffmpeg to overlay.

usage: hookbox.py "<text>" out.png [size]
"""
import os, sys
import regex
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
FONT = os.path.join(HERE, 'fonts', 'TikTokSans-SemiBold.ttf')
EMOJI = next((p for p in ('/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf', os.path.join(HERE, 'fonts', 'NotoColorEmoji.ttf')) if os.path.exists(p)), None)
SS = 4   # supersampling


def is_emoji(cluster):
    return any(ord(ch) >= 0x1F000 or 0x2600 <= ord(ch) <= 0x27BF or 0x2B00 <= ord(ch) <= 0x2BFF or ord(ch) in (0x200D, 0xFE0F, 0x2122, 0x00A9, 0x00AE)
               for ch in cluster) and not cluster.isalnum()


def runs(text):
    """Split into (is_emoji, text) runs on grapheme clusters, so ZWJ sequences like ❤️‍🩹 stay whole."""
    out = []
    for cl in regex.findall(r'\X', text):
        e = is_emoji(cl)
        if out and out[-1][0] == e:
            out[-1] = (e, out[-1][1] + cl)
        else:
            out.append((e, cl))
    return out


class Painter:
    def __init__(self, size):
        self.size = size * SS
        self.font = ImageFont.truetype(FONT, self.size, layout_engine=ImageFont.Layout.RAQM)
        self.emoji = ImageFont.truetype(EMOJI, 109, layout_engine=ImageFont.Layout.RAQM) if EMOJI else None   # CBDT: 109 px only
        self.em_h = round(self.size * 1.08)
        self.cache = {}

    def emoji_img(self, cl):
        if cl not in self.cache:
            box = self.emoji.getbbox(cl)
            im = Image.new('RGBA', (box[2] - box[0] + 4, box[3] - box[1] + 4), (0, 0, 0, 0))
            ImageDraw.Draw(im).text((2 - box[0], 2 - box[1]), cl, font=self.emoji, embedded_color=True)
            im = im.crop(im.getbbox() or (0, 0, im.width, im.height))      # trim side bearings so emoji sit close
            w = round(im.width * self.em_h / im.height)
            self.cache[cl] = im.resize((w, self.em_h), Image.LANCZOS)
        return self.cache[cl]

    def width(self, text):
        w = 0
        for e, t in runs(text):
            if e and self.emoji:
                w += sum(self.emoji_img(cl).width + self.size * 0.1 for cl in regex.findall(r'\X', t))
            else:
                w += self.font.getlength(t)
        return w

    def draw(self, img, x, y_base, text):
        d = ImageDraw.Draw(img)
        for e, t in runs(text):
            if e and self.emoji:
                for cl in regex.findall(r'\X', t):
                    em = self.emoji_img(cl)
                    img.alpha_composite(em, (round(x + self.size * 0.05), round(y_base - self.size * 0.84)))
                    x += em.width + self.size * 0.1
            else:
                d.text((x, y_base), t, font=self.font, fill=(0, 0, 0, 255), anchor='ls')
                x += self.font.getlength(t)


def wrap(p, text, max_w):
    """Balanced lines like TikTok's own wrapping: as few lines as fit, as even as possible, the longer line at the bottom."""
    words = text.split()
    if not words:
        return []
    width = lambda ws: p.width(' '.join(ws))
    if width(words) <= max_w:
        return [' '.join(words)]
    # how many lines greedy needs, then the most even split into that many (2 or 3 lines; greedy beyond)
    n, cur = 1, []
    for w in words:
        if cur and width(cur + [w]) > max_w:
            n += 1; cur = [w]
        else:
            cur.append(w)
    best = None
    if n == 2:
        for i in range(1, len(words)):
            ls = [words[:i], words[i:]]
            ws = [width(x) for x in ls]
            if max(ws) > max_w:
                continue
            score = max(ws) + (0.04 * max_w if ws[0] > ws[1] else 0)          # bottom-heavy reads like TikTok
            best = min(best or (1e18, None), (score, ls), key=lambda t: t[0])
    elif n == 3:
        for i in range(1, len(words) - 1):
            for j in range(i + 1, len(words)):
                ls = [words[:i], words[i:j], words[j:]]
                ws = [width(x) for x in ls]
                if max(ws) > max_w:
                    continue
                score = max(ws) - min(ws) * 0.3
                best = min(best or (1e18, None), (score, ls), key=lambda t: t[0])
    if best and best[1]:
        lines = [' '.join(x) for x in best[1]]
    else:
        lines, cur = [], []
        for w in words:
            if cur and width(cur + [w]) > max_w:
                lines.append(' '.join(cur)); cur = [w]
            else:
                cur.append(w)
        lines.append(' '.join(cur))
    # never leave emoji alone on a line
    if len(lines) > 1 and all(is_emoji(c) or c.isspace() for c in regex.findall(r'\X', lines[-1])):
        lines[-2] += ' ' + lines.pop()
    return lines


def render(text, out_png, size=54, max_w=860, frame_w=1080):
    p = Painter(size)
    lines = wrap(p, ' '.join(text.split()), max_w * SS)
    pad_x, pad_y, r = 0.42 * p.size, 0.26 * p.size, 0.36 * p.size
    asc, desc = p.font.getmetrics()
    lh = round(asc * 0.78 + desc * 0.55 + 2 * pad_y)          # cap height + a bit of descender, like TikTok's tight boxes
    widths = [p.width(t) + 2 * pad_x for t in lines]
    # near-equal neighbours get the same width: a step smaller than two corner radii looks like a mistake
    for i in range(1, len(widths)):
        if abs(widths[i] - widths[i - 1]) < 2.2 * r:
            widths[i] = widths[i - 1] = max(widths[i], widths[i - 1])
    W = round(max(widths) + 4 * r)
    H = round(lh * len(lines) + 2 * r)
    cx, top = W / 2, r
    mask = Image.new('L', (W, H), 0)
    m = ImageDraw.Draw(mask)
    boxes = [(cx - w / 2, top + i * lh, cx + w / 2, top + (i + 1) * lh) for i, w in enumerate(widths)]
    for b in boxes:
        m.rectangle(b, fill=255)

    def convex(x, y, sx, sy):
        """Round the corner at (x, y); (sx, sy) point from the corner into the box."""
        m.rectangle((min(x, x + sx * r), min(y, y + sy * r), max(x, x + sx * r), max(y, y + sy * r)), fill=0)
        c = (x + sx * r, y + sy * r)
        m.ellipse((c[0] - r, c[1] - r, c[0] + r, c[1] + r), fill=255)

    def concave(x, y, sx, sy):
        """Fill the inside step at (x, y) with a curve; (sx, sy) point away from the narrower box, into empty space."""
        m.rectangle((min(x, x + sx * r), min(y, y + sy * r), max(x, x + sx * r), max(y, y + sy * r)), fill=255)
        c = (x + sx * r, y + sy * r)
        m.ellipse((c[0] - r, c[1] - r, c[0] + r, c[1] + r), fill=0)

    for i, (x0, y0, x1, y1) in enumerate(boxes):
        above = boxes[i - 1] if i else None
        below = boxes[i + 1] if i + 1 < len(boxes) else None
        if not above or above[2] - above[0] < x1 - x0 - 1:      # nothing wider above: top corners are outer
            convex(x0, y0, 1, 1); convex(x1, y0, -1, 1)
            if above:                                           # narrower line above: curve into its sides
                concave(above[0], y0, -1, -1); concave(above[2], y0, 1, -1)
        if not below or below[2] - below[0] < x1 - x0 - 1:
            convex(x0, y1, 1, -1); convex(x1, y1, -1, -1)
            if below:
                concave(below[0], y1, -1, 1); concave(below[2], y1, 1, 1)
    img = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    img.paste(Image.new('RGBA', (W, H), (255, 255, 255, 255)), (0, 0), mask)
    for i, t in enumerate(lines):
        y_base = top + i * lh + pad_y + asc * 0.78
        p.draw(img, cx - p.width(t) / 2, y_base, t)
    out = img.resize((round(W / SS), round(H / SS)), Image.LANCZOS)
    out.save(out_png)
    return {'w': out.width, 'h': out.height, 'lines': lines, 'pad': round(r / SS)}


if __name__ == '__main__':
    print(render(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 54))
