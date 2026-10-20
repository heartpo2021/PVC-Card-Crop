#!/usr/bin/env python3
"""
PVC Card Auto Tool  (offline)
=============================
PDF  ->  auto detect card  ->  Front + Back  ->  85.6 x 54 mm (CR80)  ->  PNG (600 DPI)

Supported (auto detect):
  * Aadhaar (UIDAI letter PDF)       -> bottom 2 cards (front/back)
  * Voter ID (e-EPIC)                -> 2 cards
  * Ayushman / ABHA card             -> 2 cards
  * e-Nirman / Worker card           -> 2 cards
  * Any other PDF where the card is a picture/box of card shape (PAN, DL, RC ...)
    -> generic detector (image objects + pixel based box finder)

Run:
    python pvc_card_tool.py            -> window (GUI) khule che
    python pvc_card_tool.py a.pdf b.pdf -o output_folder
    python pvc_card_tool.py "C:\\pdfs"  -o output_folder
"""
import argparse
import os
import sys
import threading
import traceback
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c

# ----------------------------------------------------------------------------
CARD_W_MM = 85.6
CARD_H_MM = 54.0
CARD_AR = CARD_W_MM / CARD_H_MM          # 1.5852
MAX_STRETCH = 0.32                       # card shape ma 32% sudhi farak hoy to stretch kari fit karvu
MIN_CARD_W_PT = 120                      # PDF ma card ni minimum width (points)
DEFAULT_DPI = 600


def default_in_dir():
    """PDF select karva nu default folder: C:\\Users\\harsh\\Downloads"""
    for p in (Path(r"C:\Users\harsh\Downloads"), Path.home() / "Downloads", Path.home()):
        if p.exists():
            return p
    return Path.cwd()


def default_out_dir():
    """Output default: Downloads\\crop"""
    return Path.home() / "Downloads" / "Crop"


def out_size(dpi):
    return round(CARD_W_MM / 25.4 * dpi), round(CARD_H_MM / 25.4 * dpi)


# ----------------------------------------------------------------------------
# Coordinate helper: PDF (unrotated, y-up)  ->  visual page (top-left origin, points)
# ----------------------------------------------------------------------------
def make_to_visual(page):
    W, H = page.get_size()                 # visual (rotated) size
    rot = page.get_rotation()

    def to_visual(x, y):
        if rot == 0:
            return x, H - y
        if rot == 90:
            return y, x
        if rot == 180:
            return W - x, y
        return W - y, H - x                # 270

    return to_visual, W, H


def rect_to_visual(to_visual, l, b, r, t):
    p1 = to_visual(l, b)
    p2 = to_visual(r, t)
    return (min(p1[0], p2[0]), min(p1[1], p2[1]), max(p1[0], p2[0]), max(p1[1], p2[1]))


# ----------------------------------------------------------------------------
# Card finders  (each returns list of (x0,y0,x1,y1) in visual points)
# ----------------------------------------------------------------------------
def _dedupe(rects):
    rects = sorted(rects, key=lambda r: (r[2] - r[0]) * (r[3] - r[1]), reverse=True)
    keep = []
    for r in rects:
        area = (r[2] - r[0]) * (r[3] - r[1])
        dup = False
        for k in keep:
            ix = max(0, min(r[2], k[2]) - max(r[0], k[0]))
            iy = max(0, min(r[3], k[3]) - max(r[1], k[1]))
            if ix * iy > 0.7 * area:
                dup = True
                break
        if not dup:
            keep.append(r)
    return keep


def find_aadhaar_cards(page):
    """UIDAI letter: outer border -> bottom row has 2 cards (front | back)."""
    to_visual, W, H = make_to_visual(page)
    paths = []
    for o in page.get_objects():
        if o.type == pdfium_c.FPDF_PAGEOBJ_PATH:
            l, b, r, t = o.get_bounds()
            v = rect_to_visual(to_visual, l, b, r, t)
            w, h = v[2] - v[0], v[3] - v[1]
            if w > 0.6 * W and h > 0.5 * H:
                paths.append(v)
    if not paths:
        return []
    bx0, by0, bx1, by1 = min(paths, key=lambda v: (v[2] - v[0]) * (v[3] - v[1]))  # inner border
    cw = (bx1 - bx0) / 2.0
    ch = cw / CARD_AR
    mid = bx0 + cw
    top = by1 - ch
    # border line / dashed cut line mathi thodu andar
    e_out = 1.2        # bahar ni border line
    e_cut = 5.0        # vachhe ni dashed cut-line + scissors
    e_top = 3.5        # upar ni dashed line + scissors
    e_bot = 1.2
    front = (bx0 + e_out, top + e_top, mid - e_cut, by1 - e_bot)
    back = (mid + e_cut, top + e_top, bx1 - e_out, by1 - e_bot)
    return [front, back]


def find_image_cards(page):
    """Card jevi shape vala image objects (voter, ayushman, e-nirman, pan, dl ...)."""
    to_visual, W, H = make_to_visual(page)
    rects = []
    for o in page.get_objects():
        if o.type != pdfium_c.FPDF_PAGEOBJ_IMAGE:
            continue
        l, b, r, t = o.get_bounds()
        v = rect_to_visual(to_visual, l, b, r, t)
        w, h = v[2] - v[0], v[3] - v[1]
        if w < MIN_CARD_W_PT or h < 60:
            continue
        if w > 0.92 * W and h > 0.92 * H:
            continue
        ar = w / h
        if not (1.35 <= ar <= 2.1):
            continue
        rects.append(v)
    return _dedupe(rects)


def find_pixel_cards(page, scale=2.0):
    """Last fallback: page render kari ne card jevi box shodhvu (OpenCV joiye)."""
    try:
        import cv2
    except ImportError:
        return []
    img = page.render(scale=scale).to_pil().convert("L")
    a = np.asarray(img)
    mask = (a < 245).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    W, H = page.get_size()
    rects = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        wp, hp = w / scale, h / scale
        if wp < MIN_CARD_W_PT or hp < 60:
            continue
        if wp > 0.92 * W and hp > 0.92 * H:
            continue
        if not (1.35 <= wp / hp <= 2.1):
            continue
        rects.append((x / scale, y / scale, (x + w) / scale, (y + h) / scale))
    return _dedupe(rects)



def _sig_images(page):
    """Moti image objects ni list (visual rect)."""
    to_visual, W, H = make_to_visual(page)
    out = []
    for o in page.get_objects():
        if o.type != pdfium_c.FPDF_PAGEOBJ_IMAGE:
            continue
        l, b, r, t = o.get_bounds()
        v = rect_to_visual(to_visual, l, b, r, t)
        if (v[2] - v[0]) >= MIN_CARD_W_PT and (v[3] - v[1]) >= 60:
            out.append(v)
    return out


def find_pair_image(page):
    """PAN jevi PDF: ek j wide image ma front | back (ar ~3). Image bounds clip hoy etle bahu-side thodu vadhari ne
    pachhi rang (saturation) thi be card alag kari shodhva."""
    W, H = page.get_size()
    for v in _sig_images(page):
        w, h = v[2] - v[0], v[3] - v[1]
        if w >= 300 and 2.5 <= w / h <= 3.6:
            return [(max(0, v[0] - 40), max(0, v[1] - 4), min(W, v[2] + 40), min(H, v[3] + 4))]
    return []


def find_stacked_pair(page):
    """e-Shram jevi PDF: ek j image ma 2 card upar-niche (ke bajubaju) + shadow."""
    imgs = _sig_images(page)
    if len(imgs) != 1:
        return []
    img = page.render(scale=1.0).to_pil().convert("L")
    a = np.asarray(img)
    ys, xs = np.where(a < 250)
    if len(xs) == 0:
        return []
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    w, h = x1 - x0, y1 - y0
    if w < MIN_CARD_W_PT or h < 60:
        return []
    ar = w / h
    if 0.70 <= ar <= 0.92:                       # upar-niche
        m = (y0 + y1) / 2
        return [(x0, y0, x1, m), (x0, m, x1, y1)]
    if 2.6 <= ar <= 3.5:                         # bajubaju
        m = (x0 + x1) / 2
        return [(x0, y0, m, y1), (m, y0, x1, y1)]
    return []


def find_fullpage_card(page):
    """Page j card size no hoy (cropped PDF)."""
    W, H = page.get_size()
    if 1.45 <= W / H <= 1.75:
        return [(0, 0, W, H)]
    return []


def reading_order(rects):
    if len(rects) <= 1:
        return rects
    avg_h = sum(r[3] - r[1] for r in rects) / len(rects)
    return sorted(rects, key=lambda r: (round(((r[1] + r[3]) / 2) / (avg_h * 0.8)), r[0]))


def detect_cards(page, force_pixel=False):
    """returns (kind, rects)"""
    if not force_pixel:
        try:
            txt = page.get_textpage().get_text_range()
        except Exception:
            txt = ""
        low = " ".join(txt.lower().split())          # line-break / extra space hataavo
        if "unique identification authority" in low or "uidai" in low or "aadhaar" in low:
            r = find_aadhaar_cards(page)
            if len(r) == 2:
                return "aadhaar", r
        r = find_pair_image(page)
        if r:
            return "pair", r
        r = find_stacked_pair(page)
        if r:
            return "stack", r
        r = find_image_cards(page)
        if r:
            return "image", reading_order(r)
        r = find_fullpage_card(page)
        if r:
            return "full", r
    r = find_pixel_cards(page)
    return "pixel", reading_order(r)


# ----------------------------------------------------------------------------
# Image helpers
# ----------------------------------------------------------------------------
def trim_dark_border(img, max_trim=14):
    """Card ni bahar ni kali/grey pateli line (frame) kapi nakhva."""
    arr = np.asarray(img.convert("RGB")).astype(np.int16)
    h, w, _ = arr.shape

    def dark_gray(line):                    # line: (n,3)
        m = line.mean(axis=0)
        return m.mean() < 125 and (m.max() - m.min()) < 28

    l, t, r, b = 0, 0, w, h
    for _ in range(max_trim):
        if dark_gray(arr[t:b, l]):
            l += 1
        else:
            break
    for _ in range(max_trim):
        if dark_gray(arr[t:b, r - 1]):
            r -= 1
        else:
            break
    for _ in range(max_trim):
        if dark_gray(arr[t, l:r]):
            t += 1
        else:
            break
    for _ in range(max_trim):
        if dark_gray(arr[b - 1, l:r]):
            b -= 1
        else:
            break
    if (l, t, r, b) != (0, 0, w, h):
        pad = 2                              # anti-alias ni half-dark pixel pan kadhi nakho
        l, t, r, b = l + pad, t + pad, r - pad, b - pad
        return img.crop((l, t, r, b))
    return img


def _runs(idx):
    """sorted index array -> list of (start,end) continuous runs"""
    out = []
    if len(idx) == 0:
        return out
    st = prev = idx[0]
    for i in idx[1:]:
        if i != prev + 1:
            out.append((st, prev))
            st = i
        prev = i
    out.append((st, prev))
    return out


def refine_by_frame(img, max_line=16):
    """PDF ma image bounds card thi motha hoy (clip) to, card ni kali frame line shodhi ne tya sudhi j rakhvu."""
    rgb = np.asarray(img.convert("RGB")).astype(np.int16)
    h, w, _ = rgb.shape
    # fakt kali/grey (colour vagar) pixel -> red/green design lines frame na gane
    dark = (rgb.mean(axis=2) < 100) & ((rgb.max(axis=2) - rgb.min(axis=2)) < 35)
    # ---- left / right
    frac = dark.mean(axis=0)
    runs = [r for r in _runs(np.where(frac > 0.90)[0]) if (r[1] - r[0] + 1) <= max_line]
    left = [r for r in runs if r[1] < 0.25 * w]
    right = [r for r in runs if r[0] > 0.60 * w]
    x0, x1 = 0, w
    if left:
        x0 = min(left)[0]
    if right:
        x1 = min(right)[1] + 1          # sauthi andar vali (card ni potani) frame
    if x1 - x0 < 0.5 * w:
        x0, x1 = 0, w
    # ---- top / bottom
    frac_r = dark[:, x0:x1].mean(axis=1)
    rruns = [r for r in _runs(np.where(frac_r > 0.90)[0]) if (r[1] - r[0] + 1) <= max_line]
    top = [r for r in rruns if r[1] < 0.08 * h]
    bot = [r for r in rruns if r[0] > 0.90 * h]
    y0, y1 = 0, h
    if top:
        y0 = min(top)[0]
    if bot:
        y1 = max(bot)[1] + 1
    if y1 - y0 < 0.5 * h:
        y0, y1 = 0, h
    if (x0, x1, y0, y1) == (0, w, 0, h):
        return img
    return img.crop((x0, y0, x1, y1))



def tight_by_saturation(img):
    """Rangeen (blue/pink) card body ni tight bounding box (PAN: dashed line + safed margin kadhva)."""
    a = np.asarray(img.convert("RGB")).astype(np.int16)
    sat = (a.max(axis=2) - a.min(axis=2)) > 28
    h, w = sat.shape
    cols = np.where(sat.mean(axis=0) > 0.5)[0]
    rows = np.where(sat.mean(axis=1) > 0.5)[0]
    if len(cols) == 0 or len(rows) == 0:
        return img
    x0, x1, y0, y1 = cols.min(), cols.max() + 1, rows.min(), rows.max() + 1
    if (x1 - x0) < 0.5 * w or (y1 - y0) < 0.5 * h:
        return img
    return img.crop((x0, y0, x1, y1))


def _long_runs(mask, min_len):
    idx = np.where(mask)[0]
    runs = []
    if len(idx):
        st = prev = idx[0]
        for i in idx[1:]:
            if i != prev + 1:
                runs.append((st, prev + 1))
                st = i
            prev = i
        runs.append((st, prev + 1))
    return [r for r in runs if r[1] - r[0] >= min_len]


def split_pair_by_saturation(img):
    """Be card (front|back, rounded body, bahar dashed line + safed margin) ne alag crop kare.
    Card body = 'safed nathi' vali jagya; dashed line patli hoy etle min-length thi nikdi jay."""
    g = np.asarray(img.convert("L")).astype(np.int16)
    h, w = g.shape
    band = g[int(h * 0.35): int(h * 0.65)] < 245            # vachhe ni patti ma body hamesha non-white
    runs = _long_runs(band.mean(axis=0) > 0.8, int(w * 0.10))
    if not runs:
        return []
    L, R = runs[0][0], runs[-1][1]
    cw = max(r[1] - r[0] for r in runs)               # sauthi saaf (back) card ni width
    if len(runs) < 2 and (R - L) < 1.6 * cw:
        return []
    # be card same size: ek baju thi L+cw, biji baju thi R-cw  (QR/photo thi run tuti jay to pan chale)
    cols = [(L, L + cw), (R - cw, R)]
    # height: sauthi saaf column mathi
    best = max(runs, key=lambda r: r[1] - r[0])
    rr = _long_runs((g[:, best[0]:best[1]] < 245).mean(axis=1) > 0.8, int(h * 0.30))
    if not rr:
        return []
    y0, y1 = min(r[0] for r in rr), max(r[1] for r in rr)
    return [img.crop((x0, y0, x1, y1)) for x0, x1 in cols]


def shrink_to_body(img, tol=10, max_frac=0.08):
    """Shadow / blur rim kadhi ne card ni (ek rang vali) body sudhi shrink karvu (e-Shram)."""
    a = np.asarray(img.convert("L")).astype(np.int16)
    h, w = a.shape
    med = np.median(a[h // 4: 3 * h // 4, w // 4: 3 * w // 4])
    ok = np.abs(a - med) <= tol
    l, t, r, b = 0, 0, w, h
    lim_x, lim_y = int(w * max_frac), int(h * max_frac)
    for _ in range(lim_x):
        if ok[t + h // 10: b - h // 10, l].mean() < 0.85:
            l += 1
        else:
            break
    for _ in range(lim_x):
        if ok[t + h // 10: b - h // 10, r - 1].mean() < 0.85:
            r -= 1
        else:
            break
    for _ in range(lim_y):
        if ok[t, l + w // 10: r - w // 10].mean() < 0.85:
            t += 1
        else:
            break
    for _ in range(lim_y):
        if ok[b - 1, l + w // 10: r - w // 10].mean() < 0.85:
            b -= 1
        else:
            break
    return img.crop((l + 1, t + 1, r - 1, b - 1))


def fit_to_cr80(img, dpi):
    """Remove only empty outer whitespace, then fill CR80 without white bands.

    Large aspect mismatches are not forcibly cropped: preserve the card instead.
    """
    ow, oh = out_size(dpi)
    img = img.convert("RGB")
    arr = np.asarray(img)
    # Look for nearly-white blank bands; require a row/column to be >99.5% white.
    white = np.all(arr >= 248, axis=2)
    good_rows = np.where(white.mean(axis=1) < 0.995)[0]
    good_cols = np.where(white.mean(axis=0) < 0.995)[0]
    if len(good_rows) and len(good_cols):
        y0, y1 = int(good_rows[0]), int(good_rows[-1]) + 1
        x0, x1 = int(good_cols[0]), int(good_cols[-1]) + 1
        if x1-x0 > 0.65*img.width and y1-y0 > 0.65*img.height:
            img = img.crop((x0, y0, x1, y1))
    ar = img.width / img.height
    deviation = abs(ar / CARD_AR - 1)
    if deviation <= 0.04:
        # Tiny mismatch: fill the card with center crop; never create white bands.
        scale = max(ow / img.width, oh / img.height)
        nw, nh = max(ow, round(img.width*scale)), max(oh, round(img.height*scale))
        if (nw, nh) != img.size:
            img = img.resize((nw, nh), Image.Resampling.LANCZOS)
        x, y = (nw-ow)//2, (nh-oh)//2
        return img.crop((x,y,x+ow,y+oh)), "white margins removed; CR80 filled"
    # Preserve full card contents and fill the entire print area, as in the original project.
    # This may alter proportions if the PDF's detected card ratio is unusual.
    return img.resize((ow, oh), Image.Resampling.LANCZOS), f"filled CR80; source ratio differs by {deviation:.1%}"


# ----------------------------------------------------------------------------
# Main processing
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Quality improve
# ----------------------------------------------------------------------------
def page_effective_dpi(page):
    """Page ni sauthi moti image no asli (native) DPI. Ochho hoy to image 'blocky' dekhay."""
    best, best_area = None, 0.0
    for o in page.get_objects():
        if o.type != pdfium_c.FPDF_PAGEOBJ_IMAGE:
            continue
        try:
            l, b, r, t = o.get_bounds()
            m = o.get_metadata()
        except Exception:
            continue
        area = (r - l) * (t - b)
        if area > best_area and (r - l) > 40 and (t - b) > 25 and m.width and m.height:
            best_area = area
            best = (m.width * m.height / area) ** 0.5 * 72.0
    return best


def enhance_card(img, dpi, eff_dpi):
    """Preserve native PDF detail: do not blur, oversharpen or invent pixels.

    Raster sources with insufficient resolution cannot gain real detail through
    filters. The original rendered colors and QR modules remain untouched.
    """
    return img


# ----------------------------------------------------------------------------
# Password protected PDF
# ----------------------------------------------------------------------------
class PasswordCancelled(Exception):
    pass


def _is_password_error(e):
    code = getattr(e, "err_code", None)
    return code == getattr(pdfium_c, "FPDF_ERR_PASSWORD", 4) or "password" in str(e).lower()


def open_pdf(path, ask_password=None, known=None, log=print):
    """PDF kholvo. Password joiye to: (1) pehla thi yaad rakhela passwords try, (2) pachhi user ne puchhvu (3 vaar)."""
    known = known if known is not None else []
    try:
        return pdfium.PdfDocument(str(path))
    except pdfium.PdfiumError as e:
        if not _is_password_error(e):
            raise
    log("  (PDF password protected che)")
    for pw in list(known):                                   # aagal na PDF no password same hoy to
        try:
            return pdfium.PdfDocument(str(path), password=pw)
        except pdfium.PdfiumError as e:
            if not _is_password_error(e):
                raise
    for attempt in range(3):
        if ask_password is None:
            break
        pw = ask_password(Path(path).name, attempt)
        if pw is None:
            break
        try:
            doc = pdfium.PdfDocument(str(path), password=pw)
            if pw not in known:
                known.append(pw)
            return doc
        except pdfium.PdfiumError as e:
            if not _is_password_error(e):
                raise
            log("  ! password khotu che")
    raise PasswordCancelled()


def process_pdf(pdf_path, out_dir, dpi=DEFAULT_DPI, force_pixel=False, log=print,
                ask_password=None, known_passwords=None, enhance=True):
    pdf_path = Path(pdf_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = pdf_path.stem
    pdf = open_pdf(pdf_path, ask_password, known_passwords, log)
    scale = dpi / 72.0

    cards, kinds, effs = [], [], []
    for pi in range(min(len(pdf), 4)):               # cropped PDF ma front/back alag page hoy
        page = pdf[pi]
        kind, rects = detect_cards(page, force_pixel)
        if not rects:
            continue
        eff = page_effective_dpi(page)
        full = page.render(scale=scale, force_halftone=False).to_pil().convert("RGB")
        for (x0, y0, x1, y1) in rects:
            crop = full.crop((round(x0 * scale), round(y0 * scale), round(x1 * scale), round(y1 * scale)))
            if kind == "pair":
                for part in split_pair_by_saturation(crop):
                    cards.append(part)
                    kinds.append(kind)
                    effs.append(eff)
                continue
            if kind in ("image", "pixel"):
                crop = refine_by_frame(crop)
            elif kind == "stack":
                crop = shrink_to_body(crop)
            if kind != "aadhaar":
                crop = trim_dark_border(crop)
            cards.append(crop)
            kinds.append(kind)
            effs.append(eff)
        if len(cards) >= 2:
            break

    if not cards:
        log(f"  ! {pdf_path.name}: card detect na thayu (manual check karo)")
        return []
    names = ["front", "back"] if len(cards) == 2 else (["front"] if len(cards) == 1 else [f"card{i + 1}" for i in range(len(cards))])
    results = []
    for name, crop, kind, eff in zip(names, cards, kinds, effs):
        card, note = fit_to_cr80(crop, dpi)
        if enhance:
            card = enhance_card(card, dpi, eff)
        p = out_dir / f"{stem}_{name}.png"
        card.save(p, dpi=(dpi, dpi), compress_level=3)
        if eff is not None and eff < 250:
            log(f"  SOURCE QUALITY WARNING: raster approx {eff:.0f} DPI; resizing cannot restore missing detail")
        results.append(p)
        log(f"  OK {p.name}  [{kind}] {note}")
    return results


def collect_pdfs(inputs):
    files = []
    for s in inputs:
        p = Path(s)
        if p.is_dir():
            files += sorted(p.glob("*.pdf")) + sorted(p.glob("*.PDF"))
        elif p.suffix.lower() == ".pdf":
            files.append(p)
    seen, out = set(), []
    for f in files:
        k = str(f.resolve()).lower()
        if k not in seen:
            seen.add(k)
            out.append(f)
    return out


def run(inputs, out_dir, dpi=DEFAULT_DPI, force_pixel=False, log=print,
        password=None, ask_password=None, enhance=True):
    files = collect_pdfs(inputs)
    if not files:
        log("Koi PDF malyu nathi.")
        return
    out_dir = Path(out_dir)
    ok = 0
    known = [password] if password else []
    for f in files:
        log(f"> {f.name}")
        try:
            if process_pdf(f, out_dir, dpi, force_pixel, log, ask_password, known, enhance):
                ok += 1
        except PasswordCancelled:
            log("  ! password na malvathi aa PDF skip karyu")
        except Exception as e:  # noqa
            log(f"  ! error: {e}")
            log(traceback.format_exc())
    log(f"\nPuru thayu: {ok}/{len(files)} PDF.  Output folder: {out_dir}")


# ----------------------------------------------------------------------------
# GUI
# ----------------------------------------------------------------------------
def launch_gui():
    import tkinter as tk
    from tkinter import filedialog, scrolledtext, simpledialog, ttk, messagebox
    try:
        from tkinterdnd2 import TkinterDnD, DND_FILES
        root = TkinterDnD.Tk()
        dnd_available = True
    except ImportError:
        root = tk.Tk()
        dnd_available = False

    root.title("PVC Card Studio  |  Front + Back")
    root.geometry("860x670")
    root.minsize(720, 560)
    root.configure(bg="#eef2f7")
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure("Primary.TButton", padding=(16, 12), font=("Segoe UI", 11, "bold"), background="#2563eb", foreground="white")
    style.map("Primary.TButton", background=[("active", "#1d4ed8"), ("disabled", "#94a3b8")])
    style.configure("Soft.TButton", padding=(10, 7), font=("Segoe UI", 10))
    style.configure("TEntry", padding=5)
    files = []
    out_var = tk.StringVar(value=str(default_out_dir()))
    dpi_var = tk.StringVar(value=str(DEFAULT_DPI))
    pw_var = tk.StringVar()
    enh_var = tk.BooleanVar(value=False)
    count_var = tk.StringVar(value="No PDFs selected")
    status_var = tk.StringVar(value="Ready. Add PDFs, then click START.")

    outer = tk.Frame(root, bg="#eef2f7", padx=22, pady=18)
    outer.pack(fill="both", expand=True)
    tk.Label(outer, text="PVC CARD STUDIO", bg="#eef2f7", fg="#0f172a", font=("Segoe UI", 21, "bold")).pack(anchor="w")
    tk.Label(outer, text="Drop PDF files  •  Automatic front/back  •  85.6 × 54 mm", bg="#eef2f7", fg="#64748b", font=("Segoe UI", 10)).pack(anchor="w", pady=(0, 14))

    drop = tk.Frame(outer, bg="#ffffff", highlightbackground="#93c5fd", highlightthickness=2, height=120)
    drop.pack(fill="x")
    drop.pack_propagate(False)
    drop_text = tk.Label(drop, text="⬇  DROP PDF FILES HERE" if dnd_available else "SELECT PDF FILES BELOW", bg="white", fg="#1d4ed8", font=("Segoe UI", 15, "bold"))
    drop_text.pack(pady=(25, 4))
    tk.Label(drop, text="Multiple PDFs supported  |  Processing begins only after START", bg="white", fg="#64748b", font=("Segoe UI", 9)).pack()

    def add_paths(paths):
        found = collect_pdfs(paths)
        existing = {str(Path(f).resolve()).lower() for f in files}
        for f in found:
            if str(Path(f).resolve()).lower() not in existing:
                files.append(str(f))
                existing.add(str(Path(f).resolve()).lower())
        refresh_list()

    def refresh_list():
        file_list.delete(0, "end")
        for f in files:
            file_list.insert("end", Path(f).name)
        count_var.set(f"{len(files)} PDF(s) ready")

    def on_drop(event):
        # Tk's splitlist safely handles Windows paths containing spaces and braces.
        add_paths(root.tk.splitlist(event.data))
        return event.action if hasattr(event, "action") else None

    if dnd_available:
        for widget in (drop, drop_text):
            widget.drop_target_register(DND_FILES)
            widget.dnd_bind("<<Drop>>", on_drop)

    controls = tk.Frame(outer, bg="#eef2f7")
    controls.pack(fill="x", pady=(12, 7))
    def pick_files():
        selection = filedialog.askopenfilenames(title="Select PDFs", initialdir=str(default_in_dir()), filetypes=[("PDF files", "*.pdf")])
        if selection: add_paths(selection)
    def pick_folder():
        folder = filedialog.askdirectory(title="Select PDF folder", initialdir=str(default_in_dir()))
        if folder: add_paths([folder])
    def remove_selected():
        for i in reversed(file_list.curselection()): files.pop(i)
        refresh_list()
    for label, cmd in (("+ Add PDFs", pick_files), ("+ Add Folder", pick_folder), ("Remove Selected", remove_selected), ("Clear All", lambda: (files.clear(), refresh_list()))):
        ttk.Button(controls, text=label, command=cmd, style="Soft.TButton").pack(side="left", padx=(0, 7))
    tk.Label(controls, textvariable=count_var, bg="#eef2f7", fg="#334155").pack(side="right")

    file_list = tk.Listbox(outer, height=5, selectmode="extended", bg="white", fg="#0f172a", relief="flat", highlightthickness=1, highlightbackground="#cbd5e1", font=("Segoe UI", 10))
    file_list.pack(fill="x", pady=(0, 12))

    settings = tk.Frame(outer, bg="#eef2f7")
    settings.pack(fill="x", pady=(0, 10))
    tk.Label(settings, text="Save to:", bg="#eef2f7").grid(row=0, column=0, sticky="w")
    ttk.Entry(settings, textvariable=out_var).grid(row=0, column=1, sticky="ew", padx=8)
    def pick_out():
        folder = filedialog.askdirectory(initialdir=str(default_in_dir()))
        if folder: out_var.set(folder)
    def open_out():
        folder = Path(out_var.get())
        folder.mkdir(parents=True, exist_ok=True)
        if hasattr(os, "startfile"): os.startfile(str(folder))
    ttk.Button(settings, text="Browse", command=pick_out).grid(row=0, column=2, padx=3)
    ttk.Button(settings, text="Open Folder", command=open_out).grid(row=0, column=3)
    tk.Label(settings, text="DPI:", bg="#eef2f7").grid(row=1, column=0, sticky="w", pady=10)
    ttk.Combobox(settings, textvariable=dpi_var, values=["300", "450", "600", "900"], state="readonly", width=8).grid(row=1, column=1, sticky="w", padx=8)
    ttk.Checkbutton(settings, text="Optional smooth/sharpen (may affect QR codes)", variable=enh_var).grid(row=1, column=2, columnspan=2, sticky="w")
    tk.Label(settings, text="PDF Password:", bg="#eef2f7").grid(row=2, column=0, sticky="w")
    ttk.Entry(settings, textvariable=pw_var, show="*", width=24).grid(row=2, column=1, sticky="w", padx=8)
    tk.Label(settings, text="Leave empty to be prompted when needed", bg="#eef2f7", fg="#64748b").grid(row=2, column=2, columnspan=2, sticky="w")
    settings.columnconfigure(1, weight=1)

    btn = ttk.Button(outer, text="▶  START  —  CROP FRONT & BACK", style="Primary.TButton")
    btn.pack(fill="x", pady=(2, 8))
    tk.Label(outer, textvariable=status_var, bg="#eef2f7", fg="#475569").pack(anchor="w")
    logbox = scrolledtext.ScrolledText(outer, height=7, bg="#0f172a", fg="#dbeafe", insertbackground="white", font=("Consolas", 9), relief="flat")
    logbox.pack(fill="both", expand=True, pady=(7, 0))

    def log(msg):
        def append():
            logbox.insert("end", str(msg) + "\n")
            logbox.see("end")
        root.after(0, append)

    def ask_password(name, attempt):
        ev, result = threading.Event(), [None]
        def show():
            prompt = "Enter password:" if not attempt else "Wrong password. Try again:"
            result[0] = simpledialog.askstring("PDF Password", f"{name}\n\n{prompt}", show="*", parent=root)
            ev.set()
        root.after(0, show)
        ev.wait()
        return result[0]

    def start():
        if not files:
            messagebox.showinfo("Add PDFs", "Please add one or more PDF files first.")
            return
        btn.configure(state="disabled")
        status_var.set("Processing... please wait")
        selected = list(files)
        destination = out_var.get()
        dpi = int(dpi_var.get())
        password = pw_var.get() or None
        enhance = enh_var.get()
        def work():
            try:
                run(selected, destination, dpi, False, log, password, ask_password, enhance)
            finally:
                root.after(0, lambda: (btn.configure(state="normal"), status_var.set("Finished. You can add more PDFs and click START again.")))
        threading.Thread(target=work, daemon=True).start()
    btn.configure(command=start)
    root.mainloop()

# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="PDF -> PVC card (CR80) front/back")
    ap.add_argument("inputs", nargs="*", help="PDF files ya folder")
    ap.add_argument("-o", "--out", default=str(default_out_dir()))
    ap.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    ap.add_argument("--force-pixel", action="store_true", help="pixel detector vapravo (test)")
    ap.add_argument("-p", "--password", default=None, help="PDF password (jo protected hoy to)")
    ap.add_argument("--no-enhance", action="store_true", help="quality improve band karvu")
    a = ap.parse_args()
    if not a.inputs:
        launch_gui()
    else:
        import getpass

        def ask(name, attempt):
            if not sys.stdin.isatty():
                return None
            try:
                return getpass.getpass(f"{name} - password: ") or None
            except (EOFError, KeyboardInterrupt):
                return None

        run(a.inputs, a.out, a.dpi, a.force_pixel, print, a.password, ask, not a.no_enhance)


if __name__ == "__main__":
    main()
