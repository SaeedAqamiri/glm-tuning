#!/usr/bin/env python3
# Build the Persian RTL DOCX report
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.section import WD_SECTION
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

FONT = 'Vazirmatn'
ACCENT = RGBColor(0x1F, 0x4E, 0x79)
ACCENT2 = RGBColor(0x22, 0x73, 0xB8)
GREEN = RGBColor(0x1F, 0x8A, 0x63)
RED = RGBColor(0xC2, 0x45, 0x5E)
GRAY = RGBColor(0x55, 0x5F, 0x6B)
DARK = RGBColor(0x21, 0x2B, 0x36)

doc = Document()

# ---------- page setup: A4 + margins ----------
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
sec.left_margin = sec.right_margin = Cm(2.0)
sec.top_margin, sec.bottom_margin = Cm(2.0), Cm(1.8)

# ---------- default style ----------
st = doc.styles['Normal']
st.font.name = FONT
st.font.size = Pt(11)
st.font.color.rgb = DARK
rpr = st.element.get_or_add_rPr()
rf = rpr.find(qn('w:rFonts'))
if rf is None:
    rf = OxmlElement('w:rFonts'); rpr.append(rf)
rf.set(qn('w:ascii'), FONT); rf.set(qn('w:hAnsi'), FONT); rf.set(qn('w:cs'), FONT)
szcs = OxmlElement('w:szCs'); szcs.set(qn('w:val'), '22'); rpr.append(szcs)

for hname, hsz, hcolor in [('Heading 1', 16, ACCENT), ('Heading 2', 13, ACCENT2), ('Heading 3', 11.5, GREEN)]:
    hs = doc.styles[hname]
    hs.font.name = FONT; hs.font.size = Pt(hsz); hs.font.bold = True; hs.font.color.rgb = hcolor
    hrpr = hs.element.get_or_add_rPr()
    hrf = hrpr.find(qn('w:rFonts'))
    if hrf is None:
        hrf = OxmlElement('w:rFonts'); hrpr.append(hrf)
    hrf.set(qn('w:ascii'), FONT); hrf.set(qn('w:hAnsi'), FONT); hrf.set(qn('w:cs'), FONT)
    hszcs = OxmlElement('w:szCs'); hszcs.set(qn('w:val'), str(int(hsz*2))); hrpr.append(hszcs)

# ---------- helpers ----------
def cs_run(run, size=None, bold=None, color=None, italic=None):
    rpr = run._r.get_or_add_rPr()
    rtl = OxmlElement('w:rtl'); rtl.set(qn('w:val'), '1'); rpr.append(rtl)
    rf = rpr.find(qn('w:rFonts'))
    if rf is None:
        rf = OxmlElement('w:rFonts'); rpr.append(rf)
    rf.set(qn('w:ascii'), FONT); rf.set(qn('w:hAnsi'), FONT); rf.set(qn('w:cs'), FONT)
    if size:
        run.font.size = Pt(size)
        s = OxmlElement('w:szCs'); s.set(qn('w:val'), str(int(size*2))); rpr.append(s)
    if bold:
        run.font.bold = True
        b = OxmlElement('w:bCs'); b.set(qn('w:val'), '1'); rpr.append(b)
    if italic:
        run.font.italic = True
    if color:
        run.font.color.rgb = color
    return run

def para(text='', size=11, bold=False, color=None, align=None, style=None, space_after=6, space_before=0):
    p = doc.add_paragraph(style=style)
    pPr = p._p.get_or_add_pPr()
    bidi = OxmlElement('w:bidi'); bidi.set(qn('w:val'), '1'); pPr.append(bidi)
    p.alignment = align if align is not None else WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.line_spacing = 1.25
    if text:
        cs_run(p.add_run(text), size=size, bold=bold, color=color)
    return p

def head(text, level=1, page_break=False):
    if page_break:
        doc.add_page_break()
    h = doc.add_heading('', level=level)
    hPr = h._p.get_or_add_pPr()
    bidi = OxmlElement('w:bidi'); bidi.set(qn('w:val'), '1'); hPr.append(bidi)
    h.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    cs_run(h.add_run(text), size={1:16, 2:13, 3:11.5}[level], bold=True,
           color={1:ACCENT, 2:ACCENT2, 3:GREEN}[level])
    return h

def bullet(text, bold_prefix=None, size=10.5):
    p = para('', size=size, space_after=3)
    pPr = p._p.get_or_add_pPr()
    bidi = OxmlElement('w:bidi'); bidi.set(qn('w:val'), '1'); pPr.append(bidi)
    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    pPr2 = p._p.get_or_add_pPr()
    ind = OxmlElement('w:ind'); ind.set(qn('w:right'), '340'); pPr2.append(ind)
    r = p.add_run('•  '); cs_run(r, size=size, color=ACCENT2, bold=True)
    if bold_prefix:
        cs_run(p.add_run(bold_prefix + ' — '), size=size, bold=True)
    cs_run(p.add_run(text), size=size)
    return p

def shade(cell, hexcolor):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd'); shd.set(qn('w:val'), 'clear'); shd.set(qn('w:fill'), hexcolor)
    tcPr.append(shd)

def table(headers, rows, widths=None, fill='1F4E79', fsize=9.5, ltr_cols=None):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = 'Table Grid'
    t.alignment = 1
    tblPr = t._tbl.tblPr
    bv = OxmlElement('w:bidiVisual'); tblPr.append(bv)
    ltr_cols = ltr_cols or []
    for i, h in enumerate(headers):
        c = t.rows[0].cells[i]
        shade(c, fill)
        p = c.paragraphs[0]; p.alignment = 1
        cs_run(p.add_run(h), size=fsize, bold=True, color=RGBColor(0xFF, 0xFF, 0xFF))
    for ri, row in enumerate(rows):
        cells = t.add_row().cells
        for ci, val in enumerate(row):
            c = cells[ci]
            if ri % 2 == 1:
                shade(c, 'EEF3F9')
            p = c.paragraphs[0]
            p.alignment = 1
            if ci in ltr_cols:
                r = p.add_run(str(val)); r.font.size = Pt(fsize); r.font.name = FONT
            else:
                cs_run(p.add_run(str(val)), size=fsize)
    if widths:
        for ci, w in enumerate(widths):
            for row in t.rows:
                row.cells[ci].width = Cm(w)
    return t

def pic(path, caption, width=15.5):
    doc.add_picture(path, width=Cm(width))
    doc.paragraphs[-1].alignment = 1
    cp = para(caption, size=9, color=GRAY, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=10)
    return cp

# ---------- footer with page number (hidden on cover) ----------
sec.different_first_page_header_footer = True
footer = sec.footer
fp = footer.paragraphs[0]; fp.alignment = 1
run = fp.add_run()
fld = OxmlElement('w:fldSimple'); fld.set(qn('w:instr'), 'PAGE')
rEl = OxmlElement('w:r'); tEl = OxmlElement('w:t'); tEl.text = '1'
rEl.append(tEl); fld.append(rEl)
fp._p.append(fld)
fr = fp.add_run('  |  گزارش بهینه‌سازی CPU — GLM-5.3-Flash')
cs_run(fr, size=8.5, color=GRAY)

# ============================================================
# COVER
# ============================================================
for _ in range(5):
    para('', space_after=8)
p = para('گزارش جامع بهینه‌سازی', size=26, bold=True, color=ACCENT, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=4)
para('اجرای CPU مدل GLM-5.3-Flash', size=22, bold=True, color=ACCENT2, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=14)
para('از موتور اولیه تا سرور آزمایشی امروز', size=14, color=DARK, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=4)
para('مسیر اعتبارسنجی‌شدهٔ چهار فاز: AMX و NUMA، عملیات گراف، MTP، و کِرنل‌های Prefill', size=11.5, color=GRAY, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=30)
para('سرور: g11 — HPE DL380 Gen11 | 2× Intel Xeon Platinum 8480+ | 1 TiB RAM | بدون GPU', size=10, color=GRAY, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=2)
para('مدل: GLM-5.3-Flash UD-Q4_K_XL (GGUF شش‌پاره، ۱۸۶ گیابایت)', size=10, color=GRAY, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=2)
para('۲۰ مهر ۱۴۰۵ — 12 October 2026', size=10, color=GRAY, align=WD_ALIGN_PARAGRAPH.CENTER)
doc.add_page_break()

# ---------- TOC (manual) ----------
head('فهرست مطالب', 1)
toc = [
    ('۱.', 'خلاصهٔ مدیریتی'),
    ('۲.', 'سخت‌افزار، مدل و وضعیت اولیه'),
    ('۳.', 'روش‌شناسی اعتبارسنجی'),
    ('۴.', 'مسیر بهینه‌سازی در یک نگاه'),
    ('۵.', 'فاز ۱ — اصلاحات پایه: AMX، دیسپچ دسته‌ای و NUMA'),
    ('۶.', 'فاز ۲ — عملیات گراف: CONCAT و GET_ROWS'),
    ('۷.', 'فاز ۳ — MTP و کِرنل‌های VNNI'),
    ('۸.', 'فاز ۴ — کِرنل‌های Prefill و عبور از ۲۰۰'),
    ('۹.', 'دقت، صحت و بحث کوانتیزاسیون'),
    ('۱۰.', 'زیرساخت فعلی: سرورها، کش پرامپت و کانتکست بزرگ'),
    ('۱۱.', 'ابزار تست: رابط کاربری'),
    ('۱۲.', 'جمع‌بندی و آنچه باقی مانده است'),
    ('۱۳.', 'پیوست — بازتولید، فایل‌ها و سابقهٔ تغییرات'),
]
for num, title in toc:
    p = para('', size=11.5, space_after=4, align=WD_ALIGN_PARAGRAPH.RIGHT)
    cs_run(p.add_run(f'{num}  '), size=11.5, bold=True, color=ACCENT2)
    cs_run(p.add_run(title), size=11.5)

# ============================================================
# 1. Executive summary
# ============================================================
head('۱. خلاصهٔ مدیریتی', 1, page_break=True)
para('این گزارش شرح کامل و مستندِ مسیر بهینه‌سازی اجرای CPU مدل GLM-5.3-Flash روی سرور g11 است: از موتور اولیهٔ llama.cpp با سرعت ۵۷.۶ توکن بر ثانیه در prefill و ۶.۲ توکن بر ثانیه در decode، تا وضعیت فعلی با ۲۰۶.۷ توکن بر ثانیه prefill (اعتبارسنجی‌شده؛ تا ۲۳۶ در اندازه‌گیری زنده) و ۱۷.۷ تا ۱۸.۱ توکن بر ثانیه decode. نکتهٔ محوری این مسیر آن است که هر بهینه‌سازی پیش از پذیرش، با تست‌های عددی بایت‌به‌بایت و مقایسهٔ خروجی کامل مدل روی پرامپت‌های واقعی راستی‌آزمایی شده و در صورت نبودِ سود، صریحاً رد و ثبت شده است.')
para('بهبود کلی در طول مسیر:', bold=True, space_after=4)
table(
    ['سنجه', 'موتور اولیه', 'وضعیت فعلی', 'ضریب بهبود'],
    [
        ['Prefill — پرامپت واقعی انگلیسی (۱۸۶۰ توکن)', '57.6*', '206.7 (زنده: تا 236.3)', '3.6× تا 4.1×'],
        ['Decode — پرامپت واقعی انگلیسی', '6.2*', '17.7 – 18.1', '≈2.9×'],
        ['Prefill گرم با کش پرامپت (نوبت دوم مکالمه)', '—', '0.89s به‌جای 8.36s', '9.4×'],
        ['کانتکست هر اسلات', '8192', '131072', '16×'],
    ],
    widths=[7.2, 3.2, 4.6, 2.7], ltr_cols=[1, 2, 3],
)
para('* اعداد «موتور اولیه» از بنچمارک متن تکراری ۵۱۲/۲۰۴۸ توکنی است که تنها مبنای مقایسهٔ آن دوره است؛ زنجیرهٔ پرامپت واقعی از ردیف‌های فاز ۱ در بخش ۴ آغاز می‌شود.', size=9, color=GRAY)
para('در کنار موتور، یک رابط کاربری فارسی تستِ سرعت ساخته شد که استریم زنده، نمایش تفکر مدل، متریک‌های Prefill/Decode/MTP/کش و حالت حافظهٔ مکالمه را ارائه می‌دهد و تمام سنجه‌های این گزارش را از مرورگر قابل اندازه‌گیری می‌کند.')

# ============================================================
# 2. Hardware / baseline
# ============================================================
head('۲. سخت‌افزار، مدل و وضعیت اولیه', 1)
para('تمام کارها روی سرور g11 (saeed@192.168.20.126) انجام شده است: سرور HPE DL380 Gen11 با دو پردازندهٔ Intel Xeon Platinum 8480+ (مجموعاً ۱۱۲ هستهٔ فیزیکی، با پشتیبانی Intel AMX)، ۱ ترابایت RAM و بدون هیچ GPU. کل محاسبات مدل به‌طور کامل روی CPU و با تراشه‌های ماتریسی AMX انجام می‌شود.')
table(
    ['جزء', 'مشخصات'],
    [
        ['سرور', 'HPE DL380 Gen11 — دو سوکت'],
        ['پردازنده', '2× Intel Xeon Platinum 8480+ — ۱۱۲ هستهٔ فیزیکی، پشتیبانی AMX'],
        ['حافظه', '1 TiB DDR5 (NUMA دوگره‌ای)، بدون GPU'],
        ['مدل', 'GLM-5.3-Flash UD-Q4_K_XL — شش شارد GGUF، حدود ۱۸۶ گیابایت'],
        ['موتور', 'llama.cpp (کپی خصوصی روی سرور) — ساخت Release، بدون OpenMP، GGML_NATIVE + GGML_AMX'],
        ['مدل جایگزین موجود', 'Q8E (۲۸۱ گیابایت) — برای مقایسهٔ دقت کوانتیزاسیون؛ تا امروز بارگذاری نشده'],
    ],
    widths=[4.2, 12.5],
)
head('وضعیت اولیه', 2)
para('در نقطهٔ شروع، موتور اصلی بدون هیچ تغییری این ویژگی‌ها را داشت: اجرای MoE بدون پشتیبانی بافر AMX (سقوط به کِرنل‌های CPU عمومی)، ثردینگ ۳۲/۵۶، بدون بازپک وزن‌ها، بدون جانمایی NUMA و با دیسپچ دسته‌ای معیوب. تنها مبنای عددی آن دوره، بنچمارک متن تکراری انگلیسی بود: ۵۷.۶۰ prefill و ۶.۲۰ decode توکن بر ثانیه. هدف اعلامی اولیه (۵۰ prefill / ۱۰ decode روی متن تکراری) به‌سرعت پاس شد، اما تست روی پرامپت‌های واقعی نشان داد هدف واقعی باید بسیار بالاتر باشد؛ به همین دلیل زنجیرهٔ اصلی گزارش بر «پرامپت‌های واقعی» بنا شد: یک پرامپت انگلیسی ۱۸۶۰ توکنی و یک پرامپت فارسی ۴۳۹۷ توکنی (بازبینی کد واقعی از سورس موتور)، همیشه با temperature=0، seed=42، بدون کش پرامپت و تک‌درخواست.')

# ============================================================
# 3. Methodology
# ============================================================
head('۳. روش‌شناسی اعتبارسنجی', 1, page_break=True)
para('قاعدهٔ حاکم بر تمام مسیر این بود: «هیچ تغییری بدون اثبات صحت و سود واقعی ادغام نمی‌شود.» هر ایده از حلقهٔ زیر عبور می‌کرد:')
pic('d3_loop.png', 'شکل ۱ — حلقهٔ اعتبارسنجی هر بهینه‌سازی', 11.5)
bullet('هر کِرنل در برابر مرجع میزبان (و در مواردی مرجع مستقل بایت‌به‌بایت) تست می‌شد؛ مثلاً ۳۸۴ مورد QKV، ۷۲ مورد MoE، ۱۲۸ مورد GET_ROWS و ۱۹۲ مورد fuse گیت.', 'تست عددی بایت‌دقیق')
bullet('دو پرامپت واقعی ثابت با seed و دمای صفر و بدون کش؛ پذیرش تنها با «خروجی کامل عیناً برابر» چک‌پوینت قبلی مجاز بود.', 'اعتبارسنجی HTTP واقعی')
bullet('هر چک‌پوینت موفق با تگ و باندل Git محفوظ شد (validated-real-8p6 تا validated-real-mtp-18p4) تا همیشه امکان بازگشت باشد.', 'نسخه‌بندی و باندل')
bullet('بهبود میکروبنچمارک به‌تنهایی هرگز ملاک پذیرش نبود؛ چند کِرنل با بنچمارک بهتر، روی درخواست واقعی کندتر بودند و رد شدند.', 'بی‌اعتمادی به میکروبنچمارک')
bullet('هر آزمایش در worktree جداگانه اعمال و به‌صورت patch ذخیره شد؛ پروفایل پیش از هر فاز، مسیر بعدی را تعیین می‌کرد.', 'ایزوله‌سازی و پروفایل')

# ============================================================
# 4. Journey overview
# ============================================================
head('۴. مسیر بهینه‌سازی در یک نگاه', 1, page_break=True)
pic('d4_timeline.png', 'شکل ۲ — چهار فاز مسیر با نرخ‌های اعتبارسنجی‌شده (توکن بر ثانیه، پرامپت واقعی انگلیسی)', 5.2)
para('جدول زیر زنجیرهٔ کامل چک‌پوینت‌های اعتبارسنجی‌شده روی پرامپت واقعی است. همهٔ اعداد توکن بر ثانیه، تک‌درخواست، بدون کش و با خروجی کامل عیناً برابر چک‌پوینت قبلی‌اند:')
table(
    ['چک‌پوینت / تگ', 'تغییر کلیدی', 'EN: Prefill', 'EN: Decode', 'FA: Prefill', 'FA: Decode'],
    [
        ['MTP7، p_min=0.7', 'وضعیت پیش از اصلاحات (MTP قدیمی)', '64.67', '5.32', '64.20', '5.61'],
        ['— (پیش از NUMA)', 'خاموش‌کردن MTP قدیمی', '69.46', '7.26', '68.18', '7.45'],
        ['validated-real-8p6', 'اصلاح NUMA dense (فاز ۱)', '70.55', '8.43', '68.66', '8.59'],
        ['validated-real-9p4 (c9b8ee9)', 'مسیر سریع CONCAT (فاز ۲)', '93.83', '9.26', '89.03', '9.39'],
        ['validated-real-13p3 (d079289)', 'تکه‌تکه‌کردن GET_ROWS (فاز ۲)', '91.18', '13.35', '86.47', '13.10'],
        ['prefill-100 (f4e9512)', 'MTP پایدار + VNNI دوم‌سطری (فاز ۳)', '105.58', '16.37', '103.75', '16.23'],
        ['validated-real-mtp-18p4 (1a1d1ab)', 'تکمیل فاز ۳ — بیس فاز ۴', '154.54', '18.09', '147.25', '17.82'],
        ['prefill-final', 'کِرنل‌های Q8 signed و Q4/Q5 scaled-K (فاز ۴)', '206.66', '17.68', '206.05', '17.51'],
        ['اجرای زندهٔ امروز (8017)', 'همان پیکربندی، سرور خلوت', '221.7 – 236.3', '18.1 – 18.3', '—', '—'],
    ],
    widths=[3.9, 5.3, 1.9, 1.9, 1.9, 1.9], fsize=8.5, ltr_cols=[2, 3, 4, 5],
)
para('نمودار زیر همان زنجیره را نشان می‌دهد — میله‌ها prefill و خط پُررنگ‌تر decode است:')
pic('d1_chart.png', 'شکل ۳ — پیشرفت prefill و decode در طول مسیر', 15.5)
para('دو نکتهٔ صداقت علمی: نخست اینکه فاصلهٔ چک‌پوینت‌های mtp-16p3 و mtp-18p4 (پرش decode از ۱۶.۴ به ۱۸.۴) جزئیات مستند در فایل‌های محلی ندارد و تنها باندل آن محفوظ است. دوم اینکه اعداد مطلق بین روزها تا ~۱۰٪ (و در یک مورد بیشتر) نوسان می‌کنند؛ به همین دلیل مقایسه‌های معتبر فقط «هم‌جلسه» تلقی شده‌اند (بخش ۸).')

# ============================================================
# 5. Phase 1
# ============================================================
head('۵. فاز ۱ — اصلاحات پایه: AMX، دیسپچ دسته‌ای و NUMA', 1, page_break=True)
head('راه‌اندازی AMX و رفع دیسپچ', 2)
para('نخستین گام، اثبات سلامت سخت‌افزار بود: یک تست مستقل INT8 مجوز XTILEDATA را گرفت، پیکربندی تایل را ساخت و هر ۲۵۶ خروجی را در برابر مقدار مرجع ۳۸۴ راستی‌آزمایی کرد — نتیجه: AMX سالم است و نیازی به تغییر BIOS نیست. سپس پشتیبانی بافر AMX برای ضرب MoE اضافه شد، تخصیص ماتریس‌های expert/batched اصلاح شد و باگ‌های بایت/المان و استرید اکتیویشن در مسیر batched برطرف گردید. عملیاتِ وزن‌های پک‌شدهٔ پشتیبانی‌نشده به‌جای سقوط به کِرنل خام، صریحاً رد می‌شود. دیسپچ FP16 روی AMX به‌دلیل عبور از حد تحمل خطا غیرفعال ماند.')
head('جانمایی NUMA و ثردینگ', 2)
para('پروفایل decode روی درخواست واقعی انگلیسی نشان داد ۵۹٪ نمونه‌های چرخهٔ CPU در نوارباری (barrier) و ۱۹.۷٪ در کِرنل برداری Q8_0 است — یعنی گلوگاه اصلی، دسترسی‌های دور در سیستم دو سوکتی است. راه‌حل: نیمه‌های ردیف خروجی هر expert و ماتریس‌های dense/batched به گرهٔ NUMA همان سوکت متصل شد و هر worker به هستهٔ فیزیکی پین شد؛ نوارباری دوگروهی با همگام‌سازی acquire/release و فعال‌سازی FTZ/DAZ برای نخ‌های worker. نتیجه: decode واقعی ۱۵–۱۶٪ بهبود یافت (۷.۲۶ → ۸.۴۳ انگلیسی) با خروجی عیناً برابر.')
table(
    ['پیکربندی آن دوره (متن تکراری)', 'ورودی', 'Prefill', 'Decode'],
    [
        ['موتور اولیه بدون بازپک — t32/tb56', '512', '57.60', '6.20'],
        ['همان — ورودی بزرگ‌تر', '2048', '56.84', '6.10'],
        ['AMX MoE تک‌سوکت — t24/tb56، ub2048', '512', '65.50', '7.75'],
        ['AMX دسته‌ای + ردیف‌های محلی expert + MTP7/p0', '2048', '67.14', '11.43'],
    ],
    widths=[8.5, 2.2, 2.7, 2.7], ltr_cols=[1, 2, 3],
)
bullet('کِرنل MoE عریض‌تر: تست عددی را پاس کرد اما روی درخواست واقعی کندتر بود (۷.۹۲) — رد شد.', 'ردشدهٔ این فاز:')
bullet('تقسیم جمع صحیح Q8 به زنجیره‌های مستقل: تست پاس، عملکرد واقعی بدتر — رد شد.')
bullet('یک آزمایش دیسپچ ناایمنِ ابتدایی متن خراب تولید کرد و فوراً کنار گذاشته شد.')

# ============================================================
# 6. Phase 2
# ============================================================
head('۶. فاز ۲ — عملیات گراف: CONCAT و GET_ROWS', 1)
head('CONCAT موازی', 2)
para('پروفایل زمان دیوار decode نشان داد سهم بزرگی از زمان در CONCAT و GET_ROWS است. مسیر سریع F32/I32 برای CONCAT ردیف‌های به‌هم‌پیوسته نوشته شد: تقسیم تخت ردیف‌ها بین workerها و کپی سگمنت‌ها. هر ۳۲ تست صحت (ابعاد ۰ تا ۳، ۱/۳/۸/۴۸ ثرد، ورودی پیوسته و جابه‌جاشده) پاس شد. نتیجهٔ واقعی: انگلیسی از ۷۰.۶ به ۹۳.۸ prefill و از ۸.۴ به ۹.۳ decode (تگ validated-real-9p4، کامیت c9b8ee9) — پس از بازگشت شبکهٔ سرور که مستندسازی آن در CHECKPOINT.md آمده است.')
head('GET_ROWS تکه‌تکه‌شده — جهش decode به ۱۳+', 2)
para('پروفایل‌بندی نشان داد هر توکن decode حدود ۳۲.۴ میلی‌ثانیه فقط در GET_ROWS صرف می‌کند (۳۱.۸ میلی‌ثانیه آن محاسبهٔ واقعی نخ ۰، نه انتظار نوارباری) — صفحات حالت بازگشتی با ردیف‌های تا ۱,۰۴۸,۵۷۶ مقدار. مسیر GGML_ROWS_CHUNKS بلوک‌های ستونی ۱۰۲۴تایی پیوسته را بین workerهای گراف پخش می‌کند. هر ۱۲۸ تست مرجع مستقل بایت‌دقیق پاس شد (ردیف‌های میلیون‌عضوی، F32/I32، ثردینگ‌های مختلف، تانسورهای جابه‌جاشده). نتیجه: decode انگلیسی ۹.۳ → ۱۳.۳۵ و فارسی به ۱۳.۱۰ — عبور از هدف ۱۰ توکن بر ثانیه روی پرامپت واقعی (تگ validated-real-13p3، کامیت d079289).')
bullet(' fuse SCALE→SIGMOID→SCALE با ۱۹۲ تست بایت‌دقیق پاس شد اما روی پرامپت واقعی سودی نشان نداد (۱۳.۲۷/۱۳.۰۶) و فقط به‌صورت opt-in با FUSE_GATE=1 باقی ماند.', 'آزمایش مکمل:')

# ============================================================
# 7. Phase 3
# ============================================================
head('۷. فاز ۳ — MTP و کِرنل‌های VNNI', 1, page_break=True)
para('پس از عبور decode از ۱۳، هدف بعدی ۲۰ توکن بر ثانیه شد و مسیر اصلی، کدگذاری پیش‌گویانهٔ MTP (سر حد مدل با هد NextN) و کِرنل‌های فشرده‌تر شد. نکتهٔ مهم این فاز، جداسازی «پیکربندی» از «کِرنل» بود: همان MTP با پیکربندی اشتباه (cap=1، p_min=0، استخر batched 96 برای وارسی) فقط ۶.۸ توکن بر ثانیه می‌داد و رد شد؛ اما با استخرهای پایدار اختصاصی، وارسی موازی و زنجیره‌های کوچک به ۱۵.۱ رسید.')
table(
    ['مرحله', 'تغییر', 'EN Prefill', 'EN Decode', 'پذیرش MTP'],
    [
        ['مبنا (بدون MTP)', 'چک‌پوینت 13p3', '91.18', '13.35', '—'],
        ['MTP پیکربندی ردشده', 'cap1، p0، استخر batched96', '98.43', '6.84', '76–77٪'],
        ['MTP استخر پایدار', 'استخرهای اختصاصی + CPY موازی نواری + زنجیرهٔ کوچک', '105.27', '15.09', '158/208'],
        ['VNNI دوم‌سطری', 'اشتراک بار وزن بین دو ردیف اکتیویشن + آستانهٔ 8192', '105.58', '16.37', '158/208'],
        ['چک‌پوینت‌های بعدی', 'mtp-16p3 و سپس mtp-18p4 (1a1d1ab)', '154.54*', '18.09', '—'],
    ],
    widths=[3.6, 6.2, 2.1, 2.1, 2.2], fsize=8.5, ltr_cols=[2, 3, 4],
)
para('* عدد ۱۵۴.۵ «بیس‌لاین تازه» روی همان کد mtp-18p4 با کِرنل‌های prefill خاموش است که مبنای سنجش فاز ۴ شد.', size=9, color=GRAY)
bullet('fuse جفت QKV و جفت gate/up با فعال‌ساز مشترک (۹۶ تست کوچک و ۲۴ تست بزرگ بایت‌دقیق).', 'تغییرات مکمل این فاز:')
bullet('چیدمان interleave برای K و Q8 و مسیر small-batch تا ۸ توکن روی استخر decode.')
bullet('VNNI تک‌ردیفی برای decode عادی کندتر بود و رد شد؛ نسخهٔ دوم‌سطری ماندگار شد.')
bullet('از ۱۶.۴ تا ۱۸.۴: جزئیات این جهش فقط در باندل mtp-18p4 محفوظ است و در اسناد محلی شرح داده نشده است.', 'خلأ مستندسازی:')

# ============================================================
# 8. Phase 4
# ============================================================
head('۸. فاز ۴ — کِرنل‌های Prefill و عبور از ۲۰۰', 1, page_break=True)
para('در این فاز فقط یک فایل تغییر کرد (mmq.cpp در worktree جداگانهٔ prefill-400) و همهٔ تغییرات به بلاک‌های ماتریسی با بیش از دو سطر محدود شد تا مسیر decode دست‌نخورده بماند. دو ایدهٔ اصلی حفظ شدند:')
bullet('وزن‌های Q8 پک‌شدهٔ unsigned یک‌بار به «اسکرچ signed» تبدیل می‌شوند و جبرانِ تکراری هر سطر از حلقهٔ درونی حذف می‌شود؛ مانع‌های کامپایلری خواندن اسکرچ را برای tile intrinsics صریح می‌کنند.', '۱) GGML_Q8_PREFILL_SIGNED:')
bullet('مقیاس‌های صحیح روی بایت‌پلن‌های کم/زیاد اعمال می‌شود، بلاک کامل ۲۵۶ عضوی با AMX جمع می‌شود و سپس ترتیب اصلی جمع اعشاری و min-correction حفظ می‌گردد — در نتیجه خروجی دقیقاً با مسیر مرجع یکی است.', '۲) GGML_PREFILL_SCALED_K:')
para('همراه این دو: batch/ubatch=4096 و ۱۱۲ ثرد prefill (سوییپ‌ها نشان دادند ۵۶ و ۲۲۴ ثرد و batchهای ۱۰۲۴/۲۰۴۸ ضعیف‌ترند). صحت: ۳۸۴ تست QKV/raw-extreme، ۷۲ تست MoE و ۱۲ تست MoE بزرگ، همه بایت‌به‌بایت. نتیجه روی پرامپت واقعی: انگلیسی ۱۵۴.۵ → ۲۰۶.۷ (+۳۳.۷٪) و فارسی ۱۴۷.۳ → ۲۰۶.۱ (+۳۹.۹٪) با decode تقریباً دست‌نخورده (‎-۱.۷٪ تا -۲.۳٪). هدف ۴۰۰ محقق نشد و صادقانه در اسناد ثبت شده است.')
head('ablation زندهٔ فلگ‌های کِرنل (همین هفته)', 2)
para('برای پاسخ به «سهم هر فلگ چقدر است»، هر چهار ترکیب پشت‌سرهم در یک جلسه اندازه‌گیری شد (ری‌استارت کامل سرور برای هر ترکیب، دو ران در هر ترکیب، ران دوم ملاک به‌دلیل گرم‌شدن ~۸٪):')
table(
    ['ترکیب فلگ‌ها', 'Prefill توکن/ثانیه (ران گرم)', 'اختلاف'],
    [
        ['هر دو خاموش (کِرنل مرجع)', '214.8', '—'],
        ['فقط Q8 signed', '226.5', '‎+5.4%'],
        ['فقط Q4/Q5 scaled-K', '224.1', '‎+4.3%'],
        ['هر دو (پیکربندی validated)', '236.3', '‎+9.9%'],
    ],
    widths=[7.5, 5.0, 3.0], ltr_cols=[1, 2],
)
para('دو درس مهم: سهم دو فلگ جمع‌شدنی و در حد نویز (~۲ توکن) جداشدنی است؛ و اعداد مطلق بین جلسات نوسان بزرگی دارند — بیسِ بی‌کِرنل در همان جلسه ۲۱۴.۸ گرفت در حالی که در جلسهٔ ثبت اصلی ۱۵۴.۵ بود. بنابراین مقایسهٔ معتبر فقط هم‌جلسه است و این هشدار در PREFILL-RESULTS.md ثبت شده است.')

# ============================================================
# 9. Accuracy
# ============================================================
head('۹. دقت، صحت و بحث کوانتیزاسیون', 1, page_break=True)
para('پرسش طبیعی: آیا Q4 بودنِ وزن‌ها دقت را می‌اندازد و آیا کِرنل‌های تازه چیزی را خراب کردند؟ پاسخ دقیق دو لایه دارد:')
bullet('تمام کِرنل‌های تازه از نظر ریاضی همان مسیر مرجع را محاسبه می‌کنند (ترتیب جمع و تصحیح‌ها حفظ شده) و خروجی بایت‌به‌بایت با مرجع یکی است. مجموع تست‌های صحت در طول مسیر: ۳۸۴+۲۴۰ QKV، ۷۲+۱۲+۲۴×۲ MoE، ۱۲۸ GET_ROWS، ۳۲ CONCAT، ۱۹۲ fuse گیت، ۱۹۲ زنجیرهٔ کوچک، ۸۰ batched و ۴۶۸ مورد فاز prefill.', 'کِرنل‌ها صفر تغییر دقت:')
bullet('مدل GLM-5.3-Flash از ابتدا با کوانت UD-Q4_K_XL (~۴.۵ بیت به‌ازای وزن) به کار گرفته شده است. این انتخاب lossy ذاتی خودش را دارد اما در این پروژه هرگز در برابر Q8 اندازه‌گیری نشده است؛ آنچه ثابت شد فقط «تغییرنکردنِ خروجی پس از بهینه‌سازی» است، نه «برابری Q4 با Q8». روی سرور نسخهٔ Q8E (۲۸۱ گیابایت) موجود است و می‌تواند مبنای یک مطالعهٔ مقایسه‌ای واقعی باشد.', 'کوانتیزاسیون مدل:')
bullet('برای هر تغییر، متن کامل پاسخ دو پرامپت واقعی با چک‌پوینت قبلی مقایسه می‌شد و در صورت کوچک‌ترین تفاوت، تغییر پذیرفته نمی‌شد؛ فیلد exact_complete_output در فایل‌های نتیجه این را ثبت می‌کند.', 'شاهد عملی:')

# ============================================================
# 10. Infrastructure
# ============================================================
head('۱۰. زیرساخت فعلی: سرورها، کش پرامپت و کانتکست بزرگ', 1)
pic('d2_arch.png', 'شکل ۴ — معماری استقرار فعلی روی g11', 9.5)
table(
    ['ویژگی', 'سرور پایدار :8016', 'سرور آزمایشی :8017'],
    [
        ['چک‌پوینت کد', 'validated-real-13p3 (d079289)', 'prefill-final (بیس 1a1d1ab + patch)'],
        ['Prefill / Decode واقعی', '≈91 / ≈13.4', '≈206 (زنده تا ۲۳۶) / ≈17.7–18.3'],
        ['کانتکست', '32768 (۴ اسلات × ۸۱۹۲)', '524288 (۴ اسلات × ۱۳۱۰۷۲)'],
        ['ثرد decode / prefill', '48 / 56', '64 / 112'],
        ['کش پرامپت', 'خاموش', 'روشن (--no-cache-prompt حذف شد)'],
        ['کِرنل‌های prefill', '—', 'Q8 signed + Q4/Q5 scaled-K'],
        ['MTP', 'خاموش', 'draft-mtp، cap=1، p_min=0'],
    ],
    widths=[4.2, 5.6, 6.9], fsize=9,
)
head('کش پرامپت — اندازه‌گیری واقعی', 2)
table(
    ['نوبت', 'توکن پرامپت', 'کش‌شده', 'زمان prefill', 'توضیح'],
    [
        ['نوبت ۱ (سرد)', '1854', '0', '8.36s', 'prefill کامل — ۲۲۱.۷ توکن/ثانیه'],
        ['نوبت ۲ (گرم)', '1868', '1850 (۹۹٪)', '0.89s', 'فقط ۱۸ توکن جدید پردازش شد'],
    ],
    widths=[3.2, 2.8, 2.8, 2.9, 5.0], ltr_cols=[1, 2, 3],
)
para('یعنی ادامهٔ مکالمه ۹.۴ برابر سریع‌تر پاسخ می‌دهد و نرخ decode دست‌نخورده می‌ماند. توجه: در حالت کش، متریک «توکن بر ثانیه» prefill معنا نمی‌دهد (۱۸ توکنِ باقیمانده همگی سربارند)؛ سنجهٔ درست، صرفه‌جویی زمان جدول بالا است. افزایش کانتکست به ۵۱۲K هیچ رشد معناداری در مصرف RAM نداد (ساختار hybrid GDN مدل، کش حالت فشرده‌ای دارد) و پس از تخصیص واقعی هم ثابت ماند.')
head('فلگ‌های محیطی فعال سرور 8017', 2)
para('run-prefill-400.sh (قابل بازتولید، در پیوست):', size=10, color=GRAY)
table(
    ['گروه', 'فلگ‌ها'],
    [
        ['AMX / NUMA', 'GGML_AMX_MOE، GGML_AMX_NUMA_ROWS، GGML_AMX_NUMA_DENSE، GGML_NUMA_PIN، GGML_CPU_FTZ، GGML_TREE_BARRIER'],
        ['گراف', 'GGML_CONCAT_CHUNKS، GGML_ROWS_CHUNKS، GGML_COPY_ROWS'],
        ['کِرنل‌ها', 'GGML_VNNI_ROWS2، GGML_K_INTERLEAVE، GGML_Q8_INTERLEAVE، GGML_FUSE_QKV، GGML_FUSE_MOE_PAIR، GGML_Q8_PREFILL_SIGNED، GGML_PREFILL_SCALED_K'],
        ['MTP / مسیر کوچک', 'GGML_MTP_SHARED_POOL، GGML_CPU_SMALL_CHAIN(+MAX=8192)، GGML_SMALL_BATCH_MAX=8، LLAMA_MTP_ADAPTIVE=0'],
        ['کانتکست و کش', 'CTX=524288 (پیش‌فرض 32768)، کش فعال (NO_CACHE_PROMPT=1 برای خاموش‌کردن)'],
    ],
    widths=[3.4, 13.3], fsize=9,
)

# ============================================================
# 11. UI
# ============================================================
head('۱۱. ابزار تست: رابط کاربری', 1, page_break=True)
para('برای اینکه همهٔ این سنجه‌ها بدون اسکریپت، از مرورگر و قابل استفاده باشند، یک رابط فارسی RTL (GLM CPU Lab) ساخته و روی سرور مستقر شد (پورت 8018، پروکسی به 8017). قابلیت‌ها پس از تکمیل در این دوره:')
bullet('استریم زندهٔ SSE توکن‌به‌توکن با بج tok/s لحظه‌ای، بدون بافر شدن در پروکسی.')
bullet('نمایش «فرایند تفکر مدل» (reasoning_content) در بخش جمع‌شونده — پیش‌تر پاسخ‌های فقط‌استدلالی خالی نمایش داده می‌شدند.')
bullet('کارت‌های متریک: Prefill، Decode، توکن ورودی/خروجی، نرخ پذیرش MTP (draft_n_accepted/draft_n) و توکن کش‌شده.')
bullet('حالت «حافظهٔ گفتگو»: ارسال تاریخچهٔ مکالمه با cache_prompt=true برای سنجش سرعت مکالمهٔ پیوسته + دکمهٔ «گفتگوی جدید».')
bullet('حداکثر خروجی تا ۱۰۰,۰۰۰ توکن برای تست بلندمدت حالت استدلال (هماهنگ با کانتکست ۱۳۱K هر اسلات).')
bullet('تاریخچهٔ ۵۰ اجرای آخر در مرورگر + نمودار روند + خروجی JSON + توقف با حفظ خروجی ناقص.')
para('خود UI هم یک لایهٔ اندازه‌گیری است: اعداد جدول کش در بخش ۱۰ مستقیماً از همین مسیر (پروکسی 8018) گرفته شد و استریم آن به‌صورت تدریجی راستی‌آزمایی شد (۴۷ چانک در ۳.۷ ثانیه، بدون جهش بافری).')

# ============================================================
# 12. Summary
# ============================================================
head('۱۲. جمع‌بندی و آنچه باقی مانده است', 1)
table(
    ['هدف', 'وضعیت', 'بهترین عدد'],
    [
        ['۵۰ prefill / ۱۰ decode (متن تکراری)', 'انجام شد — فاز ۱', '67.1 / 11.4'],
        ['۱۰ decode روی پرامپت واقعی', 'انجام شد — فاز ۲', '13.35'],
        ['۱۰۰ prefill / ۲۰ decode', 'prefill انجام شد؛ decode نرسید', '105.6 / 18.09'],
        ['۴۰۰ prefill (آرزوی فاز ۴)', 'نرسید', '206.7 اعتبارسنجی‌شده / ۲۳۶.۳ زنده'],
        ['۲۰ decode (هدف قدیمی)', 'نرسید', '18.09 (mtp-18p4)'],
    ],
    widths=[6.5, 5.2, 5.0],
)
para('گام‌های منطقی بعدی بر اساس پروفایل فعلی: پروفایل‌گیری از کِرنل‌های نهایی prefill (پروفایل قبلی dense/expert matmul و سپس flash attention را گلوگاه نشان می‌داد)، آزمایش مقایسهٔ دقت با Q8E، و اگر هدف ۲۰ decode جدی ماند، کار روی نرخ پذیرش MTP (الان ~۷۶–۸۳٪ با cap=1).')

# ============================================================
# 13. Appendix
# ============================================================
head('۱۳. پیوست — بازتولید، فایل‌ها و سابقهٔ تغییرات', 1, page_break=True)
head('اجرای سرورها', 2)
para('سرور آزمایشی 8017:', bold=True, space_after=2)
p = para('', space_after=6, align=WD_ALIGN_PARAGRAPH.LEFT)
r = p.add_run('cd /home/saeed/glm-tuning-20261011 && CTX=524288 setsid nohup ./run-prefill-validated.sh > log 2>&1 &')
r.font.name = 'DejaVu Sans Mono'; r.font.size = Pt(9)
para('سرور پایدار 8016:', bold=True, space_after=2)
p = para('', space_after=6, align=WD_ALIGN_PARAGRAPH.LEFT)
r = p.add_run('bash /home/saeed/glm-tuning-20261011/run-server.sh   (از run-prefill-400.sh با NO_CACHE_PROMPT=1 برای بنچمارک کانونیکال بدون کش)')
r.font.name = 'DejaVu Sans Mono'; r.font.size = Pt(9)
head('مستندات و شواهد در ورک‌اسپیس محلی', 2)
table(
    ['فایل', 'محتوا'],
    [
        ['RESULTS.md', 'روایت کامل فازهای ۱ و ۲ با جدول‌ها و تست‌ها'],
        ['CHECKPOINT.md', 'چک‌پوینت‌های validated، تگ‌ها و رویدادهای شبکه'],
        ['EXPERIMENT-100-20.md', 'دفتر آزمایش‌های فاز ۳ (پذیرفته‌ها و ردشده‌ها با اعداد)'],
        ['PREFILL-400-STATE.md و prefill-final/PREFILL-RESULTS.md', 'وضعیت نهایی فاز ۴ + ablation زنده'],
        ['prefill-final/prefill-ablation-20261012.json', 'دادهٔ خام ablation چهار ترکیبی'],
        ['validated-*.bundle (۸ فایل، ~۳۶MB هرکدام)', 'باندل‌های Git همهٔ چک‌پوینت‌های validated'],
        ['cpu-tuning.patch، patch_moe.py، prefill-final/prefill-final.patch', 'پچ‌های سورس نسبت به مبنا'],
        ['test_*.cpp (amx، concat، rows، gate، prefill)', 'سورس تست‌های عددی بایت‌دقیق'],
        ['real-*.jsonl و prefill-final/real-*.jsonl', 'نتایج خام پرامپت واقعی هر مرحله'],
        ['test-ui/ (server.py + dist/)', 'سورس رابط کاربری — مستقر روی g11 در ~/glm-test-ui'],
    ],
    widths=[6.8, 9.9], fsize=9,
)
head('سابقهٔ گیت این مخزن محلی', 2)
table(
    ['کامیت', 'موضوع'],
    [
        ['512e5a8', 'ایجاد مخزن: ورک‌اسپیس tuning، بنچمارک‌ها و UI'],
        ['703844c', 'تکمیل UI: استریم، تفکر مدل، متریک MTP'],
        ['88c9c5c', 'ablation زندهٔ فلگ‌های prefill'],
        ['4c038df', 'کش پرامپت روی سرور آزمایشی + حالت حافظهٔ UI'],
        ['2a7eb8a', 'خروجی بلند (۱۰۰K) و کانتکست ۵۱۲K'],
    ],
    widths=[3.0, 13.7], ltr_cols=[0],
)
para('منبع دیاگرام‌ها: نسخهٔ HTML هر چهار شکل mermaid در پوشهٔ report/mermaid-src/ نگهداری می‌شود و با headless Chrome رندر شده‌اند.', size=9, color=GRAY)

doc.save('/home/arman/alef/glm-tuning/report/GLM-CPU-Tuning-Report-FA.docx')
print('DOCX saved')
