"""Generate editable PPTX and a matching PDF from slides/content.json.

Requires python-pptx and reportlab. Pass --font/--bold-font for a Chinese
TrueType font outside Windows. Fonts are embedded as subsets in the PDF.
"""
import argparse
import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.util import Pt
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
COLORS = dict(navy='11243A', blue='1D638B', teal='007F78', orange='C66A32',
              ink='203349', muted='5A6B7D', pale='EFF4F8')

class Renderer:
    def __init__(self, pdf, slide, audit):
        self.pdf, self.slide, self.audit = pdf, slide, audit

    def rect(self, x, y, w, h, color):
        shape = self.slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Pt(x), Pt(y), Pt(w), Pt(h))
        shape.fill.solid()
        shape.fill.fore_color.rgb = RGBColor.from_string(color)
        shape.line.fill.background()
        self.pdf.setFillColor(HexColor('#' + color))
        self.pdf.rect(x, 540-y-h, w, h, fill=1, stroke=0)

    def text(self, x, y, w, h, text, size=18, color='203349', bold=False):
        font = 'ChineseBold' if bold else 'Chinese'
        lines = []
        for paragraph in text.split('\n'):
            line = ''
            for char in paragraph:
                if line and pdfmetrics.stringWidth(line+char, font, size) > w:
                    lines.append(line.rstrip())
                    line = char.lstrip()
                else:
                    line += char
            lines.append(line.rstrip())
        line_height = size * 1.35
        needed = size * 1.2 + line_height * (len(lines)-1)
        if needed > h + 1:
            self.audit.append(dict(slide=self.slide.slide_id, text=text, height=h, required_height=needed))
        shape = self.slide.shapes.add_textbox(Pt(x),Pt(y),Pt(w),Pt(h))
        frame = shape.text_frame
        frame.word_wrap = False
        frame.margin_left = frame.margin_right = frame.margin_top = frame.margin_bottom = 0
        self.pdf.setFillColor(HexColor('#'+color))
        self.pdf.setFont(font,size)
        for i,line in enumerate(lines):
            paragraph = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
            paragraph.text = line
            paragraph.line_spacing = Pt(line_height)
            paragraph.space_before = paragraph.space_after = 0
            paragraph.font.name = 'Microsoft YaHei'
            paragraph.font.size = Pt(size)
            paragraph.font.bold = bold
            paragraph.font.color.rgb = RGBColor.from_string(color)
            self.pdf.drawString(x, 540-y-size*1.08-i*line_height, line)

    def card(self,x,y,w,h,heading,body,accent='1D638B'):
        self.rect(x,y,w,h,COLORS['pale'])
        self.rect(x,y,4,h,accent)
        self.text(x+18,y+16,w-36,32,heading,21,accent,True)
        self.text(x+18,y+61,w-36,h-70,body,18)

    def table(self,x,y,widths,rows,row_height=46,size=17):
        for r,row in enumerate(rows):
            cx=x
            for c,w in enumerate(widths):
                fill=COLORS['navy'] if r==0 else ('EFF4F8' if r%2==0 else 'F8FAFC')
                self.rect(cx,y+r*row_height,w-2,row_height-2,fill)
                self.text(cx+12,y+r*row_height+11,w-25,row_height-15,str(row[c]),size,
                          'FFFFFF' if r==0 else COLORS['ink'],r==0)
                cx+=w

    def picture(self,rel,x,y,w,h):
        path=ROOT/rel
        with Image.open(path) as image:
            iw,ih=image.size
        ratio=min(w/iw,h/ih)
        pw,ph=iw*ratio,ih*ratio
        px,py=x+(w-pw)/2,y+(h-ph)/2
        self.slide.shapes.add_picture(str(path),Pt(px),Pt(py),width=Pt(pw),height=Pt(ph))
        self.pdf.drawImage(str(path),px,540-py-ph,width=pw,height=ph)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--font',type=Path,default=Path('C:/Windows/Fonts/msyh.ttc'))
    parser.add_argument('--bold-font',type=Path,default=Path('C:/Windows/Fonts/msyhbd.ttc'))
    args=parser.parse_args()
    pdfmetrics.registerFont(TTFont('Chinese',str(args.font),subfontIndex=0))
    pdfmetrics.registerFont(TTFont('ChineseBold',str(args.bold_font),subfontIndex=0))
    items=json.loads((ROOT/'slides/content.json').read_text(encoding='utf-8'))
    ppt=Presentation()
    ppt.slide_width,ppt.slide_height=Pt(960),Pt(540)
    ppt.core_properties.title='STEER: literature review and MATH500 reproduction'
    ppt.core_properties.author='叶顺'
    ppt.core_properties.subject='核对版；论文与个人复现分别列示；2026-11-11'
    output=ROOT/'slides/STEER_literature_and_reproduction'
    pdf=canvas.Canvas(str(output.with_suffix('.pdf')),pagesize=(960,540))
    pdf.setTitle('STEER 文献解读与 MATH500 复现')
    pdf.setAuthor('叶顺')
    audit=[]
    for index,item in enumerate(items,1):
        slide=ppt.slides.add_slide(ppt.slide_layouts[6])
        r=Renderer(pdf,slide,audit)
        r.rect(0,0,960,7,COLORS['teal'])
        r.text(42,21,790,20,item['section'],10,COLORS['teal'],True)
        r.text(42,55,880,55,item['title'],31,COLORS['navy'],True)
        r.rect(42,491,876,1,'DCE5EC')
        r.text(42,503,825,27,item['source'],9,COLORS['muted'])
        r.text(884,502,40,22,f'{index:02}',11,COLORS['muted'])
        for op in item['operations']:
            getattr(r,op['kind'])(*op['args'])
        pdf.showPage()
    if audit:
        raise ValueError('Text overflow: '+json.dumps(audit,ensure_ascii=False))
    pdf.save()
    ppt.save(str(output.with_suffix('.pptx')))
    print(f'Created {len(items)} slides: editable PPTX and matching PDF; no text overflow.')

if __name__=='__main__':
    main()
