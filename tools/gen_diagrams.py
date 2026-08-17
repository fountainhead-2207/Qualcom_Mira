#!/usr/bin/env python3
"""Sinh sơ đồ kiến trúc SVG cho README.

Viết ra file thay vì dùng mermaid vì mermaid tự sắp bố cục: hai pipeline bị nó
xếp chồng lên nhau và người đọc không lần ra hộp nào thuộc bên nào. Ở đây toạ
độ do mình đặt.

Mỗi sơ đồ xuất hai bản, light và dark, để README dùng <picture> cho GitHub tự
đổi theo theme của người xem — SVG một bản luôn sai màu chữ ở một trong hai nền.

    python3 tools/gen_diagrams.py      # ghi vào docs/img/
"""
import pathlib

OUT = pathlib.Path(__file__).resolve().parent.parent / "docs" / "img"

THEMES = {
    "light": dict(ink="#14202B", sub="#5A6B78", box="#FFFFFF", edge="#B9C8D4",
                  line="#7C8F9D", accent="#1B6FA8", ok="#2E7D52",
                  board="#EAF3F9", pc="#F2F1FA", gpu="#FDF1E8", grp="#D5E1EA"),
    "dark":  dict(ink="#E6EFF5", sub="#93A8B6", box="#16212A", edge="#31485A",
                  line="#6E8798", accent="#63B8E8", ok="#6FC894",
                  board="#102029", pc="#1C1A2C", gpu="#2A1D15", grp="#2B3E4C"),
}

SANS = "ui-sans-serif,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif"
MONO = "ui-monospace,SFMono-Regular,Menlo,Consolas,monospace"


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class Canvas:
    def __init__(self, w, h, t):
        self.w, self.h, self.t, self.p = w, h, t, []

    def group(self, x, y, w, h, label, fill):
        t = self.t
        self.p.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" '
            f'fill="{t[fill]}" stroke="{t["grp"]}" stroke-width="1.5"/>')
        self.p.append(
            f'<text x="{x+14}" y="{y+22}" font-family="{MONO}" font-size="11.5" '
            f'letter-spacing="1.4" fill="{t["sub"]}">{esc(label.upper())}</text>')

    def box(self, x, y, w, h, title, sub=None, accent=False, tint=None):
        t = self.t
        col = t["accent"] if accent else t["edge"]
        fill = t[tint] if tint else t["box"]
        self.p.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" '
            f'fill="{fill}" stroke="{col}" stroke-width="{2 if accent else 1.5}"/>')
        cx = x + w / 2
        if sub:
            self.p.append(
                f'<text x="{cx}" y="{y+h/2-4}" text-anchor="middle" '
                f'font-family="{SANS}" font-size="13.5" font-weight="600" '
                f'fill="{t["ink"]}">{esc(title)}</text>')
            self.p.append(
                f'<text x="{cx}" y="{y+h/2+14}" text-anchor="middle" '
                f'font-family="{MONO}" font-size="11" '
                f'fill="{t["sub"]}">{esc(sub)}</text>')
        else:
            self.p.append(
                f'<text x="{cx}" y="{y+h/2+5}" text-anchor="middle" '
                f'font-family="{SANS}" font-size="13.5" font-weight="600" '
                f'fill="{t["ink"]}">{esc(title)}</text>')

    def arrow(self, pts, label=None, dashed=False, ok=False, lab_seg=0):
        """lab_seg = chỉ số điểm đầu của đoạn mang nhãn.

        Mặc định đoạn đầu tiên. Với đường gấp khúc thì phải chỉ định, nếu
        không nhãn rơi vào đoạn dọc và đè lên đáy hộp phía trên.
        """
        t = self.t
        col = t["ok"] if ok else t["line"]
        d = " ".join(f"{x},{y}" for x, y in pts)
        dash = ' stroke-dasharray="6 5"' if dashed else ""
        self.p.append(
            f'<polyline points="{d}" fill="none" stroke="{col}" '
            f'stroke-width="1.8"{dash} marker-end="url(#a{"ok" if ok else ""})"/>')
        if label:
            mx, my = pts[lab_seg]
            nx, ny = pts[lab_seg + 1]
            self.p.append(
                f'<text x="{(mx+nx)/2}" y="{(my+ny)/2-8}" text-anchor="middle" '
                f'font-family="{MONO}" font-size="10.5" '
                f'fill="{t["sub"]}">{esc(label)}</text>')

    def render(self):
        t = self.t
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" '
            f'height="{self.h}" viewBox="0 0 {self.w} {self.h}">'
            f'<defs>'
            f'<marker id="a" markerWidth="9" markerHeight="9" refX="8" refY="3" '
            f'orient="auto"><path d="M0,0 L8,3 L0,6 z" fill="{t["line"]}"/></marker>'
            f'<marker id="aok" markerWidth="9" markerHeight="9" refX="8" refY="3" '
            f'orient="auto"><path d="M0,0 L8,3 L0,6 z" fill="{t["ok"]}"/></marker>'
            f'</defs>' + "".join(self.p) + "</svg>")


def pipeline1(theme):
    c = Canvas(1040, 360, THEMES[theme])
    c.group(16, 16, 1008, 328, "Pipeline 1 · toàn bộ trên board", "board")

    c.box(44, 62, 108, 50, "mic")
    c.box(196, 62, 200, 50, "Zipformer-30M", "int8 · 189ms")
    c.box(440, 62, 208, 50, "wake word", "+ lệnh cùng câu?", accent=True)
    c.box(760, 52, 236, 60, "SO-101 arm", "13 cử chỉ", accent=True)

    c.box(440, 176, 236, 62, "Qwen2.5-0.5B", "Q4_0 · llama.cpp · 4 luồng")
    c.box(716, 176, 112, 62, "Piper TTS")
    c.box(868, 176, 128, 62, "loa BT")

    c.box(44, 268, 200, 50, "2 cameras", "25fps")
    c.box(292, 268, 188, 50, "dashboard", ":8088")

    c.arrow([(152, 87), (190, 87)])
    c.arrow([(396, 87), (434, 87)])
    c.arrow([(648, 87), (754, 87)], "khớp", ok=True)
    c.arrow([(544, 112), (544, 170)], "không khớp")
    c.arrow([(676, 207), (710, 207)])
    c.arrow([(828, 207), (862, 207)])
    # Đi lên hành lang trống y=150 (giữa hàng quyết định và hàng LLM) rồi mới
    # rẽ phải, thay vì cắt ngang qua hộp TTS và loa như bản đầu.
    c.arrow([(558, 176), (558, 150), (878, 150), (878, 116)], "cử chỉ", lab_seg=1)
    c.arrow([(244, 293), (286, 293)])
    return c.render()


def pipeline2(theme):
    # Xếp theo LUỒNG, không theo máy. Bản trước xếp mỗi máy một cột, nên đường
    # trả kết quả về tay máy phải vòng qua cả ba nhóm và đè lên tiêu đề nhóm.
    # Ở đây màu hộp cho biết máy nào, còn hướng đọc luôn là trái sang phải.
    t = THEMES[theme]
    c = Canvas(1330, 330, t)

    # chú giải màu
    for i, (lbl, tint) in enumerate((("board", "board"), ("Linux PC", "pc"),
                                     ("GPU box", "gpu"))):
        x = 40 + i * 132
        c.p.append(f'<rect x="{x}" y="20" width="14" height="14" rx="3" '
                   f'fill="{t[tint]}" stroke="{t["edge"]}" stroke-width="1.2"/>')
        c.p.append(f'<text x="{x+22}" y="32" font-family="{MONO}" font-size="11.5" '
                   f'fill="{t["sub"]}">{esc(lbl)}</text>')

    # Khoảng hở giữa hai hộp phải đủ rộng cho nhãn của mũi tên nối chúng:
    # 24px cho mũi tên trần, ~100px khi có chữ. Bản trước dùng 24px khắp nơi
    # nên mọi nhãn đều tràn đè lên hộp bên cạnh.
    c.box(40, 80, 118, 56, "mic", tint="board")
    c.box(182, 80, 150, 56, "Zipformer-30M", "int8 · 189ms", tint="board")
    c.box(356, 80, 152, 56, "wake word", "+ lệnh?", accent=True, tint="board")
    c.box(604, 80, 140, 56, "SSH tunnel", ":8766", tint="pc")
    c.box(768, 80, 194, 56, "Qwen3-4B FP8", "0.33–0.60s · 9.6GB", tint="gpu")

    c.box(1082, 96, 196, 100, "SO-101 arm", "13 cử chỉ", accent=True, tint="board")

    c.box(40, 214, 118, 56, "2 cameras", "25fps", tint="board")
    c.box(182, 214, 150, 56, "dashboard", ":8090", tint="pc")
    c.box(356, 214, 194, 56, "MolmoAct2", "0.84–1.63s · 10.9GB", tint="gpu")
    c.box(646, 214, 194, 56, "image → joint map", tint="gpu")

    c.arrow([(158, 108), (176, 108)])
    c.arrow([(332, 108), (350, 108)])
    c.arrow([(508, 108), (598, 108)], "không khớp")
    c.arrow([(744, 108), (762, 108)])
    c.arrow([(962, 108), (1076, 108)], "reply + cử chỉ", ok=True)
    # hành lang y=165 nằm giữa hai hàng, không hộp nào chạm tới
    c.arrow([(432, 136), (432, 165), (1076, 165)], "khớp", ok=True, lab_seg=1)
    c.arrow([(158, 242), (176, 242)])
    c.arrow([(332, 242), (350, 242)])
    c.arrow([(550, 242), (640, 242)], "toạ độ vật")
    # rẽ lên ở x=1010, dừng ở y=188 nên không cắt hành lang "khớp" (y=165)
    c.arrow([(840, 242), (1010, 242), (1010, 188), (1076, 188)], "gắp",
            dashed=True, lab_seg=0)
    return c.render()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in (("pipeline1", pipeline1), ("pipeline2", pipeline2)):
        for theme in THEMES:
            p = OUT / f"{name}-{theme}.svg"
            p.write_text(fn(theme), encoding="utf-8")
            print(f"  {p.relative_to(OUT.parent.parent)}  {p.stat().st_size} B")


if __name__ == "__main__":
    main()
