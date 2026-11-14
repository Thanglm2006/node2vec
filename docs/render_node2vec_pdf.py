"""Render the Node2Vec HTML handout to a Unicode PDF without external packages."""

from html.parser import HTMLParser
import argparse
from pathlib import Path
import re
import textwrap

import matplotlib
matplotlib.use("Agg")
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure


ROOT = Path(__file__).resolve().parent
HTML_PATH = ROOT / "node2vec_algorithm.html"
PDF_PATH = ROOT / "node2vec_algorithm.pdf"


class HandoutParser(HTMLParser):
    """Extract readable blocks while preserving code and table rows."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.blocks = []
        self.block_tag = None
        self.block_text = []
        self.table_cells = []
        self.cell_text = []
        self.in_cell = False

    def _flush_block(self):
        if self.block_tag is None:
            return
        text = "".join(self.block_text).strip()
        if text:
            self.blocks.append((self.block_tag, text))
        self.block_tag = None
        self.block_text = []

    def _start_block(self, tag):
        self._flush_block()
        self.block_tag = tag
        self.block_text = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in {"h1", "h2", "h3", "p", "pre", "li"}:
            self._start_block(tag)
        elif tag == "div" and "formula" in attrs.get("class", "").split():
            self._start_block("formula")
        elif tag == "tr":
            self._flush_block()
            self.table_cells = []
        elif tag in {"td", "th"}:
            self.cell_text = []
            self.in_cell = True
        elif tag == "br" and self.block_tag is not None:
            self.block_text.append("\n")

    def handle_endtag(self, tag):
        if tag in {"h1", "h2", "h3", "p", "pre", "li"}:
            self._flush_block()
        elif tag == "div" and self.block_tag == "formula":
            self._flush_block()
        elif tag in {"td", "th"} and self.in_cell:
            self.table_cells.append("".join(self.cell_text).strip())
            self.cell_text = []
            self.in_cell = False
        elif tag == "tr" and self.table_cells:
            self.blocks.append(("table", " | ".join(self.table_cells)))
            self.table_cells = []

    def handle_data(self, data):
        if self.in_cell:
            self.cell_text.append(data)
        elif self.block_tag is not None:
            self.block_text.append(data)


def clean_text(value):
    value = value.replace("\u00a0", " ")
    value = re.sub(r"[ \t]+", " ", value)
    return value.strip()


class PdfWriter:
    WIDTH = 8.27
    HEIGHT = 11.69
    LEFT = 0.66
    RIGHT = 0.66
    TOP = 0.68
    BOTTOM = 0.58
    BLUE = "#0f4c81"
    INK = "#1e293b"
    MUTED = "#475569"

    def __init__(self, output):
        self.pdf = PdfPages(output)
        self.page = 0
        self.fig = None
        self.y = None
        self.new_page()

    def new_page(self):
        if self.fig is not None:
            self.fig.text(0.5, 0.035, f"Trang {self.page}", ha="center", va="bottom", fontsize=7.5, color=self.MUTED)
            self.pdf.savefig(self.fig, bbox_inches=None)
        self.page += 1
        self.fig = Figure(figsize=(self.WIDTH, self.HEIGHT), dpi=150, facecolor="white")
        self.fig.subplots_adjust(0, 0, 1, 1)
        self.y = self.HEIGHT - self.TOP

    def ensure(self, height):
        if self.y - height < self.BOTTOM:
            self.new_page()

    def write_lines(self, lines, *, fontsize=9.1, color=None, family="DejaVu Sans", weight="normal", line_height=None, x=None):
        if line_height is None:
            line_height = fontsize / 72 * 1.45
        if x is None:
            x = self.LEFT / self.WIDTH
        height = len(lines) * line_height
        self.ensure(height + 0.02)
        for line in lines:
            self.fig.text(x, self.y / self.HEIGHT, line, transform=self.fig.transFigure,
                          fontsize=fontsize, color=color or self.INK, family=family,
                          weight=weight, va="top")
            self.y -= line_height
        return height

    def block(self, kind, text):
        if kind in {"h1", "h2", "h3"}:
            size = {"h1": 20, "h2": 14.2, "h3": 11.2}[kind]
            if kind == "h2" and self.y < self.HEIGHT - self.TOP - 1.2:
                self.ensure(0.75)
                self.y -= 0.08
            self.y -= 0.04
            lines = textwrap.wrap(clean_text(text), width=78) or [""]
            self.write_lines(lines, fontsize=size, color=self.BLUE, weight="bold", line_height=size / 72 * 1.3)
            if kind == "h2":
                self.y -= 0.05
                self.fig.add_artist(matplotlib.lines.Line2D(
                    [self.LEFT / self.WIDTH, (self.WIDTH - self.RIGHT) / self.WIDTH],
                    [self.y / self.HEIGHT, self.y / self.HEIGHT],
                    transform=self.fig.transFigure, color=self.BLUE, linewidth=1.1,
                ))
                self.y -= 0.11
            else:
                self.y -= 0.04
            return

        if kind == "pre":
            raw_lines = text.expandtabs(2).splitlines() or [""]
            lines = []
            for line in raw_lines:
                lines.extend(textwrap.wrap(line, width=94, replace_whitespace=False,
                                           drop_whitespace=False) or [""])
            line_height = 7.6 / 72 * 1.38
            height = len(lines) * line_height + 0.15
            self.ensure(height)
            left = self.LEFT - 0.04
            bottom = (self.y - height + 0.03) / self.HEIGHT
            rect = matplotlib.patches.FancyBboxPatch(
                (left / self.WIDTH, bottom),
                (self.WIDTH - self.LEFT - self.RIGHT + 0.08) / self.WIDTH,
                height / self.HEIGHT,
                boxstyle="round,pad=0.008,rounding_size=0.008",
                transform=self.fig.transFigure, facecolor="#f8fafc", edgecolor="#cbd5e1", linewidth=0.6,
            )
            self.fig.add_artist(rect)
            self.y -= 0.06
            self.write_lines(lines, fontsize=7.6, family="DejaVu Sans Mono", line_height=line_height)
            self.y -= 0.12
            return

        if kind == "formula":
            lines = textwrap.wrap(clean_text(text), width=88) or [""]
            height = len(lines) * 0.18 + 0.18
            self.ensure(height)
            self.y -= 0.07
            for line in lines:
                self.fig.text(0.5, self.y / self.HEIGHT, line, transform=self.fig.transFigure,
                              fontsize=9.1, color=self.INK, family="DejaVu Sans",
                              weight="bold", ha="center", va="top")
                self.y -= 0.18
            self.y -= 0.08
            return

        prefix = "• " if kind == "li" else ""
        text = prefix + clean_text(text)
        lines = textwrap.wrap(text, width=102) or [""]
        self.write_lines(lines, fontsize=9.0, line_height=0.185)
        self.y -= 0.035

    def close(self):
        self.fig.text(0.5, 0.035, f"Trang {self.page}", ha="center", va="bottom", fontsize=7.5, color=self.MUTED)
        self.pdf.savefig(self.fig, bbox_inches=None)
        self.pdf.close()


def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--input", type=Path, default=HTML_PATH)
    argument_parser.add_argument("--output", type=Path, default=PDF_PATH)
    argument_parser.add_argument("--cover-label", default="NODE2VEC")
    argument_parser.add_argument("--cover-title", default="Node2Vec trong project\nnode2vec")
    argument_parser.add_argument(
        "--cover-subtitle",
        default="Từ biased random walk đến embedding bằng\nSkip-gram với negative sampling",
    )
    argument_parser.add_argument(
        "--cover-meta",
        default="Tài liệu kỹ thuật theo mã nguồn hiện tại\n"
                "Preprocess.py · Node2vecWalk.py · Node2Vec.py · Skip_gram.py",
    )
    args = argument_parser.parse_args()
    cover_title = args.cover_title.replace("\\n", "\n")
    cover_subtitle = args.cover_subtitle.replace("\\n", "\n")
    cover_meta = args.cover_meta.replace("\\n", "\n")

    parser = HandoutParser()
    parser.feed(args.input.read_text(encoding="utf-8"))
    blocks = parser.blocks

    writer = PdfWriter(args.output)
    # Use a deliberate cover page so the long technical content starts cleanly.
    cover_x = writer.LEFT / writer.WIDTH
    writer.fig.text(cover_x, 0.64, args.cover_label, fontsize=12, color=writer.BLUE, weight="bold")
    writer.fig.text(cover_x, 0.56, cover_title, fontsize=25, color=writer.BLUE,
                    weight="bold", va="top", linespacing=1.2)
    writer.fig.text(cover_x, 0.39, cover_subtitle,
                    fontsize=13.5, color=writer.MUTED, va="top", linespacing=1.35)
    writer.fig.text(cover_x, 0.22, cover_meta,
                    fontsize=9.5, color=writer.MUTED, va="top", linespacing=1.5)
    writer.new_page()

    # The PDF cover replaces everything before the first section heading.
    content_start = next(
        (index for index, (kind, _text) in enumerate(blocks) if kind == "h2"),
        4,
    )
    for kind, text in blocks[content_start:]:
        writer.block(kind, text)
    writer.close()
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
