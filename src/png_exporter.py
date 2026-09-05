"""
Export termovíznych snímkov do PNG s textovým overlayom.
Vytvára finálny report obrázok s originálnym termovízny snímkom a AI analýzou.
"""

import textwrap
from pathlib import Path
from typing import Tuple, Optional, Union, List, Dict, Any
from dataclasses import dataclass
from enum import Enum
from PIL import Image, ImageDraw, ImageFont, ImageColor
import numpy as np
import logging

logger = logging.getLogger(__name__)


class FontStyle(Enum):
    REGULAR = "regular"
    BOLD = "bold"
    MONOSPACE = "monospace"


@dataclass
class ExportConfig:
    """Konfigurácia exportu."""
    canvas_height: int = 200
    font_size: int = 14
    font_family: str = "DejaVuSansMono"
    text_color: Tuple[int, int, int] = (255, 255, 255)
    background_color: Tuple[int, int, int] = (0, 0, 0)
    padding: int = 10
    line_spacing: float = 1.5
    header_font_size: int = 16
    header_color: Tuple[int, int, int] = (255, 255, 100)
    section_colors: Dict[str, Tuple[int, int, int]] = None
    max_lines: int = 30
    wrap_width: int = 80
    show_timestamp: bool = True
    show_temperature_scale: bool = True
    scale_bar_height: int = 30
    scale_bar_width: int = 200

    def __post_init__(self):
        if self.section_colors is None:
            self.section_colors = {
                'LOKALITA': (100, 255, 100),
                'TEPLOTNÉ ANOMÁLIE': (255, 200, 100),
                'MOŽNÁ PRÍČINA': (255, 100, 100),
                'ODPORÚČANIE': (100, 200, 255),
            }


class ThermalPNGExporter:
    """Exportér termovíznych snímkov do PNG s textom."""

    def __init__(self, config: Optional[ExportConfig] = None):
        self.config = config or ExportConfig()
        self._font_cache: Dict[Tuple[str, int, FontStyle], ImageFont.FreeTypeFont] = {}

    def _get_font(self, size: int, style: FontStyle = FontStyle.REGULAR) -> ImageFont.FreeTypeFont:
        """Získanie fontu s cachingom."""
        key = (self.config.font_family, size, style)
        if key not in self._font_cache:
            try:
                if style == FontStyle.BOLD:
                    font_path = self._find_font_file(self.config.font_family + "-Bold")
                elif style == FontStyle.MONOSPACE:
                    font_path = self._find_font_file("DejaVuSansMono")
                else:
                    font_path = self._find_font_file(self.config.font_family)

                if font_path:
                    self._font_cache[key] = ImageFont.truetype(font_path, size)
                else:
                    self._font_cache[key] = ImageFont.load_default()
            except Exception as e:
                logger.warning(f"Font loading failed: {e}, using default")
                self._font_cache[key] = ImageFont.load_default()
        return self._font_cache[key]

    def _find_font_file(self, name: str) -> Optional[str]:
        """Hľadanie fontového súboru v systéme."""
        import os
        font_dirs = [
            "/usr/share/fonts",
            "/usr/local/share/fonts",
            "/home/strane/.local/share/fonts",
            "/home/strane/.fonts",
            "C:/Windows/Fonts",
        ]

        extensions = [".ttf", ".otf", ".TTF", ".OTF"]

        for font_dir in font_dirs:
            if not os.path.exists(font_dir):
                continue
            for root, dirs, files in os.walk(font_dir):
                for file in files:
                    if name.lower() in file.lower() and any(file.endswith(ext) for ext in extensions):
                        return os.path.join(root, file)
        return None

    def create_report_image(self,
                           thermal_image: Union[Image.Image, np.ndarray, str, Path],
                           analysis_text: str,
                           temperature_stats: Optional[Dict[str, Any]] = None,
                           metadata: Optional[Dict[str, Any]] = None) -> Image.Image:
        """
        Vytvorenie report obrázku s termovíziou a textom.

        Args:
            thermal_image: Termovízny obrázok (PIL, numpy array, alebo cesta)
            analysis_text: Text analýzy od AI
            temperature_stats: Štatistiky teplôt
            metadata: Dodatočné metadáta

        Returns:
            PIL Image s reportom
        """
        # Načítanie termovízneho obrázku
        if isinstance(thermal_image, (str, Path)):
            thermal_pil = Image.open(thermal_image).convert('RGB')
        elif isinstance(thermal_image, np.ndarray):
            if thermal_image.dtype != np.uint8:
                thermal_image = ((thermal_image - thermal_image.min()) /
                                (thermal_image.max() - thermal_image.min() + 1e-8) * 255).astype(np.uint8)
            thermal_pil = Image.fromarray(thermal_image).convert('RGB')
        elif isinstance(thermal_image, Image.Image):
            thermal_pil = thermal_image.convert('RGB')
        else:
            raise ValueError(f"Unsupported image type: {type(thermal_image)}")

        # Rozmery
        img_width, img_height = thermal_pil.size
        canvas_height = self.config.canvas_height
        total_height = img_height + canvas_height

        # Vytvorenie plátna
        canvas = Image.new('RGB', (img_width, total_height), self.config.background_color)
        draw = ImageDraw.Draw(canvas)

        # Vloženie termovízneho obrázku
        canvas.paste(thermal_pil, (0, 0))

        # Teplotná ládka (farebná škála)
        y_offset = img_height
        if self.config.show_temperature_scale and temperature_stats:
            y_offset = self._draw_temperature_scale(draw, img_width, y_offset, temperature_stats)

        # Časová pečiatka
        if self.config.show_timestamp:
            y_offset = self._draw_timestamp(draw, img_width, y_offset, metadata)

        # Textová analýza
        self._draw_analysis_text(draw, img_width, y_offset, analysis_text)

        return canvas

    def _draw_temperature_scale(self, draw: ImageDraw.Draw, width: int,
                                y_start: int, stats: Dict[str, Any]) -> int:
        """Vykreslenie teplotnej ládky."""
        bar_height = self.config.scale_bar_height
        bar_width = min(self.config.scale_bar_width, width - 2 * self.config.padding)
        x_start = self.config.padding
        y_end = y_start + bar_height

        # Globálne min/max
        global_stats = stats.get('global', {})
        t_min = global_stats.get('min', 0)
        t_max = global_stats.get('max', 100)

        # Gradient
        for i in range(bar_width):
            ratio = i / bar_width
            # Inferno-like gradient
            r = int(255 * (ratio ** 0.5))
            g = int(255 * (ratio ** 1.5))
            b = int(255 * (ratio ** 2.5))
            draw.line([(x_start + i, y_start), (x_start + i, y_end)],
                     fill=(r, g, b))

        # Ohraničenie
        draw.rectangle([x_start, y_start, x_start + bar_width, y_end],
                      outline=(255, 255, 255), width=1)

        # Popisky
        font = self._get_font(self.config.font_size - 2)
        draw.text((x_start, y_end + 2), f"{t_min:.1f}°C", fill=self.config.text_color, font=font)
        draw.text((x_start + bar_width - 40, y_end + 2), f"{t_max:.1f}°C",
                 fill=self.config.text_color, font=font)
        draw.text((x_start, y_end + 18), "Teplotná ládka", fill=self.config.text_color, font=font)

        return y_end + 35

    def _draw_timestamp(self, draw: ImageDraw.Draw, width: int,
                        y_start: int, metadata: Optional[Dict[str, Any]]) -> int:
        """Vykreslenie časovej pečiatky a metadát."""
        from datetime import datetime
        font = self._get_font(self.config.font_size - 2)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        text = f"Analýza vygenerovaná: {timestamp}"

        if metadata:
            if 'source_file' in metadata:
                text += f" | Zdroj: {Path(metadata['source_file']).name}"
            if 'format' in metadata:
                text += f" | Formát: {metadata['format']}"

        draw.text((self.config.padding, y_start), text, fill=(180, 180, 180), font=font)
        return y_start + self.config.font_size + 5

    def _draw_analysis_text(self, draw: ImageDraw.Draw, width: int,
                           y_start: int, analysis_text: str):
        """Vykreslenie textovej analýzy s formátovaním."""
        font = self._get_font(self.config.font_size)
        header_font = self._get_font(self.config.header_font_size, FontStyle.BOLD)
        mono_font = self._get_font(self.config.font_size, FontStyle.MONOSPACE)

        x = self.config.padding
        y = y_start
        line_height = int(self.config.font_size * self.config.line_spacing)
        max_width = width - 2 * self.config.padding

        # Rozdelenie textu na sekcie
        sections = self._parse_sections(analysis_text)

        for section_name, content in sections:
            if y > self.config.max_lines * line_height:
                break

            # Header sekcie
            color = self.config.section_colors.get(section_name.upper(), self.config.header_color)
            draw.text((x, y), section_name, fill=color, font=header_font)
            y += self.config.header_font_size + 4

            # Obsah sekcie
            if content:
                wrapped = textwrap.wrap(content, width=self.config.wrap_width)
                for line in wrapped:
                    if y > self.config.max_lines * line_height:
                        break
                    draw.text((x + 5, y), line, fill=self.config.text_color, font=mono_font)
                    y += line_height
            y += 4  # medzera medzi sekciami

    def _parse_sections(self, text: str) -> List[Tuple[str, str]]:
        """Parsovanie textu na sekcie."""
        sections = []
        known_sections = [
            'LOKALITA', 'TEPLOTNÉ ANOMÁLIE', 'TEPLOTNE ANOMALIE', 'ANOMÁLIE',
            'MOŽNÁ PRÍČINA', 'MOZNA PRICINA', 'PRÍČINA', 'PRICINA',
            'ODPORÚČANIE', 'ODPORUCANIE'
        ]

        current_section = "ANALÝZA"
        current_content = []

        lines = text.split('\n')
        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                continue

            # Check if line starts a new section
            found_section = None
            for sec in known_sections:
                if line_stripped.upper().startswith(sec):
                    found_section = sec
                    break

            if found_section:
                # Ulož predchádzajúcu sekciu
                if current_content:
                    sections.append((current_section, '\n'.join(current_content)))
                current_section = found_section
                # Odstrán prefix
                content = line_stripped[len(found_section):].strip()
                if content.startswith(':'):
                    content = content[1:].strip()
                current_content = [content] if content else []
            else:
                current_content.append(line_stripped)

        # Posledná sekcia
        if current_content:
            sections.append((current_section, '\n'.join(current_content)))

        return sections

    def save(self, image: Image.Image, output_path: Union[str, Path],
             quality: int = 95, optimize: bool = True):
        """Uloženie obrázku."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path, quality=quality, optimize=optimize)
        logger.info(f"Report uložený: {path}")

    def export_from_analysis(self,
                            thermal_img_path: Union[str, Path],
                            analysis_result,
                            temperature_stats: Optional[Dict[str, Any]] = None,
                            output_path: Optional[Union[str, Path]] = None) -> Image.Image:
        """
        Kompletný export z výsledkov analýzy.

        Args:
            thermal_img_path: Cesta k pôvodnému termovíznemu obrázku
            analysis_result: AnalysisResult objekt alebo text
            temperature_stats: Štatistiky teplôt
            output_path: Výstupná cesta (voliteľné)

        Returns:
            PIL Image s reportom
        """
        # Načítanie analýzy
        if hasattr(analysis_result, 'to_formatted_text'):
            analysis_text = analysis_result.to_formatted_text()
        else:
            analysis_text = str(analysis_result)

        # Metadáta
        metadata = {}
        if hasattr(analysis_result, 'raw_response'):
            metadata['raw_response'] = analysis_result.raw_response

        # Vytvorenie reportu
        report = self.create_report_image(
            thermal_img_path,
            analysis_text,
            temperature_stats,
            metadata
        )

        # Uloženie
        if output_path:
            self.save(report, output_path)

        return report


def create_exporter(config: Optional[ExportConfig] = None) -> ThermalPNGExporter:
    """Factory funkcia pre vytvorenie exportéra."""
    return ThermalPNGExporter(config)


# CLI pre testovanie
if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Export termovízneho reportu do PNG")
    parser.add_argument("image", help="Termovízny obrázok")
    parser.add_argument("output", help "Výstupný PNG")
    parser.add_argument("--text", help="Text analýzy (ak nie je poskytnutý, použije sa dummy)")
    parser.add_argument("--canvas-height", type=int, default=200, help="Výška plátna pre text")
    parser.add_argument("--font-size", type=int, default=14, help "Veľkosť fontu")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    config = ExportConfig(canvas_height=args.canvas_height, font_size=args.font_size)
    exporter = ThermalPNGExporter(config)

    # Dummy analýza pre test
    if args.text:
        analysis_text = args.text
    else:
        analysis_text = """LOKALITA: Motorový kompartment, ľavý bok
TEPLOTNÉ ANOMÁLIE: Lokálny prehrev 87.3°C na pozícii (320, 245), ozadí 35.2°C
MOŽNÁ PRÍČINA: Porucha ložiska motoru
ODPORÚČANIE: Okamžitá údržba, výmena ložiska do 24h"""

    # Dummy štatistiky
    temp_stats = {
        'global': {'min': 20.5, 'max': 87.3, 'mean': 42.1, 'std': 12.3}
    }

    report = exporter.create_report_image(args.image, analysis_text, temp_stats)
    exporter.save(report, args.output)
    print(f"Report uložený: {args.output}")