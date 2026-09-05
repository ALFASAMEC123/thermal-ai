"""
Hlavný pipeline pre spracovanie termovíznych snímkov.
Spája konverziu, AI analýzu a export do jedného workflow.
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from .config_manager import get_config, ConfigManager
from .thermal_converter import ThermalConverter, ThermalImage, ThermalFormat
from .vlm_analyzer import ThermalVLMAnalyzer, AnalysisResult, VLMConfig
from .png_exporter import ThermalPNGExporter, ExportConfig, create_exporter

logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """Výsledok spracovania jedného súboru."""
    input_file: Path
    thermal_image: Optional[ThermalImage] = None
    analysis: Optional[AnalysisResult] = None
    temperature_stats: Optional[Dict[str, Any]] = None
    output_image: Optional[Path] = None
    success: bool = False
    error: Optional[str] = None


class ThermalProcessingPipeline:
    """Hlavný spracovací pipeline."""

    def __init__(self, config_path: Optional[str] = None):
        self.config_manager = ConfigManager()
        if config_path:
            self.config_manager.load(config_path)
        self.settings = self.config_manager.settings

        # Inicializácia komponentov
        self.converter = ThermalConverter()
        self.analyzer = ThermalVLMAnalyzer(VLMConfig(
            host=self.settings.ollama.host,
            model=self.settings.ollama.model,
            timeout=self.settings.ollama.timeout,
            temperature=self.settings.ollama.temperature,
            top_p=self.settings.ollama.top_p,
        ))
        self.exporter = create_exporter(ExportConfig(
            canvas_height=self.settings.export.canvas_height,
            font_size=self.settings.export.font_size,
            font_family=self.settings.export.font_family,
            text_color=tuple(self.settings.export.text_color),
            background_color=tuple(self.settings.export.background_color),
            padding=self.settings.export.padding,
            line_spacing=self.settings.export.line_spacing,
        ))

    def process_file(self, input_path: Union[str, Path],
                    output_dir: Optional[Union[str, Path]] = None,
                    overwrite: bool = False) -> ProcessingResult:
        """Spracovanie jedného súboru."""
        input_path = Path(input_path)
        result = ProcessingResult(input_file=input_path)

        try:
            # 1. Konverzia
            logger.info(f"Konvertujem: {input_path}")
            thermal_img = self.converter.load(input_path)

            # Uloženie medzivýsledku
            if self.settings.processing.save_intermediate:
                inter_dir = Path("data/processed")
                inter_dir.mkdir(parents=True, exist_ok=True)
                temp_png = inter_dir / f"{input_path.stem}_thermal.png"
                self.converter.save_as_image(
                    thermal_img, temp_png,
                    normalize_method=self.settings.conversion.normalize_method,
                    colormap=self.settings.conversion.colormap,
                    target_size=tuple(self.settings.conversion.target_size)
                )

            result.thermal_image = thermal_img

            # 2. Teplotné štatistiky
            result.temperature_stats = self.converter.get_temperature_stats(thermal_img)

            # 3. AI Analýza
            logger.info(f"Analyzujem cez VLM: {input_path}")
            pil_image = self.converter.to_pil(
                thermal_img,
                normalize_method=self.settings.conversion.normalize_method,
                colormap=self.settings.conversion.colormap,
                target_size=tuple(self.settings.conversion.target_size)
            )

            analysis = self.analyzer.analyze(
                pil_image,
                temperature_data=result.temperature_stats
            )
            result.analysis = analysis

            if not analysis.success:
                raise RuntimeError(f"VLM analýza zlyhala: {analysis.error}")

            # 4. Export do PNG s textom
            output_dir = Path(output_dir or "data/output")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{input_path.stem}_report.png"

            if output_path.exists() and not overwrite:
                logger.warning(f"Súbor existuje, preskakujem: {output_path}")
                result.output_image = output_path
                result.success = True
                return result

            logger.info(f"Exportujem report: {output_path}")
            self.exporter.export_from_analysis(
                thermal_img_path=pil_image,
                analysis_result=analysis,
                temperature_stats=result.temperature_stats,
                output_path=output_path
            )

            result.output_image = output_path
            result.success = True

        except Exception as e:
            logger.error(f"Chyba pri spracovaní {input_path}: {e}")
            result.error = str(e)
            result.success = False

        return result

    def process_batch(self, input_paths: List[Union[str, Path]],
                     output_dir: Optional[Union[str, Path]] = None,
                     max_workers: Optional[int] = None,
                     overwrite: bool = False) -> List[ProcessingResult]:
        """Dávkové spracovanie viacerých súborov."""
        max_workers = max_workers or self.settings.processing.max_workers
        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.process_file, p, output_dir, overwrite): p
                for p in input_paths
            }

            for future in tqdm(as_completed(futures), total=len(futures), desc="Spracovávanie"):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    path = futures[future]
                    logger.error(f"Chyba vo vlákne pre {path}: {e}")
                    results.append(ProcessingResult(
                        input_file=Path(path),
                        success=False,
                        error=str(e)
                    ))

        return results

    def process_directory(self, input_dir: Union[str, Path],
                         output_dir: Optional[Union[str, Path]] = None,
                         recursive: bool = True,
                         overwrite: bool = False) -> List[ProcessingResult]:
        """Spracovanie všetkých podporovaných súborov v adresári."""
        input_dir = Path(input_dir)
        extensions = self.settings.conversion.input_formats

        files = []
        if recursive:
            for ext in extensions:
                files.extend(input_dir.rglob(f"*{ext}"))
        else:
            for ext in extensions:
                files.extend(input_dir.glob(f"*{ext}"))

        logger.info(f"Nájdených {len(files)} súborov na spracovanie")
        return self.process_batch(files, output_dir, overwrite=overwrite)


def create_pipeline(config_path: Optional[str] = None) -> ThermalProcessingPipeline:
    """Factory funkcia."""
    return ThermalProcessingPipeline(config_path)


# CLI
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Termovízny procesing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Príklady:
  # Spracovanie jedného súboru
  python -m src.pipeline input.irb -o output/

  # Spracovanie celého adresára
  python -m src.pipeline data/raw/ -o data/output/ --recursive

  # S vlastnou konfiguráciou
  python -m src.pipeline data/raw/ --config config/custom.yaml
        """
    )

    parser.add_argument("input", help="Vstupný súbor alebo adresár")
    parser.add_argument("-o", "--output", default="data/output", help="Výstupný adresár")
    parser.add_argument("-c", "--config", help="Konfiguračný súbor")
    parser.add_argument("-r", "--recursive", action="store_true", help="Rekurzívne pre adresáre")
    parser.add_argument("--overwrite", action="store_true", help="Prepísať existujúce súbory")
    parser.add_argument("--workers", type=int, help="Počet vlákien")
    parser.add_argument("--model", help="VLM model (prekonfiguruje Ollama model)")
    parser.add_argument("--list-formats", action="store_true", help="Zobrazí podporované formáty")
    parser.add_argument("--check-ollama", action="store_true", help="Skontroluje pripojenie k Ollama")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Zobrazenie formátov
    if args.list_formats:
        print("Podporované formáty:")
        for fmt in ThermalFormat:
            print(f"  {fmt.value}")
        return

    # Vytvorenie pipeline
    pipeline = create_pipeline(args.config)

    # Override modelu
    if args.model:
        pipeline.analyzer.config.model = args.model

    # Check Ollama
    if args.check_ollama:
        if pipeline.analyzer.client.check_connection():
            print(f"✓ Pripojené k Ollama: {pipeline.analyzer.config.host}")
            models = pipeline.analyzer.client.list_models()
            print(f"Dostupné modely: {', '.join(models)}")
        else:
            print("✗ Nepodarilo sa pripojiť k Ollama", file=sys.stderr)
            sys.exit(1)
        return

    input_path = Path(args.input)

    if not input_path.exists():
        print(f"CHYBA: Cesta neexistuje: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Spracovanie
    if input_path.is_file():
        results = [pipeline.process_file(input_path, args.output, args.overwrite)]
    elif input_path.is_dir():
        results = pipeline.process_directory(
            input_path, args.output, args.recursive, args.overwrite
        )
    else:
        print(f"CHYBA: Neplatná cesta: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Štatistiky
    successful = sum(1 for r in results if r.success)
    failed = len(results) - successful

    print(f"\n=== VÝSLEDOK ===")
    print(f"Úspešné: {successful}")
    print(f"Zlyhané: {failed}")

    if failed > 0:
        print("\nChyby:")
        for r in results:
            if not r.success:
                print(f"  {r.input_file}: {r.error}")
        sys.exit(1)


if __name__ == "__main__":
    main()