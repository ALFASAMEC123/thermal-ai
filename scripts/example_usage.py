#!/usr/bin/env python3
"""
Príklad použitia - spracovanie termovíznych snímkov.
"""

import sys
from pathlib import Path

# Pridaj src do path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src import create_pipeline, create_converter, create_analyzer, create_exporter
from src.config_manager import ConfigManager


def example_single_file():
    """Spracovanie jedného súboru."""
    print("=== Príklad 1: Spracovanie jedného súboru ===")

    # Vytvorenie pipeline s default konfiguráciou
    pipeline = create_pipeline()

    # Spracovanie
    result = pipeline.process_file(
        "data/raw/sample.irb",
        "data/output/"
    )

    if result.success:
        print(f"✓ Úspešne spracované: {result.input_file}")
        print(f"  Výstup: {result.output_image}")
        print(f"  Čas analýzy: {result.analysis.processing_time:.2f}s")
        print("\nAnalýza:")
        print(result.analysis.to_formatted_text())
    else:
        print(f"✗ Chyba: {result.error}")


def example_batch_processing():
    """Dávkové spracovanie adresára."""
    print("\n=== Príklad 2: Dávkové spracovanie ===")

    pipeline = create_pipeline()

    results = pipeline.process_directory(
        "data/raw/",
        "data/output/",
        recursive=True,
        overwrite=False
    )

    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    print(f"Spracovaných: {len(results)}")
    print(f"Úspešné: {len(successful)}")
    print(f"Zlyhané: {len(failed)}")

    for r in failed:
        print(f"  ✗ {r.input_file}: {r.error}")


def example_converter_only():
    """Použitie len konvertéra."""
    print("\n=== Príklad 3: Len konverzia ===")

    converter = create_converter()

    # Načítanie .raw s manuálnym zadanie rozlíšenia
    thermal_img = converter.load(
        "data/raw/thermal.raw",
        width=640,
        height=512,
        dtype="uint16"
    )

    print(f"Rozlíšenie: {thermal_img.width}x{thermal_img.height}")
    print(f"Teplotný rozsah: {thermal_img.min_temp:.1f}°C - {thermal_img.max_temp:.1f}°C")
    print(f"Priemerná teplota: {thermal_img.mean_temp:.1f}°C")

    # Uloženie ako PNG
    converter.save_as_image(
        thermal_img,
        "data/output/converted.png",
        normalize_method="minmax",
        colormap="inferno"
    )

    # Teplotné štatistiky
    stats = converter.get_temperature_stats(thermal_img)
    print(f"Globálne min: {stats['global']['min']:.1f}°C")
    print(f"Globálne max: {stats['global']['max']:.1f}°C")


def example_vlm_only():
    """Použitie len VLM analyzéra."""
    print("\n=== Príklad 4: Len VLM analýza ===")

    from src.vlm_analyzer import VLMConfig
    from PIL import Image

    config = VLMConfig(
        host="http://localhost:11434",
        model="qwen2-vl:7b",
        temperature=0.1
    )
    analyzer = create_analyzer(config)

    # Skontroluj pripojenie
    if not analyzer.client.check_connection():
        print("Ollama nedostupné!")
        return

    # Načítanie obrázku
    image = Image.open("data/processed/thermal.png")

    # Analýza
    result = analyzer.analyze(image)

    if result.success:
        print("Analýza:")
        print(result.to_formatted_text())
    else:
        print(f"Chyba: {result.error}")


def example_custom_prompt():
    """Vlastný prompt pre analýzu."""
    print("\n=== Príklad 5: Vlastný prompt ===")

    from src.vlm_analyzer import ThermalVLMAnalyzer, VLMConfig

    custom_prompt = """Si inžinier tepelnej diagnostiky. Analyzuj termovízny snímok a odpovedz stručne:

SEKCIA: [názov sekcie/komponentu]
PROBLEM: [popis problému]
TEPLOTA_MAX: [max teplota v °C]
TEPLOTA_REF: [referenčná teplota v °C]
ROZDIL: [rozdiel v °C]
HODNENIE: [OK / POZOR / KRITICKÉ]
AKCIA: [konkrétna akcia]

Odpovedz SLOVENSKY."""

    config = VLMConfig(model="qwen2-vl:7b")
    analyzer = ThermalVLMAnalyzer(config)

    image = "data/processed/thermal.png"
    result = analyzer.analyze(image, prompt=custom_prompt)

    if result.success:
        print(result.raw_response)


def example_export():
    """Export s vlastnou konfiguráciou."""
    print("\n=== Príklad 5: Vlastný export ===")

    from src.png_exporter import ThermalPNGExporter, ExportConfig
    from PIL import Image

    config = ExportConfig(
        canvas_height=300,
        font_size=16,
        font_family="DejaVuSansMono",
        text_color=(255, 255, 255),
        background_color=(0, 0, 0),
        show_temperature_scale=True,
        section_colors={
            'LOKALITA': (100, 255, 100),
            'TEPLOTNÉ ANOMÁLIE': (255, 200, 50),
            'MOŽNÁ PRÍČINA': (255, 100, 100),
            'ODPORÚČANIE': (100, 200, 255),
        }
    )

    exporter = ThermalPNGExporter(config)

    # Načítanie termovízneho obrázku
    thermal_img = Image.open("data/processed/thermal.png")

    # Vlastný text analýzy
    analysis_text = """LOKALITA: Transformátorná stanica T-15, fáza L2
TEPLOTNÉ ANOMÁLIE: Izolátor L2 prehrety na 112°C, ostatné fázy 45-50°C
MOŽNÁ PRÍČINA: Čiastočný výboj izolátora, vlhkosť
ODPORÚČANIE: Okamžité odpojenie, výmena izolátora do 4h"""

    report = exporter.create_report_image(
        thermal_img,
        analysis_text,
        temperature_stats={'global': {'min': 20.0, 'max': 112.0, 'mean': 48.5}}
    )

    exporter.save(report, "data/output/custom_report.png")
    print("Vlastný report uložený")


def example_finetuning_dataset():
    """Vytvorenie datasetu pre fine-tuning."""
    print("\n=== Príklad 6: Vytvorenie datasetu ===")

    from src.finetuning import create_dataset_builder

    builder = create_dataset_builder("data/finetune_dataset")

    # Pridanie vzoriek z existujúcich analýz
    pipeline = create_pipeline()

    for file in Path("data/raw").glob("*.irb"):
        result = pipeline.process_file(file)
        if result.success:
            builder.add_from_analysis(file, result.analysis, result.temperature_stats)

    # Vytvorenie JSONL v rôznych formátoch
    builder.build_jsonl("train_unsloth.jsonl", format_type="unsloth")
    builder.build_jsonl("train_axolotl.jsonl", format_type="axolotl")
    builder.build_jsonl("train_llamafactory.jsonl", format_type="llamafactory")

    print(f"Vytvorený dataset s {len(builder.samples)} vzorkami")

    # Rozdelenie train/val/test
    train, val, test = builder.split_dataset(0.8, 0.1, 0.1)
    print(f"Train: {train}, Val: {val}, Test: {test}")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    # Spusti príklady (odkomentuj podľa potreby)
    # example_single_file()
    # example_batch_processing()
    # example_converter_only()
    # example_vlm_only()
    # example_custom_prompt()
    # example_export()
    # example_finetuning_dataset()

    print("Príklady sú k dispozícii. Odkomentuj v kóde podľa potreby.")
    print("\nDostupné funkcie:")
    print("  - example_single_file()")
    print("  - example_batch_processing()")
    print("  - example_converter_only()")
    print("  - example_vlm_only()")
    print("  - example_custom_prompt()")
    print("  - example_export()")
    print("  - example_finetuning_dataset()")