#!/usr/bin/env python3
"""
Porovnanie modelov - pôvodný vs fine-tuned.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
from src.vlm_analyzer import ThermalVLMAnalyzer, VLMConfig
from src.thermal_converter import ThermalConverter
from PIL import Image


def compare_models(image_path: str, model1: str, model2: str, prompt: str = None):
    """Porovnanie dvoch modelov na tom istom obrázku."""

    print(f"Porovnanie: {model1} vs {model2}")
    print(f"Obrázok: {image_path}")
    print("=" * 60)

    # Načítanie a konverzia
    converter = ThermalConverter()
    thermal_img = converter.load(image_path) if not image_path.endswith('.png') else None

    if thermal_img:
        pil_image = converter.to_pil(thermal_img, target_size=(640, 480))
    else:
        pil_image = Image.open(image_path).convert('RGB')
        pil_image = pil_image.resize((640, 480), Image.LANCZOS)

    # Model 1
    print(f"\n--- {model1} ---")
    analyzer1 = ThermalVLMAnalyzer(VLMConfig(model=model1, temperature=0.1))
    result1 = analyzer1.analyze(pil_image, prompt=prompt)

    if result1.success:
        print(result1.to_formatted_text())
    else:
        print(f"CHYBA: {result1.error}")

    # Model 2
    print(f"\n--- {model2} ---")
    analyzer2 = ThermalVLMAnalyzer(VLMConfig(model=model2, temperature=0.1))
    result2 = analyzer2.analyze(pil_image, prompt=prompt)

    if result2.success:
        print(result2.to_formatted_text())
    else:
        print(f"CHYBA: {result2.error}")

    # Uloženie porovnania
    output = {
        "image": image_path,
        "model1": {"name": model1, "result": result1.to_dict()},
        "model2": {"name": model2, "result": result2.to_dict()},
    }

    with open("model_comparison.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("\nPorovnanie uložené do model_comparison.json")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Porovnanie dvoch VLM modelov")
    parser.add_argument("image", help="Obrázok na analýzu")
    parser.add_argument("--model1", default="qwen2-vl:7b", help="Prvý model")
    parser.add_argument("--model2", default="qwen2-vl-thermal", help="Druhý model (fine-tuned)")
    parser.add_argument("--prompt", help="Vlastný prompt")

    args = parser.parse_args()

    compare_models(args.image, args.model1, args.model2, args.prompt)