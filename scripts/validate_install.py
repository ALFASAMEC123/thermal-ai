#!/usr/bin/env python3
"""
Testovací skript pre validáciu inštalácie.
"""

import sys
import subprocess
from pathlib import Path

# Pridaj src do path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_imports():
    """Test importov všetkých modulov."""
    print("Testovanie importov...")

    try:
        from src import (
            create_pipeline,
            create_converter,
            create_analyzer,
            create_exporter,
            ThermalConverter,
            ThermalImage,
            ThermalVLMAnalyzer,
            ThermalPNGExporter,
            ThermalProcessingPipeline,
        )
        print("  ✓ Hlavné moduly")
    except ImportError as e:
        print(f"  ✗ Hlavné moduly: {e}")
        return False

    try:
        from src.config_manager import ConfigManager, get_config, Settings
        print("  ✓ Config manager")
    except ImportError as e:
        print(f"  ✗ Config manager: {e}")
        return False

    try:
        from src.thermal_converter import ThermalFormat, NormalizeMethod, Colormap
        print("  ✓ Thermal converter enums")
    except ImportError as e:
        print(f"  ✗ Thermal converter enums: {e}")
        return False

    try:
        from src.vlm_analyzer import VLMConfig, AnalysisResult, OllamaVLMClient
        print("  ✓ VLM analyzer")
    except ImportError as e:
        print(f"  ✗ VLM analyzer: {e}")
        return False

    try:
        from src.png_exporter import ExportConfig
        print("  ✓ PNG exporter")
    except ImportError as e:
        print(f"  ✗ PNG exporter: {e}")
        return False

    try:
        from src.finetuning import DatasetBuilder, ThermalTrainer, FinetuningConfig
        print("  ✓ Fine-tuning moduly")
    except ImportError as e:
        print(f"  ✗ Fine-tuning moduly: {e}")
        return False

    return True


def test_dependencies():
    """Test dostupnosti závislostí."""
    print("\nTestovanie závislostí...")

    deps = [
        ("numpy", "numpy"),
        ("PIL", "PIL"),
        ("cv2", "cv2"),
        ("pandas", "pandas"),
        ("yaml", "yaml"),
        ("requests", "requests"),
        ("tqdm", "tqdm"),
        ("tifffile", "tifffile"),
    ]

    all_ok = True
    for name, import_name in deps:
        try:
            __import__(import_name)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - CHÝBA")
            all_ok = False

    # Voliteľné
    optional = [
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("unsloth", "unsloth"),
        ("trl", "trl"),
        ("datasets", "datasets"),
        ("peft", "peft"),
        ("bitsandbytes", "bitsandbytes"),
    ]

    for name, import_name in optional:
        try:
            __import__(import_name)
            print(f"  ✓ {name} (voliteľné)")
        except ImportError:
            print(f"  - {name} (voliteľné, nenainštalované)")

    return all_ok


def test_ollama():
    """Test pripojenia k Ollama."""
    print("\nTestovanie Ollama...")

    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            model_names = [m['name'] for m in models]
            print(f"  ✓ Ollama beží, dostupné modely: {', '.join(model_names)}")

            # Skontroluj VLM modely
            vlm_models = [m for m in model_names if any(x in m for x in ['qwen', 'llava', 'bakllava'])]
            if vlm_models:
                print(f"  ✓ VLM modely: {', '.join(vlm_models)}")
            else:
                print("  ⚠ Žiadne VLM modely (qwen2-vl, llava, bakllava)")
                print("    Spusti: ollama pull qwen2-vl:7b")
            return True
        else:
            print(f"  ✗ Ollama odpovedá chybou: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("  ✗ Ollama nebeží alebo nie je dostupné na localhost:11434")
        print("    Spusti: ollama serve")
        return False
    except Exception as e:
        print(f"  ✗ Chyba: {e}")
        return False


def test_exiftool():
    """Test exiftool pre .irb súbory."""
    print("\nTestovanie exiftool...")

    try:
        result = subprocess.run(["exiftool", "-ver"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"  ✓ exiftool verzia: {result.stdout.strip()}")
            return True
        else:
            print("  ✗ exiftool nebeží správne")
            return False
    except FileNotFoundError:
        print("  ✗ exiftool nenainštalované")
        print("    Ubuntu/Debian: sudo apt-get install libimage-exiftool-perl")
        return False
    except Exception as e:
        print(f"  ✗ Chyba: {e}")
        return False


def test_fonts():
    """Test dostupnosti fontov."""
    print("\nTestovanie fontov...")

    try:
        from PIL import ImageFont
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 14)
        print("  ✓ DejaVuSansMono dostupný")
        return True
    except Exception:
        try:
            font = ImageFont.load_default()
            print("  ⚠ Používa sa default font (DejaVu nenájdený)")
            return True
        except Exception as e:
            print(f"  ✗ Žiadne fonty: {e}")
            return False


def test_config():
    """Test konfigurácie."""
    print("\nTestovanie konfigurácie...")

    try:
        from src.config_manager import ConfigManager
        config = ConfigManager()
        settings = config.load("config/settings.yaml")
        print(f"  ✓ Konfigurácia načítaná")
        print(f"    Ollama model: {settings.ollama.model}")
        print(f"    VLM host: {settings.ollama.host}")
        print(f"    Export canvas height: {settings.export.canvas_height}")
        return True
    except Exception as e:
        print(f"  ✗ Chyba konfigurácie: {e}")
        return False


def test_pipeline_creation():
    """Test vytvorenia pipeline."""
    print("\nTestovanie vytvorenia pipeline...")

    try:
        from src import create_pipeline
        pipeline = create_pipeline("config/settings.yaml")
        print("  ✓ Pipeline vytvorený")
        print(f"    Converter: {type(pipeline.converter).__name__}")
        print(f"    Analyzer: {type(pipeline.analyzer).__name__}")
        print(f"    Exporter: {type(pipeline.exporter).__name__}")
        return True
    except Exception as e:
        print(f"  ✗ Chyba pipeline: {e}")
        return False


def main():
    """Hlavná testovacia funkcia."""
    print("=" * 50)
    print("VALIDÁCIA INŠTALÁCIE TERMOVÍZNEHO SYSTÉMU")
    print("=" * 50)

    tests = [
        ("Importy", test_imports),
        ("Závislosti", test_dependencies),
        ("Konfigurácia", test_config),
        ("Pipeline", test_pipeline_creation),
        ("Fonty", test_fonts),
        ("Exiftool", test_exiftool),
        ("Ollama", test_ollama),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"  ✗ {name}: výnimka {e}")
            results.append((name, False))

    print("\n" + "=" * 50)
    print("ZHRNUTIE")
    print("=" * 50)

    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status} - {name}")
        if not passed:
            all_passed = False

    print("=" * 50)
    if all_passed:
        print("✓ VŠETKY TESTY PREŠLI - Systém je pripravený!")
        return 0
    else:
        print("✗ NIEKTORÉ TESTY ZLYHALI - Oprav chyby pred použitím")
        return 1


if __name__ == "__main__":
    sys.exit(main())