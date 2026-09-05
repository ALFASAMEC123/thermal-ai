"""
Integrácia s lokálnym Vision-Language Modelom cez Ollama API.
Podporuje Qwen2-VL, LLaVA a iné modely kompatibilné s Ollama.
"""

import base64
import json
import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass, field
from enum import Enum
import requests
from PIL import Image
import io

logger = logging.getLogger(__name__)


class VLMModel(Enum):
    QWEN2_VL_7B = "qwen2-vl:7b"
    QWEN2_VL_2B = "qwen2-vl:2b"
    LLAVA_7B = "llava:7b"
    LLAVA_13B = "llava:13b"
    LLAVA_34B = "llava:34b"
    BAKLLAVA = "bakllava:7b"
    CUSTOM = "custom"


@dataclass
class VLMConfig:
    host: str = "http://localhost:11434"
    model: str = "qwen2-vl:7b"
    timeout: int = 120
    temperature: float = 0.1
    top_p: float = 0.9
    max_tokens: int = 2048
    keep_alive: str = "5m"


@dataclass
class AnalysisResult:
    """Výsledok analýzy termovízneho snímku."""
    locality: str = ""
    thermal_anomalies: str = ""
    possible_cause: str = ""
    recommendation: str = ""
    raw_response: str = ""
    success: bool = False
    error: Optional[str] = None
    processing_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'locality': self.locality,
            'thermal_anomalies': self.thermal_anomalies,
            'possible_cause': self.possible_cause,
            'recommendation': self.recommendation,
            'raw_response': self.raw_response,
            'success': self.success,
            'error': self.error,
            'processing_time': self.processing_time
        }

    def to_formatted_text(self) -> str:
        """Formátovaný text pre export do PNG."""
        lines = []
        if self.locality:
            lines.append(f"LOKALITA: {self.locality}")
        if self.thermal_anomalies:
            lines.append(f"TEPLOTNÉ ANOMÁLIE: {self.thermal_anomalies}")
        if self.possible_cause:
            lines.append(f"MOŽNÁ PRÍČINA: {self.possible_cause}")
        if self.recommendation:
            lines.append(f"ODPORÚČANIE: {self.recommendation}")
        return "\n".join(lines)


class OllamaVLMClient:
    """Klient pre komunikáciu s Ollama API."""

    def __init__(self, config: VLMConfig):
        self.config = config
        self.base_url = config.host.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({'Content-Type': 'application/json'})

    def check_connection(self) -> bool:
        """Overenie pripojenia k Ollama."""
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Ollama connection failed: {e}")
            return False

    def list_models(self) -> List[str]:
        """Zoznam dostupných modelov."""
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                data = response.json()
                return [m['name'] for m in data.get('models', [])]
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
        return []

    def pull_model(self, model: str) -> bool:
        """Sťahovanie modelu."""
        try:
            response = self.session.post(
                f"{self.base_url}/api/pull",
                json={"name": model},
                timeout=300
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to pull model {model}: {e}")
            return False

    def encode_image(self, image: Union[Image.Image, str, Path, bytes]) -> str:
        """Kódovanie obrázku do base64."""
        if isinstance(image, (str, Path)):
            with open(image, 'rb') as f:
                img_bytes = f.read()
        elif isinstance(image, bytes):
            img_bytes = image
        elif isinstance(image, Image.Image):
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            img_bytes = buffer.getvalue()
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        return base64.b64encode(img_bytes).decode('utf-8')

    def generate(self, prompt: str, images: List[Union[Image.Image, str, Path, bytes]],
                 **kwargs) -> Dict[str, Any]:
        """Generovanie odpovede od VLM."""
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "images": [self.encode_image(img) for img in images],
            "stream": False,
            "options": {
                "temperature": kwargs.get('temperature', self.config.temperature),
                "top_p": kwargs.get('top_p', self.config.top_p),
                "num_predict": kwargs.get('max_tokens', self.config.max_tokens),
            },
            "keep_alive": self.config.keep_alive
        }

        try:
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            raise TimeoutError(f"Request timed out after {self.config.timeout}s")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ollama API error: {e}")

    def chat(self, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """Chat completion endpoint (pre modely podporujúce chat)."""
        payload = {
            "model": self.config.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": kwargs.get('temperature', self.config.temperature),
                "top_p": kwargs.get('top_p', self.config.top_p),
                "num_predict": kwargs.get('max_tokens', self.config.max_tokens),
            },
            "keep_alive": self.config.keep_alive
        }

        try:
            response = self.session.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            raise TimeoutError(f"Request timed out after {self.config.timeout}s")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ollama API error: {e}")


# Šablóny promptov pre termovíznú analýzu
THERMAL_ANALYSIS_PROMPT = """Si expert na termovíziu a prediktívnu údržbu. Analyzuj termovízny snímok a poskytni podrobnú analýzu v nasledujúcom formáte:

**LOKALITA:** [Presná lokalita/oblasť na snímku - napr. "Motorový kompartment, ľavý bok, výmenník tepla", "Elektrická rozvádzková škriňa, stredná časť", "Izolácia potrubia, úsek 15-20m"]

**TEPLOTNÉ ANOMÁLIE:** [Popis anomálií s konkrétnymi teplotami - napr. "Lokálny prehrev na (x=320, y=245) s teplotou 87.3°C, ozadí 35.2°C, rozdiel +52.1°C. Oblast 15x12 px s teplotou >80°C.", "Rovnomerný prehrev celej plochy na 65°C pri očakávanych 45°C."]

**MOŽNÁ PRÍČINA:** [Technická analýza možných príčin - napr. "Porucha ložiska motoru (vibrácie + prehrev)", "Pobrežná korózia kontaktu v rozvádzke", "Poškodená izolačná vrstva potrubia", "Preťaženie fasádneho kábela"]

**ODPORÚČANIE:** [Konkrétne kroky - napr. "Okamžitá údržba: výmena ložiska do 24h", "Plánovaná kontrola: uťahovanie spojov do 1 týždňa", "Monitorovanie: sledovanie teploty každé 4h", "Žiadna akcia: teplota v normálnom rozsahu"]

Pravidlá:
- Buď presný a technicky korektný
- Uvádzaj konkrétne teploty a polohy
- Rozlišuj medzi kritickými a varovnými stavmi
- Ak nie sú anomálie, uveď "Žiadne významné anomálie nenašlé"
- Odpovedz SLOVENSKY
"""

THERMAL_ANALYSIS_PROMPT_SHORT = """Analyzuj termovízny snímok a vypíš výsledok v tomto formáte:

LOKALITA: [lokalita]
TEPLOTNÉ ANOMÁLIE: [anomálie s teplotami]
MOŽNÁ PRÍČINA: [príčina]
ODPORÚČANIE: [odporúčanie]

Odpovedz SLOVENSKY, buď konkrétny."""


class ThermalVLMAnalyzer:
    """Analyzátor termovíznych snímkov pomocou VLM."""

    def __init__(self, config: VLMConfig):
        self.client = OllamaVLMClient(config)
        self.config = config

    def analyze(self, image: Union[Image.Image, str, Path, bytes],
                prompt: Optional[str] = None,
                temperature_data: Optional[Dict[str, Any]] = None) -> AnalysisResult:
        """Analýza termovízneho snímku."""
        start_time = time.time()

        if prompt is None:
            prompt = THERMAL_ANALYSIS_PROMPT

        # Pridanie teplotných metadát do promptu ak sú k dispozícii
        if temperature_data:
            meta_prompt = self._build_metadata_prompt(temperature_data)
            prompt = meta_prompt + "\n\n" + prompt

        try:
            result = self.client.generate(prompt, [image])
            raw_response = result.get('response', '')

            parsed = self._parse_response(raw_response)
            parsed.raw_response = raw_response
            parsed.success = True
            parsed.processing_time = time.time() - start_time

            return parsed

        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return AnalysisResult(
                success=False,
                error=str(e),
                processing_time=time.time() - start_time
            )

    def _build_metadata_prompt(self, temp_data: Dict[str, Any]) -> str:
        """Vytvorenie promptu s teplotnými metadátami."""
        lines = ["TEPLOVÉ METADÁTA:"]
        if 'global' in temp_data:
            g = temp_data['global']
            lines.append(f"  Globálny min: {g.get('min', 'N/A'):.1f}°C")
            lines.append(f"  Globálny max: {g.get('max', 'N/A'):.1f}°C")
            lines.append(f"  Priemerná teplota: {g.get('mean', 'N/A'):.1f}°C")

        for name, stats in temp_data.items():
            if name != 'global':
                lines.append(f"  {name}: min={stats.get('min', 'N/A'):.1f}°C, "
                           f"max={stats.get('max', 'N/A'):.1f}°C, "
                           f"mean={stats.get('mean', 'N/A'):.1f}°C")

        return "\n".join(lines)

    def _parse_response(self, response: str) -> AnalysisResult:
        """Parsovanie odpovede do štruktúrovaného formátu."""
        result = AnalysisResult()

        # Hľadanie sekcií
        sections = {
            'locality': ['LOKALITA:', 'LOKALITA'],
            'thermal_anomalies': ['TEPLOTNÉ ANOMÁLIE:', 'TEPLOTNE ANOMALIE:', 'ANOMÁLIE:'],
            'possible_cause': ['MOŽNÁ PRÍČINA:', 'MOZNA PRICINA:', 'PRÍČINA:', 'PRICINA:'],
            'recommendation': ['ODPORÚČANIE:', 'ODPORUCANIE:', 'ODPORÚČANIE']
        }

        for field_name, markers in sections.items():
            for marker in markers:
                idx = response.find(marker)
                if idx != -1:
                    start = idx + len(marker)
                    # Najdi koniec sekcie (ďalší marker alebo koniec)
                    end = len(response)
                    for other_markers in sections.values():
                        for om in other_markers:
                            oidx = response.find(om, start)
                            if oidx != -1 and oidx < end:
                                end = oidx
                    value = response[start:end].strip()
                    setattr(result, field_name, value)
                    break

        # Fallback: ak sa nepodarilo parsovať, ulož celú odpoveď
        if not any([result.locality, result.thermal_anomalies, 
                    result.possible_cause, result.recommendation]):
            result.raw_response = response
            # Skús rozparsovať jednoduchší formát
            lines = response.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('LOKALITA:'):
                    result.locality = line[9:].strip()
                elif line.startswith('TEPLOTNÉ ANOMÁLIE:') or line.startswith('TEPLOTNE ANOMALIE:'):
                    result.thermal_anomalies = line.split(':', 1)[1].strip()
                elif line.startswith('MOŽNÁ PRÍČINA:') or line.startswith('MOZNA PRICINA:'):
                    result.possible_cause = line.split(':', 1)[1].strip()
                elif line.startswith('ODPORÚČANIE:') or line.startswith('ODPORUCANIE:'):
                    result.recommendation = line.split(':', 1)[1].strip()

        return result

    def analyze_batch(self, images: List[Union[Image.Image, str, Path, bytes]],
                      prompt: Optional[str] = None,
                      temperature_data_list: Optional[List[Dict[str, Any]]] = None) -> List[AnalysisResult]:
        """Dávková analýza viacerých snímkov."""
        results = []
        for i, img in enumerate(images):
            temp_data = temperature_data_list[i] if temperature_data_list else None
            result = self.analyze(img, prompt, temp_data)
            results.append(result)
        return results


def create_analyzer(config: Optional[VLMConfig] = None) -> ThermalVLMAnalyzer:
    """Factory funkcia pre vytvorenie analyzátora."""
    if config is None:
        config = VLMConfig()
    return ThermalVLMAnalyzer(config)


# CLI pre testovanie
if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Test VLM analýzy")
    parser.add_argument("image", help="Obrázok na analýzu")
    parser.add_argument("--model", default="qwen2-vl:7b", help="Model v Ollama")
    parser.add_argument("--host", default="http://localhost:11434", help="Ollama host")
    parser.add_argument("--prompt", choices=["full", "short"], default="full", help="Typ promptu")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    config = VLMConfig(host=args.host, model=args.model)
    analyzer = ThermalVLMAnalyzer(config)

    if not analyzer.client.check_connection():
        print("CHYBA: Nemôžem sa pripojiť k Ollama", file=sys.stderr)
        sys.exit(1)

    print(f"Použitý model: {args.model}")
    prompt = THERMAL_ANALYSIS_PROMPT if args.prompt == "full" else THERMAL_ANALYSIS_PROMPT_SHORT

    result = analyzer.analyze(args.image, prompt)

    if result.success:
        print("\n=== VÝSLEDOK ANALÝZY ===")
        print(result.to_formatted_text())
        print(f"\nČas spracovania: {result.processing_time:.2f}s")
    else:
        print(f"CHYBA: {result.error}", file=sys.stderr)
        sys.exit(1)