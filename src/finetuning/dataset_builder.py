"""
Priprava datasetu pre fine-tuning VLM modelu na termovíziu.
Vytvára JSONL formát kompatibilný s Unsloth, Axolotl, LLaMA-Factory.
"""

import json
import base64
import io
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from PIL import Image
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class ThermalSample:
    """Jeden vzorkový tréningový príklad."""
    image_path: str
    locality: str
    thermal_anomalies: str
    possible_cause: str
    recommendation: str
    temperature_stats: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class DatasetBuilder:
    """Budova datasetu pre fine-tuning."""

    def __init__(self, output_dir: Union[str, Path]):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.samples: List[ThermalSample] = []

    def add_sample(self, sample: ThermalSample):
        """Pridať vzorku do datasetu."""
        self.samples.append(sample)

    def add_from_analysis(self,
                         image_path: Union[str, Path],
                         analysis_result,
                         temperature_stats: Optional[Dict[str, Any]] = None):
        """Pridať vzorku z výsledku analýzy."""
        if hasattr(analysis_result, 'locality'):
            sample = ThermalSample(
                image_path=str(image_path),
                locality=analysis_result.locality,
                thermal_anomalies=analysis_result.thermal_anomalies,
                possible_cause=analysis_result.possible_cause,
                recommendation=analysis_result.recommendation,
                temperature_stats=temperature_stats
            )
        else:
            # Parsovanie z textu
            text = str(analysis_result)
            parsed = self._parse_analysis_text(text)
            sample = ThermalSample(
                image_path=str(image_path),
                **parsed,
                temperature_stats=temperature_stats
            )
        self.samples.append(sample)

    def _parse_analysis_text(self, text: str) -> Dict[str, str]:
        """Parsovanie textovej analýzy."""
        sections = {
            'locality': '',
            'thermal_anomalies': '',
            'possible_cause': '',
            'recommendation': ''
        }
        known = [
            ('locality', ['LOKALITA:']),
            ('thermal_anomalies', ['TEPLOTNÉ ANOMÁLIE:', 'TEPLOTNE ANOMALIE:']),
            ('possible_cause', ['MOŽNÁ PRÍČINA:', 'MOZNA PRICINA:']),
            ('recommendation', ['ODPORÚČANIE:', 'ODPORUCANIE:'])
        ]

        current_field = None
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue

            found = False
            for field, markers in known:
                for marker in markers:
                    if line.upper().startswith(marker):
                        if current_field:
                            sections[current_field] = sections[current_field].strip()
                        current_field = field
                        content = line[len(marker):].strip()
                        if content.startswith(':'):
                            content = content[1:].strip()
                        sections[field] = content
                        found = True
                        break
                if found:
                    break

            if not found and current_field:
                sections[current_field] += ' ' + line

        if current_field:
            sections[current_field] = sections[current_field].strip()

        return sections

    def build_jsonl(self,
                   output_file: str = "train.jsonl",
                   format_type: str = "unsloth",
                   include_image_base64: bool = False,
                   system_prompt: Optional[str] = None) -> Path:
        """
        Vytvorenie JSONL súboru pre tréning.

        Args:
            output_file: Názov výstupného súboru
            format_type: 'unsloth', 'axolotl', 'llamafactory', 'sharegpt'
            include_image_base64: Či zakódovať obrázok do base64 (pre niektoré formáty)
            system_prompt: Vlastný systémový prompt

        Returns:
            Cesta k vytvorenému súboru
        """
        output_path = self.output_dir / output_file

        default_system = """Si expert na termovíziu a prediktívnu údržbu. Analyzuj termovízne snímky a poskytuj podrobnú technickú analýzu v štruktúrovanom formáte."""

        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in self.samples:
                record = self._create_record(sample, format_type, include_image_base64, 
                                           system_prompt or default_system)
                f.write(json.dumps(record, ensure_ascii=False) + '\n')

        logger.info(f"Dataset uložený: {output_path} ({len(self.samples)} vzoriek)")
        return output_path

    def _create_record(self, sample: ThermalSample, format_type: str,
                      include_image: bool, system_prompt: str) -> Dict[str, Any]:
        """Vytvorenie jedného záznamu podľa formátu."""

        # Formátovanie odpovede
        assistant_response = self._format_response(sample)

        # Užívateľský prompt
        user_prompt = "Analyzuj tento termovízny snímok a poskytni podrobnú analýzu v štruktúrovanom formáte."

        if format_type == "unsloth":
            # Unsloth/Alpaca format
            return {
                "instruction": user_prompt,
                "input": "",
                "output": assistant_response,
                "system": system_prompt,
                "image": sample.image_path if not include_image else self._encode_image(sample.image_path)
            }
        elif format_type == "axolotl":
            # Axolotl chat format
            return {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": assistant_response}
                ],
                "images": [sample.image_path]
            }
        elif format_type == "llamafactory":
            # LLaMA-Factory format
            return {
                "conversations": [
                    {"from": "human", "value": f"<image>{user_prompt}"},
                    {"from": "gpt", "value": assistant_response}
                ],
                "images": [sample.image_path]
            }
        elif format_type == "sharegpt":
            # ShareGPT format
            return {
                "id": f"thermal_{len(self.samples)}",
                "image": sample.image_path,
                "conversations": [
                    {"from": "human", "value": user_prompt},
                    {"from": "gpt", "value": assistant_response}
                ]
            }
        else:
            raise ValueError(f"Unknown format: {format_type}")

    def _format_response(self, sample: ThermalSample) -> str:
        """Formátovanie odpovede do štruktúrovaného textu."""
        parts = []
        if sample.locality:
            parts.append(f"LOKALITA: {sample.locality}")
        if sample.thermal_anomalies:
            parts.append(f"TEPLOTNÉ ANOMÁLIE: {sample.thermal_anomalies}")
        if sample.possible_cause:
            parts.append(f"MOŽNÁ PRÍČINA: {sample.possible_cause}")
        if sample.recommendation:
            parts.append(f"ODPORÚČANIE: {sample.recommendation}")
        return "\n".join(parts)

    def _encode_image(self, image_path: str) -> str:
        """Kódovanie obrázku do base64."""
        try:
            with Image.open(image_path) as img:
                buffer = io.BytesIO()
                img.save(buffer, format='PNG')
                return base64.b64encode(buffer.getvalue()).decode('utf-8')
        except Exception as e:
            logger.warning(f"Failed to encode image {image_path}: {e}")
            return ""

    def split_dataset(self, train_ratio: float = 0.8, val_ratio: float = 0.1,
                     test_ratio: float = 0.1, seed: int = 42) -> Tuple[Path, Path, Path]:
        """Rozdelenie datasetu na train/val/test."""
        import random
        random.seed(seed)
        indices = list(range(len(self.samples)))
        random.shuffle(indices)

        n_train = int(len(self.samples) * train_ratio)
        n_val = int(len(self.samples) * val_ratio)

        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]

        # Vytvorenie pod-datasetov
        train_builder = DatasetBuilder(self.output_dir / "train")
        val_builder = DatasetBuilder(self.output_dir / "val")
        test_builder = DatasetBuilder(self.output_dir / "test")

        for i in train_indices:
            train_builder.samples.append(self.samples[i])
        for i in val_indices:
            val_builder.samples.append(self.samples[i])
        for i in test_indices:
            test_builder.samples.append(self.samples[i])

        train_path = train_builder.build_jsonl("train.jsonl")
        val_path = val_builder.build_jsonl("val.jsonl")
        test_path = test_builder.build_jsonl("test.jsonl")

        return train_path, val_path, test_path

    def create_from_directory(self,
                             images_dir: Union[str, Path],
                             annotations_file: Union[str, Path],
                             image_ext: str = ".png") -> int:
        """
        Vytvorenie datasetu z adresára s obrázkami a anotačným súborom.

        Anotačný súbor (JSON):
        {
            "image1.png": {
                "locality": "...",
                "thermal_anomalies": "...",
                "possible_cause": "...",
                "recommendation": "..."
            }
        }
        """
        images_dir = Path(images_dir)
        with open(annotations_file, 'r', encoding='utf-8') as f:
            annotations = json.load(f)

        count = 0
        for img_name, ann in annotations.items():
            img_path = images_dir / img_name
            if not img_path.exists():
                # Skús s inou príponou
                for ext in ['.png', '.jpg', '.jpeg', '.tiff', '.tif']:
                    test_path = images_dir / (Path(img_name).stem + ext)
                    if test_path.exists():
                        img_path = test_path
                        break

            if img_path.exists():
                sample = ThermalSample(
                    image_path=str(img_path),
                    locality=ann.get('locality', ''),
                    thermal_anomalies=ann.get('thermal_anomalies', ''),
                    possible_cause=ann.get('possible_cause', ''),
                    recommendation=ann.get('recommendation', ''),
                    metadata=ann.get('metadata')
                )
                self.add_sample(sample)
                count += 1
            else:
                logger.warning(f"Obrázok nenájdený: {img_name}")

        return count


def create_dataset_builder(output_dir: Union[str, Path]) -> DatasetBuilder:
    """Factory funkcia."""
    return DatasetBuilder(output_dir)


# CLI
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Dataset builder pre fine-tuning")
    parser.add_argument("--images", required=True, help="Adresár s obrázkami")
    parser.add_argument("--annotations", required=True, help="JSON s anotaciami")
    parser.add_argument("--output", required=True, help "Výstupný adresár")
    parser.add_argument("--format", default="unsloth", choices=["unsloth", "axolotl", "llamafactory", "sharegpt"])

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    builder = create_dataset_builder(args.output)
    count = builder.create_from_directory(args.images, args.annotations)
    builder.build_jsonl(format_type=args.format)
    print(f"Spracovaných {count} vzoriek")