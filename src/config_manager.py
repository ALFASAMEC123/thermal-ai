"""Konfiguračný manažér pre termovízny systém."""
import sys
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class OllamaConfig:
    host: str = "http://localhost:11434"
    model: str = "qwen2-vl:7b"
    timeout: int = 120
    temperature: float = 0.1
    top_p: float = 0.9


@dataclass
class ConversionConfig:
    input_formats: list = field(default_factory=lambda: [".irb", ".raw", ".csv", ".txt"])
    output_format: str = "PNG"
    normalize_method: str = "minmax"
    colormap: str = "inferno"
    target_size: list = field(default_factory=lambda: [640, 480])


@dataclass
class ExportConfig:
    canvas_height: int = 200
    font_size: int = 14
    font_family: str = "DejaVuSansMono"
    text_color: list = field(default_factory=lambda: [255, 255, 255])
    background_color: list = field(default_factory=lambda: [0, 0, 0])
    padding: int = 10
    line_spacing: float = 1.5


@dataclass
class FinetuningConfig:
    base_model: str = "Qwen/Qwen2-VL-7B-Instruct"
    output_dir: str = "./models/finetuned"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])
    learning_rate: float = 2e-4
    num_epochs: int = 3
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 2048
    warmup_steps: int = 50
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    fp16: bool = True
    bf16: bool = False
    gradient_checkpointing: bool = True


@dataclass
class ProcessingConfig:
    batch_size: int = 4
    max_workers: int = 2
    save_intermediate: bool = True
    overwrite_existing: bool = False


@dataclass
class Settings:
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    conversion: ConversionConfig = field(default_factory=ConversionConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    finetuning: FinetuningConfig = field(default_factory=FinetuningConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)


class ConfigManager:
    _instance: Optional['ConfigManager'] = None
    _settings: Optional[Settings] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._settings is None:
            self.load()

    def load(self, config_path: str = "config/settings.yaml") -> Settings:
        path = Path(config_path)
        
        # For PyInstaller frozen executable, also check next to executable
        if getattr(sys, 'frozen', False):
            exe_dir = Path(sys.executable).parent
            exe_config = exe_dir / "config" / "settings.yaml"
            if exe_config.exists():
                path = exe_config
        
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            self._settings = self._parse_config(data)
        else:
            self._settings = Settings()
            self.save(config_path)
        return self._settings

    def _parse_config(self, data: Dict[str, Any]) -> Settings:
        settings = Settings()
        if 'ollama' in data:
            settings.ollama = OllamaConfig(**data['ollama'])
        if 'conversion' in data:
            settings.conversion = ConversionConfig(**data['conversion'])
        if 'export' in data:
            settings.export = ExportConfig(**data['export'])
        if 'finetuning' in data:
            settings.finetuning = FinetuningConfig(**data['finetuning'])
        if 'processing' in data:
            settings.processing = ProcessingConfig(**data['processing'])
        return settings

    def save(self, config_path: str = "config/settings.yaml"):
        path = Path(config_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            'ollama': self._settings.ollama.__dict__,
            'conversion': self._settings.conversion.__dict__,
            'export': self._settings.export.__dict__,
            'finetuning': self._settings.finetuning.__dict__,
            'processing': self._settings.processing.__dict__,
        }
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

    @property
    def settings(self) -> Settings:
        return self._settings


def get_config() -> Settings:
    return ConfigManager().settings