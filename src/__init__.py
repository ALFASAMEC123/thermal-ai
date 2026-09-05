"""
Termovízny spracovací systém - balíček.
"""

from .config_manager import get_config, ConfigManager, Settings
from .thermal_converter import (
    ThermalConverter,
    ThermalImage,
    ThermalFormat,
    NormalizeMethod,
    Colormap,
    create_converter
)
from .vlm_analyzer import (
    ThermalVLMAnalyzer,
    AnalysisResult,
    VLMConfig,
    OllamaVLMClient,
    create_analyzer
)
from .png_exporter import (
    ThermalPNGExporter,
    ExportConfig,
    create_exporter
)
from .pipeline import (
    ThermalProcessingPipeline,
    ProcessingResult,
    create_pipeline
)

__version__ = "1.0.0"
__all__ = [
    "get_config",
    "ConfigManager",
    "Settings",
    "ThermalConverter",
    "ThermalImage",
    "ThermalFormat",
    "NormalizeMethod",
    "Colormap",
    "create_converter",
    "ThermalVLMAnalyzer",
    "AnalysisResult",
    "VLMConfig",
    "OllamaVLMClient",
    "create_analyzer",
    "ThermalPNGExporter",
    "ExportConfig",
    "create_exporter",
    "ThermalProcessingPipeline",
    "ProcessingResult",
    "create_pipeline",
]