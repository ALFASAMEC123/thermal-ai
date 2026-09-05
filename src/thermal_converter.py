"""
Konverzný modul pre termovízne súbory (.irb, .raw, .csv, .txt).
Podporuje rôzne formáty termovíznych kamier (FLIR, Optris, Thermoteknix, všeobecné RAW).
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Union, Dict, Any
from dataclasses import dataclass
from enum import Enum
import struct
import logging

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import tifffile
    TIFF_AVAILABLE = True
except ImportError:
    TIFF_AVAILABLE = False

logger = logging.getLogger(__name__)


class ThermalFormat(Enum):
    IRB_FLIR = "irb_flir"
    RAW_GENERIC = "raw_generic"
    RAW_OPTRIS = "raw_optris"
    RAW_THERMOTEK = "raw_thermotek"
    CSV_TEMPERATURE = "csv_temperature"
    TXT_TEMPERATURE = "txt_temperature"
    UNKNOWN = "unknown"


class NormalizeMethod(Enum):
    MINMAX = "minmax"
    ZSCORE = "zscore"
    HISTOGRAM_EQ = "histogram_eq"
    PERCENTILE = "percentile"


class Colormap(Enum):
    INFERNO = "inferno"
    JET = "jet"
    HOT = "hot"
    VIRIDIS = "viridis"
    PLASMA = "plasma"
    MAGMA = "magma"
    TURBO = "turbo"
    GRAY = "gray"


@dataclass
class ThermalImage:
    """Reprezentácia termovízneho snímku s metadátami."""
    data: np.ndarray  # Teplotné hodnoty v °C
    width: int
    height: int
    metadata: Dict[str, Any]
    format: ThermalFormat
    unit: str = "celsius"  # celsius, fahrenheit, kelvin, raw

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.height, self.width)

    @property
    def min_temp(self) -> float:
        return float(np.nanmin(self.data))

    @property
    def max_temp(self) -> float:
        return float(np.nanmax(self.data))

    @property
    def mean_temp(self) -> float:
        return float(np.nanmean(self.data))


class ThermalConverter:
    """Hlavná trieda na konverziu termovíznych súborov."""

    def __init__(self, config=None):
        self.config = config

    def detect_format(self, file_path: Union[str, Path]) -> ThermalFormat:
        """Automatická detekcia formátu súboru."""
        path = Path(file_path)
        suffix = path.suffix.lower()

        if suffix == '.irb':
            return self._detect_irb_format(path)
        elif suffix == '.raw':
            return self._detect_raw_format(path)
        elif suffix == '.csv':
            return ThermalFormat.CSV_TEMPERATURE
        elif suffix == '.txt':
            return ThermalFormat.TXT_TEMPERATURE
        elif suffix in ['.tiff', '.tif'] and TIFF_AVAILABLE:
            return self._detect_tiff_format(path)
        return ThermalFormat.UNKNOWN

    def _detect_irb_format(self, path: Path) -> ThermalFormat:
        """Detekcia FLIR .irb formátu."""
        try:
            with open(path, 'rb') as f:
                header = f.read(32)
                if b'FLIR' in header or b'flir' in header:
                    return ThermalFormat.IRB_FLIR
        except Exception:
            pass
        return ThermalFormat.UNKNOWN

    def _detect_raw_format(self, path: Path) -> ThermalFormat:
        """Detekcia RAW formátu na základe veľkosti a obsahu."""
        file_size = path.stat().st_size
        # Bežné rozlíšenia termovíznych kamier
        common_resolutions = [
            (640, 512), (640, 480), (384, 288), (320, 256),
            (160, 120), (80, 60), (1280, 1024), (1024, 768)
        ]

        for w, h in common_resolutions:
            # 2 bytes per pixel (uint16)
            if file_size == w * h * 2:
                return ThermalFormat.RAW_GENERIC
            # 4 bytes per pixel (float32)
            if file_size == w * h * 4:
                return ThermalFormat.RAW_GENERIC

        # Skuska čítania hlavičky pre známe formáty
        try:
            with open(path, 'rb') as f:
                header = f.read(128)
                if b'OPTRIS' in header or b'optris' in header:
                    return ThermalFormat.RAW_OPTRIS
                if b'THERMOTEK' in header or b'thermotek' in header:
                    return ThermalFormat.RAW_THERMOTEK
        except Exception:
            pass

        return ThermalFormat.RAW_GENERIC

    def _detect_tiff_format(self, path: Path) -> ThermalFormat:
        """Detekcia TIFF formátu s teplotnými dátami."""
        try:
            with tifffile.TiffFile(path) as tif:
                for page in tif.pages:
                    if page.tags.get('ImageDescription'):
                        desc = page.tags['ImageDescription'].value
                        if 'temperature' in str(desc).lower() or 'thermal' in str(desc).lower():
                            return ThermalFormat.RAW_GENERIC
        except Exception:
            pass
        return ThermalFormat.UNKNOWN

    def load(self, file_path: Union[str, Path], **kwargs) -> ThermalImage:
        """Univerzálne načítanie termovízneho súboru."""
        path = Path(file_path)
        fmt = self.detect_format(path)

        if fmt == ThermalFormat.IRB_FLIR:
            return self._load_irb_flir(path, **kwargs)
        elif fmt in [ThermalFormat.RAW_GENERIC, ThermalFormat.RAW_OPTRIS, ThermalFormat.RAW_THERMOTEK]:
            return self._load_raw(path, fmt, **kwargs)
        elif fmt == ThermalFormat.CSV_TEMPERATURE:
            return self._load_csv(path, **kwargs)
        elif fmt == ThermalFormat.TXT_TEMPERATURE:
            return self._load_txt(path, **kwargs)
        else:
            raise ValueError(f"Nepodporovaný formát: {fmt} pre súbor {path}")

    def _load_irb_flir(self, path: Path, **kwargs) -> ThermalImage:
        """Načítanie FLIR .irb formátu."""
        # FLIR .irb je vlastný formát, často vyžaduje FLIR SDK
        # Tu implementujeme základnú podporu cez exiftool alebo přímé čítanie
        try:
            import subprocess
            result = subprocess.run(
                ['exiftool', '-b', '-RawThermalImage', str(path)],
                capture_output=True, check=False
            )
            if result.returncode == 0 and result.stdout:
                # Raw thermal data extracted
                raw_data = np.frombuffer(result.stdout, dtype=np.uint16)
                # Reshape based on known FLIR resolutions
                for w, h in [(640, 512), (640, 480), (320, 256), (160, 120)]:
                    if len(raw_data) == w * h:
                        data = raw_data.reshape(h, w)
                        # Convert to temperature (FLIR specific formula)
                        temp_data = self._flir_raw_to_temp(data)
                        return ThermalImage(
                            data=temp_data,
                            width=w, height=h,
                            metadata={'source': 'FLIR IRB', 'extracted_by': 'exiftool'},
                            format=ThermalFormat.IRB_FLIR
                        )
        except Exception as e:
            logger.warning(f"exiftool extraction failed: {e}")

        # Fallback: skusíme čítať ako RAW
        return self._load_raw(path, ThermalFormat.RAW_GENERIC, **kwargs)

    def _flir_raw_to_temp(self, raw: np.ndarray, 
                          emissivity: float = 0.95,
                          distance: float = 1.0,
                          reflected_temp: float = 20.0,
                          atmospheric_temp: float = 20.0,
                          relative_humidity: float = 50.0) -> np.ndarray:
        """Konverzia FLIR raw hodnôt na teplotu (°C)."""
        # FLIR Planck R1/R2/B/F/O formula
        # Simplified version - for production use flirpy or official SDK
        R1 = 21106.77
        R2 = 0.012545258
        B = 1501.0
        F = 1.0
        O = -7340.0

        # Emissivity correction
        raw_corrected = raw / emissivity

        # Planck formula
        temp_k = B / np.log(R1 / (R2 * (raw_corrected + O)) + F)
        temp_c = temp_k - 273.15

        return temp_c

    def _load_raw(self, path: Path, fmt: ThermalFormat, 
                  width: Optional[int] = None,
                  height: Optional[int] = None,
                  dtype: str = 'uint16',
                  byte_order: str = 'little',
                  **kwargs) -> ThermalImage:
        """Načítanie RAW binárnych dát."""
        file_size = path.stat().st_size

        # Auto-detect dimensions if not provided
        if width is None or height is None:
            width, height = self._guess_dimensions(file_size, dtype)

        # Determine numpy dtype
        np_dtype = self._get_numpy_dtype(dtype, byte_order)
        bytes_per_pixel = np.dtype(np_dtype).itemsize

        expected_size = width * height * bytes_per_pixel
        if file_size != expected_size:
            logger.warning(f"File size mismatch: expected {expected_size}, got {file_size}")

        with open(path, 'rb') as f:
            raw_data = np.fromfile(f, dtype=np_dtype, count=width*height)

        if len(raw_data) < width * height:
            raise ValueError(f"Insufficient data: got {len(raw_data)}, expected {width*height}")

        data = raw_data.reshape(height, width)

        # Apply format-specific conversion
        if fmt == ThermalFormat.RAW_OPTRIS:
            data = self._optris_raw_to_temp(data)
        elif fmt == ThermalFormat.RAW_THERMOTEK:
            data = self._thermotek_raw_to_temp(data)
        else:
            # Generic: assume raw values are in 0.01°C or 0.1°C
            data = data.astype(np.float32) * 0.01  # Default scaling

        metadata = {
            'source_file': str(path),
            'format': fmt.value,
            'original_dtype': dtype,
            'byte_order': byte_order,
            'width': width,
            'height': height
        }

        return ThermalImage(
            data=data,
            width=width,
            height=height,
            metadata=metadata,
            format=fmt
        )

    def _guess_dimensions(self, file_size: int, dtype: str) -> Tuple[int, int]:
        """Odhad rozlíšenia na základe veľkosti súboru."""
        bytes_per_pixel = {'uint8': 1, 'uint16': 2, 'int16': 2, 'float32': 4, 'float64': 8}.get(dtype, 2)
        total_pixels = file_size // bytes_per_pixel

        common_resolutions = [
            (1280, 1024), (1024, 768), (640, 512), (640, 480),
            (384, 288), (320, 256), (160, 120), (80, 60)
        ]

        for w, h in common_resolutions:
            if w * h == total_pixels:
                return w, h

        # Fallback: assume square-ish
        side = int(np.sqrt(total_pixels))
        return side, side

    def _get_numpy_dtype(self, dtype: str, byte_order: str) -> np.dtype:
        """Získanie numpy dtype s byte order."""
        prefix = '<' if byte_order == 'little' else '>'
        dtype_map = {
            'uint8': 'u1', 'int8': 'i1',
            'uint16': 'u2', 'int16': 'i2',
            'uint32': 'u4', 'int32': 'i4',
            'float32': 'f4', 'float64': 'f8'
        }
        return np.dtype(prefix + dtype_map.get(dtype, 'u2'))

    def _optris_raw_to_temp(self, raw: np.ndarray) -> np.ndarray:
        """Konverzia Optris RAW na teplotu."""
        # Optris typically stores temperature in 0.1°C or 0.01°C
        return raw.astype(np.float32) * 0.1

    def _thermotek_raw_to_temp(self, raw: np.ndarray) -> np.ndarray:
        """Konverzia Thermoteknix RAW na teplotu."""
        return raw.astype(np.float32) * 0.01

    def _load_csv(self, path: Path, **kwargs) -> ThermalImage:
        """Načítanie CSV s teplotnými dátami."""
        import pandas as pd
        delimiter = kwargs.get('delimiter', ',')
        has_header = kwargs.get('header', True)

        df = pd.read_csv(path, delimiter=delimiter, header=0 if has_header else None)
        data = df.values.astype(np.float32)

        return ThermalImage(
            data=data,
            width=data.shape[1],
            height=data.shape[0],
            metadata={'source_file': str(path), 'format': 'csv'},
            format=ThermalFormat.CSV_TEMPERATURE
        )

    def _load_txt(self, path: Path, **kwargs) -> ThermalImage:
        """Načítanie TXT s teplotnými dátami."""
        delimiter = kwargs.get('delimiter', None)  # whitespace
        data = np.loadtxt(path, delimiter=delimiter, dtype=np.float32)

        if data.ndim == 1:
            # Try to reshape
            side = int(np.sqrt(len(data)))
            if side * side == len(data):
                data = data.reshape(side, side)
            else:
                data = data.reshape(1, -1)

        return ThermalImage(
            data=data,
            width=data.shape[1],
            height=data.shape[0],
            metadata={'source_file': str(path), 'format': 'txt'},
            format=ThermalFormat.TXT_TEMPERATURE
        )

    def normalize(self, thermal_img: ThermalImage, 
                  method: NormalizeMethod = NormalizeMethod.MINMAX,
                  percentile_range: Tuple[float, float] = (1, 99)) -> np.ndarray:
        """Normalizácia teplotných dát do rozsahu 0-255 pre vizualizáciu."""
        data = thermal_img.data.copy()
        valid_mask = ~np.isnan(data)

        if not np.any(valid_mask):
            return np.zeros_like(data, dtype=np.uint8)

        valid_data = data[valid_mask]

        if method == NormalizeMethod.MINMAX:
            vmin, vmax = valid_data.min(), valid_data.max()
        elif method == NormalizeMethod.PERCENTILE:
            vmin, vmax = np.percentile(valid_data, percentile_range)
        elif method == NormalizeMethod.ZSCORE:
            mean, std = valid_data.mean(), valid_data.std()
            vmin, vmax = mean - 3*std, mean + 3*std
        elif method == NormalizeMethod.HISTOGRAM_EQ:
            return self._histogram_equalize(data, valid_mask)
        else:
            vmin, vmax = valid_data.min(), valid_data.max()

        # Clip and normalize
        data = np.clip(data, vmin, vmax)
        normalized = ((data - vmin) / (vmax - vmin + 1e-8) * 255).astype(np.uint8)
        normalized[~valid_mask] = 0

        return normalized

    def _histogram_equalize(self, data: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Histogram equalization pre termovízne dáta."""
        if not CV2_AVAILABLE:
            logger.warning("OpenCV not available, using minmax instead")
            valid = data[mask]
            vmin, vmax = valid.min(), valid.max()
            return np.clip((data - vmin) / (vmax - vmin + 1e-8) * 255, 0, 255).astype(np.uint8)

        valid_data = data[mask].astype(np.uint8)
        eq = cv2.equalizeHist(valid_data)
        result = np.zeros_like(data, dtype=np.uint8)
        result[mask] = eq
        return result

    def apply_colormap(self, normalized: np.ndarray, 
                       colormap: Colormap = Colormap.INFERNO) -> np.ndarray:
        """Aplikácia farebnej mapy na normalizovaný obrázok."""
        if not CV2_AVAILABLE:
            # Fallback bez OpenCV - grayscale
            return np.stack([normalized]*3, axis=-1)

        cmap_map = {
            Colormap.INFERNO: cv2.COLORMAP_INFERNO,
            Colormap.JET: cv2.COLORMAP_JET,
            Colormap.HOT: cv2.COLORMAP_HOT,
            Colormap.VIRIDIS: cv2.COLORMAP_VIRIDIS,
            Colormap.PLASMA: cv2.COLORMAP_PLASMA,
            Colormap.MAGMA: cv2.COLORMAP_MAGMA,
            Colormap.TURBO: cv2.COLORMAP_TURBO,
            Colormap.GRAY: cv2.COLORMAP_BONE,
        }

        cv_cmap = cmap_map.get(colormap, cv2.COLORMAP_INFERNO)
        colored = cv2.applyColorMap(normalized, cv_cmap)
        return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    def resize(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Zmena veľkosti obrázku."""
        if not CV2_AVAILABLE:
            from PIL import Image
            pil_img = Image.fromarray(image)
            pil_img = pil_img.resize(target_size, Image.LANCZOS)
            return np.array(pil_img)

        return cv2.resize(image, target_size, interpolation=cv2.INTER_LANCZOS4)

    def to_pil(self, thermal_img: ThermalImage,
               normalize_method: NormalizeMethod = NormalizeMethod.MINMAX,
               colormap: Colormap = Colormap.INFERNO,
               target_size: Optional[Tuple[int, int]] = None) -> "Image.Image":
        """Konverzia ThermalImage na PIL Image."""
        if not PIL_AVAILABLE:
            raise RuntimeError("PIL/Pillow nie je nainštalované")

        normalized = self.normalize(thermal_img, normalize_method)
        colored = self.apply_colormap(normalized, colormap)

        if target_size:
            colored = self.resize(colored, target_size)

        return Image.fromarray(colored)

    def save_as_image(self, thermal_img: ThermalImage, output_path: Union[str, Path],
                      normalize_method: NormalizeMethod = NormalizeMethod.MINMAX,
                      colormap: Colormap = Colormap.INFERNO,
                      target_size: Optional[Tuple[int, int]] = None):
        """Uloženie termovízneho snímku ako obrázok."""
        pil_img = self.to_pil(thermal_img, normalize_method, colormap, target_size)
        pil_img.save(output_path)
        logger.info(f"Uložené: {output_path}")

    def get_temperature_stats(self, thermal_img: ThermalImage, 
                              regions: Optional[Dict[str, Tuple[int, int, int, int]]] = None) -> Dict[str, Any]:
        """Výpočet štatistík teploty pre celý obrázok alebo oblasti."""
        data = thermal_img.data
        stats = {
            'global': {
                'min': float(np.nanmin(data)),
                'max': float(np.nanmax(data)),
                'mean': float(np.nanmean(data)),
                'std': float(np.nanstd(data)),
                'median': float(np.nanmedian(data)),
            }
        }

        if regions:
            for name, (x, y, w, h) in regions.items():
                roi = data[y:y+h, x:x+w]
                valid = roi[~np.isnan(roi)]
                if len(valid) > 0:
                    stats[name] = {
                        'min': float(valid.min()),
                        'max': float(valid.max()),
                        'mean': float(valid.mean()),
                        'std': float(valid.std()),
                        'pixel_count': int(len(valid))
                    }

        return stats


def create_converter(config=None) -> ThermalConverter:
    """Factory funkcia pre vytvorenie konvertéra."""
    return ThermalConverter(config)


# CLI pre priamu konverziu
if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Konverzia termovíznych súborov")
    parser.add_argument("input", help="Vstupný súbor (.irb, .raw, .csv, .txt)")
    parser.add_argument("output", help="Výstupný súbor (.png, .jpg)")
    parser.add_argument("--width", type=int, help="Šírka pre RAW formát")
    parser.add_argument("--height", type=int, help="Výška pre RAW formát")
    parser.add_argument("--dtype", default="uint16", help="Dátový typ pre RAW")
    parser.add_argument("--normalize", default="minmax", choices=["minmax", "zscore", "histogram_eq", "percentile"])
    parser.add_argument("--colormap", default="inferno", choices=["inferno", "jet", "hot", "viridis", "plasma", "magma", "turbo", "gray"])
    parser.add_argument("--resize", nargs=2, type=int, metavar=("W", "H"), help="Cieľové rozlíšenie")
    parser.add_argument("--stats", action="store_true", help="Vypíš teplotné štatistiky")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    converter = ThermalConverter()
    thermal_img = converter.load(args.input, width=args.width, height=args.height, dtype=args.dtype)

    if args.stats:
        stats = converter.get_temperature_stats(thermal_img)
        print(f"Teplotné štatistiky:")
        for k, v in stats.items():
            print(f"  {k}: {v}")

    converter.save_as_image(
        thermal_img, args.output,
        normalize_method=NormalizeMethod(args.normalize),
        colormap=Colormap(args.colormap),
        target_size=tuple(args.resize) if args.resize else None
    )