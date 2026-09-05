# Termovízny Spracovací Systém s Lokálnym AI (VLM)

Kompletné riešenie pre automatizované spracovanie termovíznych snímkov (.irb, .raw) s integráciou lokálneho Vision-Language Modelu cez Ollama a možnosťou fine-tuningu.

## 🎯 Funkcie

- **Konverzia** .irb (FLIR), .raw (generické), .csv, .txt → PNG s teplotnými dátami
- **AI Analýza** cez lokálne VLM (Qwen2-VL, LLaVA) s štruktúrovaným výstupom
- **Export** do PNG s textovým overlayom (lokalita, anomálie, príčina, odporúčanie)
- **Fine-tuning** s LoRA/QLoRA (Unsloth, Axolotl) pre minimálnu chybovosť
- **Dávkové spracovanie** s paralelným spracovaním

## 📋 Požiadavky

### Systémové závislosti
```bash
# Ubuntu/Debian
sudo apt-get update && sudo apt-get install -y \
    libimage-exiftool-perl \
    python3-dev \
    build-essential

# pre OpenCV
sudo apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1-mesa-glx
```

### Python závislosti
```bash
# Základné
pip install -r requirements.txt

# Unsloth (pre rýchly fine-tuning)
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Axolotl (alternatíva)
pip install axolotl
```

### Ollama + VLM modely
```bash
# Inštalácia Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Sťahovanie modelov (vyber jeden)
ollama pull qwen2-vl:7b      # Odporúčaný - najlepší pre VLM úlohy
ollama pull llava:7b         # Alternatíva
ollama pull llava:13b        # Väčší, presnejší
ollama pull bakllava:7b      # Rýchlejší
```

## 🚀 Rýchly štart

### 1. Konfigurácia
```bash
# Skopíruj a upravi konfiguráciu
cp config/settings.yaml config/settings.local.yaml
# Uprav: ollama.host, ollama.model, cesty, atď.
```

### 2. Spracovanie jedného súboru
```bash
# Pomocou CLI
python -m src.pipeline data/raw/thermal_image.irb -o data/output/

# Alebo programovo
from src import create_pipeline
pipeline = create_pipeline("config/settings.yaml")
result = pipeline.process_file("data/raw/image.irb", "data/output/")
print(result.analysis.to_formatted_text())
```

### 3. Dávkové spracovanie adresára
```bash
python -m src.pipeline data/raw/ -o data/output/ --recursive --workers 4
```

### 4. Kontrola Ollama pripojenia
```bash
python -m src.pipeline --check-ollama
```

## 📁 Štruktúra projektu

```
.
├── config/
│   └── settings.yaml          # Hlavná konfigurácia
├── data/
│   ├── raw/                   # Vstupné .irb/.raw súbory
│   ├── processed/             # Medzivýsledky (PNG)
│   └── output/                # Finálne reporty s textom
├── models/
│   └── finetuned/             # Fine-tuned modely
├── src/
│   ├── config_manager.py      # Správa konfigurácie
│   ├── thermal_converter.py   # Konverzia .irb/.raw → PIL/OpenCV
│   ├── vlm_analyzer.py        # Ollama VLM klient + prompt engineering
│   ├── png_exporter.py        # Export do PNG s text overlay
│   ├── pipeline.py            # Hlavný pipeline
│   └── finetuning/
│       ├── dataset_builder.py # Priprava JSONL datasetu
│       └── train_unsloth.py   # Fine-tuning s Unsloth
├── scripts/
│   ├── prepare_dataset.py     # Skript na vytvorenie datasetu
│   └── train_model.py         # Spustenie fine-tuningu
├── tests/
└── requirements.txt
```

## 🔧 Konfigurácia (config/settings.yaml)

```yaml
ollama:
  host: "http://localhost:11434"
  model: "qwen2-vl:7b"
  timeout: 120
  temperature: 0.1       # Nízka pre konzistentné výstupy
  top_p: 0.9

conversion:
  input_formats: [".irb", ".raw", ".csv", ".txt"]
  normalize_method: "minmax"   # minmax, zscore, histogram_eq, percentile
  colormap: "inferno"          # inferno, jet, hot, viridis, plasma
  target_size: [640, 480]      # Rozlíšenie pre VLM

export:
  canvas_height: 200           # Výška textovej časti
  font_size: 14
  font_family: "DejaVuSansMono"
  text_color: [255, 255, 255]
  background_color: [0, 0, 0]

finetuning:
  base_model: "Qwen/Qwen2-VL-7B-Instruct"
  lora_r: 16
  lora_alpha: 32
  learning_rate: 2e-4
  num_epochs: 3
  batch_size: 2
```

## 🖼️ Podporované formáty termovíznych súborov

| Formát | Popis | Poznámky |
|--------|-------|----------|
| `.irb` | FLIR IRB | Vyžaduje `exiftool` na extrakciu raw dát |
| `.raw` | Generické RAW | Auto-detekcia rozlíšenia (640x512, 640x480, 384x288, ...) |
| `.raw` | Optris | Detekcia cez hlavičku |
| `.raw` | Thermoteknix | Detekcia cez hlavičku |
| `.csv` | Teplotná matica | Číselné hodnoty v °C |
| `.txt` | Teplotná matica | Medzerami/tabuľkou oddelené |
| `.tiff` | TIFF s metadátami | Ak obsahuje temperature tagy |

### Konverzia RAW súborov (špecifikácia rozlíšenia)
```bash
# Ak auto-detekcia zlyhá, zadaj rozlíšenie manuálne
python -m src.thermal_converter input.raw output.png --width 640 --height 512 --dtype uint16
```

## 🤖 AI Analýza - Šablóna promptu

Systém používa štruktúrovaný prompt pre konzistentné výstupy:

```
LOKALITA: [Presná lokalita na snímku]
TEPLOTNÉ ANOMÁLIE: [Popis anomálií s konkrétnymi teplotami a polohami]
MOŽNÁ PRÍČINA: [Technická analýza príčiny]
ODPORÚČANIE: [Konkrétne kroky na riešenie]
```

### Príklad výstupu:
```
LOKALITA: Motorový kompartment, ľavý bok, výmenník tepla
TEPLOTNÉ ANOMÁLIE: Lokálny prehrev na (x=320, y=245) s teplotou 87.3°C, ozadí 35.2°C, rozdiel +52.1°C. Oblast 15x12 px s teplotou >80°C.
MOŽNÁ PRÍČINA: Porucha ložiska motoru (kombinácia vibrácií a prehrevu)
ODPORÚČANIE: Okamžitá údržba: výmena ložiska do 24h,monitorovanie teploty každé 2h
```

## 📊 Export do PNG s textom

Vytvára report obrázok s:
- **Horná časť**: Originálny termovízny snímok s farebnou ládkou
- **Dolná časť**: Čierne plátno s formátovanou AI analýzou
- **Hlavička**: Časová pečiatka, zdrojový súbor, formát
- **Teplotná ládka**: Min/Max teploty s farebným gradientom

## 🎯 Fine-tuning - Príručka

### 1. Príprava datasetu

#### Z existujúcich analyzovaných snímkov:
```python
from src.finetuning import create_dataset_builder
from src import create_pipeline

builder = create_dataset_builder("data/finetune_dataset")
pipeline = create_pipeline()

# Spracuj súbory a zbieraj výsledky
for file in Path("data/raw").glob("*.irb"):
    result = pipeline.process_file(file)
    if result.success:
        builder.add_from_analysis(file, result.analysis, result.temperature_stats)

# Vytvor JSONL
builder.build_jsonl("train.jsonl", format_type="unsloth")
```

#### Z manuálnych anotácií (JSON):
```json
{
  "image1.png": {
    "locality": "Elektrická rozvádzka, stred",
    "thermal_anomalies": "Kontakt L1 prehrety na 95°C, ozadí 40°C",
    "possible_cause": "Voľný svorkovník, zvýšený kontaktový odpor",
    "recommendation": "Uťahať svorkovník, kontrolovať 24h"
  }
}
```

```bash
python -m src.finetuning.dataset_builder \
    --images data/annotated_images/ \
    --annotations data/annotations.json \
    --output data/finetune_dataset/ \
    --format unsloth
```

### 2. Tréning (Unsloth - odporúčané)

```bash
# Základný tréning
python -m src.finetuning.train_unsloth \
    --train-data data/finetune_dataset/train.jsonl \
    --val-data data/finetune_dataset/val.jsonl \
    --model Qwen/Qwen2-VL-7B-Instruct \
    --output models/finetuned/qwen2-vl-thermal \
    --epochs 3 \
    --batch-size 2 \
    --lr 2e-4 \
    --lora-r 16 \
    --lora-alpha 32

# S merge-ovaním LoRA do base modelu
python -m src.finetuning.train_unsloth ... --merge
```

### 3. Tréning (Axolotl - alternatíva)

```bash
# Generuj konfiguráciu
python -c "
from src.finetuning import generate_axolotl_config, FinetuningConfig
config = FinetuningConfig()
generate_axolotl_config(config, 'data/finetune_dataset/train.jsonl', 'data/finetune_dataset/val.jsonl', 'axolotl_config.yaml')
"

# Spusti tréning
axolotl train axolotl_config.yaml
```

### 4. Použitie fine-tuned modelu v Ollama

```bash
# Konvertuj do GGUF (automaticky sa robí pri tréningu s --merge)
# Potom vytvor Modelfile pre Ollama:

cat > Modelfile << 'EOF'
FROM ./models/finetuned/qwen2-vl-thermal_gguf/q4_k_m.gguf
TEMPLATE """{{- if .System }}
<|im_start|>system
{{ .System }}<|im_end|>
{{- end }}
{{- if .Prompt }}
<|im_start|>user
{{ .Prompt }}<|im_end|>
{{- end }}
<|im_start|>assistant
"""
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|im_start|>"
EOF

ollama create qwen2-vl-thermal -f Modelfile
```

### 5. Aktualizácia konfigurácie pre fine-tuned model
```yaml
# config/settings.yaml
ollama:
  model: "qwen2-vl-thermal"  # Tvoj fine-tuned model
```

## 💡 Tipy pre minimálnu chybovosť

### Dataset kvalita
- **Minimálne 100-200 kvalitných vzoriek** pre dobrý fine-tuning
- **Rôzne typy anomálií**: prehrev, chladnutie, rovnomerné/nerovnomerné
- **Rôzne zariadenia**: motory, transformátory, panely, potrubia, káble
- **Konzistentné formátovanie** - vždy rovnaká štruktúra výstupu

### Tréning parametre
```yaml
# Pre malé datasety (50-100 vzoriek)
lora_r: 8
lora_alpha: 16
num_epochs: 5
learning_rate: 1e-4

# Pre stredné datasety (200-500)
lora_r: 16
lora_alpha: 32
num_epochs: 3
learning_rate: 2e-4

# Pre veľké datasety (500+)
lora_r: 32
lora_alpha: 64
num_epochs: 2
learning_rate: 1e-4
```

### Validácia
```bash
# Test na hold-out sade
python -m src.pipeline data/test/ -o data/test_results/ --model qwen2-vl-thermal

# Porovnaj s pôvodným modelom
python scripts/compare_models.py --original qwen2-vl:7b --finetuned qwen2-vl-thermal
```

## 🐛 Riešenie problémov

### Ollama sa nepripojí
```bash
# Skontroluj či beží
ollama serve

# Skontroluj port
curl http://localhost:11434/api/tags
```

### CUDA Out of Memory
```yaml
# Zmenš batch size a zväčš gradient accumulation
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
# Alebo použij 4-bit kvantizáciu (load_in_4bit: true)
```

### Exiftool nenájdený pre .irb
```bash
sudo apt-get install libimage-exiftool-perl
# Alebo stiahni standalone: https://exiftool.org/
```

### Fonty pre export
```bash
# Inštalácia DejaVu fonts
sudo apt-get install fonts-dejavu-core
# Alebo nastav iný font v config.yaml
font_family: "LiberationMono"
```

## 📝 Licencia

MIT License - slobodné použitie aj komerčne.

## 🤝 Prispevanie

1. Forkni repozitár
2. Vytvor feature branch
3. Commitni zmeny
4. Vytvor Pull Request

## 📞 Podpora

Pre otázky a problémy vytvor Issue v repozitári.