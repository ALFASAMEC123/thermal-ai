"""
Fine-tuning skript pomocou Unsloth (rýchly a pamäťovo efektívny).
Podporuje QLoRA/LoRA pre Qwen2-VL, LLaVA a iné VLM modely.
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

import torch
from datasets import load_dataset, Dataset
from transformers import TrainingArguments, TrainerCallback
from trl import SFTTrainer

try:
    from unsloth import FastVisionModel
    from unsloth import is_bfloat16_supported
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False
    FastVisionModel = None

logger = logging.getLogger(__name__)


@dataclass
class FinetuningConfig:
    """Konfigurácia fine-tuningu."""
    # Model
    model_name: str = "Qwen/Qwen2-VL-7B-Instruct"
    max_seq_length: int = 2048
    dtype: Optional[torch.dtype] = None  # Auto
    load_in_4bit: bool = True
    load_in_8bit: bool = False

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "v_proj", "k_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    use_rslora: bool = True
    use_gradient_checkpointing: str = "unsloth"

    # Training
    output_dir: str = "./models/finetuned"
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 50
    num_train_epochs: int = 3
    learning_rate: float = 2e-4
    fp16: bool = not is_bfloat16_supported() if UNSLOTH_AVAILABLE else True
    bf16: bool = is_bfloat16_supported() if UNSLOTH_AVAILABLE else False
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    evaluation_strategy: str = "steps"
    save_total_limit: int = 3
    optim: str = "adamw_8bit"
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    seed: int = 42
    remove_unused_columns: bool = False

    # Data
    dataset_text_field: str = "output"
    dataset_num_proc: int = 2
    packing: bool = False


class ThermalTrainer:
    """Tréner pre termovízne VLM modely."""

    def __init__(self, config: FinetuningConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None

    def load_model(self):
        """Načítanie modelu s Unsloth optimalizáciami."""
        if not UNSLOTH_AVAILABLE:
            raise RuntimeError("Unsloth nie je nainštalované. Inštaluj: pip install unsloth")

        logger.info(f"Načítavam model: {self.config.model_name}")

        self.model, self.tokenizer = FastVisionModel.from_pretrained(
            self.config.model_name,
            max_seq_length=self.config.max_seq_length,
            dtype=self.config.dtype,
            load_in_4bit=self.config.load_in_4bit,
            load_in_8bit=self.config.load_in_8bit,
        )

        # LoRA konfigurácia
        self.model = FastVisionModel.get_peft_model(
            self.model,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules,
            use_rslora=self.config.use_rslora,
            use_gradient_checkpointing=self.config.use_gradient_checkpointing,
        )

        logger.info("Model načítaný s LoRA adapters")
        self._print_trainable_parameters()

    def _print_trainable_parameters(self):
        """Výpis trénovateľných parametrov."""
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Trénovateľné parametre: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    def prepare_dataset(self, data_path: Union[str, Path], split: str = "train") -> Dataset:
        """Príprava datasetu pre tréning."""
        data_path = Path(data_path)

        if data_path.suffix == '.jsonl':
            dataset = load_dataset('json', data_files=str(data_path), split=split)
        elif data_path.is_dir():
            dataset = load_dataset(str(data_path), split=split)
        else:
            raise ValueError(f"Nepodporovaný formát datasetu: {data_path}")

        # Formátovanie pre VLM
        def format_example(example):
            # Unsloth očakáva pole 'text' alebo určené dataset_text_field
            if 'messages' in example:
                # Chat format -> text
                text = self._format_messages(example['messages'])
            elif 'conversations' in example:
                # ShareGPT/LLaMA-Factory format
                text = self._format_conversations(example['conversations'])
            elif 'instruction' in example and 'output' in example:
                # Alpaca format
                text = self._format_alpaca(example)
            else:
                text = example.get(self.config.dataset_text_field, '')

            return {"text": text}

        dataset = dataset.map(format_example, num_proc=self.config.dataset_num_proc)
        return dataset

    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """Formátovanie chat správ do textu."""
        parts = []
        for msg in messages:
            role = msg['role']
            content = msg['content']
            if role == 'system':
                parts.append(f"<|system|>\n{content}")
            elif role == 'user':
                parts.append(f"<|user|>\n{content}")
            elif role == 'assistant':
                parts.append(f"<|assistant|>\n{content}")
        return "\n".join(parts) + "<|end|>"

    def _format_conversations(self, conversations: List[Dict[str, str]]) -> str:
        """Formátovanie ShareGPT konverzácií."""
        parts = []
        for conv in conversations:
            frm = conv['from']
            val = conv['value']
            if frm == 'human':
                parts.append(f"<|user|>\n{val}")
            elif frm == 'gpt':
                parts.append(f"<|assistant|>\n{val}")
            elif frm == 'system':
                parts.append(f"<|system|>\n{val}")
        return "\n".join(parts) + "<|end|>"

    def _format_alpaca(self, example: Dict[str, str]) -> str:
        """Formátovanie Alpaca formátu."""
        instruction = example.get('instruction', '')
        input_text = example.get('input', '')
        output = example.get('output', '')

        if input_text:
            user_content = f"{instruction}\n\n{input_text}"
        else:
            user_content = instruction

        return f"<|user|>\n{user_content}\n<|assistant|>\n{output}<|end|>"

    def train(self,
              train_dataset: Dataset,
              eval_dataset: Optional[Dataset] = None,
              resume_from_checkpoint: Optional[str] = None):
        """Spustenie tréningu."""
        if self.model is None:
            self.load_model()

        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            warmup_steps=self.config.warmup_steps,
            num_train_epochs=self.config.num_train_epochs,
            learning_rate=self.config.learning_rate,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            evaluation_strategy=self.config.evaluation_strategy,
            save_total_limit=self.config.save_total_limit,
            optim=self.config.optim,
            weight_decay=self.config.weight_decay,
            lr_scheduler_type=self.config.lr_scheduler_type,
            seed=self.config.seed,
            remove_unused_columns=self.config.remove_unused_columns,
            report_to="none",  # Disable wandb/tensorboard
            dataloader_pin_memory=False,
        )

        self.trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            dataset_text_field="text",
            max_seq_length=self.config.max_seq_length,
            args=training_args,
            packing=self.config.packing,
        )

        logger.info("Začínam tréning...")
        self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        # Uloženie finálneho modelu
        self.save_model()

    def save_model(self, output_dir: Optional[str] = None):
        """Uloženie modelu."""
        save_dir = output_dir or self.config.output_dir
        logger.info(f"Ukladam model do: {save_dir}")

        # Uloženie LoRA adapterov
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)

        # Uloženie v GGUF formáte pre Ollama (voliteľné)
        try:
            self.model.save_pretrained_gguf(
                save_dir + "_gguf",
                quantization_method="q4_k_m"
            )
            logger.info("GGUF model uložený pre Ollama")
        except Exception as e:
            logger.warning(f"GGUF export zlyhal: {e}")

    def merge_and_save(self, output_dir: Optional[str] = None):
        """Zlúčenie LoRA adapterov do base modelu a uloženie."""
        save_dir = output_dir or (self.config.output_dir + "_merged")
        logger.info(f"Zlúčam a ukladalem merged model do: {save_dir}")

        self.model = self.model.merge_and_unload()
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)

    def push_to_hub(self, repo_id: str, token: Optional[str] = None):
        """Nahranie modelu na Hugging Face Hub."""
        self.model.push_to_hub(repo_id, token=token)
        self.tokenizer.push_to_hub(repo_id, token=token)


def create_trainer(config: Optional[FinetuningConfig] = None) -> ThermalTrainer:
    """Factory funkcia."""
    return ThermalTrainer(config or FinetuningConfig())


# Axolotl konfigurácia generátor
def generate_axolotl_config(config: FinetuningConfig, 
                           train_data: str, 
                           val_data: str,
                           output_path: str = "axolotl_config.yaml") -> str:
    """Generovanie Axolotl YAML konfigurácie."""
    import yaml

    axolotl_config = {
        "base_model": config.model_name,
        "model_type": "AutoModelForVision2Seq",
        "tokenizer_type": "AutoTokenizer",
        "load_in_4bit": config.load_in_4bit,
        "load_in_8bit": config.load_in_8bit,
        "adapter": "lora",
        "lora_r": config.lora_r,
        "lora_alpha": config.lora_alpha,
        "lora_dropout": config.lora_dropout,
        "lora_target_modules": config.target_modules,
        "sequence_len": config.max_seq_length,
        "sample_packing": config.packing,
        "pad_to_sequence_len": True,
        "datasets": [
            {
                "path": train_data,
                "type": "chat_template",
                "chat_template": "qwen2-vl",
                "data_files": [train_data]
            }
        ],
        "val_set_size": 0.1,
        "output_dir": config.output_dir,
        "num_epochs": config.num_train_epochs,
        "micro_batch_size": config.per_device_train_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "optimizer": config.optim,
        "lr_scheduler": config.lr_scheduler_type,
        "warmup_steps": config.warmup_steps,
        "weight_decay": config.weight_decay,
        "fp16": config.fp16,
        "bf16": config.bf16,
        "logging_steps": config.logging_steps,
        "save_steps": config.save_steps,
        "eval_steps": config.eval_steps,
        "save_total_limit": config.save_total_limit,
        "gradient_checkpointing": True,
        "flash_attention": True,
    }

    with open(output_path, 'w') as f:
        yaml.dump(axolotl_config, f, default_flow_style=False)

    logger.info(f"Axolotl config uložená: {output_path}")
    return output_path


# CLI
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tuning VLM pre termovíziu (Unsloth)")
    parser.add_argument("--train-data", required=True, help="Tréningové dáta (JSONL)")
    parser.add_argument("--val-data", help="Validačné dáta (JSONL)")
    parser.add_argument("--model", default="Qwen/Qwen2-VL-7B-Instruct", help="Base model")
    parser.add_argument("--output", default="./models/finetuned", help="Výstupný adresár")
    parser.add_argument("--epochs", type=int, default=3, help="Počet epoch")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--max-seq-len", type=int, default=2048, help="Max sequence length")
    parser.add_argument("--resume", help="Resume from checkpoint")
    parser.add_argument("--merge", action="store_true", help="Merge LoRA after training")
    parser.add_argument("--push-hub", help "Push to HF Hub (repo_id)")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if not UNSLOTH_AVAILABLE:
        print("CHYBA: Unsloth nie je nainštalované", file=sys.stderr)
        print("Inštalácia: pip install unsloth", file=sys.stderr)
        sys.exit(1)

    config = FinetuningConfig(
        model_name=args.model,
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        max_seq_length=args.max_seq_len,
    )

    trainer = ThermalTrainer(config)
    trainer.load_model()

    train_dataset = trainer.prepare_dataset(args.train_data)
    eval_dataset = trainer.prepare_dataset(args.val_data) if args.val_data else None

    trainer.train(train_dataset, eval_dataset, resume_from_checkpoint=args.resume)

    if args.merge:
        trainer.merge_and_save()

    if args.push_hub:
        trainer.push_to_hub(args.push_hub)

    print("Tréning dokončený!")