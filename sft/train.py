import argparse

from datasets import load_from_disk
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer

from data import load_terminal_corpus


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SFT on Nemotron-Terminal-Corpus")

    # Model
    p.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3.5-4B")
    p.add_argument("--output_dir", type=str, default="./output")

    # Data
    p.add_argument(
        "--subsets",
        nargs="+",
        default=None,
        help="Dataset subsets to use (default: all four)",
    )
    p.add_argument("--sample_frac", type=float, default=None, help="Sub-sample fraction per subset")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--dataset_num_proc", type=int, default=8)
    p.add_argument(
        "--tokenized_dataset_path",
        type=str,
        nargs="+",
        default=None,
        help="If set, load pre-tokenized dataset(s) from these paths. "
        "Multiple paths will be concatenated.",
    )

    # Training
    p.add_argument("--num_gpus", type=int, default=8, help="Total GPU count (for grad accum calc)")
    p.add_argument("--max_length", type=int, default=32768)
    p.add_argument("--num_train_epochs", type=int, default=2)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--global_batch_size", type=int, default=128)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--logging_steps", type=float, default=0.01, help="<1 = ratio of total steps")
    p.add_argument("--save_steps", type=float, default=0.05, help="<1 = ratio of total steps")
    p.add_argument("--packing", action="store_true", default=False)
    p.add_argument("--deepspeed", type=str, default=None, help="Path to DeepSpeed JSON config")

    return p.parse_args()


def main():
    args = parse_args()

    if args.deepspeed:
        import json
        with open(args.deepspeed, "r") as f:
            ds_config = json.load(f)
        
        sp_size = ds_config.get("sequence_parallel_size", 1)
        if sp_size > 1:
            try:
                from deepspeed.runtime.sequence_parallel.ulysses_sp import UlyssesSPAttentionHF
                
                UlyssesSPAttentionHF.register_with_transformers(
                    model_name_or_path=args.model_name_or_path,
                    sequence_parallel_size=sp_size,
                    seq_length=args.max_length,
                    micro_batch_size=args.per_device_train_batch_size,
                    core_attn_implementation="flash_attention_2"
                )
                print(f"Successfully registered DeepSpeed UlyssesSPAttentionHF with SP size {sp_size}")
            except ImportError as e:
                print(f"Warning: DeepSpeed sequence parallelism requested (size {sp_size}) but deepspeed module not found or version too old. Error: {e}")

    grad_accum = args.global_batch_size // (args.num_gpus * args.per_device_train_batch_size)

    if getattr(args, "tokenized_dataset_path", None):
        if isinstance(args.tokenized_dataset_path, list) and len(args.tokenized_dataset_path) > 1:
            from datasets import concatenate_datasets

            dataset = concatenate_datasets([load_from_disk(p) for p in args.tokenized_dataset_path])
        else:
            path = args.tokenized_dataset_path[0] if isinstance(args.tokenized_dataset_path, list) else args.tokenized_dataset_path
            dataset = load_from_disk(path)
    else:
        dataset = load_terminal_corpus(
            subsets=args.subsets,
            sample_frac=args.sample_frac,
            seed=args.seed,
            cache_dir=args.cache_dir,
        )

    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        max_length=args.max_length,
        bf16=True,
        fp16=False,
        optim="adamw_torch",
        adam_beta1=0.9,
        adam_beta2=0.95,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        report_to=["tensorboard", "wandb"],
        seed=args.seed,
        packing=args.packing,
        dataset_num_proc=args.dataset_num_proc,
        deepspeed=args.deepspeed,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    trainer = SFTTrainer(
        model=args.model_name_or_path,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # Temporary fix to prevent Accelerate from re-autocasting the forward pass in FSDP.
    # Accelerate wraps model_forward in convert_to_fp32 during loss computation, causing issues.
    # By manually ensuring the correct precision, we avoid FSDP mixed precision crashes.
    # if getattr(trainer.accelerator.state, "fsdp_plugin", None) is not None:
    #     trainer.accelerator.state._mixed_precision = "no"

    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    main()
