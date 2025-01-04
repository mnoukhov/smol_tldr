import os
from dataclasses import dataclass
from typing import Optional

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from trl import ModelConfig, SFTConfig, SFTTrainer, ScriptArguments, TrlParser
from trl.trainer.utils import (
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

# correct chat template for TLDR
# 1. joins messages without space since its already at start of completion
# 2. removes EOS if generation_prompt
TLDR_CHAT_TEMPLATE = """
{{- messages | selectattr("content", "defined") | map(attribute="content") | join() -}}
{%- if not add_generation_prompt %}
    {{- eos_token }}
{%- endif %}
"""


@dataclass
class MyScriptArguments(ScriptArguments):
    output_global_parent_dir: str = None
    wandb_run_id: Optional[str] = None
    sanity_check: bool = False


if __name__ == "__main__":
    parser = TrlParser((MyScriptArguments, SFTConfig, ModelConfig))
    args, config, model_config = parser.parse_args_and_config()

    if args.wandb_run_id == "slurm":
        run_id = os.environ["SLURM_JOB_ID"]
        config_name = os.path.basename(config.output_dir)
        # save to parent / slurm id / output_dir
        if args.output_global_parent_dir is not None:
            args.output_global_parent_dir = os.path.join(
                args.output_global_parent_dir,
                run_id,
            )
        os.environ["WANDB_RUN_ID"] = run_id + "_" + config_name

    if args.output_global_parent_dir is not None:
        config.output_dir = os.path.join(
            args.output_global_parent_dir, config.output_dir
        )

    ################
    # Model & Tokenizer
    ################
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    quantization_config = get_quantization_config(model_config)
    config.model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if config.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path)
    tokenizer.chat_template = TLDR_CHAT_TEMPLATE

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ################
    # Dataset
    ################
    datasets = load_dataset(args.dataset_name)

    if args.sanity_check:
        for key in datasets:
            datasets[key] = datasets[key].select(range(1024))

        config.report_to = []
        config.push_to_hub = False
        config.eval_stategy = "no"
        # config.save_strategy = "no"

    train_dataset = datasets[args.dataset_train_split]
    eval_dataset = datasets[args.dataset_test_split]

    ################
    # Training
    ################
    trainer = SFTTrainer(
        model=model_config.model_name_or_path,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_config),
    )

    trainer.train()

    trainer.save_model(config.output_dir)

    if config.push_to_hub:
        trainer.push_to_hub()

    if config.save_strategy != "no" and trainer.accelerator.is_main_process:
        try:
            os.remove("output_dir")
        except OSError:
            pass

        os.symlink(config.output_dir, "output_dir")
