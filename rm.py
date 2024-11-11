import os
import warnings
from dataclasses import dataclass, field

import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import (
    TrlParser,
    ModelConfig,
    RewardConfig,
    RewardTrainer,
    ScriptArguments,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    setup_chat_format,
)


@dataclass
class MyScriptArguments(ScriptArguments):
    output_global_parent_dir: str = field(default=None)
    wandb_run_id: str | None = field(default=None)
    sanity_check: bool = field(
        default=False, metadata={"help": "only train on 1000 samples"}
    )


if __name__ == "__main__":
    parser = TrlParser((MyScriptArguments, RewardConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse_args_and_config()

    if script_args.output_global_parent_dir is not None:
        run_id = os.path.basename(os.getcwd())
        training_args.output_dir = os.path.join(
            script_args.output_global_parent_dir, run_id, training_args.output_dir
        )

    if script_args.wandb_run_id == "slurm":
        run_id = os.environ["SLURM_JOB_ID"]
        config_name = os.path.basename(training_args.output_dir)
        # save to parent / slurm id / output_dir
        if script_args.output_global_parent_dir is not None:
            training_args.output_dir = os.path.join(
                script_args.output_global_parent_dir, run_id, training_args.output_dir
            )
        os.environ["WANDB_RUN_ID"] = run_id + "_" + config_name

    ################
    # Model & Tokenizer
    ################
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
        use_cache=False if training_args.gradient_checkpointing else True,
        torch_dtype=torch_dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code,
        use_fast=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_config.model_name_or_path,
        num_labels=1,
        trust_remote_code=model_config.trust_remote_code,
        **model_kwargs,
    )
    # Align padding tokens between tokenizer and model
    model.config.pad_token_id = tokenizer.pad_token_id

    # If post-training a base model, use ChatML as the default template
    if tokenizer.chat_template is None:
        model, tokenizer = setup_chat_format(model, tokenizer)

    if model_config.use_peft and model_config.lora_task_type != "SEQ_CLS":
        warnings.warn(
            "You are using a `task_type` that is different than `SEQ_CLS` for PEFT. This will lead to silent bugs"
            " Make sure to pass --lora_task_type SEQ_CLS when using this script with PEFT."
        )

    ##############
    # Load dataset
    ##############
    dataset = load_dataset(script_args.dataset_name)

    if script_args.sanity_check:
        for key in dataset:
            dataset[key] = dataset[key].select(range(1024))

        training_args.report_to = []
        training_args.push_to_hub = False
        training_args.save_strategy = "no"

    ##########
    # Training
    ##########
    trainer = RewardTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split]
        if training_args.evaluation_strategy != "no"
        else None,
        peft_config=get_peft_config(model_config),
    )
    trainer.train()

    ############################
    # Save model and push to Hub
    ############################
    trainer.save_model(training_args.output_dir)

    if training_args.eval_strategy != "no":
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
