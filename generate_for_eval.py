import gc
import json
import os
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import vllm
from datasets import load_dataset
from packaging.version import Version
from vllm import LLM, SamplingParams

from trl import TrlParser


@dataclass
class GenerateScriptArguments:
    save_generations: Optional[bool] = field(
        default=False,
        metadata={"help": "output folder"},
    )
    num_gpus: int = int(os.environ.get("NPROC", 1))
    base_model_name: Optional[str] = field(
        default=None, metadata={"help": "the model name"}
    )
    base_model_revision: Optional[str] = field(default=None)
    model_name_or_path: Optional[str] = field(
        default="EleutherAI/pythia-410m", metadata={"help": "the model name"}
    )
    model_paths: Optional[List[str]] = field(default_factory=list)
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "the tokenizer name"}
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "the dataset name"}
    )
    dataset_test_split: Optional[str] = field(
        default="validation", metadata={"help": "the dataset name"}
    )

    temperature: Optional[float] = field(
        default=0.7, metadata={"help": "Gen temperature"}
    )
    top_p: Optional[float] = field(default=1.0, metadata={"help": "Gen temperature"})
    max_new_tokens: Optional[int] = field(
        default=48, metadata={"help": "max new tokens"}
    )
    torch_dtype: Optional[str] = field(default="auto")
    sanity_check: Optional[bool] = field(default=False)
    wandb_run_id: str = None  # unused


def generate(script_args):
    dataset = load_dataset(
        script_args.dataset_name, split=script_args.dataset_test_split
    )
    prompts = dataset["prompt"]

    sampling_params = SamplingParams(
        temperature=script_args.temperature,
        max_tokens=script_args.max_new_tokens,
        top_p=script_args.top_p,
        n=1,
        include_stop_str_in_output=True,
        skip_special_tokens=False,
    )

    gens = {}
    trainer_states = {}
    model_paths = [script_args.model_name_or_path]
    # path with possible checkpoint subfolders
    if os.path.exists(script_args.model_name_or_path):
        checkpoint_subfolders = [
            path
            for path in os.listdir(script_args.model_name_or_path)
            if path.startswith("checkpoint")
            and (not script_args.model_paths or path in script_args.model_paths)
        ]

        # if there are checkpoint subfolders, use those instead of model_path
        if checkpoint_subfolders:
            model_paths = [
                os.path.join(script_args.model_name_or_path, subfolder)
                for subfolder in checkpoint_subfolders
            ]

    for model_name_or_path in model_paths:
        print(f"generating {model_name_or_path}")
        model_or_checkpoint_name = os.path.basename(model_name_or_path)

        if 16 % script_args.num_gpus == 0:
            tensor_parallel_size = script_args.num_gpus
        else:
            tensor_parallel_size = max(
                divisor for divisor in [1, 2, 4, 8] if divisor < script_args.num_gpus
            )

        llm = LLM(
            model=model_name_or_path,
            tokenizer=script_args.tokenizer_name,
            dtype=script_args.torch_dtype,
            trust_remote_code=True,
            tensor_parallel_size=tensor_parallel_size,
        )

        generations = llm.generate(prompts, sampling_params)

        texts = [output.prompt + output.outputs[0].text for output in generations]

        gens[model_or_checkpoint_name] = texts

        dataset = dataset.add_column(f"generations_{model_or_checkpoint_name}", texts)

        # delete old model
        destroy_model_parallel()
        del llm.llm_engine.model_executor.driver_worker
        del llm
        gc.collect()
        torch.cuda.empty_cache()

        trainer_state_path = os.path.join(model_name_or_path, "trainer_state.json")
        if os.path.exists(trainer_state_path):
            with open(trainer_state_path, "r") as f:
                state = json.load(f)
                trainer_states[model_or_checkpoint_name] = state
        else:
            trainer_states[model_or_checkpoint_name] = {}

    if script_args.save_generations:
        # TODO add hash to dataset path
        # sampling_str = str(sampling_params)
        # sampling_hash = hashlib.sha256(sampling_str.encode()).hexdigest()[:10]

        # TODO fix model name or path string
        dataset_path = os.path.join(
            script_args.model_name_or_path,
            "_generations",
        )
        os.makedirs(dataset_path, exist_ok=True)
        print("saving dataset to")
        print(dataset_path)
        dataset.save_to_disk(dataset_path)
        with open(os.path.join(dataset_path, "sampling_params.txt"), "w") as f:
            print(sampling_params, file=f)

        with open(os.path.join(dataset_path, "trainer_states.json"), "w") as f:
            json.dump(trainer_states, f)

    print(f"generated {len(gens)} steps")


if __name__ == "__main__":
    parser = TrlParser([GenerateScriptArguments])
    args = parser.parse_args_and_config()[0]

    if Version(vllm.__version__) > Version("0.4.1"):
        if args.num_gpus > 1:
            raise NotImplementedError("haven't implemented multigpu with vllm > 0.4.1")

        from vllm.distributed.parallel_state import destroy_model_parallel
    else:
        from vllm.model_executor.parallel_utils.parallel_state import (
            destroy_model_parallel,
        )

    if args.sanity_check:
        checkpoint_subfolders = [
            path
            for path in os.listdir(args.model_name_or_path)
            if path.startswith("checkpoint")
        ]
        args.model_paths = checkpoint_subfolders[:2]

    print("GENERATING")
    generate(args)
