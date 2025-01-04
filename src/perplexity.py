import warnings

import torch
from torch.nn import CrossEntropyLoss
from transformers import Pipeline


class PerplexityPipeline(Pipeline):
    label_pad_token_id = -100

    def __init__(self, **kwargs):
        self.loss_fct = CrossEntropyLoss(reduction="none")
        super().__init__(**kwargs)

    def __call__(self, inputs, **kwargs):
        inputs = (inputs,)
        return super().__call__(*inputs, **kwargs)

    def _sanitize_parameters(
        self, prompt_template="TL;DR:", dataset_text_field=None, **tokenizer_kwargs
    ):
        self.prompt_template = prompt_template
        self.prompt_template_tokens = self.tokenizer.encode(
            self.prompt_template, add_special_tokens=False, return_tensors="pt"
        ).squeeze()
        preprocess_params = {
            "dataset_text_field": dataset_text_field,
            **tokenizer_kwargs,
        }

        postprocess_params = {}

        return preprocess_params, {}, postprocess_params

    def preprocess(self, inputs, dataset_text_field=None, **tokenizer_kwargs):
        if dataset_text_field is not None:
            inputs = inputs[dataset_text_field]
        inputs = self.tokenizer(
            inputs,
            text_target=inputs,
            return_tensors="pt",
            padding=True,
            truncation=False,
            **tokenizer_kwargs,
        )
        inputs = ignore_prompt_labels(
            inputs, self.prompt_template_tokens, self.label_pad_token_id, self.tokenizer
        )

        return inputs

    def _forward(self, model_inputs, pad_token_id=None):
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = pad_token_id

        out_logits = self.model(
            input_ids=model_inputs["input_ids"],
            attention_mask=model_inputs["attention_mask"],
            labels=model_inputs["labels"],
            use_cache=False,
        ).logits

        shift_logits = out_logits[..., :-1, :]
        shift_labels = model_inputs["labels"][..., 1:]
        shift_attention_mask_batch = model_inputs["attention_mask"][..., 1:]

        nll_batch = (
            self.loss_fct(shift_logits.transpose(1, 2), shift_labels)
            * shift_attention_mask_batch
        ).sum(1) / shift_attention_mask_batch.sum(1)

        return nll_batch

    def postprocess(self, model_outputs, function_to_apply=None, top_k=1, _legacy=True):
        nll_tensor = model_outputs
        ppl_tensor = torch.exp(nll_tensor)

        return [
            {"nll": nll.item(), "ppl": ppl.item()}
            for nll, ppl in zip(nll_tensor, ppl_tensor)
        ]


def ignore_prompt_labels(batch, response_token_ids, ignore_index=-100, tokenizer=None):
    for i in range(batch["labels"].size(0)):
        response_token_ids_start_idx = None

        for idx in torch.where(batch["labels"][i] == response_token_ids[0])[0]:
            # `response_token_ids` is `'### Response:\n'`, here we are just making sure that the token IDs match
            if torch.equal(
                response_token_ids,
                batch["labels"][i][idx : idx + len(response_token_ids)],
            ):
                response_token_ids_start_idx = idx

        if response_token_ids_start_idx is None:
            warnings.warn("Could not find response key, ignoring")
            batch["labels"][i, :] = ignore_index
        else:
            response_token_ids_end_idx = response_token_ids_start_idx + len(
                response_token_ids
            )

            # Make pytorch loss function ignore all tokens up through the end of the response key
            batch["labels"][i, :response_token_ids_end_idx] = ignore_index

    return batch


if __name__ == "__main__":
    from transformers import pipeline

    query_response = [
        "This is a post\n TL;DR: This is a summary ",
        "This is another post\n TL;DR: This is another summary ",
    ]

    ppl_pipeline = pipeline(
        task="perplexity",
        model="mnoukhov/pythia410m-sft-tldr",
    )

    ppls = ppl_pipeline(query_response, prompt_template="TL;DR:")
