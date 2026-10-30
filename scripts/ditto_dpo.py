import argparse
import asyncio
import json
import math
import random
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import tinker
import torch
import wandb
from dotenv import load_dotenv

from inkling_renderer import get_renderer_name
from tinker_cookbook import renderers
from tinker_cookbook.preference.train_dpo import compute_dpo_loss
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised.common import datum_from_model_input_weights
from tinker_cookbook.tokenizer_utils import get_tokenizer


@dataclass(frozen=True)
class PreferencePair:
    prompt_id: int
    chosen: str
    rejected: str
    kind: str
    chosen_round: int | None
    rejected_round: int


class ReplayBuffer:
    """Cache generations by prompt and sampling round."""

    def __init__(self):
        self.responses = {}

    def add_round(self, round_id, outputs):
        for prompt_id, texts in outputs.items():
            if not texts or any(not text.strip() for text in texts):
                raise ValueError(
                    f"Empty generation for prompt {prompt_id}"
                )

            history = self.responses.setdefault(prompt_id, {})

            if round_id in history:
                raise ValueError("Cannot overwrite a sampling round")

            history[round_id] = list(dict.fromkeys(texts))

    def save(self, path):
        path.write_text(
            json.dumps(self.responses, indent=2),
            encoding="utf-8",
        )


class DITTODataset:
    """Build preference pairs from demonstrations and cached outputs."""

    def __init__(self, demos, replay, seed=42):
        self.demos = demos
        self.replay = replay
        self.rng = random.Random(seed)

    def _candidate(self, kind, round_id):
        prompt_id = self.rng.randrange(len(self.demos))
        history = self.replay.responses[prompt_id]
        older_rounds = sorted(r for r in history if r < round_id)

        if kind == "on_policy":
            # Human demonstration beats the current policy.
            chosen = self.demos[prompt_id]["output"]
            chosen_round = None
            rejected_round = round_id

        elif kind == "replay":
            # Human demonstration beats an older policy.
            chosen = self.demos[prompt_id]["output"]
            chosen_round = None
            rejected_round = self.rng.choice(older_rounds)

        else:
            # Newer policy beats an older policy.
            available_rounds = sorted(
                r for r in history if r <= round_id
            )
            rejected_round, chosen_round = sorted(
                self.rng.sample(available_rounds, 2)
            )
            chosen = self.rng.choice(history[chosen_round])

        rejected = self.rng.choice(history[rejected_round])

        return PreferencePair(
            prompt_id=prompt_id,
            chosen=chosen,
            rejected=rejected,
            kind=kind,
            chosen_round=chosen_round,
            rejected_round=rejected_round,
        )

    def get_batch(self, round_id, batch_size):
        if batch_size <= 0 or batch_size % 10:
            raise ValueError(
                "Pair batch size must be a positive multiple of 10"
            )

        if round_id == 0:
            # Bootstrap: there are no historical policies yet.
            kinds = ["on_policy"] * batch_size
        else:
            unit = batch_size // 10
            kinds = (
                ["on_policy"] * (7 * unit)
                + ["replay"] * (2 * unit)
                + ["intermodel"] * unit
            )

        pairs = []

        for kind in kinds:
            for _ in range(1000):
                pair = self._candidate(kind, round_id)

                # Identical completions provide no preference signal.
                if pair.chosen.strip() != pair.rejected.strip():
                    pairs.append(pair)
                    break
            else:
                raise ValueError(
                    f"Cannot find distinct {kind} responses. "
                    "Try generating more samples per prompt."
                )

        self.rng.shuffle(pairs)
        return pairs


def load_demos(path):
    demos = json.loads(Path(path).read_text(encoding="utf-8"))

    if not isinstance(demos, list) or len(demos) != 10:
        raise ValueError(
            "Use data.json containing your 10 original demonstrations"
        )

    for row in demos:
        for key in ("input", "output"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(
                    f"Each demonstration needs a nonempty '{key}'"
                )

    return demos


def make_datum(
    renderer,
    prompt,
    answer,
    max_length,
    reduction="none",
):
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": answer},
    ]

    model_input, weights = renderer.build_supervised_example(
        messages,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )

    # Avoid silently truncating a preferred or rejected response.
    if model_input.length > max_length:
        raise ValueError(
            "Example exceeds max-length. Increase --max-length."
        )

    return datum_from_model_input_weights(
        model_input,
        weights,
        max_length,
        reduction=reduction,
    )


async def generate_round(sampler, renderer, demos, args):
    semaphore = asyncio.Semaphore(4)

    async def generate_one(prompt_id, row):
        texts = []

        prompt = renderer.build_generation_prompt(
            [{"role": "user", "content": row["input"]}]
        )

        # Retry incomplete or malformed responses.
        for _ in range(3):
            async with semaphore:
                result = await sampler.sample_async(
                    prompt=prompt,
                    num_samples=args.samples_per_prompt - len(texts),
                    sampling_params=tinker.SamplingParams(
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                        stop=renderer.get_stop_sequences(),
                    ),
                )

            for sequence in result.sequences:
                message, termination = renderer.parse_response(
                    sequence.tokens
                )

                if isinstance(termination, bool):
                    valid = termination
                else:
                    valid = termination.is_stop_sequence

                text = renderers.get_text_content(message).strip()

                if valid and text:
                    texts.append(text)

            if len(texts) >= args.samples_per_prompt:
                return prompt_id, texts

        raise RuntimeError(
            f"Too few complete outputs for prompt {prompt_id}. "
            "Try increasing --max-tokens."
        )

    results = await asyncio.gather(
        *[
            generate_one(i, row)
            for i, row in enumerate(demos)
        ]
    )
    return dict(results)


async def reference_scores(reference, datums):
    """Compute fixed-reference completion log probabilities."""

    async def score_one(datum):
        targets = datum.loss_fn_inputs["target_tokens"].data

        # model_input excludes the final target token.
        full_sequence = datum.model_input.append_int(
            int(targets[-1])
        )

        values = await reference.compute_logprobs_async(full_sequence)
        values = values[1:]

        weights = datum.loss_fn_inputs["weights"].data

        if len(values) != len(weights):
            raise ValueError(
                "Reference logprobs and completion mask do not align"
            )

        score = 0.0

        for value, weight in zip(values, weights):
            if weight != 0:
                if value is None or not math.isfinite(value):
                    raise ValueError(
                        "Missing or nonfinite reference log probability"
                    )
                score += value * weight

        return score

    return await asyncio.gather(
        *[score_one(datum) for datum in datums]
    )


def dpo_step(policy, reference, datums, learning_rate, beta):
    fixed_scores = asyncio.run(
        reference_scores(reference, datums)
    )

    def loss_fn(batch, logprobs):
        policy_scores = []

        for datum, token_logprobs in zip(batch, logprobs):
            weights = torch.as_tensor(
                datum.loss_fn_inputs["weights"].data,
                dtype=token_logprobs.dtype,
                device=token_logprobs.device,
            )

            # Only score assistant completion tokens.
            mask = weights != 0

            score = (
                token_logprobs[mask] * weights[mask]
            ).sum()

            policy_scores.append(score)

        reference_tensors = [
            score.new_tensor(value)
            for score, value in zip(policy_scores, fixed_scores)
        ]

        # Datums alternate: chosen, rejected, chosen, rejected...
        return compute_dpo_loss(
            chosen_logprobs=policy_scores[::2],
            rejected_logprobs=policy_scores[1::2],
            chosen_ref_logprobs=reference_tensors[::2],
            rejected_ref_logprobs=reference_tensors[1::2],
            dpo_beta=beta,
        )

    result = policy.forward_backward_custom(
        datums, loss_fn
    ).result()

    policy.optim_step(
        tinker.AdamParams(learning_rate=learning_rate)
    ).result()

    return result.metrics


def save_checkpoint(policy, directory, name):
    record = {
        "name": name,
        "state_path": policy.save_state(name).result().path,
        "sampler_path": (
            policy.save_weights_for_sampler(name).result().path
        ),
    }

    with (directory / "checkpoints.jsonl").open(
        "a", encoding="utf-8"
    ) as f:
        f.write(json.dumps(record) + "\n")

    print("Saved checkpoints:", record, flush=True)
    return record


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", default="data.json")
    parser.add_argument(
        "--model",
        default="thinkingmachines/Inkling-Small",
    )
    parser.add_argument(
        "--sft-state-path",
        help="Optional state checkpoint trained on the 10 REAL demos",
    )
    parser.add_argument("--sft-steps", type=int, default=10)
    parser.add_argument("--sft-target-nll", type=float, default=1.0)
    parser.add_argument("--sft-lr", type=float, default=1e-4)

    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--updates-per-round", type=int, default=5)
    parser.add_argument("--pair-batch-size", type=int, default=10)
    parser.add_argument("--samples-per-prompt", type=int, default=2)

    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--log-dir", default=None)
    parser.add_argument(
        "--wandb-project",
        default="tinker_personalization",
    )
    parser.add_argument(
        "--wandb-name",
        default="anushka-ditto",
    )

    args = parser.parse_args()

    if min(
        args.rounds,
        args.updates_per_round,
        args.samples_per_prompt,
    ) < 1:
        parser.error("Round, update, and sample counts must be positive")

    if args.pair_batch_size < 10 or args.pair_batch_size % 10:
        parser.error("--pair-batch-size must be a multiple of 10")

    if not args.sft_state_path and args.sft_steps < 1:
        parser.error(
            "Provide a demo-SFT checkpoint or enable SFT warm-up"
        )

    if args.sft_state_path and "/weights/" not in args.sft_state_path:
        parser.error(
            "--sft-state-path requires a state path containing /weights/"
        )

    load_dotenv()
    demos = load_demos(args.data)

    directory = Path(
        args.log_dir
        or f"results/ditto_{datetime.now():%Y%m%d_%H%M%S}"
    )

    # Start a fresh run rather than accidentally overwriting old results.
    directory.mkdir(parents=True, exist_ok=False)

    (directory / "config.json").write_text(
        json.dumps(vars(args), indent=2),
        encoding="utf-8",
    )
    (directory / "demonstrations.json").write_text(
        json.dumps(demos, indent=2),
        encoding="utf-8",
    )

    renderer_name = get_renderer_name(args.model)
    tokenizer = get_tokenizer(args.model)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    warmup_data = [
        make_datum(
            renderer,
            row["input"],
            row["output"],
            args.max_length,
            reduction="mean",
        )
        for row in demos
    ]

    with wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        config=vars(args),
    ) as run:
        print("W&B run:", run.url, flush=True)
        service = tinker.ServiceClient()

        if args.sft_state_path:
            policy = service.create_training_client_from_state(
                args.sft_state_path
            )
        else:
            # Brief SFT initialization on the original demonstrations.
            policy = service.create_lora_training_client(
                base_model=args.model,
                rank=args.lora_rank,
            )

            for step in range(args.sft_steps):
                result = policy.forward_backward(
                    warmup_data,
                    loss_fn="cross_entropy",
                ).result()

                # Each example has normalized completion-token weights.
                nll = float(result.metrics["loss:sum"]) / len(warmup_data)

                policy.optim_step(
                    tinker.AdamParams(learning_rate=args.sft_lr)
                ).result()

                run.log({
                    "warmup/nll": nll,
                    "warmup/step": step,
                })

                print(
                    f"Warm-up {step + 1}: pre-update NLL={nll:.4f}",
                    flush=True,
                )

                if nll <= args.sft_target_nll:
                    break

            initial = save_checkpoint(
                policy, directory, "demo_sft"
            )

            # Start DPO with a fresh optimizer and the SFT weights.
            policy = service.create_training_client_from_state(
                initial["state_path"]
            )

        # Freeze the reference ONCE; do not refresh it each round.
        reference = policy.save_weights_and_get_sampling_client()

        replay = ReplayBuffer()
        dataset = DITTODataset(demos, replay, args.seed)

        for round_id in range(args.rounds):
            # This sampler DOES refresh to reflect the current policy.
            sampler = policy.save_weights_and_get_sampling_client()

            generated = asyncio.run(
                generate_round(sampler, renderer, demos, args)
            )

            replay.add_round(round_id, generated)
            replay.save(directory / "replay_buffer.json")

            for update in range(args.updates_per_round):
                pairs = dataset.get_batch(
                    round_id,
                    args.pair_batch_size,
                )

                # Save the actual comparisons for inspection.
                with (directory / "pairs.jsonl").open(
                    "a", encoding="utf-8"
                ) as f:
                    for pair in pairs:
                        f.write(
                            json.dumps({
                                "round": round_id,
                                "update": update,
                                **asdict(pair),
                            }) + "\n"
                        )

                datums = []

                for pair in pairs:
                    prompt = demos[pair.prompt_id]["input"]

                    for answer in (pair.chosen, pair.rejected):
                        datums.append(
                            make_datum(
                                renderer,
                                prompt,
                                answer,
                                args.max_length,
                                reduction="none",
                            )
                        )

                metrics = dpo_step(
                    policy,
                    reference,
                    datums,
                    args.lr,
                    args.beta,
                )

                counts = Counter(pair.kind for pair in pairs)

                logged = {
                    "ditto/round": round_id,
                    "ditto/update": update,
                }
                logged.update({
                    f"dpo/{key}": value
                    for key, value in metrics.items()
                })
                logged.update({
                    f"pairs/{key}": value
                    for key, value in counts.items()
                })

                run.log(logged)

                print(
                    f"Round {round_id}, update {update}: {metrics}",
                    flush=True,
                )

            checkpoint_name = (
                "final"
                if round_id == args.rounds - 1
                else f"round_{round_id:03d}"
            )

            record = save_checkpoint(
                policy,
                directory,
                checkpoint_name,
            )

        run.summary["final_sampler_path"] = record["sampler_path"]
        run.summary["final_state_path"] = record["state_path"]

        print(
            "Finished. Evaluate on new prompts, not the 10 training demos.",
            flush=True,
        )


if __name__ == "__main__":
    main()