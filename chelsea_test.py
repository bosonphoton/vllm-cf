import torch
import vllm

from vllm.v1.sample.sampler import Sampler


MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
PROMPT = "My name is Chelsea, and I love"
GUMBEL_SEED = 123
TOP_K = 5
MAX_TOKENS = 64
TEMPERATURE = 1.0


# Run a 1-token generation to fetch full logprobs for the first token.
def get_first_token_output(llm: vllm.LLM, prompt: str) -> vllm.RequestOutput:
    sampling_params = vllm.SamplingParams(
        temperature=TEMPERATURE,
        max_tokens=1,
        logprobs=-1,
        gumbel_seed=GUMBEL_SEED,
    )
    return llm.generate([prompt], sampling_params)[0]


# Determine the sampling position used for the first generated token.
def get_first_token_position(
    output: vllm.RequestOutput, tokenizer: object, prompt: str
) -> int:
    if output.prompt_token_ids is not None:
        prompt_token_ids = output.prompt_token_ids
    else:
        prompt_token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    return len(prompt_token_ids) - 1


# Build a dense logprobs tensor indexed by token id.
def build_logprobs_tensor(
    logprobs_dict: dict[int, object], vocab_size: int, device: torch.device
) -> torch.Tensor:
    logprobs = torch.full((vocab_size,), float("-inf"), device=device)
    for token_id, info in logprobs_dict.items():
        logprobs[token_id] = float(info.logprob)
    if torch.isinf(logprobs).any():
        missing = int(torch.isinf(logprobs).sum().item())
        raise ValueError(f"Missing logprobs for {missing} tokens.")
    return logprobs


# Compute top-k tokens by gumbel logits and their gumbel-softmax probabilities.
def compute_gumbel_topk(
    logprobs: torch.Tensor,
    gumbel_seed: int,
    position: int,
    top_k: int,
) -> tuple[list[int], list[float]]:
    sampler = Sampler()
    positions = torch.tensor([position], device=logprobs.device)
    gumbel_noise = sampler._sample_gumbel(
        (1, logprobs.shape[0]),
        logprobs.device,
        logprobs.dtype,
        {0: gumbel_seed},
        positions,
    )
    gumbel_logits = logprobs.unsqueeze(0) + gumbel_noise
    topk_vals, topk_ids = torch.topk(gumbel_logits[0], k=top_k)
    gumbel_probs = torch.softmax(gumbel_logits[0], dim=-1)[topk_ids]
    return topk_ids.tolist(), gumbel_probs.tolist()


# Decode a single token id for display.
def decode_token(tokenizer: object, token_id: int) -> str:
    if hasattr(tokenizer, "convert_ids_to_tokens"):
        return tokenizer.convert_ids_to_tokens(token_id)
    return tokenizer.decode([token_id])


# Generate a completion with an optional gumbel flip at the first token.
def generate_completion(
    llm: vllm.LLM, prompt: str, position: int, rank: int
) -> str:
    extra_args = None
    if rank > 1:
        extra_args = {
            "gumbel_flip_positions": [position],
            "gumbel_flip_ranks": [rank],
        }
    sampling_params = vllm.SamplingParams(
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        gumbel_seed=GUMBEL_SEED,
        extra_args=extra_args,
    )
    return llm.generate([prompt], sampling_params)[0].outputs[0].text


# Run the full counterfactual sampling test.
def main() -> None:
    llm = vllm.LLM(model=MODEL_NAME, max_logprobs=-1)
    tokenizer = llm.get_tokenizer()

    output = get_first_token_output(llm, PROMPT)
    completion = output.outputs[0]
    if completion.logprobs is None or not completion.logprobs:
        raise ValueError("No logprobs returned; set logprobs=-1 for inspection.")

    position = get_first_token_position(output, tokenizer, PROMPT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = len(tokenizer)
    logprobs = build_logprobs_tensor(completion.logprobs[0], vocab_size, device)

    topk_ids, topk_probs = compute_gumbel_topk(
        logprobs, GUMBEL_SEED, position, TOP_K
    )

    print("Prompt:", PROMPT)
    print(f"First-token position: {position}")
    print("Top 5 gumbel-ranked tokens:")
    for rank, (token_id, prob) in enumerate(zip(topk_ids, topk_probs), start=1):
        token_str = decode_token(tokenizer, token_id)
        print(f"{rank}. id={token_id} token={token_str!r} gumbel_prob={prob:.6f}")

    print("\nCounterfactual completions (rank 1-5):")
    for rank in range(1, TOP_K + 1):
        completion_text = generate_completion(llm, PROMPT, position, rank)
        token_str = decode_token(tokenizer, topk_ids[rank - 1])
        print(f"[rank {rank}] first_token={token_str!r}")
        print(PROMPT + completion_text)


if __name__ == "__main__":
    main()
