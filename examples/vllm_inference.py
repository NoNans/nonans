"""
vLLM inference integration with NoNans.

Demonstrates the inference surface — long-context attention and large-batch
serving where standard runtimes hit numerical singularities (softmax overflow,
attention denominator collapse) at sequence > 128K and batch > 64.

The integration shape is identical to training: wrap the engine, run as
usual. The detection layer surfaces events; the runtime resolves them when
present.
"""

# Note: this example assumes vLLM is installed. If you do not use vLLM,
# the wrap() call is identical for any inference engine that exposes a
# nn.Module-compatible surface.

try:
    from vllm import LLM, SamplingParams
    HAVE_VLLM = True
except ImportError:
    HAVE_VLLM = False

import nonans


def main() -> None:
    if not HAVE_VLLM:
        print("vLLM not installed; this example requires `pip install vllm`.")
        return

    llm = LLM(
        model="meta-llama/Llama-3.1-70B-Instruct",
        max_model_len=1_000_000,    # 1M-token context
        tensor_parallel_size=8,
        dtype="float8",             # FP8 inference
    )

    # Wrap the engine. The detection layer hooks into the attention
    # kernels; the runtime resolves any singularities at the kernel boundary.
    llm.llm_engine.model_executor = nonans.wrap(
        llm.llm_engine.model_executor,
        mode="auto",
    )

    prompts = [
        "Summarize the following document. " + ("..." * 200_000),
    ]
    sampling = SamplingParams(temperature=0.7, max_tokens=2048)

    outputs = llm.generate(prompts, sampling)
    for o in outputs:
        print(o.outputs[0].text[:1000])


if __name__ == "__main__":
    main()
