---
title: >-
  [Paper Note] River-LLM: Large Language Model Seamless Exit Based on KV Share
description: >-
  [ACL 2026][Code Intelligence][Early Exit] This paper proposes River-LLM, a training-free framework that addresses the KV Cache absence problem in Early Exit for decoder-only architectures by constructing a lightweight KV…
tags:
  - "ACL 2026"
  - "Code Intelligence"
  - "Early Exit"
  - "KV Cache"
  - "Dynamic Inference"
  - "Model Acceleration"
  - "Quantization"
date: 2026-05-08
content_hash: 1cb41e0310a45e95
---

# River-LLM: Large Language Model Seamless Exit Based on KV Share

**Conference**: ACL 2026
**arXiv**: [2604.18396](https://arxiv.org/abs/2604.18396)
**Code**: N/A
**Area**: Code Intelligence
**Keywords**: Early Exit, KV Cache, Dynamic Inference, Model Acceleration, Quantization

## TL;DR
This paper proposes River-LLM, a training-free framework that addresses the KV Cache absence problem in Early Exit for decoder-only architectures by constructing a lightweight KV-shared exit channel (Exit River). It leverages state transition similarity to guide exit decisions, achieving 1.71×–2.16× real wall-clock inference speedup with near-lossless generation quality.

## Background & Motivation

**Background**: Early Exit is a mainstream approach for dynamic inference acceleration in LLMs, reducing computation by dynamically skipping redundant layers based on input complexity. Prior methods such as SkipDecode (monotonically decreasing exit), EE-LLM (batch recomputation), CALM (state propagation), and D-LLM (KV masking) have each attempted to address this problem from different angles.

**Limitations of Prior Work**: In decoder-only architectures, the efficiency of Early Exit is severely bottlenecked by the **KV Cache absence problem**. When a token exits early, the skipped layers cannot provide the necessary historical KV states for subsequent tokens. Empirical analysis in this paper shows that although theoretically more than 50% of tokens can exit at early layers, the actual wall-clock speedup is negligible.

**Key Challenge**: The four existing KV recovery strategies all have fundamental drawbacks: batch recomputation introduces significant latency overhead; monotonically decreasing exit severely limits exit flexibility; state propagation sacrifices accuracy for speed; and KV masking leads to severe accuracy degradation. No existing method can simultaneously satisfy "per-token free exit" and "KV integrity."

**Goal**: To design a "Seamless Exit" mechanism enabling individual tokens to exit independently at any layer (free granularity), while KV caches for skipped layers are automatically populated as a byproduct of exit path execution (intrinsic KV integrity), requiring no post-exit recovery or recomputation.

**Key Insight**: Inspired by research on KV cache redundancy, the authors find that lightweight quantized exit layers can replicate the KV generation of backbone decoders at minimal overhead, "substituting" for skipped layers to complete KV population. The cosine similarity between KV outputs of exit layers and backbone layers remains above 0.97.

**Core Idea**: Construct a lightweight "Exit River" (KV-Shared Exit River) with a one-to-one mapping to the backbone decoder, using 4-bit quantized weights to accelerate token throughput through the exit channel (2.4× throughput gain), while naturally generating KV caches compatible with the backbone.

## Method

### Overall Architecture
River-LLM inference proceeds in two phases: the Prefill phase employs sequence-level exit (all tokens exit at a uniform depth to maintain parallel attention efficiency); the Generation phase switches to token-level exit (each token terminates at its optimal depth). When a token triggers the exit condition, remaining computation is offloaded to the quantization-accelerated exit layer sequence, ultimately reaching the original LM Head to generate logits. The exit layers simultaneously produce complete KV caches, eliminating the KV absence problem for subsequent tokens.

### Key Designs

1. **KV-Shared Exit Layer**:

    - **Function**: Serves as a lightweight substitute for backbone decoders, generating compatible KV caches while accelerating token throughput.
    - **Mechanism**: Exit layers inherit the architecture and parameters of the corresponding backbone layers, then apply 4-bit weight quantization (W4A16) to both Attention and FFN blocks while retaining KV Cache in FP16 format to preserve representational fidelity. Each exit layer shares the same KV Cache addressing scheme as its corresponding backbone decoder. Through quantization and partial graph-compiled inference kernels, exit layers achieve 2.4× throughput improvement, and the generated KV maintains a cosine similarity above 0.97 with native backbone KV.
    - **Design Motivation**: The core insight is that KV caches do not need to be perfectly precise—the error introduced by 4-bit quantization is within an acceptable range while the computational savings are substantial. The entire weight transfer process typically completes within one minute, requiring no training.

2. **Exit Decision via State Transition Similarity**:

    - **Function**: Predicts cumulative quantization error to guide precise exit timing.
    - **Mechanism**: The cosine similarity between the input and output of each decoder block (state transition similarity) is used as the exit indicator. The exit decision is defined as $\mathcal{D}^{(l)} = \mathbb{I}(\min_{b \in \mathcal{B}} s_{t,b}^{(l)} > \tau)$, where $s_{t,b}^{(l)} = \frac{\mathbf{h}_{t,b}^{(l-1)\top} \mathbf{h}_{t,b}^{(l)}}{\|\mathbf{h}_{t,b}^{(l-1)}\| \|\mathbf{h}_{t,b}^{(l)}\|}$. The authors find a moderate positive correlation ($r=0.5536$) between state transition similarity at early layers and backbone-exit value vector similarity at final layers, making the former a viable predictor of the latter.
    - **Design Motivation**: State transition similarity exhibits a roughly monotonically increasing trend, consistent with the empirical regularity of Early Exit (most layers after the exit point also satisfy the exit condition). The computational complexity of exit determination is only $\mathcal{O}(d)$, approximately 100 microseconds, accounting for merely 0.0688% of total inference time.

3. **Backbone Offloading**:

    - **Function**: Further reduces GPU memory footprint.
    - **Mechanism**: Since the vast majority of tokens terminate backbone traversal at early stages, the framework automatically evicts subsequently sparsely activated backbone layers from main memory. The model operates at near-fully-quantized baseline memory consumption, while the Exit River remains resident in memory to provide continuous semantic completion.
    - **Design Motivation**: River-LLM's advantage over full model quantization lies in selective computational fidelity—"hard" or high-entropy tokens traverse the backbone at full precision, while "easy" tokens are offloaded to the Exit River.

### Loss & Training
River-LLM is a completely training-free framework. Exit layer weights are directly copied from backbone layers and subjected to PTQ quantization, without any fine-tuning. Flexible accuracy-speed tradeoffs are achieved by adjusting the threshold $\tau$.

## Key Experimental Results

### Main Results
Wall-clock speedup comparison on GSM8K, MATH, and HumanEval.

| Model | Task | Backbone Acc | Full Quant. Acc | River-LLM Acc | River-LLM Speedup |
|------|------|------|------|------|------|
| Llama3.2 1B | GSM8K | 33.2 | 25.1 | 29.3 | 2.16× |
| Llama3.2 1B | MATH | 17.8 | 12.2 | 14.6 | 1.88× |
| Llama3.1 8B | GSM8K | 78.2 | 69.8 | 74.4 | 1.78× |
| Llama3.1 8B | HumanEval | 57.3 | 50.2 | 55.5 | 1.77× |
| Ministral3 8B | MATH | 48.1 | 46.0 | 46.6 | 1.85× |

### Ablation Study

| KV Strategy | Actual Latency | Accuracy Retention | Notes |
|---------|---------|---------|------|
| KV Mask | Highest backbone latency | Poor | Requires deeper layers to compensate for accuracy loss |
| KV Recompute | High computational overhead | Good | Overhead accumulates in long-sequence generation |
| State Propagation | Moderate | Moderate | Accuracy-speed tradeoff |
| Mono-Decreasing | Moderate | Good | Limits exit flexibility |
| KV Share (Ours) | Lowest | Good | No recovery operations required |

### Key Findings
- River-LLM executes an average of only 3–4 backbone layers to achieve accuracy close to the full model, with most tasks on Llama3.1 8B terminating before the median layer.
- On HumanEval, River-LLM even surpasses the full-model baseline (57.3 vs. 55.5), possibly by reducing accumulated noise or "overthinking" through skipping redundant deep layers.
- Compared to the full quantization baseline, River-LLM throughput is approximately 10% lower, but accuracy retention is far superior.
- The exit decision logic requires only approximately 100 microseconds, accounting for 0.0688% of total inference time—negligible overhead.
- GPU memory consumption is significantly lower than the backbone model and existing Early Exit baselines, approaching that of fully quantized models.

## Highlights & Insights
- **The conceptual definition of "Seamless Exit"** is highly valuable: free granularity + intrinsic KV integrity clearly distinguishes River-LLM from all prior methods, and this definition itself constitutes a contribution to Early Exit research.
- **The idea of using quantized exit layers as KV proxies** is elegant: rather than pursuing exact KV recovery, 4-bit quantized layers "approximately" generate KV, and cosine similarity above 0.97 is sufficient to sustain autoregressive generation quality. This exploits the intrinsic redundancy of KV caches.
- Being completely training-free is a major practical advantage; weight transfer completes within one minute and can be applied plug-and-play to any decoder-only model.
- The quantization backend is replaceable (replacing HQQ with AWQ yields further accuracy improvements), giving the framework good extensibility.

## Limitations & Future Work
- Current evaluation covers models up to 8B parameters only; behavior on 24B and 70B models remains unvalidated.
- Speedup is marginal for prefill-dominated tasks (e.g., MMLU), as the prefill phase employs sequence-level exit.
- The exit threshold $\tau$ requires manual selection; the optimal value may differ across models and tasks.
- Cumulative quantization error at very early exit points still exists (though manageable), and its impact on very long sequence generation has not been thoroughly studied.

## Related Work & Insights
- **vs. LayerSkip/SpecEE**: These methods combine Early Exit with speculative decoding but are constrained by sequence-level exit or short draft sequences; River-LLM achieves truly free token-level exit.
- **vs. CALM**: CALM uses state propagation to fill KV, representing an accuracy-speed tradeoff; River-LLM generates high-fidelity KV through quantized exit layers, eliminating this tradeoff.
- **vs. Full Model Quantization**: Full quantization imposes uniform precision loss on all tokens; River-LLM selectively routes "hard" tokens through the full-precision backbone and "easy" tokens through the quantized Exit River, achieving a superior Pareto frontier.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The concept of a KV-shared Exit River is novel and elegant, clearly addressing the core bottleneck of Early Exit.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Evaluation across four models and multiple benchmarks with comparisons to full quantization and existing strategies is comprehensive, though validation on models larger than 8B is absent.
- **Writing Quality**: ⭐⭐⭐⭐ Motivation is clearly developed and figures are informative, though some content is repetitive.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] KV Cache Transform Coding for Compact Storage in LLM Inference](../../ICLR2026/code_intelligence/kv_cache_transform_coding_for_compact_storage_in_llm_inference.md)
- [\[ACL 2026\] Precise Debugging Benchmark: Is Your Model Debugging or Regenerating?](precise_debugging_benchmark_is_your_model_debugging_or_regenerating.md)
- [\[ICLR 2026\] Training Large Language Models To Reason In Parallel With Global Forking Tokens](../../ICLR2026/code_intelligence/training_large_language_models_to_reason_in_parallel_with_global_forking_tokens.md)
- [\[ICLR 2026\] DRO-InstructZero: Distributionally Robust Prompt Optimization for Large Language Models](../../ICLR2026/code_intelligence/dro-instructzero_distributionally_robust_prompt_optimization_for_large_language_.md)
- [\[AAAI 2026\] SPAN: Benchmarking and Improving Cross-Calendar Temporal Reasoning of Large Language Models](../../AAAI2026/code_intelligence/span_benchmarking_and_improving_cross-calendar_temporal_reasoning_of_large_langu.md)

</div>

<!-- RELATED:END -->
