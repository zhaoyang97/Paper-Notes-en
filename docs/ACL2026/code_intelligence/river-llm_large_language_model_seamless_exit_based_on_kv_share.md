---
title: >-
  [Paper Note] River-LLM: Large Language Model Seamless Exit Based on KV Share
description: >-
  [ACL 2026][Code Intelligence][Early exit] This paper proposes River-LLM, a training-free framework that addresses the KV Cache missing problem in decoder-only Early Exit by constructing a lightweight KV-shared exit chann…
tags:
  - "ACL 2026"
  - "Code Intelligence"
  - "Early exit"
  - "KV cache"
  - "Dynamic inference"
  - "Model acceleration"
  - "Quantization"
date: 2026-05-08
content_hash: b648a7b0316b0366
---

# River-LLM: Large Language Model Seamless Exit Based on KV Share

**Conference**: ACL 2026  
**arXiv**: [2604.18396](https://arxiv.org/abs/2604.18396)  
**Code**: None  
**Area**: Code Intelligence  
**Keywords**: Early exit, KV cache, Dynamic inference, Model acceleration, Quantization

## TL;DR
This paper proposes River-LLM, a training-free framework that addresses the KV Cache missing problem in decoder-only Early Exit by constructing a lightweight KV-shared exit channel (Exit River). By utilizing state transition similarity to guide exit decisions, it achieves 1.71×-2.16× real-world inference speedup while maintaining near-lossless generation quality.

## Background & Motivation

**Background**: Early Exit is a mainstream direction for LLM dynamic inference acceleration, which reduces computation by skipping redundant layers based on input complexity. Existing methods such as SkipDecode (monotonic decreasing exit), EE-LLM (batch recomputation), CALM (state propagation), and D-LLM (KV masking) attempt to solve this from different perspectives.

**Limitations of Prior Work**: In decoder-only architectures, the efficiency of Early Exit is severely bottlenecked by the **KV Cache missing problem**. When a token exits early, the skipped layers fail to provide the necessary historical KV states for subsequent tokens. Empirical analysis shows that while theoretically over 50% of tokens can exit at early layers, the actual wall-clock speedup remains minimal.

**Key Challenge**: The four existing KV recovery strategies have fundamental flaws: batch recomputation introduces significant latency overhead; monotonic decreasing exit severely limits exit flexibility; state propagation sacrifices precision for speed; and KV masking leads to severe accuracy loss. No existing method satisfies both "per-token free exit" and "KV integrity" simultaneously.

**Goal**: To design a "Seamless Exit" mechanism where individual tokens can exit independently at any layer (granular freedom), while the KV cache for skipped layers is automatically filled as a byproduct of the exit path execution (intrinsic KV integrity), without requiring post-exit recovery or recomputation.

**Key Insight**: Inspired by research on KV cache redundancy, the authors found that lightweight exit layers using quantized weights can replicate the KV generation of the backbone decoder with extremely low overhead. The cosine similarity between the KV outputs of the exit layers and the backbone layers remains above 0.97.

**Core Idea**: Construct a lightweight "Exit River" (KV-Shared Exit River) that maps one-to-one with the backbone decoder. It uses 4-bit quantized weights to accelerate the passage of tokens through the exit channel (2.4× throughput gain) while naturally generating KV caches compatible with the backbone.

## Method

### Overall Architecture
The inference of River-LLM is divided into two phases: the Prefill phase uses sequence-level exit (all tokens exit at a uniform depth to maintain parallel attention efficiency), and the Generation phase switches to token-level exit (each token terminates at its optimal depth). Once a token triggers the exit condition, the remaining computation is offloaded to a sequence of quantized exit layers, finally reaching the original LM Head to generate logits. The exit layers simultaneously produce complete KV caches, eliminating the KV missing problem for subsequent tokens.

### Key Designs

1.  **KV-Shared Exit Layer**:

    - **Function**: Acts as a lightweight replacement for the backbone decoder, generating compatible KV caches while accelerating token passage.
    - **Mechanism**: The exit layer inherits the architecture and parameters of the backbone layer, applying 4-bit weight quantization (W4A16) to the Attention and FFN blocks while maintaining the KV Cache in FP16 format to preserve representation density. Each exit layer shares the same KV Cache addressing scheme as its corresponding backbone decoder. Through quantization and inference kernels optimized with partial graph compilation, the exit layer achieves a 2.4× throughput gain, keeping the cosine similarity between generated KV and native backbone KV above 0.97.
    - **Design Motivation**: The core insight is that KV caches do not require absolute precision—the error introduced by 4-bit quantization is within an acceptable range, but the computational savings are massive. The entire weight migration process typically completes within one minute without any training.

2.  **State Transition Similarity Exit Decision**:

    - **Function**: Predicts cumulative quantization errors to guide precise exit timing.
    - **Mechanism**: Utilizes the cosine similarity between the input and output of a decoder block (state transition similarity) as the exit metric. The exit decision is defined as $\mathcal{D}^{(l)} = \mathbb{I}(\min_{b \in \mathcal{B}} s_{t,b}^{(l)} > \tau)$, where $s_{t,b}^{(l)} = \frac{\mathbf{h}_{t,b}^{(l-1)\top} \mathbf{h}_{t,b}^{(l)}}{\|\mathbf{h}_{t,b}^{(l-1)}\| \|\mathbf{h}_{t,b}^{(l)}\|}$. The authors observed a moderate positive correlation ($r=0.5536$) between early-layer state transition similarity and final-layer backbone-exit value vector similarity, allowing the former to predict the latter.
    - **Design Motivation**: State transition similarity generally follows a monotonically increasing trend, aligning with the logic of Early Exit (most layers after the exit point also satisfy the condition). The computational complexity of the exit decision is only $\mathcal{O}(d)$, taking approximately 100 microseconds, which accounts for only 0.0688% of the total inference time.

3.  **Backbone Offloading**:

    - **Function**: Further reduces GPU memory footprint.
    - **Mechanism**: Since most tokens terminate backbone traversal early, the framework can automatically evict subsequently sparsely activated backbone layers from the main VRAM. The model runs with a memory footprint close to a full-quantization baseline, while the Exit River resides permanently in VRAM to provide continuous semantic completion.
    - **Design Motivation**: The advantage of River-LLM over full-model quantization lies in selective computational fidelity—"difficult" or high-entropy tokens pass through the backbone in full precision, while "easy" tokens are offloaded to the Exit River.

### Loss & Training
River-LLM is a completely training-free framework. Exit layer weights are copied directly from the backbone and subjected to PTQ quantization without any fine-tuning. A flexible trade-off between accuracy and speed is achieved by adjusting the threshold $\tau$.

## Key Experimental Results

### Main Results
Comparison of actual wall-clock speedup on GSM8K, MATH, and HumanEval.

| Model | Task | Backbone Acc | Full Quant. Acc | River-LLM Acc | River-LLM Gain |
|------|------|------|------|------|------|
| Llama3.2 1B | GSM8K | 33.2 | 25.1 | 29.3 | 2.16× |
| Llama3.2 1B | MATH | 17.8 | 12.2 | 14.6 | 1.88× |
| Llama3.1 8B | GSM8K | 78.2 | 69.8 | 74.4 | 1.78× |
| Llama3.1 8B | HumanEval | 57.3 | 50.2 | 55.5 | 1.77× |
| Ministral3 8B | MATH | 48.1 | 46.0 | 46.6 | 1.85× |

### Ablation Study

| KV Strategy | Actual Latency | Precision Retention | Description |
|---------|---------|---------|------|
| KV Mask | Highest Backbone Latency | Poor | Requires deeper execution to compensate for precision loss |
| KV Recompute | High Compute Overhead | Good | Overhead accumulates in long sequence generation |
| State Propagation | Medium | Medium | Precision-speed trade-off |
| Mono-Decreasing | Medium | Good | Limits exit flexibility |
| KV Share (Ours) | Lowest | Good | No recovery operations required |

### Key Findings
- River-LLM on average only executes 3-4 backbone layers to reach accuracy close to the full model; on Llama3.1 8B, most tasks terminate before the median layer.
- On HumanEval, River-LLM even outperformed the full-model baseline (57.3 vs 55.5), possibly by skipping redundant deep layers to reduce cumulative noise or "overthinking."
- Compared to the full-quantization baseline, River-LLM throughput is slightly lower by about 10%, but precision retention is far superior.
- The exit decision logic takes only about 100 microseconds, 0.0688% of total inference time, making its overhead negligible.
- GPU memory consumption is significantly lower than the backbone model and existing Early Exit baselines, approaching that of a fully quantized model.

## Highlights & Insights
- The **conceptual definition of "Seamless Exit"** is valuable: Granular Freedom + Intrinsic KV Integrity, clearly distinguishing River-LLM from all prior methods. This definition itself is a contribution to Early Exit research.
- The idea of using **quantized exit layers as KV proxies** is ingenious: instead of seeking exact KV recovery, it uses 4-bit quantized layers for "approximate" generation. A cosine similarity of 0.97+ is sufficient to maintain autoregressive generation quality. This leverages the inherent redundancy of the KV cache.
- Being completely training-free is a major practical advantage; weight migration completes in a minute, allowing it to be plug-and-play for any decoder-only model.
- The quantization backend is replaceable (accuracy improved further after moving from HQQ to AWQ), giving the framework good scalability.

## Limitations & Future Work
- Current evaluations only cover models up to 8B parameters; behavior on 24B and 70B models has not been verified.
- Acceleration is less significant for prefill-dominant tasks (e.g., MMLU) because the prefill stage uses sequence-level exit.
- The exit threshold $\tau$ requires manual selection, and optimal values may vary across different models and tasks.
- Cumulative quantization errors still exist at very early exit points (though controllable), and their impact on extremely long sequence generation has not been fully studied.

## Related Work & Insights
- **vs LayerSkip/SpecEE**: These methods combine Early Exit with speculative decoding but are limited by sequence-level exits or short draft sequences. River-LLM achieves true token-level free exit.
- **vs CALM**: CALM uses state propagation to fill KV, which is a precision-speed trade-off; River-LLM generates high-fidelity KV via quantized exit layers, eliminating this trade-off.
- **vs Full Model Quantization**: Full quantization imposes uniform precision loss on all tokens. River-LLM selectively allows "difficult" tokens to use the full-precision backbone and "easy" tokens to use the quantized Exit River, achieving a superior Pareto front.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The concept of the KV-Shared Exit River is novel and elegant, clearly solving the core bottleneck of Early Exit.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Four models, multiple benchmarks, and sufficient comparisons with full quantization and existing strategies, though validation on models >8B is missing.
- **Writing Quality**: ⭐⭐⭐⭐ Clear derivation of motivation and informative charts, though some content is repetitive.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] KV Cache Transform Coding for Compact Storage in LLM Inference](../../ICLR2026/code_intelligence/kv_cache_transform_coding_for_compact_storage_in_llm_inference.md)
- [\[ACL 2026\] Precise Debugging Benchmark: Is Your Model Debugging or Regenerating?](precise_debugging_benchmark_is_your_model_debugging_or_regenerating.md)
- [\[ACL 2026\] KoCo-Bench: Can Large Language Models Leverage Domain Knowledge in Software Development?](koco-bench_can_large_language_models_leverage_domain_knowledge_in_software_devel.md)
- [\[ICLR 2026\] Training Large Language Models To Reason In Parallel With Global Forking Tokens](../../ICLR2026/code_intelligence/training_large_language_models_to_reason_in_parallel_with_global_forking_tokens.md)
- [\[ICML 2026\] Poison with Style: A Practical Poisoning Attack on Code Large Language Models](../../ICML2026/code_intelligence/poison_with_style_a_practical_poisoning_attack_on_code_large_language_models.md)

</div>

<!-- RELATED:END -->
