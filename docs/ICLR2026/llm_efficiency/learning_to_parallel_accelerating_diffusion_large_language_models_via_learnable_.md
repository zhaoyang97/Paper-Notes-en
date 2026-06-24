---
title: >-
  [Paper Note] Learning to Parallel: Accelerating Diffusion Large Language Models via Learnable Parallel Decoding
description: >-
  [ICLR 2026][LLM Efficiency][diffusion LLM] Addressing the pain point that parallel decoding in diffusion large language models (dLLMs) relies on fixed heuristics (e.g., confidence thresholds) and lacks adaptability to different inputs, this paper employs an extremely lightweight learnable filter (2-layer MLP, ~2k parameters, 6-minute training) to approximate an oracle strategy of "finalize immediately once predicted correctly." Coupled with End-of-Text early stopping…
tags:
  - "ICLR 2026"
  - "LLM Efficiency"
  - "diffusion LLM"
  - "parallel decoding"
  - "learnable filter"
  - "inference acceleration"
  - "LLaDA"
  - "KV-Cache"
date: 2026-05-08
content_hash: 9a557524b1d4cf0f
---

# Learning to Parallel: Accelerating Diffusion Large Language Models via Learnable Parallel Decoding

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=bFJ8Sdr224](https://openreview.net/forum?id=bFJ8Sdr224)  
**Code**: Open-sourced (Project page + GitHub, provided in the paper abstract)  
**Area**: LLM Inference Acceleration / Diffusion Language Models  
**Keywords**: diffusion LLM, parallel decoding, learnable filter, inference acceleration, LLaDA, KV-Cache  

## TL;DR
Addressing the pain point that parallel decoding in diffusion large language models (dLLMs) relies on fixed heuristics (e.g., confidence thresholds) and lacks adaptability to different inputs, this paper employs an extremely lightweight learnable filter (2-layer MLP, ~2k parameters, 6-minute training) to approximate an oracle strategy of "finalize immediately once predicted correctly." Coupled with End-of-Text early stopping, it achieves up to 22.58× acceleration on LLaDA-8B with almost no performance loss, reaching 57.51× when combined with KV-Cache.

## Background & Motivation
- **Background**: Token-by-token decoding in autoregressive LLMs requires $O(n)$ serial steps, limiting throughput. Diffusion language models (LLaDA, Dream, DiffuLLaMA, etc.) refresh entire sequences iteratively through denoising, theoretically allowing for parallel token generation. They generally adopt "block-wise left-to-right" semi-autoregressive decoding to balance quality and parallelism.
- **Limitations of Prior Work**: The key to harvesting the parallel dividend lies in "which tokens to finalize and which to remask at each step." Existing methods (confidence thresholds, Fast-dLLM, SlowFast-Sampling, Prophet, etc.) are **static, input-independent heuristics**—one-size-fits-all rules cannot adapt across different tasks/samples, leading to sub-optimal speed-quality trade-offs.
- **Key Challenge**: The authors' measurements reveal serious **redundant repetitive decoding** in dLLMs—many tokens are predicted correctly early on but are repeatedly masked and recomputed due to conservative remasking strategies. On GSM8K, most tokens are decoded over 10 times after their first correct prediction. According to EGP oracle statistics, the median steps per block is only 2, whereas the vanilla approach takes 32. Furthermore, when the generation length is set to 1024, approximately **89.59%** of computational power is wasted on repeatedly decoding padding tokens after the `[EoT]`.
- **Goal**: Replace "one-size-fits-all static rules" with a "per-sample, per-token adaptive" parallel decoding strategy to maximize parallel potential without modifying main model parameters or losing accuracy.
- **Core Idea**: **[Oracle Approximation]** First, define an ideal strategy, EGP (Extremely Greedy Parallel), using ground-truth (finalize immediately if predicted correctly), proving its 15-20× acceleration potential. Then, train a **lightweight filter using only confidence signals** to approximate this oracle during inference. **[Early Stopping]** Additionally, use EoTP to terminate subsequent padding when `[EoT]` appears.

## Method

### Overall Architecture
Learn2PD models the decision of "whether to finalize a token" as a binary classification problem. At each step, the main diffusion model provides predicted tokens and confidence $c_i$ for each position. These confidences are fed into a frozen, lightweight filter $f_\theta$, which outputs a logit for "no remasking needed" for each position. Positions exceeding a threshold $\tau$ are finalized immediately and removed from the mask set. Main model parameters are frozen throughout; $f_\theta$ is trained with minimal computation during a post-training phase. The EoTP early stopping module is added to handle long generation scenarios.

```mermaid
flowchart LR
    A[Masked Sequence X] --> B[Main Diffusion Model M<br/>Frozen]
    B -->|Predicted token + Confidence conf| C[Filter Model fθ<br/>2-layer MLP]
    C -->|logit > τ?| D{Finalization Decision}
    D -->|Yes| E[Finalize token<br/>Remove from mask set]
    D -->|No| F[Keep [MASK]<br/>Recompute next step]
    E --> G[EoTP: Detect EoT<br/>Discard subsequent positions]
    G --> B
    F --> B
```

### Key Designs
**1. EGP oracle: Quantifying "redundant decoding" as an approachable upper bound.** The authors first establish an ideal strategy, Extremely Greedy Parallel: at step $k$, a position is unmasked if and only if the model predicts $M(x_k)_i = y_i$ (where $y_i$ is the reference answer), and correct tokens are never remasked. While unavailable during inference due to its dependence on ground truth, it serves as a ceiling for "how fast parallelism can be"—achieving 15-20× acceleration without quality loss, with a median of only 2 steps per block vs. 32 for vanilla. This transforms a vague "acceleration space" into a clear, supervised learning target.

**2. Learnable filter $f_\theta$: Approximating the oracle with confidence patterns.** A key observation is that the confidence of diffusion models follows predictable fluctuation patterns—the confidence itself carries information about whether the model has "truly accepted" the prediction, sufficient to judge if a token has converged. Thus, the EGP finalization decision is distilled into a binary classification, training the filter with BCE loss:
$$\mathcal{L}_{\text{BCE}} = -\frac{1}{m}\sum_{i=1}^{m}\Big[y_i\log\sigma(z_i) + (1-y_i)\log(1-\sigma(z_i))\Big]$$
where $y_i\in\{0,1\}$ is the label from EGP (1=finalize, 0=remask), $z_i=f_\theta(\text{conf})$ is the filter logit, which is discretized after $\sigma$ by comparing with threshold $\tau$. Training involves two stages: first, running the EGP strategy to collect confidence and labels (approx. 3 hours on 4×A6000), then training the filter on this data (only 6 minutes on a T4). Surprisingly, **the simplest two-layer MLP (only 2,112 trainable parameters for block size 32) is sufficient**—the block-level confidence patterns are highly informative, requiring no complex structures or feature engineering. During inference, the filter is frozen with no gradient updates, making the additional overhead negligible.

**3. End-of-Text Prediction (EoTP): Eliminating the computational black hole of padding.** When the generation length is set to 1024 but the true answer is much shorter, extra positions are filled with `[EoT]`, and the model repeatedly decodes these paddings—accounting for 89.59% of total computation. The EoTP approach is straightforward: once an `[EoT]` token is resolved within a block at any step, **all subsequent positions are discarded**, using the shortened sequence as input for the next step to dynamically compress the effective length during denoising. It is orthogonal to Learn2PD and brings significant additional speedups specifically in long generation scenarios (a large portion of the 22.58× comes from this module).

## Key Experimental Results

### Main Results (LLaDA-8B-Instruct, TPS=tokens/sec, Score=Task Accuracy)

| Task | Method | Gen Len | TPS | Gain | Score |
|------|------|---------|-----|------|-------|
| GSM8K (5-shot) | LLaDA baseline | 1024 | 0.54 | 1.00× | 77.60 |
|  | + Learn2PD | 1024 | 6.63 | 12.21× | 77.26 |
|  | + Learn2PD + EoTP | 1024 | 12.26 | **22.58×** | 79.83 |
| Math (4-shot) | + Learn2PD + EoTP | 1024 | 12.27 | 7.22× | 34.60 |
| HumanEval (0-shot) | + Learn2PD + EoTP | 1024 | 6.63 | 12.55× | 35.98 |
| MBPP (3-shot) | + Learn2PD + EoTP | 1024 | 9.89 | 17.16× | 11.02 |

Acceleration is typically 3-5× at length 256 and 6-22× at length 1024. Accuracy generally remains within $\pm 1-2$ points of the baseline, with some tasks (GSM8K) even showing slight improvements.

### KV-Cache Compatibility + Ablation Study

| Configuration | TPS | Gain | Score |
|------|-----|------|-------|
| Learn2PD & EoTP | 12.26 | 22.58× | 79.83 |
| + Dual Cache | 31.23 | **57.51×** | 74.00 |
| + Prefix Cache | 14.79 | 27.23× | 77.71 |

| Filter Depth | TPS | Gain | Score |
|-------------|-----|------|-------|
| 1-layer | 8.77 | 2.57× | 78.62 |
| **2-layer** | 14.07 | 4.13× | 78.62 |
| 4-layer | 11.41 | 3.35× | 78.85 |

### Key Findings
- **The method is completely orthogonal to KV-Cache**, combining with Dual Cache pushes acceleration to 57.51× (accuracy drops slightly to 74.00), while Prefix Cache reaches 27.23× with almost no performance loss.
- **A 2-layer MLP is the sweet spot**: A single layer lacks representation power, and four layers slightly increase accuracy but decrease speed; 2 layers provide the best efficiency/quality trade-off.
- **Longer generation lengths yield higher returns**: Acceleration increases from 3.36× at 128 to 22.58× at 1024, as EoTP can eliminate more padding redundancy in longer sequences.

## Highlights & Insights
- **Proving the "acceleration space" before approximation**: The EGP oracle step is elegant—quantifying the 15-20× ceiling with ground truth proves that redundancy is both real and massive, followed by distillation with a learnable filter, creating a convincing logical loop.
- **Extremely lightweight post-training**: Only 2k parameters trained in 6 minutes on a T4 with the main model frozen. It is a plug-and-play solution with near-zero cost and high engineering friendliness.
- **Orthogonal stacking**: Learn2PD (addressing intra-block redundant remasking) + EoTP (addressing long sequence padding) + KV-Cache (addressing inter-step recomputation) multiply as three independent paths, leading to double-digit or even 57× composite acceleration.

## Limitations & Future Work
- **Validated only on LLaDA-8B**: Generalization to other dLLMs like Dream or DiffuLLaMA has not been fully demonstrated.
- **Filter supervision relies on "reference answers"**: Labels come from reference answers generated by LLaDA's own standard decoding, essentially distilling its own behavior. The filter's upper bound is constrained by the main model's quality; if the main model is wrong, the filter learns to "confidently finalize a wrong token."
- **Quality risks of aggressive finalization**: Accuracy drops from 79.83 to 74.00 when Dual Cache is added, indicating that quality is not entirely lossless when pursuing extreme speed. The combination of $\tau$ and cache strategies requires careful tuning.
- **Training data consists of 2640 samples across 66 FLAN categories**: The cross-domain robustness of the filter, especially for out-of-distribution tasks, requires further verification.

## Related Work & Insights
- **Comparison with static acceleration methods**: Fast-dLLM (confidence threshold + approximate KV-Cache), SlowFast-Sampling (two-stage sampler), Prophet (early submission based on top-2 logit gap), and dllm-Cache/FreeCache (training-free caching)—they are all fixed rules; the primary difference of this paper is making the "finalization decision" **learnable and input-adaptive**.
- **Insight**: Explicitly defining an unavailable oracle (dependent on ground truth), quantifying its upper bound, and then approximating it with the cheapest possible supervised model is a universal and cost-effective acceleration paradigm. It can be transferred to other "to stop or to trust" decision scenarios like speculative decoding or early-exit. The fact that confidence signals alone are sufficient input features suggests that the internal uncertainty structure of diffusion models is highly readable and worth further exploration.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — The combination of "EGP oracle quantification + lightweight filter approximation" is the first learnable strategy for dLLM parallel decoding, with a clear and solid rationale.
- **Experimental Thoroughness**: ⭐⭐⭐ — Four benchmarks + KV-Cache compatibility + depth/length ablations are relatively complete, but the lack of cross-dLLM generalization (using only LLaDA) is a drawback.
- **Writing Quality**: ⭐⭐⭐⭐ — The narrative chain from "discovering redundancy $\rightarrow$ defining oracle $\rightarrow$ proving potential $\rightarrow$ distillation approximation" is very smooth and well-supported by figures.
- **Value**: ⭐⭐⭐⭐ — Near-zero cost, plug-and-play, and orthogonal to existing accelerations; it has direct practical value for dLLM deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Hierarchy Decoding: A Training-free Parallel Decoding Strategy for Diffusion Large Language Models](hierarchy_decoding_a_training-free_parallel_decoding_strategy_for_diffusion_larg.md)
- [\[ICLR 2026\] dParallel: Learnable Parallel Decoding for dLLMs](dparallel_learnable_parallel_decoding_for_dllms.md)
- [\[ICLR 2026\] Fast-dLLM: Training-free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding](fast-dllm_training-free_acceleration_of_diffusion_llm_by_enabling_kv_cache_and_p.md)
- [\[ACL 2026\] CreditDecoding: Accelerating Parallel Decoding in Diffusion Large Language Models with Trace Credit](../../ACL2026/llm_efficiency/creditdecoding_accelerating_parallel_decoding_in_diffusion_large_language_models.md)
- [\[ICLR 2026\] ReFusion: A Diffusion Large Language Model with Parallel Autoregressive Decoding](refusion_a_diffusion_large_language_model_with_parallel_autoregressive_decoding.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[ICLR 2026\] dParallel: Learnable Parallel Decoding for dLLMs](dparallel_learnable_parallel_decoding_for_dllms.md)
- [\[ACL 2026\] CreditDecoding: Accelerating Parallel Decoding in Diffusion Large Language Models with Trace Credit](../../ACL2026/llm_efficiency/creditdecoding_accelerating_parallel_decoding_in_diffusion_large_language_models.md)
- [\[ICLR 2026\] Hierarchy Decoding: A Training-free Parallel Decoding Strategy for Diffusion Large Language Models](hierarchy_decoding_a_training-free_parallel_decoding_strategy_for_diffusion_larg.md)
- [\[ICLR 2026\] ReFusion: A Diffusion Large Language Model with Parallel Autoregressive Decoding](refusion_a_diffusion_large_language_model_with_parallel_autoregressive_decoding.md)
- [\[ICLR 2026\] Parallel Sampling from Masked Diffusion Models via Conditional Independence Testing](parallel_sampling_from_masked_diffusion_models_via_conditional_independence_test.md)

</div>

<!-- RELATED:END -->
