---
title: >-
  [Paper Note] DyLLM: Efficient Diffusion LLM Inference via Saliency-based Token Selection and Partial Attention
description: >-
  [ICML2026][Image Restoration][Diffusion Language Models] DyLLM is a training-free inference acceleration framework for Diffusion LLMs. It identifies "salient tokens" by calculating the cosine similarity of attention cont…
tags:
  - "ICML2026"
  - "Image Restoration"
  - "Diffusion Language Models"
  - "Inference Acceleration"
  - "Temporal Sparsity"
  - "Salient Token Selection"
  - "Approximate Attention"
date: 2026-05-08
content_hash: 76edbb703e1a0ea5
---

# DyLLM: Efficient Diffusion LLM Inference via Saliency-based Token Selection and Partial Attention

**Conference**: ICML2026  
**arXiv**: [2603.08026](https://arxiv.org/abs/2603.08026)  
**Code**: https://github.com/scale-snu/DyLLM.git (Available)  
**Area**: LLM Efficiency  
**Keywords**: Diffusion Language Models, Inference Acceleration, Temporal Sparsity, Salient Token Selection, Approximate Attention

## TL;DR
DyLLM is a training-free inference acceleration framework for Diffusion LLMs. It identifies "salient tokens" by calculating the cosine similarity of attention contexts between adjacent denoising steps, recomputing FFN and attention only for these tokens. Combined with saliency-aware approximate attention, it achieves throughput gains of 7.6× and 9.6× on LLaDA and Dream, respectively, with negligible performance degradation.

## Background & Motivation

**Background**: Masked Diffusion Language Models (MDLM, e.g., LLaDA, Dream, Gemini Diffusion) use bidirectional attention to "fill in the blanks" for an entirely masked sequence at once. They have approached the performance of AR LLMs like Llama on benchmarks such as GSM8K and MBPP, offering the potential for parallel decoding and breaking the sequential token-by-token constraint.

**Limitations of Prior Work**: MDLMs recompute the **entire sequence** at each denoising step ("repeated prefill") because bidirectional attention lacks a fixed causal order, preventing incremental KV cache reuse as in AR models. Consequently, FFN dominates the computation in every step, and overall throughput is consumed by "repeated prefixes," leaving no advantage over AR decoding enhanced by vLLM.

**Key Challenge**: Parallel decoding (multiple tokens per step, bidirectional attention) and cache reuse (requiring position/context stability) are inherently in conflict. Existing solutions either refresh based on fixed blocks/schedules (e.g., PrefixCache/DualCache in Fast-dLLM, dKV-Cache) or select tokens based on global thresholds (e.g., dLLM-Cache, Elastic-Cache). These do not capture the fine-grained structure where only a small number of tokens change in each layer at each step.

**Goal**: To reduce the FFN + Attention computation in each step from the full sequence to a **layer-adaptive and token-adaptive** small subset without retraining or modifying model weights, while maintaining generation quality.

**Key Insight**: The authors plotted the distribution of cosine similarity $s_{t,l}^{(i)}$ for attention contexts $C_{t,l}^{(i)}$ between adjacent steps in LLaDA (Fig. 2). Most tokens show $s\approx 1$ across all layers, but the "tails" in deeper layers are thicker, indicating: (1) **Temporal sparsity** truly exists; (2) Sparsity **varies by layer**, with deeper layers requiring more updates. This provide a per-layer sparse selector.

**Core Idea**: Utilize the cosine similarity of attention contexts from adjacent steps as a saliency measure. Only recompute "salient tokens" for FFN and attention in each layer/step. Updates for non-salient token attention contexts are approximated as "gathering $\Delta V$ only from salient columns," thereby simultaneously exploiting sparsity in both FFN and attention.

## Method

### Overall Architecture
DyLLM wraps a "Saliency Scheduler" around the standard MDLM decoding loop. For each step $t$ and each layer $l$, it performs three actions:

1.  **Saliency Calculation**: For each token $i$, calculate $s_{t,l}^{(i)} = \cos(C_{t,l}^{(i)}, C_{t-1,l}^{(i)})$. Tokens below a threshold $\tau$ (0.99 for LLaDA / 0.995 for Dream) enter the salient set $\mathcal{A}_{t,l}$.
2.  **Sparse Forwarding**: FFN is only recomputed for tokens within $\mathcal{A}_{t,l}$; others directly read from the FFN output cache. Attention is computed via a "dual-path" approach (Sec 3.3).
3.  **Cold Start & Response-only Steps**: The first $T_{full}=4$ steps involve full computation to warm up the cache. Subsequently, most steps only feed response tokens into the model, with a full sequence including the prompt supplemented every 4 steps.

Essentially, the "prefill → decode" two-stage concept is adapted for diffusion decoding: initial full steps fill the cache, followed by "Salient-only" stages.

### Key Designs

1.  **Layer-adaptive Salient Token Selection (Saliency Selection)**:
    - **Function**: Identifies the subset of tokens $\mathcal{A}_{t,l} = \{i \mid s_{t,l}^{(i)} < \tau\}$ that truly require updating at each layer $l$ and step $t$, using the directional drift of adjacent attention contexts as a proxy.
    - **Mechanism**: Uses the directional cosine similarity of $C_{t,l}^{(i)}$ (the result of $\text{softmax}(QK^T)V$) as the metric. Non-salient tokens reuse the FFN output cache from the previous step, skipping FFN and corresponding attention row recomputation. This is theoretically supported by Prop 3.1 (RMSNorm scale-insensitivity under linear projection) and Prop 3.2 (FFN input perturbation upper bound $\delta \le \kappa(W_o)\sqrt{2(1-s_{t,l})}$), ensuring near-zero error as $s\to 1$.
    - **Design Motivation**: Sparsity varies significantly across layers (near 1 in early layers, thicker tails in deep layers). Fixed-threshold methods like dLLM-Cache/Fast-dLLM either waste computation or over-prune. Per-layer thresholding naturally allows aggressive pruning in early layers and automatic conservation in deep layers, distributing the "error budget" layer-wise.

2.  **Saliency-aware Approximate Attention**:
    - **Function**: Decomposes attention updates into "exact calculation for salient rows + approximation using salient columns for non-salient rows," compressing complexity from $O(N^2 d)$ to $O(N \cdot |\mathcal{A}_{t,l-1}| d)$.
    - **Mechanism**: Expands context increments as $\Delta C_{t,l} = S_{t,l}\Delta V_{t,l} + (\Delta S) V_{t-1,l}$. For salient queries $i\in\mathcal{A}_{t,l-1}$, the full attention row is recomputed (row-sparse path). For non-salient queries, $\Delta S^{(i,\cdot)}\approx 0$, so the update simplifies to $\Delta C_{t,l}^{(i)} \approx S_{t,l}^{(i,\cdot)} \Delta V_{t,l}$. Since $\Delta V_{t,l}$ is non-zero only in salient columns, one only needs to gather attention scores from salient columns (column-sparse path).
    - **Design Motivation**: Skipping FFN alone is insufficient given the quadratic overhead of attention. This decomposition allows the salient set to determine both which query rows are recomputed and which KV columns are aggregated. Attention and FFN share the same sparsity mask, facilitating implementation in a FlashAttention-like fused kernel.

3.  **Response-only Step Scheduling**:
    - **Function**: Leverages the locality bias from RoPE's relative distance decay, causing salient tokens to cluster in the response segment. Prompt tokens are included only every fixed number of steps (every 4 steps in this paper).
    - **Mechanism**: Unlike prior cache works (dKV-Cache / Fast-dLLM) which utilize a rigid "full token refresh" cycle that drags throughput back to baseline, DyLLM avoids full refreshes. Even when prompts are included, computation remains sparse based on the salient set; the stable prompt context almost entirely hits the cache.
    - **Design Motivation**: This removes the primary bottleneck of prior work—"periodic full refreshes"—allowing DyLLM to maintain linear gains as $n_u$ (number of parallel tokens) increases.

### Loss & Training
Completely training-free with no fine-tuning or extra loss. Hyperparameters include only the saliency threshold $\tau$ (model-dependent, task-agnostic) and initial full steps $T_{full}=4$. Custom sparse attention/cache kernels (FlashAttention-like fused design) were written in CUDA to avoid the high overhead of native PyTorch sparse operators for fine-grained token sparsity.

## Key Experimental Results

### Main Results

Evaluated on LLaDA 8B Instruct / Dream 7B Instruct with $n_u=1$ on a single H100 80GB PCIe card. Comparisons against Original / Fast-dLLM (Prefix & Dual) / dLLM-Cache (Throughput unit: tokens/s):

| Model | Bench | Original (acc / tput) | DyLLM (acc / tput / speedup) | Fast-dLLM Dual | dLLM-Cache |
| :--- | :--- | :--- | :--- | :--- | :--- |
| LLaDA 8B | GSM8K | 77.79 / 11.47 | **79.08 / 87.21 / ×7.60** | 78.24 / 75.24 (×6.56) | 77.18 / 36.77 (×3.21) |
| LLaDA 8B | MATH | 33.22 / 15.81 | **38.68 / 96.98 / ×6.13** | 32.36 / 93.26 (×5.90) | 24.70 / 36.56 (×2.31) |
| LLaDA 8B | MBPP | 29.20 / 33.11 | **30.00 / 169.62 / ×5.12** | 25.40 / 165.44 (×5.00) | 29.00 / 93.04 (×2.81) |
| Dream 7B | GSM8K | 75.59 / 12.57 | **79.30 / 111.79 / ×8.89** ($\tau$=0.9975) | 68.39 / 153.21 (×12.19) | 72.40 / 46.19 (×3.67) |
| Dream 7B | MATH | 37.60 / 17.64 | **45.12 / 130.57 / ×7.40** | 36.06 / 191.56 (×10.86) | 44.98 / 48.76 (×2.76) |
| Dream 7B | MMLU-Pro | 47.94 / 12.60 | 47.45 / 83.10 / ×6.60 | 46.73 / 128.52 (×10.20) | 49.30 / 27.19 (×2.16) |

Note: Fast-dLLM DualCache achieves higher throughput on Dream but accuracy on GSM8K drops to 68.39 (−7.2 acc) due to periodic full refreshes and block-cache mis-pruning. DyLLM consistently leads dLLM-Cache in throughput by 2.16–3.67× while maintaining accuracy.

### Ablation Study

| Config (LLaDA, GSM8K) | Acc | Description |
| :--- | :--- | :--- |
| Original | 77.79 | Baseline |
| Salient-only FFN (τ=0.995) | 78.09 | FFN skip by saliency only; full attention |
| Salient + Approx. (τ=0.995) | 78.01 | Full DyLLM (FFN + Approx. attention) |
| τ=0.99 | 79.08 | Aggressive threshold; accuracy increases (softmax noise suppression) |
| τ=0.985 | 78.62 | Lower threshold starts to show slight accuracy drop |

When combined with confidence-aware parallel decoding: DyLLM (τ=0.9975) increases average $n_u$ to **3.92** with 77.10 acc, whereas Fast-dLLM Dual achieves $n_u=3.68$ but with acc dropping to 67.85. This proves that sparse updates and parallel decoding are compatible, provided salient tokens are precisely updated.

### Key Findings
- **"Passive Denoising" of Softmax Noise**: Lower thresholds (τ=0.99) slightly improve accuracy. The authors attribute this to the fact that softmax always assigns positive weights to all tokens; as sequences lengthen, the cumulative contribution of low-relevance tokens introduces noise. DyLLM explicitly truncates these based on saliency, acting as an implicit regularizer.
- **Greater Benefits from GQA**: Dream uses GQA, making attention relatively cheap while FFN accounts for 70%+ of runtime. DyLLM's FFN sparsity targets this primary workload, leading to nearly 2× higher speedups on Dream compared to LLaDA.
- **$n_u$ Scalability as a Killer Feature**: Fast-dLLM must perform a full refresh every $B=32$ steps. As $n_u$ increases, the refresh frequency rises, and throughput degrades. DyLLM has no full refresh steps, allowing the throughput slope to remain steeper as $n_u\in\{1,2,4\}$.
- **τ is Model-dependent, Task-agnostic**: The threshold only needs tuning once per model (LLaDA=0.99 / Dream=0.995) and generalizes across GSM8K, MBPP, MATH, and MMLU-Pro.

## Highlights & Insights
- Reconceptualizes "Diffusion LLM acceleration" from an engineering perspective of KV/activation caching to a more physical perspective of "updating only the tokens that are truly moving" at each layer and step. The saliency measure (attention context cosine similarity) is simple and provable (Prop 3.2).
- **A single sparsity mask drives both FFN skipping and dual-path attention.** This converges sparse patterns into one mask, enabling efficient implementation in FlashAttention-like fused kernels. This "shared active set" idea can be migrated to any iterative model with sequence-wide recomputation (e.g., multi-step refinement in image diffusion or world models).
- Solves the "Achilles' heel" of cache-based methods—performance degradation during $n_u$ increases due to full refreshes. This observation is valuable for any work combining parallel decoding with caching.

## Limitations & Future Work
- **Extra Memory Overhead**: Requires storing attention output and FFN output caches in addition to KV cache. Memory expands by factor of approx $(2d/g + 2d)/(2d/g)$. While manageable for small batches, this may stress VRAM in large-batch or edge deployments.
- **$\tau$ requires per-model calibration**: Although task-agnostic, changing models (e.g., Gemini Diffusion) requires re-scanning $\tau$. There is a lack of automatic or layer-adaptive thresholding mechanisms.
- **Dependence on Temporal Sparsity**: Benefits may diminish in early denoising steps or for models where sparsity is less pronounced.
- **Online Serving Scenarios**: All results are offline batched on H100s. Variable batch sizes, prefill/decode decoupling, and KV cache reuse across requests remain open questions.

## Related Work & Insights
- **vs. Fast-dLLM**: Fast-dLLM uses fixed block caching and periodic full refreshes, which can ignore important tokens and hurt throughput. DyLLM uses per-layer/step adaptive active sets with **no full refreshes**, offering superior $n_u$ scalability.
- **vs. dKV-Cache**: dKV-Cache uses a time-windowed cache without token-level selectivity. DyLLM is sparse in both token and layer dimensions.
- **vs. dLLM-Cache / Elastic-Cache**: Both use activation similarity for token selection but rely on global thresholds requiring per-model/dataset tuning. DyLLM is faster (2.16–3.67×) and generalizes across tasks better.
- **vs. Sparse Attention**: Traditional sparse attention focuses on "which KV columns to attend to"; DyLLM focuses on "which query rows are actually changing," which is better suited for the iterative refinement paradigm of diffusion models.

## Rating
- Novelty: ⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Encoder-Decoder Diffusion Language Models for Efficient Training and Inference](../../NeurIPS2025/image_restoration/encoder-decoder_diffusion_language_models_for_efficient_training_and_inference.md)
- [\[ICLR 2026\] Skip to the Good Part: Representation Structure & Inference-Time Layer Skipping in Diffusion vs. Autoregressive LLMs](../../ICLR2026/image_restoration/skip_to_the_good_part_representation_structure_inference-time_layer_skipping_in_.md)
- [\[ICML 2026\] DAPD: Dependency-Aware Parallel Decoding via Attention for Diffusion LLMs](dapd_dependency-aware_parallel_decoding_via_attention_for_diffusion_llms.md)
- [\[ICLR 2026\] Beyond Scattered Acceptance: Fast and Coherent Inference for DLMs via Longest Stable Prefixes](../../ICLR2026/image_restoration/beyond_scattered_acceptance_fast_and_coherent_inference_for_dlms_via_longest_sta.md)
- [\[NeurIPS 2025\] Spiking Meets Attention: Efficient Remote Sensing Image Super-Resolution with Attention Spiking Neural Networks](../../NeurIPS2025/image_restoration/spiking_meets_attention_efficient_remote_sensing_image_super-resolution_with_att.md)

</div>

<!-- RELATED:END -->
