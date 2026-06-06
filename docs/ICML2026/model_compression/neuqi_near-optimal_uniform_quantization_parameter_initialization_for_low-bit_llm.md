---
title: >-
  [Paper Note] NeUQI: Near-Optimal Uniform Quantization Parameter Initialization for Low-Bit LLMs
description: >-
  [ICML 2026][Model Compression][Post-Training Quantization] This paper points out that mainstream post-training quantization (PTQ) methods inherit the Min-Max formula to initialize scale and zero-point…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "Post-Training Quantization"
  - "Uniform Quantization"
  - "Quantization Parameter Initialization"
  - "Scale-Zero Point Optimization"
  - "LLM Deployment"
date: 2026-05-08
content_hash: da65e6eb883492a8
---

# NeUQI: Near-Optimal Uniform Quantization Parameter Initialization for Low-Bit LLMs

**Conference**: ICML 2026  
**arXiv**: [2505.17595](https://arxiv.org/abs/2505.17595)  
**Code**: None currently public (inference supported by BitBLAS for floating-point zero-points)  
**Area**: Model Compression / Low-Bit Quantization / PTQ  
**Keywords**: Post-Training Quantization, Uniform Quantization, Quantization Parameter Initialization, Scale-Zero Point Optimization, LLM Deployment

## TL;DR
This paper points out that mainstream post-training quantization (PTQ) methods inherit the Min-Max formula to initialize scale and zero-point, which imposes two long-overlooked constraints: "parameters determined by extreme values" and "zero-point must be an integer." The authors propose NeUQI, which breaks these constraints by analytically solving for the optimal zero-point given a scale and employing a coarse-to-fine scale search. On LLaMA-2 7B 2-bit per-channel quantization, it reduces the C4 perplexity from the Prev. SOTA of 47.55 (MagR) to 17.50, and outperforms much more expensive PV-tuning after lightweight distillation.

## Background & Motivation

**Background**: Local deployment of LLMs (consumer-grade GPUs, laptops) is gaining momentum. The mainstream approach is PTQ to quantize BF16 weights to int2/3/4. Among all PTQ paradigms, **uniform quantization (affine quantization)** is the de facto industry standard due to native hardware support and simple inference kernels. The academic community has refined quantization methods through GPTQ → AWQ → QuIP → QuaRot → MagR → GPTAQ.

**Limitations of Prior Work**: For the two core parameters—scale $s$ and zero-point $z$—nearly all methods use the same legacy formula (Jacob et al. 2017 Min-Max) for initialization: $s = (\max(\bm{x}) - \min(\bm{x}))/(2^k - 1)$ and $z = -\lfloor \min(\bm{x})/s \rceil$. While sufficient for 8-bit/4-bit, performance collapses at 3-bit/2-bit (e.g., GPTQ's C4 PPL on LLaMA-2 7B 2-bit is 2592). The lack of attention to initialization while methods become increasingly complex is a blind spot in the field.

**Key Challenge**: The Min-Max formula hides two long-ignored hard constraints: (i) **Extreme-value-determined parameters**: Both $s$ and $z$ are bound by the extreme values of $\bm{x}$. Search algorithms (e.g., LeanQuant) can only search on a massive $T^2$ ($T=2048$) 2D grid of extreme-value candidate pairs; however, searching directly in the $(s, z)$ space requires only $2^k T$, which is orders of magnitude smaller. (ii) **Integer zero-point constraint**: Restricting $z$ to a $k$-bit unsigned integer compresses the parameter space; at $k=2$, only 4 candidates remain, leading to high search failure rates.

**Goal**: (i) Eliminate the two Min-Max constraints to allow floating-point $z$ and decoupled optimization of $s$ and $z$. (ii) Analytically solve for the optimal floating-point $z$ for a given $s$ in $\mathcal{O}(n \log n)$ time. (iii) Demonstrate that "better initialization" can outperform "more complex fine-tuning."

**Key Insight**: The authors re-examine the weight quantization loss (GPTQ-style, Hessian diagonal approximation) $\mathcal{L}(s, z) = \sum_i H_{i,i} (Q_{s,z}(w_i) - w_i)^2$ and discover that **for a fixed scale, $\mathcal{L}(z)$ is a piecewise quadratic function** with $n(2^k - 1)$ turning points. This means for every fixed scale, the optimal zero-point can be solved exactly in closed form, reducing 2D joint optimization to 1D.

**Core Idea**: Decompound "joint scale and zero-point search" into "outer 1D scale search + inner analytical zero-point optimization." Acceleration via transition-point reduction and coarse-to-fine search reduces per-layer quantization time from 112 seconds to several seconds.

## Method

### Overall Architecture
- **Input**: Single-layer weight matrix $\bm{W}$ and proxy Hessian $\bm{H} = \mathbb{E}_{\bm{X}}[\bm{X}^\top \bm{X}]$ (GPTQ-style diagonal approximation $H_{i,i}$).
- **Row-wise Quantization**: Optimize $(s, z)$ independently for each row $\bm{w}$ of $\bm{W}$.
- **Outer Loop**: Starting from the upper bound given by Min-Max, uniformly sample $T$ candidate scales. Use $T_c = O(\sqrt{T})$ candidates for a coarse search to find $s^c$, then perform a fine search in the neighborhood of $s^c$ with roughly $T/T_c$ candidates.
- **Inner Loop**: For each candidate scale, invoke Algorithm 1 (global minimum search for piecewise quadratic functions) to analytically find the optimal floating-point zero-point $z^*$ and its loss.
- **Output**: Choose $\arg\min_s \mathcal{L}(s, z^*(s))$ as the final quantization parameters.

### Key Designs

1.  **Closed-form Optimal Zero-point Solver**:
    - **Function**: Given scale $s$, find the floating-point $z^*$ that minimizes quantization error in $\mathcal{O}(n \log n)$ time, free from integer constraints.
    - **Mechanism**: The single-element loss $\mathcal{L}_i(z) = h_i (x_i + z - \mathrm{clip}(\lfloor x_i+z \rceil, 0, 2^k-1))^2$ (where $x_i = w_i/s$, $h_i = H_{i,i} s^2$) is a piecewise quadratic function of $z$ with $2^k-1$ turning points. The total loss $\mathcal{L}(z) = \sum_i \mathcal{L}_i(z)$ is also piecewise quadratic on the union of all turning points (totaling $n(2^k-1)$). Algorithm 1 uses **incremental updates** to maintain the quadratic function $\mathcal{L}^I(z) \leftarrow \mathcal{L}^I(z) + \delta(z)$ between adjacent intervals. The complexity is $\mathcal{O}(n \cdot 2^k \log(n \cdot 2^k))$, dominated by sorting.
    - **Design Motivation**: The floating-point zero-point is a "free parameter" locked by legacy formulas. Once unlocked, the optimal $z$ does not necessarily fall on the $w_i + j$ grid. The incremental update trick reduces complexity from $\mathcal{O}(n^2 2^k)$ to $\mathcal{O}(n 2^k \log(n 2^k))$ by utilizing the adjacent incremental structure.

2.  **Transition-point Reduction**:
    - **Function**: Further reduces inner loop complexity from $\mathcal{O}(n 2^k \log(n 2^k))$ to $\mathcal{O}(n \log n)$.
    - **Mechanism**: $\mathcal{L}_i(z)$ is capped by the rounding loss upper bound $h_i / 4$ in the middle segment $[-1/2 - x_i, 2^k - 1/2 - x_i]$. By constructing a surrogate function $\mathcal{L}_i^S(z)$ that replaces the middle with a constant $h_i/4$, each sample is left with only two transition points. Algorithm 1 finds an approximate $z^S$, followed by local refinement in $[z^S - 1, z^S + 1]$ using the original $\mathcal{L}_i(z)$.
    - **Design Motivation**: While there are $n \cdot 2^k$ breaks, the global minimum won't be pulled away by distant saturated tails. A coarse localization followed by local refinement is a standard "outer bound + local refine" strategy applied cleanly to quantization.

3.  **Coarse-to-fine Scale Search**:
    - **Function**: Reduces $T = 2048$ inner solver calls to $\mathcal{O}(\sqrt{T}) \approx 90$.
    - **Mechanism**: Candidate set $\mathcal{S}_T = \{ ((\max(\bm{w}) - \min(\bm{w}))/(2^k - 1)) \cdot (i/T) : i = 1, \dots, T \}$. It first searches $T_c = O(\sqrt{T})$ points to find $s^c$, then refines in its vicinity.
    - **Design Motivation**: Empirically, $\mathcal{L}(s, z^*(s))$ is unimodal or smooth with respect to $s$, making full grid search unnecessary.

### Loss & Training
**Pure PTQ, no gradient training**. The loss is the diagonal Hessian-weighted MSE from Eq. 5. It can optionally be combined with strong downstream fine-tuning (PV-tuning, EfficientQAT). NeUQI provides a superior starting point, allowing fine-tuning to reach or exceed original performance with fewer resources. The calibration set is consistent with GPTQ (small WikiText/C4 samples).

## Key Experimental Results

### Main Results

LLaMA-2 7B 2-bit per-channel quantization (Wiki2 ↓ / C4 ↓ PPL, average ↑ Acc across five zero-shot benchmarks):

| Method | Wiki2 PPL | C4 PPL | Avg Acc | Remarks |
| :--- | :--- | :--- | :--- | :--- |
| GPTQ | 6953 | 2592 | 35.08% | Complete collapse |
| GPTAQ | 1269 | 246 | 34.97% | Still collapsed |
| MagR† | 129 | 47.55 | 39.54% | Prev. SOTA |
| **Ours (NeUQI)** | **17.14** | **17.50** | **47.24%** | ~3× Gain / +7.7 pp |

Consistent trends are observed on LLaMA-3 8B, LLaMA-2 70B, and Qwen-2.5 7B/14B. On LLaMA-2 70B 2-bit, NeUQI achieves Wiki2 7.03 / C4 8.88 / Acc 65.13% (vs. MagR† at 13.95 / 68.62 / 61.80%).

### Ablation Study

Efficiency ablation (LLaMA-2 7B 2-bit, per-layer query projection time):

| Configuration | Relative Loss | Relative Time | Absolute Time (s) |
| :--- | :--- | :--- | :--- |
| Baseline (No acceleration) | 1.0000× | 1.000× | 112 |
| w/ transition-point reduction | 1.00197× | 0.548× | ~61 |
| **Ours (NeUQI Full)** | **1.00193×** | **0.027×** | **~3** |

- Transition-point reduction saves ~50% time; coarse-to-fine search removes another ~95%.
- Total loss degradation from all accelerations is < 0.2%.

### Key Findings
- **Superior initialization is more cost-effective than expensive fine-tuning**: NeUQI without fine-tuning reduces LLaMA-2 7B 2-bit C4 PPL from 47.55 to 17.50. With distillation, it surpasses PV-tuning.
- **Integer zero-points are the primary constraint**: Allowing floating-point $z$ reduces PPL by up to 15.54% and increases average Acc by 3.95 pp at the same bit-width. The overhead is minimal (~0.05 bit per channel).
- **3-bit NeUQI Qwen-2.5 can outperform BF16 at equivalent memory footprints**: Figure 1 shows NeUQI 3-bit is strictly better than smaller BF16 models with the same footprint.

## Highlights & Insights
- **Revisiting "standard" engineering formulas**: The Min-Max formula has been used since 2017. While everyone modified the quantization method, few questioned the initialization. This paper identifies specific, tackleable constraints within legacy baselines.
- **Clean algorithm engineering**: Reducing an $\mathcal{O}(n^2 2^k)$ brute-force problem to $\mathcal{O}(n \log n)$ using geometric motivations (bounds, incremental updates, fine search) is a transferable approach for piecewise convex quantization losses.
- **Counter-intuitive "Initialization > Fine-tuning"**: While current trends rely on heavy fine-tuning to save low-bit models, this work shows that fine-tuning cannot fully recover a poor starting point.

## Limitations & Future Work
- **Weight-only per-channel focus**: Activation quantization and group-wise/mixed-precision settings were not systematically tested.
- **Hessian dependency**: Relies on GPTQ-style diagonal Hessian approximations, which may ignore cross-weight interactions in highly coupled attention matrices.
- **Hardware friendliness**: While the authors argue for hardware support in Appendix E, actual deployment throughput data for floating-point zero-points is missing.
- **Generalization**: The approach could potentially be extended to non-uniform schemes like logarithmic or power-of-two quantization.

## Related Work & Insights
- **vs GPTQ / GPTAQ**: These use OBS for weight rounding but retain Min-Max for parameters. NeUQI modifies the shared initialization layer and is orthogonal to rounding strategies.
- **vs LeanQuant**: LeanQuant performs a $T^2 \approx 4M$ 2D grid search. NeUQI’s decoupled analytical solver is over 1000× theoretically more efficient.
- **vs OmniQuant / EfficientQAT**: These treat clipping as learnable parameters. NeUQI achieves similar or better results analytically without backpropagation.

## Rating
- Novelty: ⭐⭐⭐⭐ High; identifying constraints in 2017-era formulas is an underrated insight.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage across multiple model families and bit-widths.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, though Algorithm 1 details are relegated to the Appendix.
- Value: ⭐⭐⭐⭐⭐ Vital for making 2-bit 70B models usable on consumer hardware.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] WUSH: Near-Optimal Adaptive Transforms for LLM Quantization](wush_near-optimal_adaptive_transforms_for_llm_quantization.md)
- [\[ICML 2026\] LFQ: Logit-aware Final-block Quantization for Boosting the Generation Quality of Low-Bit Quantized LLMs](lfq_logit-aware_final-block_quantization_for_boosting_the_generation_quality_of_.md)
- [\[ICML 2026\] NanoQuant: Efficient Sub-1-Bit Quantization of Large Language Models](nanoquant_efficient_sub-1-bit_quantization_of_large_language_models.md)
- [\[ICML 2026\] OSAQ: Outlier Self-Absorption for Accurate Low-bit LLM Quantization](osaq_outlier_self-absorption_for_accurate_low-bit_llm_quantization.md)
- [\[AAAI 2026\] SpecQuant: Spectral Decomposition and Adaptive Truncation for Ultra-Low-Bit LLMs Quantization](../../AAAI2026/model_compression/specquant_spectral_decomposition_and_adaptive_truncation_for_ultra-low-bit_llms_.md)

</div>

<!-- RELATED:END -->
