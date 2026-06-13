---
title: >-
  [Paper Note] ProjQ: Project-and-Quantize for Adapter-Aware LLM Compression
description: >-
  [ICML 2026][Model Compression][Post-Training Quantization] ProjQ actively "shapes" the quantization noise of PTQ into a low-rank subspace and leaves this part for subsequent LoRA adapters to eliminate…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "Post-Training Quantization"
  - "LoRA"
  - "Subspace Projection"
  - "Activation-Aware"
  - "Low-Rank Noise Shaping"
date: 2026-05-08
content_hash: e7c4b7df4c577907
---

# ProjQ: Project-and-Quantize for Adapter-Aware LLM Compression

**Conference**: ICML 2026  
**arXiv**: [2606.00494](https://arxiv.org/abs/2606.00494)  
**Code**: https://github.com/yy9301/ProjQ  
**Area**: Model Compression / LLM Efficiency  
**Keywords**: Post-Training Quantization, LoRA, Subspace Projection, Activation-Aware, Low-Rank Noise Shaping

## TL;DR
ProjQ actively "shapes" the quantization noise of PTQ into a low-rank subspace and leaves this part for subsequent LoRA adapters to eliminate, thereby preserving LoRA's capacity to learn downstream tasks. On LLaMA-2 / Qwen2.5 / Qwen3, it achieves performance at 3 bits comparable to standard 4-bit baselines.

## Background & Motivation

**Background**: Compressing large models via PTQ (e.g., GPTQ, AWQ, QuIP) to 3-4 bits followed by attaching a LoRA adapter for downstream fine-tuning (e.g., QLoRA, LoftQ) has become the de facto standard pipeline for LLM deployment.

**Limitations of Prior Work**: Current practices treat quantization and adaptation as two independent stages. The PTQ stage focuses solely on minimizing "weight reconstruction error," thus scattering noise uniformly across all singular directions. In the LoRA stage, an adapter with a very small rank $r$ must use a large portion of its capacity to "clean" these multi-directional noises, resulting in a "double burden."

**Key Challenge**: LoRA can only repair errors in low-rank directions, while PTQ tends to produce full-rank, isotropic noise. The geometric misalignment between the two stages—where the repairable subspace is small but the error fills the entire space—means LoRA is perpetually patching holes rather than learning tasks.

**Goal**: To "squeeze" the noise into a low-rank subspace that LoRA can repair during the quantization stage, minimizing unrepairable residuals in the orthogonal complement space, thus releasing LoRA's capacity from "noise patching" to "task learning."

**Key Insight**: The authors start from a geometric observation: "repairing a low-rank error of rank $r_d$" $\equiv$ "finding an orthogonal projection $P$ of rank $r_d$ that maximizes the captured error." While downstream task directions are unknown, the subspace structure is controllable. Therefore, the quantizer should prioritize "repairability" over "absolute magnitude." Additionally, the objective is shifted from weight space to activation space $(W-\widehat{W})X$, as output error ultimately determines downstream performance.

**Core Idea**: Enable PTQ to learn a LoRA-friendly quantization solution—squeezing inevitable quantization noise into a low-rank subspace that LoRA can digest.

## Method

### Overall Architecture
ProjQ reformulates "PTQ + LoRA initialization" as an activation-aware bi-level optimization: $\min_{\widehat{W}\in\mathcal{Q}}\,\min_{B,A}\,\lVert (W-\widehat{W}+BA)X\rVert_F^2$. The outer layer is responsible for "shaping the noise," while the inner layer "absorbs noise with the low-rank adapter." The algorithm consists of two stages: (i) alternating optimization using a **design rank** $r_d$ to concentrate noise into an $r_d$-dimensional subspace; (ii) closed-form covariance-aware SVD using the actual **adapter rank** $r_a$ to obtain the optimal $B_{init}, A_{init}$ initialization. Inputs are pre-trained weights $W$ and calibration activations $X$; outputs are the quantized backbone $\widehat{W}$ and the LoRA initialization pair.

### Key Designs

1. **Projection Equivalence**:

    - **Function**: Equivalently transforms "finding the optimal rank $r_d$ adapter" into "finding a rank $r_d$ orthogonal projection $P$."
    - **Mechanism**: The authors prove that $\min_{B,A}\lVert R+BAX\rVert_F^2 = \min_{P\in\mathcal{P}_{r_d}}\lVert R(I-P)\rVert_F^2$, where $R=(W-\widehat{W})X$. In other words, the optimal adapter corresponds to the $r_d$-dimensional subspace with the maximum residual energy. The optimal $P$ is given by the top-$r_d$ right singular vectors $V_{r_d}$ of $R$, such that $P=V_{r_d}V_{r_d}^\top$. This step is crucial for eliminating the "continuous low-rank matrix optimization," leaving two sets of variables—discrete weights $\widehat{W}$ and projection $P$—which can be solved alternately.
    - **Design Motivation**: The original problem is a mixed discrete-continuous optimization where $\widehat{W}$ and $B, A$ are strongly coupled and nearly intractable. Projection equivalence reformulates the objective into a clean "subspace identification" problem, making alternating minimization feasible and providing a clear geometric interpretation of "error energy allocation."

2. **Project-and-Quantize Alternating Minimization**:

    - **Function**: Finds the "repairable subspace" in the P-Step and optimizes the PTQ only for the "unrepairable" portion in the W-Step.
    - **Mechanism**: In the P-Step, $\widehat{W}^{(t)}$ is fixed, and a truncated SVD is performed on the residual $R^{(t)}=(W-\widehat{W}^{(t)})X$ to extract the top-$r_d$ right singular vectors, yielding $P^{(t+1)}$. In the W-Step, activations are projected into the orthogonal complement $X_\perp=X(I-P^{(t+1)})$, and a standard activation-aware PTQ (e.g., GPTQ) is called to solve $\min_{\widehat{W}\in\mathcal{Q}}\lVert (W-\widehat{W})X_\perp\rVert_F^2$. Since $X_\perp$ has already "removed" the directions LoRA can fix, the PTQ automatically spends its bit budget on the unrepairable tail. The authors prove that as long as the PTQ sub-solver does not increase error, the alternating sequence converges monotonically, stabilizing in 1-3 rounds in practice.
    - **Design Motivation**: This step is the soul of ProjQ—it transforms the "quantizer" from "minimizing error in the full space" to "minimizing error only in the orthogonal complement that LoRA cannot reach," effectively directing bit resources precisely where they are needed most.

3. **Covariance-aware Adapter Initialization**:

    - **Function**: Provides a mathematically optimal starting point for the LoRA training phase.
    - **Mechanism**: After obtaining $\widehat{W}^{(T)}$, let $\Delta W = W-\widehat{W}^{(T)}$. Eigen-decomposition is performed on the activation covariance $XX^\top=U_X\Lambda_X U_X^\top$ to construct a whitening matrix $Y=U_X\Lambda_X^{1/2}$. A rank $r_a$ truncated SVD $U_{r_a}\Sigma_{r_a}V_{r_a}^\top$ is performed on $\Delta W Y$, and the closed-form outputs are $B_{init}=U_{r_a}\Sigma_{r_a}^{1/2}$ and $A_{init}=\Sigma_{r_a}^{1/2}V_{r_a}^\top\Lambda_X^{-1/2}U_X^\top$. Since $\Lambda_X$ is strictly diagonal, $\Lambda_X^{-1/2}$ is a $O(n)$ element-wise inverse, avoiding $O(n^3)$ overhead.
    - **Design Motivation**: The first stage (shaping) uses the design rank $r_d$ (focused on "shape"), while the fine-tuning stage uses the adapter rank $r_a$ (focused on "capacity"). Decoupling the two allows for arbitrary configuration while providing optimal initialization, minimizing the "start-up penalty" so that LoRA aligns with the maximum error energy directions from the beginning.

### Loss & Training
ProjQ does not introduce additional training losses; it only calls the PTQ solver during the quantization stage. The outer objective is $\lVert (W-\widehat{W})X(I-P)\rVert_F^2$, and the inner PTQ reuses established solutions like GPTQ. Theoretically, the authors prove that under "sufficient adapter capacity" ($r_a\ge r_d+r_s$), ProjQ's separable upper bound $\mathcal{U}(W_p)\le \mathcal{U}(W_c)$ is strictly better than classic PTQ. They also provide a strong conclusion $f(W_p)\le f(W_c)$ for the zero task-drift case ($r_s=0$), indicating ProjQ is both an engineering trick and a theoretically sound strategy.

## Key Experimental Results

### Main Results
Extreme 2-bit quantization was performed on LLaMA-2 / Qwen2.5-Instruct / Qwen3, with calibration set at $r_a=r_d=64$, comparing against GPTQ+SVD-LLM, AWQ+SVD-LLM, CALDERA, and LoftQ. C4 Perplexity (lower is better):

| Model | GPTQ+SVD-LLM | AWQ+SVD-LLM | CALDERA | LoftQ | ProjQ |
|-------|--------------|-------------|---------|-------|-------|
| LLaMA2-7B | 26.26 | 1.7e5 | 21.59 | 28.77 | **21.50** |
| LLaMA2-13B | 14.50 | 9.5e4 | 13.56 | 14.14 | **12.48** |
| Qwen2.5-7B-Ins | 62.27 | NAN | 50.57 | 34.17 | **33.50** |
| Qwen2.5-14B-Ins | 28.94 | NAN | 23.33 | 31.74 | **22.22** |
| Qwen2.5-32B-Ins | 16.96 | — | — | 16.95 | **14.33** |
| Qwen3-32B | 26.24 | — | — | 25.71 | **20.74** |

In the extreme 2-bit range, AWQ degrades to $10^5$ level PPL, while ProjQ achieves the lowest PPL across all models; trends on WikiText are similar. The paper also reports that compensation loss is approximately $2\times$ lower than baselines.

### Ablation Study

| Configuration | Meaning | Trend |
|---------------|---------|-------|
| Full ProjQ | Alternating optimization + Covariance-aware initialization | Baseline |
| w/o Alternating Iteration | No W-Step, LoftQ-style initialization only | Performance degrades significantly |
| w/o Activation Weighting | Regresses to weight-space LoftQ ($X=I$) | Most severe collapse at low bits |
| Varying Design Rank $r_d$ | $r_d$ too small → cannot contain error; too large → orthogonal complement degrades | Moderate $r_d$ is optimal |
| Varying Iterations $T$ | Fast convergence between $T=1$ and $T=3$ | Stable within 1-3 rounds |

### Key Findings
- Enabling both "activation awareness" and "low-rank shaping" is critical—neither alone provides full benefits, suggesting ProjQ's mechanisms are synergistic rather than just additive.
- ProjQ's advantages are most pronounced in ultra-low bit scenarios (2-bit, 3-bit); at 4-bit, all methods approach FP performance, narrowing the gap. This aligns with theory: higher noise increases LoRA's "double burden," making shaping benefits more significant.
- ProjQ at 3-bit matches the language modeling performance of standard 4-bit baselines, effectively saving 25% storage and bandwidth without performance loss.

## Highlights & Insights
- The concept of "shaping noise into a form LoRA can consume" is elegant—it moves beyond the "PTQ should be as accurate as possible" mindset by acknowledging downstream LoRA presence and treating LoRA as part of the quantization objective. This co-design approach can be transferred to any "compression + adaptation" pipeline.
- The Projection Equivalence theorem, which replaces "low-rank matrix optimization" with "orthogonal projection optimization," is a valuable simplification for future work involving $\min_{B,A}\lVert R+BA\rVert$ objectives.
- The decoupling of design rank $r_d$ and adapter rank $r_a$ is highly practical—$r_d$ controls the "fidelity" of shaping while $r_a$ controls actual capacity, allowing independent tuning based on memory budgets and task difficulty.

## Limitations & Future Work
- ProjQ remains a "layer-wise independent optimization" and does not explicitly account for cross-layer error propagation; in very deep models, cross-layer coupling might cause local optima to deviate from global optima.
- The algorithm assumes the calibration set $X$ represents the downstream distribution; if calibration data shifts, the shaped subspace might be misaligned. Robustness under OOD calibration is not fully discussed.
- Current experiments focus on NLU/language modeling PPL; systematic evaluation of "softer" metrics like generation quality, long context, and instruction following is lacking.
- The theoretical section relies on the assumption that "classic PTQ noise spectrum spreads while ProjQ noise spectrum concentrates," which may not hold in extreme 1-bit quantization.

## Related Work & Insights
- **vs LoftQ**: LoftQ also seeks synergy between quantization and LoRA but minimizes $\lVert W-\widehat{W}-BA\rVert$ in weight space. ProjQ minimizes $\lVert (W-\widehat{W}+BA)X\rVert$ in activation space, weighting errors by their contribution to the output, effectively serving as an "activation-aware + subspace shaping" upgrade to LoftQ.
- **vs GPTQ / AWQ**: Classic PTQ cares only about compression and ignores subsequent fine-tuning, resulting in scattered errors. ProjQ uses GPTQ as a W-Step sub-solver but feeds it projected activations $X_\perp$, wrapping GPTQ in a "shaping shell."
- **vs QLoRA**: QLoRA is fully decoupled PTQ + LoRA, simple but wasteful of LoRA capacity. ProjQ is an "aligned" version of QLoRA—equally lightweight but with a LoRA starting point closer to the optimal solution.
- **vs CALDERA / EoRA / SVD-LLM**: CALDERA uses low-rank + quantization decomposition. EoRA / SVD-LLM propose covariance-aware low-rank approximation. ProjQ incorporates EoRA's closed-form initialization in its second stage and adds "active shaping" in the first, integrating these strengths into a complete pipeline.

## Rating
- Novelty: ⭐⭐⭐⭐ Reforming PTQ as a "shaping tool for LoRA" is a clear new perspective, supported by the solid projection equivalence theorem.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers three model families with multiple bit-widths and baselines; lacks long-context and generation quality evaluations.
- Writing Quality: ⭐⭐⭐⭐ Clear progression between geometric intuition, algorithm, theory, and experiments; rigorous proposition and theorem statements.
- Value: ⭐⭐⭐⭐ Directly addresses the demand for "edge-deployed LLMs"; 3-bit matching 4-bit is a very practical gain.

## Rating
- Novelty: To be evaluated
- Experimental Thoroughness: To be evaluated
- Writing Quality: To be evaluated
- Value: To be evaluated

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Multi-Adapter Representation Interventions via Energy Calibration](multi-adapter_representation_interventions_via_energy_calibration.md)
- [\[ICML 2026\] Breaking the MoE LLM Trilemma: Dynamic Expert Clustering with Structured Compression](breaking_the_moe_llm_trilemma_dynamic_expert_clustering_with_structured_compress.md)
- [\[ICML 2026\] Preserve-Then-Quantize: Balancing Rank Budgets for Quantization Error Reconstruction in LLMs](preserve-then-quantize_balancing_rank_budgets_for_quantization_error_reconstruct.md)
- [\[ACL 2026\] Quantize What Counts: More for Keys, Less for Values](../../ACL2026/model_compression/quantize_what_counts_more_for_keys_less_for_values.md)
- [\[ICML 2026\] Compress then Merge: From Multiple LoRAs into One Low-Rank Adapter](compress_then_merge_from_multiple_loras_into_one_low-rank_adapter.md)

</div>

<!-- RELATED:END -->
