---
title: >-
  [Paper Note] Preserve-Then-Quantize: Balancing Rank Budgets for Quantization Error Reconstruction in LLMs
description: >-
  [ICML 2026][Model Compression][PTQ] The authors propose SRR (Structured Residual Reconstruction), which explicitly splits the fixed low-rank budget $r$ in QER (Quantization Error Reconstruction) into two parts: "preserve…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "PTQ"
  - "Quantization Error Reconstruction"
  - "Low-rank Compensation"
  - "QPEFT"
  - "Rank Budget Allocation"
date: 2026-05-08
content_hash: ccb1116093266b6c
---

# Preserve-Then-Quantize: Balancing Rank Budgets for Quantization Error Reconstruction in LLMs

**Conference**: ICML 2026  
**arXiv**: [2602.02001](https://arxiv.org/abs/2602.02001)  
**Code**: https://ai-isl.github.io/srr (project page)  
**Area**: Model Compression / Low-bit Quantization for LLMs  
**Keywords**: PTQ, Quantization Error Reconstruction, Low-rank Compensation, QPEFT, Rank Budget Allocation

## TL;DR
The authors propose SRR (Structured Residual Reconstruction), which explicitly splits the fixed low-rank budget $r$ in QER (Quantization Error Reconstruction) into two parts: "preserve the top $k$ principal singular directions before quantization" and "use the remaining $r-k$ ranks to fit the residual." They provide a closed-form criterion requiring only a single random probe to select $k^\star$ per layer, consistently outperforming LQER/QERA in 2/3-bit PTQ and QPEFT.

## Background & Motivation

**Background**: Low-bit PTQ for LLMs suffers significant accuracy drops when compressing weights to 3/2 bits. The mainstream remedy is QER: approximate weights as $\mathbf{W}\approx \mathbf{Q}+\mathbf{L}\mathbf{R}$, where $\mathbf{Q}=\mathcal{Q}(\mathbf{W})$ is the quantized result and $\mathbf{L}\mathbf{R}$ is a rank $\le r$ correction term to recover quantization errors. Methods like ZeroQuant-V2, LQER, and QERA follow this approach, often combined with a diagonal scaling matrix $\mathbf{S}$ computed from calibration activations, performing truncated SVD in the scaled space $\mathbf{S}\mathbf{W}$.

**Limitations of Prior Work**: All existing methods allocate the entire rank budget $r$ to fitting the residual $\mathbf{S}(\mathbf{W}-\mathbf{Q})$. However, in low-bit regimes, quantization errors are typically dense and high-rank, while the truly "low-rank" component is $\mathbf{S}\mathbf{W}$ itself—transformer weights in the activation scaling space are highly anisotropic, with energy concentrated in a few principal singular directions. As a result, quantization first destroys these high-energy directions, leaving a noisy, high-rank residual for the limited rank budget to fit, leading to diminishing returns.

**Key Challenge**: The "purpose" of the rank budget is implicitly locked to "residual compensation," but in reality, a rank budget can be used in two more efficient ways—either to preserve the principal subspace before quantization (structural preservation) or to repair the residual after quantization (error reconstruction). Which is more effective depends on the spectral shape of each layer.

**Goal**: Given a fixed rank budget $r$, address two subproblems—(i) Is there a unified framework for "preserving $k$-dimensional principal structure first, then quantizing, then using $r-k$ to repair the residual"? (ii) How can one select the optimal $k$ per layer/matrix without brute-force enumeration?

**Key Insight**: The authors treat the singular spectrum of $\mathbf{S}\mathbf{W}$ as a "prior signal"—the faster the spectral decay and the more concentrated the energy, the more worthwhile it is to allocate rank to preservation; the flatter the decay, the more worthwhile to allocate rank to the residual. With the additional assumption that quantization noise is approximately isotropic, the choice of $k$ can be formulated as minimizing $\rho_k(\mathbf{S}\mathbf{W})\cdot\rho_{r-k}(\mathbf{S}\mathbf{E})$, where $\rho_p(\mathbf{A})$ is the energy ratio remaining after rank-$p$ truncation, and $\mathbf{E}$ can be approximated by a single $\mathcal{U}[-1,1]$ random matrix.

**Core Idea**: Reformulate QER as a three-step "preserve-quantize-reconstruct" process ($\mathbf{W}\approx \mathbf{L}^{(1)}\mathbf{R}^{(1)} + \mathbf{Q} + \mathbf{L}^{(2)}\mathbf{R}^{(2)}$), and use a one-shot random probe to determine the optimal rank split $k^\star$.

## Method

### Overall Architecture
SRR is a plug-and-play, post-PTQ method requiring no fine-tuning. For each linear layer weight $\mathbf{W}\in\mathbb{R}^{m\times n}$ and its activation scaling matrix $\mathbf{S}$, given quantizer $\mathcal{Q}$ and total rank budget $r$, SRR proceeds in four steps: (1) Draw a random matrix $\mathbf{E}_{ij}\sim\mathcal{U}[-1,1]$, select the rank split as $k^\star=\arg\min_k \rho_k(\mathbf{S}\mathbf{W})\rho_{r-k}(\mathbf{S}\mathbf{E})$; (2) Take the top-$k^\star$ singular components of $\mathbf{S}\mathbf{W}$ and map back to the original space, yielding $\mathbf{L}^{(1)}\mathbf{R}^{(1)}=\mathbf{S}^{-1}\mathrm{SVD}_{k^\star}(\mathbf{S}\mathbf{W})$; (3) Quantize only the remaining components $\mathbf{Q}=\mathcal{Q}(\mathbf{W}-\mathbf{L}^{(1)}\mathbf{R}^{(1)})$; (4) Use the remaining $r-k^\star$ ranks in the scaled space to fit the induced quantization error $\mathbf{E}_k=\mathbf{W}-\mathbf{L}^{(1)}\mathbf{R}^{(1)}-\mathbf{Q}$, yielding $\mathbf{L}^{(2)}\mathbf{R}^{(2)}=\mathbf{S}^{-1}\mathrm{SVD}_{r-k^\star}(\mathbf{S}\mathbf{E}_k)$. The two low-rank blocks are concatenated as $\mathbf{L},\mathbf{R}$, and inference remains in the form $\widehat{\mathbf{W}}=\mathbf{Q}+\mathbf{L}\mathbf{R}$, fully compatible with existing QER inference kernels.

### Key Designs

1. **Differentiable "Preserve-Quantize-Reconstruct" Parameterization**:

    - **Function**: Upgrades fixed-split QER to a unified framework with tunable split point $k\in\{0,\dots,r\}$; $k=0$ recovers traditional QER (e.g., ZeroQuant-V2/LQER/QERA), $k=r$ recovers LQ-LoRA/SVDQuant-style "preserve structure first" schemes, and intermediate values correspond to a new, previously unexplored regime.
    - **Mechanism**: Minimize scaled-space reconstruction error $\min_{0\le k\le r}\|\mathbf{S}(\mathbf{W}-(\Delta_1+\mathcal{Q}(\mathbf{W}-\Delta_1)+\Delta_2))\|_F$, where $\Delta_1$ is the rank-$k$ preserved term and $\Delta_2$ is the rank-$(r-k)$ residual correction. By the Eckart-Young theorem, the optimal $\Delta_1,\Delta_2$ reduce to truncated SVDs of the respective matrices, leaving $k$ as the sole scalar degree of freedom.
    - **Design Motivation**: The authors observe that different projection matrices (Query/Output/MLP up/down) within the same layer and model have widely varying optimal $k$ for minimal reconstruction error (e.g., in LLaMA-2 7B layer 10, the optimal $k$ for Q projection differs greatly from Output projection), indicating that rank allocation must be layer/matrix-specific.

2. **Closed-form $k$ Selection Criterion Based on "Quantization Noise Approximate Constant Ratio"**:

    - **Function**: Avoids the need to run quantization and SVD on $\mathbf{S}\mathbf{E}_k$ for every candidate $k$ (which would require $O(r)$ expensive computations), reducing the cost of selecting $k$ to "compute the singular spectrum of $\mathbf{S}\mathbf{W}$ once + sample a random $\mathbf{E}$ and compute its spectrum."
    - **Mechanism**: Under two assumptions, expand $\mathcal{L}(k)^2=\|\mathbf{S}\mathbf{E}_k\|_F^2\cdot\rho_{r-k}(\mathbf{S}\mathbf{E}_k)$. Assumption 1 (quantization error energy ratio is approximately constant $\eta_\mathcal{Q}$) gives $\|\mathbf{S}\mathbf{E}_k\|_F^2\approx \eta_\mathcal{Q}^2\rho_k(\mathbf{S}\mathbf{W})\|\mathbf{S}\mathbf{W}\|_F^2$; Assumption 2 (the normalized spectrum of the quantization residual is approximately independent of $k$) allows replacing $\mathbf{E}_k$ with a random $\mathbf{E}\sim\mathcal{U}[-1,1]$. Together, this yields $k^\star=\arg\min_k \rho_k(\mathbf{S}\mathbf{W})\rho_{r-k}(\mathbf{S}\mathbf{E})$, requiring only the spectrum of $\mathbf{S}\mathbf{W}$ and a single random probe.
    - **Design Motivation**: The proxy's $k$ selection closely matches the true reconstruction error curve (see Figure 2 in the paper), and different random probes typically yield $k^\star$ differing by only $\pm 1$, with spectral concentration ensuring one-shot stability. Randomized SVD is supported, and the computational overhead is only $1.06\times$ that of original QER (on LLaMA-2 7B).

3. **Two-stage QPEFT Initialization + Gradient Decay Decoupling**:

    - **Function**: Extends SRR automatically to Quantized PEFT scenarios, providing LoRA-style adapters with initialization that is both close to the original weights and stable for training.
    - **Mechanism**: Use $\mathbf{Q}$ as the frozen backbone, and initialize the trainable adapter as $\mathbf{L}\mathbf{R}=\mathbf{L}^{(1)}\mathbf{R}^{(1)}+\mathbf{L}^{(2)}\mathbf{R}^{(2)}$; since the singular values of the preserved component $\mathbf{L}^{(1)}\mathbf{R}^{(1)}$ are much larger than those of the residual component $\mathbf{L}^{(2)}\mathbf{R}^{(2)}$ (using the same learning rate would cause the former to be over-updated or the latter under-trained), the authors introduce gradient decay for the preserved component: $\nabla_{\mathbf{L}^{(1)}\mathbf{R}^{(1)}}\mathcal{L}\leftarrow \gamma\nabla_{\mathbf{L}^{(1)}\mathbf{R}^{(1)}}\mathcal{L}$ ($\gamma\in(0,1)$, typically $0.1$ or $0.5$), leaving the residual component's gradient unchanged.
    - **Design Motivation**: The preserved directions correspond to the "backbone semantics" of the original weights and should remain stable; the residual directions are the true "task adaptation capacity." Decoupling allows each to fulfill its role without interference, and QPEFT achieves a $5.9$ pp average improvement on GLUE over the baseline at 2-bit.

### Loss & Training
No training is required during the PTQ stage; everything is handled by SVD + quantization + random probe. The QPEFT stage uses standard downstream task losses (e.g., cross-entropy/Pearson for GLUE), with the only modification being the gradient scaling $\gamma$ above. All SVDs use randomized SVD (Halko et al.), requiring only the top-$r$ singular values.

## Key Experimental Results

### Main Results
Systematic comparison across 6 models (TinyLlama 1.1B, Gemma-2 2B, LLaMA-2 7B/13B, LLaMA-3.1 8B/70B), two rank budgets ($r=32, 64$), and 3 QER baselines (LQER, QERA-approx, QERA-exact). Representative data points (WikiText2 PPL ↓, 3-bit MXINT quantization):

| Model | Rank | QER Baseline (QERA-exact) | + SRR | Gain |
|-------|------|---------------------------|-------|------|
| TinyLlama 1.1B | $r=64$ | $19.59$ | $18.71$ | $-0.88$ |
| Gemma-2 2B | $r=64$ | $19.36$ | $18.30$ | $-1.06$ |
| LLaMA-2 7B | $r=64$ | $10.68$ | $10.59$ | $-0.09$ |
| LLaMA-3.1 8B | $r=64$ | $11.00$ | $10.78$ | $-0.22$ |
| LLaMA-3.1 70B | $r=32$ | $6.68$ | $6.63$ | $-0.05$ |

SRR consistently reduces perplexity across all (model, rank, baseline) combinations; improvements are most pronounced for small models (TinyLlama, Gemma-2) and in the 3-bit regime. Zero-shot five-task average accuracy ($r=64$, 3-bit):

| Model | BF16 | w-only | QERA-exact | + SRR |
|-------|------|--------|------------|-------|
| Gemma-2 2B | $59.26$ | $45.12$ | $52.15$ | $54.38$ |
| LLaMA-2 7B | $58.90$ | $52.50$ | $55.28$ | $56.56$ |
| LLaMA-3.1 8B | $67.34$ | $51.17$ | $59.05$ | $60.79$ |

### Ablation Study

| Configuration | Key Metric | Description |
|---------------|-----------|-------------|
| QER ($k=0$) | baseline PPL | All rank for residual, mainstream approach |
| LQ-LoRA-style ($k=r$) | Slightly worse than $k=0$ | All rank for structure, no residual repair |
| SRR with single random probe | $\le\pm 1$ from optimal $k^\star$ | One-shot spectral proxy is stable enough |
| QPEFT w/o gradient decay $\gamma$ | Unstable training, washed out | Backbone directions over-updated |
| QPEFT $\gamma\in\{0.1, 0.5\}$ | Both outperform baseline | Insensitive to $\gamma$, main benefit from better initialization |

### Key Findings
- The crucial factor is not "how much to preserve," but "knowing how much to preserve per layer"—the optimal $k^\star$ varies greatly across projection matrices within the same model, and layer-wise adaptation is key to SRR's effectiveness.
- The reason a one-shot random probe works is that transformer layer dimensions are large enough and the singular spectrum of random matrices is highly concentrated (Appendix B shows repeated sampling yields $k^\star$ differing by only $\pm 1$).
- The $5.9$ pp GLUE gain in QPEFT mainly comes from SRR initialization's fidelity to backbone structure, while gradient decay acts as a "stabilizer" against drift and is not sensitive to the value of $\gamma$.

## Highlights & Insights
- Explicitly parameterizing a resource (rank budget) that previously seemed to have only one use, and directly selecting the solution via spectral ratios in closed form. This "surfacing hidden design choices" approach is highly transferable: for example, the rank in LoRA, the number of experts in MoE, and the retention ratio in KV cache can all leverage the same "spectral proxy + closed-form selection" template.
- The ability of a one-shot random matrix to serve as a proxy for the quantization noise spectrum fundamentally leverages the spectral concentration properties of large random matrices (Marchenko-Pastur law). LLM layer dimensions (thousands) fall squarely in this regime, so a single sample suffices.
- The inference form $\widehat{\mathbf{W}}=\mathbf{Q}+\mathbf{L}\mathbf{R}$ is fully compatible with existing QER kernels, making engineering adoption nearly cost-free—any deployment already using LQER/QERA can simply swap the initialization script.

## Limitations & Future Work
- Assumption 1 (quantization error energy ratio is approximately constant) and Assumption 2 (residual spectrum is independent of $k$) may not hold in the 1-bit extreme; the paper evaluates down to 2-bit. For more aggressive binarization, SRR's closed-form criterion may need to be revised.
- The scaling matrix $\mathbf{S}$ still depends on calibration data, and SRR's optimal $k^\star$ changes with $\mathbf{S}$; ensuring SRR's stability under calibration distribution shift is a direction for future work.
- The gradient decay in QPEFT is a simple stop-gradient style trick; replacing it with second-order/preconditioned optimizers may eliminate the need for tuning $\gamma$.
- All experiments focus on weight quantization, not activation or KV quantization; whether the rank-splitting idea can be extended to these scenarios remains unknown.

## Related Work & Insights
- **vs LQER / QERA-approx / QERA-exact**: All correspond to $k=0$, allocating all rank to the residual. SRR shows within a unified framework that $k=0$ is suboptimal for highly anisotropic layers in low-bit regimes, and provides a zero-training upgrade.
- **vs LQ-LoRA / SVDQuant**: Both correspond to the $k=r$ extreme, preserving structure before quantization. SRR reveals this is also suboptimal for layers where the residual remains high-rank, and allows for intermediate values.
- **vs LoftQ / QLoRA**: QPEFT baselines typically use iterative QER to initialize adapters; SRR achieves $+5.9$ pp on 2-bit GLUE with a one-shot closed-form initialization, indicating "initialization quality > iteration count."

## Rating
- Novelty: ⭐⭐⭐⭐ Explicitly allocating the implicit rank budget is a simple but previously unexplored perspective, and the closed-form criterion is natural.
- Experimental Thoroughness: ⭐⭐⭐⭐ 6 models × 2 ranks × 3 QER baselines + QPEFT downstream validation on GLUE, broad coverage.
- Writing Quality: ⭐⭐⭐⭐ Clear chain from motivation→formulation→assumption→algorithm, with Figure 2 providing intuitive alignment between proxy and ground truth.
- Value: ⭐⭐⭐⭐ Fully compatible with existing QER inference kernels, almost zero extra training, engineering-friendly; also provides a reusable template for future "how to split rank budget" questions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Quantization Error Propagation: Revisiting Layer-Wise Post-Training Quantization](../../NeurIPS2025/model_compression/quantization_error_propagation_revisiting_layer-wise_post-training_quantization.md)
- [\[ICML 2026\] ScaLoRA: Optimally Scaled Low-Rank Adaptation for Efficient High-Rank Fine-Tuning](scalora_optimally_scaled_low-rank_adaptation_for_efficient_high-rank_fine-tuning.md)
- [\[AAAI 2026\] Put the Space of LoRA Initialization to the Extreme to Preserve Pre-trained Knowledge](../../AAAI2026/model_compression/put_the_space_of_lora_initialization_to_the_extreme_to_preserve_pre-trained_know.md)
- [\[NeurIPS 2025\] FiRA: Can We Achieve Full-Rank Training of LLMs Under Low-Rank Constraint?](../../NeurIPS2025/model_compression/fira_can_we_achieve_full-rank_training_of_llms_under_low-rank_constraint.md)
- [\[ACL 2026\] IMPACT: Importance-Aware Activation Space Reconstruction](../../ACL2026/model_compression/impact_importance-aware_activation_space_reconstruction.md)

</div>

<!-- RELATED:END -->
