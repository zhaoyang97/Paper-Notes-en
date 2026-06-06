---
title: >-
  [Paper Note] Preserve-Then-Quantize: Balancing Rank Budgets for Quantization Error Reconstruction in LLMs
description: >-
  [ICML 2026][Model Compression][PTQ] The authors propose SRR (Structured Residual Reconstruction), which explicitly splits the fixed low-rank budget $r$ used in QER (Quantization Error Reconstruction) for compensating qua…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "PTQ"
  - "Quantization Error Reconstruction"
  - "Low-rank Compensation"
  - "QPEFT"
  - "Rank Budget Allocation"
date: 2026-05-08
content_hash: b6f6ee8a852133ba
---

# Preserve-Then-Quantize: Balancing Rank Budgets for Quantization Error Reconstruction in LLMs

**Conference**: ICML 2026  
**arXiv**: [2602.02001](https://arxiv.org/abs/2602.02001)  
**Code**: https://ai-isl.github.io/srr (Project Page)  
**Area**: Model Compression / LLM Low-bit Quantization  
**Keywords**: PTQ, Quantization Error Reconstruction, Low-rank Compensation, QPEFT, Rank Budget Allocation

## TL;DR
The authors propose SRR (Structured Residual Reconstruction), which explicitly splits the fixed low-rank budget $r$ used in QER (Quantization Error Reconstruction) for compensating quantization residuals into two parts: "preserving $k$ principal singular directions before quantization" and "using $r-k$ ranks to fit the residual." A closed-form criterion requiring only a one-shot random probe is provided to select $k^\star$ layer-by-layer. SRR consistently outperforms LQER/QERA in 2/3-bit PTQ and QPEFT.

## Background & Motivation

**Background**: Low-bit PTQ for LLMs suffers significant precision loss when compressing weights to 3/2-bit. The mainstream remedy is QER: approximating the weight as $\mathbf{W}\approx \mathbf{Q}+\mathbf{L}\mathbf{R}$, where $\mathbf{Q}=\mathcal{Q}(\mathbf{W})$ is the direct quantization result and $\mathbf{L}\mathbf{R}$ is a correction term with rank $\le r$ used to restore quantization errors. Representative methods like ZeroQuant-V2, LQER, and QERA perform truncated SVD in the scaled space $\mathbf{S}\mathbf{W}$ using a diagonal scaling matrix $\mathbf{S}$ calculated from calibration activations.

**Limitations of Prior Work**: All existing methods allocate the entire rank budget $r$ to fitting the residual $\mathbf{S}(\mathbf{W}-\mathbf{Q})$. However, in low-bit regimes, quantization errors are usually dense and high-rank, whereas $\mathbf{S}\mathbf{W}$ itself is actually "low-rank"—transformer weights are highly anisotropic in the activation-scaled space, with energy concentrated in a few principal singular directions. Consequently, quantization first destroys these high-energy directions, leaving a noisy, high-rank residual for the limited rank budget to fix, which is inefficient.

**Key Challenge**: The "use case" of the rank budget is locked into "residual compensation" by default. In reality, a rank budget can be used in two more cost-effective ways: either preserving the principal subspace before quantization (structure preservation) or repairing the residual after quantization (error reconstruction). Which one is more effective depends on the specific spectral shape of the layer.

**Goal**: Given a fixed rank budget $r$, answer two sub-questions: (i) whether a unified framework exists for "preserving $k$-dimensional principal structure, then quantizing, then repairing residuals with $r-k$ dimensions"; (ii) how to select the optimal $k$ for each layer and matrix without brute-force enumeration.

**Key Insight**: The authors treat the singular spectrum shape of $\mathbf{S}\mathbf{W}$ as a "prior signal"—the faster the spectral decay and the more concentrated the energy, the more worthwhile it is to spend rank on preservation; the flatter the decay, the more rank should be reserved for residuals. Combined with the assumption that "quantization noise is approximately isotropic," the selection of $k$ can be formulated as a minimization problem of $\rho_k(\mathbf{S}\mathbf{W})\cdot\rho_{r-k}(\mathbf{S}\mathbf{E})$, where $\rho_p(\mathbf{A})$ is the energy ratio remaining after rank-$p$ truncation of $\mathbf{A}$, and $\mathbf{E}$ can be proxied by a simple $\mathcal{U}[-1,1]$ random matrix.

**Core Idea**: Rewrite QER as a three-step "Preserve-Quantize-Reconstruct" process ($\mathbf{W}\approx \mathbf{L}^{(1)}\mathbf{R}^{(1)} + \mathbf{Q} + \mathbf{L}^{(2)}\mathbf{R}^{(2)}$) and solve for the optimal rank split $k^\star$ using a one-shot random probe.

## Method

### Overall Architecture
SRR is a plug-and-play, training-free PTQ post-processing method. For each linear layer weight $\mathbf{W}\in\mathbb{R}^{m\times n}$ and its activation scaling matrix $\mathbf{S}$, given a quantizer $\mathcal{Q}$ and total rank budget $r$, SRR follows four steps: (1) Sample a random matrix $\mathbf{E}_{ij}\sim\mathcal{U}[-1,1]$ and select the rank split $k^\star=\arg\min_k \rho_k(\mathbf{S}\mathbf{W})\rho_{r-k}(\mathbf{S}\mathbf{E})$; (2) Extract the top-$k^\star$ singular components of $\mathbf{S}\mathbf{W}$ and map them back to the original space to get $\mathbf{L}^{(1)}\mathbf{R}^{(1)}=\mathbf{S}^{-1}\mathrm{SVD}_{k^\star}(\mathbf{S}\mathbf{W})$; (3) Quantize only the remaining components $\mathbf{Q}=\mathcal{Q}(\mathbf{W}-\mathbf{L}^{(1)}\mathbf{R}^{(1)})$; (4) Use the remaining $r-k^\star$ ranks to fit the induced quantization error $\mathbf{E}_k=\mathbf{W}-\mathbf{L}^{(1)}\mathbf{R}^{(1)}-\mathbf{Q}$ in the scaled space to get $\mathbf{L}^{(2)}\mathbf{R}^{(2)}=\mathbf{S}^{-1}\mathrm{SVD}_{r-k^\star}(\mathbf{S}\mathbf{E}_k)$. Finally, concatenate the two low-rank blocks into $\mathbf{L},\mathbf{R}$. The inference form remains $\widehat{\mathbf{W}}=\mathbf{Q}+\mathbf{L}\mathbf{R}$, which is fully compatible with existing QER inference kernels.

### Key Designs

1.  **Differentiable "Preserve-Quantize-Reconstruct" Parameterization**:
    *   **Function**: Upgrades the fixed QER to a unified framework with an adjustable split point $k\in\{0,\dots,r\}$, where $k=0$ reduces to traditional QER (e.g., ZeroQuant-V2/LQER/QERA), and $k=r$ reduces to "structure-preserving" schemes like LQ-LoRA/SVDQuant. Intermediate values cover a new region previously unexplored in literature.
    *   **Mechanism**: Minimizes the reconstruction error in the scaled space $\min_{0\le k\le r}\|\mathbf{S}(\mathbf{W}-(\Delta_1+\mathcal{Q}(\mathbf{W}-\Delta_1)+\Delta_2))\|_F$, where $\Delta_1$ is the rank-$k$ preservation term and $\Delta_2$ is the rank-$(r-k)$ residual correction. By the Eckart-Young theorem, optimal $\Delta_1$ and $\Delta_2$ reduce to the truncated SVD of the respective matrices, leaving $k$ as the only scalar degree of freedom.
    *   **Design Motivation**: The authors observed that different projection matrices (Query/Output/MLP up/down) in the same layer of the same model reach minimum reconstruction error at vastly different $k$ (e.g., the optimal $k$ for Q projection and Output projection in LLaMA-2 7B layer 10 are far apart), implying that rank allocation must be performed at the layer/matrix level.

2.  **Closed-form $k$ Selection Criterion Based on "Constant Quantization Noise Ratio"**:
    *   **Function**: Avoids running quantization and SVD of $\mathbf{S}\mathbf{E}_k$ for every candidate $k$ ($O(r)$ expensive computations), compressing the cost of selecting $k$ to "calculating the singular spectrum of $\mathbf{S}\mathbf{W}$ once + calculating the singular spectrum of one random $\mathbf{E}$."
    *   **Mechanism**: Expand $\mathcal{L}(k)^2=\|\mathbf{S}\mathbf{E}_k\|_F^2\cdot\rho_{r-k}(\mathbf{S}\mathbf{E}_k)$ under two assumptions. Assumption 1 (Quantization error energy ratio is approximately constant $\eta_\mathcal{Q}$) allows $\|\mathbf{S}\mathbf{E}_k\|_F^2\approx \eta_\mathcal{Q}^2\rho_k(\mathbf{S}\mathbf{W})\|\mathbf{S}\mathbf{W}\|_F^2$. Assumption 2 (Normalized spectrum of quantization residuals is approximately independent of $k$) allows replacing $\mathbf{E}_k$ with a random $\mathbf{E}\sim\mathcal{U}[-1,1]$. Combining these yields $k^\star=\arg\min_k \rho_k(\mathbf{S}\mathbf{W})\rho_{r-k}(\mathbf{S}\mathbf{E})$, using only the spectrum of $\mathbf{S}\mathbf{W}$ and a single random probe.
    *   **Design Motivation**: The $k$ selection results from the proxy are highly consistent with the true reconstruction error curves (Figure 2 in the paper), and $k^\star$ selected by different random probes usually fluctuates by only $\pm 1$. Spectral concentration ensures one-shot stability. It also supports randomized SVD, adding only $1.06\times$ computation compared to original QER (on LLaMA-2 7B).

3.  **Two-stage QPEFT Initialization + Decoupled Gradient Decay**:
    *   **Function**: Automatically extends SRR to Quantized PEFT scenarios, providing LoRA-style adapters with an initialization that is both close to the original weights and stable for training.
    *   **Mechanism**: Uses $\mathbf{Q}$ as the frozen backbone and initializes the trainable adapter as $\mathbf{L}\mathbf{R}=\mathbf{L}^{(1)}\mathbf{R}^{(1)}+\mathbf{L}^{(2)}\mathbf{R}^{(2)}$. Since singular values of the preservation component $\mathbf{L}^{(1)}\mathbf{R}^{(1)}$ are much larger than the residual component $\mathbf{L}^{(2)}\mathbf{R}^{(2)}$ (leading to over-updating the former or under-learning the latter with the same learning rate), the authors introduce gradient decay $\nabla_{\mathbf{L}^{(1)}\mathbf{R}^{(1)}}\mathcal{L}\leftarrow \gamma\nabla_{\mathbf{L}^{(1)}\mathbf{R}^{(1)}}\mathcal{L}$ ($\gamma\in(0,1)$, typically $0.1$ or $0.5$), while keeping the residual component gradient unchanged.
    *   **Design Motivation**: Preservation directions correspond to the "backbone semantics" of original weights and should remain stable; residual directions represent the true "task adaptation capacity." After decoupling, the two types of directions perform their respective roles without interference. QPEFT achieves an average GLUE improvement of $5.9$ pp over baselines at 2-bit.

### Loss & Training
The PTQ stage requires no training and is completed entirely by SVD, quantization, and random probes. The QPEFT stage follows standard downstream task losses (e.g., cross-entropy/Pearson for GLUE), with the only change being the gradient scaling $\gamma$ mentioned above. All SVDs use randomized SVD (Halko et al.), requiring only top-$r$ singular values.

## Key Experimental Results

### Main Results
Systematic comparison across 6 models (TinyLlama 1.1B, Gemma-2 2B, LLaMA-2 7B/13B, LLaMA-3.1 8B/70B), two rank budgets ($r=32, 64$), and 3 QER baselines (LQER, QERA-approx, QERA-exact). Representative data points (WikiText2 PPL ↓, 3-bit MXINT quantization):

| Model | Rank | QER Baseline (QERA-exact) | + SRR | Gain |
|------|----|-----------------------|-------|------|
| TinyLlama 1.1B | $r=64$ | $19.59$ | $18.71$ | $-0.88$ |
| Gemma-2 2B | $r=64$ | $19.36$ | $18.30$ | $-1.06$ |
| LLaMA-2 7B | $r=64$ | $10.68$ | $10.59$ | $-0.09$ |
| LLaMA-3.1 8B | $r=64$ | $11.00$ | $10.78$ | $-0.22$ |
| LLaMA-3.1 70B | $r=32$ | $6.68$ | $6.63$ | $-0.05$ |

SRR consistently reduces perplexity across all (model, rank, baseline) combinations; improvements are most significant for small models (TinyLlama, Gemma-2) and in 3-bit regimes. Zero-shot 5-task average accuracy ($r=64$, 3-bit):

| Model | BF16 | w-only | QERA-exact | + SRR |
|------|------|--------|-----------|-------|
| Gemma-2 2B | $59.26$ | $45.12$ | $52.15$ | $54.38$ |
| LLaMA-2 7B | $58.90$ | $52.50$ | $55.28$ | $56.56$ |
| LLaMA-3.1 8B | $67.34$ | $51.17$ | $59.05$ | $60.79$ |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| QER ($k=0$) | baseline PPL | All rank for residuals, the mainstream approach |
| LQ-LoRA-style ($k=r$) | Slightly worse than $k=0$ | All rank for structure, no residual repair |
| SRR with one-shot random probe | Diff from optimal $k^\star$ $\le \pm 1$ | One-shot spectral proxy is sufficiently stable |
| QPEFT w/o gradient decay $\gamma$ | Training unstable, washed out | Backbone directions are over-updated |
| QPEFT $\gamma\in\{0.1, 0.5\}$ | Both better than baseline | Insensitive to $\gamma$, verifying gain from better initialization |

### Key Findings
- The most critical factor is not "how much to preserve," but "knowing how much each layer should preserve"—optimal $k^\star$ varies greatly across projection matrices within the same model. Layer-wise adaptation is the core of SRR's success.
- One-shot random probes work because transformer layer dimensions are large enough that random matrix singular spectra are highly concentrated (Appendix B shows $k^\star$ selected by repeated sampling varies by only $\pm 1$).
- The $5.9$ pp GLUE gain in QPEFT stems primarily from the fidelity of SRR initialization to the backbone structure; gradient decay acts as an "anti-drift" stabilizer and is insensitive to the $\gamma$ value.

## Highlights & Insights
- Explicitly parameterizing a resource (rank budget) that seemed to have a "sole use" and selecting a closed-form solution via spectral ratios. This "bringing hidden design choices to the surface" pattern is highly transferable: e.g., LoRA rank, number of MoE experts, or KV cache retention ratios could all leverage similar "spectral proxy + closed-form selection" templates.
- The use of a one-shot random matrix as a proxy for quantization noise spectra essentially benefits from the spectral concentration of large-dimensional random matrices (Marchenko-Pastur family). LLM layer dimensions (thousands) fall exactly within the concentration zone, making one sample stable.
- The inference form $\widehat{\mathbf{W}}=\mathbf{Q}+\mathbf{L}\mathbf{R}$ is zero-cost for deployment, as it is fully compatible with existing QER kernels. Any deployment already using LQER/QERA only needs to change the initialization script.

## Limitations & Future Work
- Assumption 1 (constant energy ratio) and Assumption 2 (residual spectrum independent of $k$) might fail in extreme 1-bit regimes; the paper evaluates down to 2-bit. In more aggressive binarization, SRR's closed-form criterion might need re-derivation.
- The scaling matrix $\mathbf{S}$ still depends on calibration data, and the optimal $k^\star$ of SRR changes with $\mathbf{S}$. Maintaining SRR stability under calibration distribution drift is a potential future direction.
- Gradient decay in QPEFT is a simple stop-gradient style trick; replacing it with second-order or preconditioned optimizers might eliminate the need for hyperparameter $\gamma$ selection.
- Experiments only cover weight quantization, leaving activation and KV quantization untouched; whether the rank-splitting idea extends to these scenarios remains unknown.

## Related Work & Insights
- **vs LQER / QERA-approx / QERA-exact**: All three correspond to $k=0$, spending all rank on residuals. SRR proves $k=0$ is sub-optimal for highly anisotropic layers at low bits in a unified framework and provides a zero-training upgrade.
- **vs LQ-LoRA / SVDQuant**: These correspond to the $k=r$ extreme, preserving structure first. SRR reveals this extreme is also sub-optimal for layers where residuals remain high-rank and allows for intermediate values.
- **vs LoftQ / QLoRA**: QPEFT baselines usually use iterative QER to initialize adapters. SRR achieves $+5.9$ pp on 2-bit GLUE with a one-shot closed-form initialization, demonstrating that "initialization quality > iteration count."

## Rating
- Novelty: ⭐⭐⭐⭐ Explicitly parameterizing hidden rank allocation is simple but previously unaddressed; the closed-form criterion is natural.
- Experimental Thoroughness: ⭐⭐⭐⭐ 6 models × 2 ranks × 3 QER baselines + QPEFT downstream validation on GLUE, broad coverage.
- Writing Quality: ⭐⭐⭐⭐ Clear chain from motivation → formulation → assumption → algorithm. Figure 2 provides intuitive alignment between proxy and ground truth.
- Value: ⭐⭐⭐⭐ Zero-modification compatibility with existing QER inference kernels and almost zero extra training; friendly for engineering deployment while providing a reusable template for "how to split rank budgets."

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Compress then Merge: From Multiple LoRAs into One Low-Rank Adapter](compress_then_merge_from_multiple_loras_into_one_low-rank_adapter.md)
- [\[CVPR 2026\] Quant Experts: Token-aware Adaptive Error Reconstruction with Mixture of Experts for Large Vision-Language Models Quantization](../../CVPR2026/model_compression/quant_experts_token_aware_vlm_quantization.md)
- [\[ICML 2026\] ProjQ: Project-and-Quantize for Adapter-Aware LLM Compression](projq_project-and-quantize_for_adapter-aware_llm_compression.md)
- [\[ICML 2026\] GEMQ: Global Expert-Level Mixed-Precision Quantization for MoE LLMs](gemq_global_expert-level_mixed-precision_quantization_for_moe_llms.md)
- [\[ICML 2026\] NeUQI: Near-Optimal Uniform Quantization Parameter Initialization for Low-Bit LLMs](neuqi_near-optimal_uniform_quantization_parameter_initialization_for_low-bit_llm.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[ICML 2026\] Compress then Merge: From Multiple LoRAs into One Low-Rank Adapter](compress_then_merge_from_multiple_loras_into_one_low-rank_adapter.md)
- [\[CVPR 2026\] Quant Experts: Token-aware Adaptive Error Reconstruction with Mixture of Experts for Large Vision-Language Models Quantization](../../CVPR2026/model_compression/quant_experts_token_aware_vlm_quantization.md)
- [\[ICML 2026\] ProjQ: Project-and-Quantize for Adapter-Aware LLM Compression](projq_project-and-quantize_for_adapter-aware_llm_compression.md)
- [\[ICML 2026\] GEMQ: Global Expert-Level Mixed-Precision Quantization for MoE LLMs](gemq_global_expert-level_mixed-precision_quantization_for_moe_llms.md)
- [\[ICML 2026\] NeUQI: Near-Optimal Uniform Quantization Parameter Initialization for Low-Bit LLMs](neuqi_near-optimal_uniform_quantization_parameter_initialization_for_low-bit_llm.md)

</div>

<!-- RELATED:END -->
