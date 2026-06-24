---
title: >-
  [Paper Note] MergOPT: A Merge-Aware Optimizer for Robust Model Merging
description: >-
  [ICLR 2026][Model Compression][Model Merging] By shifting "merging" considerations forward to the fine-tuning stage, MergOPT models "other experts to be merged" as adversarial perturbations in the weight space during training. Using distributionally robust optimization (DRO), it trains experts that are naturally more robust to merging, achieving a gain of 3.5% (up to 9.5%) in subsequent merging with almost no additional training cost.
tags:
  - "ICLR 2026"
  - "Model Compression"
  - "Model Merging"
  - "Distributionally Robust Optimization"
  - "Task Vectors"
  - "Merge-Aware Fine-Tuning"
  - "LLM"
date: 2026-05-08
content_hash: bedf584fdaf0928e
---

# MergOPT: A Merge-Aware Optimizer for Robust Model Merging

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=C21rz8mo65](https://openreview.net/forum?id=C21rz8mo65)  
**Code**: To be confirmed  
**Area**: Model Merging / Model Compression  
**Keywords**: Model Merging, Distributionally Robust Optimization, Task Vectors, Merge-Aware Fine-Tuning, LLM  

## TL;DR
By shifting "merging" considerations forward to the fine-tuning stage, MergOPT models "other experts to be merged" as adversarial perturbations in the weight space during training. Using distributionally robust optimization (DRO), it trains experts that are naturally more robust to merging, achieving a gain of 3.5% (up to 9.5%) in subsequent merging with almost no additional training cost.

## Background & Motivation
**Background**: Model Merging aims to fuse multiple independent experts fine-tuned on different tasks into a single multi-task model at the parameter level, circumventing the privacy and cost issues associated with centralized data collection. Mainstream approaches focus on the **merging stage**: Weight Averaging, Task Arithmetic (linear combinations of task vectors $\theta_k-\theta_0$), TIES/DARE (sparsity/low-rank projection to trim conflicts), and Permutation Alignment (aligning experts into the same loss basin).

**Limitations of Prior Work**: These methods assume experts are fine-tuned independently using **standard optimizers** (AdamW/SGD), remaining entirely unaware of future merging. Consequently, fine-tuning often pushes models into sharp regions near the local minima of their specific tasks that are highly sensitive to merging, leading to significant performance drops after fusion. The few works focusing on the fine-tuning stage are either expensive (tangent-space linearization is $2\sim3\times$ slower during inference) or double the training time (e.g., SAFT-Merge using SAM for flat loss landscapes).

**Key Challenge**: Merging performance depends on both the "fine-tuning" and "merging" stages, yet existing research overwhelmingly optimizes only the latter. Methods that do address fine-tuning often introduce heavy training or inference overhead.

**Goal**: Design an **efficient and effective** fine-tuning scheme that produces experts inherently "robust to merging" with an overhead comparable to standard optimizers.

**Core Idea (Merging as Weight Perturbation + DRO)**: Explicitly decompose the merging operation as a "merge-induced shift" $\zeta$ applied to the current expert parameters. Merging robustness is thus reformulated as a **Weight-space Distributionally Robust Optimization (WRO)** problem—keeping losses low under the worst-case merging shifts—actively defending against future merging impacts during fine-tuning.

## Method

### Overall Architecture
The MergOPT pipeline follows these steps: **① Rewrite merging as a shift** (merged parameters $=$ current expert $+$ a shift $\zeta$ determined by other experts, coefficients, and task counts) → **② Formulate a min–max robust objective** (minimizing task loss in the outer loop while finding the worst merging configuration in an inner feasible set $\mathcal{B}$) → **③ Approximate the inner max** using empirical priors and single-step sampling to yield a lightweight optimizer that only requires one extra shift sampling and a normal backpropagation per step.

```mermaid
flowchart TD
    A[Current Expert θ_k = θ0 + Δθ_k] --> B[Rewrite merging as shift<br/>θ_merged = θ_k + ζ]
    B --> C[ζ determined by 3 unknowns:<br/>coeff α / count K / other task vectors Δθ]
    C --> D[Inner max: Worst-case merging config]
    D -. Intractable/Expensive .-> E[Feasible Set Approximation]
    E --> E1[Δθ ~ Laplace Distribution]
    E --> E2[α from discrete prior set ~0.3]
    E --> E3[K ≤ Kmax small integer]
    F[Single-step sampling ζ = (Kα-1)z] --> G[Compute task loss at θ_k+ζ & update θ_k]
    G --> A
    E1 & E2 & E3 --> F
```

### Key Designs

**1. Rewriting Merging as "Weight-space Shift" $\zeta$**: To make merging visible during fine-tuning, the expert parameters for task $k$ are defined as $\theta_k=\theta_0+\Delta\theta_k$. When $K$ tasks are merged via Task Arithmetic, the result can be algebraically transformed: $\theta_{\text{merged}}=\theta_k+\underbrace{\big((\alpha-1)\Delta\theta_k+\alpha\sum_{j\neq k}\Delta\theta_j\big)}_{\zeta(\alpha,K,\Delta\theta)}$. This step is the pivot of the paper: it translates "merging"—a future black-box operation—into a **deterministic shift vector** $\zeta$ applied to the current parameters, enabling the fine-tuning process to anticipate merging impacts. The shift depends only on the merging coefficient $\alpha$, the number of tasks $K$, and the task vectors $\Delta\theta_j$ of other experts.

**2. Weight-space Distributionally Robust Optimization (WRO) Objective**: Conventional DRO is typically applied to the data space; here, it is applied to the **weight space**. The parameters of other experts to be merged are treated as distributional uncertainty. The objective is $\min_{\theta_k}\ \sup_{(\alpha,K,\Delta\theta)\in\mathcal{B}}\ \mathbb{E}\big[\ell_k(\phi(\theta_k,\zeta(\alpha,K, \Delta\theta)))\big]$, where $\mathcal{B}=\{(\alpha,K,\Delta\theta):\alpha\in\mathcal{A},\ K\le K_{\max},\ \Delta\theta\in\mathcal{Z}\subseteq\mathrm{span}\{\Delta\theta_1,\dots,\Delta\theta_K\}\}$. This objective ensures **performance preservation** for the current task (outer min) while forcing **robustness** against diverse merging configurations (inner sup).

**3. Prior Approximation of the Feasible Set**: The fundamental obstacle to solving the inner max is that independent developers do not have access to others' $\Delta\theta_j$, nor are $\alpha$ or $K$ known. MergOPT characterizes the feasible set using three empirical priors: (i) For **task vectors**, measurements across 3 LLM architectures and 7 tasks show that task vector elements are concentrated around 0 and can be well-fitted by a **Laplace distribution** $\mathrm{Laplace}(\mu,b)=\frac{1}{2b}\exp(-|x-\mu|/b)$; thus, $z$ is sampled from a fitted Laplace. (ii) For the **merging coefficient**, existing practices often use $\alpha \approx 0.3$, leading to a discrete candidate set $\mathcal{A}$. (iii) For the **number of tasks**, performance usually drops as $K$ increases, and $K$ rarely exceeds 10 in practice (often 2–3 for LLMs), so $K$ is bounded by a small integer $K_{\max}$.

**4. Single-step Shift Sampling**: Even with defined variables, the feasible set size explodes exponentially. Instead of multi-step projected gradient ascent, MergOPT performs a **single sampling** of $(\alpha,K,z)$ per training step and constructs the shift directly: $\zeta(\alpha,K,z)=(K\alpha-1)z$ (approximating all task vectors with the same sampled $z$). The practical objective becomes $\min_{\theta_k}\mathbb{E}_{\alpha,K,z}\big[L_{\text{task}}(\phi(\theta_k,\zeta(\alpha,K,z)))\big]$, where $\alpha\sim\mathrm{Uniform}(\mathcal{A})$, $K\sim\mathrm{Uniform}(\{1,\dots,K_{\max}\})$, and $z\sim\mathrm{Laplace}(\mu,b)$. Since $z$ follows the distribution of real task vectors, repeated sampling naturally hits shifts near the worst-case direction. This keeps training costs near standard optimization while providing significant merging robustness.

## Key Experimental Results

### Main Results
Merging **7 independent experts** (TraceBench tasks), comparing experts from "Standard Fine-tuning" vs. "MergOPT Fine-tuning":

| Base Model | Merging Method | Standard FT Avg. | MergOPT Avg. | Gain |
|---|---|---|---|---|
| Llama-3.2-1B | Weight Averaging | 0.3992 | 0.4123 | +3.28% |
| Llama-3.2-1B | Task Arithmetic | 0.4055 | 0.4165 | +2.71% |
| Llama-3.2-1B | TIES-Merging | 0.4055 | 0.4123 | +1.68% |
| Llama-3.2-1B | DARE | 0.3786 | 0.4147 | **+9.54%** |
| Llama-3.2-3B | Weight Averaging | 0.4866 | 0.4897 | +0.64% |
| Llama-3.2-3B | Task Arithmetic | 0.4871 | 0.5045 | +3.56% |
| Llama-3.2-3B | TIES-Merging | 0.4898 | 0.5098 | +4.09% |
| Llama-3.2-3B | DARE | 0.4906 | 0.5048 | +2.89% |

The average relative improvement across four strategies is approximately **3.5%**, peaking at **9.5%** (DARE on 1B). Notably, MergOPT experts maintain **single-task** performance comparable to standard fine-tuning (0.5250 vs 0.5254 on 1B), indicating that robustness gains do not sacrifice individual task accuracy.

### Ablation Study
4-task grouped merging (Llama-3.2-1B, Task Arithmetic) also shows consistent benefits:

| Setting | Group 1 Avg. | Group 2 Avg. |
|---|---|---|
| Task Arithmetic | 0.3708 | 0.4959 |
| Task Arithmetic w/ MergOPT | 0.3851 (+3.86%) | — |

The appendix further verifies **optimizer independence** (SGD instantiation, comparison with SAM), the impact of different $\alpha$ values (Tab. 13), and the validity of the Laplace fitting for individual task vectors.

### Key Findings
- **Fine-tuning is a neglected lever**: Without changing the merging algorithms, simply switching the fine-tuning optimizer yields uniform gains across four merging strategies.
- **Aggressive merging benefits most from MergOPT**: The largest gain (+9.5%) occurs on DARE, which is most prone to performance drops, showing that MergOPT specifically compensates for "merging shock."
- **Robustness is nearly free**: Single-step sampling ensures training overhead is on par with standard AdamW, outperforming SAM (double training) and tangent-space methods ($2\sim3\times$ slower inference).
- Conclusions are consistent across four LLM scales (Llama 1B/3B/8B, Qwen 1.5B) and a vision model.

## Highlights & Insights
- **Impressive Perspective Shift**: The core contribution is not a new merging operator but the insight that "merging performance is co-determined by fine-tuning and merging." The idea of "pre-training for downstream operations" can be transferred to other post-processing scenarios like pruning and quantization.
- **Algebraic Identity as an Anchor**: The simple derivation $\theta_{\text{merged}}=\theta_k+\zeta$ is the critical leap that allows a single expert's training to "see" the merging process.
- **Solving the Intractable with Empirical Priors**: Using Laplace fitting for task vectors combined with single-step sampling reduces an exponentially complex min–max problem to "one extra shift sampling per step," making it extremely lightweight in practice.
- **SAFT-Merge Objective with Better Solver**: While the shift modeling is similar to SAFT-Merge, using DRO/sampling instead of SAM provides a significant efficiency advantage.

## Limitations & Future Work
- **Coarse Approximation in Single-step Sampling**: Approximating all other task vectors with a single sampled $z$ in $(K\alpha-1)z$ is a gap from the true worst-case scenario. Small gains in certain settings (e.g., +0.64% for 3B Weight Averaging) suggest unstable approximation quality.
- **Heuristic Reliance**: The candidate set $\mathcal{A}$, $K_{\max}$, and Laplace assumptions are derived from current practices. If future merging paradigms change (e.g., merging dozens of experts), these priors may fail.
- **Task Arithmetic Specificity**: The derivation of $\zeta$ is primarily based on the Task Arithmetic family. Whether closed-form derivations hold for more complex methods like permutation alignment or low-rank subspaces is not fully discussed.
- **Lack of Direct Measurement for Hit Rate**: The paper qualitatively argues that repeated sampling likely hits worst-case directions but lacks a quantitative analysis of the gap between sampling and the true min–max solution.

## Related Work & Insights
- **Merging-stage Methods**: Weight Averaging, Task Arithmetic, TIES-Merging, DARE, Fisher Merging, AdaMerging, and Permutation Alignment (Git Re-Basin/ZipIt). MergOPT is orthogonal and can be stacked with these.
- **Fine-tuning-stage Methods**: Tangent-space linearized fine-tuning (Ortiz-Jimenez 2023, improves weight disentanglement but slow) and SAFT-Merge (SAM-based flatness seeking, slow). MergOPT is one of the few "zero-extra-cost" schemes and the first to systematically validate on LLMs/text generation.
- **Inspiration**: Modeling "future non-differentiable/black-box post-processing" as adversarial training perturbations via DRO is a generalizable paradigm—quantization-aware and pruning-aware training are fundamentally similar. MergOPT provides a clean instance of "merge-aware training."

## Rating
- **Novelty**: ⭐⭐⭐⭐ The perspective of merge-aware fine-tuning via weight-space DRO is refreshing and practical. The solver (prior approximation + single-step sampling) has independent value despite sharing formulas with SAFT-Merge.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers various LLM scales, vision models, four merging strategies, and seven tasks. Includes optimizer independence and Laplace fitting; however, some gains are small, and quantitative analysis of approximation quality is missing.
- **Writing Quality**: ⭐⭐⭐⭐ Logical flow from algebraic deformation to DRO to engineering approximation is clear. Formulas and tables are well-organized.
- **Value**: ⭐⭐⭐⭐ Near-free improvement in merging robustness orthogonal to merging algorithms. Highly relevant for decentralized expert ensembles and has potential for transfer to other post-processing tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] DisTaC: Conditioning Task Vectors via Distillation for Robust Model Merging](distac_conditioning_task_vectors_via_distillation_for_robust_model_merging.md)
- [\[ICLR 2026\] LS-Merge: Merging Language Models in Latent Space](ls-merge_merging_language_models_in_latent_space.md)
- [\[ICLR 2026\] AdaRank: Adaptive Rank Pruning for Enhanced Model Merging](adarank_adaptive_rank_pruning_for_enhanced_model_merging.md)
- [\[ICLR 2026\] Navigating the Accuracy-Size Trade-Off with Flexible Model Merging](navigating_the_accuracy-size_trade-off_with_flexible_model_merging.md)
- [\[ICML 2026\] Saliency-Aware Model Merging](../../ICML2026/model_compression/saliency-aware_model_merging.md)

</div>

<!-- RELATED:END -->
