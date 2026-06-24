---
title: >-
  [Paper Note] Forget Many, Forget Right: Scalable and Precise Concept Unlearning in Diffusion Models
description: >-
  [ICLR 2026][Image Generation][concept unlearning] ScaPre utilizes a training-free and data-free closed-form solution to simultaneously address "update conflicts" and "collateral damage to similar concepts" in large-scale concept unlearning. It stably forgets 50 concepts within 120 seconds, unlearning 5 times more concepts than the strongest baseline without degrading generation quality.
tags:
  - "ICLR 2026"
  - "Image Generation"
  - "concept unlearning"
  - "diffusion model"
  - "closed-form editing"
  - "cross-attention"
  - "mutual information"
date: 2026-05-08
content_hash: 7af9225e1c34b342
---

# Forget Many, Forget Right: Scalable and Precise Concept Unlearning in Diffusion Models

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=zt7IPzsXrT](https://openreview.net/forum?id=zt7IPzsXrT)  
**Code**: [https://github.com/kaiyuan02415/scapre](https://github.com/kaiyuan02415/scapre)  
**Area**: Image Generation / Diffusion Model Concept Unlearning  
**Keywords**: concept unlearning, diffusion model, closed-form editing, cross-attention, mutual information  

## TL;DR
ScaPre utilizes a training-free and data-free closed-form solution to simultaneously address "update conflicts" and "collateral damage to similar concepts" in large-scale concept unlearning. It stably forgets 50 concepts within 120 seconds, unlearning 5 times more concepts than the strongest baseline without degrading generation quality.

## Background & Motivation
**Background**: Text-to-image diffusion models (Stable Diffusion, DALL·E, Imagen) can synthesize realistic images but also pose risks of copyright infringement, harmful content, and misuse of sensitive information. Machine unlearning aims to precisely erase specific objects, styles, or identities from a pre-trained model while preserving other generative capabilities. While single-concept unlearning is relatively mature, multi-concept unlearning (usually 10-20) has been explored by methods like MACE (multi-LoRA), SPM (composable adapters), and UCE/RECE (closed-form editing).

**Limitations of Prior Work**: When scaling from a dozen to dozens of concepts, existing methods fail, exposing three persistent challenges: (i) **Update Conflicts**: Weight updates for different concepts interfere with each other, causing some targets to remain or harming irrelevant parameters, leading to quality collapse; (ii) **Lack of Precision Mechanism**: Unlearning "overflows" to backgrounds or visually similar non-target concepts (e.g., erasing Golden Retrievers inadvertently destroys Pugs); (iii) **Scalability Bottlenecks**: Most methods rely on extra data, sub-models, or adapters, where computational costs explode as the number of concepts increases.

**Key Challenge**: It is difficult to simultaneously achieve scale, precision, and efficiency. Closed-form methods (UCE/RECE) are fast but suffer generation collapse at scale, while fine-tuning methods (MACE) offer better precision but are slow and require extra modules.

**Goal**: A unified, lightweight framework that achieves stable unlearning, precise isolation, and efficient computation at scale (dozens of concepts) without sacrificing the quality of non-target generation.

**Core Idea**: **Model unlearning as a quadratic optimization problem with "conflict-aware regularization" and "information-theoretic decoupling," which simplifies to a single-step closed-form solution (Sylvester equation).** This approach retains the extreme efficiency of closed-form methods while suppressing large-scale conflicts and collateral damage through carefully designed regularization terms.

## Method

### Overall Architecture
ScaPre (Scalable-Precise Concept Unlearning) directly edits the Key/Value projection matrices $W$ of the cross-attention layers. The pipeline consists of three steps: first, a **Spectral Trace Regularizer** shapes a stable optimization space and suppresses conflict directions; second, an **Informax Decoupler** calculates per-channel relevance to the target concepts to restrict updates to relevant subspaces; these are combined into a quadratic objective to solve for a closed-form intermediate solution $W^\star$ (the unlearning path). Finally, **Geometry Alignment** performs proximal correction by pulling $W^\star$ back toward the pre-trained reference $W_0$ along the Bures geodesic to protect global structure. The entire process is training-free and requires no additional data.

```mermaid
flowchart LR
    A["Target Concept C_E<br/>Cross-Attention K/V"] --> B["Spectral Trace Regularizer L_t<br/>Stable Optimization Space<br/>Suppress Conflict Directions"]
    B --> C["Informax Decoupler α<br/>MI for Channel Relevance<br/>Restrict Update to Target Subspace"]
    C --> D["Solve Sylvester Equation<br/>Closed-form Intermediate W*"]
    D --> E["Geometry Alignment L_g<br/>Bures Proximal Correction<br/>Pull Back to Pre-trained W_0"]
    E --> F["Unlearning Complete<br/>Non-target Quality Preserved"]
```

### Key Designs

**1. Spectral Trace Regularizer: Shaping the optimization space using second-order statistical dynamics to "brake" conflict directions.** During large-scale simultaneous unlearning, updates for different concepts compete in shared directions, distorting the optimization landscape. ScaPre formulates the regularizer as $L_t(W)=\mathrm{tr}\big(W(\lambda I + S + R)W^\top\big)$. Here, $\lambda I$ is a standard numerical stability term. $S=\sum_k\sum_t c_{k,t}c_{k,t}^\top$ aggregates second-order statistics of all target concept context features; its large eigenvalue directions correspond to areas prone to conflict and noise in large-scale unlearning, which the regularizer suppresses. $R$ specifically modulates concept interactions within the target subspace—by applying SVD to the concept embedding matrix $C_E = U\,\mathrm{diag}(\sigma)\,V^\top$, large singular values $\sigma_i$ represent directions where concepts overlap strongly. A smooth gating function $\tilde\sigma_i=(1-\mathrm{sigmoid}(\sigma_i))\,\sigma_i$ is used to softly attenuate large singular values while keeping small ones intact, reconstructing $R=U\,\mathrm{diag}(\tilde\sigma)\,U^\top$. This suppresses high-conflict overlapping directions while preserving the low-conflict directions of independent concepts.

**2. Geometry Alignment: Matching covariance structures using Bures distance to protect the pre-trained global structure more effectively than $\ell_2$.** Existing methods often use the $\ell_2$ norm to penalize weight differences, which only pulls weights element-wise and fails to preserve high-order feature correlations. ScaPre treats each row of $W$ as a covariance factor and aligns the covariance matrix $WW^\top$ with the pre-trained reference $W_0W_0^\top$ using the Bures distance: $L_g(W)=\mathrm{tr}(WW^\top)+\mathrm{tr}(W_0W_0^\top)-2\,\mathrm{tr}\big[((WW^\top)^{1/2}W_0W_0^\top(WW^\top)^{1/2})^{1/2}\big]$. By matching covariance structures rather than element differences, it preserves high-order feature correlations and maintains stability for unrelated features during large-scale unlearning, complementing the spectral trace regularizer.

**3. Informax Decoupler: Quantifying the coupling between each channel and target concepts via mutual information to restrict updates to relevant parameters.** Contribution to target concepts varies significantly across weights—some are strongly tied to the target, while others support irrelevant backgrounds. ScaPre discretizes the activation for each channel $i$ as $z=\mathbb{1}\{a_i(s)>\tau_i\}$ and uses input labels $y\in\{0,1\}$ (1 for target concept, 0 for neutral). The empirical joint distribution is estimated to calculate mutual information: $MI_i=\sum_{z,y}p_i(z,y)\log\frac{p_i(z,y)}{p_i(z)p_i(y)}$. A higher $MI_i$ indicates that the channel is more predictive of the presence of the target concept. For multiple targets, the maximum score is taken: $MI_i=\max_k MI_i^{(k)}$. Normalized decoupling weights $\alpha_i=MI_i/\max_j MI_j\in[0,1]$ are used to decouple concept-related parameters from irrelevant ones, adaptively re-weighting updates.

**4. Unified Closed-form Solution (Sylvester Equation): Combining the three components into a quadratic objective for a single-step solution.** Let $A=\lambda I+S+R$ and $B=\mathrm{diag}(\alpha)$. The overall objective is $\min_W \mathrm{tr}(WAW^\top)+\beta L_g(W)+\mathrm{tr}(W^\top BW)-\mathrm{tr}(WV^*C_E^\top)$, where $V^*$ is the target replacement (set to zero for full unlearning). Ignoring the non-quadratic geometry alignment term, setting the derivative with respect to $W$ to zero yields the Sylvester equation $BW+WA=V^*C_E^\top$. The vectorized closed-form solution is $\mathrm{vec}(W^\star)=(I_{d_{in}}\otimes B+A^\top\otimes I_{d_{out}})^{-1}\mathrm{vec}(V^*C_E^\top)$. Since the geometry alignment term contains nested matrix square roots, it is handled via proximal correction: $W^\star W^{\star\top}$ is shifted toward $W_0W_0^\top$ along the Bures geodesic to obtain a new covariance, which is then mapped back to the weight space using orthogonal Procrustes rotation, preserving the main unlearning directions while enforcing global stability.

## Key Experimental Results

Experiments were conducted using Stable Diffusion v1.4 & v1.5 on a single RTX A6000. Large-scale unlearning used Imagenette (10 classes) and a custom ImageNet-Diversi50 (50 classes). Precise unlearning used a custom ImageNet-Confuse5 (5 groups of visually similar concepts). Explicit content removal used I2P, style unlearning used 50 artists, and generation quality was assessed via MS COCO-30K.

### Main Results (Imagenette, 10 Concepts)

| Metric | SD v1.5 | FMN | SPM | ESD | MACE | UCE | RECE | SP | **ScaPre** |
|---|---|---|---|---|---|---|---|---|---|
| Avg Acc (↓) | 89.9 | 71.9 | 47.4 | 38.7 | 78.5 | 8.5 | 4.9 | 9.6 | **0.8** |
| CLIPcoco (↑) | 31.43 | 30.62 | 30.81 | 30.14 | 31.02 | 29.45 | 29.27 | 29.25 | **30.43** |
| UQ (↑) | — | 37.35 | 49.89 | 47.84 | 35.07 | 37.23 | 32.60 | 31.78 | **64.09** |

On ImageNet-Diversi50 (50 concepts), ScaPre achieved a UQ of 65.30, significantly leading (ESD 56.35, SP 51.28), while UCE/RECE suffered generation collapse (CLIP dropped to ~22, UQ only 22-25).

### Ablation Study / Precise Unlearning (ImageNet-Confuse5)

| Metric | SD v1.5 | ESD | MACE | UCE | RECE | SP | **ScaPre** |
|---|---|---|---|---|---|---|---|
| Unlearn Acc (↓) | 83.9 | 55.6 | 76.4 | 2.9 | 3.1 | 55.0 | **5.8** |
| Preserve Acc (↑) | 86.6 | 57.7 | 78.6 | 5.6 | 5.5 | 57.1 | **76.3** |
| Overall Acc (↑) | 27.2 | 50.2 | 36.3 | 10.6 | 10.4 | 50.3 | **84.3** |
| UQ (↑) | 38.27 | 46.69 | 42.13 | 31.88 | 20.41 | 47.47 | **65.49** |

**Key Finding**: While UCE/RECE erased targets effectively (Unlearn Acc ~3), their Preserve Acc dropped to ~5, destroying similar concepts. ScaPre erased targets effectively (5.8) while preserving similar concepts (76.3). Its Overall Acc of 84.3 makes it the only method to achieve true precision.

### Key Findings
- **5x Scale**: Within acceptable generation quality, ScaPre unlearns 5 times more concepts than the strongest baseline. While other methods degrade or collapse as concept count increases, ScaPre remains stable.
- **Superior Efficiency**: 50 concepts in just 120 seconds. GPU-hours and peak VRAM are among the lowest; while UCE/RECE are also fast, their unlearning and quality performance are far inferior.
- **Style Unlearning**: Achieved a CLIPx (CLIPcoco − CLIPart) of 3.44, surpassing the second-best MACE (2.72), representing the best trade-off between unlearning and quality.

## Highlights & Insights
- **Decoupling "Anti-conflict" and "Anti-collateral damage"**: The spectral trace regularizer (with $S$ statistics and $R$ SVD gating) handles optimization stability, while the Informax decoupler handles parameter-level precision.
- **Bures distance as a replacement for $\ell_2$**: A key insight is that unlearning should preserve the covariance/correlation structure of features rather than element-wise weights, explaining why $\ell_2$ fails at scale.
- **Parameter selection via Mutual Information**: MI directly measures how much a channel's activation predicts the presence of a target concept, providing a more principled approach than heuristic masks.
- **Fully closed-form, training-free, and data-free**: Pushes efficiency to its limit, making it ideal for real-world deployment.

## Limitations & Future Work
- Only validated on Stable Diffusion v1.4/v1.5 (UNet cross-attention); transferability to newer architectures like DiT, SDXL, or Flux is unknown.
- The closed-form solution depends on inverting $(I\otimes B+A^\top\otimes I)$; the numerical and memory costs as model size increases were not deeply explored.
- The proximal correction for geometry alignment is a step-wise approximation rather than a joint optimal solution; the theoretical gap remains uncharacterized.
- Mutual information estimation depends on the discretization threshold $\tau_i$ and the split of positive/neutral samples; sensitivity analysis for these factors could be deepened.

## Related Work & Insights
- **Single-concept Unlearning**: Fine-tuning (FMN, SA, SalUn, AC), weight editing (TIME, SPEED), pruning (ConceptPrune, MS, SEMU).
- **Multi-concept Unlearning**: MACE (multi-LoRA fine-tuning up to hundreds), SPM (composable adapters + latent anchoring), Sculpting Memory (dynamic masks), ESD (negative guidance fine-tuning), UCE (unified closed-form), RECE (efficient closed-form via iterative embedding derivation). ScaPre extends the "closed-form paradigm" of UCE/RECE by using conflict-aware regularization and info-decoupling to address their collapse and precision issues at scale.
- **Insight**: Closed-form editing is not antithetical to precision or stability. The key lies in encoding domain priors (conflict directions, relevant parameter identification, global structure metrics) into the regularization structure of the quadratic objective.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The dual-term $S+R$ spectral regularizer, Bures distance for geometry alignment, and Informax decoupler are original and elegantly unified in a closed-form solution. It remains within the UCE/RECE paradigm but offers significant extensions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers large-scale (10/50 concepts), precision (Confuse5), style, explicit content, efficiency, and adversarial robustness. Includes 3 custom benchmarks and 8 baselines; however, limited to SD v1.x.
- **Writing Quality**: ⭐⭐⭐⭐ Clear mapping between challenges and components. Pipeline figures and derivations are complete.
- **Value**: ⭐⭐⭐⭐ Extends concept unlearning from "a dozen" to "dozens" while maintaining both precision and efficiency (120s for 50 concepts). Highly practical for safe generative model deployment. Code is open-sourced.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] SPEED: Scalable, Precise, and Efficient Concept Erasure for Diffusion Models](speed_scalable_precise_and_efficient_concept_erasure_for_diffusion_models.md)
- [\[ICML 2026\] Forget-It-All: Multi-Concept Machine Unlearning via Concept-Aware Neuron Masking](../../ICML2026/image_generation/forget-it-all_multi-concept_machine_unlearning_via_concept-aware_neuron_masking.md)
- [\[ICLR 2026\] Many-for-Many: Unify the Training of Multiple Video and Image Generation and Manipulation Tasks](many-for-many_unify_the_training_of_multiple_video_and_image_generation_and_mani.md)
- [\[ECCV 2024\] Challenging Forgets: Unveiling the Worst-Case Forget Sets in Machine Unlearning](../../ECCV2024/image_generation/challenging_forgets_unveiling_the_worst-case_forget_sets_in_machine_unlearning.md)
- [\[ICLR 2026\] Continual Unlearning for Text-to-Image Diffusion Models: A Regularization Perspective](continual_unlearning_for_text-to-image_diffusion_models_a_regularization_perspec.md)

</div>

<!-- RELATED:END -->
