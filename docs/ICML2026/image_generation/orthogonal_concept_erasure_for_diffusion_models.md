---
title: >-
  [Paper Note] Orthogonal Concept Erasure for Diffusion Models
description: >-
  [ICML 2026][Image Generation][Concept Erasure] The authors reformulate concept erasure in T2I diffusion models from "additive parameter editing" (e.g., UCE/SPEED) to a multiplicative "layer-wise orthogonal rotation $W^* = QW$." Combined with a subspace-level erasure target and solved via a closed-form Procrustes solution, the method erases 100 celebrity concepts in
tags:
  - ICML 2026
  - Image Generation
  - Concept Erasure
date: 2026-05-08
content_hash: b3dfc5ae166e2f02
---
# Orthogonal Concept Erasure for Diffusion Models

**Conference**: ICML 2026 Oral  
**arXiv**: [2605.28902](https://arxiv.org/abs/2605.28902)  
**Code**: https://github.com/HansSunY/OCE  
**Area**: AI Security / Concept Erasure / Diffusion Models  
**Keywords**: Concept Erasure, Orthogonal Transformation, Closed-form Solution, Subspace Projection, Multi-concept Erasure

## TL;DR
The authors reformulate concept erasure in T2I diffusion models from "additive parameter editing" (e.g., UCE/SPEED) to a multiplicative "layer-wise orthogonal rotation $W^* = QW$." Combined with a subspace-level erasure target and solved via a closed-form Procrustes solution, the method erases 100 celebrity concepts in 4.3 seconds with nearly zero damage to non-target concepts.

## Background & Motivation

**Background**: T2I diffusion models are prone to generating copyrighted, sensitive, or private content. The industry uses "concept erasure" to precisely remove specific concepts while maintaining overall generation capabilities. Existing methods fall into three categories: inference-time intervention (easy to bypass), training-based (e.g., ESD/MACE, effective but requires multi-round fine-tuning and is slow), and editing-based (e.g., UCE/RECE/SPEED, uses closed-form solutions to modify cross-attention $W_k, W_v$ in seconds). Editing-based methods are the preferred direction for deployment.

**Limitations of Prior Work**: All editing-based methods formulate erasure as an **additive update** $W^* = W + \Delta$, solving for $\Delta$ via least squares. This formulation inherently struggles with the trade-off between "clean erasure" and "integrity preservation"—aggressive erasure damages unrelated concepts, while strict preservation leads to incomplete erasure. Furthermore, multi-concept erasure often leads to conflicts.

**Key Challenge**: The authors identify the root cause of these contradictions through toy experiments. Applying three controlled perturbations to $W$ for "cat" generation revealed: (A) scaling $\tilde W = \alpha W$ has almost no effect; (B) independent neuron rotation $\tilde w_i = Q_i w_i$ preserves magnitude but destroys relative angles, collapsing image quality; (C) layer-shared rotation $\tilde W = QW$ preserves both magnitudes and inter-neuron angles, causing a clear semantic shift for "cat." The conclusion: **Concept semantics are encoded in neuron directions**, while **overall generation capability is supported by the hyperspherical geometry (angles) between neurons**. Additive $\Delta$ disturbs direction, magnitude, and angles simultaneously, inevitably coupling erasure and preservation.

**Goal**: Find a parameter update method that precisely rotates neuron directions (for erasure) while strictly preserving magnitudes and inter-neuron angles (for capability), while being naturally friendly to multi-concept erasure.

**Key Insight**: Combining direction rotation, invariant magnitude, and invariant angles mathematically defines a "layer-wise orthogonal transformation $W^* = QW$, $Q^\top Q = I$." This corresponds to Case C in the toy experiments.

**Core Idea**: Replace the additive $W + \Delta$ with a multiplicative orthogonal $QW$. Elevate the erasure target from "vector-level alignment" to "subspace-level suppression," resulting in a standard orthogonal Procrustes problem with a one-step SVD closed-form solution.

## Method

### Overall Architecture

OCE addresses the traditional trade-off where editing-based concept erasure is either incomplete or damaging. It replaces the additive update $W^*=W+\Delta$ common in existing methods with a multiplicative orthogonal update $W^*=PW$. The erasure target is elevated from vector-level alignment to subspace-level suppression, solved via SVD for the orthogonal Procrustes problem. Inputs include a pre-trained SD (or FLUX), target concept embeddings $C_1$, anchor concept embeddings $C_*$ (acting as "surrogates" for target concepts), and preservation concept embeddings $C_0=[C_g, C_n]$ ($C_g$ is a general prior computed on COCO-30k, $C_n$ is the local preservation set). The output is a layer-wise orthogonal matrix $P$ applied to $W_k, W_v$ in cross-attention. For DiT models like FLUX without explicit cross-attention, the operation is applied to selected embedding layers.

### Key Designs

**1. Layer-wise Orthogonal Update vs. Additive Update: Decoupling Direction from Geometry**

The limitation lies in the additive formula. The authors' toy experiments show that neuron **directions** encode semantics, while the **angular geometry** between neurons supports generation capability. Additive $\Delta$ modifies magnitude $\|w_i\|$, direction $\cos\theta_i$, and angles $\cos\phi_{ij}$ simultaneously. OCE uses an orthogonal matrix $P$ ($P^\top P=I$) for multiplicative updates, which mathematically rotates directions while locking magnitudes and angles.

To solve for $P$: starting from a vector-wise target $\min_{P^\top P=I}\|PWC_1-WC_*\|_F^2+\|PWC_0-WC_0\|_F^2$, where the first term rotates target concepts toward anchors and the second pins preservation concepts. By stacking into $A=[WC_1, WC_0]$ and $B=[WC_*, WC_0]$, the problem simplifies to $\min_{P^\top P=I}\|PA-B\|_F^2$, equivalent to $\max_{P^\top P=I}\mathrm{tr}(PM)$ where $M=BA^\top=W(C_*C_1^\top+C_0C_0^\top)W^\top$. SVD of $M=U\Sigma V^\top$ yields $P=UV^\top$ directly without iterations or learning rates.

**2. Subspace-level Erasure + Global Preservation Prior $K_0$: Conflict-Free Multi-Concept Erasure**

Vector-wise alignment is precise for single concepts but creates conflicting constraints when aligning 100 targets to unique anchors. OCE moves erasure to "suppressing the target subspace into the orthogonal complement of the anchor": using Gram–Schmidt to obtain orthogonal bases $G, G_*$ for $WC_1$ and $WC_*$, the target becomes $\min_{P^\top P=I}-\|PR-R_{*,\perp}\|_F^2+\|PWC_0-WC_0\|_F^2$, where $R, R_*$ are projections. This structural constraint is softer and avoids multi-concept conflicts.

The preservation term includes a global prior $K_0=C_gC_g^\top=\mathbb{E}_c[cc^\top]$ (computed offline on COCO-30k in ~3s). The final maximization form is $M_{\text{total}}=-R(I-R_*)+W(K_0+C_nC_n^\top)W^\top$. This decouples general semantic priors from task-specific ones, reducing multi-concept FID from 22.76 to 18.33 without increasing inference costs.

**3. Asymmetric Granularity: Subspace for Erasure, Vector for Preservation**

The third design is the intentional mix of granularities. Erasure uses subspace projections (coarse-grained), while preservation remains vector-wise (fine-grained). The logic is that erasure can be "flexible" since targets don't need exact alignment with a specific surrogate. Conversely, preservation must be "precise" for each individual embedding to maintain fidelity. Ablation studies (Tab. 5) confirm this: the asymmetric combination ($H_o=95.48$) outperforms dual vector-wise ($91.70$) and dual subspace ($94.22$) approaches.

### Loss & Training

There is no "training" in the traditional sense. The process follows: stack $C_1, C_*, C_n$ $\rightarrow$ compute $M_{\text{total}}=-R(I-R_*)+W(K_0+C_nC_n^\top)W^\top$ $\rightarrow$ solve SVD $M_{\text{total}}=U\Sigma V^\top$ $\rightarrow$ $P=UV^\top$ $\rightarrow$ update $W^*=PW$. $K_0$ is precomputed once. Erasing 100 celebrities on SD v1.4 takes 4.3s (A100), compared to 1800s for ESD or MACE.

## Key Experimental Results

### Main Results

| Task | Metric | Prev. SOTA | OCE | Note |
|--------|------|------|----------|------|
| CIFAR-10 Single Object (Avg 5) | $\text{Acc}_e \downarrow$ / $\text{Acc}_s \uparrow$ / $H_o \uparrow$ | 8.32 / 96.92 / 94.23 (MACE) | **4.61 / 98.68 / 97.01** | Cleaner erasure, minimal drop in unrelated classes |
| Artistic Style (Van Gogh) | CS $\downarrow$ / COCO FID $\downarrow$ / COCO CS $\uparrow$ | 21.22 / 14.53 / 26.45 (UCE) | **21.08 / 7.15 / 26.52** | FID reduced by half |
| Multi-concept (100 Celebrities) | $\text{Acc}_e \downarrow$ / $\text{Acc}_s \uparrow$ / $H_o \uparrow$ / Time | 8.02 / 91.60 / 91.79 / 1800 s (MACE) | **3.44 / 94.42 / 95.48 / 4.3 s** | ~420× faster than MACE |
| Multi-concept vs. SPEED | $H_o$ / Time | 93.72 / 5.0 s | **95.48 / 4.3 s** | Higher $H_o$ at similar speed |
| I2P Implicit NSFW (w/ AT) | I2P / MMA / Ring-A-Bell ↓ | 0.10 / 0.01 / 0.00 (SPEED w/ AT) | **0.05 / 0.01 / 0.00** | Strongest among editing methods |

### Ablation Study

| Configuration | $\text{Acc}_e \downarrow$ | $\text{Acc}_s \uparrow$ | $H_o \uparrow$ | FID ↓ | Note |
|------|---------|---------|---------|---|------|
| Full OCE (Subspace E + Vector P) | 3.44 | 94.42 | **95.48** | 18.33 | Complete proposal |
| Vector E + Vector P | 7.59 | 91.37 | 91.70 | 20.79 | Multi-concept conflicts |
| Subspace E + Subspace P | 4.54 | 93.01 | 94.22 | 18.10 | Preservation too loose, lost detail |
| Without Global Prior $K_0$ | 6.72 | 94.32 | 93.80 | 22.76 | Significant FID degradation |
| $K_0$ with 1/3 COCO | 4.47 | 93.44 | 94.47 | 19.31 | More general prior is better |
| $K_0$ with 2/3 COCO | 3.85 | 93.63 | 94.87 | 18.60 | Monotone improvement |

### Key Findings
- Asymmetric design is critical: subspace erasure prevents multi-concept conflict, while vector preservation ensures fine-grained fidelity. Changing either granularity degrades performance.
- The global semantic prior $K_0$ is a "free lunch": a 3s offline budget reduces multi-concept FID from 22.76 to 18.33 by decoupling general generation capability from the specific task.
- Seamless migration to DiT architectures (FLUX.1 dev) by applying the formula to embedding layers, covering objects, styles, celebrities, and NSFW categories.
- Speed advantage scales to 400× for 100 concepts compared to training-based methods. Unlike SPEED, OCE requires no multi-step preprocessing; it is a true "one-step" solution.

## Highlights & Insights
- The toy experiments pinpoint exactly why additive updates fail: direction, magnitude, and angles are coupled in the additive formula but decoupled in the multiplicative orthogonal one. This geometric analysis is more elegant than empirical tricks.
- The "additive to multiplicative" paradigm shift could apply to various parameter editing tasks (model merging, unlearning, style injection). Methods relying on closed-form $W + \Delta$ should consider if they only intend to rotate directions.
- Asymmetric granularity is counter-intuitive but logical: erasure constraints are too brittle if hard, while preservation constraints are too loose if soft. The mix hits the "sweet spot."
- $K_0$ explicitly represents "general image generation capability" as a matrix. It is reusable, distributable, and could serve as a "capability fingerprint" in future model cards.

## Limitations & Future Work
- SVD faces computational challenges on much larger models. Subspace constraints cause erased semantics to land in a "middle ground" near the anchor rather than aligning perfectly, which might be insufficient for sensitive editing. Implicit concepts like relationships or watermarks are yet to be validated.
- Experiments focused on SD v1.4 and FLUX.1 dev; validation on larger models like SDXL or PixArt is missing. Anchor concept selection lacks a systematic discussion and is often a practical bottleneck. Adversarial robustness relies on RECE-style adversarial training (Ours w/ AT) rather than being an inherent property of OCE.
- Future Work: Replacing $P$ with structured orthogonality (block diagonal, Cayley parametrization) could mitigate SVD costs for large models. Automated anchor mining via VLMs could alleviate the "middle ground" issue.

## Related Work & Insights
- **vs. UCE / RECE / SPEED**: All are editing-based closed-form solutions but use additive $W + \Delta$. OCE replaces this with multiplicative orthogonality and subspace targets, outperforming them in both efficiency and effectiveness, especially for multi-concept scenarios.
- **vs. MACE / ESD (Training-based)**: Training methods rely on fine-tuning. OCE surpasses them with a one-step closed-form solution that is 100× cheaper, proving editing methods can match training-based results.
- **vs. OFT / Cayley Parametrization**: OFT uses orthogonal transforms for PEFT to stabilize training. OCE uses them as a "geometric scalpel" for targeted concept removal.
- **vs. CURE (NeurIPS 2025)**: Similar in name (Orthogonal Representation Editing), but OCE operates on cross-attention weights while CURE operates on the representation layer.

## Rating
- Novelty: ⭐⭐⭐⭐½ Sophisticated systematic shift from additive to multiplicative with deep geometric motivation.
- Experimental Thoroughness: ⭐⭐⭐⭐½ Covers single/multi-concept, styles, NSFW, adversarial, and DiT architectures.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear logical chain from geometric analysis to closed-form solution.
- Value: ⭐⭐⭐⭐⭐ High practical value for production T2I safety pipelines (4.3s for 100 concepts, zero training).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Mass Concept Erasure in Diffusion Models with Concept Hierarchy](../../AAAI2026/image_generation/mass_concept_erasure_in_diffusion_models_with_concept_hierarchy.md)
- [\[ICLR 2026\] SPEED: Scalable, Precise, and Efficient Concept Erasure for Diffusion Models](../../ICLR2026/image_generation/speed_scalable_precise_and_efficient_concept_erasure_for_diffusion_models.md)
- [\[CVPR 2026\] Prototype-Guided Concept Erasure in Diffusion Models](../../CVPR2026/image_generation/prototype-guided_concept_erasure_in_diffusion_models.md)
- [\[CVPR 2026\] Closed-Form Concept Erasure via Double Projections](../../CVPR2026/image_generation/closed-form_concept_erasure_via_double_projections.md)
- [\[CVPR 2025\] Precise, Fast, and Low-cost Concept Erasure in Value Space: Orthogonal Complement Matters](../../CVPR2025/image_generation/precise_fast_and_low-cost_concept_erasure_in_value_space_orthogonal_complement_m.md)

</div>

<!-- RELATED:END -->
