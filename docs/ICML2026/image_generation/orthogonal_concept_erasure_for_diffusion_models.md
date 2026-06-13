---
title: >-
  [Paper Note] Orthogonal Concept Erasure for Diffusion Models
description: >-
  [ICML 2026][Image Generation][Concept Erasure] This work reformulates "additive parameter editing" based concept erasure (e.g., UCE/SPEED) in T2I diffusion models as a multiplicative "layer-wise orthogonal rotation $W^*…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "Concept Erasure"
  - "Orthogonal Transformation"
  - "Closed-form Solution"
  - "Subspace Projection"
  - "Multi-concept Erasure"
date: 2026-05-08
content_hash: 6ae1f3e1a2ef8e48
---

# Orthogonal Concept Erasure for Diffusion Models

**Conference**: ICML 2026 Oral  
**arXiv**: [2605.28902](https://arxiv.org/abs/2605.28902)  
**Code**: https://github.com/HansSunY/OCE  
**Area**: AI Safety / Concept Erasure / Diffusion Models  
**Keywords**: Concept Erasure, Orthogonal Transformation, Closed-form Solution, Subspace Projection, Multi-concept Erasure

## TL;DR
This work reformulates "additive parameter editing" based concept erasure (e.g., UCE/SPEED) in T2I diffusion models as a multiplicative "layer-wise orthogonal rotation $W^* = QW$". Combined with a subspace-level erasure objective and the Procrustes closed-form solution, the method computes $Q$ via SVD in one step. It erases 100 celebrity concepts in 4.3 seconds with near-zero damage to non-target concepts.

## Background & Motivation

**Background**: T2I diffusion models are prone to generating copyrighted, sensitive, or private content. The industry uses "concept erasure" to precisely remove specific concepts while preserving remaining generation capabilities. Existing methods fall into three categories: inference-time intervention (easy to bypass), training-based (e.g., ESD/MACE, effective but requires multi-round fine-tuning and is slow), and editing-based (e.g., UCE/RECE/SPEED, directly modifies cross-attention $W_k,W_v$ via closed-form solutions in seconds). Editing-based methods are the preferred direction for deployment.

**Limitations of Prior Work**: All editing-based methods formulate erasure as an **additive update** $W^* = W + \Delta$, solving for $\Delta$ via least squares. This formulation suffers from a trade-off between "clean erasure" and "integrity preservation"—aggressive erasure damages unrelated concepts, while strict preservation leads to incomplete erasure. Interference also occurs when erasing multiple concepts simultaneously.

**Key Challenge**: The authors identify the root cause of this conflict through a set of toy experiments. Three controlled perturbations are applied to $W$ during "cat" generation: (A) Scaling only $\tilde W = \alpha W$, which has almost no effect; (B) Independent neuron rotation $\tilde w_i = Q_i w_i$, which preserves norms but destroys relative angles, causing image quality collapse; (C) Layer-wide shared rotation $\tilde W = QW$, which preserves both norms and inter-neuron angles, but causes a clear shift in "cat" semantics. **Conclusion**: **Concept semantics are encoded in neuron directions**, while **overall generation capability is supported by the hyperspherical geometry (angles) between neurons**. Additive updates $\Delta$ simultaneously perturb direction, norm, and angle, inevitably coupling erasure and preservation.

**Goal**: To find a parameter update method that precisely rotates neuron directions (to achieve erasure) while strictly maintaining norms and mutual angles (to preserve capability), and remains inherently friendly to multi-concept erasure.

**Key Insight**: Combining direction rotation, constant norm, and constant angles mathematically results in a "layer-wise orthogonal transformation $W^* = QW$, where $Q^\top Q = I$". This corresponds to Case C in the toy experiments.

**Core Idea**: Rewrite additive $W + \Delta$ as multiplicative orthogonal $QW$, lift the erasure objective from "vector-level alignment" to "subspace-level suppression", and solve the resulting standard orthogonal Procrustes problem with a one-step SVD closed-form solution.

## Method

### Overall Architecture

Input: Pre-trained SD (or FLUX), target concept embeddings $C_1$, anchor concept embeddings $C_*$ (each target assigned a semantic neighbor as a "surrogate"), and preservation concept embeddings $C_0 = [C_g, C_n]$ ($C_g$ is a general prior pre-calculated on COCO-30k, $C_n$ is the local preservation set for the current task). Output: A layer-wise orthogonal matrix $P$ applied to cross-attention $W_k, W_v$ to obtain $W^* = PW$. The process involves only three steps: "construct matrix $M \rightarrow$ SVD $\rightarrow P = UV^\top$" without iterative training. For DiT models like FLUX without explicit cross-attention, the operation is applied to selected embedding layers as per UCE.

### Key Designs

1. **Layer-wise Orthogonal Updates instead of Additive Updates**:
    - **Function**: Unifies editing-based erasure as a multiplicative orthogonal transformation $W^* = PW$, $P^\top P = I$, directly rotating all neuron directions while keeping norms and angles intact.
    - **Mechanism**: Following the "direction encodes semantics, angles encode capability" conclusion, the vector-wise objective is defined as $\min_{P^\top P=I} \|PWC_1 - WC_*\|_F^2 + \|PWC_0 - WC_0\|_F^2$. Stacking these into $A=[WC_1, WC_0]$ and $B=[WC_*, WC_0]$, the problem becomes $\min_{P^\top P=I}\|PA-B\|_F^2$, equivalent to $\max_{P^\top P=I}\mathrm{tr}(PM)$ where $M = BA^\top = W(C_*C_1^\top + C_0C_0^\top)W^\top$. This is the classic orthogonal Procrustes problem; SVD of $M = U\Sigma V^\top$ yields $P = UV^\top$. The process requires no learning rate or iterations.
    - **Design Motivation**: Additive $\Delta$ mathematically alters $\|w_i\|$, $\cos\theta_i$, and $\cos\phi_{ij}$ simultaneously. Orthogonal $Q$ specifically modifies only directions and automatically locks the latter two, matching the toy experiment conclusions for better erasure and stability.

2. **Subspace-level Erasure Objective + Global Preservation Prior $K_0$**:
    - **Function**: Lifts erasure from "point-to-anchor alignment" to "suppressing the target subspace into the anchor's orthogonal complement" to resolve multi-concept interference; introduces a pre-calculated global preservation matrix $K_0$ from COCO-30k.
    - **Mechanism**: Gram–Schmidt orthogonalization is applied to $WC_1$ and $WC_*$ to obtain bases $G, G_*$, defining projections $R = GG^\top$ and $R_* = G_*G_*^\top$. The new objective is $\min_{P^\top P=I} -\|PR - R_{*,\perp}\|_F^2 + \|PWC_0 - WC_0\|_F^2$, leading to $M_{\text{total}} = -R(I - R_*) + W(K_0 + C_n C_n^\top)W^\top$. Here, $K_0 = C_g C_g^\top = \mathbb{E}_c[cc^\top]$ is calculated offline once (3s on A100) and reused. SVD on $M_{\text{total}}$ yields $P$.
    - **Design Motivation**: Vector-wise alignment is too rigid for 100 simultaneous targets, leading to conflicts. Subspace-level objectives require only that the target subspace be pushed away from the anchor, providing a softer, structured constraint that naturally avoids multi-concept conflicts. $K_0$ decouples general semantic priors from specific tasks, enhancing preservation without increasing inference cost.

3. **Asymmetric "Erasure Subspace + Preservation Vector" Design**:
    - **Function**: Uses subspace-level granularity for erasure and vector-level granularity for preservation.
    - **Mechanism**: The erasure term operates on subspace projections $R, R_*$, while the preservation term remains vector-wise $\|PWC_0 - WC_0\|_F^2$ (ensuring point-to-point invariance for each embedding in $C_0$).
    - **Design Motivation**: Erasure can be "coarse-in, coarse-out" since target alignment doesn't need to be exact; however, preservation must be "pinpoint accurate". Since non-target concepts have no conflicting constraints, vector-wise protection provides the highest fidelity without side effects. Ablation (Tab. 5) shows $H_o=95.48$ for this asymmetric combination, outperforming dual vector-wise ($91.70$) or dual subspace ($94.22$) approaches.

### Loss & Training
There is no "training". The pipeline is: stack $C_1, C_*, C_n \rightarrow$ compute $M_{\text{total}} = -R(I - R_*) + W(K_0 + C_n C_n^\top)W^\top \rightarrow$ SVD $M_{\text{total}} = U\Sigma V^\top \rightarrow P = UV^\top \rightarrow$ write back $W^* = PW$. $K_0$ is pre-calculated once. Erasing 100 celebrities on SD v1.4 takes only 4.3s (A100), compared to 1800s for ESD or MACE.

## Key Experimental Results

### Main Results

| Task | Metric | Prev. SOTA | OCE | Note |
|--------|------|------|----------|------|
| CIFAR-10 Object (Avg. top 5) | $\text{Acc}_e \downarrow$ / $\text{Acc}_s \uparrow$ / $H_o \uparrow$ | 8.32 / 96.92 / 94.23 (MACE) | **4.61 / 98.68 / 97.01** | Cleaner erasure, no drop in unrelated classes |
| Style Erasure (Van Gogh) | CS $\downarrow$ / COCO FID $\downarrow$ / COCO CS $\uparrow$ | 21.22 / 14.53 / 26.45 (UCE) | **21.08 / 7.15 / 26.52** | FID reduced by half |
| Multi-concept (100 Celebrities) | $\text{Acc}_e \downarrow$ / $\text{Acc}_s \uparrow$ / $H_o \uparrow$ / Time | 8.02 / 91.60 / 91.79 / 1800 s (MACE) | **3.44 / 94.42 / 95.48 / 4.3 s** | ~420× faster than MACE |
| Multi-concept (100) vs SPEED | $H_o$ / Time | 93.72 / 5.0 s | **95.48 / 4.3 s** | Higher $H_o$ at similar speed |
| I2P Implicit NSFW (w/ AT) | I2P / MMA / Ring-A-Bell ↓ | 0.10 / 0.01 / 0.00 (SPEED w/ AT) | **0.05 / 0.01 / 0.00** | Strongest among editing methods |

### Ablation Study

| Configuration | $\text{Acc}_e \downarrow$ | $\text{Acc}_s \uparrow$ | $H_o \uparrow$ | FID ↓ | Note |
|------|---------|---------|---------|---|------|
| Full OCE (Subspace E + Vector P) | 3.44 | 94.42 | **95.48** | 18.33 | Full solution |
| Vector E + Vector P | 7.59 | 91.37 | 91.70 | 20.79 | Multi-concept conflicts |
| Subspace E + Subspace P | 4.54 | 93.01 | 94.22 | 18.10 | Loose preservation, loses detail |
| Without Global Prior $K_0$ | 6.72 | 94.32 | 93.80 | 22.76 | Significant FID degradation |
| $K_0$ with 1/3 COCO | 4.47 | 93.44 | 94.47 | 19.31 | More priors are better |
| $K_0$ with 2/3 COCO | 3.85 | 93.63 | 94.87 | 18.60 | Monotonic improvement |

### Key Findings
- Asymmetric design is critical: Subspace erasure avoids multi-concept conflicts, while vector preservation ensures fine-grained fidelity. Switching granularity on either side results in performance drops.
- The "Global Semantic Prior" $K_0$ is a "free lunch": With an offline budget of 3s, multi-concept FID improved from 22.76 to 18.33. This decouples task-agnostic preservation from task-specific preservation.
- The framework transfers seamlessly to DiT architectures (e.g., FLUX.1 dev) by switching the target layer from cross-attention to MMDiT embedding layers, succeeding across objects, styles, celebrities, and NSFW categories.
- Speed advantages scale up to 400× for 100 concepts. Unlike SPEED, which requires 3 pre-processing steps, OCE is a true "one-step" method.

## Highlights & Insights
- The toy experiment identifies the root cause of additive update failures: direction, norm, and angles are coupled in additive formulas but decoupled in multiplicative orthogonal formulas. This "geometric analysis first" approach is more elegant than simply stacking tricks.
- The "additive to multiplicative" paradigm shift could apply to various parameter editing tasks (model merging, unlearning, style injection). Methods relying on closed-form solutions for $W + \Delta$ should consider if they only intend to rotate directions.
- The "coarseness for erasure, fineness for preservation" asymmetric design is counter-intuitive but effective: hard constraints for erasure cause conflicts, while soft constraints for preservation lose detail. This hybrid approach hits the sweet spot.
- $K_0$ explicitly encodes "universal image generation capability" into a matrix that is reusable and distributable, potentially serving as a "capability fingerprint" for model cards.

## Limitations & Future Work
- SVD may face computational pressure on larger models. Subspace constraints cause erased semantics to fall into a "middle ground" near the anchor rather than achieving exact alignment, which may be insufficient for certain editing tasks. Implicit concepts like relations, compositions, and watermarks remain unverified.
- Experiments focused on smaller/medium models (SD v1.4, FLUX.1 dev); verification on larger models like SDXL or PixArt is missing. Anchor concept selection lacks a systematic discussion and is ofte a performance bottleneck. Adversarial robustness relies on RECE-style adversarial editing ("Ours w/ AT") rather than being an inherent property of OCE.
- Future Work: Replacing $P$ with structured orthogonal matrices (block-diagonal, Cayley parametrization, butterfly) to reduce SVD overhead for large models; automated anchor mining via VLMs to mitigate the "middle ground" issue.

## Related Work & Insights
- **vs UCE / RECE / SPEED**: All are editing-based closed-form solutions using $W + \Delta$. OCE replaces this with multiplicative orthogonality and subspace targets, significantly outperforming them in both efficacy and efficiency (especially for multi-concept scenarios).
- **vs MACE / ESD (Training-based)**: While training-based methods rely on multi-round fine-tuning, OCE's one-step solution surpasses them while being 100× cheaper, proving that editing-based methods are not inherently inferior.
- **vs OFT / Cayley Parametrization**: OFT uses orthogonal transformations for PEFT to stabilize training or customize generation. OCE applies it as a "geometric scalpel" for targeted concept removal in a novel scenario.
- **vs CURE (NeurIPS 2025)**: CURE also explores orthogonal representation editing, but OCE operates on cross-attention weights while CURE operates at the representation layer.

## Rating
- **Novelty**: ⭐⭐⭐⭐½ The shift from additive to multiplicative is a subtle but previously unexploited change, supported by a thorough geometric motivation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐½ Covers single/multi-concept, style, NSFW, adversarial, and DiT architectures. Ablations clearly explain the asymmetric design and $K_0$.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Logical flow from geometric analysis to derivation and closed-form solution.
- **Value**: ⭐⭐⭐⭐⭐ High practical value for T2I safety pipelines—erasing 100 concepts in 4.3s with zero training.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Mass Concept Erasure in Diffusion Models with Concept Hierarchy](../../AAAI2026/image_generation/mass_concept_erasure_in_diffusion_models_with_concept_hierarchy.md)
- [\[ICLR 2026\] SPEED: Scalable, Precise, and Efficient Concept Erasure for Diffusion Models](../../ICLR2026/image_generation/speed_scalable_precise_and_efficient_concept_erasure_for_diffusion_models.md)
- [\[CVPR 2026\] Prototype-Guided Concept Erasure in Diffusion Models](../../CVPR2026/image_generation/prototype-guided_concept_erasure_in_diffusion_models.md)
- [\[CVPR 2026\] Neighbor-Aware Localized Concept Erasure in Text-to-Image Diffusion Models](../../CVPR2026/image_generation/neighbor-aware_localized_concept_erasure_in_text-to-image_diffusion_models.md)
- [\[CVPR 2026\] GrOCE: Graph-Guided Online Concept Erasure for Text-to-Image Diffusion Models](../../CVPR2026/image_generation/groce_graph-guided_online_concept_erasure_for_text-to-image_diffusion_models.md)

</div>

<!-- RELATED:END -->
