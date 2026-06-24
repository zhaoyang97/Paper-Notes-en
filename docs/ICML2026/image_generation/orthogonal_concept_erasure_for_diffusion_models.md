---
title: >-
  [Paper Note] Orthogonal Concept Erasure for Diffusion Models
description: >-
  [ICML 2026 Oral][Image Generation][Concept Erasure] This paper reformulates "additive parameter editing" concept erasure (e.g., UCE/SPEED) in T2I diffusion models as a multiplicative "layered orthogonal rotation $W^* = QW$". By combining a subspace-level erasure target with a Procrustes closed-form solution, it calculates $Q$ in a single step—erasing 100 celebrity concepts in 4.3 seconds with near-zero damage to non-target concepts.
tags:
  - "ICML 2026 Oral"
  - "Image Generation"
  - "Concept Erasure"
  - "Orthogonal Transformation"
  - "Closed-form Solution"
  - "Subspace Projection"
  - "Multi-concept Erasure"
date: 2026-05-08
content_hash: 80654d3eca595ee3
---

# Orthogonal Concept Erasure for Diffusion Models

**Conference**: ICML 2026 Oral  
**arXiv**: [2605.28902](https://arxiv.org/abs/2605.28902)  
**Code**: https://github.com/HansSunY/OCE  
**Area**: AI Safety / Concept Erasure / Diffusion Models  
**Keywords**: Concept Erasure, Orthogonal Transformation, Closed-form Solution, Subspace Projection, Multi-concept Erasure

## TL;DR
This paper reformulates "additive parameter editing" concept erasure (e.g., UCE/SPEED) in T2I diffusion models as a multiplicative "layered orthogonal rotation $W^* = QW$". By combining a subspace-level erasure target with a Procrustes closed-form solution, it calculates $Q$ in a single step—erasing 100 celebrity concepts in 4.3 seconds with near-zero damage to non-target concepts.

## Background & Motivation

**Background**: T2I diffusion models risk generating copyrighted, sensitive, or private content. The industry uses "concept erasure" to precisely remove specific concepts while maintaining general generation capabilities. Existing methods fall into three categories: inference-time intervention (easily bypassed), training-based (e.g., ESD/MACE, effective but slow due to multi-round fine-tuning), and editing-based (e.g., UCE/RECE/SPEED, directly modifying $W_k, W_v$ of cross-attention via closed-form solutions for second-level results). Editing-based methods are the preferred direction for deployment.

**Limitations of Prior Work**: All editing-based methods formulate erasure as an **additive update** $W^* = W + \Delta$, solving for $\Delta$ via least squares. This formulation inherently struggles with the trade-off between "thorough erasure" and "integrity preservation"—aggressive erasure damages unrelated concepts, while strict preservation leads to incomplete erasure. Furthermore, multi-concept erasure often leads to internal conflicts.

**Key Challenge**: The authors use a set of toy experiments to identify the root cause of the conflict. By applying three types of controlled perturbations to $W$ to test "cat" generation: (A) pure scaling $\tilde W = \alpha W$ has almost no effect; (B) independent neuron rotation $\tilde w_i = Q_i w_i$ preserves magnitude but breaks relative angles, causing image quality to collapse; (C) shared layer-wise rotation $\tilde W = QW$ keeps both magnitude and inter-neuron angles intact, but the "cat" semantics shift significantly. Conclusion: **Concept semantics are encoded in neuron directions**, while **overall generation capability is supported by the angular hyperspherical geometry between neurons**. Additive $\Delta$ disturbs direction, magnitude, and angles simultaneously, inevitably coupling erasure and preservation.

**Goal**: Find a parameter update method that precisely rotates neuron directions (to achieve erasure) while strictly maintaining magnitudes and relative angles (to preserve capability), while being naturally friendly to multi-concept erasure.

**Key Insight**: Combining direction rotation, magnitude preservation, and angle preservation is mathematically equivalent to a "layered orthogonal transformation $W^* = QW$ where $Q^\top Q = I$". This corresponds to Case C in the toy experiment.

**Core Idea**: Rewrite the additive $W + \Delta$ as a multiplicative orthogonal $QW$ and elevate the erasure target from "vector-level alignment" to "subspace-level suppression". The problem is then reduced to a standard orthogonal Procrustes problem, solved via a one-step SVD closed-form solution.

## Method

### Overall Architecture

OCE addresses the persistent issue of editing-based erasure being either incomplete or damaging. It transforms the common additive update $W^*=W+\Delta$ into a multiplicative orthogonal update $W^*=PW$. The erasure target is elevated from point-wise alignment to subspace-level suppression, resulting in a standard orthogonal Procrustes problem solved in one SVD step. Inputs include a pre-trained SD (or FLUX), target concept embeddings $C_1$, anchor embeddings $C_*$ (a semantic neighbor for each target), and preservation embeddings $C_0=[C_g,C_n]$ (where $C_g$ is a general prior pre-calculated on COCO-30k and $C_n$ is a local preservation set). The output is a layered orthogonal matrix $P$, acting on $W_k,W_v$ to obtain $W^*=PW$. The process involves no iterative training, only matrix construction and SVD. For DiT models like FLUX without explicit cross-attention, the operation is applied to chosen embedding layers.

### Key Designs

**1. Layer-wise Orthogonal Update replacing Additive Update: Moving only direction, not magnitude or angles**

The limitation lies in the additive formula itself. After decoupling the contradictions via toy experiments, the authors found that neuron **directions** encode semantics, while the **angular geometry** between neurons supports generation capability. Additive $\Delta$ mathematically changes magnitude $\|w_i\|$, direction $\cos\theta_i$, and inter-neuron angles $\cos\phi_{ij}$ simultaneously, coupling erasure and preservation. OCE's counter-strategy is a multiplicative update with an orthogonal matrix $P$ ($P^\top P=I$). Orthogonal transformations rotate directions while locking magnitudes and angles, matching the "move semantics without hurting capability" perturbation from the toy experiment.

To solve for $P$: First, define a vector-wise objective $\min_{P^\top P=I}\|PWC_1-WC_*\|_F^2+\|PWC_0-WC_0\|_F^2$, where the first term rotates target concepts toward anchors and the second pins preservation concepts. By stacking into $A=[WC_1,WC_0]$ and $B=[WC_*,WC_0]$, the problem simplifies to $\min_{P^\top P=I}\|PA-B\|_F^2$, equivalent to $\max_{P^\top P=I}\mathrm{tr}(PM)$ where $M=BA^\top=W(C_*C_1^\top+C_0C_0^\top)W^\top$. This is a classic orthogonal Procrustes problem; performing SVD on $M=U\Sigma V^\top$ yields $P=UV^\top$ without learning rates or iterations.

**2. Subspace-level Erasure + Global Preservation Prior $K_0$: Resolving Multi-concept Conflicts**

Vector-wise alignment is precise for single concepts, but 100 targets trying to align strictly to their respective anchors create conflicting constraints, resulting in poor erasure and collateral damage. OCE elevates the target to "compressing the target subspace into the orthogonal complement of the anchor": performing Gram–Schmidt on $WC_1$ and $WC_*$ gives orthogonal bases $G,G_*$, defining projections $R=GG^\top$ and $R_*=G_*G_*^\top$. The objective becomes $\min_{P^\top P=I}-\|PR-R_{*,\perp}\|_F^2+\|PWC_0-WC_0\|_F^2$. This "subspace suppression" is softer and more structured than point-wise alignment, preventing conflicts when stacking multiple concepts.

The preservation term incorporates a reusable global prior. The preservation matrix is split into $K_0=C_gC_g^\top=\mathbb{E}_c[cc^\top]$ (calculated offline on COCO-30k in ~3s on an A100) and a task-specific local term $C_nC_n^\top$. The final trace maximization form is $M_{\text{total}}=-R(I-R_*)+W(K_0+C_nC_n^\top)W^\top$, followed by one SVD. $K_0$ decouples the "general semantic prior" from specific tasks, reducing multi-concept FID from 22.76 to 18.33 at zero inference cost.

**3. Asymmetric Granularity: Subspace for Erasure, Vector for Preservation**

The third design choice is the intentional mix of granularities: erasure uses subspace projections $R,R_*$ (coarse), while preservation remains vector-level $\|PWC_0-WC_0\|_F^2$ (fine, point-wise invariance for each $C_0$ embedding). The rationale is that the two sides have opposite requirements—erasure can be "loose" because target concepts don't need point-perfect alignment with anchors (strict constraints cause conflict), but preservation must be "pinpoint" because non-target concepts have no conflicting constraints, and vector-wise protection yields the highest fidelity. Ablations (Tab. 5) confirm this: dual vector-wise $H_o=91.70$, dual subspace $H_o=94.22$, while the asymmetric combination reaches $H_o=95.48$.

### Loss & Training

There is no "training." The pipeline from input to output involves: stacking $C_1,C_*,C_n$ → calculating $M_{\text{total}}=-R(I-R_*)+W(K_0+C_nC_n^\top)W^\top$ → SVD $M_{\text{total}}=U\Sigma V^\top$ → $P=UV^\top$ → updating $W^*=PW$. $K_0$ is pre-calculated once. Erasing 100 celebrities on SD v1.4 takes only 4.3s (A100), compared to 1800s for ESD or MACE.

## Key Experimental Results

### Main Results

| Task | Metric | Prev. SOTA | OCE | Note |
|--------|------|------|----------|------|
| CIFAR-10 Single Object Erasure (Avg. of top 5) | $\text{Acc}_e \downarrow$ / $\text{Acc}_s \uparrow$ / $H_o \uparrow$ | 8.32 / 96.92 / 94.23 (MACE) | **4.61 / 98.68 / 97.01** | Cleaner erasure, zero drop in unrelated classes |
| Artistic Style Erasure (Van Gogh) | CS $\downarrow$ / COCO FID $\downarrow$ / COCO CS $\uparrow$ | 21.22 / 14.53 / 26.45 (UCE) | **21.08 / 7.15 / 26.52** | FID halved |
| Multi-concept (100 Celebrities) | $\text{Acc}_e \downarrow$ / $\text{Acc}_s \uparrow$ / $H_o \uparrow$ / Time | 8.02 / 91.60 / 91.79 / 1800 s (MACE) | **3.44 / 94.42 / 95.48 / 4.3 s** | ~420× faster than MACE |
| Multi-concept vs SPEED | $H_o$ / Time | 93.72 / 5.0 s | **95.48 / 4.3 s** | Same speed tier, higher $H_o$ |
| I2P Implicit NSFW (with AT version) | I2P / MMA / Ring-A-Bell ↓ | 0.10 / 0.01 / 0.00 (SPEED w/ AT) | **0.05 / 0.01 / 0.00** | Strongest among editing methods |

### Ablation Study

| Configuration | $\text{Acc}_e \downarrow$ | $\text{Acc}_s \uparrow$ | $H_o \uparrow$ | FID ↓ | Note |
|------|---------|---------|---------|---|------|
| Full OCE (Subspace Erasure + Vector Pres.) | 3.44 | 94.42 | **95.48** | 18.33 | Full solution |
| Vector Erasure + Vector Pres. | 7.59 | 91.37 | 91.70 | 20.79 | Multi-concept conflicts |
| Subspace Erasure + Subspace Pres. | 4.54 | 93.01 | 94.22 | 18.10 | Loose preservation, detail loss |
| W/o Global Prior $K_0$ | 6.72 | 94.32 | 93.80 | 22.76 | Significant FID degradation |
| $K_0$ with 1/3 COCO | 4.47 | 93.44 | 94.47 | 19.31 | More prior is better |
| $K_0$ with 2/3 COCO | 3.85 | 93.63 | 94.87 | 18.60 | Monotonic improvement |

### Key Findings
- **Asymmetric design is crucial**: Subspace erasure avoids multi-concept conflict, while vector preservation ensures fine-grained fidelity. Changing either granularity degrades performance.
- **$K_0$ is a "free lunch"**: A 3s offline budget reduces multi-concept FID from 22.76 to 18.33 by decoupling general and task-specific generation abilities.
- **Transferable to DiT**: On FLUX.1 dev, the formula applies directly by switching the target from cross-attention to MMDiT embedding layers.
- **Speed advantage**: At a 100-concept scale, OCE is 400× faster. SPEED's reported runtime omits necessary preprocessing; OCE is truly a "one-step" method.

## Highlights & Insights
- The toy experiment identifies the root cause: direction, magnitude, and angles are coupled in additive formulas but decoupled in multiplicative orthogonal ones. This "geometric analysis first" approach is more elegant than empirical tricks.
- The shift from "additive to multiplicative" can potentially be applied to other parameter editing tasks (model merging, unlearning), where one may only need to rotate directions.
- Mixed granularity ("coarse erasure, fine preservation") is counter-intuitive but effective, hitting the sweet spot between conflict resolution and detail retention.
- The $K_0$ trick explicitly formulates "general generation capability" as a matrix—reusable, distributable, and a potential "capability fingerprint" for model cards.

## Limitations & Future Work
- The authors admit SVD may face computational pressure on significantly larger models. Subspace constraints cause erased semantics to drift toward a "middle ground" near the anchor rather than aligning perfectly.
- Implicit concepts like relationships, compositions, or watermarks remain unverified.
- All experiments focus on SD v1.4 / FLUX.1 dev; validation on larger models like SDXL or PixArt is missing. Anchor selection remains a manual bottleneck. Adversarial robustness relies on RECE-style adversarial editing (AT) rather than being inherent to OCE.
- Future Work: Replacing the $d \times d$ orthogonal matrix with structured orthogonality (block diagonal, Cayley parametrization) to handle larger scales; automated anchor discovery via VLMs.

## Related Work & Insights
- **vs UCE / RECE / SPEED**: All are editing-based closed-form solutions using additive $W + \Delta$. OCE crushes them in both effectiveness and efficiency (especially for multi-concept) by changing the formula to multiplicative orthogonal.
- **vs MACE / ESD (Training-based)**: Training methods rely on multi-round fine-tuning. OCE's one-step closed-form solution outperforms them while being 100× cheaper, proving editing methods aren't inherently inferior.
- **vs OFT / Cayley Parametrization**: OFT uses orthogonal transforms for PEFT for stable training; OCE uses it as a "geometric scalpel" for targeted erasure.
- **vs CURE (NeurIPS 2025)**: Also uses orthogonal editing but at the representation layer, whereas OCE operates on cross-attention weights.

## Rating
- Novelty: ⭐⭐⭐⭐½ Changing additive to multiplicative is a significant conceptual shift backed by sound geometric motivation.
- Experimental Thoroughness: ⭐⭐⭐⭐½ Covers single/multi-concept, style, NSFW, adversarial, and DiT architectures.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear logic chain: "geometric analysis → derivation → closed-form solution."
- Value: ⭐⭐⭐⭐⭐ Extreme value for production T2I safety pipelines—4.3s for 100 concepts, zero training.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Mass Concept Erasure in Diffusion Models with Concept Hierarchy](../../AAAI2026/image_generation/mass_concept_erasure_in_diffusion_models_with_concept_hierarchy.md)
- [\[ICLR 2026\] SPEED: Scalable, Precise, and Efficient Concept Erasure for Diffusion Models](../../ICLR2026/image_generation/speed_scalable_precise_and_efficient_concept_erasure_for_diffusion_models.md)
- [\[CVPR 2026\] Prototype-Guided Concept Erasure in Diffusion Models](../../CVPR2026/image_generation/prototype-guided_concept_erasure_in_diffusion_models.md)
- [\[CVPR 2026\] Closed-Form Concept Erasure via Double Projections](../../CVPR2026/image_generation/closed-form_concept_erasure_via_double_projections.md)
- [\[CVPR 2026\] GenErase: Generalizable and Semantically-Aware Concept Erasure in Diffusion Models](../../CVPR2026/image_generation/generase_generalizable_and_semantically-aware_concept_erasure_in_diffusion_model.md)

</div>

<!-- RELATED:END -->
