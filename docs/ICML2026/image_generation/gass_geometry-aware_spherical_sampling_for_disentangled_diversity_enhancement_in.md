---
title: >-
  [Paper Note] GASS: Geometry-Aware Spherical Sampling for Disentangled Diversity Enhancement in Text-to-Image Generation
description: >-
  [ICML 2026][Image Generation][T2I Diversity] The authors project the sample diversity of T2I under the same prompt onto the CLIP unit hypersphere…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "T2I Diversity"
  - "CLIP Spherical Geometry"
  - "Orthogonal Decomposition"
  - "Test-time Guidance"
  - "Prompt-independent Variation"
date: 2026-05-08
content_hash: 2ee69ebe144faf3f
---

# GASS: Geometry-Aware Spherical Sampling for Disentangled Diversity Enhancement in Text-to-Image Generation

**Conference**: ICML 2026  
**arXiv**: [2602.17200](https://arxiv.org/abs/2602.17200)  
**Code**: https://github.com/L-YeZhu/GASS_T2I (Available)  
**Area**: Diffusion Models / Image Generation  
**Keywords**: T2I Diversity, CLIP Spherical Geometry, Orthogonal Decomposition, Test-time Guidance, Prompt-independent Variation

## TL;DR
The authors project the sample diversity of T2I under the same prompt onto the CLIP unit hypersphere, expanding the projection spread along the "text direction $\mathbf{e}_t$" and the "orthogonal principal residual direction $\mathbf{u}_{\text{ind}}$". By performing gradient optimization on the predicted clean image $\hat{x}_{0|t}$ to map this geometric expansion back to the diffusion/flow sampling trajectory, they simultaneously enhance prompt-dependent (pose, composition) and prompt-independent (background, style) diversity in SD2.1 and SD3-M with almost no loss in quality or alignment.

## Background & Motivation

**Background**: Modern T2I models (diffusion and rectified flow models like SD2.1 and SD3-M) are already strong in fidelity and text alignment. However, repeatedly sampling from the same prompt often yields highly similar images, lacking diversity. Existing test-time enhancement methods (PG, CADS, IG, SPELL) mostly follow the paradigm of "maximizing intra-batch sample dissimilarity / embedding space entropy," aligning with metrics like Vendi Score.

**Limitations of Prior Work**: Pure entropy maximization treats all variation directions equally, failing to distinguish between "semantic changes (perspective, pose)" and "changes not constrained by the prompt (background, style, lighting)." In practice, these methods often only perturb the foreground while the background is blurred into uniform blocks—so-called "diversity gains" come mainly from semantic jittering, while background diversity is largely ignored. Recent methods like Scendi / SPARKE attempt disentanglement using Schur complement entropy but require the number of prompts to equal the number of images, degrading to standard VS in fixed-prompt settings and losing disentanglement capability.

**Key Challenge**: T2I diversity is inherently multi-source—given the prompt "A black colored car," there are prompt-dependent variations (car model, angle) and prompt-independent variations (background, lighting). However, existing metrics and sampling methods provide only a single scalar, without disentangling these two axes for independent measurement and intervention.

**Goal**: (i) Provide a metric capable of geometrically separating prompt-dependent and prompt-independent diversity; (ii) design a test-time sampling intervention that can controllably amplify the spread along either or both axes; (iii) provide a plug-and-play solution for frozen T2I backbones without additional training.

**Key Insight**: CLIP embeddings are naturally normalized on the unit hypersphere $\mathbb{S}^{d-1}$, where text and images share the same manifold—this provides geometric convenience for "anchoring an orthogonal decomposition at $\mathbf{e}_t$." Any image embedding $\mathbf{e}_i$ can be decomposed into a "component along $\mathbf{e}_t$" (semantic alignment direction, essentially CLIPScore) and a "residual in the orthogonal complement." While the residual subspace is high-dimensional, deep representations typically concentrate on low-dimensional manifolds, allowing a principal direction $\mathbf{u}_{\text{ind}}$ to approximate prompt-independent variation.

**Core Idea**: Measure diversity using the "sum of projection ranges along the $\mathbf{e}_t$ and $\mathbf{u}_{\text{ind}}$ axes," $SPP = \mathcal{D}_{\text{dep}} + \mathcal{D}_{\text{ind}}$. During sampling, explicitly push the target CLIP embedding of each image along these two axes by a random amount, then backpropagate this "imagined more dispersed embedding" through the CLIP encoder gradients to modify the predicted clean image $\hat{x}_{0|t}$.

## Method

### Overall Architecture
GASS is a "plug-in" sampling guidance that works entirely at inference time: it uses a frozen T2I backbone (UNet or DiT, supporting both diffusion and rectified flow) and a frozen CLIP image encoder $\mathcal{E}_I$. Given a prompt $c$ and batch size $B$, each GASS step performs three operations:

1. **Disentangled Basis Identification**: Embeddings $\{\mathbf{e}_i\}_{i=1}^B$ are encoded from the current estimated $\hat{x}_{0|t}$. Using $\mathbf{e}_t$ as the first basis, $N=10$ Gram-Schmidt mutually orthogonal directions are randomly generated in its orthogonal complement. The direction with the largest batch mean absolute projection is selected as the principal residual direction $\mathbf{u}_{\text{ind}}$.
2. **Spherical Spread Expansion**: A uniformly sampled perturbation $\delta_i^{\text{dep}}, \delta_i^{\text{ind}} \sim \mathcal{U}[-r, r]$ is added to each embedding along $\mathbf{e}_t$ and $\mathbf{u}_{\text{ind}}$. These are recombined into new target embeddings $\tilde{\mathbf{e}}_i$ and re-normalized back to the unit sphere.
3. **Gradient Write-back**: With the objective $\mathcal{L}_{\text{SPP}} = \sum_i (1 - \mathcal{E}_I(\hat{x}_{i,0|t})^\top \tilde{\mathbf{e}}_i)$, Adam is used for direct gradient descent on $\hat{x}_{0|t}$ (without moving the backbone). The optimized $\hat{x}^*_{0|t}$ is fed back into the solver's next prediction step, pushing the generation trajectory toward a geometrically more dispersed direction.

The entire process is sparse: it is only enabled for 10–20 sampling steps, adding 2.93–3.68 seconds per batch on an A100.

### Key Designs

1. **Spherical Spread Score (SPP)**:
    - **Function**: Decomposes batch diversity under the same prompt into two independently measurable scalars: prompt-dependent and prompt-independent.
    - **Mechanism**: Using the normalized text embedding as the first basis $\mathbf{u}_1 = \mathbf{e}_t$, each image embedding is expanded as $\mathbf{e}_i = (\mathbf{e}_i^\top \mathbf{e}_t)\mathbf{e}_t + \sum_{k\ge 2} (\mathbf{e}_i^\top \mathbf{u}_k)\mathbf{u}_k$. The first term is exactly the CLIPScore. The residual uses the random Gram-Schmidt search in Algo. 1 to find the direction with the strongest response $\mathbf{u}_{\text{ind}} = \arg\max_{\mathbf{r}} \tfrac{1}{B}\sum_i |\mathbf{e}_i^\top \mathbf{r}|$. The diversity metric is the sum of projection ranges on both axes: $\mathcal{D}_{\text{dep}} = \max_i(\mathbf{e}_i^\top \mathbf{e}_t) - \min_i(\mathbf{e}_i^\top \mathbf{e}_t)$, with $\mathcal{D}_{\text{ind}}$ defined isomorphically. $SPP = \mathcal{D}_{\text{dep}} + \mathcal{D}_{\text{ind}}$.
    - **Design Motivation**: Avoids Scendi's limitation of requiring multi-prompt covariance for disentanglement—here, a single prompt batch provides two independent scalars for evaluation or intervention. Validation on ImageNet shows $SPP \approx 0.220$ for real images, about 50% higher than the $0.126$–$0.146$ of SD2.1/SD3-M, indicating the metric's sensitivity to "real vs. generated diversity."

2. **Latent Dynamic Spherical Guidance**:
    - **Function**: Constructs a "geometrically more dispersed" set of target embeddings $\tilde{\mathcal{P}}$ on the CLIP sphere as the objective for subsequent gradient guidance.
    - **Mechanism**: Bounded uniform noise is injected into each image along the two disentangled axes: $\tilde{\mathbf{e}}_i = (\mathbf{e}_i^\top \mathbf{e}_t + \delta_i^{\text{dep}})\mathbf{e}_t + (\mathbf{e}_i^\top \mathbf{u}_{\text{ind}} + \delta_i^{\text{ind}})\mathbf{u}_{\text{ind}} + \mathbf{r}_i$, where $\mathbf{r}_i$ is the initial residual after removing the two principal components, ensuring other image details are preserved. Finally, $\tilde{\mathbf{e}}_i \leftarrow \tilde{\mathbf{e}}_i / \|\tilde{\mathbf{e}}_i\|_2$ is projected back to the unit sphere. The authors also provide Prop. 4.1: in expectation, the Gram determinant (hypervolume) of the batch point set strictly increases, providing theoretical backing that geometric spread is indeed expanded.
    - **Design Motivation**: Compared to the high-dimensional isotropic perturbations in PG / SPELL, restricting the perturbation to two semantically interpreted directions allows for targeted amplification of background or pose changes without disrupting the main semantics. Re-normalization pulls the target back to high-density regions of the CLIP training distribution; ablation shows that removing it drops ImageReward from 0.778 to 0.732.

3. **SPP Gradient Optimization for Predicted Clean Image (Backbone-agnostic)**:
    - **Function**: Translates the geometric targets on the CLIP sphere back to pixel space to actually change the sampling results.
    - **Mechanism**: Since CLIP lacks a decoder and backpropagation through the large T2I backbone is undesirable, gradients are computed directly for each predicted $\hat{x}_{i,0|t}$: $\mathcal{L}_{\text{SPP}} = \sum_i (1 - \mathcal{E}_I(\hat{x}_{i,0|t})^\top \tilde{\mathbf{e}}_i)$. Using Adam (lr $1\times 10^{-4}$, max 60 steps, early stopping patience 4, tolerance $5\times 10^{-4}$), $\hat{x}^*_{i,0|t} \leftarrow \hat{x}_{i,0|t} - \eta \nabla \mathcal{L}_{\text{SPP}}$ is computed and fed back into the solver's state transition equation (applicable to both DDIM and flow ODE).
    - **Design Motivation**: Bypassing gradient backpropagation through the generation network makes the method a complete black box to the backbone, allowing plug-and-play for UNet/DiT and diffusion/flow. Additionally, early stopping and sparse scheduling (active only during steps 10–20) limit the additional overhead per batch to approximately 3 seconds, making it a rare "stable and cheap" design among SOTA test-time methods.

### Loss & Training
No training, entirely test-time. The guidance loss is $\mathcal{L}_{\text{SPP}}$ as defined above. Hyperparameters: $r_{\text{dep}} = r_{\text{ind}} = 0.02$. SD2.1 uses 50 steps, and SD3-M uses 28 steps for sampling. GASS defaults to 20 steps of uniform enablement, with $N = 10$ candidate directions.

## Key Experimental Results

### Main Results

ImageNet (SD3-M, 50 images/class, 1000 classes, "A photo of [class]" template):

| Method | Density↑ | Coverage↑ | VS↑ | ClipScore↑ | SPP↑ |
|------|---------|----------|-----|-----------|------|
| CFG | 1.105 | 0.588 | 28.119 | 0.308 | 0.137 |
| PG (ICLR'24) | 1.103 | 0.586 | 28.119 | 0.308 | 0.129 |
| CADS (ICLR'24) | 1.374 | 0.636 | 28.456 | 0.309 | 0.133 |
| IG (NeurIPS'24) | 1.389 | 0.627 | 27.415 | 0.310 | 0.129 |
| SPELL (ICML'25) | 1.105 | 0.585 | 28.433 | 0.302 | 0.128 |
| **GASS** | 1.164 | 0.611 | **28.877** | **0.313** | **0.141** |

DrawBench (SD3-M, 200 prompts × 10 images): VS 8.115 → **8.212**, ImageReward 0.779 → 0.778, ClipScore 0.318 → **0.320**, SPP 0.113 → **0.114**. On SD2.1, GASS is also the only method to achieve the highest scores in VS, ClipScore, and SPP simultaneously (VS 8.847, ClipScore 0.307, SPP 0.135).

### Ablation Study

| Configuration | VS↑ | ImageReward↑ | ClipScore↑ | SPP↑ |
|------|-----|-------------|-----------|------|
| **GASS (full, $r=0.02$)** | 8.212 | 0.778 | **0.320** | **0.114** |
| IP (Random isotropic perturbation on both axes) | 8.203 | 0.774 | 0.308 | 0.113 |
| RD (Preserve $\mathbf{e}_t$, $\mathbf{u}_{\text{ind}}$ randomly orthogonal) | 8.206 | 0.778 | 0.313 | 0.113 |
| w/o Re-normalization | **8.876** | 0.732 | 0.313 | 0.123 |
| $r_{\text{dep}}=0, r_{\text{ind}}=0.02$ (Background axis expansion only) | 8.207 | 0.787 | 0.319 | 0.111 |
| $r_{\text{dep}}=0.02, r_{\text{ind}}=0$ (Semantic axis expansion only) | 8.206 | 0.780 | 0.320 | 0.112 |
| $r=0.05$ (Excessive expansion) | 8.205 | 0.778 | 0.320 | 0.112 |
| GASS steps 10 (early consecutive) | 8.215 | **0.808** | 0.318 | 0.114 |

### Key Findings
- **Basis selection is key**: After IP replaced $\mathbf{e}_t$ with a random direction, ClipScore fell from 0.320 to 0.308, proving that the geometric decomposition anchoring the text direction and principal residual direction is not just a "random pick."
- **Re-normalization as a quality guardrail**: Removing it caused VS to surge to 8.876, but ImageReward dropped from 0.778 to 0.732—indicating that images start to distort once embeddings fly off the unit sphere. This simple constraint is crucial for the "diversity vs. quality" trade-off.
- **Dual-axis expansion > Single-axis**: Expanding either axis alone yielded a lower SPP than expanding both (0.114 vs. 0.111/0.112), confirming that diversity is indeed composed of two independent additive sources.
- **GASS shows a larger margin on long prompts**: In DrawBench, divided into short/medium/long by word count, the VS increase for long prompts (≥15 words) was the most significant (7.549 → 7.935), counter-intuitively compensating for the phenomenon where complex prompts lead to lower VS.
- **Early consecutive vs. Uniform scheduling**: Early consecutive guidance for 10 steps yielded the highest ImageReward (0.808), but the generated images had lower saturation; uniform scheduling resulted in more natural colors—a trick worth noting for different downstream needs.

## Highlights & Insights
- **Geometric perspective replaces entropy perspective**: By switching "diversity" from "information entropy" to "spherical projection spread," a naturally disentangled, controllable, and visualizable metric is obtained. This is the most "Aha!" moment of the paper—after changing the coordinate system, two previously entangled variables are sliced apart by a single axis $\mathbf{e}_t$.
- **First sampling method to explicitly increase background diversity**: The authors state that GASS is the first sampling-based method to introduce meaningful background variation without modifying the prompt. The diversity of other methods is concentrated in the foreground because their perturbations are isotropic; after being diluted by semantic priors, the "budget" left for the background is essentially zero.
- **Grading on predicted clean images instead of noise prediction**: This trick allows GASS to completely bypass backpropagation through the T2I backbone, which is key to its plug-and-play capability across UNet/DiT and diffusion/flow. This "guidance in $\hat{x}_0$ space" paradigm can be transferred to any CLIP-based evaluation target (de-biasing with FairFace, style control, etc.) simply by replacing $\mathcal{L}_{\text{SPP}}$.
- **Elegant hypervolume guarantee in Prop. 4.1**: Upgrading the desired "increase in diversity" from an empirical observation to a provable proposition where the expected Gram determinant is strictly greater than the original value provides a clean theoretical template for future geometric guidance work.

## Limitations & Future Work
- **Selecting only one principal residual direction**: The authors admit this is a simplification for computational efficiency, assuming prompt-independent variation concentrates on a low-dimensional manifold. For very underspecified prompts (e.g., "an object"), a single $\mathbf{u}_{\text{ind}}$ may be insufficient to cover all residual sub-directions like background, style, lighting, and perspective.
- **Random search for $N=10$ candidate directions is somewhat brute-force**: Future work could use batch SVD / PCA to directly extract top-k principal residual directions, avoiding reliance on random seeds.
- **Dependence on CLIP image encoder as a proxy**: All geometric guidance is built on the CLIP manifold structure; GASS cannot directly amplify details invisible to CLIP (fine-grained textures, high-frequency details). Using stronger representation spaces like DINOv2 or SigLIP is an obvious extension.
- **Early stopping + 60-step Adam still incurs constant overhead**: While 2.93–3.68 s/batch is acceptable, the cost scales linearly at larger batch sizes or higher resolutions (≥1024). Combining this with quantized CLIP or distilled encoders might reduce the cost by an order of magnitude.
- **Not validated on multi-prompt / multi-condition (layout, reference images)**: The authors themselves list "extension to multi-condition input" as a future direction; current geometric decomposition only explicitly handles a single text anchor.

## Related Work & Insights
- **vs. PG (Corso et al., 2024) / SPELL (Kirchhof et al., 2025) / CADS (Sadat et al., 2024) / IG (Kynkäänniemi et al., 2024)**: These methods all use isotropic random perturbations in latent or conditioning spaces to maximize batch dissimilarity, equivalent to blindly increasing VS. GASS restricts perturbations to two geometrically interpretable orthogonal axes, resulting in more controllable diversity, significantly better background effects, and lower quality loss.
- **vs. Scendi / SPARKE (Ospanov et al., 2025; Jalali et al., 2025a)**: These also aim for prompt-dependent/independent disentanglement but rely on the text-image covariance matrix; for a fixed prompt, this matrix is singular/degrades to VS. GASS bypasses this using the geometric projection of a single prompt batch, allowing it to serve as both a metric and an intervention target.
- **vs. CLIP latent editing series (Park et al., 2023; Baumann et al., 2025)**: That line of work mainly performs geometric control for editing/personalization. GASS is among the few works applying these geometric tools to "diversity," a problem seemingly orthogonal to editing, suggesting that "spherical direction control" is a more universal toolkit than previously thought.

## Rating
- Novelty: ⭐⭐⭐⭐ Moving the diversity problem from an information entropy framework to CLIP spherical geometry, with part provable hypervolume guarantees; the disentanglement perspective is not the first, but the execution is clean.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers SD2.1/SD3-M (U-Net + DiT, diffusion + flow), ImageNet + DrawBench, 4 latest SOTA baselines + multiple ablation sets, and analyzes by prompt complexity; however, it lacks extension to 1024 resolution and SDXL/FLUX.
- Writing Quality: ⭐⭐⭐⭐ Geometric derivations are clear, Algo. 1/2 are explicit, and Figs. 1/2 aid intuition; formulas are somewhat dense, potentially steep for readers without a diffusion background.
- Value: ⭐⭐⭐⭐ Plug-and-play, black-box to backbones, almost no quality cost—immediately usable in engineering. The "spherical disentanglement + $\hat{x}_0$ gradient guidance" paradigm has potential for migration to fairness and controllable generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Geometry-Aware Tabular Diffusion](geometry-aware_tabular_diffusion.md)
- [\[ICML 2026\] Envisioning Beyond the Few: Disentangled Semantics and Primitives for Few-Shot Atypical Layout-to-Image Generation](envisioning_beyond_the_few_disentangled_semantics_and_primitives_for_few-shot_at.md)
- [\[ICLR 2026\] GeoDiv: Framework for Measuring Geographical Diversity in Text-to-Image Models](../../ICLR2026/image_generation/geodiv_framework_for_measuring_geographical_diversity_in_text-to-image_models.md)
- [\[CVPR 2026\] SeeThrough3D: Occlusion Aware 3D Control in Text-to-Image Generation](../../CVPR2026/image_generation/seethrough3d_occlusion_aware_3d_control_in_text-to-image_generation.md)
- [\[ICML 2026\] Rethinking FID Through the Geometry of the Reference Dataset](rethinking_fid_through_the_geometry_of_the_reference_dataset.md)

</div>

<!-- RELATED:END -->
