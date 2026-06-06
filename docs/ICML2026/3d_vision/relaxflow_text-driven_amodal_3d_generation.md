---
title: >-
  [Paper Note] RelaxFlow: Text-Driven Amodal 3D Generation
description: >-
  [ICML 2026][3D Vision][amodal 3D generation] RelaxFlow formalizes "text-driven completion of occluded 3D objects" as a **decoupling of control granularity for dual objectives**. It proposes a training-free dual-branch in…
tags:
  - "ICML 2026"
  - "3D Vision"
  - "amodal 3D generation"
  - "text-driven"
  - "training-free"
  - "low-pass relaxation"
  - "dual-branch flow model"
date: 2026-05-08
content_hash: 49087174c3d5d17d
---

# RelaxFlow: Text-Driven Amodal 3D Generation

**Conference**: ICML 2026  
**arXiv**: [2603.05425](https://arxiv.org/abs/2603.05425)  
**Code**: https://github.com/viridityzhu/RelaxFlow  
**Area**: 3D Vision / Diffusion Models / Multimodal VLM  
**Keywords**: amodal 3D generation, text-driven, training-free, low-pass relaxation, dual-branch flow model

## TL;DR
RelaxFlow formalizes "text-driven completion of occluded 3D objects" as a **decoupling of control granularity for dual objectives**. It proposes a training-free dual-branch inference framework: the observation branch maintains pixel-level hard constraints, while the semantic prior branch achieves low-pass relaxation through "multi-prior consensus + attention logit Gaussian blurring." The study theoretically proves that this relaxation is equivalent to low-pass filtering of the generated vector field, reducing Point-FID from 100.38 to 81.11 on SOTA models like SAM3D and TRELLIS.

## Background & Motivation
**Background**: Image-to-3D generation (feedforward models like TRELLIS, SAM3D, and Trellis-XL) can transform a single image into usable 3D assets by feeding image tokens into a conditioned rectified flow to predict sparse structures (occupancy grids) and structured latents (appearance).

**Limitations of Prior Work**: When input images are severely occluded, visible pixels are insufficient to uniquely determine object categories (e.g., a headboard could belong to a bed, sofa, or dresser). Feedforward models, accepting only image tokens, suffer from "semantic under-determinacy" and **collapse to the most frequent explanation (observation over-fitting)** without user intervention. Conversely, optimization-based SDS-style editing methods, while following text, tend to over-smooth or destroy visible evidence due to conflicts between semantic and reconstruction gradients.

**Key Challenge**: Existing methods apply a **unified control granularity** to enforce two objectives simultaneously—observations must be **rigidly obeyed** (visual fidelity), whereas text serves only as **flexible structural guidance** (tolerating local deviations to fit observations). When both are placed in the same conditional branch competing for attention, a trade-off occurs: either the text is suppressed, or the observation is compromised.

**Goal**: (1) Formalize the new task of text-driven amodal 3D generation; (2) Design an **inference-time solution without retraining the generator** that satisfies both "hard observation constraints + soft text guidance"; (3) Provide an interpretable theoretical explanation for stable convergence.

**Key Insight**: The authors observe that the oracle "semantic transport vector field" $\bm{v}_{\rm sem}$ is **band-limited** in the frequency domain—category-level geometry (e.g., "bed shape") occupies only low frequencies. In contrast, instance details and texture conflicts introduced by text/image tokens are **high-frequency noise**. By applying a low-pass filter to the semantic branch's velocity field, one can preserve the "global geometric corridor" while discarding high-frequency jitter that disrupts observations.

**Core Idea**: The generation process is split into **two ODE flows with shared states and independent conditions**. The observation branch runs the original $v_\theta(x_t, t, c_{\rm obs})$, while the semantic branch applies **Gaussian blurring to attention logits** to obtain a relaxed velocity field $\tilde v_\theta = \mathcal R_\sigma[v_\theta(x_t, t, c_{\rm prior})]$. These are fused using time-dependent weights, where the semantic branch dominates global modes early on, and the observation branch refines details later.

## Method

### Overall Architecture
RelaxFlow is a **training-free inference-time module** compatible with any "image token + rectified flow" image-to-3D generator (the paper uses TRELLIS and SAM3D). Inputs include an occluded image $I$, its visibility mask $M$, and the user's text prompt $p$. The output is a complete 3D asset decoded after two-stage flow sampling: Sparse Structure (SS, predicting a $64^3$ occupancy grid) and Structured Latent (SLAT, predicting voxel-level features). The key modification replaces the standard Euler update

$$x_{k+1}=x_k+\Delta t\,v(x_k,t_k,c)$$

with an interpolation between **shared-state dual branches**: the observation branch uses $c_{\rm obs}=E(I,M)$ as usual; the semantic branch converts text $p$ into $N=3$ visual proxy images, encodes them as $c_{\rm prior}$, and applies attention logit blurring to yield $\tilde v_{\rm prior}$. Finally, the velocities are fused using a time-dependent weight $\alpha_k$ and visibility mask $m_i$. The core of the pipeline is this "text → visual proxy → low-pass velocity field" channel, retrofitted without adapter training.

### Key Designs

1.  **Multi-Prior Consensus**:
    - **Function**: Converts the text prompt $p$ into native visual tokens while erasing "instance style pollution" from a single reference image.
    - **Mechanism**: For a prompt $p$, multiple reference images $\{(I_p^n,M_p^n)\}$ with shared semantics but varied appearances (e.g., different "red-billed birds") are retrieved or generated via Z-Image. Their token sequences are **concatenated into one long sequence for cross-attention**. Shared attributes gain cumulative attention, while conflicting textures are naturally diluted. This consensus approximates $\delta_{\rm prior}$ (the residual between visual proxies and true intent) in the Wasserstein bound of §3.2, enabling text following without an adapter.
    - **Design Motivation**: Modern feedforward 3D generators use visual tokens rather than text embeddings. One must either retrain with adapters (expensive and prone to distribution shift) or "translate" text into visual proxies. The latter offers a unified interface for single-prior, multi-prior, and user-provided images, making it the most robust and cost-effective approach.

2.  **Low-Pass Relaxation via Attention Logit Smoothing**:
    - **Function**: Applies a low-pass filter $\mathcal R_\sigma$ to the semantic velocity field to retain global geometry (e.g., "bed/sofa") and suppress instance-level texture conflicts, preventing the semantic branch from clashing with observations.
    - **Mechanism**: The authors prove (Proposition A.4 + Theorem A.9) that if $\bm v_{\rm sem}$ is band-limited and errors are high-frequency, $\tilde v_\theta=\mathcal R_\sigma[v_\theta]$ strictly reduces the $L_2$ path norm semantic error $\mathcal E_{\rm sem}$. This tightens the Wasserstein-2 upper bound: $\mathcal W_2(p,\hat p)\le C(\mathcal E_{\rm obs}+\mathcal E_{\rm sem}(\tilde v)+\delta_{\rm prior})$. Implementation-wise, instead of expensive **direct convolution on the vector field**, 1D Gaussian convolutions $\tilde L=G_\sigma^{(q)}*_q L *_k G_\sigma^{(k)}$ are applied along the query and key indices of the prior branch's cross-attention logit matrix $L_{i,j}=q_i^\top k_j/\sqrt d$. Given the 2D/3D grid arrangement of tokens, this is equivalent to separable 2D Gaussian blurring and induces an equivalent velocity field relaxation (Appendix A.4). The default $\sigma=1.0$ is robust within $[0.5, 2]$.
    - **Design Motivation**: Directly weighting two prompts (SDS/CFG) causes high-frequency instance conflicts to pull the state inconsistently at each step, shattering the "semantic corridor." "Thickening" this corridor (Fig. 2) allows the ODE solution to converge toward coarse modes like "bed" without being driven away by instance noise.

3.  **Visibility-Aware Dual-Branch Fusion**:
    - **Function**: Ensures the semantic prior **only** acts on "early stages + truly occluded voxels," returning control to the observation branch for visible regions and late-stage refinement to avoid overpainting.
    - **Mechanism**: In the time dimension, a linear cutoff schedule $\alpha_k=\max(1-k/K,\,0)\cdot\mathbb 1[k\le\lfloor\rho K\rfloor]$ is used (default $\rho=0.2$, meaning the prior acts only in the first 20% of steps), matching the induction bias of diffusion models (global structure first, details later). Spatially, for each voxel, a depth difference $\Delta_i=z_i-D'(u_i,v_i)$ is calculated via z-buffer projection from the known object pose, yielding soft visibility weights $m_i\in(0,1]$ through Gaussian falloff. The SLAT fusion is $v_i=v_{\rm obs,i}+(1-m_i)\alpha_k(\tilde v_{\rm prior,i}-v_{\rm obs,i})$, ensuring visible voxels retain observation velocities while occluded voxels receive prior offsets.
    - **Design Motivation**: Ablation shows that removing the visibility mask causes a larger performance drop than removing low-pass relaxation (81.1→92.3 vs. 81.1→87.1). This confirms that "obeying observations where appropriate" is more critical than the smoothness of the semantic corridor.

### Loss & Training
**Zero training and zero fine-tuning**. All backbone parameters (SS + SLAT flow models from SAM3D / TRELLIS) are frozen. Modifications occur only during inference in cross-attention calculations and Euler update formulas. Default settings: $N=3$ prior images, $\sigma=1.0$, $\rho=0.2$, running on a single A40 GPU. ExtremeOcc-3D enables the module only during $SS$ (semantics affecting geometry), while AmbiSem-3D enables it for both $SS$ and $SLAT$ (semantics affecting structure and appearance).

## Key Experimental Results

### Main Results

**ExtremeOcc-3D** (264 samples from 3D-FUTURE/3D-FRONT with occlusion $\ge 80\%$, category-level text priors):

| Backbone | Method | CLIP_img↑ | CLIP_txt↑ | FID↓ | LPIPS↓ | Point-FID↓ |
|---|---|---|---|---|---|---|
| TRELLIS | baseline | 0.78 | 23.14 | 122.68 | 0.83 | 141.48 |
| TRELLIS | **+ RelaxFlow** | 0.80 | 24.09 | 100.75 | 0.80 | **97.79** |
| SAM3D | baseline | 0.84 | 24.08 | 50.73 | 0.54 | 100.38 |
| SAM3D | Amodal2D+SAM3D | 0.76 | 21.59 | 94.38 | 0.56 | 127.27 |
| SAM3D | Amodal3R | 0.77 | 22.29 | 118.49 | 0.60 | 129.46 |
| SAM3D | **+ RelaxFlow** | **0.87** | **27.26** | **39.44** | **0.51** | **81.11** |

Improvements across all metrics for both backbones. CLIP_img and LPIPS improved despite following text, indicating that the semantic prior operates only where necessary without sacrificing fidelity.

**AmbiSem-3D** (21 multi-solution ambiguous samples from ObjaverseXL + User Study, n=32):

| Method | CLIP_img↑ | CLIP_txt↑ | Text Alignment↑ | 3D Fidelity↑ | Total Preference↑ |
|---|---|---|---|---|---|
| SAM3D | 0.85 | 26.29 | 4.84% | 13.59% | 9.22% |
| TRELLIS (multi-view) | 0.80 | 26.59 | 3.75% | 8.28% | 6.02% |
| SDXL → TRELLIS | 0.81 | 26.76 | 6.09% | 8.91% | 7.50% |
| SDXL → SAM3D | 0.79 | 26.71 | 11.41% | 6.09% | 8.75% |
| **RelaxFlow (ours)** | **0.87** | **27.23** | **73.91%** | **63.13%** | **68.52%** |

Human preference is overwhelmingly superior (68.52%). Automated metrics are highest for both observation and text, proving that RelaxFlow "understands the text without hallucinating."

### Ablation Study (ExtremeOcc-3D, SAM3D backbone)

| Configuration | Point-FID↓ | Description |
|---|---|---|
| Full RelaxFlow | **81.1** | Complete model |
| w/o Low-Pass Relax | 87.1 | Attention blurring removed; semantic noise leaks in |
| w/o Visibility Mask | 92.3 | Prior pollutes visible regions; largest drop |
| cutoff $\rho=0.4$ | 86.5 | Prior intervenes too long |
| cutoff $\rho=1.0$ | 89.9 | Prior active throughout; destroys details |
| LP Relax $\sigma=2.5$ | 95.2 | Excessive blurring; semantic signal is lost |
| Generated priors (Z-Image) | 82.7 | Generated proxies instead of retrieved ones |

### Key Findings
- **Visibility mask is more critical than low-pass relaxation**: Removing the former drops Point-FID by 11.2, whereas removing the latter drops it by 6.0. Counter-intuitively, "who to listen to" is more important than "how clean the semantic signal is," validating the dual-branch decoupling as the core contribution.
- **$\sigma$ and $\rho$ exhibit "goldilocks" behavior**: Over-extending the prior ($\rho=1.0$) is worse than $\rho=0.4$; $\sigma=2.5$ filters out the semantic signal. Low-pass relaxation must be precisely tuned, not maximized.
- **Generated vs. Retrieved priors**: The gap is small (82.7 vs. 81.1), meaning T2I models can generate proxy images effectively when no retrieval pool is available.

## Highlights & Insights
- **Elegant framing of "decoupling control granularity"**: Reinterprets past conflicts (like SDS/CFG merging prompts) as the need for different frequency-band velocity fields for hard constraints versus soft guidance.
- **Duality of theory and implementation**: Relaxation is modeled as a low-pass filter on the vector field (theoretical Wasserstein bound) and implemented as 2D Gaussian blurring on attention logits. This "spec-code isomorphism" is highly portable.
- **Completely training-free and non-invasive**: No adapters, LoRA, or flow retraining required. Any new image-to-3D backbone can integrate this mechanism at zero cost.
- **High transferability**: The "attention logit blurring" trick can be generalized to image editing (low-pass text tokens to preserve details), video generation (low-pass time dimension for stability), and LLM controllable generation.

## Limitations & Future Work
- **Dependency on cross-attention backbones**: Blurring $\mathcal{R}_\sigma$ is applied to attention logits; not directly applicable to pure MLP or discrete token-based backbones.
- **Inference overhead**: Dual-branch inference requires two velocity estimations and logit convolutions per step, roughly doubling runtime.
- **Dependency on pose estimation**: Backbones like TRELLIS that lack pose estimation cannot use the visibility mask, which was shown to be the most critical component.
- **Small diagnostic benchmark**: AmbiSem-3D has only 21 samples and no GT. While user studies are significant, broader verification is needed.
- **Future Directions**: Replacing the visibility mask with self-supervised masks (e.g., SAM + Depth-Anything) to unlock uncalibrated real-world scenes; exploring token-adaptive low-pass relaxation.

## Related Work & Insights
- **vs. SAM3D / TRELLIS / Amodal3R**: These models either cannot accept text control or require retraining (Amodal3R). RelaxFlow enables text-driven disambiguation as a plug-in and outperforms the original backbone's Point-FID (81.1 vs. 100.4), proving the retrofit approach can compete with retraining.
- **vs. SDXL → 3D pipelines**: Two-stage solutions introduce geometric inconsistencies. RelaxFlow solves the problem within the 3D flow, bypassing 2D editing drift.
- **vs. FlowEdit / CFG**: CFG weights prompt at the score level without distinguishing spectral properties. RelaxFlow explicitly maintains full spectrum for hard constraints and low-pass for soft guidance, effectively adding a frequency domain prior.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Formalizes a real-world problem (text-driven amodal 3D) with both framing breakthroughs and mathematical grounding.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers dual backbones, dual benchmarks, user studies, and ablations. One star off for the small size of AmbiSem-3D and missing runtime statistics.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear structure (task → theory → implementation), excellent conceptual diagrams, and complete proofs.
- Value: ⭐⭐⭐⭐⭐ Training-free and plug-and-play for cross-attention 3D generators. The "low-pass relaxation" paradigm has broad implications for controllable generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] NURBGen: High-Fidelity Text-to-CAD Generation through LLM-Driven NURBS Modeling](../../AAAI2026/3d_vision/nurbgen_high-fidelity_text-to-cad_generation_through_llm-driven_nurbs_modeling.md)
- [\[CVPR 2026\] Text–Image Conditioned 3D Generation](../../CVPR2026/3d_vision/text-image_conditioned_3d_generation.md)
- [\[CVPR 2026\] SeeThrough3D: Occlusion Aware 3D Control in Text-to-Image Generation](../../CVPR2026/3d_vision/seethrough3d_occlusion_aware_3d_control_in_text-to-image_generation.md)
- [\[AAAI 2026\] AnchorDS: Anchoring Dynamic Sources for Semantically Consistent Text-to-3D Generation](../../AAAI2026/3d_vision/anchords_anchoring_dynamic_sources_for_semantically_consiste.md)
- [\[AAAI 2026\] MR-CoSMo: Visual-Text Memory Recall and Direct Cross-Modal Alignment Method for Query-Driven 3D Segmentation](../../AAAI2026/3d_vision/mr-cosmo_visual-text_memory_recall_and_direct_cross-modal_alignment_method_for_q.md)

</div>

<!-- RELATED:END -->
