---
title: >-
  [Paper Note] Multi-turn Consistent Image Editing
description: >-
  [ICCV2025][Image Generation][multi-turn editing] This paper proposes a multi-turn image editing framework based on flow matching. By incorporating dual-objective LQR guidance and an adaptive attention mechanism…
tags:
  - "ICCV2025"
  - "Image Generation"
  - "multi-turn editing"
  - "flow matching"
  - "LQR control"
  - "attention guidance"
  - "FLUX"
  - "image inversion"
date: 2026-05-08
content_hash: fc6307771e306254
---

# Multi-turn Consistent Image Editing

**Conference**: ICCV2025
**arXiv**: [2505.04320](https://arxiv.org/abs/2505.04320)
**Code**: To be confirmed
**Area**: Image Generation
**Keywords**: multi-turn editing, flow matching, LQR control, attention guidance, FLUX, image inversion

## TL;DR

This paper proposes a multi-turn image editing framework based on flow matching. By incorporating dual-objective LQR guidance and an adaptive attention mechanism, it effectively suppresses error accumulation across editing rounds, enabling flexible and controllable iterative editing while maintaining content consistency.

## Background & Motivation

### Problem Definition

Existing image editing methods primarily focus on single-turn editing, i.e., completing an edit in one pass given a text instruction. In practice, however—such as in product design, interactive retouching, and artistic creation—user editing requirements are often vague and incremental, necessitating multiple rounds of iteration to achieve satisfactory results. Naively chaining single-turn methods for multi-turn editing leads to severe **error accumulation**: each round of inversion and sampling introduces truncation errors that compound over successive rounds, causing rapid degradation in image quality, including artifacts, semantic drift, and structural collapse.

### Limitations of Prior Work

**Diffusion-based inversion methods** (e.g., DDIM Inversion, Null-text Inversion): Insufficient inversion accuracy, with error accumulation being especially pronounced in multi-turn editing.

**Flow matching single-turn editing methods** (e.g., RF-Solver, FireFlow): Although second-order ODE solvers reduce single-step truncation error, cumulative error across multiple rounds remains significant.

**RF-Inversion**: Employs single-objective LQR with the source image as reference, achieving good single-turn editing performance; however, in multi-turn settings it references only the previous round's result and gradually drifts from the original image.

**Attention control methods** (e.g., Prompt2Prompt, MasaCtrl): Their applicability to DiT architectures such as FLUX has not been sufficiently explored.

### Core Motivation

The authors distinguish two critical issues:
- **Single-step error vs. multi-turn error**: Higher-order solvers can reduce single-step error, but accumulated error over multiple rounds remains non-negligible and requires a global strategy.
- **Single-step guidance vs. multi-turn guidance**: Single-objective LQR that references only the previous round's result leads to progressive drift; simultaneous reference to the original image is necessary to establish long-range dependency.

## Method

The overall framework is illustrated in Figure 4 and comprises three core components:

### 1. High-Precision Inversion Based on Flow Matching

The method adopts the Rectified Flow framework, modeling the transformation between the image space $x_0 \sim \pi_0$ and the Gaussian noise space $x_1 \sim \pi_1$ as a straight-line path $x_t = tx_1 + (1-t)x_0$. A second-order ODE solver (midpoint method) is used to improve discretization accuracy:

$$X_{t+\Delta t} = X_t + v(\theta, t + \frac{\Delta t}{2}) \Delta t$$

Compared to the first-order Euler method, the truncation error is reduced from $\mathcal{O}(\Delta t^2)$ to $\mathcal{O}(\Delta t^3)$. Combined with the acceleration technique from FireFlow—caching intermediate velocity field results—high-quality inversion is achieved in only 8 steps.

### 2. Dual-objective LQR Guidance

This constitutes the paper's most central contribution. Conventional single-objective LQR (e.g., RF-Inversion) references only the previous round's editing result $X_{k-1,0}$ during sampling, causing gradual deviation from the original image over multiple rounds. This paper proposes simultaneously referencing the **original image** $X_{0,0}$ and the **previous round's result** $X_{k-1,0}$:

**Inversion stage**: Single-objective LQR maps the image to the Gaussian noise space, combined with a second-order ODE solver:

$$X_{t+\Delta t} = X_t + [v_{t+\frac{\Delta t}{2}}(X_t) + \eta(v_{t+\frac{\Delta t}{2}}(X_t \mid X_0) - v_{t+\frac{\Delta t}{2}}(X_t))] \Delta t$$

**Sampling stage**: Dual-objective guidance is introduced by constructing a weighted reference $X_{\text{dual}} = X_{0,0} + \lambda(X_{k-1,0} - X_{0,0})$:

$$X_{t-\Delta t} = X_t + [-v_{t-\frac{\Delta t}{2}}(X_t) - \eta(v_{t-\frac{\Delta t}{2}}(X_t \mid X_{\text{dual}}) - v_{t-\frac{\Delta t}{2}}(X_t))] \Delta t$$

where $\eta$ controls guidance strength and $\lambda$ controls the balance between the previous round's result and the original image. The paper demonstrates that multi-objective LQR is equivalent to single-objective LQR applied to a weighted average of multiple targets (Proposition 1), keeping the framework concise and efficient.

Key parameter settings: $\eta = 0.9$, $\lambda = 0.7$; LQR guidance is applied only during the first 4 sampling steps.

### 3. Adaptive Attention Guidance

Dual-objective LQR ensures multi-turn editing stability, but its strong constraints may suppress editing flexibility. To address this, the paper analyzes the attention behavior across the 19 double blocks of the FLUX model, finding that different layers serve distinct editing functions:

- **High-activation layers** (e.g., blocks 1, 3): Affect global structure and are prone to disrupting image layout.
- **Medium-activation layers** (e.g., blocks 16, 18): Precisely localize to the target editing region.
- **Low-activation layers**: Focus on fine-grained details.

Based on this observation, attention maps from **medium-to-low activation** layers are selected as editing guidance:

1. Each block's attention map $s_{k,l}$ is normalized and mapped via sigmoid.
2. Blocks are ranked by activation magnitude; attention maps ranked 10th to 14th are selected.
3. The selected maps are averaged, and a threshold $\tau$ is applied to generate a binary mask $M_k$.
4. The mask is used to weight attention in the subsequent step: $s_{k+1,l} = \text{softmax}(\frac{QK^T}{\sqrt{d}}) \odot M_k$.

In the mask, editing regions are amplified by $h_{\text{factor}} = 2.0$ and non-editing regions are suppressed by $r_{\text{factor}} = 0.8$, enabling precise local editing.

## Key Experimental Results

### Dataset

The evaluation benchmark is constructed by extending PIE-Bench (a single-turn editing benchmark): GPT-4 Turbo is used to generate 4 additional rounds of editing instructions for each image, yielding a multi-turn editing evaluation benchmark.

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| FID ↓ | Generation quality; measures the naturalness of edited images |
| CLIP-T ↑ | Text–image consistency; measures editing fidelity to the prompt |
| CLIP-I ↑ | Source–edit image similarity; measures content preservation |

### Main Results — Round 4 Quantitative Results (Table 1)

| Method | FID ↓ | CLIP-T ↑ | CLIP-I ↑ | Steps |
|--------|-------|----------|----------|-------|
| RF-Inversion | 5.740 | 24.094 | 0.904 | 28 |
| StableFlow | 20.624 | 24.234 | 0.899 | 50 |
| FlowEdit | 14.547 | 26.703 | 0.894 | 28 |
| RF-Solver | 11.581 | 25.516 | 0.906 | 25 |
| FireFlow | 7.970 | 26.500 | 0.897 | 8 |
| MasaCtrl | 10.811 | 23.797 | 0.886 | 50 |
| PnPInversion | 10.262 | 25.765 | 0.872 | 50 |
| **Ours (15 steps)** | **5.553** | **26.831** | **0.894** | 15 |
| **Ours (8 steps)** | **5.396** | **25.828** | **0.902** | 8 |

The proposed method achieves the best FID (5.396/5.553) while remaining competitive on CLIP-T and CLIP-I, demonstrating that it continues to produce natural and faithful images after multiple rounds of editing.

### Ablation Study (Table 2, Round 4)

| Variant | FID ↓ | CLIP-T ↑ | CLIP-I ↑ |
|---------|-------|----------|----------|
| Single-objective LQR (reference to previous round only) | 9.886 | 26.484 | 0.892 |
| High-activation attention guidance | 6.316 | 26.878 | 0.891 |
| No attention guidance | 6.678 | 26.760 | 0.889 |
| **Full method** | **5.553** | **26.831** | **0.894** |

- Removing dual-objective LQR causes a significant FID increase (+4.3), confirming that dual-objective guidance is critical for suppressing distributional drift.
- Removing attention guidance increases FID by approximately 1.1 and lowers CLIP-I, indicating that attention guidance aids content preservation.
- Using high-activation attention underperforms medium-to-low activation attention, validating the authors' analysis of layer-wise attention functions.

### Multi-turn Reconstruction Experiment

In pure reconstruction experiments (without editing) over 1/2/4/8 rounds, the proposed method consistently preserves color, background, structure, and semantics across all rounds, outperforming all baselines. Although RF-Solver and FireFlow achieve accurate single-step reconstruction, their cumulative errors become apparent over multiple rounds.

## Highlights & Insights

1. **Precise problem formulation**: The paper is the first to systematically distinguish between "single-step error" and "cumulative error" in multi-turn editing, as well as the distinct requirements of "single-step guidance" and "multi-turn guidance," providing a clear conceptual framework for this research direction.
2. **Elegant dual-objective LQR design**: By simultaneously anchoring to the original image and the previous round's result—unified as a single-objective LQR problem via weighted averaging—the approach offers both theoretical guarantees (Proposition 1) and implementation simplicity.
3. **In-depth attention layer analysis**: The paper conducts detailed empirical analysis of the functional roles of FLUX's 19 double blocks, uncovering a global→local→detail hierarchy, and designs an adaptive selection strategy accordingly.
4. **Benchmark contribution**: The multi-turn editing benchmark extended from PIE-Bench fills a gap in evaluation resources for this direction.
5. **Strong practicality**: High-quality editing is completed in 8 steps, offering far superior inference efficiency compared to 50-step diffusion-based methods.

## Limitations & Future Work

1. **Limited dataset scale**: Evaluation relies solely on the multi-turn editing dataset extended from PIE-Bench, lacking validation on larger-scale and more diverse scenarios.
2. **Bounded editing rounds**: Experiments test at most 4–8 rounds of editing; performance on longer editing chains (e.g., 20+ rounds) remains unknown.
3. **Manual token selection**: The current approach requires manually specifying text tokens related to the edit for attention map extraction; future work should automate this step.
4. **Limited editing types**: Validation primarily covers attribute modification, object replacement, and accessory addition; the method's capability for complex geometric transformations or large-scale structural modifications is not fully assessed.
5. **FLUX model dependency**: The attention analysis and guidance strategy are tailored to the FLUX architecture; transferring them to other DiT models requires re-analysis.
6. **Fixed $\lambda$ and $\eta$**: Guidance parameters remain constant across all editing rounds; adaptive adjustment may further improve performance.

## Related Work & Insights

- **RF-Inversion** [Rout et al.]: Proposes single-objective LQR control for flow matching-based editing; the direct predecessor of this work.
- **FireFlow** [Deng et al., 2024]: Second-order ODE solver with velocity field caching for acceleration; this paper directly adopts its acceleration strategy.
- **RF-Solver** [Wang et al.]: Another second-order ODE solver that improves inversion accuracy via the midpoint method.
- **StableFlow** [Avrahami et al., 2024]: Analyzes key layers in FLUX for training-free editing.
- **Prompt2Prompt** [Hertz et al., 2022]: Achieves structure-preserving editing through cross-attention replacement, pioneering the research direction of attention manipulation.
- **FlowEdit** [Kulal et al.]: A flow matching-based editing method, though artifacts accumulate progressively across rounds.
- **ChatEdit** [Cui et al., 2023]: Leverages LLMs for multi-turn interactive editing, but relies on external language models rather than optimizing the generative model itself.

**Implications for future research**:
- The error control strategy for multi-turn editing can be transferred to inter-frame consistency in video editing.
- The weighted averaging idea underlying dual-objective LQR can be extended to multi-condition generation with additional references (e.g., style, pose).
- The methodology for analyzing layer-wise functions in DiT architectures can be applied to editing control in other DiT-based models (e.g., SD3, Hunyuan).

## Rating
- Novelty: ⭐⭐⭐⭐ (Both dual-objective LQR and adaptive attention selection are proposed for the first time; problem formulation is clear)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Multi-metric, multi-baseline comparison with thorough ablation; dataset is relatively small)
- Writing Quality: ⭐⭐⭐⭐ (Clear logic, complete mathematical derivations, intuitive figure design)
- Value: ⭐⭐⭐⭐ (Multi-turn editing is an important and underexplored direction with high practical value)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] ALE: Attribute-Leakage-free Editing for Text-based Image Editing](ale_attribute_leakage_free_editing.md)
- [\[ICCV 2025\] MMAIF: Multi-task and Multi-degradation All-in-One for Image Fusion with Language Guidance](mmaif_multi-task_and_multi-degradation_all-in-one_for_image_fusion_with_language.md)
- [\[ICCV 2025\] OmniPaint: Mastering Object-Oriented Editing via Disentangled Insertion-Removal Inpainting](omnipaint_mastering_object-oriented_editing_via_disentangled_insertion-removal_i.md)
- [\[ICCV 2025\] CharaConsist: Fine-Grained Consistent Character Generation](characonsist_fine-grained_consistent_character_generation.md)
- [\[ICCV 2025\] EEdit: Rethinking the Spatial and Temporal Redundancy for Efficient Image Editing](eedit_rethinking_the_spatial_and_temporal_redundancy_for_efficient_image_editing.md)

</div>

<!-- RELATED:END -->
