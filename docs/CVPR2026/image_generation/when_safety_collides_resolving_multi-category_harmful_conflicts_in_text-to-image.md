---
title: >-
  [Paper Note] When Safety Collides: Resolving Multi-Category Harmful Conflicts in Text-to-Image Diffusion via Adaptive Safety Guidance
description: >-
  [Image Generation] This paper proposes Conflict-aware Adaptive Safety Guidance (CASG), a training-free plug-and-play framework that resolves safety degradation caused by directional conflicts in multi-category aggregatio…
tags:
  - "Image Generation"
date: 2026-05-08
content_hash: 0384582b2676f8ce
---

# When Safety Collides: Resolving Multi-Category Harmful Conflicts in Text-to-Image Diffusion via Adaptive Safety Guidance

## Basic Information

- **Conference**: CVPR 2026
- **arXiv**: [2602.20880](https://arxiv.org/abs/2602.20880)
- **Code**: [GitHub](https://github.com/tmllab/2026_CVPR_CASG)
- **Area**: Image Generation / T2I Safety
- **Keywords**: Diffusion model safety, harmful content mitigation, safety guidance, multi-category conflict, training-free framework

## TL;DR

This paper proposes Conflict-aware Adaptive Safety Guidance (CASG), a training-free plug-and-play framework that resolves safety degradation caused by directional conflicts in multi-category aggregation. CASG dynamically identifies the harmful category most aligned with the current generation state and applies safety guidance exclusively along that direction.

## Background & Motivation

Text-to-image (T2I) diffusion models (e.g., Stable Diffusion, Hunyuan-DiT) have achieved remarkable progress in high-quality image generation, but simultaneously introduce safety risks regarding harmful content generation (hate, pornography, violence, illegal content, etc.). Existing safety guidance methods (e.g., SLD, SAFREE) steer generation away from harmful regions by imposing safety directions in latent or text space, serving as lightweight safety solutions that require no model modification.

**Core Problem**: Existing safety guidance methods handle multi-category harmful content by naively concatenating all harmful keywords into an aggregated set and deriving a single unified safety direction. This category-agnostic design implicitly assumes that different types of harm are compatible. Through extensive experiments, however, the authors demonstrate that this assumption does not hold — different harmful categories define distinct and often incompatible safety directions, and forced aggregation produces **Harmful Conflicts** that actually degrade safety performance.

Harmful conflicts manifest in two forms:

**Directional Inconsistency** → Safety Misalignment Degradation: Safety directions from different categories point toward incompatible or even opposing directions. For example, applying a "hate" safety direction to pornographic prompts increases the harmful rate from 67.2% to 72.4%, far worse than correctly applying the "pornography" direction, which achieves 3.2%.

**Directional Attenuation** → Safety Averaging Degradation: Multi-category aggregation causes heterogeneous directions to partially cancel, weakening the net safety signal. Applying the "pornography" direction alone yields a harmful rate of 3.2%, which rises to 5.8% when "hate" is added, and further to 48.8% when all categories are aggregated.

## Method

### Overall Architecture

CASG is a plug-and-play safety correction framework consisting of two core components:

- **CaCI (Conflict-aware Category Identification)**: Dynamically identifies the harmful category most aligned with the current generation state.
- **CrGA (Conflict-resolving Guidance Application)**: Applies safety correction exclusively along the identified dominant category direction.

CASG can be instantiated with both latent-space safety mechanisms (CASG+SLD) and text-space safety mechanisms (CASG+SAFREE).

### Latent-Space Method: Conflict-aware Safety Steering (CASG+SLD)

**Step 1: Harmful Guidance Computation.** For each category $h_i$ in the predefined harmful keyword set $\mathcal{H}$, the harmful guidance in latent space is computed. Following the classifier-free guidance principle, the noise prediction under harmful conditioning is obtained:

$$\hat{\epsilon}_i = \epsilon_\theta(z_t, c_{h_i})$$

The harmful guidance direction is then derived by subtracting the unconditional noise prediction:

$$g_i = \hat{\epsilon}_i - \epsilon_\theta(z_t)$$

This yields the harmful guidance set $G = \{g_1, \ldots, g_k\}$.

**Step 2: Aligned Category Identification (CaCI).** The prompt guidance is computed analogously:

$$g_p = \epsilon_\theta(z_t, c_p) - \epsilon_\theta(z_t)$$

The alignment between each harmful guidance $g_i$ and the prompt guidance $g_p$ is measured via cosine similarity:

$$\cos\theta_i = \frac{g_i \cdot g_p}{\|g_i\| \|g_p\|}$$

A higher cosine similarity indicates greater alignment between that harmful category and the current generation trajectory. The category with maximum cosine similarity is selected as the dominant harmful direction:

$$h^* = h_{\arg\max_i \cos\theta_i}$$

**Step 3: Aligned Category Application (CrGA).** SLD safety correction is applied exclusively along the dominant harmful category $h^*$, replacing the original multi-category aggregated direction. All other SLD mechanisms and hyperparameters remain unchanged.

### Text-Space Method: Conflict-aware Orthogonal Projection (CASG+SAFREE)

**Step 1: Projection Residual Computation.** For each harmful category $h_i$, SAFREE represents its harmful concept via a subspace projection matrix $P_{h_i}$. The projection of the prompt embedding onto the orthogonal complement of the harmful subspace is computed:

$$p_{h_i}^\perp = (I - P_{h_i}) p$$

where $p_{h_i}^\perp$ denotes the residual prompt embedding after removing components aligned with harmful category $h_i$.

**Step 2: Aligned Category Identification (CaCI).** A smaller residual norm $\|p_{h_i}^\perp\|$ indicates higher alignment between the prompt and the corresponding harmful category. The category with minimum residual norm is selected:

$$h^* = h_{\arg\min_i \|p_{h_i}^\perp\|}$$

**Step 3: Aligned Category Application (CrGA).** Orthogonal projection is performed using only the subspace $P_{h^*}$ of the dominant harmful category, avoiding cross-category interference.

### Key Design Highlights

1. **Training-free and plug-and-play**: No additional training or external API calls are required; CASG is directly embedded into existing safety mechanisms.
2. **Dynamic step-wise adaptation**: The dominant category identification is updated at every denoising timestep, capturing the dynamic evolution of harmful semantics throughout the generation process.
3. **Generation-state-based rather than text-classification-based**: Unlike LLM-based pre-classification approaches (e.g., GPT+SLD), CASG makes decisions based on the actual generation trajectory, more accurately tracking the dynamic variation of harmful conflicts.

## Experiments

### Main Results

| Method | Conflict-Aware | I2P ↓ | T2VSafetyBench ↓ | Unsafe Diffusion ↓ | CoProv2 ↓ | FID ↓ | CLIP ↑ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| SD-v1.5 | - | 42.2 | 58.3 | 52.3 | 28.2 | - | 31.43 |
| ESD | ✗ | 42.0 | 57.4 | 50.6 | 28.1 | 38.15 | 31.35 |
| UCE | ✗ | 26.7 | 28.2 | 33.0 | 19.4 | 77.41 | 29.12 |
| RECE | ✗ | 21.5 | 18.9 | 22.3 | 8.6 | 67.35 | 27.67 |
| SafetyDPO | ✗ | 13.7 | 24.0 | 16.7 | 4.2 | 49.64 | 30.61 |
| SAFREE | ✗ | 20.0 | 41.5 | 24.2 | 14.2 | 43.78 | 30.53 |
| **CASG+SAFREE** | ✓ | **18.9** | **37.5** | **17.5** | **11.8** | 46.25 | 30.35 |
| SLD | ✗ | 12.7 | 25.2 | 15.7 | 7.1 | 52.11 | 29.22 |
| **CASG+SLD** | ✓ | **10.2** | **9.8** | **9.8** | **3.9** | 52.00 | 29.36 |

CASG+SLD achieves state-of-the-art performance on all four benchmarks, reducing harmful rates by up to 15.4% (from 25.2% to 9.8% on T2VSafetyBench). FID and CLIP scores remain virtually unchanged, indicating no degradation in generation quality.

### LLM-Assisted vs. CASG Ablation

| Method | I2P ↓ | T2VSafetyBench ↓ | UD ↓ |
|:---|:---:|:---:|:---:|
| SLD | 12.7 | 25.2 | 15.7 |
| GPT-4o+SLD | 11.6 | 12.3 | 20.1 |
| QwenGuard+SLD | 14.0 | 21.1 | 23.3 |
| **CASG+SLD** | **10.2** | **9.8** | **9.8** |

LLM-assisted approaches (GPT-4o or QwenGuard pre-classification followed by SLD) show limited or even negative effect for two reasons: (1) LLMs misclassify mixed or ambiguous prompts; (2) the assigned category is fixed at the start of generation and cannot adapt to conflicts that evolve dynamically during denoising. CASG updates at every timestep, substantially outperforming LLM-assisted alternatives.

### Key Findings

1. **Multi-category aggregation does not imply greater safety**: This is the paper's most central finding. Aggregating more harmful categories may actually weaken safety (e.g., full-category aggregation yields a harmful rate of 48.8% on pornographic prompts, far exceeding 3.2% for the single correct category).
2. **Conflicts are systematic**: Consistent degradation patterns are observed across different base models, safety mechanisms, and harmful keyword definitions.
3. **Dynamic identification outperforms static classification**: Step-wise identification based on generation trajectory is more effective than one-shot text classification by LLMs.
4. **Plug-and-play improvements are substantial**: CASG consistently improves both SLD and SAFREE without introducing any additional training cost.

## Highlights & Insights

- The paper identifies a widely overlooked yet consequential problem — multi-category harmful conflicts — and provides the first systematic characterization of directional inconsistency among safety guidance directions.
- The method design is remarkably elegant: dominant category selection relies solely on cosine similarity or residual norm, requiring no training and no external models.
- Strong generality: the framework applies to both latent-space (SLD) and text-space (SAFREE) safety mechanisms.
- Rigorous and comprehensive experiments: four safety benchmarks, benign generation quality evaluation, LLM-assisted comparisons, and multi-model validation.

## Limitations & Future Work

- Computing noise predictions or projection residuals separately for each harmful category at every timestep incurs computational overhead that scales linearly with the number of categories.
- Validation is limited to SD v1.5; applicability to newer architectures (e.g., DiT, FLUX variants) has not been thoroughly verified.
- The method relies on a predefined harmful keyword set; incomplete or improperly defined keyword coverage may cause certain harmful types to be overlooked.
- Selecting only one dominant category per timestep may result in missed harmful semantics for prompts containing multiple simultaneous harmful concepts.

## Rating

⭐⭐⭐⭐ (4/5)

- **Novelty**: ⭐⭐⭐⭐⭐ — The identification of "harmful conflicts" is highly valuable and challenges the prevailing intuition that aggregating more categories yields greater safety.
- **Technical Depth**: ⭐⭐⭐⭐ — The analysis of conflict mechanisms (directional inconsistency + directional attenuation) is systematic and thorough; the CDRR analysis is convincing.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Four benchmarks, multiple baselines, LLM comparisons, and ablation analysis constitute a very comprehensive evaluation.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Problem motivation is clearly presented; PCA direction visualizations and CDRR heatmaps are highly persuasive.
- **Value**: ⭐⭐⭐⭐ — A practical plug-and-play enhancement for the T2I safety domain with high utility, though the application scope is relatively focused.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] TAP: A Token-Adaptive Predictor Framework for Training-Free Diffusion Acceleration](tap_a_token-adaptive_predictor_framework_for_training-free_diffusion_acceleratio.md)
- [\[ICCV 2025\] TRCE: Towards Reliable Malicious Concept Erasure in Text-to-Image Diffusion Models](../../ICCV2025/image_generation/trce_towards_reliable_malicious_concept_erasure_in_text-to-image_diffusion_model.md)
- [\[ICCV 2025\] DynamicID: Zero-Shot Multi-ID Image Personalization with Flexible Facial Editability](../../ICCV2025/image_generation/dynamicid_zero-shot_multi-id_image_personalization_with_flexible_facial_editabil.md)
- [\[ICCV 2025\] FlowTok: Flowing Seamlessly Across Text and Image Tokens](../../ICCV2025/image_generation/flowtok_flowing_seamlessly_across_text_and_image_tokens.md)
- [\[ICCV 2025\] MamTiff-CAD: Multi-Scale Latent Diffusion with Mamba+ for Complex Parametric Sequence](../../ICCV2025/image_generation/mamtiff-cad_multi-scale_latent_diffusion_with_mamba_for_complex_parametric_seque.md)

</div>

<!-- RELATED:END -->
