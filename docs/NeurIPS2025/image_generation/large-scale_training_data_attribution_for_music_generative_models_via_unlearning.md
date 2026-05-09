---
title: >-
  [Paper Note] Large-Scale Training Data Attribution for Music Generative Models via Unlearning
description: >-
  [NeurIPS 2025][Image Generation][Training Data Attribution] This paper applies machine unlearning-based training data attribution (TDA) to a large-scale text-to-music diffusion model (115K tracks), identifying optimal hyperparameter configurations via grid search and comparing against non-counterfactual methods, thereby demonstrating the feasibility of unlearning-based TDA in the music generation domain.
tags:
  - NeurIPS 2025
  - Image Generation
  - Training Data Attribution
  - Machine Unlearning
  - Music Generation
  - Diffusion Models
  - Copyright Attribution
date: 2026-05-08
content_hash: 6541efd5d74a8986
---

# Large-Scale Training Data Attribution for Music Generative Models via Unlearning

**Conference**: NeurIPS 2025
**arXiv**: [2506.18312](https://arxiv.org/abs/2506.18312)
**Code**: N/A
**Area**: Image Generation
**Keywords**: Training Data Attribution, Machine Unlearning, Music Generation, Diffusion Models, Copyright Attribution

## TL;DR

This paper applies machine unlearning-based training data attribution (TDA) to a large-scale text-to-music diffusion model (115K tracks), identifying optimal hyperparameter configurations via grid search and comparing against non-counterfactual methods, thereby demonstrating the feasibility of unlearning-based TDA in the music generation domain.

## Background & Motivation

The rapid advancement of generative AI in music has raised serious copyright and attribution concerns: models may inadvertently reproduce copyrighted material, while the contributions of original creators receive insufficient recognition and compensation. **Training Data Attribution (TDA)** aims to identify which training samples contribute most to a model's specific outputs, thereby supporting fairer recognition of artistic contributions.

TDA methods fall into two categories:

**Black-box methods** (no model access): similarity-based collaborative attribution using external encoders to compute the similarity between generated outputs and training data (e.g., CLAP, CLEWS). Practical and straightforward, but dependent on the encoder's perspective and not necessarily reflective of the generative model's internal behavior.

**White-box methods** (access to model parameters): counterfactual reasoning—"How would model predictions change if a given training sample were removed?" The most direct approach is leave-one-out retraining (computationally infeasible); influence functions can approximate such changes but are also limited on large models.

**Machine unlearning** has emerged as a new direction: it "forgets" a specific training sample by maximizing its loss via gradient ascent, combined with Fisher Information Matrix (FIM) regularization to prevent catastrophic forgetting. Prior unlearning-based TDA work has been explored only in other domains; its application to music generation remains unexplored.

Existing music TDA work (Deng et al.) validated influence function methods only on Music Transformer with the MAESTRO dataset (~200 hours of piano music). This paper is the first to extend TDA to a large-scale text-to-music DiT model (115K tracks, ~4,356 hours of multi-style music).

## Method

### Overall Architecture

The attribution pipeline is as follows: given a generated sample $\hat{\mathbf{z}}$ and a training sample $\mathbf{z}_i$, the attribution score is defined as the loss difference before and after unlearning:

$$\tau(\hat{\mathbf{z}}, \mathbf{z}_i) = \mathcal{L}(\mathbf{z}_i, \theta_{\setminus \hat{\mathbf{z}}}) - \mathcal{L}(\mathbf{z}_i, \theta_0)$$

The **mirrored influence hypothesis** is leveraged: rather than unlearning each training sample individually (requiring $N$ operations), the generated sample $\hat{\mathbf{z}}$ is unlearned and the resulting change in loss across training samples is observed. Only one unlearning operation is required per target sample.

### Key Designs

#### 1. Unlearning Algorithm

Directly maximizing the target sample's loss leads to catastrophic forgetting; FIM regularization is therefore incorporated:

$$\mathcal{L}_{\text{unlearn}}^{\hat{\mathbf{z}}}(\theta) = -\mathcal{L}(\hat{\mathbf{z}}, \theta) + \frac{N}{2}(\theta - \theta_0)^\top \mathbf{F} (\theta - \theta_0)$$

The first term forgets the target sample via gradient ascent; the second term applies an FIM-weighted quadratic penalty to preserve overall model performance. The FIM quantifies each parameter's influence on model outputs, imposing stronger constraints on highly influential parameters.

The derived update rule is: $\theta = \theta_0 + \frac{1}{N} \mathbf{F}^{-1} \nabla \mathcal{L}(\hat{\mathbf{z}}, \theta)$

#### 2. Fisher Information Matrix Computation

A diagonal approximation of the FIM is used to reduce computational cost:

$$({\mathbf{F}_{\text{diag}}})_{jj} \approx \frac{1}{N} \sum_{i=1}^N \frac{1}{T} \sum_{t=1}^T \left(\frac{\partial \mathcal{L}_t(\mathbf{z}_i, \theta)}{\partial \theta_j}\right)^2$$

In diffusion models, the loss depends on the denoising timestep $t$; averaging over multiple timesteps is therefore applied.

#### 3. Silence Masking Strategy

Music generation models handle variable-length audio by zero-padding short segments. Three masking schemes are proposed:
- **No masking**: neither unlearning nor attribution computation applies masking → short tracks are distorted by zero-padding, leading to inaccurate rankings
- **Dual masking** ($M_U + M_L$): both steps apply masking → attribution rankings improve, but very short tracks rank anomalously high
- **Mixed strategy** ($M_U$ only): masking applied during unlearning but not during loss computation → best overall performance

**Design Motivation**: Masking during unlearning ensures that zero-padded regions do not interfere with the forgetting process; not masking during loss computation maintains consistency with the training setup and avoids unpredictable model behavior.

### Loss & Training

- Model: Latent DiT (based on Stable Audio); VAE encodes 44.1 kHz stereo audio into a 64-dimensional latent space
- Diffusion process: v-objective, supporting up to ~2 minutes of audio (2,584 latent frames)
- Conditioning: CLAP embeddings (text-to-music) + timing conditions (variable-length generation)
- FIM computation: gradients averaged over 2,048 random timesteps per unlearning step
- Single unlearning step: ~20 minutes (NVIDIA H100); full training-set loss computation: ~5 hours (8 × H100)

## Key Experimental Results

### Main Results: Self-Influence Experiment (Train-to-Train)

40 diverse training samples selected via k-means clustering on CLAP embeddings for grid search:

| Target Layers | $M_U$ | $M_L$ | $R(\mathbf{z}_{tar})$ | $\text{CLAP}_{topk}$ | $\text{CLAP}_{botk}$ | $\text{FD}_{openl3}$ |
|--------|-------|-------|-----|------|------|------|
| Cross-Attention to_kv | ✓ | - | 103.2 | 0.38 | 0.35 | 110.5 |
| Cross-Attention Layers | ✓ | - | 1.4 | 0.60 | 0.32 | 110.4 |
| Self-Attention Layers | ✓ | - | 1.1 | 0.63 | 0.30 | 110.5 |
| **All Transformer Layers** | **✓** | **✓** | **1.0** | **0.80** | **0.38** | **110.5** |
| All Transformer Layers | - | - | 6615.7 | 0.82 | 0.42 | 110.5 |
| **All Transformer Layers (Mixed)** | **✓** | **-** | **1.0** | **0.66** | **0.26** | **110.5** |

- Learning rate $10^{-6}$ with 1 update step is the optimal configuration
- $R(\mathbf{z}_{tar}) = 1.0$ indicates the target sample ranks first in attribution (unlearning successful)
- $\text{FD}_{openl3}$ remains unchanged, confirming that unlearning does not degrade overall generation quality

### Ablation Study: Comparison with Non-Counterfactual Methods (Test-to-Train)

16 two-minute audio tracks are generated and five attribution methods are compared:

| Method | Type | Pearson Correlation with Unlearning |
|------|------|------|
| LPIPS | White-box (model internal activation similarity) | **0.56** |
| CLAP | Black-box (audio embedding similarity) | 0.46 |
| CLEWS | Black-box (music identity embedding) | 0.32 |
| RPS (Representer Point) | White-box (gradient information) | 0.11 |

### Key Findings

1. **Higher attribution concentration in unlearning**: attribution score distributions exhibit sharply concentrated patterns, with influence concentrated in a very small number of training samples
2. **Ranking consistency across methods**: highest correlation with LPIPS (also white-box and utilizing model-internal information), validating the consistency of internal representations
3. **Model-internal vs. external information**: methods leveraging model-internal information (Unlearning, LPIPS) exhibit high mutual correlation; external embedding methods (CLAP, CLEWS) also correlate well with each other, with moderate cross-group correlation
4. **RPS captures distinct patterns**: RPS shows low correlation with all other methods, suggesting it captures a qualitatively different attribution signal
5. **Overall model performance unaffected**: $\text{FD}_{openl3}$ remains at 110.5 (original value) after unlearning, confirming the effectiveness of regularization

## Highlights & Insights

- **Domain pioneering**: the first exploration of unlearning-based TDA on a large-scale text-to-music DiT, addressing real-world scale (115K tracks, 4,356 hours) and stylistic diversity
- **Mixed masking strategy** elegantly resolves zero-padding interference in variable-length audio processing—excluding irrelevant silence during unlearning while maintaining training consistency during evaluation
- **Rigorous experimental design**: self-influence experiments first validate method correctness (whether the target sample is correctly identified) before proceeding to real attribution analysis
- **Contribution to AI ethics**: provides a technical foundation for copyright attribution and creator compensation in the music AI domain

## Limitations & Future Work

1. **High computational cost**: each unlearning step requires ~20 min (H100) and full dataset loss computation requires ~5h (8×H100), making large-scale deployment challenging
2. **Single-step unlearning only**: grid search identifies 1 step as optimal, but multi-step unlearning may theoretically improve precision and warrants further investigation
3. **Diagonal FIM approximation**: inter-parameter correlations are discarded, potentially affecting attribution accuracy
4. **Proprietary validation data**: the 115K dataset is not publicly available, limiting full reproducibility
5. **Absence of human evaluation**: the musical relevance of attribution results has not been assessed by professional musicians
6. **Single architecture evaluated**: the method has not been validated on autoregressive models or other music generation architectures

## Related Work & Insights

- **Complementarity with influence function methods**: Deng et al. apply influence functions to small-scale piano data; this paper extends unlearning-based methods to large-scale, multi-style settings—the two approaches can serve as mutual validation
- **Domain transfer of general TDA methods**: the FIM-regularized unlearning approach of Wang et al., originally designed for image classification, is successfully adapted to diffusion-based audio generation
- **Potential impact on music copyright frameworks**: if TDA becomes sufficiently accurate and computationally efficient, it could enable automated royalty distribution systems

## Rating
- Novelty: ⭐⭐⭐⭐ — First application of unlearning-based TDA to large-scale music generation; mixed masking strategy is novel
- Experimental Thoroughness: ⭐⭐⭐ — Self-influence validation is rigorous, but test scale is limited (40 + 16 samples) and human evaluation is absent
- Writing Quality: ⭐⭐⭐⭐ — Method derivations are clear, experimental design is well-structured, and figures are highly informative
- Value: ⭐⭐⭐⭐ — Opens an important direction for music AI ethics and copyright attribution

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Fast Data Attribution for Text-to-Image Models](fast_data_attribution_for_text-to-image_models.md)
- [\[NeurIPS 2025\] Hephaestus: Mixture Generative Modeling with Energy Guidance for Large-scale QoS Degradation](hephaestus_mixture_generative_modeling_with_energy_guidance_for_large-scale_qos_.md)
- [\[NeurIPS 2025\] UltraHR-100K: Enhancing UHR Image Synthesis with A Large-Scale High-Quality Dataset](ultrahr-100k_enhancing_uhr_image_synthesis_with_a_large-scale_high-quality_datas.md)
- [\[NeurIPS 2025\] ICEdit: Enabling Instructional Image Editing with In-Context Generation in Large Scale Diffusion Transformer](in-context_edit_enabling_instructional_image_editing_with_in-context_generation_.md)
- [\[NeurIPS 2025\] UtilGen: Utility-Centric Generative Data Augmentation with Dual-Level Task Adaptation](utilgen_utility-centric_generative_data_augmentation_with_dual-level_task_adapta.md)

<!-- RELATED:END -->
