---
title: >-
  [Paper Note] Diffusion Probe: Generated Image Result Prediction Using CNN Probes
description: >-
  [CVPR 2026][Image Generation][Diffusion Models] This work discovers that the cross-attention distribution in early denoising steps of diffusion models is highly correlated with final image quality. It proposes Diffusion…
tags:
  - "CVPR 2026"
  - "Image Generation"
  - "Diffusion Models"
  - "Probe"
  - "Cross-Attention"
  - "Early Quality Prediction"
  - "Generation Acceleration"
date: 2026-05-08
content_hash: 6cb0780ae8b8e6e4
---

# Diffusion Probe: Generated Image Result Prediction Using CNN Probes

**Conference**: CVPR 2026
**arXiv**: [2602.23783](https://arxiv.org/abs/2602.23783)  
**Code**: None  
**Area**: Diffusion Models / Image Quality Prediction
**Keywords**: Diffusion Models, Probe, Cross-Attention, Early Quality Prediction, Generation Acceleration

## TL;DR

This work discovers that the cross-attention distribution in early denoising steps of diffusion models is highly correlated with final image quality. It proposes Diffusion Probe — a lightweight CNN that predicts generation quality from early attention maps — enabling pre-filtering of low-quality generation trajectories after only 10% of denoising steps, thereby accelerating prompt optimization, seed selection, and GRPO training.

## Background & Motivation

### State of the Field

**Background**: Text-to-image (T2I) diffusion models face a fundamental efficiency bottleneck: quality is unpredictable.

### Limitations of Prior Work

**Limitations of Prior Work**: Users often require multiple attempts (varying prompts or seeds) to obtain satisfactory results, with each attempt requiring a complete denoising pass.

### Root Cause

**Key Challenge**: Academic methods such as IC-Edit (repeated generation) and Flow-GRPO (multi-candidate ranking) similarly depend on full generation.

### Starting Point

**Key Insight**: Existing early prediction methods either incur high computational overhead (ICEdit requires a 72B VLM for decoding) or cannot be automated (PromptCharm relies on manual interpretation of attention maps).

Core finding: Early cross-attention maps in diffusion models contain predictive signals for final image quality — tokens whose attention is dispersed or fragmented correspond to objects that are missing or distorted in the final image.

## Method

### Overall Architecture

At an early denoising step (default $t=5$ out of 25 total steps), intermediate cross-attention maps $\mathcal{A}$ are extracted and, together with timestep embeddings, fed into a lightweight CNN probe network $E_\theta$, which outputs a scalar quality score prediction $\hat{q} = f_\theta(E_\theta(\mathcal{A}, t))$. The probe is trained offline on a dataset using an MSE regression loss. The predicted score is used for early-exit decisions in downstream tasks.

### Key Designs

1. **Empirical Discovery of Attention–Quality Mapping**: A systematic audit of the FLUX model reveals two key phenomena: (a) even during high-noise denoising stages, tokens corresponding to semantically salient objects induce sharp, localized high-attention regions — early object localization emerges rapidly; (b) when final image quality is poor (missing objects, distortions, semantic inconsistencies), the early attention maps of the corresponding tokens are visibly diffuse and fragmented. These two observations establish the theoretical basis for using cross-attention as a probe for generation trajectories.

2. **Lightweight CNN Probe Architecture**: The probe consists of multiple DownBlocks (with residual layers) and an OutputLayer (normalization + pooling + convolution). For UNet-based models (e.g., SDXL), attention maps are extracted from the last 10 encoder blocks; for DiT-based models (e.g., FLUX), they are extracted from 10 consecutive intermediate blocks. The architecture is extremely lightweight and does not modify any parameters of the base model — it is fully plug-and-play. The training objective is an MSE loss:
$$\mathcal{L} = \|\hat{q} - q\|_2^2$$
where $q$ is the quality score assigned by a pretrained reward model (e.g., ImageReward) on fully generated images.

3. **Three Downstream Applications**: (a) **Prompt Optimization**: An LLM is invoked to rewrite the prompt only when the predicted score falls below a threshold $\tau$, avoiding unnecessary LLM overhead; (b) **Seed Selection**: For a pool of candidate seeds, only $T_0 \ll T$ denoising steps are run, after which the probe predicts quality and the best seed is selected for a single full generation; (c) **Efficient Flow-GRPO Training**: Probe predictions replace full generation to rapidly mine preference pairs $(x^+, x^-)$, significantly accelerating policy convergence. The core logic across all three applications is consistent: replace costly full-generation evaluation with cheap early-stage prediction.

### Loss & Training

- MSE regression loss, with labels from the ImageReward pretrained reward model
- Training data: 15K prompts (MS-COCO); evaluation: 5K disjoint prompts
- Low-score samples are oversampled to address class imbalance
- Default extraction step: $t=5$ (the 5th of 25 steps); prediction accuracy improves most sharply from $t=1$ to $t=5$, with diminishing returns thereafter

## Key Experimental Results

### Main Results (Prediction Accuracy, 1024×1024 Resolution)

| Base Model | Steps | SRCC↑ | AUC-ROC↑ | KTC↑ | PCC↑ |
|------------|-------|-------|----------|------|------|
| SDXL | 5 | 0.73 | 0.86 | 0.57 | 0.72 |
| SDXL | 10 | 0.76 | 0.89 | 0.61 | 0.75 |
| FLUX | 5 | 0.76 | 0.88 | 0.60 | 0.75 |
| FLUX | 10 | **0.79** | **0.91** | **0.64** | **0.78** |
| Qwen-Image | 10 | 0.72 | 0.87 | 0.56 | 0.71 |

Strong consistency is observed across architectures (UNet/DiT); AUC > 0.9 indicates excellent discriminative performance for binary quality classification.

### Ablation Study (Downstream Task Performance)

| Model | Task | Method | CLIP Score↑ | ImageReward↑ | Aesthetic↑ |
|-------|------|--------|-------------|-------------|------------|
| SDXL | Prompt Optimization | Baseline | 28.31 | 0.71 | 5.13 |
| SDXL | Prompt Optimization | **+Probe** | 30.24 | 0.72 | 5.29 |
| SDXL | Prompt Optimization | +LLM | 30.80 | 0.73 | 5.34 |
| FLUX | Seed Selection | Random | 31.37 | 1.02 | 5.67 |
| FLUX | Seed Selection | **+Probe** | 31.41 | 1.06 | **5.79** |

The probe approaches LLM-level performance on prompt optimization at substantially lower cost; seed selection yields notable improvements in aesthetic scores.

### Key Findings

- Probe prediction accuracy reaches near-peak performance at step 5 (PCC 0.75 vs. peak 0.78), using only 20% of total denoising steps
- Consistent performance across three distinct architectures (UNet/DiT) validates the model-agnostic nature of the approach
- In Flow-GRPO training, the probe increases the proportion of high-quality samples by 2.5×, yielding smoother convergence curves
- The degree of attention map diffuseness directly corresponds to object rendering failure modes (missing, distorted, or attribute-misaligned objects)

## Highlights & Insights

- This work is the first to introduce the LLM probing paradigm into diffusion models — "diagnosing generation trajectories with probes" represents an entirely novel perspective
- The core insight is elegant: concentrated early attention implies correct object rendering; diffuse early attention implies generation failure
- The probe is fully decoupled from the base model, zero-invasive, and applicable to any T2I model with attention mechanisms
- Practical application scenarios are diverse and practically relevant (prompt iteration, seed filtering, RL training acceleration)

## Limitations & Future Work

- Prediction has only been validated for the ImageReward metric; predictive capacity for other quality dimensions (e.g., text rendering quality, spatial relationships) remains to be evaluated
- The probe must be trained separately for each base model; generalization to new models requires re-collecting training data
- The extraction step $t$ and block layer are currently selected manually, which may be suboptimal for different models
- The MSE loss is insensitive to ranking; ranking losses or contrastive learning could be explored
- For prompts that are extremely simple or highly complex, attention patterns may not carry equivalent predictive power

## Related Work & Insights

- Distinction from DAAM (attention attribution visualization): DAAM performs post-hoc analysis, whereas Diffusion Probe performs prediction
- Complementary to attention manipulation methods such as Attend-and-Excite — the latter improves attention, while the former predicts its outcome
- Compared to ICEdit's early prediction: ICEdit requires decoding with a VLM (72B parameters), whereas the probe requires only a lightweight CNN
- The probing paradigm is well-established in NLP (probes predicting linguistic properties); extending it to visual generation is a natural progression
- The approach can be combined with acceleration methods such as DPCache — early filtering followed by accelerated generation

## Rating

- Novelty: ⭐⭐⭐⭐ First application of probing to diffusion models; the attention–quality correlation is a valuable finding
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated across three models × multiple step counts × three downstream tasks
- Writing Quality: ⭐⭐⭐⭐ Motivation is clear, visualizations are intuitive, and experimental organization is well-structured
- Value: ⭐⭐⭐⭐ High practical value, especially for scenarios requiring extensive sampling (RL training, agent-based generation)
- Value: Pending

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Diversity over Uniformity: Rethinking Representation in Generated Image Detection](diversity_over_uniformity_rethinking_representation_in_generated_image_detection.md)
- [\[CVPR 2026\] Attribution as Retrieval: Model-Agnostic AI-Generated Image Attribution](attribution_as_retrieval_modelagnostic_aigenerated.md)
- [\[CVPR 2026\] Markovian Scale Prediction: A New Era of Visual Autoregressive Generation](markovian_scale_prediction_a_new_era_of_visual_autoregressive_generation.md)
- [\[ICML 2026\] Latent Diffusion Pretraining for Crystal Property Prediction](../../ICML2026/image_generation/latent_diffusion_pretraining_for_crystal_property_prediction.md)
- [\[ICLR 2026\] From Prediction to Perfection: Introducing Refinement to Autoregressive Image Generation](../../ICLR2026/image_generation/from_prediction_to_perfection_introducing_refinement_to_autoregressive_image_gen.md)

</div>

<!-- RELATED:END -->
