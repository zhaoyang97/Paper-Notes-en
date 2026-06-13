---
title: >-
  [Paper Note] SeMoBridge: Semantic Modality Bridge for Efficient Few-Shot Adaptation of CLIP
description: >-
  [ICLR 2026][Image Generation][CLIP adaptation] SeMoBridge is proposed as a lightweight semantic modality bridge that maps image embeddings into the text modality…
tags:
  - "ICLR 2026"
  - "Image Generation"
  - "CLIP adaptation"
  - "modality gap"
  - "intra-modal misalignment"
  - "few-shot classification"
  - "pseudo EOS token"
date: 2026-05-08
content_hash: 61eb860e3c1e7473
---

# SeMoBridge: Semantic Modality Bridge for Efficient Few-Shot Adaptation of CLIP

**Conference**: ICLR 2026
**arXiv**: [2509.26036](https://arxiv.org/abs/2509.26036)  
**Code**: [https://github.com/christti98/semobridge](https://github.com/christti98/semobridge)  
**Area**: Few-Shot Learning / Vision-Language Models
**Keywords**: CLIP adaptation, modality gap, intra-modal misalignment, few-shot classification, pseudo EOS token

## TL;DR

SeMoBridge is proposed as a lightweight semantic modality bridge that maps image embeddings into the text modality, converting unreliable intra-modal (image-to-image) comparisons into reliable inter-modal (text-to-image) comparisons, achieving state-of-the-art few-shot classification performance with minimal training overhead.

## Background & Motivation

CLIP aligns image and text representations into a shared embedding space via contrastive learning, demonstrating strong zero-shot performance. However, **intra-modal misalignment** arises in few-shot classification:

- CLIP exhibits an inherent **modality gap**—a systematic separation between image and text embeddings.
- The contrastive training objective focuses solely on cross-modal alignment, leaving the semantic structure within each modality uncalibrated.
- Consequently, query images may be incorrectly placed closer to the few-shot centroids of wrong classes.

Limitations of prior work:
- Methods such as Tip-X and APE operate at the level of logit scores, failing to fully exploit CLIP's inter-modal semantic priors.
- Cross the Gap addresses the issue through per-sample optimization, but incurs prohibitive computational cost.

## Method

### Core Idea

Image embeddings are mapped into the text modality while preserving semantic content, thereby transforming unreliable image-to-image comparisons into reliable image-to-text inter-modal comparisons.

### Key Design 1: Pseudo EOS Token Derivation

The directional alignment guaranteed by CLIP's training objective is exploited:

$$\frac{\mathbf{f}_{\text{img}}}{\|\mathbf{f}_{\text{img}}\|} \approx \frac{\hat{\mathbf{f}}_{\text{txt}}}{\|\hat{\mathbf{f}}_{\text{txt}}\|}$$

Back-projection is performed via the Moore-Penrose pseudoinverse of the text projection matrix, followed by rescaling:

$$\hat{\mathbf{f}}_{\text{eos}} \approx \frac{\|\mathbf{T}_{\text{eos}}\|}{\|\mathbf{W}_{\text{txt}}^+ \mathbf{f}_{\text{img}}\|} \mathbf{W}_{\text{txt}}^+ \mathbf{f}_{\text{img}}$$

The final bridged embedding is:

$$\hat{\mathbf{f}}_{\text{txt}} = \mathbf{W}_{\text{txt}} \hat{\mathbf{f}}_{\text{eos}} \approx \frac{\|\mathbf{T}_{\text{eos}}\|}{\|\mathbf{W}_{\text{txt}}^+ \mathbf{f}_{\text{img}}\|} \mathbf{f}_{\text{img}}$$

Since $\mathbf{W}_{\text{txt}}\mathbf{W}_{\text{txt}}^+$ approximates the identity matrix, the transformation reduces to a scaling of the original image embedding.

### Key Design 2: Triple Logit Score Fusion

$$\mathbf{z}_q = \lambda_1 \mathbf{z}_1 + \lambda_2 \mathbf{z}_2 + \lambda_3 \mathbf{z}_3$$

- $\mathbf{z}_1$: zero-shot prior (query image vs. class text prompts)
- $\mathbf{z}_2$: original few-shot vs. bridged query (bridged query compared to few-shot images in text space)
- $\mathbf{z}_3$: original query vs. bridged few-shot (inverted signal for enhanced robustness)

### Key Design 3: Multimodal Supervised Training (SeMoBridge-T)

Class-Specific Bias (CSB) $\hat{\boldsymbol{\tau}} \in \mathbb{R}^{C \times d_t}$ is introduced:

$$\hat{\mathbf{F}}_{\text{eos}}^c \approx \frac{\|\mathbf{T}_{\text{eos}}\|}{\|\hat{\mathbf{W}}_{\text{txt}}^+ \mathbf{F}_{\text{img}}^c + \hat{\boldsymbol{\tau}}^c\|} (\hat{\mathbf{W}}_{\text{txt}}^+ \mathbf{F}_{\text{img}}^c + \hat{\boldsymbol{\tau}}^c)$$

The multimodal loss is:

$$\mathcal{L} = \lambda_{\text{it}} \mathcal{L}_{\text{img}} + (1-\lambda_{\text{it}})\frac{\mathcal{L}_{\text{txte}} + \mathcal{L}_{\text{txtp}}}{2} + \lambda_c \mathcal{L}_{\text{cons}} + \lambda_b \mathcal{L}_{\text{bias}}$$

- $\mathcal{L}_{\text{img}}$: alignment between bridged and original image embeddings
- $\mathcal{L}_{\text{txte}}, \mathcal{L}_{\text{txtp}}$: alignment with class description EOS tokens and projections
- $\mathcal{L}_{\text{cons}}$: intra-class few-shot consistency
- $\mathcal{L}_{\text{bias}}$: CSB regularization

Only the bridge parameters are updated during training; CLIP is fully frozen.

## Experiments

### Training Efficiency Comparison

| Method | Parameters | Avg. Training Time | Avg. Accuracy |
|---|---|---|---|
| CoOp | 0.01M | 10h 00min | 63.90% |
| PromptSRC | 0.05M | 1h 42min | 77.90% |
| APE-T | 0.51M | 3min 30s | 77.18% |
| LDC | 69M | 2min | 77.17% |
| **SeMoBridge-T** | **0.77M** | **27s** | **78.15%** |

SeMoBridge-T achieves the highest accuracy with only 27 seconds of training.

### Few-Shot Classification Results

- **Training-free SeMoBridge** outperforms APE on 7 out of 11 datasets.
- **SeMoBridge-T** achieves the best overall performance in low-shot settings (1/2/4-shot).
- The most significant improvements are observed on datasets with visually similar categories, such as OxfordPets.

### Out-of-Distribution Generalization (16-shot ImageNet)

| Method | ImageNet | ImageNet-V2 | ImageNet-Sketch |
|---|---|---|---|
| APE | 71.81 | 64.81 | 49.95 |
| SeMoBridge | 71.86 | 64.90 | 49.55 |
| APE-T | 74.13 | 66.21 | 49.73 |
| SeMoBridge-T | Competitive with APE-T | — | — |

### Key Findings

- Intra-modal misalignment is the primary cause of CLIP's failure in few-shot settings.
- A simple modality bridge (scaling + projection) is sufficient to effectively address this issue.
- CSB benefits datasets with large numbers of categories such as ImageNet, but is less critical for smaller datasets.
- A fixed 1:1 balance between image and text losses in the multimodal objective is adequate.

## Highlights & Insights

- The method is remarkably concise and elegant—the core computation reduces to a pseudoinverse matrix multiplication.
- Training time is extremely short (27 seconds), an order of magnitude faster than the next best method.
- The training-free variant is already competitive, with the trained variant yielding further gains.
- The theoretical motivation is well-grounded, deriving the bridge directly from CLIP's training objective.
- The advantage is most pronounced in low-shot regimes (1/2/4-shot).

## Limitations & Future Work

- The approximation that $\mathbf{W}_{\text{txt}}\mathbf{W}_{\text{txt}}^+$ is close to the identity matrix may not hold for certain CLIP variants.
- CSB is applied during training but not to queries at inference, potentially introducing a train-inference distribution mismatch.
- Evaluation is primarily conducted on ViT-B/16; the effectiveness on larger models remains underexplored.
- The weights for triple logit fusion require validation-set tuning.

## Related Work & Insights

- **CLIP Adaptation**: prompt/adapter-based methods including CoOp, Tip-Adapter, and APE.
- **Modality Gap Research**: the modality gap in CLIP's embedding space as identified by Liang et al.
- **Modality Inversion**: per-sample optimization approaches (OTI/OVI) and closed-form projection (SD-IPC).

## Rating

- Novelty: ⭐⭐⭐⭐ — The modality bridge concept and its adaptation from SD-IPC are elegant.
- Simplicity: ⭐⭐⭐⭐⭐ — The method is clean, intuitive, and easy to reproduce.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers 11 datasets, training efficiency comparisons, and OOD generalization.
- Value: ⭐⭐⭐⭐⭐ — 27-second training with an extremely low barrier to adoption.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Uni-DAD: Unified Distillation and Adaptation of Diffusion Models for Few-step Few-shot Image Generation](../../CVPR2026/image_generation/uni-dad_unified_distillation_and_adaptation_of_diffusion_models_for_few-step_few.md)
- [\[CVPR 2026\] V-Bridge: Bridging Video Generative Priors to Versatile Few-shot Image Restoration](../../CVPR2026/image_generation/v-bridge_bridging_video_generative_priors_to_versatile_few-shot_image_restoratio.md)
- [\[ICLR 2026\] When One Modality Rules Them All: Backdoor Modality Collapse in Multimodal Diffusion Models](when_one_modality_rules_them_all_backdoor_modality_collapse_in_multimodal_diffus.md)
- [\[CVPR 2026\] Few-shot Acoustic Synthesis with Multimodal Flow Matching](../../CVPR2026/image_generation/few-shot_acoustic_synthesis_with_multimodal_flow_matching.md)
- [\[ICML 2026\] Envisioning Beyond the Few: Disentangled Semantics and Primitives for Few-Shot Atypical Layout-to-Image Generation](../../ICML2026/image_generation/envisioning_beyond_the_few_disentangled_semantics_and_primitives_for_few-shot_at.md)

</div>

<!-- RELATED:END -->
