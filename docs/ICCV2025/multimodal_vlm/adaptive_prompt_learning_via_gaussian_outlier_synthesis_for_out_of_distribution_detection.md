---
title: >-
  [Paper Note] Adaptive Prompt Learning via Gaussian Outlier Synthesis for Out-of-distribution Detection
description: >-
  [ICCV 2025][Multimodal VLM][OOD detection] This paper proposes APLGOS, a framework that leverages prompt learning in vision-language models to synthesize virtual OOD prompts and images by sampling from low-probability re…
tags:
  - "ICCV 2025"
  - "Multimodal VLM"
  - "OOD detection"
  - "prompt learning"
  - "Gaussian outlier synthesis"
  - "vision-language model"
  - "contrastive learning"
date: 2026-05-08
content_hash: 269e3366fa4e4b1a
---

# Adaptive Prompt Learning via Gaussian Outlier Synthesis for Out-of-distribution Detection

**Conference**: ICCV 2025
**Code**: N/A  
**Area**: Multimodal VLM
**Keywords**: OOD detection, prompt learning, Gaussian outlier synthesis, vision-language model, contrastive learning

## TL;DR

This paper proposes APLGOS, a framework that leverages prompt learning in vision-language models to synthesize virtual OOD prompts and images by sampling from low-probability regions of class-conditional Gaussian distributions, thereby enforcing more compact decision boundaries between in-distribution (ID) and out-of-distribution (OOD) categories. The method achieves state-of-the-art performance on four mainstream benchmarks.

## Background & Motivation

OOD detection aims to enable a detector to distinguish at test time between categories seen during training (ID) and unseen categories (OOD). Existing methods predominantly extract pseudo-OOD samples from ID data to regularize model decision boundaries, but suffer from two fundamental issues: (1) the quality of pseudo-OOD samples derived from ID data is difficult to control, making it hard to adequately cover the true OOD distribution; and (2) large amounts of ID data are required for effective training. Synthesis-based approaches (e.g., GAN-based generation, VOS virtual outliers) partially alleviate these issues but remain limited. The authors observe that vision-language models (VLMs) possess rich pre-trained knowledge and strong representational capacity that can facilitate higher-quality virtual outlier generation; however, no prior work has applied prompt learning to OOD detection.

## Method

### Overall Architecture

APLGOS consists of two main modules: a Prompt Learning Module (PLM) and a Text-Image Alignment Module (TAM). PLM is responsible for generating ID prompts and synthesizing OOD pseudo-prompts, while TAM aligns multimodal data by computing image-prompt similarities via contrastive learning. Training proceeds in three stages: Stage 1 learns ID prompts and alignment; Stage 2 introduces synthesized OOD prompts; Stage 3 further incorporates synthesized OOD images. All OOD data (prompts and images) are virtually synthesized, with only ID images drawn from real datasets.

### Key Designs

1. **ID Prompt Generation (ChatGPT-Normalized Q&A)**: A predefined Q&A template incorporating location coordinates and category names is used (e.g., "Q: What is in the region with coordinates <loc1>,<loc2>,<loc3>,<loc4>? A: That's a <CLS>."). ChatGPT is employed in multi-turn interactions to normalize and diversify the sentence set. During training, learnable ID prompts are initialized by randomly sampling from this set, and location tokens are introduced to enable finer-grained region-level prompting.

2. **OOD Prompt Synthesis (Gaussian Outlier Sampling)**: The ID prompt embeddings in the latent space are assumed to follow a class-conditional multivariate Gaussian distribution. Per-class empirical means and tied covariance matrices are computed, and virtual OOD prompts are sampled from the low-probability ($\varepsilon$-likelihood) regions of these Gaussian distributions. A learnable Gaussian noise matrix $\varepsilon$ is introduced to expand the sampling space and prevent over-reliance on ID class distributions. OOD images are synthesized analogously by sampling from low-probability regions in the image embedding space.

3. **Text-Image Alignment Module (TAM)**: Normalized similarity scores between image and prompt embeddings are computed via contrastive learning. An alignment loss $\mathcal{L}_{\text{align}}$ is combined with a location loss $\mathcal{L}_{\text{loc}}$ (which implicitly incorporates coordinate information), a classification loss $\mathcal{L}_{\text{cls}}$, and a regularization term to constrain the decision boundaries for both ID and OOD data.

### Loss & Training

The total loss is defined as:

$$\mathcal{L} = \xi_1[\gamma_1\tau\mathcal{L}_{\text{align}}^{\text{ID}} + \gamma_2(1-\tau)\mathcal{L}_{\text{align}}^{\text{OOD}}] + \gamma_3\xi_2[\kappa\mathcal{L}_{\text{loc}}^{\text{ID}} + (1-\kappa)\mathcal{L}_{\text{loc}}^{\text{OOD}}] + \gamma_4\xi_3\mathcal{L}_{\text{cls}} + \gamma_5\xi_4\mathcal{L}_{\text{reg}} + W$$

where $\xi$, $\tau$, and $\kappa$ control which loss components are active at each training stage. The three-stage training strategy progressively introduces ID alignment, OOD prompt alignment, and virtual OOD images. The ID-to-OOD data ratio is approximately 1:1, and the number of sampled OOD prompts is set to $K = 10{,}000$.

## Key Experimental Results

### Main Results

| ID Dataset | OOD Dataset | Metric | APLGOS (RegX4.0) | Prev. SOTA VOS (RegX4.0) | Gain |
|---|---|---|---|---|---|
| PASCAL VOC | MS-COCO | FPR95↓ | 45.96% | 50.53% | −4.57% |
| PASCAL VOC | OpenImages | FPR95↓ | 47.10% | 50.27% | −3.17% |
| BDD-100k | MS-COCO | FPR95↓ | 39.48% | 42.82% | −3.34% |
| BDD-100k | OpenImages | FPR95↓ | 19.79% | 27.55% | −7.76% |
| PASCAL VOC | — | mAP↑ | 49.4% | 49.1% | +0.3% |

The largest gain is observed on the BDD-100k + OpenImages combination, where FPR95 is reduced by 7.76%.

### Ablation Study

- **Prompt strategy**: The combination of location tokens `<LOC>` and ChatGPT-normalized random prompts (RP) yields the best performance, reducing FPR95 from 50.53% to 45.96%.
- **OOD prompt sample count $K$**: $K = 10{,}000$ is optimal; smaller values fail to adequately cover the boundary region, while larger values introduce excessive randomness.
- **Gaussian noise intensity $\alpha$**: $\alpha = 1.0$ is optimal; smaller values overly restrict the sampling space, while larger values excessively broaden it.
- **ID/OOD ratio**: A 1:1 ratio is optimal; deviations in either direction degrade performance.

### Key Findings

- Synthesizing OOD prompts in the latent space is more effective than extracting pseudo-OOD samples directly from ID data.
- Location information is critical for region-level OOD detection; introducing coordinate tokens yields significant performance improvements.
- The proposed method maintains superior performance even when limited ID data is available.

## Highlights & Insights

- This is the first work to introduce prompt learning into OOD detection, cleverly leveraging VLM knowledge to produce higher-quality OOD synthesis.
- ID prompts, OOD prompts, and OOD images are all virtually generated, reducing dependence on real OOD data.
- The ChatGPT-normalized Q&A strategy for diverse prompt generation is both novel and practically effective.
- The low-probability region sampling strategy under Gaussian distributions is intuitive and grounded in solid theoretical foundations.

## Limitations & Future Work

- The method relies on ChatGPT to generate the prompt set, introducing additional preprocessing costs.
- The Gaussian distribution assumption may not hold for all feature distributions encountered in practice.
- Evaluation is limited to the object detection setting; extension to image classification, semantic segmentation, and other tasks is not explored.
- The three-stage training pipeline is relatively complex, with numerous hyperparameters ($\alpha$, $\beta$, $\gamma$, $\xi$, $K$, etc.).

## Related Work & Insights

- VOS [Du et al.] is the primary baseline, synthesizing virtual outliers in the feature space.
- Prompt learning methods such as CoOp and CoCoOp provide the foundation for the prompt design in this work.
- The proposed approach is transferable to OOD detection in other safety-critical applications.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First to combine prompt learning with Gaussian outlier synthesis for OOD detection.
- **Technical Depth**: ⭐⭐⭐⭐ — The prompt synthesis and alignment framework is well-designed and technically sound.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Four datasets, comprehensive ablations, and visualization analyses.
- **Writing Quality**: ⭐⭐⭐⭐ — Well-organized with effective integration of figures and text.
- **Value**: ⭐⭐⭐ — Relatively high complexity poses a barrier to practical deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] FA: Forced Prompt Learning of Vision-Language Models for Out-of-Distribution Detection](fa_forced_prompt_learning_of_vision-language_models_for_out-of-distribution_dete.md)
- [\[ICCV 2025\] Exploiting Vision Language Model for Training-Free 3D Point Cloud OOD Detection](exploiting_vision_language_model_for_training-free_3d_point_cloud_ood_detection_.md)
- [\[ICCV 2025\] PRO-VPT: Distribution-Adaptive Visual Prompt Tuning via Prompt Relocation](pro-vpt_distribution-adaptive_visual_prompt_tuning_via_prompt_relocation.md)
- [\[ICCV 2025\] Advancing Textual Prompt Learning with Anchored Attributes](advancing_textual_prompt_learning_with_anchored_attributes.md)
- [\[ICCV 2025\] NegRefine: Refining Negative Label-Based Zero-Shot OOD Detection](negrefine_refining_negative_label-based_zero-shot_ood_detection.md)

</div>

<!-- RELATED:END -->
