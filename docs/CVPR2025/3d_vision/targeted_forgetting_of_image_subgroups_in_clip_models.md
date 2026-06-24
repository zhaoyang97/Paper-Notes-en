---
title: >-
  [Paper Note] Targeted Forgetting of Image Subgroups in CLIP Models
description: >-
  [CVPR 2025][3D Vision][CLIP unlearning] A three-stage CLIP subgroup image forgetting framework (forgetting → reminding → restoring) is proposed. It selects key layers for LoRA fine-tuning using relative Fisher Information, aligns the retain data distribution utilizing BatchNorm statistics, and restores zero-shot capability via model souping, achieving precise subgroup forgetting (target ↓ to 0%) on ImageNet-1K and CIFAR-10 while maintaining 85-93% of the overall score.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "CLIP unlearning"
  - "subgroup image forgetting"
  - "relative Fisher Information"
  - "model merging"
  - "knowledge distillation"
date: 2026-05-08
content_hash: 36e3d5f577ff100a
---

# Targeted Forgetting of Image Subgroups in CLIP Models

**Conference**: CVPR 2025  
**arXiv**: [2506.03117](https://arxiv.org/abs/2506.03117)  
**Code**: None  
**Area**: Model Compression / Machine Unlearning  
**Keywords**: CLIP unlearning, subgroup image forgetting, relative Fisher Information, model merging, knowledge distillation

## TL;DR
A three-stage CLIP subgroup image forgetting framework (forgetting → reminding → restoring) is proposed. It selects key layers for LoRA fine-tuning using relative Fisher Information, aligns the retain data distribution utilizing BatchNorm statistics, and restores zero-shot capability via model souping, achieving precise subgroup forgetting (target ↓ to 0%) on ImageNet-1K and CIFAR-10 while maintaining 85-93% of the overall score.

## Background & Motivation

**Background**: Foundation models like CLIP are pre-trained on large-scale web data such as LAION-5B, possessing powerful zero-shot classification capabilities. However, the training data contains harmful content (discriminatory images, copyright violations, personal information, etc.), causing such problematic knowledge to be inevitably encoded into the model parameters.

**Limitations of Prior Work**: Existing unlearning methods face three core challenges: (1) Inaccessibility of pre-training data—the complete LAION-5B dataset cannot be obtained, making traditional unlearning methods inapplicable; (2) Over-forgetting due to coarse-grained labels—users might only want to forget "Boeing airplanes" rather than all "airplanes," but coarse labels cannot distinguish subgroups; (3) Distribution shift—there is a significant distribution gap between the forget/retain data and the pre-training data, where direct fine-tuning leads to catastrophic forgetting.

**Key Challenge**: The need to precisely forget specific subgroups within the same class (e.g., forgetting "marmoset" while retaining other "monkeys") without accessing original pre-training data, while maintaining the zero-shot generalizability of CLIP on all other datasets.

**Goal**: To achieve fine-grained subgroup image forgetting in CLIP without relying on pre-training data.

**Key Insight**: Analysis reveals that direct Gradient Ascent (GA) causes catastrophic over-forgetting, where even similar subgroups are forgotten. The root cause is that parameter updates propagate uncontrollably to the feature spaces associated with the retain data.

**Core Idea**: A three-stage method: first use relative Fisher Information to select layers that are "important to the forget samples but less important to the retain samples" for selective unlearning; then use the distribution-aligned retain data to remind the model of the retained knowledge; and finally, leverage model merging to restore zero-shot capability.

## Method

### Overall Architecture
Input: Forget dataset $D^f$ (target subgroup images) + manually constructed retain dataset $D^r$ (images of other subgroups in the same class). Three-stage workflow: (1) Forgetting: layer selection using relative Fisher Information + LoRA fine-tuning for unlearning; (2) Reminding: fine-tuning on distribution-aligned retain data + EMA to prevent overfitting; (3) Restoring: model souping to restore zero-shot performance.

### Key Designs

1. **Relative Fisher for Layer Selection**:

    - **Function**: To identify critical layers that are important to the forget data but have minimal impact on the retain data.
    - **Mechanism**: Compute the relative Fisher Information $\mathcal{I}^l = \frac{\mathbb{E}_{D^f}[\nabla^2_{\theta^l} \text{sim}]}{\mathbb{E}_{D^r}[\nabla^2_{\theta^l} \text{sim}]}$ for each layer $l$. A higher ratio indicates that the layer is more sensitive to forget data and insensitive to retain data. Layers with the highest ratios are selected for LoRA fine-tuning, while others are frozen.
    - **Design Motivation**: Traditional Fisher Information only considers sensitivity to the forget data and ignores the impact on other data. The relative ratio balances "unlearning efficacy" and "retention safety."

2. **Distribution-Aligned Reminding**:

    - **Function**: To recover from the over-forgetting of the forgetting stage on the retain data, while reducing the distribution gap between the retain data and the pre-training data.
    - **Mechanism**: Learnable perturbations $\delta_i$ are added to the retain data. By minimizing $\mathcal{L}_a = \sum_l \|\mu_l^{img}(x_i + \delta_i) - BN_l^\mu\| + \|\sigma_l^{img}(x_i + \delta_i) - BN_l^\sigma\|$, the intermediate feature statistics of the retain samples are aligned with the pre-training distribution recorded by the BN layers. The model is then fine-tuned on the aligned data, with EMA ($\theta^{ema} = \alpha \theta^{ema} + (1-\alpha) \theta$) applied to prevent overfitting.
    - **Design Motivation**: Direct fine-tuning with small-scale retain data introduces distribution bias and leads to overfitting; the BN layer statistics implicitly encode the global distribution information of the pre-training data.

3. **Model Souping Restoration**:

    - **Function**: To restore the generalized zero-shot capability of CLIP.
    - **Mechanism**: Use a small calibration dataset $D_m$ to search for the optimal merging coefficient $\alpha$, and perform $\theta = \alpha \theta^f + (1-\alpha) \theta^{ori}$. Empirically, model merging locates the flat optima in the loss landscape, enhancing generalization capability.
    - **Design Motivation**: Both the forgetting and reminding stages shift the original representation space of CLIP. Model merging pulls the model back by interpolating with the original weights.

### Loss & Training
The forgetting stage minimizes $\mathcal{L}_f = \sum_i \frac{g^{img}(x_i^{img}) \cdot g^{txt}(x^{txt})}{\|g^{img}(x_i^{img}) \cdot g^{txt}(x^{txt})\|}$ (with maximizing image-text alignment acting as the unlearning loss). The reminding stage uses the standard contrastive loss to recover performance on the aligned retain data. LoRA adapters are applied to key layers throughout the process, greatly reducing the amount of parameter updates.

## Key Experimental Results

### Main Results: ImageNet-1K Subgroup Forgetting
Forgetting the "marmoset" subgroup while retaining the other 3 monkey subclasses:

| Backbone | Method | Target ↓ | Retain ↑ | ImageNet All ↑ | CIFAR ↑ | Food ↑ | STL ↑ | ObjectNet ↑ | Score |
|---|---|---|---|---|---|---|---|---|---|
| RN50 | Original | 51.0 | 54.7 | 59.8 | 70.4 | 60.9 | 92.0 | 68.9 | – |
| | GA | 0.0 | 0.9 | 32.2 | 16.4 | 22.9 | 63.3 | 22.1 | 45.3 |
| | EMMN | 0.0 | 56.7 | 28.4 | 13.1 | 20.4 | 54.9 | 20.5 | 55.6 |
| | LIP | 0.7 | 0.2 | 1.3 | 10.7 | 0.2 | 10.3 | 1.6 | 18.6 |
| | **Ours** | **0.0** | **50.2** | **54.5** | **85.7** | **62.9** | **81.5** | **45.2** | **91.0** |
| RN101 | GA | 0.0 | 0.3 | 36.2 | 19.9 | 29.3 | 58.6 | 19.4 | 47.3 |
| | **Ours** | **0.0** | **47.7** | **58.5** | **69.7** | **56.9** | **93.5** | **45.9** | **92.9** |

### Forgetting "box turtle" Subgroup (RN50)

| Method | Target ↓ | Retain ↑ | All ↑ | CIFAR ↑ | Food ↑ | STL ↑ | ObjNet ↑ | Score |
|---|---|---|---|---|---|---|---|---|
| GA | 0.0 | 13.3 | 42.1 | 14.6 | 40.0 | 77.8 | 21.3 | 55.8 |
| EMMN | 0.1 | 57.2 | 28.1 | 12.5 | 22.1 | 64.7 | 17.6 | 54.0 |
| **Ours** | **0.0** | **69.4** | **50.6** | **54.5** | **50.5** | **87.6** | **43.1** | **85.9** |

### Ablation Study

| Configuration | Target ↓ | Score |
|---|---|---|
| Full method (Ours) | 0.0 | 91.0 |
| w/o Relative Fisher (Full-layer fine-tuning) | 0.0 | 72.3 |
| w/o Distribution Alignment | 0.0 | 78.5 |
| w/o Model Souping | 0.0 | 82.1 |
| w/o LoRA (full FT) | 0.0 | 68.7 |

### Key Findings
- **GA and LIP lead to catastrophic over-forgetting**: GA's accuracy on ImageNet-All plummets from 59.8% to 32.2%, and on CIFAR from 70.4% to 16.4%. LIP is even more extreme, with CIFAR dropping to 10.7%.
- **Ours achieves precise unlearning**: While the accuracy on Target drops to 0.0%, Retain keeps 50.2% (original 54.7%), ImageNet-All maintains 54.5% (original 59.8%), and cross-dataset zero-shot capabilities remain almost unaffected.
- **Absolute performance lead in Score**: On RN50, 91.0 vs. the second-best EMMN 55.6 (+35.4); on RN101, 92.9 vs. 49.5 (+43.4).
- **All three stages are indispensable**: Removing Relative Fisher, Distribution Alignment, or Model Souping drops the Score from 91.0 to 72.3, 78.5, and 82.1 respectively.

## Highlights & Insights
- **Subgroup-level precise unlearning**: Instead of unlearning an entire category, the targeted approach handles specific subgroups within a category (e.g., Boeing vs. generic airplanes), which is highly aligned with real-world needs (such as copyright and privacy scenarios).
- **Ingenious utilization of BN statistics**: Leveraging the pre-training distribution information implicitly stored in the BN layers of CLIP to compensate for the inaccessibility of the pre-training data. This trick can be generalized to any scenario requiring distribution alignment.
- **Clean three-stage design**: Each stage addresses a clear and distinct problem: selective unlearning → preventing over-forgetting → restoring generalization capability, resulting in a cohesive logical flow.

## Limitations & Future Work
- The retain dataset needs to be manually constructed, which incurs non-negligible labor costs in practical applications.
- CLIP experiments were only conducted on ResNet-based backbones rather than ViTs; results on ViT-based CLIP are currently missing.
- Model souping requires searching for the merging coefficient $\alpha$, which increases hyperparameter tuning overhead.
- The permanence of the unlearning effect is unverified—whether the model will "re-memorize" target subgroups after continuing training remains unexplored.

## Related Work & Insights
- **vs. CLIP-LIP**: Applies LRP + LoRA to the CLIP text encoder for concept forgetting, but yields a Score of only 33.2-44.4 in subgroup scenarios, whereas the proposed three-stage method achieves 85.9-92.9.
- **vs. EMMN**: Although the error minimization-maximization framework requires no data, it suffers from severe over-forgetting, achieving scores of only 49.5-55.6.
- **vs. Traditional Fisher-based unlearning**: Utilizing only the Fisher Information of the forget data tends to select layers that are also sensitive to the retain data, leading to over-forgetting. The relative Fisher Information effectively addresses this problem.

## Rating
- Novelty: ⭐⭐⭐⭐ The formulation of the subgroup unlearning problem and the proposed three-stage solution are both valuable contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated across ImageNet, CIFAR, and multiple cross-domain datasets, but lacks validation on ViT backbones.
- Writing Quality: ⭐⭐⭐⭐ Clearly structured, but shows some inconsistencies in mathematical notations.
- Value: ⭐⭐⭐⭐ High practicality in subgroup unlearning scenarios, with scalable methodology.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Kiss3DGen: Repurposing Image Diffusion Models for 3D Asset Generation](kiss3dgen_repurposing_image_diffusion_models_for_3d_asset_generation.md)
- [\[CVPR 2025\] Video Depth Without Video Models](video_depth_without_video_models.md)
- [\[CVPR 2025\] Gaussian Eigen Models for Human Heads](gaussian_eigen_models_for_human_heads.md)
- [\[CVPR 2025\] Scaling Properties of Diffusion Models for Perceptual Tasks](scaling_properties_of_diffusion_models_for_perceptual_tasks.md)
- [\[AAAI 2026\] CLIPPan: Adapting CLIP as A Supervisor for Unsupervised Pansharpening](../../AAAI2026/3d_vision/clippan_adapting_clip_as_a_supervisor_for_unsupervised_pansharpening.md)

</div>

<!-- RELATED:END -->
