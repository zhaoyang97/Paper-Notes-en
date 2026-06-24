---
title: >-
  [Paper Note] Bridging the Vision-Brain Gap with an Uncertainty-Aware Blur Prior
description: >-
  [CVPR 2025][LLM Pretraining][Brain signal decoding] This work introduces the concepts of "System GAP" and "Random GAP" for the first time to describe the information mismatch between brain signals and visual stimuli. By dynamically adjusting the image blur level through an Uncertainty-Aware Blur Prior (UBP) to alleviate overfitting during training, it achieves a 50.9% top-1 accuracy on the 200-way zero-shot brain-image retrieval task, outperforming the previous SOTA by 13.7 p…
tags:
  - "CVPR 2025"
  - "LLM Pretraining"
  - "Brain signal decoding"
  - "EEG"
  - "Uncertainty-aware"
  - "Blur prior"
  - "Vision-brain gap"
date: 2026-05-08
content_hash: 739dca1672b55562
---

# Bridging the Vision-Brain Gap with an Uncertainty-Aware Blur Prior

**Conference**: CVPR 2025  
**arXiv**: [2503.04207](https://arxiv.org/abs/2503.04207)  
**Code**: [https://github.com/](https://github.com/) (to be released)  
**Area**: LLM Pre-training  
**Keywords**: Brain signal decoding, EEG, Uncertainty-aware, Blur prior, Vision-brain gap

## TL;DR

This work introduces the concepts of "System GAP" and "Random GAP" for the first time to describe the information mismatch between brain signals and visual stimuli. By dynamically adjusting the image blur level through an Uncertainty-Aware Blur Prior (UBP) to alleviate overfitting during training, it achieves a 50.9% top-1 accuracy on the 200-way zero-shot brain-image retrieval task, outperforming the previous SOTA by 13.7 percentage points.

## Background & Motivation

**Background**: Visual neural decoding aims to retrieve or reconstruct original visual stimuli from brain signals such as EEG/fMRI. Mainstream methods (e.g., BraVL, NICE) align the output of the brain signal encoder with CLIP image features via contrastive learning.

**Limitations of Prior Work**: Existing methods directly align brain signals with original image features, ignoring the information gap between them. When the human brain processes visual information, high-frequency details are lost (System GAP), and factors like attentional fluctuations, cognitive associations, and signal acquisition noise introduce further random variations (Random GAP).

**Key Challenge**: Under scarce paired data conditions, models are forced to learn how to bridge these gaps, leading to severe overfitting to the training set and failure to generalize to new data.

**Goal**: Introduce prior knowledge to alleviate the impact of both gaps and improve the alignment quality of brain-vision contrastive learning.

**Key Insight**: The human visual system itself acts as a low-pass filter—with high resolution in the fovea and low resolution in the periphery. Therefore, brain signals do not contain all the high-frequency information of an image. Blurring can bring the image closer to the information level of the brain signals.

**Core Idea**: Apply Gaussian blur to training images to simulate information loss (alleviating the System GAP) and dynamically adjust the blur intensity by estimating the uncertainty of each sample pair (alleviating the Random GAP).

## Method

### Overall Architecture

UBP adds two components to the standard vision-brain contrastive learning framework: (1) Blur Prior—applying foveated Gaussian blur to images before extracting features; (2) Uncertainty Quantification—dynamically adjusting the blur radius $r$ based on the similarity of paired samples.

### Key Designs

1. **Foveated Blur Prior**:

    - **Function**: Simulates the information loss in the human visual system and removes high-frequency details from the image.
    - **Mechanism**: First, a uniform Gaussian blur $x_\text{blur}$ is applied to the image, which is then blended with the original image using a distance-decay weight: $\tilde{x}_v = \alpha \cdot x + (1-\alpha) \cdot x_\text{blur}$, where $\alpha(i,j) = \exp(-\lambda \cdot d(i,j)/L)$, and $d(i,j)$ represents the distance from the pixel to the fovea (center of the image). Consequently, the center remains clear while the periphery becomes progressively blurred, mimicking human visual resolution distribution. The degree of blur is controlled by the Gaussian kernel radius $r$.
    - **Design Motivation**: Directly aligning original images rich in high-frequency details with brain signals lacking these details forces the model to learn impossible mappings. The blur prior reduces the difficulty of this alignment.

2. **Uncertainty-Aware Dynamic Blur**:

    - **Function**: Adaptively adjusts the blur strength based on the uncertainty of each paired sample.
    - **Mechanism**: The similarity matrix $M = h_b \cdot h_v^\top \cdot \text{softplus}(\tau)$ of paired samples in the same batch is calculated, and the diagonal is taken to obtain the similarity $S$ of $N$ pairs. $S$ is assumed to approximately follow a normal distribution $\mathcal{N}(\hat\mu, \hat\sigma^2)$. Samples falling outside the confidence interval are considered high-uncertainty samples. For samples with extremely low similarity (implying a large Random GAP), the blur radius is increased to $r_0 + c$. For those with extremely high similarity, it is reduced to $r_0 - c$, while remaining at $r_0$ for samples within the normal range.
    - **Design Motivation**: Random GAP factors (attentional shift, cognitive associations, signal noise) result in inconsistent quality of brain signals. Stronger blur must be applied to "poorly aligned" samples to reduce the difficulty of the alignment target.

### Loss & Training

Symmetric Cross-Entropy (SCE) loss is used for contrastive learning, freezing the CLIP vision encoder $f_V$ and training the brain signal encoder $f_B$. The blur radius is dynamically updated via uncertainty estimation during training.

## Key Experimental Results

### Main Results

| Method | Top-1 Avg | Top-5 Avg |
|------|-----------|-----------|
| BraVL | 5.8 | 17.5 |
| NICE | 17.2 | 44.4 |
| ATM-S | 37.2 | 69.9 |
| **UBP (ours)** | **50.9** | **79.7** |

*THINGS-EEG 200-way zero-shot retrieval, UBP outperforms the previous SOTA by 13.7/9.8 percentage points.*

### Ablation Study

| Configuration | Top-1 | Top-5 |
|------|-------|-------|
| Baseline (No Blur) | 37.2 | 69.9 |
| +Blur Prior (Fixed $r$) | 46.3 | 76.1 |
| +Uncertainty-aware | **50.9** | **79.7** |

### Key Findings
- Adding the blur prior alone yields a 9.1% top-1 improvement, confirming the existence and impact of the System GAP.
- The uncertainty-aware dynamic adjustment contributes an additional 4.6%, validating the importance of the Random GAP.
- Performance variance across different subjects is negatively correlated with EEG signal variability (high variability → low performance), supporting the Random GAP theory.
- The method is simple and general, allowing plug-and-play integration into any brain-vision contrastive learning framework.

## Highlights & Insights
- **Conceptualization of System GAP and Random GAP**: This work systematically analyzes the sources of mismatch between brain signals and visual stimuli for the first time, establishing a theoretical framework for this field.
- **Biological Plausibility of Foveated Blur**: Simulating the resolution degradation of the human visual system presents a rare case of integrating biological perception neuroscience insights into deep learning optimization.
- **Simplicity of the Method**: The core only involves blurring and dynamic adjustment with almost zero extra computational overhead but yields significant gains, embodying the philosophy of "correct reasoning over complex methodology."

## Limitations & Future Work
- The method has only been validated on EEG signals; its applicability to high-spatial-resolution modalities like fMRI remains to be tested.
- The blur prior assumes that brain signals primarily lose high-frequency information, but actual information loss patterns could be more complex.
- Uncertainty quantification is based on a simple normal distribution assumption and three-stage adjustment; more sophisticated modeling might be more effective.

## Related Work & Insights
- **vs ATM-S**: ATM-S improves performance via better brain encoders, while UBP accelerates optimization from the alignment target (image side), rendering them complementary.
- **vs NICE**: NICE performs direct contrastive learning, whereas UBP introduces a prior to reduce target noise in alignment.
- The underlying concept is transferrable to other contrastive learning settings featuring noisy paired data (e.g., weak labeling, distant supervision).

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The conceptualization of System/Random GAP and the blur prior solution are highly commendable.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive experiments over 10 subjects, comprehensive ablation studies, and in-depth analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear motivation, closely combining theory with experiments.
- Value: ⭐⭐⭐⭐ Significantly advances the performance of brain signal decoding.
---
title: >-
  [Paper Reading] Bridging the Vision-Brain Gap with an Uncertainty-Aware Blur Prior
description: >-
  [CVPR 2025][LLM/NLP][Vision-Brain Gap] Proposes an uncertainty-aware blur prior to offer a physically plausible image degradation model for reconstructing visual stimuli from brain signals (fMRI/EEG), mitigating the impact of high-frequency information loss during brain encoding on reconstruction quality.
tags:
  - CVPR 2025
  - LLM/NLP
  - Vision-Brain Gap
  - Uncertainty-aware
  - Blur prior
  - fMRI Decoding
  - Image Reconstruction
---

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Image Intrinsic Scale Assessment: Bridging the Gap Between Quality and Resolution](../../ICCV2025/llm_pretraining/image_intrinsic_scale_assessment_bridging_the_gap_between_quality_and_resolution.md)
- [\[ICML 2025\] Position: The Future of Bayesian Prediction Is Prior-Fitted](../../ICML2025/llm_pretraining/position_the_future_of_bayesian_prediction_is_prior-fitted.md)
- [\[NeurIPS 2025\] Does Object Binding Naturally Emerge in Large Pretrained Vision Transformers?](../../NeurIPS2025/llm_pretraining/does_object_binding_naturally_emerge_in_large_pretrained_vision_transformers.md)
- [\[ICML 2025\] Bayesian Neural Scaling Law Extrapolation with Prior-Data Fitted Networks](../../ICML2025/llm_pretraining/bayesian_neural_scaling_law_extrapolation_with_prior-data_fitted_networks.md)
- [\[ACL 2025\] SCAR: Data Selection via Style Consistency-Aware Response Ranking for Efficient Instruction-Tuning](../../ACL2025/llm_pretraining/scar_style_consistency_data_selection.md)

</div>

<!-- RELATED:END -->
