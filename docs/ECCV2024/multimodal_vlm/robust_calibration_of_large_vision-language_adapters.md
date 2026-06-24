---
title: >-
  [Paper Note] Robust Calibration of Large Vision-Language Adapters
description: >-
  [ECCV 2024][Multimodal VLM][CLIP adaptation] This paper discovers that CLIP adaptation methods (Adapter/Prompt Learning/TTA) severely impair the calibration capability of the zero-shot baseline in OOD scenarios, reveals that increased logit range (rather than increased logit norm) is the root cause of miscalibration, and proposes three simple and model-agnostic logit range constraint schemes (ZS-Norm, Penalty, and SaLS) that effectively mitigate miscalibration while maintaini…
tags:
  - "ECCV 2024"
  - "Multimodal VLM"
  - "CLIP adaptation"
  - "model calibration"
  - "out-of-distribution generalization"
  - "logit range constraint"
  - "uncertainty estimation"
date: 2026-05-08
content_hash: 1644c5c6b695c2ac
---

# Robust Calibration of Large Vision-Language Adapters

**Conference**: ECCV 2024  
**arXiv**: [2407.13588](https://arxiv.org/abs/2407.13588)  
**Code**: [https://github.com/Bala93/CLIPCalib](https://github.com/Bala93/CLIPCalib)  
**Area**: Multimodal VLM  
**Keywords**: CLIP adaptation, model calibration, out-of-distribution generalization, logit range constraint, uncertainty estimation

## TL;DR
This paper discovers that CLIP adaptation methods (Adapter/Prompt Learning/TTA) severely impair the calibration capability of the zero-shot baseline in OOD scenarios, reveals that increased logit range (rather than increased logit norm) is the root cause of miscalibration, and proposes three simple and model-agnostic logit range constraint schemes (ZS-Norm, Penalty, and SaLS) that effectively mitigate miscalibration while maintaining discriminative performance.

## Background & Motivation
- **Background**: Large-scale VLMs such as CLIP demonstrate strong zero-shot generalization capabilities through pre-training. To adapt to downstream tasks, the community has developed three major types of methods: Prompt Learning (CoOp, CoCoOp, etc.), black-box Adapters (CLIP-Adapter, TIP-Adapter, etc.), and test-time adaptation (TPT)
- **Limitations of Prior Work**: Although these adaptation methods improve discriminative accuracy, the authors found that they severely degrade the calibration capability of the models—adapted models tend to be overconfident, providing high confidence even when making incorrect predictions. This is particularly dangerous in safety-sensitive domains such as healthcare
- **Key Challenge**: Existing literature on CLIP adaptation focuses almost entirely on improving discriminative performance, neglecting model calibration (the accuracy of uncertainty estimation), which is a key metric for reliable deployment
- **Key Insight**: Prior work (such as LogitNorm) suggested that miscalibration in fully supervised models stems from the growth of the logit norm. However, this paper theoretically and experimentally demonstrates that in the context of CLIP adaptation, the increase in logit **range** (max - min) is the true cause of miscalibration. Adding a constant offset can increase the norm without changing softmax probabilities, whereas scaling the logit vector simultaneously increases both the range and the softmax probabilities

## Method

### Overall Architecture
The authors propose a general constrained optimization framework: while minimizing the adaptation objective function $\mathcal{H}$, the logits of each sample are constrained to stay within the logit range of its zero-shot prediction. Specifically, three implementation schemes are proposed, which can be flexibly applied during either the training or the inference stage.

### Key Designs

1. **ZS-Norm (Zero-Shot Logit Normalization)**: During training, the logits output by the adapted model are re-normalized to scale their range to match that of the corresponding zero-shot predictions. Core formula: $\mathbf{l}_i' = \frac{l_i^{\text{ZS-max}} - l_i^{\text{ZS-min}}}{l_i^{\text{max}} - l_i^{\text{min}}}(\mathbf{l}_i - l_i^{\text{min}}\mathbf{1}) + l_i^{\text{ZS-min}}\mathbf{1}$. The motivation is to directly enforce that the logit range does not exceed the zero-shot baseline during forward propagation, thereby preserving calibration characteristics.

2. **Penalty (Explicit Penalty Term)**: The constraint is converted into a ReLU penalty term added to the primary loss: $\lambda\sum_{i}\sum_{k}(\text{ReLU}(l_{ik} - l_i^{\text{ZS-max}}) + \text{ReLU}(l_i^{\text{ZS-min}} - l_{ik}))$. Gradient signals are generated to correct the logits when they exceed the zero-shot range. $\lambda$ is fixed to 10.

3. **SaLS (Sample-adaptive Logit Scaling)**: During inference, the ZS-Norm formula is used to scale the logits of each test sample. This is equivalent to an unsupervised, sample-wise temperature scaling that requires no validation set and naturally adapts to distribution shifts. This is the simplest and most effective scheme.

### Theoretical Support
- **Proposition 1**: Adding a positive constant $a$ to the logit vector increases the norm but leaves the softmax probability unchanged $\to$ increased norm $\neq$ degraded calibration
- **Proposition 2**: Multiplying the logit vector by $a>1$ increases the range and increases the maximum class softmax probability $\to$ wider range $\to$ overconfidence $\to$ miscalibration

### Loss & Training
- ZS-Norm and Penalty are integrated **during training**, modifying the learning objectives of the adaptation process.
- SaLS is an **inference-time** post-processing method, which does not modify the training workflow at all.
- All three methods are agnostic to specific adaptation strategies and can be directly integrated into any method, such as CoOp, CLIP-Adapter, or TPT.

## Key Experimental Results

### Main Results (OOD Domain Generalization, Average of ImageNet $\to$ 4 OOD Datasets)

| Method | Backbone | ACC | ECE | ECE Improvement |
|------|----------|-----|-----|---------|
| Zero-Shot | ViT-B/16 | 57.15 | 4.78 | baseline |
| TIP-Ad(f) | ViT-B/16 | 25.86 | 63.63 | +58.85↑ |
| TIP-Ad(f)+Penalty | ViT-B/16 | 49.23 | 40.98 | -22.65↓ |
| TIP-Ad(f)+SaLS | ViT-B/16 | 25.86 | 44.37 | -19.26↓ |
| TaskRes | ViT-B/16 | 58.01 | 7.52 | +2.74↑ |
| TaskRes+SaLS | ViT-B/16 | 58.01 | 6.21 | -1.31↓ |
| CoOp+ZS-Norm | ViT-B/16 | 58.75 | 4.35 | -2.26↓ |
| CoCoOp+Penalty | ViT-B/16 | 60.20 | 3.89 | -0.94↓ |

### Test-time Adaptation (11 Fine-grained Datasets, RN50)

| Method | ACC | ECE | ECE Improvement |
|------|-----|-----|---------|
| Zero-Shot | 56.03 | 5.04 | baseline |
| TPT | 58.03 | 7.67 | +2.63↑ |
| TPT+SaLS | 58.03 | 5.69 | -1.98↓ |
| C-TPT+SaLS | 57.54 | 6.79 | -0.88↓ |

### Ablation Study

| Configuration | Key Metrics | Description |
|------|---------|------|
| Logit Norm Constraint | Limited ECE reduction | Verifies that norm is not the primary cause of miscalibration |
| Logit Range Constraint | Significant ECE reduction | Verifies that range is the primary cause of miscalibration |
| SaLS vs TS | SaLS is superior | Sample-wise adaptation outperforms global temperature scaling |

### Key Findings
- The logit norm actually **decreases** after CLIP adaptation, yet ECE increases—directly refuting the traditional view that "increased norm leads to miscalibration"
- There is a clear positive correlation between logit range and ECE
- As an inference-time post-processing method, SaLS effectively reduces ECE in almost all configurations without compromising ACC
- The Penalty method can even simultaneously improve ACC and reduce ECE on certain Adapters

## Highlights & Insights
- **Outstanding Theoretical Contribution**: Two propositions clearly distinguish the different impacts of logit norm and logit range on calibration, correcting conventional misconceptions in the field
- **Highly Practical SaLS**: A zero-cost, training-free, and model-agnostic inference-time scheme that can be directly deployed to any CLIP adaptation method
- **Valuable Problem Definition**: Systematically reveals, for the first time, the calibration degradation issue of CLIP adaptation methods in OOD scenarios

## Limitations & Future Work
- ZS-Norm worsens performance on certain Adapters, indicating that normalization during training may lead to overfitting
- Only classification tasks are considered, without exploring calibration in downstream tasks like detection and segmentation
- It is assumed that the calibration of the zero-shot model is superior, but in some specific domains, the zero-shot model itself might be poorly calibrated
- Combining SaLS with other post-processing calibration methods (such as hybrid strategies) could be explored

## Related Work & Insights
- LogitNorm (ICML 2022) proposed constraining the logit norm to improve calibration, whereas this paper points out that the logit range should be constrained in the context of CLIP adaptation
- Temperature Scaling requires validation data and uses a global parameter, whereas SaLS achieves unsupervised sample-wise temperature adaptation
- This can inspire applying logit range constraints to other transfer learning scenarios (such as domain adaptation)

## Rating
- Novelty: ⭐⭐⭐⭐ The theoretical insight (range vs. norm) is novel, but the proposed solutions are relatively simple
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers three major categories of adaptation methods, two backbones, and two task setups
- Writing Quality: ⭐⭐⭐⭐⭐ Clear problem definition, rigorous theoretical derivation, and systematic experimental organization
- Value: ⭐⭐⭐⭐ Discovers a previously overlooked yet critical problem, and the SaLS scheme is highly practical

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Don't Miss the Forest for the Trees: Attentional Vision Calibration for Large Vision Language Models](../../ACL2025/multimodal_vlm/dont_miss_the_forest_for_the_trees_attentional_vision_calibration_for_large_visi.md)
- [\[ECCV 2024\] Vary: Scaling up the Vision Vocabulary for Large Vision-Language Models](vary_scaling_up_the_vision_vocabulary_for_large_visionlanguag.md)
- [\[ECCV 2024\] Attention Prompting on Image for Large Vision-Language Models](attention_prompting_on_image_for_large_visionlanguage_models.md)
- [\[ECCV 2024\] SQ-LLaVA: Self-Questioning for Large Vision-Language Assistant](sq-llava_self-questioning_for_large_vision-language_assistant.md)
- [\[ECCV 2024\] MarvelOVD: Marrying Object Recognition and Vision-Language Models for Robust Open-Vocabulary Object Detection](marvelovd_marrying_object_recognition_and_vision-language_models_for_robust_open.md)

</div>

<!-- RELATED:END -->
