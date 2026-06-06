---
title: >-
  [Paper Note] The Impact of Scaling Training Data on Adversarial Robustness
description: >-
  [NeurIPS 2025][Audio & Speech][Adversarial Robustness] A systematic evaluation of 36 state-of-the-art vision models under 6 categories of black-box attacks reveals that attack success rate (ASR) decreases logarithmically…
tags:
  - "NeurIPS 2025"
  - "Audio & Speech"
  - "Adversarial Robustness"
  - "Scaling Laws"
  - "Black-box Attacks"
  - "Data Quality"
  - "Vision Models"
date: 2026-05-08
content_hash: 8282124c5b57af13
---

# The Impact of Scaling Training Data on Adversarial Robustness

**Conference**: NeurIPS 2025
**arXiv**: [2509.25927](https://arxiv.org/abs/2509.25927)  
**Code**: None  
**Area**: Audio/Speech (Adversarial Robustness)
**Keywords**: Adversarial Robustness, Scaling Laws, Black-box Attacks, Data Quality, Vision Models

## TL;DR

A systematic evaluation of 36 state-of-the-art vision models under 6 categories of black-box attacks reveals that attack success rate (ASR) decreases logarithmically with training data volume and model scale; however, **data quality and model scale are more critical than data volume alone**.

## Background & Motivation

**Background**: DNNs have achieved remarkable success on CV tasks, yet adversarial examples remain a fundamental challenge for deploying models in safety-critical applications. Recent years have seen the emergence of diverse training paradigms—ViT, DINOv2, CLIP, etc.—with training datasets ranging from 1.2 million to 22 billion images.

**Limitations of Prior Work**: It has been commonly assumed that larger datasets and more sophisticated training objectives lead to stronger robustness; however, empirical observations show that some models trained on small, carefully curated datasets are more robust than those trained on orders-of-magnitude larger datasets.

**Key Challenge**: The independent contributions of data volume, data quality, model scale, and training paradigm to robustness have not been isolated or quantified.

**Goal**: Establish quantitative relationships (scaling laws) between training data characteristics and adversarial robustness.

**Key Insight**: A large-scale systematic evaluation spanning 36 models and 6 categories of semantic attacks.

**Core Idea**: Adversarial robustness decreases logarithmically with data volume and model size; however, the effect of model scale far outweighs that of data volume, and models trained on high-quality curated data (e.g., DINOv2) can substantially outperform CLIP models trained on 100× more data.

## Method

### Overall Architecture

A comprehensive black-box evaluation framework is constructed:
- **36 models**: ViT, ResNet, CLIP, DINOv1/v2, Swin/v2, ConvNeXt, YOLO, ViT-MAE, PaliGemma, BEiT/v2, SigLIP/v2
- **6 attack categories**: Random Perturbations, GeometricMasksV1, GeometricMasksV2, COCO Objects, ImageNet-C, ImageNet-R
- **Evaluation**: ImageNet-1K validation set

### Key Designs

#### Evaluation Metrics
- **Accuracy**: $\text{Acc}(C, \mathcal{D}) = \frac{1}{|\mathcal{D}|} \sum_{(x,y) \in \mathcal{D}} \mathbf{1}[C(x) = y]$
- **Attack Success Rate (ASR)**: $\text{ASR} = \frac{1}{|\mathcal{S}_{\text{correct}}|} \sum_{(x,y) \in \mathcal{S}_{\text{correct}}} \mathbf{1}[C(A(x)) \neq y]$
- For scenarios where original clean images are unavailable, a proxy dataset is used to approximate ASR, with an error of 3.09 pp ($\sigma=1.93$ pp)

#### Adversarial Fine-tuning Experiments
Three ResNet50 variants are fine-tuned under different GeometricMasksV2 configurations:
- v1: 3-4-2 C1 (opacity=64, 50% adversarial examples)
- v2: 3-4-2 C1&C2
- v3: Random C1&C2

#### Human Evaluation
- GeometricMasksV2 6-7-2 C1, with 4 difficulty levels (opacity 0/64/96/128)
- ImageNette dataset, 6 human participants

### Loss & Training

- CLIP models are evaluated zero-shot using the prompt "a photo of a {class name}"
- DINOv1, ViT-MAE, and PaliGemma use frozen backbones with a linear classification head
- Adversarial fine-tuning: initialized from ImageNet pre-trained weights, batch=64, 3 epochs

## Key Experimental Results

### Main Results — Scaling Laws

| Dimension | Univariate Scaling Law | Interpretation |
|-----------|----------------------|----------------|
| Data volume | $\text{ASR} = -3.16 \log_{10}(x) + 55.53$ | 10× more data → ASR drops ~3.2 pp |
| Model scale | $\text{ASR} = -13.39 \log_{10}(x) + 141.18$ | 10× more parameters → ASR drops ~13.4 pp |

**Bivariate scaling law** (after PCA-based disentanglement):
$$\text{ASR} = -0.46 \log_{10}(x_{\text{data}}) - 12.53 \log_{10}(x_{\text{model}}) + 137.67$$

The independent contribution of model scale far exceeds that of data volume.

### Model Rankings

| Model | Training Data Volume | Overall Mean ASR |
|-------|---------------------|-----------------|
| DINOv2-G | 142M | **10.3%** (best) |
| DINOv2-L | 142M | ~12% |
| Swinv2-L-384 | 14.2M | 16.8% |
| ResNet50 | 1.2M | ~50% (worst) |

### Human vs. Model Comparison

| Difficulty (opacity) | Human | DINOv2-B | ResNet-v1 (fine-tuned) | ResNet50 |
|----------------------|-------|----------|----------------------|---------|
| 0 (clean) | ~100% | ~99% | ~98% | ~96% |
| 64 | ~97% | ~92% | ~93% | ~65% |
| 128 | ~93% | ~87% | ~87% | ~35% |

### Ablation Study — Adversarial Fine-tuning

- Generalizes to structural variations (shape, scale, rotation) ✅
- Does not transfer to color distribution changes ❌ → geometric and color invariance are learned independently

### Key Findings

1. Training paradigm (supervised / self-supervised / contrastive learning) has limited impact on robustness—contrastive learning achieves 27.9% vs. supervised 34.3% ASR
2. DINOv2 is trained on only 142M images yet achieves substantially lower ASR than CLIP models trained on 22B images
3. Without controlling for data quality, scaling CLIP yields limited robustness gains
4. Human participants consistently outperform the best models at all difficulty levels; even the best model misclassifies ~13% of samples at high difficulty

## Highlights & Insights

- **First bivariate scaling law for adversarial robustness of vision models**, disentangling the independent contributions of data volume and model scale
- **Strong evidence for "quality over quantity"**: DINOv2 (142M high-quality curated data) substantially outperforms CLIP (billions of web-crawled images)
- **Limits of adversarial fine-tuning**: geometric robustness transfers across configurations but color robustness does not, revealing the modular nature of visual feature learning
- **Persistent gap between biological and artificial vision**: the robustness of the human visual system remains an upper bound that artificial models have yet to approach

## Limitations & Future Work

1. White-box gradient-based attacks (e.g., PGD, AutoAttack) are not evaluated
2. Analysis is limited to classification tasks and does not extend to detection or segmentation
3. Training dataset documentation lacks standardization, making precise variable control difficult
4. Future work should verify whether the observed scaling trends hold under gradient-based attacks

## Related Work & Insights

- **RobustBench** (Croce et al., 2021): standardized robustness benchmark
- **DINOv2** (Oquab et al., 2024): self-supervised visual representation learning
- **Bartoldson et al., 2024**: robustness scaling studies for language models
- **Insight**: Under resource constraints, prioritizing data curation and model scale is more effective than simply increasing data volume

## Rating

⭐⭐⭐⭐ (4/5)
The large-scale systematic evaluation provides highly valuable quantitative insights; however, the attack types are limited to black-box settings and the tasks are restricted to classification, leaving room for broader experimental coverage.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Data-Juicer 2.0: Cloud-Scale Adaptive Data Processing for and with Foundation Models](data-juicer_20_cloud-scale_adaptive_data_processing_for_and_with_foundation_mode.md)
- [\[NeurIPS 2025\] AdaptDel: Adaptable Deletion Rate Randomized Smoothing for Certified Robustness](adaptdel_adaptable_deletion_rate_randomized_smoothing_for_ce.md)
- [\[NeurIPS 2025\] AVRobustBench: Benchmarking the Robustness of Audio-Visual Recognition Models at Test-Time](textttavrobustbench_benchmarking_the_robustness_of_audio-visual_recognition_mode.md)
- [\[CVPR 2026\] Unlocking Strong Supervision: A Data-Centric Study of General-Purpose Audio Pre-Training Methods](../../CVPR2026/audio_speech/unlocking_strong_supervision_a_data-centric_study_of_general-purpose_audio_pre-t.md)
- [\[NeurIPS 2025\] Sensorium Arc: AI Agent System for Oceanic Data Exploration and Interactive Eco-Art](sensorium_arc_ai_agent_system_for_oceanic_data_exploration_and_interactive_eco-a.md)

</div>

<!-- RELATED:END -->
