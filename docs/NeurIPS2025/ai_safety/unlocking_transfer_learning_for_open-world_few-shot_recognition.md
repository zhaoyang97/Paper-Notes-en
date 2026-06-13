---
title: >-
  [Paper Note] Unlocking Transfer Learning for Open-World Few-Shot Recognition
description: >-
  [NeurIPS 2025 (Workshop)][AI Safety][Few-shot open-set recognition] A two-stage framework is proposed that combines open-set-aware meta-learning with open-set-free transfer learning…
tags:
  - "NeurIPS 2025 (Workshop)"
  - "AI Safety"
  - "Few-shot open-set recognition"
  - "transfer learning"
  - "meta-learning"
  - "open world"
  - "pseudo open-set samples"
date: 2026-05-08
content_hash: c1c10feab5440463
---

# Unlocking Transfer Learning for Open-World Few-Shot Recognition

**Conference**: NeurIPS 2025 (Workshop)
**arXiv**: [2411.09986](https://arxiv.org/abs/2411.09986)  
**Code**: N/A  
**Area**: Few-Shot Learning / Open-World Recognition
**Keywords**: Few-shot open-set recognition, transfer learning, meta-learning, open world, pseudo open-set samples

## TL;DR

A two-stage framework is proposed that combines open-set-aware meta-learning with open-set-free transfer learning, achieving the first successful application of the transfer learning paradigm to few-shot open-set recognition (FSOSR) and reaching SOTA on miniImageNet and tieredImageNet.

## Background & Motivation

Few-Shot Open-Set Recognition (FSOSR) presents a dual challenge:

**Closed-set classification**: Classifying inputs into known categories using only a handful of examples

**Open-set detection**: Simultaneously identifying inputs that do not belong to any known category

Although transfer learning has become the dominant paradigm for closed-set few-shot learning, its application to open-world scenarios exposes a fundamental limitation: **fine-tuning on closed-set samples alone provides no modeling of open-set boundaries**. This paper aims to "unlock" the potential of transfer learning for FSOSR.

## Method

### Overall Architecture

A two-stage pipeline:
1. **Stage 1 — Open-Set-Aware Meta-Learning**: Trains the model to establish a metric space conducive to open-set detection.
2. **Stage 2 — Open-Set-Free Transfer Learning**: Fine-tunes the model for specific tasks via transfer learning.

### Key Designs

1. **Open-Set-Aware Meta-Learning**:

    - Introduces simulated open-set samples during the meta-learning stage.
    - Trains the model to not only discriminate among known classes but also recognize unseen inputs.
    - The resulting metric space provides a favorable initialization for subsequent transfer learning.

2. **Pseudo Open-Set Sample Generation Strategies**:

    - **Dataset modification**: A subset of training classes is held out to simulate open-set inputs.
    - **Generation**: Out-of-distribution samples are synthesized in feature space.
    - The two strategies are complementary and can be used jointly.

3. **Open-Set-Free Transfer Learning**:

    - At test time, fine-tuning is performed using only the closed-set support set.
    - No additional open-set calibration samples are required.
    - Open-set detection capability is maintained through the metric space established in Stage 1.

### Loss & Training

$$\mathcal{L} = \mathcal{L}_{\text{CE}} + \lambda \mathcal{L}_{\text{open}}$$

- $\mathcal{L}_{\text{CE}}$: Cross-entropy loss for closed-set classification
- $\mathcal{L}_{\text{open}}$: Open-set detection loss (margin-based contrastive loss)

## Key Experimental Results

### Main Results

| Method | miniImageNet AUROC ↑ | miniImageNet Acc ↑ | tieredImageNet AUROC ↑ | tieredImageNet Acc ↑ |
|------|--------------------|--------------------|----------------------|--------------------|
| ProtoNet + threshold | 68.5 | 65.2 | 70.2 | 67.8 |
| PEELER | 72.3 | 68.5 | 73.8 | 70.5 |
| SnaTCHer | 74.1 | 70.2 | 75.5 | 72.3 |
| OpenFSL | 75.8 | 71.5 | 77.2 | 73.8 |
| TANE | 76.5 | 72.8 | 78.1 | 74.5 |
| **Ours** | **79.2** | **75.5** | **80.8** | **77.2** |

### Training Efficiency Comparison

| Method | Training Epochs | Additional Overhead | 5-shot AUROC | 1-shot AUROC |
|------|-----------|-----------|------------|------------|
| PEELER | 200 | +50% | 72.3 | 65.8 |
| SnaTCHer | 200 | +30% | 74.1 | 67.2 |
| OpenFSL | 200 | +40% | 75.8 | 68.5 |
| **Ours** | **200** | **+1.5%** | **79.2** | **72.5** |

### Ablation Study

| Configuration | AUROC ↑ | Closed-set Acc ↑ |
|------|---------|-----------------|
| Full method | 79.2 | 75.5 |
| Meta-learning only (no transfer learning) | 75.5 | 70.2 |
| Transfer learning only (no meta-learning) | 71.8 | 74.8 |
| Without pseudo open-set samples | 74.5 | 73.5 |
| Dataset modification pseudo open-set | 77.8 | 74.8 |
| Generative pseudo open-set | 78.5 | 75.2 |
| Both pseudo open-set strategies | 79.2 | 75.5 |

### Key Findings

1. Transfer learning is indeed viable for FSOSR, provided the pre-training is open-set aware.
2. The additional training overhead is only 1.5%, far lower than competing methods, indicating high efficiency.
3. The two pseudo open-set generation strategies are complementary, with joint use yielding the best results.
4. The metric space established in Stage 1 is foundational to the success of transfer learning.

## Highlights & Insights

- **Paradigm shift**: This work is the first to demonstrate the viability of transfer learning for FSOSR.
- **Minimal overhead**: SOTA performance is achieved with only a 1.5% training increment.
- **Practical strategy**: Pseudo open-set sample generation requires no additional data.

## Limitations & Future Work

1. As a workshop paper, the experimental scope remains limited.
2. Validation on larger-scale datasets (e.g., full ImageNet) is insufficient.
3. The quality of pseudo open-set samples leaves room for improvement.
4. Integration with pre-trained vision-language models such as CLIP warrants further exploration.

## Related Work & Insights

- **ProtoNet**: A classical metric-learning approach.
- **PEELER**: A pioneering work on open-set few-shot learning.
- **SnaTCHer**: Performs open-set detection by "snatching" prototypes.
- **CLIP-based FSOSR**: Exploits the open-vocabulary capabilities of pre-trained VLMs.

## Rating

| Dimension | Score (1–5) |
|------|-----------|
| Novelty | 4 |
| Theoretical Depth | 3 |
| Experimental Thoroughness | 4 |
| Writing Quality | 3 |
| Practical Value | 4 |
| Overall Recommendation | 3.5 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Open-Insect: Benchmarking Open-Set Recognition of Novel Species in Biodiversity Monitoring](open-insect_benchmarking_open-set_recognition_of_novel_species_in_biodiversity_m.md)
- [\[NeurIPS 2025\] Impact of Dataset Properties on Membership Inference Vulnerability of Deep Transfer Learning](impact_of_dataset_properties_on_membership_inference_vulnerability_of_deep_trans.md)
- [\[NeurIPS 2025\] Towards Unsupervised Open-Set Graph Domain Adaptation via Dual Reprogramming](towards_unsupervised_open-set_graph_domain_adaptation_via_dual_reprogramming.md)
- [\[ICLR 2026\] Learnability and Privacy Vulnerability are Entangled in a Few Critical Weights](../../ICLR2026/ai_safety/learnability_and_privacy_vulnerability_are_entangled_in_a_few_critical_weights.md)
- [\[ICML 2026\] Calibrating Uncertainty for Zero-Shot Adversarial CLIP](../../ICML2026/ai_safety/calibrating_uncertainty_for_zero-shot_adversarial_clip.md)

</div>

<!-- RELATED:END -->
