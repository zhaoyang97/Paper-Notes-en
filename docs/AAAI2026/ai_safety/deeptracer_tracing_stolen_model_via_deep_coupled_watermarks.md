---
title: >-
  [Paper Note] DeepTracer: Tracing Stolen Model via Deep Coupled Watermarks
description: >-
  [AAAI 2026][AI Safety][Model watermarking] This paper proposes DeepTracer, a robust watermarking framework that achieves deep coupling between the watermark task and the main task through adaptive source-class selection (K-Means clustering for feature space coverage) + same-class coupling loss (aligning watermark samples with target-class samples in output space) + two-stage key sample filtering. Under 6 model stealing attacks (including hard-label and data-free settings), the watermark success rate averages 77–100%, substantially outperforming existing methods.
tags:
  - AAAI 2026
  - AI Safety
  - Model watermarking
  - model stealing
  - deep coupled watermark
  - black-box verification
  - intellectual property protection
date: 2026-05-08
content_hash: 7875e544b98cf466
---

# DeepTracer: Tracing Stolen Model via Deep Coupled Watermarks

**Conference**: AAAI 2026
**arXiv**: [2511.08985](https://arxiv.org/abs/2511.08985)
**Code**: [GitHub](https://github.com/yangyunfei16/DeepTracer)
**Area**: AI Security / Model Copyright Protection
**Keywords**: Model watermarking, model stealing, deep coupled watermark, black-box verification, intellectual property protection

## TL;DR

This paper proposes DeepTracer, a robust watermarking framework that achieves deep coupling between the watermark task and the main task through adaptive source-class selection (K-Means clustering for feature space coverage) + same-class coupling loss (aligning watermark samples with target-class samples in output space) + two-stage key sample filtering. Under 6 model stealing attacks (including hard-label and data-free settings), the watermark success rate averages 77–100%, substantially outperforming existing methods.

## Background & Motivation

- **State of the Field**: Model watermarking is the mainstream DNN intellectual property protection approach. Black-box watermarking embeds special trigger samples into training data so that the model learns watermark behavior, which can later be verified by querying the deployed model.
- **Limitations of Prior Work**: Existing watermarking methods are prone to failure against model stealing attacks. Stolen models trained by querying the victim model tend not to inherit watermark behavior—especially under strong attack scenarios such as hard-label (only top-1 labels available) and data-free (no real data available).
- **Root Cause**: Traditional watermarking methods (OOD types such as Abstract and Noise) use trigger samples whose feature distributions do not overlap with the main task, activating different neuron regions in overparameterized networks. Since model stealing focuses on reproducing main task functionality, OOD watermark behavior is naturally discarded. In-distribution (ID) approaches such as MEA-Defender improve distributional alignment but still achieve insufficient coupling—designing watermarks at the input feature level is not enough; coupling must also be established in the output space.
- **Starting Point**: If the watermark task distribution is entirely a subset of the main task distribution and is deeply coupled with the main task along the complete feature-to-output pipeline, a stolen model cannot learn the main task without simultaneously learning the watermark task.

## Method

### Overall Architecture

DeepTracer consists of four stages: (1) watermark sample construction—adaptively selecting 4 source classes and composing concatenated samples; (2) coupled watermark embedding—using same-class coupling loss to reinforce output-space coupling between the main task and the watermark task; (3) key sample generation—two-stage filtering to select the most reliable verification samples; (4) copyright verification—black-box querying of the suspect model for ownership determination.

### Key Designs

1. **Adaptive Source-Class Selection and Watermark Sample Construction**

    - Function: Select 4 most representative classes as watermark source classes; resize their samples to 1/4 of the original size and concatenate them.
    - Mechanism: Compute the feature centroid of each class $c_j = \frac{1}{N_j}\sum f_i^j$, cluster centroids using K-Means ($K=4$), and select the class closest to each cluster center as a source class. The target label is the class for which the benign model yields the lowest prediction probability on the watermark samples.
    - Design Motivation: Randomly selected source classes may concentrate in a single region of the feature space, failing to uniformly cover the main task distribution. K-Means clustering ensures source classes are spread across the entire feature space, making the watermark a genuine distributional subset of the main task.

2. **Same-Class Coupling Loss**

    - Function: Force watermark samples to align with normal samples of their target class in the output space.
    - Mechanism: The intra-class loss $L_{intra} = \frac{1}{N}\sum\|f_i - c_{y_i}\|_2^2$ pulls same-label samples toward their class centroid; the inter-class loss $L_{inter} = \frac{1}{N}\sum\sum \max(0, margin - \|f_i - c_j\|_2)^2$ pushes apart samples of different labels.
    - Total training loss: $L = L_{pri} + \lambda_1 L_{wm} + \lambda_2 L_{cpl}$
    - Design Motivation: Coupling at the input feature level alone is insufficient—model stealing mimics the victim at the output level, so the watermark must be bound to the main task in output space. When the last-layer feature distribution of watermark samples completely overlaps with that of normal samples of the target class, any stolen model that successfully reproduces the main task is compelled to simultaneously reproduce the watermark behavior.

3. **Two-Stage Key Sample Filtering**

    - Stage 1: From the initial watermark sample set $S_0$, retain samples that simultaneously satisfy three conditions: (a) the victim model correctly predicts the watermark label; (b) a surrogate model simulating the stealing process also correctly predicts the watermark label; (c) the benign model does not predict the watermark label.
    - Stage 2: From the Stage 1 results, select the Top-$M$ samples for which the surrogate model assigns the highest confidence to the target label.
    - Design Motivation: Not all watermark samples are equally effective. Pre-validating through a surrogate model that simulates real stealing selects samples most likely to succeed on actual stolen models, improving verification reliability.

4. **Evidence of Deep Coupling—Activation Map Analysis**

    - Activation heatmaps of Abstract, MEA-Defender, and DeepTracer are compared.
    - Abstract watermark samples activate entirely different neuron regions from normal samples.
    - DeepTracer watermark samples activate nearly identical regions as normal samples—direct evidence of deep coupling.

### Loss & Training

- Main task loss $L_{pri}$ (cross-entropy) + watermark classification loss $L_{wm}$ + same-class coupling loss $L_{cpl} = \lambda_3 L_{intra} + \lambda_4 L_{inter}$
- Models evaluated: VGG-like / ResNet-18 / ResNet-34 on FMNIST / CIFAR-10 / CIFAR-100
- Stealing attacks covered: JBDA (seed-sample-based), Knockoff (substitute-data-based), DFME / DFMS-HL (data-free)
- Both soft-label and hard-label attack settings are evaluated

## Key Experimental Results

### Main Results Table (Watermark Impact on Main Task, CIFAR-10)

| Method | Benign Model Acc | Watermarked Model Acc (Δ) | Watermarked Model WSR | Benign Model WSR (False Positive) |
|--------|-----------------|--------------------------|----------------------|-----------------------------------|
| EWE | 85.12 | 80.98 (−4.14) | 19.44 | 0.91 |
| MEA-Defender | 84.26 | 83.44 (−0.82) | 91.82 | 2.01 |
| **DeepTracer** | 85.31 | **85.59 (+0.28)** | **100.00** | **0.00** |

### Robustness Against Stealing Table (FMNIST, JBDA Attack)

| Method | Soft-Label WSR | Hard-Label WSR |
|--------|---------------|---------------|
| Abstract | 19.04 | 18.30 |
| MEA-Defender | 46.17 | 8.61 |
| **DeepTracer** | **91.65** | **86.90** |

### Key Findings

- **Zero main task accuracy loss**: DeepTracer's accuracy on CIFAR-10 even increases slightly by 0.28% after watermarking—the same-class coupling loss acts as a regularizer.
- **Largest advantage under hard-label attacks**: DeepTracer achieves a WSR of 86.90% under hard-label attacks, compared to only 8.61% for MEA-Defender—a 10× gap.
- **Zero false positive rate**: The benign model WSR is 0.00%, indicating the watermark does not produce false alarms on non-stolen models.
- **Robust under data-free attacks**: WSR remains 100% under DFME/DFMS-HL data-free stealing, where other methods generally fail.
- **Activation map validation**: DeepTracer watermark sample activations are highly consistent with those of normal samples, confirming deep coupling.

## Highlights & Insights

- **Progressive coupling from distributional to output space**: DeepTracer not only makes the watermark a distributional subset of the main task at the input feature level, but also enforces alignment in output space via same-class coupling loss. This complete, multi-level coupling is key to high robustness.
- **K-Means adaptive source-class selection**: Simple yet effective—elevating watermark design from random selection to strategically space-covering selection. This idea is transferable to other scenarios requiring representative sample selection.
- **Watermarking as regularization**: DeepTracer achieves a slight accuracy improvement after watermarking, demonstrating that deeply coupled watermarks can be compatible with—or even enhance—the main task, which prior watermarking methods could not achieve.

## Limitations & Future Work

- Access to a benign model (i.e., an unwatermarked model of the same architecture) is required for target label selection and sample filtering.
- The watermark sample pattern—four sub-images concatenated—may be targeted by adaptive defenses if an attacker is aware of this construction scheme.
- Experiments are limited in scale (up to CIFAR-100 with 100 classes); large-scale ImageNet experiments are absent.
- The number of source classes is fixed at 4; whether different datasets require different values of $K$ is not thoroughly investigated.

## Related Work & Insights

- **vs. OOD watermarks (Abstract / Noise / Unrelated)**: OOD watermarks activate independent neuron regions and are naturally lost after stealing; DeepTracer activates the same regions, making the watermark inseparable during stealing.
- **vs. MEA-Defender (ID watermarking SOTA)**: MEA-Defender couples only at the input feature level, whereas DeepTracer additionally couples in output space—the hard-label WSR gap from 8.61% to 86.90% demonstrates the necessity of output-space coupling.

## Rating

- Novelty: ⭐⭐⭐⭐ The complete deep coupling design chain (feature coverage → output alignment → sample filtering) constitutes a systematic innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Six stealing attacks × soft/hard-label × 3 datasets × 10 watermarking baselines—extremely comprehensive.
- Writing Quality: ⭐⭐⭐⭐ The logic from root-cause analysis of OOD watermark failure to DeepTracer's design is clearly presented.
- Value: ⭐⭐⭐⭐ Directly applicable to model intellectual property protection, particularly in MLaaS scenarios.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] Fair Model-Based Clustering](fair_model-based_clustering.md)
- [\[ICLR 2026\] Traceable Black-box Watermarks for Federated Learning](../../ICLR2026/ai_safety/traceable_black-box_watermarks_for_federated_learning.md)
- [\[AAAI 2026\] Privacy Auditing of Multi-Domain Graph Pre-Trained Model under Membership Inference Attack](privacy_auditing_of_multi-domain_graph_pre-trained_model_under_membership_infere.md)
- [\[CVPR 2026\] Monte Carlo Stochastic Depth for Uncertainty Estimation in Deep Learning](../../CVPR2026/ai_safety/mcsd_uncertainty_estimation.md)
- [\[CVPR 2026\] FedRE: A Representation Entanglement Framework for Model-Heterogeneous Federated Learning](../../CVPR2026/ai_safety/fedre_a_representation_entanglement_framework_for_model-heterogeneous_federated_.md)

<!-- RELATED:END -->
