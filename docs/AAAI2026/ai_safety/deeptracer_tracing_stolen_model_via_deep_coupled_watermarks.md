---
title: >-
  [Paper Note] DeepTracer: Tracing Stolen Model via Deep Coupled Watermarks
description: >-
  [AAAI 2026][AI Safety][Model Watermarking] This work proposes DeepTracer, a robust watermarking framework. By leveraging adaptive source class selection (covering the feature space via K-Means clustering), same-class coupling loss (reducing the distance between watermark samples and the target class in the output space), and two-stage key sample filtering, the watermarking task is deeply coupled with the primary task. DeepTracer achieves an average watermark success rate of 7…
tags:
  - "AAAI 2026"
  - "AI Safety"
  - "Model Watermarking"
  - "Model Stealing"
  - "Deep Coupled Watermarks"
  - "Black-Box Verification"
  - "Intellectual Property Protection"
date: 2026-05-08
content_hash: 9c259d90743282f7
---

# DeepTracer: Tracing Stolen Model via Deep Coupled Watermarks

**Conference**: AAAI 2026  
**arXiv**: [2511.08985](https://arxiv.org/abs/2511.08985)  
**Code**: [GitHub](https://github.com/yangyunfei16/DeepTracer)  
**Area**: AI Security / Model Copyright Protection  
**Keywords**: Model Watermarking, Model Stealing, Deep Coupled Watermarks, Black-Box Verification, Intellectual Property Protection

## TL;DR

This work proposes DeepTracer, a robust watermarking framework. By leveraging adaptive source class selection (covering the feature space via K-Means clustering), same-class coupling loss (reducing the distance between watermark samples and the target class in the output space), and two-stage key sample filtering, the watermarking task is deeply coupled with the primary task. DeepTracer achieves an average watermark success rate of 77-100% under 6 types of model stealing attacks (including hard-label and data-free), significantly outperforming existing methods.

## Background & Motivation

- **Background**: Model watermarking is currently the primary solution for protecting DNN intellectual property. Black-box watermarking methods embed watermark behavior by mixing specially marked samples into the training data, and verify copyright through queries after deployment.
- **Limitations of Prior Work**: Existing watermarking methods easily fail when encountering model stealing attacks. The stealing model trains a surrogate model by querying the victim model, during which the watermark behavior often fails to transfer—especially under strong attack scenarios such as hard-label (relying only on top-1 labels) and data-free (without real data).
- **Key Challenge**: The watermark features used by traditional watermarking methods (OOD categories like Abstract or Noise) do not overlap with the feature distribution of the primary task, activating different neural regions in over-parameterized networks. Model stealing attacks focus on replicating primary task functionality and naturally ignore OOD watermarks. Although ID-based methods (e.g., MEA-Defender) improve distribution alignment, the coupling between the watermark and the primary task remains insufficient—designing watermarks purely at the input feature level is far from enough; coupling must also be achieved in the output space.
- **Key Insight**: If the distribution of the watermarking task is entirely a subset of the main task distribution, and is deeply coupled with the main task along the entire chain from features to outputs, the stealer will inevitably learn the watermarking task while learning the main task.

## Method

### Overall Architecture

DeepTracer comprises four phases: (1) Watermark Sample Construction—adaptively selecting 4 source classes and combining them into spliced samples; (2) Coupled Watermark Embedding—strengthening the output space coupling between the main task and the watermarking task using same-class coupling loss; (3) Key Sample Generation—filtering to select the most reliable verification samples in two stages; (4) Copyright Verification—querying suspect models in a black-box manner to determine copyright.

### Key Designs

1. **Adaptive Source Class Selection and Watermark Sample Construction**

    - **Function**: Selects four most representative classes as watermark source classes, scales their samples to 1/4 of the original size, and splices them.
    - **Mechanism**: Computes the feature centroid $c_j = \frac{1}{N_j}\sum f_i^j$ for each class, performs K-Means ($K=4$) clustering on these centroids, and selects the class closest to the cluster center in each cluster as a source class. The target label is chosen as the class with the lowest predicted probability by the benign model on the watermark samples.
    - **Design Motivation**: Randomly selecting source classes might concentrate them in a single area of the feature space, failing to uniformly cover the primary task distribution. K-Means clustering ensures that the source classes are scattered across the entire feature space, making the watermark truly a distribution subset of the main task.

2. **Same-Class Coupling Loss**

    - **Function**: Enforces alignment between watermark samples and normal samples of their target class in the output space.
    - **Mechanism**: The intra-class loss $L_{intra} = \frac{1}{N}\sum\|f_i - c_{y_i}\|_2^2$ pulls samples with the same label closer to the class centroid; the inter-class loss $L_{inter} = \frac{1}{N}\sum\sum \max(0, margin - \|f_i - c_j\|_2)^2$ pushes samples with different labels apart.
    - **Total Training Loss**: $L = L_{pri} + \lambda_1 L_{wm} + \lambda_2 L_{cpl}$
    - **Design Motivation**: Coupling merely at the input feature level is insufficient—since stealing attacks imitate at the output level, the binding between the watermark and the primary task must be established in the output space. When the feature distribution of watermark samples in the last layer completely overlaps with that of normal samples from the target class, any stealing model that successfully duplicates the primary task is forced to replicate the watermark behavior simultaneously.

3. **Two-Stage Key Sample Filtering**

    - **First Stage**: From the initial watermark sample set $S_0$, filter samples that simultaneously satisfy three conditions: (a) the victim model correctly identifies the watermark label; (b) the surrogate model simulating the theft also correctly identifies it; (c) the benign model does not identify it as the watermark label.
    - **Second Stage**: Select the Top-M samples with the highest confidence in the target label from the surrogate model among the results from the first stage.
    - **Design Motivation**: Not all watermark samples are equally effective. By using a surrogate model to simulate real stealing scenarios for pre-verification, the most reliable verification samples likely to succeed on the actual stolen model are selected, improving verification reliability.

4. **Deeply Coupled Verification—Heatmap Evidence**

    - Compare the neural network activation heatmaps of Abstract, MEA-Defender, and DeepTracer.
    - Abstract watermark samples activate completely different neural regions compared to normal samples.
    - DeepTracer's watermark samples activate almost identical regions as normal samples—providing visual evidence of deep coupling.

### Loss & Training

- Primary task loss $L_{pri}$ (cross-entropy) + watermark classification loss $L_{wm}$ + same-class coupling loss $L_{cpl} = \lambda_3 L_{intra} + \lambda_4 L_{inter}$
- Evaluated on FMNIST/CIFAR-10/CIFAR-100 using models such as VGG-like, ResNet-18, and ResNet-34.
- Stealing attack coverage: JBDA (seed sample-based), Knockoff (surrogate data-based), DFME/DFMS-HL (data-free based).
- Evaluated under both soft-label and hard-label attack settings.

## Key Experimental Results

### Main Results (Watermark Impact on Main Task, CIFAR-10)

| Method | Benign Model Acc | Watermarked Model Acc (Δ) | Watermarked Model WSR | Benign Model WSR (False Positive) |
|------|-----------|---------------|-----------|---------------------|
| EWE | 85.12 | 80.98 (-4.14) | 19.44 | 0.91 |
| MEA-Defender | 84.26 | 83.44 (-0.82) | 91.82 | 2.01 |
| **DeepTracer** | 85.31 | **85.59 (+0.28)** | **100.00** | **0.00** |

### Stealing Resistance (FMNIST, JBDA Attack)

| Method | Soft-Label WSR | Hard-Label WSR |
|------|---------------|---------------|
| Abstract | 19.04 | 18.30 |
| MEA-Defender | 46.17 | 8.61 |
| **DeepTracer** | **91.65** | **86.90** |

### Key Findings

- **Zero Primary Task Accuracy Loss**: The accuracy of DeepTracer on CIFAR-10 even slightly increases by 0.28% after watermarking—showing that the same-class coupling loss plays a role in regularization.
- **Most Significant Performance under Hard-Label Attacks**: DeepTracer achieves a WSR of 86.90% under hard-label attacks, whereas MEA-Defender only obtains 8.61%—a 10x performance gap.
- **Zero False Positive Rate**: The WSR of the benign model is 0.00%, showing that the watermark does not cause false alarms on untampered models.
- **Robust Under Data-Free Attacks**: The WSR remains at 100% under DFME/DFMS-HL data-free stealing attacks, while other methods generally fail.
- **Intuitive Heatmap Verification**: The activation of DeepTracer watermark samples is highly consistent with that of normal samples, confirming deep coupling.

## Highlights & Insights

- **Evolution from "Distribution Coupling" to "Output Coupling"**: It not only makes the watermark a subset of the primary task at the input feature distribution level but also enforces alignment in the output space using same-class coupling loss. This complete coupling from shallow to deep is the key to high robustness.
- **K-Means Adaptive Source Class Selection**: Simple yet effective—elevating watermark design from random selection to strategic spatial coverage, which can be transferred to other scenarios requiring representative sample selection.
- **Watermark as Regularization**: The accuracy of DeepTracer increases rather than decreases after fine-tuning, indicating that deeply coupled watermarks can be compatible with or even enhance the primary task—which was unachievable by previous watermarking methods.

## Limitations & Future Work

- Requires access to a benign model (i.e., an unwatermarked model with the same architecture) to choose the target label and filter samples.
- The watermark samples are in a visual pattern of 4 spliced sub-images; if an attacker knows this construction scheme, targeted defenses might be developed.
- The experimental scale is limited (up to CIFAR-100 with 100 classes); large-scale experiments on ImageNet are missing.
- The number of source classes is fixed to 4; whether different K values are needed for different datasets has not been thoroughly studied.

## Related Work & Insights

- **vs. OOD Watermarking (Abstract/Noise/Unrelated)**: OOD watermarks activate independent neural regions and are naturally lost after stealing; DeepTracer activates the same regions, making it inseparable during theft.
- **vs. MEA-Defender (State-of-the-Art ID Watermarking)**: MEA-Defender only couples at the input feature level, while DeepTracer further couples in the output space. The performance gap in hard-label WSR (from 8.61% to 86.90%) demonstrates the necessity of output space coupling.

## Rating

- Novelty: ⭐⭐⭐⭐ The complete design chain of deep coupling (feature coverage $\rightarrow$ output alignment $\rightarrow$ sample filtering) represents a systemic innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely comprehensive, covering 6 stealing attacks $\times$ soft/hard-label $\times$ 3 datasets $\times$ 10 watermarking baselines.
- Writing Quality: ⭐⭐⭐⭐ Clear logic, from the root cause analysis of OOD watermarking failure to the design of DeepTracer.
- Value: ⭐⭐⭐⭐ Direct practical value for model intellectual property protection, especially in MLaaS scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Traceable Black-box Watermarks for Federated Learning](../../ICLR2026/ai_safety/traceable_black-box_watermarks_for_federated_learning.md)
- [\[CVPR 2026\] RAVEN: Erasing Invisible Watermarks via Novel View Synthesis](../../CVPR2026/ai_safety/raven_erasing_invisible_watermarks_via_novel_view_synthesis.md)
- [\[AAAI 2026\] Fair Model-Based Clustering](fair_model-based_clustering.md)
- [\[ICLR 2026\] NoisePrints: Distortion-Free Watermarks for Authorship in Private Diffusion Models](../../ICLR2026/ai_safety/noiseprints_distortion-free_watermarks_for_authorship_in_private_diffusion_model.md)
- [\[ICLR 2026\] On Optimal Hyperparameters for Differentially Private Deep Transfer Learning](../../ICLR2026/ai_safety/on_optimal_hyperparameters_for_differentially_private_deep_transfer_learning.md)

</div>

<!-- RELATED:END -->
