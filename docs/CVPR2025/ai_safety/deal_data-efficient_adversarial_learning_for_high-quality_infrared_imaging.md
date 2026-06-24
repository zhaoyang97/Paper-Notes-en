---
title: >-
  [Paper Note] DEAL: Data-Efficient Adversarial Learning for High-Quality Infrared Imaging
description: >-
  [CVPR 2025][AI Safety][infrared imaging] This work proposes DEAL (Data-Efficient Adversarial Learning), an adversarial learning framework trained on only 50 clean infrared images. Through dynamic adversarial degradation synthesis and a dual-channel interaction network (Scale Transform + Spiking Neurons), it simultaneously addresses three types of infrared degradations (stripe noise, low resolution, and low contrast) with an ultra-lightweight parameter size of 0.96M.
tags:
  - "CVPR 2025"
  - "AI Safety"
  - "infrared imaging"
  - "adversarial learning"
  - "data-efficient"
  - "spiking neural network"
  - "degradation synthesis"
date: 2026-05-08
content_hash: db36b5ae289af539
---

# DEAL: Data-Efficient Adversarial Learning for High-Quality Infrared Imaging

**Conference**: CVPR 2025  
**arXiv**: [2503.00905](https://arxiv.org/abs/2503.00905)  
**Code**: [https://github.com/LiuZhu-CV/DEAL](https://github.com/LiuZhu-CV/DEAL)  
**Area**: AI Safety  
**Keywords**: infrared imaging, adversarial learning, data-efficient, spiking neural network, degradation synthesis

## TL;DR
This work proposes DEAL (Data-Efficient Adversarial Learning), an adversarial learning framework trained on only 50 clean infrared images. Through dynamic adversarial degradation synthesis and a dual-channel interaction network (Scale Transform + Spiking Neurons), it simultaneously addresses three types of infrared degradations (stripe noise, low resolution, and low contrast) with an ultra-lightweight parameter size of 0.96M.

## Background & Motivation

**Background**: Infrared imaging is crucial in scenarios such as security monitoring, autonomous driving, and military reconnaissance. However, constrained by the physical characteristics of focal plane array sensors, infrared images generally suffer from three types of degradation: stripe noise (non-uniform sensor response), low resolution (thermal radiation propagation characteristics), and low contrast (signal processing limitations).

**Limitations of Prior Work**: (1) Existing methods typically address only a single type of degradation and fail to handle composite degradation scenarios. (2) High-quality infrared training data is extremely scarce, as collecting paired data requires precise optical equipment and highly controlled environments. (3) Cascading different degradation-processing models leads to error accumulation.

**Key Challenge**: Infrared image enhancement requires a large volume of paired training data, which is extremely expensive to acquire. Concurrently, the three types of degradation are coupled (e.g., stripe noise superimposed on low resolution), making a simple cascaded approach highly ineffective.

**Goal**: How to train a unified model using an extremely small amount of data (50 images) to simultaneously handle three types of infrared degradation?

**Key Insight**: Adversarial learning—allowing a degradation generator to dynamically synthesize various degradation combinations, enabling the enhancement network to learn to handle all degradation types during adversarial game-playing. Combined with the spiking mechanism of SNNs, it is naturally suited for high-intensity anomaly detection in infrared images.

**Core Idea**: Utilizing adversarial learning to let the degradation generator and the enhancement network engage in a game, thereby dynamically synthesizing and handling composite infrared degradations. Coupled with SNN spike signals to precisely isolate stripe noise, the framework requires only 50 clean images for training.

## Method

### Overall Architecture
A hierarchical minimax framework: the enhancement network $\mathcal{N}_E$ minimizes reconstruction loss, while the degradation generator $\mathcal{N}_G$ maximizes reconstruction loss (generating more challenging degradations). The two are optimized alternately during training, allowing the enhancement network to ultimately learn to handle composite degradations of varying intensities.

### Key Designs

1. **Dynamic Adversarial Solution (DAS)**

    - **Function**: Dynamically synthesizes combinations of different degradation types and intensities.
    - **Mechanism**: $\hat{x}^{(i+1)} = \sum_{j=1}^N a_{ij} D_j(\hat{x}^i)$, where a classifier generates a learnable weight matrix $\mathbf{a}$ to control the intensity and type of degradation.
    - **Design Motivation**: Fixed degradations (e.g., average pooling or fully random) perform significantly worse than dynamic adversarial synthesis (VIF 0.747 vs 0.931).
    - **Alternative Training**: Warm-start the enhancement network first, then in each step, update the degradation generator (gradient ascent) before updating the enhancement network (gradient descent).

2. **Scale Transform Module (STM)**

    - **Function**: Handles multi-scale degradations through up/down-sampling.
    - **Mechanism**: Dense-connection-based up/down-sampling blocks with kernel sizes of 3 (residual blocks) and 7 (sampling layers).
    - **Design Motivation**: Infrared degradations affect features at different scales; thus, multi-scale processing is more effective than single-scale processing.

3. **Spiking-Guided Separation Module (SSM)**

    - **Function**: Achieves precise isolation of stripe noise using spiking neural networks.
    - **Mechanism**: Leaky Integrate-and-Fire (LIF) neurons encode spatial features into binary spike sequences, firing spikes when the intensity anomaly (such as stripes) exceeds a threshold.
    - **Threshold-related Batch Normalization** ensures stable training.
    - **Design Motivation**: Stripe noise manifests as high-intensity anomalies in space, and the "all-or-none" characteristic of spiking signals is naturally suited for such binary separation tasks.
    - **Ultra-lightweight parameters**: 0.96M vs competing methods BTC (22.4M) and KXNet (6.51M).

### Loss & Training
- $\mathcal{L} = \alpha\mathcal{L}_{pixel} + \beta\mathcal{L}_{SSIM}$, where $\alpha=0.75$, $\beta=1.1$.
- Degradation optimization: $-\mathcal{L}(\hat{y}; y) + \mathcal{L}(\hat{x}; x)$.
- Training data: 50 clean images from the M3FD dataset, 840 epochs.
- PyTorch on NVIDIA V100, SGD (degradation) + Adam (enhancement).

## Key Experimental Results

### Main Results

**Stripe noise removal (comparison of 7 methods):**

| Metric | Best Competing Method | DEAL | Rank |
|------|------------|------|------|
| MI (moderate stripe) | 3.23 | **3.397** | 1st |
| VIF (moderate) | 0.92 | **0.961** | 1st |
| MI (severe stripe) | 3.12 | **3.244** | 1st |
| VIF (severe) | 1.03 | **1.098** | 1st |

**Object Detection (YOLOv5 on M3FD):**

| Category | Degraded Image | Best Cascade | DEAL |
|------|----------|---------|------|
| People | 0.522 | 0.700 | **0.737** |
| Car | 0.345 | 0.772 | **0.826** |
| Bus | 0.137 | 0.678 | **0.726** |
| mAP | 0.262 | 0.612 | **0.660** |

### Ablation Study

| Degradation Strategy | VIF | Q^AB/F |
|---------|-----|--------|
| Average Pooling | 0.747 | 0.389 |
| Fully Random | 0.716 | 0.397 |
| **DAS (Ours)** | **0.931** | **0.482** |

| Data Volume | VIF | Q^AB/F |
|--------|-----|--------|
| 20 images | 0.829 | 0.445 |
| **50 images** | **0.931** | **0.482** |
| 100 images | 0.935 | 0.485 |

### Key Findings
- DAS improves VIF by 25% over fixed degradation (0.747 $\to$ 0.931), demonstrating the critical importance of dynamic degradation synthesis in adversarial learning.
- 50 images represent the optimal sweet spot; 100+ images yield marginal returns.
- With only 0.96M parameters, the model achieves performance comparable to a 22.4M parameter model, showing that the spiking mechanism in SSM is highly advantageous in terms of parameter efficiency.
- In downstream object detection, the mAP increases by 15.2% (0.262 $\to$ 0.660), indicating that image enhancement directly benefits high-level vision tasks.

## Highlights & Insights
- **Impressive data efficiency with only 50 images**: The data augmentation effect achieved through adversarial learning successfully compensates for the scarcity of data.
- **Clever application of SNN in image processing**: The binary nature of spiking signals is naturally suited for detecting and isolating stripes (characterized by anomalous high-intensity signals).
- **Unified model for three degradations**: It avoids the error accumulation typical of cascaded approaches.
- The dynamic degradation synthesis strategy can be transferred to other data-scarce image enhancement tasks.

## Limitations & Future Work
- Validated only on infrared scenarios, limiting the immediate scope of application.
- Training for 840 epochs still requires some time.
- The comparison lacks recent Transformer-based image restoration methods.
- The spike threshold in SSM is manually set; an adaptive threshold might yield better performance.

## Related Work & Insights
- **vs Cascade Methods (LINF+SEID)**: Unified processing avoids error accumulation, improving mAP by +4.8%.
- **vs KXNet**: Parameter size of 0.96M vs 6.51M, delivering better performance while being more lightweight.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of SNN and adversarial learning for handling infrared degradation is highly novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Coordinated evaluation across three degradations, composite degradation, and downstream object detection.
- Writing Quality: ⭐⭐⭐⭐ The problem is clearly motivated.
- Value: ⭐⭐⭐⭐ Direct application value to the infrared imaging community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Data-free Universal Adversarial Perturbation with Pseudo-Semantic Prior](data-free_universal_adversarial_perturbation_with_pseudo-semantic_prior.md)
- [\[CVPR 2025\] A Simple Data Augmentation for Feature Distribution Skewed Federated Learning](a_simple_data_augmentation_for_feature_distribution_skewed_federated_learning.md)
- [\[CVPR 2025\] Joint Out-of-Distribution Filtering and Data Discovery Active Learning](joint_out-of-distribution_filtering_and_data_discovery_active_learning.md)
- [\[ICCV 2025\] Towards Adversarial Robustness via Debiased High-Confidence Logit Alignment](../../ICCV2025/ai_safety/towards_adversarial_robustness_via_debiased_high-confidence_logit_alignment.md)
- [\[CVPR 2026\] Fractal Camouflage: A Bio-Inspired Approach for Multi-Scale Adversarial Attacks in the Infrared Domain](../../CVPR2026/ai_safety/fractal_camouflage_a_bio-inspired_approach_for_multi-scale_adversarial_attacks_i.md)

</div>

<!-- RELATED:END -->
