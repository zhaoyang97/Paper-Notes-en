---
title: >-
  [Paper Note] A Simple Data Augmentation for Feature Distribution Skewed Federated Learning
description: >-
  [CVPR 2025][AI Safety][Federated Learning] Proposes FedRDN—an extremely simple data augmentation method for federated learning. During training, it randomly uses the channel-wise mean/standard deviation from other clients for data normalization (instead of relying fixedly on local statistics). Requiring only a few lines of code, it significantly mitigates the feature distribution skew problem and consistently improves performance across multiple FL methods.
tags:
  - "CVPR 2025"
  - "AI Safety"
  - "Federated Learning"
  - "Feature Distribution Skew"
  - "Data Normalization Augmentation"
  - "Plug-and-Play"
  - "Non-IID Data"
date: 2026-05-08
content_hash: 90541b93ebab2f3f
---

# A Simple Data Augmentation for Feature Distribution Skewed Federated Learning

**Conference**: CVPR 2025  
**arXiv**: [2306.09363](https://arxiv.org/abs/2306.09363)  
**Code**: Coming soon  
**Area**: AI Safety / Federated Learning  
**Keywords**: Federated Learning, Feature Distribution Skew, Data Normalization Augmentation, Plug-and-Play, Non-IID Data

## TL;DR

Proposes FedRDN—an extremely simple data augmentation method for federated learning. During training, it randomly uses the channel-wise mean/standard deviation from other clients for data normalization (instead of relying fixedly on local statistics). Requiring only a few lines of code, it significantly mitigates the feature distribution skew problem and consistently improves performance across multiple FL methods.

## Background & Motivation

**Background**: The non-IID issue in federated learning is a core challenge. Feature distribution skew is a common scenario where data across different clients originate from different devices or environments (e.g., MRI scanners from different hospitals), leading to differences in $P_k(x)$ while $P_k(y|x)$ remains the same.

**Limitations of Prior Work**: Existing methods (such as FedBN, HarmoFL, and FedFA) tackle feature skew at the model level (e.g., BN parameters, frequency-domain features, and feature augmentation), overlooking the data itself. FedFA requires modifying the network architecture and incurs extra communication overhead. FedMix, the only input-level augmentation, requires sharing average images, posing privacy risks.

**Key Challenge**: The root cause of feature skew is the distinct data distribution across different clients, yet clients in FL cannot directly access data from other clients. How can global distribution information be injected without compromising privacy?

**Core Idea**: During the training phase, a pair of statistics is randomly selected from the collection of all clients' statistics $\{(\mu^k, \sigma^k)\}_{k=1}^K$ to perform data normalization: $\hat{x}_i^k = (x_i^k - \mu^j) / \sigma^j$. This exposes the model to the same sample under multiple distributions. Only dataset-level statistics are shared (from which individual images cannot be reconstructed), ensuring privacy security.

## Method

### Overall Architecture

Pre-training phase: Each client calculates the channel-wise mean/standard deviation of local data $\rightarrow$ sends them to the server for aggregation $\rightarrow$ the server distributes them to all clients. Training phase: The normalization statistics for each sample are **randomly selected** from the set of statistics of all clients (instead of being fixedly set to local statistics). Testing phase: Local statistics are used for normalization.

### Key Designs

1. **Data Distribution Statistics**:

    - Each client $k$ computes channel-wise $\mu^k, \sigma^k \in \mathbb{R}^C$.
    - This requires only one communication round, after which the statistics are reused throughout the entire training process.

2. **Random Data Normalization**:

    - For each sample in every epoch, a client's statistics are randomly selected for normalization.
    - Over multiple epochs, each sample is "seen" under various distributions, implicitly injecting global information.
    - Key point: The selection is performed **independently and randomly** for each individual image, rather than applying the same statistics to the entire batch.

3. **Plug-and-Play Design**:

    - It only requires replacing `transforms.Normalize()` in `transforms.Compose()`.
    - It does not modify the network architecture, add training epochs, or increase computational overhead.
    - It can be combined with any FL method: FedAvg, FedProx, FedBN, etc.

### Privacy Security

Only dataset-level means and standard deviations (one scalar per channel) are shared, making it impossible to reconstruct any individual images. This is significantly more secure than FedMix, which shares average images.

## Key Experimental Results

### Main Results: Office-Caltech-10 with various FL methods + FedRDN

| Baseline Method | Original Avg | +Traditional Norm | +FedMix | +**FedRDN** |
|---------|:-:|:-:|:-:|:-:|
| FedAvg | 62.51 | 61.46↓ | 63.59↑1.1 | **69.80↑7.3** |
| FedProx | 61.84 | 62.57↑0.7 | 63.92↑2.1 | **69.71↑7.9** |
| FedNova | 60.71 | 63.15↑2.4 | 63.20↑2.5 | **69.40↑8.7** |

### Ablation Study (DomainNet)

| Baseline Method | Original Avg | +FedRDN |
|---------|:-:|:-:|
| FedAvg | 42.32 | 43.55↑1.2 |
| FedProx | 42.85 | 44.63↑1.8 |

### Key Findings

- **Consistent Improvements**: FedRDN brings positive gains across all tested FL methods with no negative cases.
- **Improvement Margins Far Exceed Other Augmentations**: On Office-Caltech-10, it achieves +7.3% vs. +1.1% for FedMix and -1.1% for traditional normalization.
- **Traditional Normalization Can Be Harmful**: In certain setups, fixed statistics actually degrade performance (FedAvg -1.05%), indicating that the key lies in the "randomness" rather than "normalization" itself.
- **Equally Effective on MRI Segmentation (ISIC Skin Cancer Detection)**: The AUC increases from 74.0 to 77.6.
- **Complementary to FedFA**: It can be stacked with FedFA because they operate at the input level and the feature level, respectively.

## Highlights & Insights

- **Extreme Simplicity**: The entire core of the method is a single line of code—changing `Normalize(fixed_mean, fixed_std)` to `Normalize(random_mean, random_std)`. However, the insight is profound.
- **Thinking about FL from a Data Perspective**: While the vast majority of FL research focuses on optimization/aggregation strategies, this paper returns to the data itself: "If distribution skew is caused by the data, why not handle the data directly?"
- **Randomness is Key**: Instead of normalizing with some "optimal" statistics, the method performs random selection. This allows the model to observe from multiple distributional perspectives, similar to the randomness principle in data augmentation.

## Limitations & Future Work

- It only validates feature distribution skew scenarios, leaving label distribution skew and joint skew untested.
- The statistics are calculated only once before training and are not dynamically updated as the model trains.
- The improvement margin on large-scale datasets (such as DomainNet 6 domains) is relatively small (+1.2%), possibly due to insufficient granularity of the statistics.
- Channel-wise statistics may be insufficient to capture complex distributional differences (such as spatial structure differences).

## Rating

- Novelty: ⭐⭐⭐⭐ Strikingly simple method with profound insight—the idea of using random normalization for distribution augmentation is highly novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated on 3 datasets, 6 FL methods, and compared with 4 augmentation methods, though it lacks verification on more non-IID scenarios.
- Writing Quality: ⭐⭐⭐⭐ The methodology is clearly described, and the pseudocode in the Algorithm is concise.
- Value: ⭐⭐⭐⭐⭐ Plug-and-play, zero extra overhead, privacy-secure, and consistent improvements—highly adopting-ready for FL practitioners.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Domain-Skewed Federated Learning with Feature Decoupling and Calibration](../../CVPR2026/ai_safety/domain-skewed_federated_learning_with_feature_decoupling_and_calibration.md)
- [\[CVPR 2025\] FedAWA: Adaptive Optimization of Aggregation Weights in Federated Learning Using Client Vectors](fedawa_adaptive_optimization_of_aggregation_weights_in_federated_learning_using_.md)
- [\[CVPR 2025\] Geometric Knowledge-Guided Localized Global Distribution Alignment for Federated Learning](geometric_knowledge-guided_localized_global_distribution_alignment_for_federated.md)
- [\[CVPR 2025\] Joint Out-of-Distribution Filtering and Data Discovery Active Learning](joint_out-of-distribution_filtering_and_data_discovery_active_learning.md)
- [\[ICCV 2025\] Client2Vec: Improving Federated Learning by Distribution Shifts Aware Client Indexing](../../ICCV2025/ai_safety/client2vec_improving_federated_learning_by_distribution_shifts_aware_client_inde.md)

</div>

<!-- RELATED:END -->
