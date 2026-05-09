---
title: >-
  [Paper Note] FecalFed: Privacy-Preserving Poultry Disease Detection via Federated Learning
description: >-
  [CVPR 2026][AI Safety][Federated Learning] This paper proposes FecalFed, a privacy-preserving federated learning framework that first removes 46.89% duplicate contamination from public poultry fecal datasets via a dual-hash deduplication pipeline and releases a clean benchmark of 8,770 images (poultry-fecal-fl). Under highly non-IID conditions (Dirichlet α=0.5), FedAdam + Swin-Small recovers accuracy from a collapsed 64.86% (single-farm) to 90.31%, only 4.79% below the centralized upper bound of 95.10%. The edge-optimized Swin-Tiny (28M parameters) still achieves 89.74%, providing an efficient and practical solution for on-farm deployment.
tags:
  - CVPR 2026
  - AI Safety
  - Federated Learning
  - Privacy Preservation
  - Poultry Disease Detection
  - Data Deduplication
  - Non-IID
  - Vision Transformer
date: 2026-05-08
content_hash: 2a4cee5122baf33a
---

# FecalFed: Privacy-Preserving Poultry Disease Detection via Federated Learning

**Conference**: CVPR 2026
**arXiv**: [2604.00559](https://arxiv.org/abs/2604.00559)
**Code**: None
**Area**: AI Safety / Privacy Preservation
**Keywords**: Federated Learning, Privacy Preservation, Poultry Disease Detection, Data Deduplication, Non-IID, Vision Transformer

## TL;DR

This paper proposes FecalFed, a privacy-preserving federated learning framework that first removes 46.89% duplicate contamination from public poultry fecal datasets via a dual-hash deduplication pipeline and releases a clean benchmark of 8,770 images (poultry-fecal-fl). Under highly non-IID conditions (Dirichlet α=0.5), FedAdam + Swin-Small recovers accuracy from a collapsed 64.86% (single-farm) to 90.31%, only 4.79% below the centralized upper bound of 95.10%. The edge-optimized Swin-Tiny (28M parameters) still achieves 89.74%, providing an efficient and practical solution for on-farm deployment.

## Background & Motivation

**Real-World Problem**: Highly pathogenic avian influenza (HPAI) continues to cause global outbreaks, while endemic infections such as coccidiosis, Newcastle disease (NCD), and salmonellosis impose massive annual economic losses. Traditional surveillance relies on laboratory methods such as PCR, which are time-consuming and costly. Deep learning has demonstrated high-accuracy, non-invasive diagnostic potential through fecal image analysis.

**Privacy Bottleneck**: Training robust AI models requires aggregating large volumes of data, but farms refuse to upload sensitive health data to central servers due to biosecurity risks, commercial interests, and reputational concerns. Data silos force each farm to train independently, but performance collapses severely under naturally heterogeneous distributions.

**Data Contamination Crisis**: Public datasets appear to circumvent privacy constraints, yet the authors discover serious undocumented contamination—analysis of over 16,000 public images reveals a 46.89% duplication rate, with 77.4% of images in the Roboflow dataset being downscaled copies of Zenodo originals, causing severe train/test leakage and inflated reported performance.

**Infeasibility of Single-Farm Training**: Under realistic non-IID conditions, isolated single-farm training yields only 64.86% accuracy (±24.95%), more than 30 percentage points below the centralized baseline of 95.10%, with extremely high variance that makes reliable disease diagnosis impossible.

**Mechanism**: Federated learning enables a "data stays local, models travel" paradigm—each farm retains its raw data and only communicates model weight updates to the central server, preserving privacy while recovering the performance benefits of collaborative training.

## Method

### Overall Architecture

FecalFed consists of two main modules: (1) a data cleaning pipeline—multi-source public data aggregation → dual-hash perceptual deduplication → standardized preprocessing → release of the clean benchmark poultry-fecal-fl; and (2) a cross-farm federated learning framework—orchestrated via the Flower (flwr) framework, where 10 simulated farms train locally under non-IID conditions and the central server performs adaptive aggregation. Raw fecal images remain on local farms at all times; only classification head weights are transmitted over the network.

### Key Designs

**1. Dual-Algorithm Perceptual Hash Deduplication Pipeline**

- **Function**: Identifies and removes near-duplicate images from 16,513 raw images aggregated across multiple public repositories, eliminating data leakage.
- **Core Idea**: Each image is simultaneously hashed with a 256-bit Average Hash (aHash, capturing macroscopic brightness patterns) and a Perceptual Hash (pHash, capturing frequency-domain features via discrete cosine transform). An image pair is flagged as a duplicate when both Hamming distances satisfy: $D_{aHash}(x,y) \leq 5 \wedge D_{pHash}(x,y) \leq 5$
- **Design Motivation**: A single hash algorithm is prone to missed detections or false positives; cross-validation with two algorithms improves recall while controlling false positives. The strict threshold (≤5) ensures only genuine near-duplicates are removed rather than visually similar but distinct samples.
- **Key Findings**: The deduplication rate reaches 46.89% (7,743 images removed); 77.4% of the Roboflow dataset consists of downsampled copies of Zenodo data. Additionally, 19 cross-label conflict groups (identical images labeled under different disease categories across repositories) were identified and fully removed. The resulting clean dataset contains 8,770 unique images across 4 classes (Healthy / Coccidiosis / NCD / Salmonella).

**2. Dirichlet Non-IID Data Partitioning**

- **Function**: Partitions the clean dataset across 10 simulated farms, generating highly heterogeneous label distributions.
- **Core Idea**: Dirichlet distribution (concentration parameter α=0.5) is used to sample per-client class proportions; smaller α values produce more skewed distributions. Consequently, some farms hold almost exclusively salmonellosis samples while others are dominated by healthy samples.
- **Design Motivation**: In real agricultural settings, disease outbreaks are highly localized—one region may experience a coccidiosis outbreak while others remain entirely healthy. The IID assumption is therefore severely at odds with reality. The α=0.5 setting creates sufficiently extreme distributional skew to stress-test the robustness of federated algorithms.

**3. Frozen Backbone + Classification Head Fine-Tuning Only**

- **Function**: Freezes the ImageNet-pretrained Swin/ViT feature extraction backbone; only classification head parameters are updated during federated training.
- **Core Idea**: This substantially reduces the number of parameters communicated per round (from tens of millions to the few hundred thousand of the classification head), while lowering memory and computational requirements on local devices.
- **Design Motivation**: Edge devices on farms (smartphones, NVIDIA Jetson Nano, and similar embedded systems) lack the computational power and memory to fine-tune an entire Vision Transformer. Freezing the backbone makes local training on edge devices feasible with minimal communication overhead—manageable even over low-bandwidth rural networks.

**4. FedAdam Adaptive Server-Side Optimization**

- **Function**: Replaces the simple weighted averaging of standard FedAvg by feeding aggregated pseudo-gradients into a server-side Adam optimizer.
- **Core Idea**: Under non-IID conditions, client update directions diverge significantly, and naive weighted averaging can cause the global model to diverge or stagnate—particularly for larger models. FedAdam uses first- and second-moment estimates to adaptively adjust per-parameter update magnitudes, stabilizing aggregation under non-IID settings.
- **Design Motivation**: Proximal regularization methods such as FedProx may interfere with the fine-tuning of pretrained models, whereas FedAdam operates server-side without modifying the client's local training process, making it more compatible with pretrained weights.
- **Key Hyperparameters**: Server learning rate η=0.1, moment decay β₁=0.9, β₂=0.99, adaptivity τ=0.001.

### Loss & Training

- **Loss Function**: Standard cross-entropy loss.
- **Client Sampling**: 50% of clients (5 out of 10 farms) are randomly sampled per round.
- **Local Training**: E=1 local epoch per round, batch size=256, simulated on an NVIDIA A100.
- **Global Communication Rounds**: 10 rounds by default; extended to 20 rounds in ablation experiments.
- **Data Preprocessing**: Uniform resize to 224×224; training augmentations include random resized crop, horizontal/vertical flips, random rotation (≤30°), and color jitter (brightness/contrast/saturation variance 0.2, hue variance 0.1); normalization using ImageNet mean and standard deviation.
- **Evaluated Architectures**: Swin-Small (50M), ViT-B/16 (86M), Swin-Tiny (28M), ViT-S/16 (22M).

## Key Experimental Results

### Main Results: Model Performance Across Training Paradigms

| Model | Params | Single-Farm (Non-IID) | Centralized (Upper Bound) | FL (FedAvg) | FL (FedAdam) | Best FL vs. Centralized |
|---|---|---|---|---|---|---|
| Swin-Small | 50M | 64.86% ±24.95% | 95.10% | 89.74% | **90.31%** | -4.79% |
| ViT-B/16 | 86M | 60.97% ±23.05% | 94.81% | 88.77% | **90.02%** | -4.79% |
| Swin-Tiny | 28M | 64.03% ±24.04% | 93.04% | 86.89% | **89.74%** | -3.30% |
| ViT-S/16 | 22M | 65.87% ±22.06% | 92.99% | **89.28%** | 85.12% | -3.71% |

### Ablation Study: Effect of Communication Rounds on Convergence (ViT-B/16 + FedAvg)

| Communication Rounds | Test Accuracy | Notes |
|---|---|---|
| 5 rounds | 80.79% | Early convergence; performance not yet fully realized |
| 10 rounds | 88.77% | Default configuration; majority of performance recovered |
| 20 rounds | 91.05% | Continued improvement without stagnation; exceeds FedAdam at 10 rounds (90.02%) |

### Data Deduplication Statistics

| Metric | Value |
|---|---|
| Raw aggregated images | 16,513 |
| Unique images after deduplication | 8,770 |
| Overall duplication rate | 46.89% |
| Duplication rate in Roboflow dataset | 77.4% |
| Cross-label conflict groups | 19 |
| Disease categories | 4 (Healthy / Coccidiosis / NCD / Salmonella) |

### Key Findings

1. **Single-farm training collapses universally**: All architectures drop to the 60–66% range under α=0.5 non-IID conditions, with variance as high as ±22–25%, demonstrating that isolated training is fundamentally infeasible under heterogeneous data.
2. **FedAdam vs. FedAvg depends on model scale**: Larger models (Swin-Small 50M, ViT-B/16 86M) benefit more from FedAdam (+0.57%, +1.25%), whereas the smaller ViT-S/16 performs better with FedAvg (89.28% vs. 85.12%), suggesting that adaptive optimizers may overfit in low-parameter regimes.
3. **Swin-Tiny is the optimal choice for edge deployment**: At 28M parameters and 89.74% accuracy, it trails the 86M ViT-B/16 (90.02%) by only 0.28% while using less than one-third of the parameters, placing it at the Pareto frontier of accuracy–efficiency trade-offs.
4. **Increasing communication rounds remains effective**: Since only classification head weights are transmitted, communication bandwidth overhead is minimal. At 20 rounds, FedAvg achieves 91.05%, surpassing FedAdam at 10 rounds (90.02%), indicating that increasing rounds is a more practical strategy than switching algorithms in low-bandwidth settings.
5. **FecalFed achieves substantial recovery**: Taking Swin-Small as an example, FedAdam improves accuracy by +25.45 percentage points over single-farm training (64.86% → 90.31%), demonstrating the critical role of federated collaboration under non-IID conditions.

## Highlights & Insights

1. **Exposing data hygiene issues is of high value**: A 46.89% duplication rate and 77.4% synthetic data derived from downscaled copies indicate severely insufficient reliability in public agricultural AI datasets. This finding is an important warning to the broader agricultural AI community—any performance figures reported on these datasets may be substantially inflated.
2. **Complete end-to-end pipeline from data to deployment**: Data cleaning → non-IID partitioning → federated training → edge optimization addresses the critical challenge chain for real-world agricultural AI deployment holistically.
3. **Highly efficient communication**: Freezing the backbone and transmitting only classification head parameters renders the approach practical even over low-bandwidth rural networks.
4. **Rigorous experimental design**: The paper provides a complete comparison of the centralized upper bound, single-farm lower bound, and two federated strategies, clearly quantifying the contribution of each component.

## Limitations & Future Work

1. Only 4 disease classes are covered; the actual variety of poultry diseases is far larger, and fine-grained diagnostic requirements are not addressed.
2. The scale of 10 simulated clients is limited and does not adequately validate scalability to scenarios involving hundreds of farms.
3. Domain shift arising from different capture devices (phone models, lighting conditions, shooting angles) is not considered.
4. Although the frozen backbone strategy is communication-efficient, it limits the model's ability to adapt to domain-specific agricultural features; full fine-tuning may yield better performance in high-bandwidth settings.
5. Proximal regularization methods such as FedProx are excluded on theoretical grounds without empirical evaluation.
6. Stronger privacy guarantees such as differential privacy (DP) are not considered; the current scheme relies solely on not transmitting raw data.
7. All experiments are simulated on a single A100; latency and memory footprint on real edge devices are not validated.

## Related Work & Insights

- **FedAvg (McMahan et al., 2017)**: The foundational federated learning method, used as a baseline in this paper; results confirm its competitiveness even under non-IID conditions.
- **FedAdam (Reddi et al., 2020)**: Adaptive federated optimization; this paper demonstrates its stabilizing effect on large models under non-IID settings.
- **FedProx (Li et al., 2020)**: Proximal regularization for drift mitigation; not evaluated experimentally but discussed as potentially interfering with pretrained model fine-tuning.
- **Degu et al., 2023**: Smartphone CNN-based poultry fecal disease diagnosis, establishing the feasibility of mobile deployment.
- **Luong & Nguyen, 2024**: Interpretable poultry diagnostics via ViT + Integrated Gradients, but reliant on centralized data.
- **Insight**: Data cleaning (deduplication, leakage inspection) should be a standard first step in any AI dataset work involving multi-source aggregation. The effectiveness of federated learning under non-IID conditions depends on the alignment between model scale and optimization strategy.

## Rating

- **Novelty**: ⭐⭐⭐ — The combination of federated learning and disease classification is not novel per se, but the systematic exposure of data contamination and the release of a clean benchmark constitute independent contributions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Complete comparison across multiple architectures and strategies, plus communication round ablation; however, FedProx experiments and differential privacy evaluation are missing.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure, rich data, and well-motivated problem formulation.
- **Value**: ⭐⭐⭐⭐ — Provides the agricultural AI community with a reproducible federated learning baseline and a rigorously deduplicated public dataset.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] FedMeNF: Privacy-Preserving Federated Meta-Learning for Neural Fields](../../ICCV2025/ai_safety/fedmenf_privacy-preserving_federated_meta-learning_for_neural_fields.md)
- [\[CVPR 2026\] FedRE: A Representation Entanglement Framework for Model-Heterogeneous Federated Learning](fedre_a_representation_entanglement_framework_for_model-heterogeneous_federated_.md)
- [\[CVPR 2026\] Federated Active Learning Under Extreme Non-IID and Global Class Imbalance](federated_active_learning_extreme_noniid.md)
- [\[CVPR 2026\] FedDAP: Domain-Aware Prototype Learning for Federated Learning under Domain Shift](feddap_domain-aware_prototype_learning_for_federated_learning_under_domain_shift.md)
- [\[CVPR 2026\] Domain-Skewed Federated Learning with Feature Decoupling and Calibration](domain-skewed_federated_learning_with_feature_decoupling_and_calibration.md)

</div>

<!-- RELATED:END -->
