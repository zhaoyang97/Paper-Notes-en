---
title: >-
  [Paper Note] Detecting Backdoor Attacks in Federated Learning via Direction Alignment Inspection
description: >-
  [CVPR 2025][AI Safety][federated learning] Proposes the AlignIns defense method, which identifies malicious model updates in federated learning through dual-granularity direction alignment detection (global direction + fine-grained sign analysis), outperforming existing defense methods under both IID and non-IID settings.
tags:
  - "CVPR 2025"
  - "AI Safety"
  - "federated learning"
  - "backdoor attack"
  - "defense"
  - "direction alignment"
  - "sign analysis"
  - "anomaly detection"
date: 2026-05-08
content_hash: b75a70876bcdaf69
---

# Detecting Backdoor Attacks in Federated Learning via Direction Alignment Inspection

**Conference**: CVPR 2025  
**arXiv**: [2503.07978](https://arxiv.org/abs/2503.07978)  
**Code**: [JiiahaoXU/AlignIns](https://github.com/JiiahaoXU/AlignIns)  
**Institution**: University of Nevada, Reno  
**Area**: AI Security / Federated Learning  
**Keywords**: federated learning, backdoor attack, defense, direction alignment, sign analysis, anomaly detection

## TL;DR
Proposes the AlignIns defense method, which identifies malicious model updates in federated learning through dual-granularity direction alignment detection (global direction + fine-grained sign analysis), outperforming existing defense methods under both IID and non-IID settings.

## Background & Motivation

**Background**: Federated learning (FL) is inherently vulnerable to backdoor attacks due to its distributed training nature, where malicious clients can submit poisoned model updates to manipulate the global model. Various attack methods (e.g., Badnet, DBA, Scaling, PGD, Neurotoxin) threaten the security of FL.

**Limitations of Prior Work**:
   - **Magnitude-based defenses** (Manhattan/Euclidean distance): When the model approaches convergence, all update magnitudes become small, making malicious updates difficult to distinguish from benign ones based on magnitude.
   - **Cosine similarity-based defenses** (e.g., FoolsGold): Only capture global directional similarity, ignoring fine-grained information (such as parameter sign distributions).
   - Under non-IID data scenarios, the update directions of benign clients are inherently diverse, making anomaly detection even more challenging.
   - There is a lack of theoretical analysis on filtering-based defense methods under non-IID data.

**Key Challenge**: The dual objective of backdoor attacks (maintaining main-task accuracy + maximizing backdoor accuracy) forces malicious updates to mimic benign updates in magnitude, but they may expose anomalies in fine-grained directional features.

**Key Insight**: Checking directional alignment at two granularities—global temporal direction alignment and fine-grained major parameter sign alignment.

**Core Idea**: Temporal Direction Alignment (TDA) + Major Parameter Sign Alignment (MPSA) + MZ-score anomaly detection + Post-filtering clipping = Robust backdoor defense.

## Method

### AlignIns Overall Workflow
Receiving all client model updates → Direction alignment detection (two-step) → Filtering malicious updates → Clipping → Aggregation

### Key Designs

1. **Temporal Direction Alignment (TDA)**

    - **Function**: Evaluates the cosine similarity between each model update and the direction of the latest global model.
    - **Mechanism**: Benign updates should align broadly with the global convergence direction, whereas malicious updates may exhibit abnormal alignment patterns.
    - **Calculation**: $\text{TDA}_i = \cos(\Delta_i^t, \theta^t)$
    - Employs MZ-score for anomaly detection; updates exceeding the radius $\lambda_c$ are flagged as suspicious.

2. **Major Parameter Sign Alignment (MPSA)**

    - **Function**: Analyzes the sign distribution of major parameters in model updates.
    - **Mechanism**: Extracts parameters in the top-$k$ ($k = 0.3 \times d$) in terms of magnitude from each update, and computes the alignment ratio of their signs with the principal signs of all updates.
    - **Principal signs**: The majority voted signs across all updates.
    - **Effect**: Captures fine-grained anomalies that cannot be detected by global cosine similarity.

3. **MZ-score Anomaly Detection**

    - Uses the robust Modified Z-score (based on median instead of mean).
    - Minimal hyperparameters: requires only two filtering radii, $\lambda_c$ and $\lambda_s$.
    - Default values: $\lambda_c = 1.0$, $\lambda_s = 1.0$.

4. **Post-filtering Clipping**

    - Extra clipping is applied to updates passing the direction detection to mitigate abnormally large magnitudes.
    - Defends against magnitude-based attacks that might bypass direction detection.

### Theoretical Contributions
- Provides a theoretical analysis of the robustness of AlignIns.
- Proves that the propagation error of AlignIns during FL training is bounded.
- Presents the first theoretical robustness analysis for filtering-based defenses under non-IID data.

## Key Experimental Results

### Main Results on IID CIFAR-10 (ResNet9, 20% Attackers, 50% Poisoning Rate)

| Method | Clean MA↑ | Badnet BA↓ | DBA BA↓ | Neurotoxin BA↓ | Avg. RA↑ |
|------|---------|-----------|---------|---------------|---------|
| FedAvg (No Defense) | 89.47 | 67.61 | 70.42 | 79.40 | — |
| FoolsGold | — | — | — | — | Low |
| Multi-Metrics | — | — | — | — | Medium |
| **AlignIns** | **Best** | **Lowest** | **Lowest** | **Lowest** | **Best** |

### Cross-Device FL Settings (100 Clients, CIFAR-10)

| Method | IID RA↑ | Non-IID RA↑ |
|------|---------|------------|
| FoolsGold | 82.99 | Low |
| **AlignIns** | **Best**| **Best** |

AlignIns remains effective under cross-device (large-scale client) environments.

### Ablation Study (CIFAR-10)

| Configuration | IID Avg. RA↑ | IID BA↓ | Non-IID Avg. RA↑ |
|------|-------------|---------|-----------------|
| MPSA Only (30%) | 88.55 | 2.88 | — |
| TDA + MPSA (Complete) | **Best** | **Lowest** | **Best** |

### Key Findings
- TDA and MPSA are complementary: TDA captures global direction anomalies, while MPSA captures fine-grained sign anomalies.
- Demonstrates a more significant advantage in non-IID scenarios because MPSA is unaffected by the diversity of benign updates.
- Effective against 5 SOTA attacks (Badnet, DBA, Scaling, PGD, Neurotoxin).

## Supplementary Experimental Settings

### Datasets and FL Configurations

| Parameter | Default Value |
|------|-------|
| Number of clients | 20 (cross-silo) / 100 (cross-device) |
| Adversarial ratio | 20% (4/20 malicious clients) |
| Poisoning rate | 50% |
| Non-IID degree | Dirichlet $\beta=0.5$ |
| Local training epochs | 2 |
| CIFAR-10 training rounds | 150 |
| CIFAR-100 training rounds | 100 |
| MPSA parameter k | $0.3\times d$ (Top-30% parameters) |

## Highlights & Insights
- **Dual-granularity detection** with clear logic: global direction + parameter sign distribution, covering anomalies at different levels from coarse to fine.
- Novel **MPSA metric**: leverages the sign distribution of major parameters instead of their magnitudes, maintaining discriminative power even when the model converges.
- **MZ-score** is more robust than standard Z-score and is insensitive to outliers.
- Comprehensive theoretical analysis: the first work to prove bounded robustness of filtering-based defenses under non-IID conditions.
- Fully compatible with existing FL frameworks without requiring modifications to client local training processes.
- The appendix validates robustness against adaptive attacks such as trigger optimization.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Infighting in the Dark: Multi-Label Backdoor Attack in Federated Learning](infighting_in_the_dark_multi-label_backdoor_attack_in_federated_learning.md)
- [\[CVPR 2025\] Geometric Knowledge-Guided Localized Global Distribution Alignment for Federated Learning](geometric_knowledge-guided_localized_global_distribution_alignment_for_federated.md)
- [\[CVPR 2025\] DeDe: Detecting Backdoor Samples for SSL Encoders via Decoders](dede_detecting_backdoor_samples_for_ssl_encoders_via_decoders.md)
- [\[CVPR 2026\] Batman: Benign Knowledge Alignment Through Malicious Null Space in Federated Backdoor Attack](../../CVPR2026/ai_safety/batman_benign_knowledge_alignment_through_malicious_null_space_in_federated_back.md)
- [\[ECCV 2024\] Fisher Calibration for Backdoor-Robust Heterogeneous Federated Learning](../../ECCV2024/ai_safety/fisher_calibration_for_backdoor-robust_heterogeneous_federated_learning.md)

</div>

<!-- RELATED:END -->
