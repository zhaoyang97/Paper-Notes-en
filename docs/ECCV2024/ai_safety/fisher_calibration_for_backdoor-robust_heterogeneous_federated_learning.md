---
title: >-
  [Paper Note] Fisher Calibration for Backdoor-Robust Heterogeneous Federated Learning
description: >-
  [ECCV 2024][AI Safety][Federated Learning] This paper proposes Self-Driven Fisher Calibration (SDFC), which utilizes Fisher information to measure differences in parameter importance across different distributions. SDFC effectively distinguishes malicious backdoor clients and performs parameter calibration in heterogeneous federated learning scenarios, overcoming the limitations of existing defense methods that rely on data homogeneity and minority malicious node assumptions.
tags:
  - "ECCV 2024"
  - "AI Safety"
  - "Federated Learning"
  - "Backdoor Attack"
  - "Fisher Information"
  - "Heterogeneity"
  - "Parameter Calibration"
date: 2026-05-08
content_hash: 34113a7b17070ef8
---

# Fisher Calibration for Backdoor-Robust Heterogeneous Federated Learning

**Conference**: ECCV 2024  
**Code**: [GitHub](https://github.com/WenkeHuang/SDFC)  
**Area**: AI Security / Federated Learning  
**Keywords**: Federated Learning, Backdoor Attack, Fisher Information, Heterogeneity, Parameter Calibration

## TL;DR

This paper proposes Self-Driven Fisher Calibration (SDFC), which utilizes Fisher information to measure differences in parameter importance across different distributions. SDFC effectively distinguishes malicious backdoor clients and performs parameter calibration in heterogeneous federated learning scenarios, overcoming the limitations of existing defense methods that rely on data homogeneity and minority malicious node assumptions.

## Background & Motivation

1. **Background**: Federated Learning, as a privacy-preserving distributed learning paradigm, has shown great potential in collaborative computer vision tasks. However, federated learning systems are highly vulnerable to backdoor attacks, where malicious clients inject triggers into training data to induce targeted misclassifications of trigger-containing inputs during inference.

2. **Limitations of Prior Work**: Existing backdoor defense schemes usually rely on two core assumptions: (a) data distributions across clients are homogeneous (IID), and (b) malicious clients are in the minority. Based on these assumptions, existing methods use client-level anomaly detection rules (such as norm clipping of model updates, median aggregation, etc.) to identify and filter malicious clients. However, these assumptions often do not hold in real-world scenarios—different clients naturally have different data distributions (Non-IID/heterogeneity), and the proportion of malicious clients can be high.

3. **Key Challenge**: In heterogeneous federated learning, both benign clients with heterogeneous data and backdoor attackers generate gradient updates that deviate from the global optimization direction. This makes it extremely difficult to distinguish them purely based on gradient/parameter deviations; normal data heterogeneity and malicious backdoor injection produce similar "anomaly" signals at the parameter update level, which traditional methods cannot accurately distinguish.

4. **Goal**: How to effectively defend against backdoor attacks in heterogeneous federated learning without relying on data homogeneity and minority malicious node assumptions.

5. **Key Insight**: Although both heterogeneous clients and backdoor attackers cause parameter deviation, their deviations differ in the dimension of parameter importance. Fisher Information can be used to measure the importance of each parameter to a specific distribution. The parameter deviation of benign heterogeneous clients is concentrated on parameters important to their local distribution (reasonable deviation), whereas the deviation of backdoor attackers occurs on parameters that are important to the trigger pattern but unimportant to the normal distribution (malicious deviation).

6. **Core Idea**: Use Fisher Information to measure the importance differences of parameters relative to "meaningful distributions" (local data and global validation sets), and calibrate the parameter updates of heterogeneous clients accordingly to naturally filter out backdoor signals.

## Method

### Overall Architecture

SDFC introduces two core mechanisms, parameter calibration based on Fisher information and adaptive aggregation weight allocation, into the standard federated learning aggregation framework. In each communication round: (1) the server distributes the global model to each client; (2) clients train locally on their data, then upload both model updates and local Fisher information matrix estimations; (3) the server computes importance differences for each parameter using Fisher information and calibrates the parameters that exhibit significant importance differences between the local and global distributions; (4) differentiated aggregation weights are assigned to clients, giving larger contributions to clients closer to the global distribution.

### Key Designs

1. **Fisher Information for Parameter Importance Measurement**: The core innovation of SDFC is using the Fisher Information Matrix (FIM) to characterize the importance of model parameters for specific data distributions. For each client $k$, two sets of Fisher information are computed: (a) the Fisher information matrix $F_k^{local}$ with respect to the local data distribution, reflecting parameter importance to the local task; (b) the Fisher information matrix $F^{global}$ with respect to the global validation distribution, reflecting importance to the global target distribution. By comparing the difference between these two matrices $|F_k^{local} - F^{global}|$, parameters whose local importance is significantly inconsistent with global importance can be identified. Although normal heterogeneous clients have different data distributions, their parameter importance patterns still maintain high consistency with the global distribution (since they are sub-distributions of the same task). Conversely, backdoor attackers exhibit abnormally high local importance on parameters that are unimportant for the normal distribution but critical for the trigger response due to the injected trigger patterns.

2. **Parameter Calibration**: Based on the Fisher information differences, SDFC calibrates parameters with large importance differences. Specifically, for parameters where the local importance is much higher than the global importance (potentially containing backdoor signals), their influence weights during aggregation are reduced. For parameters with small importance differences (usually features useful for the task itself), their normal contributions are maintained. This calibration operates at a fine-grained, parameter-by-parameter level, rather than simply discarding or retaining the entire client model. This enables SDFC to precisely excise backdoor signals while retaining useful knowledge contributions even if most of a malicious client's parameter updates are benign (with only a few parameters contaminated by the backdoor).

3. **Adaptive Aggregation Weight Allocation**: In addition to parameter-level calibration, SDFC assigns differentiated aggregation weights at the client level. The weights are allocated based on the total magnitude of parameter differences: the smaller the overall difference between local and global Fisher information, the closer the client's data distribution is to the global distribution, and thus the higher the aggregation weight assigned. This strategy encourages clients with more representative data to contribute more to the global model. Because backdoor attackers introduce abnormal trigger distributions, their overall Fisher information differences are typically larger, naturally resulting in lower aggregation weights.

### Loss & Training

- Clients use standard cross-entropy loss for local training.
- Fisher information is calculated via diagonal approximation of the second-order derivative of the loss function with respect to the parameters: $F_{ii} = \mathbb{E}[(\partial \mathcal{L}/\partial w_i)^2]$
- Global Fisher information is calculated using a small validation set maintained on the server side (free of sensitive user data).
- Parameter calibration is executed during server-side aggregation without adding computational overhead to clients.
- Compatible with mainstream federated learning aggregation algorithms such as FedAvg and FedProx.

## Key Experimental Results

### Main Results

| Dataset/Scenario | Metric | Ours | Prev. SOTA | Gain |
|------------|------|------|------------|------|
| CIFAR-10 (Non-IID + Backdoor) | Main Task Accuracy | High | Baseline Methods | Maintain Normal Performance |
| CIFAR-10 (Non-IID + Backdoor) | Backdoor Attack Success Rate↓ | **Significantly Reduced** | Limited Defense Effect | Significant Improvement |
| CIFAR-100 (Non-IID + Backdoor) | Main Task Accuracy | High | Baseline Methods | Stable |
| CIFAR-100 (Non-IID + Backdoor) | Backdoor Attack Success Rate↓ | **Significantly Reduced** | High | Drastic Decrease |

### Ablation Study

| Configuration | Backdoor Success Rate | Main Task Accuracy | Description |
|------|-----------|-------------|------|
| FedAvg (No Defense) | High | Baseline | Backdoor attack is almost 100% successful |
| + Norm Clipping | Still High | Slight Decrease | Simple clipping is insufficient for defense |
| + Krum/Multi-Krum | Medium | Significant Decrease | Filters out benign heterogeneous clients |
| + SDFC (Param-only Calibration) | Low | Maintained | Parameter-level calibration effectively excises backdoor |
| + SDFC (Calibration + Weights) | **Lowest** | Maintained Highest | Complete scheme achieves optimal performance |

### Key Findings

- Under the joint challenge of Non-IID + backdoor attacks, traditional defense methods (such as Krum, median aggregation, etc.) mistakenly discard useful updates from benign heterogeneous clients, resulting in a severe degradation in main task performance.
- Fisher information can indeed effectively distinguish parameter deviations caused by normal heterogeneity from those caused by backdoor attacks.
- Fine-grained parameter-level calibration is more effective than coarse-grained client-level filtering, excising backdoors while preserving useful knowledge.
- SDFC maintains effective defense even when the proportion of malicious clients is high (e.g., 40-50%).

## Highlights & Insights

1. **Precise Problem Definition**: Clearly points out the core difficulty of backdoor defense in heterogeneous federated learning—the entanglement of heterogeneity and malice at the parameter level—and proposes an effective resolution strategy.
2. **Novel Application of Fisher Information**: Brings Fisher information from traditional fields like model compression and continual learning into federated learning security, cleverly utilizing its "parameter importance measurement" characteristics to solve a new problem.
3. **Parameter-Level Fine-Grained Operation**: Instead of simply discarding or accepting the entire client model, SDFC calibrates each parameter individually, which can both excise backdoors and preserve knowledge.
4. **Combining Theory and Practice**: Fisher information provides a solid theoretical foundation while the method implementation remains simple and efficient.

## Limitations & Future Work

1. The calculation of Fisher information requires the server side to maintain a validation set to estimate the global distribution, which may be impractical in certain privacy-strict scenarios.
2. The diagonal approximation of Fisher information is a commonly used but coarse approximation, which may lose correlation information between parameters.
3. For more stealthy adaptive backdoor attacks (where attackers adjust their attack methods based on defense strategies), the robustness of SDFC needs further verification.
4. Experiments are mainly conducted on image classification tasks, and the effectiveness on more complex vision tasks (object detection, segmentation) is worth exploring.
5. Communication overhead may increase, as clients need to upload an additional Fisher information matrix.

## Related Work & Insights

- **FedAvg (McMahan et al., 2017)**: The baseline algorithm for federated learning, with no defense capability against backdoor attacks.
- **Krum/Multi-Krum (Blanchard et al., 2017)**: Aggregates updates by selecting clients closest to the majority, but may mistakenly filter benign clients in heterogeneous scenarios.
- **EWC (Kirkpatrick et al., 2017)**: Was the first to use Fisher information to measure parameter importance in continual learning, with which SDFC's idea shares theoretical connections.
- **FLTrust (Cao et al., 2022)**: Uses a server-side validation set to construct trust scores, showing similarities to SDFC in the usage of server-side validation data.
- **Insight**: The application of Fisher information in the security domain deserves further exploration, such as detecting data poisoning and model inversion attacks in federated learning.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Using Fisher information for backdoor defense is a novel entry point, and the idea of parameter-level calibration is creative.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive comparison across multiple attack methods, multiple heterogeneity settings, and multiple baseline schemes.
- **Writing Quality**: ⭐⭐⭐⭐ Problem motivation is clearly stated, and the analysis of the core contradiction is thorough.
- **Value**: ⭐⭐⭐⭐ Addresses practical pain points of backdoor defenses in heterogeneous federated learning, which is of great significance for the secure deployment of federated learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Domain-Skewed Federated Learning with Feature Decoupling and Calibration](../../CVPR2026/ai_safety/domain-skewed_federated_learning_with_feature_decoupling_and_calibration.md)
- [\[ECCV 2024\] SkyMask: Attack-Agnostic Robust Federated Learning with Fine-Grained Learnable Masks](skymask_attack-agnostic_robust_federated_learning_with_fine-grained_learnable_ma.md)
- [\[CVPR 2025\] Infighting in the Dark: Multi-Label Backdoor Attack in Federated Learning](../../CVPR2025/ai_safety/infighting_in_the_dark_multi-label_backdoor_attack_in_federated_learning.md)
- [\[CVPR 2026\] FedRE: A Representation Entanglement Framework for Model-Heterogeneous Federated Learning](../../CVPR2026/ai_safety/fedre_a_representation_entanglement_framework_for_model-heterogeneous_federated_.md)
- [\[CVPR 2025\] Detecting Backdoor Attacks in Federated Learning via Direction Alignment Inspection](../../CVPR2025/ai_safety/detecting_backdoor_attacks_in_federated_learning_via_direction_alignment_inspect.md)

</div>

<!-- RELATED:END -->
