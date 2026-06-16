---
title: >-
  [Paper Note] Traceable Black-box Watermarks for Federated Learning
description: >-
  [ICLR 2026][AI Safety][Federated Learning] This paper proposes TraMark, which partitions the model parameter space into a main-task region and a watermark region and employs masked aggregation to prevent watermark collis…
tags:
  - "ICLR 2026"
  - "AI Safety"
  - "Federated Learning"
  - "Black-box Watermarking"
  - "Traceability"
  - "Model Leakage Detection"
  - "Masked Aggregation"
date: 2026-05-08
content_hash: e7f4e935e47fbeff
---

# Traceable Black-box Watermarks for Federated Learning

**Conference**: ICLR 2026
**arXiv**: [2505.13651](https://arxiv.org/abs/2505.13651)  
**Code**: [GitHub](https://github.com/JiiahaoXU/TraMark)  
**Area**: AI Security
**Keywords**: Federated Learning, Black-box Watermarking, Traceability, Model Leakage Detection, Masked Aggregation

## TL;DR

This paper proposes TraMark, which partitions the model parameter space into a main-task region and a watermark region and employs masked aggregation to prevent watermark collision. TraMark achieves server-side traceable black-box watermark injection in federated learning for the first time, attaining a verification rate of 99.58% with only a 0.54% drop in main-task accuracy.

## Background & Motivation

1. **Model leakage risk in federated learning**: In FL systems, every client has access to the global model. Malicious clients may copy and illegally distribute the model, threatening the collective interests of all participants. Protecting the intellectual property of FL-trained models has become a critical challenge.

2. **Limitations of existing watermarking methods**: Parameter-level watermarks (e.g., FedTracker) require white-box access to model parameters for verification, which is infeasible in practical deployments (e.g., API access). Backdoor-based watermarks support black-box verification, but existing approaches either do not support tracing or require modifications to the local training protocol and access to client data.

3. **Tension between traceability and watermark collision**: Achieving traceability requires injecting a distinct watermark for each client; however, FedAvg aggregation merges all clients' parameters, causing watermark confusion (watermark collision) that undermines traceability. This is the core tension between traceability and federated aggregation.

4. **Lack of formal definitions**: Despite empirical progress, the literature still lacks formal definitions and mathematical modeling of the traceable black-box watermarking problem in FL, impeding the development of systematic solutions.

## Method

### Overall Architecture: TraMark

- **Function**: Creates a personalized, traceable watermarked model for each client on the server side, enabling verifiers to detect model leakage and identify the leakage source under a black-box setting.
- **Design Motivation**: Performing watermark injection entirely on the server side prevents malicious clients from tampering with the watermarking process; parameter space partitioning and masked aggregation resolve the watermark collision problem.
- **Mechanism**: Three core components — (1) Constrained watermark region: partitioning model parameters into a main-task region and a watermark region; (2) Masked aggregation: aggregating only the main-task region while preserving the independence of each client's watermark region; (3) Independent watermark injection: training each client's model on a unique watermark dataset within the watermark region.

### Key Design 1: Parameter Space Partitioning and Constrained Watermark Region

Existing methods perform watermark injection over the entire parameter space, causing two problems: (a) watermark perturbations spread across the full parameter space, degrading main-task performance; (b) watermarks from different clients are merged during aggregation, leading to collision. TraMark strictly partitions the parameter space into two regions using complementary binary masks:

- **Watermark mask** $\mathbf{M}_w \in \{0,1\}^d$: marks the parameters used for watermark learning, with proportion $k$ (default 1%)
- **Main-task mask** $\mathbf{M}_m \in \{0,1\}^d$: marks the parameters used for main-task learning, with proportion $1-k$

satisfying $\mathbf{M}_w + \mathbf{M}_m = \mathbf{1}^d$. The watermark region is selected based on parameter importance: after $\alpha \times T$ warm-up rounds, the $k \times d$ parameters with the smallest absolute values are selected as the watermark region, ensuring minimal impact on the main task.

### Key Design 2: Masked Aggregation and Watermark Injection

**Masked aggregation**: Unlike FedAvg, which uniformly averages all parameters, TraMark constructs a personalized global model for each client $i$ as follows:

$$\tilde{\theta}_i = \theta_i + \mathbf{M}_m \odot \frac{1}{n}\sum_{i=1}^{n}\Delta_i + \mathbf{M}_w \odot \Delta_i$$

The main-task region undergoes standard FedAvg aggregation (sharing knowledge), while the watermark region retains only the client's own updates (preserving unique watermarks).

**Watermark injection**: Each personalized model $\tilde{\theta}_i$ is trained for $\tau_w$ steps on its corresponding unique watermark dataset $\mathcal{D}_i^w$, with gradients updated exclusively in the watermark region:

$$\tilde{\theta}_i^{s+1} = \tilde{\theta}_i^s - \eta_w g_i^s \odot \mathbf{M}_w$$

Gradient masking ensures that watermark knowledge does not propagate into the main-task region.

### Key Design 3: Unique Watermark Dataset Construction

To maximally avoid watermark collision, the watermark dataset assigned to each client must differ in both trigger patterns and output distribution:

- **Non-overlapping triggers**: $\mathcal{X}_i^w \cap \mathcal{X}_j^w = \emptyset$, using OOD samples unrelated to the main-task distribution (e.g., samples from individual MNIST classes)
- **Distinct predefined outputs**: $\phi_i(x) \neq \phi_j(x)$, mapping each client to different target labels

During verification, a watermarked model produces the predefined output only for triggers from its own watermark dataset, while responding with random guesses to other clients' triggers, thereby enabling reliable tracing.

## Key Experimental Results

### Experimental Setup

- **Datasets**: FMNIST (CNN), CIFAR-10 (AlexNet), CIFAR-100 (VGG-16), Tiny-ImageNet (ViT)
- **FL configuration**: 10 clients, 5 local training rounds, learning rate 0.01; both IID and non-IID (Dirichlet $\gamma=0.5$) settings
- **Watermark configuration**: MNIST as watermark source, 100 samples per class, watermark learning rate 1e-4, $\tau_w=5$, $k=1\%$, $\alpha=0.5$
- **Baselines**: FedAvg (no watermark), WAFFLE (black-box, non-traceable), FedTracker (white-box, traceable)
- **Metrics**: Main-task accuracy (MA) and verification rate (VR)

### Main Results

| Dataset | FedAvg MA | WAFFLE MA | FedTracker MA/VR | TraMark MA/VR |
|---|---|---|---|---|
| FMNIST | 92.60 | 92.21 | 89.95 / 100.0 | 91.20 / 96.67 |
| FMNIST (N) | 91.52 | 91.41 | 67.50 / 100.0 | 91.31 / 100.0 |
| CIFAR-10 | 89.15 | 89.16 | 87.56 / 60.0 | 88.58 / 100.0 |
| CIFAR-10 (N) | 87.01 | 86.75 | 83.42 / 50.0 | 86.26 / 100.0 |
| CIFAR-100 | 61.91 | 61.68 | 61.05 / 100.0 | 61.13 / 100.0 |
| Tiny-ImageNet | 21.05 | 21.24 | 20.40 / 100.0 | 20.91 / 100.0 |
| **Average** | **65.44** | **65.31** | **61.25 / 87.50** | **64.90 / 99.58** |

### Ablation Study: Key Hyperparameter Analysis

| Hyperparameter | Configuration | MA (%) | VR (%) |
|---|---|---|---|
| Partition ratio k=0.5% | Low watermark capacity | 65.70 | 84.17 |
| **Partition ratio k=1.0% (default)** | **Balanced** | **65.66** | **99.17** |
| Partition ratio k=5.0% | High watermark capacity | 65.16 | 100.0 |
| Watermark dataset 50 samples | Insufficient data | 65.85 | 54.17 |
| **Watermark dataset 100 samples (default)** | **Balanced** | **65.51** | **99.17** |
| Watermark dataset 200 samples | Sufficient data | 65.57 | 100.0 |
| Warm-up ratio α=0 | No warm-up | 59.50 | 100.0 |
| **Warm-up ratio α=0.5 (default)** | **With warm-up** | **64.15** | **100.0** |
| Warm-up ratio α=0.7 | Excessive warm-up | 65.20 | Degraded |

### Key Findings

1. **TraMark achieves high tracing rate with minimal accuracy loss**: Average VR of 99.58% (vs. 87.50% for FedTracker), with MA dropping only 0.54% (vs. 4.19% for FedTracker), validating the effectiveness of the parameter space partitioning strategy.
2. **Robustness against attacks**: VR remains stable under 30%–70% pruning rates and shows no significant degradation after 30 rounds of fine-tuning attacks, indicating strong coupling between watermark-region parameters and main-task parameters.
3. **Warm-up training is critical**: Without warm-up, MA drops by approximately 5%, as randomly initialized parameters cannot accurately reflect importance, leading to improper partitioning.
4. **Flexibility in watermark dataset selection**: MNIST, SVHN, and WafflePattern all achieve 100% VR, with no statistically significant differences ($p \geq 0.05$).

## Highlights & Insights

- Provides the first formal definition of the traceable black-box watermarking problem in federated learning, introducing the concepts of watermark collision and traceability constraints.
- The combination of parameter space partitioning and masked aggregation is elegant and concise, simultaneously preventing watermark collision and preserving main-task performance.
- Fully server-side operation requires no client cooperation and provides inherent resistance against malicious clients.
- Watermark injection incurs negligible overhead (0.67 seconds per client) and can be seamlessly integrated into existing FedAvg frameworks.

## Limitations & Future Work

- The watermark dataset requires assigning distinct OOD trigger categories to each client, limiting scalability for tasks with few labels (a 10-class main task with 10 clients already requires 20 categories).
- Evaluation is limited to classification tasks; applicability to other task types such as generation and detection remains unexplored.
- Although $k=1\%$ suffices, the parameter space partitioning strategy relies on simple magnitude-based ranking; more refined importance measures may further improve main-task performance.

## Rating

| Dimension | Score |
|---|---|
| Novelty | ⭐⭐⭐⭐ |
| Effectiveness | ⭐⭐⭐⭐ |
| Reproducibility | ⭐⭐⭐⭐⭐ |
| Practicality | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Toward Enhancing Representation Learning in Federated Multi-Task Settings](toward_enhancing_representation_learning_in_federated_multi-task_settings.md)
- [\[CVPR 2026\] FedDAP: Domain-Aware Prototype Learning for Federated Learning under Domain Shift](../../CVPR2026/ai_safety/feddap_domain-aware_prototype_learning_for_federated_learning_under_domain_shift.md)
- [\[AAAI 2026\] DeepTracer: Tracing Stolen Model via Deep Coupled Watermarks](../../AAAI2026/ai_safety/deeptracer_tracing_stolen_model_via_deep_coupled_watermarks.md)
- [\[CVPR 2026\] Domain-Skewed Federated Learning with Feature Decoupling and Calibration](../../CVPR2026/ai_safety/domain-skewed_federated_learning_with_feature_decoupling_and_calibration.md)
- [\[ICML 2026\] FedHPro: Federated Hyper-Prototype Learning via Gradient Matching](../../ICML2026/ai_safety/fedhpro_federated_hyper-prototype_learning_via_gradient_matching.md)

</div>

<!-- RELATED:END -->
