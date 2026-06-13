---
title: >-
  [Paper Note] HealSplit: Towards Self-Healing through Adversarial Distillation in Split Federated Learning
description: >-
  [AAAI 2026][AI Safety][Split Federated Learning] This paper proposes HealSplit, the first unified defense framework for Split Federated Learning (SFL). It identifies poisoned samples via topology-aware scoring (TAS) on a…
tags:
  - "AAAI 2026"
  - "AI Safety"
  - "Split Federated Learning"
  - "Data Poisoning Attack"
  - "Topology-Aware Detection"
  - "Adversarial Distillation"
  - "Self-Healing Defense"
date: 2026-05-08
content_hash: 33e56f4a4fb3c197
---

# HealSplit: Towards Self-Healing through Adversarial Distillation in Split Federated Learning

**Conference**: AAAI 2026
**arXiv**: [2511.11240](https://arxiv.org/abs/2511.11240)  
**Code**: None  
**Area**: AI Security
**Keywords**: Split Federated Learning, Data Poisoning Attack, Topology-Aware Detection, Adversarial Distillation, Self-Healing Defense

## TL;DR

This paper proposes HealSplit, the first unified defense framework for Split Federated Learning (SFL). It identifies poisoned samples via topology-aware scoring (TAS) on a graph built over smashed data, generates semantically consistent substitute representations using a GAN, and trains a consistency-validated student model through adversarial multi-teacher distillation. This end-to-end detect-and-recover pipeline substantially outperforms ten SOTA defense methods across five categories of poisoning attacks.

## Background & Motivation

**Split Federated Learning (SFL)** combines the advantages of Federated Learning (FL) and Split Learning (SL): each client performs local forward propagation and transmits intermediate representations (smashed data) to the server. However, SFL exposes multiple attack surfaces:

**Types of poisoning attacks**:
- Label poisoning (LP): modifies labels as $y_j' = (y_j + \delta_y) \mod C$
- Data poisoning (DP): modifies inputs as $x_j' = x_j + \delta_x$
- Smashed data poisoning (SP): modifies intermediate representations as $z_j' = g_\phi(x_j) + \delta_z$
- Weight poisoning (WP): modifies model parameters as $\theta' = \theta + \Delta_\theta$
- Multi-vector combined attacks (DP+SP, WP+SP, etc.)

**Limitations of existing defenses**:
- Statistical aggregation methods such as Krum, Trimmed Mean, and Bulyan assume access to complete model updates or raw gradients — assumptions that do not hold under SFL's split architecture
- Advanced defenses such as FLTrust and DnC are primarily designed for single attack vectors and generalize poorly
- ShieldFL employs encrypted cosine similarity yet remains vulnerable to combined attacks

**Core motivation**: In SFL, smashed data constitutes the primary channel for poisoning attacks — securing smashed data is thus sufficient to defend against a broad range of attack types. However, detection alone is insufficient; a recovery mechanism is needed to avoid the utility degradation caused by discarding flagged samples.

## Method

### Overall Architecture

HealSplit comprises three core components forming an end-to-end detect → replace → verify pipeline:

1. **Topology-Aware Detection** — constructs a graph over smashed data and identifies poisoned samples via TAS
2. **Semantically Consistent Substitution** — generates substitute representations using a GAN
3. **Adversarial Multi-Teacher Distillation** — trains a consistency-validated student model to ensure substitution quality

### Key Designs

1. **Topology-Aware Malicious Data Detection**

   **Core observation**: Poisoned samples tend to form locally dense but globally isolated clusters in feature space — highly similar to each other yet weakly connected to benign data.

   **Graph construction**: A KNN weighted graph is built over smashed data with adjacency matrix:
   $W_{kj} = \begin{cases} \exp(-\gamma \|z_k - z_j\|^2), & \text{if } z_j \in \mathcal{N}_k \text{ and } z_k \in \mathcal{N}_j \\ 0, & \text{otherwise} \end{cases}$

   **Topology Anomaly Scoring (TAS)**: Personalized PageRank (PPR) is applied for iterative propagation to capture both local and global structural anomalies:
   $r_k^{(t+1)} = \mathbb{I}_{[t=0]} \cdot \frac{1}{d_k + \epsilon} + \mathbb{I}_{[t \geq 1]} \cdot \left(\alpha \sum_{w \in \mathcal{N}(k)} \frac{r_w^{(t)}}{d_w} + (1-\alpha) v_k \right)$

   **Adaptive thresholding**: The detection threshold is determined automatically via kernel density estimation (KDE):
   $T = \min\left(\underset{r}{\operatorname{argmin}} \hat{f}(r), Q_\rho(\{r_k\})\right)$

   Samples with TAS below the threshold are flagged as poisoned.

   **Design motivation**: This design is inspired by graph propagation theory from social network analysis — the propagation patterns of anomalous nodes differ detectably from those of normal nodes in the topological dimension. PPR accounts for both local neighborhoods and global graph structure, making it more robust than purely feature-space distance measures.

2. **Semantically Consistent Substitution**

   A vanilla GAN is trained on the cleaned smashed data to generate substitute representations:
   $\mathcal{L}_D = -\mathbb{E}_z[\log D(z)] - \mathbb{E}_{\tilde{z}}[\log(1 - D(\tilde{z}))]$
   $\mathcal{L}_G = -\mathbb{E}_{\tilde{z}}[\log D(\tilde{z})]$

   The GAN is trained solely on the smashed data from the current round. Due to limited training data, generated samples may be semantically inconsistent and thus require filtering by the consistency-validated student model.

3. **Adversarial Multi-Teacher Distillation**

   Two complementary teachers are employed:

   **(a) AD (Anomaly-Influence Debiasing) Teacher**:
   - Defines three tasks: poisoning identification (a), client identification (b), and class classification (c)
   - Computes a Gradient Interaction Score (GIS) matrix $\mathbf{G}_p$ measuring inter-task gradient alignment
   - Constructs an inter-task influence score matrix $\mathbf{M}_p$ combining the TAS matrix $\mathbf{R}$ and the GIS matrix
   - The loss function dynamically adjusts label influence weights via the influence score matrix

   **(b) Vanilla Teacher**: Trained exclusively on cleaned data to preserve semantic integrity

   The student model is distilled using KL divergence:
   $\mathcal{L}_{VS} = \tau^2 \cdot KL(LogSoftmax(h_{T_{van}}(z_i)/\tau), Softmax(h_S(z_i)/\tau))$
   $\mathcal{L}_{IS} = \tau^2 \cdot KL(LogSoftmax(h_{T_{AD}}(z_i)/\tau), Softmax(h_S(z_i)/\tau))$

### Loss & Training

**Total student loss**: $\mathcal{L}_{Stu} = \sum_k (\mathcal{L}_a + \lambda_b \mathcal{L}_b + \mu \mathcal{L}_{VS} + \eta \mathcal{L}_{IS})$

**Momentum-adaptive optimization**: The contributions of the two teachers are dynamically balanced to prevent either from dominating:
$$\mu_t = m \cdot \mu_{t-1} + (1-m) \cdot \sigma\left(\kappa \cdot \frac{\mathcal{L}_{VS} - \mathcal{L}_{IS}}{\mathcal{L}_{VS} + \mathcal{L}_{IS} + \epsilon}\right)$$

**Theoretical guarantee**: It is proved that HealSplit reduces the server-side gradient variance (SGV) by increasing gradient similarity, thereby providing a convergence-theoretic justification for its robustness.

## Key Experimental Results

### Main Results: Robustness Across Attack Types

Evaluated on MNIST with 10 clients (20% malicious) and a ResNet-18 backbone:

| Defense | No Attack | DP | WP | SP | LP | DP+SP | WP+SP | LP+SP |
|---|---|---|---|---|---|---|---|---|
| FedAvg | 96.90 | 10.12 | 44.74 | 96.90 | 79.23 | 9.19 | 68.22 | 64.82 |
| Krum | 96.66 | 76.77 | 15.91 | 71.62 | 82.95 | 70.48 | 76.20 | 70.68 |
| ShieldFL | 97.58 | 83.73 | 84.24 | 96.35 | 78.18 | 75.54 | 75.16 | 12.97 |
| DnC | 97.27 | 80.58 | 82.18 | 95.33 | 80.43 | 76.34 | 78.82 | 75.33 |
| FLTrust | 96.52 | 76.48 | 48.70 | 94.42 | 55.56 | 73.39 | **11.33** | 32.41 |
| **HealSplit** | 97.17 | **96.86** | **95.99** | **96.75** | **96.72** | **93.88** | **92.44** | **93.88** |

Key findings:
- HealSplit maintains above 92% accuracy across all attack scenarios with minimal variance
- The SOTA method FLTrust collapses to 11.33% under the WP+SP combined attack
- ShieldFL similarly collapses to 12.97% under LP+SP
- HealSplit is attack-agnostic — it requires no prior knowledge of the attack type

### Ablation Study

| Component | MNIST | F-MNIST | CIFAR-10 | HAM10k |
|---|---|---|---|---|
| **HealSplit (Full)** | **93.88** | **84.11** | **53.87** | **72.27** |
| w/o Vanilla Teacher | 90.99 | 80.64 | 51.27 | 69.64 |
| w/o AD Teacher | 87.34 | 75.17 | 46.40 | 63.20 |
| w/o Distillation | 74.38 | 69.65 | 42.75 | 59.61 |
| w/o Adversarial | 92.74 | 82.59 | 51.55 | 70.40 |

- The AD Teacher and distillation mechanism contribute most significantly (removing each causes drops of ~6.5% and ~19.5%, respectively)
- The Vanilla Teacher provides training stability
- The adversarial mechanism enhances robustness against strong attacks

### Key Findings

1. **Cross-dataset generalization**: HealSplit outperforms all baselines on MNIST, F-MNIST, CIFAR-10, and HAM10000 (non-IID)
2. **Cross-architecture generalization**: Consistent superiority is observed across ResNet-18, ResNet-152, and VGG16
3. **Robustness to client scale**: HealSplit maintains stable high accuracy as the number of clients increases, whereas DnC degrades significantly
4. **Robustness to malicious ratio**: As the fraction of malicious clients increases from 10% to 50%, HealSplit exhibits only a marginal performance drop
5. **Robustness to adaptive attacks**: Even when attackers attempt to minimize TAS discrepancy to evade detection, HealSplit still surpasses the strongest baseline

## Highlights & Insights

1. **First unified defense framework for SFL**: Covers five attack categories rather than targeting a single attack type
2. **End-to-end detect-and-recover design**: Rather than simply discarding suspicious samples, substitute representations are generated and validated for consistency
3. **Topological perspective as a novel contribution**: Graph propagation theory is applied to analyze anomalous patterns in smashed data, offering greater robustness than feature-space distance metrics
4. **Theoretical grounding**: HealSplit is proved to reduce SGV, providing a convergence-theoretic basis for its robustness
5. **No attack prior required**: Anomalies are automatically detected in real time with adaptive thresholding, eliminating the need for manual tuning

## Limitations & Future Work

- Evaluation is limited to image classification tasks; applicability to NLP or other modalities remains unvalidated
- GAN training overhead may affect SFL system efficiency, as it must be synchronized with training rounds
- The topological detection assumes poisoned samples form "locally dense, globally isolated" clusters — adversaries who deliberately scatter poisoned samples may evade detection
- While adaptive attack experiments demonstrate a performance drop that still leads all baselines, stronger adaptive attacks (e.g., gradient-based evasion) warrant further investigation
- Validation is confined to classification tasks; applicability to generative or regression tasks remains to be examined

## Related Work & Insights

- **Distinction from FL defenses**: Conventional FL defenses (Krum, Bulyan, FLTrust) assume access to complete model updates, which is not feasible under SFL's split architecture
- **Application of adversarial distillation**: The dual-teacher frameworks of DTDBD and B-MTARD inspired the AD Teacher + Vanilla Teacher design in this work
- **Practical significance**: As SFL becomes an increasingly popular paradigm for privacy-preserving distributed learning, HealSplit provides the first targeted security guarantee for this setting

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (First unified defense framework for SFL + organic integration of topological detection and adversarial distillation)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (4 datasets × 5 attack types × 10 baselines × multi-dimensional ablation and generalization experiments)
- Writing Quality: ⭐⭐⭐⭐ (Complex framework presented in an organized manner with clear mathematical derivations)
- Value: ⭐⭐⭐⭐⭐ (Fills a critical gap in SFL security defense)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] FedAFD: Multimodal Federated Learning via Adversarial Fusion and Distillation](../../CVPR2026/ai_safety/fedafd_multimodal_federated_learning_via_adversarial_fusion_and_distillation.md)
- [\[AAAI 2026\] CoRe-Fed: Bridging Collaborative and Representation Fairness via Federated Embedding Distillation](core-fed_bridging_collaborative_and_representation_fairness_via_federated_embedd.md)
- [\[AAAI 2026\] Credal Ensemble Distillation for Uncertainty Quantification](credal_ensemble_distillation_for_uncertainty_quantification.md)
- [\[AAAI 2026\] InfoDecom: Decomposing Information for Defending Against Privacy Leakage in Split Inference](infodecom_decomposing_information_for_defending_against_privacy_leakage_in_split.md)
- [\[ICLR 2026\] Hide and Find: A Distributed Adversarial Attack on Federated Graph Learning](../../ICLR2026/ai_safety/hide_and_find_a_distributed_adversarial_attack_on_federated_graph_learning.md)

</div>

<!-- RELATED:END -->
