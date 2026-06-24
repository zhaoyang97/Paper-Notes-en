---
title: >-
  [Paper Note] HealSplit: Towards Self-Healing through Adversarial Distillation in Split Federated Learning
description: >-
  [AAAI 2026][AI Safety][Split Federated Learning] Proposes HealSplit, the first unified defense framework for Split Federated Learning (SFL). It integrates Topology-Aware Detection (TAS) to identify poisoned samples, GANs to generate semantically consistent substitute representations, and adversarial multi-teacher distillation to train a consistency-validating student model. This achieves end-to-end detection and recovery, outperforming ten SOTA defense methods under five type…
tags:
  - "AAAI 2026"
  - "AI Safety"
  - "Split Federated Learning"
  - "Data Poisoning Attacks"
  - "Topology Anomaly Detection"
  - "Adversarial Distillation"
  - "Self-Healing Defense"
date: 2026-05-08
content_hash: 423808ac5b1a28f6
---

# HealSplit: Towards Self-Healing through Adversarial Distillation in Split Federated Learning

**Conference**: AAAI 2026  
**arXiv**: [2511.11240](https://arxiv.org/abs/2511.11240)  
**Code**: None  
**Area**: AI Safety  
**Keywords**: Split Federated Learning, Data Poisoning Attacks, Topology Anomaly Detection, Adversarial Distillation, Self-Healing Defense

## TL;DR

Proposes HealSplit, the first unified defense framework for Split Federated Learning (SFL). It integrates Topology-Aware Detection (TAS) to identify poisoned samples, GANs to generate semantically consistent substitute representations, and adversarial multi-teacher distillation to train a consistency-validating student model. This achieves end-to-end detection and recovery, outperforming ten SOTA defense methods under five types of poisoning attacks.

## Background & Motivation

**Split Federated Learning (SFL)** combines the advantages of Federated Learning (FL) and Split Learning (SL), where each client executes local forward propagation and transmits intermediate representations (smashed data) to the server. However, SFL suffers from multiple attack surfaces:

**Multiple Poisoning Attack Types**:
   - Label Poisoning (LP): Modifying labels $y_j' = (y_j + \delta_y) \mod C$
   - Data Poisoning (DP): Modifying inputs $x_j' = x_j + \delta_x$
   - Smashed Data Poisoning (SP): Modifying intermediate representations $z_j' = g_\phi(x_j) + \delta_z$
   - Weight Poisoning (WP): Modifying model parameters $\theta' = \theta + \Delta_\theta$
   - Multi-vector joint attacks (such as DP+SP, WP+SP, etc.)

**Limitations of Prior Work**:
   - Statistical aggregation methods like Krum, Trimmed Mean, Bulyan → Assume access to full model updates or raw gradients, which does not hold under the split architecture of SFL.
   - Advanced defenses like FLTrust, DnC → Primarily target a single attack vector, resulting in poor generalization.
   - ShieldFL using encrypted cosine similarity → Remains vulnerable under joint attacks.

**Core Motivation**: Smashed data in SFL is the primary channel for poisoning attacks → securing smashed data can defend against a wide range of attack types. However, a recovery mechanism is needed post-detection; otherwise, discarding samples degrades model utility.

## Method

### Overall Architecture

HealSplit contains three core components, forming an end-to-end detection-substitution-verification pipeline:

1. **Topology-Aware Detection** → Constructs graphs on smashed data and identifies poisoned samples using TAS.
2. **Semantically Consistent Substitution** → Generates substitute representations using GANs.
3. **Adversarial Multi-Teacher Distillation** → Trains a consistency-validating student model to ensure the quality of substitutions.

### Key Designs

1. **Topology-Aware Detection**

   **Core Observation**: Poisoned samples tend to form locally dense but globally isolated clusters in the feature space—that is, they are highly similar to each other but weakly connected to benign data.

   **Graph Construction**: Constructs a KNN weighted graph on the smashed data, with the adjacency matrix:
   $$W_{kj} = \begin{cases} \exp(-\gamma \|z_k - z_j\|^2), & \text{if } z_j \in \mathcal{N}_k \text{ and } z_k \in \mathcal{N}_j \\ 0, & \text{otherwise} \end{cases}$$

   **Topology Anomaly Score (TAS)**: Uses Personalized PageRank (PPR) for iterative propagation to capture local and global structural anomalies:
   $$r_k^{(t+1)} = \mathbb{I}_{[t=0]} \cdot \frac{1}{d_k + \epsilon} + \mathbb{I}_{[t \geq 1]} \cdot \left(\alpha \sum_{w \in \mathcal{N}(k)} \frac{r_w^{(t)}}{d_w} + (1-\alpha) v_k \right)$$

   **Adaptive Threshold**: Automatically determines the detection threshold via Kernel Density Estimation (KDE):
   $$T = \min\left(\underset{r}{\operatorname{argmin}} \hat{f}(r), Q_\rho(\{r_k\})\right)$$

   Samples with a TAS below the threshold are flagged as poisoned samples.

   **Design Motivation**: Inspired by graph propagation theory in social networks—the propagation patterns of anomalous nodes exhibit detectable differences from normal nodes in topological dimensions. PPR considers both local neighborhoods and global graph structure, being more robust than mere feature distances.

2. **Semantically Consistent Substitution**

   Uses a vanilla GAN trained on the cleansed smashed data to generate substitute representations:
   $$\mathcal{L}_D = -\mathbb{E}_z[\log D(z)] - \mathbb{E}_{\tilde{z}}[\log(1 - D(\tilde{z}))]$$
   $$\mathcal{L}_G = -\mathbb{E}_{\tilde{z}}[\log D(\tilde{z})]$$

   The GAN is trained using only the smashed data of the current round. Since limited training data may lead to the generation of semantically inconsistent samples, a consistency-validating student model is required for filtering.

3. **Adversarial Multi-Teacher Distillation**

   Two complementary teachers:

   **(a) AD (Anomaly-Influence Debiasing) Teacher**:
    - Defines three tasks: Poisoning Identification (a), Client Identification (b), and Class Classification (c).
    - Computes the Gradient Interaction Score (GIS) matrix $\mathbf{G}_p$: measures the degree of gradient alignment between tasks.
    - Constructs the inter-task Influence Score Matrix $\mathbf{M}_p$: combines the TAS matrix $\mathbf{R}$ and the GIS matrix.
    - The loss function dynamically adjusts label influence weights via the influence score matrix.

   **(b) Vanilla Teacher**: Trained exclusively on cleansed data to maintain semantic integrity.

   The distillation loss of the student model uses KL divergence:
   $$\mathcal{L}_{VS} = \tau^2 \cdot KL(LogSoftmax(h_{T_{van}}(z_i)/\tau), Softmax(h_S(z_i)/\tau))$$
   $$\mathcal{L}_{IS} = \tau^2 \cdot KL(LogSoftmax(h_{T_{AD}}(z_i)/\tau), Softmax(h_S(z_i)/\tau))$$

### Loss & Training

**Total Student Loss**: $\mathcal{L}_{Stu} = \sum_k (\mathcal{L}_a + \lambda_b \mathcal{L}_b + \mu \mathcal{L}_{VS} + \eta \mathcal{L}_{IS})$

**Momentum Adaptive Optimization**: Dynamically balances the contributions of the two teachers to prevent domination by either side:
$$\mu_t = m \cdot \mu_{t-1} + (1-m) \cdot \sigma\left(\kappa \cdot \frac{\mathcal{L}_{VS} - \mathcal{L}_{IS}}{\mathcal{L}_{VS} + \mathcal{L}_{IS} + \epsilon}\right)$$

**Theoretical Guarantee**: Proves that HealSplit reduces server-side gradient variance (SGV) by improving gradient similarity, thereby improving convergence.

## Key Experimental Results

### Main Results: Robustness under Multiple Attack Types

On MNIST, 10 clients (20% malicious), ResNet-18 backbone:

| Defense Method | No Attack | DP | WP | SP | LP | DP+SP | WP+SP | LP+SP |
|---------|--------|----|----|----|----|-------|-------|-------|
| FedAvg | 96.90 | 10.12 | 44.74 | 96.90 | 79.23 | 9.19 | 68.22 | 64.82 |
| Krum | 96.66 | 76.77 | 15.91 | 71.62 | 82.95 | 70.48 | 76.20 | 70.68 |
| ShieldFL | 97.58 | 83.73 | 84.24 | 96.35 | 78.18 | 75.54 | 75.16 | 12.97 |
| DnC | 97.27 | 80.58 | 82.18 | 95.33 | 80.43 | 76.34 | 78.82 | 75.33 |
| FLTrust | 96.52 | 76.48 | 48.70 | 94.42 | 55.56 | 73.39 | **11.33** | 32.41 |
| **HealSplit** | 97.17 | **96.86** | **95.99** | **96.75** | **96.72** | **93.88** | **92.44** | **93.88** |

Key Findings:
- HealSplit maintains 92%+ accuracy across all attack scenarios with minimal fluctuation.
- The SOTA method FLTrust plummets to 11.33% under the WP+SP joint attack.
- ShieldFL also plummets to 12.97% under LP+SP.
- HealSplit is attack-agnostic—requiring no prior knowledge of attack types.

### Ablation Study

| Component | MNIST | F-MNIST | CIFAR-10 | HAM10k |
|------|-------|---------|----------|--------|
| **HealSplit (Full)** | **93.88** | **84.11** | **53.87** | **72.27** |
| w/o Vanilla Teacher | 90.99 | 80.64 | 51.27 | 69.64 |
| w/o AD Teacher | 87.34 | 75.17 | 46.40 | 63.20 |
| w/o Distillation | 74.38 | 69.65 | 42.75 | 59.61 |
| w/o Adversarial | 92.74 | 82.59 | 51.55 | 70.40 |

- The AD teacher and distillation mechanism contribute the most (decreasing by ~6.5% and ~19.5% respectively when removed).
- The Vanilla teacher provides training stability.
- The adversarial mechanism enhances robustness against strong attacks.

### Key Findings

1. **Cross-dataset Generalization**: Outperforms all baselines across MNIST, F-MNIST, CIFAR-10, and HAM10000 (Non-IID).
2. **Cross-model Generalization**: Consistently leads when using ResNet-18, ResNet-152, and VGG16.
3. **Robustness to Client Number**: HealSplit maintains stable high accuracy as the number of clients increases, whereas DnC degrades significantly.
4. **Robustness to Malicious Ratio**: HealSplit drops only slightly when the ratio of malicious clients increases from 10% to 50%.
5. **Robustness to Adaptive Attacks**: Even if attackers attempt to evade detection by minimizing TAS discrepancies, HealSplit still outperforms the strongest baselines.

## Highlights & Insights

1. **First Unified SFL Defense Framework**: Covers five types of attacks rather than targeting a single attack as prior work did.
2. **End-to-End Design of Detection + Recovery**: Instead of simply discarding suspicious samples, it generates substitutes and verifies consistency.
3. **Innovation from a Topological Perspective**: Uses graph propagation theory to analyze anomalous patterns in smashed data, which is more robust than feature space distances.
4. **Theoretical Support**: Proves that HealSplit reduces SGV, offering theoretical convergence guarantees for robustness.
5. **No Prior Attack Knowledge Required**: Automatically detects anomalies in real-time and adaptively adjusts thresholds without manual parameter tuning.

## Limitations & Future Work

- Still relies on image classification tasks for evaluation; the applicability to NLP or other modalities remains unverified.
- GAN training latency may impact SFL system efficiency (needs to be synchronized with training rounds).
- Topological detection assumes that poisoned samples form "locally dense, globally isolated" clusters → if attackers deliberately disperse poisoned samples, they might bypass detection.
- Although adaptive attack experiments showed performance decay, it still leads → stronger adaptive attacks (e.g., gradient-based evasion) warrant further exploration.
- Extensively validated only on classification tasks; applicability to generative or regression tasks needs to be investigated.

## Related Work & Insights

- **Difference from FL Defenses**: Traditional FL defenses (Krum, Bulyan, FLTrust) assume access to full model updates, which is invalid in SFL.
- **Application of Adversarial Distillation**: The dual-teacher frameworks of DTDBD and B-MTARD inspired the design of the AD Teacher + Vanilla Teacher in this work.
- **Practical Significance**: SFL is emerging as a popular paradigm for privacy-preserving distributed learning; HealSplit provides the first targeted security guarantee for it.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (The pioneering nature of the SFL unified defense framework + the organic integration of topological detection and adversarial distillation)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (4 datasets × 5 attack types × 10 baselines × multidimensional ablation + generalization experiments)
- Writing Quality: ⭐⭐⭐⭐ (Complex framework but systematically written, clear formula derivations)
- Value: ⭐⭐⭐⭐⭐ (Fills an important gap in SFL security defense)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] FedAFD: Multimodal Federated Learning via Adversarial Fusion and Distillation](../../CVPR2026/ai_safety/fedafd_multimodal_federated_learning_via_adversarial_fusion_and_distillation.md)
- [\[AAAI 2026\] CoRe-Fed: Bridging Collaborative and Representation Fairness via Federated Embedding Distillation](core-fed_bridging_collaborative_and_representation_fairness_via_federated_embedd.md)
- [\[AAAI 2026\] Credal Ensemble Distillation for Uncertainty Quantification](credal_ensemble_distillation_for_uncertainty_quantification.md)
- [\[AAAI 2026\] InfoDecom: Decomposing Information for Defending Against Privacy Leakage in Split Inference](infodecom_decomposing_information_for_defending_against_privacy_leakage_in_split.md)
- [\[CVPR 2026\] Stealing Split Learning Bottom Models by Recovering Embedding Geometry](../../CVPR2026/ai_safety/stealing_split_learning_bottom_models_by_recovering_embedding_geometry.md)

</div>

<!-- RELATED:END -->
