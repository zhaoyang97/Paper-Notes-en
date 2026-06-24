---
title: >-
  [Paper Note] CoRe-Fed: Bridging Collaborative and Representation Fairness via Federated Embedding Distillation
description: >-
  [AAAI 2026][AI Safety][Federated Learning] The CoRe-Fed framework is proposed to simultaneously address both representation fairness and collaborative fairness in federated learning through two synergistic modules: embedding-level contrastive alignment and contribution-aware aggregation, significantly improving the fairness and generalization of the global model under heterogeneous data distributions.
tags:
  - "AAAI 2026"
  - "AI Safety"
  - "Federated Learning"
  - "Fairness"
  - "Representation Alignment"
  - "Contrastive Learning"
  - "Knowledge Distillation"
date: 2026-05-08
content_hash: 2d0b9d34eba0743b
---

# CoRe-Fed: Bridging Collaborative and Representation Fairness via Federated Embedding Distillation

**Conference**: AAAI 2026  
**arXiv**: [2602.00647](https://arxiv.org/abs/2602.00647)  
**Code**: [Available](https://github.com/Noorain1/CoRe-Fed)  
**Area**: AI Security  
**Keywords**: Federated Learning, Fairness, Representation Alignment, Contrastive Learning, Knowledge Distillation

## TL;DR

The CoRe-Fed framework is proposed to simultaneously address both representation fairness and collaborative fairness in federated learning through two synergistic modules: embedding-level contrastive alignment and contribution-aware aggregation, significantly improving the fairness and generalization of the global model under heterogeneous data distributions.

## Background & Motivation

Federated learning (FL) allows multiple clients to collaboratively train models without sharing raw data. However, traditional FL algorithms face three types of bias under heterogeneous data distributions:

**Performance Bias**: Statistical heterogeneity and label correlations in client datasets cause the model to perform well on some clients while neglecting others.

**Representation Bias**: Uneven data distribution leads to poorly aligned embedding vectors of different clients in the global latent space, e.g., CIFAR-10 car/truck images producing angular deviations across different clients.

**Collaborative Bias**: Standard aggregation strategies (such as FedAvg) may suppress updates from underrepresented clients or give noisy/misaligned contributions the same weight as high-quality ones.

Existing works either focus solely on level-based fairness or deal exclusively with representation fairness, rarely considering the systematic connection between representation alignment and aggregation fairness. The core insight of CoRe-Fed is that **representation fairness and collaborative fairness are mutually reinforcing**—representation fairness improves embedding quality, while collaborative fairness ensures that high-quality embeddings are not diluted during aggregation.

## Method

### Overall Architecture

CoRe-Fed consists of two collaborative modules executed on the server side:

1. **Representation Alignment Module**: Aligns client embeddings through contrastive learning and knowledge distillation.
2. **Contribution-Aware Aggregation Module**: Dynamically adjusts client weights based on participation frequency and representation similarity.

Workflow: Client local training $\to$ embedding vector extraction $\to$ server computes global embedding $\to$ contrastive loss quantifies representation similarity $\to$ construct alignment vector to guide knowledge distillation $\to$ fair aggregation based on participation frequency and alignment scores.

### Key Designs

**Embedding Extraction and Normalization**: Each client $i$ extracts features from its local dataset to obtain an $L_2$-normalized average embedding vector. The global embedding is obtained by averaging the embeddings of the participating client set.

**Contrastive Learning Alignment**: A temperature-scaled NT-Xent loss aligns client embeddings with the global embedding while contrasting them with other clients' embeddings. The temperature parameter $\tau_c = 0.07$ controls the sharpness of the distribution.

**Embedding-Level Knowledge Distillation**: Computes the alignment vector (cosine similarity alignment score multiplied by the global embedding) to soft-align client embeddings toward the global direction:

- Update formula: $\tilde{z}_i = z_i + \beta (\tilde{z}_g^{(i)} - z_i)$
- $\beta \in [0,1]$ controls the distillation strength, preserving local client features while moving toward the global semantic space.

**Participation Frequency Estimation**: Counts client participation frequency within a dynamic sliding window $\tau = M/|C_t|$ to ensure less active clients are not forgotten.

**Sigmoid-Modulated Fair Weights**: Computes aggregation weights by combining participation frequency and representation alignment:

$$w_i = \frac{(1/f_i)^\gamma \cdot \sigma(k \cdot \rho_i)}{\sum_l (1/f_l)^\gamma \cdot \sigma(k \cdot \rho_l)}$$

- $(1/f_i)^\gamma$ amplifies the influence of less active clients.
- $\sigma(k \cdot \rho_i)$ favors clients well-aligned with the global model.
- Establishes a reward-penalty mechanism: low-frequency participation + high representation alignment = high weight.

**Gradient Reuse Strategy**: For temporarily offline clients, their historical gradients are reused within the sliding window, allowing them to still influence the global model updates.

### Loss & Training

- Local training uses standard cross-entropy loss.
- Representation alignment is performed via contrastive loss on the server side.
- Global model updates combine fair weights and (potentially reused) gradients.
- Key hyperparameters: $\gamma \in \{0.5, 2\}$ (fairness exponent), $k \in \{0.5, 2\}$ (sigmoid slope), $\beta = 0.5$ (distillation coefficient), $\tau_c = 0.07$ (temperature).
- SGD optimizer, learning rate $\eta = 0.1$, decay factor of 0.999/round, local epochs $E = 1$.

## Key Experimental Results

### Main Results

On FMNIST and CIFAR-10, with a Dirichlet($\alpha=0.5$) non-IID partition, 100 clients are set up with 20 selected to participate in each round:

| Algorithm | FMNIST Acc | D_Cosine | D_Manhattan | CIFAR-10 Acc | D_Cosine | D_Manhattan |
|---|---|---|---|---|---|---|
| FedRDN | 0.870 | 0.746 | 116.8 | 0.569 | 1.077 | 180.9 |
| FedMDFG | 0.874 | 0.587 | 88.1 | 0.681 | 0.766 | 116.3 |
| FedMGDA+ | 0.849 | 0.421 | 79.5 | 0.549 | 0.719 | 48.4 |
| Ditto | 0.862 | 0.536 | 106.5 | 0.663 | 1.251 | 104.2 |
| qFedAvg | 0.884 | 0.401 | 76.2 | 0.628 | 0.702 | 52.9 |
| **CoRe-Fed** | **0.891** | **0.294** | **73.5** | **0.722** | **0.430** | **36.0** |

CoRe-Fed improves the accuracy on CIFAR-10 by 6.0% compared to the best baseline, reducing cosine distance by 43.9% and Manhattan distance by 69%.

### Ablation Study

| Scenario | Setting | Comparison with Full Model |
|---|---|---|
| Co-Fed | Contribution-aware aggregation only | Slight improvement in fairness but accuracy drops |
| Re-Fed | Contrastive learning + knowledge distillation only | Representation bias is reduced but aggregation is still dominated by active clients |
| CoRe-Fed | Combination of both | Highest accuracy and lowest distance |

Trade-off between hyperparameters $k$ and $\gamma$: $k=2.0, \gamma=0.5$ compared to $k=0.5, \gamma=2.0$ improves accuracy by 0.81%, reduces cosine distance by 0.16%, and reduces Manhattan distance by 0.32%. An excessively high $\gamma$ will over-amplify stale updates from inactive clients.

### Key Findings

- The two types of fairness are **complementary**: using either module in isolation is less effective than their combination.
- CoRe-Fed performs particularly stably under a small batch size (50), while FedMGDA+ and FedRDN exhibit significant oscillations.
- Per-client accuracy analysis shows that CoRe-Fed achieves both high average accuracy and low variance across clients.

## Highlights & Insights

1. **Design Philosophy of a Unified Framework**: First to unify representation fairness and collaborative fairness within a single framework, demonstrating their mutually reinforcing relationship.
2. The **Sigmoid-modulated weight mechanism** cleverly combines the reciprocal of participation frequency with the alignment cosine similarity, establishing an intuitively clear reward-and-punishment mechanism.
3. The **gradient reuse strategy** allows temporarily offline clients to still influence the global model within a certain window, enhancing applicability to real-world scenarios.
4. **Embedding-level distillation instead of model-level distillation** reduces computational overhead and offers an intuitive theoretical formulation.

## Limitations & Future Work

1. Experiments are only validated on FMNIST and CIFAR-10, two relatively simple datasets, lacking large-scale evaluation.
2. The model architecture is limited to MLP and simple CNNs, without validation on modern architectures like ResNet/ViT.
3. The setting of the gradient reuse window $\tau$ is highly heuristic and lacks theoretical optimality analysis.
4. Robustness under malicious client (Byzantine robustness) scenarios is not explored.
5. The expressiveness of scalar multiplication in the alignment vector is limited; richer transformations could be considered.

## Related Work & Insights

- **FedMDFG** (Pan et al. 2023): Performs aggregation via multi-directional fair gradients, serving as the primary baseline for comparison in this work.
- **FedRDN** (Yan et al. 2025): Uses data augmentation to mitigate feature drift, but does not consider collaborative fairness in aggregation.
- **Shapley value methods** (Tastan et al. 2024): Weighted aggregation using approximations of the Shapley values of the last-layer gradients.
- Insight: Embedding space alignment can serve as a universal bridge for cross-client coordination, showing potential for generalization to heterogeneous federated learning.

## Rating

- Novelty: 4/5 - The unified framework addressing both types of fairness is a novel contribution.
- Technical Depth: 3/5 - The components are relatively mature; their combination is the primary innovation.
- Experimental Thoroughness: 3/5 - Detailed ablation and analysis, but the datasets and model scales are relatively small.
- Writing Quality: 4/5 - Clear motivation of the problem, intuitive illustrations.
- Overall: 3.5/5

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Bridging Fairness and Explainability: Can Input-Based Explanations Promote Fairness in Hate Speech Detection?](../../ICLR2026/ai_safety/bridging_fairness_and_explainability_can_input-based_explanations_promote_fairne.md)
- [\[AAAI 2026\] HealSplit: Towards Self-Healing through Adversarial Distillation in Split Federated Learning](healsplit_towards_self-healing_through_adversarial_distillation_in_split_federat.md)
- [\[ICLR 2026\] Toward Enhancing Representation Learning in Federated Multi-Task Settings](../../ICLR2026/ai_safety/toward_enhancing_representation_learning_in_federated_multi-task_settings.md)
- [\[AAAI 2026\] Credal Ensemble Distillation for Uncertainty Quantification](credal_ensemble_distillation_for_uncertainty_quantification.md)
- [\[CVPR 2026\] FedAFD: Multimodal Federated Learning via Adversarial Fusion and Distillation](../../CVPR2026/ai_safety/fedafd_multimodal_federated_learning_via_adversarial_fusion_and_distillation.md)

</div>

<!-- RELATED:END -->
