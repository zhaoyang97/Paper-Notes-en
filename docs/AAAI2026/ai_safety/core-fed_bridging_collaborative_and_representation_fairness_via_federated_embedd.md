---
title: >-
  [Paper Note] CoRe-Fed: Bridging Collaborative and Representation Fairness via Federated Embedding Distillation
description: >-
  [AAAI 2026][AI Safety][Federated Learning] This paper proposes CoRe-Fed, a framework that simultaneously addresses representation fairness and collaborative fairness in federated learning through two synergistic modules—embedding-level contrastive alignment and contribution-aware aggregation—achieving significant improvements in both fairness and generalization of the global model under heterogeneous data distributions.
tags:
  - AAAI 2026
  - AI Safety
  - Federated Learning
  - Fairness
  - Representation Alignment
  - Contrastive Learning
  - Knowledge Distillation
date: 2026-05-08
content_hash: 559cb7874af2bbb0
---

# CoRe-Fed: Bridging Collaborative and Representation Fairness via Federated Embedding Distillation

**Conference**: AAAI 2026
**arXiv**: [2602.00647](https://arxiv.org/abs/2602.00647)
**Code**: [Available](https://github.com/Noorain1/CoRe-Fed)
**Area**: AI Safety
**Keywords**: Federated Learning, Fairness, Representation Alignment, Contrastive Learning, Knowledge Distillation

## TL;DR

This paper proposes CoRe-Fed, a framework that simultaneously addresses representation fairness and collaborative fairness in federated learning through two synergistic modules—embedding-level contrastive alignment and contribution-aware aggregation—achieving significant improvements in both fairness and generalization of the global model under heterogeneous data distributions.

## Background & Motivation

Federated learning (FL) enables multiple clients to collaboratively train models without sharing raw data. However, conventional FL algorithms suffer from three types of bias under heterogeneous data distributions:

**Performance Bias**: Statistical heterogeneity and label correlations across client datasets cause the model to perform well on some clients while neglecting others.

**Representation Bias**: Uneven data distributions lead to poor alignment of client embedding vectors in the global latent space—for example, CIFAR-10 images of automobiles and trucks may produce angular discrepancies across clients.

**Collaborative Bias**: Standard aggregation strategies such as FedAvg may suppress updates from disadvantaged clients, or assign equal weight to noisy or misaligned contributions alongside high-quality ones.

Existing work either focuses solely on group-level fairness or handles only representation fairness, rarely addressing the systematic interplay between representation alignment and aggregation fairness. The core insight of CoRe-Fed is that **representation fairness and collaborative fairness are mutually reinforcing**—improved representation quality enhances embeddings, while fair aggregation ensures high-quality embeddings are not diluted during aggregation.

## Method

### Overall Architecture

CoRe-Fed comprises two synergistic modules executed on the server side:

1. **Representation Alignment Module**: Aligns client embeddings via contrastive learning and knowledge distillation.
2. **Contribution-Aware Aggregation Module**: Dynamically adjusts client weights based on participation frequency and representation similarity.

Workflow: local client training → embedding extraction → server computes global embeddings → contrastive loss quantifies representation similarity → alignment vectors guide knowledge distillation → fair aggregation based on participation frequency and alignment scores.

### Key Designs

**Embedding Extraction and Normalization**: Each client $i$ extracts features from its local dataset to obtain an L2-normalized mean embedding vector. The global embedding is computed as the average of embeddings from all participating clients.

**Contrastive Learning Alignment**: A temperature-scaled NT-Xent loss aligns each client's embedding with the global embedding while contrasting against embeddings from other clients. The temperature parameter $\tau_c = 0.07$ controls distribution sharpness.

**Embedding-Level Knowledge Distillation**: An alignment vector is computed as the cosine similarity alignment score multiplied by the global embedding, softly steering the client embedding toward the global direction:

- Update rule: $\tilde{z}_i = z_i + \beta \cdot (\tilde{z}_g^{(i)} - z_i)$
- $\beta \in [0,1]$ controls distillation strength, preserving local client characteristics while encouraging convergence toward the global semantic space.

**Participation Frequency Estimation**: Client participation frequency is tracked within a dynamic sliding window $\tau = M / |C_t|$, ensuring low-activity clients are not forgotten.

**Sigmoid-Modulated Fair Weights**: Aggregation weights are computed jointly from participation frequency and representation alignment:

- Weight formula: $w_i = \frac{(1/f_i)^\gamma \cdot \sigma(k \cdot \rho_i)}{\sum_l (1/f_l)^\gamma \cdot \sigma(k \cdot \rho_l)}$
- $(1/f_i)^\gamma$ amplifies the influence of low-participation clients.
- $\sigma(k \cdot \rho_i)$ favors clients well-aligned with the global model.
- This forms a reward-penalty mechanism: low frequency + high alignment = high weight.

**Gradient Reuse Strategy**: For temporarily offline clients, historical gradients within the sliding window are reused so that these clients continue to influence global model updates.

### Loss & Training

- Local training uses standard cross-entropy loss.
- The server performs representation alignment via contrastive loss.
- Global model updates incorporate fair weights and (potentially reused) gradients.
- Key hyperparameters: $\gamma \in \{0.5, 2\}$ (fairness exponent), $k \in \{0.5, 2\}$ (sigmoid slope), $\beta = 0.5$ (distillation coefficient), $\tau_c = 0.07$ (temperature).
- SGD optimizer with learning rate $\eta = 0.1$, decay factor $0.999$/round, local epochs $E = 1$.

## Key Experimental Results

### Main Results

Evaluated on FMNIST and CIFAR-10 with Dirichlet($\alpha=0.5$) non-IID partitioning, 100 clients with 20 selected per round:

| Algorithm | FMNIST Acc | D_Cosine | D_Manhattan | CIFAR-10 Acc | D_Cosine | D_Manhattan |
|---|---|---|---|---|---|---|
| FedRDN | 0.870 | 0.746 | 116.8 | 0.569 | 1.077 | 180.9 |
| FedMDFG | 0.874 | 0.587 | 88.1 | 0.681 | 0.766 | 116.3 |
| FedMGDA+ | 0.849 | 0.421 | 79.5 | 0.549 | 0.719 | 48.4 |
| Ditto | 0.862 | 0.536 | 106.5 | 0.663 | 1.251 | 104.2 |
| qFedAvg | 0.884 | 0.401 | 76.2 | 0.628 | 0.702 | 52.9 |
| **CoRe-Fed** | **0.891** | **0.294** | **73.5** | **0.722** | **0.430** | **36.0** |

On CIFAR-10, CoRe-Fed improves accuracy by 6.0% over the best baseline, reduces angular distance by 43.9%, and reduces Manhattan distance by 69%.

### Ablation Study

| Variant | Setting | Comparison to Full Model |
|---|---|---|
| Co-Fed | Contribution-aware aggregation only | Marginal fairness gain but accuracy drop |
| Re-Fed | Contrastive learning + distillation only | Reduced representation bias but aggregation still dominated by majority clients |
| CoRe-Fed | Both modules combined | Highest accuracy and lowest distance |

Hyperparameter trade-off: $k=2.0, \gamma=0.5$ yields +0.81% accuracy, −0.16 cosine distance, and −0.32 Manhattan distance compared to $k=0.5, \gamma=2.0$. Excessively large $\gamma$ over-amplifies stale updates from inactive clients.

### Key Findings

- The two fairness objectives are **complementary**: neither module alone matches the performance of their combination.
- CoRe-Fed is particularly stable under small batch sizes (50), whereas FedMGDA+ and FedRDN exhibit notable oscillations.
- Per-client accuracy analysis demonstrates that CoRe-Fed achieves both high mean accuracy and low inter-client variance.

## Highlights & Insights

1. **Unified Framework Design Philosophy**: CoRe-Fed is the first to jointly address representation fairness and collaborative fairness within a single framework, demonstrating their mutually reinforcing relationship.
2. **Sigmoid-Modulated Weight Mechanism** elegantly combines the inverse of participation frequency with alignment cosine similarity, yielding an intuitive reward-penalty scheme.
3. **Gradient Reuse Strategy** allows temporarily offline clients to continue influencing global model updates within a defined window, improving applicability to real-world scenarios.
4. **Embedding-Level Distillation Rather Than Model-Level Distillation** incurs lower computational overhead and is theoretically straightforward.

## Limitations & Future Work

1. Experiments are conducted only on FMNIST and CIFAR-10, two relatively simple datasets, lacking large-scale validation.
2. Model architectures are limited to MLPs and simple CNNs; generalization to modern architectures such as ResNet and ViT remains unverified.
3. The sliding window parameter $\tau$ is set heuristically, without theoretical optimality analysis.
4. Robustness under Byzantine settings (malicious clients) is not discussed.
5. The scalar multiplication in the alignment vector has limited expressiveness; richer transformations could be explored.

## Related Work & Insights

- **FedMDFG** (Pan et al. 2023): Performs aggregation via multi-directional fair gradients; serves as the primary baseline.
- **FedRDN** (Yan et al. 2025): Mitigates feature drift through data augmentation but does not address aggregation fairness.
- **Shapley Value Methods** (Tastan et al. 2024): Approximate Shapley values via last-layer gradients for weighted aggregation.
- Insight: Embedding space alignment can serve as a general coordination bridge across clients, with potential applicability to model-heterogeneous federated learning.

## Rating

- Novelty: 4/5 — A unified framework jointly addressing both fairness objectives is a genuine contribution.
- Technical Depth: 3/5 — Individual components are well-established; the primary innovation lies in their combination.
- Experimental Thoroughness: 3/5 — Ablation and analysis are detailed, but dataset and model scales are limited.
- Writing Quality: 4/5 — Problem motivation is clear and figures are intuitive.
- Overall: 3.5/5

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Bridging Fairness and Explainability: Can Input-Based Explanations Promote Fairness in Hate Speech Detection?](../../ICLR2026/ai_safety/bridging_fairness_and_explainability_can_input-based_explanations_promote_fairne.md)
- [\[AAAI 2026\] HealSplit: Towards Self-Healing through Adversarial Distillation in Split Federated Learning](healsplit_towards_self-healing_through_adversarial_distillation_in_split_federat.md)
- [\[ICLR 2026\] Toward Enhancing Representation Learning in Federated Multi-Task Settings](../../ICLR2026/ai_safety/toward_enhancing_representation_learning_in_federated_multi-task_settings.md)
- [\[CVPR 2026\] FedAFD: Multimodal Federated Learning via Adversarial Fusion and Distillation](../../CVPR2026/ai_safety/fedafd_multimodal_federated_learning_via_adversarial_fusion_and_distillation.md)
- [\[CVPR 2026\] FedRE: A Representation Entanglement Framework for Model-Heterogeneous Federated Learning](../../CVPR2026/ai_safety/fedre_a_representation_entanglement_framework_for_model-heterogeneous_federated_.md)

<!-- RELATED:END -->
