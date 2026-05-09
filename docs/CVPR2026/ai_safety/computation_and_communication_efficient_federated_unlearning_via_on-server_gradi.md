---
title: >-
  [Paper Note] Computation and Communication Efficient Federated Unlearning via On-server Gradient Conflict Mitigation and Expression
description: >-
  [CVPR 2026][AI Safety][Federated Unlearning] This paper proposes FOUL, a two-stage framework that decouples causal and non-causal features during training and performs on-server gradient conflict matching during unlearning, achieving efficient federated unlearning with low communication overhead without accessing client data.
tags:
  - CVPR 2026
  - AI Safety
  - Federated Unlearning
  - Gradient Conflict
  - Causal Disentanglement
  - On-server Aggregation
  - Privacy
date: 2026-05-08
content_hash: 117757efa6fa707c
---

# Computation and Communication Efficient Federated Unlearning via On-server Gradient Conflict Mitigation and Expression

**Conference**: CVPR 2026
**arXiv**: [2603.13795](https://arxiv.org/abs/2603.13795)
**Code**: Paper claims reproducibility but no public link provided
**Area**: AI Safety
**Keywords**: Federated Unlearning, Gradient Conflict, Causal Disentanglement, On-server Aggregation, Privacy

## TL;DR

This paper proposes FOUL, a two-stage framework that decouples causal and non-causal features during training and performs on-server gradient conflict matching during unlearning, achieving efficient federated unlearning with low communication overhead without accessing client data.

## Background & Motivation

**Privacy regulation drivers**: Data protection laws such as GDPR grant users the "right to be forgotten," requiring federated learning (FL) models to remove specific clients' data contributions.

**Prohibitive retraining cost**: Although full retraining (Retrain) serves as the gold standard for federated unlearning (FUL), its computational and communication costs are prohibitively high for large-scale real-world deployments.

**Dilemma of approximate unlearning**: Most existing approximate methods rely on simultaneous access to both retained and forgotten data locally ($\mathcal{L}_{\text{unlearn}}=\mathcal{L}_{\text{retain}}-\mathcal{L}_{\text{forget}}$), which breaks down in client-wise unlearning scenarios where the forgetting client cannot access retained data.

**Repeated overhead of subnetwork identification**: Existing methods identify subnetworks to be forgotten via gradient conflict detection, but must repeat this process as the model evolves each round, incurring significant additional computation.

**Causal structure motivation**: Causal invariant representations ($\mathcal{Z}_K$) are domain-invariant, while non-causal representations ($\mathcal{Z}_V$) encode domain-specific information. Unlearning should target only the non-causal component to avoid corrupting general knowledge.

**Missing server-side aggregation**: Existing FUL methods cannot simultaneously exploit gradient information from both retained and forgotten clients at the server to perform efficient unlearning.

## Method

### Overall Architecture

FOUL operates in two stages:

- **Stage 1 — Learning to Unlearn (L2U)**: During federated training, the feature extractor is disentangled into a causal subnetwork $\theta_K$ (extracting domain-invariant features) and a non-causal subnetwork $\theta_V$ (extracting domain-specific features), preparing the model for future unlearning.
- **Stage 2 — On-server Gradient Matching**: Upon an unlearning request, only $\theta_V$ is updated. The server aggregates gradients via conflict matching — aligning the aggregated gradient with retained clients' gradients while opposing the forgotten client's gradient.

### Key Design 1: Causal/Non-causal Feature Disentanglement

- **Function**: Each client's local model is parameterized as $\theta_u = \{\theta_E, \theta_K, \theta_V, \theta_D, \theta_C\}$ (shallow extractor, causal encoder, non-causal encoder, decoder, classifier), jointly trained with four loss terms.
- **Mechanism**: The causal encoder applies a prototypical network loss $\mathcal{L}_K$ to enforce compact intra-class representations (domain-invariant); the non-causal encoder applies a hinge loss $\mathcal{L}_V$ to maximize intra-class variance (capturing domain-specific variation); the classification loss $\mathcal{L}_{\text{gtc}}$ ensures causal representations are predictive of labels; the reconstruction loss $\mathcal{L}_{\text{rec}}$ ensures the two representations jointly reconstruct the original input.
- **Design Motivation**: Based on the causal structured model (Definition 5), causal factors $\mathcal{Z}_K$ and non-causal factors $\mathcal{Z}_V$ are independent and sufficient. Unlearning only requires intervention on $\mathcal{Z}_V$, leaving $\mathcal{Z}_K$ intact and thereby preserving retained knowledge.

### Key Design 2: On-server Gradient Conflict Matching

- **Function**: The server identifies an aggregated gradient $g_\text{FOUL}$ that has a positive inner product with retained clients' gradients (aligned direction) and a negative inner product with the forgotten client's gradient (conflicting direction).
- **Mechanism**: Learnable weight coefficients $\Gamma = \{\gamma_u^{(r)}\}$ are introduced, reducing the high-dimensional gradient optimization to a low-dimensional optimization over $U$ coefficients. Theorem 1 provides a closed-form solution: $\nabla_\text{FOUL}^{(r)} = \nabla_\text{FL}^{(r)} + \frac{\kappa \|\nabla_\text{FL}^{(r)}\|}{\|\nabla_{\Gamma_\mathcal{R}}^{(r)} - \nabla_{\Gamma_\mathcal{F}}^{(r)}\|}(\nabla_{\Gamma_\mathcal{R}}^{(r)} - \nabla_{\Gamma_\mathcal{F}}^{(r)})$.
- **Design Motivation**: Directly optimizing in the full parameter space is computationally prohibitive and generalizes poorly without data guidance. Coefficient parameterization drastically reduces the search space ($U \ll d$) while requiring no access to any client's local data, preserving privacy.

### Key Design 3: Hinge Variance Loss for the Non-causal Subnetwork

- **Function**: A hinge loss formulation constrains the intra-class variance of non-causal representations: $\mathcal{L}_V = \frac{1}{C}\sum_{c=1}^{C}\max(0, 1-\sqrt{\text{Var}(\mathcal{Z}_V^c)+\epsilon})$.
- **Mechanism**: The non-causal encoder maximizes intra-class variance to capture domain-specific information, with the hinge bound preventing trivial suppression — starting from a high value in early training to avoid being overshadowed by other losses, while the zero lower bound prevents $\mathcal{L}_V$ from dominating in later stages.
- **Design Motivation**: Directly maximizing MSE-based variance causes it to be suppressed by other high-magnitude losses during early training (when variance is low) and over-dominates other objectives at convergence. The hinge formulation stabilizes multi-task training dynamics.

## Loss & Training

Total loss during the L2U stage (per client):

$$\mathcal{L}_{\text{L2U}}(\theta_u) = \alpha_V \mathcal{L}_V + \alpha_K \mathcal{L}_K + \alpha_{\text{gtc}} \mathcal{L}_{\text{gtc}} + \mathcal{L}_{\text{rec}}$$

- $\mathcal{L}_K$: Prototypical network contrastive loss (Eq. 4), pulling intra-class causal representations closer
- $\mathcal{L}_V$: Hinge variance loss (Eq. 5), pushing intra-class non-causal representations apart
- $\mathcal{L}_{\text{gtc}}$: Cross-entropy classification loss (Eq. 7), causal representations → labels
- $\mathcal{L}_{\text{rec}}$: MSE reconstruction + KL divergence (Eq. 8), causal + non-causal → reconstructed input

During the unlearning stage, only $\theta_V$ is updated while $\theta_K$ is frozen; server-side unlearning is executed via the closed-form gradient aggregation in Theorem 1.

## Key Experimental Results

| Method | FA ↓ | RA ↑ | TA ↑ | MIA ↓ | Comm. (MB) | Comp. (FLOPs) |
|--------|------|------|------|-------|------------|---------------|
| Retrain | 70.51 | 82.84 | 77.45 | 50.02 | 42.73 | 5.81e16 |
| FATS | 74.45 | 80.91 | 75.98 | 55.72 | 42.73 | 5.81e16 |
| NoT | 73.24 | 79.25 | 75.28 | 59.13 | 34.72 | 3.38e16 |
| FUSED | 75.94 | 79.34 | 76.86 | 58.72 | 0.98 | 2.81e16 |
| **FOUL (L2U)** | **69.53** | **93.11** | **77.14** | 53.82 | **16.02** | **2.35e16** |
| **FOUL (L2U+Unlearn)** | 70.97 | 92.33 | 76.43 | **51.93** | 16.02 | 2.35e16 |

> PACS dataset, ResNet-18 backbone, 20 clients, IID setting

| Method | FA ↓ | RA ↑ | TA ↑ | MIA ↓ |
|--------|------|------|------|-------|
| Retrain | 30.64 | 42.41 | 38.94 | 50.71 |
| FATS | 33.07 | 40.96 | 37.34 | 60.81 |
| **FOUL (L2U)** | **27.97** | **43.81** | 38.16 | **56.40** |
| **FOUL (L2U+Unlearn)** | 29.92 | 42.13 | **39.16** | 57.11 |

> TerraIncognita dataset, ResNet-50 backbone

**Time-to-Forget (T2F)**: FOUL reaches optimal FA within <50 rounds with T2F > 0.32/round, compared to Retrain which requires 75 rounds at T2F ≈ 0.13/round — approximately 2.5× faster unlearning speed.

## Highlights & Insights

- **Elegant causal disentanglement**: The paper introduces the causal invariance principle from domain generalization into federated unlearning, theoretically grounding the insight that unlearning need only act on the non-causal subnetwork, eliminating repeated subnetwork identification overhead.
- **Zero-data server-side unlearning**: The entire unlearning stage is executed at the server via gradient coefficient optimization, requiring no access to any client's local data and genuinely preserving privacy.
- **Dimensionality collapse optimization**: Gradient optimization in $d$-dimensional space is reduced to $U$ coefficient variables ($U \ll d$), substantially lowering server-side computational complexity and improving generalization.
- **RA surpassing Retrain**: FOUL achieves RA of 93.11 on PACS versus 82.84 for Retrain, demonstrating that the disentangled causal subnetwork retains purer general knowledge.
- **Communication cost halved**: Transmitting only non-causal subnetwork parameters reduces communication overhead from 42.73 MB to 16.02 MB.
- **T2F metric introduced**: A new metric for measuring unlearning speed, filling a gap in the FUL evaluation framework along the efficiency dimension.

## Limitations & Future Work

- Validation is limited to domain generalization benchmarks (PACS/VLCS/OfficeHome/TerraIncognita); the method has not been tested in real-world FL deployments (e.g., medical or mobile settings).
- Experiments are confined to ResNet-18/50 backbones; generalization to modern architectures such as ViT remains unverified.
- The quality of causal/non-causal disentanglement depends on hyperparameters $\alpha_V, \alpha_K, \alpha_{\text{gtc}}$, and the paper provides insufficient robustness analysis.
- The unlearning stage assumes full client participation; partial dropout and asynchronous scenarios are not discussed.
- Most experiments are conducted under IID settings; performance under Non-IID distributions warrants deeper investigation.

## Related Work & Insights

- **Federated unlearning baselines**: FedRecovery, FFMU, FUSED, MoDE, and others approach unlearning from different angles, but none simultaneously resolves the computation–communication–privacy trilemma.
- **Causal representation learning**: The causal structured model provides a theoretical foundation for disentanglement, which may transfer to other selective-forgetting scenarios such as continual learning and domain adaptation.
- **Gradient conflict and multi-task learning**: The idea of gradient direction alignment/conflict originates from MTL and domain generalization (e.g., PCGrad, Fish); this paper is the first to apply it in federated unlearning for dual objectives of retention and forgetting.
- **Insight**: The proactive strategy of embedding disentangled structures during training to reduce future unlearning costs is worth exploring in the context of RLHF unlearning for LLM fine-tuning.

## Rating

| Dimension | Score |
|-----------|-------|
| Novelty | ⭐⭐⭐⭐ |
| Technical Depth | ⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐ |
| Practicality | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Efficient Fairness-Performance Pareto Front Computation](../../NeurIPS2025/ai_safety/efficient_fairness-performance_pareto_front_computation.md)
- [\[AAAI 2026\] SecMoE: Communication-Efficient Secure MoE Inference via Select-Then-Compute](../../AAAI2026/ai_safety/secmoe_communication-efficient_secure_moe_inference_via_select-then-compute.md)
- [\[CVPR 2026\] SubFLOT: Submodel Extraction for Efficient and Personalized Federated Learning via Optimal Transport](subflot_submodel_extraction_for_efficient_and_personalized_federated_learning_vi.md)
- [\[NeurIPS 2025\] Efficient Verified Machine Unlearning for Distillation](../../NeurIPS2025/ai_safety/efficient_verified_machine_unlearning_for_distillation.md)
- [\[AAAI 2026\] Robust Watermarking on Gradient Boosting Decision Trees](../../AAAI2026/ai_safety/robust_watermarking_on_gradient_boosting_decision_trees.md)

<!-- RELATED:END -->
