---
title: >-
  [Paper Note] Toward Enhancing Representation Learning in Federated Multi-Task Settings
description: >-
  [ICLR 2026][AI Safety][Federated multi-task learning] This paper proposes the Muscle loss — an N-tuple-level multi-model contrastive learning objective whose minimization is equivalent to maximizing a lower bound on the mutual information among all model representations. Building on this, the FedMuscle algorithm aligns the representation spaces of heterogeneous models via a public dataset, naturally handling both model and task heterogeneity. FedMuscle consistently outperforms state-of-the-art baselines across CV/NLP multi-task settings, with gains of up to +28.65%.
tags:
  - ICLR 2026
  - AI Safety
  - Federated multi-task learning
  - contrastive learning
  - Muscle loss
  - model heterogeneity
  - mutual information maximization
date: 2026-05-08
content_hash: 63172456ed37c8f6
---

# Toward Enhancing Representation Learning in Federated Multi-Task Settings

**Conference**: ICLR 2026
**arXiv**: [2602.01626](https://arxiv.org/abs/2602.01626)
**Code**: Available (provided in supplementary materials)
**Institution**: Huawei Noah's Ark Lab, Montreal
**Area**: AI Safety
**Keywords**: Federated multi-task learning, contrastive learning, Muscle loss, model heterogeneity, mutual information maximization

## TL;DR

This paper proposes the Muscle loss — an N-tuple-level multi-model contrastive learning objective whose minimization is equivalent to maximizing a lower bound on the mutual information among all model representations. Building on this, the FedMuscle algorithm aligns the representation spaces of heterogeneous models via a public dataset, naturally handling both model and task heterogeneity. FedMuscle consistently outperforms state-of-the-art baselines across CV/NLP multi-task settings, with gains of up to +28.65%.

## Background & Motivation

**Background**: Federated multi-task learning (FMTL) enables users with different tasks and models to collaboratively train without sharing private data. With the proliferation of foundation models (FMs), users may fine-tune different pretrained models subject to their resource constraints, making model and task heterogeneity the norm.

**Limitations of Prior Work**: Existing FMTL methods (FeSTA, FedBone, FedHCA2, FedLPS, etc.) assume fully or partially homogeneous model architectures (e.g., shared encoders), restricting users' freedom to select their own models.

**Inadequacy of Pairwise Alignment**: When more than two models are involved, existing methods apply InfoNCE pairwise to every model pair: $\mathcal{L}^n_{Pairwise} = \sum_{m \neq n} \mathcal{L}^{n,m}_{InfoNCE}$. This decomposition captures only binary dependencies and fails to model the joint dependencies among N model representations.

**Limitations of Knowledge Distillation Methods**: KD-based methods such as FedDF and FCCL require models to share the same logit dimensionality, i.e., all models must be associated with the same task — making cross-task heterogeneity infeasible.

**Lack of Theoretical Grounding for Gramian Contrastive Loss**: The Gramian contrastive loss proposed by Cicchetti et al. (2025) can align multiple models simultaneously but lacks theoretical justification and incurs high computational cost (requiring the determinant of a Gramian matrix, leading to $(M+1)^3\times$ higher computation).

**Core Insight**: The fundamental purpose of sharing model parameters is to establish a shared representation space. Therefore, one should directly learn a shared representation space rather than enforcing parameter sharing. N-tuple-level contrastive learning combined with mutual information maximization theory offers a principled approach to achieving this goal.

## Method

### 1. Muscle Loss Function

The core innovation is the extension from pairwise to N-tuple joint alignment. Given N models, the anchor is $\bm{z}_i^n$, positive samples are the representations of all models on the same data point $i$, and negative samples are combinations where at least one model corresponds to a different data point:

$$\mathcal{L}^n_{\text{Muscle}}(\bm{z}_i^n) = -\log \frac{\alpha_{(i,...,i)} \exp\left(\bm{z}_i^n \cdot \sum_{m \neq n} \bm{z}_i^m / \tau^{(N)}_{n,m}\right)}{\sum_{\bm{j} \in \mathcal{J}^n} \alpha_{\bm{j}} \exp\left(\bm{z}_i^n \cdot \sum_{m \neq n} \bm{z}^m_{j_m} / \tau^{(N)}_{n,m}\right)}$$

### 2. Key Design: Weighting Coefficient $\alpha_{\bm{j}}$

$$\alpha_{\bm{j}} = \exp\left(-\frac{1}{2} \sum_{m \neq n} \sum_{m' \neq n,m} \gamma^{(N)}_{m,m'} \bm{z}^m_{j_m} \cdot \bm{z}^{m'}_{j_{m'}}\right)$$

where $\gamma^{(N)}_{m,m'} = 1/\tau^{(N-1)}_{m,m'} - 1/\tau^{(N)}_{m,m'}$ is always positive. This implies:
- The less similar the non-anchor model representations are within a negative sample combination → the larger $\alpha_{\bm{j}}$
- The Muscle loss places greater emphasis on negative sample combinations that are already highly dissimilar among themselves — information entirely ignored by pairwise methods
- The weighting coefficients are theoretically motivated, derived from the optimal density ratio rather than designed heuristically

### 3. Theoretical Guarantee via Mutual Information Maximization (Theorem 1)

$$I(\bm{z}_i^n; \{\bm{z}_i^m\}_{m \neq n}) \geq (N-1)\log(B) - \mathbb{E}\mathcal{L}^n_{\text{Muscle}}(\bm{z}_i^n)$$

Minimizing the Muscle loss is equivalent to maximizing a lower bound on the mutual information among all model representations, providing a theoretical guarantee for the effectiveness of knowledge transfer.

### 4. FedMuscle Algorithm

1. Each communication round: each client trains for $E$ epochs on local data $\mathcal{D}^n$, updating the full model $\bm{\theta}^n$.
2. Contrastive learning phase ($T$ rounds): each client extracts a representation matrix $\bm{Z}^n \in \mathbb{R}^{B \times d}$ on the public dataset $\mathcal{D}$ and uploads it to the server.
3. Server computation: for each client $n$, randomly sample $M$ representation matrices from the remaining $N-1$ clients; compute the aggregated matrix $\bm{S}^n$ and weight vector $\bm{\alpha}^n$; return to the client.
4. Client update: minimize the CL loss $\mathcal{L}^n_{CL}$, updating only the representation model $\bm{w}^n$.

### 5. Communication Efficiency

- Uplink: each client transmits a $B \times d$ representation matrix (e.g., $32 \times 256$).
- Downlink: randomly sample $M$ clients' representations (rather than all $N-1$), reducing complexity from $B^{N-1}$ to $B^M$.
- No model parameters are transmitted, providing additional privacy protection — particularly important for pretrained FMs.

## Key Experimental Results

### Table 1: Setup 1 — Uni-Modal Benchmark (Pascal VOC Public Dataset)

| Method | User1 MLC | User4 IC100 | User6 IC10 | Δ (%) |
|--------|-----------|-------------|------------|-------|
| Local Training | 42.17 | 24.77 | 43.77 | 0.00 |
| CoFED | 47.47 | 24.67 | 43.40 | +5.83 |
| SimCLR | 40.80 | 27.43 | 49.03 | +3.57 |
| SAGE | 41.97 | 24.50 | 43.33 | +0.96 |
| FedHeNN | 41.27 | 24.10 | 41.63 | −0.41 |
| **FedMuscle** | **46.33** | **36.67** | **66.57** | **+26.70** |

### Table 2: Setup 2 — Multi-Modal + Multi-Task (CV + NLP, 10 Clients)

| Method | MLC (User 1–3) | IC100 (User 4–5) | IC10 (User 6) | SS (User 7–8) | TC (User 9–10) | Δ (%) |
|--------|----------------|------------------|---------------|---------------|----------------|-------|
| Local Training | 42–44 | 24–25 | 43.77 | 32–34 | 41–56 | 0.00 |
| **FedMuscle** | **47–51** | **29–36** | **61.60** | **33–34** | **46–54** | **+14.39** |

### Table 3: CreamFL Integration (35 Clients, 5K Test Images)

| Method | i2t_R@1 | t2i_R@1 | Δ (%) |
|--------|---------|---------|-------|
| Local Training | 24.78 | 17.72 | 0.00 |
| CreamFL | 24.48 | 17.96 | +0.88 |
| **CreamFL + Muscle** | **25.50** | **18.20** | **+1.94** |

## Key Findings

1. **Muscle loss consistently outperforms all baselines**: Across three different public datasets (Pascal VOC / COCO / CIFAR-100), FedMuscle achieves Δ of +26.70% / +28.65% / +16.88%, far exceeding the second-best method CoFED at +5.83% / +9.85% / +5.99%.

2. **Public dataset quality affects performance**: Datasets with rich visual detail (COCO / Pascal VOC) yield the best results; CIFAR-100 performs slightly lower due to limited image detail. Nevertheless, FedMuscle remains effective regardless of the public dataset used.

3. **Muscle vs. Gramian vs. Pairwise**: On Pascal VOC / COCO / CIFAR-100, Muscle outperforms the Gramian loss by 11.2% / 28.4% / 11.1%, demonstrating the clear advantage of the theoretically derived weighting coefficients.

4. **Effectiveness under Non-IID settings**: With 12 clients × 4 tasks × Dirichlet($\alpha=0.1$) non-IID partitioning, FedMuscle achieves Δ = +17.40%, demonstrating strong robustness.

5. **$M=3$ offers the optimal cost-performance trade-off**: As $M$ increases from 1 to 5, Δ improves from +17.90% to +27.74%, but communication overhead grows exponentially from 0.004 GB to 381.565 GB per round. $M=3$ (Δ = +26.70%, 0.956 GB) represents the optimal balance.

6. **Muscle is plug-and-play**: Replacing LCR/GCA in CreamFL with Muscle improves multi-modal retrieval performance, demonstrating strong generalizability.

## Highlights & Insights

- **Paradigm shift**: From "sharing parameters" to "sharing representation space" — the core objective of FL is not parameter synchronization but representation alignment. This perspective is more fundamental and naturally accommodates model heterogeneity.
- **Theoretical necessity of N-tuple formulation**: Analogous to the many-body problem, the joint dependencies among N models cannot be decomposed into $\binom{N}{2}$ pairwise dependencies. The weighting coefficients $\alpha_{\bm{j}}$ precisely encode these higher-order interactions.
- **Tight mutual information lower bound**: The MI lower bound becomes tighter as batch size $B$ increases, consistent with empirical observations (larger $B$ yields better performance).
- **Practicality of LoRA fine-tuning**: Applying LoRA (rank = 16) to pretrained FMs enables parameter-efficient fine-tuning with heterogeneous model support, closely reflecting realistic deployment scenarios.

## Limitations & Future Work

1. **Exponential growth of communication overhead with $M$**: The downlink communication cost scales as $B^M \times d$, reaching 381 GB per round at $M=5$, limiting scalability to large-scale client deployments.
2. **Dependence on public datasets**: The method requires a public dataset accessible to all clients (5,000 samples), which may be unavailable in privacy-sensitive scenarios.
3. **Limited cross-modal knowledge transfer**: In Setup 2, gains on SS and TC tasks are relatively modest (SS clients improve by only +0.6–3% mIoU), indicating room for improvement in cross-modal knowledge transfer.
4. **Manual temperature parameter tuning**: $\tau^{(N)}_{n,m}$ and $\tau^{(N-1)}_{n,m}$ are fixed at 0.2 and 0.15, respectively, without an adaptive temperature adjustment mechanism.

## Related Work & Insights

| Dimension | FedMuscle (Ours) | FedHeNN (Makhija 2022) | CreamFL (Yu 2023) |
|-----------|-----------------|----------------------|-------------------|
| Alignment | N-tuple Muscle loss | CKA proximal term | LCR + GCA (pairwise) |
| Theoretical guarantee | MI lower bound | None (CKA reliability questionable) | None |
| Model heterogeneity | Fully supported | Supported | Supported (requires global model) |
| Task heterogeneity | Fully supported | Partially supported | Not supported (same task only) |
| Communication content | Representation matrix | Model parameters | Representations + gradients |
| Objective | Each client's local model | Each client's local model | Global model |

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — N-tuple multi-model contrastive learning with MI theoretical guarantee and theoretically derived weighting coefficients; exceptionally original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — CV + NLP multi-modal evaluation, diverse heterogeneous settings, and extensive ablation studies; lacks validation at larger scale (> 12 clients).
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Theoretical derivations are rigorous and clear, notation is consistent, and the logical flow from motivation to method to experiments is coherent.
- **Value**: ⭐⭐⭐⭐ — Makes a principled contribution to heterogeneous FL; exponential growth in communication overhead remains a bottleneck for practical deployment.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Active Membership Inference Test (aMINT): Enhancing Model Auditability with Multi-Task Learning](../../ICCV2025/ai_safety/active_membership_inference_test_amint_enhancing_model_auditability_with_multi-t.md)
- [\[CVPR 2026\] FedRE: A Representation Entanglement Framework for Model-Heterogeneous Federated Learning](../../CVPR2026/ai_safety/fedre_a_representation_entanglement_framework_for_model-heterogeneous_federated_.md)
- [\[ICLR 2026\] Dataless Weight Disentanglement in Task Arithmetic via Kronecker-Factored Approximate Curvature](dataless_weight_disentanglement_in_task_arithmetic_via_kronecker-factored_approx.md)
- [\[ICLR 2026\] Adaptive Methods Are Preferable in High Privacy Settings: An SDE Perspective](adaptive_methods_are_preferable_in_high_privacy_settings_an_sde_perspective.md)
- [\[ICLR 2026\] Sample-Efficient Distributionally Robust Multi-Agent Reinforcement Learning via Online Interaction](sample-efficient_distributionally_robust_multi-agent_reinforcement_learning_via_.md)

<!-- RELATED:END -->
