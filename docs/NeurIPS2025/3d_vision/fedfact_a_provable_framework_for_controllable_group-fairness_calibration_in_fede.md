---
title: >-
  [Paper Note] FedFACT: A Provable Framework for Controllable Group-Fairness Calibration in Federated Learning
description: >-
  [NeurIPS 2025][3D Vision][Federated Learning] This paper proposes FedFACT, a framework that characterizes the structure of the **Bayes-optimal fair classifier** under federated learning…
tags:
  - "NeurIPS 2025"
  - "3D Vision"
  - "Federated Learning"
  - "Group Fairness"
  - "Bayes-Optimal Classifier"
  - "Cost-Sensitive Learning"
  - "Post-Processing Calibration"
date: 2026-05-08
content_hash: 9baa4ac926b8ce3d
---

# FedFACT: A Provable Framework for Controllable Group-Fairness Calibration in Federated Learning

**Conference**: NeurIPS 2025
**arXiv**: [2506.03777](https://arxiv.org/abs/2506.03777)  
**Code**: N/A  
**Area**: AI Safety
**Keywords**: Federated Learning, Group Fairness, Bayes-Optimal Classifier, Cost-Sensitive Learning, Post-Processing Calibration

## TL;DR

This paper proposes FedFACT, a framework that characterizes the structure of the **Bayes-optimal fair classifier** under federated learning, and reduces fair federated learning to **personalized cost-sensitive learning** (in-processing) and **bi-level optimization** (post-processing), respectively. It is the first to achieve controllable coordination between global and local fairness in multi-class settings, with convergence and generalization guarantees.

## Background & Motivation

**Background**: Federated learning is increasingly deployed in high-stakes domains such as healthcare and finance, where ensuring group fairness across sensitive attributes (e.g., gender, race) is essential. Two notions of fairness arise in FL: **global fairness** (aggregate model disparity across all clients) and **local fairness** (disparity within each individual client).

**Limitations of Prior Work**:
   - **Inherent tension between global and local fairness**: Statistical heterogeneity across client data distributions causes the two fairness objectives to conflict.
   - **Uncontrollable accuracy–fairness trade-off**: The **non-decomposability** and **non-differentiability** of fairness metrics make optimization extremely challenging.
   - Existing methods largely rely on surrogate fairness losses, whose inevitable approximation gap leads to suboptimal performance.

**Key Challenge**: Existing methods either focus exclusively on global fairness (sacrificing local) or local fairness (sacrificing global), and most are limited to binary classification. The core challenge is: under multi-class × multi-sensitive-attribute × joint global/local fairness constraints, how can one find an optimal classifier with minimal accuracy degradation?

**Goal**: To design a theoretically optimal and controllable framework for group-fairness calibration in federated learning that simultaneously guarantees global and local fairness for multi-class classification.

**Key Insight**: The paper adopts a **Bayes-optimal** perspective—first characterizing the structure of the optimal classifier under fairness constraints in the federated setting, then deriving practical algorithms from this characterization.

**Core Idea**: The federated Bayes-optimal fair classifier can be accurately approximated via per-client cost-sensitive classification or plug-in-based bi-level optimization, where the cost matrix $\mathbf{M}^{\lambda,\mu}(a,k)$ controls the degree of global/local fairness through dual variables.

## Method

### Overall Architecture

FedFACT comprises two complementary approaches:
- **In-processing**: Intervenes during training by modifying the training loss to implicitly satisfy fairness constraints.
- **Post-processing**: After training, directly calibrates the output probabilities of a pre-trained model.

Both approaches are grounded in a theoretical characterization of the federated Bayes-optimal fair classifier.

### Key Designs

#### 1. Characterization of the Federated Bayes-Optimal Fair Classifier (Propositions 1 & 2)

- **Function**: Identifies the classifier that minimizes classification risk subject to global and local fairness constraints.
- **Core Structure** (Proposition 2):
$$h_k^*(x) = e_y, \quad y \in \arg\max_{j \in [m]} \left(\sum_{a \in \mathcal{A}} \mathbb{P}(A=a|x,k) [\mathbf{M}^{\lambda,\mu}(a,k)]^\top \eta(x,a,k)\right)_j$$
  where $\mathbf{M}^{\lambda,\mu}(a,k) = \mathbf{I} - \frac{1}{p_{a,k}}[\text{global dual term} - \text{local dual term}]$
- **Design Motivation**: Proposition 1 establishes that the optimal classifier decomposes into a linear combination of deterministic per-client classifiers, providing the theoretical basis for federated optimization.

#### 2. In-Processing: Personalized Cost-Sensitive Learning (Algorithm 1)

- **Function**: Reduces the fairness-constrained federated optimization to a cost-sensitive classification problem at each client.
- **Core Formula** (Proposition 3):
$$\ell_k(y, \mathbf{s}(x), a) = -\sum_{i=1}^{m} \overline{\mathbf{M}}^{\lambda,\mu}_{y,i}(a,k) \log \frac{\exp([\mathbf{s}(x)]_i)}{\sum_j \exp([\mathbf{s}(x)]_j)}$$
  where $\overline{\mathbf{M}} = \mathbf{M}^{\lambda,\mu} + \kappa \mathbf{1}_{m \times m}$ (to ensure positive definiteness).
- **Training Procedure**:
  1. Each round: clients integrate a shared global model $\theta$ and a personalized model $\phi_k$.
  2. Adaptive weight $w_k$ balances the contributions of both models.
  3. Local dual parameter $\mu_k$ is updated (controlling local fairness).
  4. The server aggregates $\theta$ and the global dual parameter $\lambda$ (controlling global fairness).
- **Design Motivation**: The cost matrix $\mathbf{M}$ encodes fairness constraints and is dynamically adjusted via dual variables, enabling controllable accuracy–fairness trade-offs.

#### 3. Post-Processing: Plug-In-Based Bi-Level Optimization (Algorithm 2)

- **Function**: Calibrates the classification probabilities of a pre-trained federated model to satisfy fairness constraints.
- **Core Formula** (Theorem 5):
$$\min_{\lambda \in \Lambda} \left\{\hat{F}(\lambda) = \sum_k \hat{p}_k \hat{F}_k(\lambda)\right\}, \quad \hat{F}_k(\lambda) := \min_{\mu_k \in \mathcal{M}_k} \hat{H}_k(\lambda, \mu_k)$$
- **Key Design**: Soft-max replaces hard-max to ensure smoothness; $\mu_k$ is updated locally (local fairness) and $\lambda$ is aggregated at the server (global fairness).
- **Design Motivation**: Post-processing leaves model representations unchanged, incurring low overhead and making it suitable for resource-constrained settings.

### Loss & Training

- **In-processing**: Personalized cost-sensitive cross-entropy loss with online learning updates for mixture weights.
- **Post-processing**: Gradient descent with projection in the bi-level optimization.
- **Fairness Constraints**: Supports DP (Demographic Parity), EOP (Equalized Opportunity), and EO (Equalized Odds).
- **Convergence Rate**: $\mathcal{O}(1/\sqrt{T})$ (Theorem 4).

## Key Experimental Results

### Main Results (4 datasets, $\gamma=0.5$ partition)

| Method | Compas Acc | Compas $\mathscr{D}^g$ | Compas $\mathscr{D}^l$ | Adult Acc | Adult $\mathscr{D}^g$ | Adult $\mathscr{D}^l$ |
|------|-----------|-----------|-----------|-----------|-----------|-----------|
| FedAvg | 69.73 | 0.2766 | 0.3590 | 84.52 | 0.1765 | 0.2310 |
| FairFed | 59.39 | 0.1008 | 0.1022 | 80.73 | 0.0983 | 0.1434 |
| FedFACT$^g$(Post) | **67.27** | **0.0128** | 0.0660 | **82.83** | 0.0173 | 0.0276 |
| FedFACT$^l$(Post) | 67.49 | 0.0315 | **0.0552** | 82.79 | 0.0154 | **0.0267** |
| FedFACT$^{g\&l}$(Post) | 67.33 | 0.0139 | 0.0641 | 82.74 | **0.0134** | 0.0274 |

| Method | CelebA Acc | CelebA $\mathscr{D}^g$ | CelebA $\mathscr{D}^l$ | ENEM Acc | ENEM $\mathscr{D}^g$ | ENEM $\mathscr{D}^l$ |
|------|-----------|-----------|-----------|-----------|-----------|-----------|
| FedAvg | 89.14 | 0.1435 | 0.1308 | 67.56 | 0.2620 | 0.2462 |
| FedFACT$^g$(Post) | 87.25 | **0.0089** | 0.0253 | 66.15 | 0.0175 | **0.0181** |
| FedFACT$^l$(Post) | **87.36** | 0.0127 | **0.0163** | **66.54** | 0.0197 | 0.0240 |
| FedFACT$^{g\&l}$(Post) | 87.06 | 0.0093 | 0.0172 | 66.52 | **0.0162** | 0.0184 |

### Ablation Study

| Variable | Finding |
|------|------|
| In-processing vs. Post-processing | Post-processing achieves higher accuracy but slightly weaker fairness control; in-processing provides finer-grained fairness regulation. |
| Global $\xi^g$ from 0.00 to 0.10 | Fairness disparity increases continuously from 0.01 to 0.10 with a corresponding accuracy gain—demonstrating controllable trade-off. |
| Local $\xi^k$ from 0.00 to 0.10 | Similar controllable behavior observed. |
| Heterogeneous (Hetero) partition | FedFACT consistently outperforms baselines under stronger data heterogeneity. |

### Key Findings

1. **Post-processing mode achieves the highest accuracy**: FedFACT(Post) maintains accuracy close to FedAvg on nearly all datasets while improving fairness by 10–20×.
2. **Joint global and local constraints are effective**: FedFACT$^{g\&l}$ reduces both global and local disparity simultaneously.
3. **Controllability**: Continuously tunable accuracy–fairness trade-offs are achieved by adjusting $\xi^g$ and $\xi^k$.
4. **Robustness under data heterogeneity**: FedFACT demonstrates greater improvements under the Hetero partition setting.

## Highlights & Insights

1. **First theoretical characterization of the multi-class federated Bayes-optimal fair classifier**: Algorithms are derived from optimality principles rather than heuristics.
2. **Reduction of fair FL to classical problems**: In-processing reduces to cost-sensitive learning; post-processing reduces to bi-level optimization—an elegant and practical insight.
3. **Unified perspective via the cost matrix $\mathbf{M}^{\lambda,\mu}$**: A single matrix encodes fairness constraints, sensitive attributes, and client-specific distributional information.
4. **Rigorous convergence and generalization guarantees**: The framework is theoretically guaranteed to approximate the optimum, not merely empirically effective.
5. **Complementary in-/post-processing modes**: The former modifies representations but incurs higher cost; the latter is lightweight but operates only on outputs.

## Limitations & Future Work

1. **Limited experimental scale**: All four datasets are tabular or simple image benchmarks; validation on large-scale vision or NLP tasks is absent.
2. **Communication overhead**: Transmission of dual parameters $\lambda$ and confusion matrices introduces additional cost.
3. **Assumes Bayes scores are known or estimable**: Post-processing relies on accurate estimation of $\eta(x,a,k)$.
4. **In-processing and post-processing are incompatible**: The two modes cannot be used in combination theoretically.
5. **Sensitive attributes must be known**: In practice, sensitive attributes may be missing or imprecisely recorded.

## Related Work & Insights

- **Relation to LogoFair**: LogoFair also derives Bayes-optimal classifiers but is restricted to binary classification and post-processing only; FedFACT extends to multi-class settings with both in-processing and post-processing modes.
- **Relation to FairFed/FedFB**: These methods rely on heuristic reweighting, whereas FedFACT is grounded in optimality theory.
- **Insight**: The application of cost-sensitive learning to fairness deserves deeper exploration—the classification cost should vary across samples according to their group membership.

## Rating

⭐⭐⭐⭐ (4/5)
- Solid theoretical contributions; the first work to address controllable multi-class global and local fairness jointly.
- Experimental scope is limited; applicability to complex real-world scenarios remains to be validated.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Copresheaf Topological Neural Networks: A Generalized Deep Learning Framework](copresheaf_topological_neural_networks_a_generalized_deep_learning_framework.md)
- [\[NeurIPS 2025\] Fair Representation Learning with Controllable High Confidence Guarantees via Adversarial Inference](fair_representation_learning_with_controllable_high_confidence_guarantees_via_ad.md)
- [\[ICCV 2025\] Faster and Better 3D Splatting via Group Training](../../ICCV2025/3d_vision/faster_and_better_3d_splatting_via_group_training.md)
- [\[ICCV 2025\] Boost 3D Reconstruction using Diffusion-based Monocular Camera Calibration](../../ICCV2025/3d_vision/boost_3d_reconstruction_using_diffusion-based_monocular_camera_calibration.md)
- [\[ICCV 2025\] LACONIC: A 3D Layout Adapter for Controllable Image Creation](../../ICCV2025/3d_vision/laconic_a_3d_layout_adapter_for_controllable_image_creation.md)

</div>

<!-- RELATED:END -->
