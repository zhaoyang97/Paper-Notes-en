---
title: >-
  [Paper Note] Dataless Weight Disentanglement in Task Arithmetic via Kronecker-Factored Approximate Curvature
description: >-
  [ICLR 2026][AI Safety] This work elegantly bridges classical curvature approximation theory (KFAC) with the practical demands of task arithmetic…
tags:
  - "ICLR 2026"
  - "AI Safety"
date: 2026-05-08
content_hash: 417049df088f5e68
---

# Dataless Weight Disentanglement in Task Arithmetic via Kronecker-Factored Approximate Curvature

**Conference**: ICLR 2026
**arXiv**: [2602.17385](https://arxiv.org/abs/2602.17385)  
**Code**: [https://github.com/aimagelab/mammoth](https://github.com/aimagelab/mammoth)  
**Area**: AI Safety / Model Editing

## TL;DR
This work elegantly bridges classical curvature approximation theory (KFAC) with the practical demands of task arithmetic, proposing a data-free weight disentanglement regularization method. The theoretical derivation is clear, with a coherent logical chain from representation drift regularization → Jacobian Gramian → GGN → KFAC. Experiments span multiple model scales across both vision and language domains, and the robustness analysis with respect to the $\alpha$ hyperparameter is practically valuable. Limitations include the $O(d^2)$ storage overhead of KFAC for large models and a remaining gap relative to data-dependent methods in the text domain.

## Rating

⭐⭐⭐⭐

This work elegantly bridges classical curvature approximation theory (KFAC) with the practical demands of task arithmetic, proposing a data-free weight disentanglement regularization method. The theoretical derivation is clear, with a coherent logical chain from representation drift regularization → Jacobian Gramian → GGN → KFAC. Experiments span multiple model scales across both vision and language domains, and the robustness analysis with respect to the $\alpha$ hyperparameter is practically valuable. Limitations include the $O(d^2)$ storage overhead of KFAC for large models and a remaining gap relative to data-dependent methods in the text domain.

---

## Background & Motivation

### State of the Field

Task Arithmetic produces task vectors $\boldsymbol{\tau}_t = \boldsymbol{\theta}_t^{\star} - \boldsymbol{\theta}_0$ by fine-tuning a base model, and merges multi-task capabilities via linear combination $\boldsymbol{\theta}_0 + \sum_t \alpha_t \boldsymbol{\tau}_t$. This paradigm requires no additional training and supports knowledge reuse across domains and even across backbone architectures, offering substantial flexibility and scalability.

### Limitations of Prior Work

Naïve linear combination leads to cross-task interference—adding a new task vector modifies shared representations, disrupting the representations of other tasks and degrading the performance of the merged model. Promoting weight disentanglement is therefore necessary, ensuring that different task vectors only influence the input-space regions corresponding to their respective tasks.

### Root Cause

Existing representation drift regularization methods (e.g., $\tau$Jp) can effectively promote weight disentanglement, but require access to training data from other tasks. This is infeasible under practical constraints such as privacy requirements, decentralized training, or data non-shareability, and contradicts the modular spirit of task arithmetic.

### Paper Goals

This paper proposes TAK (Task Arithmetic with KFAC regularization): under a linearized fine-tuning framework, the representation drift regularization is reformulated as a quadratic form of the Jacobian Gramian, which is a special instance of the Generalized Gauss-Newton (GGN) matrix. By approximating the GGN with KFAC, pre-computed Kronecker factors can serve as a regularizer without requiring any data. A cumulative regularization strategy is further proposed to merge multi-task KFAC factors into a single surrogate, achieving $O(1)$ complexity with respect to the number of tasks.

---

## Method

### Overall Architecture

TAK follows a two-stage training pipeline:
1. **Pre-computation stage**: For each task $t$, KFAC factors $\{(\boldsymbol{B}_t^l, \boldsymbol{A}_t^l)\}_l$ are computed from the task's training data and merged into a single surrogate.
2. **Fine-tuning stage**: The KFAC regularization term is incorporated into the linearized fine-tuning objective:

$$\min_{\boldsymbol{\tau}_{t'}} \mathcal{L}_{\mathcal{D}_{t'}}(\boldsymbol{\tau}_{t'}) + \beta \sum_{t \neq t'} \lambda_t \boldsymbol{\tau}_{t'}^\top \boldsymbol{G}_t(\boldsymbol{\theta}_0) \boldsymbol{\tau}_{t'}$$

### Key Design 1: From Representation Drift to KFAC

Under the linearized model $f_\text{lin}(\boldsymbol{x}, \boldsymbol{\theta}) = f(\boldsymbol{x}, \boldsymbol{\theta}_0) + \mathrm{J}_{\boldsymbol{\theta}} f(\boldsymbol{x}, \boldsymbol{\theta}_0)(\boldsymbol{\theta} - \boldsymbol{\theta}_0)$, representation drift simplifies to:

$$\Delta_{t \to t,t'}(\boldsymbol{x}) = \alpha_{t'}^2 \| \mathrm{J}_{\boldsymbol{\theta}} f(\boldsymbol{x}, \boldsymbol{\theta}_0) \boldsymbol{\tau}_{t'} \|_2^2$$

The regularization term becomes $\boldsymbol{\tau}_{t'}^\top \boldsymbol{G}_t \boldsymbol{\tau}_{t'}$, where the Jacobian Gramian $\boldsymbol{G}_t$ is a special instance of the GGN matrix (corresponding to squared loss where $\nabla^2 c = \boldsymbol{I}$). KFAC approximates the GGN as block-diagonal with each block being a Kronecker product:

$$\boldsymbol{G}(\boldsymbol{\theta}^l) \approx \boldsymbol{B}^l \otimes \boldsymbol{A}^l$$

where $\boldsymbol{A}^l$ is the input covariance and $\boldsymbol{B}^l$ is the output gradient covariance.

### Key Design 2: Cumulative Factor Merging

The naïve approach requires storing KFAC factors for each task at $O(T)$ complexity. A heuristic merging strategy is proposed:

$$\boldsymbol{G}_{-t'} \approx \left(\sum_{t \neq t'} \boldsymbol{B}_t^l\right) \otimes \left(\frac{1}{T-1} \sum_{t \neq t'} \boldsymbol{A}_t^l\right)$$

Theoretical analysis shows the approximation error is bounded by $\|E\|_F \leq T \sigma_A \sigma_B$, and approximation accuracy is high when KFAC factors vary little across tasks (as expected with a shared pre-trained backbone).

### Key Design 3: Task Localization and OOD Detection

KFAC regularization naturally induces task localization: $\| \mathrm{J}_{\boldsymbol{\theta}} f(\boldsymbol{x}, \boldsymbol{\theta}_0) \boldsymbol{\tau}_t \|_2^2$ serves as a "normality score" for task $t$. After regularization, the scores of out-of-distribution samples are pushed toward zero, achieving localized influence of task vectors in input space.

---

## Key Experimental Results

### Main Results: 8-Task Vision Addition

| Method | Data-free | $\alpha$ | ViT-B/32 (Abs.) | ViT-B/16 (Abs.) | ViT-L/14 (Abs.) |
|--------|-----------|---------|-----------------|-----------------|-----------------|
| Pre-trained | - | - | 48.4% | 55.4% | 65.0% |
| Linear FT | - | 1.0 | 76.7% | 80.2% | 88.0% |
| $\tau$Jp | ✗ | 1.0 | 85.0% | 88.2% | 90.9% |
| Diag. GGN | ✓ | 1.0 | 80.1% | 82.9% | 87.9% |
| **TAK (Ours)** | **✓** | **1.0** | **85.8%** | **88.3%** | **91.6%** |
| $\tau$Jp | ✗ | Best | 85.6% | **88.6%** | 91.1% |
| **TAK (Ours)** | **✓** | **Best** | **86.0%** | 88.3% | **91.6%** |

TAK matches or surpasses the data-dependent $\tau$Jp method without requiring external data, and achieves near-optimal performance at $\alpha=1.0$.

### Ablation Study & Analysis

| Analysis Dimension | Key Results |
|-------------------|-------------|
| Task unlearning | TAK reduces target task accuracy to **3.4** (ViT-B/32) while maintaining task retention at **62.4%** |
| Cumulative vs. naïve | Gap < 0.3 on ViT-B/16, validating the effectiveness of the merging strategy |
| KFAC data quantity | Performance saturates with 128–256 samples |
| Monte Carlo sampling | 1–2 samples/data point suffice; more samples lead to degraded performance |
| KFAC compression | Block-8 strategy achieves 87% memory reduction with only ~1 point accuracy loss |
| Training overhead | With MC=1, pre-computing all factors requires only 3.9 minutes |
| Language tasks (T5-base) | TAK: 78.7 Abs. / 98.9 Norm.; $\tau$Jp: **81.3%** / **100** |

---

## Limitations & Future Work

**Strengths**:
- Rigorous theoretical derivation elegantly connecting representation drift regularization with GGN/KFAC
- Data-free, satisfying privacy and modularity constraints
- Highly robust to $\alpha$, eliminating the need for hyperparameter search
- Comprehensive experiments covering vision and language domains with thorough ablation analysis
- Cumulative merging strategy scales to an arbitrary number of tasks at $O(1)$ complexity

**Weaknesses**:
- KFAC factor storage grows quadratically with layer width, potentially becoming a bottleneck for very large models
- A gap remains relative to the data-dependent $\tau$Jp in the text domain (T5-base)
- Theoretical analysis relies on linearization assumptions; while effective in nonlinear experiments, formal guarantees are absent
- Applicability to parameter-efficient fine-tuning scenarios (e.g., LoRA) remains unexplored

## Highlights & Insights
- The method design is concise and effective, with a clear core mechanism
- Experimental validation is comprehensive with thorough ablation analysis
- The work offers a novel solution to a key problem in the field

## Limitations & Future Work
- The method may have limitations under certain conditions; generalizability warrants further verification
- Computational efficiency and scalability can be further optimized
- Integration with additional related methods deserves further exploration

## Related Work & Insights
- **vs. representative methods in the field**: This work makes a distinctive contribution in method design and is complementary to existing approaches
- **vs. traditional methods**: The proposed method achieves significant improvements on key metrics compared to conventional solutions
- **Insights**: The technical approach of this work provides important reference value for subsequent related research

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Toward Enhancing Representation Learning in Federated Multi-Task Settings](toward_enhancing_representation_learning_in_federated_multi-task_settings.md)
- [\[ICML 2026\] Limits of Convergence-Rate Control for Open-Weight Safety](../../ICML2026/ai_safety/limits_of_convergence-rate_control_for_open-weight_safety.md)
- [\[NeurIPS 2025\] Reconstruction and Secrecy under Approximate Distance Queries](../../NeurIPS2025/ai_safety/reconstruction_and_secrecy_under_approximate_distance_queries.md)
- [\[NeurIPS 2025\] Preserving Task-Relevant Information Under Linear Concept Removal](../../NeurIPS2025/ai_safety/preserving_task-relevant_information_under_linear_concept_removal.md)
- [\[AAAI 2026\] Privacy Auditing of Multi-Domain Graph Pre-Trained Model under Membership Inference Attack](../../AAAI2026/ai_safety/privacy_auditing_of_multi-domain_graph_pre-trained_model_under_membership_infere.md)

</div>

<!-- RELATED:END -->
