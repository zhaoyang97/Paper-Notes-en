---
title: >-
  [Paper Note] Representation-Guided Parameter-Efficient LLM Unlearning
description: >-
  [ACL 2026][LLM Safety][LLM unlearning] Ours proposes the ReGLU framework, shifting LLM unlearning from the "parameter importance" paradigm to the "representation space geometry" paradigm. By using Representation-guided L…
tags:
  - "ACL 2026"
  - "LLM Safety"
  - "LLM unlearning"
  - "representation space geometry"
  - "LoRA initialization"
  - "orthogonal regularization"
  - "parameter-efficient"
date: 2026-05-08
content_hash: cf22f9070d6bae13
---

# Representation-Guided Parameter-Efficient LLM Unlearning

**Conference**: ACL 2026 Findings  
**arXiv**: [2604.17396](https://arxiv.org/abs/2604.17396)  
**Code**: [https://github.com/sustech-nlp/ReGLU](https://github.com/sustech-nlp/ReGLU)  
**Area**: Model Compression  
**Keywords**: LLM unlearning, representation space geometry, LoRA initialization, orthogonal regularization, parameter-efficient

## TL;DR

Ours proposes the ReGLU framework, shifting LLM unlearning from the "parameter importance" paradigm to the "representation space geometry" paradigm. By using Representation-guided LoRA Initialization (RILA), unlearning updates are aligned with the most discriminative subspace of the forget/retain sets, combined with Representation Orthogonal Loss (ROL) to ensure updates do not interfere with retain set knowledge.

## Background & Motivation

**Background**: LoRA-based LLM unlearning methods have demonstrated performance comparable to or even better than full fine-tuning, but still face the difficulty of the "unlearning-retention tradeoff"—reducing forget set performance often comes at the cost of decreasing retain set performance.

**Limitations of Prior Work**: Methods such as FILA and VILA rely on parameter importance metrics like Fisher Information to identify parameters "related only to the forget set." However, due to the superposition phenomenon, LLM parameters exhibit polysemanticity—a single parameter participates in the representation of multiple concepts simultaneously. Consequently, parameter-importance-based methods cannot reliably separate parameters associated with unlearning from those associated with retention.

**Key Challenge**: Parameter-level importance measures are unreliable due to polysemanticity, yet knowledge for unlearning and retention does indeed have distinct representations within the model. A more reliable signal is needed to guide selective unlearning.

**Goal**: Leverage the geometric properties of the representation subspace (rather than parameter importance) to achieve precise forget-retain separation.

**Key Insight**: While polysemanticity exists at the parameter level due to superposition, representation subspaces can be more effectively decoupled. By constraining unlearning updates to a subspace "aligned with forget set representations and orthogonal to retain set representations," unlearning knowledge can be isolated more precisely.

**Core Idea**: (1) RILA—Construct a balanced covariance matrix $\text{Cov}_\Delta = (1-\beta)\text{Cov}_F - \beta\text{Cov}_R$ and use its top-r eigenvectors to initialize LoRA, ensuring initial updates maximize forget set variance while minimizing retain set variance; (2) ROL—Constrain the LoRA up-projection matrix B to be orthogonal to the principal subspace of the retain set representations.

## Method

### Overall Architecture

ReGLU consists of two complementary components: RILA determines the initialization direction for LoRA (pointing to the subspace to be forgotten), and ROL continuously constrains updates during training to avoid the retain set subspace. The total loss is defined as $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{forget}} + \gamma \mathcal{L}_{\text{retain}} + \lambda \mathcal{L}_{\text{ROL}}$.

### Key Designs

1.  **Representation-Guided LoRA Initialization (RILA)**:
    - **Function**: Aligns the initial update direction of LoRA to maximize the discriminability between the forget and retain sets.
    - **Mechanism**: For each linear layer, output representations of the forget and retain sets are collected after passing through the layer to compute their respective covariance matrices $\text{Cov}_F$ and $\text{Cov}_R$. A balanced covariance $\text{Cov}_\Delta = (1-\beta)\text{Cov}_F - \beta\text{Cov}_R$ is constructed, and its top-r eigenvectors form $Q_r$. Initialization is set as $B_{\text{init}} = Q_r$ and $A_{\text{init}} = Q_r^\top W_0$. It is theoretically proven that this maximizes the objective function at initialization.
    - **Design Motivation**: Methods like FILA use parameter-level Fisher Information to initialize LoRA, but parameter polysemanticity makes these importance measures unreliable. Covariance in the representation space more directly reflects "which directions carry forget/retention knowledge."

2.  **Representation Orthogonal Loss (ROL)**:
    - **Function**: Continuously prevents LoRA updates from interfering with retain set knowledge during the training process.
    - **Mechanism**: A basis $P_B \in \mathbb{R}^{d_{\text{out}} \times k}$ is formed using the top-k eigenvectors of the retain set representation covariance matrix (capturing the primary directions of the retain set). A regularization term $\mathcal{L}_{\text{ROL}} = \|B^\top P_B\|_F^2$ is added to the loss. This forces the column vectors of the LoRA up-projection matrix B to be orthogonal to the principal directions of the retain set, ensuring $\Delta h = B(Ax)$ resides in the orthogonal complement of the retain set representation subspace.
    - **Design Motivation**: Even with correct initialization, gradient updates during training may drift from the ideal subspace. ROL provides continuous geometric constraints, "trapping" updates in a space that does not interfere with the retain set.

3.  **Compatibility with existing unlearning losses**:
    - **Function**: ReGLU can be combined with any unlearning objective function.
    - **Mechanism**: $\mathcal{L}_{\text{forget}}$ can be any unlearning loss such as Gradient Ascent (GA), NPO, SimNPO, or IHL. ReGLU provides the initialization strategy and regularization, complementing the unlearning targets orthogonally.
    - **Design Motivation**: Different unlearning losses have various strengths and weaknesses; as a general framework, ReGLU is not tied to a specific unlearning objective.

### Loss & Training

$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{forget}} + \gamma \mathcal{L}_{\text{retain}} + \lambda \mathcal{L}_{\text{ROL}}$. Evaluation is performed on TOFU and WMDP benchmarks using models including Llama-2-7B, Phi-1.5B, and Zephyr-7B-beta.

## Key Experimental Results

### Main Results

| Model/Method | TOFU Forget 1% | Forget 5% | Forget 10% | Average |
| :--- | :--- | :--- | :--- | :--- |
| Phi-1.5B IHL | -1.3 | -11.5 | -12.4 | -8.4 |
| Phi-1.5B IHL+FILA | -2.5 | -9.3 | -10.3 | -7.4 |
| Phi-1.5B IHL+ReGLU | **-0.1** | **-5.4** | **-7.7** | **-4.4** |

### Ablation Study

| Configuration | Effect | Description |
| :--- | :--- | :--- |
| RILA only (No ROL) | Improvement but insufficient | Correct initialization but drifts during training |
| ROL only (Random Init) | Limited improvement | Effective constraint but poor starting point |
| RILA + ROL | Optimal | Synergy between initialization and continuous constraint |

### Key Findings

- ReGLU consistently outperforms FILA and VILA across all unlearning loss functions.
- IHL + ReGLU improves the average metric on Phi-1.5B from -7.4 (FILA) to -4.4.
- Geometric diagnostics confirm that ReGLU successfully decouples representations of unlearning and retention.
- Consistent advantages are demonstrated on the WMDP benchmark, proving cross-task generalization.

## Highlights & Insights

- **Paradigm shift from "parameter importance" to "representation geometry" is the core contribution**: Polysemanticity makes parameter-level signals unreliable, whereas the geometric structure of representation subspaces provides more stable separation signals. This insight may drive a methodological shift in the field of LLM unlearning.
- **Elegant construction of the balanced covariance matrix**: The eigenvectors of $\text{Cov}_\Delta = (1-\beta)\text{Cov}_F - \beta\text{Cov}_R$ naturally correspond to directions where "forget set variance is high but retain set variance is low," a concept that is intuitive and theoretically grounded.
- **Complementary design of RILA and ROL**: One manages "where to start," while the other ensures the update "does not drift into prohibited areas."

## Limitations & Future Work

- Requires collecting representations of both forget and retain sets to calculate covariance, involving upfront computational costs.
- Hyperparameters $\beta$ (balancing coefficient) and $k$ (ROL basis dimension) require tuning.
- Validated only on relatively small-scale models (1.5B-7B).
- The quality of covariance estimation depends on the number of samples; extremely small forget sets (1%) may introduce noise.

## Related Work & Insights

- **vs FILA/VILA (Parameter Importance Methods)**: Parameter selection based on Fisher Information is limited by superposition; ReGLU bypasses this issue by utilizing representation geometry.
- **vs ETW (Token-level Methods)**: ETW focuses on "which tokens to penalize," while ReGLU focuses on "in which subspace to update." The two are orthogonal and can be combined.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The paradigm shift from parameter importance to representation geometry is a substantial innovation with solid theoretical support.
- Experimental Thoroughness: ⭐⭐⭐⭐ Sufficiently validated across two benchmarks, three models, and various unlearning targets.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation and rigorous theoretical derivation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] FairLLaVA: Fairness-Aware Parameter-Efficient Fine-Tuning for Large Vision-Language Models](../../CVPR2026/llm_safety/fairllava_fairness-aware_parameter-efficient_fine-tuning_for_large_vision-langua.md)
- [\[ACL 2026\] Modeling LLM Unlearning as an Asymmetric Two-Task Learning Problem](modeling_llm_unlearning_as_an_asymmetric_two-task_learning_problem.md)
- [\[ICLR 2026\] LLM Unlearning with LLM Beliefs](../../ICLR2026/llm_safety/llm_unlearning_with_llm_beliefs.md)
- [\[ACL 2026\] Maximizing Local Entropy Where It Matters: Prefix-Aware Localized LLM Unlearning](maximizing_local_entropy_where_it_matters_prefix-aware_localized_llm_unlearning.md)
- [\[ACL 2026\] SWAN: Semantic Watermarking with Abstract Meaning Representation](swan_semantic_watermarking_with_abstract_meaning_representation.md)

</div>

<!-- RELATED:END -->
