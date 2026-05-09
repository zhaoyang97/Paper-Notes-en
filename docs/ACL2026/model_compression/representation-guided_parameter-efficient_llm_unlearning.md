---
title: >-
  [Paper Note] Representation-Guided Parameter-Efficient LLM Unlearning
description: >-
  [ACL 2026][Model Compression][To be supplemented] To be supplemented after thorough reading.
tags:
  - ACL 2026
  - Model Compression
  - To be supplemented
date: 2026-05-08
content_hash: a7c4815165a2204a
---

# Representation-Guided Parameter-Efficient LLM Unlearning

**Conference**: ACL 2026
**arXiv**: [2604.17396](https://arxiv.org/abs/2604.17396)
**Code**: [https://github.com/sustech-nlp/ReGLU](https://github.com/sustech-nlp/ReGLU)
**Area**: Model Compression
**Keywords**: LLM unlearning, representation space geometry, LoRA initialization, orthogonal regularization, parameter efficiency

## TL;DR

This paper proposes ReGLU, a framework that shifts LLM unlearning from a "parameter importance" paradigm to a "representation space geometry" paradigm. It introduces Representation-guided Initialization for LoRA Adaptation (RILA), which aligns unlearning updates to the most discriminative subspace between the forget and retain sets, and a Representation Orthogonality Loss (ROL) that constrains updates from interfering with retain-set knowledge.

## Background & Motivation

**Background**: LoRA-based LLM unlearning methods have demonstrated performance comparable to or better than full fine-tuning, yet the forget–retain trade-off remains challenging — reducing performance on the forget set often comes at the cost of degraded performance on the retain set.

**Limitations of Prior Work**: Methods such as FILA and VILA rely on parameter importance metrics (e.g., Fisher information) to identify parameters exclusively associated with the forget set. However, due to the superposition phenomenon, LLM parameters are polysemantic — individual parameters participate in representing multiple concepts simultaneously. Consequently, parameter-importance-based methods cannot reliably disentangle forget-relevant and retain-relevant parameters.

**Key Challenge**: Parameter-level importance metrics are unreliable due to polysemanticity, yet forget and retain knowledge are indeed encoded differently within the model. A more reliable signal is needed to guide selective unlearning.

**Goal**: To leverage the geometric properties of representation subspaces — rather than parameter importance — to achieve precise forget–retain separation.

**Key Insight**: Although superposition-induced polysemanticity exists at the parameter level, representation subspaces can be decoupled more effectively. By constraining unlearning updates to operate within a subspace that is aligned with forget-set representations and orthogonal to retain-set representations, forget knowledge can be more precisely isolated.

**Core Idea**: (1) RILA — constructs a balanced covariance matrix $\text{Cov}_\Delta = (1-\beta)\text{Cov}_F - \beta\text{Cov}_R$, whose top-$r$ eigenvectors initialize the LoRA adapter, ensuring that initial updates maximize forget-set variance while minimizing retain-set variance; (2) ROL — constrains the LoRA up-projection matrix $B$ to remain orthogonal to the principal subspace of retain-set representations.

## Method

### Overall Architecture

ReGLU consists of two complementary components: RILA determines the initialization direction of the LoRA adapter (i.e., toward which subspace unlearning is directed), while ROL continuously constrains updates during training to avoid encroaching on the retain-set subspace. The total loss is $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{forget}} + \gamma \mathcal{L}_{\text{retain}} + \lambda \mathcal{L}_{\text{ROL}}$.

### Key Designs

1. **Representation-guided Initialization for LoRA Adaptation (RILA)**:

    - **Function**: Directs the initial LoRA update toward the subspace that maximizes forget–retain discriminability.
    - **Mechanism**: For each linear layer, the output representations of forget-set and retain-set samples passing through that layer are collected, and their respective covariance matrices $\text{Cov}_F$ and $\text{Cov}_R$ are computed. A balanced covariance matrix $\text{Cov}_\Delta = (1-\beta)\text{Cov}_F - \beta\text{Cov}_R$ is constructed, and its top-$r$ eigenvectors form $Q_r$. The adapter is initialized as $B_{\text{init}} = Q_r$ and $A_{\text{init}} = Q_r^\top W_0$. Theoretical analysis shows this maximizes the objective function at initialization.
    - **Design Motivation**: Methods such as FILA initialize LoRA using parameter-level Fisher information, but parameter polysemanticity renders these importance estimates unreliable. Covariance in representation space more directly captures the directions along which forget/retain knowledge is encoded.

2. **Representation Orthogonality Loss (ROL)**:

    - **Function**: Continuously prevents LoRA updates from interfering with retain-set knowledge during training.
    - **Mechanism**: The top-$k$ eigenvectors of the retain-set representation covariance matrix form a basis $P_B \in \mathbb{R}^{d_{\text{out}} \times k}$ capturing the principal directions of the retain set. A regularization term $\mathcal{L}_{\text{ROL}} = \|B^\top P_B\|_F^2$ is then added to the loss, enforcing that the column vectors of the up-projection matrix $B$ remain orthogonal to the principal retain-set directions, so that $\Delta h = B(Ax)$ lies in the orthogonal complement of the retain-set representation subspace.
    - **Design Motivation**: Even when initialization is properly directed, gradient updates during training may drift away from the ideal subspace. ROL provides a persistent geometric constraint that confines updates to the region that does not interfere with the retain set.

3. **Compatibility with Existing Unlearning Losses**:

    - **Function**: ReGLU can be combined with any unlearning objective.
    - **Mechanism**: $\mathcal{L}_{\text{forget}}$ can be any unlearning loss, including gradient ascent (GA), NPO, SimNPO, IHL, etc. ReGLU contributes only an initialization strategy and a regularizer, which are orthogonal to and compatible with any unlearning objective.
    - **Design Motivation**: Different unlearning losses have distinct strengths and weaknesses; by not binding to a specific objective, ReGLU serves as a general-purpose framework.

### Loss & Training

$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{forget}} + \gamma \mathcal{L}_{\text{retain}} + \lambda \mathcal{L}_{\text{ROL}}$. Experiments are conducted on the TOFU and WMDP benchmarks using Llama-2-7B, Phi-1.5B, and Zephyr-7B-beta.

## Key Experimental Results

### Main Results

| Model / Method | TOFU Forget 1% | Forget 5% | Forget 10% | Average |
|---|---|---|---|---|
| Phi-1.5B IHL | -1.3 | -11.5 | -12.4 | -8.4 |
| Phi-1.5B IHL+FILA | -2.5 | -9.3 | -10.3 | -7.4 |
| Phi-1.5B IHL+ReGLU | **-0.1** | **-5.4** | **-7.7** | **-4.4** |

### Ablation Study

| Configuration | Effect | Notes |
|---|---|---|
| RILA only (no ROL) | Improvement but insufficient | Correct initialization but drift during training |
| ROL only (random init) | Improvement but limited | Constraint effective but poor starting point |
| RILA + ROL | Optimal | Synergy of initialization and persistent constraint |

### Key Findings

- ReGLU consistently outperforms FILA and VILA across all unlearning loss functions.
- IHL + ReGLU improves the average metric on Phi-1.5B from -7.4 (FILA) to -4.4.
- Geometric diagnostics confirm that ReGLU successfully decouples forget and retain representations.
- Consistent advantages on the WMDP benchmark demonstrate cross-task generalizability.

## Highlights & Insights

- **The paradigm shift from "parameter importance" to "representation geometry" is the central contribution**: The superposition phenomenon renders parameter-level signals unreliable, whereas the geometric structure of representation subspaces provides a more stable separation signal. This insight may catalyze a methodological shift across the LLM unlearning field.
- **The construction of the balanced covariance matrix is elegant**: The eigenvectors of $\text{Cov}_\Delta = (1-\beta)\text{Cov}_F - \beta\text{Cov}_R$ naturally correspond to directions with high forget-set variance and low retain-set variance — conceptually intuitive and theoretically grounded.
- **The complementary design of RILA and ROL**: RILA governs where the optimization starts; ROL governs where it must not drift.

## Limitations & Future Work

- Computing covariance matrices requires collecting representations from both the forget and retain sets, incurring non-trivial preprocessing costs.
- The hyperparameters $\beta$ (balance coefficient) and $k$ (ROL basis dimensionality) require tuning.
- Evaluation is conducted only on relatively small-scale models (1.5B–7B parameters).
- The quality of covariance estimation depends on sample size; very small forget sets (1%) may introduce noise.

## Related Work & Insights

- **vs. FILA/VILA (parameter importance methods)**: Fisher-information-based parameter selection is limited by the superposition phenomenon; ReGLU circumvents this by operating in representation space.
- **vs. ETW (token-level methods)**: ETW focuses on which tokens to penalize, whereas ReGLU focuses on which subspace to update within — the two approaches are orthogonal and potentially composable.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ The paradigm shift from parameter importance to representation geometry constitutes a substantive innovation with rigorous theoretical support.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Two benchmarks, three models, and multiple unlearning objectives provide reasonably comprehensive evaluation.
- **Writing Quality**: ⭐⭐⭐⭐ Motivation is clearly articulated; theoretical derivations are rigorous.

**Code**: To be confirmed
**Area**: model_compression
**Keywords**: To be supplemented

## TL;DR
To be supplemented after thorough reading.

## Background & Motivation
To be supplemented after thorough reading.

## Method
To be supplemented after thorough reading.

## Key Experimental Results
To be supplemented after thorough reading.

## Highlights & Insights
To be supplemented after thorough reading.

## Limitations & Future Work
To be supplemented after thorough reading.

## Related Work & Insights
To be supplemented after thorough reading.

## Rating
- Novelty: Pending
- Experimental Thoroughness: Pending
- Writing Quality: Pending
- Value: Pending

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] DASH-KV: Accelerating Long-Context LLM Inference via Asymmetric KV Cache Hashing](dash-kv_accelerating_long-context_llm_inference_via_asymmetric_kv_cache_hashing.md)
- [\[ACL 2026\] Latent-Condensed Transformer for Efficient Long Context Modeling](latent-condensed_transformer_for_efficient_long_context_modeling.md)
- [\[ICLR 2026\] Reference-Guided Machine Unlearning](../../ICLR2026/model_compression/reference-guided_machine_unlearning.md)
- [\[ACL 2026\] Calibrated Speculative Decoding: Frequency-Guided Candidate Selection for Efficient Inference](calibrated_speculative_decoding_frequency-guided_candidate_selection_for_efficie.md)
- [\[ACL 2026\] LLM Prompt Duel Optimizer: Efficient Label-Free Prompt Optimization](llm_prompt_duel_optimizer_efficient_label-free_prompt_optimization.md)

<!-- RELATED:END -->
