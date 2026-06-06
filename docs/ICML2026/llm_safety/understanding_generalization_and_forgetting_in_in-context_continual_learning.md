---
title: >-
  [Paper Note] Understanding Generalization and Forgetting in In-Context Continual Learning
description: >-
  [ICML 2026][LLM Safety][In-Context Learning] This paper establishes the first theoretical framework for In-Context Continual Learning (ICCL)…
tags:
  - "ICML 2026"
  - "LLM Safety"
  - "In-Context Learning"
  - "Continual Learning"
  - "Forgetting"
  - "Task Interference"
  - "Transformer Theory"
date: 2026-05-08
content_hash: 04107b0710da3314
---

# Understanding Generalization and Forgetting in In-Context Continual Learning

**Conference**: ICML 2026  
**arXiv**: [2605.28705](https://arxiv.org/abs/2605.28705)  
**Code**: TBD  
**Area**: LLM Reasoning / Continual Learning / Attention Mechanism  
**Keywords**: In-Context Learning, Continual Learning, Forgetting, Task Interference, Transformer Theory

## TL;DR
This paper establishes the first theoretical framework for In-Context Continual Learning (ICCL), revealing that attention mechanisms inherently produce systematic bias and task interference when processing multi-task sequences, resulting in task-order-dependent decay of generalization performance and task memory.

## Background & Motivation

**Background**: LLMs adapt to new tasks during inference via In-Context Learning (ICL) without parameter updates. However, existing theories primarily focus on single-task ICL, whereas real-world prompts often contain sequences of multiple heterogeneous tasks.

**Limitations of Prior Work**: Classical continual learning investigates forgetting caused by parameter updates. In contrast, ICL is a parameter-frozen, pure attention-based adaptation. A theoretical gap exists between these fields—it remains unknown whether LLMs implicitly perform continual learning when processing long-context multi-task sequences and when forgetting occurs.

**Key Challenge**: Single-task ICL theory (assuming all examples come from the same function) cannot capture inter-task dependencies. Classical continual learning theory is based on training-time parameter updates, which fundamentally differs from inference-time attention adaptation.

**Goal**: (1) Build the first theoretical framework for In-Context Continual Learning (ICCL); (2) Quantify generalization and forgetting in multi-task sequences; (3) Explain experimental phenomena such as task-order sensitivity and performance saturation in long prompts.

**Key Insight**: The fundamental mechanism lies in how the attention mechanism aggregates historical context. Standard linear attention aggregates uniformly (leading to look-ahead leakage), while masked linear attention aggregates causally (respecting task order but still generating systematic interference).

**Core Idea**: Through an explicit **bias-variance-interference decomposition**, the authors characterize how task similarity, context length, and task order jointly determine whether historical information leads to positive or negative transfer.

## Method

### Overall Architecture
The model concatenates examples and queries from $T$ regression tasks into a single prompt sequence, processed by a shared attention layer. For each task $t$, the model infers task parameters $w_t$ from $M$ in-context examples $\{(x_{t,i}, y_{t,i})\}_{i=1}^{M}$ and predicts the query $y_{t,q} = \langle w_t, x_{t,q} \rangle$. The core analysis focuses on generalization error $G_t$ and forgetting error $F_t$.

### Key Designs

1.  **Masked Linear Attention Model**:
    *   **Function**: Encodes task sequence structure and prevents future information leakage.
    *   **Mechanism**: $f_{\text{MSA}}(P;\theta) = P + WP \cdot \mathrm{mask}(P^\top VP)$, where $\mathrm{mask}(A)_{i,j} = \begin{cases} \frac{1}{j}A_{i,j}, & i \le j \\ 0, & i > j \end{cases}$, ensuring information flows only forward.
    *   **Design Motivation**: To address information leakage in standard linear attention while maintaining analytical tractability.

2.  **Bias-Variance-Interference Decomposition**:
    *   **Function**: Explicitly quantifies three sources of error.
    *   **Mechanism**: The prediction error $\mathbb{E}[(\hat{y}_{t,q} - y_{t,q})^2]$ for task $t$ is decomposed into: (i) irreducible error; (ii) finite-sample variance $\sim O(1/M)$; (iii) task mean bias $\|\frac{1}{t}\sum_{s=1}^t w_s - w_t\|_2^2$.
    *   **Design Motivation**: To reveal when increasing context is beneficial (variance-dominated) versus harmful (bias-dominated).

3.  **Task Similarity and Order Dependence Coefficients**:
    *   **Function**: Characterizes interference intensity between tasks during forgetting.
    *   **Mechanism**: Forgetting stems from attention weight readjustment, with coefficients $\alpha_i(t) = \begin{cases} c_t < 0, & i \le t \\ d > 0, & i > t \end{cases}$; past task weights are negative (offset), while future task weights are positive (creating interference).
    *   **Design Motivation**: Explains why long contexts cannot eliminate forgetting—the mean term interference is independent of $M$ and depends only on task alignment.

## Key Experimental Results

### Main Results

| Factor | Impact on Generalization | Impact on Forgetting |
| :--- | :--- | :--- |
| Context Length $M$ | Reduces variance; however, bias dominates after crossing an alignment threshold, leading to saturation or degradation. | Variance term decays at $O(1/M)$; mean interference term remains constant (forgetting persists even as $M \to \infty$). |
| Task Similarity | Historical information reduces bias when tasks are aligned; bias grows rapidly when tasks are heterogeneous. | Interference term $\mathrm{tr}(\mu_i\mu_j^\top\Gamma^{-2}\Lambda)$ increases significantly when future tasks align with the current task. |
| Task Order | Irrelevant for single tasks; influences historical aggregation patterns in multi-task settings. | Order-sensitive: The degree of alignment between subsequent and preceding tasks determines interference intensity. |

### Key Findings

| Experiment | Phenomenon | Verification |
| :--- | :--- | :--- |
| Non-monotonicity | When processing Task 2, increasing $M$ from 3 to 19 caused the error to jump from a minimum to 0.99. | Aligns perfectly with theoretical predictions. |
| Task Clustering Effect | Performance is superior within similar clusters {Task 1, 3, 5}, while cross-cluster tasks cause strong interference. | Consistent with theory. |
| Real LLMs | Qwen2.5 on SST-2 → AG News sequence showed catastrophic forgetting as Task A accuracy dropped from 0.934 to 0.472 at $M=1$. | Theoretical predictions match empirical observations. |

## Highlights & Insights
*   **First Theoretical Bridge**: Unifies two independent fields, ICL and Continual Learning, explaining the mechanism of forgetting under frozen parameters via attention weight readjustment.
*   **Analyzable Simplified Model with Transferability**: Assumptions from linear regression remain valid even on non-linear two-layer ReLU networks.
*   **Practical Implications**: Task sequences should be intentionally designed for multi-task prompting, or smaller $M$ should be used to balance negative transfer.

## Limitations & Future Work
*   The assumption of linear task distributions may not fully capture the complexity of real-world NLP tasks.
*   The analysis is based on masked linear attention; differences in the multi-head aggregation mechanism of softmax attention were not explored.
*   The awareness of task boundaries via positional encodings was not considered.
*   Future Directions: Designing "task-boundary-aware" attention mechanisms; investigating prompt reordering to mitigate forgetting; extending to non-linear self-attention and multi-head mechanisms.

## Related Work & Insights
*   **Vs. Single-task ICL Theory (Bai et al. 2023, Von Oswald et al. 2023)**: These works assume all examples are homogeneous; this paper introduces task sequence heterogeneity.
*   **Vs. Classic Continual Learning (Lin et al. 2023)**: They focus on forgetting due to parameter updates; this paper provides the first proof that attention aggregation inherently leads to forgetting even with frozen parameters.
*   **Vs. Empirical ICCL Research (Kang et al. 2025)**: They demonstrated that ICL can perform continual learning; this paper answers the fundamental theoretical question of **when it succeeds and why it fails**.

## Rating
*   Novelty: ⭐⭐⭐⭐⭐ First to unify multi-task continual learning with ICL theory.
*   Experimental Thoroughness: ⭐⭐⭐⭐⭐ Step-by-step validation from simplified models to GPT-2, non-linear tasks, and real Qwen models.
*   Writing Quality: ⭐⭐⭐⭐ Theoretically rigorous with intuitive mathematical explanations.
*   Value: ⭐⭐⭐⭐⭐ Directly guides multi-task prompt design and explains performance saturation in long contexts.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] FoeGlass: Simple In-Context Learning Is Enough for Red Teaming Audio Deepfake Detectors](foeglass_simple_in-context_learning_is_enough_for_red_teaming_audio_deepfake_det.md)
- [\[AAAI 2026\] Attention Retention for Continual Learning with Vision Transformers](../../AAAI2026/llm_safety/attention_retention_for_continual_learning_with_vision_transformers.md)
- [\[NeurIPS 2025\] Finding Structure in Continual Learning](../../NeurIPS2025/llm_safety/finding_structure_in_continual_learning.md)
- [\[AAAI 2026\] CATFormer: When Continual Learning Meets Spiking Transformers With Dynamic Thresholds](../../AAAI2026/llm_safety/catformer_when_continual_learning_meets_spiking_transformers_with_dynamic_thresh.md)
- [\[ACL 2026\] Membership Inference Attacks on In-Context Learning Recommendation](../../ACL2026/llm_safety/membership_inference_attacks_on_llm-based_recommender_systems.md)

</div>

<!-- RELATED:END -->
