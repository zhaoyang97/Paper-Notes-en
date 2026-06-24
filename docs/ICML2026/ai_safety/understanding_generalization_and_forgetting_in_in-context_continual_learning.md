---
title: >-
  [Paper Note] Understanding Generalization and Forgetting in In-Context Continual Learning
description: >-
  [ICML 2026][AI Safety][In-context learning] Establishes the first theoretical framework for In-Context Continual Learning (ICCL)—revealing that attention mechanisms inevitably produce systematic bias and task interference when processing multi-task sequences, leading to task-order-dependent degradation in generalization and memory.
tags:
  - "ICML 2026"
  - "AI Safety"
  - "In-context learning"
  - "continual learning"
  - "forgetting"
  - "task interference"
  - "Transformer theory"
date: 2026-05-08
content_hash: e16fe79fa1b30d6d
---

# Understanding Generalization and Forgetting in In-Context Continual Learning

**Conference**: ICML 2026  
**arXiv**: [2605.28705](https://arxiv.org/abs/2605.28705)  
**Code**: TBD  
**Area**: LLM Reasoning / Continual Learning / Attention Mechanism  
**Keywords**: In-context learning, continual learning, forgetting, task interference, Transformer theory

## TL;DR
Establishes the first theoretical framework for In-Context Continual Learning (ICCL)—revealing that attention mechanisms inevitably produce systematic bias and task interference when processing multi-task sequences, leading to task-order-dependent degradation in generalization and memory.

## Background & Motivation

**Background**: LLMs adapt to new tasks during inference via In-Context Learning (ICL) without parameter updates. However, existing theories primarily focus on single-task ICL, whereas real-world prompts often contain sequences of heterogeneous tasks.

**Limitations of Prior Work**: Classical continual learning investigates forgetting caused by parameter updates; however, ICL is a parameter-frozen, purely attention-based adaptation. A theoretical gap exists between these domains—it remains unclear whether LLMs implicitly perform continual learning when processing long-context multi-task sequences and when forgetting occurs.

**Key Challenge**: Single-task ICL theory (assuming all examples come from the same function) cannot capture inter-task dependencies; classical continual learning theory is based on training-time updates, which fundamentally differs from inference-time pure attention adaptation.

**Goal**: (1) Construct the first theoretical framework for In-Context Continual Learning (ICCL); (2) Quantify generalization and forgetting in multi-task sequences; (3) Explain experimental phenomena such as task-order sensitivity and performance saturation in long prompts.

**Key Insight**: The aggregation method of attention mechanisms on historical context. Standard linear attention aggregates uniformly (leading to future information leakage); masked linear attention aggregates causally (respecting task order but still producing systematic interference).

**Core Idea**: Characterize how task similarity, context length, and task order jointly determine whether historical information leads to positive or negative transfer through an explicit **bias-variance-interference decomposition**.

## Method

### Overall Architecture
Ours formalizes In-Context Continual Learning (ICCL) as an analytically solvable regression problem: prompt sequences are formed by concatenating examples and queries from $T$ regression tasks and fed into a shared attention layer. For each task $t$, the model must infer task parameters $w_t$ from $M$ in-context examples $\{(x_{t,i}, y_{t,i})\}_{i=1}^{M}$ and predict the query $y_{t,q} = \langle w_t, x_{t,q} \rangle$. Within this setting, the authors derive closed-form expressions for generalization error $G_t$ (current task prediction accuracy) and forgetting error $F_t$ (degradation of previous tasks when processing subsequent tasks), transforming the question of "when positive transfer or forgetting occurs" into a calculable quantity.

### Key Designs

**1. Masked Linear Attention Model: Encoding task order with causal masks to prevent future information leakage**

Standard linear attention, commonly used in single-task ICL theory, aggregates all contexts uniformly. In multi-task sequences, this allows task $t$ to "peek" at upcoming examples, violating the causality of "continual learning." The authors employ masked linear attention $f_{\text{MSA}}(P;\theta) = P + WP \cdot \mathrm{mask}(P^\top VP)$, where $\mathrm{mask}(A)_{i,j} = \frac{1}{j}A_{i,j}$ (if $i \le j$) and $0$ otherwise, forcing information flow only from past to present. This respects task arrival order while maintaining the closed-form analyzability of linear attention, laying the foundation for error decomposition.

**2. Bias-variance-interference decomposition: Splitting prediction error into three parts to clarify context utility**

To determine if increasing context length is beneficial, one must identify the dominant error source. The authors explicitly decompose the prediction error of task $t$, $\mathbb{E}[(\hat{y}_{t,q} - y_{t,q})^2]$, into three components: (i) irreducible error; (ii) finite-sample variance, of magnitude $\sim O(1/M)$, which decays as context examples increase; and (iii) task mean bias $\|\frac{1}{t}\sum_{s=1}^t w_s - w_t\|_2^2$, representing the deviation of the averaged historical task parameters from the current $w_t$. The key insight is that while the variance term can be suppressed by increasing $M$, the bias term depends solely on task similarity and is independent of $M$. Consequently, adding context is helpful when tasks are aligned and variance dominates; however, when tasks are heterogeneous and bias dominates, additional context pulls predictions toward an incorrect historical mean.

**3. Task similarity and order dependency coefficients: Attributing forgetting to the redistribution of attention weights**

Forgetting in this framework is not parameter overwriting but the recalibration of attention weights over historical positions when processing new tasks. The authors introduce dependency coefficients $\alpha_i(t) = c_t < 0$ (for $i \le t$) and $\alpha_i(t) = d > 0$ (for $i > t$) to characterize this redistribution: learned past tasks are assigned negative weights (canceling their contribution, thus degrading), while unlearned future tasks are assigned positive weights (interfering with the current task). Expanding the forgetting error yields two types of terms: intra-task variance noise terms multiplied by $\alpha_i^2(t) = O(1/M^2)$, which decay overall at $O(1/M)$ (suppressible by longer context); and inter-task mean interference terms $\mathrm{tr}(\mu_i\mu_j^\top\Gamma^{-2}\Lambda)$, which include $M^2$ but are offset by $O(1/M^2)$ coefficients. Thus, **these terms converge to a non-zero constant rather than decaying to 0 as $M$ increases**. This explains why long contexts cannot eliminate forgetting—as long as subsequent tasks are not orthogonal to current task means, residual interference persists, and higher task alignment can paradoxically increase this interference constant.

## Key Experimental Results

### Main Results

| Factor | Impact on Generalization | Impact on Forgetting |
|------|-----------|-----------|
| Context Length $M$ | Reduces variance; however, performance saturates or worsens after the alignment threshold as bias dominates | Variance term decays $O(1/M)$; mean interference term remains constant (forgetting persists even as $M \to \infty$) |
| Task Similarity | Historical information reduces bias when tasks align; bias grows rapidly when tasks are heterogeneous | Interference term $\mathrm{tr}(\mu_i\mu_j^\top\Gamma^{-2}\Lambda)$ increases significantly when future tasks align with the current task |
| Task Order | Irrelevant for single tasks; determines the historical aggregation method in multi-task scenarios | Order-sensitive: the alignment level between subsequent and previous tasks determines interference intensity |

### Key Findings

| Experiment | Phenomenon | Verification |
|------|------|------|
| Non-monotonicity | Increasing $M$ from 3 to 19 for Task 2 caused error to jump from a minimum to 0.99 | Fully aligns with theoretical predictions |
| Task Clustering Effects | Performance is high within clusters {Task 1, 3, 5}, while cross-cluster tasks cause strong interference | Consistent with theory |
| Real LLM | Task A forgetting in Qwen2.5 on SST-2→AG News sequence dramatically decreased 0.934→0.472 at $M=1$ | Theoretical predictions match perfectly |

## Highlights & Insights
- **First theoretical bridge**: Unifies the independent fields of ICL and continual learning, explaining the forgetting mechanism under frozen parameters through attention weight recalibration.
- **Analyzable but transferable simplified models**: Linear regression assumptions remain valid for non-linear two-layer ReLU networks.
- **Practice Insight**: Multi-task prompts should be designed with specific task orders, or smaller $M$ should be used to balance negative transfer.

## Limitations & Future Work
- Assumes linear task distributions; real-world NLP tasks are more complex.
- Analysis is based on masked linear attention; differences in softmax multi-head attention aggregation mechanisms are not explored.
- Does not account for positional encoding's ability to perceive task boundaries.
- Future directions: Design "task-boundary-aware" attention mechanisms; study prompt reordering to mitigate forgetting; extend analysis to non-linear self-attention and multi-head mechanisms.

## Related Work & Insights
- **vs. Single-task ICL theory (Bai et al. 2023, Von Oswald et al. 2023)**: They assume all examples are i.i.d.; ours introduces task sequence heterogeneity.
- **vs. Classical Continual Learning (Lin et al. 2023)**: They focus on forgetting via parameter updates; ours proves for the first time that attention aggregation inevitably leads to forgetting even with frozen parameters.
- **vs. Empirical ICCL studies (Kang et al. 2025)**: They demonstrate that ICL can perform continual learning; ours provides fundamental theoretical reasons for **when it succeeds and when it fails**.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to unify multi-task continual learning and ICL theory.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Verified incrementally from simplified models → GPT-2 → non-linear tasks → real Qwen models.
- Writing Quality: ⭐⭐⭐⭐ Rigorous theory with intuitive mathematics.
- Value: ⭐⭐⭐⭐⭐ Directly guides multi-task prompt design and explains performance saturation in long contexts.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] FoeGlass: Simple In-Context Learning Is Enough for Red Teaming Audio Deepfake Detectors](foeglass_simple_in-context_learning_is_enough_for_red_teaming_audio_deepfake_det.md)
- [\[ICML 2026\] Active Continual Learning with Metaplastic Binary Bayesian Neural Networks](active_continual_learning_with_metaplastic_binary_bayesian_neural_networks.md)
- [\[ICML 2026\] Robust In-Context Reinforcement Learning Under Reward Poisoning Attacks](robust_in-context_reinforcement_learning_under_reward_poisoning_attacks.md)
- [\[ICML 2026\] Forgetting is Not Deletion: An Investigation of Reversibility in LLM Machine Unlearning](unlearning_isnt_deletion_investigating_reversibility_of_machine_unlearning_in_ll.md)
- [\[ICML 2026\] Forget to Know, Remember to Use: Context-Aware Unlearning for Large Language Models](forget_to_know_remember_to_use_context-aware_unlearning_for_large_language_model.md)

</div>

<!-- RELATED:END -->
