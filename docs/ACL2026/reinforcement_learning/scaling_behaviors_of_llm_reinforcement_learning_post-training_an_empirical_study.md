---
title: >-
  [Paper Note] Scaling Behaviors of LLM Reinforcement Learning Post-Training: An Empirical Study
description: >-
  [ACL 2026][Reinforcement Learning][reinforcement learning post-training] This paper presents the first systematic study of scaling behaviors in LLM reinforcement learning post-training. Using the Qwen2.5 series (0.5B-72B…
tags:
  - "ACL 2026"
  - "Reinforcement Learning"
  - "reinforcement learning post-training"
  - "scaling laws"
  - "mathematical reasoning"
  - "learning efficiency"
  - "data reuse"
date: 2026-05-08
content_hash: cfbad105910b8457
---

# Scaling Behaviors of LLM Reinforcement Learning Post-Training: An Empirical Study

**Conference**: ACL 2026  
**arXiv**: [2509.25300](https://arxiv.org/abs/2509.25300)  
**Code**: [GitHub](https://github.com/reasoning360/rl-scaling)  
**Area**: Reinforcement Learning / Scaling Laws  
**Keywords**: reinforcement learning post-training, scaling laws, mathematical reasoning, learning efficiency, data reuse

## TL;DR

This paper presents the first systematic study of scaling behaviors in LLM reinforcement learning post-training. Using the Qwen2.5 series (0.5B-72B), it identifies a power-law relationship between performance and training resources, revealing that learning efficiency tends to saturate as model scale increases.

## Background & Motivation

**Background**: Scaling laws in the pre-training phase have been extensively researched; Kaplan et al. and Chinchilla established power-law relationships between loss and model size, data volume, and computation. While RL post-training (e.g., GRPO, RLHF) has become the mainstream paradigm for enhancing LLM reasoning capabilities, its scaling behaviors remain almost entirely unexplored.

**Limitations of Prior Work**: Practitioners lack guidance when conducting RL post-training: What model size should be selected? How much computational resource should be allocated? Should data be reused when samples are scarce? These critical questions lack quantitative answers, leading to significant trial-and-error and resource waste.

**Key Challenge**: Do pre-training scaling laws apply to RL post-training? RL post-training has unique characteristics—utilizing verifiable rewards instead of cross-entropy loss and employing on-policy sampling instead of i.i.d. data—differences that may lead to distinct scaling behaviors.

**Goal**: Systematically characterize the relationships between model scale, data volume, computation, and performance in RL post-training through large-scale experiments to establish predictable scaling formulas.

**Core Idea**: Test loss in RL post-training follows a log-linear (power-law) relationship with resource consumption: $\log L(N,X) = -k(N) \cdot \log X + E(N)$, where learning efficiency $k(N)$ improves with model scale but approaches a saturation limit $K_{\max}$.

## Method

### Overall Architecture

This study is a systematic empirical investigation. By training 63 LLMs (covering the full Qwen2.5 series from 0.5B to 72B, including both base and instruct variants) on mathematical reasoning tasks using the GRPO algorithm, the researchers measure performance across three resource-constrained scenarios and fit scaling formulas.

### Key Designs

1.  **Power-Law Scaling Formulation**:
    - **Function**: Establishes a predictable relationship between test loss, model scale, and resource consumption in RL post-training.
    - **Mechanism**: Findings show $\log L(N,X) = -k(N) \cdot \log X + E(N)$, where $X$ represents computation $C$ or data volume $D$. Learning efficiency $k(N)$ is modeled using a saturation function: $k(N) = \frac{K_{\max}}{1+N_0/N}$, indicating that larger models possess higher learning efficiency, though marginal returns diminish.
    - **Design Motivation**: Analogous to pre-training scaling laws, but incorporating a saturation term to capture the diminishing marginal effects unique to RL post-training.

2.  **Inter/Intra-model Prediction**:
    - **Function**: Validates the predictive power of the scaling formulas.
    - **Mechanism**: Inter-model prediction—using fitted parameters from 0.5B-32B models to predict the performance of the 72B model. Intra-model prediction—using early training steps to predict subsequent training trajectories. Goodness-of-fit $R^2 > 0.99$ was achieved under both protocols.
    - **Design Motivation**: Predictability is the core value of scaling laws; these protocols verify the ability to predict across both model scales and training stages.

3.  **Data Reuse Analysis**:
    - **Function**: Addresses the practical question of whether data should be reused when it is insufficient.
    - **Mechanism**: By fixing the total data volume $D_{\mathrm{total}}$ and varying the reuse factor $\tau$ ($D_{\mathrm{unique}} \times \tau = D_{\mathrm{total}}$), experiments found that within the range of $\tau \leq 25$, performance is insensitive to the reuse factor. Performance is primarily determined by total training volume rather than the uniqueness of samples.
    - **Design Motivation**: High-quality reasoning data is a common bottleneck; verifying the effectiveness of data reuse carries significant practical weight.

### Loss & Training

The standard GRPO algorithm is used with binary reward signals (Correct=1, Incorrect=0). The primary evaluation metric is test loss $L = 1 - R/R_{\max}$. Training data utilizes the guru-RL-92k math subset (approximately 50k problems), sorted by increasing difficulty to implement curriculum learning. Each configuration is repeated 3 times to ensure robustness.

## Key Experimental Results

### Main Results

| Model Scale | Compute Efficiency $k_C(N)$ | Data Efficiency $k_D(N)$ | Description |
| :--- | :--- | :--- | :--- |
| 0.5B-3B | Rapid growth | Rapid growth | Small model stage; significant scale benefits. |
| 7B-32B | Growth slows | Growth slows | Efficiency gains begin to saturate. |
| 72B | Approaches $K_{\max}$ | Approaches $K_{\max}$ | Saturation trend is evident. |

### Prediction Accuracy

| Prediction Type | $R^2$ | Description |
| :--- | :--- | :--- |
| Inter-model (0.5B-32B → 72B) | >0.99 | Accurately predicts large model performance. |
| Intra-model (Early → Late) | >0.99 | Accurately predicts training trajectories. |
| Cross-architecture (Llama 3) | >0.99 | Formulas exhibit architectural independence. |

### Key Findings
- **Larger models are consistently superior in compute and data efficiency**, but marginal returns diminish after 32B, where gains decrease significantly.
- **Data reuse is highly effective**: No significant performance degradation occurs when $\tau \leq 25$; overfitting only appears when $\tau=100$.
- **Limited domain transfer**: RL post-training generalizes well within the math domain but provides almost no benefit for OOD tasks such as code or logical reasoning, and may even harm certain capabilities.
- **Crossover between 32B and 72B**: Under the same compute budget, 32B may outperform 72B in the early stages (as it can undergo more training steps), revealing a trade-off between model scale and training steps.

## Highlights & Insights
- **RL version of "Chinchilla"**: This work establishes scaling formulas for RL post-training analogous to pre-training scaling laws for the first time, filling a major theoretical gap.
- **Discovery of saturation trends**: The saturation of learning efficiency suggests that infinitely increasing model scale is not the optimal strategy for RL post-training, providing an upper-bound reference for resource allocation.
- **Practical value of data reuse**: Validating that moderate data reuse causes almost no performance loss provides direct guidance for data-scarce scenarios.
- **Cross-architecture validation**: Validation on the Llama 3 series strengthens the universality of the conclusions.
- **Warning on domain specialization**: The high degree of specialization in RL post-training (which may even harm other capabilities) serves as a vital practical reminder.

## Limitations & Future Work
- Experiments only cover the mathematical reasoning domain; scaling behaviors for multi-domain RL post-training remain unknown.
- The largest model is only 72B; the saturation trend for models over 100B cannot be empirically verified.
- Findings are based only on the GRPO algorithm; whether other RL algorithms (e.g., DAPO, PPO) exhibit different scaling behaviors remains to be explored.
- Only dense models were studied; scaling behaviors for MoE architectures in RL post-training were not covered.
- Absolute coefficients of the scaling formulas depend on the evaluation dataset and task difficulty, making them difficult to generalize universally.

## Related Work & Insights
- **vs Kaplan et al. (2020)**: Classic pre-training scaling laws focus on cross-entropy loss; this paper extends scaling laws to reward optimization in RL post-training.
- **vs Chinchilla (Hoffmann et al., 2022)**: Chinchilla provided compute-optimal model-data ratios; this paper provides a similar resource allocation guide for RL post-training.
- **vs Hilton et al. (2023)**: While that work found power-law relationships in CNN+RL environments, this paper validates similar patterns in the LLM+GRPO context.

## Rating
- Novelty: ⭐⭐⭐⭐ First systematic study of RL post-training scaling behaviors, filling an important gap.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Large-scale experiments involving 63 models, multiple scales, architectures, and scenarios, with 3 repetitions per configuration.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, rigorous formula derivation, and rich visualizations.
- Value: ⭐⭐⭐⭐⭐ Provides quantitative guidance for resource allocation in RL post-training, offering extreme practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] How Reasoning Evolves from Post-Training Data: An Empirical Study Using Chess](../../ICML2026/reinforcement_learning/how_reasoning_evolves_from_post-training_data_an_empirical_study_using_chess.md)
- [\[ACL 2026\] Why Does Reinforcement Learning Generalize? A Feature-Level Mechanistic Study of Post-Training in Large Language Models](why_does_reinforcement_learning_generalize_a_feature-level_mechanistic_study_of_.md)
- [\[ACL 2026\] Deliberative Searcher: Improving LLM Reliability via Reinforcement Learning with Constraints](deliberative_searcher_improving_llm_reliability_via_reinforcement_learning_with_.md)
- [\[ACL 2026\] LearnAlign: Data Selection for LLM Reinforcement Learning with Improved Gradient Alignment](learnalign_data_selection_for_llm_reinforcement_learning_with_improved_gradient_.md)
- [\[ACL 2026\] CE-GPPO: Coordinating Entropy via Gradient-Preserving Clipping Policy Optimization in Reinforcement Learning](ce-gppo_coordinating_entropy_via_gradient-preserving_clipping_policy_optimizatio.md)

</div>

<!-- RELATED:END -->
