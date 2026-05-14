---
title: >-
  [Paper Note] RankLLM: Weighted Ranking of LLMs by Quantifying Question Difficulty
description: >-
  [ICLR 2026][LLM Evaluation][question difficulty] This paper proposes RankLLM, a non-parametric framework based on bidirectional score propagation over a directed bipartite graph…
tags:
  - "ICLR 2026"
  - "LLM Evaluation"
  - "question difficulty"
  - "model competency"
  - "bipartite graph"
  - "score propagation"
  - "benchmark"
date: 2026-05-08
content_hash: 2f088fce638a4db9
---

# RankLLM: Weighted Ranking of LLMs by Quantifying Question Difficulty

**Conference**: ICLR 2026
**arXiv**: [2602.12424](https://arxiv.org/abs/2602.12424)
**Code**: Not released (a HuggingFace leaderboard platform has been established)
**Area**: LLM Evaluation
**Keywords**: LLM evaluation, question difficulty, model competency, bipartite graph, score propagation, benchmark

## TL;DR

This paper proposes RankLLM, a non-parametric framework based on bidirectional score propagation over a directed bipartite graph, which jointly estimates question difficulty and model competency to achieve difficulty-aware LLM ranking, reaching 90% agreement with human judgments.

## Background & Motivation

Mainstream LLM evaluation benchmarks (e.g., MMLU-Pro, MATH, GSM8K) typically compress performance into per-category accuracy, implicitly treating all questions as equally important. This approach has several critical issues:

**Difficulty differences are ignored**: Treating a simple arithmetic problem and a multi-step calculus derivation as equivalent fails to distinguish pattern matching from advanced reasoning.

**Unstable rankings**: When the ratio of easy to hard questions changes, model rankings may reverse.

**Fine-grained differences are obscured**: Capability gaps between models with similar overall accuracy are masked.

Although Item Response Theory (IRT) methods attempt to model question difficulty, they require parameterized logistic fitting for each item, incurring high computational cost and proving impractical under small sample sizes or large datasets.

## Method

### Overall Architecture

RankLLM models questions and models as nodes in a directed bipartite graph $\mathcal{G}=(\mathcal{V}, \mathcal{E})$, jointly estimating question difficulty $\pi_q$ and model competency $\pi_m$ via damped bidirectional score propagation. The core intuition is: **a model gains higher competency scores for answering hard questions correctly, and a question gains higher difficulty scores for stumping strong models**.

### Key Designs

**Bipartite graph construction**: The vertex set $\mathcal{V}=\mathcal{M}\cup\mathcal{Q}$ contains $M$ models and $Q$ questions. The edge set is divided into two types:
- **Competency edges** $\mathcal{E}_{\text{Comp}}$: $q_i \to m_j$ indicating model $m_j$ answered question $q_i$ correctly.
- **Difficulty edges** $\mathcal{E}_{\text{Fail}}$: $m_j \to q_i$ indicating model $m_j$ failed to answer question $q_i$.

**Performance matrices**:
- Competency matrix $A \in \{0,1\}^{Q \times M}$, where $A_{ij}=1$ denotes model $m_j$ answered question $q_i$ correctly.
- Difficulty matrix $\hat{A} = (\mathbf{1}^{Q \times M} - A)^\top$.

Questions answered correctly or incorrectly by all models (approximately 2%) are excluded during preprocessing to ensure graph connectivity.

**Transition matrices**:
- Competency transition: $P_{Q \to M} = \text{diag}(A\mathbf{1}_M)^{-1} A$
- Difficulty transition: $P_{M \to Q} = \text{diag}(\hat{A}\mathbf{1}_Q)^{-1} \hat{A}$

### Iterative Score Propagation

A damping factor $\alpha \in (0,1)$ (analogous to PageRank teleportation) is introduced to resolve the 2-periodicity of bipartite graphs:

$$\pi_Q^{(t+1)} = \alpha P_{M \to Q}^\top \pi_M^{(t)} + (1-\alpha)\frac{\mathbf{1}_Q}{Q}$$

$$\pi_M^{(t+1)} = \alpha P_{Q \to M}^\top \pi_Q^{(t+1)} + (1-\alpha)\frac{\mathbf{1}_M}{M}$$

This iterative process constitutes an ergodic Markov chain, with convergence to a unique stationary distribution guaranteed by the Perron-Frobenius theorem.

### Continuous Score Extension

For benchmarks providing partial scores, the binary matrix $A$ is replaced by a continuous matrix $A_c \in [0,1]^{Q \times M}$, with all subsequent formulas remaining formally identical.

### Loss & Training

RankLLM requires no training or optimization objective; it converges directly through iterative propagation. The entire process is non-parametric, with only a single damping hyperparameter $\alpha$.

## Key Experimental Results

### Main Results

Evaluation is conducted on 6 benchmarks, 35,550 questions, and 30 models:

| Dataset | # Questions |
|---------|-------------|
| BBH | 6,511 |
| GPQA | 448 |
| GSM8K | 1,320 |
| HellaSwag | 10,000 |
| MATH | 5,000 |
| MMLU-Pro | 12,102 |

**Human alignment**: RankLLM achieves **90%** agreement with human consensus, significantly outperforming Simple Rank (62.9%), 1PL-IRT (50.0%), 2PL-IRT (51.4%), and Multi-IRT (52.9%).

**Key ranking findings**: The Kendall's Tau between RankLLM scores and accuracy is 0.8492, indicating overall consistency but significant reordering among adjacent models. For example, Qwen2-0.5B (accuracy 20.2%) ranks higher than DeepSeek-Chat-Lite (accuracy 30.49%), because the former achieves a 5.5% vs. 2.4% correct rate on hard questions.

### Ablation Study / Efficiency Analysis

| Method | Convergence Time (s) |
|--------|----------------------|
| **RankLLM** | **0.00597** |
| 1PL-IRT | 1,782.75 |
| 2PL-IRT | 3,787.03 |
| Multi-IRT (3D) | 18.76 |

RankLLM is more than 3,100× faster than the fastest IRT baseline.

**Robustness**: When $k$ models ($k$ = 1–15) are randomly removed, question difficulty Spearman correlation remains above 0.938 and model competency correlation above 0.993.

**Scalability**: Tests scaling to $Q=1{,}000{,}000$ and $M=2{,}000$ consistently converge within 9 iterations, with complexity linear in $Q \times M$.

### Key Findings

1. **Dataset difficulty distributions**: MATH and MMLU-Pro exhibit wider difficulty distributions, making them suitable for evaluating advanced reasoning; GSM8K and HellaSwag skew easier.
2. **Model family consistency**: Models within the same family maintain stable difficulty distribution patterns across parameter scales (observed for Llama, Qwen, and Yi); scaling primarily affects absolute accuracy.
3. **Reliability of open-weight models**: Difficulty estimates derived solely from open-weight models are highly correlated with those from the full model pool (Spearman 0.96, Pearson 0.94).
4. **Diversity gains**: A mixed-scale model pool reduces extreme estimation errors by 83% and achieves the highest agreement with human judgments (90% consensus).

## Highlights & Insights

1. **Elegant non-parametric design**: The entire method relies on a single damping hyperparameter, requiring no per-question parameter fitting — far simpler than IRT.
2. **Exceptional computational efficiency**: Evaluation of 30 models × 35K questions completes in 0.006 seconds on consumer hardware.
3. **Strong human alignment**: 90% consensus agreement represents one of the highest levels reported in this area.
4. **Theoretical guarantees**: Convergence proof based on the Perron-Frobenius theorem ensures methodological reliability.
5. **Rich practical insights**: Counter-intuitive findings are revealed, such as Qwen2-0.5B outperforming DeepSeek-Chat-Lite on hard questions.

## Limitations & Future Work

1. Difficulty is defined based on success/failure patterns across a model pool; estimates may be biased when the pool is homogeneous.
2. Only answer correctness is considered; the quality of reasoning processes is not captured.
3. There is no theoretically optimal guidance for selecting the damping factor $\alpha$.
4. The human evaluation is limited in scale (20 annotators, 70 question pairs), and statistical power could be further improved.

## Related Work & Insights

- **Relation to IRT**: IRT (1PL/2PL) requires parameterized fitting, is computationally expensive, and is unstable under small sample sizes; RankLLM is a purely graph-based algorithm with linear complexity.
- **Relation to PageRank**: The damped propagation in RankLLM is essentially a bipartite variant of PageRank, adapted for the evaluation setting with bidirectional competency–difficulty propagation.
- **Implications for evaluation practice**: A mixed-scale model pool constitutes the optimal evaluation configuration; homogeneous model pools introduce systematic bias in difficulty estimation.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Non-parametric joint estimation of difficulty and competency; concise and elegant.
- **Practicality**: ⭐⭐⭐⭐⭐ — Extremely low computational cost; HuggingFace leaderboard already available.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 6 benchmarks, 30 models, human evaluation, and robustness analysis all included.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear exposition with complete mathematical derivations.
- **Overall**: ⭐⭐⭐⭐ — Simple and effective method, though theoretical contributions are relatively modest.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] EvaLearn: Quantifying the Learning Capability and Efficiency of LLMs via Sequential Problem Solving](../../NeurIPS2025/llm_evaluation/evalearn_quantifying_the_learning_capability_and_efficiency_of_llms_via_sequenti.md)
- [\[ICLR 2026\] Benchmarking Overton Pluralism in LLMs](benchmarking_overton_pluralism_in_llms.md)
- [\[ICLR 2026\] Truthfulness Despite Weak Supervision: Evaluating and Training LLMs Using Peer Prediction](truthfulness_despite_weak_supervision_evaluating_and_training_llms_using_peer_pr.md)
- [\[ICLR 2026\] GuidedSampling: Steering LLMs Towards Diverse Candidate Solutions at Inference-Time](guidedsampling_steering_llms_towards_diverse_candidate_solutions_at_inference-ti.md)
- [\[ICLR 2026\] Enabling Fine-Grained Operating Points for Black-Box LLMs](enabling_fine-grained_operating_points_for_black-box_llms.md)

</div>

<!-- RELATED:END -->
