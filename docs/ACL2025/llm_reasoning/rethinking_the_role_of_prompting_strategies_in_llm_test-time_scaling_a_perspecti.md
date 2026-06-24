---
title: >-
  [Paper Note] Rethinking the Role of Prompting Strategies in LLM Test-Time Scaling: A Perspective of Probability Theory
description: >-
  [ACL 2025][Reasoning][Test-time scaling] Through systematic experiments across 6 LLMs $\times$ 8 prompting strategies $\times$ 6 benchmarks, this work discovers that as the number of majority voting samples increases, simple CoT consistently outperforms complex prompting strategies. This phenomenon is theoretically proven from a probability perspective, and an $O(1)$ complexity scaling performance prediction method along with two improvement strategies are proposed.
tags:
  - "ACL 2025"
  - "Reasoning"
  - "Test-time scaling"
  - "majority voting"
  - "prompting strategies"
  - "Chain-of-Thought"
  - "probability analysis"
date: 2026-05-08
content_hash: 5dd5b7cf4ebf9b14
---

# Rethinking the Role of Prompting Strategies in LLM Test-Time Scaling: A Perspective of Probability Theory

**Conference**: ACL 2025  
**arXiv**: [2505.10981](https://arxiv.org/abs/2505.10981)  
**Code**: [GitHub](https://github.com/MraDonkey/rethinking_prompting)  
**Area**: LLM Inference  
**Keywords**: Test-time scaling, majority voting, prompting strategies, Chain-of-Thought, probability analysis

## TL;DR
Through systematic experiments across 6 LLMs $\times$ 8 prompting strategies $\times$ 6 benchmarks, this work discovers that as the number of majority voting samples increases, simple CoT consistently outperforms complex prompting strategies. This phenomenon is theoretically proven from a probability perspective, and an $O(1)$ complexity scaling performance prediction method along with two improvement strategies are proposed.

## Background & Motivation

**Background**: Test-time scaling (scaling computation during inference) is a recent hot topic in LLM inference, which improves reasoning capabilities by increasing computation during the inference phase (e.g., multi-sample querying + majority voting). Concurrently, researchers have designed various prompting strategies (such as CoT, ToT, Self-Refine, Step-Back) to enhance reasoning.

**Limitations of Prior Work**: Complex prompting strategies usually outperform simple CoT in pass@1 performance, but **how the relative performance of different strategies changes during test-time scaling (large-scale sampling + majority voting)** has not been systematically investigated. It has been widely assumed that "better pass@1 performance leads to better scaling performance," but this assumption remains unverified.

**Key Challenge**: Although complex strategies (e.g., ToT, MAD) are more accurate in a single inference, their per-inference cost is significantly higher. Under the same compute budget, they consume several times more tokens than simple CoT.

**Goal**
   - Which prompting strategy performs best under test-time scaling?
   - Why does this phenomenon occur? Is there a theoretical explanation?
   - Can the performance of different strategies across various sampling sizes be efficiently predicted?

**Key Insight**: From a probability theory perspective, majority voting is modeled as a mode selection problem of a multinomial distribution. The analysis investigates how the proportion of "easy questions" versus "hard questions" determines the scaling curves of different strategies.

**Core Idea**: Simple CoT consistently wins in test-time scaling because it yields more "easy questions" (where the correct answer has the highest probability) and the distribution of incorrect answers is more dispersed.

## Method

### Overall Architecture

Input: LLM $\mathcal{M}$, prompting strategy $\mathbf{P}_i$, sampling count $N$, dataset $\mathfrak{D}$  
Process: Sample $N$ times for each question, then select the final answer using majority voting  
Evaluation: Compare the accuracy of different strategies under a fixed sample count $N$ or a fixed inference cost $O$

### Key Designs

1. **Question Difficulty Definition (Definition 1)**:

    - **Function**: Categorizes questions into three difficulty levels for a given prompting strategy.
    - **Mechanism**: Let the answer space be $\mathcal{A} = \{a_1, ..., a_m\}$, where $p_{i,j}$ is the probability of strategy $\mathbf{P}_i$ outputting answer $a_j$. If the correct answer $a_1$ uniquely has the maximum probability, the question is classified as **Easy**; if the maximum probability is shared, it is **Medium**; if an incorrect answer has the maximum probability, it is **Hard**.
    - **Design Motivation**: Intuitively, the accuracy of easy questions converges to 100% after scaling, while that of hard questions converges to 0%.

2. **Scaling Convergence Theorems (Theorem 1-3)**:

    - **Function**: Proves the behavior of the three difficulty levels as $N \to \infty$.
    - **Mechanism**: **Easy questions** converge to 1; **Medium questions** converge to $1/|\mathcal{S}|$; **Hard questions** converge to 0.
    - **Design Motivation**: CoT has more easy questions and fewer hard questions (validated in Table 1).

3. **Strategy Flip Theorem (Theorem 4)**:

    - **Function**: Identifies under what conditions an initially inferior strategy will outperform the other as $N$ becomes large.
    - **Mechanism**: If a strategy exhibits a larger gap between the correct answer probability and the maximum incorrect answer probability, it will overtake the other when $N$ is large.
    - **Design Motivation**: The error distribution of CoT is more dispersed and does not concentrate on a specific incorrect answer.

4. **$O(1)$ Scaling Performance Prediction Method**:

    - **Function**: Estimates probability distributions from a small number of samples to predict performance under any $N$.
    - **Design Motivation**: Avoids the high computational cost of actual inference at large $N$ scales.

### Improvement Strategies

- **Adaptive scaling**: Query fewer samples for easy questions and more samples for hard questions.
- **Dynamic strategy selection**: Select the optimal prompting strategy dynamically for each question.

## Key Experimental Results

### Main Results (CoT vs. Other Strategies, Qwen2.5-7B as an Example)

| Strategy | Proportion of Easy Questions | Proportion of Hard Questions | Asymptotic Accuracy |
|------|-----------|---------|-----------|
| CoT | **88.1%** | **11.6%** | **88.2%** |
| L2M | 87.4% | 12.3% | 87.6% |
| SBP | 87.1% | 12.8% | 87.2% |
| DiP | 86.3% | 13.4% | 86.4% |

### Ablation Study (Effects of Improvement Strategies)

| Configuration | GSM8K Maj@10 | MATH-500 Maj@10 |
|------|-------------|-----------------|
| Pure CoT | 86.0% | 15.2% |
| + Combination of Both Improvements | **97.4%** | **61.0%** |

### Key Findings
- Across all six investigated LLMs, CoT becomes the optimal strategy when $N$ is sufficiently large, a trend that holds true for approximately 80% of the model-strategy combinations.
- Self-Refine exhibits the poorest performance under test-time scaling, falling even behind Direct Prompting.
- On stronger base models, Direct Prompting can eventually achieve top performance at large $N$.

## Highlights & Insights

- **Counter-intuitive Discovery**: Strategies with superior pass@1 performance do not necessarily scale better under test-time scaling.
- **Elegant and Practical Probability Analysis**: Majority voting is successfully modeled as the mode of a multinomial distribution.
- **$O(1)$ Prediction Method**: Enables rapid selection of the optimal strategy and sample count in practical deployments.

## Limitations & Future Work

- **Sole Focus on Majority Voting**: Other methods such as Best-of-N and Process Reward Models (PRMs) are left uninvestigated.
- **Answer Extraction Relies on Regular Expressions**: Direct application to open-ended generation tasks remains difficult.
- **Theoretical Assumptions on Independent Questions**: In practical scenarios, the distributions over different questions may exhibit correlation.

## Related Work & Insights

- **vs. Self-Consistency (Wang et al., 2023)**: This work systematically compares multiple prompting strategies within the framework of Self-Consistency.
- **vs. OpenAI o1/o3**: While o1 scaling utilizes trained reasoning, this paper focuses specifically on the prompting dimension.
- **vs. Scaling Laws**: The proposed prediction method can be viewed as a scaling law for test-time majority voting.

## Rating
- Novelty: ⭐⭐⭐⭐ Counter-intuitive discovery combined with theoretical proofs, supported by large-scale, comprehensive experiments.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 6 LLMs $\times$ 8 strategies $\times$ 6 benchmarks.
- Writing Quality: ⭐⭐⭐⭐ The theoretical portion is clear, though heavily laden with mathematical notation.
- Value: ⭐⭐⭐⭐ Provides direct guidance for practical test-time scaling implementations.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Understanding the Role of Training Data in Test-Time Scaling](../../ICLR2026/llm_reasoning/understanding_the_role_of_training_data_in_test-time_scaling.md)
- [\[ACL 2025\] Revisiting the Test-Time Scaling of o1-like Models: Do they Truly Possess Test-Time Scaling Capabilities?](revisiting_the_test-time_scaling_of_o1-like_models_do_they_truly_possess_test-ti.md)
- [\[ACL 2025\] Linguistic Generalizability of Test-Time Scaling in Mathematical Reasoning](mclm_multilingual_test_time_scaling.md)
- [\[NeurIPS 2025\] Rethinking Optimal Verification Granularity for Compute-Efficient Test-Time Scaling](../../NeurIPS2025/llm_reasoning/rethinking_optimal_verification_granularity_for_compute-efficient_test-time_scal.md)
- [\[ACL 2025\] Is That Your Final Answer? Test-Time Scaling Improves Selective Question Answering](test_time_scaling_selective_qa.md)

</div>

<!-- RELATED:END -->
