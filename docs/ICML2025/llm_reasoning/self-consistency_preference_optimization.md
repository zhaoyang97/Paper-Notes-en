---
title: >-
  [Paper Note] Self-Consistency Preference Optimization
description: >-
  [ICML 2025][Reasoning][Self-consistency] Introduce the concept of self-consistency from inference into the training phase, construct preference pairs through a voting mechanism, and perform iterative training using a weighted DPO loss. This significantly improves the mathematical and logical reasoning capabilities of LLMs without requiring gold labels.
tags:
  - "ICML 2025"
  - "Reasoning"
  - "Self-consistency"
  - "Preference Optimization"
  - "Self-training"
  - "Unsupervised Inference Alignment"
  - "DPO"
date: 2026-05-08
content_hash: 5d2c2a69f64d9380
---

# Self-Consistency Preference Optimization

**Conference**: ICML 2025  
**arXiv**: [2411.04109](https://arxiv.org/abs/2411.04109)  
**Code**: None  
**Area**: LLM Alignment/RLHF  
**Keywords**: Self-consistency, Preference Optimization, Self-training, Unsupervised Inference Alignment, DPO

## TL;DR

Introduce the concept of self-consistency from inference into the training phase, construct preference pairs through a voting mechanism, and perform iterative training using a weighted DPO loss. This significantly improves the mathematical and logical reasoning capabilities of LLMs without requiring gold labels.

## Background & Motivation

Self-alignment of LLMs is currently a popular research area, with the core goal of enabling models to self-improve without human annotations. However, existing methods have limited effectiveness on **complex reasoning tasks**, with the main bottlenecks as follows:

**LLMs cannot reliably self-evaluate the correctness of reasoning**: Huang et al. (2024) demonstrate that LLMs struggle to judge whether their answers are correct on complex problems with unique correct answers, causing methods like Self-Rewarding to fail on reasoning tasks.

**Poor generalization of external Reward Models (RMs)**: Even if an RM is trained on reasoning tasks, it still fails when facing out-of-distribution problems.

**Existing iterative training methods rely on gold labels**: Methods like STaR and IRPO require gold standard answers to construct training data, which limits their scalability.

Meanwhile, **self-consistency**, as an inference-time technique, has been proven effective—sampling multiple answers for the same problem and selecting the most frequent answer. The intuition behind this is that model errors are usually random, and the model is unlikely to generate the same incorrect answer multiple times.

**Key Insight**: Can this inference-time consistency signal be utilized in the training phase? Consistent answers are more likely to be correct, and inconsistent answers are more likely to be incorrect—which naturally forms preference pairs.

## Method

### Overall Architecture

ScPO is an **unsupervised iterative training framework**, where each iteration consists of three steps:

1. **Question Generation and Filtering**: Generate new reasoning questions using the current model, and filter out unanswerable questions based on consistency.
2. **Constructing Self-Consistency Preference Pairs**: Sample multiple answers for each question, selecting the most consistent one as the chosen response and the most inconsistent one as the rejected response.
3. **Weighted Preference Optimization Training**: Train the model with a weighted loss function based on the voting margin.

The entire process starts from a seed model $M_0$ and iteratively generates $M_1, M_2$ (experiments show that 2 iterations are sufficient).

### Key Designs

1. **Voting Function and Preference Pair Construction**: For each question $x$, sample $k$ answers $\{y_1, ..., y_k\}$ from the current model. The voting function $\mathcal{V}(y)$ computes the number of answers that share the same final answer as $y$. The answer with the highest vote count is selected as the chosen $y^+$, while the one with the lowest is selected as the rejected $y^-$, with a requirement that $\mathcal{V}(y^+) \geq \tau$ (threshold filtering). The elegance of this design lies in not needing to know the correct answer, but judging the quality solely based on the consistency among the answers.

2. **Consistency-Based Question Generation and Filtering**: Use few-shot prompting to let the model generate new questions (only requiring target question templates, not the corresponding answers). Then, sample answers for the generated questions and compute consistency, filtering out questions where all answers have a vote count $< \tau$—these are likely unanswerable or poorly formatted questions. This approach allows generating more diverse questions because it does not require generating the correct answers simultaneously.

3. **Instance-Level Weighting Mechanism**: The quality of preference pairs varies—pairs with large voting margins are more reliable, whereas those with small margins are noisier. ScPO introduces an instance weight $w(x) = (\mathcal{V}(y^+) - \mathcal{V}(y^-)) / k$, normalized to $[0, 1]$. High-confidence pairs receive larger weights, while weights of low-confidence pairs approach 0. When chosen and rejected have the same number of votes, $w(x)=0$, automatically ignoring the sample.

4. **Threshold Increment in Iterative Training**: As training iterates, the model becomes more consistent (validated by experiments), so the filtering threshold needs to be increased. For example, on GSM8K, $M_1$ training uses $\tau=0.5k$, and $M_2$ training increases it to $\tau=0.7k$. Meanwhile, to still sample rejected answers after the model becomes more consistent, an extra 8 answers are sampled with a higher temperature (1.2) to increase diversity.

5. **Semi-Supervised Extension (ScPO Semi-Sup.)**: When part of the data has gold labels, preference pairs are constructed directly using correct/incorrect answers for labeled questions and the weight is set to $w(x)=1$; for unlabeled questions, self-consistency is still used. When all data are labeled, it degenerates to the IRPO loss. This design allows ScPO to seamlessly integrate existing annotated data.

### Loss & Training

The ScPO loss function consists of two parts:

$$\mathcal{L}_{\text{ScPO}}(y^+, y^- | x) = \underbrace{-w(x) \log \sigma\left(\beta \log \frac{M_\theta(y^+|x)}{M_t(y^+|x)} - \beta \log \frac{M_\theta(y^-|x)}{M_t(y^-|x)}\right)}_{\text{Weighted DPO Loss}} \underbrace{- \frac{\alpha \cdot w(x)}{|y^+|} \log M_\theta(y^+|x)}_{\text{Weighted NLL Loss}}$$

- **Weighted DPO term**: Boosts the probability of chosen and suppresses the probability of rejected, with the weight determined by the voting margin.
- **Weighted NLL term**: Performs supervised learning directly on the chosen answer to prevent the model from drifting too much.

Key hyperparameter settings: $\beta=0.5$, $\alpha=1$, learning rate $5 \times 10^{-6}$ (cosine schedule), batch size 16, and training for 10 epochs. Each iteration uses the model from the previous round $M_t$ as the reference model for DPO. GSM8K/MATH uses $k=8$, and ZebraLogic uses $k=16$ (since logic puzzles are difficult for sampling consistent answers).

## Key Experimental Results

### Main Results

**GSM8K (Llama-3 Base 8B)**:

| Method | Iteration | Greedy Acc. | SC Acc. | Description |
|------|------|-------------|---------|------|
| Seed (M0) | - | 41.17% | 51.80% | Zero-shot baseline |
| IRPO_RM | M2 | 50.11% | 61.25% | External RM scoring |
| LMSI | M2 | 56.71% | 62.55% | Consistency + NLL fine-tuning |
| **ScPO_Unsup.** | **M2** | **63.91%** | **71.11%** | **+22.74% vs seed** |
| IRPO_Gold | M2 | 64.29% | 72.56% | Upper bound with gold labels |
| **ScPO_Semi-Sup.** | **M2** | **66.64%** | **74.75%** | **Surpasses gold-label IRPO** |

**MATH (Llama-3 Base 8B)**:

| Method | Iteration | Greedy Acc. | SC Acc. | Description |
|------|------|-------------|---------|------|
| Seed (M0) | - | 14.46% | 18.20% | Zero-shot baseline |
| IRPO_RM | M2 | 18.08% | 22.64% | External RM scoring |
| LMSI | M2 | 16.96% | 20.20% | Consistency + NLL fine-tuning |
| **ScPO_Unsup.** | **M2** | **19.72%** | **24.58%** | **+5.26% vs seed** |
| IRPO_Gold | M2 | 20.32% | 26.88% | Upper bound with gold labels |
| **ScPO_Semi-Sup.** | **M1** | **19.88%** | **27.35%** | **SC surpasses IRPO_Gold** |

**ZebraLogic (Llama-3 Instruct 8B)**:

| Method | Puzzle Acc. | Easy | Hard | Cell Acc. |
|------|------------|------|------|-----------|
| Llama-3 70B | 17.2% | 52.1% | 3.6% | 42.9% |
| Gemma-2 27B | 16.3% | 50.7% | 2.9% | 41.2% |
| Claude-3 Haiku | 14.3% | 47.9% | 1.2% | 37.9% |
| M0 (8B) | 11.6% | 40.0% | 0.4% | 39.1% |
| **ScPO M2 (8B)** | **18.1%** | **58.2%** | **2.5%** | **45.2%** |

### Ablation Study

**Weighted loss vs unweighted ablation**:

| Configuration | GSM8K Greedy | MATH Greedy | Description |
|------|-------------|-------------|------|
| M1 w/ w(x)=1 | 58.53% | 15.92% | Unweighted |
| M1 w/ ScPO | 61.03% | 17.36% | Weighted, +2.5%/+1.44% |
| M2 w/ w(x)=1 | 62.62% | 18.74% | Unweighted |
| M2 w/ ScPO | 63.91% | 19.72% | Weighted, +1.29%/+0.98% |

**Threshold ablation (MATH)**:

| Threshold τ | Accuracy Gap | Training Data Size | Test Accuracy |
|--------|----------|-----------|-----------|
| 0.1k | 18% | 6.7K | 15.44% |
| 0.3k | 44% | 2.4K | 16.34% |
| **0.5k** | **57%** | **1.8K** | **17.36%** |
| 0.7k | 68% | 0.7K | 14.76% |

### Key Findings

1. **Unsupervised ScPO performance is within 1% of IRPO with gold labels**: On GSM8K, the greedy accuracy gap is only 0.38%, demonstrating that self-consistency can effectively replace gold labels.
2. **An 8B model trained with ScPO surpasses 70B models**: On ZebraLogic, the 8B model trained with ScPO outperforms Llama-3 70B (+0.9%), Gemma-2 27B (+1.8%), and Claude-3 Haiku (+3.8%).
3. **Self-consistency is more reliable than external RMs**: Analysis shows that ArmoRM exhibits more misrankings when classifying preference pairs than self-consistency (especially in OOD scenarios like ZebraLogic, where it has 12.3% more misrankings).
4. **Model consistency increases with iterations**: The vote share increases significantly after each round of training, indicating that ScPO effectively distills the self-consistency distribution into the model's single-sample distribution.
5. **Optimal trade-off point exists for the threshold**: A threshold of $\tau=0.5k$ achieves the best balance—lower thresholds introduce too much noise, while higher thresholds lead to insufficient data.

## Highlights & Insights

- **Elegant design of unsupervised signals**: Converting an inference-time technique (self-consistency) into training signals avoids the limitations of both LLM self-evaluation and external RMs. Using consistency as a proxy for correctness is both simple and effective.
- **Clear theoretical intuition for weighted loss**: The voting margin reflects the model's confidence in preference labeling. High-confidence pairs should receive more weight, which aligns perfectly with the idea of uncertainty weighting.
- **Strong scalability**: Not relying on gold labels means training data can be expanded infinitely with model-generated questions, breaking through the limitations of existing datasets.
- **Complementary to inference-time SC**: Utilizing ScPO during training and SC during inference can yield additional gains, though the marginal benefits of SC during inference diminish as iterations progress (since the model itself becomes more consistent).

## Limitations & Future Work

1. **Dependency on parsable final answers**: The current approach requires extracting discrete final answers from responses to compute consistency, making it inapplicable to open-ended tasks (such as summarization or translation). The authors suggest extending this with universal self-consistency in the future.
2. **Base model capability bottlenecks**: If the seed model is completely unable to generate consistent answers for a specific task, ScPO cannot perform bootstrapping—this explains why the data size for MATH is significantly smaller than for GSM8K.
3. **Only validated at 8B scale**: It remains unknown whether larger models (70B+) still exhibit similar gains.
4. **Diminishing returns in iterations**: There is almost no gain in the third iteration, and the inference-time SC benefit also decreases.
5. **Lack of integration with Process Reward Models (PRMs)**: Combining this method with step-level verification might yield further improvements.

## Related Work & Insights

- **IRPO (Pang et al., 2024)**: The loss function of ScPO is based on the DPO+NLL structure of IRPO, but introduces instance-level weighting.
- **Self-Rewarding LM (Yuan et al., 2024)**: A similar iterative training framework, but using LLM-as-a-judge instead of self-consistency, which yields poor performance on reasoning tasks.
- **STaR (Zelikman et al., 2022)**: Iteratively bootstraps reasoning capabilities but relies on gold labels for filtering.
- **LMSI (Huang et al., 2023)**: Also utilizes self-consistency for unsupervised training, but relies solely on NLL loss instead of preference optimization.

**Insight**: Self-consistency might be an underestimated training signal. In broader scenarios, any form of "internal model consistency" (across sampling, reformulations, or languages) could serve as a proxy label for preference optimization.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Converting an inference-time technique into training signals is a novel direction, although individual components (DPO, self-consistency, iterative training) are derived from existing work.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Three datasets, multiple baselines, and comprehensive ablations (weighting/thresholds/consistency analysis).
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure, well-articulated motivation, and intuitive diagrams.
- **Value**: ⭐⭐⭐⭐ — A practical method that approaches supervised performance in settings without gold labels, though limited to reasoning tasks with parsable answers.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Unsupervised Visual Chain-of-Thought Reasoning via Preference Optimization](../../ICCV2025/llm_reasoning/unsupervised_visual_chain-of-thought_reasoning_via_preference_optimization.md)
- [\[ACL 2025\] Ranked Voting based Self-Consistency of Large Language Models](../../ACL2025/llm_reasoning/ranked_voting_based_self-consistency_of_large_language_models.md)
- [\[ACL 2025\] Revisiting Self-Consistency from Dynamic Distributional Alignment Perspective on Answer Aggregation](../../ACL2025/llm_reasoning/revisiting_self-consistency_from_dynamic_distributional_alignment_perspective_on.md)
- [\[NeurIPS 2025\] A Theoretical Study on Bridging Internal Probability and Self-Consistency for LLM Reasoning](../../NeurIPS2025/llm_reasoning/a_theoretical_study_on_bridging_internal_probability_and_sel.md)
- [\[ICLR 2026\] Toward Effective Tool-Integrated Reasoning via Self-Evolved Preference Learning](../../ICLR2026/llm_reasoning/toward_effective_tool-integrated_reasoning_via_self-evolved_preference_learning.md)

</div>

<!-- RELATED:END -->
