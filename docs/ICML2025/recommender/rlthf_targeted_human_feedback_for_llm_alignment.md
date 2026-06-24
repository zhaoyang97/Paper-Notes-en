---
title: >-
  [Paper Note] RLTHF: Targeted Human Feedback for LLM Alignment
description: >-
  [ICML 2025][Recommender Systems][RLHF] RLTHF proposes a hybrid human-AI framework for LLM alignment. By analyzing the reward distribution of the reward model to identify "hard samples" mislabeled by LLMs, it selectively annotates only these samples with human feedback, achieving or even surpassing the alignment quality of full-scale human annotation at only 6-7% of the cost.
tags:
  - "ICML 2025"
  - "Recommender Systems"
  - "RLHF"
  - "LLM Alignment"
  - "Human Feedback"
  - "Reward Model"
  - "Active Learning"
date: 2026-05-08
content_hash: 2d655cf21b77153f
---

# RLTHF: Targeted Human Feedback for LLM Alignment

**Conference**: ICML 2025  
**arXiv**: [2502.13417](https://arxiv.org/abs/2502.13417)  
**Code**: None  
**Area**: Recommender Systems  
**Keywords**: RLHF, LLM Alignment, Human Feedback, Reward Model, Active Learning

## TL;DR

RLTHF proposes a hybrid human-AI framework for LLM alignment. By analyzing the reward distribution of the reward model to identify "hard samples" mislabeled by LLMs, it selectively annotates only these samples with human feedback, achieving or even surpassing the alignment quality of full-scale human annotation at only 6-7% of the cost.

## Background & Motivation

RLHF (Reinforcement Learning from Human Feedback) is a core technology for current LLM alignment, but its effectiveness heavily relies on high-quality human annotations, which are extremely costly. To reduce costs, RLAIF (Reinforcement Learning from AI Feedback) emerged to substitute LLMs for human annotators. However, the issues with RLAIF are also evident: limited by prompt optimization, task complexity, and model biases, LLM annotations cannot fully replace human judgment, especially on **hard-to-distinguish samples**.

Key Challenge: LLMs can handle most "simple" preference judgments, but they are prone to errors on "hard samples" where fine-grained alignment is most needed—yet **precisely these hard samples are the most critical for model fine-tuning** (Ethayarajh et al., 2024). Full AI annotation offers insufficient quality, full human annotation is prohibitively expensive, and random-sampling-based human annotation is highly inefficient (failing to target the most valuable samples).

Key Insight: **Leverage the reward distribution of reward models to locate samples mislabeled by LLMs**. Specifically, when the reward model's prediction on training data is inconsistent with the training labels, it suggests that these samples may contain annotation errors. By analyzing the shape characteristics of the reward distribution curve ("elbow" and "knee"), human effort can be highly efficiently directed to the most valuable areas.

## Method

### Overall Architecture

RLTHF is divided into three stages:

1. **Initial Alignment**: Perform coarse-grained alignment on unlabeled data using a general-purpose LLM (e.g., GPT-4o).
2. **Iterative Alignment Improvement**: Locate mislabeled samples using the reward model's reward distribution $\rightarrow$ Selective human annotation $\rightarrow$ Retrain the reward model $\rightarrow$ Iterate.
3. **Knowledge Transfer**: Apply the aligned reward model to downstream tasks (via DPO or PPO).

### Key Designs

1. **Reward Distribution Analysis**: After training a reward model on the LLM-annotated dataset $\mathcal{D}_{\Lambda_{LLM}}$, calculate the reward difference for each sample pair $\Delta_\Lambda \hat{r}_\Lambda = \hat{r}(\rho_c) - \hat{r}(\rho_r)$. Sorting the samples by this value yields a monotonically decreasing reward distribution curve $\vartheta(\cdot)$. Samples above the curve (high positive reward difference) strongly align with LLM annotations and are highly likely to be correctly annotated "simple" samples; samples below (low or negative reward difference) contradict the training labels, likely representing LLM mislabels. Design Motivation: The reward model essentially learns the dominant preference features in the data; samples that conflict with these dominant features naturally receive low rewards.

2. **"Elbow" and "Knee" Localization**: Detect two key points via the first derivative of the reward distribution curve: the **"elbow"** indicates the transition to a high-accuracy region, and the **"knee"** indicates the transition to a low-accuracy region. The mirror reflection point of the "elbow" corresponds to samples with an extremely high probability of being mislabeled by the LLM (where their preference features highly conflict with the dominant features). For samples below the reflection point, their labels are directly flipped, while samples in the interval between the reflection point and the knee are handed over for human annotation. Design Motivation: This strategy precisely directs human effort to the region with the "highest cost-performance ratio," avoiding waste on simple samples that the LLM can already judge correctly.

3. **Iterative Training and Two Hyperparameters**: After each iteration, train a new reward model using human-annotated data + high-confidence LLM-annotated data. Two key hyperparameters control the balance between data quality and coverage:

    - **Back-off ratio ($\beta$)**: Controls how far back to shift to the left of the "knee" to select training samples. High $\beta$ = cleaner but lower coverage.
    - **Amplification ratio ($\alpha$)**: Amplifies the influence of human-annotated samples in training through oversampling. Excessive $\alpha$ leads to overfitting.
   
   Recommendation: Use high $\alpha$ and high $\beta$ in early iterations, and gradually decrease them in later iterations. Design Motivation: Early-stage data contains substantial noise and requires high-quality filtering, while later-stage data becomes increasingly clean, allowing for looser constraints to expand coverage.

4. **Randomized Shard Downsampling**: Run the iterative alignment first on a $1/4$ randomized shard of the dataset, and once satisfactory performance is achieved, use the final reward model to annotate the entire dataset. Design Motivation: Concentrate human effort in a smaller space and then propagate the alignment to the full dataset using the generalization capability of the reward model.

### Loss & Training

The reward model is trained using the standard Bradley-Terry model:

$$\mathcal{L}(\hat{r}) = -\mathbb{E}_{(x,y) \sim \mathcal{D}}[\log \sigma(\hat{r}(\rho_{c}) - \hat{r}(\rho_{r}))]$$

Downstream tasks are trained using DPO, and the evaluation relies on AlpacaEval with Claude 3.5 Sonnet as the judge.

## Key Experimental Results

### Main Results

| Dataset | Metric | RLTHF (4o) | AI-only (4o) | Random | Human | Human Annotation Volume |
|--------|------|-----------|-------------|--------|-------|----------|
| HH-RLHF | Preference Accuracy | 89.6% | 74.7% | - | 91.8% | 6% |
| TL;DR | Preference Accuracy | 88.0% | 78.8% | - | 89.6% | 7% |
| HH-RLHF | DPO Win Rate | 58.1% | 49.2% | 52.5% | 55.7% | 6% |
| TL;DR | DPO Win Rate | 62.3% | 59.2% | 59.8% | 60.2% | 7% |

Key Observation: RLTHF achieves a preference accuracy close to full-scale human annotation using only 6-7% of the human annotation budget; in downstream DPO training, RLTHF even **surpasses the win rate of full-scale human annotation**.

### Ablation Study

| Configuration | HH-RLHF Itr-5 Accuracy | TL;DR Itr-5 Accuracy | Description |
|------|---------------------|-------------------|------|
| Full RLTHF | 87.7% | 83.7% | Full Method |
| No Annotation (Pure Self-Improvement) | 75.7% | 75.2% | No human annotation, unable to surpass AI baseline |
| No Ampl./Back-off | 75.8% | 76.0% | No hyperparameter control, only marginal improvement |

ROI Comparison: Compared to Random annotation, the return on investment (ROI) of RLTHF is 15.9 times higher on HH-RLHF and 5.3 times higher on TL;DR.

### Key Findings

- **Pure AI Self-Improvement is Infeasible**: Without human annotation, relying solely on iterative training cannot break through the upper bound of AI preference accuracy.
- **Random Annotation is Highly Inefficient**: Under the same annotation budget, randomly selecting samples for annotation yields only marginal improvements.
- **Why RLTHF Surpasses Full Human Annotation**: The back-off mechanism effectively filters out the inherent noise and bias in human-annotated data (these "noisy samples" tend to gather in the "knee" region of the reward distribution curve).
- **Robustness to Weak AI Annotators**: Even when starting with GPT-4o mini (which is weaker than GPT-4o), the gap narrows to $< 0.5\%$ after 10% human annotation.
- **Iterative is Superior to One-shot**: Distributing the annotation budget across multiple iterations yields up to a $4.2\%$ improvement compared to one-shot annotation.

## Highlights & Insights

- **The reward distribution perspective** is highly elegant: it transforms the annotation quality problem into a visual and actionable distribution analysis task.
- Cost analysis (Appendix F) reveals that even when accounting for the costs of LLM annotations and additional RM training, the overall cost is still reduced by 84-86%.
- The geometric intuition of "elbow/knee/reflection point" is simple yet powerful, making it easy to understand and implement.
- Consideration of practical application scenarios (e.g., data invisibility when providing fine-tuning services to third-party clients) gives the method strong engineering feasibility.

## Limitations & Future Work

- Only validated on two preference datasets (HH-RLHF and TL;DR), without involving more complex alignment tasks such as code generation or mathematical reasoning.
- Relies on the Bradley-Terry model assumption that preferences can be modeled with scalar rewards, leaving more general preference structures undiscussed.
- The detection of "elbow" and "knee" relies on first-derivative heuristics, which might be unstable when the distribution shape is irregular.
- Although empirical recommendations are provided for tuning the hyperparameters $\alpha$ and $\beta$, automated methods are still lacking.
- Lacks a systematic comparison with other active learning strategies (e.g., uncertainty sampling, query-by-committee).

## Related Work & Insights

- The core idea shares similarities with active learning, but the novelty lies in using the **reward distribution** rather than traditional uncertainty metrics for sample selection.
- The comparison with SER (Huang et al., 2024) is meaningful: while SER pursues pure LLM self-improvement, this work demonstrates that it is infeasible and that human intelligence must be introduced.
- Food for thought: In recommender systems, user feedback also possesses a "simple/hard" distinction—user feedback indicating explicit satisfaction or dissatisfaction with recommendations is easy to label, but how to handle the "barely acceptable" gray area remains a key problem.

## Rating

- Novelty: ⭐⭐⭐⭐ The idea of using reward distribution analysis for annotation selection is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Solid ablations and detailed hyperparameter analysis, though only conducted on two datasets.
- Writing Quality: ⭐⭐⭐⭐ Clear method descriptions and intuitive schematics.
- Value: ⭐⭐⭐⭐⭐ Holds significant practical value for reducing RLHF annotation costs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Position: Towards Bidirectional Human-AI Alignment](../../NeurIPS2025/recommender/position_towards_bidirectional_human-ai_alignment.md)
- [\[ICML 2025\] PARM: Multi-Objective Test-Time Alignment via Preference-Aware Autoregressive Reward Model](parm_multi-objective_test-time_alignment_via_preference-aware_autoregressive_rew.md)
- [\[ICML 2025\] MATCHA: Toward Safe and Human-Aligned Game Conversational Recommendation via Multi-Agent Decomposition](toward_safe_and_human-aligned_game_conversational_recommendation_via_multi-agent.md)
- [\[AAAI 2026\] Align³GR: Unified Multi-Level Alignment for LLM-based Generative Recommendation](../../AAAI2026/recommender/align3gr_unified_multi-level_alignment_for_llm-based_generat.md)
- [\[AAAI 2026\] Preference is More Than Comparisons: Rethinking Dueling Bandits with Augmented Human Feedback](../../AAAI2026/recommender/preference_is_more_than_comparisons_rethinking_dueling_bandits_with_augmented_hu.md)

</div>

<!-- RELATED:END -->
