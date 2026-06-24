---
title: >-
  [Paper Note] Language Model Probabilities are Not Calibrated in Numeric Contexts
description: >-
  [ACL 2025][LLM Evaluation][Probability calibration] This paper systematically investigates the probability calibration of language models in numeric contexts. It reveals that even in simple scenarios (such as drawing marbles from a bag), all tested models, including GPT-4o, are severely miscalibrated. They exhibit systematic biases based on token order, token frequency, and token identity (e.g., some models consistently select the first option…
tags:
  - "ACL 2025"
  - "LLM Evaluation"
  - "Probability calibration"
  - "numerical reasoning"
  - "language model bias"
  - "mode collapse"
  - "systematic bias"
date: 2026-05-08
content_hash: c52aaee7f5de324e
---

# Language Model Probabilities are Not Calibrated in Numeric Contexts

**Conference**: ACL 2025  
**arXiv**: [2410.16007](https://arxiv.org/abs/2410.16007)  
**Code**: Not released  
**Authors**: Charles J. Lovering, Michael Krumdick, Viet Dac Lai, Varshini Reddy, Seth Ebner, Nilesh Kumar, Rik Koncel-Kedziorski, Chris Tanner  
**Affiliations**: Kensho Technologies, Adobe, RIT, Apple  
**Area**: LLM Evaluation  
**Keywords**: Probability calibration, numerical reasoning, language model bias, mode collapse, systematic bias

## TL;DR

This paper systematically investigates the probability calibration of language models in numeric contexts. It reveals that even in simple scenarios (such as drawing marbles from a bag), all tested models, including GPT-4o, are severely miscalibrated. They exhibit systematic biases based on token order, token frequency, and token identity (e.g., some models consistently select the first option, while others select the second). Furthermore, instruction tuning is found to exacerbate mode collapse.

## Background & Motivation

**Problem Definition**: Some texts have deterministic unique continuations (e.g., "The Eiffel Tower is in [Paris]"), while others have natural probability distributions (e.g., "The coin flip landed on [heads/tails]"). Ideally, the output probability of a language model should match the implicit numerical information in the context.

**Why It Matters**:
   - If a bag contains 98 blue marbles and 99 red marbles, the model should output "red" with a probability of approximately 50.2%.
   - Miscalibration may not significantly impact a single interaction, but it can cause systematic harm across large user populations or repeated use.
   - Typical scenarios: In recommendation systems, a model might consistently recommend the same restaurant due to irrelevant factors like the restaurant's name; in medical diagnosis, miscalibration could lead to incorrect advice.

**Pre-training Data Bias**: Numbers appear with different frequencies in pre-training datasets (e.g., numbers ending in 5 are more frequent than those ending in 7), which can introduce bias regarding different numeric values.

**Relationship with Mathematical Ability**: Poor mathematical reasoning implies substandard representation and utilization of numbers, which is a prerequisite for calibration.

## Method

### Overall Architecture: Three Template Datasets

The paper introduces three template datasets to systematically test the LLM's calibration in numeric contexts:

1. **colors Dataset** (165k questions): Randomly drawing a marble from $N_1$ marbles of one color and $N_2$ marbles of another color. The model must distribute probabilities between the two colors. It uses 5 templates, 3 number ranges (1-10, 10-100, 100-999), and 110 color combinations.
2. **wordproblems Dataset** (33.6k questions): More natural scenarios (e.g., "There are 17 spruces and 99 cedars in the forest. Which tree was struck by lightning?"), featuring 10 templates and 4-10 pairs of options.
3. **distributions Dataset** (4.5k questions): Sampling integers from a uniform distribution (e.g., "sampling from the interval [2,5)"), with 5 templates and 320 pairs of defining interval numbers.

### Key Designs: Measurement of Probability Mass

For each question instance, the context $C$ has a set of valid continuation tokens $T = \{t_1, t_2, ..., t_n\}$, where each token corresponds to an ideal probability $p_i$. The model's output probability $\pi_i$ is obtained by summing the probabilities of common tokenization variants (e.g., casing, spaces).

### Evaluation Metrics

1. **Probability Mass (PM)**: The total probability mass allocated to valid tokens $PM(T) = \sum \pi_t$. A high PM indicates that the model understands the task and assigns probabilities to valid options.
2. **Wasserstein Distance (WD)**: Measures the distance between the model's probability distribution and the ideal calibrated distribution. Lower WD indicates better calibration.
3. **Relative Entropy (RE)**: The difference in entropy between the model's distribution and the ideal distribution $RE = H(\Pi) - H(P)$. $RE < 0$ indicates overconfidence/concentration (mode collapse), while $RE > 0$ indicates underconfidence/dispersion.

### Reference Behavior Classification

The paper defines 6 reference behaviors to describe the models' systematic bias patterns:

- **Null**: PM is close to zero.
- **Calibrated**: The ideal case $\Pi = P$.
- **Pick Higher**: Consistently choosing the option with the larger quantity.
- **Pick Lower**: Consistently choosing the option with the smaller quantity.
- **Pick First**: Consistently choosing the first option that appears in the prompt.
- **Pick Second**: Consistently choosing the second option that appears.

### Evaluated Models

- **Open-source Models** (Base + Chat versions): Mistral-7B-v0.1/v0.3, Mixtral-8x7B, Yi-1.5-9B/34B, Llama-3.1-8B, gemma-2-9b/27b
- **Closed-source Models**: gpt-3.5, gpt-4-turbo, gpt-4o-mini, gpt-4o

## Main Results

### Main Results 1: Probability Mass

| Model | colors (Base→Chat) | wordproblems (Base→Chat) | distributions (Base→Chat) |
|------|:---:|:---:|:---:|
| Llama-3.1-8B | 0.38 → 0.80 | 0.54 → 0.86 | 0.82 → 0.78 |
| Mixtral-8x7B | 0.36 → 0.99 | 0.57 → 0.97 | 0.96 → 1.00 |
| gemma-2-27b | 0.54 → 1.00 | 0.59 → 1.00 | 0.96 → 1.00 |
| gpt-4o | - → 1.00 | - → 0.60 | - → 0.95 |

**Key Findings**: The PM of instruction-tuned models is statistically significantly higher than that of their Base versions (probability concentrates on valid options), indicating that the models understand the task. However, a high PM is only a prerequisite for calibration, not a guarantee of good calibration.

### Main Results 2: Calibration Results (WD)

| Model | colors | wordproblems | distributions |
|------|:---:|:---:|:---:|
| Pick Higher Baseline | 0.47 | 0.44 | - |
| Pick Higher(p=0.7) Baseline | **0.15** | **0.17** | - |
| Llama-3.1-8B | 0.40 | 0.48 | 0.43 |
| gemma-2-27b | 0.40 | 0.48 | 0.59 |
| gpt-4o-mini | 0.40 | 0.57 | 0.57 |
| gpt-4o | 0.40 | 0.57 | 0.49 |

**Core Conclusion**: **All models calibrate poorly**. A simple "Pick Higher (p=0.7)" baseline (assigning 70% probability to the option with the larger number) outperforms all models. This suggests that while models can identify valid options, they fail to distribute probabilities correctly among them.

### Main Results 3: Relative Entropy

- The RE of all models across all datasets is **statistically significantly below calibrated levels**.
- Instruction tuning leads to a **sharp drop** in entropy, with average decreases of 0.50/0.36/1.19 bits across the three datasets.
- This implies that instruction-tuned models retain only 47%/42%/55% of the entropy of the ideal calibrated distribution—marking **mode collapse**.
- The best-calibrated models (gpt-*) also have the lowest relative entropy, indicating that no model is close to being well-calibrated.

### Ablation Study: Impact of Option Identity and Order

The paper conducts a detailed analysis of option pairs in the colors dataset (using gpt-4o-mini as an example):

1. **Diagonal Asymmetry**: Listing color A as the first or second option significantly changes model behavior. For example, when "white" is listed first, the model tends to "Pick Higher"; when listed second, it tends to "Pick First".
2. **Token Identity Impact**: Different color words trigger diverse bias patterns. Certain color pairs (e.g., purple-white) prompt the model to choose the first option almost 100% of the time.
3. **Word Frequency Effect**: Differences in the frequency of numbers and colors in the pre-training corpus impact calibration performance.

### Key Findings Summary

1. Models can identify valid options (high PM) but **fail to distribute probabilities correctly among them**.
2. While instruction tuning improves PM, it leads to **severe mode collapse** (over-concentration on a single option).
3. Different models exhibit different systematic biases: gpt-4o-mini tends to pick the first option, whereas Llama-3.1-8B tends to pick the second.
4. The **token identity** of the options (which specific color word is used) and the **word order** (which color appears first) significantly impact the direction of the bias.
5. Models fail to achieve basic calibration even in extremely simple numerical reasoning scenarios.

## Highlights & Insights

1. **Underestimated Importance of the Problem**: Calibration bias is barely noticeable in single interactions but can cause systematic unfairness in large-scale applications (e.g., recommendation systems, medical diagnosis).
2. **Simple and Powerful Experimental Design**: Using the simplest probability scenario (marble drawing) exposes the fundamental flaws of LLMs without requiring complex mathematical reasoning.
3. **Revealing the Side Effects of Instruction Tuning**: RLHF/SFT makes models "more confident" but not "more correct", which is the root cause of mode collapse.
4. **Discovery of Systematic Biases**: Biases are not random; they exhibit predictable, systematic patterns related to word order and token identity.
5. **Reference Behavior Classification**: Abstracting model behavior into 6 reference behaviors provides a clear analytical tool.

## Limitations & Future Work

1. Tested numbers only up to 999; behavior on larger numbers remains unexplored.
2. Did not test whether prompting strategies like Chain-of-Thought can improve calibration.
3. The datasets only cover simple ratios and probabilities, leaving complex concepts like Bayes' theorem and conditional independence unexamined.
4. The method of accumulating probability mass (summing over tokenization variants) is an imperfect approximation.
5. Mitigation strategies for the calibration issue were not explored.

## Related Work & Insights

- **Predictive Calibration**: Guo et al. (2017) study the alignment of confidence and accuracy; Wei et al. (2024) report good calibration in GPT, but Phan et al. (2025) find the opposite.
- **Linguistic Calibration**: Yona et al. (2024) find that models struggle to express internal uncertainty in text; Kumar et al. (2024) measure consistency between internal (logits) and external (Likert scale) confidence.
- **Simulating Randomness**: Van Koevering & Kleinberg (2024) find that LLMs simulate coin flips biased toward heads and options mentioned first in the prompt.
- **Position Bias**: Pezeshkpour & Hruschka (2024) show that LLMs favor options in specific positions in multiple-choice questions.

## Rating ⭐⭐⭐⭐

- **Novelty**: ⭐⭐⭐⭐ First to systematically study the problem of probability calibration in numeric contexts, with a clear problem definition.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Three datasets, 16 models, multiple evaluation metrics, and detailed bias analysis.
- **Value**: ⭐⭐⭐⭐ Exposes the fundamental flaws of LLMs in probability reasoning, serving as an important warning for downstream applications.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure, excellent visualization, and intuitive "Reference Behavior" classification.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] La Leaderboard: A Large Language Model Leaderboard for Spanish Varieties and Languages of Spain and Latin America](la_leaderboard_spanish.md)
- [\[ICML 2025\] Communicating Activations Between Language Model Agents](../../ICML2025/llm_evaluation/communicating_activations_between_language_model_agents.md)
- [\[NeurIPS 2025\] Bayesian Evaluation of Large Language Model Behavior](../../NeurIPS2025/llm_evaluation/bayesian_evaluation_of_large_language_model_behavior.md)
- [\[ICLR 2026\] Rewarding Doubt: A Reinforcement Learning Approach to Calibrated Confidence Expression of Large Language Models](../../ICLR2026/llm_evaluation/rewarding_doubt_a_reinforcement_learning_approach_to_calibrated_confidence_expre.md)
- [\[ICLR 2026\] Pitfalls in Evaluating Language Model Forecasters](../../ICLR2026/llm_evaluation/pitfalls_in_evaluating_language_model_forecasters.md)

</div>

<!-- RELATED:END -->
