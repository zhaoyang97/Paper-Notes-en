---
title: >-
  [Paper Note] Ranking Unraveled: Recipes for LLM Rankings in Head-to-Head AI Combat
description: >-
  [ACL 2025][LLM (Other)][LLM ranking] This study systematically evaluates the performance of four ranking algorithms (Elo, Bradley-Terry, Glicko, Markov Chain) in head-to-head LLM evaluations. By defining three core ranking criteria (transitivity, prediction accuracy, and hyperparameter sensitivity), the authors reveal that the widely used Elo rating system suffers from severe deficiencies in stability and consistency, and recommend Glicko for large…
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "LLM ranking"
  - "Elo"
  - "Bradley-Terry"
  - "Glicko"
  - "Markov Chain"
  - "head-to-head evaluation"
  - "Chatbot Arena"
date: 2026-05-08
content_hash: d35a20db35c86073
---

# Ranking Unraveled: Recipes for LLM Rankings in Head-to-Head AI Combat

**Conference**: ACL 2025  
**arXiv**: [2411.14483](https://arxiv.org/abs/2411.14483)  
**Code**: Open-sourced (the paper mentions releasing all code, data, and models)  
**Authors**: Roland Daynauth, Christopher Clarke, Krisztian Flautner, Lingjia Tang, Jason Mars  
**Institution**: University of Michigan  
**Area**: LLM Evaluation / Ranking Algorithms  
**Keywords**: LLM ranking, Elo, Bradley-Terry, Glicko, Markov Chain, head-to-head evaluation, Chatbot Arena  

## TL;DR

This study systematically evaluates the performance of four ranking algorithms (Elo, Bradley-Terry, Glicko, Markov Chain) in head-to-head LLM evaluations. By defining three core ranking criteria (transitivity, prediction accuracy, and hyperparameter sensitivity), the authors reveal that the widely used Elo rating system suffers from severe deficiencies in stability and consistency, and recommend Glicko for large, uneven datasets and Bradley-Terry for small, controlled datasets.

## Background & Motivation

**LLM Evaluation Dilemma**: Traditional benchmarks (such as GLUE, SuperGLUE, LM-Eval, etc.) rely on predefined ground truth, making them unable to evaluate open-ended text generation, dialogue, and other tasks that require subjective human judgment. Research shows that the correlation between these benchmark evaluation results and human LLM evaluation is very weak.

**Rise of Pairwise Ranking**: Platforms represented by Chatbot Arena allow users to vote on responses from two models via "head-to-head battles," and then use ranking algorithms to obtain overall rankings. This approach has become the de facto standard for LLM evaluation.

**Core Problem**: Ranking algorithms like Elo were originally designed for structured competitions such as chess. Applying them to LLM evaluation faces many challenges—different algorithms can produce different rankings on the same data (as shown in Figure 1), and there is a lack of systematic research to guide algorithm selection.

## Method

### 3.1 Four Ranking Algorithms

**Elo (1960)**:
- Derived from chess, calculating win rates using a logistic function: $p_{ij} = 1/(1+10^{(\theta_j-\theta_i)/400})$
- Ratings are updated sequentially after each match: $\theta_i' = \theta_i + k \times (S_{ij} - p_{ij})$
- Key hyperparameter: k-factor (determines the magnitude of the impact of a single match on ratings)

**Bradley-Terry (1952)**:
- A probabilistic model that simultaneously calculates the strength parameters of all models using Maximum Likelihood Estimation (MLE).
- Computed concurrently based on all match results rather than updating match-by-match.
- No hyperparameter tuning required.

**Glicko (1995)**:
- An improved version of Elo, introducing the Rating Deviation (σ) parameter to measure the reliability of ratings.
- σ acts as a confidence interval and is dynamically adjusted based on the number of matches—rating updates for new models (with fewer matches) are more conservative.
- Suitable for handling environments where new models are frequently added.

**Markov Chain**:
- A non-parametric ranking algorithm that treats models as graph nodes and matches as edges.
- A random walker moves to the winner node with probability p, and the total vote count represents the ranking.
- The hyperparameter p controls the degree to which win rate affects ranking (default p=0.8).

### 3.2 Three Core Ranking Criteria

1. **Transitivity**: If A beats B and B beats C, then the ranking should show A > B > C. Ranking algorithms should maintain the transitive structure in the data as much as possible.
2. **Prediction Accuracy**: The ability of the ranking system to correctly predict unseen match outcomes, measuring its alignment with human preferences.
3. **Sensitivity**: The stability of ranking results against hyperparameter variations. A system that is too sensitive can lead to drastic ranking changes with minor parameter updates.

### 3.3 Two Evaluation Scenarios

- **Arena Style**: Represented by Chatbot Arena, with 57 models and 244,978 matches. The match distribution is highly uneven (maximum of 30,416 matches vs. minimum of 954 matches).
- **Controlled Style**: Represented by SLAM, with 11 models and 2,858 matches. The number of matches per model is uniform (501–529 matches).

## Key Experimental Results

### Transitivity Preservation

| Algorithm | Arena (%) | SLAM (%) |
|------|-----------|----------|
| Elo | 68.24 | 52.5 |
| Markov | 51.38 | 51.67 |
| Glicko | 56.54 | 53.33 |
| **Bradley-Terry** | **77.29** | **56.67** |

**Analysis**: Bradley-Terry achieves the best consistency because MLE considers all match results simultaneously, making it independent of order. Elo's match-by-match updates lead to order-sensitivity.

### Prediction Accuracy (F1 Score)

| Algorithm | Arena (F1) | SLAM (F1) |
|------|-----------|----------|
| **Elo** | **0.90** | 0.87 |
| Markov | 0.77 | 0.88 |
| Glicko | 0.88 | 0.88 |
| Bradley-Terry | 0.82 | 0.87 |

**Analysis**:
- Elo predicts most accurately on the Arena dataset—sequential updates adapt better to uneven distributions.
- Markov performs worst on Arena—severely affected by sparse matches.
- Bradley-Terry is affected by the "strong model" problem on Arena (e.g., the win-loss ratio of gpt-4-turbo is 12288/3979, leading to an overestimation of strength, which is the "rare events" problem in logistic regression).
- All algorithms perform similarly on SLAM—the uniform distribution eliminates differences among algorithms.

### Hyperparameter Sensitivity

- **Elo**: Highly sensitive to the k-factor, showing the largest F1 distribution fluctuation across 100 different hyperparameter settings. It is particularly unstable on small-scale SLAM datasets. The optimal k-value is lower than the commonly used 32.
- **Glicko**: Consistently maintains predictive performance on both datasets and is minimally affected by hyperparameter changes—this is because a large number of matches allows the system to dynamically adjust the rating deviation.
- **Markov**: Stable on the controlled SLAM dataset but struggles on the large-scale Arena.

### Correlation Between Algorithms

| | Elo | Markov | Glicko | BT | Win-Rate |
|-|-----|--------|--------|----|----------|
| Elo | 1.00/1.00 | 0.74/0.89 | 0.86/0.93 | 0.94/0.95 | 0.93/0.91 |
| Glicko | - | 0.76/0.99 | 1.00/1.00 | 0.81/0.99 | 0.89/0.99 |
| BT | - | - | - | 1.00/1.00 | 0.91/0.98 |

(Format: Arena/SLAM Spearman correlation coefficient)

On SLAM, the correlation of all algorithms with the win rate is extremely high ($\ge 0.91$), indicating that the simple win rate is already a good ranking indicator under a uniform distribution.

## Recommended Best Practices

| Dataset Characteristics | Recommended Algorithm | Reason |
|-----------|---------|------|
| Small, uniformly distributed | **Bradley-Terry** | Best transitivity, comparable prediction accuracy, no hyperparameter tuning required |
| Large, unevenly distributed | **Glicko** | Rating deviation handles new models/low match count scenarios, insensitive to hyperparameters |
| Small, unevenly distributed | **Bradley-Terry** | No hyperparameters required, efficient in handling small data |
| Large, uniformly distributed | **Bradley-Terry** | Good scalability, highly interpretable results |
| **Not Recommended** | **Elo** | Cannot achieve stable rankings even with >1000 permutations, sensitive to hyperparameters, mediocre transitivity |

## Highlights & Insights

1. **"Demystifying" Elo**: Although Elo is the default choice for platforms like Chatbot Arena, it fails to stabilize rankings even after 1000+ permutations. This directly challenges the conclusions of previous studies suggesting that "increasing the number of permutations can resolve Elo instability."
2. **The "Rare Events" Problem of Bradley-Terry**: When a model's win rate is extremely high (e.g., gpt-4-turbo's win-loss ratio is 12288/3979), MLE overestimates the model's strength. This is a known issue in logistic regression, and weighted regression does not effectively solve it.
3. **The Power of Uniform Distribution**: All algorithms exhibit nearly equivalent performance on the SLAM dataset, suggesting that efforts spent controlling the distribution of evaluation matchups may be more important than the choice of algorithm.
4. **Practical Decision Framework**: The paper eventually provides a clear algorithm selection guide (Table 4), which is highly suited for practical application.

## Limitations & Future Work

1. **Scalability**: The number of comparisons for pairwise evaluation grows quadratically with the number of models, limiting large-scale assessment.
2. **Human Feedback Variability**: Human judgment is influenced by personal background, expertise, and context understanding, introducing ranking noise.
3. **Lack of In-depth Analysis on Ties**: The paper treats ties as a score of 0.5, but the frequency and distribution of ties may affect different algorithms differently.
4. **Only Two Datasets Considered**: The conclusions may be limited by the specific distributional characteristics of Chatbot Arena and SLAM.

## Related Work & Insights

- **LLM Evaluation Methods**: Pairwise human evaluation platforms like Chatbot Arena (Chiang et al., 2024), AlpacaFarm (Dubois et al., 2024), etc.
- **Elo Analysis**: Boubdir et al. (2023) analyzed the robustness issues of Elo in LLM ranking, but only focused on Elo itself.
- **Ranking Algorithms**: Elo (1978), Bradley-Terry (1952), Glicko (1999), Markov Chain (Callaghan et al., 2003)

## Rating ⭐⭐⭐⭐

- **Novelty**: ⭐⭐⭐⭐ — First systematic comparison of multiple ranking algorithms in LLM evaluation.
- **Value**: ⭐⭐⭐⭐⭐ — Provides a clear algorithm selection guide with direct practical significance for LLM evaluation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Two evaluation scenarios, three evaluation criteria, multidimensional analysis.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure, explicit conclusions, and highly practical recommendations.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Analyzing the Rapid Generalization of SFT via the Perspective of Attention Head Activation Patterns](analyzing_the_rapid_generalization_of_sft_via_the_perspective_of_attention_head_.md)
- [\[ACL 2025\] MHA2MLA: Towards Economical Inference by Enabling DeepSeek's Multi-Head Latent Attention in Any Transformer-based LLMs](mha2mla_deepseek_latent_attention.md)
- [\[ACL 2025\] Are Optimal Algorithms Still Optimal? Rethinking Sorting in LLM-Based Pairwise Ranking with Batching and Caching](are_optimal_algorithms_still_optimal_rethinking_sorting_in_llm-based_pairwise_ra.md)
- [\[ACL 2025\] The AI Gap: How Socioeconomic Status Affects Language Technology Interactions](the_ai_gap_how_socioeconomic_status_affects_language_technology_interactions.md)
- [\[ACL 2025\] AI as a Novel Ethical Agent: Exploring Moral Judgments by Large Language Models](ai_as_a_novel_ethical_agent_exploring_moral_judgments_by_large_language_models.md)

</div>

<!-- RELATED:END -->
