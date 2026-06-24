---
title: >-
  [Paper Note] DataDecide: How to Predict Best Pretraining Data with Small Experiments
description: >-
  [LLM Evaluation] > This work constructs DataDecide—the largest open model suite to date (25 data recipes $\times$ 14 model scales $\times$ 3 random seeds)—to systematically study how small-scale experiments can predict the best pretraining data. The study reveals that a single small-scale ranking (e.g., at 150M parameters) achieves approximately 80% pairwise decision accuracy, and continuous likelihood proxy metrics require only 0.01% of the target compute to reach over 80% p…
tags:
  - "LLM Evaluation"
date: 2026-05-08
content_hash: 359d38038a1b168d
---

# DataDecide: How to Predict Best Pretraining Data with Small Experiments

| Attribute | Content |
|------|------|
| **Conference** | ICML 2025 |
| **arXiv** | [2504.11393](https://arxiv.org/abs/2504.11393) |
| **Code** | [HuggingFace DataDecide Collection](https://huggingface.co/collections/datadecide) |
| **Area** | Data Selection |
| **Keywords** | pretraining data selection, Scaling Laws, small-scale experiments, decision accuracy, proxy metrics |

## TL;DR

> This work constructs DataDecide—the largest open model suite to date (25 data recipes $\times$ 14 model scales $\times$ 3 random seeds)—to systematically study how small-scale experiments can predict the best pretraining data. The study reveals that a single small-scale ranking (e.g., at 150M parameters) achieves approximately 80% pairwise decision accuracy, and continuous likelihood proxy metrics require only 0.01% of the target compute to reach over 80% prediction accuracy across multiple benchmarks.

## Background & Motivation

### Core Problem

Large Language Model (LLM) pretraining is extremely costly, making full-scale training on different datasets to select the best data impractical. In practice, researchers typically rely on small-scale experiments for data decisions. The critical question remains: **Which benchmarks and decision methods can most accurately predict the optimal data for large models from small-scale experiments?**

### Limitations of Prior Work

**Lack of counterfactual validation**: Existing methods only indirectly validate decision correctness by outputting a single "well-performing" large model, failing to observe what would happen if alternative data were chosen.

**Insufficient data recipe coverage**: Prior works consider at most 2 (Pythia) or 6 (Paloma, Brandfonbrener et al.) data recipes, which is insufficient for a systematic evaluation of decision methods.

**Inadequate Scaling Law validation**: Although existing scaling law methods can reduce prediction errors, they lack evaluations on the mapping from prediction errors to actual decision accuracy.

**Mismatch between evaluation metrics and scales**: Discrete accuracy metrics perform unstably on small-scale models, which may compromise prediction quality.

### Design Motivation

To make "data selection decision" research quantifiable and reproducible, the authors construct the DataDecide model suite. Controlled pretraining experiments are conducted across 25 data recipes, covering scales from 4M to 1B parameters with 3 random seeds per configuration. This registers over 30K model checkpoints in total, all of which are open-sourced.

## Method

### Overall Architecture

The core mechanism of DataDecide can be summarized in three steps:

1. **Construct a large-scale controlled model suite**: Fix the architecture, optimizer, and hyperparameters, while varying only the data recipes and model scales.
2. **Define prediction tasks**: For each pair of data recipes $(A, B)$, predict which performs better at the target scale (1B).
3. **Measure prediction accuracy**: Use Decision Accuracy to evaluate the correctness of the prediction method across all recipe pairs.

### DataDecide Model Suite

- **Data Recipes**: 25 types, covering major open-source corpora such as Dolma 1.7, DCLM-Baseline, FineWeb, Falcon/RefinedWeb, C4, along with different deduplication, filtering, domain ablation, and mixing strategies.
- **Model Scales**: 14 scales, from 4M (3.7M parameters) to 1B (1176.8M parameters), leveraging OLMo's model ladder framework to automatically configure hyperparameters.
- **Token-to-Parameter Ratio**: Fixed at 100 (i.e., $5 \times$ Chinchilla optimal ratio), reflecting current mainstream overtraining strategies.
- **Random Seeds**: 3 seeds per configuration; all seeds for 1B models are fully trained, while other scales are early-stopped after 25% of the compute on the 2nd and 3rd seeds.
- **Total Models**: $25 \times 14 \times 3 = 1{,}050$ models.

### Prediction Methods

#### Method 1: Single Scale Ranking

Train models for all 25 data recipes at a fixed small scale (e.g., 150M parameters), and directly use the downstream performance ranking of the small models as the predicted ranking for the large models.

$$\hat{y}_A > \hat{y}_B \iff \text{Acc}_{\text{small}}(A) > \text{Acc}_{\text{small}}(B)$$

#### Method 2: Multi-Scale Extrapolation (Scaling Law Extrapolation)

Train models at multiple small scales, fit scaling law curves, and extrapolate to the target scale's performance. The two-step approach of Bhagia et al. (2024) is adopted:

**Step 1 - Compute-to-Loss Mapping:**

$$L(C) = \frac{A}{C^{\alpha}} + E$$

where $A, \alpha, E$ are parameters to be optimized, and $C = 6ND$ (theoretical FLOPs).

**Step 2 - Loss-to-Accuracy Mapping:**

$$\text{Acc}(L) = \frac{a}{1 + e^{-k(L - L_0)}} + b$$

where $a, b, k, L_0$ are parameters to be optimized. A total of 8 scaling law variants were tested (2-parameter, 3-parameter, 5-parameter, single-step fitting, helper point injection, filtering early checkpoints, etc.).

### Evaluation System

#### Decision Accuracy

The core evaluation metric. For all pairs of data recipes $(A, B)$, it measures whether the prediction correctly identifies the winner at the target scale:

$$\text{Decision Accuracy} = \frac{1}{|\mathcal{P}|} \sum_{(A,B) \in \mathcal{P}} \mathbb{I}\big(\text{sign}(\hat{y}_A - \hat{y}_B) = \text{sign}(y_A - y_B)\big)$$

The "gold standard" ranking at the target scale is based on the average performance across 3 random seeds.

#### Compute Budget Ratio (%C)

$$\%C = \frac{c}{C} \times 100\%$$

where $c$ represents the FLOPs of the prediction experiments, and $C$ represents the FLOPs of the target scale.

#### Downstream Evaluation

Evaluations utilize 10 multiple-choice benchmarks from the OLMES framework: MMLU, HellaSwag, ARC Challenge, ARC Easy, PIQA, CommonsenseQA, SocialIQA, OpenBookQA, BoolQ, and WinoGrande.

### Proxy Metrics

To mitigate the instability of discrete accuracy at small scales, five continuous proxy metrics are introduced:

| Metric | Definition |
|------|------|
| **Correct Prob** | Average probability of the correct option |
| **Margin** | Difference between the probability of the correct option and the highest incorrect option |
| **Norm Correct Prob** | Proportion of the correct option's probability relative to the sum of all options' probabilities |
| **Total Prob** | Sum of probabilities of all options (both correct and incorrect) |
| **Accuracy** | Proportion where the correct option has the highest probability (discrete) |

Each metric has both `per_token` and `per_char` length-normalization variants; experiments show that `per_char` performs best on most tasks.

## Key Experimental Results

### Main Results: Compute Budget vs. Decision Accuracy

**Key Findings** (Figure 1):

- On the OLMES 10-task aggregated metric, the compute budget and decision accuracy exhibit an approximately log-linear relationship.
- A small model with 150M parameters can achieve approximately **80%** pairwise decision accuracy.
- Utilizing continuous likelihood metrics requires only **0.01%** of the target compute budget to achieve $>80\%$ decision accuracy on benchmarks such as MMLU, ARC, HellaSwag, MBPP, and HumanEval.

### Prediction Difficulty of Different Tasks (Figure 2)

| Task | Prediction Difficulty | Features |
|------|----------|------|
| ARC Easy | Extremely easy to predict | High decision accuracy achieved with minimal compute |
| MMLU | Relatively easy to predict | Low variance across runs |
| ARC Challenge | Moderate | Wide performance distribution across data recipes |
| HellaSwag | Relatively difficult | Requires more compute before effective prediction begins |
| SocialIQA, WinoGrande | Difficult | Shows a distinct insensitive period followed by log-linear growth |
| BoolQ | Extremely difficult | Has non-trivial decision accuracy only at intermediate checkpoints near target compute |

### Scaling Law Comparison (Figure 3)

**Key Conclusion**: None of the 8 scaling law variants outperform the compute-decision accuracy frontier of single-scale ranking.

| Scaling Law Variant | Relative Error | Absolute Error |
|------------------|----------|----------|
| 3-parameter + helpers + >50% checkpoints | 5.6 | 2.6 |
| 3-parameter + helpers | 6.0 | 2.8 |
| 3-parameter | 6.5 | 3.1 |
| 2-parameter | 6.5 | 3.2 |
| 5-parameter single-step | 42.8 | 17.4 |
| 5-parameter | 230.8 | 65.4 |

The 2-parameter and 3-parameter variants yield similar and optimal prediction errors, while the 5-parameter variant degrades significantly due to overfitting.

### Proxy Metrics Experiment (Figure 4)

**Key Findings**:

- **Correct Prob and Total Prob** provide the best or equivalent decision accuracy at small scales (0.01% to 1% of target compute).
- On 5 tasks including MMLU, ARC Easy, and PIQA, continuous proxy metrics significantly outperform discrete Accuracy at small scales.
- As the target scale is approached (the final order of magnitude), Accuracy and continuous metrics that penalize incorrect answers (Norm Correct Prob, Margin) surpass Correct Prob and Total Prob.
- **Breakthrough in Code Tasks** (Figure 6): HumanEval and MBPP, which are nearly unpredictable when using Accuracy, see their decision accuracy rise to approximately 80% when switching to Correct Prob.

### Two Key Factors of Predictability (Figure 5)

Decision accuracy depends on:
1. **Inter-run Variance (Noise)**: The lower the standard deviation of performance across 3 random seeds, the more accurate the prediction.
2. **Performance Distribution Across Recipes (Spread)**: The larger the performance difference among different data recipes, the easier they are to distinguish.

For example, MMLU is easy to predict primarily due to low inter-run variance; ARC Easy is easy to predict because the performance distribution across various data recipes is very wide. The advantage of the Correct Prob proxy metric often manifests in improving at least one of these two features.

## Highlights & Insights

1. **Counter-intuitive Finding**: Simple single-scale ranking achieves approximately 80% decision accuracy. Complex scaling law extrapolation methods are not only more computationally expensive but also fail to yield significant improvements. This challenges the intuition of "more complex is better."
2. **Value of Proxy Metrics**: Continuous likelihood metrics can render certain tasks that are otherwise unpredictable at small scales (such as code generation) predictable. The core reason is that they reduce evaluation noise or increase the performance spread among recipes.
3. **Clear Practical Recommendations**: (a) Select highly predictable benchmarks (e.g., MMLU, ARC) for small-scale data decisions; (b) Use Correct Prob as the proxy metric; (c) The 150M parameter model offers the most cost-effective prediction scale.
4. **Open Ecosystem Construction**: Releasing 30K+ checkpoints, complete models of 25 data recipes, and all evaluation results allows the community to run new evaluations and experiment with new prediction methods at zero cost.
5. **Novel Decision Accuracy Metric**: Distinct from traditional scaling law works that focus on the absolute magnitude of prediction errors, DataDecide directly measures "whether the prediction can correctly make a pairwise decision," which aligns better with practical application needs.

## Limitations & Future Work

1. **Fixed Token-to-Parameter Ratio**: Only the 100:1 ratio ($5 \times$ Chinchilla optimal) is evaluated. Although this aligns with current mainstream practices, it does not cover all scenarios.
2. **Limited Target Scale**: The maximum target scale is 1B parameters, failing to directly validate whether the findings generalize to larger models (e.g., 7B, 70B).
3. **Single Model Architecture**: The OLMo architecture is fixed throughout, leaving the impact of architectural variations on data selection decisions untested.
4. **Limited Evaluation Tasks**: Only 10 OLMES multiple-choice benchmarks are used. Although the paper demonstrates feasibility on code tasks, scalability to mathematical reasoning tasks remains questionable (which remain difficult to predict even with alternative proxy metrics).
5. **Potentially Sub-optimal Scaling Law Implementations**: The 8 scaling law variants are straightforward baseline implementations; more meticulously designed fitting methods might yield better performance.

## Related Work & Insights

- **Scaling Law Prediction**: Kaplan et al. (2020) and Hoffmann et al. (2022) established the foundations of how language model loss scales with compute; Gadre et al. (2024) and Bhagia et al. (2024) extended this to downstream performance prediction; Choshen et al. (2024) provided a practical guide to scaling law estimation.
- **Data Selection Suites**: Pythia (Biderman et al., 2023) first provided controlled comparisons of 2 data recipes; Paloma (Magnusson et al., 2024) extended this to 6 recipes; DCLM (Li et al., 2024) heavily leveraged single-scale ranking but did not open-source ablation models.
- **Data Mixture Optimization**: Kang et al. (2024) and Ye et al. (2024) optimized data source mixture proportions via scaling laws, which represents a special case of the DataDecide application scope.
- **Emergent Abilities and Metric Selection**: Schaeffer et al. (2023) pointed out that discrete metrics can lead to "illusions of emergent abilities" and that continuous metrics better capture progressive improvements—a finding that resonates with the proxy metric insights in this work.

## Rating

| Dimension | Rating |
|------|------|
| Novelty | ⭐⭐⭐⭐ |
| Rigorousness | ⭐⭐⭐⭐⭐ |
| Practicality | ⭐⭐⭐⭐⭐ |
| Clarity | ⭐⭐⭐⭐ |
| **Overall** | **⭐⭐⭐⭐** |

> This work is a solid empirical contribution that answers a highly practical question through large-scale controlled experiments. The experimental design is rigorous, the conclusions are clear and actionable, and the open-source resources are abundant. Although limitations exist such as the target scale being capped at 1B and the architectural/ratio uniformity, the methodological framework is directly generalizable, serving as a valuable reference for pretraining data selection in both industry and academia.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] The Best of Both Worlds: Bridging Quality and Diversity in Data Selection with Bipartite Graph](the_best_of_both_worlds_bridging_quality_and_diversity_in_data_selection_with_bi.md)
- [\[ACL 2025\] Retrieval Models Aren't Tool-Savvy: Benchmarking Tool Retrieval for Large Language Models](../../ACL2025/llm_evaluation/retrieval_models_arent_tool-savvy_benchmarking_tool_retrieval_for_large_language.md)
- [\[ACL 2025\] Browsing Lost Unformed Recollections: A Benchmark for Tip-of-the-Tongue Search and Reasoning](../../ACL2025/llm_evaluation/browsing_lost_unformed_recollections_a_benchmark_for_tip-of-the-tongue_search_an.md)
- [\[ACL 2025\] Atomic Calibration of LLMs in Long-Form Generations](../../ACL2025/llm_evaluation/atomic_calibration_of_llms_in_long-form_generations.md)
- [\[ACL 2025\] AndroidLab: Training and Systematic Benchmarking of Android Autonomous Agents](../../ACL2025/llm_evaluation/androidlab_autonomous_agent.md)

</div>

<!-- RELATED:END -->
