---
title: >-
  [Paper Note] Forest vs Tree: The (N, K) Trade-off in Reproducible ML Evaluation
description: >-
  [AAAI 2026][Reproducibility] This paper investigates the optimal trade-off between the number of samples $N$ and the number of annotators per sample $K$ in machine learning evaluation. Under a fixed budget $N \times K$…
tags:
  - "AAAI 2026"
  - "Reproducibility"
  - "Human Annotation"
  - "Annotator Disagreement"
  - "Evaluation Reliability"
  - "Optimal Budget Allocation"
date: 2026-05-08
content_hash: c468a7f80978739b
---

# Forest vs Tree: The (N, K) Trade-off in Reproducible ML Evaluation

**Conference**: AAAI 2026
**arXiv**: [2508.03663](https://arxiv.org/abs/2508.03663)  
**Code**: None  
**Area**: Other
**Keywords**: Reproducibility, Human Annotation, Annotator Disagreement, Evaluation Reliability, Optimal Budget Allocation

## TL;DR

This paper investigates the optimal trade-off between the number of samples $N$ and the number of annotators per sample $K$ in machine learning evaluation. Under a fixed budget $N \times K$, by analyzing multi-annotator datasets and simulated distributions, the study finds that $K > 10$ is generally optimal when annotator disagreement is considered, and the required total budget $N \times K$ typically does not exceed 1000.

## Background & Motivation

**Background**: Reproducibility is the cornerstone of scientific validation and the basis upon which results are granted authority. In machine learning evaluation, ground truth labels are typically obtained from human annotators. However, inter-annotator disagreement is prevalent, and the evaluation literature surprisingly lacks investigation into the effects of effectively ignoring such disagreement.

**Limitations of Prior Work**: (1) Budgets for collecting human-annotated evaluation data are limited—increasing the number of annotators $K$ per sample substantially raises per-item annotation costs; (2) most ML evaluations use majority voting or a single annotation as the ground truth, entirely discarding inter-annotator disagreement information; (3) systematic guidance is lacking to help practitioners decide how to allocate budgets between "more samples" and "more annotators per sample."

**Key Challenge**: Under a fixed budget, there is a fundamental trade-off between increasing the number of samples $N$ (broader coverage but fewer annotations per item) and increasing the number of annotators $K$ per item (more reliable labels but narrower coverage). Prior intuition has favored maximizing $N$, which neglects the noise introduced by annotator disagreement.

**Goal**: (1) Systematically study the $(N, K)$ trade-off; (2) identify optimal $(N, K)$ configurations under different evaluation metrics; (3) provide practical guidance to help ML practitioners optimize evaluation budget allocation.

**Key Insight**: The authors analyze multiple real-world multi-annotator datasets and simulated distributions fitted to these datasets, systematically exploring the optimal $(N, K)$ for the core task of reliably comparing the performance of two ML models.

**Core Idea**: In ML evaluation, "more annotators per sample" is often more effective than "more samples"—configurations with $K > 10$ outperform maximum-$N$ configurations with $K = 1$ in most scenarios.

## Method

### Overall Architecture

The research framework consists of three stages: (1) collecting multiple real-world multi-annotator datasets; (2) systematically experimenting with different $(N, K)$ configurations on each dataset, evaluating the reproducibility of model comparisons; (3) analyzing how optimal $(N, K)$ configurations vary with evaluation metrics and dataset characteristics.

### Key Designs

1. **Reproducibility Measurement Framework**:

    - Function: Quantifies the repeatability of evaluation conclusions (i.e., model A outperforms model B).
    - Mechanism: For a given $(N, K)$ configuration, multiple rounds of sampling are performed from the full dataset, each drawing $N$ samples with $K$ randomly selected annotators. Evaluation metrics are computed for two models on each sample and compared. Reproducibility is defined as the proportion of sampling rounds yielding the same conclusion (i.e., the same model is better). High reproducibility indicates stable and reliable evaluation conclusions.
    - Design Motivation: Significance tests commonly reported in ML papers assume deterministic labels; however, when labels are obtained from annotators with disagreements, traditional significance tests may yield irreproducible conclusions.

2. **Systematic Comparison Across Multiple Evaluation Metrics**:

    - Function: Reveals differences in sensitivity to the $(N, K)$ trade-off across different evaluation metrics.
    - Mechanism: Multiple commonly used metrics are compared—accuracy (using majority-vote labels), weighted F1, AUC, and distribution-sensitive metrics (e.g., cross-entropy, Jensen-Shannon divergence). Different metrics exhibit varying sensitivity to annotator disagreement: accuracy entirely ignores minority opinions, whereas distribution-sensitive metrics retain full disagreement information.
    - Design Motivation: The optimal $(N, K)$ is likely to vary across metrics. Distribution-sensitive metrics should theoretically benefit more from higher $K$, as they leverage richer annotator information.

3. **Dual Validation via Real Data and Simulation**:

    - Function: Ensures the robustness and generalizability of the findings.
    - Mechanism: Experiments are conducted on real multi-annotator datasets spanning multiple domains (including NLP, computer vision, and social science), while parametric simulated distributions fitted to the statistical properties of real data are used to generate large-scale data for validating patterns across a continuous parameter space.
    - Design Motivation: Real datasets are limited in number and scale; simulated data enables exploration of a broader range of conditions. Using simulation alone may not reflect real-world scenarios, so combining both approaches is most reliable.

### Loss & Training

Not applicable (this paper is a study in evaluation methodology and does not involve model training).

## Key Experimental Results

### Main Results

| Finding | Details | Notes |
|---------|---------|-------|
| Optimal $K$ | $K > 10$ is almost always superior | Across nearly all datasets and metrics |
| Total budget $N \times K$ | ≤ 1000 is generally sufficient | Often well below 1000 in most cases |
| Metric dependence | Distribution-sensitive metrics benefit more from high $K$ | Accuracy is less sensitive to $K$ |
| Existence of $(N, K)$ trade-off | Depends on the evaluation metric | A genuine trade-off exists under certain metrics |

### Ablation Study

| Configuration | Reproducibility | Notes |
|---------------|----------------|-------|
| High $N$, low $K$ (e.g., N=500, K=2) | Moderate | Many samples but high label noise |
| Low $N$, high $K$ (e.g., N=50, K=20) | High | More reliable labels compensate for fewer samples |
| Balanced configuration (e.g., N=100, K=10) | Optimal or near-optimal | Close to optimal in most scenarios |
| Extreme $K=1$ (majority vote) | Lowest | Ignoring disagreement leads to unstable conclusions |

### Key Findings

- **Core Finding**: When human annotator disagreement is considered, the optimal budget allocation requires a total budget $N \times K$ of no more than 1000 (often considerably less), and the optimum is almost always achieved at $K > 10$.
- Distribution-sensitive metrics (e.g., cross-entropy) benefit more from high $K$ than accuracy-based metrics, as they can exploit the full annotator distribution information.
- The nature of the $(N, K)$ trade-off—and even whether it exists—depends on the evaluation metric: some metrics exhibit a monotonic preference for higher $K$, while others have a genuine optimal balance point.
- The proposed methodology can be directly applied by ML practitioners to plan evaluation data collection budgets.

## Highlights & Insights

- **High Practical Value**: The direct output of this paper is a methodology to help ML researchers make more informed budget decisions when collecting evaluation data.
- **Challenges the "Maximize N" Intuition**: Conventional wisdom holds that more samples are always better; this paper demonstrates that in many scenarios, increasing the number of annotators is more effective than increasing the number of samples.
- **Annotator Disagreement as Signal, Not Noise**: The paper implicitly argues for the importance of retaining annotator disagreement information—rather than collapsing it into majority votes—for reliable evaluation.

## Limitations & Future Work

- The study focuses on label annotation for classification tasks; applicability to regression, ranking, and other evaluation settings remains to be validated.
- Annotator quality is assumed to be uniform; the influence of annotator capability heterogeneity on the optimal $(N, K)$ is not considered.
- The scenario in which annotator disagreement is modeled as data uncertainty to improve model training (rather than evaluation alone) is not explored.
- Integration with active learning frameworks could be investigated—if certain samples exhibit greater annotator disagreement, it may be appropriate to allocate more annotators to those samples.

## Related Work & Insights

- **vs. Standard Evaluation Practice**: The standard practice is $K=1$ (or $K=3$ with majority voting); the analysis presented here shows this is far from optimal.
- **vs. Annotation Aggregation Methods (e.g., Dawid-Skene)**: Annotation aggregation methods address the optimal aggregation of given annotations, whereas this paper focuses on the upstream question of "how many annotations should be collected."
- **vs. Statistical Power Analysis**: Classical power analysis assumes deterministic labels; this paper incorporates annotation uncertainty into the analysis, more closely reflecting the realities of ML evaluation.

## Rating

- Novelty: ⭐⭐⭐⭐ The systematic study of the $(N, K)$ trade-off offers a novel perspective that challenges mainstream intuition
- Experimental Thoroughness: ⭐⭐⭐⭐ Dual validation via multiple real datasets and simulation, covering diverse metrics
- Writing Quality: ⭐⭐⭐⭐ Problem formulation is clear and the analysis is rigorous
- Value: ⭐⭐⭐⭐⭐ Offers direct and broad practical guidance for ML evaluation

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] On the Complexity-Faithfulness Trade-off of Gradient-Based Explanations](../../ICCV2025/others/on_the_complexity-faithfulness_trade-off_of_gradient-based_explanations.md)
- [\[AAAI 2026\] MicroEvoEval: A Systematic Evaluation Framework for Image-Based Microstructure Evolution Prediction](microevoeval_a_systematic_evaluation_framework_for_image-based_microstructure_ev.md)
- [\[AAAI 2026\] Extreme Value Monte Carlo Tree Search for Classical Planning](extreme_value_monte_carlo_tree_search_for_classical_planning.md)
- [\[AAAI 2026\] DFDT: Dynamic Fast Decision Tree for IoT Data Stream Mining on Edge Devices](dfdt_dynamic_fast_decision_tree_for_iot_data_stream_mining_on_edge_devices.md)
- [\[ICML 2026\] Over-Alignment vs Over-Fitting: The Role of Feature Learning Strength in Generalization](../../ICML2026/others/over-alignment_vs_over-fitting_the_role_of_feature_learning_strength_in_generali.md)

</div>

<!-- RELATED:END -->
