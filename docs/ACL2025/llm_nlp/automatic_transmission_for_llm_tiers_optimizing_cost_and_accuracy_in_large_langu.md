---
title: >-
  [Paper Note] LLM-AT: Automatic Transmission for LLM Tiers Optimizing Cost and Accuracy
description: >-
  [ACL 2025][LLM (Other)][LLM Tiers] This paper proposes the LLM-AT framework, a training-free iterative pipeline consisting of Starter (an accuracy estimator based on historical inference logs that selects the initial LLM tier) $\rightarrow$ Generator (generates answers) $\rightarrow$ Judge (evaluates validity, automatically upgrading to a higher tier if invalid). On the MATH dataset, LLM-AT achieves near-o1 accuracy at only 59.37% of the cost of a single o1 inference…
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "LLM Tiers"
  - "Model Selection"
  - "Cost Optimization"
  - "Automatic Upgrade"
  - "Accuracy Estimation"
  - "Training-Free Routing"
date: 2026-05-08
content_hash: 2448cb5f570c09b4
---

# LLM-AT: Automatic Transmission for LLM Tiers Optimizing Cost and Accuracy

**Conference**: ACL 2025  
**arXiv**: [2505.20921](https://arxiv.org/abs/2505.20921)  
**Code**: [GitHub](https://github.com/hyudsl/LLM-AT)  
**Area**: LLM/NLP  
**Keywords**: LLM Tiers, Model Selection, Cost Optimization, Automatic Upgrade, Accuracy Estimation, Training-Free Routing

## TL;DR

This paper proposes the LLM-AT framework, a training-free iterative pipeline consisting of Starter (an accuracy estimator based on historical inference logs that selects the initial LLM tier) $\rightarrow$ Generator (generates answers) $\rightarrow$ Judge (evaluates validity, automatically upgrading to a higher tier if invalid). On the MATH dataset, LLM-AT achieves near-o1 accuracy at only 59.37% of the cost of a single o1 inference, and achieves comparable performance on MCQA at only 12% of the o1 cost.

## Background & Motivation

**Background**: LLM providers (such as OpenAI) offer multiple tiers of models (o1, o1-mini, GPT-4o, GPT-4o-mini), where higher-tier models perform better but are more expensive. As NLP tasks become increasingly complex and modular (e.g., Tree of Thoughts requiring dozens of LLM calls), selecting the appropriate model tier for each subtask has become a critical challenge.

**Limitations of Prior Work**: (1) Existing LLM routing methods (Ding et al. 2024; Ong et al. 2024) require a large amount of annotated data to train routers, which incurs high coloring/labeling costs; (2) they require retraining whenever new models are released; (3) they exhibit poor out-of-distribution generalization; (4) simple cascading approaches (Chen et al. 2024a) always start from the lowest tier, wasting substantial calls on complex problems.

**Key Challenge**: A training-free tier selection method is needed—one that can skip unnecessary low-tier attempts while saving costs when high tiers are not required.

**Goal**: Design a training-free automatic LLM tier selection framework that achieves Pareto optimality in accuracy and cost.

**Key Insight**: Adopt an analogy to automatic transmissions in cars—automatically selecting the appropriate gear (model tier) based on task difficulty (road conditions), combined with self-verification and automatic upgrade mechanisms.

**Core Idea**: Build an accuracy estimator based on historical inference records to select the initial tier, and automatically verify and upgrade using a Judge to realize training-free cost-accuracy optimization.

## Method

### Overall Architecture

LLM-AT consists of three modules: (1) Starter—estimates the accuracy of each tier on the current problem and selects the lowest-cost tier meeting the threshold as the starting point; (2) Generator—generates answers using the selected tier (via CoT or PoT prompting); (3) Judge—uses a same-tier LLM to evaluate answer validity. If judged invalid, the system automatically upshifts to the next higher tier to regenerate and verify, repeating this until a valid answer is obtained or the highest tier is reached.

### Key Designs

1. **Accuracy Estimator**:

    - **Function**: Estimates the accuracy of each tier for the current problem without labeled data.
    - **Mechanism**: Maintains a History database recording historical inference results. For a new query $q$, the system retrieves the top-k most similar historical queries, calculates the ratio of correct/incorrect pseudo-labels for each tier, and uses Bayesian smoothing to estimate the accuracy: $P_j(q) = \frac{n_j^T + \alpha^T}{n_j^T + n_j^F + \alpha^T + \alpha^F}$, where $\alpha^T = \lambda \cdot Acc^{Bench}$ and $\alpha^F = \lambda \cdot (1 - Acc^{Bench})$ represent benchmark-based priors.
    - **Design Motivation**: Simply relying on public benchmark scores cannot capture the difficulty variance of individual queries, whereas statistics based on similar queries can capture local difficulty features.

2. **Pseudo-labeling**:

    - **Function**: Generates correctness labels for historical inference logs without human annotation.
    - **Mechanism**: (a) If judged valid by the Judge, it is labeled as correct; (b) if a certain tier is valid, all higher tiers are also labeled as correct; (c) lower tiers producing the identical answer are labeled as correct, and those with different answers are labeled as incorrect; (d) tiers skipped by the Starter remain empty.
    - **Design Motivation**: Leverages the verification results of the Judge and the assumption of performance monotonicity across tiers to accumulate labels without human annotation.

3. **Similarity-Weighted Accuracy Estimation**:

    - **Function**: Ensures more similar historical queries contribute more to the estimation.
    - **Mechanism**: $n_j^T = \sum_{q' \in \text{top}(q)} \text{sim}(q, q') \cdot \mathbb{1}(l_{j,q'} \text{ is correct})$, utilizing cosine similarity as the weight instead of a simple count.
    - **Design Motivation**: Highly similar historical queries better reflect the difficulty and type characteristics of the current query.

4. **Judge Module and Special Configurations**:

    - **Function**: Evaluates the validity of the Generator's answers.
    - **Mechanism**: Uses the same-tier LLM as the Judge, taking the query and answer as input and outputting "yes"/"no". The Judge for the lowest tier (GPT-4o-mini) utilizes a higher-tier LLM (GPT-4o) to compensate for the weak model's insufficient self-verification capability. GPT-4o-mini also supports an "abstention" option—complex queries directly upshift, bypassing the Judge to save costs.
    - **Design Motivation**: Self-verification in low-tier models is unreliable (Huang et al. 2023), which is compensated for by a stronger Judge. The abstention mechanism avoids useless attempts by weak models on complex queries.

## Key Experimental Results

### Main Results (Compared with single-model baselines)

| Method | MATH Accuracy | MATH Cost ($) | MATH Time (min) | MCQA Accuracy | MCQA Cost ($) |
|------|-----------|-------------|---------------|-----------|-------------|
| o1 (Single) | Highest | 41.56 | 110.73 | Highest | 59.52 |
| o1-mini (Iterative) | - | - | 123.73 | - | - |
| LLM-AT (o1 Upper Bound) | Close to o1 | **16.89** (-59.37%) | **88.79** (-19.81%) | Close to o1 | **7.14** (-88.01%) |

### Judge Reliability (MATH)

| Model | Judge F1 | Generator Accuracy |
|------|---------|-----------------|
| GPT-4o-mini (Special Judge) | 0.828 | 0.531 |
| GPT-4o | 0.799 | 0.610 |
| o1-mini | 0.876 | 0.749 |

### Cold Start Robustness

| Historical Data Quantile | MATH Accuracy | MCQA Accuracy | Note |
|--------------|-----------|-----------|------|
| Q1 (First 25%) | 0.71 | 0.835 | Cold start phase |
| Q2 (25-50%) | 0.79 | 0.893 | Rapid improvement |
| Q3 (50-75%) | 0.75 | 0.941 | Stable |
| Q4 (Last 25%) | 0.86 | 0.955 | Optimal |

### Key Findings
- LLM-AT consistently outperforms single-inference and iterative-inference baselines on the Pareto frontiers of accuracy-cost and accuracy-time.
- Cost savings are extremely significant: 59.37% on MATH and 88.01% on MCQA.
- The median of the accuracy estimator aligns well with the actual accuracy trend, validating the effectiveness of label-free estimation.
- The cold-start effect exists but fades quickly—performance is significantly enhanced with a few hundred samples, making it suitable for practical deployment.
- In subcategories where model performance is reversed (lower tier outperforms higher tier), LLM-AT adaptively selects lower-tier models more often, demonstrating robustness.
- The F1 score of the Judge is consistently higher than the accuracy of the Generator, confirming the reliability of the pseudo-labeling scheme.

## Highlights & Insights
- Elegant design philosophy—it compares the LLM routing problem to an automatic transmission, with a clear division of labor among the Starter, Generator, and Judge modules. Being entirely training-free is the biggest advantage, as accommodating a new model only requires adding it as a tier.
- The accuracy estimator relies on the simple assumption that "similar questions perform similarly on similar tiers," but achieves unexpectedly powerful results through Bayesian smoothing and similarity weighting.

## Limitations & Future Work
- Validated only on the OpenAI tier system (o1/o1-mini/GPT-4o/GPT-4o-mini), without testing open-source model tiers.
- Using the same-tier LLM as the Judge may introduce systematic biases—such as over-relying on its own answers.
- The pseudo-labeling mechanism assumes that higher tiers do not perform worse than lower tiers (performance monotonicity), but experiments show this is not always true.
- The accuracy estimator depends on the quality of the embedding model and the selection of the top-k parameter.

## Related Work & Insights
- **vs Chen et al. 2024a**: Both use cascade mechanisms, but they always start from the lowest tier; LLM-AT's Starter can skip unnecessary low tiers, saving time and cost.
- **vs Ding et al. 2024; Ong et al. 2024**: Training-based routing requires labeled data and lacks generalization; LLM-AT is training-free and adaptively adjusts by accumulating History during runtime.
- **vs Madaan et al. 2023 (Self-Refine)**: While Self-Refine iteratively improves within the same tier, LLM-AT iterates across tiers—upshifting to a stronger model upon invalidation rather than having the same model retry.

## Rating
- Novelty: ⭐⭐⭐⭐ Training-free routing combined with history-based accuracy estimation is an innovative paradigm, though the upgrade strategy is relatively intuitive.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated on two datasets of different difficulty gradients with multiple deep-dive analyses (cold start, robustness, Judge reliability).
- Writing Quality: ⭐⭐⭐⭐ The automatic transmission analogy is intuitive, and the framework diagram is clear.
- Value: ⭐⭐⭐⭐ Directly applicable to multi-tier LLM deployments; its training-free nature allows rapid adaptation when new models are released.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Mind Your Tone: Investigating How Prompt Politeness Affects LLM Accuracy](mind_your_tone_investigating_how_prompt_politeness_affects_llm_accuracy_short_pa.md)
- [\[ACL 2025\] BFS-Prover: Scalable Best-First Tree Search for LLM-Based Automatic Theorem Proving](bfs-prover_scalable_best-first_tree_search_for_llm-based_automatic_theorem_provi.md)
- [\[ACL 2025\] AutoExp: Automatic Experiment Design and Execution by LLMs](autoexp_automatic_experiment_design_and_execution_by_llms.md)
- [\[ACL 2025\] LLM Braces: Straightening Out LLM Predictions with Relevant Sub-Updates](llm_braces_straightening.md)
- [\[ACL 2025\] AutoGUI: Scaling GUI Grounding with Automatic Functionality Annotations from LLMs](autogui_scaling_gui_grounding_with_automatic.md)

</div>

<!-- RELATED:END -->
