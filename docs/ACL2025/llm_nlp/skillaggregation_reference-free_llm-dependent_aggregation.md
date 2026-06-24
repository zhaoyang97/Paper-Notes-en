---
title: >-
  [Paper Note] SkillAggregation: Reference-free LLM-Dependent Aggregation
description: >-
  [ACL 2025][LLM (Other)][LLM Evaluation] This paper proposes SkillAggregation, a method that learns context-dependent skill weights of LLM judges and performs inference using posterior estimation. It effectively aggregates the predictions of multiple LLM judges without reference labels, outperforming existing aggregation methods across multiple tasks.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "LLM Evaluation"
  - "Multi-Model Aggregation"
  - "Reference-Free Aggregation"
  - "LLM-as-a-Judge"
  - "Crowdsourced Annotation"
date: 2026-05-08
content_hash: 7cfe8aad3a90fd41
---

# SkillAggregation: Reference-free LLM-Dependent Aggregation

**Conference**: ACL 2025  
**arXiv**: [2410.10215](https://arxiv.org/abs/2410.10215)  
**Code**: None  
**Area**: LLM/NLP  
**Keywords**: LLM Evaluation, Multi-Model Aggregation, Reference-Free Aggregation, LLM-as-a-Judge, Crowdsourced Annotation

## TL;DR

This paper proposes SkillAggregation, a method that learns context-dependent skill weights of LLM judges and performs inference using posterior estimation. It effectively aggregates the predictions of multiple LLM judges without reference labels, outperforming existing aggregation methods across multiple tasks.

## Background & Motivation

LLM-as-a-judge has emerged as an important alternative for evaluating NLP tasks. However, individual LLMs suffer from issues such as self-preference bias, verbosity bias, and prompt sensitivity. While employing multiple LLM judges can improve performance, the key challenge lies in effectively aggregating their judgments.

**Limitations of Prior Work**:

**Equally-weighted methods** (e.g., majority voting, average probability) ignore the skill differences among judges; for instance, GPT-4 is typically superior to GPT-3 and should be assigned a higher weight.

**Task-specific methods** (e.g., CrossCheckGPT) are designed only for specific tasks (e.g., hallucination detection) and cannot generalize to other evaluation scenarios.

**Overly restrictive constraints** (e.g., PRD) require each judge to evaluate all other judges, which limits practical applicability.

Inspired by crowdsourcing aggregation, the authors treat LLM judges as "workers" and propose a general, reference-free, context-aware aggregation method.

## Method

### Overall Architecture

SkillAggregation is developed based on improvements to the Crowdlayer method and consists of three core components:
1. A context encoder (a pre-trained language model, such as GPT-2) encodes textual inputs into vector representations.
2. A bottleneck layer projects the encoded representations into a 2-dimensional class distribution estimation.
3. Learnable skill-estimate vectors capture the capability of each LLM judge.

### Key Designs

1. **Skill-Estimate Vectors**:

    - Each LLM judge k corresponds to a pair of scalars p̂₀^(n,k) and p̂₁^(n,k).
    - p̂₀^(n,k) ≈ P(b_{n,k}=0|c_n=0, X_n): Given that the true label is 0 and the context is given, the probability that the judge correctly predicts as 0.
    - p̂₁^(n,k) ≈ P(b_{n,k}=1|c_n=1, X_n): Given that the true label is 1 and the context is given, the probability that the judge correctly predicts as 1.
    - Skills can be task-specific (sharing a set of parameters across all samples) or context-specific (SkillAggregation-X, mapped from context via a linear layer + Sigmoid).

2. **Regularization Term**:

    - Analysis reveals that predictions can be rewritten as (p̂₀ + p̂₁ - 1)·s_{n,0} + (1 - p̂₁), depicting a linear relationship between the LLM's judgment and the true label.
    - Overconfident LLMs lead to an excessively large slope (p̂₀ + p̂₁ - 1), amplifying their influence.
    - The regularization term L_reg = Σ(p̂₀ + p̂₁ - 1)² is proposed to penalize excessively large slopes.
    - Total loss: L = L_CE + λ·L_reg

3. **Posterior Estimation Inference**:

    - Compared to Crowdlayer, which relies solely on the bottleneck layer's output for inference, SkillAggregation leverages the LLM judgments during inference.
    - It assumes that LLMs are conditionally independent given the ground truth and context (CI assumption).
    - The posterior P(c_n|X_n, b_n) is computed via Bayes' theorem, using the learned skill-estimate vectors and bottleneck output to approximate the true skills and prior.
    - Final decision: compare the posterior ratio r_n, and predict positive when r_n > 1.

### Loss & Training

- Training Objective: Minimize the cross-entropy loss between predicted LLM judgments and actual LLM judgments + the regularization term.
- Context Encoder: GPT-2 base (117M parameters), with the last hidden state used as the context representation.
- Completely reference-free: Learned directly on the entire test set without reference labels.
- Model Selection: 250 labeled development set samples are used for hyperparameter selection.
- Training Time: Only takes 20-30 minutes on a single NVIDIA RTX 6000 Ada GPU.

## Key Experimental Results

### Main Results

| Method | HaluEval 7B(%) | TruthfulQA 7B(%) | Chatbot Arena 7B(%) |
|------|------|----------|------|
| Average Probability | 76.28 | 68.06 | 63.24 |
| Majority Voting | 76.16 | 67.47 | 63.93 |
| DawidSkene | 76.78 | 67.84 | 64.71 |
| Train on MV | 78.78 | 67.32 | 63.77 |
| Crowdlayer | 79.27 | 67.74 | 64.06 |
| SkillAgg w/o Reg | 80.22 | 68.07 | 64.17 |
| **SkillAgg** | **80.83** | **68.74** | **64.22** |
| **SkillAgg-X** | **81.06** | **68.77** | **64.43** |

SkillAggregation-X achieves an absolute accuracy improvement of 4.9% on HaluEval, 1.3% on TruthfulQA, and 0.5% on Chatbot Arena (compared to majority voting).

| Method | HaluEval ~70B(%) | TruthfulQA ~70B(%) | Chatbot Arena ~70B(%) |
|------|------|----------|------|
| Majority Voting | 80.81 | 83.63 | 70.61 |
| **SkillAgg-X** | **84.79** | **84.57** | **70.72** |

When using 70B-scale LLM judges, the overall performance increases significantly, but the gain brought by the aggregation method decreases.

### Ablation Study

| Configuration | Key Metrics | Description |
|------|---------|------|
| Regularization effect | Stable improvements across all three datasets | Prevents overconfident LLMs from dominating posterior estimation |
| Context encoder replacement (RoBERTa/Gemma-2B) | Similar performance | The method is insensitive to the choice of encoder |
| Judge subset analysis | Close to DawidSkene with weak judges, clear advantages with strong judges | Requires sufficiently good judges to learn effective priors |
| Dataset size | Unstable performance with 1000 samples, stable with 5000 samples | Requires sufficient samples to learn skill estimates effectively |
| Post-bias mitigation | Diminished gains | Portion of the gain comes from implicit debiasing effects |

### Key Findings

1. Differentiated weighting methods (DawidSkene, SkillAggregation) outperform equally-weighted methods (majority voting, average probability) across all tasks.
2. The learned skill estimates are highly correlated with the actual accuracy of LLMs (PCC = 93.6% on HaluEval).
3. The improvement is greatest on HaluEval, because the context encoder inherently possesses a certain level of task comprehension.
4. The improvement is smallest on Chatbot Arena, as human preference evaluation is inherently noisier.
5. The regularization term contributes significantly to the 7B/8B models, effectively mitigating the overconfidence issue in smaller models.

## Highlights & Insights

1. **Reference-free learning**: Learns aggregation weights solely from LLM judgments without human-annotated data, making it highly valuable in practical applications.
2. **Posterior estimation inference**: Compared to Crowdlayer, which only uses prior prediction, the design of introducing LLM judgments for posterior updates during inference is clever and effective.
3. **Theoretical motivation of the regularization term**: By analyzing the meaning of the slope in the linear relationship, the necessity of the regularization term is naturally derived.
4. **Transfer from crowdsourcing to LLM evaluation**: Adapting mature crowdsourcing annotation theories (e.g., Dawid-Skene) to the LLM evaluation scenario is methodologically inspiring.
5. **Lightweight training**: Requires only GPT-2 base and 20-30 minutes of training, resulting in extremely low deployment costs.

## Limitations & Future Work

1. Only focuses on binary classification tasks and has not been extended to regression or multi-class evaluation scenarios.
2. The conditional independence assumption may not hold; multiple LLMs might make correlated errors on the same sample.
3. Calibration performance is not considered, as the focus is solely on accuracy.
4. The development set still requires 250 labeled samples, meaning it is not completely unsupervised.
5. Whether more powerful context encoders can further improve performance remains unexplored.

## Related Work & Insights

This paper introduces the Worker Aggregation theory from crowdsourced annotation into the LLM evaluation field, contrasting and complementing works like PoLL (equally-weighted multi-judge), CrossCheckGPT (aggregation dedicated to hallucinations), and PRD (rank aggregation). This method can be extended to any scenario requiring the aggregation of multiple model predictions, such as model ensembles and multi-agent decision-making.

## Rating

- Novelty: ⭐⭐⭐⭐ Restructures Crowdlayer into SkillAggregation and introduces posterior inference and regularization, with clear core innovations.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated on three tasks, various judge configurations, and multi-dimensional ablation analyses, providing a relatively comprehensive evaluation.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous mathematical derivations, in-depth analysis, and clear charts and tables.
- Value: ⭐⭐⭐⭐ Provides direct guidance for LLM evaluation practices; the proposed method is simple and highly efficient.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Probabilistic Aggregation and Targeted Embedding Optimization for Collective Moral Reasoning](probabilistic_aggregation_and_targeted_embedding_optimization_for_collective_mor.md)
- [\[ACL 2025\] Training-free LLM Merging for Multi-task Learning](training-free_llm_merging_for_multi-task_learning.md)
- [\[ACL 2025\] A Training-free LLM-based Approach to General Chinese Character Error Correction](a_training-free_llm-based_approach_to_general_chinese_character_error_correction.md)
- [\[ACL 2025\] Cuckoo: An IE Free Rider Hatched by Massive Nutrition in LLM's Nest](cuckoo_an_ie_free_rider_hatched_by_massive_nutrition_in_llms_nest.md)
- [\[ACL 2025\] Leveraging Self-Attention for Input-Dependent Soft Prompting in LLMs](input_dependent_soft_prompting.md)

</div>

<!-- RELATED:END -->
