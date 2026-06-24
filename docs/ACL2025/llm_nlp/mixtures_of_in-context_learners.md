---
title: >-
  [Paper Note] Mixtures of In-Context Learners
description: >-
  [ACL2025][LLM (Other)][in-context learning] This paper proposes MoICL, which partitions the set of demonstrations in ICL into multiple subsets (experts) and blends their next-token distributions using a learnable weight function. This approach significantly improves the accuracy, robustness, and efficiency of ICL without modifying LLM parameters.
tags:
  - "ACL2025"
  - "LLM (Other)"
  - "in-context learning"
  - "mixture of experts"
  - "demonstration selection"
  - "product of experts"
  - "robustness"
date: 2026-05-08
content_hash: fd41bbe34de0631b
---

# Mixtures of In-Context Learners

**Conference**: ACL2025  
**arXiv**: [2411.02830](https://arxiv.org/abs/2411.02830)  
**Code**: -  
**Area**: LLM/NLP  
**Keywords**: in-context learning, mixture of experts, demonstration selection, product of experts, robustness  

## TL;DR

This paper proposes MoICL, which partitions the set of demonstrations in ICL into multiple subsets (experts) and blends their next-token distributions using a learnable weight function. This approach significantly improves the accuracy, robustness, and efficiency of ICL without modifying LLM parameters.

## Background & Motivation

In-context learning (ICL) is a transformative technology in current NLP, which enables LLMs to perform various tasks purely by providing demonstrations within the context, without requiring finetuning of model parameters. However, ICL exhibits several key limitations:

**Limited Context Length**: The maximum context length of Transformers constrains the number of demonstrations that can be used, whereas more demonstrations typically lead to better performance.

**Quadratic Complexity**: As the number of demonstrations increases, the computational and memory overhead of self-attention grows quadratically.

**Sensitivity to Demonstration Selection**: The performance of ICL is highly sensitive to the selection of demonstrations; different selections can yield drastically different results.

**Lack of Quality Distinction**: Traditional ICL treats all demonstrations with equal weights, failing to distinguish between high-quality and low-quality (or even noisy) examples.

Most existing demonstration selection methods rely on heuristics and fail to quantify the contribution of each example to generalization performance. These challenges collectively motivate the authors to propose a more expressive learning-based approach to combine the outputs of multiple demonstration sets.

## Method

### Overall Architecture

The core idea of MoICL (Mixtures of In-Context Learners) is as follows:

1. **Partitioning Demonstrations**: Partition the demonstration set D into k disjoint subsets D_1, D_2, ..., D_k.
2. **Creating Experts**: Each subset D_i along with the input text x is fed into the LLM, producing a next-token distribution p(y|D_i, x), which is referred to as an "expert".
3. **Weighted Fusion**: Blend the output distributions of all experts using a learnable weight w ∈ ℝᵏ.

The final predictive distribution is formalized in the Product of Experts form:

$$p(y|D, x) \propto \exp\left[\sum_{i=1}^{k} w_i \log p(y|D_i, x)\right]$$

where w_i represents the contribution weight of the i-th expert. This formulation allows negative weights, enabling certain experts to act as "anti-experts".

### Key Designs

**Two Implementations of the Weight Function**:

1. **Scalar Weights (Scalar)**: Directly learn k scalar weight parameters w ∈ ℝᵏ, initialized to 1/k. The advantage is simplicity and efficiency, while the disadvantage is that weights are tied to fixed demonstration subsets.

2. **Hyper-network**: Use a small T5 model as a hyper-network h_ϕ(·), which takes the concatenation of all demonstration subsets as input to dynamically generate weights. The advantage is the ability to generalize to unseen demonstrations during training.

**Sparsified Mixture Weights**:

To reduce inference overhead (which requires calling the LLM k times for each token), the authors propose to sparsify weights:

$$w = w' \odot \text{top-}k'(m)$$

where m is a learnable masking coefficient, and the top-k' function retains only the largest k' weights. Since top-k' is a discrete operation, Implicit Maximum Likelihood Estimation (IMLE) is employed for gradient estimation to achieve end-to-end training.

**Anti-Expert Mechanism**:

Allowing negative weights is a crucial design choice of MoICL. A negative weight implies that the corresponding demonstration subset is not only unhelpful but is also utilized as a negative reference. Experiments demonstrate that restricting weights to be positive leads to a significant performance drop (from 81.33% to 76.05%).

### Loss & Training

The weight parameters are learned by maximizing the conditional log-likelihood on the training set D_T. The training process only updates the parameters of the weight function (either scalar weights or hyper-network parameters), requiring no modifications to the LLM parameters.

## Experiments

### Main Results

Evaluations are conducted on classification tasks using Llama-3-8B-Instruct with 30 demonstrations:

| Method | Offensive | Hate | SST2 | PAWS | QNLI |
|------|-----------|------|------|------|------|
| Concat-based ICL | 76.44 | 53.54 | 95.46 | 78.12 | 89.08 |
| Random Search | 77.88 | 58.09 | 95.76 | 78.88 | 89.99 |
| LENS | 78.70 | 53.20 | 93.81 | 75.60 | 89.04 |
| LoRA (PEFT) | 79.79 | 53.76 | 85.89 | 54.82 | 57.24 |
| **MoICL scalar k=10** | **79.42** | **66.52** | 95.32 | **79.42** | **90.44** |
| **MoICL scalar k=30** | **81.33** | 63.45 | 94.79 | **79.50** | 90.11 |

MoICL outperforms all baseline methods on 5 out of 7 datasets, achieving the largest improvement on the Hate dataset (+13%). Notably, whereas LoRA requires access to model weights for finetuning, MoICL does not.

### Robustness Experiments

**OOD Demonstration (Out-of-Domain Examples)**: When 70% of the demonstrations are drawn from a different dataset (SST2), MoICL scalar still maintains an accuracy of 80.19%, whereas concat-based ICL drops to 68.49%, demonstrating a gain of up to +11% for MoICL. Weight visualizations reveal that in-domain demonstrations receive positive weights while out-of-domain demonstrations receive negative weights, indicating that MoICL successfully learns to distinguish between in-domain and out-of-domain examples.

**Label Imbalance**: Under an extremely imbalanced setup where only 1 out of 30 demonstrations is "neutral" and 29 are "offensive", the accuracy of concat-based ICL plummets from 76.44% to 28.49%, whereas MoICL scalar only drops from 81.33% to 77.77%, achieving a gain of up to +49%.

**Noisy Demonstration (Label Noise)**: In the NQ-Open generation task, even when the answers of 10/12 demonstrations are replaced with random values, MoICL scalar maintains stable performance (over +35% improvement), whereas concat-based ICL performance continuously declines. Weight analysis shows that noisy demonstrations receive weights close to 0 or negative values.

### Ablation Study

1. **Impact of the Number of Subsets**: Under the scalar weight setting, increasing the number of subsets k (i.e., fewer demonstrations per subset) paradoxically improves performance. Although fewer demonstrations per subset weaken individual experts, the increased number of tunable weights enhances flexibility, compensating for this loss.

2. **Sparsification Effect**: The IMLE top-k' mask achieves an accuracy of 76.07% when selecting only 5 experts, demonstrating strong sparsification performance and stability.

3. **Impact of Model Scale**: Across the 7B, 13B, and 70B versions of Llama-2, MoICL scalar consistently outperforms concat-based ICL, with the largest gain observed on the 70B model (82.26% vs 69.42%).

### Efficiency Analysis

- **Data Efficiency**: Requires only around 20 labeled demonstrations to outperform concat-based ICL.
- **Time Efficiency**: For the same level of performance, MoICL requires less inference time.
- **Computational Complexity**: The complexity of MoICL is k·(n/k+1)²·C_LLM, which is superior to the (n+1)²·C_LLM complexity of concat-based ICL.

## Highlights & Insights

1. **Value of Anti-Experts**: Allowing negative weights enables MoICL to learn from "negative examples." This is especially crucial in OOD and noisy scenarios, where poor demonstrations are not only ignored but also actively exploited.
2. **No Need for Model Weights**: MoICL only requires access to the output logits of the LLM without modifying the model parameters, which is suitable for semi-black-box scenarios.
3. **Adapting MoE to ICL**: Migrating the Mixture of Experts concept to ICL—where demonstration subsets serve as experts and weights are learned via gradient descent—presents an elegant analogy.
4. **Pareto Frontier Improvements**: MoICL holds an advantage on the accuracy-efficiency Pareto frontier, achieving superior performance with a shorter context length.

## Limitations & Future Work

1. **Requirement of Logits Access**: MoICL requires access to the vocabulary distribution logits of the LLM, making it inapplicable to pure black-box APIs (e.g., models like GPT-4 that do not expose logits).
2. **Need for Training Data**: Weight tuning requires a small amount of labeled data to serve as the training set.
3. **Limited Experimental Scale**: Experiments are only conducted on the Llama-2 and Llama-3 series, without verification on larger or closed-source models.
4. **Weights Tied to Demonstration Subsets**: The scalar weight method requires a fixed partitioning of demonstrations; although the hyper-network resolves this issue, it introduces a small computational overhead.

## Related Work & Insights

- **Concat-based ICL** (Brown et al., 2020): The standard ICL approach, which concatenates all demonstrations into the context.
- **Ensemble-based ICL** (Min et al., 2022): Each demonstration is fed independently into the LLM, and the output distributions are combined via product fusion.
- **LENS** (Li and Qiu, 2023): An improved ICL method based on retrieval.
- **Mixtures of In-Context Experts** (Le et al., 2022): Uses cosine similarity to compute weights; it is the direct predecessor of this work.

## Rating

⭐⭐⭐⭐ (4/5)

Highly innovative, elegantly applying the MoE concept to ICL. The experimental design is comprehensive with in-depth analysis, among which the robustness analysis is particularly impressive. However, it is limited by the requirement of logits access, restricting its applicability in the era of pure black-box LLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] ExpeTrans: LLMs Are Experiential Transfer Learners](expetrans_llms_are_experiential_transfer_learners.md)
- [\[ACL 2025\] Large Language Models are Good Relational Learners](large_language_models_are_good_relational_learners.md)
- [\[ACL 2025\] Problem-Solving Logic Guided Curriculum In-Context Learning for LLMs Complex Reasoning](problem-solving_logic_guided_curriculum_in-context_learning_for_llms_complex_rea.md)
- [\[ACL 2025\] Uncertainty Unveiled: Can Exposure to More In-context Examples Mitigate Uncertainty for Large Language Models?](uncertainty_unveiled_can_exposure_to_more_in-context_examples_mitigate_uncertain.md)
- [\[AAAI 2026\] Vision Transformers are Circulant Attention Learners](../../AAAI2026/llm_nlp/vision_transformers_are_circulant_attention_learners.md)

</div>

<!-- RELATED:END -->
