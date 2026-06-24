---
title: >-
  [Paper Note] Understanding Impact of Human Feedback via Influence Functions
description: >-
  [ACL 2025][LLM Alignment][influence functions] Applying influence functions to feedback data auditing for RLHF reward models for the first time, combined with OPORP vector compression to achieve a 2.5x speedup. It outperforms GPT-4o in bias detection (AUC 0.8 vs. 0.747) and identifies 47% of mislabeled samples from the Anthropic-HH dataset.
tags:
  - "ACL 2025"
  - "LLM Alignment"
  - "influence functions"
  - "RLHF"
  - "reward model"
  - "bias detection"
  - "data quality"
  - "OPORP"
date: 2026-05-08
content_hash: c2a7f841d63142ae
---

# Understanding Impact of Human Feedback via Influence Functions

**Conference**: ACL 2025  
**arXiv**: [2501.05790](https://arxiv.org/abs/2501.05790)  
**Code**: [https://github.com/mintaywon/IF_RLHF](https://github.com/mintaywon/IF_RLHF)  
**Area**: LLM Alignment / RLHF Data Analysis  
**Keywords**: influence functions, RLHF, reward model, bias detection, data quality, OPORP

## TL;DR

Applying influence functions to feedback data auditing for RLHF reward models for the first time, combined with OPORP vector compression to achieve a 2.5x speedup. It outperforms GPT-4o in bias detection (AUC 0.8 vs. 0.747) and identifies 47% of mislabeled samples from the Anthropic-HH dataset.

## Background & Motivation

**RLHF relies on high-quality feedback**: Reward model (RM) training completely depends on human preference annotations. Feedback quality directly determines alignment performance—garbage in, garbage out. However, human annotators inevitably introduce systematic biases in practice.

**Diverse bias types**: Known biases include length bias (annotators tend to choose longer responses) and sycophancy bias (tending to choose responses that align with the user's view). These biases are propagated through the RM to the final LLM behaviors.

**Lack of systematic auditing methods**: Current quality auditing for RLHF data heavily relies on manual spot-checking or LLM-assisted evaluation (e.g., using GPT-4o for individual reviews). The former is not scalable, while the latter is expensive and yields limited accuracy.

**Theoretical advantages of influence functions**: Influence functions can accurately quantify the contribution of each training sample to model predictions without retraining the model, making them ideal data auditing tools. However, they suffer from severe computational bottlenecks at LLM scale.

**Characteristics of preference learning**: Standard influence functions are derived based on classification/regression loss, whereas preference learning uses the pairwise loss of the Bradley-Terry model, requiring specialized theoretical derivation and engineering adaptations.

**Key Insight**: Through DataInf approximation + OPORP gradient compression (160MB to 256KB), influence functions become computationally feasible on Llama-3-8B scale RMs. Additionally, bias-sensitive validation sets are designed to target and detect specific bias types.

## Method

### Overall Architecture

The overall workflow is divided into four stages: (1) fine-tuning Llama-3-8B as RM on the Anthropic-HH dataset using LoRA; (2) extracting gradients for each training/validation sample through forward propagation and compressing them to 256KB using OPORP; (3) approximating the influence scores of each training sample on the validation set loss using the DataInf formulation; (4) identifying samples with the lowest influence scores (the largest negative contributions) as the most suspicious biased/mislabeled samples.

### Key Designs

**1. Bradley-Terry Influence Function Derivation**

- **Function**: Extending classical influence function theory from classification loss to the Bradley-Terry loss formulation of preference learning.
- **Mechanism**: Computing the change in model parameters after removing the $i$-th training sample, thereby obtaining the change in validation loss. Analytical forms for the Hessian and gradient of the BT loss are derived.
- **Design Motivation**: The BT loss involves the reward difference between both chosen and rejected responses for each sample. Its gradient structure differs from standard cross-entropy, preventing the direct application of existing formulations.

**2. Efficient Approximation with DataInf + OPORP**

- **Function**: Reducing the computation of Hessian-inverse-vector products to a feasible scale.
- **Mechanism**: DataInf uses diagonal approximation to decompose the Hessian, decoupling the influence computation of $n$ samples. OPORP (Orthogonal Random Projection) compresses the gradient from 160MB to 256KB using a random orthogonal matrix, preserving the dot product between gradients.
- **Design Motivation**: The parameter count of Llama-3-8B's LoRA (rank=16) remains large. Full DataInf requires 28.8 hours, whereas OPORP compression reduces it to only 92.3 seconds (achieving 2.5x acceleration), enabling large-scale application.

**3. Bias-Sensitive Validation Set Design**

- **Function**: Constructing specialized validation sets so that influence scores are highly sensitive to specific types of bias (e.g., length/sycophancy).
- **Mechanism**: For length bias detection, the validation set is composed of sample pairs with obvious length differences but equivalent content quality. For sycophancy bias detection, the validation set contains pairings of "sycophantic but incorrect" vs. "disagreeing but correct".
- **Design Motivation**: Influence scores measure the "contribution of training samples to the validation loss." The directional bias of the validation set determines what biases can be detected—general validation sets can only find "generally problematic samples."

**4. Annotation Strategy Analysis and Guidance**

- **Function**: Extending influence score analysis to serve as a guidance tool for annotator behavior.
- **Mechanism**: Comparing the differences in annotation characteristics between high-influence (beneficial) and low-influence (harmful) samples to extract implicit strategies of expert annotators, and feeding these strategies back to non-expert annotators.
- **Design Motivation**: Rather than just detecting problematic data, the goal is to mine "what constitutes a good annotation" from the data, establishing a closed loop from data auditing to annotation quality improvement.

### Loss & Training

The RM training utilizes the standard Bradley-Terry preference loss. The model is Llama-3-8B + LoRA (rank=16). The influence function is a post-training analysis tool and does not alter the training process itself.

## Key Experimental Results

### Main Results

| Detection Task | Method | AUC | Remarks |
|----------|------|-----|------|
| Length Bias Detection | **IF (Ours)** | **0.800** | Best |
| Length Bias Detection | GPT-4o | 0.747 | Requires LLM inference, high cost |
| Length Bias Detection | Mahalanobis | 0.600 | Statistical baseline |
| Sycophancy Bias Detection | **IF (Ours)** | **0.711** | Best |
| Sycophancy Bias Detection | Baselines | ~0.600 | Close to random |
| Mislabel Detection (Top-100) | **IF (Ours)** | **47/100** | 47% confirmed as mislabeled |
| Mislabel Detection (Top-100) | Random Sampling | 13/100 | Baseline |

### Ablation Study

| Configuration | Computation Time | Influence Score Quality |
|------|----------|------------|
| DataInf (No compression) | 28.8 hours | Baseline |
| DataInf + OPORP | 92.3 seconds | Highly consistent with baseline |
| Compression Ratio 160MB→256KB | 2.5x acceleration | Gradient dot product preserved |
| LoRA rank=16 | — | Optimal precision-efficiency balance |

### Key Findings

- Among the top 100 samples with the lowest influence scores, 47% were manually verified as mislabeled, significantly outperforming the random baseline of 13%.
- Outperforms GPT-4o in length bias detection (+5.3% AUC) without requiring LLM inference costs.
- Influence scores reveal systematic bias patterns of annotators: samples with high negative influence have a significantly higher proportion of long responses selected as preferred.
- Annotation quality of non-expert annotators can be significantly improved by learning the characteristics of high-influence samples.
- OPORP compression has almost no impact on the relative ranking of influence scores, while bringing over 1000x memory savings.

## Highlights & Insights

- **Influence function as a paradigm for data auditing**: Different from traditional data cleaning (based on rules or anomaly detection), the influence function provides a causal quantification of sample contributions, directly answering "is this sample beneficial or harmful to the model?"
- **Generality of OPORP compression**: The gradient compression technology can be transferred to other scenarios requiring large-scale gradient analysis (e.g., data selection, model interpretability).
- **The validation set determines the detection direction**: This insight implies that the influence function serves as a programmable auditing framework—switching the validation set allows detecting novel types of biases, offering strong flexibility.
- **A closed loop from auditing to improvement**: Not only detecting problematic data but also distilling effective annotation strategies to feed back to annotators.

## Limitations & Future Work

- OPORP gradient compression still suffers from information loss. Extremely fine-grained biases might be missed, and the trade-off between compression ratio and detection accuracy deserves further investigation.
- The design of validation sets relies on prior knowledge of bias types. If the bias types are unknown or novel, automated methods for constructing validation sets need to be explored.
- Validation was conducted only at the RM level; the application of influence functions during the policy model (PPO/DPO) training phase remains unexplored.
- The diagonal approximation of DataInf might be imprecise when parameters are highly correlated.
- Currently, only length and sycophancy biases are covered. Other known bias types, such as formatting bias and positional bias, remain to be validated.

## Related Work & Insights

- **vs. GPT-4o instance-by-instance review**: GPT-4o requires inference on each data point, with costs scaling linearly with data size. In contrast, the influence function is computed once, allowing subsequent fast queries of any sample.
- **vs. Outlier detection (Mahalanobis distance)**: Statistical anomaly detection only focuses on how anomalous a sample is in the feature space, regardless of its actual impact on model predictions.
- **vs. TracIn/TRAK**: These influence function variants are effective in classification tasks but are not adapted to the BT loss of preference learning.
- **Insight**: The combination of influence functions and gradient compression can be extended to full-pipeline data auditing in DPO/RLHF.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Applying influence functions systematically to RLHF data auditing for the first time.
- Experimental Thoroughness: ⭐⭐⭐⭐ Sufficient comparisons across two applications: bias detection and annotation improvement.
- Writing Quality: ⭐⭐⭐⭐ Clear combination of theoretical derivation and application scenarios.
- Value: ⭐⭐⭐⭐⭐ Providing a scalable tool for RLHF data quality control.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Curiosity-Driven Reinforcement Learning from Human Feedback](curiosity_driven_rlhf.md)
- [\[NeurIPS 2025\] Strategyproof Reinforcement Learning from Human Feedback](../../NeurIPS2025/llm_alignment/strategyproof_reinforcement_learning_from_human_feedback.md)
- [\[ICLR 2026\] RLBFF: Binary Flexible Feedback to Bridge Between Human Feedback & Verifiable Rewards](../../ICLR2026/llm_alignment/rlbff_binary_flexible_feedback_to_bridge_between_human_feedback_verifiable_rewar.md)
- [\[ACL 2025\] Balancing the Budget: Understanding Trade-offs Between Supervised and Preference-Based Finetuning](balancing_the_budget_understanding_trade-offs_between_supervised_and_preference-.md)
- [\[ICLR 2026\] What's In My Human Feedback? Learning Interpretable Descriptions of Preference Data](../../ICLR2026/llm_alignment/whats_in_my_human_feedback_learning_interpretable_descriptions_of_preference_dat.md)

</div>

<!-- RELATED:END -->
