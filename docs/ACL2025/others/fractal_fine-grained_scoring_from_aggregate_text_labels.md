---
title: >-
  [Paper Note] FRACTAL: Fine-Grained Scoring from Aggregate Text Labels
description: >-
  [ACL 2025][Fine-Grained Scoring] Proposes the FRACTAL method, which decomposes response-level aggregate labels into sentence-level pseudo-labels. By combining multi-instance learning (MIL) and learning from label proportions (LLP) techniques with prior information (document-sentence cosine similarity), it trains a sentence-level scoring model covering four types of tasks: retrieval, question answering, summarization, and mathematical reasoning.
tags:
  - "ACL 2025"
  - "Fine-Grained Scoring"
  - "Multi-Instance Learning"
  - "Learning from Label Proportions"
  - "Pseudo-Labels"
  - "LLM Evaluation"
date: 2026-05-08
content_hash: 09af64fc165dfdd5
---

# FRACTAL: Fine-Grained Scoring from Aggregate Text Labels

**Conference**: ACL 2025  
**arXiv**: [2404.04817](https://arxiv.org/abs/2404.04817)  
**Code**: None  
**Area**: Other  
**Keywords**: Fine-Grained Scoring, Multi-Instance Learning, Learning from Label Proportions, Pseudo-Labels, LLM Evaluation

## TL;DR

Proposes the FRACTAL method, which decomposes response-level aggregate labels into sentence-level pseudo-labels. By combining multi-instance learning (MIL) and learning from label proportions (LLP) techniques with prior information (document-sentence cosine similarity), it trains a sentence-level scoring model covering four types of tasks: retrieval, question answering, summarization, and mathematical reasoning.

## Background & Motivation

Evaluation and optimization feedback for LLMs are usually provided at the response level (e.g., quality ratings for the entire response). Although this coarse-grained feedback is efficient and low-cost, it has clear limitations:

**Imprecise Localization**: Response-level labels cannot identify which sentences in the response are high quality and which have issues (such as factual errors, redundancies, or irrelevancy).

**Poor Interpretability**: They cannot provide fine-grained optimization objectives for LLM fine-tuning.

**Recent studies show**: Sentence-level labels can provide more accurate and interpretable feedback for LLM optimization (Amplayo et al., 2022; Lightman et al., 2023). However, collecting fine-grained human feedback is often extremely costly, especially for Side-by-Side (SxS) preference evaluations.

**Core Problem**: **Can useful sentence-level fine-grained labels be inferred from easily obtainable response-level coarse-grained labels?**

This is essentially a weakly supervised learning problem: given the aggregate labels of "bags" (responses), the goal is to infer the labels of "instances" (sentences).

## Method

### Overall Architecture

FRACTAL consists of three key components:
1. Loss function design: Models responses as "bags" and sentences as "instances", designing a joint loss that considers both bag labels and prior information.
2. Differentiable approximation of aggregation functions: Provides differentiable alternatives for aggregation operations such as MIN, MAX, and AVG.
3. Maximum likelihood pseudo-labels: Generates instance-level pseudo-labels consistent with bag labels using the predictions of the trained model, followed by a second round of training.

### Key Designs

1. **Mapping Tasks to MIL/LLP**: Different NLP tasks correspond to different aggregation functions:

    - **Retrieval**: The aggregation function is AVG (the average of sentence-level relevance reflects the overall relevance), which corresponds to the LLP problem.
    - **QA**: The aggregation function is MAX (at least one sentence contains the answer), which corresponds to the MIL problem.
    - **Summarization**: The aggregation function is MIN (all sentences must be entailed), which corresponds to the MIL problem.
    - **Mathematical Reasoning**: The aggregation function is MIN (all steps must be correct for the overall response to be correct), which corresponds to the MIL problem.

2. **Bag Loss with Priors (PriorsBagLoss)**: Introduces two types of prior information on top of standard BagLoss:

    - **Cosine Similarity Prior P1**: Measures the similarity between the sentence embedding and the reference document/query embedding, normalized to [0,1]: $P1(x) = \frac{1}{2}(1 + \frac{\langle x, U \rangle}{\|x\|_2 \|U\|_2})$
    - **Relevance Prior P2**: Measures the Pearson correlation coefficient between sentence pairs: $P2(x, z) = \frac{1}{2}(1 + \rho_{xz})$
   
   The total loss is a weighted combination of the bag loss and the prior losses: $L_{tot} = \lambda L_{totbag} + \lambda_1 L_{totprior1} + \lambda_2 L_{totprior2}$

3. **Differentiable Approximation of Aggregation Functions**:

    - Uses TensorFlow's built-in `tf_reduce_min` for MIN.
    - Derives MAX by applying MIN to the flipped variables.
    - AVG is inherently differentiable.

4. **Maximum Likelihood Pseudo-Labels (PsLab)**: Generates the maximum likelihood label configuration consistent with the bag label for instances in each bag, utilizing the predictions of the trained model $M$. For cases with MIN aggregation and a bag label of 0: first binarize all instances by $M(x) > 0.5$; if all instances are labeled 1, flip the instance with the smallest $M(z)$ to 0 (ensuring at least one 0). Then, retrain the model on the generated pseudo-labels.

5. **Preference Bag Loss (PrefBagLoss)**: For SxS preference labels (pairwise comparisons), the preference loss is defined using the Bradley-Terry model: $L_{pref}(B_1, B_2, y_{B_1 B_2}) = y_{B_1 B_2} \log \frac{y_{B_2}}{y_{B_1}}$

### Loss & Training

Mini-batch training is adopted: at each step, $q$ bags are sampled to calculate the weighted sum of bag loss and prior losses, and the model weights are updated using standard optimizers. For preference label scenarios, bag pairs are sampled for training. Weight hyperparameters in the loss function satisfy $\lambda, \lambda_1, \lambda_2 \in [0, 1]$ and $\lambda + \lambda_1 + \lambda_2 = 1$.

## Key Experimental Results

### Main Results

| Dataset | Task | Metric | FRACTAL | BagLoss | Supervised | Description |
|--------|------|------|---------|---------|------------|------|
| MultiSpanQA | Retrieval | AUC-ROC | **0.693** | 0.661 | 0.729 | Narrows the gap to the supervised upper bound |
| QA-Feedback | Preference QA | AUC-ROC | **0.532** | 0.509 | 0.651 | Preference label scenario |
| AquaMuSe | Summarization | AUC-ROC | **0.814** | 0.751 | 0.876 | Maximum improvement of +6.3% |
| WikiCatSum | Summarization | AUC-ROC | **0.645** | 0.477 | 0.837 | Improvement of +16.8% |
| PRM800K | Mathematical Reasoning | AUC-ROC | **0.597** | 0.569 | 0.613 | Close to the supervised upper bound |
| FirA | Retrieval Regression | MAE↓ | **0.294** | 0.304 | 0.283 | Regression task |

### Ablation Study

| Configuration | Key Metrics | Description |
|------|---------|------|
| BagLoss vs PriorBagLoss | Improvement across most datasets | Prior information is effective |
| PriorBagLoss vs PsLab | PsLab performs better on 4/5 datasets | Second-round training on pseudo-labels is effective |
| cos-sim directly as scorer | Much lower than training methods | Pure prior is insufficient |
| NLI scorer | Performs well on summarization tasks | But requires a T5x-11B model |

### Key Findings

- FRACTAL outperforms the BagLoss baseline on 5 out of 6 datasets, with its performance always falling between BagLoss and Supervised (the fully supervised upper bound) across all datasets.
- Contribution of prior information: The cosine similarity prior is more effective in retrieval and QA tasks, the relevance prior is more effective in summarization tasks, and the combination of both yields the best results on QA-Feedback.
- The PsLab pseudo-labeling method performs best in tasks with deterministic labels (0/1 binary, such as MultiSpanQA and AquaMuSe) and is not applicable to preference labels or continuous scenarios.
- In the PRM800K mathematical reasoning task, FRACTAL scores very close to the fully supervised upper bound (0.597 vs 0.613), indicating that localization of step-level errors in mathematical reasoning is highly sensitive to response-level signals.

## Highlights & Insights

- **Elegant Problem Modeling**: Unifies different NLP tasks under the MIL/LLP framework, with clear definitions and rigorous mathematical derivations.
- **Clever Prior Design**: Document-sentence cosine similarity and inter-sentence correlation are both easy to compute and universally applicable signals, requiring no additional human annotation.
- **Simple and Effective Pseudo-labeling Strategy**: Employs maximum likelihood configuration assignment plus consistency checks, avoiding complex EM or regularization techniques.
- **End-to-End Fine-Tuning Validation**: Evaluates not only the accuracy of sentence-level scoring but also validates that fine-tuning LLMs with the inferred sentence-level labels can yield performance comparable to models trained with human-annotated fine-grained labels.
- **Unified Cross-Task Approach**: The same framework applies to four vastly different tasks: retrieval, QA, summarization, and mathematical reasoning.

## Limitations & Future Work

1. The choice of aggregation functions (MIN/MAX/AVG) relies on manual specification for each task; automatic selection of aggregation functions is worth investigating.
2. The prior information currently only utilizes cosine similarity and Pearson correlation; richer priors (such as NLI scores, syntactic features) can be explored in the future.
3. The PsLab pseudo-labeling method is not applicable to preference labels and continuous label scenarios, which limits its generalizability.
4. Evaluations are mainly conducted on medium-scale datasets, and the scalability to large-scale LLM evaluation scenarios remains to be validated.
5. The quality of pseudo-labels highly depends on the accuracy of the first-stage model, and errors might be amplified in the second round of training.

## Related Work & Insights

- Classic work on MIL began with Dietterich et al. (1997) for drug activity detection and was later extended to domains like information retrieval and medical imaging. FRACTAL is the first to systematically apply it to fine-grained evaluation in NLP scenarios.
- In terms of LLP, the bag loss method in DLLP (Ardehaly & Culotta, 2017) serves as an important baseline.
- It is directly related to the PRM800K work by Lightman et al. (2023)—while the latter collects step-level annotations, this paper studies how to infer step-level scores from aggregate labels.
- Important implications for RLHF/RLAIF: If fine-grained signals can be inferred from easily obtainable response-level feedback, manual annotation costs can be significantly reduced.

## Rating

- Novelty: ⭐⭐⭐⭐ Introducing MIL/LLP into NLP fine-grained evaluation is novel, and the design of priors is creative
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 6 datasets, 4 task types, various baselines and variants, and end-to-end fine-tuning validation
- Writing Quality: ⭐⭐⭐⭐ Mathematically rigorous, but with many symbols and some dense paragraphs
- Value: ⭐⭐⭐⭐ Directly valuable for LLM evaluation and fine-grained feedback in RLHF

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] TROVE: A Challenge for Fine-Grained Text Provenance via Source Sentence Tracing and Relationship Classification](trove_a_challenge_for_finegrained_text.md)
- [\[ACL 2025\] Tuna: Comprehensive Fine-grained Temporal Understanding Evaluation on Dense Dynamic Videos](tuna_temporal_understanding.md)
- [\[ACL 2025\] Guidelines for Fine-grained Sentence-level Arabic Readability Annotation](guidelines_for_fine-grained_sentence-level_arabic_readability_annotation.md)
- [\[ACL 2025\] A Spatio-Temporal Point Process for Fine-Grained Modeling of Reading Behavior](a_spatio-temporal_point_process_for_fine-grained_modeling_of_reading_behavior.md)
- [\[ACL 2025\] Barec: A Large and Balanced Corpus for Fine-grained Arabic Readability Assessment](a_large_and_balanced_corpus_for_fine-grained_arabic_readability_assessment.md)

</div>

<!-- RELATED:END -->
