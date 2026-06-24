---
title: >-
  [Paper Note] UPME: An Unsupervised Peer Review Framework for Multimodal Large Language Model Evaluation
description: >-
  [CVPR 2025][Multimodal VLM][MLLM Evaluation] This paper proposes the UPME framework, which enables multiple MLLMs to generate questions and review each other using only image data through an **unsupervised peer review mechanism**, a **vision-language scoring system**, and **dynamic weight optimization**. It achieves a Pearson correlation of 0.944 with human evaluation on MMStar, effectively mitigating the reliance of MLLM evaluation on human annotations and addressing review…
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "MLLM Evaluation"
  - "Unsupervised Evaluation"
  - "Peer Review"
  - "Vision-Language Scoring"
  - "Dynamic Weight Optimization"
date: 2026-05-08
content_hash: b078f869f33caefd
---

# UPME: An Unsupervised Peer Review Framework for Multimodal Large Language Model Evaluation

**Conference**: CVPR 2025  
**arXiv**: [2503.14941](https://arxiv.org/abs/2503.14941)  
**Code**: Unreleased  
**Area**: Multimodal VLM  
**Keywords**: MLLM Evaluation, Unsupervised Evaluation, Peer Review, Vision-Language Scoring, Dynamic Weight Optimization

## TL;DR
This paper proposes the UPME framework, which enables multiple MLLMs to generate questions and review each other using only image data through an **unsupervised peer review mechanism**, a **vision-language scoring system**, and **dynamic weight optimization**. It achieves a Pearson correlation of 0.944 with human evaluation on MMStar, effectively mitigating the reliance of MLLM evaluation on human annotations and addressing review bias.

## Background & Motivation
The field of MLLM evaluation faces two core challenges. First, traditional VQA benchmark methods (such as MMMU, MMStar) rely on massive human-designed question-answer pairs, which is labor-intensive and limits the evaluation scope. Second, although MLLM-as-a-Judge methods reduce the human burden, they introduce **redundancy bias** (favoring longer answers) and **self-preference bias** (favoring their own outputs), causing the evaluation to deviate from true understanding of visual content. **Key Challenge**: How to achieve objective MLLM evaluations highly consistent with human evaluation completely without human-annotated data? **Key Insight**: This work draws inspiration from academic peer review—letting models generate questions, score each other, and iteratively optimize confidence weights to eliminate the bias of weaker models.

## Method

### Overall Architecture
UPME consists of three core modules: (1) **Peer Review Mechanism**—in each iteration round, two candidate models and one reviewer model are randomly drawn from the MLLM pool, where the reviewer model generates questions based on images and evaluates the candidate models' answers; (2) **Vision-Language Scoring System**—a comprehensive score is calculated from three dimensions: answer correctness, visual understanding & reasoning, and image-text relevance; (3) **Dynamic Weight Optimization**—confidence weights are initialized for each model and iteratively optimized through MSE loss to maximize consistency between weights and estimated scores.

### Key Designs
1. **Peer Review Mechanism**:
    - **Function**: Achieves fully unsupervised multi-model peer review, eliminating reliance on human annotations.
    - **Mechanism**: Given an image set $\mathcal{I}$ and a model pool $\mathcal{M}$, a reviewer model $M_r$ is randomly selected in each round to generate a question $Q_i^r = M_r(I_i)$ for an image $I_i$. Two candidate models $M_j$ and $M_k$ answer the question, and the reviewer model evaluates them via pairwise comparison using the vision-language scoring system $S_{VL}$.
    - **Design Motivation**: Pairwise comparison is more accurate than independent scoring (supported by existing studies). Randomly rotating roles allows every model to act as both a candidate and a reviewer, converging towards a stable evaluation over many rounds.

2. **Vision-Language Scoring System**:
    - **Function**: Establishes a multi-dimensional scoring standard, compensating for the neglect of visual content in text-only peer reviews.
    - **Mechanism**: The final score is a weighted sum of three terms: $S_{VL} = \gamma_1 S_{Correct} + \gamma_2 S_{Visual} + \gamma_3 S_{Clip}$. Here, $S_{Correct}$ evaluates answer correctness based on pairwise comparison outcomes (1/0.5/0); $S_{Visual}$ comprehensively assesses four visual dimensions—description, reasoning, localization, and relations—via function $\Gamma$; $S_{Clip}$ utilizes a CLIP model to compute image-text alignment scores.
    - **Design Motivation**: The introduction of the CLIP score is a key novelty. Acting as an objective indicator independent of the reviewer model, it effectively mitigates redundancy bias (longer answers do not necessarily yield higher CLIP scores) and self-preference bias (CLIP does not favor any specific model).

3. **Dynamic Weight Optimization**:
    - **Function**: Leverages iterative optimization to assign higher review weights to stronger models, thereby improving overall evaluation accuracy.
    - **Mechanism**: The estimated score of each model is represented as $\hat{G}_{M_j} = \sum_i \sum_{k \neq j} \sum_{r \neq k, r \neq j} \text{Review}_i^{j,k,r} \times w_r$. An MSE loss $\mathcal{L}_{MSE} = \frac{1}{m} \sum_{j=1}^{m} (\hat{G}_{M_j} - w_{M_j})^2$ is used to iteratively update the consistency between weights $w$ and scores $\hat{G}$.
    - **Design Motivation**: Preliminary experiments indicate that allocating higher weights to stronger models significantly improves evaluation accuracy. Dynamic optimization automatically discovers this weight assignment.

## Key Experimental Results

### Main Results (Pearson / Spearman Correlation with Human Evaluation)

| Method | MMStar Pearson↑ | MMStar Spearman↑ | ScienceQA Pearson↑ | ScienceQA Spearman↑ |
|------|----------------|------------------|--------------------|--------------------|
| GPT-4o (Single Model Reviewer) | 0.878 | 0.875 | 0.617 | 0.625 |
| Peer Review (Original) | 0.725 | 0.771 | 0.463 | 0.686 |
| Majority Vote | 0.757 | 0.757 | 0.509 | 0.524 |
| Rating Vote | 0.795 | 0.743 | 0.623 | 0.629 |
| PRD (Semi-supervised) | 0.838 | 0.864 | 0.636 | 0.694 |
| **UPME** | **0.944** | **0.972** | **0.814** | **0.886** |

### Ablation Study

| Scoring Components | MMStar Pearson | ScienceQA Pearson |
|----------|----------------|-------------------|
| Correctness only | 0.854 | 0.713 |
| Visual only | 0.873 | 0.701 |
| CLIP only | 0.785 | 0.548 |
| Visual + CLIP | 0.903 | 0.775 |
| All (UPME) | **0.944** | **0.814** |

### Human Preference Alignment

| Dataset | Method | Human Agreement Rate | Model Agreement Rate |
|--------|------|-----------|-----------|
| MMStar | Peer Review | 71.1% | 67.5% |
| MMStar | UPME | **95.9%** | **89.8%** |
| ScienceQA | Peer Review | 68.2% | 61.8% |
| ScienceQA | UPME | **87.4%** | **82.6%** |

### Key Findings
- Stable convergence is achieved with a sample size of just 25 images. Further increasing the sample size (up to 100) results in negligible metric changes.
- The average review accuracy of all models in UPME increases from 61.56% to 74.48% (correctness dimension), and reaches 73.93% in the visual understanding dimension.
- Redundancy bias and self-preference bias are effectively suppressed in UPME (p-values from Chi-Square tests show no significance).

## Highlights & Insights
- **Truly Unsupervised Evaluation**: No human-annotated QA pairs are required; evaluation is completed using only images. This marks an important paradigm shift in the evaluation field.
- **Clever Incorporation of CLIP Score**: Serving as a third-party objective referee, CLIP introduces image-text alignment signals external to the reviewer model, which is key to eliminating bias.
- **Guaranteed Convergence**: UPME converges within 30 epochs across 64 different initialization settings, demonstrating the robustness and reliability of the framework.
- **Exceptionally High Alignment with Human Preferences**: A 95.9% human agreement rate on MMStar indicates that UPME can practically substitute human evaluation.

## Limitations & Future Work
- The model pool contains only 6 models (5 closed-source + 1 open-source); the effectiveness with larger-scale model pools remains to be validated.
- CLIP scores themselves may be unreliable in certain scenarios (e.g., fine-grained spatial relationship understanding).
- The current framework lacks explicit control over the quality of questions generated by the reviewer models; weaker models might generate lower-quality questions.
- The correlation on ScienceQA (0.814) is lower than on MMStar (0.944), illustrating room for improvement in scientific problems requiring deep reasoning.

## Related Work & Insights
- Inherits the "user voting" philosophy of Chatbot Arena, but replaces human voting with automated model peer reviews.
- The strategy of using CLIP scores as a bias corrector can be generalized to other MLLM evaluation scenarios.
- Insight: When building new MLLM evaluation benchmarks, UPME can be deployed first to quickly filter valuable evaluation images before conducting manual annotation.

## Supplementary Analysis
- The computational cost of UPME mainly stems from multi-round model inference (requiring 3 models per round). The API call cost for 25 images across multiple iterations is approximately 1/10 of a single benchmark evaluation.
- The framework exhibits strong fault tolerance toward the weakest models in the pool. Experiments show that even when including LLaMA-3.2-11B (with a Pearson correlation of only 0.314), the overall evaluation quality remains at 0.944.
- The essence of dynamic weight optimization is a self-consistency constraint—strong models naturally perform better both when being reviewed and when reviewing others, a consistency captured mathematically.

## Rating
- **Novelty**: ⭐⭐⭐⭐ First fully unsupervised peer-review framework for MLLMs; the combination of peer review and dynamic weight optimization is highly innovative.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Ablation, bias analysis, and human preference alignment experiments are complete, although the model pool scale is relatively small.
- **Writing Quality**: ⭐⭐⭐⭐ Clear framework description, standardized mathematical notation, and intuitive diagrams.
- **Value**: ⭐⭐⭐⭐ Provides a cost-effective and efficient pathway for MLLM evaluation, but its practicality requires larger-scale validation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] PRISMM-Bench: A Benchmark of Peer-Review Grounded Multimodal Inconsistencies](../../ICLR2026/multimodal_vlm/prismm-bench_a_benchmark_of_peer-review_grounded_multimodal_inconsistencies.md)
- [\[ACL 2025\] FlagEvalMM: A Flexible Framework for Comprehensive Multimodal Model Evaluation](../../ACL2025/multimodal_vlm/flagevalmm_a_flexible_framework_for_comprehensive_multimodal_model_evaluation.md)
- [\[CVPR 2025\] Distraction is All You Need for Multimodal Large Language Model Jailbreaking](distraction_is_all_you_need_for_multimodal_large_language_model_jailbreaking.md)
- [\[CVPR 2025\] Period-LLM: Extending the Periodic Capability of Multimodal Large Language Model](period-llm_extending_the_periodic_capability_of_multimodal_large_language_model.md)
- [\[CVPR 2025\] CoLLM: A Large Language Model for Composed Image Retrieval](collm_a_large_language_model_for_composed_image_retrieval.md)

</div>

<!-- RELATED:END -->
