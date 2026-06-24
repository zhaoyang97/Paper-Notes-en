---
title: >-
  [Paper Note] Probing the Geometry of Truth: Consistency and Generalization of Truth Directions
description: >-
  [ACL 2025][Interpretability][truth direction] This work systematically investigates the consistency and generalization of the internal "truth direction" in LLMs, finding that only highly capable models stably exhibit a consistent truth direction, and that truthfulness probes trained on simple atomic statements can generalize to logical transformations, question-answering tasks, and in-context knowledge scenarios.
tags:
  - "ACL 2025"
  - "Interpretability"
  - "truth direction"
  - "truthness probing"
  - "LLM internal representation"
  - "linear probe"
  - "generalization"
date: 2026-05-08
content_hash: a9d5a2671669fc8c
---

# Probing the Geometry of Truth: Consistency and Generalization of Truth Directions

**Conference**: ACL 2025  
**arXiv**: [2506.00823](https://arxiv.org/abs/2506.00823)  
**Authors**: Yuntai Bao, Xuhong Zhang, Tianyu Du, Xinkui Zhao, Zhengwen Feng, Hao Peng, Jianwei Yin (Zhejiang University, Zhejiang Normal University)  
**Code**: [GitHub](https://github.com/colored-dye/truthfulness_probe_generalization)  
**Area**: Interpretability  
**Keywords**: truth direction, truthness probing, LLM internal representation, linear probe, generalization

## TL;DR

This work systematically investigates the consistency and generalization of the internal "truth direction" in LLMs, finding that only highly capable models stably exhibit a consistent truth direction, and that truthfulness probes trained on simple atomic statements can generalize to logical transformations, question-answering tasks, and in-context knowledge scenarios.

## Background & Motivation

### Background
After being trained on large-scale corpora, LLMs possess rich knowledge, but their outputs often contain confidently asserted untruths. Prior work (Burns et al. 2022; Marks & Tegmark 2023) discovered a linear feature inside LLMs—the "truth direction"—allowing the categorization of statement truthfulness from the model's hidden states using lightweight classifiers (probes).

### Limitations of Prior Work
- Prior works generally **assume that a consistent truth direction exists in all LLMs**, lacking a systematic validation of this assumption.
- Levinstein & Herrmann (2024) claim that probes fail to generalize across logical negation, but attribute this failure to the lack of complexity in the probe design.
- The **syntactic form generalization** of truthfulness probes from declarative statements to question-answering scenarios has not been sufficiently studied.
- Whether probes can generalize from **parametric knowledge** to **in-context knowledge** (e.g., reading comprehension, summary generation) remains unclear.

### Core Problem
- **RQ1**: Do LLMs universally represent truthfulness as a linear feature?
- **RQ2**: Are complex probing techniques required to identify the truth direction?
- **RQ3**: To what extent does the truth direction generalize?

## Method

### Probe Formalization

Given a Transformer language model, an input token sequence $t = (t_1, t_2, \ldots, t_n)$ is processed through $L$ layers to obtain representations $\boldsymbol{h}_i^{(l)} \in \mathbb{R}^d$ for each layer. For autoregressive models, the representation at the last token position of the $l$-th layer $\boldsymbol{h}_{-1}^{(l)}$ is used as the probe input. Given a labeled dataset $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^M$, the goal is to learn a probe $\Phi$ to minimize classification error:

$$\underset{\Phi}{\arg\min} \frac{1}{M} \sum_{i=1}^{M} J(\Phi, \boldsymbol{h}_i, y_i)$$

### Key Designs

#### Design 1: Geometry-Oriented Probes

Based on the "truth direction hypothesis"—that true/false representations can be separated by a hyperplane, with the hyperplane's normal vector being the truth direction:

- **Linear SVM Probe**: Maximizes the margin of separation, using Platt scaling posterior calibration to obtain probability outputs. Implemented using NuSVC with $\nu=0.5$, and 5-fold cross-validation for Platt scaling.
- **Mass-Mean (MM) Probe**: Computes the centroids of true and false representations, using the direction of the line connecting the centroids as the truth direction (Marks & Tegmark 2023). Classification is based on distance voting to the two centroids.

#### Design 2: Statistics-Oriented Probes

Without assuming any geometric structure, directly maximizes label likelihood:

- **Logistic Regression (LR)**: Optimized using L-BFGS, serving as a general baseline.
- **Multi-Layer Perceptron (MLP/SAPLMA)**: Hidden layer structure $(512, 128, 64)$, tanh activation, trained with the Adam optimizer until convergence.

### Optimal Layer Selection Strategy
The optimal layer is selected by computing the **ratio of between-class variance to within-class variance** for each layer. For Llama-3.1-8B, the 12th layer (zero-indexed) has the highest variance ratio; for Llama-2-7B, a distinct peak appears only in the sp_en_trans topic, suggesting inconsistent truth directions in weaker models.

### Dataset Construction
Employs the analytical factual statements dataset curated by Bürger et al. (2024), covering 6 topics: animal_class, cities, element_symb, facts, inventors, and sp_en_trans. Each topic contains four variants: affirmative statements, negative statements, logical conjunctions, and logical disjunctions.

## Key Experimental Results

### Experiment 1: Consistency of Truth Directions (Generalization Across Logical Negation)

Probes are trained on affirmative statements and tested on negative statements. Models are listed below from weakest to strongest:

| Model | Number of Topics with Successful Generalization (/6) | Model Size |
|------|---------------------|-----------|
| Llama-2-7B | 0/6 | 7B |
| Llama-2-7B-Chat | 0/6 | 7B |
| Llama-2-13B | 4/6 | 13B |
| Llama-2-13B-Chat | 4/6 | 13B |
| Llama-3.1-8B | 4/6 | 8B |
| Llama-3.1-8B-Instruct | 4/6 | 8B |
| Llama-3.1-70B | 5/6 | 70B |
| Llama-3.1-70B-Instruct | **6/6** | 70B |

Conclusion: The consistency of the truth direction positively correlates with model capability, and only the strongest models exhibit a consistent truth direction across all topics.

### Experiment 2: Generalization to QA Tasks (MMLU & TriviaQA)

Probes are trained on atomic factual statements and tested on MMLU and TriviaQA (using Llama-3.1-8B):

| Dataset | Prompt Setting | SVM AUROC↑ | SVM ECE↓ | SVM BS↓ |
|--------|-----------|------------|----------|---------|
| MMLU | zero-shot | ~0.60 | ~0.15 | ~0.24 |
| MMLU | TTTTT (5-shot all correct) | ~0.65 | ~0.12 | ~0.22 |
| MMLU | TTFFF (with incorrect exemplars) | ~0.65 | ~0.12 | ~0.22 |
| TriviaQA | 5-shot | ~0.70 | ~0.15 | ~0.22 |
| TriviaQA | 20-shot | ~0.72 | ~0.10 | ~0.20 |

Key Finding: The performance with incorrect few-shot exemplars is nearly identical to that with all-correct exemplars—the probe focus remains solely on the truthfulness of the final (Q, A) pair, treating preceding exemplars merely as context.

### Experiment 3: Selective QA Application

Using the TriviaQA 20-shot setting, an SVM probe is used to filter LLM answers:

| Metric | Value |
|------|------|
| Accuracy of all answers | 55.29% |
| Proportion judged as true by the probe | 80.26% |
| Accuracy of the filtered subset | **64.06%** |

Accuracy is improved by approximately 9 percentage points through probe filtering.

### Experiment 4: Generalization to In-Context Knowledge

Probes are trained on atomic factual statements and tested on tasks requiring in-context knowledge (using Llama-3.1-8B):

| Dataset | Task Type | Generalization Result |
|--------|---------|---------|
| SciQ | In-context multi-choice QA | AUROC > 0.5, successfully generalized |
| BoolQ | In-context yes/no QA | AUROC > 0.5, successfully generalized |
| XSum | Summary faithfulness detection | AUROC > 0.5, successfully generalized |

## Key Findings

1. **Not all LLMs have a consistent truth direction**: Llama-2-7B fails to generalize across negation at all, whereas Llama-3.1-70B-Instruct generalizes perfectly across all topics.
2. **Simple probes are sufficient to identify the truth direction**: When the model is powerful enough, performance differences among the four simple probes (LR, SVM, MLP, MM) are negligible.
3. **The truth direction is a product of pre-training**: The probe AUROC on a randomly initialized model is around 0.5 (random chance), while it reaches 1.0 on pre-trained weights.
4. **Generalization on logical conjunctions is superior to logical disjunctions**: This is likely because computing the truth value of disjunctions is more challenging for LLMs.
5. **Probes are robust to incorrect few-shot exemplars**: The probe extracts truthfulness only from the final (Q, A) pair, unaffected by erroneous instances in the preceding context.
6. **Out-of-distribution probes outperform in-distribution probes**: On MMLU, probes trained on atomic statements surprisingly outperform probes trained within the MMLU domain itself.

## Highlights & Insights

- **Systematically addresses three core questions**: Instead of assuming the universal existence of a truth direction, it provides data-backed answers through extensive experiments over 8 models, 4 probe types, and multiple tasks.
- **Challenges previous conclusions**: Demonstrates that the generalization failure in Levinstein & Herrmann (2024) is not a probing issue, but rather stems from the model itself lacking consistent truth representations.
- **Practical application demonstration**: Showcases the practical value of probes in the selective QA scenario, which yields increased response reliability without modifying the model parameters.
- **Rigorous experimental design**: Experiments on randomized models rule out the possibility of probes "factoring in" a non-existent truth direction; few-shot experiments with incorrect exemplars reveal the underlying robustness mechanisms of the probes.

## Limitations & Future Work

- **Vague definition of "truthfulness"**: Probes might capture widely held human consensus rather than objective facts, making their applicability to superhuman AI systems questionable.
- **Evaluation limited to short-text QA**: More complex scenarios such as long-form QA and instruction following are not covered.
- **Unclear causal relationship**: There is no proof that LLMs actually utilize the truth direction during generation; the study only demonstrates correlation.
- **Computational resource constraints**: The largest evaluated model is only 70B, leaving the hypotheses unverified on stronger models like GPT-4.
- **Predominant focus on the Llama series**: Mistral is only evaluated as a supplementary appendix, lacking a broader comparison across model families.

## Related Work & Insights

- **Burns et al. (2022, CCS)**: Unsupervised method designed for yes/no QA; this work uses supervised probes and evaluates a wider range of generalization scenarios.
- **Marks & Tegmark (2023)**: Introduced the concept of truth direction and the MM probe; this work systematically validates its consistency and generalization boundaries.
- **Bürger et al. (2024, TTPD)**: Proposed the TTPD probe to distinguish between affirmative and general truth directions; this work finds that such a distinction is unnecessary in strong models.
- **Levinstein & Herrmann (2024)**: Claimed that probes fail to generalize across negation; this work shows that this is an issue of model capability rather than the probe design.
- **Azaria & Mitchell (2023, SAPLMA)**: Used MLP probes; this work demonstrates that simple linear probes are equally effective.
- **Sky et al. (2024)**: Detected hallucinations in generation with context; this work generalizes probes trained on factual statements to in-context tasks.

## Rating

- Novelty: ⭐⭐⭐⭐ — Systematically answers three open questions regarding the truth direction, challenging previous beliefs.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Comprehensive evaluation covering 8 models, 4 probe designs, 6 topic areas, and varied downstream tasks.
- Writing Quality: ⭐⭐⭐⭐ — Well-structured, with clearly formulated research questions and layered experimental design.
- Value: ⭐⭐⭐⭐ — Highly informative for evaluating LLM trustworthiness and safety alignment; the selective QA application highlights practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Emergence of Linear Truth Encodings in Language Models](../../NeurIPS2025/interpretability/emergence_of_linear_truth_encodings_in_language_models.md)
- [\[ACL 2026\] How Context Shapes Truth: Geometric Transformations of Statement-level Truth Representations in LLMs](../../ACL2026/interpretability/how_context_shapes_truth_geometric_transformations_of_statement-level_truth_repr.md)
- [\[ACL 2025\] Probing Subphonemes in Morphology Models](probing_subphonemes_in_morphology_models.md)
- [\[ICLR 2026\] Diagnosing Generalization Failures from Representational Geometry Markers](../../ICLR2026/interpretability/diagnosing_generalization_failures_from_representational_geometry_markers.md)
- [\[NeurIPS 2025\] Representation Consistency for Accurate and Coherent LLM Answer Aggregation](../../NeurIPS2025/interpretability/representation_consistency_for_accurate_and_coherent_llm_answer_aggregation.md)

</div>

<!-- RELATED:END -->
