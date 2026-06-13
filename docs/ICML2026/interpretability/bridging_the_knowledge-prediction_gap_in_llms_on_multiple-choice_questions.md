---
title: >-
  [Paper Note] Bridging the Knowledge-Prediction Gap in LLMs on Multiple-Choice Questions
description: >-
  [ICML 2026][Interpretability][Knowledge-prediction gap] This paper reveals a widespread "knowledge-prediction gap" in LLMs on multiple-choice questions (MCQs)—where hidden layers linearly encode the correct answer while…
tags:
  - "ICML 2026"
  - "Interpretability"
  - "Knowledge-prediction gap"
  - "linear probes"
  - "subspace alignment"
  - "inference-time intervention"
  - "multiple-choice questions"
date: 2026-05-08
content_hash: ecaa0d8779f6e10f
---

# Bridging the Knowledge-Prediction Gap in LLMs on Multiple-Choice Questions

**Conference**: ICML 2026  
**arXiv**: [2509.23782](https://arxiv.org/abs/2509.23782)  
**Code**: https://github.com/holi-lab/KAPPA  
**Area**: Interpretability  
**Keywords**: Knowledge-prediction gap, linear probes, subspace alignment, inference-time intervention, multiple-choice questions  

## TL;DR

This paper reveals a widespread "knowledge-prediction gap" in LLMs on multiple-choice questions (MCQs)—where hidden layers linearly encode the correct answer while final predictions deviate. Through geometric analysis, this gap is attributed to the misalignment between knowledge and prediction subspaces. The authors propose KAPPA, an inference-time intervention that uses closed-form affine transformations to align these subspaces, consistently narrowing the gap and improving accuracy across models and benchmarks.

## Background & Motivation

**Background**: Evaluating LLMs on multiple-choice question (MCQ) benchmarks is a standard practice. However, models frequently exhibit "capability inconsistency"—providing the correct answer in free-generation scenarios while failing in MCQ formats for the same question. Prior research indicates that even when a model fails, correct answers can be extracted via simple linear classifiers applied to hidden layers, suggesting that sufficient knowledge is already encoded internally.

**Limitations of Prior Work**: Previous studies largely attributed MCQ errors to "surface factors" such as option bias, surface cues, or stylistic artifacts, but lacked an explanatory framework unifying these failures with internal representations. Research on the "knowledge-prediction gap" has also been confined to narrow scenarios like truthfulness detection or simple arithmetic, failing to generalize to diverse MCQ tasks.

**Key Challenge**: The residual stream of a model simultaneously encodes the correct answer (knowledge signal) and the actual output answer (prediction signal) linearly. However, these signals are routed along geometrically distinct directions, leading the prediction signal to "override" the knowledge signal during final generation. This is not a lack of knowledge, but a failure in knowledge utilization.

**Goal**: (1) Quantify the prevalence and severity of the knowledge-prediction gap across diverse MCQ benchmarks and model families; (2) Explain the structural causes of the gap from the perspective of residual stream geometry; (3) Design an inference-time intervention requiring no additional training to bridge the gap.

**Key Insight**: By training two linear probes—a knowledge probe to predict the ground truth and a prediction probe to predict model outputs—the respective weight matrices can be viewed as basis vectors defining subspaces. If the two subspaces are aligned, model predictions should yield consistency with internal knowledge; empirical results show significant principal angle deviation between these subspaces on benchmarks with large gaps.

**Core Idea**: Use minimal $l_2$ perturbation to correct the coordinates of hidden states in the prediction subspace to match their coordinates in the knowledge subspace—essentially "aligning prediction to knowledge."

## Method

### Overall Architecture

Given an MCQ input, KAPPA extracts the hidden state $h$ at intermediate layers of the residual stream. It projects $h$ into two $k$-dimensional subspaces defined by the weights of the knowledge and prediction probes to calculate their respective coordinates (probe logits). When the two sets of coordinates are inconsistent, KAPPA applies a closed-form affine transformation to $h$, aligning its coordinates in the prediction subspace with those in the knowledge subspace. The modified $h'$ is then written back into the residual stream for continued forward propagation. The entire process requires no gradient updates and relies solely on two sets of pre-trained linear probe weights.

### Key Designs

1.  **Quantifying the Knowledge-Prediction Gap via Dual Probes**:
    *   **Function**: Characterize the signal strength and direction of "what the model knows" versus "what the model outputs" in the residual stream.
    *   **Mechanism**: Activation $h^l(x)$ is extracted at each layer $l$. Two datasets are constructed: $D_{\text{know}}^{(l)} = \{(h^l(x), y)\}$ paired with ground-truth labels, and $D_{\text{pred}}^{(l)} = \{(h^l(x), \tilde{y})\}$ paired with model-predicted labels. These are used to train $k$-class linear classifiers. Two complementary metrics are introduced: the Agreement Rate $\text{AGR}(x) = \mathbb{I}[\arg\max p_K(x) = \arg\max p_M(x)]$ to measure decision-level alignment, and KL Divergence $\text{KLD}(x) = \text{KL}(p_M \| p_K)$ to measure distribution-level alignment.
    *   **Design Motivation**: Accuracy alone cannot compare the gap size across benchmarks and models. AGR captures hard differences in "correct/incorrect" choices, while KLD captures soft differences in confidence distributions.

2.  **Subspace Geometric Analysis**:
    *   **Function**: Provide a mechanistic geometric attribution for the knowledge-prediction gap.
    *   **Mechanism**: Column vectors of each probe's weight matrix $W \in \mathbb{R}^{d \times k}$ are treated as subspace bases. The alignment between knowledge and prediction subspaces is measured using mean principal angles and CKA. Experiments show that in deep layers, principal angles approach $90^\circ$ (near random baseline), and CKA stays in the mid-range of 0.4–0.8, indicating that both signals coexist in the same residual stream but propagate along different geometric directions. Spearman correlation analysis across 8 benchmarks shows that the degree of subspace misalignment is highly correlated with the measured gap (Llama 3.1 8B: $\rho = 0.976, p = 0.001$).
    *   **Design Motivation**: Anchorage of the abstract phenomenon "model ignores its own knowledge" to geometric structures provides a theoretical basis for subsequent alignment interventions.

3.  **KAPPA Inference-time Alignment Intervention**:
    *   **Function**: Modify hidden states during inference to ensure model predictions are faithful to internally encoded knowledge.
    *   **Mechanism**: A constrained optimization problem is established: $\min_{\tilde{h}'} \|\tilde{h}' - \tilde{h}\|_2^2$ subject to $\tilde{W}_{\text{pred}}^\top \tilde{h}' = \tilde{W}_{\text{know}}^\top \tilde{h}$. The closed-form solution is $h' = h + W_{\text{pred}}(W_{\text{pred}}^\top W_{\text{pred}})^{-1}(\tilde{W}_{\text{know}}^\top \tilde{h} - \tilde{W}_{\text{pred}}^\top \tilde{h})$. An extended version introduces hyperparameters $\alpha, \beta$ to control alignment strength: $\tilde{W}_{\text{pred}}^\top \tilde{h}' = \alpha \cdot \tilde{W}_{\text{know}}^\top \tilde{h} + \beta \cdot \text{sign}(\tilde{W}_{\text{know}}^\top \tilde{h})$, where $\alpha$ amplifies relative differences between options and $\beta$ pushes each option logit toward extremes.
    *   **Design Motivation**: Unlike fixed-direction activation steering, KAPPA dynamically calculates the minimal perturbation for each input, only modifying components within the prediction subspace while preserving information in orthogonal directions. The closed-form solution avoids iterative optimization and incurs negligible overhead.

## Key Experimental Results

### Main Results

On six benchmarks with significant gaps, KAPPA consistently improves ACC and AGR across models:

| Benchmark (#Options) | Model | Base ACC | KAPPA(6) ACC | Δ ACC | Base AGR | KAPPA(6) AGR |
|---|---|---|---|---|---|---|
| TruthfulQA (4) | Llama 3.1 8B | 56.7 | 73.5 | +16.8 | 62.1 | 77.6 |
| TruthfulQA (4) | Qwen 2.5 7B | 58.8 | 64.1 | +5.3 | 61.8 | 67.3 |
| BBQ-Age (3) | Llama 3.1 8B | 59.9 | 76.8 | +16.9 | 59.2 | 81.1 |
| BBH-Algo (4) | Llama 3.1 8B | 45.1 | 50.1 | +5.0 | 62.1 | 82.5 |
| GSM8k (4) | Llama 3.1 8B | 32.6 | 36.6 | +4.0 | 53.7 | 75.9 |
| BBH-NLP (4) | Qwen 2.5 7B | 61.1 | 63.6 | +2.5 | 69.8 | 74.9 |

Cross-model TruthfulQA results (KAPPA(6) vs Base):

| Model | Base ACC | KAPPA(6) ACC | Base AGR | KAPPA(6) AGR |
|---|---|---|---|---|
| Mistral 7B v0.3 | 40.7 | 58.3 | 46.6 | 62.3 |
| Llama 3.1 8B | 56.7 | 73.5 | 62.1 | 77.6 |
| Qwen 2.5 7B | 58.8 | 64.1 | 61.8 | 67.3 |
| Qwen3 4B | 56.5 | 61.4 | 60.0 | 66.1 |
| Qwen3 14B | 71.6 | 77.7 | 76.0 | 83.7 |

### Ablation Study

| Analysis Dimension | Key Metric | Description |
|---|---|---|
| Comparing CAA/DoLA | KAPPA superior in 12/12 settings | Existing interventions fail to systematically reduce the gap. |
| Intervention Layers (1/3/6) | 6 layers > 3 layers > 1 layer | Cumulative effects of multi-layer intervention are stronger. |
| $\alpha, \beta$ Hyperparam Sweep | Increasing $\alpha$ or $\beta$ monotonically boosts AGR | Both hyperparameters causally control alignment strength. |
| Sensitivity to training data | 10% data still outperforms Base | Effective even in low-data scenarios. |
| Cross-dataset transfer | TruthfulQA → BBQ-Age: +5.72 AGR | Subspaces are partially shared between tasks requiring similar skills. |
| Free-gen transfer | TruthfulQA ACC: 41.7 → 44.2 | MCQ probes can generalize to open-ended generation. |

### Key Findings

*   The knowledge-prediction gap is largest in truthfulness/bias benchmarks (knowledge probes perform +19–21 points higher than the model in TruthfulQA), followed by reasoning benchmarks, and minimal in knowledge-intensive benchmarks.
*   The degree of subspace misalignment is highly correlated with the gap ($\rho = 0.976$), confirming the geometric roots of the phenomenon.
*   KAPPA does not directly modify the logits of answer tokens (the principal angle between the prediction subspace at intervention layers and the logit space is approximately $65^\circ$–$70^\circ$). Instead, it indirectly influences subsequent decisions by modifying abstract representations in intermediate layers.

## Highlights & Insights

*   **Closed-form Minimal Perturbation Alignment**: Modeling knowledge-prediction alignment as a constrained optimization problem and deriving a closed-form solution allows for negligible computational overhead without iterative optimization. The philosophy of "using math where gradients are not needed" is highly instructive for inference-time interventions.
*   **Dual Probes as Diagnostic Tools**: Training two linear probes with different targets on the same hidden state, then comparing their subspace geometries, provides a general framework for diagnosing "internal signal divergence" within models. This can be transferred to tasks like hallucination detection and alignment auditing.
*   **Cross-format Generalization**: Probes and intervention strategies trained on MCQs can transfer to free-form generation, suggesting that intermediate subspaces encode abstract semantic directions rather than specific answer tokens. This deepens the understanding of the hierarchical structure of LLM internal representations.

## Limitations & Future Work

*   Only linearly accessible knowledge signals are addressed; deeper knowledge encoded non-linearly remains untouched.
*   Probe training requires labeled data and model-predicted labels, making it unsuitable for fully black-box scenarios.
*   Performance gain in free-form generation transfer is limited (GSM8k free-gen accuracy slightly decreased by 0.9 points), suggesting differences still exist between MCQ and open-generation subspaces.
*   Future Work: High-dimensional non-linear alignment, unsupervised probe discovery, and joint use with Chain-of-Thought (CoT) prompting to simultaneously bridge gaps at both the reasoning and representation levels.

## Related Work & Insights

*   **Knowledge-Prediction Gap Literature**: Marks & Tegmark (2024) first discovered that hidden layers could extract correct answers in truthfulness tasks; this paper generalizes the phenomenon to general MCQs and provides a geometric explanation.
*   **Inference-time Intervention**: CAA (Rimsky et al., 2024) uses mean difference vectors for activation steering, while DoLA (Chuang et al., 2024) contrasts logits across layers. Neither is designed specifically for the knowledge-prediction gap, and experiments show their limited effectiveness here.
*   **Mechanistic Interpretability**: Results align with findings by Geva et al. (2023) and Park et al. (2024) regarding how high-level features in the residual stream are transformed by subsequent layers into token predictions. KAPPA's effectiveness further supports this information flow landscape.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] PINE: Pruning Boosted Tree Ensembles with Conformal In-Distribution Prediction Equivalence](pine_pruning_boosted_tree_ensembles_with_conformal_in-distribution_prediction_eq.md)
- [\[ICLR 2026\] Bridging Explainability and Embeddings: BEE Aware of Spuriousness](../../ICLR2026/interpretability/bridging_explainability_and_embeddings_bee_aware_of_spuriousness.md)
- [\[ACL 2026\] Rhetorical Questions in LLM Representations: A Linear Probing Study](../../ACL2026/interpretability/rhetorical_questions_in_llm_representations_a_linear_probing_study.md)
- [\[ICML 2026\] MAAT: Knowledge-Guided Kernel Regression for Heterogeneous Partially Observed State Reconstruction](knowledge-informed_kernel_state_reconstruction_from_heterogeneous_partial_observ.md)
- [\[ACL 2026\] Tracing Relational Knowledge Recall in Large Language Models](../../ACL2026/interpretability/tracing_relational_knowledge_recall_in_large_language_models.md)

</div>

<!-- RELATED:END -->
