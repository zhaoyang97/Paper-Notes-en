---
title: >-
  [Paper Note] Bridging the Knowledge-Prediction Gap in LLMs on Multiple-Choice Questions
description: >-
  [ICML 2026][Interpretability][Paper Note] This paper reveals a ubiquitous "knowledge-prediction gap" in LLMs on multiple-choice questions (MCQs)—where hidden layers linearly encode the correct answer, but the final prediction deviates. Through geometric analysis, this gap is attributed to the misalignment between the knowledge subspace and the prediction subsp
tags:
  - ICML 2026
  - Interpretability
date: 2026-05-08
content_hash: b88cc99dabff0f0d
---
# Bridging the Knowledge-Prediction Gap in LLMs on Multiple-Choice Questions

**Conference**: ICML 2026  
**arXiv**: [2509.23782](https://arxiv.org/abs/2509.23782)  
**Code**: https://github.com/holi-lab/KAPPA  
**Area**: Interpretability  
**Keywords**: Knowledge-Prediction Gap, Linear Probes, Subspace Alignment, Inference-time Intervention, Multiple-Choice Questions  

## TL;DR

This paper reveals a ubiquitous "knowledge-prediction gap" in LLMs on multiple-choice questions (MCQs)—where hidden layers linearly encode the correct answer, but the final prediction deviates. Through geometric analysis, this gap is attributed to the misalignment between the knowledge subspace and the prediction subspace. The authors propose KAPPA, a method that uses closed-form affine transformations at inference time to align these two subspaces, consistently narrowing the gap and improving accuracy across various models and benchmarks.

## Background & Motivation

**Background**: Evaluating LLMs on MCQ benchmarks is a mainstream practice. However, models frequently exhibit "capability inconsistency"—providing the correct answer in free-form generation but failing when switched to an MCQ format. Existing research indicates that even when a model fails, correct answers can be extracted by applying simple linear classifiers to its hidden layers, implying that sufficient knowledge is already encoded internally.

**Limitations of Prior Work**: Previous work primarily attributed MCQ errors to "surface factors" such as option bias, surface cues, or stylistic artifacts, lacking an explanatory framework that unified these failures with internal representations. Research on the "knowledge-prediction gap" has also been limited to narrow scenarios like truthfulness detection and simple arithmetic, failing to generalize to diverse MCQ tasks.

**Key Challenge**: Correct answers (knowledge signals) and actual output answers (prediction signals) are both linearly encoded within the residual stream, but these signals are routed along geometrically distinct directions. Consequently, the prediction signal "overrides" the knowledge signal during final generation. This represents a failure in knowledge utilization rather than a lack of knowledge.

**Goal**: (1) Quantify the prevalence and severity of the knowledge-prediction gap across multiple MCQ benchmarks and model families; (2) Explain the structural causes of the gap from the perspective of residual stream geometry; (3) Design an inference-time intervention that requires no additional training to bridge this gap.

**Key Insight**: By training two linear probes—a knowledge probe to predict the ground truth and a prediction probe to predict the model's output—their respective weight matrices can be viewed as basis vectors defining subspaces. If the two subspaces are aligned, model predictions should be consistent with internal knowledge. In practice, however, benchmarks with large gaps show severe deviation in the mean principal angles between these two subspaces.

**Core Idea**: Use minimum $\ell_2$ perturbation to correct the coordinates of hidden states in the prediction subspace to match their coordinates in the knowledge subspace—effectively "aligning prediction to knowledge."

## Method

### Overall Architecture

Given an MCQ input, KAPPA extracts the hidden state $h$ at an intermediate layer of the residual stream. It projects $h$ into two $k$-dimensional subspaces defined by the weights of a knowledge probe and a prediction probe, calculating the respective coordinates (probe logits). When these two sets of coordinates are inconsistent, KAPPA applies a closed-form affine transformation to $h$, aligning its coordinates in the prediction subspace with those in the knowledge subspace. The modified state $h'$ is written back to the residual stream to continue forward propagation. This process requires no gradient updates, using only the weights of two pre-trained linear probes.

### Key Designs

**1. Dual-Probe Gap Quantification: Separating "What the Model Knows" from "What it Outputs"**

Previous work could identify model failures but could not horizontally compare the magnitude of the gap across benchmarks and models. KAPPA extracts residual stream activations $h^l(x)$ at each layer $l$ and constructs two parallel datasets: a knowledge dataset $D_{\text{know}}^{(l)} = \{(h^l(x), y)\}$ pairing activations with ground-truth labels, and a prediction dataset $D_{\text{pred}}^{(l)} = \{(h^l(x), \tilde{y})\}$ pairing the same activations with the model's own output labels. Two $k$-class linear classifiers are trained to obtain the knowledge distribution $p_K$ and prediction distribution $p_M$. The discrepancy is characterized by two complementary metrics: Prediction Agreement Rate $\text{AGR}(x) = \mathbb{I}[\arg\max p_K(x) = \arg\max p_M(x)]$ for hard differences, and KL Divergence $\text{KLD}(x) = \text{KL}(p_M \| p_K)$ for soft shifts in confidence.

**2. Subspace Geometric Analysis: Anchoring Knowledge Utilization Failures to Residual Stream Misalignment**

To explain the gap's origin, KAPPA treats the column vectors of each probe's weight matrix $W \in \mathbb{R}^{d \times k}$ as the basis for a subspace—the "knowledge subspace" and the "prediction subspace." Alignment is measured using mean principal angles and CKA. Results show that in deeper layers, the mean principal angle approaches $90^\circ$ (near the random baseline), while CKA falls between 0.4 and 0.8, indicating that knowledge and prediction signals coexist in the same residual stream but propagate in nearly orthogonal directions. Crucially, Spearman correlation analysis across 8 benchmarks shows that more severe subspace misalignment correlates with a larger empirical gap ($\rho = 0.976, p = 0.001$ for Llama 3.1 8B). This identifies a measurable structural cause for abstract "knowledge utilization failure."

**3. KAPPA Inference-time Alignment: Pulling Prediction back to Knowledge via Closed-form Minimum Perturbation**

KAPPA directly modifies hidden states during inference to align coordinates in the prediction subspace with those in the knowledge subspace. This is formulated as a constrained optimization problem: $\min_{\tilde{h}'} \|\tilde{h}' - \tilde{h}\|_2^2$ subject to $\tilde{W}_{\text{pred}}^\top \tilde{h}' = \tilde{W}_{\text{know}}^\top \tilde{h}$. This problem has a closed-form solution:

$$h' = h + W_{\text{pred}}(W_{\text{pred}}^\top W_{\text{pred}})^{-1}(\tilde{W}_{\text{know}}^\top \tilde{h} - \tilde{W}_{\text{pred}}^\top \tilde{h})$$

The modified $h'$ is returned to the residual stream. An extended version introduces two hyperparameters to tighten alignment: $\tilde{W}_{\text{pred}}^\top \tilde{h}' = \alpha \cdot \tilde{W}_{\text{know}}^\top \tilde{h} + \beta \cdot \text{sign}(\tilde{W}_{\text{know}}^\top \tilde{h})$, where $\alpha$ amplifies relative differences between options and $\beta$ pushes logits toward extremes. Unlike CAA, which uses fixed activation steering, KAPPA dynamically calculates the "just enough" minimum perturbation for each input, only modifying components within the prediction subspace and preserving information in orthogonal directions with negligible computational overhead.

## Key Experimental Results

### Main Results

On six benchmarks with significant gaps, KAPPA consistently improves ACC and AGR across models:

| Benchmark (options) | Model | Base ACC | KAPPA(6) ACC | Δ ACC | Base AGR | KAPPA(6) AGR |
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

| Dimension | Metric | Description |
|---|---|---|
| Comparison vs CAA/DoLA | KAPPA superior in 12/12 settings | Existing interventions fail to systematically narrow the gap |
| Intervention Layers (1/3/6) | 6 layers > 3 > 1 (most settings) | Multi-layer interventions have a stronger cumulative effect |
| $\alpha, \beta$ Hyperparam Sweep | Increasing $\alpha$ or $\beta$ monotonically improves AGR | Hyperparameters causally control alignment strength |
| Training Data Sensitivity | Outperforms Base with only 10% data | Remains effective in low-data scenarios |
| Cross-dataset Transfer | TruthfulQA → BBQ-Age: +5.72 AGR | Subspaces are partially shared between similar skill tasks |
| Free-generation Transfer | TruthfulQA ACC: 41.7 → 44.2 | MCQ probes generalize to open-ended generation |

### Key Findings

- The knowledge-prediction gap is largest in truthfulness/bias benchmarks (TruthfulQA knowledge probe is +19–21 points higher than the model), followed by reasoning, and smallest in knowledge-intensive benchmarks.
- Subspace misalignment is highly correlated with the gap ($\rho = 0.976$), confirming a geometric root.
- KAPPA does not directly modify the logits of answer tokens (the mean principal angle between the intervention layer's prediction subspace and the logit space is ~65°–70°); instead, it indirectly influences decision-making through abstract representations.

## Highlights & Insights

- **Closed-form Minimum Perturbation Alignment**: Modeling knowledge-prediction alignment as a constrained optimization problem and deriving a closed-form solution provides a mathematically grounded, efficient alternative to iterative optimization.
- **Dual Probes as Diagnostic Tools**: Training two probes with different targets on the same hidden state and comparing their geometric relationships offers a general framework for diagnosing "internal signal divergence," applicable to hallucination detection and alignment auditing.
- **Cross-format Generalization**: Probes and interventions trained on MCQs can transfer to free generation, suggesting that intermediate subspaces encode abstract semantic directions rather than specific symbols, deepening the understanding of LLM representation hierarchy.

## Limitations & Future Work

- Only addresses linearly accessible knowledge signals; deeper non-linearly encoded knowledge remains untouched.
- Probe training requires annotated data and model prediction labels, making it unsuitable for fully black-box scenarios.
- Transfer effect to free generation is limited (e.g., GSM8k accuracy slightly decreased by 0.9 points), indicating differences between MCQ and open-generation subspaces.
- Future work could explore high-dimensional non-linear alignment, unsupervised probe discovery, and integration with CoT prompting to bridge gaps at both reasoning and representation levels.

## Related Work & Insights

- **Knowledge-Prediction Gap**: Marks & Tegmark (2024) first identified that hidden layers can extract correct answers in truthfulness tasks; this work generalizes the phenomenon and provides a geometric explanation.
- **Inference-time Intervention**: CAA (Rimsky et al., 2024) uses mean difference vectors for steering, and DoLA (Chuang et al., 2024) contrasts layer logits—neither is specifically designed for the knowledge-prediction gap.
- **Mechanistic Interpretability**: Results align with findings by Geva et al. (2023) and Park et al. (2024) regarding how high-level features are converted into token predictions, supporting the information flow perspective of the residual stream.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Rhetorical Questions in LLM Representations: A Linear Probing Study](../../ACL2026/interpretability/rhetorical_questions_in_llm_representations_a_linear_probing_study.md)
- [\[ICML 2026\] PINE: Pruning Boosted Tree Ensembles with Conformal In-Distribution Prediction Equivalence](pine_pruning_boosted_tree_ensembles_with_conformal_in-distribution_prediction_eq.md)
- [\[ICLR 2026\] Bridging Explainability and Embeddings: BEE Aware of Spuriousness](../../ICLR2026/interpretability/bridging_explainability_and_embeddings_bee_aware_of_spuriousness.md)
- [\[ACL 2026\] Tracing Relational Knowledge Recall in Large Language Models](../../ACL2026/interpretability/tracing_relational_knowledge_recall_in_large_language_models.md)
- [\[ICLR 2026\] Closing the Curvature Gap: Full Transformer Hessians and Their Implications for Scaling Laws](../../ICLR2026/interpretability/closing_the_curvature_gap_full_transformer_hessians_and_their_implications_for_s.md)

</div>

<!-- RELATED:END -->
