---
title: >-
  [Paper Note] Bridging the Knowledge-Prediction Gap in LLMs on Multiple-Choice Questions
description: >-
  [ICML 2026][Interpretability][Paper Note] This paper reveals a widespread "knowledge-prediction gap" in LLMs on MCQs—correct answers are already linearly encoded in hidden layers, but the final predictions deviate. Through geometric analysis, this gap is attributed to the misalignment between knowledge and prediction subspaces. The authors propose KAPPA, which
tags:
  - ICML 2026
  - Interpretability
date: 2026-05-08
content_hash: 8c06f8c51e59c51b
---
# Bridging the Knowledge-Prediction Gap in LLMs on Multiple-Choice Questions

**Conference**: ICML 2026  
**arXiv**: [2509.23782](https://arxiv.org/abs/2509.23782)  
**Code**: https://github.com/holi-lab/KAPPA  
**Area**: Interpretability  
**Keywords**: knowledge-prediction gap, linear probes, subspace alignment, inference-time intervention, multiple-choice questions  

## TL;DR

This paper reveals a widespread "knowledge-prediction gap" in LLMs on MCQs—correct answers are already linearly encoded in hidden layers, but the final predictions deviate. Through geometric analysis, this gap is attributed to the misalignment between knowledge and prediction subspaces. The authors propose KAPPA, which uses closed-form affine transformations to align these subspaces during inference, consistently closing the gap and improving accuracy across models and benchmarks.

## Background & Motivation

**Background**: Evaluating LLMs on multiple-choice question (MCQ) benchmarks is a standard practice. However, models frequently exhibit "capability inconsistency"—giving the correct answer in free-form generation but failing when switched to an MCQ format. Existing research shows that even when a model fails, correct answers can be extracted from its hidden layers using simple linear classifiers, suggesting that sufficient knowledge is already encoded internally.

**Limitations of Prior Work**: Previous work primarily attributes MCQ errors to "surface-level factors" such as option bias, surface cues, or stylistic artifacts. However, there is a lack of an explanatory framework that unifies these failures with the model's internal representations. Research on the "knowledge-prediction gap" has also been limited to narrow scenarios like truthfulness detection and simple arithmetic, without generalization to diverse MCQ tasks.

**Key Challenge**: Both the correct answer (knowledge signal) and the actual output (prediction signal) are linearly encoded in the residual stream, but these signals are routed along geometrically distinct directions. This causes the prediction signal to "override" the knowledge signal during final generation. This is not a lack of knowledge, but a failure in knowledge utilization.

**Goal**: (1) Quantify the prevalence and severity of the knowledge-prediction gap across various MCQ benchmarks and model families; (2) Explain the structural cause of the gap from a residual stream geometry perspective; (3) Design training-free inference-time interventions to bridge the gap.

**Key Insight**: By training two linear probes—a knowledge probe to predict the ground truth and a prediction probe to predict model outputs—their weight matrices can be viewed as basis vectors defining subspaces. If the two subspaces are aligned, model predictions should be consistent with internal knowledge; empirical measurements show that principal angles deviate significantly in benchmarks with large gaps.

**Core Idea**: Use minimal $\ell_2$ perturbations to correct the coordinates of hidden states in the prediction subspace to match their coordinates in the knowledge subspace—essentially "aligning prediction to knowledge."

## Method

### Overall Architecture

Given an MCQ input, KAPPA extracts the hidden state $h$ from intermediate layers of the residual stream. It projects $h$ onto two $k$-dimensional subspaces defined by the weights of the knowledge and prediction probes to calculate their respective coordinates (probe logits). When the two sets of coordinates are inconsistent, KAPPA applies a closed-form affine transformation to $h$ to align its coordinates in the prediction subspace with those in the knowledge subspace. The modified $h'$ is written back to the residual stream for further forward propagation. The entire process requires no gradient updates and only utilizes two sets of pre-trained linear probe weights.

### Key Designs

**1. Dual-Probe Gap Quantization: Separating "what the model knows" from "what it outputs" into comparable signals**

Prior work could only state that a model failed, but could not compare the magnitude of the gap across different benchmarks and models. KAPPA extracts residual stream activations $h^l(x)$ at each layer $l$ and constructs two parallel datasets: a knowledge dataset $D_{\text{know}}^{(l)} = \{(h^l(x), y)\}$ pairing activations with ground-truth labels, and a prediction dataset $D_{\text{pred}}^{(l)} = \{(h^l(x), \tilde{y})\}$ pairing the same activations with the model's own output labels. Two $k$-class linear classifiers are trained to obtain the knowledge distribution $p_K$ and prediction distribution $p_M$. The gap is characterized by two complementary metrics: the Agreement Rate $\text{AGR}(x) = \mathbb{I}[\arg\max p_K(x) = \arg\max p_M(x)]$ captures hard differences in selection, while the KL Divergence $\text{KLD}(x) = \text{KL}(p_M \| p_K)$ captures soft shifts in confidence distributions.

**2. Subspace Geometric Analysis: Anchoring "internal disagreement" to geometric misalignment in the residual stream**

KAPPA treats the column vectors of each probe's weight matrix $W \in \mathbb{R}^{d \times k}$ as the basis for a subspace—the "knowledge subspace" for the knowledge probe and the "prediction subspace" for the prediction probe. The alignment between these two subspaces is measured using the mean principal angle and CKA. Results show that in deep layers, the mean principal angle approaches $90°$ (near random baseline), and CKA falls in the 0.4–0.8 range, indicating that knowledge and prediction signals coexist in the same residual stream but propagate in nearly orthogonal geometric directions. Spearman correlation analysis across 8 benchmarks confirms: the more severe the subspace misalignment, the larger the measured gap ($\rho = 0.976, p = 0.001$ on Llama 3.1 8B).

**3. KAPPA Inference-time Alignment: Pulling prediction back to knowledge via closed-form minimal perturbation**

KAPPA directly modifies the hidden state during inference to align its coordinates in the prediction subspace with its coordinates in the knowledge subspace. Formally, this is a constrained optimization problem: $\min_{\tilde{h}'} \|\tilde{h}' - \tilde{h}\|_2^2$ such that $\tilde{W}_{\text{pred}}^\top \tilde{h}' = \tilde{W}_{\text{know}}^\top \tilde{h}$. This problem has a closed-form solution:

$$h' = h + W_{\text{pred}}(W_{\text{pred}}^\top W_{\text{pred}})^{-1}(\tilde{W}_{\text{know}}^\top \tilde{h} - \tilde{W}_{\text{pred}}^\top \tilde{h})$$

The modified $h'$ is written back for the remaining forward pass. An extended version introduces two hyperparameters to tighten the alignment: $\tilde{W}_{\text{pred}}^\top \tilde{h}' = \alpha \cdot \tilde{W}_{\text{know}}^\top \tilde{h} + \beta \cdot \text{sign}(\tilde{W}_{\text{know}}^\top \tilde{h})$, where $\alpha$ amplifies relative differences between options and $\beta$ pushes logits toward extremes. Unlike activation steering methods like CAA that use fixed directions, KAPPA dynamically calculates the "just enough" minimal perturbation for each input.

## Key Experimental Results

### Main Results

On 6 benchmarks with significant gaps, KAPPA consistently improves ACC and AGR across models:

| Benchmark (options) | Model | Base ACC | KAPPA(6) ACC | Gain | Base AGR | KAPPA(6) AGR |
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

| Analysis Dimension | Key Metrics | Note |
|---|---|---|
| vs. CAA/DoLA | KAPPA superior in 12/12 settings | Existing interventions fail to systematically close the gap |
| Number of layers (1/3/6) | 6 layers > 3 layers > 1 layer | Cumulative effect of multi-layer intervention is stronger |
| α, β Hyperparameters | Increasing α or β monotonically improves AGR | Both parameters causally control alignment strength |
| Training Data Sensitivity | Superior to Base with only 10% data | Effective in low-data scenarios |
| Cross-dataset Transfer | TruthfulQA → BBQ-Age: +5.72 AGR | Subspaces are partially shared among similar skill tasks |
| Free Generation Transfer | TruthfulQA ACC: 41.7 → 44.2 | MCQ probes can generalize to open-ended generation |

### Key Findings

- The knowledge-prediction gap is largest on truthfulness/bias benchmarks (TruthfulQA knowledge probes outperform models by +19–21 points), followed by reasoning benchmarks, and is smallest on knowledge-intensive benchmarks.
- Subspace misalignment is highly correlated with the gap ($\rho = 0.976$), confirming the geometric root of the gap.
- KAPPA does not directly modify the logits of answer tokens (the angle between the prediction subspace and logit space in intervention layers is approx. 65°–70°); instead, it indirectly influences subsequent decisions by modifying abstract representations in intermediate layers.

## Highlights & Insights

- **Closed-form Minimal Perturbation Alignment**: Modeling knowledge-prediction alignment as a constrained optimization problem and deriving a closed-form solution allows for negligible computational overhead and guaranteed minimal modification. This "math over gradients" approach is highly efficient for inference-time intervention.
- **Dual Probes as a Diagnostic Tool**: Training two linear probes with different targets on the same hidden state and comparing their subspace geometry provides a general framework for diagnosing internal model signal divergence, applicable to hallucination detection or alignment auditing.
- **Cross-format Generalization**: Probes and interventions trained on MCQs can transfer to free-form generation, suggesting that intermediate subspaces encode abstract semantic directions rather than specific answer tokens. This deepens the understanding of the hierarchical structure of LLM internal representations.

## Limitations & Future Work

- Currently only addresses linearly accessible knowledge signals; deeper non-linearly encoded knowledge remains untouched.
- Probe training requires labeled data and model predictions, making it inapplicable in purely black-box scenarios.
- Transfer effectiveness to free generation is limited (GSM8k accuracy slightly decreased by 0.9 points), indicating differences between MCQ and open-ended generation subspaces.
- Future Work: Explore high-dimensional non-linear alignment, unsupervised probe discovery, and integration with CoT prompting to bridge gaps at both the reasoning and representation levels.

## Related Work & Insights

- **Knowledge-Prediction Gap Literature**: Marks & Tegmark (2024) first observed that hidden layers could extract correct answers in truthfulness tasks; this work generalizes the phenomenon to general MCQs and provides a geometric explanation.
- **Inference-time Intervention**: CAA (Rimsky et al., 2024) uses mean difference vectors for activation steering, and DoLA (Chuang et al., 2024) uses contrastive layer logits—neither was designed for the knowledge-prediction gap, and experiments show their limited efficacy.
- **Mechanistic Interpretability**: Consistent with Geva et al. (2023) and Park et al. (2024) regarding how high-level features in the residual stream are transformed into token predictions by subsequent layers, the effectiveness of KAPPA further corroborates this view of information flow.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Rhetorical Questions in LLM Representations: A Linear Probing Study](../../ACL2026/interpretability/rhetorical_questions_in_llm_representations_a_linear_probing_study.md)
- [\[ICML 2026\] PINE: Pruning Boosted Tree Ensembles with Conformal In-Distribution Prediction Equivalence](pine_pruning_boosted_tree_ensembles_with_conformal_in-distribution_prediction_eq.md)
- [\[ICLR 2026\] Bridging Explainability and Embeddings: BEE Aware of Spuriousness](../../ICLR2026/interpretability/bridging_explainability_and_embeddings_bee_aware_of_spuriousness.md)
- [\[ACL 2026\] Tracing Relational Knowledge Recall in Large Language Models](../../ACL2026/interpretability/tracing_relational_knowledge_recall_in_large_language_models.md)
- [\[CVPR 2026\] Where Culture Fades: Revealing the Cultural Gap in Text-to-Image Generation](../../CVPR2026/interpretability/where_culture_fades_revealing_the_cultural_gap_in_text-to-image_generation.md)

</div>

<!-- RELATED:END -->
