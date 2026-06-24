---
title: >-
  [Paper Note] Enhancing Training Data Attribution with Representational Optimization
description: >-
  [NeurIPS 2025][LLM Pretraining][training data attribution] This paper proposes AirRep (Attentive Influence Ranking Representation), a representation learning-based training data attribution method that employs a trainable encoder and attention pooling mechanism. AirRep achieves attribution accuracy on par with or superior to state-of-the-art gradient-based methods while being approximately 80× faster at inference.
tags:
  - "NeurIPS 2025"
  - "LLM Pretraining"
  - "training data attribution"
  - "representation learning"
  - "influence functions"
  - "attention pooling"
  - "data selection"
date: 2026-05-08
content_hash: d7d99b57c028cdd3
---

# Enhancing Training Data Attribution with Representational Optimization

**Conference**: NeurIPS 2025
**arXiv**: [2505.18513](https://arxiv.org/abs/2505.18513)  
**Code**: [github.com/sunnweiwei/AirRep](https://github.com/sunnweiwei/AirRep)  
**Area**: LLM Pretraining
**Keywords**: training data attribution, representation learning, influence functions, attention pooling, data selection

## TL;DR

This paper proposes AirRep (Attentive Influence Ranking Representation), a representation learning-based training data attribution method that employs a trainable encoder and attention pooling mechanism. AirRep achieves attribution accuracy on par with or superior to state-of-the-art gradient-based methods while being approximately 80× faster at inference.

## Background & Motivation

Training Data Attribution (TDA) aims to quantify how training data influences model predictions, which is critical for AI transparency and accountability. Existing methods fall into two broad categories:

**Gradient-based methods** (e.g., influence functions):
- Theoretically grounded, approximating changes in model predictions via gradients and inverse Hessians
- Computationally prohibitive (requiring gradient computation and Hessian approximation), and rely on assumptions of loss convexity and model optimality that do not hold for modern neural networks

**Representation-based methods** (e.g., embedding similarity):
- Efficient and scalable, suitable for large-scale applications
- Rely on heuristically designed representation spaces not optimized for attribution tasks, limiting accuracy

Furthermore, both categories adopt a simple summation linear assumption when estimating **group influence**, failing to capture interaction effects among samples.

The core problem addressed in this paper: **Can a method be designed that combines the accuracy of gradient-based approaches with the efficiency of representation-based approaches?**

## Method

### Overall Architecture

AirRep consists of a trainable encoder $\text{Enc}$ and an attention pooling layer $\text{Agg}$. Given a target sample $x$ and a training set $S$, the influence score is computed as:

$$f_{\text{AirRep}}(x, S) = \text{Enc}(x)^\top \cdot \text{Agg}(\text{Enc}(z_i) \mid z_i \in S)$$

### Key Designs

#### 1. Attention-based Influence Pooling

Conventional methods estimate group influence via simple summation, ignoring inter-sample interactions. AirRep introduces an attention mechanism for weighted aggregation:

$$f_{\text{AirRep}}(x, S) = \text{Enc}(x)^\top \cdot \sum_{i=1}^{n} \alpha_i \, \text{Enc}(z_i)$$

Attention weights are defined as:
$$\alpha_i = \frac{\exp(|\text{Enc}(x)^\top \cdot \text{Enc}(z_i)|)}{\sum_{j \in [n]} \exp(|\text{Enc}(x)^\top \cdot \text{Enc}(z_j)|)}$$

Core intuition: influence scores are typically sparse — each test sample depends on only a small subset of training points, with the remainder contributing noise. The attention mechanism enables selective pooling by focusing on the most relevant training samples.

Mathematical connection: attention pooling can be shown to relate to sample weights in higher-order group influence functions (Basu et al.'s second-order term analysis), providing theoretical grounding.

#### 2. Trainable Encoder

AirRep is built on GTE-Small (30M parameters) augmented with a randomly initialized projection matrix. The encoder is optimized through task-aware training to adapt the embedding space for attribution tasks rather than generic text similarity.

#### 3. Automatic Data Generation Pipeline

The pipeline for constructing training attribution signals proceeds as follows:
1. Sample $N_v = 10^4$ validation samples and $N_t = 10^5$ training samples from the corpus
2. Randomly sample $M = 100$ subsets from the training set, each containing $n = 1000$ samples
3. Fine-tune an LLM (Qwen2.5-0.5B) on each subset and evaluate validation loss
4. Compute normalized loss as attribution labels:

$$\hat{r}(x, S_i) = -\frac{\ell(x; \theta_i) - \text{Mean}(\{\ell(x; \theta_j)\})}{\text{Var}(\{\ell(x; \theta_j)\})}$$

100 cross-validation instances are constructed, yielding a total of $10^4$ training subsets and $10^7$ training samples.

### Loss & Training

A **weighted pairwise ranking loss** is adopted to optimize the ordering of attribution scores rather than exact value matching:

$$\mathcal{L}(x, \mathcal{S}) = -\sum_{i,j \in M} \mathbb{1}_{r_i > r_j} \, w_{i,j} \, \log \sigma(f_i - f_j)$$

A weight function handles label noise:

$$w_{i,j} = \begin{cases} 0, & \text{if } |r_i - r_j| < T_{\min} \\ \min\{|r_i - r_j|, T_{\max}\}, & \text{if } T_{\min} \leq |r_i - r_j| \end{cases}$$

where $T_{\min} = 0.1$ and $T_{\max} = 5.0$. Pairs with low label difference are ignored (unreliable labels), while high-difference pairs are clipped to mitigate the influence of outliers.

Training details: AdamW optimizer, lr=$10^{-4}$, up to 2000 steps, with distributed training to maximize GPU utilization.

## Key Experimental Results

### Main Results

**LDS evaluation** (Qwen2.5-0.5B, averaged over 4 datasets):

| Method | Dim | Avg | FLAN | Alpaca | Tulu | SafeRLHF |
|--------|-----|-----|------|--------|------|----------|
| LoGra | 18432 (48×) | 18.45 | 19.75 | 12.38 | 14.88 | 26.82 |
| Dsdm | 18432 (48×) | 18.02 | 19.67 | 12.15 | 14.31 | 25.94 |
| LESS | 8196 (21×) | 16.16 | 16.40 | 9.59 | 13.02 | 25.63 |
| TracIn | 18432 (48×) | 11.33 | 14.75 | 9.21 | 10.75 | 10.60 |
| TF-IDF | - | 9.98 | 2.52 | 7.24 | 5.24 | 24.94 |
| **AirRep** | **384 (1×)** | **26.23** | **21.11** | **22.58** | **15.14** | **46.08** |

Using only 1/48 of the storage, AirRep surpasses all gradient-based methods by an average LDS margin of **7.78 points**.

**Cross-model generalization** (AirRep trained solely on Qwen2.5-0.5B):
- Maintains leading performance on Qwen2.5-1.5B, 3B, and 7B, demonstrating that AirRep trained on smaller models transfers to larger ones
- Also robust across different architectures (Llama-1B, TinyLlama, GPT-2)

**Data classification accuracy**:

| Method | FLAN | Tulu | SafeRLHF |
|--------|------|------|----------|
| LoGra (18432) | 85.44 | 86.00 | 83.20 |
| GTE-Small | 50.59 | 76.60 | **90.60** |
| **AirRep** | **86.41** | **88.20** | 87.20 |

### Ablation Study

Starting from the base GTE (7.65), components are incrementally added:
1. **+Encoder optimization** (without attention) → 19.82 (+12.17), demonstrating that encoder optimization is the core contributor
2. **+Attention pooling** → 26.23 (+6.41), confirming that attention pooling substantially improves group influence estimation
3. Adding attention directly to GTE or LoGra yields only marginal improvement, indicating that optimizing the weight distribution is more important than simple reweighting

### Key Findings

1. **Substantial efficiency advantage**: AirRep achieves approximately 80× faster inference, ~50× better storage efficiency, and can encode hundreds of thousands of samples per second
2. **Amortizable training cost**: Beyond a crossover point of approximately 475K samples, the total cost of AirRep (including retraining) is lower than LoGra. AirRep processes 100M+ samples in 24 GPU-hours versus LoGra's 2M
3. **Cross-model and cross-task generalization**: Trained on Qwen-0.5B, AirRep is directly applicable to 7B models and different architectures
4. **Unsupervised learning of task information**: AirRep training uses no data labels yet learns task-relevant representations (FLAN classification accuracy: 86.41%)

## Highlights & Insights

1. **Bridging the gradient–representation divide**: Task-aware training elevates representation-based methods to the accuracy level of gradient-based methods while retaining the efficiency of the former
2. **Theoretically grounded attention pooling**: A mathematical connection to higher-order group influence functions is established, moving beyond purely heuristic design
3. **Elegant weighted ranking loss design**: Clipping and ignoring label differences gracefully handles label noise arising from the stochasticity of LLM training
4. **Large-scale scalability**: Processing 100M samples in 24 GPU-hours makes AirRep genuinely applicable to LLM pretraining data attribution
5. **Training cost amortization analysis**: A quantitative analysis of the crossover point at which training overhead is offset by inference efficiency serves as a practical reference

## Limitations & Future Work

1. **Training data generation cost**: Obtaining attribution labels requires training 100 LLM subset models; although amortizable, the upfront cost is non-trivial
2. **Evaluation limited to the fine-tuning stage**: Data attribution during pretraining is more challenging and remains to be validated
3. **Underperformance on SafeRLHF relative to GTE**: The training data (UltraChat) contains no harmful content, lacking safety-relevant learning signals
4. **Modality limitation**: Only text tasks have been validated; while the method is claimed to be modality-agnostic, vision and multimodal scenarios have not been experimentally verified
5. **GTE-Small as the base encoder**: It remains an open question whether the 30M parameter capacity limits representational expressiveness and whether larger encoders could yield further improvements

## Related Work & Insights

- **Influence function family** (Koh & Liang → LoGra → TRAK): Comparison baselines for AirRep; highlight the computational bottleneck of gradient-based methods at LLM scale
- **DCLM / FineWeb-Edu**: Employ representation-based methods for data selection without attribution-specific optimization
- **Datamodels** (Ilyas et al.): The source of the LDS evaluation framework; AirRep follows their experimental setup
- **Insights**: (1) Task-aware representation learning is the key pathway to improving TDA; (2) Group influence estimation requires moving beyond linear assumptions; (3) The "train on small models, apply to large models" transfer paradigm is equally effective in data attribution

## Rating

- Novelty: ⭐⭐⭐⭐ — The combination of attention pooling and trainable encoder, together with the ranking optimization paradigm, represents significant innovation
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — LDS, data selection, data classification, ablation, cost analysis, and cross-model/cross-architecture generalization are comprehensively covered
- Writing Quality: ⭐⭐⭐⭐ — Motivation is clearly articulated, the technical approach is logically coherent, and experiments are well organized
- Value: ⭐⭐⭐⭐⭐ — Enabling efficient and accurate data attribution at LLM scale has important implications for data curation and model interpretability

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Final-Model-Only Data Attribution with a Unifying View of Gradient-Based Methods](final-model-only_data_attribution_with_a_unifying_view_of_gradient-based_methods.md)
- [\[NeurIPS 2025\] Quantifying Task-Relevant Representational Similarity Using Decision Variable Correlation](quantifying_task-relevant_representational_similarity_using_decision_variable_co.md)
- [\[NeurIPS 2025\] Conformal Risk Training: End-to-End Optimization of Conformal Risk Control](conformal_risk_training_end-to-end_optimization_of_conformal_risk_control.md)
- [\[ICML 2025\] LLM Data Selection and Utilization via Dynamic Bi-level Optimization](../../ICML2025/llm_pretraining/llm_data_selection_and_utilization_via_dynamic_bi-level_optimization.md)
- [\[NeurIPS 2025\] Predict Training Data Quality via Its Geometry in Metric Space](predict_training_data_quality_via_its_geometry_in_metric_space.md)

</div>

<!-- RELATED:END -->
