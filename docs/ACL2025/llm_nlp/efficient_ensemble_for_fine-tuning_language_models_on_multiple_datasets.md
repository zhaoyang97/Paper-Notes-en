---
title: >-
  [Paper Note] Efficient Ensemble for Fine-tuning Language Models on Multiple Datasets
description: >-
  [ACL 2025][LLM (Other)][Ensemble Learning] Proposed EnsembleLoRA—an efficient ensemble method for multi-dataset fine-tuning. It utilizes a first-order Taylor approximation to rapidly estimate task affinity for dataset grouping, trains one adapter per group, and combines them via weighted aggregation. This achieves a 10% average test accuracy improvement over QLoRA on 10 SuperGLUE tasks with only a 9% additional computational cost.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Ensemble Learning"
  - "LoRA"
  - "Multi-Dataset Fine-Tuning"
  - "Task Affinity"
  - "Gradient Approximation"
  - "Parameter-Efficient Fine-Tuning"
date: 2026-05-08
content_hash: 2cc2fb564bc30b7d
---

# Efficient Ensemble for Fine-tuning Language Models on Multiple Datasets

**Conference**: ACL 2025  
**arXiv**: [2505.21930](https://arxiv.org/abs/2505.21930)  
**Code**: [https://github.com/VirtuosoResearch/EnsembleLoRA](https://github.com/VirtuosoResearch/EnsembleLoRA)  
**Authors**: Dongyue Li, Ziniu Zhang, Lu Wang, Hongyang R. Zhang  
**Institution**: Northeastern University, University of Michigan  
**Area**: NLP / Efficient Fine-Tuning  
**Keywords**: Ensemble Learning, LoRA, Multi-Dataset Fine-Tuning, Task Affinity, Gradient Approximation, Parameter-Efficient Fine-Tuning

## TL;DR

Proposed EnsembleLoRA—an efficient ensemble method for multi-dataset fine-tuning. It utilizes a first-order Taylor approximation to rapidly estimate task affinity for dataset grouping, trains one adapter per group, and combines them via weighted aggregation. This achieves a 10% average test accuracy improvement over QLoRA on 10 SuperGLUE tasks with only a 9% additional computational cost.

## Background & Motivation

**Background**: Parameter-efficient fine-tuning (PEFT) methods like LoRA/QLoRA are highly efficient for single-dataset adaptation. However, in practical scenarios, evaluation protocols often involve mixtures of multiple datasets or tasks. How to efficiently adapt to multiple datasets remains a problem.

**Limitations of Prior Work**:
   - **Training a single adapter** on all datasets $\rightarrow$ negative transfer among tasks. Experiments show that performance degradation occurs in 33 out of 45 pairwise combinations for QLoRA.
   - **Pre-training followed by fine-tuning (MTL-FT)** $\rightarrow$ double the computational overhead (pre-training + task-specific fine-tuning).
   - **One adapter per task** $\rightarrow$ memory overhead scales linearly with the number of tasks (10 tasks = 10 adapters).
   - Key Challenge: How to understand the relationships between datasets and find an appropriate grouping strategy.

**Core Motivation**: Leverage the property that the weights of LoRA fine-tuned models are extremely close to those of the base model (relative distance < 0.2%). By utilizing a first-order expansion to rapidly estimate task affinity, the datasets can be efficiently grouped and ensembled using a small number of adapters.

## Method

### Overall Architecture

Given $n$ datasets and a base adaptation method (e.g., QLoRA), the algorithm proceeds in three steps:

1. **Task Affinity Grouping**: Estimate the $n \times n$ task affinity matrix $\rightarrow$ cluster into $m$ groups ($m \ll n$).
2. **Adapter Training**: Train one adapter per group $\rightarrow$ $m$ adapters.
3. **Gradient Boosting Refinement**: Train additional adapters for groups with the largest losses $\rightarrow$ final $M = m + b$ adapters $\rightarrow$ weighted ensemble.

### Key Designs

#### Key Design 1: First-Order Approximation for Estimating Fine-Tuning Performance

**Core Observation**: The relative distance between the weights of PEFT methods (such as LoRA/Adapter) after fine-tuning and those of the base model is extremely small:

| Method | Llama-3-1B | Llama-3-3B | Llama-3-8B |
|------|-----------|-----------|-----------|
| LoRA | 0.16% | 0.14% | 0.12% |
| QLoRA | 0.18% | 0.16% | 0.11% |
| Adapter | 0.09% | 0.05% | 0.08% |
| QAdapter | 0.11% | 0.08% | 0.07% |

Therefore, the model output can be approximated using a first-order Taylor expansion:

$$h_X(s, y) \approx h_{\theta^*}(s, y) + [\nabla_X h_{\theta^*}(s, y)]^{\top}(X - \theta^*)$$

Experiments demonstrate that the approximation error is < 1% (LoRA/Adapter) or < 3% (QLoRA/QAdapter) on Llama/GPT-J models (up to 34B parameters).

#### Key Design 2: Gradient Projection + Regression Estimation

1. Compute gradients once on the base model for all training samples.
2. Use random projection to reduce gradient dimensions to several hundred (leveraging the distance-preserving property of Johnson-Lindenstrauss).
3. For any dataset subset $S$, estimate the fine-tuned adapter weights $\hat{\theta}_S$ by solving a logistic regression (which completes in a few seconds on CPU).
4. Evaluate the estimated performance $\hat{f}_i(S)$ of each task using $\hat{\theta}_S$.
5. Iterate through $k$ random subsets to compute the task affinity matrix $T_{i,j}$.

**Estimation Results** (Projection dimension $d$ vs. estimation error):

| $d$ | Llama-3-1B | Llama-3-3B | Llama-3-8B | Speedup |
|-----|-----------|-----------|-----------|--------|
| 200 | 8.2% | 8.1% | 7.0% | $10^5 \times$ |
| 400 | 4.7% | 4.8% | 4.3% | $10^5 \times$ |
| 800 | 4.6% | 4.4% | 4.2% | $10^5 \times$ |

With a speedup of $10^5 \times$, the estimation error is kept within 5%.

#### Key Design 3: Clustering and Grouping

A clustering algorithm based on semidefinite programming relaxation maximizes intra-group affinity density and automatically determines the number of groups $m$ using trace regularization.

#### Key Design 4: Gradient Boosting Refinement

After training the initial groups, new adapters are added to the groups with the largest training losses:
- Fit residuals (negative gradient $1 - p_s$).
- Solve linear regression still using gradient approximation.
- The first boosting step reduces training error by 18%, corresponding to a 0.4% increase in test accuracy.

### Computational/Memory Overhead Comparison

| Method | Runtime | Memory |
|------|---------|------|
| Base Fine-Tuning | $T$ | $A$ |
| MTL-FT | $\approx 2T$ | $nA$ |
| **EnsembleLoRA** | $T + G$ | $MA$ |

where $T$ is the base fine-tuning time, $G$ is the time for a single gradient computation ($G \ll T$), and $M$ is the number of adapters in the ensemble.

## Key Experimental Results

### Experimental Setup

- Base models: Llama-3.1-8B, CodeLlama-34B-Instruct
- 10 SuperGLUE tasks (divided into 5 categories: Sentence Completion, Natural Language Inference, Coreference Resolution, Question Answering, Word Sense Disambiguation)
- Baselines: Base fine-tuning, MTL-FT (Liu et al., 2019), TAG (Fifty et al., 2021)

### Llama-3-8B + QLoRA Main Results

| Method | Average Accuracy | FLOPs | GPU Memory |
|------|-----------|-------|---------|
| Full FT | 84.6% | $6.0 \times 10^{19}$ | 73.0 GB |
| QLoRA (Single adapter) | ~70% | Baseline | Baseline |
| MTL-FT | On par with Ours | 45% higher | 45% higher |
| **EnsembleLoRA** | **QLoRA +10%** | QLoRA +9% | QLoRA +9GB |

### Key Findings

1. **Severe negative transfer in single-adapter setup**: Among the 45 pairwise combinations for QLoRA, 73% (33/45) exhibit negative transfer.
2. **Ensemble is effective but scales need to be controlled**: Naively training a 10-adapter ensemble improves performance by 10.8%, but increases memory overhead by 4 times.
3. **Grouping + Gradient Boosting = Efficient Ensemble**: EnsembleLoRA achieves comparable performance with only about 3 adapters.
4. **Scalable to 500 datasets** (in a federated learning setup), reducing computation by 90% and memory by 91%.
5. **Theoretical analysis of generalization**: LoRA adapters with low rank possess the lowest generalization error, which is further reduced by ensembling.

### CodeLlama-34B Results

On the 34B parameter model, EnsembleQLoRA improves accuracy by 3% compared to a single QLoRA, while only increasing FLOPs by 8%.

## Highlights & Insights

1. **Compelling Core Insights**: The post-fine-tuning weights of PEFT methods are extremely close to the base model (< 0.2%), which makes first-order approximation possible. This observation transforms the complex multi-dataset fine-tuning problem into a regression problem in the gradient space.
2. **$10^5 \times$ faster estimation** is highly practical—all affinity estimations complete in a few seconds on a CPU.
3. **Strong Generalization**: Applicable to four PEFT methods: LoRA, Adapter, QLoRA, and QAdapter.
4. **Combining Theory with Practice**: Besides generalization error analysis, the generalization advantage of low-rank adapters is validated through sharpness measurements.

## Limitations & Future Work

1. Only classification tasks (SuperGLUE) are evaluated; generative tasks (e.g., summarization, translation) are not verified.
2. First-order approximation error for QLoRA is slightly larger (< 3% vs < 1%), which might fail under extreme quantization.
3. The clustering algorithm is based on semidefinite programming, which can be computationally heavy when the number of tasks is very large (e.g., 500).
4. Strategies for managing severely imbalanced dataset sizes are not discussed.
5. Tuning is still required for the choice of the number of boosting steps $b$ and the number of groups $m$.

## Related Work

- **Parameter-Efficient Fine-Tuning**: LoRA (Hu et al., 2021), QLoRA (Dettmers et al., 2023), Adapter-tuning (Houlsby et al., 2019)
- **Multi-Task Learning**: MTL-FT (Liu et al., 2019), TAG (Fifty et al., 2021)
- **Influence Functions / Data Modeling**: Koh & Liang (2017), Ilyas et al. (2022), Park et al. (2023)

## Rating

⭐⭐⭐⭐⭐ (5/5)

This paper makes a solid technical contribution. The core observation (that PEFT weights are extremely close to the base model) is simple yet profound, and the resulting method is both theoretically grounded and practically efficient. The experiments are extensive, covering models from 1B to 34B parameters and datasets from 10 to 500. It provides a highly hands-on solution for the practical and important scenario of multi-dataset fine-tuning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] A Semantic-Aware Layer-Freezing Approach to Computation-Efficient Fine-Tuning of Language Models](a_semantic-aware_layer-freezing_approach_to_computation-efficient_fine-tuning_of.md)
- [\[ACL 2025\] Refining Salience-Aware Sparse Fine-Tuning Strategies for Language Models](refining_salience-aware_sparse_fine-tuning_strategies_for_language_models.md)
- [\[ACL 2025\] HFT: Half Fine-Tuning for Large Language Models](hft_half_fine-tuning_for_large_language_models.md)
- [\[ACL 2025\] GORP: Continual Gradient Low-Rank Projection Fine-Tuning for LLMs](gorp_continual_gradient_projection.md)
- [\[ACL 2025\] PiFi: Plug-in and Fine-tuning: Bridging the Gap between Small Language Models and Large Language Models](plugin_finetuning_bridge.md)

</div>

<!-- RELATED:END -->
