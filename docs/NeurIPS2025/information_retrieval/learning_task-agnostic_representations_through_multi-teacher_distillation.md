---
title: >-
  [Paper Note] Learning Task-Agnostic Representations through Multi-Teacher Distillation
description: >-
  [NeurIPS 2025][Information Retrieval & RAG][Knowledge Distillation] This paper proposes a task-agnostic multi-teacher distillation framework based on mutual information maximization. By estimating the conditional distribution of teacher embeddings via Gaussian kernels, the student model learns high-information-density general-purpose representations without relying on any downstream task labels, achieving state-of-the-art performance among same-scale models across text…
tags:
  - "NeurIPS 2025"
  - "Information Retrieval & RAG"
  - "Knowledge Distillation"
  - "Multi-Teacher"
  - "Task-Agnostic"
  - "Representation Learning"
  - "Mutual Information"
date: 2026-05-08
content_hash: 84739e1a664bab33
---

# Learning Task-Agnostic Representations through Multi-Teacher Distillation

**Conference**: NeurIPS 2025
**arXiv**: [2510.18680](https://arxiv.org/abs/2510.18680)  
**Code**: None  
**Area**: Information Retrieval
**Keywords**: Knowledge Distillation, Multi-Teacher, Task-Agnostic, Representation Learning, Mutual Information

## TL;DR

This paper proposes a task-agnostic multi-teacher distillation framework based on mutual information maximization. By estimating the conditional distribution of teacher embeddings via Gaussian kernels, the student model learns high-information-density general-purpose representations without relying on any downstream task labels, achieving state-of-the-art performance among same-scale models across text, vision, and molecular modeling domains.

## Background & Motivation

1. **Diversity of embedding models remains underexploited**: Embedding models trained with different architectures, paradigms, and objectives capture different aspects of the input, yet existing methods struggle to consolidate such diverse knowledge into a single compact model.

2. **Existing knowledge distillation methods are task-bound**: Conventional multi-teacher distillation either aligns logits on specific tasks or relies on task-specific labels, making it unable to generalize to unseen tasks and requiring the distillation process to be repeated for each new task.

3. **Task-agnostic distillation research is scarce and overly restrictive**: The few existing works require the student and teachers to share the same architecture (Liang et al., 2023), the same embedding dimensionality (SEED), or necessitate fine-tuning the teachers, severely limiting generality.

4. **MSE regression is unstable in high-dimensional spaces**: Methods such as SimReg use MSE loss for point estimation to reconstruct teacher features, but this approach is unstable in high-dimensional spaces, as extensively documented in the reinforcement learning literature (Farebrother et al., 2024).

5. **Student models must perform well across multiple tasks**: An ideal distilled student should perform well on diverse downstream tasks—including classification, regression, clustering, and semantic similarity—rather than excelling on only a single task.

6. **Inference efficiency requirements**: Large embedding models (e.g., 7B parameters) are unavailable in resource-constrained environments, making it imperative to compress their knowledge into 20M–335M scale models without significant performance loss.

## Method

### Overall Architecture: Task-Agnostic Distillation via "Majority Voting"

- **Function**: Trains a student embedding model such that the predictions of its Bayes classifier agree with the majority vote of multiple teachers' Bayes classifiers on arbitrary downstream tasks.
- **Design Motivation**: Directly optimizing the prediction agreement between the student and teachers over all possible tasks is intractable; however, the authors prove that this agreement is upper-bounded by the conditional entropy of teacher embeddings given the student embeddings, which is task-agnostic.
- **Mechanism**:
  1. Define the ideal loss as the average probability that the student's Bayes classifier $C_S$ disagrees with each of the $K$ teachers' Bayes classifiers $C_{T_k}$.
  2. Apply Proposition 3.1 to establish: $\Pr(C_S \neq C_{T_k}) \leq 1 - \exp(-h(T_k(X) \mid S(X)))$.
  3. Apply Jensen's inequality to obtain the task-agnostic upper bound: $\mathcal{L}^* \leq 1 - \exp\!\left(-\frac{1}{K}\sum_k h(T_k(X) \mid S(X))\right)$.
  4. Minimizing this upper bound is equivalent to maximizing the mutual information $I(T_k(X); S(X))$ between the student and each teacher.

### Key Design 1: Conditional Distribution Estimation via Gaussian Kernels

- **Function**: Estimates the conditional distribution of each teacher's embeddings given the student's embeddings using a parameterized Gaussian model.
- **Design Motivation**: The conditional entropy cannot be computed directly; it must be approximated through a differentiable training objective via conditional distribution estimation. Compared to the point estimation of MSE, interval estimation via Gaussian kernels is more stable and effective in high-dimensional spaces.
- **Mechanism**:
  1. For each teacher $k$, learn a mapping from student embeddings to Gaussian parameters, outputting mean $\mu_k(S(X))$ and covariance $\Sigma_k(S(X))$.
  2. The final loss is the negative log-likelihood: $\mathcal{L} = \frac{1}{K}\sum_k \mathbb{E}_X\left[-\log \mathcal{N}(T_k(X) \mid \mu_k(S(X)), \Sigma_k(S(X)))\right]$.
  3. The student network and all Gaussian kernel parameters are jointly trained end-to-end; the Gaussian kernels are discarded after training.

### Key Design 2: Pre-Computed Teacher Embeddings for Efficient Training

- **Function**: Encodes the entire training set with all teacher models in advance and stores the resulting embeddings; during training, pre-computed embeddings are loaded directly by batch index.
- **Design Motivation**: This avoids repeated forward passes through multiple large teacher models for every batch, substantially reducing computational overhead; adding one additional teacher increases training step time by less than 1%.
- **Mechanism**: Teacher models are run sequentially to obtain embeddings for all samples, which are stored to disk. During training, embeddings are retrieved by batch index, and the student and Gaussian kernels are updated end-to-end using the Adam optimizer.

## Key Experimental Results

### Experiment 1: NLP Text Embedding Distillation

| Model | Parameters | Avg Classification (12 tasks) |
|-------|------------|-------------------------------|
| GIST-xs | 23M | 72.7 |
| MSE Student-xs | 23M | 72.9 |
| **NLL Student-xs** | **23M** | **74.0** |
| GIST-s | 33M | 76.1 |
| **NLL Student-s** | **33M** | **76.7** |
| GIST-m | 109M | 76.0 |
| **NLL Student-m** | **109M** | **76.7** |
| bge-large-en-v1.5 | 335M | 76.0 |
| **NLL Student-l** | **335M** | **76.5** |

**Key Findings**:
- The NLL-distilled 109M student surpasses all 335M models on MTEB classification tasks, demonstrating exceptionally high information density.
- The student resides on the Pareto frontier across all model size categories; the xs model (23M) ranks first on the majority of tasks.
- NLL distillation consistently outperforms MSE distillation, validating the theoretical expectation that interval estimation is superior to point estimation.

### Experiment 2: Molecular Modeling Distillation

| Method | Avg Rank (Regression) | Avg Rank (Classification) |
|--------|-----------------------|---------------------------|
| ChemBERTaMTR (teacher) | ~3.5 | ~4.0 |
| 3D-infomax (teacher) | ~4.0 | ~3.5 |
| MSE (8-teacher) | ~3.0 | ~3.0 |
| Cosine (8-teacher) | ~3.5 | ~3.5 |
| **NLL (8-teacher)** | **~1.5** | **~2.0** |

**Key Findings**:
- Eight-teacher distillation substantially outperforms single- and dual-teacher distillation, confirming the value of multi-teacher diversity.
- NLL distillation achieves the best average rank on both regression and classification tasks, surpassing all individual teacher models.
- Computational overhead is minimal: each additional teacher adds only 1.57 ms/step (<1%), enabling efficient scaling of the teacher pool.

### Experiment 3: Vision Embedding Distillation

- The student is PVTv2 (3.7M); teachers are Swin/DINOv2/ViT/BEiT (~87M each).
- The distilled student consistently lies on the Pareto frontier across DTD, FGVCAircraft, CUB, CIFAR10, SVHN, and STL10.
- Performance is competitive with large ViT teachers that have 20× more parameters.

## Highlights & Insights

- **Theoretical elegance**: Starting from Bayes classifier agreement, the framework rigorously derives a task-agnostic conditional entropy upper bound and arrives at a practical, optimizable loss via mutual information maximization, forming a complete theoretical chain.
- **Cross-modal generality**: The same framework, without modification, achieves state-of-the-art performance across three fundamentally different modalities: text, vision, and molecular graphs.
- **Strong practicality**: The pre-computed teacher embedding strategy makes training cost nearly linear in the number of teachers with negligible per-teacher overhead.
- **No architectural constraints**: The student and teachers may have different architectures and embedding dimensionalities, overcoming the limitations of most prior methods.

## Limitations & Future Work

- **Embedding space structure is not preserved**: Since mutual information is invariant to invertible transformations, the optimization objective does not guarantee preservation of structural properties such as cosine similarity in the teacher's embedding space, resulting in limited gains on clustering and STS tasks that rely on dot products.
- **Suboptimal for single-task scenarios**: When the downstream task is known and unique, task-specific distillation may be more effective.
- **Dependence on teacher quality and relevance**: The quality of student embeddings depends on the relevance of the teachers to downstream tasks; if the teachers are unrelated to the target domain, the benefit is limited.
- **Storage overhead**: Pre-computing and storing all teacher embeddings requires substantial disk space (up to ~100 GB for the largest text teachers).

## Related Work & Insights

| Dimension | Ours (NLL Distillation) | SimReg (Navaneet et al., 2022) |
|-----------|------------------------|-------------------------------|
| Distillation objective | Task-agnostic (mutual information maximization) | Task-agnostic (MSE reconstruction) |
| Loss function | Gaussian kernel NLL (interval estimation) | MSE + cross-encoder head (point estimation) |
| Theoretical guarantee | Yes (Bayes classifier agreement upper bound) | No |
| High-dimensional stability | Strong (interval estimation is inherently stable) | Weak (MSE unstable in high dimensions) |
| Multi-teacher scalability | Natively supported with negligible overhead | Requires cross-encoder heads; poor scalability |

| Dimension | Ours (NLL Distillation) | CompRess (Abbasi Koohpayegani et al., 2020) |
|-----------|------------------------|---------------------------------------------|
| Distillation objective | Conditional distribution matching | Nearest-neighbor graph preservation |
| Multi-teacher support | Natively supported | Unstable (neighbor graph conflicts with multiple teachers) |
| Architectural constraints | None (arbitrary student/teacher architectures) | None |
| Applicable modalities | Text / Vision / Molecular | Primarily vision |

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The theoretical path from Bayesian majority-vote classifier to mutual information upper bound is novel; Gaussian kernels as a systematic replacement for MSE in distillation is applied here for the first time.
- **Technical Depth**: ⭐⭐⭐⭐ — The theoretical derivation is rigorous and complete, with a clear equivalence relationship from conditional entropy to mutual information; the practical implementation is relatively straightforward.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Comprehensive evaluation across three modalities (NLP/Vision/Molecular) × multiple benchmarks × multiple model scales, with ablation studies covering teacher count and distillation method comparisons.
- **Value**: ⭐⭐⭐⭐ — Pre-computed embeddings enable efficient and scalable training with released models; gains on clustering/STS scenarios are limited.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] FLARE: Task-Agnostic Embedding Model Evaluation via Normalizing Flows](../../ACL2026/information_retrieval/flare_task-agnostic_embedding_model_evaluation_through_a_normalization_process.md)
- [\[NeurIPS 2025\] Scaling Language-Centric Omnimodal Representation Learning](scaling_language-centric_omnimodal_representation_learning.md)
- [\[NeurIPS 2025\] Retrieval is Not Enough: Enhancing RAG Reasoning through Test-Time Critique and Optimization](retrieval_is_not_enough_enhancing_rag_reasoning_through_test-time_critique_and_o.md)
- [\[ACL 2025\] LDIR: Low-Dimensional Dense and Interpretable Text Embeddings with Relative Representations](../../ACL2025/information_retrieval/ldir_low-dimensional_dense_and_interpretable_text_embeddings_with_relative_repre.md)
- [\[ICCV 2025\] Aligning Information Capacity Between Vision and Language via Dense-to-Sparse Feature Distillation](../../ICCV2025/information_retrieval/aligning_information_capacity_between_vision_and_language_via_dense-to-sparse_fe.md)

</div>

<!-- RELATED:END -->
