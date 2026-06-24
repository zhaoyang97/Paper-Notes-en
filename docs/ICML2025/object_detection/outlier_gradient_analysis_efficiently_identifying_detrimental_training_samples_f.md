---
title: >-
  [Paper Note] Outlier Gradient Analysis: Efficiently Identifying Detrimental Training Samples for Deep Learning Models
description: >-
  [ICML 2025 Oral][Object Detection][Data-Centric Learning] Outlier Gradient Analysis (OGA) is proposed to reformulate the identification of detrimental training samples in influence functions as an outlier detection problem in the gradient space. This sidesteps the high computational overhead of Hessian matrix inversion while outperforming traditional influence function methods on tasks such as noisy label correction, NLP data filtering, and LLM influence data identification.
tags:
  - "ICML 2025 Oral"
  - "Object Detection"
  - "Data-Centric Learning"
  - "Influence Functions"
  - "Gradient Space Outlier Detection"
  - "Detrimental Sample Identification"
  - "Noisy Label Correction"
date: 2026-05-08
content_hash: 69e0b4e563ae181b
---

# Outlier Gradient Analysis: Efficiently Identifying Detrimental Training Samples for Deep Learning Models

**Conference**: ICML 2025 Oral  
**arXiv**: [2405.03869](https://arxiv.org/abs/2405.03869)  
**Code**: None  
**Area**: Object Detection  
**Keywords**: Data-Centric Learning, Influence Functions, Gradient Space Outlier Detection, Detrimental Sample Identification, Noisy Label Correction

## TL;DR

Outlier Gradient Analysis (OGA) is proposed to reformulate the identification of detrimental training samples in influence functions as an outlier detection problem in the gradient space. This sidesteps the high computational overhead of Hessian matrix inversion while outperforming traditional influence function methods on tasks such as noisy label correction, NLP data filtering, and LLM influence data identification.

## Background & Motivation

One of the core challenges in Data-Centric Learning is identifying training samples that are detrimental to model performance. Influence Functions are the most commonly used tools for this task, estimating the contribution of individual samples by measuring the impact of infinitesimal weight perturbations on model parameters without retraining the model from scratch.

However, influence functions face two key bottlenecks in deep learning scenarios:

**Convexity Assumption**: Influence functions require the loss function to be strictly convex to guarantee the invertibility of the Hessian matrix, whereas the loss functions of deep models are typically non-convex.

**Computational Cost**: The computational cost of the Hessian matrix and its inverse is extremely high, making it virtually infeasible for deep models with huge parameter sizes and large-scale datasets.

Although many approximation methods (such as LiSSA, DataInf, Kronecker factorization, etc.) have been proposed to alleviate the computational burden, they fundamentally still rely on some form of the Hessian. This paper takes an alternative path—completely bypassing the Hessian and directly performing outlier detection in the gradient space to identify detrimental samples.

## Method

### Overall Architecture

The core mechanism of this paper can be summarized in three steps:

1. **Compute Gradients**: For each training sample $z_j = (x_j, y_j)$, compute the gradient of its loss with respect to the model parameters, $\nabla_{\hat{\theta}} \ell(z_j; \hat{\theta})$.
2. **Gradient Space Outlier Detection**: Form a gradient set $\mathcal{G}$ using the gradients of all training samples, and run an outlier detection algorithm $\mathcal{A}$ on this space.
3. **Pruning and Retraining**: Label the samples detected as outliers as detrimental samples, remove them, and retrain the model on the pruned dataset.

This process is formalized in Algorithm 1 as follows:

- Input: Training set $T$, loss function $\ell$, model parameters $\hat{\theta}$, outlier detection algorithm $\mathcal{A}$, pruning budget $k$
- Output: Set of detrimental/beneficial labels $L$, pruned training set $T^*$

### Key Designs

#### Bridge from Influence Functions to Outlier Detection

The formulation of the classic influence function is:

$$\mathcal{I}(z_j) = -\sum_{z \in T/V} \nabla_{\hat{\theta}} \ell(z; \hat{\theta})^\top \mathbf{H}_{\hat{\theta}}^{-1} \nabla_{\hat{\theta}} \ell(z_j; \hat{\theta})$$

The authors observe that this equation consists of three components: (1) the gradient term in the summation, (2) the inverse Hessian matrix, and (3) the gradient of sample $z_j$. The first two components are shared across all training samples, whereas **the third component, $\nabla_{\hat{\theta}} \ell(z_j; \hat{\theta})$, is the only term that uniquely depends on the specific sample $z_j$**. Consequently, it plays a decisive role in determining whether a sample is beneficial or detrimental.

Based on this, the authors propose two key assertions:

- **Observation 3.1**: For a converged ERM model, the vast majority of training samples contribute positively to the model, and detrimental samples constitute a very small minority—meaning detrimental samples are "outliers."
- **Assumption 3.2**: There exists an outlier detection algorithm capable of identifying detrimental samples in the gradient space, achieving performance equivalent to evaluating the discrete influence of samples via influence functions.

The core of this transition lies in: identifying detrimental samples using influence functions only requires knowing the "positive/negative" sign (discrete influence $\tilde{\mathcal{I}}$) rather than precise continuous influence scores. Outlier detection in the gradient space is precisely capable of providing this binary classification.

#### Selection of Outlier Detection Algorithms

The authors select three outlier detection methods:

1. **Isolation Forest (iForest)**: Linear time complexity, low memory overhead, friendly to high-dimensional gradient spaces; performs subspace detection by constructing iTree ensembles, which can effectively handle non-linearly separable outliers.
2. **L1-norm Thresholding**: Computes the L1-norm of each sample's gradient, marking those exceeding a threshold as outliers.
3. **L2-norm Thresholding**: Similar to L1-norm, but using the L2-norm.

Among these, iForest is the preferred method as it strikes the best balance between performance and efficiency.

#### Extended Design for LLM Tasks

For LLM influence data identification tasks, it is necessary to measure the similarity between training and test samples. The authors train an individual iForest estimator for each category of prompt (10 estimators in total), with each estimator built solely on the gradient space of training prompts from that specific category. For unseen test prompts, anomaly scores are generated by all category-specific iForest estimators, thereby achieving cross-set influence estimation.

### Loss & Training

This paper does not modify the original training loss function. Instead, it serves as a **post-processing data cleaning strategy**:

- First, train the model normally until convergence.
- Compute the gradients of the training samples.
- Run outlier detection in the gradient space and remove detected detrimental samples based on a pruning budget $k$ (default is 5% of the training set).
- Retrain the model on the pruned dataset.

This workflow of "training $\rightarrow$ detection $\rightarrow$ pruning $\rightarrow$ retraining" is simple and can be integrated with other methods (such as loss function corrections).

## Key Experimental Results

### Main Results

**Outlier Detection and Classification Accuracy on Synthetic Dataset (Two Half Moons)**:

| Method | Outlier Detection Accuracy (%) | Classification Accuracy after Pruning (%) |
|---|---|---|
| Multilayer Perceptron (baseline) | - | 90.0 |
| Exact Hessian | 90.0 | 90.0 |
| LiSSA | 82.0 | 91.0 |
| DataInf | 82.0 | 91.0 |
| Gradient Tracing | 82.0 | 91.0 |
| **Outlier Gradient (iForest)** | **96.0** | **96.0** |
| Outlier Gradient (L1) | 98.0 | 87.0 |
| Outlier Gradient (L2) | 98.0 | 87.0 |

**CIFAR-10N / CIFAR-100N Noisy Label Correction (ResNet-34)**:

| Method | Aggregate | Random | Worst | Noisy100 |
|---|---|---|---|---|
| Cross Entropy (baseline) | 90.87 | 89.17 | 82.27 | 57.36 |
| LiSSA | 91.49 | 90.05 | 83.38 | 60.48 |
| DataInf | 91.46 | 90.05 | 83.40 | 60.70 |
| Self-LiSSA | 92.07 | 89.58 | 83.01 | 59.48 |
| Outlier Gradient (L1) | 91.86 | **90.66** | **84.20** | 60.32 |
| **Outlier Gradient (L2)** | **92.21** | 90.25 | 82.99 | **61.40** |
| Outlier Gradient (iForest) | 91.36 | 90.20 | 83.72 | 60.99 |

### Ablation Study

| Configuration | Key Metric | Description |
|---|---|---|
| Pruning budget k=2.5% | Slightly lower accuracy than 5% | Insufficient pruning |
| Pruning budget k=5% (default) | Best balance | Consistent performance across multiple datasets |
| Pruning budget k=12.5% | Drop in accuracy | Over-pruning damages beneficial samples |
| iForest parameter tuning | Insensitive to parameters | Default parameters yield stable performance |
| Base model changed to ResNet-18 | Consistent trend | Method is architecture-independent |
| ImageNet scale-up | Consistent trend | Equally effective on large-scale datasets |

### Key Findings

1. **Synthetic Data Validation**: On non-convex MLP models, traditional influence function scores confound detrimental and normal samples (poor discrimination of influence scores), whereas detrimental samples are clearly non-linearly separable in the gradient space and effectively detected by iForest.
2. **Computational Efficiency Advantage**: The time complexity of OGA is linear (with respect to both the number of samples and parameters), which is significantly superior to methods requiring the Hessian inverse. The runtime on CIFAR-10N is several orders of magnitude faster than LiSSA.
3. **Perfect Performance in LLM Scenarios**: On three benchmarks using Llama-2-13B-chat, OGA achieves a perfect score of 1.000 in both Class Detection AUC and Recall, significantly outperforming DataInf (AUC 0.999) and Gradient Tracing (some AUCs as low as 0.72).
4. **NLP Data Filtering**: During LoRA fine-tuning of RoBERTa on GLUE tasks, OGA significantly outperforms all baseline methods on QNLI, SST2, and QQP.

## Highlights & Insights

1. **Simple yet Profound Core Concept**: Simplifying complex influence function computation into a two-step "compute gradients $\rightarrow$ outlier detection" operation, with clear theoretical motivation and self-consistent logic behind Observation 3.1 and Assumption 3.2.
2. **Enormous Advantages of Being Hessian-Free**: Completely bypassing the Hessian matrix makes the method directly applicable to any deep model without the restrictions of convexity assumptions or the accumulation of errors introduced by Hessian approximations.
3. **High Generalizability**: From vision models (ResNet) to NLP Transformers (RoBERTa) and LLMs (Llama-2-13B), the proposed method shows consistent effectiveness across models of various scales and modalities.
4. **Complementary to Existing Methods**: As a data pruning method, OGA can be combined with noise-tolerant learning methods that modify the loss function or model architecture to potentially achieve greater performance gains.

## Limitations & Future Work

1. **Selection of Pruning Budget $k$**: This remains a common hyperparameter challenge in outlier detection—how to automatically determine the optimal pruning ratio, which currently still requires manual tuning.
2. **Requirement of Double Training**: The pipeline inherently requires training the model twice (once to collect gradients and a second time to retrain after pruning), which still incurs notable costs for extremely large models.
3. **Gradient Dimensionality Reduction**: The paper does not thoroughly explore whether the high dimensionality of gradient vectors impacts outlier detection effectiveness when the model parameters scale up (e.g., to tens of billions in LLMs). The low-dimensionality of LoRA parameters might be a key factor in the success of the LLM experiments.
4. **Validation Limited to Discrete Influence**: The proposed method only makes binary "detrimental/beneficial" classifications. It cannot provide a continuous ranking of influence like traditional influence functions, restricting its use in domains requiring fine-grained data valuation.
5. **Preliminary Adaptation to Distribution Shift**: Although preliminary attempts were made using a semi-supervised OneClassSVM, the performance under substantial train-test distribution shifts warrants more systematic validation.

## Related Work & Insights

- **Koh & Liang (2017)**: The pioneering work on influence functions in deep learning, serving as the starting point for this paper.
- **DataInf (Kwon et al., 2024)**: An efficient influence estimation method, representing one of the primary baselines compared in this study, which similarly addresses the efficiency challenges in large models.
- **Pruthi et al. (2020) Gradient Tracing**: A representative method that directly utilizes gradients for influence estimation, though it lacks an outlier detection perspective.
- **Isolation Forest (Liu et al., 2008)**: The core outlier detection component utilized in this paper, whose linear complexity and subspace ensemble properties are key to the success of this method.
- **Insights for Object Detection**: Although the experiments in this paper are primarily focused on image classification and NLP, the core idea of OGA can be directly applied to data cleaning in object detection—such as detecting incorrectly annotated bounding boxes or cleaning adversarial samples—especially given the ubiquity of noisy annotations in large-scale detection datasets (e.g., COCO, Objects365).

## Rating

| Dimension | Score (1-5) | Description |
|---|---|---|
| Novelty | ⭐⭐⭐⭐ | Novel perspective; the transition from influence functions to outlier detection is simple yet powerful |
| Theoretical Depth | ⭐⭐⭐ | Observations and assumptions are reasonable but heavily empirically driven, lacking rigorous theoretical proofs |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ | Broad coverage spanning synthetic $\rightarrow$ vision $\rightarrow$ NLP $\rightarrow$ LLMs, with comprehensive ablation studies |
| Practical Value | ⭐⭐⭐⭐ | Simple, versatile, and highly efficient; directly applicable to real-world pipelines |
| Writing Quality | ⭐⭐⭐⭐ | Clearly structured with rich tables and figures |
| **Overall** | **⭐⭐⭐⭐** | A solid piece of work with practical impact in the field of data-centric learning |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information](../../ECCV2024/object_detection/yolov9_learning_what_you_want_to_learn_using_programmable_gradient_information.md)
- [\[ICML 2026\] EARL: Towards a Unified Analysis-Guided Reinforcement Learning Framework for Egocentric Interaction Reasoning and Pixel Grounding](../../ICML2026/object_detection/earl_towards_a_unified_analysis-guided_reinforcement_learning_framework_for_egoc.md)
- [\[ICCV 2025\] Uncertainty-Aware Gradient Stabilization for Small Object Detection](../../ICCV2025/object_detection/uncertainty-aware_gradient_stabilization_for_small_object_detection.md)
- [\[CVPR 2025\] Integration of deep generative Anomaly Detection algorithm in high-speed industrial line](../../CVPR2025/object_detection/integration_of_deep_generative_anomaly_detection_algorithm_in_high-speed_industr.md)
- [\[ECCV 2024\] Weak-to-Strong Compositional Learning from Generative Models for Language-based Object Detection](../../ECCV2024/object_detection/weak-to-strong_compositional_learning_from_generative_models_for_language-based_.md)

</div>

<!-- RELATED:END -->
