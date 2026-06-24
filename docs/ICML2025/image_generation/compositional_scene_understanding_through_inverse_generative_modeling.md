---
title: >-
  [Paper Note] Compositional Scene Understanding through Inverse Generative Modeling
description: >-
  [ICML2025][Image Generation][compositional generation] This paper proposes the Inverse Generative Modeling (IGM) framework, which reformulates scene understanding tasks as an inversion problem of searching for optimal conditional parameters within compositional generative models. By composing multiple small diffusion models to represent complex scenes, the method achieves robust out-of-distribution generalization capabilities and directly leverages pre-trained text-to-image m…
tags:
  - "ICML2025"
  - "Image Generation"
  - "compositional generation"
  - "inverse generative modeling"
  - "diffusion models"
  - "scene understanding"
  - "object discovery"
date: 2026-05-08
content_hash: f5d08461dfdbec34
---

# Compositional Scene Understanding through Inverse Generative Modeling

**Conference**: ICML2025  
**arXiv**: [2505.21780](https://arxiv.org/abs/2505.21780)  
**Code**: [energy-based-model/compositional-inference](https://energy-based-model.github.io/compositional-inference)  
**Area**: Image Generation  
**Keywords**: compositional generation, inverse generative modeling, diffusion models, scene understanding, object discovery  

## TL;DR

This paper proposes the Inverse Generative Modeling (IGM) framework, which reformulates scene understanding tasks as an inversion problem of searching for optimal conditional parameters within compositional generative models. By composing multiple small diffusion models to represent complex scenes, the method achieves robust out-of-distribution generalization capabilities and directly leverages pre-trained text-to-image models for zero-shot multi-object perception.

## Background & Motivation

Traditional scene understanding tasks are dominated by discriminative models that learn direct mappings from input images to visual attributes. However, numerous studies have shown that the performance of discriminative models drops significantly under test distribution shifts, with even minor distribution changes leading to severe performance decay.

Generative models have long been recognized for their superior generalization potential, but it was not until the recent advent of diffusion models that they demonstrated competitive results on visual reasoning tasks. However, existing generative inference methods primarily focus on single-label classification tasks. Performing broader scene understanding tasks (such as object discovery and multi-object classification) on scenes much more complex than the training set remains an open question.

The core motivation of this paper is inspired by Richard Feynman's quote, "What I cannot create, I do not understand"—understanding through generation. The authors argue that if a generative model can accurately reconstruct a scene, its conditional parameters naturally capture the understanding of that scene. More crucially, through compositional modeling, simple scene fragments seen during training can be composed to understand far more complex scenes encountered at test time.

## Method

### Overall Architecture

This paper formalizes scene understanding as an **inverse generative modeling** problem. Given an image $\boldsymbol{x}$, the goal is to infer a set of visual concepts $\{c^1, c^2, \cdots, c^K\}$ that describe the image. The overall architecture consists of two stages:

1. **Compositional Generative Modeling**: Construct a generative model composed of multiple small generative models.
2. **Inverse Inference**: Solve an optimization problem to find the conditional parameters that best fit the given image.

### Key Design 1: Compositional Generative Models

To model the conditional probability distribution $p(\boldsymbol{x}|c^1, c^2, \ldots, c^K)$, the authors employ a product-of-experts approximation:

$$p(\boldsymbol{x}|c^1, \ldots, c^K) \propto \prod_{k=1}^{K} p(\boldsymbol{x}|c^k)$$

Each $p(\boldsymbol{x}|c^k)$ is parameterized as $e^{-E_\theta(\boldsymbol{x}|c^k)}$ using an Energy-Based Model (EBM), meaning the product distribution simplifies to a sum of energies:

$$p(\boldsymbol{x}|c^1, \ldots, c^K) \propto e^{-\sum_{k=1}^{K} E_\theta(\boldsymbol{x}|c^k)}$$

Approximating $\nabla_{\boldsymbol{x}} E_\theta(\boldsymbol{x}|c^k)$ with the denoising function of a diffusion model $\epsilon_\theta(\boldsymbol{x}^t, t | c^k)$, the combined denoising function is:

$$\epsilon_\theta^{\text{comb}}(\boldsymbol{x}^t, t) = \sum_{k=1}^{K} \epsilon_\theta(\boldsymbol{x}^t, t | c^k)$$

### Key Design 2: Joint Training of Combined Score Functions

Distinct from prior works that compose independently trained denoising functions only at test time, this work proposes to directly train the combined score function:

$$\mathcal{L}_\theta = \mathbb{E}_{\boldsymbol{x}, \epsilon, t} \left\| \epsilon - \sum_{k=1}^{K} \epsilon_\theta(\boldsymbol{x}^t, t | c^k) \right\|^2$$

This joint training enables each denoising function to perform more accurately when composed, while still allowing the addition of more terms at test time to construct more complex scenes.

### Key Design 3: Inverse Inference

Scene understanding is formalized as maximizing the log-likelihood of the given image:

$$\hat{c}^1, \ldots, \hat{c}^K = \arg\min_{c^1, \ldots, c^K} \mathbb{E}_{\epsilon, t} \left\| \epsilon - \sum_{k=1}^{K} \epsilon_\theta(\boldsymbol{x}^t, t | c^k) \right\|^2$$

- **Discrete Concept Inference**: Enumerate all possible concept configurations and select the one with the minimal denoising error.
- **Continuous Concept Inference**: Employ stochastic gradient descent with a multi-random-initialization strategy to prevent local optima.

### Key Design 4: Estimation of the Number of Concepts

By solving the optimization problem under different values of $K$, the $\hat{K}$ that maximizes the likelihood (minimizes the denoising error) is selected:

$$\hat{K} = \arg\min_{K \in [K_{min}, K_{max}]} \left\{ \min_{c^1, \ldots, c^K} \mathbb{E}_{\epsilon, t} \left\| \epsilon - \sum_{k=1}^{K} \epsilon_\theta(\boldsymbol{x}^t, t | c^k) \right\|^2 \right\}$$

### Loss & Training

The training stage employs the standard denoising diffusion objective, but performs end-to-end training directly on the combined denoising function:

$$\mathcal{L}_\theta = \mathbb{E}_{\boldsymbol{x}, \epsilon, t} \left\| \epsilon - \sum_{k=1}^{K} \epsilon_\theta(\boldsymbol{x}^t, t, c^k) \right\|^2$$

During the inference phase for optimizing concept parameters, stochastic gradient descent (SGD) is used. It requires only a single sample of $\epsilon_n, t_n$ per step, reducing the sampling complexity from $N$ to 1.

## Key Experimental Results

### Main Results 1: Object Discovery (CLEVR Dataset)

The training set contains images with 3-5 objects, and the testing is split into in-distribution (3-5 objects) and out-of-distribution (6-8 objects):

| Model | In-Dist. Perception Rate↑ | In-Dist. Estimation Error↓ | OOD Perception Rate↑ | OOD Estimation Error↓ |
|------|-------|---------|-------|---------|
| ResNet-50 | 5.3% | 19.4e-2 | 2.9% | 19.7e-2 |
| SlotAttn | 80.4% | 8.7e-4 | 53.3% | 1.3e-3 |
| DINOSAUR | 82.5% | 8.4e-4 | 59.0% | 1.2e-3 |
| GC | 82.2% | 6.0e-4 | 58.7% | 1.2e-3 |
| **IGM (Ours)** | **94.7%** | **1.4e-4** | **85.3%** | **3.5e-4** |

### Main Results 2: Facial Attribute Prediction (CelebA Dataset)

The training set only contains female faces, and the out-of-distribution test set contains male faces:

| Model | In-Distribution Accuracy | Out-of-Distribution Accuracy |
|------|---------|---------|
| ResNet-50 | 79.6% | 62.2% |
| GC | 79.1% | 61.7% |
| **IGM (Ours)** | **80.8%** | **65.6%** |

### Main Results 3: Zero-Shot Multi-Object Perception

Utilizing pre-trained Stable Diffusion without any additional training:

| Model | Accuracy↑ |
|------|------|
| Diffusion Classifier | 70.4% |
| DC Variant | 73.2% |
| **IGM (Ours)** | **87.3%** |

### Ablation Study

| Variant | In-Distribution Perception Rate | Out-of-Distribution Perception Rate |
|------|---------|---------|
| IGM w/o Multi-Initialization | 72.8% | 68.0% |
| IGM w/ Multi-Initialization | 94.7% | 85.3% |

The multi-random-initialization strategy yields a boost of approximately 22% in-distribution and 17% out-of-distribution performance.

### Key Findings

1. Compositional modeling has a significant advantage in out-of-distribution (OOD) generalization: it outperforms the best baseline by 26.6% on the OOD object discovery task.
2. Concept number inference is highly effective: the ground-truth number of concepts consistently yields the lowest denoising error.
3. Zero-shot scenarios: Directly utilizing pre-trained Stable Diffusion achieves an 87.3% multi-object perception accuracy.
4. Multi-initialization is crucial for the success of continuous concept inference.

## Highlights & Insights

1. **Elegant Problem Reformulation**: By converting the discriminative task of scene understanding into an inverse problem of generative models, this approach leverages the inherent compositional generalization capability of generative models.
2. **Compositionality as Generalization**: By decomposing a complex scene into a combination of simple concepts, it naturally generalizes to more complex scenes without having observed them during training.
3. **Unified Framework**: The same framework can handle discrete concepts (classification), continuous concepts (coordinates), and pre-trained models (zero-shot), demonstrating exceptional flexibility.
4. **Denoising Error as a Likelihood Proxy**: It cleverly uses the denoising error of diffusion models to approximate log-likelihood, bypassing the need to explicitly compute the partition function.
5. **Minimal Training Overhead**: The zero-shot scheme directly reuses pre-trained models without requiring any additional training.

## Limitations & Future Work

1. **Low Inference Efficiency**: Discrete concept inference requires enumerating all possible configurations, leading to exponential computational complexity $O(M^K)$ when the number of concepts is large.
2. **Concept Independence Assumption**: It assumes independence among concepts in the scene, neglecting actual object interactions (such as occlusions and spatial relationships), which may lead to errors in realistic scenes.
3. **Sensitivity of Continuous Inference to Initialization**: Although the multi-initialization strategy mitigates local optima issues, it increases computational overhead.
4. **Limited Evaluation Scale**: Zero-shot multi-object perception is only validated on a small dataset of three animal categories, lacking large-scale evaluations in real-world scenes.
5. **Product Approximation Bias**: Approximating the joint distribution as the product of marginal distributions introduces bias, which may fail in highly correlated scenarios.

## Related Work & Insights

- **Generative Classifiers**: Li et al. (2024) leverage the denoising error of diffusion models for single-label classification, which this work extends to multi-concept compositional inference.
- **Compositional Generative Models**: Du & Kaelbling (2024) propose a compositional generative modeling paradigm; based on this, this work utilizes composition for scene understanding rather than generation.
- **Slot Attention**: Locatello et al. (2020) discover objects using attention mechanisms, but demonstrate limited generalization ability.
- **DINOSAUR**: Seitzer et al. (2022) combine self-supervised features for object discovery.

## Rating

⭐⭐⭐⭐ (4/5)

Strong novelty, elegant method, and convincing motivation for compositional generalization. The performance gain on out-of-distribution object discovery is significant. However, inference efficiency and evaluation scale are clear weaknesses, and the zero-shot experimental setup is overly simplistic.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] DCTdiff: Intriguing Properties of Image Generative Modeling in the DCT Space](dctdiff_intriguing_properties_of_image_generative_modeling_in_the_dct_space.md)
- [\[ICML 2026\] Compositional Generative Modeling from Decentralized Data](../../ICML2026/image_generation/compositional_generative_modeling_from_decentralized_data.md)
- [\[ICML 2025\] Action-Minimization Meets Generative Modeling: Efficient Transition Path Sampling with the Onsager-Machlup Functional](action-minimization_meets_generative_modeling_efficient_transition_path_sampling.md)
- [\[ICML 2025\] Understanding and Mitigating Memorization in Generative Models via Sharpness of Probability Landscapes](understanding_and_mitigating_memorization_in_generative_models_via_sharpness_of_.md)
- [\[ICML 2025\] Efficient Generative Modeling with Residual Vector Quantization-Based Tokens](efficient_generative_modeling_with_residual_vector_quantization-based_tokens.md)

</div>

<!-- RELATED:END -->
