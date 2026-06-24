---
title: >-
  [Paper Note] Inference-Time Decomposition of Activations (ITDA): A Scalable Approach to Interpreting Large Language Models
description: >-
  [ICML 2025][Interpretability][Sparse Autoencoders] ITDA is proposed, an inference-time activation decomposition method based on Matching Pursuit. It achieves comparable reconstruction performance at only 1% of the training cost of SAEs, scales to a 405B parameter model, and inherently supports cross-model representation comparison.
tags:
  - "ICML 2025"
  - "Interpretability"
  - "Sparse Autoencoders"
  - "Matching Pursuit"
  - "Dictionary Learning"
  - "Representational Similarity"
date: 2026-05-08
content_hash: e179d4ee7a7133ea
---

# Inference-Time Decomposition of Activations (ITDA): A Scalable Approach to Interpreting Large Language Models

**Conference**: ICML 2025  
**arXiv**: [2505.17769](https://arxiv.org/abs/2505.17769)  
**Code**: [github.com/pleask/itda](https://github.com/pleask/itda)  
**Area**: Interpretability  
**Keywords**: Sparse Autoencoders, Interpretability, Matching Pursuit, Dictionary Learning, Representational Similarity

## TL;DR

ITDA is proposed, an inference-time activation decomposition method based on Matching Pursuit. It achieves comparable reconstruction performance at only 1% of the training cost of SAEs, scales to a 405B parameter model, and inherently supports cross-model representation comparison.

## Background & Motivation

Sparse Autoencoders (SAEs) are currently the mainstream approach to decomposing LLM activations into interpretable latent variables, but they suffer from two key bottlenecks:

**Prohibitive Training Costs**: SAEs require hundreds of millions to billions of tokens of model activation data for training, and the parameter size of the SAE itself can exceed that of the LLM being analyzed (e.g., the SAE for Gemma 2 2B surprisingly has 5 billion parameters). Currently, open-source SAEs only cover models with $\le$ 27B parameters.

**Incomparability Across Models**: The latent variables of SAEs are learned via gradient descent in the activation space of specific models. Consequently, there is no inherent correspondence between the SAE latents of different models, rendering direct cross-model comparison impossible.

Inspired by relative representation similarity methods (Moschella et al., 2022)—which advocate that while the absolute representations of different models vary, the angular relationships between elements in the representation space remain invariant—the authors propose a lightweight alternative.

## Method

### Overall Architecture

The core idea of ITDA is: **instead of training an encoder, it uses the Matching Pursuit algorithm during inference to decompose activations onto a dictionary composed of real activations**. The overall process consists of three steps:

1. **Dictionary Construction** (Offline, Greedy Sampling): Iteratively selects activation vectors from training data to construct a dictionary $\mathbf{D} \in \mathbb{R}^{n \times d}$
2. **Sparse Coding** (Online, Matching Pursuit): For a target activation $\mathbf{x}$, solve for sparse coefficients $\mathbf{a} = \text{MP}(\mathbf{x}, \mathbf{D}, L_0)$ using Matching Pursuit.
3. **Reconstruction**: $\hat{\mathbf{x}} = \mathbf{a}\mathbf{D}$

Comparison with SAEs: SAEs utilize a learned encoder-decoder framework $\mathbf{f}(\mathbf{x}) = \sigma(\mathbf{W}^{\text{enc}}\mathbf{x} + \mathbf{b}^{\text{enc}})$ and $\hat{\mathbf{x}} = \mathbf{W}^{\text{dec}}\mathbf{f} + \mathbf{b}^{\text{dec}}$. In contrast, ITDA completely eliminates the parameterized encoder, relying instead on inference-time optimization.

### Key Designs

#### 1. Theoretical Foundation of Relative Representations

The absolute representations $e^{(i)} = E_\theta(\mathbf{x}^{(i)})$ learned by different models may differ by a rotation or affine transformation $T$. However, the angular relationships remain invariant:

$$\angle(\mathbf{e}^{(i)}, \mathbf{e}^{(j)}) = \angle(T\mathbf{e}^{(i)}, T\mathbf{e}^{(j)})$$

ITDA utilizes cosine similarity $S_C(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{||\mathbf{a}|| \cdot ||\mathbf{b}||}$ as the similarity function, preserving this angular invariance naturally.

#### 2. Inference-Time Sparse Coding (Matching Pursuit)

Given input $\mathbf{x} \in \mathbb{R}^d$ and dictionary $\mathbf{D}$, the sparse coding problem is solved:

$$\min_{\mathbf{a} \in \mathbb{R}^n} ||\mathbf{x} - \mathbf{a}\mathbf{D}|| \quad \text{s.t.} \quad ||\mathbf{a}||_0 \leq L_0$$

Each iteration of the MP algorithm involves:

- **Selection**: Find the dictionary atom $\mathbf{d}_j$ with the highest correlation to the current residual.
- **Update**: Project the residual onto the direction of the newly selected atom and subtract it.
- **Repeat**: Iterate for $L_0$ steps to reach the target sparsity.

The correlation here is equivalent to unnormalized cosine similarity, aligning with the relative representation framework.

#### 3. Greedy Dictionary Construction

Unlike Moschella et al., who randomly sample anchor points, ITDA constructs the dictionary **deterministically and iteratively**:

**Algorithm 1: ITDA Dictionary Training**

Input: Training data $\{x_i\}$, sparsity $L_0$, threshold $\tau$

1. Initialize dictionary $\mathbf{D}$ (selecting high-frequency activations or random sampling)
2. For each sample $\mathbf{x}$ in training batch $\mathcal{B}$:
    - Compute sparse coding $\mathbf{a} = \text{OMP}(\mathbf{x}, \mathbf{D}, $L_0$)$
    - Reconstruct $\hat{\mathbf{x}} = \mathbf{a}\mathbf{D}$
    - Compute reconstruction loss $\ell(\mathbf{x}) = ||\mathbf{x} - \hat{\mathbf{x}}||_2^2$
    - If $\ell(\mathbf{x}) > \tau$, append normalized $\mathbf{x}$ to the dictionary
3. Filter duplicate activations in the dictionary

The key parameter $\tau$ (loss threshold) controls the dictionary size: **lower $\tau$ $\to$ larger dictionary $\to$ lower reconstruction error**. This contrasts with the fixed dictionary size design of SAEs—where the dictionary size is preset before training, whereas ITDA's is adaptively determined by a reconstruction quality threshold.

#### 4. Interpretable Labels

A unique advantage of ITDA dictionary atoms: each atom inherently possesses an interpretable label—specifically, the **prompt + token** pair that produced the activation. While SAE latents require separate automated interpretability analysis (e.g., inspecting highly activating samples) to understand their meaning, ITDA labels provide natural semantic information.

#### 5. Cross-Model Representation Similarity Metric

Based on the interpretable labels of the ITDA dictionary, the authors propose a novel representational similarity metric:

- Build an ITDA dictionary for each of the two models, where each dictionary atom corresponds to a (prompt, token) pair.
- Compare the (prompt, token) label sets of the two dictionaries using **Jaccard similarity (IoU)**.
- This bypasses direct comparison of activation values between models (which exist in different spaces) and instead compares "which inputs are important to the models".

### Loss & Training

ITDA does not undergo a traditional loss function optimization process. Its "training" is essentially dictionary construction, with key hyperparameters being:

- **Sparsity $L_0$**: The number of dictionary atoms used per activation.
- **Threshold $\tau$**: The reconstruction error threshold that determines when to add a new dictionary atom.
- **Dictionary Initialization**: Using the most high-frequency activations as initial dictionary elements.

In comparison to the SAE training loss: $\mathcal{L}(\mathbf{x}) = ||\mathbf{x} - \hat{\mathbf{x}}||_2^2 + \lambda \mathcal{S}(\mathbf{f}(\mathbf{x})) + \alpha \mathcal{L}_{\text{aux}}$, ITDA requires neither sparsity regularization (sparsity is guaranteed by the $L_0$ hard constraint) nor auxiliary losses.

## Key Experimental Results

### Main Results

#### Training Efficiency Comparison

| Method | Training Tokens | GPT-2 Training Time | Max Model Supported |
|------|-------------|--------------|-------------|
| SAE | Hundreds of millions to billions | Hours | 27B (Open-sourced) |
| **ITDA** | **~1 million** | **Minutes** | **405B** |
| Gain | **~100×** | **~100×** | **~15×** |

#### Reconstruction Performance Comparison

| Model | Method | Reconstruction Quality | Cross-Entropy Degradation |
|------|------|---------|-----------|
| Pythia series | SAE | Baseline | Baseline |
| Pythia series | **ITDA** | **Comparable** | **Similar or slightly worse** |
| Gemma-2 | SAE | Baseline | Baseline |
| Gemma-2 | **ITDA** | Worse | Significantly worse |
| Llama-3.1 70B | **ITDA** | ✓ (First time) | — |
| Llama-3.1 405B | **ITDA** | ✓ (First time) | — |

### Ablation Study

| Configuration | Key Metrics | Description |
|------|---------|------|
| Threshold $\tau$ ↓ | Dictionary grows, reconstruction error ↓ | $\tau$ is the core knob controlling the accuracy-efficiency trade-off |
| Threshold $\tau$ ↑ | Dictionary shrinks, reconstruction error ↑ | In extreme cases, minimal atoms can be used |
| Randomly sampled dictionary | Poor reconstruction | Greedy strategy significantly outperforms random sampling |
| Deterministic greedy dictionary | Good reconstruction | Ensures reproducibility across runs |
| Sparsity $L_0$ ↑ | Reconstruction ↑, Compute ↑ | Controls number of non-zero elements in sparse codes |

### Key Findings

1. **Representational Similarity SOTA**: The ITDA metric, based on Jaccard dictionary distance, outperforms CKA, SVCCA, and relative representation methods on the layer-matching benchmark of Kornblith et al., achieving state-of-the-art performance.

2. **Sparsity Scaling Breakthrough**: ITDA applies sparse dictionary decomposition to 70B and 405B parameter LLMs for the first time, which is an order of magnitude larger than the largest models covered by current open-source SAEs.

3. **Comparable Automated Interpretability Scores**: The automated interpretability scores of ITDA dictionary atoms match those of SAE latents, demonstrating that greedily sampled real activations also possess monosemantic properties.

4. **Model Dependency**: ITDA performs close to SAEs on the Pythia model series, but exhibits much more severe cross-entropy degradation on Gemma-2, indicating model dependency in the method's effectiveness.

5. **Reproduction of Layer Freezing Experiments**: The ITDA dictionary distance metric successfully reproduces the conclusions from the layer-freezing experiments of Raghu et al. (2017), further validating its reliability as a representational similarity tool.

## Highlights & Insights

1. **Remarkably Simple Concept**: It replaces deep learning (SAE) with a classical signal processing method (Matching Pursuit), addressing a computational bottleneck through a "no-learning" paradigm. The 100× acceleration stems from simplifying "learning a dictionary + encoder" into "sampling a dictionary + inference encoding".

2. **Labels as Interpretability**: The (prompt, token) labels of ITDA atoms are an elegant design—while SAEs require auxiliary steps to explain the meaning of each latent variable, ITDA's dictionary inherently carries semantic information.

3. **A New Paradigm for Cross-Model Comparison**: It translates "comparing activation spaces of different models" (difficult, involving alignment issues) into "comparing which inputs are selected into the dictionary" (easy, only requiring the Jaccard similarity of label sets). This perspective could inspire fields like model merging and knowledge distillation.

4. **Adaptive Dictionary Size**: Controlling the dictionary scale via the threshold $\tau$ rather than a preset size allows the method to adaptively fit the actual complexity of the activation space—simpler representation spaces use smaller dictionaries, and complex ones use larger ones.

## Limitations & Future Work

1. **Underperformance on Gemma-2**: The cross-entropy degradation on Gemma-2 is significantly worse than that of SAEs, indicating that Matching Pursuit has insufficient expressive power in the activation space of certain model architectures. This suggests a need to explore other inference-time optimization algorithms (such as improved variants of OMP or FISTA).

2. **Inference-Time Computational Overhead**: Matching Pursuit requires calculating correlations with the entire dictionary during inference; larger dictionaries lead to slower inference. In contrast, SAE encoding only requires a single matrix multiplication + activation function, making it more computationally efficient during inference.

3. **Unexplored Model Diffing**: The authors mention the potential of using ITDA to identify behavioral discrepancies caused by fine-tuning (e.g., scheming, sycophancy), but no empirical experiments are presented in the paper, which remains an important future direction.

4. **Dictionary Reliance on Training Data**: All dictionary atoms originate from real activations within the training set, potentially failing to cover activation patterns of out-of-distribution (OOD) inputs, whereas SAE decoders can reconstruct learned combinations of unseen directions.

5. **Redundancy in Batch Processing**: Similar inputs in the same batch may be redundantly annexed to the dictionary. While post-processing filters exist, the efficiency of this resolution remains suboptimal.

## Related Work & Insights

- **Return of Classical Sparse Coding** (K-SVD, Matching Pursuit, FISTA): The deep learning era tends to use neural networks for everything; this work demonstrates that classical algorithms still hold advantages in specific scenarios.
- **Crosscoders** (Lindsey et al., 2024): Training SAEs on multi-model representations to discover cross-model features; ITDA's labeling approach offers a much lighter alternative.
- **Relative Representations** (Moschella et al., 2022): ITDA is a natural extension of this idea—progressing from simple cosine similarity vectors to sparse decomposition via Matching Pursuit.
- **Future Directions**: Combining ITDA with circuit analysis to rapidly analyze computational circuits in ultra-large models.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Introduces classical Matching Pursuit to LLM interpretability, presenting a novel and rational approach.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Multi-model comparisons and scalability are well-demonstrated, though the underperformance on Gemma-2 warrants deeper analysis.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — The motivation is clear, the method description is rigorous, and comparison with SAE is woven throughout the entire text.
- **Value**: ⭐⭐⭐⭐ — The 100× speedup and 405B scalability offer significant practical utility, opening new avenues for cross-model comparisons.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Scalable Circuit Learning for Interpreting Large Language Models](../../ICML2026/interpretability/scalable_circuit_learning_for_interpreting_large_language_models.md)
- [\[ICML 2025\] Supernova Event Dataset: Interpreting Large Language Models' Personality through Critical Event Analysis](supernova_event_dataset_interpreting_large_language_models_personality_through_c.md)
- [\[ACL 2025\] Mechanistic Interpretability of Emotion Inference in Large Language Models](../../ACL2025/interpretability/mechanistic_interpretability_of_emotion_inference_in_large_language_models.md)
- [\[ICLR 2026\] Structural Inference: Interpreting Small Language Models with Susceptibilities](../../ICLR2026/interpretability/structural_inference_interpreting_small_language_models_with_susceptibilities.md)
- [\[ICML 2025\] A Reasoning-Based Approach to Cryptic Crossword Clue Solving](a_reasoning-based_approach_to_cryptic_crossword_clue_solving.md)

</div>

<!-- RELATED:END -->
