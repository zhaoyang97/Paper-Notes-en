---
title: >-
  [Paper Note] Beyond Communication Overhead: A Multilevel Monte Carlo Approach for Mitigating Compression Bias in Distributed Learning
description: >-
  [ICML 2025][Model Compression][Distributed Learning] This paper proposes a gradient compression scheme based on Multilevel Monte Carlo (MLMC), which constructs statistically unbiased gradient estimators from biased compressors, turning compression bias into manageable variance. This allows the approach to enjoy the theoretical guarantees of unbiased methods while maintaining the empirical efficiency of biased compressors. Combined with adaptive probability optimization…
tags:
  - "ICML 2025"
  - "Model Compression"
  - "Distributed Learning"
  - "Gradient Compression"
  - "Multilevel Monte Carlo"
  - "Unbiased Estimation"
  - "Communication Efficiency"
date: 2026-05-08
content_hash: 0690cd2384c77db1
---

# Beyond Communication Overhead: A Multilevel Monte Carlo Approach for Mitigating Compression Bias in Distributed Learning

**Conference**: ICML 2025  
**arXiv**: [2507.05508](https://arxiv.org/abs/2507.05508)  
**Code**: None  
**Area**: Model Compression  
**Keywords**: Distributed Learning, Gradient Compression, Multilevel Monte Carlo, Unbiased Estimation, Communication Efficiency

## TL;DR

This paper proposes a gradient compression scheme based on Multilevel Monte Carlo (MLMC), which constructs statistically unbiased gradient estimators from biased compressors, turning compression bias into manageable variance. This allows the approach to enjoy the theoretical guarantees of unbiased methods while maintaining the empirical efficiency of biased compressors. Combined with adaptive probability optimization, its superiority is validated on BERT fine-tuning and CIFAR-10.

## Background & Motivation

### Background
In distributed learning, multiple worker nodes calculate gradients in parallel and send them to a central server. Communication overhead is the primary bottleneck, and gradient compression is a standard solution.

### Limitations of Prior Work

**Unbiased Compressors** (such as Rand-k, QSGD):
- Guarantee $\mathbb{E}[C(x)] = x$, which simplifies theoretical analysis.
- Randomly select gradient components without prioritizing the most important information $\rightarrow$ poor empirical performance.

**Biased Compressors** (such as Top-k, SignSGD):
- Retain the most important information, providing good empirical results.
- $\mathbb{E}[C(x)] \neq x$, introducing bias.
- Require additional error correction mechanisms (such as Error Feedback), which degrades theoretical guarantees.
- Limited scalability for parallelization: EF21-SGDM only supports up to $M = O(\sqrt{T})$ machines without degradation.

### Key Challenge
Biased methods show excellent empirical performance but suffer from poor theoretical properties and limited parallel scalability; unbiased methods have solid theoretical guarantees but perform poorly in practice. Can one achieve the best of both worlds?

### Key Insight
This work exploits the core property of the MLMC method—**converting bias into variance**—to construct unbiased estimators using biased compressors. Since the estimator is unbiased, it can be directly integrated into standard data-parallel SGD analysis frameworks, yielding convergence guarantees of the same form as uncompressed SGD.

## Method

### Overall Architecture

**Core Idea**: Given a biased multilevel compressor $C^l$ ($l$ from 1 to $L$, where $L$ corresponds to no compression), construct the MLMC gradient estimate:

$$\tilde{g}_{t,i} = g_{t,i}^0 + \frac{1}{p^l}(g_{t,i}^l - g_{t,i}^{l-1}), \quad l \sim p^l$$

where $g_{t,i}^l = C^l(v_{t,i})$, $g_{t,i}^0 = 0$, and $\{p^l\}$ is the level probability distribution.

**Key Lemma 3.2**: Regardless of the multilevel compressor used, the MLMC estimator is always an unbiased estimate of the true gradient:
$$\mathbb{E}[\tilde{g}_{t,i} | x_t] = \nabla f_i(x_t), \quad \forall t, i$$

### Key Designs

#### 1. Multilevel Compressor Definition

Existing compressors are parameterized by level $l$:
- **Top-k**: $l$ corresponds to the number of retained elements $k$; $L = d$ (no compression).
- **s-Top-k**: Divides the vector into segments of length $s$, where $l$ corresponds to the number of retained segments.
- **Fixed-point Quantization**: $l$ corresponds to the number of retained significant bits (1 to 63).
- **Floating-point Quantization**: $l$ corresponds to the number of significant bits in the mantissa.

#### 2. Fixed-point MLMC Compression

The residual $g^l - g^{l-1}$ contains only 2 bits per element (1 bit of information + 1 bit for sign). The total communication cost is $2d + 64 + \lceil\log_2(63)\rceil$ bits per iteration per machine, achieving a **32x compression ratio** compared to the $64d$ bits of the uncompressed strategy.

**Optimal Probability Distribution** (Lemma 3.3):
$$p^l = \frac{2^{-l}}{1 - 2^{-63}}$$

Lower levels (fewer bits, higher bias) are assigned higher probabilities because their residuals are larger, necessitating higher sampling rates.

#### 3. Adaptive MLMC-Top-k Compression

Utilizing the non-uniform gradient distribution, the level probabilities are adaptively optimized for each sample:

**Adaptive Optimal Probability** (Lemma 3.4):
$$p_{t,i}^l = \frac{\|g_{t,i}^l - g_{t,i}^{l-1}\|}{\sum_{l'=1}^L \|g_{t,i}^{l'} - g_{t,i}^{l'-1}\|}$$

Intuition: Levels with larger residual norms are assigned higher probabilities.

**Relationship to Importance Sampling**: For Top-k, MLMC is equivalent to sampling the $l$-th largest element with probability $p_{t,i}^l$ and scaling it by $1/p_{t,i}^l$. However, the MLMC framework is more general—it is equally applicable to structured compressors like quantization, where importance sampling has no natural definition.

#### 4. Special Analysis for Exponentially Decaying Gradients

**Assumption 3.5**: The sorted gradient elements exhibit exponential decay, satisfying $|v(j)| = |v(0)| e^{-rj/2}$ (commonly observed in deep learning, mimicking Gaussian-like distributions).

**Lemma 3.6**: Under this assumption, the variance of adaptive MLMC s-Top-k is $O(1/(rs))$, whereas that of Rand-k is $O(d/s)$. When $1/r < d$ (meaning gradients are non-uniform), MLMC significantly outperforms Rand-k.

### Loss & Training

**Optimization Framework** (Algorithms 2 and 3):
1. Server broadcasts the model $x_t$.
2. Each machine computes the stochastic gradient $v_{t,i}$.
3. Samples compression level $l \sim p^l$ (or adaptive $p_{t,i}^l$).
4. Computes the MLMC estimate $\tilde{g}_{t,i}$ and sends it.
5. Server aggregates $\tilde{g}_t = \frac{1}{M}\sum_i \tilde{g}_{t,i}$ and updates $x_{t+1} = x_t - \eta \tilde{g}_t$.

This is fully consistent with standard data-parallel SGD, with the only difference being that gradients are replaced by MLMC estimates.

## Key Experimental Results

### Main Results: BERT Fine-tuning on GLUE SST-2 (Communication Efficiency)

| Method | k=0.01n | k=0.05n | k=0.1n | k=0.5n |
|------|---------|---------|--------|--------|
| Rand-k | Slow convergence | Slow | Moderate | Close to Full |
| Top-k | Diverged | Slow | Moderately slow | Good |
| EF21-SGDM | Slow convergence | Moderately slow | Moderate | Good |
| **Adaptive MLMC-Top-k** | **Optimal** | **Optimal** | **Optimal** | **Optimal** |
| Uncompressed SGD | — | — | — | Baseline |

Adaptive MLMC-Top-k achieves the best communication and iteration efficiency across all sparsity levels and numbers of machines ($M=4$, $M=32$).

### Ablation Study: Bit Compression on CIFAR-10 ResNet18

| Method | M=4 Final Accuracy | M=32 Final Accuracy | Communication Volume |
|------|------------|-------------|--------|
| 2-bit Fixed-point (Biased) | ~89% | ~87% | 2d bits |
| 2-bit QSGD (Unbiased) | ~88% | ~85% | 2d bits |
| **MLMC Fixed-point** | **~91%** | **~90%** | 2d bits |
| Uncompressed SGD | ~92% | ~91% | 64d bits |

Under the same $2d$ bits communication volume, MLMC fixed-point compression significantly outperforms both 2-bit biased and unbiased baselines in final accuracy.

### Key Findings

1. **Win-win in both communication and iteration efficiency**: Reaches higher accuracy with the same number of bits, and converges faster within the same number of iterations.
2. **Superior parallelization efficiency**: Convenges faster for $M=32$ compared to $M=4$ (consistent with Theorem 4.1).
3. **Adaptive probability is crucial**: For Top-k style compressors, adaptive MLMC (Algorithm 3) significantly outperforms the fixed-probability counterpart.
4. **Plug-and-play feature**: Merely requires replacing the gradients in SGD with MLMC estimates, without modifying the optimizer itself.

## Highlights & Insights

1. **Ingenious combination of MLMC and gradient compression**: The core characteristic of "bias-to-variance" perfectly resolves the contradiction between biased and unbiased compressors.
2. **Clear theoretical advantages**: Supports parallelization on up to $M = O(T)$ machines (vs. $M = O(\sqrt{T})$ for EF21), showcasing a greater advantage in large-scale parallel scenarios.
3. **General plugin framework**: Applicable to any multilevel compressor satisfying Definition 3.1, not restricted to a specific compression method.
4. **Adaptive probability exploits gradient structure**: Explicitly leverages the non-uniformity (Gaussian-like/exponential distribution) of gradient components in deep learning.
5. **Highly efficient residual communication**: For Top-k, the residual $g^l - g^{l-1}$ contains only 1 element; for s-Top-k, it contains only 1 segment.
6. **Recovers Importance Sampling as a special case**: For Top-k, it is equivalent to IS, but the overall framework is much more general.

## Limitations & Future Work

1. **Adaptive version requires sorting precomputation**: Computing the optimal probabilities in Algorithm 3 requires sorting the gradient elements, incurring extra computational overhead.
2. **Limitations of Assumption 3.5**: The exponential decay assumption might not hold during later stages of training or on certain architectures.
3. **No integration with MARINA/DASHA**: Although the paper mentions that MLMC can be combined with variance reduction methods, this is not experimentally validated.
4. **Weak fixed-probability version**: For non-Top-k methods (e.g., bit compression), additional analysis is required to determine the optimal probabilities.
5. **Server-to-worker compression**: Considers only worker-to-server communication compression, without addressing bi-directional compression.
6. **Non-convex theory only provides gradient norm convergence**: Lacks direct guarantees for loss suboptimality.

## Related Work & Insights

- **QSGD (Alistarh et al. 2017)**: Classic unbiased quantization $\rightarrow$ representative of good theory but poor empirical results.
- **Top-k + Error Feedback**: EF (Seide et al. 2014), EF21 (Richtárik et al. 2021) $\rightarrow$ biased + error-correction paradigm.
- **EF21-SGDM (Fatkhullin et al. 2023)**: Current SOTA $\rightarrow$ parallelizability of $O(\sqrt{T})$, whereas ours achieves $O(T)$.
- **MARINA (Gorbunov et al. 2021)**: Variance-reduced unbiased compression $\rightarrow$ can be combined with MLMC.
- **MLMC (Giles 2013)**: Classic review of Multilevel Monte Carlo $\rightarrow$ theoretical foundation.
- **Horváth & Richtárik 2021**: Constructing unbiased estimators from two biased compressors $\rightarrow$ approximately doubles the communication cost.
- **Insight**: The "bias-to-variance" conversion of MLMC serves as a general tool, which may play a role in other ML scenarios requiring unbiased estimation (e.g., reinforcement learning, federated learning).

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Cross-domain innovation combining MLMC and gradient compression, with an elegant idea and a powerful unified framework.
- Experimental Thoroughness: ⭐⭐⭐⭐ Sufficient experiments across two tasks (BERT/ResNet) and multiple levels of sparsity/machines, but lacks ultra-large-scale verification.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear structure, advancing step-by-step from intuition to theory and then experiments; Algorithms are well-presented.
- Value: ⭐⭐⭐⭐⭐ Resolves the fundamental contradiction between biased and unbiased compression in distributed learning, offering an easy-to-use plug-and-play framework.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] SCAN: Self-Denoising Monte Carlo Annotation for Robust Process Reward Learning](../../NeurIPS2025/model_compression/scan_self-denoising_monte_carlo_annotation_for_robust_process_reward_learning.md)
- [\[ICML 2025\] Toward Data-centric Directed Graph Learning: An Entropy-driven Approach](toward_data-centric_directed_graph_learning_an_entropy-driven_approach.md)
- [\[ACL 2025\] Mitigating Selection Bias with Node Pruning and Auxiliary Options](../../ACL2025/model_compression/selection_bias_node_pruning.md)
- [\[ICML 2025\] Eigenspectrum Analysis of Neural Networks without Aspect Ratio Bias](eigenspectrum_analysis_of_neural_networks_without_aspect_ratio_bias.md)
- [\[ACL 2025\] Beyond Text Compression: Evaluating Tokenizers Across Scales](../../ACL2025/model_compression/beyond_text_compression_tokenizers.md)

</div>

<!-- RELATED:END -->
