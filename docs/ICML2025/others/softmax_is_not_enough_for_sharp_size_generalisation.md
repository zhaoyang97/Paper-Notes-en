---
title: >-
  [Paper Note] Softmax is not Enough (for Sharp Size Generalisation)
description: >-
  [ICML2025][softmax] This work theoretically proves that softmax attention **inevitably undergoes coefficient dispersion** as the input scale increases, failing to maintain sharp focus on a small number of key elements, and proposes adaptive temperature as a mitigation method.
tags:
  - "ICML2025"
  - "softmax"
  - "attention dispersion"
  - "size generalization"
  - "adaptive temperature"
  - "sharp function"
  - "Transformer theory"
date: 2026-05-08
content_hash: c237b2c71d186e9e
---

# Softmax is not Enough (for Sharp Size Generalisation)

**Conference**: ICML2025  
**arXiv**: [2410.01104](https://arxiv.org/abs/2410.01104)  
**Code**: No official open-source code (the paper provides JAX implementation snippets)  
**Area**: Transformer Generalization / Attention Mechanism Theory  
**Keywords**: softmax, attention dispersion, size generalization, adaptive temperature, sharp function, Transformer theory

## TL;DR

This work theoretically proves that softmax attention **inevitably undergoes coefficient dispersion** as the input scale increases, failing to maintain sharp focus on a small number of key elements, and proposes adaptive temperature as a mitigation method.

## Background & Motivation

Softmax attention in Transformers is widely believed to learn "circuits" that consistently perform sharp computations on diverse inputs—for example, concentrating all attention on a single token when searching for the maximum value. Mechanistic interpretability research has identified various sharp attention patterns, such as induction heads, comparator heads, and retrieval heads.

However, a key question remains: **Can these sharp behaviors generalize to larger-scale inputs?** Experiments show that even if LLMs can correctly find the maximum within the distribution, the accuracy drops rapidly as the input sequence grows longer. The core motivation of this paper is to **explain and prove** mathematically that this phenomenon is inevitable.

**Definition of Sharp Function**: A function is called sharp if its output depends only on a constant number of inputs (e.g., $\max$ depends on only 1). In contrast, $\text{average}$ depends on all $n$ inputs and is not sharp.

## Method

### Core Theory: Softmax Inevitably Disperses

**Lemma 2.1 (Softmax Coefficient Dispersion)**: Let $n$ logits $e_k^{(n)}$ satisfy the bounded condition $m \le e_k^{(n)} \le M$, and temperature $\theta > 0$, then as $n \to \infty$:

$$\text{softmax}_\theta(\mathbf{e}^{(n)})_k = \Theta\left(\frac{1}{n}\right)$$

**Proof Sketch**: Utilizing the upper and lower bounds of the logits yields a two-sided bound for the attention coefficients:

$$\frac{1}{n} \exp\left(-\frac{\delta}{\theta}\right) \le \alpha_k^{(n)} \le \frac{1}{n} \exp\left(\frac{\delta}{\theta}\right)$$

where $\delta = M - m$ is the spread of the logits. Since $\delta$ and $\theta$ are constants, the attention coefficients decay to zero at a rate of $\Theta(1/n)$.

**Theorem 2.2 (Softmax Inevitably Disperses in Transformers)**: For a Transformer using a finite vocabulary ($|\mathcal{X}| < |\mathbb{N}|$) and composed of MLP + softmax self-attention layers, whether BERT or GPT-style, the coefficients of all global attention heads **inevitably converge to a uniform distribution** when the number of input tokens is sufficiently large.

Key insight of the proof: Finite vocabulary $\to$ compact input space $\to$ continuous functions (MLPs) map compact sets to compact sets $\to$ convex combinations (attention) preserve compactness $\to$ bounded logits $\to$ directly apply Lemma 2.1.

**Proposition 3.1 (Sharpness Requires Large Weights)**: The logit spread satisfies:

$$\delta \le 2\sigma_{\max}^{(Q)} \sigma_{\max}^{(K)} \|\mathbf{y}\| \max_i \|\mathbf{x}_i\|$$

That is, for attention heads to remain sharp, they must **increase the maximum singular value of the $\mathbf{Q}, \mathbf{K}$ matrices**, which, however, leads to overfitting and error amplification.

### Adaptive Temperature

Directly setting the temperature $\theta = 0$ (hard attention) makes training difficult to converge. The authors propose an **inference-time adaptive temperature** scheme:

1. First compute the original softmax probability $p = \text{softmax}(\mathbf{e})$
2. Compute the Shannon entropy $H = -\sum_i p_i \log p_i$
3. Use a degree-4 polynomial to fit the relationship of $\beta = 1/\theta$ with respect to entropy $H$: $\beta \approx -0.037H^4 + 0.481H^3 - 2.3H^2 + 4.917H - 1.791$
4. Apply the correction only when $H > 0.5$ and $\beta > 1$ (without increasing the entropy)
5. Return $\text{softmax}(\mathbf{e} \cdot \beta)$

This method is compatible with Flash Attention—the entropy correction computation can be fully streamed, maintaining $O(n)$ memory.

## Key Experimental Results

### Max Retrieval

The model is trained on $\le 16$ elements and tested on different scales. The adaptive temperature is adjusted **purely during inference**, without altering model parameters:

| Input Scale | 16 (ID) | 64 | 256 | 1024 | 4096 | 16384 |
|---------|---------|-----|------|------|------|-------|
| Baseline | **98.6%** | 94.3% | 81.3% | 53.8% | 22.6% | 12.4% |
| Adaptive \theta | **98.6%** | **94.5%** | **82.1%** | **57.7%** | **24.9%** | **14.0%** |
| p-value | 0.4 | 0.002 | 2e-4 | 1e-4 | 0.02 | 4e-3 |

Statistically significant improvements are observed across all OOD scales (paired t-test), and the improvement becomes more pronounced as the scale increases (+3.9% @ 1024).

### CLRS-Text Algorithmic Reasoning Benchmark

In the Gemma 2B fine-tuning experiments, softmax in all attention heads is replaced with adaptive temperature softmax (applied during both training and inference). On the **vast majority** of the 30 algorithmic tasks, the adaptive temperature version performs better, especially on larger, OOD problem scales.

A few exceptions (Heapsort, MST Kruskal, Bubble Sort) may be due to their context windows being far beyond the training range of the polynomial fitting.

### Logit Spread $\delta$ in Practical Transformers

| Model | $\delta$ Range | $\delta$ Mean |
|------|--------|--------|
| Gemma 2B | [2.28, 14.78] | 5.69 \pm 2.05 |
| Gemma 7B | [0.09, 32.74] | 5.82 \pm 2.61 |

$\delta$ is far smaller than the theoretical limit of floating-point numbers, indicating that the dispersion effect occurs faster in practical models than in the theoretical worst case.

## Highlights & Insights

- **Solid Theoretical Contribution**: This is the first work to rigorously prove that softmax attention dispersion is inevitable in Transformers, explaining the fundamental cause of length generalization failure.
- **Elegant Analysis**: The proof chain from compactness $\to$ boundedness $\to$ dispersion is concise and powerful.
- **Practical Value**: Adaptive temperature serves as a drop-in replacement, which can be applied during inference at zero extra cost.
- **Broad Perspective**: The end of the paper systematically discusses multiple architectural directions to bypass the dispersion theorem (including unnormalized attention, sigmoid attention, selective attention, hard/local attention, discontinuous MLPs, etc.).
- **Flash Attention Compatibility**: Adaptive temperature can be streamed, without disrupting efficient attention implementations.

## Limitations & Future Work

1. **Adaptive Temperature is an Ad-hoc Method**: The authors explicitly admit that it does not escape the theoretical guarantees, but merely delays the dispersion.
2. **Questionable Generalization of Polynomial Fitting**: The degree-4 polynomial is fitted to specific tasks, requiring re-adaptation or introduction during training when migrating to different scenarios (such as CLRS-Text).
3. **CLRS-Text Demands Introduction During Training**: Unlike max retrieval which can be used purely during inference, complex tasks require re-fine-tuning.
4. **Influence of Positional Encodings (e.g., RoPE/ALiBi) Untouched**: The positional encodings of modern LLMs might interact with the dispersion effect.
5. **Lack of Validation on Real LLM Inference Tasks** (e.g., mathematical reasoning, code generation).
6. **Only Covers Softmax Attention**: This is not applicable to architectures that do not use softmax, such as linear attention and state space models (though such architectures themselves might not suffer from this issue).

## Related Work & Insights

- **Mechanistic Interpretability**: Olsson et al. 2022 (induction heads), Wang et al. 2022 (IOI), Wu et al. 2024 (retrieval heads)—this work fundamentally questions the OOD robustness of these circuits.
- **Alternative Attention Mechanisms**: Sigmoid Attention (Ramapuram et al. 2024), Selective Attention (Leviathan et al. 2025), Differential Transformer (Ye et al. 2025)—which may naturally avoid dispersion.
- **Length Generalization**: Anil et al. 2022, Chiang & Cholak 2022—this work provides a mathematical explanation for length generalization failures.
- **CLRS-Text**: Markeeva et al. 2024—algorithmic reasoning benchmark.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First to rigorously prove the softmax dispersion theorem, with a unique perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Max retrieval is clean, and CLRS-Text coverage is broad, but it lacks validation in real-world LLM scenarios.
- Writing Quality: ⭐⭐⭐⭐⭐ — The theoretical derivation is clear, the writing is elegant, and the flow from motivation to conclusion is seamless.
- Value: ⭐⭐⭐⭐⭐ — Offers profound insights for Transformer architectural design and length generalization research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Residual Matrix Transformers: Scaling the Size of the Residual Stream](residual_matrix_transformers_scaling_the_size_of_the_residual_stream.md)
- [\[NeurIPS 2025\] Out-of-distribution Generalisation is Hard: Evidence from ARC-like Tasks](../../NeurIPS2025/others/out-of-distribution_generalisation_is_hard_evidence_from_arc-like_tasks.md)
- [\[ACL 2025\] Do not Abstain! Identify and Solve the Uncertainty](../../ACL2025/others/do_not_abstain_identify_and_solve_the_uncertainty.md)
- [\[NeurIPS 2025\] Coreset for Robust Geometric Median: Eliminating Size Dependency on Outliers](../../NeurIPS2025/others/coreset_for_robust_geometric_median_eliminating_size_dependency_on_outliers.md)
- [\[ACL 2025\] All That Glitters is Not Novel: Plagiarism in AI Generated Research](../../ACL2025/others/plagiarism_ai_generated_research.md)

</div>

<!-- RELATED:END -->
