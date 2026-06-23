---
title: >-
  [Paper Note] On the Limits of Sparse Autoencoders: A Theoretical Framework and Reweighted Remedy
description: >-
  [ICLR 2026][Interpretability][Paper Note] This paper derives the first closed-form optimal solution for Sparse Autoencoders (SAEs), theoretically proving that SAEs generally fail to fully recover true monosemantic features from superimposed polysemantic features (resulting in feature shrinking and vanishing), except under conditions of extreme sparsity. For ge
tags:
  - ICLR 2026
  - Interpretability
date: 2026-05-08
content_hash: 6590a3d3a928b864
---
# On the Limits of Sparse Autoencoders: A Theoretical Framework and Reweighted Remedy

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=DSOTgzeH3w](https://openreview.net/forum?id=DSOTgzeH3w)  
**Code**: To be confirmed  
**Area**: Interpretability / Mechanistic Interpretability  
**Keywords**: Sparse Autoencoders, Superposition Hypothesis, Feature Identifiability, Monosemanticity, Reweighting

## TL;DR
This paper derives the first closed-form optimal solution for Sparse Autoencoders (SAEs), theoretically proving that SAEs generally fail to fully recover true monosemantic features from superimposed polysemantic features (resulting in feature shrinking and vanishing), except under conditions of extreme sparsity. For general sparsity levels, the authors propose WSAE, which adaptively reweights dimensions based on their degree of polysemanticity, and provide a weighting principle verified to improve monosemanticity and interpretability in both synthetic data and real-world language/vision models.

## Background & Motivation
**Background**: Large model interpretability is hindered by "black boxes," with a core challenge being **polysemanticity**—where a single neuron is activated by multiple semantically unrelated concepts. The mainstream explanation is the **superposition hypothesis**: polysemantic dimensions are approximate linear combinations of several monosemantic concepts, allowing models to represent more "features" than dimensions. To decompose superimposed polysemantic features into interpretable monosemantic ones, Sparse Autoencoders (SAEs) have become a popular tool in mechanistic interpretability, widely used on LLMs and VLMs with activation functions ranging from ReLU and JumpReLU to Top-k and BatchTopK.

**Limitations of Prior Work**: Previous works focused almost exclusively on modifying the **architecture** (gated SAE, k-sparse, KL targets) or **evaluation** of SAEs. However, none addressed the fundamental question: **Under what conditions can an SAE uniquely recover true monosemantic features from polysemantic inputs?** Consequently, the identifiability of SAEs has lacked theoretical support. Empirical observations show that SAEs "sometimes work," but why they work and when they fail remains unclear.

**Key Challenge**: SAE training optimizes the reconstruction loss of **polysemantic features $x_p$**, while the actual goal is to recover **true monosemantic features $x$**. Since $x$ is unknown, the loss is calculated against $x_p$. This introduces a risk: the superposition matrix $W_p$ may allow the reconstructed $\tilde{x}_p$ to align well with the true $x_p$, even if the recovered $x_m$ differs significantly from the true $x$. Low reconstruction loss does not equal correct feature recovery.

**Goal**: (1) Provide a theoretical framework and closed-form solution for SAEs, characterizing the conditions for recovery; (2) Propose a remedy to improve true feature recovery under general sparsity, along with an operational weighting principle.

**Key Insight**: Following the superposition hypothesis, polysemantic features are modeled as a linear transformation of true monosemantic features $x_p = W_p x$. The SAE is modeled as a single-hidden-layer encoder-decoder (ignoring bias), and the **closed-form optimal solution** for the reconstruction loss is derived. With this solution, recovery quality can be analyzed analytically rather than relying on empirical observations.

**Core Idea**: Prove via the closed-form solution that "SAEs generally cannot recover true features except under extreme sparsity," then use a reweighting strategy—downweighting polysemantic dimensions while preserving weights for monosemantic ones—to align the training loss more closely with the true feature reconstruction loss.

## Method

### Overall Architecture
The paper follows a **theoretical derivation chain**: Modeling $\to$ Closed-form solution $\to$ Revealing failures $\to$ Identifying conditions $\to$ Designing remedy $\to$ Weighting principle.

Mathematically, true monosemantic features $x \in \mathbb{R}^n$ have components $x_i$ that take the value 0 with probability $S$ and positive values with probability $1-S$ ($S$ is the sparsity factor). Polysemantic features $x_p = W_p x \in \mathbb{R}^n_p$, where $n > n_p$ (dimensionality reduction causes superposition), and columns of the superposition matrix exhibit **negative interference**, i.e., $W_{p,[:,i]}^\top W_{p,[:,j]} \le 0\ (i\neq j)$, forming digon/polygon structures geometrically. The SAE encodes via $x_m = \sigma(W_m x_p)$ and decodes via $\tilde{x}_p = W_m^\top x_m$, with the objective to reconstruct polysemantic features:

$$L_{\text{SAE}}(W_m; x_p) = \mathbb{E}_x \| W_p x - W_m^\top \sigma(W_m W_p x) \|^2.$$

"Full recovery" is defined as $x_m \sim x$ (where $\sim$ denotes equality up to permutation and zero-padding). The theoretical chain explores the conditions under which the optimal $W_m^*$ leads to $x_m \sim x$.

### Key Designs

**1. SAE Closed-form Solution: Identifying $W_p^\top$ as the Optimal Solution**

This is the pivot of the paper. Pain point: Without a closed-form solution, one can only guess when SAEs work. Under the assumption that columns of the superposition matrix form digons/polygons, the authors prove (Theorem 1): When $n_m \ge n$, $W_m^* = I^*(W_p, 0)^\top$ is one of the optimal solutions. That is, the transposed superposition matrix $W_p^\top$ (with zero-padding and row permutation) corresponds to the SAE optimal weights, and recovered features are $x_m = \sigma(W_p^\top x_p)$. This explicit form transforms the analysis of recovery quality into a calculable algebraic problem.

**2. Feature Shrinking and Vanishing: Why SAEs Generally Fail**

Using the closed-form solution, the authors disprove the illusion that SAEs always recover features. **Feature shrinking** occurs when recovered feature values from polysemantic dimensions are suppressed; a dimension that is more polysemantic (interfering with more true features) experiences more severe shrinking. In Example 1 with $x=(0.5,1.0,0.8)^\top$ and $W_p=\bigl(\begin{smallmatrix}1&0&0\\0&1&-1\end{smallmatrix}\bigr)$, recovery yields $(0.5,0.2,0)^\top$—the original top-1 feature (1.0) is suppressed, leading to **mislabeled features**. **Feature vanishing** is the extreme case: in Example 2, certain dimensions go to zero, leaving $x_m$ with fewer effective dimensions than $x_p$. These phenomena explain the systematic bias of SAEs—they naturally favor monosemantic dimensions while ignoring polysemantic ones.

**3. Unique Identifiability Under Extreme Sparsity: A Theoretical Explanation**

If recovery fails generally, why does it work in practice? The authors prove (Theorems 2, 3) that the key is **sparsity**. As $S \to 1$ and columns satisfy a weak **non-positive interference** condition, $W_m^* = I^*(W_p, 0)^\top$ remains optimal, and $I^* \sigma(W_m^* x_p) = x$ holds for any $x$. When $n_m = n$, this solution is **unique**. Intuitively, as $S \to 1$, $x$ is almost certainly 1-sparse, preventing shrinking or vanishing. This attributes empirical SAE success to the sufficient sparsity of true features in certain scenarios.

**4. Reweighted Remedy (WSAE): Aligning Training Loss with True Reconstruction**

Since ground-truth sparsity is **not controllable via training**, the authors modify the loss function. They quantify the gap between the "SAE loss" and the "true reconstruction loss" $L_{\text{GT}}(W_m;x)=\mathbb{E}_x\|x-\sigma(W_m W_p x)\|^2$. When $W_m=W_p^\top$ (Theorem 4):

$$L_{\text{SAE}} - L_{\text{GT}} = [x-\sigma(W_p^\top W_p x)]^\top (W_p^\top W_p - I_{n\times n})[x-\sigma(W_p^\top W_p x)].$$

This gap depends on the recovery error and $W_p^\top W_p - I$. Since $W_p$ is fixed by the input, the gap cannot be minimized directly.

The proposed remedy assigns a weight $\gamma_i > 0$ to each polysemantic dimension, defining the reweighted loss ($\Gamma=\mathrm{diag}(\gamma_1,\dots,\gamma_{n_p})$):

$$L_{\text{WSAE}}(W_m;x_p)=\mathbb{E}_{x_p}\|\Gamma[x_p - W_m^\top \mathrm{ReLU}(W_m x_p)]\|_2^2.$$

The gap now depends on $W_p^\top \Gamma^\top \Gamma W_p - I_{n\times n}$ (Theorem 5). By adjusting $\Gamma$, the gap can be mitigated. For monosemantic dimensions, $\gamma_i \approx 1$ is used; for polysemantic dimensions, **smaller weights** are assigned to reduce negative interference terms. Mechanism: **High weights for monosemantic dimensions, low weights for polysemantic ones**. Practically, the authors use **per-dimension variance** $s_i$ as a proxy for monosemanticity, setting $\gamma_i = s_i^\alpha$ ($\alpha=1$ in experiments).

### Loss & Training
The standard reconstruction loss $L_{\text{SAE}}$ is replaced by $L_{\text{WSAE}}$ with diagonal weights $\gamma_i = s_i^\alpha$. The monosemanticity proxy $s_i$ is calculated per dimension (variance for synthetic/LLMs, semantic consistency for VLM). Higher $\alpha$ biases reconstruction toward monosemantic dimensions. Other settings (ReLU/Top-k, hidden dimensionality) remain consistent with standard SAEs.

## Key Experimental Results

### Main Results
Synthetic data follows the toy model by Elhage et al. ($n=200, n_p=20, n_m=200$). Real-world data uses Pythia-160M (Top-k, $k=32$, 32× expansion) and ResNet-18 (ImageNet-100, 16384 dimensions, $k=16$). LLM monosemanticity is evaluated via auto-interpretability scores (Llama3.1-8B summarization + activation prediction).

| Setup | Metric | Original SAE | Weighted SAE | Gain |
|------|------|------|------|------|
| Pythia-160M (12-layer avg, $\alpha=1$) | auto-interp score (%) | ~76.4 | ~80.2 | **+3.8** |
| Pythia-160M Layer 3, $\alpha=1$ | auto-interp score (%) | 77.8 | 84.6 | **+6.8** |
| Pythia-160M Layer 8, $\alpha=1$ | auto-interp score (%) | 74.6 | 81.5 | **+6.9** |
| ResNet-18 (NCL, $\alpha=1$) | Semantic Consistency (%) | 40.2 | 42.2 | **+2.0** |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| $\alpha=0$ (Original SAE) | Baseline | WSAE reduces to SAE; the gap term is unadjustable. |
| $\alpha=0.5$ | Positive gain in most layers | Weak reweighting shows stable improvements. |
| $\alpha=1$ | +3.8% avg (12 layers) | Stronger reweighting leads to the largest and most consistent gain. |
| WSAE vs SAE (Synthetic, low $S$) | Lower $L_{\text{GT}}$ | Advantage is more pronounced when evaluated on non-sparse dimensions. |
| WSAE vs SAE (Synthetic, recon $L$) | Comparable | WSAE does not deviate from the sparsity-reconstruction Pareto frontier. |

### Key Findings
- **Sparsity is the Switch**: On synthetic data, the number of true features explaining an SAE latent decreases as $S$ increases (Fig 2), aligning with the theory that full recovery occurs only at extreme sparsity.
- **Reweighting Benefits Low Sparsity**: When $S$ is low (general sparsity), WSAE achieves significantly lower $L_{\text{GT}}$ than SAE and higher per-dimension variance, without sacrificing reconstruction of $x_p$.
- **Stronger Baseline, Greater Gain**: On LLMs, SAEs that already performed well showed more significant improvements after reweighting, consistently across layers.

## Highlights & Insights
- **First Closed-form Solution for SAEs**: Transforms the recovery problem from an empirical guess into a provable algebraic framework, identifying $W_p^\top$ as the optimal solution.
- **Concrete Failure Mechanisms**: Uses simple 3D examples to illustrate "feature shrinking/vanishing," highlighting a critical risk—SAEs can systematically misidentify top activation dimensions, potentially **misleading interpretability conclusions**.
- **Diagnosis-driven Remedy**: WSAE is derived from the $L_{\text{SAE}}-L_{\text{GT}}$ gap. It identifies exactly how to weight dimensions (downweighting polysemantic ones to suppress interference) using low-cost proxies like variance.

## Limitations & Future Work
- The theory relies on the superposition hypothesis, linear transformations ($x_p = W_p x$), and non-positive interference, which may not strictly hold in real LLMs.
- The closed-form solution (Theorem 1) assumes digon/polygon geometry. The uniqueness (Theorem 3) requires $n_m = n$, which may not apply to typical overcomplete SAEs ($n_m \gg n$).
- Identifiability is an asymptotic conclusion ($S \to 1$). In practice, sparsity is finite, meaning WSAE "improves" rather than "guarantees" recovery.
- Proxies (variance/consistency) are approximations; optimal $\alpha$ might vary across models and layers.
- Experiments are scaled to Pythia-160M and ResNet-18; validation on larger LLMs is required.

## Related Work & Insights
- **vs. Superposition Hypothesis (Elhage et al.)**: While Elhage explains "why" polysemanticity occurs, this work explores "if" SAEs can reverse it, providing closed-form solutions and failure conditions.
- **vs. Structural Improvements (k-sparse, gated SAE, KL targets)**: Unlike works that modify activations or architectures to mitigate $l_1$ shrinkage, this work modifies the **loss weights** based on theoretical gaps, making it orthogonal to structural improvements.
- **vs. SAE Evaluation (Minegishi et al.)**: This work provides a theoretical basis for why certain models remain competitive and uses standard suites to confirm improved monosemanticity.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First closed-form solution and identifiability framework for SAEs.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covered synthetic, LLM, and VLM, though real-world models were small-scale.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear progression from theorems to intuitive examples.
- Value: ⭐⭐⭐⭐⭐ Warns of systematic biases in SAEs while providing a low-cost, effective remedy.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Toward Faithful Retrieval-Augmented Generation with Sparse Autoencoders](toward_faithful_retrieval-augmented_generation_with_sparse_autoencoders.md)
- [\[ICLR 2026\] AbsTopK: Rethinking Sparse Autoencoders For Bidirectional Features](abstopk_rethinking_sparse_autoencoders_for_bidirectional_features.md)
- [\[ICLR 2026\] Sparse Autoencoders Trained on the Same Data Learn Different Features](sparse_autoencoders_trained_on_the_same_data_learn_different_features.md)
- [\[ICLR 2026\] Uncovering Conceptual Blindspots in Generative Image Models Using Sparse Autoencoders](uncovering_conceptual_blindspots_in_generative_image_models_using_sparse_autoenc.md)
- [\[ICLR 2026\] Learning Multimodal Dictionary Decompositions with Group-Sparse Autoencoders](learning_multimodal_dictionary_decompositions_with_group-sparse_autoencoders.md)

</div>

<!-- RELATED:END -->
