---
title: >-
  [Paper Note] A Law of Data Reconstruction for Random Features (and Beyond)
description: >-
  [ICLR 2026][LLM Pretraining][Data Reconstruction] This paper establishes a data reconstruction law in random feature models from information-theoretic and algebraic perspectives: when the parameter count $p \gg dn$ (wher…
tags:
  - "ICLR 2026"
  - "LLM Pretraining"
  - "Data Reconstruction"
  - "Overparameterization"
  - "Random Features"
  - "Memorization"
  - "Privacy"
date: 2026-05-08
content_hash: 7476d5ebd35a4bb5
---

# A Law of Data Reconstruction for Random Features (and Beyond)

**Conference**: ICLR 2026
**arXiv**: [2509.22214](https://arxiv.org/abs/2509.22214)  
**Code**: [https://github.com/iurada/data-reconstruction-law](https://github.com/iurada/data-reconstruction-law)  
**Area**: Machine Learning Theory / Privacy
**Keywords**: Data Reconstruction, Overparameterization, Random Features, Memorization, Privacy

## TL;DR
This paper establishes a data reconstruction law in random feature models from information-theoretic and algebraic perspectives: when the parameter count $p \gg dn$ (where $d$ is the data dimension and $n$ is the number of samples), training data can be fully reconstructed. A projection-loss-based optimization method is proposed and the universality of this threshold is validated on RF models, two-layer networks, and ResNets.

## Background & Motivation

**Background**: It is known that neural networks can interpolate (memorize labels) when $p \gg n$; classical theory equates memorization with label fitting.

**Limitations of Prior Work**: There is no theoretical characterization of reconstructing training data from model parameters (as opposed to merely fitting labels). Empirical observations suggest that larger models are easier to reconstruct from, yet no rigorous parameter-count threshold theory exists. Data extraction attacks on foundation models (e.g., GPT-4, Stable Diffusion) expose privacy risks, making it urgent to understand the conditions under which reconstruction is feasible.

**Key Challenge**: Label fitting requires $p \geq n$ degrees of freedom ($n$ equations), whereas data reconstruction requires recovering the entire $d \times n$ input matrix, intuitively demanding $p \geq dn$ degrees of freedom — yet this claim has lacked formal proof.

**Goal**: Establish a parameter-count threshold theory for data reconstruction, answering the question "how large must a model be to memorize training data (rather than merely memorizing labels)?"

**Key Insight**: Develop theory on the analytically tractable random feature (RF) model, deriving sufficient conditions for reconstruction via properties of subspaces in feature space, then empirically validate generalization to deep networks.

**Core Idea**: Data reconstruction exhibits a phase transition at $p \approx dn$ — reconstruction is impossible below this threshold and training data can be fully recovered from model parameters above it.

## Method

### Overall Architecture
The theoretical component establishes two theorems on the random feature regression model $f_{RF}(x,\theta) = \varphi(x)^\top \theta$, proving that when $p \gg dn$ the subspace spanned by the feature space uniquely determines the original training data. The experimental component proposes a projection-loss-based optimization reconstruction algorithm and validates the threshold on RF models, two-layer networks, and deep ResNets.

### Key Designs

1. **Theorem 1: Uniqueness of Reconstruction**:

    - Function: Proves that when $p \gg dn$, if the subspace spanned by the features of a candidate reconstruction $\hat{X}$ contains the original features, then each row of $\hat{X}$ must be close to some training sample.
    - Mechanism: Exploits the concentration property of the RF kernel to guarantee linear independence of the rows of $\Phi$, then analyzes the constraints of $\varphi(\hat{x}) = \sum a_i \varphi(x_i)$ via the nonlinear Hermite components. Uniform concentration over all $\hat{x}$ is achieved via an $\varepsilon$-net argument, critically relying on the condition $p \gg dn$.
    - Design Motivation: Establishes the sufficient condition "feature subspace equality $\Rightarrow$ proximity of inputs," providing a theoretical foundation for reconstruction.

2. **Theorem 2: Exclusion of Duplicates**:

    - Function: Proves that for $n=2$, reconstruction does not produce duplicates (i.e., both rows cannot be close to the same training sample).
    - Mechanism: Proof by contradiction combined with Taylor expansion. Assuming $\hat{x}_1, \hat{x}_2$ are both close to $x_1$, a contradiction is derived by projecting along the direction of the residual $\varepsilon_2 - \varepsilon_1$, with higher-order nonlinear terms handled via a generalized Stein lemma.
    - Design Motivation: Closes the gap left by Theorem 1 — ensuring the entire training set is reconstructed completely rather than partially with redundancy.

3. **Projection-Loss Reconstruction Algorithm**:

    - Function: Proposes the practical data reconstruction objective $\mathcal{L}(\hat{X}) = \|P_{\hat{\Phi}}^\perp \theta^*\|_2^2$.
    - Mechanism: Since the trained parameter $\theta^* = \Phi^+ Y \in \text{span}\{\varphi(x_i)\}$, the objective requires $\theta^*$ to lie within the subspace spanned by the reconstructed features. Gradient descent is used to optimize $\hat{X}$, with rows renormalized to the sphere after each step.
    - Design Motivation: Converts the theoretical subspace containment condition into an optimizable loss function; the method is not restricted to RF models.

### Loss & Training
The reconstruction method employs gradient descent with momentum to minimize the projection loss, without requiring training a new model. Access to the trained last-layer parameters $\theta^*$ and the random feature matrix $V$ (or equivalently the preceding layer parameters) is required.

## Key Experimental Results

### Main Results

**CIFAR-10 RF Model Reconstruction ($n=100$, $d=3072$, ReLU):**

| Parameter Count $p$ | Training Loss | Reconstruction Error $\rho$ | Status |
|---|---|---|---|
| $p = n$ | ~0 | ~1.0 | Label fitting only |
| $p = dn$ | ~0 | ~0.5 | Reconstruction begins |
| $p = 10dn$ | ~0 | **~0** | Full reconstruction |

### Ablation Study

| Configuration | Reconstruction Threshold | Notes |
|---|---|---|
| RF (spherical data) | $p \approx dn$ | Fully consistent with theory |
| Two-layer network (GD-trained) | $p^{(L)} \approx dn$ | Last-layer parameter count is decisive |
| ResNet (GD-trained) | $p^{(L)} \approx dn$ | Holds equally |
| Logistic loss (classification) | $p \approx dn$ | Not restricted to regression |
| Cross-entropy | $p^{(L)} \approx dn$ | Holds equally |

### Key Findings
- The label-fitting threshold $p = n$ and the data reconstruction threshold $p = dn$ are two distinct phase transitions — a large "gray zone" exists between them where the model memorizes labels but cannot reconstruct data.
- Sign ambiguity of ReLU: because the odd-order Hermite coefficients ($\geq 3$) of ReLU are zero, reconstruction may exhibit sign flips. This issue is resolved by using $\phi(z) = \text{ReLU}(z) + \tanh(z)$.
- The threshold $p \gg dn$ coincides with the smooth interpolation threshold identified in the adversarial robustness literature, suggesting an intrinsic connection between adversarial robustness and data reconstruction capability.
- When $\hat{n} \neq n$ (i.e., the exact sample count is unknown), setting $\hat{n} > n$ yields correct samples plus partial duplicates, while $\hat{n} < n$ causes samples to be merged.

## Highlights & Insights
- **Elegant Phase Transition**: $p = n$ is the label memorization threshold and $p = dn$ is the data memorization threshold; their ratio is exactly the data dimensionality $d$, with clear physical intuition — reconstruction requires recovering $dn$ degrees of freedom.
- **Theory-Driven Algorithm Design**: The projection loss is not designed ad hoc but naturally derived from the theoretical analysis — if $\theta^*$ lies within the feature subspace, reconstruction succeeds. This "theory $\to$ algorithm" pathway is instructive.
- **Cross-Architecture Universality**: Although the theory is established only for RF models, experiments demonstrate that the threshold applies to two-layer networks, ResNets, and ViTs alike, suggesting that overparameterization of the last layer is the key factor.

## Limitations & Future Work
- Theorem 2 only covers the case $n=2$; proving the exclusion of duplicates for $n \geq 3$ faces combinatorial explosion as a technical obstacle.
- The theory requires the activation function to satisfy specific Hermite coefficient conditions (which ReLU does not fully satisfy); in practice, ReLU still works but with sign ambiguity.
- It has not been proven that the global optimum of the projection loss necessarily corresponds to a permutation of the training data — guarantees at the non-convex optimization level are currently absent.
- The question of whether it is information-theoretically possible to reconstruct individual samples in the intermediate regime $n \ll p \ll dn$ is discussed preliminarily but left unresolved.

## Related Work & Insights
- **vs. Haim et al. 2022**: Haim reconstructs boundary samples via maximum-margin KKT conditions; the proposed method reconstructs all samples via subspace projection, without being limited to classification boundaries.
- **vs. Loo et al. 2022**: Loo derives reconstruction in the infinite-width network ($p \to \infty$) setting; this paper explicitly identifies the finite threshold $p \gg dn$.
- **vs. Adversarial Robustness**: The threshold $p \gg dn$ is simultaneously a necessary and sufficient condition for smooth interpolation, suggesting a deep connection between "capability for smooth memorization" and "susceptibility to reconstruction."

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First paper to provide a parameter-count phase transition threshold for data reconstruction, with clear and substantive theoretical contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Coverage spans from synthetic data to CIFAR-10/Tiny-ImageNet, and from RF models to ResNet/ViT.
- Writing Quality: ⭐⭐⭐⭐⭐ Theory and experiments are tightly integrated; Figure 1 is highly persuasive and proof sketches are clearly presented.
- Value: ⭐⭐⭐⭐⭐ Significant implications for privacy and security — provides a quantitative characterization of "how large a model must be before it becomes dangerous."

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Optimal Online Change Detection via Random Fourier Features](../../NeurIPS2025/llm_pretraining/optimal_online_change_detection_via_random_fourier_features.md)
- [\[ICLR 2026\] Understanding the Emergence of Seemingly Useless Features in Next-Token Predictors](understanding_the_emergence_of_seemingly_useless_features_in_next-token_predicto.md)
- [\[ICLR 2026\] Token-level Data Selection for Safe LLM Fine-tuning](token-level_data_selection_for_safe_llm_fine-tuning.md)
- [\[ICLR 2026\] Predicting Training Re-evaluation Curves Enables Effective Data Curriculums](predicting_training_re-evaluation_curves_enables_effective_data_curriculums_for_.md)
- [\[ICLR 2026\] FictionalQA: A Dataset for Studying Memorization and Knowledge Acquisition](fictionalqa_a_dataset_for_studying_memorization_and_knowledge_acquisition.md)

</div>

<!-- RELATED:END -->
