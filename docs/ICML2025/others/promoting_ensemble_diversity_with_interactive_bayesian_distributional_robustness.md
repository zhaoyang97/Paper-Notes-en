---
title: >-
  [Paper Note] IBDR: Promoting Ensemble Diversity with Interactive Bayesian Distributional Robustness
description: >-
  [ICML 2025][Bayesian Inference] This paper proposes the IBDR Bayesian inference framework. By introducing interactive loss and Wasserstein distributional robustness optimization over the product distribution space, the framework constructs a particle ensemble that balances diversity and low sharpness. Utilizing ViT-B/16, it achieves a 73.6% average accuracy on VTAB-1K, outperforming all baselines.
tags:
  - "ICML 2025"
  - "Bayesian Inference"
  - "Particle Diversity"
  - "Distributional Robustness"
  - "SAM"
  - "DPP"
  - "LoRA"
date: 2026-05-08
content_hash: 94aef407bb4e32ac
---

# IBDR: Promoting Ensemble Diversity with Interactive Bayesian Distributional Robustness

**Conference**: ICML 2025  
**arXiv**: [2506.07247](https://arxiv.org/abs/2506.07247)  
**Area**: Bayesian Inference / Ensemble Learning / Model Fine-tuning  
**Keywords**: Bayesian Inference, Particle Diversity, Distributional Robustness, SAM, DPP, LoRA

## TL;DR
This paper proposes the IBDR Bayesian inference framework. By introducing interactive loss and Wasserstein distributional robustness optimization over the product distribution space, the framework constructs a particle ensemble that balances diversity and low sharpness. Utilizing ViT-B/16, it achieves a 73.6% average accuracy on VTAB-1K, outperforming all baselines.

## Background & Motivation
**Background**: Bayesian inference provides uncertainty estimation through multi-model particle sampling. Existing methods include Markov Chain Monte Carlo (MCMC) methods such as SGLD, SGHMC, and SVGD, alongside variational inference (VI) approaches, but these suffer from computational and storage overheads on large-scale models.

**Limitations of Prior Work**: (1) In traditional Bayesian inference, particles are sampled independently from the posterior ($\theta_{1:K} \sim^{iid} Q$), lacking an explicit interaction mechanism; (2) Independent sampling easily leads to particle collapse into the same mode, resulting in insufficient ensemble diversity; (3) Although methods like SA-BNN introduce sharpness awareness, they still operate on independent distribution spaces without modeling inter-particle relationships.

**Key Challenge**: How to explicitly promote diversity among particles while maintaining low loss and low sharpness for each individual model particle?

**Key Insight**: Define a joint loss containing interactive terms on the product distribution space $Q^K = Q \odot Q \odot \cdots \odot Q$, and derive a tractable upper bound using Wasserstein Distributionally Robust Optimization (DRO) theory.

## Method

### Overall Architecture
The IBDR framework consists of three core components: (1) defining a joint loss comprising classification loss and diversity loss, expressed as $\ell(\boldsymbol{\theta};x,y) = \frac{1}{K}\sum_i l(\theta_i;x,y) + \alpha l_{div}(\theta_{1:K};x,y)$; (2) deriving the upper bound of the population loss via Theorem 4.1, which connects distributional robustness with SAM; (3) alternately optimizing the particle means $\mu_{1:K}$ and the dual variable $\lambda$.

### Key Designs
1. **Interactive Joint Loss Design**:
    - **Function**: Defines a loss function that includes interaction terms among particles on the product distribution space.
    - **Mechanism**: The total loss is formulated as $\ell(\boldsymbol{\theta};x,y) = \frac{1}{K}\sum_{i=1}^K l(\theta_i;x,y) + \alpha \cdot l_{div}(\theta_{1:K};x,y)$, where $l_{div}$ is defined based on Determinantal Point Processes (DPP) as the determinant of the Gram matrix of normalized non-target predictions: $l_{div} = \det([\tilde{f}_{-y}^i]^T [\tilde{f}_{-y}^i])$, which maximizes the volume spanned by the non-target class prediction vectors.
    - **Design Motivation**: Particles agree on the correct class (achieving low cross-entropy loss) but diversify in their "ways of making mistakes" (yielding a high DPP volume), thereby facilitating complementarity.

2. **Distributional Robustness Generalization Bound (Theorem 4.1)**:
    - **Function**: Derives the upper bound of the population loss with interactive loss on the product distribution space.
    - **Mechanism**: $$\mathcal{L}_\mathcal{D}(Q^K) \leq L\sqrt{\frac{K \cdot D_{KL}(Q,P) + \log(1/\delta)}{2N}} + \min_{\lambda \geq 0}\{\lambda\rho + \mathbb{E}_{\boldsymbol{\theta}\sim Q^K}[\max_{\boldsymbol{\theta}'}{\mathcal{L}_S(\boldsymbol{\theta}') - \lambda c^K(\boldsymbol{\theta},\boldsymbol{\theta}')}]\}$$ This upper bound unifies Sharpness-Aware Minimization (SAM) and distributional robustness as special cases (Corollary 4.2) and extends them to the product distribution space with diversity interaction terms.
    - **Design Motivation**: Directly minimizing the population loss is intractable (since $\mathcal{D}$ is inaccessible), requiring a computable upper bound. This theoretically guarantees concurrently achieving low loss, low sharpness, and high diversity.

3. **Bilevel Dual Optimization Algorithm**:
    - **Function**: Designs an implementable optimization pipeline based on the theoretical upper bound.
    - **Mechanism**: Given the posterior $Q = \frac{1}{K}\sum_i \mathcal{N}(\mu_i, \sigma^2 I)$ with a fixed $\sigma=0.1$, the algorithm alternates between: (1) performing a step of gradient ascent to obtain the adversarial perturbation $\theta_i'$ after reparameterizing $\theta_i = \mu_i + \sigma\epsilon_i$; (2) computing the loss gradient containing the diversity term at $\theta_i'$ to update $\mu_i$; and (3) updating $\lambda \geq 0$ via projected gradient descent.
    - **Design Motivation**: Similar to the two-step optimization in SAM but extended to multi-particle interaction scenarios, where $\lambda$ dynamically balances the robustness radius and the loss.

### Loss & Training
LoRA is applied to fine-tune ViT-B/16 and LLaMA-2 with each particle initializing its LoRA module independently. Hyperparameters: $K$ particles (default 3-5), diversity weight $\alpha$, regularization weight $\beta$, and dual learning rate $\alpha_\lambda$. During inference, predictions from each particle are aggregated to form the final ensemble.

## Key Experimental Results

### Main Results (VTAB-1K, ViT-B/16)

| Method | Natural(7) | Specialized(4) | Structured(8) | Average |
|------|-----------|----------------|---------------|------|
| LoRA | 79.2 | 84.3 | 60.2 | 68.4 |
| SAM | 79.5 | 83.2 | 53.4 | 70.5 |
| SA-BNN | 80.1 | 85.6 | 49.1 | 68.2 |
| SGLD | 78.4 | 83.6 | 57.3 | 68.4 |
| SVGD | 79.8 | 84.6 | 57.5 | 70.9 |
| BayesTune | 79.5 | 84.9 | 57.2 | 68.5 |
| **IBDR** | **81.5** | **86.0** | **60.3** | **73.6** |

### ECE Calibration Experiments

| Method | Natural | Specialized | Structured | Average ECE↓ |
|------|---------|-------------|------------|---------|
| LoRA | 0.19 | 0.11 | 0.21 | 0.17 |
| SAM | 0.17 | 0.14 | 0.17 | 0.16 |
| **IBDR** | **0.11** | **0.10** | **0.14** | **0.12** |

### Key Findings
- IBDR achieves the best performance in 13 out of 19 VTAB subtasks, with an average accuracy of 73.6% outperforming the second-best SVGD which scores 70.9% (+2.7%).
- The improvement is most significant in the Structured category (60.3% vs. the second-best SGLD at 57.3%), indicating that diversity is more critical for complex structured tasks.
- ECE calibration decreases from 0.17 with LoRA to 0.12, indicating that IBDR improves uncertainty estimation.
- The framework also demonstrates performance gains on LLaMA-2 commonsense reasoning tasks.

## Highlights & Insights
- Strong theoretical contributions: Theorem 4.1 unifies SAM, DRO, and particle diversity as different special cases under a single framework.
- Intuitive design of the DPP diversity loss: "agree on correct answers, but diversify in their ways of making mistakes."
- Clever zero-initialization design of $w_d(0)$: ensures equivalence to standard LoRA during single-particle degradation.
- Close alignment between theory and practice: Algorithm 1 is derived directly from Corollary 4.3.

## Limitations & Future Work
- Storage overhead scales linearly with the number of particles $K$, making $K > 5$ impractical for large-scale models.
- $\sigma = 0.1$ is fixed instead of being learned, which may limit the flexibility of the posterior.
- Computing the diversity loss requires simultaneous forward passes of all particles, resulting in a training time approximately $K$ times that of a single model.
- Validated only in LoRA fine-tuning scenarios, with other PEFT approaches like Adapter/Prompt left unexplored.

## Related Work & Insights
- **vs SA-BNN (NeurIPS23)**: SAM + Bayesian but with independent sampling. IBDR incorporates particle interactions on top of this, improving the average accuracy on VTAB-1K from 68.2% to 73.6% (+5.4%).
- **vs SVGD**: Implicitly pushes particles apart via a kernel function. IBDR explicitly models diversity in the prediction space via DPP, improving accuracy from 70.9% to 73.6%.
- **vs DeepEnsemble**: Simple independent training for ensembling. IBDR explicitly promotes diversity via the interactive loss, improving accuracy from 67.0% to 73.6%.
- **Insight**: Particle interaction in Bayesian inference is an overlooked dimension; the combination of DPP diversity and DRO robustness exhibits substantial potential for generalization.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The theoretical framework unifies SAM and DRO, and the concept of interactive Bayesian inference is highly novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers all 19 tasks of VTAB-1K, ECE calibration, and LLaMA commonsense reasoning.
- **Writing Quality**: ⭐⭐⭐⭐ Theoretical derivations are clear, though the notation is somewhat heavy.
- **Value**: ⭐⭐⭐⭐ Provides both theoretical and practical contributions to the field of Bayesian fine-tuning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Improving Generalization with Flat Hilbert Bayesian Inference](improving_generalization_with_flat_hilbert_bayesian_inference.md)
- [\[ICML 2025\] Probably Approximately Global Robustness Certification](probably_approximately_global_robustness_certification.md)
- [\[ICML 2025\] Diversity By Design: Leveraging Distribution Matching for Offline Model-Based Optimization](diversity_by_design_leveraging_distribution_matching_for_offline_model-based_opt.md)
- [\[ICML 2025\] Randomized Dimensionality Reduction for Euclidean Maximization and Diversity Measures](randomized_dimensionality_reduction_for_euclidean_maximization_and_diversity_mea.md)
- [\[ACL 2025\] Evaluating the Evaluation of Diversity in Commonsense Generation](../../ACL2025/others/evaluating_the_evaluation_of_diversity_in_commonsense_generation.md)

</div>

<!-- RELATED:END -->
