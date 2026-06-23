---
title: >-
  [Paper Note] Rectified LpJEPA: Joint-Embedding Predictive Architectures with Sparse and Maximum-Entropy Representations
description: >-
  [ICML 2026][Others][JEPA] The authors generalize the "post-projection alignment to isotropic Gaussian" in LeJEPA to "post-projection alignment to a Rectified Generalized Gaussian (RGG) distribution." By utilizing rectified and truncated generalized Gaussians, they achieve explicitly controllable expected $\ell_0$ sparsity. On ImageNet-100, a Re
tags:
  - ICML 2026
  - Others
  - JEPA
  - Rectified Generalized Gaussian
date: 2026-05-08
content_hash: 425ad8ae581c2b14
---
# Rectified LpJEPA: Joint-Embedding Predictive Architectures with Sparse and Maximum-Entropy Representations

**Conference**: ICML 2026  
**arXiv**: [2602.01456](https://arxiv.org/abs/2602.01456)  
**Code**: https://github.com (Author's homepage link; repository not directly provided)  
**Area**: Self-Supervised Learning / JEPA / Sparse Representations  
**Keywords**: JEPA, Sparse Representations, Maximum Entropy Distributions, Rectified Generalized Gaussian, Sliced Wasserstein  

## TL;DR
The authors generalize the "post-projection alignment to isotropic Gaussian" in LeJEPA to "post-projection alignment to a Rectified Generalized Gaussian (RGG) distribution." By utilizing rectified and truncated generalized Gaussians, they achieve explicitly controllable expected $\ell_0$ sparsity. On ImageNet-100, a ResNet encoder achieves a $85.08\%$ linear probe accuracy while maintaining $\ell_0$ sparsity at $\sim 73\%$, significantly outperforming the fully dense representations of LeJEPA.

## Background & Motivation

**Background**: The JEPA series (I-JEPA, LeJEPA, etc.) learn self-supervised representations by enforcing multi-view consistency in the latent space, avoiding reconstruction in the pixel space. LeJEPA (Balestriero & LeCun 2025) builds on this by using SIGReg regularization to align the marginal of each 1D random projection to a univariate Gaussian. It relies on the Cramér–Wold theorem to approximately "stretch" the entire representation distribution into an isotropic Gaussian to prevent collapse.

**Limitations of Prior Work**: Pulling representations into an isotropic Gaussian naturally leads to dense (uniformly active across all dimensions) representations. This discards a key prior that appears repeatedly in neuroscience, signal processing, and deep learning—sparsity and non-negativity. The $\ell_0$ sparsity of LeJEPA on ImageNet-100 is constantly $1.0$ (fully dense), which contradicts the "efficient coding" hypothesis found in sparse coding, ReLU, and NMF.

**Key Challenge**: To achieve sparsity, a Dirac mass must be inserted into the representation distribution. However, once the target distribution contains a Dirac mass, it is no longer a stable distribution (not closed under linear combinations). This immediately invalidates the analytical reasoning of SIGReg, which assumes the distribution remains in the same family after projection. How can a target distribution be both "controllably sparse" and "maximum entropy" within the Cramér–Wold slicing framework?

**Goal**: (i) Construct a new distribution family where both expected $\ell_p$ and expected $\ell_0$ can be analytically controlled; (ii) Design a corresponding sliced regularization term to bypass the "non-closure under projection" problem; (iii) Verify that the resulting representations are controllably sparse while maintaining downstream accuracy.

**Key Insight**: Starting from the maximum entropy principle—given a support $S$ and constraint $\mathbb{E}[\|\mathbf{x}\|_p^p]$, the maximum entropy distribution is the truncated generalized Gaussian $\mathcal{TGN}_p$. By mixing $\mathcal{TGN}_p$ with a Dirac $\delta_0$ at the origin, one obtains the Rectified Generalized Gaussian (RGG), whose expected $\ell_0$ is analytically given by $(\mu, \sigma, p)$.

**Core Idea**: Replace the "post-projection Gaussian alignment" in LeJEPA with a "two-sample sliced Wasserstein alignment to RGG." Explicitly apply ReLU rectification to the features so that the target distribution and the model output share the same $[0, \infty)$ support. This simultaneously achieves non-negativity, controllable sparsity, maximum entropy, and consistency.

## Method

### Overall Architecture
Input a pair of augmented views $(\mathbf{x}, \mathbf{x}')$ into a backbone $f_{\boldsymbol{\theta}}$ to obtain logits $\mathbf{z}_{\text{raw}}, \mathbf{z}'_{\text{raw}} \in \mathbb{R}^D$, then apply ReLU to get $\mathbf{z} = \mathrm{ReLU}(\mathbf{z}_{\text{raw}})$ and $\mathbf{z}' = \mathrm{ReLU}(\mathbf{z}'_{\text{raw}})$. Simultaneously, sample $\mathbf{y}$ from the target distribution $\prod_{i=1}^D \mathcal{RGN}_p(\mu, \sigma)$ and sample $N$ projection directions $\mathbf{c}_i$ uniformly from the unit $\ell_2$ sphere $\mathbb{S}^{D-1}_{\ell_2}$. The loss consists of two parts: view consistency $\|\mathbf{z}-\mathbf{z}'\|_2^2$ and sliced distribution matching $\sum_i \mathcal{L}(\mathbb{P}_{\mathbf{c}_i^\top \mathbf{z}} \,\|\, \mathbb{P}_{\mathbf{c}_i^\top \mathbf{y}})$, where $\mathcal{L}$ takes the form of the sorted difference for 1D sliced 2-Wasserstein. The overall pipeline follows the "backbone + projector + post-projection alignment" structure of LeJEPA, but the target distribution is changed from Gaussian to RGG, and feature rectification is enforced.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["A pair of augmented views (x, x′)"] --> B["backbone f_θ<br/>leads to logits z_raw, z′_raw ∈ ℝ^D"]
    B --> C["Feature Rectification ReLU<br/>z, z′ ∈ [0,∞)^D"]
    C --> D["View Consistency<br/>‖z − z′‖₂²"]
    E["RGG Target Distribution Sampling<br/>y ~ ∏ RGN_p(μ,σ), support [0,∞)"] --> G
    F["Unit ℓ₂ Sphere Sampling<br/>N projection directions c_i"] --> G
    C --> G["RDMReg Two-Sample Sliced Alignment<br/>Sorted 2-Wasserstein after c_i projection"]
    D --> H["Total Loss = Consistency + RDMReg"]
    G --> H
```

### Key Designs

**1. Rectified Generalized Gaussian (RGG) Target Distribution: Turning "Sparsity Intensity" into an Analytically Tunable Knob**

LeJEPA pulls representations toward an isotropic Gaussian, naturally leading to fully dense activations and discarding the "sparse + non-negative" prior. This paper requires a target distribution that is controllably sparse without losing information. The RGG is constructed by mixing a Dirac $\delta_0$ with a truncated generalized Gaussian $\mathcal{TGN}_p(\mu,\sigma,(0,\infty))$ at zero, equivalent to sampling from $\mathcal{GN}_p(\mu,\sigma)$ followed by ReLU. Its advantage is that the expected $\ell_0$ has a closed form:

$$\mathbb{E}[\|\mathbf{x}\|_0] = D \cdot \Phi_{\mathcal{GN}_p(0,1)}(\mu/\sigma),$$

Thus, a negative $\mu$ directly corresponds to high sparsity ($\mu=-3$ compresses the activation rate to $\sim 1\%$). The continuous part inherits the property of "maximum entropy under expected $\ell_p$ norm constraints" (Prop 3.3): $p=2$ reduces to Rectified Gaussian, $p=1$ to Rectified Laplace, and $0<p<1$ yields sharper sparse priors. Achieving both sparsity and information retention requires both a point mass at 0 (for hard zeros) and maximum entropy in the continuous part (to preserve task information)—RGG is the simplest structure that analytically combines these through a mixture distribution, where all knobs can be written as closed forms of known special functions $\Phi$, $\Gamma$, and $P(\cdot,\cdot)$.

**2. Two-Sample Sliced Distribution Matching (RDMReg): Bypassing the Fatal "RGG Non-Closure" Problem**

SIGReg can directly write the NLL of a univariate Gaussian density because Gaussians are closed under linear combinations; they remain Gaussian after projection. However, once the target distribution contains a Dirac mass, it is no longer closed, and the 1D marginal of $\mathbf{c}^\top \mathbf{y}$ does not belong to a closed-form family, causing analytical reasoning to fail. RDMReg breaks this by abandoning closure in favor of a two-sample approach: actual samples $\mathbf{Y}\in\mathbb{R}^{B\times D}$ are drawn from the target RGG, and for each projection direction $\mathbf{c}_i$, alignment is performed using the squared sorted 1D 2-Wasserstein distance:

$$\mathcal{L}(\cdot)=\tfrac{1}{B}\big\|(\mathbf{Z}\mathbf{c}_i)^\uparrow-(\mathbf{Y}\mathbf{c}_i)^\uparrow\big\|_2^2.$$

Theoretically, exact distribution matching requires infinite projections, but experiments show that a small $N$ independent of dimensionality is sufficient. Reducing alignment to 1D and using non-parametric 2-Wasserstein for sample alignment is the only compromise that is both compatible with arbitrary target distributions and resistant to the curse of dimensionality—this is the price paid for replacing projection closure with sparsity.

**3. Pairing Feature Rectification with Target Rectification: Ensuring Strict Consistency Between Model Output and Target Support**

The key to establishing the sparsity-accuracy trade-off is ensuring alignment occurs on the same support. The authors explicitly add a ReLU at the end of the backbone so that $\mathbf{z}\in[0,\infty)^D$, matching the $[0,\infty)$ support of the RGG. Figure 3(a) tests all four combinations—$(\mathcal{RGN}_p \mid \mathbf{z}^+)$ (Ours, both rectified), $(\mathcal{GN}_p \mid \mathbf{z})$ (Baseline, neither rectified), $(\mathcal{GN}_p \mid \mathbf{z}^+)$, and $(\mathcal{RGN}_p \mid \mathbf{z})$. Results show that only the "both rectified" setting achieves both high accuracy and high sparsity. In other cases, the model either collapses to full density or suffers a major accuracy drop. The reason is: if alignment occurs on different supports, the sliced Wasserstein can never reach 0, forcing the model to choose between compromising accuracy or abandoning sparsity. Only after support alignment can the continuous mapping theorem allow $\mathbf{z}_{\text{raw}}$ to converge to $\mathcal{GN}_p$, automatically making $\mathrm{ReLU}(\mathbf{z}_{\text{raw}})$ converge to RGG.

### Loss & Training
The complete loss is:
$$\min_{\boldsymbol{\theta}} \mathbb{E}\big[\|\mathbf{z}-\mathbf{z}'\|_2^2\big] + \mathbb{E}_{\mathbf{c}}\big[\mathcal{L}(\mathbb{P}_{\mathbf{c}^\top \mathbf{z}} \,\|\, \mathbb{P}_{\mathbf{c}^\top \mathbf{y}}) + \mathcal{L}(\mathbb{P}_{\mathbf{c}^\top \mathbf{z}'} \,\|\, \mathbb{P}_{\mathbf{c}^\top \mathbf{y}})\big]$$
where $\mathcal{L}$ is the sliced 2-Wasserstein. Besides uniform sampling of projection directions, the authors provide a variant in the appendix using "eigenvectors of the covariance of $\mathbf{Z}$ as projections," which accelerates the removal of second-order dependencies (conditionally equivalent to VICReg). The backbone uses standard configurations like ResNet/ViT/ConvNeXt + MLP projector. Linear probes are conducted on both encoder output $f_{\boldsymbol{\theta}_1}(\mathbf{x})$ and projector output $\mathbf{z}$.

## Key Experimental Results

### Main Results
ImageNet-100 linear probe (top-1 acc% / higher is better, $\ell_0$ sparsity / lower is better).

| Method | Encoder Acc1 | Projector Acc1 | $\ell_1$ Sparsity | $\ell_0$ Sparsity |
|------|-------------:|---------------:|--------------:|--------------:|
| Rectified LpJEPA $\mathcal{RGN}_{2.0}(0, \sigma_{\text{GN}})$ | **85.08** | 80.00 | 0.341 | 0.730 |
| Rectified LpJEPA $\mathcal{RGN}_{2.0}(1.0, \sigma_{\text{GN}})$ | **85.08** | 80.54 | 0.628 | 0.867 |
| Rectified LpJEPA $\mathcal{RGN}_{1.0}(0.25, \sigma_{\text{GN}})$ | 84.98 | **80.76** | 0.375 | 0.744 |
| Rectified LpJEPA $\mathcal{RGN}_{1.0}(-3.0, \sigma_{\text{GN}})$ | 82.72 | 71.88 | **0.006** | **0.010** |
| LeJEPA (Baseline, Dense) | 84.80 | 79.52 | 0.637 | 1.000 |
| VICReg | 84.18 | 78.88 | 0.795 | 1.000 |
| SimCLR | 83.44 | 77.90 | 0.634 | 1.000 |
| NCL-ReLU (Sparse Baseline) | 82.58 | 76.88 | 0.004 | 0.009 |
| NVICReg-ReLU (Sparse Baseline) | 84.48 | 77.74 | 0.521 | 0.712 |

While achieving accuracy comparable to or higher than LeJEPA, it reduces $\ell_0$ sparsity from $1.000$ to $0.730$ (i.e., $\sim 27\%$ of dimensions are permanently zero). Compared to NVICReg-ReLU, it yields $\sim 0.6\%$ higher accuracy at more sparse levels.

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| $(\mathcal{RGN}_p \mid \mathbf{z}^+)$ (Ours) | Best Accuracy-Sparsity Trade-off | Both features and target are rectified; the only setting Balancing both. |
| $(\mathcal{GN}_p \mid \mathbf{z})$ (No Rectification) | High Accuracy but Dense | $\ell_0$ sparsity is constant at 0; reduces to LeJEPA. |
| $(\mathcal{GN}_p \mid \mathbf{z}^+)$ | Sharp Accuracy Drop | Feature rectified but target not; support mismatch. |
| $(\mathcal{RGN}_p \mid \mathbf{z})$ | Sharp Accuracy Drop | Target rectified but feature not; alignment failure. |
| $\mu$ sweeping from $1.0$ to $-3.0$ | Empirical $\ell_0$ highly consistent with Prop 3.5 | Analytical formula $\mathbb{E}[\|\mathbf{x}\|_0] = D \cdot \Phi(\mu/\sigma)$ holds in practice. |
| Pareto Frontier (Sparsity vs Acc) | Acc only drops when $> 95\%$ dimensions are zero | Large margin for utilizing sparsity. |
| nHSIC Independence | RGG series significantly lower than VICReg/NVICReg | RDMReg suppresses high-order dependencies, not just covariance. |

### Key Findings
- Empirical $\ell_0$ almost overlaps with the theoretical formula $D \cdot \Phi_{\mathcal{GN}_p(0,1)}(\mu/\sigma)$ across multiple backbones, indicating RGG doesn't just "look sparse" but actually forces the model to sparsify according to analytical knobs.
- Representation sparsification is very "cheap": accuracy barely decreases until $\ell_0$ sparsity reaches $\sim 95\%$; cliff-like drops only occur at more aggressive levels.
- Compared to VICReg / NVICReg, which only penalize second-order statistics, Rectified LpJEPA shows significantly lower nHSIC, suggesting sliced Wasserstein alignment captures dependencies beyond the second order.
- Across different downstream datasets, Rectified LpJEPA exhibits "data-adaptive" sparsity (Fig 3(c)), and the sparsity statistics themselves can serve as OOD (Out-of-Distribution) indicators.

## Highlights & Insights
- By strictly extending LeJEPA's "projection closure + univariate Gaussian density" to "abandoning closure + two-sample sliced Wasserstein," this work provides an interface-level abstraction for JEPA regularizer design. Any "prior-shaped" representation (heavy-tailed, non-negative, hierarchical, etc.) can now use the same framework.
- "Sparsity" is transformed from something "indirectly obtained via ReLU / $\ell_1$" into something "analytically determined by distribution hyperparameters $(\mu, \sigma, p)$." This is particularly useful for embodied models requiring energy/bandwidth budgets, enabling activation rate estimations without retraining.
- Achieving "sparsity + maximum entropy + independence" simultaneously is difficult. The authors provide a formal argument by rewriting entropy as $d$-dimensional Rényi entropy $\mathbb{H}_d$ (bypassing the undefined differential entropy at Dirac points). This entropy notation serves as a useful toolkit for future analysis of discrete-continuous hybrid representations with hard zeros.

## Limitations & Future Work
- Evaluation is primarily on ImageNet-100; ImageNet-1K results are only in the appendix. More evidence is needed to confirm if the sparsity-accuracy trade-off holds at larger scales.
- Training introduces sliced Wasserstein, incurring $O(N \cdot B \log B)$ sorting overhead per step. The authors admit efficiency costs for extremely large batches or high-dimensional projectors.
- Choosing the target $\sigma$ ($\sigma_{\text{GN}}$ vs $\sigma_{\text{RGN}}$) requires binary search, increasing the hyperparameter search space; whether the default recipes generalize across datasets remains to be verified.
- The paper only discusses image classification; it does not touch dense prediction (detection/segmentation) where "semantic meaningfulness of non-zero positions" is critical. The true value of sparse representations has not been fully stress-tested.

## Related Work & Insights
- **vs LeJEPA (Balestriero & LeCun 2025)**: LeJEPA is a degenerate case of RGG when $\mu \to +\infty$ (no Dirac mass, no feature rectification). This work strictly generalizes it and recovers both sparsity and accuracy near $\mu = 0$.
- **vs VICReg / NVICReg**: VICReg only performs second-order matching (covariance/variance/invariance) and cannot eliminate high-order dependencies. This work proves that sliced 2-Wasserstein, even using only covariance eigenvectors as projections, strictly implies the second-order part of VICReg while achieving significantly lower nHSIC.
- **vs NCL-ReLU and Pure Sparse Baselines**: Pure sparse contrastive methods lag by $\sim 2\%$ in accuracy. Rectified LpJEPA aligns accuracy with LeJEPA by treating sparsity as a "distribution prior" rather than a "hard regularizer."
- **vs Sparse Coding / NMF Tradition**: Upgrading "non-negativity + sparsity" from an engineering bias to a target distribution shape for JEPA regularizers provides a new path for classical sparse coding in "end-to-end deep self-supervision."

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Extending the projection distribution family from Gaussian to RGG with supporting two-sample sliced Wasserstein is a clear generalization of JEPA collapse-prevention design.
- Experimental Thoroughness: ⭐⭐⭐⭐ ImageNet-100 + multiple backbones + multiple sparsity levels + nHSIC/entropy/OOD adaptation. ImageNet-1K being only in the appendix is the only regret.
- Writing Quality: ⭐⭐⭐⭐ Clear three-layer narrative of theory, intuition, and experiment. Formulas and figures are well-coordinated.
- Value: ⭐⭐⭐⭐ Provides an analytically tunable engineering template for self-supervised representations that require both expressivity and sparsity, directly reusable for embodied or low-power scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Possibilistic Predictive Uncertainty for Deep Learning](possibilistic_predictive_uncertainty_for_deep_learning.md)
- [\[ICML 2026\] On Revisiting Entropy for Identifying Mislabeled Images](on_revisiting_entropy_for_identifying_mislabeled_images.md)
- [\[ICML 2026\] Continual Learning of Domain-Invariant Representations](continual_learning_of_domain-invariant_representations.md)
- [\[ICML 2026\] nD-RoPE: A Generalized RoPE for n-Dimensional Position Embedding](nd-rope_a_generalized_rope_for_n-dimensional_position_embedding.md)
- [\[ICML 2025\] Optimal Auction Design in the Joint Advertising](../../ICML2025/others/optimal_auction_design_in_the_joint_advertising.md)

</div>

<!-- RELATED:END -->
