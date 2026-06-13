---
title: >-
  [Paper Note] Rectified LpJEPA: Joint-Embedding Predictive Architectures with Sparse and Maximum-Entropy Representations
description: >-
  [ICML 2026][JEPA] The authors generalize the "post-projection isotropic Gaussian alignment" in LeJEPA to "post-projection Rectified Generalized Gaussian (RGG) distribution alignment." By employing rectification and trunc…
tags:
  - "ICML 2026"
  - "JEPA"
  - "Sparse Representation"
  - "Maximum-Entropy Distribution"
  - "Rectified Generalized Gaussian"
  - "Sliced Wasserstein"
date: 2026-05-08
content_hash: 0468447af92dd29a
---

# Rectified LpJEPA: Joint-Embedding Predictive Architectures with Sparse and Maximum-Entropy Representations

**Conference**: ICML 2026  
**arXiv**: [2602.01456](https://arxiv.org/abs/2602.01456)  
**Code**: https://github.com (Link to author's homepage; repository not directly provided)  
**Area**: Self-Supervised Learning / JEPA / Sparse Representations  
**Keywords**: JEPA, Sparse Representation, Maximum-Entropy Distribution, Rectified Generalized Gaussian, Sliced Wasserstein  

## TL;DR
The authors generalize the "post-projection isotropic Gaussian alignment" in LeJEPA to "post-projection Rectified Generalized Gaussian (RGG) distribution alignment." By employing rectification and truncated generalized Gaussians to obtain explicitly controllable expected $\ell_0$ sparsity, the method achieves $85.08\%$ linear probe accuracy with a ResNet encoder on ImageNet-100 while maintaining $\ell_0$ sparsity at $\sim 73\%$, significantly outperforming the fully dense representations of LeJEPA.

## Background & Motivation

**Background**: The JEPA series (I-JEPA, LeJEPA, etc.) learns self-supervised representations by enforcing multi-view consistency in the latent space, avoiding reconstruction in pixel space. LeJEPA (Balestriero & LeCun 2025) builds on this by using SIGReg regularization to align the marginal of each 1D random projection to a univariate Gaussian. It relies on the Cramér–Wold theorem to approximately "stretch" the entire representation distribution into an isotropic Gaussian, thereby preventing collapse.

**Limitations of Prior Work**: Forcing representations toward isotropic Gaussians naturally leads to dense representations (where all dimensions are uniformly active). This discards key priors—sparsity and non-negativity—that frequently appear in neuroscience, signal processing, and deep learning. LeJEPA’s $\ell_0$ sparsity on ImageNet-100 is constantly $1.0$ (fully dense), which contradicts the "efficient coding" hypothesis found in sparse coding, ReLU, and NMF.

**Key Challenge**: Achieving sparsity requires embedding an $\ell_0$ constraint or a Dirac mass into the representation distribution. However, once the target distribution contains a Dirac mass, it is no longer a stable distribution (it is not closed under linear combinations). This causes the analytical reasoning of SIGReg—which assumes "projected distributions remain in the same family"—to fail immediately. The problem is how to maintain the Cramér–Wold slicing framework while making the target distribution both "sparsity-controllable" and "maximum-entropy."

**Goal**: (i) Construct a new distribution family where both expected $\ell_p$ and expected $\ell_0$ are analytically controllable; (ii) Design a corresponding sliced regularization term to bypass the "projection non-closure" issue; (iii) Verify that the resulting representations are controllably sparse while maintaining downstream accuracy.

**Key Insight**: Starting from the Maximum Entropy Principle—given support $S$ and the constraint $\mathbb{E}[\|\mathbf{x}\|_p^p]$—the maximum entropy distribution is the truncated generalized Gaussian $\mathcal{TGN}_p$. By mixing $\mathcal{TGN}_p$ with a Dirac $\delta_0$ at zero, one obtains the Rectified Generalized Gaussian (RGG), whose expected $\ell_0$ is analytically given by $(\mu, \sigma, p)$.

**Core Idea**: Replace the "post-projection Gaussian alignment" in LeJEPA with "two-sample sliced Wasserstein alignment to RGG" and explicitly add ReLU rectification to the features. This ensures that the target distribution and model output share the same $[0, \infty)$ support, simultaneously achieving non-negativity, controllable sparsity, maximum entropy, and consistency.

## Method

### Overall Architecture
For a pair of augmented views $(\mathbf{x}, \mathbf{x}')$, the backbone $f_{\boldsymbol{\theta}}$ produces raw logits $\mathbf{z}_{\text{raw}}, \mathbf{z}'_{\text{raw}} \in \mathbb{R}^D$. ReLU is applied to obtain $\mathbf{z} = \mathrm{ReLU}(\mathbf{z}_{\text{raw}})$ and $\mathbf{z}' = \mathrm{ReLU}(\mathbf{z}'_{\text{raw}})$. Simultaneously, $\mathbf{y}$ is sampled from the target distribution $\prod_{i=1}^D \mathcal{RGN}_p(\mu, \sigma)$, and $N$ projection directions $\mathbf{c}_i$ are sampled uniformly from the unit $\ell_2$ sphere $\mathbb{S}^{D-1}_{\ell_2}$. The loss consists of two parts: view consistency $\|\mathbf{z}-\mathbf{z}'\|_2^2$ and sliced distribution matching $\sum_i \mathcal{L}(\mathbb{P}_{\mathbf{c}_i^\top \mathbf{z}} \,\|\, \mathbb{P}_{\mathbf{c}_i^\top \mathbf{y}})$, where $\mathcal{L}$ takes the form of the sorted difference for 1D sliced 2-Wasserstein distance. The workflow follows the "backbone + projector + post-projection alignment" structure of LeJEPA, but replaces the Gaussian target with RGG and forces feature rectification.

### Key Designs

1.  **Rectified Generalized Gaussian (RGG) Target Distribution**:
    *   **Function**: Provides a distribution family supported on $[0,\infty)$ with maximum entropy under an expected $\ell_p$ norm and analytically controllable expected $\ell_0$, turning "sparsity intensity" into continuously adjustable hyperparameters $(\mu, \sigma, p)$.
    *   **Mechanism**: RGG is defined as a mixture of Dirac $\delta_0$ and truncated generalized Gaussian $\mathcal{TGN}_p(\mu,\sigma,(0,\infty))$ at 0, equivalent to sampling from $\mathcal{GN}_p(\mu, \sigma)$ followed by ReLU. Its expected $\ell_0$ satisfies $\mathbb{E}[\|\mathbf{x}\|_0] = D \cdot \Phi_{\mathcal{GN}_p(0,1)}(\mu/\sigma)$. Thus, a negative $\mu$ directly corresponds to high sparsity (e.g., $\mu = -3$ suppresses the activation rate to $\sim 1\%$). The continuous portion inherits the maximum entropy property under expected $\ell_p$ constraints (Prop 3.3). $p=2$ degenerates to Rectified Gaussian, $p=1$ to Rectified Laplace, and $0<p<1$ provides sharper sparse priors.
    *   **Design Motivation**: To achieve both "controllable sparsity" and "no information loss," it is necessary to have (a) a point mass at 0 (producing hard zeros) and (b) maximum entropy in the continuous part (preserving task information). RGG is the simplest structure that analytically combines these two using a mixture distribution, where all control knobs can be written as closed forms of known special functions $\Phi$, $\Gamma$, and $P(\cdot,\cdot)$.

2.  **Two-Sample Sliced Distribution Matching (RDMReg)**:
    *   **Function**: Matches high-dimensional RGG distributions under the Cramér–Wold projection framework, bypassing the fatal issue of "RGG projection non-closure."
    *   **Mechanism**: Since RGG is not closed under linear combinations, the 1D marginal of $\mathbf{c}^\top \mathbf{y}$ does not have a closed-form family member, preventing the use of "analytical Gaussian density NLL" as in SIGReg. RDMReg uses a two-sample approach: sample $\mathbf{Y} \in \mathbb{R}^{B \times D}$ from the target RGG, and for each projection $\mathbf{c}_i$, use $\mathcal{L}(\cdot) = \tfrac{1}{B}\|(\mathbf{Z}\mathbf{c}_i)^\uparrow - (\mathbf{Y}\mathbf{c}_i)^\uparrow\|_2^2$ (the squared 1D 2-Wasserstein distance after sorting). Theoretically, an infinite number of projections is required for strict equivalence to full distribution matching, but experiments show a small $N$ (independent of dimension) suffices.
    *   **Design Motivation**: The "projection closure" of the Gaussian family is why SIGReg can directly write an NLL; this property must be abandoned for sparsity. Reducing the alignment problem to 1D and using non-parametric 2-Wasserstein for sample alignment is the only compromise that remains compatible with arbitrary target distributions while resisting the curse of dimensionality.

3.  **Feature Rectification + Target Rectification Alignment Pair**:
    *   **Function**: Ensures that the model output space is strictly consistent with the target distribution support, a key constraint for the sparsity-accuracy trade-off.
    *   **Mechanism**: The authors explicitly add $\mathrm{ReLU}$ to the end of the backbone so that $\mathbf{z} \in [0,\infty)^D$, matching the RGG support. The ablation in Figure 3(a) tests four combinations: $(\mathcal{RGN}_p \mid \mathbf{z}^+)$ (Ours), $(\mathcal{GN}_p \mid \mathbf{z})$ (Baseline), $(\mathcal{GN}_p \mid \mathbf{z}^+)$, and $(\mathcal{RGN}_p \mid \mathbf{z})$. Only the "rectify both sides" setting achieves both high accuracy and high sparsity; other combinations either collapse to full density or suffer significant accuracy drops.
    *   **Design Motivation**: If alignment occurs on different supports (e.g., unrectified features aligned to RGG), the sliced Wasserstein distance will never reach 0. The model would be forced to "compromise accuracy" or "abandon sparsity." Only through support alignment and the "continuous mapping theorem" path can $\mathbf{z}_{\text{raw}}$ converge to $\mathcal{GN}_p$ while automatically making $\mathrm{ReLU}(\mathbf{z}_{\text{raw}})$ converge to RGG.

### Loss & Training
The complete loss is:
$\min_{\boldsymbol{\theta}} \mathbb{E}\big[\|\mathbf{z}-\mathbf{z}'\|_2^2\big] + \mathbb{E}_{\mathbf{c}}\big[\mathcal{L}(\mathbb{P}_{\mathbf{c}^\top \mathbf{z}} \,\|\, \mathbb{P}_{\mathbf{c}^\top \mathbf{y}}) + \mathcal{L}(\mathbb{P}_{\mathbf{c}^\top \mathbf{z}'} \,\|\, \mathbb{P}_{\mathbf{c}^\top \mathbf{y}})\big]$,
where $\mathcal{L}$ is the sliced 2-Wasserstein distance. In addition to uniform sampling for projection directions, the authors provide a variant in the appendix using the "eigenvectors of the covariance of $\mathbf{Z}$" as projections, which accelerates the removal of second-order dependencies (conditionally equivalent to VICReg). The backbone uses standard ResNet/ViT/ConvNeXt + MLP projector configurations. Linear probing is performed on both the encoder output $f_{\boldsymbol{\theta}_1}(\mathbf{x})$ and the projector output $\mathbf{z}$.

## Key Experimental Results

### Main Results
ImageNet-100 linear probe (top-1 acc% / higher is better; lower $\ell_0$ sparsity is better).

| Method | Encoder Acc1 | Projector Acc1 | $\ell_1$ Sparsity | $\ell_0$ Sparsity |
| :--- | :---: | :---: | :---: | :---: |
| Rectified LpJEPA $\mathcal{RGN}_{2.0}(0, \sigma_{\text{GN}})$ | **85.08** | 80.00 | 0.341 | 0.730 |
| Rectified LpJEPA $\mathcal{RGN}_{2.0}(1.0, \sigma_{\text{GN}})$ | **85.08** | 80.54 | 0.628 | 0.867 |
| Rectified LpJEPA $\mathcal{RGN}_{1.0}(0.25, \sigma_{\text{GN}})$ | 84.98 | **80.76** | 0.375 | 0.744 |
| Rectified LpJEPA $\mathcal{RGN}_{1.0}(-3.0, \sigma_{\text{GN}})$ | 82.72 | 71.88 | **0.006** | **0.010** |
| LeJEPA (Baseline, Dense) | 84.80 | 79.52 | 0.637 | 1.000 |
| VICReg | 84.18 | 78.88 | 0.795 | 1.000 |
| SimCLR | 83.44 | 77.90 | 0.634 | 1.000 |
| NCL-ReLU (Sparse Baseline) | 82.58 | 76.88 | 0.004 | 0.009 |
| NVICReg-ReLU (Sparse Baseline) | 84.48 | 77.74 | 0.521 | 0.712 |

While maintaining or exceeding the accuracy of LeJEPA, Ours reduces $\ell_0$ sparsity from $1.000$ to $0.730$ (i.e., $\sim 27\%$ of dimensions are permanently zero). Compared to NVICReg-ReLU, it achieves $\sim 0.6\%$ higher accuracy at more sparse levels.

### Ablation Study

| Configuration | Key Metric | Description |
| :--- | :--- | :--- |
| $(\mathcal{RGN}_p \mid \mathbf{z}^+)$ (Ours) | Best accuracy-sparsity trade-off | Both features and target are rectified; only setting to balance both. |
| $(\mathcal{GN}_p \mid \mathbf{z})$ (No rectification) | High accuracy but dense | $\ell_0$ sparsity is constant at 0; degenerates to LeJEPA. |
| $(\mathcal{GN}_p \mid \mathbf{z}^+)$ | Accuracy drop | Features rectified but target is not -> support mismatch. |
| $(\mathcal{RGN}_p \mid \mathbf{z})$ | Accuracy drop | Target rectified but features are not; alignment fails. |
| $\mu$ sweep from $1.0$ to $-3.0$ | Empirical $\ell_0$ matches Prop 3.5 | Analytical formula $\mathbb{E}[\|\mathbf{x}\|_0] = D \cdot \Phi(\mu/\sigma)$ holds true. |
| Pareto Front (Sparsity vs. Acc) | Acc only drops when $> 95\%$ dim are zero | Substantial room for exploitable sparsity. |
| nHSIC Independence | RGG series significantly lower than VICReg | RDMReg suppresses high-order dependencies, not just covariance. |

### Key Findings
*   Empirical $\ell_0$ almost coincides with the theoretical analytical formula $D \cdot \Phi_{\mathcal{GN}_p(0,1)}(\mu/\sigma)$ across multiple backbones, indicating that RGG doesn't just "look sparse"—it forces the model to sparsify according to the analytical knob.
*   Sparsifying representations is "cheap": accuracy barely decreases until $\ell_0$ sparsity reaches $\sim 95\%$; a cliff-like drop only occurs at more aggressive levels.
*   Compared to VICReg / NVICReg which only penalize second-order statistics, Rectified LpJEPA achieves significantly lower nHSIC, proving that sliced Wasserstein alignment captures dependencies beyond the second order.
*   Rectified LpJEPA exhibits "data-adaptive" sparsity across different downstream datasets (Fig 3(c)), suggesting sparse statistics themselves can be used as OOD signals.

## Highlights & Insights
*   The idea of "projection closure + univariate Gaussian density" from LeJEPA is strictly extended by "abandoning closure + two-sample sliced Wasserstein" to arbitrary target distributions. This serves as an interface-level abstraction for JEPA regularization design—any "shape prior" (heavy-tailed, non-negative, hierarchical...) can be implemented via this same framework.
*   "Sparsity" is transformed from something "obtained indirectly via ReLU/$\ell_1$" into something "analytically precise through distribution hyperparameters $(\mu, \sigma, p)$." This is especially useful for embodied models requiring energy/bandwidth budgets, as activation rates can be estimated analytically without retraining.
*   Achieving "sparsity + maximum entropy + mutual independence" simultaneously is difficult. The authors provide a formal argument by rewriting entropy as $d$-dimensional Rényi entropy $\mathbb{H}_d$ (bypassing the undefined differential entropy at Dirac points). This entropy notation is a useful toolbox for analyzing discrete-continuous mixed representations with "hard zeros."

## Limitations & Future Work
*   Evaluation is primarily on ImageNet-100; ImageNet-1K is only in the appendix. More evidence is needed to confirm if the sparsity-accuracy trade-off holds at larger scales.
*   Training introduces sliced Wasserstein, incurring $O(N \cdot B \log B)$ sorting overhead per step. The authors acknowledge efficiency costs for very large batches or high-dimensional projectors.
*   The choice of target $\sigma$ ($\sigma_{\text{GN}}$ vs $\sigma_{\text{RGN}}$) requires a binary search. The hyperparameter space increases; while the authors provide default recipes, whether cross-dataset re-tuning is necessary remains to be verified.
*   The paper only discusses image classification and does not touch upon dense prediction (detection/segmentation) where "semantic significance of non-zero positions" is truly required. The true value of sparse representations has not been fully stress-tested.

## Related Work & Insights
*   **vs LeJEPA (Balestriero & LeCun 2025)**: LeJEPA is a degenerate case of RGG when $\mu \to +\infty$ (no Dirac mass, unrectified features). Ours strictly generalizes this and recovers both sparsity and accuracy in the neighborhood of $\mu = 0$.
*   **vs VICReg / NVICReg**: VICReg only matches covariance/variance/invariance (second-order), failing to eliminate high-order dependencies. Ours proves that sliced 2-Wasserstein strictly subsumes the second-order components of VICReg (even using only covariance eigenvectors as projections) and results in significantly lower nHSIC.
*   **vs NCL-ReLU and Pure Sparse Baselines**: Pure sparse contrastive methods lag by $\sim 2\%$ in accuracy. Rectified LpJEPA aligns accuracy with LeJEPA by treating sparsity as a "distribution prior" rather than a "hard penalty."
*   **vs Sparse Coding / NMF Traditions**: Elevating "non-negativity + sparsity" from an engineering bias to a target distribution morphology in JEPA provides a new path for classic sparse coding in "end-to-end deep self-supervision."

## Rating
*   Novelty: ⭐⭐⭐⭐⭐ Formally extending the projection distribution family from Gaussian to RGG with two-sample sliced Wasserstein is a clear advancement in JEPA anti-collapse design.
*   Experimental Thoroughness: ⭐⭐⭐⭐ ImageNet-100 + multiple backbones + multiple sparsity levels + nHSIC/Entropy/OOD adaptation; ImageNet-1k only in the appendix is the only regret.
*   Writing Quality: ⭐⭐⭐⭐ Clear three-layer narrative (Theory / Intuition / Experiment) with well-coordinated formulas and figures.
*   Value: ⭐⭐⭐⭐ Provides an analytically adjustable engineering template for self-supervised representations that require both expressivity and sparsity, directly reusable for embodied/low-power scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] On Revisiting Entropy for Identifying Mislabeled Images](on_revisiting_entropy_for_identifying_mislabeled_images.md)
- [\[ICML 2026\] Possibilistic Predictive Uncertainty for Deep Learning](possibilistic_predictive_uncertainty_for_deep_learning.md)
- [\[ICML 2026\] Continual Learning of Domain-Invariant Representations](continual_learning_of_domain-invariant_representations.md)
- [\[AAAI 2026\] DECOR: Deep Embedding Clustering with Orientation Robustness](../../AAAI2026/others/decor_deep_embedding_clustering_with_orientation_robustness.md)
- [\[ICCV 2025\] Joint Asymmetric Loss for Learning with Noisy Labels](../../ICCV2025/others/joint_asymmetric_loss_for_learning_with_noisy_labels.md)

</div>

<!-- RELATED:END -->
