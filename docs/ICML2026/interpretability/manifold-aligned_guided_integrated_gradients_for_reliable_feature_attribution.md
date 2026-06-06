---
title: >-
  [Paper Note] Manifold-Aligned Guided Integrated Gradients for Reliable Feature Attribution
description: >-
  [ICML 2026][Interpretability][integrated gradients] Ours proposes MA-GIG: it transfers the "select features by low gradient magnitude" strategy of Guided IG from pixel space to the latent space of a pre-trained VAE. By u…
tags:
  - "ICML 2026"
  - "Interpretability"
  - "integrated gradients"
  - "guided IG"
  - "data manifold"
  - "VAE"
  - "path methods"
date: 2026-05-08
content_hash: b4431e61cee1dad8
---

# Manifold-Aligned Guided Integrated Gradients for Reliable Feature Attribution

**Conference**: ICML 2026  
**arXiv**: [2605.02167](https://arxiv.org/abs/2605.02167)  
**Code**: https://github.com/leekwoon/ma-gig (Available)  
**Area**: Interpretability / Feature Attribution / Integrated Gradients  
**Keywords**: integrated gradients, guided IG, data manifold, VAE, path methods

## TL;DR
Ours proposes MA-GIG: it transfers the "select features by low gradient magnitude" strategy of Guided IG from pixel space to the latent space of a pre-trained VAE. By utilizing the decoder Jacobian to map axis-aligned latent updates into updates within the tangent space of the data manifold, the method avoids high-gradient noise regions while ensuring intermediate samples on the integration path remain close to the real data manifold, leading to more reliable attribution.

## Background & Motivation

**Background**: Integrated Gradients (IG) has become a standard for path attribution due to axiomatic guarantees like completeness and sensitivity, integrating gradients along a straight line from baseline to input. Subsequent works either change the baseline (e.g., Sturmfels et al.) or the path—where Guided IG (GIG) avoids gradient noise by selecting features with low gradient magnitudes at each step, and EIG/MIG place the path in VAE latent space to align with the manifold.

**Limitations of Prior Work**: (1) The straight-line path of IG may pass through high-variance regions with extreme gradient oscillations, accumulating spurious gradients into the attribution; (2) While GIG reduces noise, it operates in pixel space where intermediate samples drift away from the natural image manifold, leading to undefined gradient behavior; (3) EIG/MIG reduce manifold deviation by walking in latent space but completely ignore the geometry of the classifier's logit surface, potentially crossing high-curvature noise regions. These three approaches only address either "manifold alignment" or "gradient noise," but not both.

**Key Challenge**: Reliable attribution requires **simultaneously** (i) keeping intermediate samples close to the manifold (in-distribution), and (ii) steering the path away from high-variance logit regions. Achieving (ii) in pixel space inevitably violates (i) because sparse axis-aligned pixel updates are unlikely to fall within the tangent space of the data manifold. Conversely, simply walking in latent space loses the logit surface geometry.

**Goal**: (1) Formalize that "GIG’s off-manifold drift" is structural rather than accidental; (2) Migrate the low-gradient selection strategy of GIG to the latent space so that "sparse axis-aligned" updates naturally become "relevant updates within the manifold tangent space" through the decoder; (3) Quantitatively compare against traditional attribution methods across multiple classifiers and datasets.

**Key Insight**: It is noted that, assuming an ideal VAE satisfies perfect autoencoding ($D(E(x)) = x$ on the manifold, and the decoder is a smooth immersion), the columns of the decoder Jacobian $J_D(z)$ span $T_{D(z)}\mathcal{M}$. Therefore, **any direction in the latent space** pushed forward by the Jacobian falls within the tangent space.

**Core Idea**: Transfer the greedy low-gradient updates of GIG from pixel space to the VAE latent space. This ensures that axis-aligned updates are automatically converted into tangential updates via the decoder Jacobian—maintaining the same denoising mechanism while manifold alignment is provided "for free" by the geometric properties of the decoder.

## Method

### Overall Architecture
Input: Image $x$, baseline $x'$, classifier $f$, pre-trained VAE encoder $E$, decoder $D$, steps $K$, selection ratio $q$, step size $\eta$.  
Mechanism: (a) Encode $z = E(x), z' = E(x')$, initialize $z^{(0)} = z'$; (b) For each step $k=0,\dots,K-2$, first decode $\hat x^{(k)} = D(z^{(k)})$, then calculate latent gradient $g^{(k)} = J_D(z^{(k)})^\top \nabla_x f(\hat x^{(k)})$, set threshold $\tau^{(k)}$ as the $q$-quantile of $|g^{(k)}|$, and advance latent dimensions in the low-gradient subset $S^{(k)} = \{j: |g^{(k)}_j| \leq \tau^{(k)}\}$ toward $z$ with $\eta$; (c) Finally, perform path integration using pixel differences and pixel gradients between adjacent decoded points $\tilde x^{(k)} = D(z^{(k)})$: $\mathcal{A}_i = \sum_k \frac{\partial f(\tilde x^{(k)})}{\partial x_i}(\tilde x^{(k+1)}_i - \tilde x^{(k)}_i)$.

### Key Designs

1.  **Formalizing the Geometric Impossibility of "Input-Space Guidance $\Rightarrow$ Manifold Deviation"**:
    - **Function**: Theoretically proves that greedy updates in GIG within pixel space must leave the manifold, motivating the move to latent space.
    - **Mechanism**: The update $\Delta x^{(k)}$ in GIG at step $k$ is an **axis-aligned sparse vector** (non-zero only in selected pixel dimensions). Decomposing into tangent-normal components $\Delta x^{(k)} = \Delta x^{(k)}_\| + \Delta x^{(k)}_\perp$, the orthogonal component $\Delta x^{(k)}_\perp$ represents off-manifold drift. Ours proves Proposition 3.1: Given a manifold reach $\tau$, if $\|\Delta x^{(k)}_\perp\| > \frac{1}{\tau}\|\Delta x^{(k)}\|^2$ and $\|\Delta x^{(k)}\| \leq \tau/2$, then $x^{(k+1)}\notin \mathcal{M}$ strictly holds. The key observation is that the orthogonal component of an axis-aligned displacement is **first-order** $\mathcal{O}(\|\Delta x\|)$, while the manifold curvature tolerance is only **second-order** $\mathcal{O}(\|\Delta x\|^2)$. Thus, for small steps, the first-order term dominates, and the path almost certainly escapes. The total deviation over $K$ steps is $d(x^{(K)}, \mathcal{M}) \leq \sum_k \|\Delta x^{(k)}_\perp\| + \mathcal{O}(\kappa)$.
    - **Design Motivation**: Prior work only empirically observed that "GIG intermediate images look unnatural." Ours upgrades this to a geometric statement: the tangent spaces of natural images are fundamentally misaligned with pixel axes, making sparse pixel updates structurally off-manifold—this is a mechanistic failure, not a hyperparameter issue.

2.  **Latent Space GIG: Moving the Greedy Strategy to $\mathcal{Z}$**:
    - **Function**: Performs GIG-style low-gradient selection and sparse advancement within $\mathcal{Z}$, letting the decoder translate this into on-manifold updates.
    - **Mechanism**: Latent gradients are computed via the chain rule and decoder Jacobian: $\nabla_z f(D(z^{(k)})) = J_D(z^{(k)})^\top \nabla_x f(D(z^{(k)}))$. A low-gradient subset $S_z^{(k)} = \{j: |\partial f / \partial z_j| \leq \tau_z^{(k)}\}$ is selected in $\mathcal{Z}$, and only these latent dimensions are updated: $\Delta z^{(k)} = \sum_{j \in S_z^{(k)}} \delta_j u_j$, where $u_j$ is the standard basis of $\mathcal{Z}$. Although $\Delta z^{(k)}$ is axis-aligned in $\mathcal{Z}$, its push-forward in pixel space $\Delta x^{(k)} \approx J_D(z^{(k)}) \Delta z^{(k)} = \delta_j \cdot \partial D / \partial z_j$ **is exactly the $j$-th column of the Jacobian**, which is the tangent vector of the decoder at that point.
    - **Design Motivation**: Under Assumption 3.2 (Perfect Autoencoder), $\mathrm{Im}(J_D(z)) = T_{D(z)}\mathcal{M}$. Thus, any direction in latent space pushed forward by the Jacobian **falls into the tangent space**. The failure of GIG in pixel space stems from $\{e_i\}$ being misaligned with the tangent space; MA-GIG replaces the basis with $\{\partial D / \partial z_j\}$, ensuring geometric alignment. This "change of basis" makes manifold alignment a "free byproduct" rather than a "hard constraint."

3.  **Baseline Encoding + Decoded Path Integration Formula**:
    - **Function**: Maps the latent path back to pixel space for final attribution, maintaining the completeness property of IG.
    - **Mechanism**: The baseline is initialized in latent space as $z^{(0)} = z' = E(x')$, and finally $z^{(K)} = z$ (anchored to the real $z$). Pixel endpoints are forced to be $\tilde x^{(0)} = x'$ and $\tilde x^{(K)} = x$ (to prevent reconstruction error from polluting attribution), while intermediate points are decoded as $\tilde x^{(k)} = D(z^{(k)})$. Attribution follows discrete IG: $\mathcal{A}_i = \sum_{k=0}^{K-1}\frac{\partial f(\tilde x^{(k)})}{\partial x_i}(\tilde x^{(k+1)}_i - \tilde x^{(k)}_i)$.
    - **Design Motivation**: Calculating $\mathcal{A}_z$ directly in latent space is uninterpretable—users need to know which **pixels** are important. Forcing endpoints to be real $x', x$ instead of $D(z'), D(z)$ addresses the completeness gap caused by imperfect VAE reconstruction.

### Loss & Training
MA-GIG is an **inference-only** algorithm and introduces no new training losses. It requires a pre-trained VAE (Ours uses MAR backbone; Appendix also tests Stable Diffusion's VAE, etc.). Hyperparameters include $K$ (steps), $q$ (fraction), and $\eta$ (step size).

## Key Experimental Results

### Main Results
Three datasets (ImageNet, Oxford-IIIT Pet, Oxford 102 Flower) and three classifiers (ResNet18, VGG16, InceptionV1). Metrics: DiffID (↑), Insertion AUC (↑), Deletion AUC (↓). Representative results on Oxford-IIIT Pet:

| Method | ResNet18 DiffID | ResNet18 Ins | ResNet18 Del | VGG16 DiffID | InceptionV1 DiffID |
|---|---|---|---|---|---|
| G×I | 0.2384 | 0.4378 | 0.1994 | 0.4060 | 0.2255 |
| IG | 0.3790 | 0.5186 | 0.1396 | 0.5255 | 0.3438 |
| IG² | 0.3823 | 0.5264 | 0.1441 | 0.6075 | 0.4273 |
| AGI | 0.2787 | 0.4453 | 0.1667 | 0.4471 | 0.3381 |
| EIG | 0.3595 | 0.4964 | 0.1369 | 0.4949 | 0.3306 |
| MIG | 0.3486 | 0.4889 | 0.1402 | 0.4850 | 0.3180 |
| **MA-GIG** | **Best/2nd** | **Best/2nd** | **Best/2nd** | **Best/2nd** | **Best/2nd** |

(Table 1 shows MA-GIG covers Best/2nd Best across all 9 backbone-dataset combinations for DiffID and Insertion, with leading Deletion scores.)

### Ablation Study

| Config | Performance | Note |
|---|---|---|
| MA-GIG (MAR VAE) | Best | Main results backbone |
| Other VAE backbones (LDM VAE, etc.) | Still leading | Robustness to generative priors |
| $q, \eta, K$ range | Stable | Not sensitive to hyperparameters |
| Degrading to pixel GIG | Significant drop | Validates role of manifold alignment |
| EIG (No greedy selection) | Inferior | Validates necessity of logit-aware selection |
| MIG (Geodesic path, no selection) | Inferior | Same as above |

### Key Findings
- **Manifold alignment + noise suppression must coexist**: EIG/MIG fail by only solving the former; GIG fails by only solving the latter. MA-GIG succeeds by proving these tasks are **complementary**.
- **Performance scales with generative prior quality but remains robust**: Leading results with various VAEs suggest imperfect VAEs still provide useful tangent space approximations.
- **Qualitative Visualization**: Attribution maps for MA-GIG are significantly more concentrated on foreground class-relevant regions with reduced background noise.
- **Completeness is preserved**: Using real endpoints $x', x$ instead of $D(z'), D(z)$ makes the completeness of IG numerically sound despite imperfect VAEs.

## Highlights & Insights
- **Proposition 3.1 is a elegant geometric statement**: It upgrades the intuitive observation "GIG samples look unnatural" to a rigorous impossibility: "axis-aligned updates + manifold geometry $\Rightarrow$ unavoidable drift," providing a hard motive for basis transformation.
- **The move to $\mathcal{Z}$ is a minimal-intervention modification**: The algorithm skeleton remains almost identical to GIG, simply changing the basis from $\{e_i\}$ to $\{u_j\}$ pushed by $J_D$. This proves that the right coordinate system can solve manifold issues for any iterative perturbation method.
- **Decoder Jacobian as a tangent basis**: The observation that Jacobian columns provide a natural basis for the manifold tangent space is used as a first-class tool and is a valuable primitive for future work.
- **Engineering detail of endpoint anchoring**: Forcing $\tilde x^{(0)} = x', \tilde x^{(K)} = x$ solves the completeness gap from reconstruction errors, which is a practical tip for VAE-based attribution.

## Limitations & Future Work
- Strict geometric guarantees rely on the **Perfect Autoencoder assumption**; real VAEs have reconstruction errors and topological defects.
- **Dependency on pre-trained VAEs**: Deployment costs are higher than IG/GIG, and the VAE must match the classifier's training domain. Application to OOD scenarios or domains without good VAEs (e.g., medical, radar) is limited.
- Computational overhead is higher than IG due to $\nabla_x f$ calculations + Jacobian-vector products + decoding per step.
- Only verified for image classification. Smooth immersion properties in VAEs for non-image modalities (text, tabular) are not guaranteed.
- Future work could explore attribution-specific VAEs or use diffusion score functions instead of decoder Jacobians for tangent space projection.

## Related Work & Insights
- **vs IG**: IG walks straight through high-variance regions; Ours uses VAE to approximate manifold paths and avoid noise.
- **vs GIG**: GIG uses low-gradient selection in pixel space but drifts off-manifold; MA-GIG solves this via latent space.
- **vs EIG / MIG**: They use latent interpolation/geodesics, which align with the manifold but ignore logit curvature; MA-GIG combines logit-aware greedy selection with manifold alignment.
- **vs AGI**: AGI integrates from adversarial starting points, leading to extreme extrapolation; MA-GIG is more stable using low-gradient paths.
- **Insight**: Rewriting iterative perturbation algorithms (adversarial attacks, reverse engineering, IG series) into "VAE latent space" versions might be a universal "free lunch"—this paper provides a clear geometric template for it.

## Rating
- Novelty: ⭐⭐⭐⭐ First IG variant to simultaneously tackle manifold alignment and logit noise suppression with strong geometric arguments.
- Experimental Thoroughness: ⭐⭐⭐⭐ 3 datasets × 3 classifiers + multiple backbones + qualitative analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Well-structured flow from geometric motivation to proof and experiments.
- Value: ⭐⭐⭐⭐ Practical improvement for the interpretability community with a transferable "basis-transformation" strategy.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Distribution-Based Feature Attribution for Explaining the Predictions of Any Classifier](../../AAAI2026/interpretability/distribution-based_feature_attribution_for_explaining_the_predictions_of_any_cla.md)
- [\[ICML 2026\] MUSE: Resolving Manifold Misalignment in Visual Tokenization via Topological Orthogonality](muse_resolving_manifold_misalignment_in_visual_tokenization_via_topological_orth.md)
- [\[CVPR 2026\] Feature Attribution Stability Suite: How Stable Are Post-Hoc Attributions?](../../CVPR2026/interpretability/feature_attribution_stability_suite_how_stable_are_post-hoc_attributions.md)
- [\[ICML 2026\] SemGrad: Gradients w.r.t. Semantics-Preserving Embeddings Tell LLM Uncertainty](gradients_with_respect_to_semantics_preserving_embeddings_tell_the_uncertainty_o.md)
- [\[ICML 2026\] Barriers to Counterfactual Credit Attribution for Autoregressive Models](barriers_to_counterfactual_credit_attribution_for_autoregressive_models.md)

</div>

<!-- RELATED:END -->
