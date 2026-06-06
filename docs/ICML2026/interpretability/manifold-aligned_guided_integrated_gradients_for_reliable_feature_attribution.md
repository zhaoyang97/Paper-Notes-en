---
title: >-
  [Paper Note] Manifold-Aligned Guided Integrated Gradients for Reliable Feature Attribution
description: >-
  [ICML 2026][Interpretability][integrated gradients] This paper proposes MA-GIG: transferring the “select low-gradient features and take a step” strategy of Guided IG from pixel space to the latent space of a pretrained V…
tags:
  - "ICML 2026"
  - "Interpretability"
  - "integrated gradients"
  - "guided IG"
  - "data manifold"
  - "VAE"
  - "path methods"
date: 2026-05-08
content_hash: fcdbd0f9e676f34e
---

# Manifold-Aligned Guided Integrated Gradients for Reliable Feature Attribution

**Conference**: ICML 2026  
**arXiv**: [2605.02167](https://arxiv.org/abs/2605.02167)  
**Code**: https://github.com/leekwoon/ma-gig (available)  
**Area**: Interpretability / Feature Attribution / Integrated Gradients  
**Keywords**: integrated gradients, guided IG, data manifold, VAE, path methods

## TL;DR
This paper proposes MA-GIG: transferring the “select low-gradient features and take a step” strategy of Guided IG from pixel space to the latent space of a pretrained VAE. By leveraging the decoder Jacobian, axis-aligned updates in latent space are mapped into updates within the tangent space of the data manifold, thus both avoiding high-gradient noise regions and ensuring that samples along the integration path remain close to the true data manifold, resulting in more reliable attributions.

## Background & Motivation

**Background**: Integrated Gradients (IG) has become the standard path-based attribution method due to axiomatic guarantees such as completeness and sensitivity, integrating gradients along a straight line from baseline to input. Subsequent work either changes the baseline (Sturmfels et al.) or the path—Guided IG (GIG) selects features with low gradient magnitude at each step to avoid noisy regions; EIG/MIG move the path into VAE latent space to stay close to the manifold.

**Limitations of Prior Work**: (1) IG’s straight-line path may traverse high-variance regions with oscillating gradients, accumulating spurious gradients into the attribution; (2) GIG reduces noise but still operates in pixel space, causing intermediate samples to deviate from the natural image manifold, where gradients are ill-defined; (3) EIG/MIG use VAE paths to reduce manifold deviation but ignore the geometry of the classifier’s logit surface, so the path may still cross high-curvature noisy regions. Each approach only addresses either “manifold alignment” or “gradient noise,” not both.

**Key Challenge**: Reliable attribution requires **simultaneously** (i) keeping intermediate samples in-distribution (on-manifold), and (ii) ensuring the path avoids high-variance logit regions. Achieving (ii) in pixel space inevitably breaks (i), as axis-aligned sparse pixel updates almost never fall within the tangent space of the data manifold; conversely, moving solely in latent space loses logit surface geometry.

**Goal**: (1) Formalize that “GIG’s off-manifold drift” is structural, not incidental; (2) Transfer GIG’s low-gradient selection strategy to latent space, so that axis-aligned updates become correlated updates within the manifold’s tangent space via the decoder; (3) Quantitatively compare with traditional attribution methods across multiple classifiers and datasets.

**Key Insight**: The authors observe that, assuming an ideal VAE with perfect autoencoding ($D(E(x)) = x$ on the manifold, decoder is a smooth immersion), the columns of the decoder Jacobian $J_D(z)$ span exactly $T_{D(z)}\mathcal{M}$, so **any direction in latent space** is mapped by the Jacobian into the tangent space.

**Core Idea**: Move GIG’s greedy low-gradient updates from pixel space to VAE latent space, so that axis-aligned updates are automatically mapped by the decoder Jacobian into tangent updates—retaining the same denoising mechanism, but with manifold alignment provided for free by the decoder’s geometry.

## Method

### Overall Architecture
Input: image $x$, baseline $x'$, classifier $f$, pretrained VAE encoder $E$, decoder $D$, number of steps $K$, selection ratio $q$, step size $\eta$.  
Procedure: (a) Encode $z = E(x), z' = E(x')$, initialize $z^{(0)} = z'$; (b) For each step $k=0,\ldots,K-2$, decode $\hat x^{(k)} = D(z^{(k)})$, compute latent gradient $g^{(k)} = J_D(z^{(k)})^\top \nabla_x f(\hat x^{(k)})$, take the $q$-quantile $\tau^{(k)}$ of $|g^{(k)}|$ as threshold, select the low-gradient subset $S^{(k)} = \{j: |g^{(k)}_j| \leq \tau^{(k)}\}$ and update those latent dimensions towards $z$ by $\eta$; (c) Finally, use the pixel differences between adjacent decoded points $\tilde x^{(k)} = D(z^{(k)})$ and pixel gradients to compute the path integral $\mathcal{A}_i = \sum_k \frac{\partial f(\tilde x^{(k)})}{\partial x_i}(\tilde x^{(k+1)}_i - \tilde x^{(k)}_i)$.

### Key Designs

1. **Formalizing the Geometric Impossibility of “Input Space Guidance ⇒ Manifold Alignment”**:

    - Function: Theoretically proves that GIG’s greedy updates in pixel space inevitably leave the manifold, motivating the switch to latent space.
    - Mechanism: At step $k$, GIG’s update $\Delta x^{(k)}$ is an **axis-aligned sparse vector** (nonzero only in selected pixel dimensions). Decompose as $\Delta x^{(k)} = \Delta x^{(k)}_\| + \Delta x^{(k)}_\perp$ (tangent + orthogonal), where $\Delta x^{(k)}_\perp$ is the off-manifold drift. Proposition 3.1: If the manifold reach is $\tau$, and $\|\Delta x^{(k)}_\perp\| > \frac{1}{\tau}\|\Delta x^{(k)}\|^2$ with $\|\Delta x^{(k)}\| \leq \tau/2$, then $x^{(k+1)}\notin \mathcal{M}$ strictly holds. The key observation: the orthogonal component of axis-aligned displacement is **first-order** $\mathcal{O}(\|\Delta x\|)$, but the manifold’s curvature tolerance is **second-order** $\mathcal{O}(\|\Delta x\|^2)$, so the first-order term dominates for small steps, making off-manifold drift almost inevitable. Over $K$ steps, total deviation $d(x^{(K)}, \mathcal{M}) \leq \sum_k \|\Delta x^{(k)}_\perp\| + \mathcal{O}(\kappa)$.
    - Design Motivation: Previous work only empirically observed that “GIG intermediate images look unnatural”; this paper elevates it to a geometric statement: the tangent space of natural images is misaligned with pixel axes, so sparse pixel updates structurally drift off-manifold—not due to poor hyperparameters, but due to the mechanism itself.

2. **Latent Space GIG: Transferring the Same Greedy Strategy to $\mathcal{Z}$**:

    - Function: Performs GIG-style low-gradient selection and sparse updates in $\mathcal{Z}$, letting the decoder map them into manifold updates.
    - Mechanism: Latent gradients are computed via chain rule + decoder Jacobian: $\nabla_z f(D(z^{(k)})) = J_D(z^{(k)})^\top \nabla_x f(D(z^{(k)}))$. In $\mathcal{Z}$, select the low-gradient subset $S_z^{(k)} = \{j: |\partial f / \partial z_j| \leq \tau_z^{(k)}\}$, and update only these latent dimensions $\Delta z^{(k)} = \sum_{j \in S_z^{(k)}} \delta_j u_j$, where $u_j$ is the standard basis of $\mathcal{Z}$. Although $\Delta z^{(k)}$ is axis-aligned in $\mathcal{Z}$, its pushforward in pixel space $\Delta x^{(k)} \approx J_D(z^{(k)}) \Delta z^{(k)} = \delta_j \cdot \partial D / \partial z_j$ **is exactly the $j$-th column of the Jacobian**, i.e., the decoder’s tangent vector at that point.
    - Design Motivation: Under Assumption 3.2 (Perfect Autoencoder), $\mathrm{Im}(J_D(z)) = T_{D(z)}\mathcal{M}$, so any latent direction is mapped **into the tangent space**. GIG fails in pixel space because $\{e_i\}$ is misaligned with the tangent space; MA-GIG replaces the basis with $\{\partial D / \partial z_j\}$, achieving geometric alignment. This “change of basis to make sparse updates automatically satisfy constraints” turns manifold alignment from a “hard constraint” into a “free byproduct,” requiring no projection or correction.

3. **Baseline Encoding + Decoded Point Path Integral Formula**:

    - Function: Converts the latent space path back to pixel space for final attribution, preserving IG’s completeness.
    - Mechanism: The baseline is initialized in latent space as $z^{(0)} = z' = E(x')$, and finally $z^{(K)} = z$ (anchored to the true $z$); pixel space endpoints are forced to be $\tilde x^{(0)} = x'$, $\tilde x^{(K)} = x$ (to avoid reconstruction error at endpoints), with intermediate points $\tilde x^{(k)} = D(z^{(k)})$. Attribution uses the discrete IG formula: $\mathcal{A}_i = \sum_{k=0}^{K-1}\frac{\partial f(\tilde x^{(k)})}{\partial x_i}(\tilde x^{(k+1)}_i - \tilde x^{(k)}_i)$.
    - Design Motivation: Attribution in latent space $\mathcal{A}_z$ is not interpretable—users care about **pixels**, not latent variables. Forcing endpoints to be the true $x', x$ (not $D(z'), D(z)$) addresses completeness gaps due to imperfect VAE; intermediate points use decoder outputs to keep the path near the manifold.

### Loss & Training
MA-GIG is a **pure inference-time** algorithm, introducing no new training losses. It requires a pretrained VAE (the paper uses MAR backbone, with additional tests on Stable Diffusion’s VAE, etc. in the appendix). Main hyperparameters are $K$ (steps), $q$ (selection ratio, as in GIG), $\eta$ (step size); details in Appendix F.

## Key Experimental Results

### Main Results
Three datasets: ImageNet / Oxford-IIIT Pet / Oxford 102 Flower; three classifiers: ResNet18 / VGG16 / InceptionV1. Metrics: DiffID (↑), Insertion AUC (↑), Deletion AUC (↓). Representative results on Oxford-IIIT Pet (higher is better except Del):

| Method | ResNet18 DiffID | ResNet18 Ins | ResNet18 Del | VGG16 DiffID | InceptionV1 DiffID |
|---|---|---|---|---|---|
| G×I | 0.2384 | 0.4378 | 0.1994 | 0.4060 | 0.2255 |
| IG | 0.3790 | 0.5186 | 0.1396 | 0.5255 | 0.3438 |
| IG² | 0.3823 | 0.5264 | 0.1441 | 0.6075 | 0.4273 |
| AGI | 0.2787 | 0.4453 | 0.1667 | 0.4471 | 0.3381 |
| EIG | 0.3595 | 0.4964 | 0.1369 | 0.4949 | 0.3306 |
| MIG | 0.3486 | 0.4889 | 0.1402 | 0.4850 | 0.3180 |
| **MA-GIG** | **Best/2nd Best** | **Best/2nd Best** | **Best/2nd Best** | **Best/2nd Best** | **Best/2nd Best** |

(Table 1 shows MA-GIG achieves best or second-best DiffID and Insertion across all 9 backbone-dataset combinations, and also leads in Deletion.)

### Ablation Study

| Configuration | Performance | Notes |
|---|---|---|
| MA-GIG (MAR VAE) | Best | Main result backbone |
| Switch to other VAE backbones (LDM VAE, etc., Appendix G.2) | Still leads | Demonstrates robustness to generative prior |
| Vary $q, \eta, K$ | Stable performance | Indicates insensitivity to hyperparameters |
| Degrade to pixel-space GIG | Significant drop | Validates the core role of manifold alignment |
| EIG (linear latent interpolation, no greedy low-gradient selection) | Inferior | Validates necessity of logit-aware selection |
| MIG (latent geodesic, no low-gradient selection) | Inferior | Same as above |

### Key Findings
- **Manifold alignment and gradient noise suppression must be combined**: EIG/MIG only address the former and lag behind, GIG only the latter and also lags; only MA-GIG combines both and wins, showing these are **complementary** rather than interchangeable.
- **Quality improves with generative prior quality but is not sensitive**: Using different VAEs still leads, indicating that even imperfect VAEs provide useful tangent space approximations (echoing the paper’s Practical Remark).
- **Qualitative visualization** (Fig. 2): MA-GIG attribution maps are more focused on foreground class regions, with significantly reduced background noise, consistent with improved DiffID.
- **Completeness is nearly preserved**: Using true endpoints $x', x$ instead of $D(z'), D(z)$ keeps IG completeness numerically reasonable even with imperfect VAEs.

## Highlights & Insights
- **Proposition 3.1 is an elegant geometric statement**: It upgrades the impressionistic observation that “GIG intermediate samples look unnatural” to a strict impossibility result: axis-aligned sparse updates + manifold reach geometry ⇒ inevitable off-manifold drift, providing a strong motivation for changing the coordinate basis.
- **Transferring the GIG strategy to $\mathcal{Z}$ is minimally invasive**: The algorithmic skeleton matches GIG almost one-to-one, only changing the basis from $\{e_i\}$ to $\{u_j\}$ and pushing forward via $J_D$, showing that “the same algorithm + correct coordinate system” can resolve the manifold issue—potentially transferable to any iterative perturbation method for images (adversarial examples, adversarial training, CAM variants).
- **Decoder Jacobian columns naturally provide a tangent basis for the manifold**: This observation is used as a first-class tool in this paper and is a valuable primitive for future work.
- **Endpoint anchoring engineering detail**: Forcing $\tilde x^{(0)} = x', \tilde x^{(K)} = x$ instead of decoded endpoints avoids completeness degradation due to imperfect VAEs—a practical trick worth reusing.

## Limitations & Future Work
- Strict geometric guarantees rely on the **Perfect Autoencoder assumption**; real VAEs have reconstruction errors and topological defects. While the paper’s Practical Remark claims imperfect VAEs still work, there is no quantitative curve relating “VAE quality → attribution quality,” so the limits are unclear.
- **Requires an extra pretrained VAE**, increasing deployment cost compared to IG/GIG, and the VAE must match the classifier’s training domain; applications are limited in OOD classifier or domains lacking suitable VAEs (e.g., medical/radar).
- Computational cost is significantly higher than IG: each step requires $\nabla_x f$ + Jacobian-vector product + decode, which may be a bottleneck for ultra-high-resolution images.
- Only tested on image classification; not validated on text, tabular, audio, or other non-image modalities, where VAEs/generative models may lack smooth immersion properties, so generalization is uncertain.
- Future work could explore learnable attribution-specific VAEs, or use diffusion score functions to replace the decoder Jacobian for tangent space projection.

## Related Work & Insights
- **vs IG**: IG uses a straight path through high-variance regions; this paper uses VAE to approximate manifold paths, avoiding noise.
- **vs GIG**: GIG uses low-gradient selection in pixel space to reduce noise but is off-manifold; MA-GIG transfers the same strategy to latent space, solving off-manifold for free.
- **vs EIG / MIG**: They use VAE linear interpolation/geodesics, aligning with the manifold but ignoring logit geometry, possibly traversing high-curvature regions; MA-GIG combines GIG’s logit-aware greediness with manifold alignment.
- **vs AGI**: AGI uses adversarial samples as starting points and integrates along the steepest ascent, leading to severe path extrapolation; MA-GIG uses low-gradient paths, yielding greater stability.
- **Insights**: Any iterative perturbation algorithm that “doesn’t work in pixel space” (adversarial attack, reverse engineering, IG variants) may benefit from a “VAE latent space” rewrite—this paper provides a clear geometric template.

## Rating
- Novelty: ⭐⭐⭐⭐ First IG variant to jointly address manifold alignment and logit noise suppression, with concise and powerful geometric arguments
- Experimental Thoroughness: ⭐⭐⭐⭐ 3 datasets × 3 classifiers + multiple VAE backbones + both qualitative and quantitative results
- Writing Quality: ⭐⭐⭐⭐⭐ Geometric motivation—assumptions—algorithm—proof—experiments are tightly connected, balancing theory and engineering
- Value: ⭐⭐⭐⭐ A practical improvement for the interpretability community, with the “change of basis for sparse updates” idea having broader applicability

## Related Papers

- [\[AAAI 2026\] Distribution-Based Feature Attribution for Explaining the Predictions of Any Classifier](../../AAAI2026/interpretability/distribution-based_feature_attribution_for_explaining_the_predictions_of_any_cla.md)
- [\[ICML 2026\] MUSE: Resolving Manifold Misalignment in Visual Tokenization via Topological Orthogonality](muse_resolving_manifold_misalignment_in_visual_tokenization_via_topological_orth.md)
- [\[CVPR 2026\] Feature Attribution Stability Suite: How Stable Are Post-Hoc Attributions?](../../CVPR2026/interpretability/feature_attribution_stability_suite_how_stable_are_post-hoc_attributions.md)
- [\[ACL 2025\] Normalized AOPC: Fixing Misleading Faithfulness Metrics for Feature Attribution Explainability](../../ACL2025/interpretability/normalized_aopc_faithfulness_metrics.md)
- [\[AAAI 2026\] FineVAU: A Novel Human-Aligned Benchmark for Fine-Grained Video Anomaly Understanding](../../AAAI2026/interpretability/finevau_a_novel_human-aligned_benchmark_for_fine-grained_video_anomaly_understan.md)

</div>

<!-- RELATED:END -->

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Distribution-Based Feature Attribution for Explaining the Predictions of Any Classifier](../../AAAI2026/interpretability/distribution-based_feature_attribution_for_explaining_the_predictions_of_any_cla.md)
- [\[ICML 2026\] SemGrad: Gradients w.r.t. Semantics-Preserving Embeddings Tell LLM Uncertainty](gradients_with_respect_to_semantics_preserving_embeddings_tell_the_uncertainty_o.md)
- [\[CVPR 2026\] Feature Attribution Stability Suite: How Stable Are Post-Hoc Attributions?](../../CVPR2026/interpretability/feature_attribution_stability_suite_how_stable_are_post-hoc_attributions.md)
- [\[ICML 2026\] Barriers to Counterfactual Credit Attribution for Autoregressive Models](barriers_to_counterfactual_credit_attribution_for_autoregressive_models.md)
- [\[CVPR 2026\] Geometry-Guided Camera Motion Understanding in VideoLLMs](../../CVPR2026/interpretability/geometry-guided_camera_motion_understanding_in_videollms.md)

</div>

<!-- RELATED:END -->
