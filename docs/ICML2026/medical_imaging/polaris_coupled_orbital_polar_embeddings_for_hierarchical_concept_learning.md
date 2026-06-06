---
title: >-
  [Paper Note] Polaris: Coupled Orbital Polar Embeddings for Hierarchical Concept Learning
description: >-
  [ICML 2026][Medical Imaging][Polar embeddings] Polaris decouples concept representations into "direction (semantics) + orbital potential (hierarchy)" signals…
tags:
  - "ICML 2026"
  - "Medical Imaging"
  - "Polar embeddings"
  - "unit hypersphere"
  - "vMF distribution"
  - "Stein variational gradient"
  - "taxonomy expansion"
date: 2026-05-08
content_hash: 680f73ae4d369765
---

# Polaris: Coupled Orbital Polar Embeddings for Hierarchical Concept Learning

**Conference**: ICML 2026  
**arXiv**: [2605.00265](https://arxiv.org/abs/2605.00265)  
**Code**: None  
**Area**: Representation Learning / Hierarchical Concept Learning / Taxonomy Expansion / Spherical Embeddings  
**Keywords**: Polar embeddings, unit hypersphere, vMF distribution, Stein variational gradient, taxonomy expansion

## TL;DR
Polaris decouples concept representations into "direction (semantics) + orbital potential (hierarchy)" signals, both learned on the unit hypersphere. It utilizes tangent space projection and exponential mapping to ensure manifold closure, employs anisotropic spherical SVGD to prevent equatorial collapse, and implements vMF KL divergence for asymmetric "parent broader than child" constraints. It improves top-K recall by up to 19 points and reduces mean rank by 60% in taxonomy expansion tasks.

## Background & Motivation
**Background**: Taxonomy expansion (placing new concepts under the correct parent in an existing tree/DAG) is a core problem for knowledge graphs, recommendations, product classification, and medical ontologies. Prevailing methods include: (1) Euclidean embeddings with symmetric similarity (TransE, TaxoExpan); (2) Hyperbolic embeddings leveraging exponential volume growth to alleviate tree crowding (Poincaré, HyperExpan); (3) Container embeddings (ConE, Box, Gumbel Box) to explicitly encode "child contained by parent" partial orders.

**Limitations of Prior Work**: Euclidean methods fail to express the naturally asymmetric parent-child relationship. Hyperbolic methods are sensitive to optimization and numerical precision. Container methods, while expressing partial orders, often conflate semantic similarity and hierarchical position, where small semantic errors escalate into large placement errors. **Polar embeddings** (using direction for semantics and radius/angle for hierarchy) can decouple these signals, but previous approaches required ad-hoc tricks: modulo operations to wrap angles, sector-specific losses, or sigmoid rescaling—all of which break manifold continuity and lead to severe high-dimensional drift under weak supervision.

**Key Challenge**: To express "semantic direction independent of hierarchical position," one must use polar geometry. However, conventional parameterization (learning $(\theta, \psi)$ directly with $\bmod 2\pi$) implicitly models the sphere as a zero-curvature cylinder, which is topologically inconsistent with a true constant-curvature sphere, leading to unstable optimization.

**Goal**: (1) Perform manifold-consistent polar learning on the unit hypersphere without hacks like wrap/mod/sigmoid; (2) Decouple semantic direction from hierarchical position; (3) Stably learn partial-order structures under noisy/weak supervision; (4) Effortlessly reduce retrieval space using structural priors.

**Key Insight**: The authors observe that since a sphere is a manifold of constant curvature, one should avoid singular angle parameterization. Instead, learn unit-norm vectors in Cartesian coordinates (via tangent space projection and exponential mapping) and use inner products as angle surrogates. Hierarchy signals are "isolated" into an orbital potential derived from the existing hierarchy rather than being forced into the same angular coordinates.

**Core Idea**: Learn direction on $\mathbb{S}^{d-1}$ for semantic encoding, use an orbital potential derived from the known hierarchy to encode "depth," and utilize vMF distributions with asymmetric KL divergence to explicitly model the "broader parent, narrower child" constraint.

## Method

### Overall Architecture
Input: A seed taxonomy (tree/DAG/multimodal) + PLM/CLIP features $\mathbf{e} \in \mathbb{R}^{d_\text{plm}}$ for new concepts. Output: A unit vector $\mathbf{z} \in \mathbb{S}^{d-1}$ and vMF parameters $(\boldsymbol\mu, \kappa)$ for each concept. The pipeline consists of four parts: (1) **Manifold-Consistent Encoding**: Projects $\mathbf{e}$ to the tangent space at the North Pole to obtain $\mathbf{v}$, then uses the exponential map $\mathbf{z}_0 = \cos(\|\mathbf{v}\|)\mathbf{p}_N + \sin(\|\mathbf{v}\|)\mathbf{v}/\|\mathbf{v}\|$ to lift it to the sphere, followed by a "spherical linear layer" (row $\ell_2$ normalization, no bias, output re-projection). (2) **Geometric Learning**: Uses the Welsch M-estimator to wrap the geodesic angle $\theta_{ij} = \arccos\langle\mathbf{z}_i, \mathbf{z}_j\rangle$ for triplet loss. (3) **Global Regularization**: Employs anisotropic spherical SVGD to push embeddings away from the equator toward the poles, preventing equatorial crowding caused by high-dimensional concentration-of-measure. (4) **Probabilistic Learning**: Models each concept as $\text{vMF}(\boldsymbol\mu_i, \kappa_i)$ and uses asymmetric KL to enforce $\kappa_p < \kappa_c$, ensuring parent distributions are broader. (5) **Inference**: Uses hierarchy-derived orbital potentials to set dynamic cosine thresholds for coarse hierarchical screening followed by fine angular ranking.

### Key Designs

1.  **Manifold-Consistent Spherical Encoding (Tangent Space Projection + Exponential Map + Spherical Linear Layer)**:
    - **Function**: Strictly maps Euclidean features from PLMs onto $\mathbb{S}^{d-1}$ and ensures all subsequent linear transformations are "manifold-closed."
    - **Mechanism**: Projects onto the tangent space at North Pole $\mathbf{p}_N$ via $\mathbf{v} = \mathbf{e} - \langle\mathbf{e}, \mathbf{p}_N\rangle\mathbf{p}_N$; then uses $\mathbf{z}_0 = \exp_{\mathbf{p}_N}(\mathbf{v}) = \cos(\|\mathbf{v}\|)\mathbf{p}_N + \sin(\|\mathbf{v}\|)\mathbf{v}/\|\mathbf{v}\|$ to lift along the geodesic. Linear layers perform three actions: (a) Enforce $\|\mathbf{w}_i\|_2 = 1$ for weights after initialization and Riemannian gradient steps; (b) Remove bias to prevent origin translation from breaking symmetry; (c) Re-project output via $\mathbf{y} = \mathbf{W}\mathbf{x}/\|\mathbf{W}\mathbf{x}\|_2$.
    - **Design Motivation**: Hacky parameterizations like $\theta \leftarrow \theta \bmod 2\pi$ treat the sphere as a cylinder, causing gradient discontinuities. Tangent space + exp map is the standard Riemannian geometric approach, ensuring continuous differentiability. Theorem 2.2 proves Welsch loss is $\mathsf{SO}(d)$ invariant, meaning it relies only on relative geometry.

2.  **Welsch Geodesic Triplets + Anisotropic Spherical SVGD Regularization**:
    - **Function**: (a) Stably learns local parent-child relations under noisy semantics; (b) Prevents embeddings from collapsing to the equator in high dimensions.
    - **Mechanism**: Geodesic angle $\theta_{ij} = \arccos\langle\mathbf{z}_i, \mathbf{z}_j\rangle$ is calculated via inner product. A bounded Welsch M-estimator $\mathcal{W}(\theta) = 1 - \exp(-\theta^2/(2c^2))$ limits outlier influence. The triplet loss $\mathcal{L}_\text{geom} = \max(0, \gamma + \mathcal{W}(\theta_{cp}) - \mathcal{W}(\theta_{cn}))$ pulls children toward parents. SVGD treats embeddings as particles with velocity field $\phi(\mathbf{z}) = \mathbb{E}_{\mathbf{z}'}[k(\mathbf{z}', \mathbf{z})\nabla\log p(\mathbf{z}') + \nabla k(\mathbf{z}', \mathbf{z})]$ using a vMF kernel $k(\mathbf{z}', \mathbf{z}) = \exp(\kappa\mathbf{z}'^\top\mathbf{z})$. The target score includes: (a) structural item $\nabla\log p_\text{struct} = [0, \dots, 0, z_d/(1-z_d^2)]^\top$ to push particles from the equator ($z_d \approx 0$) to the poles, and (b) alignment item $\nabla\log p_\text{align} = \kappa_\text{align}\boldsymbol\mu$ to keep embeddings near their anchors.
    - **Design Motivation**: Theorem 2.3 shows $\sigma\{|\langle\mathbf{z}, \mathbf{u}\rangle| \geq \epsilon\} \leq 2\exp(-d\epsilon^2/2)$—random vectors in high dimensions exponentially concentrate at the equator. Since rotation-invariant losses provide no gradient signal to counter this, SVGD is needed to inject "anti-equatorial" forces.

3.  **vMF Asymmetric KL + Orbital Retrieval**:
    - **Function**: (a) Uses distributions instead of points to express "semantic volume," making parents broader than children; (b) Uses hierarchical coarse-to-fine filtering for faster and more accurate inference.
    - **Mechanism**: vMF parameters are derived: $\boldsymbol\mu_i = f_\text{sphere}(\mathbf{z}_i; \Theta_\mu)$, $\kappa_i = \text{Softplus}(\mathbf{w}_\kappa^\top\mathbf{z}_i + b_\kappa)$. The asymmetric KL is approximated as $D_\text{KL}(\text{vMF}_c\|\text{vMF}_p) = \log C_d(\kappa_c) - \log C_d(\kappa_p) - \mathcal{A}_d(\kappa_c)(\kappa_c - \kappa_p\boldsymbol\mu_c^\top\boldsymbol\mu_p)$. This requires $\kappa_p < \kappa_c$ (parent higher entropy) and alignment between $\boldsymbol\mu_p$ and $\boldsymbol\mu_c$. A probabilistic triplet loss $\mathcal{L}_\text{vMF} = \max(0, \gamma_\text{prob} + D_\text{KL}(c\|p) - D_\text{KL}(c\|n))$ is used. Optimization employs Riemannian Adam.
    - **Design Motivation**: Distance cannot distinguish "Dog" from "Mammal"—the latter has a larger semantic volume. vMF uses $1/\kappa$ as a proxy for volume. Orbital retrieval reduces inference cost from a global search to a small gated candidate set.

### Loss & Training
Total loss: $\mathcal{L} = \mathcal{L}_\text{geom} + \lambda_\text{SVGD}\mathcal{L}_\text{SVGD} + \lambda_\text{vMF}\mathcal{L}_\text{vMF}$. Each term manages local learning, global coverage, and asymmetric constraints. Optimization uses Riemannian Adam for hypersphere parameters and standard Adam for auxiliary heads.

## Key Experimental Results

### Main Results
Evaluated on single-parent trees (Science / WordNet / Environment) against 14 baselines. Reports R@1, R@5, Wu&P, MR, and MRR (mean of 5 seeds). Results show "consistent improvements of up to ~19 points in top-K retrieval and up to ~60% reduction in mean rank."

| Dataset | Metric | Best Baseline (STEAM) | Polaris Performance | Gain |
| :--- | :--- | :--- | :--- | :--- |
| Science | R@1 / R@5 / MR↓ | 34.8 / 59.7 / 31.7 | ~44 / ~70 / ~13 | top-K +~9-10 pts, MR -~60% |
| WordNet | R@1 / R@5 / MR↓ | 24.9 / 54.5 / 61.1 | ~31 / ~60 / ~25 | top-K +~6 pts |
| Environment | R@1 / R@5 / MR↓ | 34.7 / 51.1 / 28.7 | ~39 / ~55 / ~15 | top-K +~4 pts |

Consistent improvements were also observed in multi-parent DAG and multimodal hierarchy settings.

### Ablation Study

| Configuration | Key Change | Note |
| :--- | :--- | :--- |
| Full Polaris | All components | Full model performance |
| w/o SVGD | Removed SVGD | Embeddings drifted to equator; depth signal lost |
| w/o vMF | Used point triplets | Width difference vanished; asymmetry degraded |
| w/o orbital retrieval | Global $\arg\max$ | Speed decreased; accuracy dropped without prior |
| Welsch → Squared Dist | No M-estimator | Outlier impact increased; performance dropped |
| Polar wrap baseline | $(\theta, \psi) + \bmod 2\pi$ | Unstable optimization (see Appendix I) |

### Key Findings
- **SVGD Corrects Equatorial Collapse**: Theorem 2.3 explains the concentration of random vectors; the polar-biased score from anisotropic SVGD is necessary to maintain latent depth structure.
- **vMF KL Asymmetry as "Soft" Partial Order**: Hard constraints (like ConE) often collapse; vMF softens partial orders into "higher parent entropy," which is more robust to noise.
- **Orbital Retrieval Efficiently Reduces Space**: Dynamic cosine thresholds derived from hierarchical potentials make retrieval faster and cleaner.
- **Manifold Consistency vs. Angle Hacks**: Continuous Riemannian optimization outperforms discontinuous wrapping methods.

## Highlights & Insights
- **Decoupling is Geometrically Rigorous**: Semantics follow $\mathbf{z} \in \mathbb{S}^{d-1}$ while hierarchy follows orbital potential, making the signals structurally independent.
- **SVGD as a Spherical Regularizer**: An elegant application—using SVGD to ensure particles match a "non-equatorially concentrated" target distribution, countering the high-dimensional concentration-of-measure.
- **vMF KL "Asymmetry = Partial Order"**: Using $\kappa$ as a proxy for volume and enforcing $\kappa_p < \kappa_c$ is a concise solution for modeling hierarchy with distributions.
- **Transferable Tricks**: Spherical linear layers and the combination of rotation-invariant loss with orbital bias can be applied to any contrastive learning or retrieval task on spheres.

## Limitations & Future Work
- **Hierarchy Dependence**: Requires an existing hierarchy to derive orbital potentials; not directly applicable to cold-start scenarios without an initial skeleton.
- **vMF Numerical Stability**: The Bessel ratio $\mathcal{A}_d(\kappa)$ can be unstable for extremely sharp distributions in high dimensions.
- **Uniform Curvature Assumption**: Hyperspheres assume constant curvature; future work could explore mixed-curvature (Spherical $\times$ Hyperbolic) representations.
- **Multimodal Scaling**: More verification is needed for truly large-scale multimodal taxonomies.
- **Code Availability**: Reproducibility depends on specific hyperparameters (SVGD kernel temperature, etc.) noted in the appendix.

## Related Work & Insights
- **vs Poincaré / HyperExpan**: Hyperbolic methods rely on exponential volume growth; Polaris explicitly separates "depth" into potentials, avoiding hyperbolic numerical difficulties.
- **vs ConE / Box / Gumbel Box**: Unlike hard container constraints, Polaris softens partial orders into entropy differences via vMF, proving more robust across all single-parent benchmarks.
- **vs HAKE**: HAKE uses modulus for depth, but Polaris strictly fixes the norm to 1 and moves depth to the orbital potential, avoiding modulus/angle coupling issues.
- **vs TaxoExpan / STEAM**: Polaris significantly outperforms these Euclidean GNN baselines due to the combined strength of spherical geometry and probabilistic asymmetry.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The combination of polar spherical embeddings, anti-equatorial SVGD, and asymmetric vMF KL is novel and effectively addresses "optimization instability," "equatorial collapse," and "partial order softening."
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers 3 hierarchy types, 14 baselines, and comprehensive ablations.
- **Writing Quality**: ⭐⭐⭐⭐ Clear geometric motivation and theoretical grounding; self-contained despite heavy math.
- **Value**: ⭐⭐⭐⭐ Provides a robust template for taxonomy expansion: "distributional representation + manifold consistency + global regularization."

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Bridging Explainability and Embeddings: BEE Aware of Spuriousness](../../ICLR2026/medical_imaging/bridging_explainability_and_embeddings_bee_aware_of_spuriousness.md)
- [\[ICLR 2026\] Deep Hierarchical Learning with Nested Subspace Networks for Large Language Models](../../ICLR2026/medical_imaging/deep_hierarchical_learning_with_nested_subspace_networks_for_large_language_mode.md)
- [\[AAAI 2026\] ReCoN-Ipsundrum: An Inspectable Recurrent Persistence Loop Agent with Affect-Coupled Cognition](../../AAAI2026/medical_imaging/recon-ipsundrum_an_inspectable_recurrent_persistence_loop_agent_with_affect-coup.md)
- [\[CVPR 2026\] Focus-to-Perceive Representation Learning: A Cognition-Inspired Hierarchical Framework for Endoscopic Video Analysis](../../CVPR2026/medical_imaging/focus-to-perceive_representation_learning_a_cognition-inspired_hierarchical_fram.md)
- [\[CVPR 2026\] Unlocking Positive Transfer in Incrementally Learning Surgical Instruments: A Self-reflection Hierarchical Prompt Framework](../../CVPR2026/medical_imaging/unlocking_positive_transfer_in_incrementally_learning_surgical_instruments_a_sel.md)

</div>

<!-- RELATED:END -->
