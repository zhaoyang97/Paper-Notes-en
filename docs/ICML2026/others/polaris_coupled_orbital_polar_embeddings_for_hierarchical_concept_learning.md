---
title: >-
  [Paper Note] Polaris: Coupled Orbital Polar Embeddings for Hierarchical Concept Learning
description: >-
  [ICML 2026][Polar Coordinate Embedding] Polaris decomposes concept representations into two decoupled signals—"direction (semantics)" and "orbital potential (hierarchy)"—and learns both on the unit hypersphere: tangent s…
tags:
  - "ICML 2026"
  - "Polar Coordinate Embedding"
  - "Unit Hypersphere"
  - "vMF Distribution"
  - "Stein Variational Gradient"
  - "taxonomy expansion"
date: 2026-05-08
content_hash: 8368424c02e23a29
---

# Polaris: Coupled Orbital Polar Embeddings for Hierarchical Concept Learning

**Conference**: ICML 2026  
**arXiv**: [2605.00265](https://arxiv.org/abs/2605.00265)  
**Code**: None  
**Area**: Representation Learning / Hierarchical Concept Learning / Taxonomy Expansion / Spherical Embedding  
**Keywords**: Polar Coordinate Embedding, Unit Hypersphere, vMF Distribution, Stein Variational Gradient, taxonomy expansion

## TL;DR
Polaris decomposes concept representations into two decoupled signals—"direction (semantics)" and "orbital potential (hierarchy)"—and learns both on the unit hypersphere: tangent space projection plus exponential mapping ensures manifold closure, anisotropic spherical SVGD prevents equatorial concentration, and vMF KL divergence implements the asymmetric "parent should have higher entropy than child" constraint. On taxonomy expansion tasks, Polaris improves top-K recall by up to 19 points and reduces mean rank by 60%.

## Background & Motivation
**Background**: Taxonomy expansion (attaching new concepts to the correct parent node in an existing tree/DAG) is a core problem in knowledge graphs, recommendation, product categorization, and medical ontologies. Mainstream approaches fall into three categories: (1) Euclidean embeddings with symmetric similarity (TransE, TaxoExpan); (2) Hyperbolic embeddings leverage exponential volume growth to alleviate tree crowding (Poincaré, HyperExpan); (3) Container embeddings such as cone/box explicitly encode "child node is contained by parent" (ConE, Box, Gumbel Box).

**Limitations of Prior Work**: Euclidean methods cannot express the inherently asymmetric parent-child relationship with symmetric distances; hyperbolic methods are sensitive to optimization and numerical precision; container methods, while encoding partial order, require joint optimization of "semantic similarity" and "hierarchical position" in noisy or non-tree (DAG) structures, and their entanglement often amplifies small semantic errors into large placement errors. **Polar coordinate embeddings** (using direction for semantics and radius/angle for hierarchy) can decouple these signals, but previous polar coordinate methods rely on ad-hoc stabilization tricks: angle wrapping, sector-specific losses, sigmoid rescaling—all of which break manifold continuity and suffer from high-dimensional angle drift under weak supervision.

**Key Challenge**: To express "semantic direction independent of hierarchical position," polar geometry is necessary, but conventional parameterization (directly learning $(\theta,\psi)$ then mod $2\pi$) implicitly models the sphere as a flat cylinder of zero curvature, inconsistent with the true topology of a constant-curvature sphere, leading to unstable optimization.

**Goal**: (1) Perform manifold-consistent polar coordinate learning on the unit hypersphere, without hacks like wrap/mod/sigmoid; (2) Decouple semantic direction from hierarchical position; (3) Stably learn partial-order structure even under weak supervision/noisy semantics; (4) Efficiently narrow retrieval space using structural priors.

**Key Insight**: Since the sphere is a constant-curvature manifold, avoid singular angle parameterization and directly learn unit-norm vectors in Cartesian coordinates (using tangent space projection and exponential mapping), with inner product as an angle surrogate; separate the hierarchical signal into an orbital potential derived from the existing hierarchy, rather than embedding both into the same angular coordinate.

**Core Idea**: Learn direction-encoded semantics on $\mathbb{S}^{d-1}$, encode "depth" separately via an orbital potential derived from the known hierarchy, and use vMF distributions plus asymmetric KL to explicitly enforce "parent is broader, child is narrower" in the loss.

## Method

### Overall Architecture
Input: a seed taxonomy (tree/DAG/multimodal) plus a batch of new concepts (text or image) with PLM/CLIP features $\mathbf{e}\in\mathbb{R}^{d_\text{plm}}$. Output: each concept's unit vector $\mathbf{z}\in\mathbb{S}^{d-1}$ and vMF parameters $(\boldsymbol\mu,\kappa)$. The pipeline consists of four parts: (1) **Manifold-consistent encoding** projects $\mathbf{e}$ to the north pole tangent space to obtain $\mathbf{v}$, then uses exponential mapping $\mathbf{z}_0=\cos(\|\mathbf{v}\|)\mathbf{p}_N+\sin(\|\mathbf{v}\|)\mathbf{v}/\|\mathbf{v}\|$ to lift to the sphere, followed by a "spherical linear layer" (row-wise $\ell_2$ normalization, bias removal, output re-projection). (2) **Geometric learning** uses the Welsch M-estimator on geodesic angle $\theta_{ij}=\arccos\langle\mathbf{z}_i,\mathbf{z}_j\rangle$ for triplet loss. (3) **Global regularization** applies anisotropic spherical SVGD to push embeddings away from the equator and toward the poles, preventing equatorial concentration due to high-dimensional effects. (4) **Probabilistic learning** models each concept as vMF($\boldsymbol\mu_i$, $\kappa_i$), using asymmetric KL to enforce $\kappa_p<\kappa_c$, i.e., parent distributions are broader. (5) **Inference** uses hierarchy-derived orbital potential to dynamically threshold cosine similarity for coarse parent candidate filtering, followed by fine ranking by angle.

### Key Designs

1. **Manifold-consistent Spherical Encoding (Tangent Space Projection + Exponential Mapping + Spherical Linear Layer)**:

    - **Function**: Strictly maps PLM-provided Euclidean features onto $\mathbb{S}^{d-1}$, ensuring all subsequent linear transformations remain "closed on the manifold."
    - **Mechanism**: First, project to the tangent space at the north pole $\mathbf{p}_N$: $\mathbf{v}=\mathbf{e}-\langle\mathbf{e},\mathbf{p}_N\rangle\mathbf{p}_N$; then use exponential mapping $\mathbf{z}_0=\exp_{\mathbf{p}_N}(\mathbf{v})=\cos(\|\mathbf{v}\|)\mathbf{p}_N+\sin(\|\mathbf{v}\|)\mathbf{v}/\|\mathbf{v}\|$ to lift to the sphere along the geodesic. All linear layers perform: (a) row $\mathbf{w}_i$ is forced to $\|\mathbf{w}_i\|_2=1$ at initialization and after each Riemannian gradient update; (b) bias is removed (to avoid origin translation breaking spherical symmetry); (c) output is re-projected $\mathbf{y}=\mathbf{W}\mathbf{x}/\|\mathbf{W}\mathbf{x}\|_2$.
    - **Design Motivation**: Explicitly modeling angles via $\theta\leftarrow\theta\bmod 2\pi$ is equivalent to treating the sphere as a cylinder (zero curvature), with discontinuous gradients at wrap boundaries; tangent space plus exponential mapping is the standard Riemannian geometry approach, continuously differentiable and strictly norm-preserving. Theorem 2.2 proves the Welsch loss is $\mathsf{SO}(d)$-invariant—loss depends only on relative geometry, not specific axes, a direct benefit of geometric consistency.

2. **Welsch Geodesic Triplet + Anisotropic Spherical SVGD Regularization**:

    - **Function**: (a) Stably learns local parent-child relations under noisy semantics; (b) Prevents embeddings from collapsing to the equator in high dimensions.
    - **Mechanism**: Geodesic angle $\theta_{ij}=\arccos\langle\mathbf{z}_i,\mathbf{z}_j\rangle$ is computed via inner product (rotation-invariant). The bounded Welsch M-estimator $\mathcal{W}(\theta)=1-\exp(-\theta^2/(2c^2))$ limits outlier influence; triplet loss $\mathcal{L}_\text{geom}=\max(0,\gamma+\mathcal{W}(\theta_{cp})-\mathcal{W}(\theta_{cn}))$ pulls child nodes toward parents and pushes away from negatives. SVGD treats each embedding as a particle, with velocity field $\phi(\mathbf{z})=\mathbb{E}_{\mathbf{z}'}[k(\mathbf{z}',\mathbf{z})\nabla\log p(\mathbf{z}')+\nabla k(\mathbf{z}',\mathbf{z})]$, kernel chosen as vMF $k(\mathbf{z}',\mathbf{z})=\exp(\kappa\mathbf{z}'^\top\mathbf{z})$. The target score splits into: (a) structural term $\nabla\log p_\text{struct}=[0,\dots,0,z_d/(1-z_d^2)]^\top$ pushes particles from the equator ($z_d\approx 0$) toward the poles; (b) alignment term $\nabla\log p_\text{align}=\kappa_\text{align}\boldsymbol\mu$ keeps embeddings within their anchor's attraction domain. Finally, $\phi(\mathbf{z})$ is projected onto $T_\mathbf{z}\mathbb{S}^{d-1}$ to ensure valid updates.
    - **Design Motivation**: Theorem 2.3 gives $\sigma\{|\langle\mathbf{z},\mathbf{u}\rangle|\geq\epsilon\}\leq 2\exp(-d\epsilon^2/2)$—in high dimensions, random unit vectors concentrate exponentially at the equator; rotation-invariant geometric loss provides no gradient signal for this, so SVGD must inject an "anti-equator" force, otherwise embeddings collapse to $z_d\approx 0$ and depth signals are lost.

3. **vMF Asymmetric KL + Orbital Retrieval**:

    - **Function**: (a) Expresses "concept semantic volume" via distributions rather than points, making parents "broader" than children; (b) During inference, coarse-to-fine parent candidate filtering by hierarchy accelerates and improves accuracy.
    - **Mechanism**: Each node's vMF parameters are derived from point embeddings—$\boldsymbol\mu_i=f_\text{sphere}(\mathbf{z}_i;\Theta_\mu)$, $\kappa_i=\text{Softplus}(\mathbf{w}_\kappa^\top\mathbf{z}_i+b_\kappa)$. The vMF KL between parent and child is approximated as $D_\text{KL}(\text{vMF}_c\|\text{vMF}_p)=\log C_d(\kappa_c)-\log C_d(\kappa_p)-\mathcal{A}_d(\kappa_c)(\kappa_c-\kappa_p\boldsymbol\mu_c^\top\boldsymbol\mu_p)$, where $\mathcal{A}_d(\kappa)=I_{d/2}(\kappa)/I_{d/2-1}(\kappa)$ is the modified Bessel function ratio. This objective essentially enforces $\kappa_p<\kappa_c$ (parent entropy higher) and alignment of $\boldsymbol\mu_p$ and $\boldsymbol\mu_c$. The probabilistic triplet loss is $\mathcal{L}_\text{vMF}=\max(0,\gamma_\text{prob}+D_\text{KL}(c\|p)-D_\text{KL}(c\|n))$. Inference uses hierarchy-derived orbital potential to assign each layer a dynamic cosine threshold, first coarsely filtering candidate parents by "orbit," then fine ranking by angle. Optimization uses Riemannian Adam: Euclidean gradients are projected to $T_{\mathbf{z}_t}\mathcal{M}$, momentum is parallel transported to the new tangent space, and updates are $\mathbf{z}_{t+1}=\exp_{\mathbf{z}_t}(-\eta\hat{\mathbf{m}}_t/\sqrt{\hat{\mathbf{v}}_t})$.
    - **Design Motivation**: Pointwise distance cannot distinguish "dog" from "mammal"—the latter has a larger semantic volume; vMF uses $1/\kappa$ as a proxy for volume, and the asymmetric KL structure $\kappa_c-\kappa_p\boldsymbol\mu_c^\top\boldsymbol\mu_p$ constrains both direction and "width" in a single objective. Orbital retrieval reduces inference cost from a full $\arg\max$ to a small candidate set after hierarchical gating, crucial for large-scale taxonomies.

### Loss & Training
The total loss is $\mathcal{L}=\mathcal{L}_\text{geom}+\lambda_\text{SVGD}\mathcal{L}_\text{SVGD}+\lambda_\text{vMF}\mathcal{L}_\text{vMF}$, with each term responsible for: local parent-child learning, global manifold coverage, and asymmetric probabilistic constraints. Optimization uses Riemannian Adam; parameters on the hypersphere are updated on the sphere, while auxiliary Euclidean parameters use standard Adam.

## Key Experimental Results

### Main Results
Compared against 14 baselines on single-parent trees (Science / WordNet / Environment). Reports R@1, R@5, Wu&P, MR (lower is better), and MRR, averaged over five seeds. The abstract states "consistent improvements of up to ~19 points in top-K retrieval and up to ~60% reduction in mean rank."

| Dataset | Metric | Best Baseline (STEAM) | Polaris | Gain |
|---------|--------|----------------------|---------|------|
| Science | R@1 / R@5 / MR↓ | 34.8 / 59.7 / 31.7 | ~44 / ~70 / ~13 | top-K +~9-10 pts, MR -~60% |
| WordNet | R@1 / R@5 / MR↓ | 24.9 / 54.5 / 61.1 | ~31 / ~60 / ~25 | top-K +~6 pts |
| Environment | R@1 / R@5 / MR↓ | 34.7 / 51.1 / 28.7 | ~39 / ~55 / ~15 | top-K +~4 pts |

Multi-parent DAG and multimodal hierarchy also show consistent improvements (text: 9 / 6 / 4 points).

### Ablation Study

| Configuration | Key Change | Description |
|---------------|-----------|-------------|
| Full Polaris | Spherical encoding + Welsch geom + SVGD + vMF + orbital retrieval | Full model |
| w/o SVGD | Remove anisotropic spherical SVGD | Embeddings drift to equator in high dimensions, depth signal lost |
| w/o vMF | Replace probabilistic triplet with point triplet | Parent-child "width difference" disappears, asymmetry degrades |
| w/o orbital retrieval | Inference as full $\arg\max$ | Speed drops significantly, accuracy also drops due to lack of hierarchical prior |
| Welsch → squared distance | No M-estimator | Outlier influence amplified, weak supervision degrades |
| Polar wrap baseline | Explicit $(\theta,\psi)$ + mod $2\pi$ | Optimization unstable (see Appendix I for analysis) |

### Key Findings
- **SVGD correctly implements "anti-equator"**: Theorem 2.3 explains that high-dimensional spherical random vectors concentrate exponentially near the equator; rotation-invariant angle loss alone cannot counteract this. The anisotropic SVGD's pole-biased score is a necessary trick to restore "latent space depth structure."
- **vMF KL asymmetry is a "soft partial order"**: Hard constraints like Cone easily collapse; vMF softens partial order into "parent entropy higher," making it more robust to noise and providing a confidence radius proxy $1/\kappa$ for each concept.
- **Orbital retrieval truly reduces search space**: Hierarchy-derived potential gates candidate parents faster and more accurately than pure angle ranking, with significant engineering value for taxonomies with hundreds of thousands of nodes.
- **Manifold-consistent encoding vs. angle hacks**: Forcibly modding angles is equivalent to modeling the sphere as a cylinder, breaking curvature assumptions; ablation shows wrap-based polar coordinates cause optimization oscillations under weak supervision.

## Highlights & Insights
- **"Decoupling semantic direction and hierarchical position" is geometrically realized**: Direction is on $\mathbf{z}\in\mathbb{S}^{d-1}$, hierarchy is via orbital potential; the two signals are structurally independent. Compared to cone/box methods that entangle both in a single container with hard-tuned weights, this separation allows each signal to be learned "according to its own geometry."
- **Applying Stein variational gradient to spherical manifolds as regularization**: A rare but reasonable application—SVGD is designed to match a particle set to a target distribution; here, the target is a "non-equator-concentrated uniform + alignment distribution," countering high-dimensional concentration-of-measure. This trick is transferable to any scenario "learning embeddings on the sphere but avoiding collapse" (contrastive learning, retrieval, ArcFace-style face representation).
- **vMF KL "asymmetry = partial order"**: Treating $\kappa$ as a proxy for semantic volume and enforcing $\kappa_p<\kappa_c$ is a concise way to model partial order via distributions. Especially suitable for domains like medical ontologies or product categorization where parents are naturally more ambiguous.
- **Transferable tricks**: (1) Spherical linear layers (row normalization + bias removal + output re-projection) are applicable to any retrieval/contrastive learning scenario requiring features on the unit sphere; (2) Combining a rotation-invariant loss with an anchor-biased score can counteract similar uniformity issues.

## Limitations & Future Work
- **Depends on an existing hierarchy to derive orbital potential**: This approach does not work in pure cold-start scenarios (no skeleton at all); one could consider using LLMs to generate a rough hierarchy for bootstrapping.
- **vMF's Bessel ratio $\mathcal{A}_d(\kappa)$ is numerically unstable for high dimensions and large $\kappa$**: The paper uses approximation formulas, but extremely sharp distributions may still pose numerical issues.
- **Hypersphere assumption unifies all concept semantic volumes**: In reality, different subdomains may require different curvatures; extension to mixed-curvature (sphere × hyperbolic) representations is possible.
- **Few multimodal hierarchy experiments**: Only one multimodal benchmark is tested; further validation is needed for truly large-scale image-text taxonomies.
- **Code appears unavailable**: Reproducibility (especially SVGD kernel temperature and orbital potential specifics) depends on the appendix, left for community verification.

## Related Work & Insights
- **vs Poincaré / HyperExpan**: Hyperbolic embeddings use exponential volume growth to alleviate tree crowding; Polaris explicitly encodes "depth" in orbital potential rather than relying on implicit geometric encoding, avoiding hyperbolic space numerical difficulties.
- **vs ConE / Box / Gumbel Box**: Container embeddings use hard partial-order constraints (child must lie in parent cone/box); Polaris softens this via vMF asymmetric KL, making it more robust to noise and outperforming all single-parent baselines in experiments.
- **vs HAKE / Polar coordinate methods**: HAKE uses norm for hierarchy depth but couples it with angle; Polaris strictly fixes norm to 1, with all depth information in orbital potential, avoiding norm/angle coupling optimization issues.
- **vs TaxoExpan / STEAM**: These are Euclidean GNN baselines; the strongest, STEAM, achieves R@1 = 34.8, MR = 31.7 on Science, while Polaris leads by a large margin; the gain comes from the combination of "spherical geometry + probabilistic asymmetry."
- **Insights**: Transferring the "geometry-consistent encoder + flow-based regularizer + asymmetric probabilistic loss" trio to entity linking, medical diagnosis ontologies, and shop catalogs is a natural direction; especially for medical imaging ontologies (e.g., SNOMED CT) with deep hierarchies, the vMF "parent broader" constraint is highly suitable.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of polar spherical embedding + SVGD anti-equator + vMF asymmetric KL is new; while each component has prior work, their integration elegantly solves the three longstanding issues of "unstable angle optimization, high-dimensional equatorial concentration, and softening partial order."
- Experimental Thoroughness: ⭐⭐⭐⭐ Three types of hierarchy (tree / DAG / multimodal) + 14 baselines + five seeds, ablation individually verifies SVGD / vMF / orbital components; the only shortcoming is the lack of large-scale industrial taxonomy case studies.
- Writing Quality: ⭐⭐⭐⭐ Geometric motivation is clearly explained (especially tangent space + exponential mapping vs. mod hack), Theorems 2.2-2.4 elevate the need for SVGD to a theoretical level; formulas are dense and require patience, but the paper is self-contained.
- Value: ⭐⭐⭐⭐ For the taxonomy expansion community, this is a methodological consolidation: making "distributional representation + manifold consistency + global regularization" into a reusable template, with orbital retrieval providing an engineering speedup.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Coupled Training with Privileged Information and Unlabeled Data](coupled_training_with_privileged_information_and_unlabeled_data.md)
- [\[ICML 2026\] New Bounds for Kernel Sums via Fast Spherical Embeddings](new_bounds_for_kernel_sums_via_fast_spherical_embeddings.md)
- [\[AAAI 2026\] Forget Less by Learning from Parents Through Hierarchical Relationships](../../AAAI2026/others/forget_less_by_learning_from_parents_through_hierarchical_relationships.md)
- [\[AAAI 2026\] On the Variability of Concept Activation Vectors](../../AAAI2026/others/on_the_variability_of_concept_activation_vectors.md)
- [\[ICLR 2026\] HEEGNet: Hyperbolic Embeddings for EEG](../../ICLR2026/others/heegnet_hyperbolic_embeddings_for_eeg.md)

</div>

<!-- RELATED:END -->
