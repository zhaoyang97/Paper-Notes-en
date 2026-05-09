---
title: >-
  [Paper Note] Domain Expansion: A Latent Space Construction Framework for Multi-Task Learning
description: >-
  [ICLR 2026][Robotics][multi-task learning] This paper proposes the Domain Expansion framework, which restructures the latent space into mutually orthogonal subspaces via Orthogonal Pooling, structurally preventing gradient conflicts and representation collapse in multi-objective training, and enabling interpretable, composable concept algebra.
tags:
  - ICLR 2026
  - Robotics
  - multi-task learning
  - orthogonal pooling
  - latent space construction
  - representation collapse
  - composable representations
date: 2026-05-08
content_hash: e5f91b50da78bdae
---

# Domain Expansion: A Latent Space Construction Framework for Multi-Task Learning

**Conference**: ICLR 2026
**arXiv**: [2601.20069](https://arxiv.org/abs/2601.20069)
**Code**: To be confirmed
**Area**: Representation Learning / Multi-Task Learning
**Keywords**: multi-task learning, orthogonal pooling, latent space construction, representation collapse, composable representations

## TL;DR
This paper proposes the Domain Expansion framework, which restructures the latent space into mutually orthogonal subspaces via Orthogonal Pooling, structurally preventing gradient conflicts and representation collapse in multi-objective training, and enabling interpretable, composable concept algebra.

## Background & Motivation

**Background**: Multi-task learning (MTL) aims to satisfy multiple learning objectives (e.g., classification + regression) with a single network, but conflicting gradients from competing objectives pull shared representations in opposing directions, leading to representation degradation. The authors formalize this as *latent representation collapse*—the feature space is compressed into a small compromise region that is suboptimal for all objectives.

**Limitations of Prior Work**: (a) Gradient-level MTL methods (GradNorm, PCGrad, Nash-MTL, CAGrad, MGDA, etc.) are inherently *reactive*—they reconcile conflicts only after conflicting gradients have already been produced, requiring additional gradient operations at every step. (b) These methods do not alter the structure of the latent space itself; the learned representations remain entangled and uninterpretable. A representative case: under Objective Set 2, baselines such as Nash-MTL achieve high classification accuracy but near-zero V-score, indicating that the model learns a shortcut mapping rather than meaningful internal representations.

**Key Challenge**: How can a representation space be designed such that multiple learning objectives naturally do not interfere during training—rather than requiring reconciliation after interference occurs?

**Goal**: Eliminate inter-task interference at the level of representational space design, constructing a *proactive* latent space that intrinsically supports multi-objective learning.

**Key Insight**: Analogous to anamorphic art (e.g., a pattern on a cylinder that reveals different shapes when viewed from different angles), a high-dimensional latent vector can simultaneously encode multiple independent concepts through projections along different orthogonal directions.

**Core Idea**: Partition the latent space into non-interfering concept subspaces using orthogonal eigenvector bases obtained from feature decomposition, such that gradients flow within subspaces and are zero across subspaces.

## Method

### Overall Architecture
Domain Expansion is a three-step framework executed dynamically during training, with the orthogonal basis updated once per epoch.

### Key Designs

1. **Find Principal Axes**:

   - Compute the empirical mean $\mu$ and covariance matrix $\Sigma$ of the latent features at the current epoch.
   - Perform eigendecomposition of $\Sigma$ to obtain orthogonal eigenvector bases $V = [v_0, v_1, \ldots, v_{D-1}]$.
   - Apply the Hungarian algorithm across epochs to align eigenvectors and resolve instability in early training.

2. **Define Orthogonal Domain**:

   - Select the top $M$ eigenvectors corresponding to the largest eigenvalues to form the "domain" $V_M$.
   - Each eigenvector $v_m$ is exclusively assigned to one target concept $\mathcal{C}_m$ (e.g., azimuth, category, ID).
   - Projection operator: $\text{Proj}_m = v_m v_m^\top$

3. **Orthogonal Pooling**:

   - Project latent features $f$ onto each orthogonal subspace: $f^{\text{proj},m} = \text{Proj}_m(f - \mu)$
   - Loss gradients across subspaces are naturally decoupled—learning concept A cannot affect the subspace of concept B.
   - Total loss = weighted sum of independent losses over each subspace: $\mathcal{L}_{\text{total}} = \sum_m w_m \cdot \mathcal{L}_m(\mathcal{F}_m^{\text{proj}}, \mathcal{C}_m)$

### Loss & Training
- Regression concepts (azimuth, elevation, rotation) use Rank-N-Contrast (RNC) loss (temperature $\tau=2.0$, weight 1.0).
- Classification concepts (category, ID) use a modified SupCon loss (L2 distance replacing inner product, weight 0.02).
- Two-stage training: first train the encoder while dynamically updating the orthogonal basis → freeze the encoder and train a linear decoder.

### Algebraic Properties (Concept Algebra)
- **Concept orthogonality**: $\mathcal{F}_0^{\text{proj}} \perp \mathcal{F}_1^{\text{proj}} \perp \cdots$; modifying one concept does not affect others.
- **Concept composition operator**: $f_j = f_i \pm f_\Delta^{\text{proj},m}$, supporting single-concept manipulation via vector arithmetic.
- **Reconstructability**: $f_i = \mu + \sum_m f_i^{\text{proj},m}$; the full representation can be reconstructed from subspace components.

## Key Experimental Results

### Main Results: ShapeNet (5 objectives: azimuth / elevation / rotation + category / ID)

| Method | Spearman (az/el/rot) ↑ | V-score (cat/id) ↑ | Composition Similarity ↑ |
|--------|------------------------|-------------------|--------------------------|
| Baseline | 0.41/0.34/0.35 | 0.16/0.14 | 0.22 |
| FAMO | 0.49/0.41/0.42 | 0.00/0.00 | 0.28 |
| Nash-MTL | 0.38/0.41/0.42 | 0.00/0.00 | 0.28 |
| IMTL | 0.31/0.16/0.16 | 0.39/0.28 | 0.14 |
| **Ours** | **0.95/0.87/0.85** | **0.99/0.91** | **0.95** |

### Ablation Study & Key Findings

| Finding | Evidence |
|---------|----------|
| Gradient methods learn shortcuts | Under Objective Set 2, Nash-MTL achieves high classification accuracy but V-score = 0 → representation collapse |
| Orthogonal pooling effectively decouples | Spearman improves from 0.41 → 0.95; V-score from 0.16 → 0.99 |
| Concept composition is feasible | Compositional cosine similarity reaches 0.93–0.95, far exceeding baselines at 0.14–0.28 |
| Cross-dataset generalization | Consistently effective on MPIIGaze (gaze estimation) and Rotated MNIST |
| PCA visualization | Baseline space is entangled and disordered; proposed method yields clearly organized latent axes aligned with each concept |

## Highlights & Insights
- The most fundamental conceptual contribution is the **proactive vs. reactive** distinction—rather than reconciling conflicts during optimization, the proposed method eliminates the possibility of conflict at the structural level. This is analogous to "prevention over treatment."
- **Strong interpretability**: Each orthogonal axis directly corresponds to a semantic concept, and PCA visualizations clearly reveal an organized latent space—a rare property in black-box deep learning.
- **Concept algebra**: Vector addition and subtraction enable concept-level manipulation (e.g., "change the pose of this chair"), validating the combinatorial reasoning capacity of the latent space.
- **Extremely lightweight**: Orthogonal pooling requires only PCA and matrix projection, introduces no additional learnable parameters, and incurs virtually zero extra computational cost.
- The **anamorphic art analogy** is highly intuitive—a cylinder appears rectangular from the front and circular from above; orthogonal projection allows a single vector to simultaneously "store" multiple independent perspectives.

## Limitations & Future Work
- **Dimensionality constraint**: Each concept is allocated only a 1-dimensional subspace, which may be insufficient for complex concepts (e.g., texture, composite shapes) requiring higher-dimensional representation.
- **No generative component**: The framework can represent concept combinations such as "chair + boat" but cannot generate corresponding images—integration with VAE/GAN architectures remains unexplored.
- **Limited scale**: Validation is restricted to ShapeNet, MNIST, and MPIIGaze; large-scale real-world multi-task benchmarks (e.g., NYUv2, Cityscapes) are absent.
- **Eigenvector instability**: The orthogonal basis undergoes rapid changes in early training, necessitating Hungarian alignment—instability may be more pronounced in larger models or datasets.
- **Fixed basis count**: $M$ must be specified in advance; adaptive selection mechanisms are not explored.

## Related Work & Insights
- **vs. GradNorm / PCGrad / IMTL**: These methods operate in gradient space (projection, reweighting); the proposed method operates in feature space (projection onto orthogonal subspaces), functioning at a higher level of abstraction.
- **vs. Nash-MTL / FAMO**: These pursue more sophisticated multi-task optimization objectives but do not alter representational structure—the experiments in this paper demonstrate that structural modification is more effective than optimization-level improvements.
- **vs. β-VAE and disentanglement methods**: β-VAE achieves loose disentanglement via KL divergence penalties; the proposed method enforces strict disentanglement through hard orthogonality constraints.
- **vs. contrastive learning**: SupCon and RNC are used as loss functions within each subspace—they are components of the method rather than competing approaches.
- **Insight**: The orthogonal projection paradigm can be generalized to multimodal learning, where representations from different modalities (text / image / audio) are constrained to orthogonal subspaces.

## Rating
- Novelty: ⭐⭐⭐⭐ Orthogonal pooling combined with concept algebra is a creative design
- Experimental Thoroughness: ⭐⭐⭐ Dataset scale is limited; large-scale validation is lacking
- Writing Quality: ⭐⭐⭐⭐ Mathematical definitions are rigorous; the anamorphic art analogy is intuitive
- Value: ⭐⭐⭐⭐ Offers a new perspective on representation learning for MTL

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Resolving Token-Space Gradient Conflicts: Token Space Manipulation for Transformer-Based Multi-Task Learning](../../ICCV2025/robotics/resolving_token-space_gradient_conflicts_token_space_manipulation_for_transforme.md)
- [\[ICLR 2026\] RoboCasa365: A Large-Scale Simulation Framework for Training and Benchmarking Generalist Robots](robocasa365_a_large-scale_simulation_framework_for_training_and_benchmarking_gen.md)
- [\[ACL 2026\] DeCoVec: Building Decoding Space based Task Vector for Large Language Models via In-Context Learning](../../ACL2026/robotics/decovec_building_decoding_space_based_task_vector_for_large_language_models_via_.md)
- [\[ICCV 2025\] Rep-MTL: Unleashing the Power of Representation-Level Task Saliency for Multi-Task Learning](../../ICCV2025/robotics/rep-mtl_unleashing_the_power_of_representation-level_task_saliency_for_multi-tas.md)
- [\[ICCV 2025\] Beyond Losses Reweighting: Empowering Multi-Task Learning via the Generalization Perspective](../../ICCV2025/robotics/beyond_losses_reweighting_empowering_multi-task_learning_via_the_generalization_.md)

</div>

<!-- RELATED:END -->
