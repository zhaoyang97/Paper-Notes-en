---
title: >-
  [Paper Note] SAFAG: Generalizable Actionable Part Pose Estimation Without Symmetry Annotation
description: >-
  [ICML 2026][Robotics][GAParts] SAFAG decomposes GAPart 6D pose estimation into a two-stage framework consisting of "candidate quaternion generation + tangent space refinement." By utilizing adaptive probability distribut…
tags:
  - "ICML 2026"
  - "Robotics"
  - "GAParts"
  - "6D Pose"
  - "Symmetry Self-supervision"
  - "Quaternion Refinement"
  - "Robotic Manipulation"
date: 2026-05-08
content_hash: 0eb471c4e448b0ee
---

# SAFAG: Generalizable Actionable Part Pose Estimation Without Symmetry Annotation

**Conference**: ICML 2026  
**arXiv**: [2605.17033](https://arxiv.org/abs/2605.17033)  
**Code**: TBD  
**Area**: Robotics / Embodied AI / 6D Pose Estimation  
**Keywords**: GAParts, 6D Pose, Symmetry Self-supervision, Quaternion Refinement, Robotic Manipulation

## TL;DR
SAFAG decomposes GAPart 6D pose estimation into a two-stage framework consisting of "candidate quaternion generation + tangent space refinement." By utilizing adaptive probability distributions on the $x, y, z$ axes to implicitly learn symmetry axes/planes, it reduces the rotation error of cross-category actionable parts from 5.51° to 3.23° without any symmetry annotations.

## Background & Motivation

**Background**: Cross-category object manipulation in embodied AI relies on high-quality part-level 6D pose perception. Geng et al. proposed GAParts (Generalizable and Actionable Parts, e.g., sliding drawers, hinged doors, buttons), shifting the focus of pose estimation from holistic objects to 9 categories of interactive parts. This allows robots to execute manipulation strategies "by part." Subsequent works like GAPartNet, GASEM, DFGAP, GenPose++, and RFMPose follow this line of research.

**Limitations of Prior Work**: GAParts exhibit richer symmetries than whole objects (e.g., circular knobs are 360° equivalent around an axis, drawer lids are 180° equivalent). Existing methods suffer from two main issues: (1) **Multi-solution problem**: GAPartNet uses NPCS to force a unique solution, compressing the equivalence set into a single label, which damages accuracy; (2) **Strong annotation dependency**: GASEM, DFGAP, and others design symmetry-aware losses but require pre-annotated symmetry axes or planes, which are expensive and often unavailable in real-world scenarios.

**Key Challenge**: To correctly handle the "one-to-many" mapping caused by symmetry, one must know the symmetry axis; to know the symmetry axis, dense annotation is required; to remove annotation, one falls back to the ill-posed "one-to-many" mapping under unsupervised settings.

**Goal**: Perform 6 DoF pose estimation for GAParts with zero symmetry annotation, covering both rotational and reflectional symmetries.

**Key Insight**: The authors treat the symmetry axis/plane as a discrete probability distribution over the $x, y, z$ axes (mixture weights $\pi_x, \pi_y, \pi_z$). The network learns this distribution via self-supervision from point cloud reconstruction consistency. The pose itself is handled via quaternions (compact, singularity-free, non-decoupled) through a "candidate → tangent space refinement → aggregation" pipeline.

**Core Idea**: Reformulate "finding symmetry axes" as "estimating probability distributions on three axes," using Chamfer mirror consistency as a self-supervised signal, thereby removing "symmetry annotation" entirely from the training pipeline.

## Method

### Overall Architecture
The input is a partial point cloud of the part $P \in \mathbb{R}^{N \times 3}$ ($N=1024$), and the output is the 6 DoF pose (rotation $R \in SO(3)$ + translation $t \in \mathbb{R}^3$) for each GAPart. The pipeline consists of four modules: (1) **HyperS3 backbone**: A modification of 3D-GCN into a feature extractor friendly to the $S^3$ manifold, outputting point cloud features $\mathcal{F}^{pc}$; (2) **Candidate Generation**: Concatenates $\mathcal{F}^{pc}$ with $K=64$ random noise vectors to generate 64 quaternion candidates in parallel via a shared CNN; (3) **Tangent Space Refinement**: A candidates encoder first encodes distribution statistics (mean, second-order moments, eigenvalues/vectors) of the 64 candidates into $\mathcal{F}^{embedding}$. A CNN then predicts offsets $\Delta q_i$ in the $S^3$ tangent space for each candidate, and a final linear layer aggregates them into $q^{final}$; (4) **Adaptive Symmetry Network**: Uses the axis-angle representation of $q^{final}$ and $\mathcal{F}^{pc}$ to predict three-axis probabilities $\pi_{x,y,z}$. Rotational symmetry is obtained by weighted averaging, while mirror symmetry uses Chamfer consistency to filter three candidate planes. Finally, the predicted symmetry structure constructs a ground truth equivalence set to supervise $q^{final}$.

### Key Designs

1.  **HyperS3 Convolutional Layer (Rotation-aware Local Coordinates)**:
    -   **Function**: Adds a layer to 3D-GCN that explicitly constructs $SO(3)$ local bases, ensuring features are stable relative to rotation, specifically serving quaternion regression.
    -   **Mechanism**: Finds KNN ($M=8$) for each point $p_i$ and calculates the local covariance $S_i = \frac{1}{M}\sum_{j} R_{ij} R_{ij}^\top$. Uses one-step power iteration to extract the principal direction $e_{1,i}=\frac{S_i v_0}{\|S_i v_0\|}$, then computes $e_{2,i}, e_{3,i}$ via cross-product to form an orthogonal basis $E_i$. Neighbor vectors are projected into both the local coordinate system (rotation-sensitive branch $\mathcal{F}^{align}$) and the Euclidean coordinate system (geometric structure branch $\mathcal{F}^{euclid}$). Adaptive fusion is performed as $\mathcal{F}_i = \alpha_i \mathcal{F}^{align} + (1-\alpha_i)\mathcal{F}^{euclid}$ using confidence weights $\alpha_i$ based on the trace $\mathrm{tr}(S_i)$ and anisotropy $a_i=\|S_i - S_i^{iso}\|_F^2$.
    -   **Design Motivation**: Previous backbones lacked sufficient rotation sensitivity for $S^3$ quaternion regression. Explicitly embedding "rotation awareness" into the convolutional layer through local bases prevents the downstream candidate generation from having to learn this invariance from scratch.

2.  **Two-stage Quaternion Regression with Candidates Encoder**:
    -   **Function**: Replaces single-solution regression with "initial distribution of $K=64$ candidates followed by tangent space refinement," matching the multi-solution structure caused by symmetry.
    -   **Mechanism**: In the generation stage, noise $z_i$ is concatenated with $\mathcal{F}^{pc}$ to produce 64 candidates $\{q_i\}$. Before refinement, statistics are computed: the mean $q_{mean}$ after sign alignment, residuals $r_i = q_{mean}^{-1}\otimes q_i$ mapped to the tangent space as offsets $\delta_i$, average cosine similarity $\bar{\mu} = \frac{1}{K}\sum_i |q_i^\top q_{mean}|^2$, and the top three eigenvalues/vectors $\{\lambda_j, v_j\}_{j=1}^3$ of the second-order moment matrix. These are fed into the candidates encoder. After fusing the resulting $\mathcal{F}^{embedding}$ with $\mathcal{F}^{pc}$, the CNN predicts offsets $\Delta q_i$ in the tangent space. The 64 corrected candidates $\Delta q_i \otimes q_i$ are aggregated into $q^{final}$ via a linear layer.
    -   **Design Motivation**: Direct regression of rotation matrices leads to $SO(3)$ discontinuities. Quaternions are compact and singularity-free. Updating in the tangent space represents first-order change on the manifold with clear geometric meaning. Encoding second-order moments of the candidate distribution ensures the refinement is aware of the global distribution shape.

3.  **Adaptive Symmetry Perception (Zero-label Probability Modeling)**:
    -   **Function**: Estimates symmetry structures without any ground truth for symmetry axes/planes, then uses this to construct equivalence sets for supervising the final quaternion.
    -   **Mechanism**: Models symmetry inference as a discrete mixture distribution $\pi_x, \pi_y, \pi_z$ on the $x, y, z$ axes. **Rotational Symmetry**: Concatenates the axis-angle representation of $q^{final}$ ($\mathcal{F}^{rot}$) and $\mathcal{F}^{pc}$ to predict three-axis probabilities; the symmetry axis is $n = \pi_x n_x + \pi_y n_y + \pi_z n_z$. **Mirror Symmetry**: Since the number and position of planes are unknown, $n$ is treated as the primary normal, with an additional predicted orthogonal normal $n'$, and $n''$ obtained via cross-product. For each candidate normal $u_j$, the mirrored point cloud $p_i^{(j)\prime} = p_i - 2((p_i-p_c)\cdot u_j) u_j$ is computed. Bidirectional Chamfer distance $\mathcal{L}_{geom}(P, u_j)=\tfrac{1}{2}(d(P,P^{(j)\prime})+d(P^{(j)\prime},P))$ measures mirror consistency, and normalized geometric scores $s_j = \frac{1/(\mathcal{L}_{geom}+\varepsilon)}{\sum_j 1/(\mathcal{L}_{geom}+\varepsilon)+\varepsilon}$ are used to suppress inconsistent planes.
    -   **Design Motivation**: By transforming the "finding symmetry axes" task into "estimating three-axis probabilities," the supervision signal can naturally come from the geometric consistency between the original and mirrored point clouds, eliminating the need for symmetry annotations.

### Loss & Training
The final quaternion is supervised by the "GT equivalence set constructed from the predicted symmetry structure"—the closest member of the equivalence set to the prediction is chosen as the regression target. The symmetry network is self-supervised by the Chamfer mirror consistency $\mathcal{L}_{geom}$. Symmetry types (rotational/reflectional) are provided as prior categories, but specific axis/plane parameters are learned entirely by the network.

## Key Experimental Results

### Main Results (GAPartNet, Average across 9 Categories)

| Setting | Method | Rot. (°) ↓ | Trans. (cm) ↓ |
| :--- | :--- | ---: | ---: |
| Seen | GAPartNet | 7.71 | 0.037 |
| Seen | GASEM | 9.11 | 0.036 |
| Seen | GenPose++ | 15.86 | 0.035 |
| Seen | RFMPose | 17.03 | 0.060 |
| Seen | DFGAP | 5.51 | 0.020 |
| Seen | **SAFAG (Ours)** | **3.23** | **0.016** |
| Unseen | GAPartNet | 27.59 | — |
| Unseen | GASEM | 29.45 | — |
| Unseen | GenPose++ | 31.66 | — |
| Unseen | RFMPose | 33.39 | — |

SAFAG reduces rotation error by 41% (5.51° → 3.23°) relative to the strongest baseline DFGAP on the Seen split. Translation error also drops to 0.016 cm. Improvements are most significant in highly symmetric categories such as Sd.Ld (Slider Lid), Hg.Ld (Hinged Lid), and Rd.F.Hl (Round Fixed Handle).

### Ablation Study

| Configuration | Rot. (°) | Description |
| :--- | ---: | :--- |
| Full SAFAG | 3.23 | Full model |
| w/o HyperS3 conv | Increase | Reduced to original 3D-GCN; lower rotation sensitivity |
| w/o Two-stage | Increase | Single-solution regression suffers from $SO(3)$ discontinuity |
| w/o Adaptive Symmetry | Large Increase | Degenerates to single GT supervision; symmetry ambiguity returns |

### Key Findings
-   The adaptive symmetry module contributes the most: without it, symmetric parts (sliding lids, round handles, knobs) degrade significantly, verifying that "probability modeling + Chamfer self-supervision" is key to removing annotations.
-   HyperS3's rotation-aware local basis allows the backbone to output features that are more stable for rotation.
-   $K=64$ candidates balanced with tangent space refinement achieve a good trade-off between precision and efficiency.
-   Real-world demos show that SAFAG-perceived GAParts can directly interface with manipulation policies to complete grasping tasks.

## Highlights & Insights
-   **Reformulating "Symmetry Annotation" as "Three-axis Probability + Geometric Consistency"**: This approach acknowledges that symmetry is inherently a distribution and uses Chamfer distance as a natural self-supervised signal. This paradigm of "liberating priors from labels" is transferable to other tasks requiring symmetry/equivalence information (e.g., NOCS category-level pose, symmetric objects in SLAM).
-   **Three-stage Quaternion Regression (Candidate → Refinement → Aggregation)**: This explicit candidate deployment + manifold refinement is lighter and more accurate than generative (GenPose++) or flow-matching (RFMPose) methods, suggesting it is a high-efficiency paradigm for $S^3$.
-   **Candidates Encoder for Second-order Moments**: Using the mean, eigenvalues, and eigenvectors of the candidate distribution as encoder inputs allows the refinement stage to perceive the "shape" of the candidate swarm, avoiding instability from isolated point-wise corrections.

## Limitations & Future Work
-   Symmetry types (rotational vs. reflectional) are required as prior categories; a more thorough version would automatically determine the symmetry type.
-   Main experiments are limited to GAPartNet; robustness in real-world scenarios with severe occlusion/noise lack extensive quantitative evaluation.
-   Fixed candidate count $K=64$; whether adaptive $K$ is needed for extreme symmetries (e.g., perfectly axis-symmetric knobs) is not discussed.
-   Future work could incorporate joint motion information for temporal estimation to resolve single-frame ambiguities.

## Related Work & Insights
-   **vs GAPartNet (Geng et al., 2023)**: GAPartNet forces convergence to a single solution via NPCS, losing equivalence information; SAFAG explicitly models this to outperform it significantly (Seen 7.71° → 3.23°).
-   **vs GASEM / DFGAP**: Both require symmetry axis/plane annotations; SAFAG removes these entirely via Chamfer self-supervision while still exceeding DFGAP's performance by 41%.
-   **vs GenPose++ / RFMPose**: These model multi-solutions as distribution sampling; while elegant, they require symmetry labels and high training costs. SAFAG achieves lower error with candidate refinement and no labels.

## Rating
-   Novelty: ⭐⭐⭐⭐ "Symmetry as probability distribution + Chamfer self-supervision" elegantly removes annotation dependency.
-   Experimental Thoroughness: ⭐⭐⭐⭐ Full coverage of GAPartNet categories, though lacking extensive quantitative real-world scenarios.
-   Writing Quality: ⭐⭐⭐⭐ Clear decomposition of the four modules; well-explained geometric motivations.
-   Value: ⭐⭐⭐⭐ Directly serves cross-category manipulation in embodied AI; the label-free setting has high deployment value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] StableVLA: Towards Robust Vision-Language-Action Models without Extra Data](stablevla_towards_robust_vision-language-action_models_without_extra_data.md)
- [\[AAAI 2026\] Scalable Multi-Objective and Meta Reinforcement Learning via Gradient Estimation](../../AAAI2026/robotics/scalable_multi-objective_and_meta_reinforcement_learning_via_gradient_estimation.md)
- [\[ICLR 2026\] Partially Equivariant Reinforcement Learning in Symmetry-Breaking Environments](../../ICLR2026/robotics/partially_equivariant_reinforcement_learning_in_symmetry-breaking_environments.md)
- [\[NeurIPS 2025\] Time Reversal Symmetry for Efficient Robotic Manipulations in Deep Reinforcement Learning](../../NeurIPS2025/robotics/time_reversal_symmetry_for_efficient_robotic_manipulations_in_deep_reinforcement.md)
- [\[AAAI 2026\] Coordinated Humanoid Robot Locomotion with Symmetry Equivariant Reinforcement Learning Policy](../../AAAI2026/robotics/coordinated_humanoid_robot_locomotion_with_symmetry_equivariant_reinforcement_le.md)

</div>

<!-- RELATED:END -->
