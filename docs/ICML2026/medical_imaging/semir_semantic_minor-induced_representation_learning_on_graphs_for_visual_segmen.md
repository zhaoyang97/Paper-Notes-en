---
title: >-
  [Paper Note] SEMIR: Semantic Minor-Induced Representation Learning on Graphs for Visual Segmentation
description: >-
  [ICML 2026][Medical Imaging][graph minor] SEMIR treats the voxel grid as a parent graph $G$, compresses it into a "boundary-aligned" graph minor $H$ (reducing node count from $\sim10^7$ to $\sim10^3$) via parameterized e…
tags:
  - "ICML 2026"
  - "Medical Imaging"
  - "graph minor"
  - "few-shot boundary alignment"
  - "superpixel"
  - "tumor segmentation"
  - "exact lifting"
date: 2026-05-08
content_hash: 36fd7eceda272ed1
---

# SEMIR: Semantic Minor-Induced Representation Learning on Graphs for Visual Segmentation

**Conference**: ICML 2026  
**arXiv**: [2605.12389](https://arxiv.org/abs/2605.12389)  
**Code**: None (no repository link provided)  
**Area**: Medical Image Segmentation / Graph Neural Networks  
**Keywords**: graph minor, few-shot boundary alignment, superpixel, tumor segmentation, exact lifting

## TL;DR
SEMIR treats the voxel grid as a parent graph $G$, compresses it into a "boundary-aligned" graph minor $H$ (reducing node count from $\sim10^7$ to $\sim10^3$) via parameterized edge contraction, node deletion, and edge deletion. Using only 5–20 few-shot samples, it black-box optimizes $\Theta$ to maximize boundary Dice, then applies a GNN for supernode classification on the minor, and finally performs exact lifting via a bijection between the minor and the voxel grid. On the BraTS / KiTS / LiTS tumor segmentation tasks, SEMIR consistently outperforms nnU-Net on minority class Dice, requiring only a 16GB T4 GPU.

## Background & Motivation

**Background**: Mainstream medical voxel image segmentation relies on dense convolutional or Transformer architectures like U-Net / Swin-UNETR, predicting voxel-wise softmax on the original grid. To handle $10^8$-scale voxels, approaches include patch slicing, downsampling, or manual superpixel-based pre-compression (SLIC, Felzenszwalb).

**Limitations of Prior Work**: (1) The computational cost of dense inference scales with the number of voxels, not with anatomical complexity—tumors occupy <1% of the volume but incur 100% of the computation; (2) Severe class imbalance dilutes gradient signals for minority classes (tumor, enhancing tumor); (3) Existing superpixel/pooling methods are "task-agnostic," grouping by low-level intensity and misaligned with semantic boundaries, and require interpolation when mapping predictions back to voxels, introducing boundary artifacts.

**Key Challenge**: There is a structural trade-off between "computability" and "boundary alignment." Multi-class joint segmentation forces all structures to compete for the same representation, while spatial scale differences require delicate balancing of loss weights.

**Goal**: (1) Learn a "task-adaptive, topology-preserving" intermediate graph representation, so inference cost scales with semantic boundary complexity rather than voxel count; (2) Must support exact lifting, with zero boundary artifacts; (3) Must be learnable with as few as 5–20 samples.

**Key Insight**: Graph minor theory provides a formal tool—edge contraction naturally induces a surjective parent→child partition, with each supernode corresponding to a connected subset of the original graph, i.e., a strictly non-overlapping partition. Robertson-Seymour established polynomial-time testability.

**Core Idea**: Treat graph compression itself as the representation space to be "few-shot learned"—parameters $\Theta=\{\psi,\alpha,\beta\}$ control contraction, edge-deletion, and node-deletion. Black-box optimization maximizes boundary Dice on few-shot samples, then a binary GNN is trained on the compressed graph, and finally, labels are lifted back to voxels via an exact bijection.

## Method

### Overall Architecture
Input volume $I \in \mathbb{R}^{H \times W \times D \times C}$ is encoded as an $N$-node connected grid graph $G$, represented as an expanded tensor $T \in \{0,\dots,255\}^{(2H-1)\times(2W-1)\times(2D-1)}$ (even indices encode node states, at least one odd index encodes edge states). Pipeline: (i) Use current $\Theta$ to construct the minor $H = S(T, \Theta)$; (ii) On few-shot samples $\mathcal{D}_{\text{few}}$, use ExtraTrees SMBO to search for $\Theta_{\text{opt}} = \arg\min_\Theta \mathbb{E}[1 - \text{DSC}(S_B(T,\Theta), Y_B)]$; (iii) With $\Theta_{\text{opt}}$, reconstruct the minor and extract supernode features $X(H)$ and edge features $F(H)$; (iv) A 3-layer GINE (hidden 128, Adam lr $10^{-3}$) performs supernode binary classification on $H$, with a separate model for each target structure; (v) The bijection recorded in $T$ is used for lifting, directly assigning supernode labels to their constituent voxels, with zero interpolation.

### Key Designs

1. **Parameterized Graph Minor Construction (contraction / edge-deletion / node-deletion)**:

    - **Function**: Compress the voxel grid into a sparse graph that is "boundary-aligned, topology-preserving, and supports exact lifting."
    - **Mechanism**: First, perform seed-driven flood-fill edge contraction—neighboring voxel $p$ is merged into seed $s$ if and only if $\|I_p - I_s\|_n \le \psi$ (relative to the seed, not the current supernode mean, to avoid collapsing low-contrast gradient regions); then perform node deletion to remove supernodes whose area $a_v$ or mean intensity $\bar{I}_v$ is out of bounds (parameter $\beta=(\beta_{\min}, \beta_{\max}, m_{\min}, m_{\max})$); finally, perform edge deletion—if $\|\bar{I}_{v_i}-\bar{I}_{v_j}\|_n > \alpha$, cut the corresponding edge—this directly defines the segmentation boundary. Lemma 3.1 guarantees each supernode corresponds to a connected subgraph in $G$; Theorem 3.2 guarantees the lift is an exact bijection.
    - **Design Motivation**: Seed-anchored contraction preserves low-contrast gradients as "chains of adjacent supernodes" rather than collapsing them; edge deletion turns strong gradients into explicit cuts; node deletion removes small noisy regions, with background defaulting to 0, making it conservative rather than generating false positives. The three operators together preserve topology while allowing task-driven parameter tuning.

2. **Few-Shot Black-Box Optimization of $\Theta$ (boundary-aligned representation learning)**:

    - **Function**: Replace manual superpixel threshold tuning with data-driven boundary alignment learning.
    - **Mechanism**: Model the search for $\Theta$ as minimizing the binary boundary Dice loss $L(\hat{Y}_B, Y_B) = 1 - \frac{2|\hat{Y}_B \cap Y_B|}{|\hat{Y}_B| + |Y_B|}$, using ExtraTrees surrogate-based SMBO on 5–20 labeled samples; $Y_B$ is the task-specific semantic boundary map, independent of class IDs.
    - **Design Motivation**: $\Theta$ does not parameterize a fixed model, but a family of "graph homomorphisms $\pi_\Theta: G \to H_\Theta$"—each $\Theta$ defines a partition; few-shot search targets the "partition structure" itself, so the search space is naturally constrained by physical meaning (ψ, α, β are all low-dimensional and interpretable), making 5–20 samples sufficient.

3. **Scale/Rotation-Invariant Supernode/Edge Descriptors + GNN Inference**:

    - **Function**: Enable the GNN to make robust predictions on compressed graphs for anisotropic medical volumes.
    - **Mechanism**: For each supernode, extract $a_u$ (voxel count), $\sigma_u$ (per-channel intensity std), $\Sigma_u$ (intensity covariance), principal axis $d_u$ (largest eigenvector of spatial covariance), $\text{elong}_u=\sqrt{(\lambda_{u,1}+\varepsilon)/(\lambda_{u,2}+\varepsilon)}$, boundary length $b_u$, 3D compactness $\text{comp}_u = 36\pi a_u^2/(b_u^3+\varepsilon)$; for each edge, compute scale-invariant relative differences between adjacent supernodes using log-ratio. The 3-layer GINE uses hidden 128, Adam $10^{-3}$, and early stopping on validation Dice.
    - **Design Motivation**: CT/MRI voxel spacing is anisotropic, so absolute geometry is unreliable; log-ratio and covariance eigenvectors provide natural scale and rotation invariance; compactness, elongation, and covariance together capture geometric differences such as "vessel-like elongated" vs "tumor-like mass."

### Loss & Training
The minor optimization stage uses black-box SMBO, with no differentiable gradients; the GNN stage uses standard voxel-level Dice/BCE (after lifting for comparison). Each target structure (ET, TC, tumor, liver) is trained as an independent binary model; the full multi-class setup constructs separate minors and GNNs, then merges via confidence-weighted voting or energy minimization—directly eliminating class imbalance "by construction."

## Key Experimental Results

### Main Results (same split, compared to nnU-Net, binary target-vs-rest)

| Dataset | Target | nnU-Net DSC | SEMIR DSC | Training Time |
|---------|--------|-------------|-----------|--------------|
| BraTS   | ET     | 0.812       | **0.894 ± 0.006** | 43 h vs 2.5 h (T4) |
| BraTS   | TC     | 0.829       | **0.941 ± 0.002** | 39 h vs 1.6 h (T4) |
| KiTS    | T      | 0.720       | **0.819 ± 0.006** | 19 h vs 0.8 h (T4) |
| LiTS    | T      | 0.733       | **0.891 ± 0.007** | 11 h vs 0.6 h (T4) |

Comparison with published SOTA (dataset-specific protocols, minority class Dice): BraTS ET 0.894 is close to GTMamba (0.884); KiTS T 0.819 is significantly higher than ConvOccNet (0.693) and Swin UNETR (0.343); LiTS T 0.891 exceeds most published baselines.

### Ablation Study

BraTS ET / NWPU VHR-10 IoU:

| Ablation | BraTS ET | NWPU VHR-10 | Description |
|----------|----------|-------------|-------------|
| Full SEMIR | 0.894 | 0.862 | Complete method |
| Remove edge contraction | 0.441 | 0.408 | Minor degenerates to voxel graph, fragmentation -51% |
| Remove edge deletion | 0.719 | 0.681 | No explicit boundaries, supernodes cross semantic borders |
| Remove node deletion | 0.812 | 0.749 | Noisy supernodes not pruned |
| Learned $\Theta$ (5-shot) | 0.894 | 0.789 | 5 samples suffice |
| Fixed manual $\Theta$ | 0.837 | 0.763 | Few-shot learned partition is better |
| Remove edge features | 0.725 | 0.741 | Lacks relative geometric signals |
| Remove spatial features | 0.661 | 0.629 | Compactness/elongation are key |

### Key Findings
- The minor reduces inference nodes from $\sim10^7$ to $\sim10^3$, so complexity scales with "semantic boundary complexity" rather than "voxel resolution"; this directly explains why SEMIR on a 16GB T4 outperforms nnU-Net, which requires an A100 and is 20×–60× slower.
- Only 5 samples are needed for few-shot $\Theta$ optimization to outperform the best manual tuning, demonstrating that the hypothesis space for $\Theta$ is small and well-constrained by physical priors; this is the key payoff of "learning structure" rather than "learning hyperparameters."
- On non-medical NWPU aerial images, small-object IoU still reaches 0.862 (dropping to 0.408 without edge contraction), indicating that minor construction is generally applicable to "small-object + high-resolution" visual tasks.

## Highlights & Insights
- Introduces the relatively underused graph minor tool from graph theory to segmentation, providing an algebraic foundation for "strict topology preservation + bijective lifting," achieving "no interpolation artifacts"—a longstanding issue with classic superpixel methods.
- The perspective of "optimizing the inference space rather than the segmentation model itself" is profound: fundamentally decomposing class imbalance via per-target binary segmentation, and placing task adaptation at the "partition" level rather than "network weights."
- The expanded tensor $T$ stores node and edge states in single bytes; Rust backend flood-fill constructs the minor on CPU in under a second, cleanly decoupling "compute-intensive" and "data-intensive" stages in engineering. The GPU receives precomputed graph batches—this CPU-GPU asynchronous design can be adapted to other vision tasks requiring sparsification.

## Limitations & Future Work
- Boundary statistics sensitivity: In low-contrast or poorly fused multimodal regions, mischosen $\alpha$ can cause minor boundaries to deviate; if the few-shot set does not cover rare pathologies, generalization is limited.
- Current minor construction and downstream GNN are decoupled (modular), with no end-to-end joint optimization yet; pseudo-random traversal introduces minor run-to-run jitter.
- Evaluation is limited to CT/MRI volumetric images; modalities with very different color/noise distributions (e.g., ultrasound, pathology) are untested; node deletion's "discarding outlier regions" may wrongly remove rare pathologies, requiring physician oversight in clinical deployment.

## Related Work & Insights
- **vs nnU-Net**: Dense voxel inference + multi-class joint segmentation suffers severe class imbalance on minority classes; SEMIR directly resolves this via per-target binary minors, achieving +8.2 (BraTS ET), +9.9 (KiTS T), +15.8 (LiTS T) Dice points.
- **vs SLIC / Felzenszwalb superpixels**: Task-agnostic, manually tuned, require interpolation for lifting; SEMIR offers task-aware black-box optimization + exact lifting, making "grouping" a learnable partition family.
- **vs DiffPool / MinCutPool**: These learn soft clustering but lack lifting guarantees; SEMIR uses hard partitioning via graph homomorphism for invertibility, theoretically more robust.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to use graph minor + few-shot boundary alignment as inference space representation learning.
- Experimental Thoroughness: ⭐⭐⭐⭐ 3 medical datasets + nnU-Net comparison + NWPU cross-domain ablation + full ablation; published SOTA comparisons are protocol-aware and clearly annotated.
- Writing Quality: ⭐⭐⭐⭐⭐ Narrative is clear from "density vs structure" to graph minor theory, specific contraction/deletion operators, and finally Lemma + Theorem.
- Value: ⭐⭐⭐⭐⭐ Enables a 16GB T4 to outperform nnU-Net on A100, truly game-changing for resource-constrained clinical deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] SynBrain: Enhancing Visual-to-fMRI Synthesis via Probabilistic Representation Learning](../../NeurIPS2025/medical_imaging/synbrain_enhancing_visual-to-fmri_synthesis_via_probabilistic_representation_lea.md)
- [\[ICLR 2026\] SEED: Towards More Accurate Semantic Evaluation for Visual Brain Decoding](../../ICLR2026/medical_imaging/seed_towards_more_accurate_semantic_evaluation_for_visual_brain_decoding.md)
- [\[ICML 2026\] MedCRP-CL: Continual Medical Image Segmentation via Bayesian Nonparametric Semantic Modality Discovery](medcrp-cl_continual_medical_image_segmentation_via_bayesian_nonparametric_semant.md)
- [\[CVPR 2026\] Multimodal Classification of Radiation-Induced Contrast Enhancements and Tumor Recurrence Using Deep Learning](../../CVPR2026/medical_imaging/multimodal_classification_of_radiation-induced_contrast_enhancements_and_tumor_r.md)
- [\[CVPR 2026\] Semantic Class Distribution Learning for Debiasing Semi-Supervised Medical Image Segmentation](../../CVPR2026/medical_imaging/semantic_class_distribution_learning_for_debiasing.md)

</div>

<!-- RELATED:END -->
