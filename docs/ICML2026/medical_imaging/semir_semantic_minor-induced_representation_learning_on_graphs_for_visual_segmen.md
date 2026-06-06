---
title: >-
  [Paper Note] SEMIR: Semantic Minor-Induced Representation Learning on Graphs for Visual Segmentation
description: >-
  [ICML 2026][Medical Imaging][Graph minor] SEMIR treats the voxel grid as a parent graph $G$ and compresses it into a "boundary-aligned" graph minor $H$ through parameterized edge contraction, node deletion…
tags:
  - "ICML 2026"
  - "Medical Imaging"
  - "Graph minor"
  - "few-shot boundary alignment"
  - "superpixels"
  - "tumor segmentation"
  - "exact lifting"
date: 2026-05-08
content_hash: 9ab849bf8d603068
---

# SEMIR: Semantic Minor-Induced Representation Learning on Graphs for Visual Segmentation

**Conference**: ICML 2026  
**arXiv**: [2605.12389](https://arxiv.org/abs/2605.12389)  
**Code**: None (Repository link not provided in the paper)  
**Area**: Medical Image Segmentation / Graph Neural Networks  
**Keywords**: Graph minor, few-shot boundary alignment, superpixels, tumor segmentation, exact lifting

## TL;DR
SEMIR treats the voxel grid as a parent graph $G$ and compresses it into a "boundary-aligned" graph minor $H$ through parameterized edge contraction, node deletion, and edge deletion (reducing node count from $\sim10^7$ to $\sim10^3$). It utilizes 5–20 few-shot labels and black-box optimization to maximize boundary Dice for $\Theta$. After performing supernode classification on the minor using a GNN, it maps labels back to the original grid via an exact lifting bijection—consistently outperforming nnU-Net on minority class Dice across BraTS, KiTS, and LiTS tumor segmentation tasks using only a 16GB T4 GPU.

## Background & Motivation

**Background**: The mainstream for medical voxel segmentation involves dense convolutional or Transformer architectures like U-Net and Swin-UNETR, which perform voxel-wise softmax predictions on the original grid. To maintain computational feasibility on $10^8$ voxels, these methods rely on patch-wise processing, downsampling, or pre-compression via manual superpixels (e.g., SLIC, Felzenszwalb).

**Limitations of Prior Work**: (1) The computational cost of dense inference is tied to the number of voxels rather than anatomical complexity—tumors occupying $< 1\%$ of the volume still incur 100% of the computational cost. (2) Severe class imbalance dilates gradient signals for minority classes (e.g., enhancing tumor). (3) Existing superpixel or pooling methods are task-agnostic, grouping by low-level intensity, which misaligns with semantic boundaries and introduces artifacts during interpolation when mapping back to voxels.

**Key Challenge**: A structural trade-off exists between "computability" and "boundary alignment." Multi-class joint segmentation forces all anatomical structures to compete for the same representation, while spatial scale differences force a precarious balance of loss weights.

**Goal**: (1) Learn a task-adaptive, topology-preserving intermediate graph representation that scales inference cost with semantic boundary complexity rather than voxel count. (2) Support exact lifting with zero boundary artifacts. (3) Achieve high-quality representation learning with extremely few samples (5–20).

**Key Insight**: Graph minor theory provides formal tools—edge contraction naturally induces a surjective partition from parent to child, where each supernode corresponds to a connected subset in the original graph, forming a "strictly non-overlapping" partition. Robertson-Seymour theorem ensures polynomial measurability.

**Core Idea**: Treat graph compression itself as a few-shot learnable representation space. Parameters $\Theta=\{\psi, \alpha, \beta\}$ control contraction, edge-deletion, and node-deletion. Black-box optimization is used to maximize boundary Dice on few shots, followed by binary GNN classification on the compressed graph and final lifting back to voxels.

## Method

### Overall Architecture
The input volume $I \in \mathbb{R}^{H \times W \times D \times C}$ is encoded as an $N$-connected grid graph $G$, represented as an expanded tensor $T \in \{0,\dots,255\}^{(2H-1)\times(2W-1)\times(2D-1)}$ (even indices encode node states; at least one odd index encodes edge states). The pipeline follows: (i) Construct minor $H = S(T, \Theta)$ using current $\Theta$; (ii) Use ExtraTrees SMBO on few-shot data $\mathcal{D}_{\text{few}}$ to find $\Theta_{\text{opt}} = \arg\min_\Theta \mathbb{E}[1 - \text{DSC}(S_B(T,\Theta), Y_B)]$; (iii) Re-run minor construction with $\Theta_{\text{opt}}$ and extract supernode features $X(H)$ and edge features $F(H)$; (iv) Train a 3-layer GINE (hidden 128, Adam learning rate $10^{-3}$) for binary supernode classification per target structure; (v) Perform Lift using the bijection recorded by $T$ to assign supernode labels directly to voxels with zero interpolation.

### Key Designs

1.  **Parameterized Graph Minor Construction (contraction / edge-deletion / node-deletion)**:
    - **Function**: Compresses the voxel grid into a sparse graph that is boundary-aligned, topology-preserving, and supports exact lifting.
    - **Mechanism**: First, perform seed-driven flood-fill edge contraction—merging adjacent voxel $p$ into seed $s$ if and only if $\|I_p - I_s\|_n \le \psi$ (relative to the seed rather than the current supernode mean to prevent collapsing low-contrast gradients). Second, perform node deletion to remove supernodes where area $a_v$ or mean intensity $\bar{I}_v$ exceeds bounds (parameters $\beta=(\beta_{\min}, \beta_{\max}, m_{\min}, m_{\max})$). Finally, perform edge deletion—if $\|\bar{I}_{v_i}-\bar{I}_{v_j}\|_n > \alpha$, the edge is cut, explicitly defining segmentation boundaries. Lemma 3.1 guarantees each supernode corresponds to a connected subgraph in $G$; Theorem 3.2 guarantees exact bijective lifting.
    - **Design Motivation**: Seed-anchored contraction preserves low-contrast gradients as chains of adjacent supernodes. Edge deletion turns strong gradients into explicit cuts. Node deletion filters acquisition noise. These operators preserve topology while allowing task-driven parameter tuning.

2.  **Few-shot Black-box Optimization of $\Theta$ (boundary-aligned representation learning)**:
    - **Function**: Replaces manual superpixel threshold tuning with data-driven boundary alignment learning.
    - **Mechanism**: Modeling the search for $\Theta$ as minimizing binary boundary Dice loss $L(\hat{Y}_B, Y_B) = 1 - \frac{2|\hat{Y}_B \cap Y_B|}{|\hat{Y}_B| + |Y_B|}$ using ExtraTrees surrogate-based SMBO on 5–20 annotations. $Y_B$ is derived from task-specific semantic boundary maps, independent of specific class IDs.
    - **Design Motivation**: $\Theta$ does not parameterize a fixed model but rather a family of graph homomorphisms $\pi_\Theta: G \to H_\Theta$. The few-shot optimization searches for the "partition structure" itself. The search space is constrained by physical meaning ($\psi, \alpha, \beta$ are low-dimensional), making 5–20 samples sufficient.

3.  **Scale/Rotation Invariant Supernode/Edge Descriptors + GNN Inference**:
    - **Function**: Enables the GNN to perform robust predictions on compressed graphs derived from anisotropic medical volumes.
    - **Mechanism**: Supernode features include area $a_u$, intensity standard deviation $\sigma_u$, intensity covariance $\Sigma_u$, principal axis direction $d_u$ (eigenvector of spatial covariance), elongation $\text{elong}_u=\sqrt{(\lambda_{u,1}+\varepsilon)/(\lambda_{u,2}+\varepsilon)}$, boundary length $b_u$, and 3D compactness $\text{comp}_u = 36\pi a_u^2/(b_u^3+\varepsilon)$. Edges use log-ratios between adjacent supernodes for scale-invariant relative differences. A 3-layer GINE is trained with hidden size 128 and early stopping on validation Dice.
    - **Design Motivation**: Absolute geometry is unreliable due to anisotropic voxel spacing in CT/MRI. Log-ratios and covariance eigenvectors provide scale and rotation invariance. The combination of geometric features characterizes differences like "vessel-like thin structures" vs. "tumor-like masses."

### Loss & Training
The minor optimization phase uses black-box SMBO without differentiable gradients. The GNN phase uses standard voxel-level Dice and BCE (computed after lifting). Each target structure (ET, TC, tumor, liver) is trained as an independent binary classification model. Multi-class results are integrated via separate minor/GNN construction and merged using confidence-weighted voting or energy minimization, effectively eliminating the imbalance problem by design.

## Key Experimental Results

### Main Results (Comparison with nnU-Net, binary target-vs-rest)

| Dataset | Target | nnU-Net DSC | SEMIR DSC | Training Time |
| :--- | :--- | :--- | :--- | :--- |
| BraTS | ET | 0.812 | **0.894 ± 0.006** | 43 h vs 2.5 h (T4) |
| BraTS | TC | 0.829 | **0.941 ± 0.002** | 39 h vs 1.6 h (T4) |
| KiTS | T | 0.720 | **0.819 ± 0.006** | 19 h vs 0.8 h (T4) |
| LiTS | T | 0.733 | **0.891 ± 0.007** | 11 h vs 0.6 h (T4) |

Contextual comparison with published SOTA (minority class Dice): BraTS ET 0.894 is competitive with GTMamba (0.884); KiTS T 0.819 significantly outperforms ConvOccNet (0.693) and Swin UNETR (0.343); LiTS T 0.891 is higher than most published baselines.

### Ablation Study

BraTS ET / NWPU VHR-10 IoU:

| Ablation | BraTS ET | NWPU VHR-10 | Description |
| :--- | :--- | :--- | :--- |
| Full SEMIR | 0.894 | 0.862 | Complete method |
| w/o edge contraction | 0.441 | 0.408 | Minor degrades to voxel graph; fragmentation -51% |
| w/o edge deletion | 0.719 | 0.681 | No explicit boundaries; supernodes cross semantic edges |
| w/o node deletion | 0.812 | 0.749 | Noisy supernodes not pruned |
| Learned $\Theta$ (5-shot) | 0.894 | 0.789 | 5 samples are sufficient |
| Fixed manual $\Theta$ | 0.837 | 0.763 | Few-shot learned partition is superior |
| w/o edge features | 0.725 | 0.741 | Lack of relative geometric signals |
| w/o spatial features | 0.661 | 0.629 | Compactness / elongation are critical |

### Key Findings
- Graph minor reduces inference nodes from $\sim10^7$ to $\sim10^3$; complexity scales with "semantic boundary complexity" rather than "voxel resolution." This explains why SEMIR on a 16GB T4 outperforms nnU-Net, which requires A100s and is 20×–60× slower.
- Optimized $\Theta$ from 5-shot samples outperforms the best manual tuning, proving that the hypothesis space for $\Theta$ is small and well-constrained. This is the key benefit of "learning structure" over "learning hyperparameters."
- On the non-medical NWPU aerial dataset, small-object IoU remains high at 0.862 (dropping to 0.408 without edge contraction), indicating the general applicability of graph minor construction for "small object + high resolution" vision problems.

## Highlights & Insights
- Applying graph minor theory to segmentation provides a rigorous algebraic foundation for topology preservation and bijective lifting, solving the long-standing issue of interpolation artifacts in superpixel methods.
- The shift from optimizing model weights to optimizing the "inference space" is a profound perspective: class imbalance is addressed via per-target binary decomposition, and task adaptation is shifted to the "partition" layer rather than the "network weights" layer.
- Engineering implementation using expanded tensor $T$ (single-byte states) and a Rust-backend flood-fill allows minor construction in under a second on the CPU. This asynchronous CPU-GPU design, decoupling compute-intensive and data-intensive tasks, is applicable to other vision tasks requiring sparsification.

## Limitations & Future Work
- Boundary sensitivity: In regions with low contrast or poor multi-modal fusion, incorrect $\alpha$ values can lead to misaligned minor boundaries. If the few-shot set lacks rare pathological morphologies, generalization may be limited.
- Current minor construction and downstream GNN are modular and decoupled; end-to-end joint optimization has not yet been implemented. Pseudo-random traversal also introduces slight run-to-run variance.
- Evaluation is limited to CT/MRI volumes. Modalities with significantly different color/noise distributions (e.g., ultrasound, pathology) remain unverified. Node deletion's "discarding outliers" might accidentally remove rare pathologies, requiring clinical oversight during deployment.

## Related Work & Insights
- **vs nnU-Net**: Dense voxel inference with multi-class joint learning is severely affected by class imbalance in minority classes. SEMIR mitigates this via per-target binary minors, gaining +8.2 (BraTS ET), +9.9 (KiTS T), and +15.8 (LiTS T) points.
- **vs SLIC / Felzenszwalb**: These are task-agnostic, manually tuned, and require interpolation for lifting. SEMIR provides task-aware black-box optimization and exact lifting.
- **vs DiffPool / MinCutPool**: These methods learn soft clustering and lack lift guarantees. SEMIR ensures robustness via hard partitioning based on graph homomorphisms.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First use of graph minor + few-shot boundary alignment for inference space representation learning.
- Experimental Thoroughness: ⭐⭐⭐⭐ 3 medical datasets + nnU-Net comparison + NWPU cross-domain ablation + full ablation study.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear narrative progression from density vs. structure to graph minor theory, operators, and formal proofs.
- Value: ⭐⭐⭐⭐⭐ Enables a 16GB T4 to outperform nnU-Net on A100s; a game-changer for resource-constrained clinical deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] SynBrain: Enhancing Visual-to-fMRI Synthesis via Probabilistic Representation Learning](../../NeurIPS2025/medical_imaging/synbrain_enhancing_visual-to-fmri_synthesis_via_probabilistic_representation_lea.md)
- [\[ICLR 2026\] SEED: Towards More Accurate Semantic Evaluation for Visual Brain Decoding](../../ICLR2026/medical_imaging/seed_towards_more_accurate_semantic_evaluation_for_visual_brain_decoding.md)
- [\[ICML 2026\] MedCRP-CL: Continual Medical Image Segmentation via Bayesian Nonparametric Semantic Modality Discovery](medcrp-cl_continual_medical_image_segmentation_via_bayesian_nonparametric_semant.md)
- [\[ICML 2026\] iLoRA: Bayesian Low-Rank Adaptation with Latent Interaction Graphs for Microbiome Diagnosis](ilora_bayesian_low-rank_adaptation_with_latent_interaction_graphs_for_microbiome.md)
- [\[CVPR 2026\] Semantic Class Distribution Learning for Debiasing Semi-Supervised Medical Image Segmentation](../../CVPR2026/medical_imaging/semantic_class_distribution_learning_for_debiasing.md)

</div>

<!-- RELATED:END -->
