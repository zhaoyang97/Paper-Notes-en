---
title: >-
  [Paper Note] Learning Multi-Scale Hypergraph for High-Order Brain Connectivity Analysis
description: >-
  [ICML 2026][Medical Imaging][Brain Networks] MuHL decomposes brain ROI features into multi-resolution representations using graph wavelets with learnable scales and dynamically generates soft hyperedges via a "node embed…
tags:
  - "ICML 2026"
  - "Medical Imaging"
  - "Brain Networks"
  - "Hypergraph Learning"
  - "Graph Wavelets"
  - "Neurodegenerative Diseases"
  - "Multi-scale"
date: 2026-05-08
content_hash: 71accfe60dbabcb8
---

# Learning Multi-Scale Hypergraph for High-Order Brain Connectivity Analysis

**Conference**: ICML 2026  
**arXiv**: [2606.03310](https://arxiv.org/abs/2606.03310)  
**Code**: None  
**Area**: Medical Imaging / Brain Network Analysis  
**Keywords**: Brain Networks, Hypergraph Learning, Graph Wavelets, Neurodegenerative Diseases, Multi-scale

## TL;DR
MuHL decomposes brain ROI features into multi-resolution representations using graph wavelets with learnable scales and dynamically generates soft hyperedges via a "node embedding × shared projection matrix" mechanism. It achieves 93.2% accuracy on ADNI for 5nd-stage AD classification and 76.8% on PPMI for PD classification, while providing interpretable key ROIs and hyperedges.

## Background & Motivation
**Background**: The current mainstream for analyzing brain networks (DTI structural or fMRI functional networks) is the GNN family—GCN, GAT, GCNII, and specialized models like BrainGNN, BrainGB, BrainNetTF, and ALTER. These models perform pairwise message passing between nodes (ROIs) and model high-order relationships indirectly by stacking layers.

**Limitations of Prior Work**: Abnormalities in brain function or structure are often group-wise phenomena where "multiple ROIs coexist in a state of coordinated dysfunction." Pairwise adjacency matrices cannot inherently express group-wise dependencies where 3 or more ROIs are connected simultaneously. Approximating high-order relationships by stacking GCN layers often triggers oversmoothing. Hypergraph models (HGNN, dwHGCN, HyBRiD, etc.) can explicitly represent "one hyperedge connecting a group of nodes," but most use **pre-defined** hyperedges (e.g., via KNN) or **only learn hyperedge weights** with a fixed topology, lacking flexibility.

**Key Challenge**: High-order interactions in the brain require both **learnable structures** (not pre-defined) and **multi-scale capability** (capturing both local clusters and global communities). Existing hypergraph methods fail to satisfy both requirements.

**Goal**: Without relying on manual hyperedge priors, the goal is to (i) directly learn continuous, sparse soft hyperedges and (ii) ensure ROI features at different resolutions correspond to hyperedges of different scales (small scales → compact hyperedges, large scales → inter-regional hyperedges).

**Key Insight**: Inspiration is drawn from the Spectral Graph Wavelet Transform (SGWT). The same graph signal is smoothed into versions with different receptive fields under different wavelet scales. As the scale increases, node features become more similar and naturally group into larger "sets." By treating this scale as a **learnable parameter**, the model can autonomously determine the neighborhood size each hierarchy should observe.

**Core Idea**: The pipeline consists of "learnable-scale graph wavelet decomposition → multi-scale soft hyperedge generation via a shared projection matrix $\Phi$ → multi-scale fusion via Transformer." This upgrades pairwise brain networks into learnable, cross-resolution hypergraphs for the staging and classification of neurodegenerative diseases.

## Method

### Overall Architecture
The input is a brain subject graph $\mathcal{G}$ (nodes = ROIs, edges = structural/functional connectivity) and node features $X \in \mathbb{R}^{N\times D}$ (e.g., SUVR, β-amyloid, tau, cortical thickness, or BOLD signals). The output is the disease stage label (5 classes for ADNI: CN/SMC/EMCI/LMCI/AD; 3 classes for PPMI: CN/Prodromal/PD).

The pipeline consists of three stages:

1.  **Adaptive Multi-Scale Feature Filtering (MSF)**: $X$ is decomposed into $\{X_{s_j}\}$ using a set of **learnable** scales $\{s_j\}_{j=1}^J$ via SGWT. Each $X_{s_j}$ represents ROI features at a specific receptive field.
2.  **Multi-Scale Hypergraph Structure Learning (HSL)**: For each scale $X_{s_j}$, a shared learnable projection matrix $\Phi$ calculates the incidence matrix $H_{s_j}$. Scale-specific soft hyperedges $\bar{H}_{s_j}$ are then obtained through ReLU + SoftMax + TopK sparsification.
3.  **Multi-Scale Transformer (MST)**: Hypergraph convolution is performed for each scale to obtain $F_{s_j}^{(Z)}$. These are fed into Scale-Wise Self-Attention (SWSA), where each attention head focuses on a single scale. The outputs are concatenated and passed through FFN + LN to produce $B^{(P)}$ for the classification head.

All parameters, including scales $s_j$ and projection $\Phi$, are learned end-to-end.

### Key Designs

1.  **Multi-Resolution Decomposition via Learnable Graph Wavelets**:
    - **Function**: Extracts various ROI representations $X_{s_j} = U g^2(s_j \Lambda) U^T X$ from the same graph $\mathcal{G}$ and node features based on $J$ **learnable** scales, where $U, \Lambda$ are from the spectral decomposition of the normalized Laplacian.
    - **Mechanism**: Traditional multi-resolution analysis uses multiple atlases or manual scales. Here, $s_j$ is a trainable scalar determined by backpropagation from the classification loss. Small $s$ results in slight smoothing where node differences are preserved, leading HSL to form compact hyperedges. Large $s$ spreads smoothing across more neighbors, making distant nodes similar and grouping them into larger hyperedges.
    - **Design Motivation**: To avoid the engineering burden of multiple parcellations and allow the "neighborhood size" to be a learnable hyperparameter aligned with the downstream classification task.

2.  **Soft Hyperedge Generation via Shared Projection Matrix Φ + TopK Sparsification**:
    - **Function**: Multiplies node embeddings $\bar{X}_{s_j} = X_{s_j} W$ with a learnable projection $\Phi \in \mathbb{R}^{d_h \times M}$ to obtain the incidence matrix $H_{s_j} = \bar{X}_{s_j}\Phi$. After $\tilde{H}_{s_j} = \mathrm{SoftMax}(\mathrm{ReLU}(\bar{X}_{s_j}\Phi))$, each node retains only top-$\eta$ hyperedges to form sparse $\bar{H}_{s_j}$.
    - **Mechanism**: Sharing $\Phi$ across all scales ensures a **correspondence** between the $M$ hyperedges across different resolutions—a single hyperedge might cover a compact set of nodes at a small scale but expand to a cross-regional group at a large scale. The authors prove two propositions: (1) calculating $H_s$ is equivalent to projecting $W_X(s)$ in the wavelet domain; (2) at least one hyperedge exists whose assigned node set size increases monotonically with $s$, approaching $N$ as $s \to \infty$. 
    - **Design Motivation**: Existing methods either use fixed KNN hyperedges or fixed topologies. This design allows the topology to be learned end-to-end while ensuring cross-scale semantic correspondence and sparsity through shared $\Phi$ and TopK operations.

3.  **Multi-Scale Fusion via Scale-Wise Self-Attention (SWSA)**:
    - **Function**: After hypergraph convolution (formula: $F_{s_j}^{(z)} = \sigma(\mathcal{D}_v^{-1/2}\bar{H}_{s_j} W_e \mathcal{D}_e^{-1} \bar{H}_{s_j}^T \mathcal{D}_v^{-1/2} F_{s_j}^{(z-1)} \Theta^{(z)})$), each scale is assigned an exclusive attention head to compute $A_{s_j} = \mathrm{Softmax}(Q_{s_j} K_{s_j}^T / \sqrt{d_k})$. The $J$ heads are concatenated and projected by $W_O$ followed by FFN and LN.
    - **Mechanism**: While traditional multi-head attention looks at different subspaces of the same feature, SWSA maps each head to a specific "resolution." Each head models long-range dependencies within its scale's graph before unified aggregation, explicitly embedding the "local-global" semantic hierarchy into the attention topology.
    - **Design Motivation**: Hypergraph convolution propagates messages within a scale, but cross-scale global dependencies require a dedicated fusion mechanism. SWSA makes this interaction learnable and decoupled.

### Loss & Training
The model is trained end-to-end using Cross-Entropy plus an L1 penalty on **negative** scales to ensure $s_j > 0$:

$$L = -\frac{1}{T}\sum_t\sum_c Y_{tc}\log\hat{Y}_{tc} + \alpha \frac{1}{J}\sum_j \mathbf{1}_{s<0}|s_j|$$

Default parameters: $J=3, M=16, \eta=3, d_h=16$. Evaluation is performed via 5-fold cross-validation with Adam optimization.

## Key Experimental Results

### Main Results
Two benchmarks were used: ADNI (650 subjects, 160 ROIs, 5 classes) and PPMI (181 subjects, 116 ROIs, 3 classes). MuHL was compared against 19 baselines including GNNs (GCN, BrainGNN, BrainNetTF, ALTER) and hypergraph models (HGNN, UniGCNII, HyBRiD).

| Dataset | Metric | MuHL | Prev. SOTA | Gain |
|--------|------|------|----------|------|
| ADNI (5 classes) | Acc | **93.2** | 90.8 (ALTER) | +2.4 |
| ADNI | F1 | **94.7** | 90.9 (ALTER) | +3.8 |
| PPMI (3 classes) | Acc | **76.8** | 72.9 (GAT) | +3.9 |
| PPMI | F1 | **62.4** | 56.4 (BQN) | +6.0 |

MuHL also leads in zero-shot cross-dataset/cross-stage tasks: ADNI-2 → ADNI-1/3/GO (Acc 56.0 vs. 52.8) and PPMI → TaoWu (Acc 60.5 vs. 57.0), indicating the learned hypergraph structures possess transferability.

### Ablation Study
| Configuration | ADNI Acc | PPMI Acc | Description |
|------|---------|---------|------|
| Full (MSF+HSL+MST) | **93.2** | **76.8** | Complete MuHL |
| w/o MSF | 90.0 | 72.9 | No multi-scale decomposition; single resolution only |
| w/o HSL | 76.8 | 67.4 | No hypergraph structure learning (used pre-defined); **largest drop** |
| w/o MST | 86.9 | 64.1 | No multi-scale Transformer; hypergraph convolution only |

### Key Findings
- **HSL is critical**: Removing HSL results in a 16.4% drop on ADNI and 9.4% on PPMI, proving that "making hyperedge topology learnable" is more central than multi-scale or Transformer fusion. This suggests the bottleneck of existing "weight-only" hypergraph methods.
- **Optimal Hyperedge Count M**: Performance peaks at $M=16$. Exceeding 20 introduces redundancy/noise. $d_h$ similarly saturates after 16, with larger values showing signs of overfitting.
- **Hub ROIs align with clinical priors**: On ADNI, top-10 ROIs include the bilateral globus pallidus, putamen, hippocampus, and thalamus (subcortical nuclei strongly associated with AD). On PPMI, they include the amygdala, thalamus, and supramarginal gyrus. These ROIs often appear in symmetric pairs, consistent with the bilateral spread of neurodegenerative diseases.

## Highlights & Insights
- **Scales as Learnable Continuous Variables**: Unlike previous work using fixed or discrete scales, MuHL uses $s_j$ in backpropagation, allowing the model to choose the optimal neighborhood size. This technique can be transferred to any task requiring multi-resolution graph signals (molecules, social networks, point clouds).
- **Cross-Scale Hyperedge Construction via Shared $\Phi$**: This approach simultaneously achieves cross-scale semantic correspondence and monotonic hyperedge expansion, both backed by formal proofs. It is more elegant than KNN-based or independent matrix learning.
- **Intrinsic Interpretability**: Hyperedge activation directly reflects ROI importance, and aggregated activity ranks hub ROIs. Interpretability is a natural structural product rather than an external module added after training.

## Limitations & Future Work
- **Severe Class Imbalance**: The AD group in ADNI has only 12 subjects compared to 226 in the CN group. The overall 5-class accuracy might be overly optimistic, and predictions for minority classes may be unstable.
- **No Public Code**: The lack of a GitHub link makes reproduction difficult, particularly regarding $\Phi$ initialization and training dynamics.
- **Coupling of M and ROI Count**: $M=16$ was optimal for both $N=160$ and $N=116$. It is unclear if this holds for denser atlases (e.g., Schaefer 400/1000) without re-tuning.
- **Spectral Decomposition Complexity**: SGWT requires $O(N^3)$ for the Laplacian decomposition. While manageable for 160 ROIs, it may fail for voxel-level graphs. Chebyshev polynomial approximation could be a solution.
- **Static Connectivity**: Only static functional connectivity matrices were used, ignoring the temporal dynamics of fMRI. Future work could introduce temporal scales or time-evolving hyperedges.

## Related Work & Insights
- **vs. HGNN / HNHN / UniGCNII**: Classic hypergraph GNNs require a pre-defined incidence $H$. MuHL learns $H$ end-to-end, bypassing the "hypergraph prior" problem.
- **vs. dwHGCN / HyBRiD**: These methods assume fixed topologies (learning only weights or masks). MuHL's continuous soft topology provides a higher level of flexibility, evidenced by the 16% drop when HSL is removed.
- **vs. BrainNetTF / ALTER**: Transformer-based methods typically model pairwise attention. MuHL uses hypergraphs to explicitly capture group-wise relationships, outperforming ALTER (93.2% vs 90.8% on ADNI).
- **vs. Original SGWT**: Traditional SGWT is used for denoising with fixed scales. MuHL treats it as a learnable feature pyramid connected to hypergraph learning, representing a novel application in GNNs.

## Rating
- **Novelty**: ⭐⭐⭐⭐ End-to-end learnable scale wavelets combined with shared-projection soft hypergraphs is a clean and original approach.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers major benchmarks, 19 baselines, zero-shot transfer, and sensitivity analysis. Loses one star due to class imbalance and lack of a third fMRI dataset.
- **Writing Quality**: ⭐⭐⭐⭐ Clear motivation, well-supported propositions, and coherent explanations.
- **Value**: ⭐⭐⭐⭐ Provides a strong baseline for learnable multi-resolution hypergraphs in brain analysis; however, the lack of code limits its immediate engineering utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Continual Learning for fMRI-Based Brain Disorder Diagnosis via Functional Connectivity Matrices Generative Replay](../../CVPR2026/medical_imaging/forge_continual_learning_for_fmri_based_brain_disorder_diagnosis.md)
- [\[CVPR 2026\] Modeling Spatiotemporal Neural Frames for High Resolution Brain Dynamics](../../CVPR2026/medical_imaging/modeling_spatiotemporal_neural_frames_for_high_resolution_brain_dynamic.md)
- [\[ICML 2026\] CAME-Grad: The Double Dilemma in Multi-Task Radiology Report Generation — A Gradient Dynamics Analysis and Solution](the_double_dilemma_in_multi-task_radiology_report_generation_a_gradient_dynamics.md)
- [\[NeurIPS 2025\] Riemannian Flow Matching for Brain Connectivity Matrices via Pullback Geometry](../../NeurIPS2025/medical_imaging/riemannian_flow_matching_for_brain_connectivity_matrices_via_pullback_geometry.md)
- [\[ICML 2026\] PathCTM: Thinking in Scales — Accelerating Gigapixel Pathology Image Analysis via Adaptive Continuous Reasoning](thinking_in_scales_accelerating_gigapixel_pathology_image_analysis_via_adaptive_.md)

</div>

<!-- RELATED:END -->
