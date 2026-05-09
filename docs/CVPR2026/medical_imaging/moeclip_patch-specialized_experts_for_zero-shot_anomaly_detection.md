---
title: >-
  [Paper Note] MoECLIP: Patch-Specialized Experts for Zero-shot Anomaly Detection
description: >-
  [CVPR 2026][Medical Imaging][Zero-shot anomaly detection] MoECLIP introduces Mixture-of-Experts into zero-shot anomaly detection (ZSAD), achieving patch-level dynamic expert routing and specialization via Frozen Orthogonal Feature Separation (FOFS) and an Equiangular Tight Frame (ETF) loss, attaining state-of-the-art performance across 14 industrial and medical benchmarks.
tags:
  - CVPR 2026
  - Medical Imaging
  - Zero-shot anomaly detection
  - mixture of experts
  - CLIP
  - LoRA
  - expert specialization
date: 2026-05-08
content_hash: ef54f1960d186356
---

# MoECLIP: Patch-Specialized Experts for Zero-shot Anomaly Detection

**Conference**: CVPR 2026
**arXiv**: [2603.03101](https://arxiv.org/abs/2603.03101)
**Code**: [Available](https://github.com/CoCoRessa/MoECLIP)
**Area**: Medical Imaging
**Keywords**: Zero-shot anomaly detection, mixture of experts, CLIP, LoRA, expert specialization

## TL;DR

MoECLIP introduces Mixture-of-Experts into zero-shot anomaly detection (ZSAD), achieving patch-level dynamic expert routing and specialization via Frozen Orthogonal Feature Separation (FOFS) and an Equiangular Tight Frame (ETF) loss, attaining state-of-the-art performance across 14 industrial and medical benchmarks.

## Background & Motivation

### 1. State of the Field
Visual anomaly detection (AD) identifies regions deviating from normal patterns and is critical in industrial defect inspection and medical image diagnosis. Conventional unsupervised AD (UAD) learns exclusively from normal data but still requires large collections of normal samples. Zero-shot anomaly detection (ZSAD) leverages the strong generalization capability of vision–language models such as CLIP to detect anomalies without any training data from the target category, establishing itself as an emerging paradigm.

### 2. Limitations of Prior Work
CLIP is pre-trained for global semantic understanding and is inherently ill-suited for detecting local anomalies. Existing ZSAD methods (PromptAD, AnomalyCLIP, AdaCLIP, AA-CLIP) enhance patch representations through prompt learning or adapters, yet all adopt a **patch-agnostic design**: they apply a single, uniform transformation to every patch, ignoring the distinct characteristics of different image regions (object parts, background, anomalous regions).

### 3. Root Cause
Specializing CLIP for anomaly detection must not come at the cost of its generalization ability; differentiating treatment across patches risks **functional redundancy** in a naïve multi-expert setup, where experts converge to similar functions.

### 4. Core Problem
(1) Break the patch-agnostic design constraint to enable patch-level dynamic adaptation; (2) resolve expert functional redundancy in MoE so that each expert achieves genuine specialization.

### 5. Starting Point
Combine a MoE architecture with LoRA and introduce simultaneous constraints at both the input and output ends to disentangle expert functions.

### 6. Core Idea
Route each patch dynamically to appropriate LoRA experts via a MoE architecture; apply FOFS to orthogonally partition the feature space at the input side, and enforce maximal equiangular separation via ETF loss at the output side, jointly eliminating expert redundancy.

## Method

### Overall Architecture

MoECLIP integrates MoE modules at the outputs of multiple layers (layers 6, 12, 18, and 24) of the CLIP Vision Encoder (ViT-L/14-336), with the encoder weights completely frozen. Each MoE module contains $K=4$ LoRA experts and a linear router with a Top-2 routing strategy. The model is trained with supervision on an auxiliary dataset (VisA) and evaluated on entirely unseen categories at test time.

Overall pipeline: input image → multi-layer patch feature extraction by CLIP ViT → dynamic routing and feature adaptation by per-layer MoE modules → multi-scale aggregation by PAA → cosine similarity with text features → anomaly map and anomaly score.

### Key Designs

#### 1. MoE-based Feature Adaptation

**Function**: Dynamically select the optimal expert combination for each patch at every layer to adapt its representation.

**Mechanism**: Each MoE module receives patch feature $F_i^l \in \mathbb{R}^d$; the router computes routing scores for each expert and selects the Top-$k$ experts to produce a weighted residual output $F_{i,\text{expert}}^l$. A critical technique is **norm normalization** of the MoE output (matching the $\ell_2$ norm of the original feature), followed by a weighted residual connection ($\lambda_{\text{MoE}}=0.1$) with the original feature, preventing norm mismatch from causing training instability and generalization degradation.

**Design Motivation**: LoRA experts are inherently parameter-efficient (rank=8), reducing overfitting risk; norm normalization, inspired by AA-CLIP, preserves the stability of CLIP's representation space.

#### 2. FOFS (Frozen Orthogonal Feature Separation)

**Function**: Enforce disjoint feature subspaces at the input end of LoRA experts to eliminate input-level redundancy.

**Mechanism**: The $d$-dimensional input feature space is partitioned into $K$ non-overlapping subspaces $c_1, \dots, c_K$. Each expert's LoRA down-projection matrix $A_n \in \mathbb{R}^{r \times d}$ is constructed as a block matrix — only the columns corresponding to the $n$-th subspace are filled with an orthogonal matrix $Q_n$ obtained via QR decomposition, with remaining columns set to zero. This guarantees $A_n A_m^T = 0$ ($n \neq m$), i.e., mutual orthogonality across experts. Critically, $A_n$ is frozen throughout training; only $B_n$ is learnable.

**Design Motivation**: (1) Physically forces different experts to attend to distinct feature dimensions, preventing redundancy from initialization; (2) freezing $A$ preserves CLIP's generalization ability and reduces overfitting risk, inspired by recent LoRA research showing that randomly initialized orthogonal $A$ matrices can match the performance of learned ones.

#### 3. ETF Loss (Equiangular Tight Frame Loss)

**Function**: Constrain expert output vectors to maximal equiangular separation at the output end of LoRA experts, eliminating output-level redundancy.

**Mechanism**: For each patch at each layer, the Gram matrix of the $K$ expert outputs (after $\ell_2$ normalization) is computed, and its deviation from the ideal ETF Gram matrix is penalized via the Frobenius norm. The ideal structure requires diagonal entries of 1 (unit norm) and off-diagonal entries of $-1/(K-1)$ (maximal equiangularity).

**Design Motivation**: FOFS constrains only the input side; the learnable $B$ matrices can still cause expert outputs to converge to similar subspaces. The ETF loss serves as a complementary mechanism, further enforcing expert differentiation at the output side.

#### 4. PAA (Patch Averaging Aggregation)

**Function**: Integrate multi-scale contextual information during training to improve detection of anomalies at varying sizes.

**Mechanism**: Patch embeddings are reshaped into a 2D spatial grid, and average pooling is applied over multiple sliding window scales $s \in \{1, 3, 5\}$, independently producing multiple sets of patch features. No additional parameters are introduced.

**Design Motivation**: The fixed patch size of ViT is inherently limited in detecting anomalies at different scales; existing methods apply patch aggregation only at test time, lacking multi-scale awareness during training. The benefit is especially pronounced on medical datasets.

#### 5. Depth-wise Adapter

**Function**: Provide semantically aligned global features for image-level anomaly scoring.

**Mechanism**: Inspired by MobileNet, a 1D depthwise separable convolution (Depthwise + Pointwise) processes the final-layer PAA features, followed by global average pooling to obtain an image-level vector $V_{\text{image}}$, which is compared with text features via cosine similarity to produce the anomaly score.

### Loss & Training

Total loss: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{seg}} + \mathcal{L}_{\text{ac}} + \lambda_{\text{etf}}\mathcal{L}_{\text{etf}} + \lambda_{\text{bal}}\mathcal{L}_{\text{bal}}$

- **Segmentation loss** $\mathcal{L}_{\text{seg}}$: Focal + Dice Loss, applied to multi-layer multi-scale anomaly maps
- **Classification loss** $\mathcal{L}_{\text{ac}}$: BCE Loss, applied to image-level anomaly scores
- **ETF loss**: $\lambda_{\text{etf}}=0.01$, constrains expert outputs to equiangular separation
- **Balance loss**: $\lambda_{\text{bal}}=0.01$, uses the squared coefficient of variation of routing probabilities to prevent expert collapse

Training configuration: OpenCLIP ViT-L/14-336, image size 518×518, Adam with $\text{lr}=5 \times 10^{-4}$, 20 epochs, 2× V100 16 GB.

## Key Experimental Results

### Main Results

Comparison with 6 state-of-the-art methods on 14 datasets (5 industrial + 9 medical); all models are trained on VisA (for VisA evaluation, MVTec-AD is used as the training set).

**Table 1: Image-level anomaly classification (AUROC, AP)**

| Method | MVTec-AD | VisA | BTAD | RSDD | DTD-Syn | BrainMRI | HeadCT | LiverCT | RetinaOCT | **Average** |
|--------|----------|------|------|------|---------|----------|--------|---------|-----------|-------------|
| WinCLIP | (91.8,95.1) | (78.1,77.5) | (83.3,84.1) | (85.3,65.3) | (95.0,97.9) | (45.1,80.3) | (83.7,81.6) | (66.5,56.1) | (53.7,44.3) | (75.8,75.8) |
| AnomalyCLIP | (91.9,96.2) | (82.1,85.4) | (92.5,94.2) | (74.0,73.2) | (93.3,97.7) | (70.8,90.6) | (95.1,95.3) | (68.2,63.4) | (74.7,73.9) | (82.5,85.5) |
| AA-CLIP | (90.9,96.0) | (79.2,83.7) | (94.8,97.5) | (94.9,94.2) | (92.5,97.7) | (79.6,94.4) | (95.4,94.3) | (58.4,49.7) | (83.4,83.8) | (85.5,87.9) |
| Bayes-PFL | (92.2,96.1) | (86.8,89.3) | (93.0,96.7) | (91.3,89.7) | (93.5,97.7) | (81.9,94.5) | (95.4,93.2) | (61.7,55.2) | (83.7,81.8) | (86.6,88.2) |
| **MoECLIP** | **(93.9,96.8)** | **(83.6,86.2)** | **(93.1,98.0)** | **(95.3,95.1)** | **(95.5,98.6)** | **(88.5,97.1)** | **(96.6,94.5)** | **(74.0,64.6)** | **(85.5,84.9)** | **(89.6,90.6)** |

MoECLIP achieves an average image-level AUROC of 89.6% (+3.0%) and AP of 90.6% (+2.4%).

**Table 2: Pixel-level anomaly segmentation (AUROC, AP) — selected results**

| Method | MVTec-AD | BTAD | BrainMRI | ColonDB | ClinicDB | Kvasir | **Average** |
|--------|----------|------|----------|---------|----------|--------|-------------|
| AA-CLIP | (91.6,45.4) | (95.6,49.4) | (96.7,55.1) | (82.8,31.5) | (89.2,49.8) | (86.0,52.9) | (93.2,45.8) |
| Bayes-PFL | (91.9,48.4) | (95.6,48.6) | (95.7,42.9) | (82.9,30.7) | (88.2,49.1) | (85.6,53.4) | (93.2,44.3) |
| **MoECLIP** | **(92.5,45.7)** | **(96.8,50.4)** | **(97.3,61.3)** | **(85.4,34.8)** | **(89.7,49.9)** | **(88.1,57.6)** | **(94.3,47.5)** |

Pixel-level average AUROC reaches 94.3% (+1.1%) and AP 47.5% (+1.7%), with particularly notable gains on medical datasets (BrainMRI AP +6.2%, Kvasir AP +4.2%).

### Ablation Study

**Table 3: Component ablation (Pixel AUROC, Image AUROC)**

| Configuration | MVTec-AD | DTD-Syn | HeadCT | ColonDB | Average |
|---------------|----------|---------|--------|---------|---------|
| Vanilla CLIP | (38.4,74.1) | (33.9,71.6) | (-,56.5) | (49.5,-) | (40.6,67.4) |
| w/o FOFS & ETF | (91.6,91.7) | (97.8,93.1) | (-,94.4) | (84.1,-) | (91.2,93.1) |
| w/o FOFS | (92.0,92.8) | (98.3,93.9) | (-,95.0) | (85.3,-) | (91.9,93.9) |
| w/o ETF Loss | (92.2,92.7) | (98.2,93.4) | (-,96.1) | (84.6,-) | (91.7,94.1) |
| w/o Depth Adapter | (92.0,92.5) | (98.1,93.8) | (-,94.5) | (85.0,-) | (91.7,93.6) |
| w/o PAA | (92.1,92.8) | (98.1,94.7) | (-,93.1) | (81.9,-) | (90.7,93.5) |
| **MoECLIP (full)** | **(92.5,93.9)** | **(98.8,95.5)** | **(-,96.6)** | **(85.4,-)** | **(92.2,95.3)** |

### Key Findings

1. **FOFS and ETF are complementary**: Removing either component individually degrades performance, and removing both causes a larger drop, confirming the necessity of dual constraints at the input and output ends.
2. **Functional redundancy quantified**: Inter-expert cosine similarity decreases from 0.45 (vanilla MoE) → 0.24 (after adding FOFS) → 0.02 (after adding ETF), nearly eliminating redundancy entirely.
3. **PAA is critical for the medical domain**: Removing PAA decreases performance by 3.5% on HeadCT and 3.5% on ColonDB, demonstrating the importance of multi-scale perception for medical anomaly detection.
4. **More experts is not necessarily better**: $K=4$ is optimal; $K>4$ leads to performance degradation due to functional redundancy.
5. **Cross-domain generalization**: Despite training only on industrial data, MoE experts still route and specialize effectively on medical data.

## Highlights & Insights

1. **First introduction of MoE into ZSAD**: A pioneering paradigm shift from patch-agnostic to patch-specialized processing.
2. **Elegant dual-end constraint design**: FOFS physically isolates subspaces at the input side (frozen, zero extra parameters); ETF loss enforces a geometrically optimal structure at the output side; the two mechanisms are orthogonal and mutually complementary.
3. **Intuitive visualization**: Grad-CAM clearly shows Expert 1 focusing on anomalous regions, Expert 2 on the object body, and Expert 3 on the background, confirming that routing is genuinely content-driven.
4. **The elegance of freezing $A$ in FOFS**: By leveraging the recent LoRA finding that random orthogonal $A \approx$ learned $A$, the design simultaneously achieves orthogonal separation, parameter savings, and overfitting suppression.

## Limitations & Future Work

1. **Expert count set manually**: $K=4$ is an empirical choice; no mechanism exists to adaptively determine the number of experts.
2. **FOFS partitions subspaces equally**: Feature dimensions are divided uniformly across experts, without accounting for the possibility that different experts may require different dimensionalities.
3. **Validated on ViT-L/14 only**: The effect of different backbone scales (e.g., ViT-B, ViT-H) remains unexplored.
4. **Single auxiliary training set**: VisA is consistently used as the auxiliary training set; the impact of different training sets on generalization has not been investigated.
5. **Fixed PAA window scales**: $s \in \{1,3,5\}$ is manually defined; adaptive or learnable scale selection could be considered.

## Related Work & Insights

- **Evolution of ZSAD methods**: WinCLIP → April-GAN → AnomalyCLIP → AdaCLIP → AA-CLIP → Bayes-PFL → MoECLIP, progressing from hand-crafted prompts to learned prompts, then adapters, and finally MoE-based dynamic routing.
- **Addressing functional redundancy in MoE**: Existing approaches (contrastive loss, orthogonal regularization) act solely on the output side; this work simultaneously constrains both the input and output sides, a strategy transferable to other MoE scenarios.
- **Inspiration from freezing $A$ in LoRA**: Works such as VeRA have validated the feasibility of shared/frozen down-projection matrices; combining this with orthogonal separation is worth exploring in other PEFT+MoE settings.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Pioneering introduction of MoE into ZSAD establishing a patch-specialized paradigm; the dual-end FOFS+ETF constraint is conceptually distinctive.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Comprehensive evaluation across 14 datasets (industrial + medical) with full comparisons, ablations, visualizations, and quantified expert similarity; highly rigorous.
- **Writing Quality**: ⭐⭐⭐⭐ — Motivation is clearly articulated, methodology is systematically presented, and visualizations are informative.
- **Value**: ⭐⭐⭐⭐ — Provides elegant solutions to both ZSAD and MoE functional redundancy; the approach generalizes naturally to other PEFT+MoE scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] VisualAD: Language-Free Zero-Shot Anomaly Detection via Vision Transformer](visualad_language-free_zero-shot_anomaly_detection_via_vision_transformer.md)
- [\[CVPR 2026\] Bidirectional Multimodal Prompt Learning with Scale-Aware Training for Few-Shot Multi-Class Anomaly Detection](bidirectional_multimodal_prompt_learning_with_scale-aware_training_for_few-shot_.md)
- [\[ICLR 2026\] Dual Distillation for Few-Shot Anomaly Detection](../../ICLR2026/medical_imaging/dual_distillation_for_few-shot_anomaly_detection.md)
- [\[CVPR 2026\] InvAD: Inversion-based Reconstruction-Free Anomaly Detection with Diffusion Models](invad_inversion-based_reconstruction-free_anomaly_detection_with_diffusion_model.md)
- [\[CVPR 2026\] Interpretable Cross-Domain Few-Shot Learning with Rectified Target-Domain Local Alignment](interpretable_cross-domain_few-shot_learning_with_rectified_target-domain_local_.md)

</div>

<!-- RELATED:END -->
