---
title: >-
  [Paper Note] SIGMA: Sinkhorn-Guided Masked Video Modeling
description: >-
  [ECCV 2024][LLM Evaluation][masked video modeling] This paper proposes SIGMA, which upgrades the reconstruction target of masked video modeling from the pixel level to learnable deep feature cluster assignments by introducing a projection network. Employing optimal transport with the Sinkhorn algorithm to enforce high-entropy regularization, SIGMA avoids representation collapse and comprehensively outperforms state-of-the-art methods like VideoMAE across 10 datasets and 3 ben…
tags:
  - "ECCV 2024"
  - "LLM Evaluation"
  - "masked video modeling"
  - "self-supervised learning"
  - "optimal transport"
  - "Sinkhorn algorithm"
  - "video representation"
date: 2026-05-08
content_hash: 06ee19329c0eeeb7
---

# SIGMA: Sinkhorn-Guided Masked Video Modeling

**Conference**: ECCV 2024  
**arXiv**: [2407.15447](https://arxiv.org/abs/2407.15447)  
**Code**: [https://quva-lab.github.io/SIGMA](https://quva-lab.github.io/SIGMA)  
**Area**: Self-supervised Video Learning  
**Keywords**: masked video modeling, self-supervised learning, optimal transport, Sinkhorn algorithm, video representation

## TL;DR

This paper proposes SIGMA, which upgrades the reconstruction target of masked video modeling from the pixel level to learnable deep feature cluster assignments by introducing a projection network. Employing optimal transport with the Sinkhorn algorithm to enforce high-entropy regularization, SIGMA avoids representation collapse and comprehensively outperforms state-of-the-art methods like VideoMAE across 10 datasets and 3 benchmarks.

## Background & Motivation

**Background**: Self-supervised video learning is undergoing an evolution from pretext tasks $\rightarrow$ contrastive learning $\rightarrow$ masked modeling. Masked video modeling (MVM) methods like VideoMAE pretrain ViTs by reconstructing the pixel values of masked space-time tubes, demonstrating excellent scalability.

**Limitations of Prior Work**: Pixel-level reconstruction targets are inherently low-level. Since an individual patch or space-time tube does not represent an independent semantic unit (unlike words/subwords in NLP), models tend to learn low-level texture features rather than high-level semantics. Experimental evidence shows that VideoMAE performs weakly under frozen linear probing (achieving only 20.7% on K400).

**Key Challenge**: If the reconstruction target is replaced with learnable deep features, the joint optimization of the projection network and the video model collapses to a trivial solution (where all inputs map to the same feature vector), which the L2 loss cannot prevent.

**Key Insight**: Constraining the feature space to a finite number of learnable prototypes and forcing uniform usage of all clusters through the equipartition constraint of optimal transport forms a high-entropy bottleneck. This both avoids representation collapse and infuses the features with semantic information.

**Core Idea**: Solving optimal transport via the Sinkhorn algorithm $\rightarrow$ generating pseudo-labels $\rightarrow$ symmetric prediction of cluster assignments between the video model and the projection network $\rightarrow$ jointly learning a semantically rich feature space.

## Method

### Overall Architecture

Input video $\rightarrow$ partition into space-time tubes ($2 \times 16 \times 16$) $\rightarrow$ apply 90% masking $\rightarrow$ video model $\Psi$ (encoder-decoder) processes only the unmasked parts to predict the features of masked tubes $\mathbf{x}^\Psi$ $\rightarrow$ projection network $\varphi$ views all tubes of the complete video to output features $\mathbf{x}^\varphi$ $\rightarrow$ both sets of features are projected onto shared learnable prototypes $\mathbf{C}$ $\rightarrow$ the Sinkhorn algorithm generates cluster pseudo-labels $\rightarrow$ compute symmetric cross-entropy loss.

### Key Designs

1. **From Pixel Reconstruction to Feature Reconstruction**:

    - Function: Use deep features instead of pixel values as the reconstruction targets for MVM.
    - Mechanism: Introduce a projection network $\varphi$ to embed all space-time tubes and select the features of masked tubes as targets. The L2 reconstruction loss is formulated in the feature space as:
    $$\mathcal{L}_2^F = \frac{1}{N}\sum_{i=1}^N \|\mathbf{x}_i^\varphi - \mathbf{x}_i^\Psi\|_2^2$$
    - Design Motivation: Pixel-level targets lean toward learning low-level features, whereas deep feature targets can capture more abstract semantic information.
    - **Problem**: Jointly optimizing both networks leads to a trivial solution (where all space-time tubes map to the same feature vector).

2. **Sinkhorn-Guided Clustering Bottleneck**:

    - Function: Constrain the feature space to finite clusters to prevent representation collapse.
    - Mechanism: Define $K$ learnable prototypes $\mathbf{C} = \{\mathbf{c}_1, ..., \mathbf{c}_K\}$, and model the mapping from features to prototypes as an entropy-regularized optimal transport problem:
    $$\min_{\mathbf{Q}} \langle \mathbf{Q}, -\log \mathbf{X} \rangle + \frac{1}{\lambda} \text{KL}(\mathbf{Q} \| rc^\top)$$
      where the equipartition constraints $r = \frac{1}{K} \cdot \mathbb{1}$ and $c = \frac{1}{B} \cdot \mathbb{1}$ force all prototypes to be utilized uniformly. The fast Sinkhorn-Knopp algorithm is used to solve this on GPU, generating soft pseudo-labels $\mathbf{q}$.
    - Design Motivation: (1) A finite number of cluster centers forces similar space-time tubes to share the same prototype, injecting semantic information; (2) the equipartition constraint prevents all points from collapsing into a single cluster; (3) online computation within mini-batches is enabled, making training highly efficient.

3. **Symmetric Prediction Loss**:

    - Function: Construct a self-supervised prediction task using pseudo-labels.
    - Mechanism: The video model predicts the cluster assignments of the projection network, and vice versa:
    $$\mathcal{L} = \frac{1}{B}\sum_{i=1}^B [\mathcal{L}_{CE}(\tilde{\mathbf{x}}_i^\varphi, \mathbf{q}_i^\Psi) + \mathcal{L}_{CE}(\tilde{\mathbf{x}}_i^\Psi, \mathbf{q}_i^\varphi)]$$
      where $\tilde{\mathbf{x}} = \mathbf{x}^\top \mathbf{c}$ is the inner product score of the feature and the prototype.
    - Design Motivation: (1) Symmetric prediction avoids unidirectional dependency; (2) no momentum encoder (EMA) is required, simplifying the training pipeline; (3) no data augmentation is needed to generate different views.

### Loss & Training

- Use AdamW optimizer, base learning rate $1.5e^{-4}$, weight decay 0.05, cosine decay.
- 90% tube masking ratio with a tube size of $2 \times 16 \times 16$.
- Projection network $\varphi$ options: (a) 3-layer MLP (sigma-mlp), (b) frozen DINO pretrained model (sigma-dino).
- Pretrain for 800 epochs on K400/SSv2.
- Temperature parameter $\tau$ controls softmax sharpness.

## Key Experimental Results

### Main Results — Frozen Linear Probing (ViT-B, K400 pretrained)

| Dataset | SIGMA-MLP | SIGMA-DINO | VideoMAE | MGMAE | Gain (MLP vs MAE) |
|--------|-----------|------------|----------|-------|----------|
| SSv2 | 19.9 | **20.8** | 17.5 | 16.8 | +2.4 |
| K400 | 30.7 | **47.5** | 20.7 | 24.9 | +10.0 |
| UCF-101 | 73.8 | **80.7** | 58.6 | 64.4 | +15.2 |
| HMDB-51 | 45.0 | **52.3** | 37.7 | 41.3 | +7.3 |
| IN-1K | 24.1 | **45.0** | 20.2 | 20.4 | +3.9 |
| C-100 | 46.2 | **66.7** | 40.4 | 44.1 | +5.8 |

### Full Finetuning (ViT-B, 800 epochs)

| Dataset | SIGMA-DINO | VideoMAE | MME | MGMAE | Remarks |
|--------|------------|----------|-----|-------|------|
| SSv2 (K400 pretrain) | **71.1** | 68.5 | 70.5 | - | Outperforms all methods, including those with motion guidance |
| SSv2 (SSv2 pretrain) | **70.9** | 69.6 | 70.0 | 71.0 | On par with MGMAE (which uses optical flow guidance) |
| K400 | **81.6** | 80.0 | - | 81.2 | Achieves SOTA without relying on motion guidance |

### Ablation Study

| Configuration | Accuracy (mini-SSv2) | Description |
|------|---------|------|
| VideoMAE (baseline) | 56.9 | Pixel-level reconstruction |
| SIGMA-MLP + L2 loss | 36.4 | Deep features + L2 $\rightarrow$ collapse |
| SIGMA-MLP + Eq.7 | **60.3** | Sinkhorn clustering loss resolves collapse |
| Prototypes K=1000 | 59.7 | Insufficient clusters |
| Prototypes K=4000 | **60.3** | Optimal number of clusters |
| Prototypes K=6000 | 60.1 | Slight drop |

### Key Findings

- **Huge Gap in Frozen Evaluation**: SIGMA-DINO reaches 47.5% on K400 frozen probing, whereas VideoMAE is only at 20.7%, representing a 130% relative improvement. This demonstrates that the features learned by SIGMA possess significantly stronger semantics than pixel reconstruction methods.
- **Direct Collapse of L2 Feature Reconstruction**: Only 36.4% on mini-SSv2 (lower than VideoMAE's 56.9%), completely validating the severity of the trivial solution problem.
- **No Motion Guidance Required**: SIGMA does not use optical flow or motion vectors; solely relying on random tube masking and Sinkhorn clustering, it matches or even outperforms methods that use motion guidance (MME, MGMAE).
- **Unsupervised Video Segmentation**: SIGMA-DINO achieves an overclustering mIoU of 56.5% on DAVIS (vs. 50.9% for VideoMAE), showing that the learned features possess better spatiotemporal semantic understanding.
- **SEVERE Generalization Benchmark**: Leads comprehensively across four dimensions: domain shift, sample efficiency, action granularity, and task shift, with an average score of 64.3 vs. 57.4 for VideoMAE.

## Highlights & Insights

- **An Elegant Concept of Optimal Transport as a Self-Supervised Regularizer**: The Sinkhorn equipartition constraint simultaneously prevents feature collapse and injects semantic information, killing two birds with one stone.
- **Flexibility of the Projection Network**: It can be a simple MLP (purely self-supervised) or a pretrained DINO (leveraging image priors), unified within a single framework.
- **No Reliance on Data Augmentation**: Unlike contrastive learning methods such as DINO or BYOL that require carefully designed spatiotemporal augmentation pipelines, SIGMA is sufficient with only masking and clustering, making it more scalable.
- **Frozen Probing is the Gold Standard for Assessing Pretrained Feature Quality**: The gap on this metric is much larger than in full finetuning, showing a significant improvement in the semantic quality of the features themselves.
- **Space-Time Tube-Level Clustering Grants Token-Level Semantic Meaning**: This bridges the semantic gap between visual tokens (patches) and language tokens (words).

## Limitations & Future Work

- **Not Extended to ViT-L/ViT-H**: Due to limited academic computational resources, only ViT-S and ViT-B were validated.
- **External Supervision Introduced by the DINO Variant**: SIGMA-DINO employs an ImageNet-pretrained DINO model, making it not purely self-supervised on videos.
- **Number of Clusters $K$ Requires Tuning**: Although not highly sensitive to $K$, the optimal value ($\sim 4000$) still needs to be determined experimentally.
- **Significant Performance Gap Between SIGMA-MLP and SIGMA-DINO Under Full Finetuning**: This indicates that the representational capacity of the MLP projection network is still limited.
- **Hierarchical Clustering Can Be Explored**: Currently, only a single-level prototype is used; hierarchical clustering could capture richer semantic structures.

## Related Work & Insights

- **vs. VideoMAE**: The core difference lies in the reconstruction target—pixels vs. cluster pseudo-labels. SIGMA comprehensively outperforms it across all evaluation protocols, especially with a huge gap in frozen probing.
- **vs. MME**: MME predicts HOG features + motion trajectories, requiring preprocessing to remove camera motion; SIGMA requires no preprocessing.
- **vs. MGMAE**: MGMAE uses motion vectors to guide the masking strategy, which is a complementary technique; SIGMA can also be combined with motion-guided masking for further improvement.
- **vs. SwAV/DINO (Image Domain)**: SIGMA successfully transfers the Sinkhorn clustering concept from image contrastive learning to video masked modeling.
- **vs. JEPA**: JEPA also explores latent-space prediction targets but suffers from training instability; SIGMA maintains stable training through the Sinkhorn equipartition constraint.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The combination of Sinkhorn clustering + masked video modeling is pioneering, resolving the collapse challenge in learning with deep targets.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluation across 10 datasets and 3 benchmarks (including frozen probing, fine-tuning, segmentation, and generalization) is highly comprehensive.
- Writing Quality: ⭐⭐⭐⭐⭐ The logical progression from pixel targets $\rightarrow$ feature targets $\rightarrow$ collapse issues $\rightarrow$ Sinkhorn solutions is highly natural.
- Value: ⭐⭐⭐⭐⭐ Has a paradigm-shifting impact on the self-supervised video learning domain; the Sinkhorn clustering paradigm can be widely transferred.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Eliminating Warping Shakes for Unsupervised Online Video Stitching](eliminating_warping_shakes_for_unsupervised_online_video_stitching.md)
- [\[ECCV 2024\] Deep Cost Ray Fusion for Sparse Depth Video Completion](deep_cost_ray_fusion_for_sparse_depth_video_completion.md)
- [\[ECCV 2024\] OGNI-DC: Robust Depth Completion with Optimization-Guided Neural Iterations](ogni-dc_robust_depth_completion_with_optimization-guided_neural_iterations.md)
- [\[ECCV 2024\] ColorMNet: A Memory-based Deep Spatial-Temporal Feature Propagation Network for Video Colorization](colormnet_a_memory-based_deep_spatial-temporal_feature_propagation_network_for_v.md)
- [\[ECCV 2024\] VisFocus: Prompt-Guided Vision Encoders for OCR-Free Dense Document Understanding](visfocus_prompt-guided_vision_encoders_for_ocr-free_dense_document_understanding.md)

</div>

<!-- RELATED:END -->
