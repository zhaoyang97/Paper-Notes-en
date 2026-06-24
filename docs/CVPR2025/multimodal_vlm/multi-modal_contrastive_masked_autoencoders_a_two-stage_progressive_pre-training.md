---
title: >-
  [Paper Note] A Two-Stage Progressive Pre-training using Multi-Modal Contrastive Masked Autoencoders
description: >-
  [CVPR 2025][Multimodal VLM][RGB-D pre-training] This paper proposes a progressive two-stage pre-training strategy. In the first stage, patch-level contrastive learning is used to align cross-modal representations of RGB and depth modalities. In the second stage, joint training of masked autoencoding, diffusion-inspired denoising, and feature distillation is conducted. This achieves a +1.3% mIoU improvement over Mask3D on ScanNet semantic segmentation and reaches SOTA performa…
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "RGB-D pre-training"
  - "contrastive learning"
  - "masked autoencoders"
  - "denoising diffusion"
  - "knowledge distillation"
date: 2026-05-08
content_hash: 45c941a1fb363e17
---

# A Two-Stage Progressive Pre-training using Multi-Modal Contrastive Masked Autoencoders

**Conference**: CVPR 2025  
**arXiv**: [2408.02245](https://arxiv.org/abs/2408.02245)  
**Code**: None  
**Area**: Multimodal VLM  
**Keywords**: RGB-D pre-training, contrastive learning, masked autoencoders, denoising diffusion, knowledge distillation

## TL;DR
This paper proposes a progressive two-stage pre-training strategy. In the first stage, patch-level contrastive learning is used to align cross-modal representations of RGB and depth modalities. In the second stage, joint training of masked autoencoding, diffusion-inspired denoising, and feature distillation is conducted. This achieves a +1.3% mIoU improvement over Mask3D on ScanNet semantic segmentation and reaches SOTA performance on multiple RGB-D downstream tasks.

## Background & Motivation
1. **Background**: Self-supervised learning (SSL) has become a mainstream paradigm for visual pre-training, primarily divided into two main families: masked image modeling (MIM, e.g., MAE), which learns local spatial statistics, and contrastive learning (e.g., SimCLR/MoCo), which learns invariant representations across augmentations. For RGB-D data, MultiMAE and Mask3D have attempted to pre-train ViTs utilizing multimodal data.
2. **Limitations of Prior Work**: (a) MultiMAE requires semantic segmentation labels during pre-training and multimodal inputs during fine-tuning. (b) Although Mask3D encodes 3D priors through depth reconstruction, it fails to learn cross-modal relations to capture local context. (c) CoMAE proposes a hybrid framework but is restricted to small-scale datasets and requires all modalities during fine-tuning. (d) Existing methods generally fail to capture high-frequency components of the data.
3. **Key Challenge**: Contrastive learning and MAE learn complementary features—the former learns invariant discriminant representations, while the latter captures local spatial dependencies. Simply combining both within a single framework (e.g., pixel/patch-level contrastive learning + masking) is not trivial and performs poorly in RGB-D scenarios.
4. **Goal**: (1) How to effectively combine contrastive learning and MAE to learn complementary representations from RGB-D data? (2) How to learn high-frequency components of the data? (3) How to transfer knowledge between the two stages?
5. **Key Insight**: Inspired by the success of denoising in diffusion models, the authors hypothesize that denoising can assist encoders in extracting high-frequency features, thereby complementing the low-frequency reconstruction of MAE. Meanwhile, instead of a single-framework fusion, a two-stage progressive training is designed to prevent mutual interference between the two SSL paradigms.
6. **Core Idea**: Leverage the cross-modal alignment capability of contrastive learning and the multi-level feature learning capability of MAE + denoising in a staged manner, transferring knowledge across stages via feature distillation to achieve complementary fusion.

## Method

### Overall Architecture
The inputs are RGB-D image pairs. Modality-specific ViT encoders are used. In the first stage, patch-level InfoNCE contrastive loss is used to align the representation space of RGB and depth patches. In the second stage, the encoders are initialized with the first-stage weights. After random masking of both modalities, unmasked patches are passed through the encoders, concatenated with learnable mask tokens, and processed by lightweight decoders to: reconstruct the masked patches of the depth modality (MAE target), predict the noise of unmasked patches (denoising target), and align the second-stage global embeddings to the first-stage global embeddings (distillation target).

### Key Designs

1. **Patch-Level Cross-Modal Contrastive Learning (Stage 1)**:

    - **Function**: Align representations of RGB and depth modalities at the patch level to learn local cross-modal correspondences.
    - **Mechanism**: For RGB-D pairs in a batch, RGB and depth images are passed through their respective ViT encoders to obtain patch-level features $\mathbf{z}_i^{rgb}$ and $\mathbf{z}_i^{depth}$. The InfoNCE loss is used: $\mathcal{L}_{PNCE} = -\frac{1}{N}\sum_{i=1}^N \log \frac{\exp(s_{i,i}/\tau)}{\sum_{k\neq i}\exp(s_{i,k}/\tau) + \exp(s_{i,i}/\tau)}$, where $s_{i,j} = \|\mathbf{z}_i^{rgb}\|^T\|\mathbf{z}_i^{depth}\|$. The loss is computed bidirectionally and then averaged.
    - **Design Motivation**: Instance-level contrastive learning has limited utility for dense prediction tasks (such as semantic segmentation) because it only learns high-level semantics while neglecting local discriminative features. Patch-level contrastive learning can capture cross-modal local context correspondences.

2. **Joint Masked Autoencoding + Denoising Training (Core of Stage 2)**:

    - **Function**: Learn fine-grained spatial features (MAE) while extracting high-frequency components (denoising).
    - **Mechanism**: MAE part: Randomly mask patches in both modalities, pass only unmasked patches through the encoders, and concatenate with learnable mask tokens to decode and reconstruct the masked regions of the depth modality, with loss $\mathcal{L}_{depth} = \frac{1}{n}\sum\|\mathbf{M}_i^{depth} \circ (\mathbf{x}_i^{depth} - \hat{\mathbf{x}}_i^{depth})\|_2^2$. Denoising part: Add Gaussian noise to the depth inputs $\mathbf{x}_i^{depth} \leftarrow \mathbf{x}_i^{depth} + \sigma_i^{depth}\mathbf{e}_i^{depth}$. The noise level $\sigma$ is mapped to an embedding via sinusoidal positional encoding + MLP, and added to the encoded representation (similar to the timestep embedding in diffusion models). The decoder simultaneously reconstructs masked patches and predicts the noise in unmasked patches: $\mathcal{L}_{denoise} = \frac{1}{n}\sum\|(1-\mathbf{M}_i^{depth})\circ(\sigma_i^{depth}\mathbf{e}_i^{depth} - \hat{\mathbf{x}}_i^{depth})\|_2^2$.
    - **Design Motivation**: The reconstruction task of MAE primarily captures low-frequency spatial statistics, while the denoising task forces the encoders to distinguish noise from signal, thereby extracting high-frequency components from the data. Denoising utilizes the reconstruction outputs of unmasked patches that were otherwise "wasted" in standard MAE, introducing almost zero computational overhead.

3. **Global Feature Distillation (Cross-Stage Knowledge Transfer)**:

    - **Function**: Transfer global cross-modal correspondence knowledge learned in the first stage to the second-stage model.
    - **Mechanism**: Apply max pooling to the output of the second-stage encoder to obtain global embedding $\mathbf{f_2}$ and align it with the global embedding $\mathbf{f_1}$ of the frozen first-stage model via a smooth $\ell_1$ loss. Distillation is performed separately for RGB and depth modalities, and the final distillation loss is the sum of both modalities.
    - **Design Motivation**: A pure MAE + denoising stage might lose the global cross-modal discriminative information acquired through first-stage contrastive learning. Feature distillation allows retaining this knowledge without restricting the flexibility of the second stage.

### Loss & Training
First stage: $\mathcal{L}_{stage1} = \mathcal{L}_{PNCE}$. Second stage: $\mathcal{L}_{stage2} = \alpha\mathcal{L}_{depth} + \beta\mathcal{L}_{denoise} + \gamma\mathcal{L}_{distill}$. The second-stage encoder is initialized with the first-stage weights. The first stage is pre-trained on ImageNet, and the second stage continues pre-training on ScanNet (2.5M RGB-D frames) or SUN RGB-D. Only the depth modality is reconstructed (experiments indicate that reconstructing both RGB and depth does not improve performance).

## Key Experimental Results

### Main Results

| Dataset | Task | Metric | Ours | Mask3D | MultiMAE | MAE | Gain vs Mask3D |
|--------|------|------|------|--------|----------|-----|--------------|
| ScanNet | Semantic Segmentation | mIoU | **67.5** | 66.2 | 65.1 | 64.8 | +1.3 |
| SUN RGB-D | Semantic Segmentation | mIoU | **48.7** | 47.4 | 47.1 | 47.3 | +1.3 |
| NYUv2 | Depth Estimation | δ₁ | **87.1** | 85.4 | 85.3 | 85.1 | +1.7 |
| ScanNet | Instance Segmentation | AP | **23.7** | 22.8 | 22.4 | 20.7 | +0.9 |

ViT-L scaling:

| Method | ScanNet mIoU (ViT-L) |
|------|---------------------|
| MAE | 68.2 |
| MultiMAE | 69.3 |
| **Ours** | **70.8** |

### Ablation Study

| Contrastive Learning | Reconstruction | Denoising | Distillation | ScanNet mIoU |
|---------|------|------|------|-------------|
| ✓ | ✗ | ✗ | ✗ | 63.4 |
| ✓ | ✓ | ✗ | ✗ | 66.3 |
| ✓ | ✓ | ✗ | ✓ | 66.5 |
| ✓ | ✓ | ✓ | ✗ | 67.0 |
| ✓ | ✓ | ✓ | ✓ | **67.5** |

Ablation of the denoising component:

| Configuration | mIoU | Description |
|------|------|------|
| No Noise | 66.5 | No denoising component |
| Only Noise Added | 66.9 | Noise added to input but no noise prediction |
| Full Denoising | **67.5** | Noise prediction + positional encoding |

### Key Findings
- **Significant complementarity between contrastive learning and reconstruction**: Contrastive learning alone yields 63.4 -> jumps to 66.3 with reconstruction, a 2.9 percentage point increase, validating the hypothesis that the two SSL paradigms learn complementary features.
- Denoising contributes +0.7 mIoU, where adding noise alone contributes +0.4, and full denoising (including noise prediction and positional encoding) contributes an additional +0.6, demonstrating that denoising inspired by diffusion models indeed learns high-frequency features.
- Reconstructing both RGB and depth simultaneously is not better than reconstructing depth alone (67.5 vs 66.9), because the 3D prior is primarily encoded via depth prediction.
- A high masking rate (80% × 80%) yields the best results, consistent with MAE findings, as a high masking rate makes the task more challenging.
- The advantage is more pronounced in low-data scenarios: using only 60% of the training data slightly outperforms MAE using 100% of the data.

## Highlights & Insights
- **Denoising pre-training inspired by diffusion models** is the most innovative contribution of this paper: bringing the denoising concept from diffusion models into self-supervised representation learning and leveraging sinusoidal positional encoding of noise levels to help the model distinguish noise from signal. This approach can be transferred to any pre-training scenario where learning high-frequency features is desired.
- **Two-stage progressive design** is more rational than single-framework fusion: empirical evidence shows that hybrid contrastive + MAE in a single framework behaves poorly in RGB-D scenarios, whereas progressive alignment-then-reconstruction benefits from the strengths of both.
- Cleverly leverages the "wasted" unmasked patches in MAE to compute denoising loss, incurring almost zero additional computational overhead.

## Limitations & Future Work
- Validated only in RGB-D scenarios, without extension to other multimodal combinations (e.g., RGB-Point Cloud, RGB-Thermal).
- Patch-level contrastive learning did not explore pixel-level; the authors mention that pixel-to-point contrast might be more effective but leave it for future work.
- While effective on small-scale datasets (only 10k images for SUN RGB-D) in the second stage, scaling behavior under large-scale data is not fully explored.
- The sensitivity analysis for the choice of pre-training hyperparameters ($\alpha$, $\beta$, $\gamma$) is not sufficiently detailed.
- Whether a systematic study was conducted on the selection of the denoising noise level range $[0, \sigma_{max}]$.

## Related Work & Insights
- **vs Mask3D**: Mask3D only uses depth reconstruction to encode 3D priors without learning cross-modal relationships; in contrast, this paper explicitly aligns RGB-depth via contrastive learning + reconstruction + denoising, which is more comprehensive.
- **vs MultiMAE**: MultiMAE requires semantic labels for pre-training and multimodal inputs during fine-tuning; this paper only requires RGB-D for pre-training and uses only RGB for fine-tuning, offering stronger practicality.
- **vs CoMAE**: CoMAE also blends contrastive and MAE but is restricted to small-scale scenarios and requires dual-modality inputs for fine-tuning. The proposed progressive design is effective across both large-scale and small-scale datasets.
- The idea of using denoising as a pre-training objective warrants attention and may cross-pollinate with topics like Diffusion Pre-training.

## Rating
- Novelty: ⭐⭐⭐⭐ Introducing diffusion denoising into SSL pre-training is a novel point, and the progressive two-stage design has theoretical grounding.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers semantic segmentation, depth estimation, instance segmentation, and low-data scenarios, with comprehensive and thorough ablation studies.
- Writing Quality: ⭐⭐⭐⭐ The methodology is clearly described, though the abundance of equations may slightly affect readability.
- Value: ⭐⭐⭐⭐ The progressive multi-paradigm pre-training strategy and denoising pre-training target can be widely applied to multimodal SSL.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] SCAN: Bootstrapping Contrastive Pre-training for Data Efficiency](../../ICCV2025/multimodal_vlm/scan_bootstrapping_contrastive_pre-training_for_data_efficiency.md)
- [\[CVPR 2025\] Post-pre-training for Modality Alignment in Vision-Language Foundation Models](post-pre-training_for_modality_alignment_in_vision-language_foundation_models.md)
- [\[CVPR 2025\] GeoMM: On Geodesic Perspective for Multi-Modal Learning](geomm_on_geodesic_perspective_for_multi-modal_learning.md)
- [\[CVPR 2026\] PowerCLIP: Powerset Alignment for Contrastive Pre-Training](../../CVPR2026/multimodal_vlm/powerclip_powerset_alignment_for_contrastive_pre-training.md)
- [\[CVPR 2025\] Multimodal Autoregressive Pre-training of Large Vision Encoders](multimodal_autoregressive_pre-training_of_large_vision_encoders.md)

</div>

<!-- RELATED:END -->
