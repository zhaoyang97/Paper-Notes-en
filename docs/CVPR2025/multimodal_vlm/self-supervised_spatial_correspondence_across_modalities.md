---
title: >-
  [Paper Note] Self-Supervised Spatial Correspondence Across Modalities
description: >-
  [CVPR 2025][Multimodal VLM][Cross-modal pixel correspondence] Extends the Contrastive Random Walk (CRW) framework to cross-modal pixel-level correspondence. By simultaneously learning intra-modal and inter-modal cycle-consistent feature representations, it achieves cross-modal dense matching for RGB-Depth, RGB-Thermal, Photo-Sketch, etc., without requiring paired annotations, significantly outperforming existing methods.
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "Cross-modal pixel correspondence"
  - "Contrastive Random Walk"
  - "self-supervised learning"
  - "RGB-Depth matching"
  - "RGB-Thermal matching"
date: 2026-05-08
content_hash: 67f475c96e92e5d8
---

# Self-Supervised Spatial Correspondence Across Modalities

**Conference**: CVPR 2025  
**arXiv**: [2506.03148](https://arxiv.org/abs/2506.03148)  
**Code**: [https://ayshrv.com/cmrw](https://ayshrv.com/cmrw)  
**Area**: Multimodal VLM  
**Keywords**: Cross-modal pixel correspondence, Contrastive Random Walk, self-supervised learning, RGB-Depth matching, RGB-Thermal matching

## TL;DR

Extends the Contrastive Random Walk (CRW) framework to cross-modal pixel-level correspondence. By simultaneously learning intra-modal and inter-modal cycle-consistent feature representations, it achieves cross-modal dense matching for RGB-Depth, RGB-Thermal, Photo-Sketch, etc., without requiring paired annotations, significantly outperforming existing methods.

## Background & Motivation

When different sensors (RGB, depth, thermal) capture the same scene, the information stored in pixels is entirely different (intensity vs. depth vs. temperature), making cross-modal pixel correspondence a highly challenging problem. Traditional methods face several difficulties:

1. **Failure of photometric consistency assumptions**: The pixel value meanings of depth maps and RGB images are completely different, rendering photometric losses inapplicable.
2. **Difficulties in cross-modal translation**: Translating one modality to another first (e.g., via CycleGAN) and then matching, but translation itself is a hard problem (e.g., monocular depth estimation).
3. **Requirement for paired data**: Existing multimodal learning methods usually require spatially aligned multi-sensor paired data.

Key Insight: Self-supervised tracking methods (such as CRW) learn correspondence through cycle consistency—a point walking out in a video and walking back should return to its original position. The authors extend this idea from the temporal dimension to the modal dimension. However, directly optimizing cycle consistency across modalities suffers from convergence difficulties (due to the huge domain gap between modalities), therefore requiring auxiliary intra-modal random walks to stabilize training.

## Method

### Overall Architecture

A graph is constructed where nodes are patches in images from two modalities, and edges connect cross-modal patch pairs. A global matching Transformer is trained to assign transition probabilities for the random walk. The walker starts from one modality, travels through the other modality, and returns. The cross-modal correspondence is learned by maximizing the return probability (cycle consistency).

### Key Designs

1. **Cross-modal Matching Transformer**:
    - Function: Generates matching features for image patch pairs across different modalities.
    - Mechanism: Each modality uses an independent visual encoder to extract features $\phi(I_t^m) \in \mathbb{R}^{\frac{H}{c} \times \frac{W}{c} \times d}$ ($c=4$ downsampling). The features are concatenated with 2D positional encodings and fed into a shared Transformer (6 layers of self-attention + cross-attention + FFN), outputting association features $F_t^{m_1}$ and $F_{t+k}^{m_2}$. The transition matrix is $A_{t,t+k}^{m_1,m_2} = \text{softmax}(F_t^{m_1}(F_{t+k}^{m_2})^\top / \tau)$, where $\tau = \sqrt{d}$.
    - Design Motivation: The shared Transformer allows the model to process both intra-modal ($m_1 = m_2$) and inter-modal ($m_1 \neq m_2$) matching without modality-specific assumptions. Using independent encoders avoids forcing different modalities to share low-level features.

2. **Joint Cross-modal & Intra-modal CRW**:
    - Function: Addresses the convergence difficulties of direct cross-modal optimization.
    - Mechanism: The cross-modal loss utilizes a palindromic sequence $\{I_t^{m_1}, I_{t+k}^{m_2}, I_t^{m_1}\}$, defined as $\mathcal{L}_{\text{cross-crw}} = \mathcal{L}_{\text{CE}}(A_{t,t+k}^{m_1,m_2} A_{t+k,t}^{m_2,m_1}, T_f^b(I))$. The intra-modal loss uses data-augmented crop pairs $\{I_{\text{ori}}^{m_i}, I_{\text{aug}}^{m_i}, I_{\text{ori}}^{m_i}\}$, defined as $\mathcal{L}_{\text{intra-crw}} = \sum_{i=1}^{2} \mathcal{L}_{\text{CE}}(A_{\text{ori,aug}}^{m_i} A_{\text{aug,ori}}^{m_i}, T_f^b(I))$.
    - Design Motivation: Training strictly with cross-modal loss leads to local minima (since initial random features can cause arbitrary alignments across different modalities, such as mistakenly aligning RGB bright regions with depth high-value distant regions). Intra-modal CRW provides a "scaffold"—learning to match within the same modality first, before extending across modalities.

3. **Edge-Aware Smoothness Loss**:
    - Function: Encourages spatial coherence in the predicted flow field.
    - Mechanism: $\mathcal{L}_{\text{smooth}} = \mathbb{E}_p \sum_{d \in \{x,y\}} \exp(-\lambda_c I_d(p)) |\frac{\partial^2 \mathbf{f}_{s,t}(p)}{\partial d^2}|$, which penalizes second-order derivatives of the flow field, but lowers the penalty weight at edges with large image gradients.
    - Design Motivation: Only applied when the source image is RGB (since visual edges in RGB are reliable perceptual grouping cues). For depth/thermal as the source, the visual meaning of edges is different, making this constraint unsuitable.

### Loss & Training

The total loss is the sum of three terms: $\mathcal{L}_{\text{cross-crw}} + \mathcal{L}_{\text{intra-crw}} + \lambda_s \mathcal{L}_{\text{smooth}}$

A three-stage training strategy is adopted:
1. **Stage 1**: Intra-modal CRW only (RGB-RGB and Depth-Depth / Thermal-Thermal).
2. **Stage 2**: Introduce cross-modal CRW (RGB-Depth and Depth-RGB / RGB-Thermal and Thermal-RGB).
3. **Stage 3**: Introduce edge-aware smoothness loss.

For semantic matching tasks (photo-sketch, cross-style), DINOv2 is used as a shared encoder and fine-tuned to leverage pre-trained semantic priors.

## Key Experimental Results

### Main Results

| Dataset | Direction | Ours | RAFT (Supervised) | GMFlow (Supervised) | ARFlow (SSL) |
|--------|------|------|-----------|-------------|-------------|
| NYU Depth | RGB→D | 33.5 | 7.9 | 12.7 | 7.5 |
| NYU Depth | D→RGB | 34.3 | 1.3 | 12.5 | 7.4 |
| Thermal-IM | RGB→T | 41.8 | 5.6 | 3.8 | 12.5 |
| Thermal-IM | T→RGB | 47.9 | 0.9 | 2.6 | 13.2 |
| KAIST | RGB→T | 35.2 | 29.2 | 23.1 | 31.0 |
| KAIST | T→RGB | 34.1 | 7.4 | 22.3 | 30.4 |

### Ablation Study

| Configuration | NYU RGB→D | NYU D→RGB | Thermal RGB→T | Thermal T→RGB |
|------|-----------|-----------|---------------|---------------|
| Intra-modal CRW Only | 2.5 | 2.2 | 4.9 | 6.2 |
| Cross-modal CRW Only | 5.6 | 4.5 | 6.2 | 8.3 |
| Intra-modal + Cross-modal (No Smoothness) | 19.1 | 21.1 | 30.2 | 38.5 |
| Full (+ Smoothness) | 33.5 | 34.3 | 41.8 | 47.9 |

### Semantic Matching (Photo-Sketch, PSC6K)

| Method | PCK-5 | PCK-10 |
|------|-------|--------|
| DINOv2+NN | 11.48 | 31.66 |
| SD-DINO | 33.10 | 70.50 |
| PSCNet (SOTA, Supervised) | 57.92 | 84.72 |
| Ours (DINO+full) | 53.61 | 82.20 |

### Key Findings

- **Significant outperformance over all baselines on cross-modal matching**: NYU Depth RGB→D reaches 33.5 vs. the previous best of 12.7 (GMFlow), representing a >2.6x improvement.
- **Intra-modal pre-training is crucial**: Training only with cross-modal CRW is unstable (5.6 vs. 33.5). Intra-modal pre-training provides vital initialization.
- **Smoothness loss contributes significantly**: Increasing from 19.1 to 33.5 (+14.4), edge-aware smoothness plays a massive regularizing role in cross-modal matching.
- **Competitive performance with specialized methods on semantic matching**: On Photo-Sketch, it achieves close results to supervised SOTA (53.61 vs 57.92 PCK-5), and on cross-style, it outperforms all methods including the supervised GeoAwareSC.

## Highlights & Insights

- **Simple yet powerful**: Does not rely on any hand-designed photometric consistency metrics; instead, it uses pure self-supervised learning via cycle consistency, applicable to any modality pair.
- **Progressive design of the three-stage training strategy is elegant**: It first teaches the model to "look at the same modality", then "look across modalities", and finally introduces spatial constraints.
- **Highly data-efficient**: Does not require spatially or temporally aligned paired data, needing only frames from different moments and modalities of the same scene.
- **Strong generalizability**: The same framework handles four different cross-modal matching tasks without modification.

## Limitations & Future Work

- Only four modality pairs were validated; other modalities (e.g., SAR, MRI, ultrasound) might lack matching cues like occlusion boundaries.
- Left-right symmetry confusion occasionally occurs in cross-style matching (e.g., left and right limbs of animals).
- All current datasets include the RGB modality; matching purely between non-RGB modalities remains unexplored.
- Evaluation benchmarks rely on manually annotated keypoints or fake labels generated by trackers, which limits evaluation reliability.

## Related Work & Insights

- The natural extension of CRW from temporal tracking to cross-modal matching represents an elegant conceptual transfer.
- Unlike audio-visual correspondence (patch-level), this work focuses on pixel-level dense correspondence.
- Insight: This idea can be further extended to applications like dense matching between 3D point clouds and 2D images, as well as cross-modal SLAM.

## Rating

- Novelty: ⭐⭐⭐⭐ Extending CRW to cross-modality is a natural yet effective innovation, with intra-modal CRW auxiliary training serving as a key technical contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Four tasks, comprehensive ablation, and new benchmark construction (cross-style) with rich baselines.
- Writing Quality: ⭐⭐⭐⭐ Methods are clearly described, equations are concise, and diagrams are intuitive.
- Value: ⭐⭐⭐⭐ Provides a general self-supervised solution for cross-modal correspondence, directly applicable to multi-sensor fusion scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] BadVision: Stealthy Backdoor Attack in Self-Supervised Learning Vision Encoders for Large Vision Language Models](stealthy_backdoor_attack_in_self-supervised_learning_vision_encoders_for_large_v.md)
- [\[ECCV 2024\] Self-Adapting Large Visual-Language Models to Edge Devices across Visual Modalities](../../ECCV2024/multimodal_vlm/self-adapting_large_visual-language_models_to_edge_devices_across_visual_modalit.md)
- [\[ACL 2025\] Vision-Language Models Struggle to Align Entities across Modalities](../../ACL2025/multimodal_vlm/vision-language_models_struggle_to_align_entities_across_modalities.md)
- [\[CVPR 2025\] It's a (Blind) Match! Towards Vision-Language Correspondence without Parallel Data](its_a_blind_match_towards_vision-language_correspondence_without_parallel_data.md)
- [\[ACL 2025\] Performance Gap in Entity Knowledge Extraction Across Modalities in Vision Language Models](../../ACL2025/multimodal_vlm/performance_gap_in_entity_knowledge_extraction_across_modalities_in_vision_langu.md)

</div>

<!-- RELATED:END -->
