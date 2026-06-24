---
title: >-
  [Paper Note] unMORE: Unsupervised Multi-Object Segmentation via Center-Boundary Reasoning
description: >-
  [ICML2025][Segmentation][unsupervised segmentation] This paper proposes unMORE, which achieves unsupervised multi-object segmentation by learning a three-layer object-centric representation (existence, center field, and boundary distance field) and designing a network-free multi-object reasoning module, substantially outperforming all unsupervised methods on six datasets, including COCO.
tags:
  - "ICML2025"
  - "Segmentation"
  - "unsupervised segmentation"
  - "object-centric representation"
  - "center field"
  - "boundary distance field"
  - "multi-object reasoning"
date: 2026-05-08
content_hash: a9180292482b68e1
---

# unMORE: Unsupervised Multi-Object Segmentation via Center-Boundary Reasoning

**Conference**: ICML2025  
**arXiv**: [2506.01778](https://arxiv.org/abs/2506.01778)  
**Code**: [GitHub](https://github.com/vLAR-group/unMORE)  
**Area**: Unsupervised Segmentation / Multi-Object Discovery  
**Keywords**: unsupervised segmentation, object-centric representation, center field, boundary distance field, multi-object reasoning

## TL;DR

This paper proposes unMORE, which achieves unsupervised multi-object segmentation by learning a three-layer object-centric representation (existence, center field, and boundary distance field) and designing a network-free multi-object reasoning module, substantially outperforming all unsupervised methods on six datasets, including COCO.

## Background & Motivation

Unsupervised multi-object segmentation aims to discover and segment multiple objects from a single image without relying on manual annotations. Existing methods primarily fall into two categories:

**Slot-based methods** (e.g., SlotAttention): Rely on image reconstruction targets to learn object representations, which are effective on synthetic data but difficult to scale to complex real-world scenes.

**Self-supervised feature distillation methods** (e.g., TokenCut, CutLER, CuVLER): Utilize object localization cues from DINO/v2 pretrained features to discover multiple objects, but still suffer from the **under-segmentation** problem—tending to group adjacent objects into a single one.

The key challenges are: (1) ill-defined objectness; and (2) the lack of an effective multi-object search mechanism. Drawing an analogy to the ability of human infants to learn the concept of objects from single-object images and then identify multiple objects in complex scenes, this paper proposes a two-stage pipeline.

## Method

### Overall Architecture

unMORE is a two-stage pipeline:

- **Stage 1**: Train an Objectness Network on ImageNet single-object images to learn a three-layer object-centric representation.
- **Stage 2**: Leverage the trained (frozen) Objectness Network to discover multiple objects in scene images using a network-free multi-object reasoning module.

### Data Preparation

Using the VoteCut method from CuVLER, DINO/v2 patch features are extracted for each ImageNet image, followed by constructing an affinity matrix, performing Normalized Cut, and selecting the most salient foreground mask as the coarse object mask.

### Three-layer Object-Centric Representation

**1. Object existence score $f^e$**: Binary classification, where 1 indicates the image contains a valid object, and 0 otherwise. The largest background region cropped from ImageNet images is used as negative samples.

**2. Object center field $\boldsymbol{f}^c \in \mathbb{R}^{H \times W \times 2}$**: Each pixel inside the mask is assigned a unit vector pointing to the object's bounding box center $[C_h, C_w]$, while pixels outside the mask are assigned zero vectors:

$$\boldsymbol{f}^c_{(h,w)} = \begin{cases} \frac{[h,w] - [C_h, C_w]}{\|[h,w] - [C_h, C_w]\|}, & \text{if } M_{(h,w)}=1 \\ [0,0], & \text{otherwise} \end{cases}$$

**3. Object boundary distance field $\boldsymbol{f}^b \in \mathbb{R}^{H \times W \times 1}$**: The signed distance field (positive inside the mask, negative outside, zero on the boundary) is first computed, and then normalized for foreground and background separately:

$$\boldsymbol{f}^b_{(h,w)} = \begin{cases} S_{(h,w)} / \max(\boldsymbol{S} * \boldsymbol{M}), & \text{if } M_{(h,w)}=1 \\ S_{(h,w)} / |\min(\boldsymbol{S} * (\boldsymbol{1}-\boldsymbol{M}))|, & \text{otherwise} \end{cases}$$

Key property: The maximum signed distance value $S_{(\hat h,\hat w)} = 1 / \|[\partial \boldsymbol{f}^b / \partial h, \partial \boldsymbol{f}^b / \partial w]\|$ can be back-calculated through the gradient of $\boldsymbol{f}^b$, which is used for boundary reasoning.

### Objectness Network Architecture and Training

- **Existence branch**: ResNet50 binary classifier $\rightarrow$ predicting $\tilde{f^e}$
- **Center field + boundary distance field branch**: DPT-Large + two CNN heads $\rightarrow$ predicting $\tilde{\boldsymbol{f}^c}$ and $\tilde{\boldsymbol{f}^b}$, respectively
- **Total loss**:

$$\ell = \text{CE}(\tilde{f^e}, f^e) + \ell_2(\tilde{\boldsymbol{f}^c}, \boldsymbol{f}^c) + \ell_1(\tilde{\boldsymbol{f}^b}, \boldsymbol{f}^b)$$

### Multi-Object Reasoning Module (Network-Free)

**Step 0 — Initial proposal generation**: Uniformly and randomly initialize $T$ bounding box proposals on the scene image, and uniformly scale them to $128 \times 128$.

**Step 1 — Existence verification**: Query $f^e_p$ for each proposal, and discard those below the threshold $\tau^e$.

**Step 2 — Center reasoning**: Query the center field $\boldsymbol{f}^c_p$, and apply a predefined $5 \times 5$ anti-center kernel convolution to generate the **anti-center map** $\boldsymbol{f}^{ac}_p$. If the maximum anti-center value exceeds $\tau^c$, indicating containing $\ge 2$ crowded objects, split the proposal into four sub-proposals (up/down/left/right) at that position, and return to Step 1; otherwise, segment using connected components.

**Step 3 — Boundary reasoning**: For single-object proposals, obtain the maximum boundary distance values from the four edges, back-calculate actual pixel distances using gradients, and iteratively update the coordinates of the four corners of the proposals (expansion for positive values, shrinkage for negative values) until converging to tight bounding boxes.

Finally, apply NMS to remove redundant converged proposals, and take the union of the positive boundary distance field region and non-zero center field region as the final object mask.

### Optional: Detector Training

Train a class-agnostic detector (Mask R-CNN) using the discovered objects as pseudo-labels, which constitutes the full version of unMORE.

## Key Experimental Results

### Main Results on COCO* val (With 197 Extra Class Annotations)

| Method | Type | AP50^box | AP^box | AP50^mask | AP^mask |
|------|------|----------|--------|-----------|---------|
| VoteCut | Learning-free | 10.8 | 5.5 | 9.5 | 4.6 |
| DINOSAUR | SlotAtt | 2.0 | 0.6 | 1.1 | 0.3 |
| **unMORE_disc** | **Obj.Net** | **19.1** | **10.1** | **17.8** | **9.5** |
| CutLER | Det.×3 | 26.0 | 14.7 | 22.7 | 11.8 |
| CuVLER | Det.×2 | 28.0 | 15.5 | 24.4 | 12.6 |
| **unMORE** | **Obj.Net+Det.×1**| **32.6** | **18.0** | **29.6** | **15.5** |

- unMORE_disc (without detector training) already surpasses all learning-free baselines by approximately **2$\times$ AP**.
- unMORE (trained with a single-round detector) achieves a +2.5 gain in AP^box and a +2.9 gain in AP^mask compared to CuVLER.

### Zero-Shot Cross-Dataset Generalization (Summary of Table 2)

| Dataset | CutLER AP50^box | CuVLER AP50^box | **unMORE AP50^box** |
|--------|-----------------|-----------------|---------------------|
| COCO20K | 22.4 | 24.1 | **25.9** |
| LVIS | 8.5 | 8.9 | **10.4** |

The proposed method also achieves the best unsupervised results on KITTI, VOC, Object365, and OpenImages.

### Performance in Crowded Scenes

While all baselines suffer performance degradation on crowded images, unMORE significantly outperforms them due to its splitting mechanism during center reasoning, which effectively separates adjacent objects.

## Highlights & Insights

1. **Exquisite design of the three-layer representation**: The hierarchy of existence $\rightarrow$ center $\rightarrow$ boundary systematically answers "whether, where, and what shape", mimicking the human object perception process.
2. **Network-free reasoning module**: Step 2 and Step 3 involve no learnable parameters, making iterative reasoning entirely dependent on the geometric properties of the center and boundary distance fields.
3. **Anti-center kernel design**: Convolving a $5 \times 5$ outwardly radiating unit-vector kernel with the center field elegantly detects split points between crowded objects.
4. **Gradient properties of the boundary distance field**: The inverse of the gradient after normalization is capable of recovering the object size, which is directly applied to proposal expansion and contraction.
5. **COCO* complementary annotations**: Manually supplementing 197 class annotations to COCO val provides a fairer evaluation of unsupervised methods.

## Limitations & Future Work

1. **Dependence on pretrained feature quality**: The coarse masks are generated from DINO/v2 + VoteCut. If pretrained features perform poorly in localizing objects within specific domains (e.g., medical imaging), performance will be limited.
2. **Inference efficiency**: The iterative reasoning in Step 2 and Step 3 requires multiple forward passes for each proposal, introducing considerable computational overhead when there are numerous proposals.
3. **Class-agnostic detection only**: It does not provide category information, requiring an additional classifier for semantic segmentation.
4. **Threshold sensitivity**: Hyperparameters such as $\tau^e$ and $\tau^c$ require careful tuning, and the paper does not fully discuss their robustness across different datasets.
5. **Sensitivity of boundary distance field to occlusion**: In the presence of severe occlusion, incomplete masks may lead to biased estimations of the boundary distance field.

## Related Work & Insights

- **CutLER / CuVLER**: Strong baselines but rely on multi-round self-training detectors; unMORE surpasses them with only a single round.
- **SlotAttention / DINOSAUR**: Reconstruction-driven object discovery method, which fails significantly in real-world scenes.
- **DeepSDF / Park et al.**: Signed Distance Field (SDF) has succeeded in 3D reconstruction, and this work is the first to adopt it for 2D unsupervised object discovery.
- **Insight**: Integrating explicit learning of object-centric representations with parameter-free iterative reasoning might be highly generalizable to 3D point cloud scenarios.

## Rating

- Novelty: ⭐⭐⭐⭐ — The combination of three-layer representation and network-free reasoning is highly novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Comprehensive evaluation across 6 datasets, including ablation studies and COCO* extra annotations.
- Writing Quality: ⭐⭐⭐⭐ — Clearly structured with excellent illustrations.
- Value: ⭐⭐⭐⭐ — New SOTA for unsupervised segmentation, with a breakthrough in crowded scenes.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Pixel-Level Reasoning Segmentation via Multi-turn Conversations](../../ACL2025/segmentation/pixel-level_reasoning_segmentation_via_multi-turn_conversations.md)
- [\[CVPR 2025\] M3-VOS: Multi-Phase, Multi-Transition, and Multi-Scenery Video Object Segmentation](../../CVPR2025/segmentation/m3-vos_multi-phase_multi-transition_and_multi-scenery_video_object_segmentation.md)
- [\[ECCV 2024\] Unsupervised Moving Object Segmentation with Atmospheric Turbulence](../../ECCV2024/segmentation/unsupervised_moving_object_segmentation_with_atmospheric_turbulence.md)
- [\[ICCV 2025\] Ensemble Foreground Management for Unsupervised Object Discovery](../../ICCV2025/segmentation/ensemble_foreground_management_for_unsupervised_object_discovery.md)
- [\[ICCV 2025\] Beyond Single Images: Retrieval Self-Augmented Unsupervised Camouflaged Object Detection](../../ICCV2025/segmentation/beyond_single_images_retrieval_self-augmented_unsupervised_camouflaged_object_de.md)

</div>

<!-- RELATED:END -->
