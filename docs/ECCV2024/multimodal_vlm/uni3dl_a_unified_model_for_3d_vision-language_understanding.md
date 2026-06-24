---
title: >-
  [Paper Note] Uni3DL: Unified Model for 3D and Language Understanding
description: >-
  [ECCV 2024][Multimodal VLM][3D Vision-Language Unified Model] This paper proposes Uni3DL, a unified 3D vision-language model operating directly on point clouds. By learning task-agnostic semantic/mask outputs through a Query Transformer and then combining multiple functional heads using a Task Router, it achieves functional unification across six tasks: semantic segmentation, instance segmentation, object detection, visual grounding, 3D caption generation…
tags:
  - "ECCV 2024"
  - "Multimodal VLM"
  - "3D Vision-Language Unified Model"
  - "Point Cloud Understanding"
  - "Query Transformer"
  - "Multi-Task Learning"
  - "Functional Unification"
date: 2026-05-08
content_hash: 9f8d5d4ee78f35bc
---

# Uni3DL: Unified Model for 3D and Language Understanding

**Conference**: ECCV 2024  
**arXiv**: [2312.03026](https://arxiv.org/abs/2312.03026)  
**Code**: [https://uni3dl.github.io/](https://uni3dl.github.io/)  
**Area**: Multimodal VLM / 3D Vision  
**Keywords**: 3D Vision-Language Unified Model, Point Cloud Understanding, Query Transformer, Multi-Task Learning, Functional Unification

## TL;DR

This paper proposes Uni3DL, a unified 3D vision-language model operating directly on point clouds. By learning task-agnostic semantic/mask outputs through a Query Transformer and then combining multiple functional heads using a Task Router, it achieves functional unification across six tasks: semantic segmentation, instance segmentation, object detection, visual grounding, 3D caption generation, and text-to-3D retrieval. Its performance reaches or exceeds the task-specific SOTA for each task.

## Background & Motivation

**Background**: 3D perception technologies form the foundation of applications like robotic navigation, autonomous driving, and virtual reality. Currently, the 3D perception domain is populated by a large number of task-specific models (semantic segmentation, instance segmentation, visual grounding, captioning, etc.), with each task designed with independent architectures and training routines.

**Limitations of Prior Work**:
   - **Incomplete task coverage**: Existing 3D vision-language unified models (e.g., PointCLIP v2, ULIP, 3D-VisTA) support a limited variety of tasks, with dense prediction tasks (semantic/instance segmentation) receiving particularly sparse attention.
   - **Reliance on multi-view images**: Most methods require projecting point clouds into multi-view 2D images for processing, which leads to a loss of 3D geometric information and increases model complexity.
   - **Need for task-specific fine-tuning**: Although models like 3D-VisTA perform vision-language pre-training, downstream tasks still require independent task-specific heads.

**Key Challenge**: While unified models in the 2D domain (e.g., CLIP, X-Decoder, Mask2Former) have achieved massive success, the transfer from 2D to 3D faces two main obstacles: major 2D/3D architectural discrepancies and a shortage of large-scale 3D pre-training data. Existing 3D unified models either support limited tasks or heavily rely on 2D projections.

**Goal**: Design a unified model that operates directly on point clouds, supports as many 3D vision and vision-language tasks as possible, and achieves cross-task parameter sharing and seamless task decomposition.

**Key Insight**: Borrow from the "functional unification" paradigm of the 2D domain rather than the I/O-unified seq2seq approach. By generating general semantic and mask representations through a query-based transformer, different tasks are accommodated simply by combining different functional heads.

**Core Idea**: Utilize a functionally unified architecture of Query Transformer + Task Router to achieve unified modeling of six major 3D vision-language tasks directly on point clouds.

## Method

### Overall Architecture

Uni3DL comprises four core modules:

- **Text Encoder**: Based on a CLIP tokenizer + transformer, this module extracts text features $\mathbf{F}_T \in \mathbb{R}^{L_T \times C}$.
- **Point Encoder**: Based on a MinkowskiEngine sparse 3D convolutional U-Net (Res16UNet34C), this module takes colored point clouds $\mathbf{P} \in \mathbb{R}^{N_0 \times 6}$ as input and outputs multi-level voxel features $\{\mathbf{V}_s\}_{s=1}^{S}$.
- **Query Transformer Module**: The core module where learnable latent queries $\mathbf{F}_Q \in \mathbb{R}^{Q \times C}$ and text queries attend to voxel features via cross-attention, generating task-agnostic mask outputs $\mathbf{O}_m$ and semantic outputs $\mathbf{O}_s$.
- **Task Router**: Contains multiple functional heads (classification, mask, grounding, text generation, text-3D matching). Different tasks are completed by combining different heads.

The overall formulation is: $\mathbf{O}_m, \mathbf{O}_s = \mathcal{D}(\langle \mathbf{F}_Q, \mathbf{F}_T \rangle, \mathbf{V})$

### Key Designs

1. **Point Cloud Encoder (3D U-Net)**:

    - **Function**: Extracts multi-scale voxel features from colored point clouds.
    - **Mechanism**: The input point cloud is quantized into $N_0$ voxels and processed through $S$ stages of convolution-downsampling-deconvolution-upsampling to obtain feature maps of various resolutions $\{\mathbf{V}_s \in \mathbb{R}^{N_s \times C}\}_{s=1}^{S}$. Features from the final scale $\mathbf{V}_S$ serve as point embeddings for per-point mask computation, while the intermediate features $\{\mathbf{V}_s\}_{s=1}^{S-1}$ are fed into the Query Transformer to enhance the queries.
    - **Design Motivation**: The U-Net structure preserves multi-scale information, facilitating tasks that require both global semantics (e.g., classification) and localized precision (e.g., instance segmentation). Initialization with pre-trained Mask3D weights accelerates convergence.

2. **Query Transformer Module (Core Innovation)**:

    - **Function**: Fuses latent queries, text queries, and 3D visual features to generate unified semantic and mask representations.
    - **Mechanism**: A decoder comprised of $L=15$ transformer layers, each containing:
        - **Masked Cross-Attention**: Queries attend to voxel features using the masked attention strategy of Mask2Former, forcing each query to focus only on the voxel regions corresponding to the predicted mask from the previous layer: $\langle \hat{\mathbf{F}}_Q^l, \hat{\mathbf{F}}_T^l \rangle = \text{Cross-Att}(\langle \mathbf{F}_Q^{l-1}, \mathbf{F}_T^{l-1} \rangle, \mathbf{V}_s)$.
        - **Self-Attention**: Interaction among queries, enabling latent queries and text queries to mutually reinforce each other.
        - **FFN**: Standard Feed-Forward Network.
    - **Voxel Sampling**: To handle scenes with varying numbers of points, a fixed number of voxels is sampled from each feature layer during training to ensure efficient batch training.
    - **Design Motivation**: Masked attention improves object localization (focusing only on relevant regions). Latent queries capture object-level information, while text queries capture textual semantics—both are jointly optimized in a single decoder.

3. **Task Router (Key to Functional Unification)**:

    - **Function**: Derives task-specific results from unified semantic and mask outputs by combining different functional heads.
    - **Routing Strategy** (Table 2):
        - Semantic Segmentation = Classification Head + Mask Head
        - Instance Segmentation = Classification Head + Mask Head
        - Grounded Segmentation = Mask Head + Grounding Head
        - 3D Caption Generation = Text Generation Head
        - Text-to-3D Retrieval = Text-3D Matching Head
    - **Design Motivation**: Different tasks share underlying encoder and decoder parameters and only differ in their final routing strategy, achieving true parameter sharing and task decomposition.

4. **Functional Head Details**:

    - **Object Classification Head**: Takes the first $Q$ semantic outputs $\mathbf{O}_s$ and passes all $K+1$ class names through the text encoder to obtain class embeddings $\mathbf{C}_{emb}$. The classification probability is computed as $\mathbf{O}_c = \mathbf{O}_s \cdot \mathbf{C}_{emb}^T$ (open-vocabulary classification).
    - **Mask Head**: Computes the dot product between the mask outputs and full-resolution voxel features to yield per-point masks for each query: $\mathbf{O}_m = \mathbf{O}_m \cdot \mathbf{V}_S^T$.
    - **Grounding Head**: Computes similarity scores between text embeddings and object embeddings: $\mathbf{S}_t = \text{Softmax}(e^\eta \cdot \mathbf{T}_{emb} \cdot \mathbf{O}_s^T)$, where $\eta$ is a learnable scaling parameter. Hungarian matching is used for alignment. An additional lightweight MLP predicts the object categories mentioned in the text descriptions.
    - **Text Generation Head**: Computes an affinity matrix $\mathbf{S}_{cap} \in \mathbb{R}^{L_T \times V}$ using the last $L_T$ semantic outputs and vocabulary token embeddings. Causal masking is used during training, and autoregressive generation is performed during inference.
    - **Text-3D Matching Head**: Uses the last semantic token as the shape embedding to compute a contrastive loss with the text embeddings.

### Loss & Training

The total loss is the sum of five task-specific losses:

$$\mathcal{L} = \mathcal{L}_{cls} + \mathcal{L}_{mask} + \mathcal{L}_{grd} + \mathcal{L}_{cap} + \mathcal{L}_{ret}$$

- $\mathcal{L}_{cls} = \lambda_{cls} \cdot \text{CE}(\mathbf{O}_c, C_{gt})$: Classification cross-entropy.
- $\mathcal{L}_{mask} = \lambda_{bce} \cdot \text{BCE} + \lambda_{dice} \cdot \text{DICE}$: Mask binary cross-entropy + Dice loss.
- $\mathcal{L}_{grd} = \lambda_{gc} \cdot \mathcal{L}_{gc} + \mathcal{L}_{gtxt} + \mathcal{L}_{gmask}$: Grounding matching + Category existence + Mask losses.
- $\mathcal{L}_{cap} = \lambda_{cap} \cdot \text{CE}(\mathbf{S}_{cap}, y_{cap})$: Caption generation loss.
- $\mathcal{L}_{ret} = \lambda_{ret} \cdot \text{CL}(\mathbf{S}_{ret}, y_{ret})$: CLIP-style contrastive learning loss.

Weight settings: $\lambda_{cls}=2.0$, $\lambda_{bce}=5.0$, $\lambda_{dice}=5.0$, $\lambda_{gc}=0.4$, $\lambda_{cap}=\lambda_{ret}=2.0$.

**Training Pipeline**:
- Pre-training: Jointly trained on three datasets—ScanNet(v2) + ScanRefer + Cap3D Objaverse—for 50 epochs, taking approximately 20 hours on 4×A100 GPUs.
- Fine-tuning: Downstream tasks are individually fine-tuned for 20-30 epochs with a learning rate of 1e-4 or 1e-5.
- 150 latent queries + 1 scene-level query; voxel sizes: 0.02m for 3D scans, 0.01 for normalized 3D shapes.

## Key Experimental Results

### Main Results

Performance of Uni3DL across 6 major tasks (Table 3):

| Task | Dataset | Metric | Ours | Prev. SOTA | Gain |
|------|--------|------|--------|----------|------|
| Semantic Segmentation | ScanNet Val | mIoU | **76.2** | 75.6 (Swin3D†) | +0.6 |
| Semantic Segmentation | S3DIS Area5 | mIoU | 72.7 | 73.0 (Swin3D†) | -0.3 |
| Object Detection | ScanNet Val | bAP50 | **67.7** | 63.9 (Mask-Att-Free†) | +3.8 |
| Instance Segmentation | ScanNet Val | mAP | **60.9** | 58.4 (Mask-Att-Free†) | +2.5 |
| Grounded Segmentation | ScanRefer | mIoU/Acc@0.25 | 32.3/39.4 | 27.8/37.5 (TGNN) | +4.5/+1.9 |
| 3D Captioning | Cap3D | B-1/R/M | **31.6/33.1/14.4** | 12.6/15.0/16.0 | B-1 exceeds by 19+ |
| Text-to-3D Retrieval | Text2Shape | R@1/R@5 | 5.8/19.7 | 5.1/17.2 (P2W) | +0.7/+2.5 |

**Key Findings**:
- Semantic segmentation achieves the best mIoU of 76.2 on ScanNet, even exceeding Swin3D† which uses extra data.
- The advantage in 3D captioning is particularly pronounced, with BLEU-1 and ROUGE-L exceeding the previous SOTA by over 20%.
- Outperforms TGNN (the only competitor) by a large margin on grounded segmentation.
- All tasks are served by a single unified architecture, whereas baseline competitors are task-specific designs.

### Ablation Study

**Impact of Pre-training** (Table 4):

| Configuration | Semantic Segmentation mIoU | Instance Segmentation mAP50 | Grounding Acc@0.25 | Retrieval R@1 |
|------|--------------|---------------|-------------------|---------|
| From scratch | 72.3 | 61.7 | 33.8 | 2.4 |
| Fine-tuned after pre-training | **76.2** | **65.3** | **39.4** | **4.6** |
| Gain | +3.9 | +3.6 | +5.6 | +2.2 |

**Pre-training Task Combinations** (Table 5):

| Configuration | Grounding Acc@0.25 | Captioning R | Retrieval S2T R@1 |
|------|-------------------|-------------|-------------------|
| Full model | 37.8 | 18.6 | 8.0 |
| w/o Instance Segmentation | 33.8 (-4.0) | 17.8 | 4.0 |
| w/o Retrieval | 37.7 | 15.8 (-2.8) | n/a |
| w/o Captioning | 37.9 | n/a | 3.5 (-4.5) |

### Key Findings

- Pre-training significantly benefits all downstream tasks, especially grounded segmentation (+5.6 Acc@0.25) and semantic segmentation (+3.9 mIoU).
- Mutual benefits exist between tasks: instance segmentation aids grounded segmentation (shared instance understanding capability), and captioning and retrieval reinforce each other (shared text-3D alignment representations).
- Removing instance segmentation pre-training leads to a sharp decline in grounded segmentation and retrieval, indicating that instance-level understanding is key to cross-task transfer.
- Zero-shot 3D classification performance (ModelNet10: 70.4%, ModelNet40: 57.0%) is inferior to CLIP-based methods, but Uni3DL possesses the advantage of being completely independent of 2D projection and pre-trained 2D foundation models.

## Highlights & Insights

- **Functional Unification over I/O Unification**: Instead of taking a seq2seq approach (which predicts token sequences), it outputs heterogeneous formats (masks + semantics + text) from the decoder and then dynamically combines them via the task router. This is far better suited for dense prediction tasks than I/O unification, as mask outputs are inherently high-dimensional continuous forms.
- **Direct Operations on Point Clouds**: By operating directly on point clouds, the model preserves complete 3D geometric information without reliance on multi-view projections. Though this means it cannot leverage powerful 2D pre-trained models like CLIP, it benefits from a cleaner architecture and better geometric understanding.
- **Dual-Track Query Design**: Latent queries capture object-level information (without requiring language inputs), while text queries encode linguistic semantics. Jointly optimizing both in a single decoder naturally achieves vision-language alignment.

## Limitations & Future Work

- **Underutilization of 2D Pre-trained Models**: The authors acknowledge that while operating directly on point clouds prevents information loss, it sacrifices the powerful representations of 2D pre-trained models like CLIP. Future work could explore hybrid approaches.
- **Constrained Data Scale**: The size of 3D datasets (ScanNet with 1.2K scenes, Cap3D with 660K pairs) remains far smaller than 2D datasets (LAION with 400M+ pairs), restricting the generalization capabilities of the unified model.
- **Limited Text Generation Quality**: Autoregressive text generation uses a custom, small transformer, so the generation quality is far below that of LLM-based methods. Integrating an LLM for text generation is a potential future step.
- **Suboptimal Retrieval Performance**: The R@1 score of 5.8% on Text2Shape is significantly lower than Parts2Words (12.7%), which utilizes part labels. Fine-grained shape-to-text matching remains a bottleneck.
- **Scene Scale**: Evaluation is primarily restricted to indoor scenes (ScanNet, S3DIS), while performance on large-scale outdoor scenarios remains unverified.

## Related Work & Insights

- **vs X-Decoder**: X-Decoder pioneered functional unification in the 2D domain. Uni3DL successfully extends this ideology to 3D point clouds, with key differences lying in the 3D voxel sampling strategy and the 3D U-Net backbone.
- **vs 3D-VisTA**: 3D-VisTA also circumvents multi-view projections during pre-training, but still requires separate fine-tuning of independent task heads for downstream tasks. Uni3DL enjoys higher degree of parameter sharing.
- **vs PointLLM**: PointLLM integrates Vicuna for superior text generation, but does not support dense prediction tasks. Uni3DL maintains an edge with its comprehensive multi-task coverage.
- **Insights**: The encoder-decoder-router unified framework paradigm can be generalized to other modalities (e.g., audio-language). The key is to design sufficiently versatile decoder outputs (the combination of semantics and masks is indeed highly adaptable).

## Rating

- Novelty: ⭐⭐⭐⭐ First comprehensive realization of the functional unification paradigm in 3D, though individual components are mostly combinations of existing techniques.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely thorough, with support across 6 major tasks, 5 datasets, and detailed ablation analyses.
- Writing Quality: ⭐⭐⭐⭐ Clear structure; the task-method comparison in Table 1 is immediately informative.
- Value: ⭐⭐⭐⭐ Establishes a strong baseline for 3D unified modeling, though independent training without utilizing 2D pretraining limits its practical competitiveness.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] SR-3D: 3D-Aware Region Prompted Vision Language Model](../../ICLR2026/multimodal_vlm/3d_aware_region_prompted_vision_language_model.md)
- [\[ICLR 2026\] UniF2ace: A Unified Fine-grained Face Understanding and Generation Model](../../ICLR2026/multimodal_vlm/unif2ace_a_underlineunified_underlinefine-grained_underlineface_understanding_an.md)
- [\[ECCV 2024\] UniCode: Learning a Unified Codebook for Multimodal Large Language Models](unicode_learning_a_unified_codebook_for_multimodal_large_language_models.md)
- [\[CVPR 2026\] OneCAT: Decoder-Only Auto-Regressive Model for Unified Understanding and Generation](../../CVPR2026/multimodal_vlm/onecat_decoder-only_auto-regressive_model_for_unified_understanding_and_generati.md)
- [\[ICLR 2026\] Omni-Weather: A Unified Multimodal Model for Weather Radar Understanding and Generation](../../ICLR2026/multimodal_vlm/omni-weather_a_unified_multimodal_model_for_weather_radar_understanding_and_gene.md)

</div>

<!-- RELATED:END -->
