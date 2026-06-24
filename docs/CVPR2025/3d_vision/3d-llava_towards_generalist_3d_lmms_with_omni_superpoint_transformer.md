---
title: >-
  [Paper Note] 3D-LLaVA: Towards Generalist 3D LMMs with Omni Superpoint Transformer
description: >-
  [CVPR 2025][3D Vision][3D LMM] This work proposes 3D-LLaVA, a general-purpose 3D Large Multimodal Model (LMM) with a minimalist architecture. The core is the **Omni Superpoint Transformer (OST)** acting as a versatile visual connector. It simultaneously serves as a visual feature selector, a visual prompt encoder, and a segmentation mask decoder. Using only point cloud inputs, it fully achieves state-of-the-art (SOTA) performance across five benchmarks…
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "3D LMM"
  - "Superpoint Transformer"
  - "Visual Connector"
  - "Referring Segmentation"
  - "Scene Understanding"
date: 2026-05-08
content_hash: 39a6c2c6b0267a48
---

# 3D-LLaVA: Towards Generalist 3D LMMs with Omni Superpoint Transformer

**Conference**: CVPR 2025  
**arXiv**: [2501.01163](https://arxiv.org/abs/2501.01163)  
**Code**: [https://github.com/djiajunustc/3D-LLaVA](https://github.com/djiajunustc/3D-LLaVA)  
**Area**: 3D Vision  
**Keywords**: 3D LMM, Superpoint Transformer, Visual Connector, Referring Segmentation, Scene Understanding  
**Authors**: Jiajun Deng, Tianyu He, Li Jiang, Tianyu Wang, Feras Dayoub, Ian Reid

## TL;DR
This work proposes 3D-LLaVA, a general-purpose 3D Large Multimodal Model (LMM) with a minimalist architecture. The core is the **Omni Superpoint Transformer (OST)** acting as a versatile visual connector. It simultaneously serves as a visual feature selector, a visual prompt encoder, and a segmentation mask decoder. Using only point cloud inputs, it fully achieves state-of-the-art (SOTA) performance across five benchmarks, including ScanQA (92.6 CiDEr) and ScanRefer (43.3 mIoU). 
The architecture follows the minimalist design philosophy of 2D LLaVA, assembling a 3D encoder, a visual connector, and an LLM into a unified pipeline.

## Background & Motivation
Existing 3D LMMs suffer from three key limitations: (1) Reliance on complex pipelines, such as offline multi-view feature extraction (e.g., Chat-Scene requires pre-extracting 2D+3D instance features) or additional task-specific heads; (2) Single-functional visual connectors, where MLP projections or Q-Formers only perform feature conversion and cannot handle visual prompts or segmentation; (3) Inability to ground language to 3D masks, as most 3D LMMs can only generate text. While the 2D LMM domain has a clean paradigm like LLaVA, the 3D domain lacks an equally simple yet powerful baseline. Furthermore, existing methods often require separate fine-tuning for different tasks, lacking a truly unified training scheme.

## Core Problem
How to design a unified 3D LMM architecture **without offline feature extraction or additional task-specific modules**, which can simultaneously comprehend conversations/QA/descriptions, handle visual prompt interactions, and generate 3D segmentation masks? The key challenge lies in making the visual connector satisfy two fundamentally different needs: feature dimension conversion (3D $\rightarrow$ LLM) and task versatility (text generation + mask prediction).

## Method

### Overall Architecture
Point cloud $\rightarrow$ Sparse 3D U-Net feature extraction $\rightarrow$ Superpoint pooling aggregation into ~hundreds of superpoints $\rightarrow$ OST processing (multifunctional: feature enhancement + selection + prompt encoding + mask decoding) $\rightarrow$ Projection to language space $\rightarrow$ LLM (Vicuna-7B + LoRA) $\rightarrow$ Text output + optional [SEG] to trigger mask decoding. The entire architecture takes only point clouds as input, without requiring multi-view images or offline feature extraction.

### Key Designs
1. **Omni Superpoint Transformer (OST)**: The architecture is based on self-attention (rather than cross-attention), where superpoint features serve as queries, keys, and values simultaneously. It features a three-in-one functionality:

    - **Visual Feature Selector**: Selects the top-$K$ (default 100) superpoint tokens to input into the LLM based on objectness scores, reducing computational overhead. The objectness scores are derived from instance segmentation supervision during the pre-training stage, where high-scoring superpoints correspond to meaningful object regions in the scene.
    - **Visual Prompt Encoder**: A parameter-free Visual Sampler converts click, box, or mask prompts into features using three-nearest-neighbor interpolation or average pooling. These features are concatenated with the superpoint queries and fed into the OST. Masked attention is utilized to prevent prompts from affecting the superpoints (unidirectional information flow: attention flows from superpoints $\rightarrow$ prompts, while prompts $\rightarrow$ superpoints is masked).
    - **Mask Decoder**: When the LLM outputs a [SEG] token, the corresponding hidden state is projected as a segmentation query and fed into the frozen OST, where a mask head generates a binary mask. Freezing the OST ensures consistency with the pre-trained perceptual prior.

2. **Distance-Adaptive Self-Attention**: Introduces a bias term $\sigma \cdot D$ based on the Euclidean distance between superpoints into the self-attention mechanism, guiding stronger interactions between close superpoints. Here, $\sigma$ is a learnable scalar, and $D$ is the distance matrix between superpoints.

3. **Hybrid Pre-training**: Combines instance segmentation supervision (ScanNet200 annotations) with 2D-to-3D knowledge distillation (lifting CLIP-ViT-L features from LLaVA-1.5 into 3D via geometric correspondence) to achieve two goals at once: obtaining perceptual priors and using 2D features as a bridge for 3D-language alignment. Experiments demonstrate that the distillation loss contributes the most to downstream QA tasks (CiDEr drops by ~3 points without KD), indicating that 2D pre-trained features are key to bridging the 3D-language gap.

### Loss & Training
- **Pre-training (Stage 1)**: $\mathcal{L}_{Pre} = \mathcal{L}_{Cls} + \mathcal{L}_{Mask} + \mathcal{L}_{KD}$, for 512 epochs on ScanNet200. The classification and mask losses provide perceptual priors, while the distillation loss lifts the CLIP-ViT-L features of LLaVA-1.5 into 3D via geometric correspondence, bridging the 3D-language alignment.
- **Instruction Fine-Tuning (Stage 2)**: $\mathcal{L}_{IFT} = \mathcal{L}_{text} + 0.1 \times \mathcal{L}_{mask}$, freezing the 3D encoder, OST, and the LLM backbone, while only training the projection layers and LoRA (rank=16). This freezing strategy resembles 2D LLaVA, preventing the pre-trained perceptual priors from being disrupted.
- 8 $\times$ RTX 3090, batch size 2 per GPU, 8 gradient accumulation steps, AdamW, cosine scheduler, lr=2e-4.
- Data: 295K instruction fine-tuning data entries (ScanRefer 41K + Nr3D 37K + Multi3DRefer 54K + ScanQA 26K + SQA3D 33K + Scan2Cap 36K + dialogue data 68K).

## Key Experimental Results

### Main Results (Unified Training, No Separate Fine-Tuning)

| Task | Dataset | Metric | 3D-LLaVA | Prev. SOTA | Gain |
|------|--------|------|----------|--------|------|
| 3D QA | ScanQA | CiDEr | **92.6** | 87.7 (Chat-Scene) | +4.9 |
| 3D QA | ScanQA | BLEU-4 | **17.1** | 14.3 (Chat-Scene) | +2.8 |
| Situated QA | SQA3D | EM | 54.5 | 54.6 (Chat-Scene) | -0.1 |
| 3D Dense Cap | Scan2Cap | C@0.5 | **78.8** | 77.2 (Chat-Scene) | +1.6 |
| Referring Seg | ScanRefer | mIoU | **43.3** | 41.7 (SegPoint*) | +1.6 |
| Referring Seg | Multi3DRefer | mIoU | **42.7** | 36.1 (SegPoint*) | +6.6 |

Note: 3D-LLaVA utilizes only point clouds, with no multi-view images; Chat-Scene uses PC+Image. On SQA3D, it performs on par with Chat-Scene (54.5 vs 54.6), possibly because situated QA relies more on language understanding than visual perception. The largest improvement is on Multi3DRefer (+6.6 mIoU), as multi-object referring requires precise mask-level localization.

### Ablation Study
- **Visual Prompt Encoding**: OST as an encoder (78.8 CiDEr) > direct pooling utilization (76.8) > coordinate projection MLP (68.7). Although the pooling method is simple, it loses spatial interaction information between superpoints, while MLP coordinate projection completely discards visual features.
- **Number of Visual Tokens**: 100 is optimal; at 50, CiDEr decreases by 1.5%; at 200/400, there is no significant improvement, but inference overhead increases linearly.
- **Box-level Grounding**: Although designed for the mask level, directly converting to bounding boxes yields Acc@0.25=51.2%, outperforming most competitive methods.
- **Ablation of Distance-Adaptive Self-Attention**: Removing the distance bias term decreases ScanQA's CiDEr by about 2 points, demonstrating that local spatial priors are beneficial for 3D scene understanding.
- **Ablation of Distillation Loss**: Removing $\mathcal{L}_{KD}$ during pre-training leads to a ~3-point drop in downstream CiDEr, proving that 2D-to-3D knowledge distillation is crucial to bridge 3D-language alignment.
- **Pre-training Freezing Strategy**: Freezing the 3D encoder, OST, and LLM backbone during instruction fine-tuning while only training the projection layers and LoRA protects pre-trained perceptual priors from catastrophic forgetting.

## Highlights & Insights
- **Minimalist Design**: A single OST module simultaneously solves three tasks: feature selection, prompt encoding, and mask decoding, without requiring additional modules.
- **Point Cloud-Only**: Does not rely on offline multi-view image features, making deployment simpler, yet outperforms methods utilizing 2D+3D features.
- **High Versatility**: A single training run covers all 3D understanding tasks including QA, description, and referring segmentation, making it the first unified 3D LMM that simultaneously covers text generation and 3D segmentation.
- **Hybrid Pre-training Strategy**: The combination of instance segmentation and 2D knowledge distillation provides a strong perceptual prior for the 3D encoder, an idea that can be generalized to other 3D tasks.
- **Distance-Adaptive Self-Attention** Design is Simple and Effective: Adding just a single bias term based on Euclidean distance enhances local spatial interactions with negligible computational overhead.
- **Instruction Fine-Tuning Data Organization**: Unifies 6 datasets into a question-answer format, where the segmentation task is triggered by a special [SEG] token, realizing true unified multi-task training rather than separate multi-stage training.

## Limitations & Future Work
- Only validated on ScanNet indoor scenes; the effectiveness on outdoor/large-scale scenes remains unknown.
- Box-level grounding (Acc@0.5=40.6%) is weaker than Chat-Scene (50.2%), because the method is optimized for masks rather than bounding boxes, causing accuracy loss when converting to bounding boxes.
- 3D data remains a bottleneck, with only 295K instruction data entries. The authors point out that data collection is a key focus for future work; 2D LLaVA achieved significantly better strength with 665K entries, highlighting the scarcity of 3D data as a primary constraint.
- Pre-training requires instance annotations from ScanNet200, which limits adaptability to new scenes; future work could explore unsupervised pre-training alternatives.
- Superpoint pooling compresses the raw point cloud into hundreds of superpoint tokens. While this significantly reduces computational overhead, it may discard fine-grained geometric details, which could be disadvantageous for tasks requiring precise spatial reasoning (e.g., small object localization).
- The feature extraction overhead of the Sparse 3D U-Net is not reported in detail; encoder inference latency might become a bottleneck during practical deployment.
- The 7B-scale LLM may limit complex reasoning capabilities; whether larger LLMs (such as 13B/70B) can yield further improvements remains unexplored.

## Related Work & Insights
- **Chat-Scene**: Uses offline Mask3D + 2D + 3D instance features, resulting in a complex pipeline. 3D-LLaVA achieves a 4.9% higher CiDEr on ScanQA while using only point clouds.
- **SegPoint**: The first method combining LLMs for referring segmentation, but it requires fine-tuning. 3D-LLaVA achieves a 1.6% higher ScanRefer mIoU under unified training.
- **Grounded 3D-LLM**: Also takes point cloud inputs, but 3D-LLaVA achieves a 19.9% higher CiDEr on ScanQA.
- **LEO**: While its ScanQA CiDEr of 101.4 seems higher, its setting is different (accessing ground-truth object information), making it not directly comparable.

### Qualitative Analysis
The model performs outstandingly in complex spatial reasoning tasks: it can accurately answer questions that require understanding 3D spatial orientation, such as "what is to the right of the chair". In referring segmentation, it accurately generates 3D masks even when the object description contains multiple spatial constraints (e.g., "the brown chair near the window"). Failure cases mainly occur when object appearances are highly similar or spatial relationship descriptions are ambiguous.

## Related Work & Insights
- The design concept of "one visual connector solves all tasks" is applicable to 2D multimodal models—upgrading the projection layer to a multifunctional Transformer.
- Superpoint representation as a tokenization scheme for 3D scenes preserves more information than FPS sampling, which can be generalized to other 3D-LLMs.
- The combination of instance segmentation and 2D distillation in hybrid pre-training can be transferred to other scene understanding tasks that require perceptual priors.
- The technique of using masked attention in OST to implement a unidirectional information flow is worth adopting: visual prompts only "read" superpoint information without "contaminating" the superpoint representations.

## Rating
- Novelty: ⭐⭐⭐⭐ The three-in-one OST design is elegant, but the core components (superpoints, masked Transformer) are clever combinations of existing technologies.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 5 datasets, multi-task comparisons, rich ablations, and qualitative visualizations.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear motivation, systematic comparison, and detailed method descriptions.
- Value: ⭐⭐⭐⭐⭐ Simple and unified architecture + SOTA performance + open-source code, serving as a strong baseline for 3D LMMs.
- Replicability: ⭐⭐⭐⭐⭐ Open-source code, detailed training settings (reproducible on 8$\times$ RTX 3090), and fully public datasets.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] LLaVA-3D: A Simple yet Effective Pathway to Empowering LMMs with 3D Capabilities](../../ICCV2025/3d_vision/llava-3d_a_simple_yet_effective_pathway_to_empowering_lmms_with_3d_capabilities.md)
- [\[ICCV 2025\] GaussianProperty: Integrating Physical Properties to 3D Gaussians with LMMs](../../ICCV2025/3d_vision/gaussianproperty_integrating_physical_properties_to_3d_gaussians_with_lmms.md)
- [\[CVPR 2025\] VGGT: Visual Geometry Grounded Transformer](vggt_visual_geometry_grounded_transformer.md)
- [\[CVPR 2025\] FASTer: Focal Token Acquiring-and-Scaling Transformer for Long-term 3D Object Detection](faster_focal_token_acquiring-and-scaling_transformer_for_long-term_3d_objection_.md)
- [\[ICLR 2026\] Exploring the Potential of Encoder-free Architectures in 3D LMMs](../../ICLR2026/3d_vision/exploring_the_potential_of_encoder-free_architectures_in_3d_lmms.md)

</div>

<!-- RELATED:END -->
