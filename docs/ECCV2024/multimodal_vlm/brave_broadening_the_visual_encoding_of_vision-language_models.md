---
title: >-
  [Paper Note] BRAVE: Broadening the Visual Encoding of Vision-Language Models
description: >-
  [ECCV 2024][Multimodal VLM][Multi-encoder fusion] This paper systematically analyzes the impact of different visual encoders (CLIP, DINOv2, EVA-CLIP, etc.) on VLM performance, finding that no single encoder is optimal across all tasks. Based on this, the BRAVE method is proposed, which utilizes a lightweight MEQ-Former to fuse features from multiple frozen encoders into a compact representation. Consequently, it achieves SOTA results on captioning and VQA tasks with only 116M…
tags:
  - "ECCV 2024"
  - "Multimodal VLM"
  - "Multi-encoder fusion"
  - "Q-Former"
  - "Visual encoding"
  - "VQA"
  - "Hallucination"
date: 2026-05-08
content_hash: b53bad8b29de81c6
---

# BRAVE: Broadening the Visual Encoding of Vision-Language Models

**Conference**: ECCV 2024  
**arXiv**: [2404.07204](https://arxiv.org/abs/2404.07204)  
**Code**: [Project Page](https://brave-vlms.epfl.ch)  
**Area**: Multimodal VLMs  
**Keywords**: Multi-encoder fusion, Q-Former, Visual encoding, VQA, Hallucination

## TL;DR

This paper systematically analyzes the impact of different visual encoders (CLIP, DINOv2, EVA-CLIP, etc.) on VLM performance, finding that no single encoder is optimal across all tasks. Based on this, the BRAVE method is proposed, which utilizes a lightweight MEQ-Former to fuse features from multiple frozen encoders into a compact representation. Consequently, it achieves SOTA results on captioning and VQA tasks with only 116M trainable parameters while significantly reducing visual hallucinations.

## Background & Motivation

**Background**: VLMs typically consist of a visual encoder (e.g., CLIP) + a bridging module (e.g., Q-Former/MLP) + a language model (e.g., LLaMA). Recent research has invested heavily in larger LMs and more training data, yielding significant performance improvements.

**Limitations of Prior Work**: VLMs exhibit severe limitations on the visual side:
   - **CLIP Blind Spots**: Tong et al. found that CLIP is "blind" to certain visual differences, failing to distinguish between image pairs with clear visual deviations.
   - **Visual Hallucinations**: VLMs hallucinate details that are not present in the images.
   - **Biases of a Single Encoder**: Different encoders possess distinct inductive biases due to their training objectives, datasets, and model sizes, meaning any single encoder inevitably has blind spots.

**Key Challenge**: VLMs require a comprehensive understanding of various visual attributes (color, spatial relationships, texture, semantics, etc.), but a single encoder bound by specific training objectives and datasets cannot perform optimally across all dimensions.

**Goal**: To efficiently fuse multiple encoders with different visual biases to create a more comprehensive visual representation.

**Key Insight**: First, conduct a systematic encoder benchmark (8 encoders × 5 tasks) to empirically demonstrate that "there is no silver-bullet encoder," and then propose a fusion framework.

**Core Idea**: Use a unified, lightweight Multi-Encoder Querying Transformer (MEQ-Former) to resample and fuse features from an arbitrary number of frozen encoders into a fixed-length compact representation, serving as a soft visual prompt for a frozen LM.

## Method

### Overall Architecture

Multiple frozen visual encoders (5) → Extract individual image features → Linearly project to a unified dimension → Sequence-level concatenation → MEQ-Former resamples and fuses via cross-attention → Fixed-length output → FC projection to LM input space → Input as soft visual prompt + text prompt to frozen LM → Generate output.

### Key Designs

1. **Systematic Multi-Encoder Analysis (Section 2)**:

    - **Function**: Evaluate the impact of 8 visual encoders on VLM performance under a unified framework.
    - **Encoder Selection**: CLIP-L/14, OpenCLIP-G/14, EVA-CLIP-g, SIGLIP-G/14, SILC-G/16, ViT-e, ViT-G, DINOv2-L/14.
    - **Key Findings**:
        - Different encoders exhibit significant performance discrepancies across tasks (COCO standard deviation of 4.91, VQAv2 standard deviation of 1.74).
        - No single encoder consistently outperforms others.
        - Encoders with drastically different biases can achieve similar performance (e.g., EVA-CLIP vs. ViT-e).
        - MMVP is extremely challenging for all encoders (most fall below the random guess baseline of 25%).
    - **Design Motivation**: Empirically demonstrates the necessity of multi-encoder fusion in a data-driven manner.

2. **MEQ-Former (Multi-Encoder Querying Transformer)**:

    - **Function**: Unify and fuse features from $K$ encoders into a fixed-length compact representation.
    - **Mechanism**:
        - Features from each encoder are projected to a unified dimension (1408 dimensions) via linear layers.
        - Sequence-level concatenation serves as the keys/values for cross-attention.
        - 160 learnable queries ($32 \times 5$ encoders) combined with text prompt tokens serve as queries for cross-attention.
        - 12 Transformer layers perform alternating cross-attention and self-attention.
        - The final 160 query outputs are mapped to the LM input space via an FC layer.
    - **Feature Compression Effect**: Compresses features from $1223 \times 1408$ to $160 \times 768$ (14x compression).
    - **Design Motivation**:
        - Cross-attention naturally resolves the issue of differing output dimensions across different encoders.
        - Fixed-length outputs keep the computation cost at the LM side constant, regardless of the number of encoders.
        - Forgoing encoder-specific identity embeddings allows the MEQ-Former to autonomously learn how to utilize disparate features.
        - Compared to standard Q-Former ensembling ($5 \times 110\text{M} = 550\text{M}$), MEQ-Former requires only 116M parameters.

3. **Encoder Dropout Training Strategy**:

    - **Function**: Randomly mask the features of each encoder with a 20% probability during pre-training.
    - **Mechanism**: Serves as a regularization method to prevent the MEQ-Former from over-relying on a single encoder.
    - **Design Motivation**: Avoid local optima; without dropout, the MEQ-Former might exploit a shortcut by focusing solely on the easiest-to-fit encoder.

### Loss & Training

- **Pre-training**: Train the MEQ-Former with a captioning objective on the WebLI dataset (100M image-text pairs) while keeping both visual encoders and the LM frozen.
- **VQA Fine-tuning**: Fine-tune the MEQ-Former and LM on a mixture of VQAv2 + OKVQA + VQ2A datasets (17M samples).
- **High-Resolution Fine-tuning**: Further fine-tune at $336 \times 336$ resolution.
- Total trainable parameters are only 116M (comprising approximately 1% of the total VLM parameters).

## Key Experimental Results

### Main Results (Captioning)

| Method | Trainable Params | COCO (CIDEr) ↑ | NoCaps out-domain ↑ | NoCaps overall ↑ |
|------|-----------|-------------|-------------------|-----------------| 
| PaLI-17B | 16.9B | **149.1** | - | 127.0 |
| GiT2 | 5.1B | 145.0 | 130.6 | 126.9 |
| BLIP-2 | 1.1B | 144.5 | 124.8 | 121.6 |
| InstructBLIP | 188M | - | - | 121.9 |
| **BRAVE** | **116M** | 148.0 | **133.3** | **127.6** |

### Main Results (VQA)

| Method | Trainable Params | VQAv2 ↑ | OKVQA ↑ | GQA ↑ | VizWiz ↑ | MMVP ↑ | POPE ↑ |
|------|-----------|--------|--------|------|---------|-------|-------|
| PaLI-17B | 16.9B | **84.3** | 64.5 | - | - | - | - |
| LLaVA-1.5 | 13B | 80.0 | - | 63.3 | 53.6 | 24.7 | 85.9 |
| InstructBLIP | 188M | - | 55.5 | - | 33.4 | 16.7 | 78.9 |
| SPHINX-2k | 13B | 80.7 | 62.6 | 63.1 | 44.9 | - | 87.2 |
| **BRAVE** | **3B** | 82.5 | **66.0** | **66.3** | **54.2** | **42.0** | **87.6** |

### Ablation Study

| Configuration | COCO ↑ | VQAv2 ↑ | OKVQA ↑ | Description |
|------|-------|--------|--------|------|
| A0: Full BRAVE | 147.0 | 81.8 | 65.7 | Baseline |
| A1: No LM Fine-tuning | - | 78.6 | 57.5 | LM fine-tuning is crucial for VQA |
| A1: LoRA r=128 | - | 81.0 | 62.9 | LoRA offsets 70% of the performance gap |
| A2: No Synthetic VQA Data | - | 81.1 | 64.0 | Synthetic data contributes significantly |
| A3: No Encoder Dropout | 145.3 | 81.3 | 66.0 | Captioning is more heavily affected |
| A4: No Text Input to MEQ | 145.9 | 81.4 | 64.9 | Text prompts assist in task alignment |
| A5: No High-Resolution FT | 145.2 | 79.6 | 65.0 | High resolution is important for VQA |
| A8: FlanT5-L (Smaller LM) | 142.5 | 79.9 | 65.5 | Larger LMs offer clear linguistic advantages |

### MEQ-Former vs. Q-Former Ensembling

| Bridging Method | Params | COCO ↑ | VQAv2 ↑ | OKVQA ↑ | GQA ↑ |
|----------|-------|-------|--------|--------|------|
| Q-Former Ensemble | 605M | 140.9 | 78.5 | 64.3 | 50.6 |
| **MEQ-Former** | **116M** | **145.2** | **79.6** | **65.0** | **51.5** |

### Key Findings

- The performance boost of BRAVE on MMVP is striking: 42.0% vs. the best single encoder at 27.3% (+14.7%), far exceeding the 25% random guess baseline.
- Strong performance on NoCaps out-domain (133.3) indicates that multi-encoder fusion significantly strengthens out-of-distribution (OOD) generalization.
- MEQ-Former outperforms the Q-Former ensemble with 5x fewer parameters, demonstrating that unified resampling is superior to simple concatenation.
- Removing any 2 encoders results in graceful performance degradation (good robustness); however, degradation accelerates when more than 2 are excluded.
- MEQ-Former adaptively allocates attention weights to different encoders based on downstream tasks.
- Visual-side scaling (multi-encoder) and language-side scaling (larger LM) provide complementary benefits to VLM performance.

## Highlights & Insights

- **Systematic Evidence of "No Silver-Bullet Encoder"**: The comprehensive benchmark across 8 encoders and 5 tasks quantifies this intuition within a unified framework for the first time.
- **New Paradigm for Visual-Side Scaling**: Combining multiple encoders with diverse biases represents a novel scaling dimension compared to simply widening a single encoder or scaling up the LM.
- **Extreme Parameter Efficiency**: SOTA results are achieved with only 116M trainable parameters (1% of total parameters), which is 150x fewer trainable parameters than PaLI-17B.
- **Encoder Dropout as Regularization**: A simple yet effective training trick that can be transferred to other multi-source feature fusion scenarios.
- **No Encoder-Source Labels on Features**: MEQ-Former does not need to know which encoder the features originate from, simplifying the architecture and enhancing flexibility.

## Limitations & Future Work

- Inference requires forward passes through all encoders, causing computational costs to scale linearly with the number of encoders.
- Adaptive encoder selection mechanisms remain unexplored; dynamically deciding which encoders to activate based on the input could alleviate inference overhead.
- The ensemble of encoders could be further expanded, for instance, by adding 3D prior encoders or scene understanding encoders.
- It only explores image-text modalities and could be extended to other modalities like audio and video.
- Performance remains bounded by the inherent biases and hallucination issues of LLMs.

## Related Work & Insights

- **vs. BLIP-2**: BLIP-2 bridges a single encoder with a single Q-Former. BRAVE generalizes this to multiple encoders with the MEQ-Former, achieving higher performance with fewer parameters (116M vs. 188M).
- **vs. LLaVA-1.5**: LLaVA utilizes an MLP projection to connect CLIP and the LM. Despite having 13B parameters, it falls far short of BRAVE (42.0%) on MMVP (24.7%), because CLIP's blind spots cannot be rectified by an MLP.
- **vs. LLaVA-MoF / SPHINX**: These concurrent works also explore multi-encoder architectures, but they merely concatenate features before feeding them into the LM, limiting scalability. BRAVE's unified resampling mechanism can seamlessly scale to any number of encoders.

## Rating

- Novelty: ⭐⭐⭐⭐ The idea of multi-encoder fusion itself is not entirely new, but the unified resampling design of the MEQ-Former and the systematic encoder analysis are significant contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely comprehensive, encompassing an 8-encoder benchmark, 8 downstream tasks, exhaustive ablation studies, analysis of encoder contributions, and comparisons against ensemble methods.
- Writing Quality: ⭐⭐⭐⭐⭐ Highly logical flow from analysis to methodology to experiments, with strong motivation and professional figures/tables.
- Value: ⭐⭐⭐⭐⭐ Demonstrates the clinical importance of visual-side scaling. The MEQ-Former design is elegant, highly efficient, and reusable, offering direct impact to the VLM community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] FastVLM: Efficient Vision Encoding for Vision Language Models](../../CVPR2025/multimodal_vlm/fastvlm_efficient_vision_encoding_for_vision_language_models.md)
- [\[ICLR 2026\] Revisiting Multimodal Positional Encoding in Vision-Language Models](../../ICLR2026/multimodal_vlm/revisiting_multimodal_positional_encoding_in_visionlanguage_models.md)
- [\[ECCV 2024\] Grounding Language Models for Visual Entity Recognition](grounding_language_models_for_visual_entity_recognition.md)
- [\[ECCV 2024\] Self-Adapting Large Visual-Language Models to Edge Devices across Visual Modalities](self-adapting_large_visual-language_models_to_edge_devices_across_visual_modalit.md)
- [\[ECCV 2024\] Vary: Scaling up the Vision Vocabulary for Large Vision-Language Models](vary_scaling_up_the_vision_vocabulary_for_large_visionlanguag.md)

</div>

<!-- RELATED:END -->
