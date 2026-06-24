---
title: >-
  [Paper Note] Calico: Part-Focused Semantic Co-Segmentation with Large Vision-Language Models
description: >-
  [CVPR 2025][Multimodal VLM][Part-Level Co-Segmentation] This paper proposes Calico—the first large vision-language model designed for part-level semantic co-segmentation. By establishing part-level semantic correspondence across multiple images using a Correspondence Extraction Module (CEM) and a Correspondence Adaptation Module (CAM), and fine-tuning only 0.3% of the parameters, it thoroughly outperforms existing methods on the newly constructed MixedParts benchmark…
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "Part-Level Co-Segmentation"
  - "Multi-Image Reasoning"
  - "Semantic Correspondence"
  - "Large Vision-Language Models"
  - "Parameter-Efficient Tuning"
date: 2026-05-08
content_hash: beb9bd8fec348089
---

# Calico: Part-Focused Semantic Co-Segmentation with Large Vision-Language Models

**Conference**: CVPR 2025  
**arXiv**: [2412.19331](https://arxiv.org/abs/2412.19331)  
**Code**: [https://plan-lab.github.io/calico](https://plan-lab.github.io/calico)  
**Area**: Multimodal VLM  
**Keywords**: Part-Level Co-Segmentation, Multi-Image Reasoning, Semantic Correspondence, Large Vision-Language Models, Parameter-Efficient Tuning

## TL;DR

This paper proposes Calico—the first large vision-language model designed for part-level semantic co-segmentation. By establishing part-level semantic correspondence across multiple images using a Correspondence Extraction Module (CEM) and a Correspondence Adaptation Module (CAM), and fine-tuning only 0.3% of the parameters, it thoroughly outperforms existing methods on the newly constructed MixedParts benchmark, achieving a 6.3% gain in mIoU and a 51.3% speedup in inference.

## Background & Motivation

**Background**: Large vision-language models (LVLMs) have achieved significant progress in single-image segmentation tasks. Models like LISA and GLaMM achieve pixel-level segmentation based on text instructions by introducing a segmentation token [SEG] into the LLM. Meanwhile, self-supervised ViTs such as DINOv2 have been proven capable of capturing part-level semantic correspondences across categories.

**Limitations of Prior Work**: (1) Existing LVLM segmentation models only support single-image operations, making them incapable of comparative reasoning and co-segmentation across multiple images; (2) Existing methods for part co-segmentation (e.g., SCOPS, DFF) cannot generate semantic labels or handle unique parts—they can only find shared structures and fail to name or differentiate unique features; (3) Existing part segmentation models (such as PartGLEE, VLPart) require users to specify each part name individually for segmentation, failing to automatically discover and compare shared/unique parts across multiple images.

**Key Challenge**: Simultaneously achieving the "location + comparison + naming" three-in-one part-level co-segmentation in multi-image scenarios requires the model to possess both cross-image semantic correspondence and open-ended label generation capabilities. However, these two capabilities are decoupled in existing architectures.

**Goal**: This work defines a new task, "Part-Focused Semantic Co-Segmentation"—given multiple images containing similar objects, the model automatically segments and annotates shared objects, shared parts, and unique parts.

**Key Insight**: DINOv2 naturally possesses cross-category part-level semantic correspondence capabilities, and LVLMs naturally feature open-ended text generation and reasoning capabilities. Merging the two could resolve the triple mandate of "locate + compare + name".

**Core Idea**: Integrate part-level semantic correspondence features extracted by DINOv2 into a frozen LVLM, allowing the LVLM to learn part-level reasoning and co-segmentation across multiple images via parameter-efficient adaptation modules.

## Method

### Overall Architecture

Calico is built upon an expansion of the GLaMM architecture, comprising the following pipeline: multiple input images retrieve global/semantic features via the EVA-CLIP vision encoder and the DINOv2 encoder, respectively. These features are compressed using Q-Former and fed into the Vicuna-7B LLM in an interleaved manner. The LLM outputs text containing the [SEG] token, which is then projected and fed into the SAM decoder to generate segmentation masks. The entire system contains only approximately 29M trainable parameters, accounting for 0.3% of the total parameters.

### Key Designs

1. **Correspondence Extraction Module (CEM)**:

    - **Function**: Merges part-level semantic correspondence information from DINOv2 into the global visual features of EVA-CLIP.
    - **Mechanism**: Given input images, global embeddings $\mathbf{X}_{\text{global}}$ are obtained through a frozen EVA-CLIP, and semantic embeddings $\mathbf{X}_{\text{semantic}}$ are obtained through a frozen DINOv2. A cross-attention mechanism is then employed where semantic embeddings serve as Key-Value and global embeddings serve as Query to fuse the features: $\mathbf{X}_{\text{global}}' = \mathcal{A}(\mathbf{X}_{\text{global}}, \mathbf{X}_{\text{semantic}})$. The resulting enhanced features exhibit both the global recognition capabilities of CLIP and the part-level semantic correspondence capabilities of DINOv2.
    - **Design Motivation**: Features learned by DINOv2 via self-supervised training exhibit strong semantic correspondence at the part-level granularity (e.g., "legs" of different object categories show similar activations) but lack alignment with the language space. Conversely, EVA-CLIP provides strong global features but lacks fine-grained correspondence. CEM precisely complements both.

2. **Correspondence Adaptation Module (CAM)**:

    - **Function**: Injects the semantically enriched features from the CEM into the intermediate layers of the LLM to enable instruction-guided multi-image understanding.
    - **Mechanism**: Two CAMs are placed at the 11th and 22nd layers of the LLM (at 1/3 and 2/3 of a 32-layer LLM). Each CAM first linearly projects the last text token of the current layer (which carries instruction information) into a guidance embedding, which is then added to the learnable query of the Q-Former to form a context-guided query: $\mathbf{q}' = \mathbf{q} + f_{\text{adaptation}}(\mathbf{t}_{S_T}^l)$. This query is then used to extract information from the CEM-enhanced visual features. Finally, the extracted outcomes are projected back to the language space and added to the visual tokens of the corresponding LLM layers.
    - **Design Motivation**: The two CAM layers inject semantic correspondence information at different depths of the LLM, encouraging the model to learn cross-image correspondences at different granularities (object-level and part-level). Ablation studies confirm that this setup outperforms 1-layer or 3-layer configurations.

3. **Q-Former Visual Compression + Interleaved Multi-Image Input**:

    - **Function**: Compresses the visual tokens of each image from 256/576 down to 32, supporting efficient multi-image processing.
    - **Mechanism**: Drawing inspiration from BLIP-2, 32 learnable queries of the Q-Former extract compact visual embeddings from the EVA-CLIP features via a cross-attention mechanism. The embeddings of multiple images are fed into the LLM in an interleaved manner (`image1 tokens + text + image2 tokens + text`), with each image distinguished by a unique identifier (`IMAGE1`, `IMAGE2`).
    - **Design Motivation**: Multi-image segmentation requires handling multiple sets of visual tokens. Directly using the original token count (e.g., GLaMM's 576 tokens/image) leads to a computational explosion. Q-Former compresses the number of tokens by 8 to 18 times, reducing TFLOPS by 32.6% and speeding up inference by 51.3%.

### Loss & Training

The training loss is a weighted combination of text loss and segmentation loss: $\mathcal{L} = \lambda_{\text{text}} \mathcal{L}_{\text{text}} + \mathcal{L}_{\text{mask}}$, where $\mathcal{L}_{\text{text}}$ is the standard causal LM cross-entropy loss, and $\mathcal{L}_{\text{mask}} = \lambda_{\text{focal}} \mathcal{L}_{\text{focal}} + \lambda_{\text{Dice}} \mathcal{L}_{\text{Dice}}$. The hyperparameters are set to $\lambda_{\text{text}}=1.0$, $\lambda_{\text{focal}}=2.0$, and $\lambda_{\text{Dice}}=0.5$. Parameter-efficient fine-tuning is performed using LoRA (rank=8, alpha=16) with a learning rate of 4e-4 and the AdamW optimizer, training for 10 epochs on 4 A40 GPUs.

## Key Experimental Results

### Main Results

| Method | AP50↑ | mIoU↑ | Recall↑ | SS↑ | S-IoU↑ |
|------|-------|-------|---------|-----|--------|
| Cascade (Sparkles+GPT4o+LISA) | 5.7 | 27.9 | 19.0 | 32.2 | 14.8 |
| Multi-Image PartGLEE | 1.2 | 29.3 | 9.7 | 78.5 | 63.3 |
| Multi-Image VLPart | 13.4 | 42.8 | 34.6 | 59.1 | 46.5 |
| Multi-Image GLaMM (Fine-tuned) | 42.9 | 59.9 | 54.9 | 76.8 | 71.2 |
| Multi-Image LISA (Fine-tuned) | 41.4 | 59.7 | 55.5 | 78.7 | 72.5 |
| **Ours** | **45.9** | **63.7** | **59.7** | **82.7** | **77.1** |

### Ablation Study

| Configuration | AP50 | mIoU | Recall | SS | S-IoU |
|------|------|------|--------|-----|-------|
| w/o Q-Former | 38.5 | 59.2 | 44.8 | 64.5 | 55.6 |
| w/o DINO | 43.9 | 61.7 | 57.1 | 80.2 | 74.5 |
| w/o CEM | 43.6 | 61.6 | 57.5 | 80.8 | 75.2 |
| w/o CAM | 45.9 | 63.3 | 59.7 | 82.0 | 76.5 |
| w/o CEM w/o CAM | 44.1 | 62.7 | 58.1 | 81.6 | 76.3 |
| **Ours (Full)** | **45.9** | **63.7** | **59.7** | **82.7** | **77.1** |

### Key Findings

- **CEM is the Core Contribution**: Removing CEM leads to a comprehensive drop in segmentation metrics (mIoU drops from 63.7 to 61.6), proving that DINOv2's semantic correspondence information is crucial for cross-image part understanding.
- **CAM Requires CEM Guidance to Be Effective**: When keeping CAM but removing CEM (w/o CEM), performance is actually lower than removing both (w/o CEM w/o CAM). This indicates that CAM injects redundant/noisy information when no external semantic signal is present.
- **Q-Former is Indispensable**: Removing Q-Former leads to a substantial decline in performance (AP50 drops from 45.9 to 38.5) since both CEM and CAM rely on the Q-Former architecture.
- **Selection of CAM Layers**: A uniform distribution across 2 layers (11th and 22nd layers) is optimal, with performance declining under 1-layer and 3-layer configurations.
- **Significant Efficiency Advantages**: Using 32 tokens/image (vs 256 for LISA and 576 for GLaMM) decreases TFLOPS by 32–35% and improves inference speed by 30–51%.

## Highlights & Insights

- **High Value in Task Definition**: "Part-focused semantic co-segmentation" is a well-defined and widely applicable new task that covers three sub-goals: localization, comparison, and naming. The design philosophy of this multi-image, multi-granularity visual reasoning task is highly inspiring.
- **Ingenious Fusion Paradigm of DINOv2 + CLIP**: CEM leverages cross-attention to "graft" the part-level semantic correspondence capability of DINOv2 onto CLIP features, avoiding retraining large models while keeping both encoders frozen to ensure efficiency. This "frozen + fusion" paradigm can be transferred to other tasks requiring multi-source visual features.
- **Instruction-Guided Design of CAM**: Utilizing the last token of LLM layers (which carries instruction semantics) to guide visual feature extraction enables the model to "dynamically focus on specific visual content based on user queries." This design can be extended to other scenarios requiring conditional visual understanding.

## Limitations & Future Work

- Evaluation is only conducted on a single self-constructed dataset, MixedParts, lacking validation of generalizability across other part segmentation benchmarks.
- It is assumed that DINOv2 features can fully capture part-level semantic correspondences. However, for parts that are functionally similar but geometrically or visually diverse (e.g., "handles" of different materials), this assumption may fail.
- Multi-image inputs are currently limited to 2 images, and the scalability to more images remains unverified.
- Although the MixedParts dataset is large (2.4 million samples), its part annotations originate from existing datasets, which may inherit biases from the original annotations.
- Future work can explore transferring CEM/CAM modules to the comparative analysis of organs/substructures in 3D medical imaging.

## Related Work & Insights

- **vs LISA**: LISA is a single-image segmentation LVLM based on LLaVA. Calico introduces multi-image capabilities and part-level correspondence modules on top of LISA. Although fine-tuned Multi-Image LISA performs reasonably on MixedParts, it lags behind Calico, highlighting the necessity of dedicated correspondence modules.
- **vs GLaMM**: GLaMM is the direct initialization source for Calico, and both share the same base weights. Calico's CEM+CAM modules yield significant performance gains (mIoU: 59.9 $\rightarrow$ 63.7), proving the value of correspondence information injection.
- **vs Traditional Part Co-Segmentation (SCOPS/DFF, etc.)**: These methods cannot generate semantic labels and require specifying the number of part categories beforehand. Calico automatically names discovered parts utilizing the text generation capabilities of the LLM, achieving true open-ended part co-segmentation.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First to define the part-focused semantic co-segmentation task; the CEM/CAM design is elegant and effective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive ablation studies and detailed efficiency analysis, though evaluated on only a single dataset.
- Writing Quality: ⭐⭐⭐⭐⭐ Clearly defined problems, systematic method description, and in-depth experimental analysis.
- Value: ⭐⭐⭐⭐ High value in task definition and dataset contributions, carrying inspiring significance for the multi-image LVLM field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Can Large Vision-Language Models Correct Semantic Grounding Errors By Themselves?](can_large_vision-language_models_correct_semantic_grounding_errors_by_themselves.md)
- [\[CVPR 2025\] Identifying and Mitigating Position Bias of Multi-image Vision-Language Models](identifying_and_mitigating_position_bias_of_multi-image_vision-language_models.md)
- [\[CVPR 2025\] Continual Learning with Vision-Language Models via Semantic-Geometry Preservation](continual_learning_with_vision-language_models_via_semantic-geometry_preservatio.md)
- [\[CVPR 2025\] Generalized Few-Shot 3D Point Cloud Segmentation with Vision-Language Model](generalized_few-shot_3d_point_cloud_segmentation_with_vision-language_model.md)
- [\[CVPR 2026\] Uncertainty-guided Compositional Alignment with Part-to-Whole Semantic Representativeness in Hyperbolic Vision-Language Models](../../CVPR2026/multimodal_vlm/uncertainty-guided_compositional_alignment_with_part-to-whole_semantic_represent.md)

</div>

<!-- RELATED:END -->
