---
title: >-
  [Paper Note] F-LMM: Grounding Frozen Large Multimodal Models
description: >-
  [CVPR 2025][Segmentation][Visual Grounding] F-LMM freezes all parameters of off-the-shelf LMMs and trains only a lightweight CNN mask decoder to translate the inherent word-pixel correspondences in LMM attention maps into segmentation masks, achieving competitive visual grounding performance while fully preserving conversational capability.
tags:
  - "CVPR 2025"
  - "Segmentation"
  - "Visual Grounding"
  - "Large Multimodal Models"
  - "Frozen Parameters"
  - "Attention Maps"
  - "Referring Expression Segmentation"
date: 2026-05-08
content_hash: 7568ceed42805753
---

# F-LMM: Grounding Frozen Large Multimodal Models

**Conference**: CVPR 2025  
**arXiv**: [2406.05821](https://arxiv.org/abs/2406.05821)  
**Code**: [https://github.com/wusize/F-LMM](https://github.com/wusize/F-LMM)  
**Area**: Visual Grounding / Multimodal VLM  
**Keywords**: Visual Grounding, Large Multimodal Models, Frozen Parameters, Attention Maps, Referring Expression Segmentation

## TL;DR

F-LMM freezes all parameters of off-the-shelf LMMs and trains only a lightweight CNN mask decoder to translate the inherent word-pixel correspondences in LMM attention maps into segmentation masks, achieving competitive visual grounding performance while fully preserving conversational capability.

## Background & Motivation

**Background**: Endowing Large Multimodal Models (LMMs) with visual grounding capabilities is currently a hot direction. The mainstream approach is to add a special segmentation token (e.g., [SEG]) to the LMM vocabulary, connect the LMM with a mask head such as SAM, and then fine-tune the entire model on segmentation and grounding data.

**Limitations of Prior Work**: Fine-tuning LMMs leads to catastrophic degradation of conversational capabilities. The authors comprehensively evaluated state-of-the-art (SOTA) grounding LMMs and found that models such as GLaMM, LISA, and PixelLM scored close to zero on general QA benchmarks like MME and MMBench, losing instruction-following capabilities and general knowledge understanding. For example, GLaMM is unable to answer simple yes/no questions.

**Key Challenge**: There is a fundamental conflict between grounding and conversational capabilities. Existing segmentation/grounding data only contain simple grounding prompts. During fine-tuning, the model mainly learns the relationship between words and segmentation tokens, leading to overfitting on the grounding task at the expense of conversational capabilities. Collecting high-quality "conversational data with segmentation annotations" is extremely costly and yields limited effectiveness.

**Goal**: How to enable off-the-shelf LMMs to possess both visual grounding and conversational capabilities without fine-tuning any LMM parameters.

**Key Insight**: The authors discovered that word-pixel correspondences are naturally inherent in the attention mechanisms of trained LMMs. Visualizing the attention maps via K-Means clustering reveals the geometric and spatial contours of objects. These serve as valuable segmentation priors, requiring only a lightweight decoder to translate these attention weights into masks.

**Core Idea**: Freeze the LMM and train only a CNN to translate the attention maps into segmentation masks, thereby obtaining grounding capabilities "for free".

## Method

### Overall Architecture

The input consists of an image and a text, and the output is the segmentation mask corresponding to the keyword. The overall pipeline is: (1) the frozen LMM conducts multimodal reasoning normally, while extracting the word-image attention maps from all layers and all attention heads; (2) $M \times N$ attention maps are stacked into a tensor to serve as the segmentation prior; (3) a CNN mask decoder translates the attention maps into mask logits; (4) a SAM-based mask refiner optimizes the mask using additional visual and textual cues; (5) a keyword selector automatically identifies object nouns within the text.

### Key Designs

1. **Word-Image Attention Map Extraction**:

    - **Function**: Extract word-pixel corresponding segmentation priors from the frozen LMM.
    - **Mechanism**: In the causal self-attention of the LLM, each word token $z^i$ has attention weights over all preceding tokens (including image tokens). The attention weights of word tokens on image tokens are extracted and unflattened back into a 2D spatial structure, yielding $a^i \in \mathbb{R}^{h \times w}$. For multi-word described objects, the attention maps of each word are averaged or max-pooled. A total of $MN$ attention maps from $M$ layers and $N$ heads are stacked into $A \in \mathbb{R}^{MN \times h \times w}$ and then bilinearly interpolated to $64 \times 64$.
    - **Design Motivation**: Although LMMs are not trained with pixel-level annotations, the transformer's attention mechanism naturally encodes the correspondence between text and image regions. Visualizations show that these attention maps contain coarse yet meaningful spatial contours.

2. **Mask Head (Decoder + Refiner)**:

    - **Function**: Translate coarse attention maps into fine-grained segmentation masks.
    - **Mechanism**: The mask decoder is a 3-stage U-Net. It takes the $MN$-channel attention map tensor as input and outputs binary mask logits. The mask refiner is adapted from SAM's mask head: the mask logits output by the decoder are converted into dense prompt embeddings via SAM's prompt encoder, the mask's bounding box is converted into a box embedding, and text embeddings extracted from various layers of the LLM (weighted and combined via learnable scalars) are used as sparse prompt embeddings. These three embeddings, together with the image feature from SAM's frozen ViT image encoder, are input into the refiner to generate the fine-grained mask.
    - **Design Motivation**: Attention maps provide spatial and geometric cues but have low resolution and coarse boundaries. A U-Net can effectively translate these multi-channel priors, and SAM's high-quality mask prior is utilized to optimize boundary precision. The entire mask head has an extremely small parameter count.

3. **Keyword Selector**:

    - **Function**: Automatically identify object keywords in the text that require visual grounding.
    - **Mechanism**: A linear layer is placed on top of the LLM's transformer layers to map the $d$-dimensional hidden states to 1-dimensional scores, which are normalized to $[0,1]$ via sigmoid. During training, this is supervised using BCE loss. During inference, words with scores exceeding a threshold $\lambda=0.3$ are labeled as positive. Consecutive positive words are merged into a single object.
    - **Design Motivation**: External NLP tools like SpaCy parse out too many non-object nouns, introducing noise; training a lightweight linear layer to directly determine which words need grounding is much more precise (improving F1 score from 57.8% with SpaCy to 82.8%).

### Loss & Training

The mask decoder and mask refiner are trained using BCE + DICE loss, respectively. The keyword selector is trained with BCE loss. The entire training process only utilizes the RefCOCO(+/g) and PNG datasets (approximately 190K samples), taking about 20 hours on 8 A800-40G GPUs, with a batch size of 8, for 8 epochs, using the AdamW optimizer with a learning rate of 1e-4.

## Key Experimental Results

### Main Results

| Model | MME↑ | MMBench↑ | RefCOCO cIoU↑ | PNG All↑ |
|------|------|----------|---------------|----------|
| GLaMM-FS-7B | 14/9 | 36.8 | 78.6 | 55.8 |
| LISA-7B | 1/1 | 0.4 | 74.9 | - |
| PixelLM-7B | 309/135 | 17.4 | 73.0 | 43.1 |
| F-LMM (LLaVA-1.6-M-7B) | 1501/324 | 69.5 | 75.7 | 66.5 |
| F-LMM (DeepSeekVL-7B) | 1468/298 | 73.2 | 76.1 | 65.7 |

F-LMM completely preserves the performance of the original LMM on conversational benchmarks (MME 1500+ vs. 0-300 for fine-tuning methods), while achieving competitive grounding performance.

### Ablation Study

| Configuration | PNG All | PNG Thing | PNG Stuff |
|------|---------|-----------|-----------|
| Without mask refiner | 50.8 | 48.6 | 55.9 |
| + mask prompt | 63.4 | 62.0 | 66.8 |
| + mask + box prompt | 63.7 | 62.2 | 67.1 |
| + mask + box + text prompt | 64.9 | 63.4 | 68.3 |

### Key Findings

- The SAM-based mask refiner contributes the most (+14.1% PNG), showing that though attention maps contain valid spatial priors, they require refinement.
- Textual cues (text embeddings) provide an additional 1.2% improvement to the refiner, indicating the complementarity of multimodal information.
- F-LMM significantly leads on the long-sentence subset of reasoning-segmentation (49.1% vs LISA 36.6%), benefiting from its fully preserved reasoning capabilities.
- The keyword selector achieves an F1 of 82.8%, far exceeding SpaCy's 57.8% (with precision improving from 41.1% to 72.5%).

## Highlights & Insights

- **"Frozen is Optimal" Design Philosophy**: For grounding tasks, keeping the LMM frozen and employing a lightweight decoder not only preserves conversational capabilities but also achieves competitive grounding performance. This suggests that instead of investing massive resources down the line to collect high-quality annotated data to mitigate catastrophic forgetting induced by fine-tuning, avoiding fine-tuning altogether is a better path.
- **Attention Maps as "Free" Segmentation Priors**: Although LMMs are never trained with pixel-level annotations, their attention mechanisms already encode sufficient spatial information. This finding is highly insightful for understanding the internal representations of large models.
- **Plug-and-Play with Arbitrary LMMs**: F-LMM does not rely on specific LMM architectures and can be directly applied to 10+ different LMMs, making this generalizability highly valuable.

## Limitations & Future Work

- Grounding precision (cIoU) is still about 2-3 percentage points lower than fine-tuned methods (75.7% vs. 78.6%), which may be insufficient for scenarios requiring extremely high precision.
- It is only validated on MLP-based LMMs (which preserve 2D topological structure), and extra adaptation is needed for LMMs using cross-attention (such as Flamingo).
- The quality of the attention maps depends on the LMM's own vision encoder; if the base LMM's visual understanding is weak, F-LMM's grounding performance will degrade accordingly.
- The training data only utilized RefCOCO(+/g) and PNG. Incorporating broader types of segmentation data (such as panoptic segmentation) might yield further improvements.

## Related Work & Insights

- **vs LISA/GLaMM/PixelLM**: These methods fine-tune LMMs to learn [SEG] tokens. They achieve high grounding accuracy but almost completely lose conversational capabilities, whereas F-LMM perfectly preserves conversational capabilities while delivering competitive grounding.
- **vs LLaVA-G**: LLaVA-G attempts to balance the two using annotated conversational data, which is extremely costly and still leads to a decline in conversational capabilities; F-LMM does not require any conversational annotations.
- **vs SAM**: SAM provides high-quality mask priors but requires manual prompts; F-LMM automatically discovers objects from the dialogue and generates prompts.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The approach of freezing LMMs for grounding is novel and counter-intuitive, and the discovery of attention maps as segmentation priors is highly insightful.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ It covers 10+ LMMs across five tasks (QA + RES + PNG + reasoning segmentation + GCG) with exhaustive ablation studies.
- Writing Quality: ⭐⭐⭐⭐ The problem definition is clear and the motivation is strong (the zero scores for fine-tuned methods in Table 1 are highly striking).
- Value: ⭐⭐⭐⭐⭐ An elegant "best-of-both-worlds" solution is proposed, providing highly valuable insights for building general-purpose AI assistants.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] PARTONOMY: Large Multimodal Models with Part-Level Visual Understanding](../../NeurIPS2025/segmentation/partonomy_large_multimodal_models_with_part-level_visual_understanding.md)
- [\[CVPR 2025\] StoryGPT-V: Large Language Models as Consistent Story Visualizers](storygpt-v_large_language_models_as_consistent_story_visualizers.md)
- [\[ICCV 2025\] HiMTok: Learning Hierarchical Mask Tokens for Image Segmentation with Large Multimodal Model](../../ICCV2025/segmentation/himtok_learning_hierarchical_mask_tokens_for_image_segmentation_with_large_multi.md)
- [\[ECCV 2024\] OpenPSG: Open-set Panoptic Scene Graph Generation via Large Multimodal Models](../../ECCV2024/segmentation/openpsg_open-set_panoptic_scene_graph_generation_via_large_multimodal_models.md)
- [\[CVPR 2025\] Scale Efficient Training for Large Datasets](scale_efficient_training_for_large_datasets.md)

</div>

<!-- RELATED:END -->
