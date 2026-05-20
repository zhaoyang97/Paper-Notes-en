---
title: >-
  [Paper Note] CAMU: Context Augmentation for Meme Understanding
description: >-
  [AAAI 2026][Multimodal VLM][Hateful meme detection] This paper proposes the CAMU framework, which achieves 0.807 accuracy and 0.806 F1 on the Hateful Memes dataset through visually grounded context caption generation…
tags:
  - "AAAI 2026"
  - "Multimodal VLM"
  - "Hateful meme detection"
  - "multimodal fusion"
  - "CLIP fine-tuning"
  - "visual grounding"
  - "caption generation"
date: 2026-05-08
content_hash: 943f04b5c119ae65
---

# CAMU: Context Augmentation for Meme Understanding

**Conference**: AAAI 2026
**arXiv**: [2504.17902](https://arxiv.org/abs/2504.17902)  
**Code**: To be released  
**Area**: Multimodal VLM
**Keywords**: Hateful meme detection, multimodal fusion, CLIP fine-tuning, visual grounding, caption generation

## TL;DR
This paper proposes the CAMU framework, which achieves 0.807 accuracy and 0.806 F1 on the Hateful Memes dataset through visually grounded context caption generation, a novel caption scoring network, and parameter-efficient n-layer fine-tuning of the CLIP text encoder—matching the 55B-parameter SOTA while being substantially more efficient.

## Background & Motivation

**Background**: Hateful meme detection is a core task in multimodal content moderation. Mainstream approaches leverage the cross-modal alignment capabilities of vision-language models such as CLIP via contrastive learning or projection-layer fine-tuning to classify whether a meme contains hateful content. The current SOTA, PALI-X-VPD, achieves 0.892 AUROC using a 55B-parameter large language model with code generation and chain-of-thought reasoning.

**Limitations of Prior Work**: The meaning of a meme is not a simple superposition of image and text, but a complex fusion arising from cultural context, irony, and implication. Existing methods face two core challenges: (1) the "benign confounders" problem—identical text paired with different images can shift a meme from hateful to non-hateful, making unimodal features unreliable; (2) simple projection-layer fine-tuning (e.g., Hate-CLIPper) is insufficient to capture subtle semantic relationships in memes, while large-model approaches incur prohibitive computational overhead for real-time deployment.

**Key Challenge**: High performance requires deep semantic understanding at high computational cost, yet lightweight methods cannot fully exploit context information augmented by captions.

**Goal**: Design a hierarchical, interpretable framework that achieves high-accuracy hateful content detection through multimodal context augmentation while remaining computationally efficient.

**Key Insight**: The observation that meme text typically does not describe the image but instead constructs meaning jointly with it, necessitating visual grounding to first understand image content, then generating context-augmented captions, and finally driving classification through high-quality caption selection.

**Core Idea**: Use visual grounding combined with an LVLM to generate augmented captions, employ a caption scoring network to select the most relevant caption, and fine-tune only the last $n$ layers of the CLIP text encoder for efficient classification.

## Method

### Overall Architecture
CAMU consists of three hierarchical modules: (1) Visually grounded context augmentation: RAM is used for tag generation and GroundingDINO for open-vocabulary object detection; detection results are fed into an LVLM (InternVL-2.5 / Gemini) to generate multiple candidate captions; (2) Caption scoring and selection: a novel feedforward neural network scores candidate captions and performs differentiable selection via Gumbel-Softmax; (3) Parameter-efficient CLIP fine-tuning: only the last $n$ layers of the text encoder are fine-tuned, with bidirectional cross-attention fusing image and caption features for classification.

### Key Designs

1. **Visually Grounded Context Augmentation**:

    - **Function**: Generate augmented captions that incorporate cultural context and visual details for meme images.
    - **Mechanism**: The RAM model first identifies tags in the image (e.g., "woman," "kitchen"), and GroundingDINO obtains bounding box coordinates for detected objects. This information is fed into InternVL-2.5 or Gemini-2.0-flash, with prompts instructing the LVLM to generate descriptive captions by integrating the original meme text with detected visual elements, emphasizing cultural references and latent meanings.
    - **Design Motivation**: LVLMs still hallucinate on semantically subtle images such as memes. Visual grounding helps the model "see more precisely," reducing hallucinations and capturing fine-grained visual cues relevant to hatefulness.

2. **Caption Scorer**:

    - **Function**: Select the most relevant caption for hateful content detection from multiple candidate captions.
    - **Mechanism**: A three-hidden-layer feedforward network takes the CLIP text encoder's caption feature vector (dimension $d$) as input, processes it through GELU + LayerNorm + Dropout + Weight Normalization layers, and outputs a scalar score. Gumbel-Softmax enables differentiable caption selection. A hate relevance loss $\mathcal{L}_{\text{rel}}$ directly aligns caption scores with ground-truth labels—encouraging the scorer to assign higher scores to hate-relevant captions for hateful images.
    - **Design Motivation**: Caption quality varies across different LVLMs, necessitating a selection mechanism jointly optimized with the downstream task. Conventional methods rely on cosine similarity for selection, whereas the caption scorer learns which captions are most useful for hateful content detection.

3. **Parameter-Efficient n-Layer Text Encoder Fine-Tuning**:

    - **Function**: Efficiently adapt CLIP under limited training data (~8.5K samples).
    - **Mechanism**: Only the last $n$ layers ($n = 1$–$4$) of the text encoder are fine-tuned while the image encoder remains frozen. The selected caption and image features are projected to a higher-dimensional space and fused via bidirectional cross-attention: $\mathbf{I}_{\text{enhanced}} = \mathbf{I}_p + \text{CrossAttn}(\mathbf{I}_p, \mathbf{T}_p, \mathbf{T}_p)$, with text enhancement defined analogously. The total loss is $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{cls}} + \lambda_1 \mathcal{L}_{\text{rel}} + \lambda_2 \mathcal{L}_{\text{cont}}$.
    - **Design Motivation**: Full fine-tuning is prone to overfitting in low-resource settings; the n-layer strategy strikes a balance between parameter efficiency and representational capacity.

### Loss & Training
Three losses are jointly optimized: classification loss $\mathcal{L}_{\text{cls}}$ (binary cross-entropy), hate relevance loss $\mathcal{L}_{\text{rel}}$ (direct supervision of caption scores), and contrastive loss $\mathcal{L}_{\text{cont}}$ (CLIP's original InfoNCE loss). Experiments show that removing the contrastive loss yields the best performance, suggesting that the caption scorer's signal is more precise than contrastive learning. Training uses batch size 64 with gradient accumulation to 512, learning rate $1\times10^{-4}$, 30 epochs with early stopping.

## Key Experimental Results

### Main Results

| Method | AUROC | Acc. | F1 | Params |
|--------|-------|------|----|--------|
| PALI-X-VPD (SOTA) | **0.892** | 0.808 | — | 55B |
| CAMU (XLM-R-ViT-H, n=4, w/o cont) | 0.849 | **0.807** | **0.806** | ~1.1B |
| RGCL-HateCLIPper | 0.867 | 0.788 | — | — |
| Hate-CLIPper | 0.858 | — | — | — |
| Gemini-2.0-flash (zero-shot) | 0.743 | 0.741 | 0.756 | — |

### Ablation Study

| Configuration | AUROC | Acc. | F1 |
|---------------|-------|------|----|
| CLIP-XLM-R-ViT-H/14, n=4, $\mathcal{L}_{\text{cls}}+\mathcal{L}_{\text{rel}}$ | **0.849** | **0.807** | **0.806** |
| CLIP-XLM-R-ViT-H/14, n=4, all three losses | 0.819 | 0.775 | 0.774 |
| CLIP-ViT-L/14, n=4, all three losses | 0.812 | 0.753 | 0.752 |
| CLIP-ViT-B/16, full text encoder | 0.788 | 0.632 | 0.591 |
| CLIP-ViT-L/14, projection-layer fine-tuning | 0.828 | 0.720 | 0.710 |

### Key Findings
- Removing the contrastive loss $\mathcal{L}_{\text{cont}}$ yields the best performance, indicating that the caption scorer's hate relevance loss is more effective than standard contrastive learning.
- Increasing the number of fine-tuned layers consistently improves performance: AUROC rises from 0.795 at $n=1$ to 0.819 at $n=4$ (CLIP-XLM-R-ViT-H/14).
- Projection-layer fine-tuning alone is insufficient to leverage caption information (AUROC only 0.828); deeper text encoder adaptation is required to capture subtle semantics.
- The method achieves state-of-the-art F1 (0.673) on the MultiOFF dataset, demonstrating generalization capability.

## Highlights & Insights
- The joint optimization of the caption scorer with the classifier is particularly elegant—it learns not "which caption is best" but "which caption is most useful for hateful content judgment," making this task-driven selection more effective than heuristic rules.
- The finding that contrastive loss acts as a noise source rather than a signal source in this task has broad implications for CLIP fine-tuning research: standard InfoNCE may be redundant for specific downstream tasks.
- The design of visual grounding as a "pre-understanding" layer is transferable to other tasks requiring compositional semantic understanding, such as advertisement comprehension and sarcasm detection.

## Limitations & Future Work
- The training set contains only ~8.5K samples; scaling to larger datasets such as MMHS150K may substantially improve performance.
- When the visual grounding stage misses critical visual elements (e.g., small or difficult-to-recognize objects), the entire pipeline is constrained.
- Only two candidate caption sources are considered; integrating captions from more LVLMs could further improve performance.
- Exploring intermediate-layer features from the encoder may be worthwhile, as different layers may capture distinct linguistic and semantic nuances.

## Related Work & Insights
- **vs. Hate-CLIPper**: Uses only projection-layer fine-tuning for cross-modal interaction, achieving AUROC 0.858 but unable to deeply understand meme semantics. CAMU attains higher accuracy through n-layer text encoder fine-tuning and caption augmentation.
- **vs. PALI-X-VPD**: Achieves AUROC 0.892 via chain-of-thought reasoning with 55B parameters, but at extreme computational cost. CAMU achieves comparable Accuracy and F1 with far fewer parameters.
- **vs. RGCL-HateCLIPper**: Improves performance through retrieval-augmented contrastive learning, but reliance on cosine similarity may introduce instability. CAMU's caption scorer provides a more reliable signal.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — The joint optimization design of the caption scorer with hate relevance loss is novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Ablations are comprehensive, covering multiple CLIP variants and loss combinations, though the dataset scale is relatively small.
- **Writing Quality**: ⭐⭐⭐⭐ — Structure is clear and experimental tables are informative.
- **Value**: ⭐⭐⭐⭐ — Provides practical guidance for efficient multimodal content moderation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Panda: Test-Time Adaptation with Negative Data Augmentation](panda_test-time_adaptation_with_negative_data_augmentation.md)
- [\[AAAI 2026\] Yes FLoReNce, I Will Do Better Next Time! Agentic Feedback Reasoning for Humorous Meme Detection](yes_florence_i_will_do_better_next_time_agentic_feedback_reasoning_for_humorous_.md)
- [\[ICLR 2026\] Context Tokens are Anchors: Understanding the Repetition Curse in dMLLMs from an Information Flow Perspective](../../ICLR2026/multimodal_vlm/context_tokens_are_anchors_understanding_the_repetition_curse_in_dmllms_from_an_.md)
- [\[CVPR 2026\] Text-Only Training for Image Captioning with Retrieval Augmentation and Modality Gap Correction](../../CVPR2026/multimodal_vlm/text-only_training_for_image_captioning_with_retrieval_augmentation_and_modality.md)
- [\[NeurIPS 2025\] MIDAS: Misalignment-based Data Augmentation Strategy for Imbalanced Multimodal Learning](../../NeurIPS2025/multimodal_vlm/midas_misalignment-based_data_augmentation_strategy_for_imbalanced_multimodal_le.md)

</div>

<!-- RELATED:END -->
