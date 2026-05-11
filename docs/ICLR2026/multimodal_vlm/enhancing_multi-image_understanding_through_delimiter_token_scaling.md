---
title: >-
  [Paper Note] Enhancing Multi-Image Understanding through Delimiter Token Scaling
description: >-
  [ICLR 2026][Multimodal VLM][multi-image understanding] By scaling the hidden states of image delimiter tokens in vision-language models, this work enhances inter-image information isolation and achieves performance gains…
tags:
  - "ICLR 2026"
  - "Multimodal VLM"
  - "multi-image understanding"
  - "large vision-language models"
  - "delimiter tokens"
  - "cross-image information leakage"
  - "attention mechanism"
date: 2026-05-08
content_hash: 92c75ba96f4ce31e
---

# Enhancing Multi-Image Understanding through Delimiter Token Scaling

**Conference**: ICLR 2026
**arXiv**: [2602.01984](https://arxiv.org/abs/2602.01984)
**Code**: [GitHub](https://github.com/MYMY-young/DelimScaling)
**Area**: Multimodal / VLM
**Keywords**: multi-image understanding, large vision-language models, delimiter tokens, cross-image information leakage, attention mechanism

## TL;DR
By scaling the hidden states of image delimiter tokens in vision-language models, this work enhances inter-image information isolation and achieves performance gains on multi-image understanding benchmarks (Mantis/MuirBench/MIRB/QBench2) and multi-document/multi-table understanding benchmarks (TQABench/MultiNews/WCEP-10) without introducing any additional training or inference cost.

## Background & Motivation
Large vision-language models (LVLMs, e.g., LLaVA, InternVL) have achieved strong performance on single-image tasks, but exhibit notable performance degradation when processing multi-image inputs. A central cause is **cross-image information leakage**—the model fails to distinguish information originating from different images, leading to incorrect attribution during reasoning.

Existing LVLMs already employ delimiter tokens (e.g., `<image_start>` and `<image_end>`) to demarcate the boundaries of each image, yet these delimiters fail in practice to prevent cross-image information leakage. During self-attention computation, visual tokens from different images continue to interact, diluting image-specific information.

**Key Challenge**: While delimiter tokens encode boundary information, their hidden-state magnitudes are insufficient to form effective "information barriers" in attention computation.

**Key Insight**: Remarkably straightforward—directly scale the hidden states of delimiter tokens by a scalar factor $\alpha > 1$, thereby amplifying their isolation effect within the attention mechanism. This intervention is applied at inference time without any model retraining.

## Method

### Overall Architecture
The input is a multimodal sequence containing multiple images and a text prompt, with delimiter tokens interspersed between image tokens. The method multiplies the hidden states of delimiter tokens by a scaling factor $\alpha > 1$ at intermediate layers or all layers, and the output is the model's final prediction. The entire procedure is a training-free inference-time intervention.

### Key Designs

1. **Delimiter Token Scaling**:

    - **Function**: Within Transformer layers, the hidden states of delimiter tokens (special tokens marking image boundaries) are multiplied by a scaling factor $\alpha$.
    - **Mechanism**: The amplified delimiter hidden states receive larger attention weights under softmax attention computation, thereby forming an "information bottleneck" or "isolation barrier" in the attention distribution.
    - **Design Motivation**: Analysis reveals that although delimiter tokens are present in the sequence, their hidden-state norms are not prominent relative to visual tokens, and thus fail to serve their expected role as boundary markers in attention computation. The scaling operation directly reinforces this signal.

2. **Enhanced Intra-Image Interaction and Suppressed Cross-Image Interaction**:

    - The scaled delimiter tokens act as "information barriers," causing visual tokens within the same image to attend more strongly to one another (enhanced intra-image interaction).
    - Simultaneously, attention interactions between visual tokens from different images are suppressed (reduced cross-image interaction).
    - As a result, the model better preserves image-specific information, leading to more accurate reasoning when distinguishing and comparing multiple images.

3. **Training-Free, Zero Additional Cost**:

    - The method is a purely inference-time intervention requiring no additional training.
    - No new parameters or modules are introduced.
    - Computational overhead at inference is negligible (a single scalar multiplication at specific token positions).

### Loss & Training
No training is required. The method is a direct inference-time intervention, with the only hyperparameters being the scaling factor $\alpha$ and the range of layers to which it is applied.

## Key Experimental Results

### Main Results
The paper evaluates the method on multiple multi-image understanding benchmarks:

| Dataset | Task Type | Effect |
|---------|-----------|--------|
| Mantis | Multi-image reasoning | Gain |
| MuirBench | Multi-image understanding | Gain |
| MIRB | Multi-image reasoning | Gain |
| QBench2 | Image quality comparison | Gain |

The method is further validated on text-only tasks requiring disambiguation of different textual entities:

| Dataset | Task Type | Effect |
|---------|-----------|--------|
| TQABench | Multi-table understanding | Gain |
| MultiNews | Multi-document summarization/understanding | Gain |
| WCEP-10 | Multi-document event understanding | Gain |

### Ablation Study

| Configuration | Key Findings | Notes |
|---------------|-------------|-------|
| Scaling factor $\alpha$ | An optimal range exists | Too small yields negligible effect; too large may disrupt the model's original distribution |
| Layer range | Middle layers are most effective | Early and final layers tend to contribute less |
| Delimiter type | Both start and end delimiters are effective | Scaling both token types contributes to performance gains |

### Key Findings
- Delimiter tokens in existing LVLMs, while present in the sequence, fail to function as effective boundary markers at the hidden-state level.
- A simple scaling operation substantially enhances their functionality, indicating that the problem lies not in architectural design but in insufficient learning of delimiter representations during training.
- The method is effective not only for visual delimiters but also for textual delimiters (separating multiple documents or tables), demonstrating the generality of the underlying mechanism.
- The approach is model-agnostic and can be applied to a variety of LVLMs.

## Highlights & Insights
- **Minimal intervention, significant effect**: Improving multi-image understanding through hidden-state scaling alone is remarkably compelling in its simplicity.
- **Zero cost**: A genuine "free lunch"—no training, no additional parameters, and negligible inference overhead.
- **General mechanism**: The extension from visual delimiters to textual delimiters (multi-document/multi-table settings) demonstrates that this is a general phenomenon in attention mechanisms, not one specific to the visual modality.
- **Diagnostic insight**: The analysis of why delimiter tokens fail (insufficient hidden-state norms to influence attention distributions) provides valuable understanding of the internal workings of LVLMs.
- **High practical utility**: Directly applicable to any existing LVLM without retraining, making it suitable for immediate deployment.

## Limitations & Future Work
- The scaling factor $\alpha$ requires manual tuning, and different models and tasks may demand different optimal values.
- As an inference-time intervention, explicitly incorporating delimiter learning during training may yield further improvements.
- The HTML version of the paper fails to render on ar5iv, making it difficult to obtain complete experimental details.
- For particularly long multi-image sequences (e.g., video frames), the scaling strategy may require further adaptation.
- Comparisons or combinations with other attention intervention methods (e.g., attention masking, positional encoding modifications) are not explored.
- The method has not been tested on the latest ultra-large-scale LVLMs (e.g., GPT-4V).

## Related Work & Insights
- Unlike visual token compression methods (e.g., TrimTokenator-LC, VisionTrim), which focus on efficiency, this work addresses information isolation quality in multi-image settings.
- Unlike training-based methods specifically designed for multi-image understanding, this paper provides a complementary training-free approach.
- Insight: The "signal strength" of special tokens in the attention mechanism may be an overlooked design dimension—future LVLM training may need to explicitly encourage delimiter tokens to learn stronger boundary representations.
- The simple dry intervention of "scaling hidden states" may generalize to other scenarios requiring information isolation, such as distinguishing turns in multi-turn dialogue or separating retrieved documents in RAG pipelines.

## Rating
- Novelty: ⭐⭐⭐⭐ — Both the observation and the method are novel, though technical complexity is relatively low.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers multiple benchmarks and task types, including ablation studies and cross-modal validation.
- Writing Quality: ⭐⭐⭐⭐ — Motivation is clearly articulated and the method is concisely presented (assessed from the abstract and code, as the full HTML is unavailable).
- Value: ⭐⭐⭐⭐⭐ — Extremely high practical value; any LVLM user can apply it immediately.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] DIVA-GRPO: Enhancing Multimodal Reasoning through Difficulty-Adaptive Variant Advantage](diva-grpo_enhancing_multimodal_reasoning_through_difficulty-adaptive_variant_adv.md)
- [\[ICLR 2026\] TableDART: Dynamic Adaptive Multi-Modal Routing for Table Understanding](tabledart_dynamic_adaptive_multi-modal_routing_for_table_understanding.md)
- [\[ICLR 2026\] MMR-Life: Piecing Together Real-life Scenes for Multimodal Multi-image Reasoning](mmr-life_piecing_together_real-life_scenes_for_multimodal_multi-image_reasoning.md)
- [\[ICLR 2026\] Index-Preserving Lightweight Token Pruning for Efficient Document Understanding](index-preserving_lightweight_token_pruning_for_efficient_document_understanding_.md)
- [\[CVPR 2026\] ViKey: Enhancing Temporal Understanding in Videos via Visual Prompting](../../CVPR2026/multimodal_vlm/vikey_enhancing_temporal_understanding_in_videos_via_visual_prompting.md)

</div>

<!-- RELATED:END -->
