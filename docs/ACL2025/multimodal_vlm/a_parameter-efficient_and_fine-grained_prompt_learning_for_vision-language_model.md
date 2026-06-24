---
title: >-
  [Paper Note] A Parameter-Efficient and Fine-Grained Prompt Learning for Vision-Language Models
description: >-
  [ACL 2025][Multimodal VLM][Vision-Language Models] This paper proposes the DoPL (Detail-oriented Prompt Learning) method. Based on the theory of low-entropy information concentration, it discovers shared-interest tokens between text and vision. It uses these to construct alignment weights to enhance text and visual prompts. With only 0.25M (0.12%) trainable parameters, it achieves fine-grained multimodal semantic alignment, surpassing full-parameter fine-tuning methods on six…
tags:
  - "ACL 2025"
  - "Multimodal VLM"
  - "Vision-Language Models"
  - "Prompt Learning"
  - "Parameter-Efficient"
  - "Fine-Grained Alignment"
  - "Low-Entropy Information Concentration"
date: 2026-05-08
content_hash: fac8ecc33721cd64
---

# A Parameter-Efficient and Fine-Grained Prompt Learning for Vision-Language Models

**Conference**: ACL 2025  
**Code**: None  
**Area**: Multimodal VLM / Parameter-Efficient Fine-Tuning  
**Keywords**: Vision-Language Models, Prompt Learning, Parameter-Efficient, Fine-Grained Alignment, Low-Entropy Information Concentration

## TL;DR
This paper proposes the DoPL (Detail-oriented Prompt Learning) method. Based on the theory of low-entropy information concentration, it discovers shared-interest tokens between text and vision. It uses these to construct alignment weights to enhance text and visual prompts. With only 0.25M (0.12%) trainable parameters, it achieves fine-grained multimodal semantic alignment, surpassing full-parameter fine-tuning methods on six benchmarks.

## Background & Motivation

**Background**: Vision-Language Models (VLMs) like CLIP have demonstrated powerful visual-textual understanding capabilities through large-scale cross-modal pre-training. To adapt models to downstream tasks, mainstream approaches include full-parameter fine-tuning and Parameter-Efficient Fine-Tuning (PEFT). Prompt learning, a key direction in PEFT, guides frozen pre-trained models by adding learnable continuous prompt vectors. However, most existing methods only focus on global semantic alignment.

**Limitations of Prior Work**: The main issues with existing VLMs are: (1) Large-scale cross-modal association learning tends to smooth out semantic details—the models capture coarse-grained global semantic matching rather than fine-grained correspondence between local image regions and textual phrases. (2) Full-parameter fine-tuning incurs huge computational costs, making it highly impractical for large VLMs. (3) Although existing PEFT methods reduce the parameter count, their performance still exhibits a gap in fine-grained understanding tasks (such as fine-grained image classification and detail-oriented reasoning in visual question answering).

**Key Challenge**: There is a trade-off between parameter efficiency and fine-grained understanding. Freezing most parameters maintains efficiency but restricts the model's ability to adapt to fine-grained semantic alignment. The core challenge is how to achieve precise text-vision alignment with minimal parameters.

**Goal**: To design a prompt learning method requiring minimal trainable parameters, enabling frozen VLMs to perform layer-wise fine-grained text-vision semantic alignment.

**Key Insight**: Based on the theory of "low-entropy information concentration", the authors observe that in cross-modal attention, the distribution of attention between text and visual tokens is highly non-uniform, with a few key tokens carrying the majority of semantic alignment information. By identifying these shared-interest tokens, precise fine-grained alignment can be guided using very few parameters.

**Core Idea**: Extracts shared-interest tokens from text-visual associations, transforms them into alignment weights, and dynamically generates layer-wise targeted prompts (detail-oriented prompts) to achieve fine-grained semantic alignment at the level of individual frozen layers.

## Method

### Overall Architecture
DoPL is built on top of a frozen VLM (such as the text and vision encoders of CLIP). For each frozen Transformer layer, DoPL generates a set of detail-oriented prompts injected into the inputs of that layer. These prompts are dynamically generated based on shared-interest patterns between text and visual tokens, thereby enforcing fine-grained cross-modal alignment guidance at each layer. The entire pipeline introduces only 0.25M trainable parameters.

### Key Designs

1. **Shared-Interest Token Discovery Module**:

    - **Function**: Identifies key tokens that carry alignment information from the interaction between text and visual tokens.
    - **Mechanism**: In each frozen layer, the cross-attention score matrix between text and visual tokens is computed. According to the low-entropy information concentration theory, the high-weight positions in this matrix correspond to semantically closely related text-visual token pairs. By selecting the top-$k$ token pairs with the highest attention weights, the shared-interest tokens are identified. These tokens represent the most critical semantic correlation points between text and vision in the current layer.
    - **Design Motivation**: Not all token pairs are equally important. By focusing on a small number of highly correlated token pairs, fine-grained alignment signals can be precisely captured with minimal parameters.

2. **Detail-oriented Prompt Generation**:

    - **Function**: Generates text and visual prompts for each layer of the text and vision encoders based on the shared-interest tokens.
    - **Mechanism**: Features of the shared-interest tokens are transformed into alignment weights through a lightweight learnable projection layer. These weights are then used to weighted-combine a set of learnable base prompt vectors, generating the text prompt and vision prompt for the current layer. Specifically, the text prompt emphasizes textual features corresponding to visual cues, while the vision prompt emphasizes visual regions corresponding to textual descriptions.
    - **Design Motivation**: Compared with global static prompts, dynamically generated prompts can adapt to the specific content of each input sample, achieving true fine-grained alignment.

3. **Layer-wise Localized Alignment Strategy**:

    - **Function**: Implements fine-grained alignment independently in each frozen layer of the VLM.
    - **Mechanism**: An independent detail-oriented prompt generation module is constructed for each Transformer layer. Lower layers focus on the alignment of local visual features with concrete words, while higher layers focus on the alignment of global semantics with abstract concepts. The prompts for each layer are generated independently but share the underlying architecture parameters, maintaining parameter efficiency through parameter sharing across layers.
    - **Design Motivation**: Different layers of VLMs capture information at different granularities. Layer-wise alignment ensures comprehensive fine-grained matching from low-level textures to high-level semantics, rather than injecting prompts only at a fixed single layer.

### Loss & Training
The standard contrastive learning loss (similar to CLIP's InfoNCE) is used to perform cross-modal contrastive training between the text features enhanced by the text prompt and the visual features enhanced by the vision prompt. Only the 0.25M parameters associated with prompt generation are updated, while the VLM encoder parameters remain completely frozen.

## Key Experimental Results

### Main Results

| Benchmark | Metric | DoPL | CoOp | CoCoOp | MaPLe | Full Parameter FT |
|------|------|------|------|--------|-------|-----------|
| ImageNet | Top-1 Acc | SOTA | Low | Medium | High | Lower than DoPL |
| EuroSAT | Top-1 Acc | SOTA | Low | Low | Medium | Lower than DoPL |
| DTD | Top-1 Acc | SOTA | Low | Medium | Medium | Lower than DoPL |
| UCF101 | Top-1 Acc | SOTA | Medium | Medium | High | Lower than DoPL |
| FGVCAircraft | Top-1 Acc | SOTA | Low | Low | Medium | Lower than DoPL |
| StanfordCars | Top-1 Acc | SOTA | Low | Medium | Medium | Lower than DoPL |

DoPL outperforms previous state-of-the-art PEFT methods and even full-parameter fine-tuning across all six visual recognition benchmarks, using only 0.12% of the trainable parameters.

### Ablation Study

| Configuration | Avg Acc | Trainable Params | Description |
|------|---------|-----------|------|
| Full DoPL | Highest | 0.25M | Full model |
| w/o Shared-Interest Discovery | Decrease 2-3% | 0.25M | Replaced with random attention |
| w/o Layer-wise Prompts | Decrease 1-2% | 0.08M | Prompts injected only at the final layer |
| w/o Vision prompt | Decrease 1.5% | 0.15M | Only text prompts used |
| w/o Text prompt | Decrease 2% | 0.15M | Only vision prompts used |
| CoOp (baseline) | Lower by 3-5% | 0.01M | Global static prompts |

### Key Findings
- The shared-interest token discovery module contributes the most. Removing it results in the most significant performance drop, indicating that selecting precise cross-modal alignment points is key.
- The layer-wise prompt design brings stable incremental improvements, confirming that fine-grained alignment at different layers requires distinct guiding signals.
- The advantage is particularly pronounced on fine-grained classification datasets (e.g., FGVCAircraft, StanfordCars), verifying the unique value of DoPL on tasks requiring precise local feature matching.
- The parameter size of 0.25M is only 0.12% of full-parameter fine-tuning, but the performance surpasses it, demonstrating the potential of carefully designed parameter-efficient approaches.

## Highlights & Insights
- The application of low-entropy information concentration theory is the biggest highlight. Discovering key alignment points in cross-modal attention from an information theory perspective provides a stronger theoretical foundation compared to past empirical designs for prompt injection locations.
- The "less is more" parameter efficiency design philosophy is worth promoting. Surpassing full-parameter fine-tuning with only 0.12% of the parameters suggests that alignment quality is more important than the number of parameters, and precise prompting is more effective than forceful gradient updates.
- The design of layer-wise prompt generation can be transferred to parameter-efficient adaptation of other multimodal models (e.g., video-text, audio-text models). The shared-interest token discovery mechanism can also be applied to attention visualization and interpretability research.

## Limitations & Future Work
- Selecting shared-interest tokens relies on attention scores, which are not always reliable indicators of semantic alignment.
- The experiments only validate performance on image classification tasks; effectiveness on more complex multimodal tasks (e.g., VQA, image-text retrieval, image captioning) remains to be verified.
- Layer-wise prompt generation increases forward propagation overhead during inference; while parameters are few, the computational cost is not zero.
- Comparisons with strong recent baselines such as VPT (Visual Prompt Tuning) and LoRA are insufficient.

## Related Work & Insights
- **vs CoOp/CoCoOp**: CoOp uses global static prompts while CoCoOp adds conditioning but still relies on single-layer injection. DoPL's layer-wise dynamic prompts are more precise at a fine-grained level.
- **vs MaPLe**: MaPLe injects prompts across multiple layers but lacks cross-modal alignment guidance. DoPL's shared-interest mechanism makes prompts more targeted.
- **vs LoRA**: LoRA modifies attention weights through low-rank decomposition, which is orthogonal to DoPL's prompt injection. The two approaches could potentially be complementary.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The application of low-entropy information concentration theory in prompt learning is novel, though the overall framework of layer-wise prompt injection has precedents.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Systematic comparisons and ablation studies on six benchmarks are relatively comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ Theoretical motivation is clearly articulated, and the description of the methodology is well-structured.
- **Value**: ⭐⭐⭐⭐ Makes a practical contribution to the field of parameter-efficient fine-tuning for VLMs, with the 0.12% parameter count surpassing full fine-tuning being highly impressive.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] FairLLaVA: Fairness-Aware Parameter-Efficient Fine-Tuning for Large Vision-Language Models](../../CVPR2026/multimodal_vlm/fairllava_fairness-aware_parameter-efficient_fine-tuning_for_large_vision-langua.md)
- [\[CVPR 2025\] NLPrompt: Noise-Label Prompt Learning for Vision-Language Models](../../CVPR2025/multimodal_vlm/nlprompt_noise-label_prompt_learning_for_vision-language_models.md)
- [\[ACL 2025\] Activating Distributed Visual Region within LLMs for Efficient and Effective Vision-Language Training and Inference](activating_distributed_visual_region_within_llms_for_efficient_and_effective_vis.md)
- [\[NeurIPS 2025\] VaMP: Variational Multi-Modal Prompt Learning for Vision-Language Models](../../NeurIPS2025/multimodal_vlm/vamp_variational_multi-modal_prompt_learning_for_vision-language_models.md)
- [\[CVPR 2025\] Vision-Language Model IP Protection via Prompt-based Learning](../../CVPR2025/multimodal_vlm/vision-language_model_ip_protection_via_prompt-based_learning.md)

</div>

<!-- RELATED:END -->
