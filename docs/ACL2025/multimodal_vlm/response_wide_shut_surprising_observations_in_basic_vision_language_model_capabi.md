---
title: >-
  [Paper Note] Response Wide Shut: Surprising Observations in Basic Vision Language Model Capabilities
description: >-
  [ACL 2025][Multimodal VLM][Vision-Language Models] By training linear probes on three intermediate representation spaces of VLMs (visual encoder, VL projection layer, and language decoder), this study systematically reveals a counter-intuitive phenomenon: for most visual tasks, the visual encoder and the VL projection layer actually retain sufficient visual information, and the real bottleneck lies in the representation space of the language decoder—where a significant amount…
tags:
  - "ACL 2025"
  - "Multimodal VLM"
  - "Vision-Language Models"
  - "VLM Analysis"
  - "Linear Probe"
  - "Intermediate Representation Spaces"
  - "Visual Encoder"
  - "Information Loss"
date: 2026-05-08
content_hash: 5421454cc3e7731f
---

# Response Wide Shut: Surprising Observations in Basic Vision Language Model Capabilities

**Conference**: ACL 2025  
**arXiv**: [2507.10442](https://arxiv.org/abs/2507.10442)  
**Authors**: Shivam Chandhok, Wan-Cyuan Fan, Vered Shwartz, Vineeth N Balasubramanian, Leonid Sigal (UBC, Vector Institute, IIT Hyderabad, MSR India)  
**Code**: Unreleased  
**Area**: Multimodal VLM  
**Keywords**: Vision-Language Models, VLM Analysis, Linear Probe, Intermediate Representation Spaces, Visual Encoder, Information Loss  

## TL;DR

By training linear probes on three intermediate representation spaces of VLMs (visual encoder, VL projection layer, and language decoder), this study systematically reveals a counter-intuitive phenomenon: for most visual tasks, the visual encoder and the VL projection layer actually retain sufficient visual information, and the real bottleneck lies in the representation space of the language decoder—where a significant amount of information is lost during transmission from the projection layer to the final text output.

## Background & Motivation

### Background
VLMs perform exceptionally well on complex tasks (such as chart understanding and humor detection) but frequently exhibit shortfalls in basic visual capabilities, such as failing to understand simple negation, struggling with accurate counting, and displaying poor fine-grained recognition. This paradox of "strong high-level capabilities, weak basic capabilities" suggests that the mechanism VLMs use to solve complex tasks may differ from that of humans, relying more on large-scale pattern matching and memory retrieval rather than functional visual reasoning.

### Limitations of Prior Work
Prior studies (e.g., Peng et al. 2024, Paiss et al. 2023, Kim & Ji 2024) only evaluate the final text responses of VLMs, which merely diagnoses "whether the model works" from a black-box perspective, failing to pin down "which module is at fault." Eyes Wide Shut (EWS) (Tong et al. 2024) attributes the issue to the inadequacy of the visual encoder, but this conclusion may be overly simplistic.

### Core Motivation
To open the black box of VLMs and precisely localize at which stage information is lost by dissection of the performance across three intermediate representation spaces, thereby providing targeted guidance for improving VLM architectures and training strategies.

## Method

### Overall Architecture
The VLM is dissected into representation spaces corresponding to three key modules for independent evaluation:
1. **Visual Space (Visual)**: Representations output by the visual encoder (e.g., CLIP ViT).
2. **VL Projection Space (VL Proj)**: Representations after vision-language alignment projection (e.g., output of MLP or Q-Former).
3. **Response Space (Response)**: The output of the language decoder, evaluated in two ways—training a probe on token embeddings (Probe) and directly evaluating the textual response (Text).

For the first two spaces, a single-layer linear probe (logistic regression) is trained on frozen features to assess whether the space encodes sufficient task-relevant information. For the response space, both a probe is trained (to maintain consistency in evaluation) and the textual output is directly evaluated via visual question answering (VQA) formats (the standard paradigm of using VLMs).

### Key Designs

#### Control Tasks to Validate Probe Effectiveness
To rule out the possibility that the linear probe only learns the task from the training data rather than reading information from the representation, a control task is established: retraining the probe after randomly shuffling the labels. If the probe performs well on normal labels but poorly on randomized labels, it indicates that the probe indeed utilizes the information in the representation rather than learning the task from scratch.

#### Multi-Task and Multi-Model Comprehensive Coverage
- **Task Coverage**: Coarse-grained recognition (PaintSkills, Pascal VOC), fine-grained recognition (Stanford Dogs, CUB Birds), object counting (1-4 objects), and spatial relation understanding (above, below, left, right).
- **Model Coverage**: 7 VLMs across three categories—contrastive (CLIP, ALBEF), encoder-decoder (CoCa, BLIP-2), and instruction-tuned (InstructBLIP, LLaVA-1.5, LLaVA-NEXT).
- All tasks are formulated as classification problems and converted into VQA prompt formats.

#### Background Alterations and Robustness Analysis
Eleven image transformations are designed to investigate how VLMs handle foreground/background information: pure black/white backgrounds, contour masking, inverse blurring, red circular annotations, edge maps, patch shuffling, etc. The response differences of the three spaces to these transformations are observed.

## Key Experimental Results

### Main Results

#### Table 1: Coarse-grained vs. Fine-grained Recognition Performance Comparison

| Task | Model | Visual Probe | VL Proj Probe | Response Probe | Response Text |
|------|------|-------------|---------------|---------------|---------------|
| Coarse-grained (Mean) | LLaVA-NEXT 7B | 98.6 | 98.2 | 94.2 | 95.9 |
| Coarse-grained (Mean) | InstructBLIP 7B | 99.0 | 98.9 | 96.8 | 84.0 |
| Fine-grained (Mean) | LLaVA-NEXT 7B | 90.2 | 86.1 | **30.0** | **24.5** |
| Fine-grained (Mean) | InstructBLIP 7B | 92.9 | 93.3 | **44.6** | **12.5** |
| Fine-grained (Mean) | LLaVA-1.5 7B | 92.1 | 88.0 | **30.0** | **27.3** |
| Fine-grained (Mean) | BLIP-2 7B | 92.8 | 92.6 | **39.3** | **16.8** |

While the performance gap across the three spaces is small in coarse-grained recognition (around 5%), the performance in the response space drops sharply by at least 45 percentage points in fine-grained recognition—even though both the visual encoder and the projection layer reach 90%+, the response space only yields 12–44%.

#### Table 2: Information Flow Comparison for Counting and Spatial Understanding

| Task | Model | Visual Probe | VL Proj Probe | Response Text |
|------|------|-------------|---------------|---------------|
| Counting | LLaVA-NEXT 7B | 94.4 | 95.7 | 81.2 |
| Counting | InstructBLIP 7B | 96.6 | 95.6 | 82.0 |
| Counting | BLIP-2 7B | 96.6 | 95.3 | 25.0 |
| Spatial | LLaVA-1.5 7B | 50.2 | 51.0 | **63.0** |
| Spatial | LLaVA-1.5 13B | 50.5 | 49.0 | **74.0** |
| Spatial | LLaVA-NEXT 7B | 49.8 | 49.8 | 37.6 |

The counting task shares the same trend as fine-grained recognition (strong visual space, weak response space). However, the spatial understanding task exhibits a **reversed trend**—the performance in the visual space is close to random (~50%), while the response space performs significantly better. This indicates that the bottleneck for spatial understanding indeed lies in the visual encoder, rather than the language decoder.

### Ablation Study

#### Table 3: Impact of Background Alterations on Different Spaces (COCO Dataset, LLaVA-NEXT)

| Alteration | Visual Probe | VL Proj Probe | Response Text |
|------|-------------|---------------|---------------|
| Original | 77.5 | 74.5 | 67.5 |
| Black BG | 88.3 (+10.8) | 87.0 (+12.5) | 72.5 (+5.0) |
| Inverse Blur | 87.8 (+10.3) | 87.5 (+13.0) | 74.2 (+6.7) |
| Silhouette + White BG | 47.9 (-29.6) | 44.7 (-29.8) | 20.3 (-47.2) |
| Edge Map | 62.0 (-15.5) | 60.1 (-14.4) | 52.4 (-15.1) |
| Patch Shuffle | 73.5 (-4.0) | 70.1 (-4.4) | 65.2 (-2.3) |

Removing the background leads to an improvement of approximately 11–12% in the visual and projection spaces, whereas the gains are halved in the response space (only around 5%), further confirming the loss of information from intermediate layers to the response layer. "Inverse blur" acts as the "best strategy" since it balances foreground focus and background context preservation.

## Key Findings

1. **The Visual Encoder is Not the Bottleneck**: Contrary to prior studies like EWS, the visual encoder represents sufficient information across most tasks (probe accuracy >90%), and the VL projection layer largely preserves this information.
2. **The Language Decoder is an Information Black Hole**: In fine-grained recognition, the performance of the response space plummets by more than 45% compared to the visual space; counting tasks also see a drop of at least 14%. The information is clearly present in the intermediate layers but fails to surface in the final output.
3. **Spatial Understanding is the Sole Exception**: For this task, the visual encoder (CLIP) is indeed the bottleneck, hovering around random chance (~50%), whereas the language decoder conversely compensates partially via spatial examples present in the fine-tuning data.
4. **Scaling Up LLM Size Offers Limited Benefit**: Moving LLaVA-NEXT from 7B to 34B only increases the response space accuracy in fine-grained recognition from 31.0% to 49.4%, which is still over 35% lower than the visual space.
5. **Texture Trumping Shape**: Edge maps (retaining shape but removing texture) cause severe performance drops, whereas patch shuffling (disrupting shape but preserving texture) has a minor impact, indicating that VLMs rely more heavily on texture information compared to humans.
6. **"Pseudo-Robustness" of the Response Space**: The response space seems most robust to visual corruptions, but this is not because it handles noise well; rather, it is because it underutilizes visual information to begin with.

## Highlights & Insights

- **Novel Analysis Paradigm**: This work is the first to systematically compare VLM performance across three intermediate representation spaces, refining the vague question of "where VLMs fail" into "which module loses what information."
- **Overturning Prevailing Beliefs**: It robustly refutes the mainstream assumption that "the visual encoder is the performance bottleneck for the grounding capabilities of VLMs," turning the spotlight instead on the insufficient co-fine-tuning between the VL projection layer and the language decoder.
- **Data-Driven Explanations**: It finds that only 0.17% of the LLaVA 665k fine-tuning data involves fine-grained dog breeds, and only 0.2% of LAION-2B involves spatial relationships, directly unearthing the data-driven root causes behind the performance disparities.
- **Practical Actionable Improvements**: It explicitly shows that increasing fine-grained examples and improving the co-training of the projection layer and the decoder are more effective strategies than swapping out the visual encoder.
- **Exquisitely Designed Background Alteration Experiments**: The eleven image transformations systematically cover multiple dimensions including foreground/background separation, visual prompting, and shape/texture decoupling.

## Limitations & Future Work

- **Limited to Open-source Models**: Due to the requirement of accessing intermediate features, closed-source models such as GPT-4V and Gemini could not be evaluated, although they might differ fundamentally in architecture and training data.
- **Limited Task Coverage**: The work only covers three basic tasks—recognition, counting, and spatial understanding—without spanning more complex capabilities like segmentation, visual reasoning, or OCR.
- **Model Recency**: The analyzed models (LLaVA-1.5/NEXT, BLIP-2, etc.) are no longer the absolute state-of-the-art. Whether the same issues persist in newer models (such as LLaVA-OneVision, Qwen-VL, etc.) remains to be validated.
- **Limitations of Linear Probes**: Linear probes can only detect linearly separable information, potentially underestimating non-linearly encoded visual knowledge.
- **Difficulty in Controlling Training Data**: The pre-training and fine-tuning datasets of different VLMs vary greatly, making it difficult to completely rule out the confounding effects of data.

## Related Work & Insights

- **EWS (Tong et al. 2024)** posits that the visual encoder is the bottleneck of VLM performance; this work supports this conclusion for spatial understanding but provides contrarian evidence for classification and counting tasks.
- **Kim & Ji (2024)** find that text-only LLMs already possess fine-grained classification knowledge; combining this with the current findings suggests that the issue lies in the vision-to-language modality alignment (projection layer + co-fine-tuning) rather than any standalone module.
- **Zhang et al. (2024)** compare LLaVA and CLIP on classification capabilities but restrict their focus to the response space; this work extends the depth of comparison to all three spaces.
- **Insights for Future VLM Design**: Rather than investing resources to swap or upgrade visual encoders, it is more beneficial to focus on improving the VL projection mechanism and co-fine-tuning strategies. Increasing the proportion of fine-grained and spatial reasoning samples in the fine-tuning data may offer a highly cost-effective path to improvement.

## Rating

- Novelty: ⭐⭐⭐⭐ — The three-space probe analysis framework is a novel and inspiring analysis paradigm.
- Experimental Thoroughness: ⭐⭐⭐⭐ — 7 models × 4 tasks × multiple transformations, ensuring broad coverage and well-executed control experiments.
- Writing Quality: ⭐⭐⭐⭐ — Solid and clear structure, step-by-step reasoning, and highly informative tables.
- Value: ⭐⭐⭐⭐ — Offers critical diagnostic tools and improvement pathways for the VLM community, although it lacks validation experiments (e.g., verifying whether doing the suggested improvements actually resolves the issue).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] R-VLM: Region-Aware Vision Language Model for Precise GUI Grounding](r-vlm_region-aware_vision_language_model_for_precise_gui_grounding.md)
- [\[ECCV 2024\] AddressCLIP: Empowering Vision-Language Models for City-wide Image Address Localization](../../ECCV2024/multimodal_vlm/addressclip_empowering_vision-language_models_for_city-wide_image_address_locali.md)
- [\[ACL 2025\] Centurio: On Drivers of Multilingual Ability of Large Vision-Language Model](centurio_multilingual_vlm.md)
- [\[ACL 2025\] LogicQA: Logical Anomaly Detection with Vision Language Model Generated Questions](logicqa_logical_anomaly_detection_with_vision_language_model_generated_questions.md)
- [\[ACL 2025\] Transferring Textual Preferences to Vision-Language Understanding through Model Merging](transferring_textual_preferences_to_vision-language_understanding_through_model_.md)

</div>

<!-- RELATED:END -->
