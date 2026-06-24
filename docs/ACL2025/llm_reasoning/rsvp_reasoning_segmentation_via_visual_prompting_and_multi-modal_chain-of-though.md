---
title: >-
  [Paper Note] RSVP: Reasoning Segmentation via Visual Prompting and Multi-modal Chain-of-Thought
description: >-
  [ACL 2025 (Main)][Reasoning][Reasoning Segmentation] This paper proposes the RSVP framework, unifying the reasoning capabilities of multimodal large models with visual segmentation through a two-stage structure (reasoning-driven localization + segmentation refinement). Utilizing multimodal chain-of-thought visual prompting, it outperforms the SOTA on ReasonSeg by up to +6.5 gIoU / +9.2 cIoU, and achieves 49.7 mAP on zero-shot SegInW.
tags:
  - "ACL 2025 (Main)"
  - "Reasoning"
  - "Reasoning Segmentation"
  - "Visual Prompting"
  - "Multimodal Chain-of-Thought"
  - "Visual Grounding"
  - "Language-Visual Segmentation"
date: 2026-05-08
content_hash: a83b1414f203f9d8
---

# RSVP: Reasoning Segmentation via Visual Prompting and Multi-modal Chain-of-Thought

**Conference**: ACL 2025 (Main)  
**arXiv**: [2506.04277](https://arxiv.org/abs/2506.04277)  
**Code**: None  
**Area**: LLM Reasoning  
**Keywords**: Reasoning Segmentation, Visual Prompting, Multimodal Chain-of-Thought, Visual Grounding, Language-Visual Segmentation

## TL;DR

This paper proposes the RSVP framework, unifying the reasoning capabilities of multimodal large models with visual segmentation through a two-stage structure (reasoning-driven localization + segmentation refinement). Utilizing multimodal chain-of-thought visual prompting, it outperforms the SOTA on ReasonSeg by up to +6.5 gIoU / +9.2 cIoU, and achieves 49.7 mAP on zero-shot SegInW.

## Background & Motivation

**Background**: Multimodal Large Language Models (MLLMs) have demonstrated excellent cognitive reasoning capabilities on visual reasoning tasks, enabling them to understand complex visual scenes and answer questions about image content. However, these models lack a mechanism to directly project reasoning conclusions into precise visual outputs (such as segmentation masks), resulting in a gap between cognitive reasoning and visual perception.

**Limitations of Prior Work**: Most existing reasoning segmentation methods adopt an end-to-end approach, compressing both reasoning and segmentation into a single model. This approach faces two challenges: (1) Inopaque reasoning process—the model directly jumps from queries to segmentation results without displaying any intermediate reasoning chain, making it difficult to debug and interpret; (2) Limited segmentation precision—although MLLMs excel at semantic reasoning, their outputs are typically text or coarse-grained region indicators, which cannot directly produce pixel-level precise segmentation masks.

**Key Challenge**: High-level semantic reasoning (such as understanding complex queries like "which object is most likely to cause a traffic accident") requires the powerful reasoning capabilities of MLLMs, whereas precise pixel-level segmentation requires specialized vision models. Fitting both into a single model prevents either from achieving optimal performance.

**Goal**: To design an interpretable reasoning segmentation framework where the MLLM focuses on reasoning and localization, while a specialized segmentation module focuses on generating precise masks, with the two parts bridged by structured visual representations.

**Key Insight**: Although MLLMs cannot directly output segmentation masks, they possess an inherent target localization capability—they can indicate target regions through textual descriptions or bounding boxes. Leveraging this capability, MLLMs can be guided to generate structured visual prompts (e.g., region proposals) to serve as inputs for downstream segmentation modules.

**Core Idea**: A two-stage decoupled framework—the first stage (reasoning stage) utilizes an MLLM to perform multi-step reasoning and generate interpretable region proposals, and the second stage (segmentation stage) utilizes a vision-language segmentation module to refine the region proposals into pixel-level masks.

## Method

### Overall Architecture

RSVP is a two-stage pipeline: the input is an image and a natural language query requiring reasoning (e.g., "find the object most likely to cause a traffic accident"), and the output is a precise segmentation mask of the corresponding target object. The first stage (reasoning stage) uses an MLLM to parse the query, perform multi-step reasoning, identify the target, and generate region proposals. The second stage (segmentation stage) utilizes a Vision-Language Segmentation Module (VLSM) to fuse textual and visual cues, refining the region proposals into precise segmentation masks. The two stages are connected via structured visual prompts (such as bounding boxes or visual markers).

### Key Designs

1. **Multimodal Chain-of-Thought Visual Prompting**:

    - **Function**: Guides the MLLM to perform structured multi-step reasoning, progressively deriving target localization from complex queries.
    - **Mechanism**: A prompt engineering strategy is designed to guide the MLLM to reason step-by-step like a "chain of thought": (a) first comprehend the semantic intent of the query; (b) analyze relevant objects and their relationships in the image; (c) lock onto the target object based on the reasoning; and (d) generate region proposals for the target (bounding box coordinates or visual markers). The entire reasoning process is output in natural language, offering high interpretability. The introduction of visual prompting ensures that reasoning is not confined to the language level but is also integrated with visual information for localization.
    - **Design Motivation**: Traditional methods force models to jump directly from questions to answers, bypassing the reasoning process. The chain-of-thought paradigm has proven effective in textual reasoning, and extending it to multimodal scenarios can simultaneously improve reasoning accuracy and interpretability.

2. **Vision-Language Segmentation Module (VLSM)**:

    - **Function**: Refines the coarse-grained region proposals and textual descriptions from the first stage into pixel-level precise segmentation masks.
    - **Mechanism**: The VLSM receives three inputs—the visual features of the original image, the region proposals from the first stage (acting as visual priors), and the textual descriptions generated during the reasoning process. The module fuses textual and visual cues via a cross-modal attention mechanism and utilizes the region proposals as spatial attention priors to focus on the target region, ultimately outputting the target's segmentation mask. This design enables the segmentation module to leverage semantic information to resolve visual ambiguities.
    - **Design Motivation**: Relying solely on region proposals (such as bounding boxes) for segmentation discards semantic information, whereas pure text-driven segmentation lacks accuracy when handling fine boundaries. VLSM leverages information from both modalities to complement each other.

3. **Reasoning-Segmentation Decoupling**:

    - **Function**: Allocates cognitive reasoning and visual perception to the modules that excel at each, respectively.
    - **Mechanism**: The MLLM (e.g., GPT-4V, LLaVA) is responsible for high-level reasoning—understanding query intent, analyzing scenes, and inferring target identities. The VLSM handles low-level perception—generating precise, pixel-level segmentations. The two are decoupled and connected via structured intermediate representations (region proposals + textual descriptions). This architecture allows for independent replacement or upgrading of either module.
    - **Design Motivation**: End-to-end methods conflate reasoning and segmentation, causing reasoning capacity and segmentation accuracy to constrain each other. When decoupled, each module can achieve optimal performance on its respective task, making the overall system more modular, easier to maintain, and simpler to upgrade.

### Loss & Training

The VLSM is trained using standard segmentation losses (a combination of cross-entropy and Dice loss). The reasoning stage utilizes the in-context learning capability of the MLLM and does not require additional training. The overall framework emphasizes a reasoning and compositional approach rather than end-to-end fine-tuning.

## Key Experimental Results

### Main Results

| Dataset / Metric | RSVP | Prev. SOTA | Gain |
|------------|------|---------|------|
| ReasonSeg gIoU | RSVP | Previous Best | **+6.5** |
| ReasonSeg cIoU | RSVP | Previous Best | **+9.2** |
| SegInW mAP (Zero-shot) | **49.7** | - | New Zero-shot SOTA |

On the ReasonSeg benchmark, RSVP substantially outperforms all existing methods across both gIoU and cIoU metrics. Under the zero-shot setting of Segmentation in the Wild (SegInW), RSVP achieves 49.7 mAP without any target domain training data.

### Ablation Study

| Configuration | gIoU | cIoU | Explanation |
|------|------|------|------|
| Full RSVP | Best | Best | Complete two-stage framework |
| W/o Chain-of-Thought | Decline | Decline | Removes multi-step reasoning, directly localizing |
| W/o Visual Prompting | Decline | Decline | Uses text reasoning only, without visual grounding |
| W/o VLSM (using standard segmenter) | Significant Decline | Significant Decline | Refinement without text-visual fusion |
| Single-stage End-to-end | Significant Decline | Significant Decline | Validates the necessity of the decoupled architecture |

### Key Findings

- The two-stage decoupled architecture achieves a significant improvement compared to end-to-end methods, validating the effectiveness of the "reasoning-perception separation" strategy.
- Visual prompting is crucial for localization quality—without the support of visual cues, pure text reasoning is prone to localization bias.
- Chain-of-Thought reasoning is particularly effective for complex queries, whereas the marginal utility for simple queries is minor.
- The outstanding performance in the zero-shot setting demonstrates that the reasoning capabilities of RSVP exhibit strong generalization, without depending on target domain segmentation annotations.

## Highlights & Insights

- **Win-Win for Interpretability and Precision**: The reasoning stage of RSVP outputs comprehensive natural-language reasoning chains, allowing users to see "why" the model selects a specific target. This is particularly valuable in high-stakes scenarios such as medical imaging and autonomous driving. Concurrently, the decoupled design does not sacrifice segmentation accuracy.
- **Leveraging Inherent Localization Capabilities of MLLMs**: While most works focus on training MLLMs to generate segmentation outputs directly, RSVP takes an opposite approach—by not modifying the MLLM but leveraging its native comprehension and localization capabilities, guiding reasoning through prompt engineering. This approach incurs zero training costs and automatically upgrades alongside the capabilities of the MLLM.
- **Scalability of Modular Design**: The reasoning module can be replaced with a stronger MLLM, and the segmentation module can be replaced with a more precise segmenter, allowing both to evolve independently. This philosophy can be seamlessly transferred to other multi-step "reasoning -> execution" tasks.

## Limitations & Future Work

- The two-stage methodology introduces additional reasoning latency, making it unsuitable for real-time response scenarios.
- Highly dependent on the reasoning quality of the MLLM—if target localization fails in the first stage, downstream segmentation cannot recover regardless of its precision.
- The capability to handle complex occlusion scenarios and fine-grained distinctions (e.g., "the third flower from the left") remains to be verified.
- Validation has primarily been conducted on English queries, leaving performance under multilingual reasoning segmentation scenarios unknown.
- Future directions: An auto-verification mechanism for reasoning could be introduced, enabling the model to inspect the segmentation results and correct errors within the reasoning chain.

## Related Work & Insights

- **vs LISA (Lai et al., 2024)**: LISA embeds segmentation tokens into MLLMs for end-to-end training, rendering the reasoning process invisible. RSVP guarantees interpretability through a decoupled design and demonstrates superior zero-shot capabilities.
- **vs SEEM/GroundedSAM**: These methods primarily rely on explicit visual references (such as specified class names or boxes) and lack complex semantic reasoning capabilities. RSVP is capable of processing implicit queries that require multi-step reasoning.
- **vs Set-of-Mark (SoM) Prompting**: SoM annotations regions on images as visual prompts. RSVP further integrates this concept with chain-of-thought reasoning, establishing a structured reasoning-localization-segmentation pipeline.

## Rating

- Novelty: ⭐⭐⭐⭐ A novel paradigm combining multimodal CoT and visual segmentation in a decoupled manner. The concept is clear and intuitive.
- Experimental Thoroughness: ⭐⭐⭐⭐ Demonstrates significant improvements on both ReasonSeg and SegInW, showcasing impressive zero-shot capabilities.
- Writing Quality: ⭐⭐⭐⭐ The paper is well-structured, with a well-motivated argumentation for the two-stage design.
- Value: ⭐⭐⭐⭐ Provides a practical paradigm and reference for explainable visual reasoning segmentation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] CMMCoT: Enhancing Complex Multi-Image Comprehension via Multi-Modal Chain-of-Thought and Memory Augmentation](../../AAAI2026/llm_reasoning/cmmcot_enhancing_complex_multi-image_comprehension_via_multi.md)
- [\[CVPR 2025\] Interleaved-Modal Chain-of-Thought](../../CVPR2025/llm_reasoning/interleaved-modal_chain-of-thought.md)
- [\[CVPR 2026\] Rationale-Enhanced Decoding for Multi-modal Chain-of-Thought](../../CVPR2026/llm_reasoning/rationale-enhanced_decoding_for_multi-modal_chain-of-thought.md)
- [\[ICLR 2026\] Fine-R1: Make Multi-modal LLMs Excel in Fine-Grained Visual Recognition by Chain-of-Thought Reasoning](../../ICLR2026/llm_reasoning/fine-r1_make_multi-modal_llms_excel_in_fine-grained_visual_recognition_by_chain-.md)
- [\[NeurIPS 2025\] Latent Chain-of-Thought for Visual Reasoning](../../NeurIPS2025/llm_reasoning/latent_chain-of-thought_for_visual_reasoning.md)

</div>

<!-- RELATED:END -->
