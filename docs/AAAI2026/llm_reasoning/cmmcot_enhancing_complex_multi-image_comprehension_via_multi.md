---
title: >-
  [Paper Note] CMMCoT: Enhancing Complex Multi-Image Comprehension via Multi-Modal Chain-of-Thought and Memory Augmentation
description: >-
  [AAAI 2026][LLM Reasoning][Multi-image understanding] This paper proposes the CMMCoT framework, which constructs interleaved multimodal multi-step reasoning chains (with visual region token supervision) and a test-time retrieval-based memory augmentation module (RIFREM) to enhance slow-thinking reasoning in multi-image scenarios without increasing model parameters. Built on Qwen2.5-VL-7B, the method achieves an average improvement of 1.4 points on multi-image benchmarks.
tags:
  - AAAI 2026
  - LLM Reasoning
  - Multi-image understanding
  - multimodal chain-of-thought
  - test-time memory augmentation
  - visual reasoning
  - slow thinking
date: 2026-05-08
content_hash: 49ed0004fd477956
---

# CMMCoT: Enhancing Complex Multi-Image Comprehension via Multi-Modal Chain-of-Thought and Memory Augmentation

**Conference**: AAAI 2026
**arXiv**: [2503.05255](https://arxiv.org/abs/2503.05255)
**Code**: [https://github.com/zhangguanghao523/CMMCoT](https://github.com/zhangguanghao523/CMMCoT)
**Area**: LLM Reasoning
**Keywords**: Multi-image understanding, multimodal chain-of-thought, test-time memory augmentation, visual reasoning, slow thinking

## TL;DR
This paper proposes the CMMCoT framework, which constructs interleaved multimodal multi-step reasoning chains (with visual region token supervision) and a test-time retrieval-based memory augmentation module (RIFREM) to enhance slow-thinking reasoning in multi-image scenarios without increasing model parameters. Built on Qwen2.5-VL-7B, the method achieves an average improvement of 1.4 points on multi-image benchmarks.

## Background & Motivation
Current MLLMs have achieved strong performance in single-image understanding, and O1-style CoT "slow thinking" has demonstrated notable success in mathematical reasoning. However, extending these approaches to **multi-image comprehension** leads to significant performance degradation, due to two key reasons:
1. Existing multimodal CoT methods (e.g., VoCoT, MVoT) are primarily designed for single-image scenarios. Their reasoning processes rely mainly on textual information and lack the ability to explicitly track and compare **cross-image visual concepts** within the reasoning chain.
2. When humans analyze multiple images, they simultaneously perform two cognitive operations: (a) cross-image visual comparison—matching regions of interest across different images; and (b) dynamic memory—continuously retaining key visual concepts throughout the reasoning chain. Neither operation is modeled by existing methods.

Additionally, the absence of training datasets specifically designed for multi-image multimodal CoT severely limits progress in this area.

## Core Problem
1. **Complexity of cross-image visual concept tracking**: Multi-image scenarios require associating visual objects across different images and reasoning jointly with textual information, which is substantially more challenging than single-image tasks. Existing methods supervise only the textual reasoning chain during training, neglecting supervision of the visual reasoning process.
2. **Enhancing reasoning capability at test time**: Test-time scaling is effective in simple scenarios but exhibits diminishing returns in multi-image settings. How can visual understanding be enhanced during inference without adding model parameters?

## Method

### Overall Architecture
CMMCoT is built upon Qwen2-VL / Qwen2.5-VL and consists of two phases: training and inference.
- **Training phase**: Given multiple input images and a question, the model outputs an interleaved multimodal sequence containing reasoning steps, entity coordinates, and entity images. A two-stage training strategy (multi-image data first, then mixed training) is used to learn the multimodal reasoning chain.
- **Inference phase**: The model generates reasoning text; upon predicting coordinates, it extracts entity images from the corresponding image using the predicted index and coordinates. The RIFREM module then performs cross-attention between entity features and multi-image KV pairs stored in a memory bank to enhance subsequent reasoning.

### Key Designs

1. **Interleaved Multimodal Sequence Representation**:

    - Image index tokens `<IMG>0</IMG>` are introduced to reference specific input images.
    - Entity coordinates are represented as `<|box_start|>(x0,y0),(x1,y1)<|box_end|>` (normalized to 0–1000).
    - Entity image visual tokens are delimited by `<|vision_start|>...<|vision_end|>`.
    - During training, entity images are cropped according to coordinates and image indices, encoded via a visual encoder, with a minimum resolution of 512px to capture fine-grained features.
    - **Loss is computed only on text tokens, coordinate tokens, and special tokens**; entity image tokens are excluded from the loss. This is a critical design choice: entity images serve as input context rather than prediction targets.

2. **RIFREM (Retrieval-based Image Feature Reasoning Enhancement Module)**:

    - A memory bank $\mathcal{M}$ is maintained at inference time, storing the Keys and Values of the multi-image input sequence at each decoder layer.
    - When a `</IMG>` token is encountered during reasoning, the corresponding entity image is extracted by its coordinates, injected into the decoder layers, and its query vectors are retrieved.
    - These queries perform standard scaled dot-product cross-attention with the layer-wise KV pairs stored in the memory bank: $Q' = \text{softmax}(\frac{QK_\mathcal{M}^T}{\sqrt{d_k}})V_\mathcal{M}$
    - The enhanced $Q'$ is passed to subsequent reasoning steps, enabling cross-image visual feature mining.
    - Ablation studies show: inserting RIFREM at all 28 layers incurs excessive latency; inserting only at the first and last layers degrades performance; **inserting at 8 uniformly distributed layers achieves the best accuracy–latency trade-off**.

3. **CMMCoT-260K Dataset**:

    - Constructed from GRIT, Flickr30k-Entities, VoCoT, and MANTIS datasets.
    - Covers 4 task types: Caption (50K), Co-reference (90K), Comparison (18K), and Reason (102K).
    - Construction pipeline: GPT-4o generates reasoning chains → Qwen3-235B extracts textual entities → Qwen-VL-max detects entity bounding boxes → GPT-4o verifies IoU ≥ 0.9 → spatial fusion produces unified bounding boxes.

### Loss & Training
- **Two-stage training**:
    - Stage 1: Fine-tuned on CMMCoT-260K, lr=1e-5, 2 epochs.
    - Stage 2: CMMCoT-260K mixed 1:1 with general-purpose datasets, lr=1e-6, 1 epoch, batch size 256.
- Optimizer: AdamW (β=0.95, weight decay=0.1) with cosine lr scheduler.
- DeepSpeed ZeRO-3 is used for distributed training.
- Training prompt: "Please answer the question with reasoning and identify key objects."
- Stage 2 mixed training mitigates catastrophic forgetting induced by multi-image task specialization.

## Key Experimental Results

### Multi-Image Benchmarks (Table 1)

| Model | Params | BLINK | Mantis | NLVR2 | MVBench | Q-Bench | Avg |
|-------|--------|-------|--------|-------|---------|---------|-----|
| Qwen2.5-VL | 7B | 55.3 | 69.8 | 88.3 | 74.7 | 77.7 | 73.2 |
| **Qwen2.5-VL (Ours)** | **7B** | **56.8** | **72.2** | **89.9** | **75.8** | **78.5** | **74.6** |
| InternVL3 | 8B | 55.5 | 70.1 | 88.5 | 75.4 | 75.9 | 73.1 |
| Qwen2.5-VL | 3B | 49.1 | 62.7 | 86.2 | 71.3 | 74.9 | 68.8 |
| **Qwen2.5-VL (Ours)** | **3B** | **51.4** | **68.5** | **88.9** | **73.1** | **75.2** | **71.4** |

### Single-Image Benchmarks (Table 2)

| Model | Params | MMMU | MMStar | SQA | RealWorldQA | MME | POPE | HallBench | Avg |
|-------|--------|------|--------|-----|-------------|-----|------|-----------|-----|
| Qwen2.5-VL | 7B | 58.6 | 63.9 | 89.0 | 68.4 | 82.6 | 85.9 | 51.9 | 71.4 |
| **Qwen2.5-VL (Ours)** | **7B** | 57.5 | **66.4** | **96.8** | **71.6** | **83.5** | **89.2** | **63.6** | **75.5** |
| InternVL3 | 8B | **62.7** | **68.7** | **97.9** | **71.4** | **86.5** | **90.4** | 49.0 | **75.2** |

### Ablation Study

**Module Combination Ablation (Table 3, Qwen2.5-VL-7B)**:

| Grounding | Entity Images | RIFREM | BLINK | Mantis | NLVR2 | MVBench | Q-Bench | Avg |
|-----------|---------------|--------|-------|--------|-------|---------|---------|-----|
| ✘ | ✘ | ✘ | 55.3 | 69.8 | 88.3 | 74.7 | 77.7 | 73.2 |
| ✓ | ✘ | ✘ | 55.2 | 70.4 | 88.7 | 74.5 | 77.9 | 73.3 |
| ✓ | ✓ | ✘ | 57.1 | 71.6 | 89.4 | 75.4 | 78.7 | 74.4 |
| ✓ | ✓ | ✓ | 56.8 | 72.2 | 89.9 | 75.8 | 78.5 | 74.6 |

- **Grounding alone yields only +0.1 points** (limited effect).
- **Grounding + entity images yields +1.2 points** (largest contribution, demonstrating that injecting visual entity features is the core mechanism).
- **Adding RIFREM yields an additional +0.2 points** (moderate but consistent improvement at inference time).

**RIFREM Layer Ablation**:
- Group 1 (first and last 2 layers only): performance degrades due to disruption of information flow.
- Group 3 (8 uniformly distributed layers): best accuracy–latency trade-off.
- Group 5 (all layers): highest accuracy but prohibitive latency.

**Cross-Model Fine-tuning Validation (Table 4)**:
- LLaVA-v1.5/v1.6 and Mantis show **substantial performance drops** after CMMCoT fine-tuning due to lacking grounding capability.
- LLaVA-OV shows smaller degradation owing to pre-trained grounding data.
- Qwen2-VL, with built-in grounding capability, shows **performance gains** after fine-tuning.
- Conclusion: **CMMCoT requires the base model to possess strong visual grounding capability**.

## Highlights & Insights
- **Extending CoT from single-image to multi-image is a valuable direction**, and the approach is systematic and complete (dataset + training + inference module).
- The design of **excluding entity images from loss computation** is elegant—cropped entity images are injected as reasoning context rather than prediction targets, avoiding the difficulty of visual token reconstruction.
- **RIFREM is a plug-and-play inference-time enhancement** that requires no additional trainable parameters; its design is conceptually analogous to RAG but operates at the visual token level.
- The CMMCoT-260K data construction pipeline (GPT-4o chain generation → IoU verification → spatial fusion) is practical and reusable.

## Limitations & Future Work
- **Limited performance gains**: The multi-image average improvement is only 1.4 points (73.2→74.6); single-image gains are larger (71.4→75.5) but driven primarily by large improvements on SQA and HallBench.
- **Strong dependency on base model grounding capability**: Table 4 demonstrates that CMMCoT fine-tuning is harmful for models without grounding ability, substantially limiting the generalizability of the method.
- **Latency overhead of RIFREM is not fully quantified**: Only relative comparisons are reported; absolute latency figures are absent. The additional computational cost of 8-layer RIFREM may be non-negligible in practical deployment.
- **Dataset quality depends on multiple closed-source APIs** (GPT-4o, Qwen-VL-max), making reproduction costly.
- **MMMU performance decreases** (58.6→57.5), indicating negative transfer in certain single-image understanding scenarios.
- Direct comparison experiments with concurrent multi-image CoT works (e.g., MVoT) are absent.

## Related Work & Insights
- **VoCoT / MVoT**: The most directly related prior work. VoCoT and MVoT primarily leverage visual CoT to enhance single-image reasoning; CMMCoT explicitly extends this to multi-image scenarios and introduces cross-image memory augmentation. However, VoCoT/MVoT do not require the base model to have grounding capability, making them more broadly applicable.
- **Virgo**: Uses textual reasoning capabilities to guide visual reasoning, but neglects supervision of the visual reasoning process during training. CMMCoT explicitly incorporates entity coordinates and entity images as visual supervision signals within the reasoning chain.
- **LLaVA-CoT / Visual-CoT**: Annotate textual reasoning chains using external tools. CMMCoT includes not only a textual chain but also a visual chain (coordinates + entity images), making it better suited for cross-image reasoning in multi-image scenarios.

**Transferable insights**:
- The strategy of filtering coordinate annotation quality using IoU ≥ 0.9 during dataset construction can be transferred to other datasets requiring grounding annotations.
- The idea of injecting entity images as reasoning context without computing loss over them can be generalized to key-frame reasoning in video understanding.

## Rating
- **Novelty**: ⭐⭐⭐ Extending CoT to multi-image settings is a natural direction; the RIFREM memory retrieval design is noteworthy but not a breakthrough contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Both multi-image and single-image benchmarks are evaluated, with comprehensive ablations (module combinations, RIFREM layer count, different base models); however, direct comparisons with concurrent methods are missing.
- **Writing Quality**: ⭐⭐⭐ Generally clear but somewhat disorganized in structure (tables are interspersed throughout Method and Experiments); the motivation for some design choices could be articulated more intuitively.
- **Value**: ⭐⭐⭐ The research direction is meaningful and the CMMCoT-260K dataset has reuse value, but the limited performance gains and strong dependency on base model grounding capability may restrict practical impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Rationale-Enhanced Decoding for Multi-modal Chain-of-Thought](../../CVPR2026/llm_reasoning/red_rationale_enhanced_decoding_cot.md)
- [\[AAAI 2026\] Relation-R1: Progressively Cognitive Chain-of-Thought Guided Reinforcement Learning for Unified Relation Comprehension](relation-r1_progressively_cognitive_chain-of-thought_guided_reinforcement_learni.md)
- [\[AAAI 2026\] L2V-CoT: Cross-Modal Transfer of Chain-of-Thought Reasoning via Latent Intervention](l2v-cot_cross-modal_transfer_of_chain-of-thought_reasoning_v.md)
- [\[ICLR 2026\] Fine-R1: Make Multi-modal LLMs Excel in Fine-Grained Visual Recognition by Chain-of-Thought Reasoning](../../ICLR2026/llm_reasoning/fine-r1_make_multi-modal_llms_excel_in_fine-grained_visual_recognition_by_chain-.md)
- [\[AAAI 2026\] Evaluating, Synthesizing, and Enhancing for Customer Support Conversation](evaluating_synthesizing_and_enhancing_for_customer_support_conversation.md)

</div>

<!-- RELATED:END -->
