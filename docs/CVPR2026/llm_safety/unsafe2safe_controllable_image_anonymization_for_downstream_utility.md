---
title: >-
  [Paper Note] Unsafe2Safe: Controllable Image Anonymization for Downstream Utility
description: >-
  [CVPR 2026][LLM Safety][Image Anonymization] This paper proposes Unsafe2Safe, a fully automatic privacy-preserving pipeline that realizes controllable image anonymization through a four-stage approach—VLM privacy inspection → dual captioning (private/public) → LLM editing instructions → text-guided diffusion editing. The method achieves substantial improvements on the VLMScore privacy metric while surpassing the original images in downstream accuracy on Caltech-101 classification and OK-VQA.
tags:
  - CVPR 2026
  - LLM Safety
  - Image Anonymization
  - Privacy Protection
  - Diffusion Editing
  - VLM Inspection
  - Downstream Task Preservation
date: 2026-05-08
content_hash: 0f7f20d724da6181
---

# Unsafe2Safe: Controllable Image Anonymization for Downstream Utility

**Conference**: CVPR 2026
**arXiv**: [2603.28605](https://arxiv.org/abs/2603.28605)
**Code**: [https://see-ai-lab.github.io/unsafe2safe/](https://see-ai-lab.github.io/unsafe2safe/)
**Area**: AI Safety
**Keywords**: Image Anonymization, Privacy Protection, Diffusion Editing, VLM Inspection, Downstream Task Preservation

## TL;DR

This paper proposes Unsafe2Safe, a fully automatic privacy-preserving pipeline that realizes controllable image anonymization through a four-stage approach—VLM privacy inspection → dual captioning (private/public) → LLM editing instructions → text-guided diffusion editing. The method achieves substantial improvements on the VLMScore privacy metric while surpassing the original images in downstream accuracy on Caltech-101 classification and OK-VQA.

## Background & Motivation

1. **Background**: As large-scale visual datasets (e.g., LAION) become widely used, personal privacy concerns in images—faces, license plates, health information, etc.—have attracted increasing attention. Existing anonymization methods focus primarily on facial anonymization (e.g., DeepPrivacy2, blurring/mosaicking), covering only a narrow range of privacy elements.
2. **Limitations of Prior Work**: (1) Traditional facial anonymization addresses only faces, ignoring license plates, health identifiers, personal opinions, and other privacy elements; (2) anonymized images often compromise scene semantic integrity, causing significant performance degradation on downstream tasks (classification, VQA); (3) anonymization may introduce new demographic biases (e.g., consistently replacing faces with those of a single ethnicity).
3. **Key Challenge**: Effective anonymization requires substantial modification of privacy regions, yet such modifications disrupt the semantic information needed for downstream tasks—a fundamental tension between privacy and utility.
4. **Goal**: Design a fully automatic and controllable anonymization pipeline that maximizes privacy protection while minimizing downstream task performance loss and balancing demographic distribution.
5. **Key Insight**: Leveraging the multimodal understanding capabilities of VLMs for privacy inspection and scene description, LLMs for generating plausible replacement instructions, and diffusion editors for semantics-preserving local modification.
6. **Core Idea**: Four sequential stages—VLM inspection → dual captioning → LLM instructions → diffusion editing—each addressing a specific sub-problem. A Safe Cross-Attention module performs dual-condition attention to simultaneously preserve semantics and execute edits.

## Method

### Overall Architecture

Input image → Stage 1: InternVL2.5 inspects privacy risk (binary labeling) → generate private caption $c^{\text{priv}}$ and public caption $c^{\text{pub}}$ for unsafe images → Qwen3-4B analyzes the public caption to produce pseudo-private attributes and editing instruction $c^{\text{edit}}$ → Stage 2: diffusion editor (FlowEdit/InstructPix2Pix) executes the edit → Safe Cross-Attention balances semantic preservation and privacy editing → output anonymized image.

### Key Designs

1. **VLM Privacy Inspection and Dual Captioning**

   - **Function**: Automatically identifies privacy risks and generates two versions of scene descriptions—one preserving and one removing privacy information.
   - **Mechanism**: InternVL2.5 inspects each image against predefined privacy criteria (faces, health identifiers, vehicles, personal opinions, sensitive documents), achieving a recall of 97.5% (a deliberately high Type I error rate to minimize privacy leakage). For unsafe images, the model generates $c^{\text{priv}}$ containing privacy details and $c^{\text{pub}}$ with privacy removed.
   - **Design Motivation**: The two captions serve as modality-aligned privacy-safe representations—$c^{\text{pub}}$ retains scene semantics without privacy content, while $c^{\text{edit}}$ guides modifications to privacy regions.

2. **LLM Editing Instruction Generation**

   - **Function**: Generates plausible replacement attributes and editing instructions based on the public caption.
   - **Mechanism**: Qwen3-4B-Instruct analyzes $c^{\text{pub}}$, generates pseudo-private attributes (e.g., replacing a specific face with "a middle-aged male"), and produces structured editing prompts $c^{\text{edit}}$. The final editing condition concatenates $c^{\text{edit}}$ and $c^{\text{pub}}$ as the textual prior for the diffusion editor.
   - **Design Motivation**: Delegating replacement strategy to the LLM rather than humans enables full automation; the LLM also generates diverse replacement attributes, mitigating demographic bias.

3. **Safe Cross-Attention Module**

   - **Function**: Prevents the diffusion editor from over-modifying non-private regions.
   - **Mechanism**: Embeddings of $c^{\text{pub}}$ and $c^{\text{edit}}$ are concatenated into a unified token sequence, upon which dual-condition cross-attention is performed during denoising. $c^{\text{pub}}$ provides a semantic preservation signal while $c^{\text{edit}}$ provides the target transformation signal; the two act jointly within the attention layers.
   - **Design Motivation**: Standard diffusion editors conditioned on a single instruction tend to over-edit or under-edit. Dual-condition attention allows the model to simultaneously "know what not to change" and "know what to change."

### Loss & Training

The core pipeline requires no training. Optional fine-tuning: InstructPix2Pix is fine-tuned on MS-COCO using automatically generated triplets (private caption, public caption, editing instruction). Fine-tuning employs self-attention replacement (probability 0.4) to construct training pairs.

## Key Experimental Results

### Main Results

| Method | Caltech-101 Acc | VLMScore↑ | FaceSim↓ | TextSim↓ | Race Entropy↑ |
|---|---|---|---|---|---|
| Original Images | 94.28 | 7.70 | 1.000 | 1.000 | 0.438 |
| DeepPrivacy2 | 94.60 | 11.05 | 0.392 | 0.957 | 0.732 |
| FaceAnon | 94.85 | 8.76 | 0.459 | 0.936 | 0.609 |
| **U2S (FlowEdit)** | 94.79 | **13.97** | **0.366** | **0.524** | **0.765** |
| **U2S (LLM)** | 92.88 | 12.70 | 0.343 | 0.488 | **0.875** |

### Ablation Study

| Component | Caltech-101 Acc | FaceSim↓ | Race Entropy↑ | Notes |
|---|---|---|---|---|
| Non-finetuned (edit) | 94.32 | 0.516 | 0.683 | Baseline |
| Finetuned (edit) | **95.12** | 0.591 | 0.800 | Fine-tuning improves quality |
| Finetuned + SafeAttn | 94.89 | 0.547 | **0.831** | Safe attention improves diversity |

### Key Findings

- **VQA accuracy increases after anonymization**: U2S (FlowEdit) achieves a VQA accuracy of 0.709 vs. 0.606 for original images (+10.3%), possibly because anonymization removes distracting privacy-related information.
- Demographic balance is substantially improved: the proportion of white individuals decreases from 80.28% to 37.90% (LLM variant), and Race Entropy rises from 0.438 to 0.875.
- U2S provides more comprehensive privacy protection than facial anonymization (TextSim reduced from 0.957 to 0.488), covering faces, text, vehicles, and other privacy elements.
- The high recall (97.5%) of the VLM privacy inspection ensures minimal privacy leakage.

## Highlights & Insights

- **Modular four-stage pipeline design**: Each stage can be replaced independently (e.g., with a stronger VLM or a newer diffusion editor), making the system upgradable with minimal effort.
- **Counter-intuitive VQA accuracy gain**: Anonymization may indirectly benefit downstream tasks by eliminating privacy-related visual noise—suggesting that privacy information in current datasets constitutes a form of interference.
- **Demographic balance as a by-product**: LLM-generated diverse replacement attributes naturally yield demographic balance without requiring additional fairness constraints.
- **Generalizability of Safe Cross-Attention**: The dual-condition attention mechanism is reusable in other editing tasks requiring a "preserve + modify" balance, such as local style transfer.

## Limitations & Future Work

- Unsafe2Safe is a dataset construction tool, not a privacy decision-maker—the responsibility for defining "what constitutes private information" rests with the user.
- The pipeline depends on the quality of the underlying VLM/LLM; model hallucinations may cause misjudgments (missed detections or over-detection).
- Scene classification accuracy on MIT Indoor67 decreases (80.75 vs. 83.88), indicating that global modifications negatively affect scene understanding.
- Diffusion editor artifacts may be visible at region boundaries.
- Scalability of privacy definitions—how to automatically adapt to privacy standards across different countries and cultures remains an open question.

## Related Work & Insights

- **vs. DeepPrivacy2**: Performs facial anonymization only, without addressing license plates, text, etc. U2S provides comprehensive coverage of multiple privacy elements while achieving comparable classification performance.
- **vs. FaceAnon**: A similar face-level method; its FaceSim of 0.459 is substantially worse than U2S's 0.366, demonstrating that U2S achieves more thorough anonymization.
- **vs. Traditional Blurring/Mosaicking**: These methods completely destroy semantic information, rendering downstream tasks unusable. U2S preserves semantic integrity through diffusion editing.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The four-stage pipeline design is novel; Safe Cross-Attention is a notable contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Comprehensive evaluation across five dimensions: classification, captioning, VQA, privacy, and demographics.
- **Writing Quality**: ⭐⭐⭐⭐ — Pipeline description is clear; evaluation framework is rigorously designed.
- **Value**: ⭐⭐⭐⭐⭐ — Data privacy is a core pain point in industry; a fully automatic anonymization tool has direct application value.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] PinPoint: Evaluation of Composed Image Retrieval with Explicit Negatives, Multi-Image Queries, and Paraphrase Testing](pinpoint_evaluation_of_composed_image_retrieval_with_explicit_negatives_multi-im.md)
- [\[CVPR 2026\] V-Attack: Targeting Disentangled Value Features for Controllable Adversarial Attacks on LVLMs](v-attack_targeting_disentangled_value_features_for_controllable_adversarial_atta.md)
- [\[NeurIPS 2025\] InvisibleInk: High-Utility and Low-Cost Text Generation with Differential Privacy](../../NeurIPS2025/llm_safety/invisibleink_high-utility_and_low-cost_text_generation_with_differential_privacy.md)
- [\[AAAI 2026\] LAMP: Learning Universal Adversarial Perturbations for Multi-Image Tasks via Pre-trained Models](../../AAAI2026/llm_safety/lamp_learning_universal_adversarial_perturbations_for_multi-image_tasks_via_pre-.md)
- [\[ICCV 2025\] Oasis: One Image is All You Need for Multimodal Instruction Data Synthesis](../../ICCV2025/llm_safety/oasis_one_image_is_all_you_need_for_multimodal_instruction_data_synthesis.md)

<!-- RELATED:END -->
