---
title: >-
  [Paper Note] VILA-M3: Enhancing Vision-Language Models with Medical Expert Knowledge
description: >-
  [CVPR 2025][Multimodal VLM][Medical VLM] Proposes the VILA-M3 framework, which integrates knowledge from medical domain expert models (segmentation/classification) into a generalist VLM on-demand via a four-stage training scheme. It achieves an average of ~9% SOTA improvement across multiple medical benchmarks such as VQA, report generation, and classification, with a model scale significantly smaller than Med-Gemini (3B-40B vs 1.5T).
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "Medical VLM"
  - "Expert Model Integration"
  - "Instruction Fine-Tuning"
  - "Multi-Task Medical AI"
  - "Domain Knowledge Fusion"
date: 2026-05-08
content_hash: 15c755effc350fc0
---

# VILA-M3: Enhancing Vision-Language Models with Medical Expert Knowledge

**Conference**: CVPR 2025  
**arXiv**: [2411.12915](https://arxiv.org/abs/2411.12915)  
**Code**: [https://github.com/Project-MONAI/VLM](https://github.com/Project-MONAI/VLM)  
**Area**: Multimodal VLM  
**Keywords**: Medical VLM, Expert Model Integration, Instruction Fine-Tuning, Multi-Task Medical AI, Domain Knowledge Fusion

## TL;DR

Proposes the VILA-M3 framework, which integrates knowledge from medical domain expert models (segmentation/classification) into a generalist VLM on-demand via a four-stage training scheme. It achieves an average of ~9% SOTA improvement across multiple medical benchmarks such as VQA, report generation, and classification, with a model scale significantly smaller than Med-Gemini (3B-40B vs 1.5T).

## Background & Motivation

Generalist VLMs (e.g., GPT-4o, Gemini) perform poorly in the medical domain primarily due to two reasons: (1) they rely on internet-memorized knowledge rather than medical fine-grained features, failing to accurately identify fine-grained visual details such as tumors; (2) existing medical datasets are mostly static and designed for narrow AI tasks (classification/segmentation), making them unsuitable for training large-scale VLMs. Meanwhile, there are already many validated or even FDA-approved expert models in the medical field (e.g., tumor segmentation, X-ray classification) that perform exceptionally well on specific tasks. How to leverage the "knowledge" of these expert models to enhance VLMs becomes a key issue.

Existing medical VLMs like Med-Gemini (1.5T parameters), although specially designed, do not utilize expert model information, and their huge parameter size is unfavorable for deployment. The authors argue that on top of the standard three-stage training (visual pre-training, vision-language pre-training, and instruction fine-tuning), a fourth stage is required: specialized instruction fine-tuning that focuses on medical data and integrates expert model information.

## Method

### Overall Architecture

VILA-M3 is based on the VILA architecture and employs an autoregressive multimodal LLM design: images are encoded into visual tokens, aligned with text tokens via a linear projection layer, and then fed into the LLM. The framework introduces a four-stage training process: visual encoder pre-training → VLM pre-training → generalist IFT → Expert-guided IFT. During inference, VILA-M3 can trigger external expert models on-demand and leverage their feedback to improve the output. The visual encoder uses OpenAI CLIP-L (384×384), and the LLM backbones cover various options including Vicuna, Llama-3, and Yi-34B.

### Key Designs

1. **Expert Model Triggering and Feedback Mechanism**:
    - Function: Allows the VLM to actively invoke the appropriate expert model when fine-grained analysis is needed.
    - Mechanism: VILA-M3 learns to predict keywords and parameters (e.g., `<VISTA3D(hepatic tumor)>`) to trigger expert models. The list of available expert models is provided to the model as system prompts (model card). The results returned by the expert model are formatted as conversational user prompts and fed back to the VLM—segmentation results are returned as masks overlaid on the original image + textual description, and classification results are returned as a yes/no list for 18 diseases.
    - Design Motivation: VLMs excel at connecting coarse-grained visual features with language but fail to capture subtle features in medical images (e.g., small tumors). Experiments show that without segmentation assistance, both VLMs and GPT-4o fail to detect brain tumors, but they correctly identify them immediately once expert segmentation results are integrated.

2. **2D/3D Hybrid Information Fusion**:
    - Function: Enables the VLM, which only accepts 2D inputs, to utilize 3D volumetric information.
    - Mechanism: When VILA-M3 processes a 2D input such as a CT slice, it can trigger a 3D segmentation model like VISTA3D (supporting 127 anatomical structures) to segment the complete 3D volume. For brain MRI, the MONAI BraTS model is used to perform multimodal (T1/T1c/T2/FLAIR) tumor subregion segmentation. For chest X-rays, a TorchXRayVision CNN classification model ensemble is used (requires ~1.5GB VRAM).
    - Design Motivation: Clinical tasks typically require 3D spatial information, whereas VLMs are limited by 2D inputs. Bridging this gap via expert models, where VISTA3D requires approximately 12GB VRAM.

3. **Balanced Dataset Strategy and Expert Data Construction**:
    - Function: Prevents the model from forgetting language capabilities and avoids data bias.
    - Mechanism: Frequency-weighted balancing is applied to datasets of different categories such as VQA, report generation, classification, and expert triggering. Expert model inference is run on existing datasets to automatically generate training dialogues for expert triggering. In addition, a small amount of pure medical text is mixed in to prevent LLM performance degradation. Fully unfrozen fine-tuning is performed (visual encoder + projection layer + LLM).
    - Design Motivation: The sizes of the original datasets vary drastically (MIMIC-CXR has ~370k images vs. VQA datasets with a few thousand entries), and unbalanced training leads to bias. Balancing improves average performance by approximately 4%.

### Loss & Training

Training utilizes the standard autoregressive language modeling loss. Key hyperparameters: learning rate $2 \times 10^{-5}$, cosine schedule with a warmup ratio of 0.03, no weight decay, bf16 mixed precision, and gradient checkpointing. Training all models for 2 epochs yields the best results. Computational cost ranges from 32 GPUs × 5.5 hours for the 3B model to 128 GPUs × 21 hours for the 40B model.

## Key Experimental Results

### Main Results

| Dataset | Metric | VILA-M3-40B | Med-Gemini(1.5T) | Task SOTA | Gain (vs Med-Gemini) |
|--------|------|-------------|------------------|----------|---------------------|
| VQA-Rad | Acc | 90.4 | 78.8 | 84.2 | +11.6 |
| MIMIC VQA | Acc | 86.4(13B) | 78.6 | - | +7.8 |
| PathVQA | Acc | 92.7 | 83.3 | 91.7 | +9.4 |
| MIMIC-CXR | BLEU-4 | 21.6 | 20.5 | 15.4 | +1.1 |
| MIMIC-CXR | ROUGE | 32.2 | 28.3 | 30.6 | +3.9 |
| ChestX-ray14 | F1 | 51.3 | 46.7 | 50.0 | +4.6 |
| CheXpert | F1 | 61.6(8B) | 48.3 | 51.5 | +13.3 |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| Without expert models | ChestX-ray14 Avg F1=47.1 | Unable to detect fine-grained abnormalities |
| With expert models | ChestX-ray14 Avg F1=51.3 | Expert feedback improves classification by 4.2% |
| Unbalanced data | Average across all metrics is lower by ~4% | Data bias severely impacts tasks with small datasets |
| Epoch=2 | Optimal performance | Overfitting occurs at Epoch=3 |
| 3B→8B→13B→40B | Gradual improvement | 40B degrades by 23% on the VILA benchmark (Yi backbone) |

### Key Findings

- VILA-M3 outperforms the 1.5T-parameter Med-Gemini on 7 out of 8 metrics, proving that expert knowledge integration is more effective than brute-force scaling of parameters.
- Degradation on generalist VLM benchmarks after expert-guided IFT is controllable (7% degradation for 3B, 4% for 13B), indicating that the training scheme preserves general capabilities.
- GPT-4o achieves an F1 score of only 33.1 on ChestX-ray14 classification, rendering it nearly unusable for medical classification.
- After incorporating expert information into report generation, BLEU-4 increases from 19.7 → 20.2 (3B), and the GREEN score improves to 39.4.

## Highlights & Insights

- Its **on-demand expert invocation** design, similar to the tool-use/chain-of-thought paradigm, allows the VLM to autonomously decide when to seek expert help.
- Challenges the "larger is better" paradigm: 3B parameters + domain expert knowledge > brute-force stacking of 1.5T parameters.
- Open-sources the complete pipeline for data preparation, training, and evaluation, with model weights released on Hugging Face.
- The modular design makes it easier to meet medical regulatory requirements—each expert model can obtain FDA approval independently.

## Limitations & Future Work

- Only 2D image inputs are supported for the VLM, leaving 3D information completely reliant on indirect bridging via expert models.
- The 40B model (Yi backbone) suffers from severe degradation (~23%) on the VILA benchmark, indicating cross-architecture transfer issues.
- The list of expert models must be pre-defined as a model card, lacking the ability to dynamically discover newly available tools.
- Lacks support for more medical modalities, such as whole slide pathological images and ultrasound videos.
- Evaluation metrics for report generation (BLEU/ROUGE) may not fully reflect clinical value; while the GREEN score is more reasonable, its coverage is limited.

## Related Work & Insights

- Compared to BiomedParse, VILA-M3 provides more comprehensive coverage across four types of tasks: VQA, report generation, classification, and segmentation.
- The concept of "expert models as tools" can be generalized to other professional domains (e.g., law, finance).
- Although works like LLaVA-Med and Med-Flamingo are pioneering, they lack the dimension of expert knowledge, a gap that VILA-M3 effectively bridges.
- Future directions include RAG-augmentation and multi-agent expert collaboration frameworks.

## Rating

- Novelty: ⭐⭐⭐⭐ Although the concept of integrating expert models is not entirely new, its systematic implementation in medical VLMs is solid.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers VQA, report generation, classification, and segmentation, offering a comprehensive evaluation with multi-scale models, multiple datasets, and ablations.
- Writing Quality: ⭐⭐⭐⭐ Clearly structured with rich figures/tables and well-articulated motivation.
- Value: ⭐⭐⭐⭐⭐ High actual clinical value and fully open-source, exerting a deep impact on the medical AI community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Towards Understanding How Knowledge Evolves in Large Vision-Language Models](towards_understanding_how_knowledge_evolves_in_large_vision-language_models.md)
- [\[CVPR 2025\] Florence-VL: Enhancing Vision-Language Models with Generative Vision Encoder and Depth-Breadth Fusion](florence-vl_enhancing_vision-language_models_with_generative_vision_encoder_and_.md)
- [\[ACL 2026\] MedLayBench-V: A Large-Scale Benchmark for Expert-Lay Semantic Alignment in Medical Vision Language Models](../../ACL2026/multimodal_vlm/medlaybench-v_a_large-scale_benchmark_for_expert-lay_semantic_alignment_in_medic.md)
- [\[NeurIPS 2025\] ExGra-Med: Extended Context Graph Alignment for Medical Vision-Language Models](../../NeurIPS2025/multimodal_vlm/exgra-med_extended_context_graph_alignment_for_medical_vision-language_models.md)
- [\[ACL 2025\] HSCR: Hierarchical Self-Contrastive Rewarding for Aligning Medical Vision Language Models](../../ACL2025/multimodal_vlm/hscr_hierarchical_self-contrastive_rewarding_for_aligning_medical_vision_languag.md)

</div>

<!-- RELATED:END -->
