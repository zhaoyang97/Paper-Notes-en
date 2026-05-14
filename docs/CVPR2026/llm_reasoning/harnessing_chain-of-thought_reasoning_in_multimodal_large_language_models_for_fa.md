---
title: >-
  [Paper Note] Harnessing Chain-of-Thought Reasoning in Multimodal Large Language Models for Face Anti-Spoofing
description: >-
  [CVPR 2026][LLM Reasoning][Face Anti-Spoofing] The paper introduces FaceCoT, the first CoT-VQA dataset for face anti-spoofing (FAS) with 1.08 million samples covering 14 attack types…
tags:
  - "CVPR 2026"
  - "LLM Reasoning"
  - "Face Anti-Spoofing"
  - "Chain-of-Thought Reasoning"
  - "Multimodal Large Language Models"
  - "VQA Dataset"
  - "Progressive Learning"
date: 2026-05-08
content_hash: 7607f3e94b69e4bc
---

# Harnessing Chain-of-Thought Reasoning in Multimodal Large Language Models for Face Anti-Spoofing

**Conference**: CVPR 2026
**arXiv**: [2506.01783](https://arxiv.org/abs/2506.01783)
**Code**: None
**Area**: LLM Reasoning
**Keywords**: Face Anti-Spoofing, Chain-of-Thought Reasoning, Multimodal Large Language Models, VQA Dataset, Progressive Learning

## TL;DR
The paper introduces FaceCoT, the first CoT-VQA dataset for face anti-spoofing (FAS) with 1.08 million samples covering 14 attack types, and proposes a two-stage progressive learning strategy CEPL, achieving an average AUC improvement of 4.06% and HTER reduction of 5.00% across 11 FAS benchmarks.

## Background & Motivation

**Background**: Face anti-spoofing (FAS) requires detecting presentation attacks such as print attacks, screen replay, and 3D masks. Existing methods primarily rely on single-modality CNN/ViT classifiers, which exhibit limited generalization across devices, environments, and attack types, and lack interpretability.

**Limitations of Prior Work**: While multimodal large language models (MLLMs) have achieved breakthroughs in vision-language understanding and semantic reasoning, the FAS domain lacks high-quality visual-language multimodal datasets — publicly available FAS datasets contain only images/videos with binary labels, without structured language annotations.

**Key Challenge**: Directly training MLLMs on limited-label data leads to overfitting and prevents the generation of interpretable reasoning chains. Furthermore, end-to-end joint training of CoT reasoning and classification causes task interference — the classification objective converges quickly, leaving the reasoning objective under-optimized.

**Goal**: (1) Construct a large-scale CoT-VQA dataset for the FAS domain; (2) Design a training strategy that fully leverages CoT data.

**Key Insight**: Simulating the human "global-to-local" hierarchical reasoning process, the paper designs a 6-level CoT annotation format, and constructs the dataset through a three-stage pipeline: GPT-4o annotation, human verification, and RL-fine-tuned caption model.

**Core Idea**: Structured CoT annotations are used to enhance the visual encoder of an MLLM for fine-grained facial feature perception, followed by joint training of reasoning and classification to achieve FAS.

## Method

### Overall Architecture
The method comprises two components: (1) FaceCoT dataset construction (6-level CoT format, GPT-4o annotation, RL-enhanced caption model for scaling); and (2) CoT-Enhanced Progressive Learning (CEPL), a two-stage training strategy. The input is a face image, and the output is a binary liveness prediction along with CoT reasoning text.

### Key Designs

1. **FaceCoT Dataset Construction**:

    - **Function**: Construct a FAS CoT-VQA dataset with 1.08 million samples.
    - **Mechanism**:
        - **6-Level CoT Format**: Caption (global scene) → Facial Description (facial region) → Facial Attributes (fine-grained facial details) → Reasoning (integrated analysis) → Spoofing Description (artifact characterization) → Conclusion.
        - **FaceCoT-Gold100K**: 100K samples are balanced-sampled from CelebA-Spoof and WFAS, annotated with GPT-4o, achieving approximately 98.97K/100K correct annotations; 581 hard cases are manually corrected by professional annotators.
        - **FaceCoT-Silver982K**: A FAS caption model is trained on Gold100K and optimized via RL (VRFT) — with an accuracy reward (1 point if the Conclusion matches the GT) and a format reward (adherence to the template), improving annotation accuracy from 88% (SFT) to 99.6%. This model is then used to annotate the remaining data, yielding 982K additional samples.
    - **Design Motivation**: The hierarchical CoT mimics human cognitive processes, progressing from global scene understanding to local detail and finally to logical judgment. RL optimization addresses semantic and formatting errors of the SFT caption model on out-of-distribution data.

2. **CEPL — Stage 1: Visual Enhancement Pre-training**:

    - **Function**: Enhance the visual encoder's fine-grained feature representations using CoT data.
    - **Mechanism**: Full-parameter SFT is applied to the MLLM, with face images as input and CoT reasoning text as the supervision signal. This compels the visual encoder to extract fine-grained facial features that are precisely aligned with language descriptions (texture, lighting, edge artifacts, etc.).
    - **Design Motivation**: In end-to-end joint training of CoT and classification, the rapidly converging classification loss "crowds out" the optimization of CoT reasoning, preventing the visual encoder from fully leveraging the fine-grained visual cues embedded in the CoT annotations.

3. **CEPL — Stage 2: Multi-task Joint Training**:

    - **Function**: Jointly optimize CoT reasoning and binary classification.
    - **Mechanism**: The visual encoder weights from Stage 1 are retained, while the connector and LLM decoder are re-initialized to pre-trained weights; the decoder is then fine-tuned with LoRA. Joint training is performed on CoT-annotated data and binary-label data with a multi-task loss.
    - **Design Motivation**: Re-initializing the decoder prevents the CoT-only training bias from Stage 1 from degrading classification performance. LoRA efficiently fine-tunes while preserving pre-trained knowledge. Inheriting the visual encoder ensures that fine-grained feature representations are not lost.

### Loss & Training
**Stage 1**: Standard autoregressive language modeling loss on CoT text; full-parameter SFT.
**Stage 2**: Multi-task loss = CoT generation loss + binary classification loss; LoRA fine-tuning of the LLM decoder and connector.
**RL Phase**: Combined accuracy reward (0/1) and format reward.

## Key Experimental Results

### Main Results

**Average metrics across 11 FAS benchmark datasets**

| Method | Avg HTER (%)↓ | Avg AUC (%)↑ |
|--------|--------------|-------------|
| ViTAF | 23.85 | 82.82 |
| ViT-B | 23.48 | 82.98 |
| ViT-L | ~20 | ~85 |
| FLIP | ~18 | ~87 |
| I-FAS | ~13 | ~92 |
| **Ours-All** | **~8** | **~96** |

**Key per-dataset results**

| Dataset | Ours-All HTER | Ours-All AUC | I-FAS HTER | I-FAS AUC |
|---------|--------------|-------------|-----------|----------|
| CASIA-MFSD | 0.00 | 100.00 | 1.11 | 99.88 |
| 3DMask | 0.40 | 99.98 | 6.18 | 98.40 |
| OULU-NPU | 5.86 | 97.72 | 14.86 | 91.68 |
| HiFiMask | 15.93 | 91.30 | 28.23 | 77.17 |

### Ablation Study

| Configuration | Performance | Notes |
|---------------|-------------|-------|
| Ours-CelebA | Baseline | Caption model annotations using CelebA-Spoof only |
| Ours-100K | Slightly better | FaceCoT-Gold100K (GPT-4o + human verification) |
| Ours-All | Best | Gold100K + Silver982K full dataset |
| w/o CEPL (end-to-end) | Worse | Validates the necessity of two-stage training |
| SFT-only caption model | 88% accuracy | RL significantly improves this to 99.6% |

### Key Findings
- Data scale is critical: scaling from 100K to 1.08M samples (Silver data) yields substantial gains on challenging datasets such as HiFiMask.
- The CEPL two-stage strategy substantially outperforms end-to-end training — decoupled optimization avoids task interference between CoT reasoning and classification.
- RL fine-tuning improves the caption model's annotation accuracy from 88% to 99.6%, resolving cross-domain semantic and formatting errors.
- 100% AUC is achieved on multiple datasets (e.g., CASIA-MFSD), demonstrating the effectiveness of CoT-guided fine-grained feature learning.
- 3D mask attacks (HiFiMask) remain the most challenging, yet the proposed method nearly halves the HTER (28.23 → 15.93).

## Highlights & Insights
- **The 6-level CoT annotation format is elegantly designed**: Caption → Facial Description → Facial Attributes → Reasoning → Spoofing Description → Conclusion proceeds from global to local and then to logical judgment, faithfully replicating the cognitive workflow of a human expert. This hierarchical design is not limited to FAS and can be transferred to any security detection task requiring multi-granularity visual reasoning.
- **RL-enhanced caption model enables low-cost data scaling**: Training a caption model on 100K high-quality annotations followed by RL fine-tuning enables automatic annotation of nearly 1 million samples at 99.6% accuracy, substantially reducing annotation costs. This "small gold set + RL expansion" pipeline is reusable.
- **Two-stage training decouples reasoning from classification**: Stage 1 focuses on reinforcing the visual encoder using CoT text supervision; Stage 2 re-initializes the decoder before joint training. The key insight is that the fast-converging classification loss can "starve" the optimization budget available to the reasoning loss.

## Limitations & Future Work
- Dataset construction is heavily dependent on GPT-4o; the quality of initial annotations is coupled to GPT-4o's capabilities. Replacing it with open-source models of comparable quality could reduce costs.
- At 1.08 million samples, the dataset is still not particularly large for MLLM training; more aggressive data scaling warrants exploration.
- The method is validated only on static images and does not incorporate video-level temporal cues (e.g., flickering, 3D motion).
- Full-parameter SFT in Stage 1 incurs considerable computational overhead; it remains to be explored whether LoRA could achieve comparable results.
- CoT reasoning text as an intermediate output increases inference latency, and whether it can be omitted at deployment time requires consideration.

## Related Work & Insights
- **vs. I-FAS**: I-FAS is a prior MLLM-based FAS method that does not incorporate CoT reasoning. The proposed method performs comparably to I-FAS on Rose-Youtu (where I-FAS excels) but significantly outperforms it in cross-domain scenarios.
- **vs. LLaVA-CoT**: LLaVA-CoT employs a 4-level "summary-caption-reasoning-conclusion" structure for general reasoning. The present work designs a more fine-grained 6-level structure tailored to FAS, introducing two face-specific levels: Facial Description and Facial Attributes.
- **vs. VRFT (RL strategy)**: The verifiable RL strategy from VRFT is adopted and applied to caption model training, with FAS-specific accuracy and format rewards designed accordingly.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — First FAS CoT-VQA dataset and progressive training strategy; however, the individual technical components (CoT, CEPL) are not entirely novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 11 benchmark datasets, multiple training configuration comparisons, and comprehensive ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ — The data construction pipeline is clearly described, though the methodological contribution is relatively straightforward.
- **Value**: ⭐⭐⭐⭐ — The dataset contribution is significant; the method is effective but domain-specific to FAS.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] CoRVid: Improving Multimodal Large Language Models Towards Chain-of-Thought Reasoning](../../ICCV2025/llm_reasoning/corvid_improving_multimodal_large_language_models_towards_chain-of-thought_reaso.md)
- [\[CVPR 2026\] Understanding and Mitigating Hallucinations in Multimodal Chain-of-Thought Models](understanding_and_mitigating_hallucinations_in_multimodal_chain-of-thought_model.md)
- [\[CVPR 2026\] Rationale-Enhanced Decoding for Multi-modal Chain-of-Thought](rationale-enhanced_decoding_for_multi-modal_chain-of-thought.md)
- [\[NeurIPS 2025\] ThinkSound: Chain-of-Thought Reasoning in Multimodal Large Language Models for Audio Generation and Editing](../../NeurIPS2025/llm_reasoning/thinksound_chain-of-thought_reasoning_in_multimodal_large_language_models_for_au.md)
- [\[CVPR 2026\] Understanding the Role of Hallucination in Reinforcement Post-Training of Multimodal Reasoning Models](understanding_the_role_of_hallucination_in_reinforcement_post-training_of_multim.md)

</div>

<!-- RELATED:END -->
