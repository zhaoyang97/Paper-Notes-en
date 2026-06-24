---
title: >-
  [Paper Note] Cross-modal Causal Relation Alignment for Video Question Grounding
description: >-
  [CVPR 2025][Video Understanding][Video question grounding] Eliminates spurious cross-modal associations in Video Question Grounding (VideoQG) via causal intervention. It introduces three modules—Gaussian smoothing grounding, cross-modal alignment, and explicit causal intervention—simultaneously improving grounding (+2.2 Acc@GQA) and question answering (+0.9 Acc@VQA) performance on NextGQA.
tags:
  - "CVPR 2025"
  - "Video Understanding"
  - "Video question grounding"
  - "causal inference"
  - "cross-modal alignment"
  - "Gaussian smoothing"
  - "front-door intervention"
date: 2026-05-08
content_hash: fa3f9831f703421e
---

# Cross-modal Causal Relation Alignment for Video Question Grounding

**Conference**: CVPR 2025  
**arXiv**: [2503.07635](https://arxiv.org/abs/2503.07635)  
**Code**: [https://github.com/WissingChen/CRA-GQA](https://github.com/WissingChen/CRA-GQA)  
**Area**: Video Understanding  
**Keywords**: Video question grounding, causal inference, cross-modal alignment, Gaussian smoothing, front-door intervention

## TL;DR
Eliminates spurious cross-modal associations in Video Question Grounding (VideoQG) via causal intervention. It introduces three modules—Gaussian smoothing grounding, cross-modal alignment, and explicit causal intervention—simultaneously improving grounding (+2.2 Acc@GQA) and question answering (+0.9 Acc@VQA) performance on NextGQA.

## Background & Motivation

**Background**: Video Question Grounding (VideoQG) requires models to simultaneously answer video-related questions and ground the corresponding video segment for the answer. Existing methods suffer from "unfaithfulness"—models might guess the correct answer via language shortcuts (such as keywords in the question) but ground the wrong temporal segment.

**Limitations of Prior Work**: (1) Post-hoc analysis methods (e.g., post-hoc attention analysis) suffer from poor grounding quality. (2) End-to-end methods easily learn spurious correlations, where both language bias (certain question types favoring specific answers) and visual confounding (irrelevant visual information interfering with grounding) coexist.

**Key Challenge**: The model needs to simultaneously understand "what to answer" and "where to find the answer," but their optimization objectives can conflict—shortcuts can assist in answering but destroy grounding.

**Goal**: From the perspective of causal inference, simultaneously eliminate language and visual biases, enabling the model to perform grounding and question answering based on correct causal relations.

**Key Insight**: Explicitly model the causal relations among video, question, answer, and grounding by constructing a causal graph, eliminating spurious correlations via front-door intervention (visual deconfounding) and back-door intervention (language debiasing).

**Core Idea**: Use Gaussian smoothed attention for temporal grounding, bidirectional contrastive learning for cross-modal alignment, and front-door/back-door causal intervention to eliminate visual/language biases.

## Method

### Overall Architecture
Video + Question $\rightarrow$ CLIP/RoBERTa encoding $\rightarrow$ GSG module performs temporal grounding via Gaussian-filtered cross-attention $\rightarrow$ CMA module aligns grounding regions with QA features using bidirectional InfoNCE $\rightarrow$ ECI module applies front-door intervention for visual deconfounding and back-door intervention for language debiasing $\rightarrow$ Outputs answer and temporal segment simultaneously.

### Key Designs

1. **Gaussian Smoothing Grounding (GSG)**:

    - **Function**: Generate smooth temporal attention distributions for grounding
    - **Mechanism**: Vision-language cross-attention $w = G(\text{MLP}(v \cdot l_g^T))$, where $G$ is a learnable Gaussian filter. Gaussian filtering suppresses noise spikes in the attention map, producing continuous temporal segment grounding.
    - **Design Motivation**: Ablation results show that without Gaussian smoothing, Acc@GQA is only 16.4; adding it improves the metric to 18.2 (+1.8), and IoU@0.5 increases from 8.0 to 10.6.

2. **Cross-Modal Alignment (CMA)**:

    - **Function**: Ensure consistency between the grounding region and QA semantics
    - **Mechanism**: Bidirectional InfoNCE contrastive loss—pulling the grounded visual segments closer to correct answers and pushing them away from incorrect ones.
    - **Design Motivation**: Prevent decoupling between grounding and answering—the model might ground visually salient regions that are irrelevant to the answer.

3. **Explicit Causal Intervention (ECI)**:

    - **Function**: Eliminate both sources of spurious correlations
    - **Mechanism**: Front-door intervention—using the grounded video segment as a mediator variable to cut off the direct influence of ungrounded regions on the answer. Back-door intervention—constructing a semantic structure graph (subject, predicate, object parsed by Stanza) and approximating the distribution of confounders using clustering features of the semantic graph for debiasing.
    - **Design Motivation**: CRA reduces bias errors by 1.1% and unfaithful answers by 1.4%.

### Loss & Training
Multi-task loss: QA classification loss + grounding loss + CMA contrastive loss + ECI causal loss. 32-frame video input, CLIP-L is frozen, and RoBERTa is fine-tuned.

## Key Experimental Results

### Main Results

| Method | Acc@GQA | Acc@VQA | mIoP | IoU@0.5 |
|------|---------|---------|------|---------|
| Temp[CLIP] baseline | 16.0 | 60.2 | 25.7 | 8.9 |
| TimeCraft | 18.2 | - | 28.1 | 9.6 |
| **CRA (Temp[CLIP])** | **18.2** | **61.1** | **28.6** | **10.6** |

### Ablation Study

| Module | Acc@GQA | IoU@0.5 |
|------|---------|---------|
| Baseline | 16.0 | 8.9 |
| +GSG | 16.4 (+Gaussian) → 18.2 | 10.6 |
| +CMA | Further improves alignment | - |
| +ECI | Reduces bias errors by 1.1% | - |

### Key Findings
- **Larger Models Perform Worse at Grounding**: FrozenBiLM achieves high VQA accuracy (70.2%) but exhibits poorer grounding quality than smaller models, indicating that larger language models are more prone to learning shortcuts.
- **Template-Generated QA exhibits Larger Biases**: CRA yields larger improvements on the STAR dataset (generated via templates) because templates introduce more systematic spurious correlations.

## Highlights & Insights
- **First Application of a Causal Inference Framework to VideoQG**: The dual front-door and back-door interventions cover both major sources of bias.
- The finding that **"larger models are more unfaithful"** is noteworthy—stronger language modeling capabilities may exacerbate shortcut learning.

## Limitations & Future Work
- The causal graph assumes fixed relationships among variables, whereas real-world scenarios may be more complex.
- The quality of the semantic structure graph relies heavily on the Stanza parser.
- Validation is limited to NextGQA and STAR, and the dataset scales are relatively small.

## Related Work & Insights
- **vs TimeCraft**: TimeCraft also performs temporal grounding but does not consider causal biases. CRA achieves comparable Acc@GQA while delivering better IoU.
- **vs VGT / SeViLA**: These methods have lower grounding accuracy (IoU@0.5 < 9%), whereas CRA achieves 10.6%.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of causal inference and video grounding is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Two datasets, bias analysis, and faithfulness analysis.
- Writing Quality: ⭐⭐⭐⭐ The explanation of the causal graph is clear.
- Value: ⭐⭐⭐⭐ Highly significant for improving the credibility of video question grounding.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Progressive Cross-Modal Causal Intervention for Long-Term Action Recognition](../../CVPR2026/video_understanding/progressive_cross-modal_causal_intervention_for_long-term_action_recognition.md)
- [\[CVPR 2025\] QA-TIGER: Question-Aware Gaussian Experts for Audio-Visual Question Answering](question-aware_gaussian_experts_for_audio-visual_question_answering.md)
- [\[CVPR 2025\] EgoTextVQA: Towards Egocentric Scene-Text Aware Video Question Answering](egotextvqa_towards_egocentric_scene-text_aware_video_question_answering.md)
- [\[CVPR 2025\] Video-Panda: Parameter-efficient Alignment for Encoder-free Video-Language Models](video-panda_parameter-efficient_alignment_for_encoder-free_video-language_models.md)
- [\[CVPR 2025\] Temporal Alignment-Free Video Matching for Few-Shot Action Recognition](temporal_alignment-free_video_matching_for_few-shot_action_recognition.md)

</div>

<!-- RELATED:END -->
