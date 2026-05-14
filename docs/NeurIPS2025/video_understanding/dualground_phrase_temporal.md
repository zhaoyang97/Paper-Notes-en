---
title: >-
  [Paper Note] DualGround: Structured Phrase and Sentence-Level Temporal Grounding
description: >-
  [NeurIPS 2025][Video Understanding][Video Temporal Grounding] This paper identifies that existing video temporal grounding (VTG) models over-rely on the global sentence semantics encoded in the [EOS] token while neglecti…
tags:
  - "NeurIPS 2025"
  - "Video Understanding"
  - "Video Temporal Grounding"
  - "Phrase-Level Semantics"
  - "Dual-Branch Architecture"
  - "Attention Decoupling"
  - "Highlight Detection"
date: 2026-05-08
content_hash: 57716243622a69d7
---

# DualGround: Structured Phrase and Sentence-Level Temporal Grounding

**Conference**: NeurIPS 2025
**arXiv**: [2510.20244](https://arxiv.org/abs/2510.20244)
**Code**: None
**Area**: Video Understanding
**Keywords**: Video Temporal Grounding, Phrase-Level Semantics, Dual-Branch Architecture, Attention Decoupling, Highlight Detection

## TL;DR
This paper identifies that existing video temporal grounding (VTG) models over-rely on the global sentence semantics encoded in the [EOS] token while neglecting word-level signals. It proposes DualGround, a dual-branch architecture that explicitly decouples global and local semantics via a sentence-level path (adaptive cross-attention) and a phrase-level path (recurrent phrase generation + Slot Attention), achieving state-of-the-art performance on QVHighlights and Charades-STA.

## Background & Motivation

1. **Background**: VTG has advanced significantly through pretrained vision-language models (VLMs) such as CLIP and InternVideo2, yet these models treat all text tokens uniformly in cross-modal attention.

2. **Limitations of Prior Work**: Experiments reveal that VTG models are heavily biased toward the global semantics of the [EOS] token—using only [EOS] yields performance comparable to or better than using all tokens, indicating severe underutilization of word-level cues.

3. **Key Challenge**: The [EOS] token serves as a sentence summary computed independently of visual context, and may fail to capture visually salient textual cues (e.g., "red jacket"), leading to insufficient fine-grained grounding.

4. **Goal**: Design an architecture that explicitly leverages word- and phrase-level semantics to balance global and local alignment.

5. **Key Insight**: Decompose text representations into sentence-level ([EOS]) and phrase-level (word clusters) components, and model their alignment with video separately.

6. **Core Idea**: A dual-path architecture in which the sentence path augments [EOS] attention via dummy tokens, while the phrase path clusters words into semantic phrases and computes phrase-segment context representations.

## Method

### Overall Architecture
Given video features $V \in \mathbb{R}^{T \times d}$ and text features comprising [EOS] and word tokens, the model processes them through a sentence-level path and a phrase-level path in parallel, then fuses the outputs for temporal pyramid prediction.

### Key Designs

1. **Sentence-Level Path (ACA + Dummy Tokens)**:
    - Function: Strengthen global alignment between [EOS] and video segments.
    - Mechanism: $L_d$ learnable dummy tokens are introduced as attention absorbers. Semantically irrelevant video segments attend to dummy tokens, while relevant segments attend to [EOS], producing sharper alignment signals $\alpha_i$.
    - Design Motivation: The single [EOS] token is incompatible with standard attention mechanisms; dummy tokens provide the necessary contrastive background.

2. **Phrase-Level Path (RPG + Slot Attention)**:
    - Function: Construct semantically coherent phrase representations from word tokens.
    - Mechanism: The Recurrent Phrase Generator (RPG) uses [EOS] and the previous phrase as guidance vectors to perform soft attention aggregation over word tokens. Slot Attention iteratively refines the phrases to disentangle overlapping semantics. Phrase-segment context embeddings are then computed as $C = f_{ctx}(f_p(P) \odot f_v(V)) \in \mathbb{R}^{N \times T \times d}$.
    - Design Motivation: Word-level semantics are context-dependent; phrase-level abstraction is better suited for alignment with visual content than individual words.

3. **Phrase-Guided Aggregation**:
    - Function: Aggregate $N$ phrase-segment contexts into a unified representation.
    - Mechanism: Importance weights are computed using the reconstructed global token $P_{[EOS]}$ over all phrases, and contexts are aggregated accordingly: $v_{p,t} = \sum_{n=1}^N \text{softmax}(\langle W_q P_{[EOS]}, W_k p^{(n)} \rangle / \sqrt{d}) \cdot C_{n,t}$.
    - Design Motivation: Different phrases contribute differently to different temporal segments, necessitating adaptive aggregation.

### Loss & Training
- MR loss: Focal loss (classification) + L1 (regression)
- HD loss: Ranking loss + contrastive loss (applied to saliency scores and attention weights respectively)
- Phrase loss: DQA loss (orthogonalization of phrase attention) + EOS reconstruction loss (InfoNCE aligning $P_{[EOS]}$ with $e_{[EOS]}$)
- Total loss: $\mathcal{L} = \lambda_{mr}\mathcal{L}_{mr} + \lambda_{hd}\mathcal{L}_{hd} + \lambda_{phrase}(\mathcal{L}_{DQA} + \mathcal{L}_{EOS})$

## Key Experimental Results

### Main Results

| Dataset | Metric | DualGround | FlashVTG | Gain |
|---------|--------|-----------|----------|------|
| QVHighlights Test (IV2) | R1@0.7 | **56.94%** | 53.96% | +2.98% |
| QVHighlights Test (IV2) | mAP | **52.73%** | 52.00% | +0.73% |
| QVHighlights Val (IV2) | R1@0.7 | **58.97%** | 56.06% | +2.91% |
| Charades-STA (IV2) | R1@0.7 | SOTA | - | - |

### Ablation Study

| Configuration | R1@0.7 | mAP | Note |
|---------------|--------|-----|------|
| FlashVTG baseline | 56.13 | 52.24 | Full token sequence |
| + RPG + Slot + DQA + EOS | **58.97** | **53.26** | Full DualGround |
| − DQA loss | 56.55 | 52.53 | Insufficient phrase disentanglement |
| − EOS reconstruction | 58.02 | 53.11 | Slight drop in global consistency |
| − Slot Attention | 57.83 | 53.02 | Insufficient refinement |

### Key Findings
- Performance gains are more pronounced for long queries (>20 words), confirming the phrase path's effectiveness on complex queries.
- Attention visualizations clearly demonstrate that baseline methods neglect visually salient words such as "red jacket."
- Highlight detection is sensitive to the phrase path—unregularized phrases may introduce noise.
- The DQA loss contributes +2.42 R1@0.7; the EOS reconstruction loss contributes +0.95 R1@0.7; Slot Attention contributes +1.14 R1@0.7.
- The model assumes a fixed phrase count $N$, which may not be optimal for all queries; the dual-path design also incurs additional inference overhead.
- Compared to the FlashVTG baseline (InternVideo2 features): R1@0.7 improves from 56.13% to 58.97% (+2.84), and mAP improves from 52.24% to 53.26% (+1.02).
- Controlled experiments reveal an [EOS] bias in VTG models: using only the [EOS] token performs comparably to or better than using all tokens, demonstrating severe underutilization of word-level cues. This finding directly motivates DualGround's dual-path design.

## Highlights & Insights
- Controlled experiments (words-only / EOS-only / all tokens) provide a clear and rigorous demonstration of the [EOS] bias in VTG models.
- Applying Slot Attention to refine phrase clusters is an elegant design transfer.
- The sentence + phrase dual-path design is generalizable to other visual-language alignment tasks.
- The DQA loss and EOS reconstruction loss provide effective regularization.
- The loss formulation combines Focal loss (classification) + L1 (regression) for MR, and ranking loss + contrastive loss for HD. The total loss is $\mathcal{L} = \lambda_{mr}\mathcal{L}_{mr} + \lambda_{hd}\mathcal{L}_{hd} + \lambda_{phrase}(\mathcal{L}_{DQA} + \mathcal{L}_{EOS})$.

## Limitations & Future Work
- The phrase count $N$ must be specified in advance; the optimal $N$ may vary across queries.
- The dual-path design incurs additional computational overhead at inference time.
- No comparison is made against MLLM-based methods (e.g., TimeZero).
- Validation is limited to MR and HD tasks; extension to broader tasks such as video QA is not explored.

## Related Work & Insights
- **vs. FlashVTG**: FlashVTG employs a multi-scale temporal pyramid but processes text uniformly; DualGround decouples sentence- and phrase-level semantics.
- **vs. CG-DETR**: CG-DETR introduces ACA but applies it only at the sentence level; DualGround extends it to a complete dual-path framework.
- **vs. Keyword-DETR**: Keyword-DETR emphasizes visually salient keywords but does not perform phrase clustering; DualGround achieves semantically coherent phrase abstraction through Slot Attention.

## Rating

### Implementation Details
Built on the FlashVTG framework using InternVideo2/CLIP features. $L_d$ learnable dummy tokens serve as attention absorbers.
- Novelty: ⭐⭐⭐⭐ The sentence/phrase dual-path decoupling is an effective and well-motivated design.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-benchmark comparisons, detailed ablations, and visualization analyses.
- Writing Quality: ⭐⭐⭐⭐ Motivation is rigorously established; controlled experiments are persuasive.
- Value: ⭐⭐⭐⭐ Offers meaningful reference for VTG and visual-language alignment research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Moment Quantization for Video Temporal Grounding](../../ICCV2025/video_understanding/moment_quantization_for_video_temporal_grounding.md)
- [\[NeurIPS 2025\] Structured Sparse Transition Matrices to Enable State Tracking in State-Space Models](structured_sparse_transition_matrices_to_enable_state_tracking_in_state-space_mo.md)
- [\[ICCV 2025\] VTimeCoT: Thinking by Drawing for Video Temporal Grounding and Reasoning](../../ICCV2025/video_understanding/vtimecot_thinking_by_drawing_for_video_temporal_grounding_and_reasoning.md)
- [\[ICCV 2025\] Sparse-Dense Side-Tuner for Efficient Video Temporal Grounding](../../ICCV2025/video_understanding/sparse-dense_side-tuner_for_efficient_video_temporal_grounding.md)
- [\[ICCV 2025\] TimeExpert: An Expert-Guided Video LLM for Video Temporal Grounding](../../ICCV2025/video_understanding/timeexpert_an_expert-guided_video_llm_for_video_temporal_grounding.md)

</div>

<!-- RELATED:END -->
