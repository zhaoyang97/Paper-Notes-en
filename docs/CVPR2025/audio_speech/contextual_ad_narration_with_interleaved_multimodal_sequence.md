---
title: >-
  [Paper Note] Contextual AD Narration with Interleaved Multimodal Sequence
description: >-
  [CVPR 2025][Audio & Speech][audio description] A unified framework named Uni-AD is proposed. It takes interleaved multimodal sequences (video features + text + character bank + context) as input. By aligning features through a visual mapping network, identifying main characters via a character-refinement module, and enhancing contextual consistency with a contrastive loss, it achieves SOTA performance on MAD-eval-Named.
tags:
  - "CVPR 2025"
  - "Audio & Speech"
  - "audio description"
  - "interleaved multimodal"
  - "character refinement"
  - "contrastive loss"
  - "movie understanding"
date: 2026-05-08
content_hash: 465d5573ae634ab4
---

# Contextual AD Narration with Interleaved Multimodal Sequence

**Conference**: CVPR 2025  
**arXiv**: [2403.12922](https://arxiv.org/abs/2403.12922)  
**Code**: [GitHub](https://github.com/ant-research/UniAD)  
**Area**: Audio-Visual Language  
**Keywords**: audio description, interleaved multimodal, character refinement, contrastive loss, movie understanding

## TL;DR
A unified framework named Uni-AD is proposed. It takes interleaved multimodal sequences (video features + text + character bank + context) as input. By aligning features through a visual mapping network, identifying main characters via a character-refinement module, and enhancing contextual consistency with a contrastive loss, it achieves SOTA performance on MAD-eval-Named.

## Background & Motivation

**Background**: The Audio Description (AD) task aims to generate descriptions of visual elements in movies for visually impaired individuals, which requires understanding of plot context, character identities, and actions.

**Limitations of Prior Work**:
   - Early methods (such as MM-Narrator) only used visual features, failing to refer to characters by specific names.
   - AutoAD-II introduced a character bank but suffered from coarse-grained multimodal alignment.
   - Existing methods are deficient in handling character identities and plot contexts, resulting in incoherent descriptions.

**Key Challenge**: The AD task requires simultaneous processing of video frames, subtitles, character information, and contextual relations. However, the feature spaces of different modalities vary significantly, and simple concatenation cannot achieve fine-grained alignment.

**Key Insight**: Constructing an interleaved multimodal input sequence and leveraging a pre-trained foundation model (LLM + visual encoder) to process all modalities through a unified framework.

**Core Idea**: Fine-grained alignment with a visual mapping network + identifying key characters with character refinement + ensuring contextual coherence using contrastive loss.

## Method

### Overall Architecture
Uni-AD takes interleaved multimodal sequences as input to the LLM:
- Video features (aligned to the text space via a visual mapping network)
- Subtitle text
- Character information from the character bank
- Surrounding AD descriptions (context)

### Key Designs

1. **Visual Mapping Network**

    - **Function**: Maps video features to the text feature space to achieve fine-grained multimodal alignment.
    - **Architecture**: A lightweight module consisting of linear layers and Layer Normalization.
    - **Advantages**: Preserves richer visual details compared to direct concatenation or coarse-grained alignment.
    - **Processing Flow**: Video frames $\rightarrow$ CLIP/Visual Encoder $\rightarrow$ Mapping Network $\rightarrow$ Text space features.

2. **Character-Refinement Module**

    - **Function**: Identifies characters from the character bank who play important roles in the current video clip.
    - **Mechanism**:
        - Matches characters in the current frame using face detection and recognition.
        - Filters main characters based on their appearance frequency and screen position.
        - Embeds the refined character information into the input sequence.
    - **Design Motivation**: The character bank usually contains a large number of characters, and inputting all of them would introduce noise.

3. **Contrastive Loss**

    - **Function**: Enhances the contextual coherence and temporal consistency of the generated AD.
    - **Mechanism**: Pulls the representation of the current AD closer to its context, while pushing away unrelated clips.
    - **Formula**: Standard InfoNCE format, where positive samples are temporally adjacent ADs.
    - Jointly trained with the generation loss.

4. **Contextual Information Integration**

    - **Function**: Uses preceding and succeeding AD descriptions to provide plot coherence.
    - **Implementation**: Appends several preceding AD segments as text tokens into the input sequence.
    - **Effect**: Generates smoother and more plot-consistent descriptions.

### Loss & Training
- Fine-tuned based on a pre-trained LLM.
- Multi-task loss: Language model generation loss + Contrastive loss.
- Trained on the MAD dataset.

## Key Experimental Results

### Main Results (MAD-eval dataset)

| Method | CIDEr↑ | METEOR↑ | ROUGE-L↑ |
|------|--------|---------|----------|
| MM-Narrator | — | — | — |
| AutoAD-II | Baseline | Baseline | Baseline |
| **Uni-AD** | **SOTA** | **SOTA** | **SOTA** |

### MAD-eval-Named (Evaluation with Character Names)

| Method | CIDEr↑ | Character Name Accuracy↑ |
|------|--------|-------------|
| w/o Character-Refinement | Low | Low |
| **Uni-AD (Full)** | **Highest** | **Highest** |

### Ablation Study

| Configuration | CIDEr↑ | METEOR↑ |
|------|--------|---------|
| w/o Visual Mapping Network | Decreased | Decreased |
| w/o Character-Refinement | Significantly decreased | Decreased |
| w/o Contrastive Loss | Slightly decreased | Slightly decreased |
| w/o Context | Decreased | Decreased |
| **Full Uni-AD** | **Highest** | **Highest** |

### Key Findings
- The Character-Refinement module brings the most significant improvement to the Named metrics.
- The contrastive loss improves the temporal consistency of the generated AD.
- Fine-grained alignment from the visual mapping network outperforms direct feature concatenation.
- Contextual information is crucial for generating coherent plot descriptions.

## Highlights & Insights
- A **unified framework** is proposed to process multimodal inputs cleanly and elegantly.
- The Character-Refinement module addresses the core challenge of the AD task—character identification.
- The contrastive loss acts as a lightweight yet effective consistency enhancement mechanism.
- The framework is designed with good scalability.

## Limitations & Future Work
- Face recognition-based character identification is unstable during profile shots or occlusions.
- High sensitivity to the quality of the character bank.
- Limited context window in long video scenarios.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of interleaved multimodality and character refinement is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Complete evaluations across multiple datasets and ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clear and easy-to-understand framework description.
- Value: ⭐⭐⭐⭐ Practical significance for visually impaired assistance and movie understanding.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] UniM: A Unified Any-to-Any Interleaved Multimodal Benchmark](../../CVPR2026/audio_speech/unim_a_unified_any-to-any_interleaved_multimodal_benchmark.md)
- [\[CVPR 2025\] MultiFoley: Video-Guided Foley Sound Generation with Multimodal Controls](video-guided_foley_sound_generation_with_multimodal_controls.md)
- [\[CVPR 2025\] HOP: Heterogeneous Topology-based Multimodal Entanglement for Co-Speech Gesture Generation](hop_heterogeneous_topology-based_multimodal_entanglement_for_co-speech_gesture_g.md)
- [\[ACL 2025\] Contextual Biasing with the Knowledgeable External Language Model for End-to-End Speech Recognition](../../ACL2025/audio_speech/contextual_biasing_with_the_knowledgeable_external_language_model_for_end-to-end.md)
- [\[CVPR 2025\] DistinctAD: Distinctive Audio Description Generation in Contexts](distinctad_distinctive_audio_description_generation_in_contexts.md)

</div>

<!-- RELATED:END -->
