---
title: >-
  [Paper Note] SEPT: Semantically Expanded Prompt Tuning for Audio-Language Models
description: >-
  [ACL 2026][Audio & Speech][Prompt Tuning] SEPT significantly alleviates the Base-New Tradeoff (BNT) problem in Audio-Language Model (ALM) prompt tuning by utilizing LLMs to generate semantic neighbors and designing a sem…
tags:
  - "ACL 2026"
  - "Audio & Speech"
  - "Prompt Tuning"
  - "Audio-Language Models"
  - "Semantic Expansion"
  - "Base-New Tradeoff"
  - "Generalization"
date: 2026-05-08
content_hash: df80de247f45c6ee
---

# SEPT: Semantically Expanded Prompt Tuning for Audio-Language Models

**Conference**: ACL 2026  
**arXiv**: [2601.20867](https://arxiv.org/abs/2601.20867)  
**Code**: None  
**Area**: Audio and Speech  
**Keywords**: Prompt Tuning, Audio-Language Models, Semantic Expansion, Base-New Tradeoff, Generalization

## TL;DR

SEPT significantly alleviates the Base-New Tradeoff (BNT) problem in Audio-Language Model (ALM) prompt tuning by utilizing LLMs to generate semantic neighbors and designing a semantic expansion loss with margin constraints to regularize the prompt embedding space. It establishes the first systematic evaluation benchmark for ALM prompt generalization.

## Background & Motivation

**Background**: Prompt tuning has achieved significant progress in Vision-Language Models (VLM) and is beginning to expand to Audio-Language Models (ALM, such as CLAP). Methods like CoOp learn continuous prompt vectors to replace manual templates, significantly improving performance on seen categories.

**Limitations of Prior Work**: Prompt tuning in ALMs severely overfits to base (seen) categories, leading to a substantial decline in generalization to new (unseen) categories—the Base-New Tradeoff (BNT). This issue is more severe in ALMs than in VLMs because audio benchmarks typically contain only a few dozen categories (semantic sparsity), and the learned prompts lack sufficient semantic support to maintain geometric cohesion.

**Key Challenge**: Learned prompt embeddings disrupt the semantic structure of the pre-trained text embedding space—the similarity between a category and its semantic neighbors decreases significantly after prompt tuning, rendering the model unable to utilize semantic relationships to generalize to unseen categories.

**Goal**: (1) Establish the first evaluation benchmark for ALM prompt generalization; (2) Design a plug-and-play framework to mitigate BNT.

**Key Insight**: Use LLMs to generate semantic neighbors (synonyms, acoustic variants) for each category, integrate these neighbors into the prompt tuning process, and explicitly regularize the embedding space so that each category forms a compact cluster with its semantic neighbors.

**Core Idea**: Expand the semantic coverage of each category through semantic neighbors and use a loss that pulls positive samples together while pushing negative samples apart to maintain the semantic structure of the embedding space. This maintains generalization to new categories while enhancing base performance.

## Method

### Overall Architecture

SEPT is a plug-and-play framework that can be integrated into any prompt tuning method. The workflow includes: (1) Using an LLM to generate $N$ semantic neighbors for each category; (2) Calculating margin constraints (using the average distance of $T$ manual prompts as a reference); (3) Adding a semantic expansion loss $\mathcal{L}_{se}$ on top of the standard cross-entropy loss during training. It introduces no additional computational overhead during inference.

### Key Designs

1.  **Semantic Neighbor Generation**:
    - **Function**: Expands the semantic coverage of each category.
    - **Mechanism**: Uses an LLM to generate $N$ semantically related terms $\{p_i^1, ..., p_i^N\}$ for each category name $c_i$, capturing fine-grained acoustic variants and natural language expressions. These neighbors generate embeddings via shared learnable prompts and a frozen text encoder.
    - **Design Motivation**: Audio datasets have few categories, leading to a sparse semantic space; neighbors provide additional semantic anchors.

2.  **Semantic Expansion Loss with Margin Constraints**:
    - **Function**: Regularizes the embedding space while maintaining natural semantic distances.
    - **Mechanism**: Two components—(a) Intra-class alignment loss $\mathcal{L}_{\text{intra}}$: applies a pulling force when the distance between a category embedding $\mathbf{z}_i$ and its positive neighbor $\mathbf{p}_i^n$ exceeds a pre-computed margin $m_{i,i,n}$; (b) Inter-class separation loss $\mathcal{L}_{\text{inter}}$: applies a pushing force when the distance between a category embedding and neighbors of other classes is less than a pre-computed margin $m_{i,j,n}$. Margins are calculated via the average L2 distance of $T$ manual prompts, reflecting the original semantic hierarchy.
    - **Design Motivation**: Naive pulling/pushing might over-compress positive samples or over-separate negative samples—margin constraints preserve natural semantic relationships (e.g., "bell" and "chime" should be close, while "explosion" and "birdsong" should be distant).

3.  **Plug-and-play Integration**:
    - **Function**: Seamlessly integrates into various prompt tuning baselines.
    - **Mechanism**: Total loss $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{ce}} + \lambda \cdot \mathcal{L}_{\text{se}}$, where $\lambda$ is a balancing hyperparameter. It is compatible with methods like CoOp, CoCoOp, KgCoOp, and DePT without affecting inference efficiency.
    - **Design Motivation**: Provides a universal regularization mechanism rather than being restricted to a specific method.

### Loss & Training

Standard cross-entropy + semantic expansion loss (intra-class alignment + inter-class separation, both in hinge loss form). Text and audio encoders are frozen; only prompt vectors are optimized.

## Key Experimental Results

### Main Results

**Average of 11 Audio Datasets (Base-to-New Generalization)**

| Method | Base | New | H (Harmonic Mean) |
| :--- | :--- | :--- | :--- |
| CoOp | 65.00 | 34.09 | 42.83 |
| **CoOp + SEPT** | 64.36 | **42.98** | **49.70** |
| CoCoOp | 69.13 | 36.83 | 46.26 |
| **CoCoOp + SEPT** | 68.63 | **42.59** | **50.65** |
| KgCoOp | 37.99 | 37.42 | 36.39 |
| **KgCoOp + SEPT** | **58.92** | **45.28** | **49.79** |

### Ablation Study

| Configuration | Base | New | H | Description |
| :--- | :--- | :--- | :--- | :--- |
| CoOp + SEPT (Full) | 64.36 | 42.98 | 49.70 | Optimal |
| Only $\mathcal{L}_{\text{intra}}$ | — | — | Decrease | Lacks inter-class separation |
| Only $\mathcal{L}_{\text{inter}}$ | — | — | Decrease | Lacks intra-class compactness |
| No Margin Constraint | — | — | Decrease | Over-compression/separation |

### Key Findings

- SEPT achieves the most significant Gain on New categories (CoOp: 34.09→42.98, +8.89%), while Base performance only slightly decreases (65.00→64.36).
- KgCoOp benefits the most (H: 36.39→49.79, +13.4%), indicating SEPT is complementary to existing regularization methods.
- SEPT is the first work to systematically evaluate base-to-new generalization and cross-dataset transfer in ALMs.
- Margin constraints are crucial to prevent over-compression of positive samples—performance drops significantly without them.

## Highlights & Insights

- The analysis that BNT is more severe in ALMs than in VLMs due to "semantic sparsity" is clear and persuasive, providing direct guidance for the solution.
- The design of margin constraints is ingenious—using distances between manual prompts as a "natural distance" reference is both simple and effective.
- The plug-and-play design allows for direct enhancement of multiple existing methods, offering strong practicality.

## Limitations & Future Work

- The quality of semantic neighbors depends on the LLM; domain-specific knowledge might be required for professional fields (e.g., medical audio).
- Validated only on audio classification; tasks such as audio retrieval and audio captioning were not covered.
- Margin calculation requires $T$ manual prompts, adding a preprocessing step.
- The potential applicability to vision-language models has not yet been explored.

## Related Work & Insights

- **vs CoOp/CoCoOp**: SEPT provides orthogonal regularization and can be used as a direct add-on.
- **vs KgCoOp**: KgCoOp regularizes towards manual prompts using Euclidean distance, while SEPT regularizes towards semantic structures via semantic neighbors; the approaches are different but complementary.

## Rating

- Novelty: ⭐⭐⭐⭐ The idea of semantic expansion has similar counterparts in VLMs but is systematically applied and evaluated in ALMs for the first time.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 11 datasets, four baseline methods, complete ablations, base-to-new + cross-dataset evaluations.
- Writing Quality: ⭐⭐⭐⭐ Motivation and methodology are clearly articulated.
- Value: ⭐⭐⭐⭐ Establishes a benchmark for ALM prompt generalization and provides an effective solution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Temporal Contrastive Decoding: A Training-Free Method for Large Audio-Language Models](temporal_contrastive_decoding_a_training-free_method_for_large_audio-language_mo.md)
- [\[ACL 2026\] SpeakerSleuth: Can Large Audio-Language Models Judge Speaker Consistency across Multi-turn Dialogues?](speakersleuth_can_large_audio-language_models_judge_speaker_consistency_across_m.md)
- [\[AAAI 2026\] Listening Between the Frames: Bridging Temporal Gaps in Large Audio-Language Models](../../AAAI2026/audio_speech/listening_between_the_frames_bridging_temporal_gaps_in_large_audio-language_mode.md)
- [\[ACL 2026\] Closing the Modality Reasoning Gap for Speech Large Language Models](closing_the_modality_reasoning_gap_for_speech_large_language_models.md)
- [\[ACL 2026\] Mind the Pause: Disfluency-Aware Objective Tuning for Multilingual Speech Correction with LLMs](mind_the_pause_disfluency-aware_objective_tuning_for_multilingual_speech_correct.md)

</div>

<!-- RELATED:END -->
