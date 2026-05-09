---
title: >-
  [Paper Note] SEPT: Semantically Expanded Prompt Tuning for Audio-Language Models
description: >-
  [ACL 2026][Audio & Speech][Prompt Tuning] SEPT leverages LLMs to generate semantic neighbors for each category and introduces a margin-constrained semantic expansion loss to regularize the prompt embedding space, substantially alleviating the Base-New Tradeoff (BNT) in prompt tuning for audio-language models (ALMs). It also establishes the first systematic evaluation benchmark for prompt generalization in ALMs.
tags:
  - ACL 2026
  - "Audio & Speech"
  - Prompt Tuning
  - Audio-Language Models
  - Semantic Expansion
  - Base-New Tradeoff
  - Generalization
date: 2026-05-08
content_hash: c26d231263875c02
---

# SEPT: Semantically Expanded Prompt Tuning for Audio-Language Models

**Conference**: ACL 2026
**arXiv**: [2601.20867](https://arxiv.org/abs/2601.20867)
**Code**: None
**Area**: Audio & Speech
**Keywords**: Prompt Tuning, Audio-Language Models, Semantic Expansion, Base-New Tradeoff, Generalization

## TL;DR

SEPT leverages LLMs to generate semantic neighbors for each category and introduces a margin-constrained semantic expansion loss to regularize the prompt embedding space, substantially alleviating the Base-New Tradeoff (BNT) in prompt tuning for audio-language models (ALMs). It also establishes the first systematic evaluation benchmark for prompt generalization in ALMs.

## Background & Motivation

**Background**: Prompt tuning has achieved remarkable progress in vision-language models (VLMs) and has begun to be extended to audio-language models (ALMs, e.g., CLAP). Methods such as CoOp replace hand-crafted templates with learned continuous prompt vectors, yielding significant performance gains on seen categories.

**Limitations of Prior Work**: Prompt tuning in ALMs suffers from severe overfitting to base (seen) categories, leading to a substantial drop in generalization to new (unseen) categories—the Base-New Tradeoff (BNT). This issue is more pronounced in ALMs than in VLMs, as audio benchmarks typically contain only a few dozen categories (semantic sparsity), leaving learned prompts without sufficient semantic support to maintain geometric cohesion.

**Key Challenge**: Learned prompt embeddings disrupt the semantic structure of the pre-trained text embedding space—the similarity between a category and its semantic neighbors is markedly weakened after prompt tuning, preventing the model from leveraging semantic relationships to generalize to unseen categories.

**Goal**: (1) Establish the first evaluation benchmark for prompt generalization in ALMs; (2) Design a plug-and-play framework to mitigate the BNT.

**Key Insight**: LLMs are used to generate semantic neighbors (synonyms, acoustic variants) for each category, which are then incorporated into the prompt tuning process to explicitly regularize the embedding space so that each category and its semantic neighbors form compact clusters.

**Core Idea**: By expanding the semantic coverage of each category via semantic neighbors, and employing a loss that pulls positive pairs together while pushing negative pairs apart, the approach preserves the semantic structure of the embedding space—simultaneously improving base performance and maintaining generalization to new categories.

## Method

### Overall Architecture

SEPT is a plug-and-play framework that can be integrated into any prompt tuning method. The pipeline proceeds as follows: (1) an LLM generates $N$ semantic neighbors for each category; (2) margin constraints are computed using the average distance across $T$ hand-crafted prompts as a reference; (3) during training, a semantic expansion loss $\mathcal{L}_{se}$ is added on top of the standard cross-entropy loss. No additional computational overhead is introduced at inference time.

### Key Designs

1. **Semantic Neighbor Generation**:

    - Function: Expands the semantic coverage of each category.
    - Mechanism: An LLM generates $N$ semantically related terms $\{p_i^1, \ldots, p_i^N\}$ for each class name $c_i$, capturing fine-grained acoustic variants and natural language expressions. These neighbors produce embeddings through the shared learnable prompt and the frozen text encoder.
    - Design Motivation: The scarcity of categories in audio datasets yields a sparse semantic space; semantic neighbors provide additional semantic anchors.

2. **Margin-Constrained Semantic Expansion Loss**:

    - Function: Regularizes the embedding space while preserving natural semantic distances.
    - Mechanism: Two components—(a) intra-class alignment loss $\mathcal{L}_{\text{intra}}$: applies an attractive force when the distance between a category embedding $\mathbf{z}_i$ and its positive neighbor $\mathbf{p}_i^n$ exceeds the pre-computed margin $m_{i,i,n}$; (b) inter-class separation loss $\mathcal{L}_{\text{inter}}$: applies a repulsive force when the distance between a category embedding and a neighbor of another class falls below the pre-computed margin $m_{i,j,n}$. Margins are computed as the mean L2 distance across $T$ hand-crafted prompts, reflecting the original semantic distances.
    - Design Motivation: Naïve attraction–repulsion may over-compress positive pairs or over-separate negative pairs; margin constraints preserve the natural semantic hierarchy (e.g., "bell" and "chime" should be close, while "explosion" and "birdsong" should be far apart).

3. **Plug-and-Play Integration**:

    - Function: Seamlessly integrates with various prompt tuning baselines.
    - Mechanism: The total loss is $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{ce}} + \lambda \cdot \mathcal{L}_{\text{se}}$, where $\lambda$ is a balancing hyperparameter. Compatible with methods including CoOp, CoCoOp, KgCoOp, and DePT without affecting inference efficiency.
    - Design Motivation: Provides a general regularization mechanism rather than being tied to any specific method.

### Loss & Training

Standard cross-entropy loss combined with the semantic expansion loss (intra-class alignment + inter-class separation, both in hinge loss form). The text and audio encoders are frozen; only the prompt vectors are optimized.

## Key Experimental Results

### Main Results

**Average across 11 audio datasets (Base-to-New Generalization)**

| Method | Base | New | H (Harmonic Mean) |
|--------|------|-----|-------------------|
| CoOp | 65.00 | 34.09 | 42.83 |
| **CoOp + SEPT** | 64.36 | **42.98** | **49.70** |
| CoCoOp | 69.13 | 36.83 | 46.26 |
| **CoCoOp + SEPT** | 68.63 | **42.59** | **50.65** |
| KgCoOp | 37.99 | 37.42 | 36.39 |
| **KgCoOp + SEPT** | **58.92** | **45.28** | **49.79** |

### Ablation Study

| Configuration | Base | New | H | Note |
|---------------|------|-----|---|------|
| CoOp + SEPT (Full) | 64.36 | 42.98 | 49.70 | Best |
| $\mathcal{L}_{\text{intra}}$ only | — | — | Drop | Missing inter-class separation |
| $\mathcal{L}_{\text{inter}}$ only | — | — | Drop | Missing intra-class compactness |
| Without margin constraints | — | — | Drop | Over-compression / over-separation |

### Key Findings

- SEPT yields the most pronounced improvement on new categories (CoOp: 34.09→42.98, +8.89%), while base accuracy drops only marginally (65.00→64.36).
- KgCoOp benefits the most (H: 36.39→49.79, +13.4%), indicating that SEPT is complementary to existing regularization methods.
- SEPT is the first work to systematically evaluate base-to-new generalization and cross-dataset transfer in ALMs.
- Margin constraints are critical for preventing over-compression of positive pairs—performance degrades without them.

## Highlights & Insights

- The identification of "semantic sparsity" as the reason BNT is more severe in ALMs than in VLMs is a clear and convincing analysis that directly motivates the proposed solution.
- The margin constraint design is elegant—using the distances obtained from hand-crafted prompts as a reference for "naturally appropriate distances" is both simple and effective.
- The plug-and-play design allows SEPT to directly augment multiple existing methods, making it highly practical.

## Limitations & Future Work

- The quality of semantic neighbors depends on the LLM, which may require domain-specific knowledge for specialized domains (e.g., medical audio).
- Validation is limited to audio classification; tasks such as audio retrieval and audio captioning are not covered.
- Margin computation requires $T$ hand-crafted prompts, introducing an additional preprocessing step.
- The potential applicability of SEPT to vision-language models has not been explored.

## Related Work & Insights

- **vs. CoOp/CoCoOp**: SEPT is an orthogonal regularization approach and can be directly stacked on top of these methods.
- **vs. KgCoOp**: KgCoOp regularizes toward hand-crafted prompts via Euclidean distance, whereas SEPT regularizes toward semantic structure via semantic neighbors—the two approaches differ in mechanism but are complementary.

## Rating

- Novelty: ⭐⭐⭐⭐ Similar ideas exist for VLMs, but this represents the first systematic application and evaluation in the ALM setting.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 11 datasets, four baseline methods, complete ablations, base-to-new + cross-dataset evaluations.
- Writing Quality: ⭐⭐⭐⭐ Motivation and methodology are clearly articulated.
- Value: ⭐⭐⭐⭐ Establishes a benchmark for ALM prompt generalization and provides an effective solution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Temporal Contrastive Decoding: A Training-Free Method for Large Audio-Language Models](temporal_contrastive_decoding_a_training-free_method_for_large_audio-language_mo.md)
- [\[ACL 2026\] HalluAudio: A Comprehensive Benchmark for Hallucination Detection in Large Audio-Language Models](halluaudio_a_comprehensive_benchmark_for_hallucination_detection_in_large_audio-.md)
- [\[ACL 2026\] SpeakerSleuth: Can Large Audio-Language Models Judge Speaker Consistency across Multi-turn Dialogues?](speakersleuth_can_large_audio-language_models_judge_speaker_consistency_across_m.md)
- [\[ACL 2026\] Robustness via Referencing: Defending against Prompt Injection Attacks by Referencing the Executed Instruction](robustness_via_referencing_defending_against_prompt_injection_attacks_by_referen.md)
- [\[AAAI 2026\] AHAMask: Reliable Task Specification for Large Audio Language Models without Instructions](../../AAAI2026/audio_speech/ahamask_reliable_task_specification_for_large_audio_language.md)

</div>

<!-- RELATED:END -->
