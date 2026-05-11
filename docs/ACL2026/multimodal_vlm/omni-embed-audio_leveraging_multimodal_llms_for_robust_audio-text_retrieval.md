---
title: >-
  [Paper Note] Omni-Embed-Audio: Leveraging Multimodal LLMs for Robust Audio-Text Retrieval
description: >-
  [ACL 2026][Multimodal VLM][Audio-text retrieval] This paper proposes OEA (Omni-Embed-Audio), which employs a multimodal LLM as a unified encoder to construct a retrieval-oriented audio-text embedding space. It introduces…
tags:
  - "ACL 2026"
  - "Multimodal VLM"
  - "Audio-text retrieval"
  - "CLAP"
  - "multimodal LLM"
  - "user-intent queries"
  - "hard negative discrimination"
date: 2026-05-08
content_hash: 7eb9cd3f655afb9f
---

# Omni-Embed-Audio: Leveraging Multimodal LLMs for Robust Audio-Text Retrieval

**Conference**: ACL 2026
**arXiv**: [2604.18360](https://arxiv.org/abs/2604.18360)
**Code**: [Web Demo](https://omni-embed-audio.github.io)
**Area**: Multimodal VLM / Audio Retrieval
**Keywords**: Audio-text retrieval, CLAP, multimodal LLM, user-intent queries, hard negative discrimination

## TL;DR
This paper proposes OEA (Omni-Embed-Audio), which employs a multimodal LLM as a unified encoder to construct a retrieval-oriented audio-text embedding space. It introduces the User-Intent Queries (UIQ) benchmark and hard-negative discrimination metrics (HNSR/TFR), demonstrating that the LLM backbone significantly outperforms CLAP-based methods on T2T retrieval (+22%) and hard negative discrimination (+4.3%p HNSR@10).

## Background & Motivation

**Background**: Contrastive Language-Audio Pretraining (CLAP)-based methods have become the dominant paradigm for audio-text retrieval. The latest M2D-CLAP achieves state-of-the-art performance by combining self-supervised masked modeling with CLAP. Performance on standard benchmarks (AudioCaps, Clotho) continues to improve.

**Limitations of Prior Work**: (1) Standard benchmarks use caption-style queries that diverge substantially from real-world search behavior—actual Freesound queries average only 1.8 words; (2) existing models suffer up to 16% performance degradation on paraphrase queries; (3) standard evaluation metrics only check whether the target is retrieved, without measuring whether models can suppress acoustically similar but semantically distinct distractors—i.e., discrimination ability is not evaluated.

**Key Challenge**: CLAP text encoders are lightweight and optimized for contrastive audio alignment, compressing entire queries into "bag-of-content" vectors—rendering them unable to handle negation semantics ("no thunder") or fine-grained semantic distinctions, which are precisely the core requirements of real-world search scenarios.

**Goal**: (1) Build a unified retrieval encoder based on a multimodal LLM; (2) systematically evaluate retrieval robustness across diverse real-world query types; (3) propose new metrics that measure hard negative discrimination capability.

**Key Insight**: LLMs encounter abundant negation patterns ("not," "except") during instruction-following pretraining, and their attention mechanisms can preserve compositional semantic structures—complementing the limited capacity of CLAP's lightweight text encoders.

**Core Idea**: Use a multimodal LLM with native audio understanding as a unified encoder, combined with LoRA adaptation and contrastive learning, to surpass dedicated CLAP models in retrieval quality and semantic discrimination.

## Method

### Overall Architecture
OEA uses a single shared multimodal LLM backbone to process both text and audio inputs. Text queries are encoded with a "query:" prefix, while audio inputs are processed with a "passage:" prefix through the model's native audio encoder. Both modalities obtain representations via mean pooling over the final hidden layer, which are then mapped to a shared 512-dimensional L2-normalized embedding space through modality-specific projection heads. Backbone weights are frozen; only LoRA adapters (~11–16M parameters) and projection heads are trained.

### Key Designs

1. **Unified LLM Backbone Encoder Architecture**:

    - Function: Encodes both text and audio with a single Transformer, eliminating the modality gap inherent in conventional dual encoders.
    - Mechanism: A multimodal LLM with native audio understanding (Nemotron-3B, Qwen2.5-Omni-3B/7B) is selected as the shared backbone. LoRA adapters are attached to attention layers, and modality-specific projection heads compress backbone representations into a 512-dimensional shared space. All backbone weights are frozen; only LoRA adapters and projection heads are trained (~0.29–0.36% of total parameters).
    - Design Motivation: Conventional dual encoders (e.g., CLAP) use separate lightweight text and audio encoders with limited text representational capacity; a shared LLM backbone allows audio understanding to benefit from the LLM's rich linguistic priors.

2. **User-Intent Queries (UIQ) Benchmark**:

    - Function: Systematically evaluates retrieval model robustness across diverse real-world query types.
    - Mechanism: Five query types are defined across three categories: conversational (Question—natural language questions; Imperative—command-style instructions), reformulation (Keyphrase—keyword tags; Paraphrase—synonymous rewrites), and exclusion (Negative—queries specifying content to exclude). Queries are generated using GPT-5.1 under lexical constraints and length controls, with human verification (9 annotators, mean score 4.15/5).
    - Design Motivation: Existing benchmarks test only caption-style queries, failing to reflect the diversity of real user search behavior—particularly imperative and exclusion-based queries.

3. **Hard Negative Mining and Discrimination Metrics (HNSR/TFR)**:

    - Function: Evaluates a model's ability to retrieve the target while simultaneously suppressing acoustically similar distractors.
    - Mechanism: A four-stage hard negative mining pipeline: MGA-CLAP acoustic similarity retrieval → BGE text semantic dissimilarity filtering → human verification → construction of (target audio, hard negative audio) pairs. HNSR@k is defined as the proportion of cases where the target is within top-k and the hard negative is outside top-k. $\Delta$-Rank = Rank(HN) − Rank(Target) measures separation.
    - Design Motivation: Standard R@k only checks whether the target is retrieved; for exclusion-type queries, the core challenge is whether the model can simultaneously suppress acoustically similar distractors.

### Loss & Training
Symmetric InfoNCE contrastive learning loss is used with temperature $\tau = 0.07$. Multi-stage curriculum learning is applied: initial audio-text alignment on WavCaps (275K samples), followed by fine-tuning on AudioCaps v2 (91K samples). Clotho v2 training data may optionally be added (denoted +Cl). AdamW optimizer with PyTorch DDP and BFloat16 precision training.

## Key Experimental Results

### Main Results (T2A Retrieval)

| Model | AudioCaps R@5 | Clotho R@5 | MECAT R@5 |
|------|-------------|-----------|----------|
| M2D-CLAP | **77.13** | 42.91 | 23.55 |
| OEA-Nemo3B | 72.64 | 40.57 | **24.53** |
| OEA-Qwen3B (+Cl) | 69.35 | **49.78** | 17.16 |
| OEA-Qwen7B | 72.25 | 44.78 | 23.29 |

### T2T Retrieval and Hard Negative Discrimination

| Model | Clotho T2T R@1 | MECAT T2T R@5 | HNSR@10 |
|------|---------------|--------------|---------|
| M2D-CLAP | 55.85 | 38.74 | 30.3% |
| OEA-Qwen7B (+Cl) | **63.58** | **47.41** | **34.6%** |
| Gain | +13.8% | +22.4% | +4.3%p |

### Key Findings
- On T2A retrieval, OEA and M2D-CLAP are broadly comparable; M2D-CLAP is stronger in-domain on AudioCaps, while OEA generalizes better cross-domain (Clotho/MECAT).
- On T2T retrieval, OEA leads by a substantial margin (+22% relative gain), owing to the LLM backbone's superior text comprehension compared to CLAP's lightweight text encoder.
- On imperative queries, OEA holds an exclusive advantage (+5.1%p), attributable to the LLM's instruction-following pretraining.
- On hard negative discrimination, OEA is significantly stronger (HNSR@10 +4.3%p, TFR@10 +34.7%), as the LLM's attention mechanism preserves the compositional structure of negation semantics.
- The 7B model does not consistently outperform the 3B model—retrieval quality is more constrained by contrastive alignment and data-backbone compatibility.

## Highlights & Insights
- The **"complementary capability"** argument is clearly articulated—M2D-CLAP is stronger for in-domain caption-style retrieval, while OEA excels at T2T and semantic discrimination; the two have distinct deployment scenarios, and the paper provides concrete decision rules.
- The HNSR/TFR metrics address a gap in evaluating exclusion-type queries—standard R@k cannot distinguish between "target retrieved but hard negatives also present" and "clean target retrieval."
- Transforming an LLM into a retrieval encoder by training only 0.29–0.36% of parameters demonstrates extreme parameter efficiency.

## Limitations & Future Work
- OEA depends on multimodal LLM backbones with native audio understanding, limiting the choice of base models.
- Memory footprint is substantially larger than CLAP (18.3 GB vs. ~0.6 GB); edge deployment requires quantization or distillation.
- Hard negative mining via MGA-CLAP + BGE filtering may miss certain types of acoustic confusions.
- UIQ is generated by a single LLM and, despite human verification, may not cover all real-world query styles.
- Performance in multilingual audio retrieval settings is not evaluated.
- Audio encoding latency is high (666 ms/clip for Qwen7B), requiring pre-computation for real-time scenarios.

## Related Work & Insights
- **vs M2D-CLAP (Niizumi et al., 2025)**: M2D-CLAP is stronger on T2A but lacks T2T and discrimination capability; OEA's semantic understanding advantage stems from the LLM backbone.
- **vs RobustCLAP (Selvakumar et al., 2024)**: RobustCLAP is optimized for paraphrase robustness but does not handle exclusion queries; OEA naturally handles negation semantics through the LLM.
- **vs NevIR/ExcluIR (Weller et al., 2023)**: These works find that text retrieval models perform near-randomly on negation queries; OEA demonstrates that an LLM backbone can partially address this problem.

## Rating
- Novelty: ⭐⭐⭐⭐ Using a multimodal LLM as an audio retrieval encoder is a novel perspective; the UIQ benchmark and discrimination metrics are meaningful contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three datasets, six OEA variants, four CLAP baselines, five query types, and multi-dimensional analysis.
- Writing Quality: ⭐⭐⭐⭐ Conclusions are clear, experimental design is thorough, and deployment recommendations are practical.
- Value: ⭐⭐⭐⭐ Advances the evaluation paradigm for audio retrieval; the UIQ benchmark can be broadly adopted by the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] TEMA: Anchor the Image, Follow the Text for Multi-Modification Composed Image Retrieval](tema_anchor_the_image_follow_the_text_for_multi-modification_composed_image_retr.md)
- [\[NeurIPS 2025\] Watch and Listen: Understanding Audio-Visual-Speech Moments with Multimodal LLM](../../NeurIPS2025/multimodal_vlm/watch_and_listen_understanding_audio-visual-speech_moments_with_multimodal_llm.md)
- [\[AAAI 2026\] When Eyes and Ears Disagree: Can MLLMs Discern Audio-Visual Confusion?](../../AAAI2026/multimodal_vlm/when_eyes_and_ears_disagree_can_mllms_discern_audio-visual_confusion.md)
- [\[AAAI 2026\] Leveraging Textual Compositional Reasoning for Robust Change Captioning](../../AAAI2026/multimodal_vlm/leveraging_textual_compositional_reasoning_for_robust_change_captioning.md)
- [\[ACL 2026\] Faithful-First Reasoning, Planning, and Acting for Multimodal LLMs](faithful-first_reasoning_planning_and_acting_for_multimodal_llms.md)

</div>

<!-- RELATED:END -->
