---
title: >-
  [Paper Note] Omni-Embed-Audio: Leveraging Multimodal LLMs for Robust Audio-Text Retrieval
description: >-
  [ACL 2026][Audio & Speech][Audio-text retrieval] This paper proposes OEA (Omni-Embed-Audio), which leverages multimodal LLMs as unified encoders to construct a retrieval-oriented audio-text embedding space. It introduces…
tags:
  - "ACL 2026"
  - "Audio & Speech"
  - "Audio-text retrieval"
  - "CLAP"
  - "Multimodal LLM"
  - "User-intent queries"
  - "Hard negative discrimination"
date: 2026-05-08
content_hash: 0c8b54b13d5ed183
---

# Omni-Embed-Audio: Leveraging Multimodal LLMs for Robust Audio-Text Retrieval

**Conference**: ACL 2026  
**arXiv**: [2604.18360](https://arxiv.org/abs/2604.18360)  
**Code**: [Web Demo](https://omni-embed-audio.github.io)  
**Area**: Multimodal VLM / Audio Retrieval  
**Keywords**: Audio-text retrieval, CLAP, Multimodal LLM, User-intent queries, Hard negative discrimination

## TL;DR
This paper proposes OEA (Omni-Embed-Audio), which leverages multimodal LLMs as unified encoders to construct a retrieval-oriented audio-text embedding space. It introduces the User-Intent Queries (UIQ) benchmark and hard negative discrimination metrics (HNSR/TFR), finding that the LLM backbone significantly outperforms CLAP-based methods in T2T retrieval (+22%) and hard negative discrimination (+4.3%p HNSR@10).

## Background & Motivation

**Background**: Methods based on Contrastive Language-Audio Pretraining (CLAP) have become the mainstream paradigm for audio-text retrieval. The latest M2D-CLAP achieves SOTA by combining self-supervised masked modeling with CLAP. Performance on standard benchmarks (AudioCaps, Clotho) continues to improve.

**Limitations of Prior Work**: (1) Standard benchmarks use descriptive caption-style queries, which differ significantly from real search behavior—real Freesound queries average only 1.8 words; (2) Performance drops by up to 16% when facing paraphrased queries; (3) Existing metrics only check if the target is retrieved, without measuring the model's ability to suppress acoustically similar but semantically different distractors—i.e., a lack of discrimination evaluation.

**Key Challenge**: CLAP models' text encoders are lightweight and optimized for contrastive alignment with audio, compressing the entire query into a "bag-of-content" vector—this prevents them from handling negative semantics ("no thunder") and fine-grained semantic distinctions, which are core requirements in real search scenarios.

**Goal**: (1) Build a unified retrieval encoder based on multimodal LLMs; (2) Systematically evaluate retrieval robustness under various real-world query types; (3) Propose new metrics to measure hard negative discrimination capability.

**Key Insight**: LLMs have been exposed to numerous negative patterns ("not", "except") during instruction-following pretraining, and their attention mechanisms can maintain complex semantic structures—complementing the lightweight text encoders of CLAP.

**Core Idea**: Use multimodal LLMs with native audio understanding capabilities as unified encoders, paired with LoRA adaptation and contrastive learning, to surpass specialized CLAP models in retrieval quality and semantic discrimination.

## Method

### Overall Architecture
OEA uses a single shared multimodal LLM backbone to simultaneously process text and audio inputs. Text queries are encoded with a "query:" prefix, and audio is processed by the model's native audio encoder with a "passage:" prefix. Representations are obtained via mean pooling of the last hidden layer for both modalities, then mapped to a shared 512-dimensional L2-normalized embedding space via modality-specific projection heads. The backbone weights are frozen, and only the LoRA adapters (approx. 11-16M parameters) and projection heads are trained.

### Key Designs

1. **Unified LLM Backbone Encoder Architecture**:

    - **Function**: Simultaneously encode text and audio with a single Transformer, eliminating the modality gap of traditional dual-encoders
    - **Mechanism**: Multimodal LLMs with native audio understanding (Nemotron-3B, Qwen2.5-Omni-3B/7B) are chosen as the shared backbone. LoRA adapters are attached to attention layers, and modality-specific projection heads compress backbone representations into a 512-dimensional shared space. All backbone weights are frozen; only LoRA + projection heads (approx. 0.29-0.36% of total parameters) are trained
    - **Design Motivation**: Traditional dual-encoders (e.g., CLAP) use independent lightweight text and audio encoders, where text encoder expressiveness is limited; a shared LLM backbone allows audio understanding to benefit from the LLM’s rich linguistic priors

2. **User-Intent Queries (UIQ) Benchmark**:

    - **Function**: Systematically evaluate retrieval model robustness against diverse real-world query types
    - **Mechanism**: Five query types are defined across three categories: Conversational (Question—natural language questions, Imperative—command-style instructions), Rephrased (Keyphrase—keyword tags, Paraphrase—synonym rephrasing), and Exclusionary (Negative—queries specifying excluded content). These are generated using GPT-5.1 under lexical constraints and length control, validated by humans (9 annotators, mean score 4.15/5)
    - **Design Motivation**: Existing benchmarks only test caption-style queries, failing to reflect the diversity of real user search behavior—especially imperative and exclusionary queries

3. **Hard Negative Mining and Discrimination Metrics (HNSR/TFR)**:

    - **Function**: Evaluate the model's ability to suppress acoustically similar distractors while retrieving the target
    - **Mechanism**: A four-stage hard negative mining pipeline: MGA-CLAP acoustic similarity retrieval → BGE text semantic dissimilarity filtering → Human verification → Construction of (Target Audio, Hard Negative Audio) pairs. HNSR@k is defined as the proportion where "Target is within top-k and Hard Negative is outside top-k". $\Delta$-Rank = Rank(HN) − Rank(Target) measures separation
    - **Design Motivation**: Standard R@k only checks if the target is retrieved, but for exclusionary queries, the ability to simultaneously suppress acoustically similar distractors is the core challenge

### Loss & Training
Symmetric InfoNCE contrastive learning loss is used with temperature $\tau = 0.07$. Multi-stage curriculum learning: initial audio-text alignment using WavCaps (275K samples), followed by fine-tuning with AudioCaps v2 (91K samples). Optionally, Clotho v2 training data is added (marked as +Cl). Training uses AdamW optimizer with PyTorch DDP and BFloat16 precision.

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
| Relative Gain | +13.8% | +22.4% | +4.3%p |

### Key Findings
- In T2A retrieval, OEA is roughly on par with M2D-CLAP; M2D-CLAP is stronger in-domain on AudioCaps, while OEA generalizes better across domains (Clotho/MECAT).
- In T2T retrieval, OEA leads significantly (+22% relative gain) because the LLM backbone's text understanding capability far exceeds CLAP's lightweight text encoder.
- OEA possesses a unique advantage in imperative queries (+5.1%p), stemming from the LLM's instruction-following pretraining.
- OEA is significantly stronger in hard negative discrimination (HNSR@10 +4.3%p, TFR@10 +34.7%); the LLM's attention mechanism preserves the composite structure of negative semantics.
- 7B models do not always outperform 3B models—retrieval quality is more constrained by contrastive alignment and data-backbone matching.

## Highlights & Insights
- The **"Capability Complementarity" thesis** is very clear—M2D-CLAP is stronger for in-domain caption-style retrieval, while OEA excels in T2T and semantic discrimination; the paper provides clear deployment decision rules based on these observations.
- The proposed HNSR/TFR metrics fill the gap in evaluating exclusionary queries—standard R@k cannot distinguish between "target retrieved alongside hard negatives" and "clean target retrieval."
- Encoding LLMs into retrieval models by training only 0.29-0.36% of parameters is extremely parameter-efficient.

## Limitations & Future Work
- OEA depends on multimodal LLM backbones with native audio understanding, limiting the selection of available base models.
- Memory footprint is significantly larger than CLAP (18.3GB vs ~0.6GB), requiring quantization or distillation for edge deployment.
- Hard negative filtering using MGA-CLAP + BGE might miss certain types of acoustic confusion.
- UIQ is generated by a single LLM; though human-verified, it may not cover all real-world query styles.
- Performance in multilingual audio retrieval scenarios remains unevaluated.
- High audio encoding latency (666ms/clip for Qwen7B) necessitates pre-computation for real-time applications.

## Related Work & Insights
- **vs M2D-CLAP (Niizumi et al., 2025)**: M2D-CLAP is stronger in T2A but lacks in T2T and discrimination; OEA’s semantic understanding advantage comes from the LLM backbone.
- **vs RobustCLAP (Selvakumar et al., 2024)**: RobustCLAP optimizes for paraphrase robustness but doesn't handle exclusionary queries; OEA handles negative semantics naturally through the LLM.
- **vs NevIR/ExcluIR (Weller et al., 2023)**: These works found text retrieval models perform near random on negative queries; OEA proves LLM backbones can partially solve this issue.

## Rating
- Novelty: ⭐⭐⭐⭐ Using multimodal LLMs as audio retrieval encoders is a fresh perspective; UIQ benchmark and discrimination metrics are significant contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 3 datasets, 6 OEA variants, 4 CLAP baselines, 5 query types, and multi-dimensional analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear conclusions, well-thought-out experimental design, and practical deployment suggestions.
- Value: ⭐⭐⭐⭐ Advances the audio retrieval evaluation paradigm; the UIQ benchmark can be widely adopted by the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Protecting Bystander Privacy via Selective Hearing in Audio LLMs](protecting_bystander_privacy_via_selective_hearing_in_audio_llms.md)
- [\[ACL 2026\] PlanRAG-Audio: Planning and Retrieval Augmented Generation for Long-form Audio Understanding](planrag-audio_planning_and_retrieval_augmented_generation_for_long-form_audio_un.md)
- [\[CVPR 2026\] OmniRet: Efficient and High-Fidelity Omni Modality Retrieval](../../CVPR2026/audio_speech/omniret_efficient_and_high-fidelity_omni_modality_retrieval.md)
- [\[ACL 2026\] Music Audio-Visual Question Answering Requires Specialized Multimodal Designs](music_audio-visual_question_answering_requires_specialized_multimodal_designs.md)
- [\[NeurIPS 2025\] Node-Based Editing for Multimodal Generation of Text, Audio, Image, and Video](../../NeurIPS2025/audio_speech/node-based_editing_for_multimodal_generation_of_text_audio_image_and_video.md)

</div>

<!-- RELATED:END -->
