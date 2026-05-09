---
title: >-
  [Paper Note] SpeakerSleuth: Can Large Audio-Language Models Judge Speaker Consistency across Multi-turn Dialogues?
description: >-
  [ACL 2026][Audio & Speech][Large Audio-Language Models] SpeakerSleuth introduces the first benchmark (1,818 instances) for evaluating LALMs' ability to judge speaker consistency in multi-turn dialogues. Systematic evaluation of 12 LALMs and 6 embedding methods reveals that models struggle to detect and localize acoustic inconsistencies, exhibit severe text-over-acoustics modality bias, yet perform comparatively well on comparative/ranking tasks involving acoustic variants.
tags:
  - ACL 2026
  - "Audio & Speech"
  - Large Audio-Language Models
  - Speaker Consistency
  - Multi-turn Dialogue
  - Benchmark
  - Modality Bias
date: 2026-05-08
content_hash: c1371a0bf1298e02
---

# SpeakerSleuth: Can Large Audio-Language Models Judge Speaker Consistency across Multi-turn Dialogues?

**Conference**: ACL 2026
**arXiv**: [2601.04029](https://arxiv.org/abs/2601.04029)
**Code**: [https://github.com/holi-lab/SpeakerSleuth](https://github.com/holi-lab/SpeakerSleuth)
**Area**: Audio & Speech
**Keywords**: Large Audio-Language Models, Speaker Consistency, Multi-turn Dialogue, Benchmark, Modality Bias

## TL;DR

SpeakerSleuth introduces the first benchmark (1,818 instances) for evaluating LALMs' ability to judge speaker consistency in multi-turn dialogues. Systematic evaluation of 12 LALMs and 6 embedding methods reveals that models struggle to detect and localize acoustic inconsistencies, exhibit severe text-over-acoustics modality bias, yet perform comparatively well on comparative/ranking tasks involving acoustic variants.

## Background & Motivation

**State of the Field**: Speech synthesis technology can now generate naturalistic human speech and is widely deployed in voice assistants, podcasts, film dubbing, and dialogue agents. Maintaining speaker identity consistency across multi-turn dialogues—timbre, pitch, and voice quality—is a fundamental requirement.

**Limitations of Prior Work**:
- Even state-of-the-art speech synthesis models suffer from speaker confusion, timbre drift, and voice quality variation.
- Existing evaluation methods compute pairwise similarity via embedding models, which cannot holistically assess consistency across an entire dialogue and require manual threshold tuning.
- Although LALMs can process an entire dialogue in one pass and directly produce a judgment, whether their acoustic discrimination capability is reliable remains entirely unknown.

**Root Cause**: LALMs are theoretically capable of serving as comprehensive audio-language judges, yet no systematic benchmark exists to assess whether they possess reliable acoustic discrimination capability, particularly in multi-turn dialogue settings.

**Paper Goals**: Construct a benchmark to systematically evaluate both LALMs and embedding-based methods on speaker consistency judgment in multi-turn dialogues, and to reveal their strengths, weaknesses, and fundamental limitations.

**Starting Point**: Three progressively challenging tasks are designed—detection (consistent or not) → localization (which turn is inconsistent) → discrimination (comparison and ranking of variants)—to comprehensively assess acoustic discrimination capability at different granularities.

**Core Idea**: A controlled experimental design (identical dialogue content × three scenarios: fully consistent / gender-switched / similar-speaker substitution) isolates acoustic factors for systematic evaluation, thereby exposing the modality bias of LALMs.

## Method

### Overall Architecture

The SpeakerSleuth benchmark comprises: (1) collecting multi-turn dialogue audio from 4 datasets; (2) generating 3 scenario variants per dialogue (S1: fully consistent, S2: gender-switched, S3: similar-speaker substitution); (3) quality assurance via human verification; and (4) evaluating 12 LALMs and 6 embedding methods across detection, localization, and discrimination tasks.

### Key Designs

1. **Three Controlled Scenario Designs**:
   - **Function**: Isolate acoustic discrimination capability through controlled variables.
   - **Mechanism**:
     - S1 (Fully Consistent): The original dialogue serves as the positive sample.
     - S2 (Gender-Switched): One randomly selected turn is replaced with an opposite-gender speaker via voice conversion, introducing an obvious acoustic deviation.
     - S3 (Similar Speaker): One turn is replaced with the most acoustically similar same-gender speaker (highest cosine similarity under ECAPA-TDNN embeddings), testing fine-grained discrimination.
   - **Design Motivation**: S1/S2/S3 share identical dialogue content, so performance differences directly reflect acoustic discrimination ability. The increasing difficulty from S2 to S3 probes the gradient of models' acoustic sensitivity.

2. **Three-Level Task Hierarchy**:
   - **Function**: Evaluate acoustic discrimination capability at coarse-to-fine granularities.
   - **Mechanism**:
     - Detection (absolute judgment): Determine whether all turns belong to the same speaker, requiring a stable internal threshold.
     - Localization (fine-grained analysis): Identify which specific turn is inconsistent, requiring turn-level acoustic feature discrimination.
     - Discrimination (relative comparison): Rank three candidate audio clips by acoustic similarity; assessed in both classification and ranking forms.
   - **Design Motivation**: Mirrors real TTS workflows—first detect inconsistency → localize the problematic turn → regenerate and select the best candidate.

3. **Modality Bias Experiment (Effect of Textual Context)**:
   - **Function**: Reveal modality imbalance between textual and acoustic signals in LALMs.
   - **Mechanism**: Building on the main experiment, textual context from other speakers' turns is additionally provided, and the resulting effect on detection performance is observed.
   - **Design Motivation**: In real-world applications, LALMs receive both audio and text simultaneously; it is therefore necessary to verify whether models neglect acoustic inconsistencies due to textual coherence.

### Loss & Training

SpeakerSleuth is an evaluation benchmark rather than a training framework. Data construction involves FreeVC for voice conversion, automated text filtering (Qwen3-32B), and manual audio quality verification.

## Key Experimental Results

### Main Results (Detection — Balanced Accuracy)

| Model | S1 Acc | S2 Acc | S3 Acc | Balanced Acc | Notes |
|---|---|---|---|---|---|
| Gemini-2.5-Pro | 73.9 | 71.6 | 39.3 | **64.7** | Best LALM |
| GPT-4o-audio | 72.9 | 32.8 | 29.5 | 52.0 | Weak detection |
| Pairwise (WavLM) | 91.8 | 38.4 | 37.7 | **64.9** | Best embedding method |
| Pairwise (ECAPA) | 36.0 | 88.4 | 86.3 | 61.7 | Over-detection tendency |

### Discrimination Task

| Model | Classification Acc | NDCG@1 | Exact Match | Notes |
|---|---|---|---|---|
| Gemini-2.5-Pro | **81.5** | **88.8** | **71.5** | Strong relative judgment |
| Pairwise (ECAPA) | **99.2** | **99.6** | 58.6 | Excellent embedding ranking |

### Effect of Textual Context (Detection)

| Model | S2 Audio-only | S2 +Text | Δ | Notes |
|---|---|---|---|---|
| GPT-4o-audio | 32.8 | 6.3 | **−26.5** | Severe textual interference |
| Gemini-2.5-Flash-Lite | 70.3 | 3.3 | **−67.0** | Near-complete failure |
| Gemini-2.5-Pro | 71.6 | 46.8 | −24.8 | Affected but retains some discrimination |

### Key Findings

- **Unstable Detection Threshold**: LALMs cluster along the anti-diagonal—either systematically predicting consistency (e.g., MiniCPM-o) or systematically predicting inconsistency (e.g., Qwen2.5-Omni-7B)—indicating a lack of calibrated internal thresholds.
- **Very Weak Localization**: Most models either default to marking no turns or indiscriminately mark all turns (e.g., Gemma-3n achieves 95% recall but only 19% precision).
- **Strong Discrimination Performance**: The same models perform well on relative comparison and ranking of acoustic variants (Gemini-2.5-Pro: 88.8% NDCG@1), suggesting that models possess inherent acoustic discrimination capacity but cannot make reliable absolute judgments.
- **Severe Textual Modality Bias**: When textual context is added, models prioritize textual coherence over acoustic cues, failing to detect even highly salient inconsistencies such as gender switching.
- Embedding-based methods also exhibit systematic bias: ECAPA-TDNN tends toward over-detection, while WavLM tends toward under-detection.

## Highlights & Insights

- The discovery of a fundamental text-over-acoustics modality bias in LALMs serves as an important warning for the development of reliable audio-language judges.
- The counter-intuitive finding that models detect poorly yet discriminate well reveals the root cause: the problem is not a lack of acoustic perceptual ability, but rather the absence of reliable internal decision thresholds.
- The controlled three-scenario design (consistent / gender-switched / similar speaker) is elegant in its clean isolation of acoustic factors.
- The concurrent evaluation of both LALMs and embedding-based methods provides a fair comparison and complementary insights across both paradigms.

## Limitations & Future Work

- The voice conversion tool used in benchmark construction may introduce artifacts, potentially affecting the naturalness of certain scenarios.
- Only English dialogue data are evaluated; cross-lingual assessment remains to be conducted.
- Each target speaker is fixed at 5 turns; consistency evaluation over longer dialogues is not addressed.
- The evaluation set size (1,818 instances) is relatively limited, and statistical power may be insufficient to distinguish subtle performance differences between certain models.

## Related Work & Insights

- **vs. Traditional Speaker Verification (ECAPA-TDNN)**: Traditional methods perform pairwise comparison; SpeakerSleuth evaluates holistic dialogue-level consistency judgment.
- **vs. LALM-as-Judge (Speech Quality Assessment)**: Existing LALM-based judges primarily focus on single-dimension speech quality; SpeakerSleuth is the first to evaluate cross-turn speaker consistency.
- **vs. Speaker Recognition/Diarization**: Traditional tasks identify *who is speaking*; SpeakerSleuth evaluates *whether utterances claimed to be from the same speaker are acoustically consistent*.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The first benchmark for speaker consistency evaluation in multi-turn dialogues; the modality bias finding carries significant implications.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Comprehensive coverage of 12 LALMs, 6 embedding methods, three-level tasks, textual context analysis, and reference audio influence analysis.
- **Writing Quality**: ⭐⭐⭐⭐ — The logical flow from task design to benchmark construction to experimental analysis is clear, and key findings are well summarized.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] HalluAudio: A Comprehensive Benchmark for Hallucination Detection in Large Audio-Language Models](halluaudio_a_comprehensive_benchmark_for_hallucination_detection_in_large_audio-.md)
- [\[ACL 2026\] How Hypocritical Is Your LLM Judge? Listener–Speaker Asymmetries in the Pragmatic Competence of Large Language Models](how_hypocritical_is_your_llm_judge_listener-speaker_asymmetries_in_the_pragmatic.md)
- [\[ACL 2026\] Temporal Contrastive Decoding: A Training-Free Method for Large Audio-Language Models](temporal_contrastive_decoding_a_training-free_method_for_large_audio-language_mo.md)
- [\[ACL 2026\] StressTest: Can YOUR Speech LM Handle the Stress?](stresstest_can_your_speech_lm_handle_the_stress.md)
- [\[ACL 2026\] Breaking Block Boundaries: Anchor-based History-stable Decoding for Diffusion Large Language Models](breaking_block_boundaries_anchor-based_history-stable_decoding_for_diffusion_lar.md)

<!-- RELATED:END -->
