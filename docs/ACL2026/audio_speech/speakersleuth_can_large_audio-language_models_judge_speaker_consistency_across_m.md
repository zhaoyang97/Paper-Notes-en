---
title: >-
  [Paper Note] SpeakerSleuth: Can Large Audio-Language Models Judge Speaker Consistency across Multi-turn Dialogues?
description: >-
  [ACL 2026][Audio & Speech][Large Audio-Language Models] SpeakerSleuth constructs the first benchmark (1,818 instances) to evaluate the capability of LALMs in judging speaker consistency within multi-turn dialogues. Syste…
tags:
  - "ACL 2026"
  - "Audio & Speech"
  - "Large Audio-Language Models"
  - "Speaker Consistency"
  - "Multi-turn Dialogue"
  - "Benchmark"
  - "Modality Bias"
date: 2026-05-08
content_hash: 44383a5b3824dd31
---

# SpeakerSleuth: Can Large Audio-Language Models Judge Speaker Consistency across Multi-turn Dialogues?

**Conference**: ACL 2026  
**arXiv**: [2601.04029](https://arxiv.org/abs/2601.04029)  
**Code**: [https://github.com/holi-lab/SpeakerSleuth](https://github.com/holi-lab/SpeakerSleuth)  
**Area**: Audio & Speech  
**Keywords**: Large Audio-Language Models, Speaker Consistency, Multi-turn Dialogue, Benchmark, Modality Bias

## TL;DR

SpeakerSleuth constructs the first benchmark (1,818 instances) to evaluate the capability of LALMs in judging speaker consistency within multi-turn dialogues. Systematic evaluation of 12 LALMs and 6 embedding methods reveals that models struggle with detecting and locating acoustic inconsistencies and exhibit a severe "text-over-audio" modality bias, although they perform relatively well in comparing/ranking acoustic variants.

## Background & Motivation

**Background**: Speech synthesis technology has achieved the generation of natural human speech, widely applied in voice assistants, podcasts, film dubbing, and conversational agents. Maintaining speaker identity consistency (timbre, pitch, voice quality) across multi-turn dialogues is a fundamental requirement.

**Limitations of Prior Work**:
- Even the latest speech synthesis models suffer from issues such as speaker confusion, timbre drift, and voice quality variations.
- Existing evaluation methods rely on embedding models to compute pairwise similarity, which fails to evaluate the dialogue's consistency holistically and requires manual threshold setting.
- Although LALMs can process entire dialogues and output judgments directly, whether their acoustic discriminative capabilities are reliable remains unknown.

**Key Challenge**: Theoretically, LALMs could serve as comprehensive audio-language judges, but there is a lack of systematic benchmarks to evaluate whether they possess reliable acoustic discriminative power, especially in multi-turn dialogue scenarios.

**Goal**: To build a benchmark for systematically evaluating the capabilities of LALMs and embedding methods in judging speaker consistency in multi-turn dialogues, revealing their strengths, weaknesses, and core limitations.

**Key Insight**: Designing three progressive tasks—Detection (consistency check) → Localization (identifying inconsistent turns) → Discrimination (comparing and ranking variants)—to comprehensively evaluate acoustic discriminative abilities at different levels.

**Core Idea**: Through a controlled experimental design (same dialogue × three scenarios: fully consistent / gender switch / similar speaker replacement), the study isolates acoustic factors for systematic evaluation, revealing the modality bias of LALMs.

## Method

### Overall Architecture

The SpeakerSleuth benchmark consists of: (1) multi-turn dialogue audio collected from 4 datasets; (2) 3 scenarios generated for each dialogue (Consistent S1, Gender Switch S2, Similar Speaker S3); (3) quality assurance via human verification; (4) evaluation of 12 LALMs and 6 embedding methods across detection, localization, and discrimination tasks.

### Key Designs

1. **Three Controlled Scenarios**:
    - Function: Isolating acoustic discriminative ability through controlled variables.
    - Mechanism:
        - S1 (Fully Consistent): Original dialogues serve as positive samples.
        - S2 (Gender Switch): A random turn is replaced by an opposite-gender speaker using voice conversion to create obvious acoustic deviations.
        - S3 (Similar Speaker): A turn is replaced by the most acoustically similar same-gender speaker (highest cosine similarity via ECAPA-TDNN embeddings) to test fine-grained discrimination.
    - Design Motivation: S1/S2/S3 use identical text content; performance differences directly reflect acoustic discriminative power. The increasing difficulty from S2 to S3 tests the model's acoustic sensitivity gradient.

2. **Three-level Task Hierarchy**:
    - Function: Evaluating acoustic discriminative capabilities from coarse to fine levels.
    - Mechanism:
        - Detection (Absolute Judgment): Determining if all turns belong to the same speaker, requiring a stable internal threshold.
        - Localization (Fine-grained Analysis): Identifying specific inconsistent turns, requiring turn-level acoustic feature differentiation.
        - Discrimination (Relative Comparison): Ranking three candidate audio clips based on acoustic similarity, testing relative judgment (via classification and ranking).
    - Design Motivation: This corresponds to practical TTS workflows—detecting inconsistency → locating problematic turns → regenerating and selecting the best option.

3. **Modality Bias Experiments (Impact of Textual Context)**:
    - Function: Revealing the modality imbalance between text and acoustic signals in LALMs.
    - Mechanism: Building on the main experiment, additional text context of other speaker turns is provided to observe its impact on detection performance.
    - Design Motivation: In practice, LALMs receive both audio and text; it is necessary to verify whether models ignore acoustic inconsistencies due to textual coherence.

### Loss & Training

SpeakerSleuth is an evaluation benchmark rather than a training method. Data construction involves: voice conversion using FreeVC, automated text filtering (Qwen3-32B), and manual audio quality verification.

## Key Experimental Results

### Main Results (Detection - Balanced Accuracy)

| Model | S1 Acc | S2 Acc | S3 Acc | Balanced Acc | Description |
|------|--------|--------|--------|-----------|------|
| Gemini-2.5-Pro | 73.9 | 71.6 | 39.3 | **64.7** | Strongest LALM |
| GPT-4o-audio | 72.9 | 32.8 | 29.5 | 52.0 | Weak detection capability |
| Pairwise (WavLM) | 91.8 | 38.4 | 37.7 | **64.9** | Strongest embedding method |
| Pairwise (ECAPA) | 36.0 | 88.4 | 86.3 | 61.7 | Over-detection |

### Discrimination Task

| Model | Classification Acc | NDCG@1 | Exact Match | Description |
|------|-----------|--------|---------|------|
| Gemini-2.5-Pro | **81.5** | **88.8** | **71.5** | Strong relative judgment |
| Pairwise (ECAPA) | **99.2** | **99.6** | 58.6 | Excellent ranking by embedding |

### Impact of Textual Context (Detection)

| Model | S2 Audio-only | S2 + Text | Δ | Description |
|------|-------------|---------|---|------|
| GPT-4o-audio | 32.8 | 6.3 | **-26.5** | Severe text interference |
| Gemini-2.5-Flash-Lite | 70.3 | 3.3 | **-67.0** | Near total failure |
| Gemini-2.5-Pro | 71.6 | 46.8 | -24.8 | Affected but retains some judgment |

### Key Findings
- **Unstable Detection Thresholds**: LALMs cluster on the anti-diagonals—either over-predicting consistency (e.g., MiniCPM-o) or over-predicting inconsistency (e.g., Qwen2.5-Omni-7B), indicating a lack of calibrated internal thresholds.
- **Extremely Weak Localization**: Most models either default to marking no turns or indiscriminately mark all turns (e.g., Gemma-3n with 95% recall but only 19% precision).
- **Strong Performance in Discrimination**: The same models excel at relatively comparing or ranking acoustic variants (Gemini-2.5-Pro achieves 88.8% NDCG@1), suggesting models possess inherent acoustic discriminative power, but their absolute judgments are unreliable.
- **Severe Textual Bias**: When textual context is added, models prioritize textual coherence over acoustic cues, failing to detect even obvious inconsistencies like gender switches.
- **Systemic Bias in Embeddings**: ECAPA-TDNN tends toward over-detection, while WavLM tends toward omissions.

## Highlights & Insights
- Discovered a fundamental "text-over-audio" modality bias in LALMs, providing a critical warning for building reliable audio-language judges.
- The counter-intuitive finding of "poor detection but good discrimination" reveals the essence of the problem: it is not a lack of acoustic perception, but a lack of reliable internal decision thresholds.
- The controlled design using three scenarios (consistent/gender switch/similar speaker) elegantly isolates acoustic factors.
- Simultaneous evaluation of LALMs and embedding methods provides a fair comparison and complementary insights into both approaches.

## Limitations & Future Work
- Voice conversion tools used in the benchmark might introduce artifacts, affecting naturalness in certain scenarios.
- Only English dialogue data was tested; cross-lingual evaluation requires further expansion.
- Each target speaker is fixed at 5 turns; consistency in longer dialogues remains unexplored.
- The evaluation set size (1,818 instances) is relatively limited; statistical power may be insufficient to distinguish minor differences between some models.

## Related Work & Insights
- **vs. Traditional Speaker Verification (ECAPA-TDNN)**: Traditional methods perform pairwise comparisons, while SpeakerSleuth evaluates holistic dialogue-level consistency.
- **vs. LALM-as-Judge (Speech Quality Assessment)**: Existing LALM judges primarily focus on single-dimension voice quality, while SpeakerSleuth is the first to evaluate speaker consistency across turns.
- **vs. Speaker Identification/Diarization**: Traditional tasks identify "who is speaking," whereas SpeakerSleuth evaluates "whether utterances claimed to be from the same person are acoustically consistent."

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First speaker consistency benchmark for multi-turn dialogues; the discovery of modality bias is highly insightful.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across 12 LALMs, 6 embedding methods, three-level tasks, and analyses of text/reference audio influence.
- Writing Quality: ⭐⭐⭐⭐ Clear logic from task design to benchmark construction and experimental analysis, with well-summarized key findings.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Style Amnesia: Investigating Speaking Style Degradation and Mitigation in Multi-Turn Spoken Language Models](style_amnesia_investigating_speaking_style_degradation_and_mitigation_in_multi-t.md)
- [\[ACL 2026\] Temporal Contrastive Decoding: A Training-Free Method for Large Audio-Language Models](temporal_contrastive_decoding_a_training-free_method_for_large_audio-language_mo.md)
- [\[ACL 2026\] Closing the Modality Reasoning Gap for Speech Large Language Models](closing_the_modality_reasoning_gap_for_speech_large_language_models.md)
- [\[ACL 2026\] StressTest: Can YOUR Speech LM Handle the Stress?](stresstest_can_your_speech_lm_handle_the_stress.md)
- [\[ACL 2026\] SEPT: Semantically Expanded Prompt Tuning for Audio-Language Models](generalizable_prompt_tuning_for_audio-language_models_via_semantic_expansion.md)

</div>

<!-- RELATED:END -->
