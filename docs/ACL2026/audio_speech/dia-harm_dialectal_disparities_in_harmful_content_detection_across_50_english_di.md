---
title: >-
  [Paper Note] DIA-HARM: Dialectal Disparities in Harmful Content Detection Across 50 English Dialects
description: >-
  [ACL 2026][Audio & Speech][dialectal bias] This paper introduces DIA-HARM, the first benchmark for evaluating the robustness of misinformation detectors across 50 English dialects. It reveals that human-authored dialectal content causes detection performance drops of 1.4–3.6% F1, that fine-tuned Transformers substantially outperform zero-shot LLMs (96.6% vs. 78.3%), and that some models suffer catastrophic degradation exceeding 33% on mixed-content inputs.
tags:
  - ACL 2026
  - "Audio & Speech"
  - dialectal bias
  - misinformation detection
  - robustness evaluation
  - English dialects
  - detection fairness
date: 2026-05-08
content_hash: 8e7925e12e163793
---

# DIA-HARM: Dialectal Disparities in Harmful Content Detection Across 50 English Dialects

**Conference**: ACL 2026
**arXiv**: [2604.05318](https://arxiv.org/abs/2604.05318)
**Code**: [https://github.com/jsl5710/dia-harm](https://github.com/jsl5710/dia-harm)
**Area**: Content Safety / Dialectal Robustness
**Keywords**: dialectal bias, misinformation detection, robustness evaluation, English dialects, detection fairness

## TL;DR

This paper introduces DIA-HARM, the first benchmark for evaluating the robustness of misinformation detectors across 50 English dialects. It reveals that human-authored dialectal content causes detection performance drops of 1.4–3.6% F1, that fine-tuned Transformers substantially outperform zero-shot LLMs (96.6% vs. 78.3%), and that some models suffer catastrophic degradation exceeding 33% on mixed-content inputs.

## Background & Motivation

**State of the Field**: Harmful content detectors—particularly misinformation classifiers—are predominantly developed and evaluated on Standard American English (SAE), leaving their robustness to dialectal variation largely unexplored.

**Limitations of Prior Work**: (1) Hundreds of millions of English speakers worldwide use non-SAE dialects, yet detection systems have not been validated on these varieties; (2) Dialectal transformation alters morphosyntactic structure while preserving misinformative semantics—if detectors rely on surface patterns rather than deep semantic understanding, dialectal content may evade detection; (3) Detection failures may systematically afford less protection to dialect speakers.

**Root Cause**: Misinformation detectors should make judgments based on factual veracity (semantics); however, if they rely on surface linguistic patterns, dialectal variants—which alter surface form while preserving semantics—expose this vulnerability.

**Paper Goals**: (1) Construct a misinformation detection evaluation benchmark spanning 50 English dialects; (2) Assess the dialectal robustness of 16 detection models; (3) Identify cross-dialect transfer patterns.

**Starting Point**: The paper leverages Multi-VALUE's linguistically grounded, rule-based dialect transformation toolkit to convert standard misinformation datasets into 50 dialectal variants, constructing the D3 corpus (195K samples) for systematic model evaluation.

**Core Idea**: Dialectal variation constitutes a natural perturbation for misinformation detectors—altering linguistic form without changing factual veracity—thereby revealing whether detectors comprehend semantics or merely exploit surface patterns.

## Method

### Overall Architecture

DIA-HARM comprises three components: (1) D3 corpus construction—applying rule-based dialect transformation to convert SAE misinformation data into 50 dialect variants; (2) detection model evaluation—assessing 16 models (fine-tuned Transformers and zero-shot LLMs) on dialectal variants; (3) cross-dialect transfer analysis—examining performance transfer patterns across 2,450 dialect pairs.

### Key Designs

1. **Rule-Based Dialect Transformation (Multi-VALUE)**:

    - Function: Generate linguistically valid dialectal variants.
    - Mechanism: Multi-VALUE applies morphosyntactic transformation rules (e.g., tense marking, pronoun systems, article usage) to convert SAE text into 50 English dialects spanning American, British, African, Caribbean, and Asia-Pacific regions.
    - Design Motivation: Rule-based transformation ensures linguistic validity, as opposed to simple noise injection or semantically equivalent paraphrase.

2. **Multi-Paradigm Detection Model Evaluation**:

    - Function: Comprehensively assess dialectal robustness across different detection paradigms.
    - Mechanism: Sixteen models are evaluated—fine-tuned Transformers (RoBERTa, mDeBERTa, etc.) and zero-shot LLMs (GPT-4, Llama, etc.)—distinguishing between human-authored and AI-generated dialectal content.
    - Design Motivation: Different model types may exhibit distinct vulnerability patterns—fine-tuned models may overfit to SAE, whereas zero-shot LLMs may generalize more readily.

3. **Cross-Dialect Transfer Analysis (2,450 Dialect Pairs)**:

    - Function: Identify regularities in performance transfer across dialects.
    - Mechanism: Detection performance variation is analyzed across all dialect pairs to identify which pairings yield good transfer and which cause degradation. The analysis also examines whether multilingual models (mDeBERTa) are more robust than monolingual counterparts.
    - Design Motivation: Understanding cross-dialect transfer regularities can guide model selection and data augmentation strategies.

### Loss & Training

DIA-HARM is an evaluation benchmark and does not involve novel model training. Fine-tuned models are trained on SAE data and subsequently evaluated on dialectal variants.

## Key Experimental Results

### Main Results

**Detection Performance Comparison (Best-case F1 %)**

| Model Type | SAE | Dialect (Avg.) | Worst-case Degradation |
|---|---|---|---|
| Fine-tuned Transformer (best) | 96.6 | 93–95 | −3.6 |
| Zero-shot LLM (best) | 78.3 | ~76 | −2.4 |
| Monolingual model (RoBERTa) | High | Severe degradation | >33% |
| Multilingual model (mDeBERTa) | 97.2 | 97.2 | Negligible |

### Ablation Study

| Content Type | Dialectal Impact | Notes |
|---|---|---|
| Human-authored | −1.4–3.6% F1 | Dialects significantly affect detection |
| AI-generated | Stable | AI-generated content unaffected by dialect |
| Mixed content | >33% degradation in some models | Most dangerous scenario |

### Key Findings

- Fine-tuned Transformers substantially outperform zero-shot LLMs (96.6% vs. 78.3%), though their dialectal vulnerabilities vary.
- Multilingual models (mDeBERTa: 97.2% average F1) generalize best across dialectal variants.
- Monolingual models (RoBERTa) can suffer catastrophic failure on dialectal inputs (>33% degradation).
- AI-generated dialectal content does not affect detection performance, whereas human-authored dialectal content causes significant degradation—indicating that detectors partially rely on surface patterns present in human writing.
- Certain dialect pairs (e.g., Jamaican Creole) induce particularly severe detection degradation.

## Highlights & Insights

- This is the first systematic evaluation of misinformation detectors across 50 English dialects, unprecedented in scale and coverage.
- The finding that "AI-generated dialectal content remains stable while human-authored dialectal content degrades" profoundly illuminates the dependency patterns of current detectors.
- The superior dialectal robustness of multilingual pre-trained models provides clear practical guidance for model selection.

## Limitations & Future Work

- Rule-based dialect transformation may not fully capture the complexity of authentic dialectal usage.
- The study focuses solely on misinformation detection; other harmful content types (e.g., hate speech) remain to be explored.
- The 50 dialects examined do not cover all English varieties.
- Dialect-aware training as a potential defense strategy is not investigated.

## Related Work & Insights

- **vs. Hate Speech Dialect Studies (Sap et al., 2019)**: Prior work addresses dialectal bias in hate speech detection; DIA-HARM is the first to systematically evaluate misinformation detection.
- **vs. Multi-VALUE**: Multi-VALUE provides the dialect transformation toolkit; DIA-HARM applies it to safety detection evaluation.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First robustness benchmark for misinformation detection spanning 50 dialects.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 16 models, 50 dialects, 195K samples, 2,450 dialect pair analyses.
- Writing Quality: ⭐⭐⭐⭐ Addresses an important problem with comprehensive analysis.
- Value: ⭐⭐⭐⭐⭐ Exposes critical fairness gaps in detection systems with direct implications for safety system deployment.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] HalluAudio: A Comprehensive Benchmark for Hallucination Detection in Large Audio-Language Models](halluaudio_a_comprehensive_benchmark_for_hallucination_detection_in_large_audio-.md)
- [\[ACL 2026\] Alexandria: A Multi-Domain Dialectal Arabic Machine Translation Dataset for Culturally Inclusive and Linguistically Diverse LLMs](alexandria_a_multi-domain_dialectal_arabic_machine_translation_dataset_for_cultu.md)
- [\[AAAI 2026\] A Mind Cannot Be Smeared Across Time](../../AAAI2026/audio_speech/a_mind_cannot_be_smeared_across_time.md)
- [\[ACL 2026\] SpeakerSleuth: Can Large Audio-Language Models Judge Speaker Consistency across Multi-turn Dialogues?](speakersleuth_can_large_audio-language_models_judge_speaker_consistency_across_m.md)
- [\[AAAI 2026\] Thucy: An LLM-based Multi-Agent System for Claim Verification across Relational Databases](../../AAAI2026/audio_speech/thucy_an_llm-based_multi-agent_system_for_claim_verification_across_relational_d.md)

<!-- RELATED:END -->
