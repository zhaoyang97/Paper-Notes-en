---
title: >-
  [Paper Note] How Hypocritical Is Your LLM Judge? Listener–Speaker Asymmetries in the Pragmatic Competence of Large Language Models
description: >-
  [ACL 2026][Audio & Speech][Pragmatic competence] This paper systematically compares 14 LLMs as pragmatic listeners (judging pragmatic appropriateness) and pragmatic speakers (generating pragmatically appropriate language) across three tasks—false presuppositions, antipresuppositions, and deductive reasoning—revealing pervasive listener–speaker asymmetries: most models perform substantially better as judges than as generators, and item-level analysis shows that correct judgments do not reliably predict successful generation.
tags:
  - ACL 2026
  - "Audio & Speech"
  - Pragmatic competence
  - listener–speaker asymmetry
  - LLM-as-a-judge
  - false presuppositions
  - deductive reasoning
date: 2026-05-08
content_hash: b3a22f28fd17496b
---

# How Hypocritical Is Your LLM Judge? Listener–Speaker Asymmetries in the Pragmatic Competence of Large Language Models

**Conference**: ACL 2026
**arXiv**: [2604.15873](https://arxiv.org/abs/2604.15873)
**Code**: None
**Area**: Speech Processing / Pragmatic Competence Evaluation
**Keywords**: Pragmatic competence, listener–speaker asymmetry, LLM-as-a-judge, false presuppositions, deductive reasoning

## TL;DR

This paper systematically compares 14 LLMs as pragmatic listeners (judging pragmatic appropriateness) and pragmatic speakers (generating pragmatically appropriate language) across three tasks—false presuppositions, antipresuppositions, and deductive reasoning—revealing pervasive listener–speaker asymmetries: most models perform substantially better as judges than as generators, and item-level analysis shows that correct judgments do not reliably predict successful generation.

## Background & Motivation

**Background**: LLM language capability evaluation typically adopts two paradigms: generative tasks (model as "speaker") and judgment tasks (model as "listener"/judge). The LLM-as-a-judge paradigm is increasingly popular, with models serving as substitutes for human annotators.

**Limitations of Prior Work**: (1) These two evaluation roles have rarely been directly compared—researchers implicitly assume that success in one role reflects overall language competence; (2) psycholinguistic research shows that human language comprehension and production are related but distinct tasks, and successful comprehension does not guarantee successful production; (3) the reliability of LLM-as-a-judge has not been systematically validated in the pragmatic domain.

**Key Challenge**: If a model can correctly judge the pragmatic appropriateness of a response (listener role), does it follow that it can also generate pragmatically appropriate responses itself (speaker role)?

**Goal**: To directly compare LLMs' pragmatic judgment (listener) and pragmatic generation (speaker) capabilities on the same set of items, and to examine whether the two are consistent.

**Key Insight**: Drawing on classic findings of comprehension–production asymmetries in psycholinguistics, the authors design parallel listener/speaker prompts using identical underlying test items, enabling rigorous item-level comparison.

**Core Idea**: Pragmatic judgment and pragmatic generation are partially dissociated capabilities in current LLMs—"knowing what is right" does not equal "doing it right," and LLM judges may be fundamentally "hypocritical."

## Method

### Overall Architecture

Three pragmatic tasks are selected. For each task, parallel speaker prompts (requiring generation) and listener prompts (requiring judgment) are designed for the same set of test items. Fourteen LLMs (both open-source and closed-source) are evaluated, accuracy under both roles is computed, and item-level conditional analysis is conducted.

### Key Designs

1. **False Presuppositions Task**:

    - **Function**: Tests whether models can detect and reject false presuppositions embedded in questions.
    - **Mechanism**: Two German datasets are used (False Scenarios and False Claims), containing politically sensitive questions with false presuppositions. Speaker condition: the model directly answers a question containing a false presupposition; the correct behavior is to reject the presupposition. Listener condition: the model is provided with the question, the false presupposition, and an existing response, and must judge whether the response accepts the false presupposition (three-way classification: A/N/U), compared against human annotations.
    - **Design Motivation**: Rejecting a false presupposition requires detecting an implicit assumption and actively correcting it—generation is substantially harder than judgment, making this an ideal scenario for probing listener–speaker asymmetry.

2. **Antipresuppositions Task**:

    - **Function**: Tests whether models adhere to the Maximize Presupposition! principle.
    - **Mechanism**: A German "fruit story" paradigm is used—following a context-setting passage, the model must select a definite or indefinite article. Speaker condition: fill in the correct article/quantifier at the marked position. Listener condition: judge whether a given continuation is pragmatically appropriate. Even under this highly constrained generation setting (a single-word choice), many models still exhibit a significant listener advantage.
    - **Design Motivation**: This is the "simplest" possible generation task (single-word selection); if asymmetry persists here, the problem is fundamental.

3. **Deductive Reasoning Task**:

    - **Function**: Tests the consistency of logical reasoning between evaluation and generation.
    - **Mechanism**: Based on classic logical reasoning tasks, premises and a conclusion are provided. Speaker condition: fill in the missing color term that makes the conclusion valid. Listener condition: judge whether the given conclusion follows logically from the premises (True/False). Item-level analysis uses the conditional probability $\Delta_{cond} = P(\text{task}|l=1) - P(\text{task}|l=0)$ to measure whether correct judgment predicts successful generation.
    - **Design Motivation**: Deductive reasoning involves both pragmatic and logical competence, allowing the authors to examine whether asymmetry generalizes across different cognitive dimensions.

### Loss & Training

This is an evaluation study; no training is involved. Fourteen models are evaluated: LLaMA-3-8B, Qwen-3-8B/14B, Phi-4-14B, OLMo-2-7B/13B/32B, Mistral-7B, Mixtral-8x7B, M-Prometheus-14B, GPT-4o, GPT-4.1, GPT-5, and Claude Sonnet 4.5. Each model receives a total of 990 + 504 + 180 prompts.

## Key Experimental Results

### Main Results

**Listener–Speaker Accuracy Comparison (Representative Models)**

| Model | FP-Speaker | FP-Listener | AP-Speaker | AP-Listener | Reasoning-Speaker | Reasoning-Listener |
|-------|-----------|------------|-----------|------------|------------------|--------------------|
| Mistral-7B | ~2% | ~30% | ~50% | ~86% | ~20% | ~45% |
| LLaMA-8B | ~10% | ~35% | ~55% | ~65% | ~25% | ~73% |
| Qwen-3-14B | ~30% | ~75% | ~35% | ~91% | — | — |
| GPT-4o | ~85% | ~90% | ~80% | ~85% | ~75% | ~80% |
| GPT-5 | — | — | ~100% | ~86% | ~100% | ~100% |

### Ablation Study

| Model | Task | $P(\text{task}\|l=1)$ | $P(\text{task}\|l=0)$ | $\Delta_{cond}$ |
|-------|------|--------------------|--------------------|--------------------|
| GPT-4o | FP-Scenarios | 97.1% | 3.0% | **+94.1** |
| Mistral-7B | Antipresuppositions | 58.8% | 88.9% | **−30.0** |
| GPT-4o | Antipresuppositions | 64.4% | 100.0% | **−35.6** |
| Phi-4-14B | Reasoning | 100.0% | 5.1% | **+94.9** |
| LLaMA-8B | FP-Scenarios | 8.8% | 26.0% | **−17.2** |

### Key Findings

- Listener–speaker asymmetry is pervasive: most models achieve substantially higher accuracy as judges than as generators.
- The asymmetry is most severe in open-source small models (e.g., Mistral-7B on false presuppositions: speaker 2% vs. listener 30%).
- A counterintuitive pattern emerges in the antipresuppositions task: several models correctly judge violations but select the violating option themselves during generation (negative $\Delta_{cond}$).
- Larger models (e.g., GPT-5) show greater alignment between the two roles on some tasks, though alignment is still imperfect.
- Instruction-following failure rates vary substantially across models, limiting the reliability of LLM-as-a-judge.

## Highlights & Insights

- The core methodological strength is comparing the two roles on identical items, eliminating confounds introduced by different test sets.
- The negative $\Delta_{cond}$ in the antipresuppositions task is particularly striking: correct judgment not only fails to predict successful generation but may even be negatively correlated with it. This suggests that judgment and generation may rely on distinct internal representations or reasoning pathways.
- A practical warning for the LLM-as-a-judge paradigm: a model's ability to recognize what constitutes a good response does not imply that it can generate one, and vice versa.

## Limitations & Future Work

- Speaker data for the false presuppositions task are drawn from outputs reported in prior work rather than newly generated for this study, potentially introducing temporal and version-related confounds.
- Only German (false presuppositions, antipresuppositions) and English (reasoning) are used, limiting cross-lingual generalizability.
- Constrained output formats may not fully capture "naturalistic" pragmatic competence.
- The mechanisms underlying the asymmetry are not analyzed in depth—whether they stem from differences in attention patterns, internal representations, or decoding strategies remains an open question.
- Sample sizes are small for certain model–task combinations due to instruction-following failures reducing the number of valid responses.

## Related Work & Insights

- **vs. Hu & Levy (2023)**: Their work found that metalinguistic judgments may be dissociated from a model's internal representations; this paper extends that finding to the pragmatic domain and across multiple pragmatic phenomena.
- **vs. Piot et al. (2025)**: A similar judgment–generation dissociation was found in non-pragmatic domains (content moderation, safety); the present paper independently identifies the same pattern in pragmatics, suggesting this is a general property of LLMs.
- **vs. Qiu et al. (2025)**: That work evaluates comprehension and production in interactive games, but production ability is measured only indirectly via listener success rates; this paper directly assesses speaker generation quality.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Systematically comparing LLMs across two pragmatic roles is a novel and practically meaningful perspective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ 14 models × 3 tasks × item-level analysis provides comprehensive coverage.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Psycholinguistic background is clearly established; argumentation is logically rigorous.
- **Value**: ⭐⭐⭐⭐ Carries important methodological implications for the LLM-as-a-judge paradigm and for evaluation of linguistic competence more broadly.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] SpeakerSleuth: Can Large Audio-Language Models Judge Speaker Consistency across Multi-turn Dialogues?](speakersleuth_can_large_audio-language_models_judge_speaker_consistency_across_m.md)
- [\[ACL 2026\] HalluAudio: A Comprehensive Benchmark for Hallucination Detection in Large Audio-Language Models](halluaudio_a_comprehensive_benchmark_for_hallucination_detection_in_large_audio-.md)
- [\[ACL 2026\] Temporal Contrastive Decoding: A Training-Free Method for Large Audio-Language Models](temporal_contrastive_decoding_a_training-free_method_for_large_audio-language_mo.md)
- [\[ACL 2026\] Breaking Block Boundaries: Anchor-based History-stable Decoding for Diffusion Large Language Models](breaking_block_boundaries_anchor-based_history-stable_decoding_for_diffusion_lar.md)
- [\[ACL 2026\] Do We Need Distinct Representations for Every Speech Token? Unveiling and Exploiting Redundancy in Large Speech Language Models](do_we_need_distinct_representations_for_every_speech_token_unveiling_and_exploit.md)

<!-- RELATED:END -->
