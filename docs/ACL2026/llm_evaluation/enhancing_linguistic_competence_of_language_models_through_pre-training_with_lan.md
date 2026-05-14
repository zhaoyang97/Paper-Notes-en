---
title: >-
  [Paper Note] Enhancing Linguistic Competence of Language Models through Pre-training with Language Learning Tasks
description: >-
  [ACL 2026][LLM Evaluation][linguistic competence] L2T proposes a pre-training framework that mixes 14 language learning tasks spanning four linguistic granularities (character → discourse) with standard next-token predic…
tags:
  - "ACL 2026"
  - "LLM Evaluation"
  - "linguistic competence"
  - "pre-training"
  - "language learning tasks"
  - "language acquisition"
  - "structured stimuli"
date: 2026-05-08
content_hash: 50f637ccc968f115
---

# Enhancing Linguistic Competence of Language Models through Pre-training with Language Learning Tasks

**Conference**: ACL 2026
**arXiv**: [2601.03448](https://arxiv.org/abs/2601.03448)
**Code**: [https://github.com/gucci-j/l2t](https://github.com/gucci-j/l2t)
**Area**: LLM Evaluation
**Keywords**: linguistic competence, pre-training, language learning tasks, language acquisition, structured stimuli

## TL;DR

L2T proposes a pre-training framework that mixes 14 language learning tasks spanning four linguistic granularities (character → discourse) with standard next-token prediction. At the 500M/1B parameter scale, it improves BLiMP linguistic competence scores by 2–3 percentage points and accelerates their acquisition, while preserving general reasoning performance.

## Background & Motivation

**Background**: Language models pre-trained on raw text via causal language modeling (CLM) acquire world knowledge and reasoning capabilities, yet are not explicitly optimized for *linguistic competence*—the ability to understand morphological, syntactic, and semantic phenomena.

**Limitations of Prior Work**:
- LMs tend to behave as "stochastic parrots," mimicking surface patterns without mastering underlying linguistic structure.
- This resembles human rote learning: patterns are replicated without understanding the generative rules.
- Existing improvements typically rely on architectural modifications or complex curriculum designs, increasing engineering overhead.

**Key Challenge**: CLM is a single-objective task that prioritizes learning surface-level statistical features over linguistic structure; humans, by contrast, do not acquire language through a single objective but through multi-task learning.

**Goal**: Introduce structured language learning tasks during pre-training to enhance linguistic competence and accelerate its acquisition, without compromising general reasoning performance.

**Key Insight**: Inspired by human language acquisition—where learners engage in diverse tasks such as error correction, reordering, and completion—the framework automatically converts raw text into structured input–output pairs at multiple granularities, providing explicit linguistic structure stimuli during pre-training.

**Core Idea**: Pre-training should not be limited to sequence reconstruction (CLM); it should also include diverse language learning tasks that require "extracting and reorganizing information," forming structured scaffolding that promotes linguistic competence development.

## Method

### Overall Architecture

The L2T framework automatically converts raw text into structured input–output pairs for 14 language learning tasks across four linguistic granularity levels (character, word, sentence, and discourse). These are mixed with standard CLM data and used for training from scratch. Task generation is fully automatic and requires no external annotation resources, inducing structure directly from raw text.

### Key Designs

1. **Four-Level, 14-Category Language Learning Tasks**

    - **Function**: Provide multi-granularity structured linguistic stimuli during pre-training.
    - **Mechanism**:
      - *Character-level* (4 tasks): character counting, masked character reconstruction, space restoration, typo correction — targeting morphological awareness.
      - *Word-level* (5 tasks): final-word prediction, masked word reconstruction, random word correction, word reordering, token-type counting — breaking linear sequential dependencies and promoting structural inference.
      - *Sentence-level* (2 tasks): irrelevant sentence deletion, sentence reordering — requiring inter-sentence relational understanding.
      - *Discourse-level* (3 tasks): infilling, suffix completion, word-to-text generation — supporting global coherence and disambiguation.
    - **Design Motivation**: Analogous to how humans improve morphological awareness through error correction, acquire syntactic structure through reordering, and develop global coherence through completion tasks.

2. **Two Data Scenario Designs**

    - **Function**: Validate the robustness of L2T under different resource conditions.
    - **Mechanism**:
      - *Disjoint* (data-abundant): 100B tokens split into two halves — one for CLM, the other for generating L2T samples (~36B raw + ~64B L2T).
      - *Shared* (data-limited): 42B tokens used for both CLM and L2T sample generation, totaling 100B tokens.
    - **Design Motivation**: Disjoint tests the combined effect of data diversity and structured tasks; Shared tests the effect of structured stimuli vs. repeated exposure on the same source data (analogous to "multi-task learning vs. rote memorization").

3. **Structure Induction without External Supervision**

    - **Function**: Automatically generate training signals from raw text without human annotation.
    - **Mechanism**: Each task defines a deterministic or randomized transformation that converts a text segment into an $(x, y)$ pair, where $x$ is a perturbed or query input and $y$ is the recovered or analyzed output.
    - **Design Motivation**: Unlike instruction fine-tuning (which requires external supervision), L2T induces structure directly from raw text at low cost and with high scalability.

### Loss & Training

- Loss is computed over all tokens, including both the input and output portions of L2T tasks.
- Models use the Qwen2.5 architecture with the Mistral tokenizer (32K vocabulary), pre-trained from scratch at 500M and 1B scales.
- Total pre-training budget is fixed at 100B tokens, exceeding the Chinchilla-optimal threshold to evaluate the fully-trained regime.

## Key Experimental Results

### Main Results (Linguistic Competence — BLiMP)

| Scale | Data | Raw | L2T | Gain |
|-------|------|-----|-----|------|
| 500M | Disjoint | 78.6 | **80.2** | +1.6 |
| 500M | Shared | 78.1 | **80.9** | +2.8 |
| 1B | Disjoint | 79.0 | **80.8** | +1.8 |
| 1B | Shared | 78.9 | **81.2** | +2.3 |

### General Benchmarks

| Scale | Data | Raw avg | L2T avg | Difference |
|-------|------|---------|---------|------------|
| 500M | Disjoint | — | — | −0.87 (slight drop) |
| 1B | Disjoint | — | — | −0.07 (negligible) |
| 500M | Shared | — | — | +0.15 (slight gain) |
| 1B | Shared | — | — | −1.38 (drop, mainly on ARC) |

### Ablation Study (Per-Task Analysis)

| Task | Linguistic Competence | Notes |
|------|-----------------------|-------|
| 9 of 14 tasks | Surpasses Raw baseline | Char Count, Reordering, etc. provide critical structural scaffolding |
| Space, Masked Char | Below Raw baseline | Unstable training signal when used in isolation |
| Combined L2T | Surpasses most single tasks | Multi-task complementarity yields better robustness |

### Key Findings

- Island effects show the largest improvements (+6.9–11.3 points), suggesting that multi-granularity structured tasks aid in capturing long-distance dependencies.
- L2T models surpass the Raw baseline from as early as 5B training tokens, and the advantage persists throughout training — indicating accelerated linguistic competence acquisition.
- The Shared scenario yields larger gains (+2.3–2.8 vs. +1.6–1.8), demonstrating that structured stimuli are more effective than repeated exposure to the same data.
- L2T also improves broader cognitive capabilities, including fluid reasoning (RPM +5.4%) and numerical competence.

## Highlights & Insights

- The analogy of "language models as rote learners" is highly insightful; L2T addresses the surface-pattern learning problem of single-objective CLM through multi-task structured stimuli.
- Task design is theoretically grounded: each task category corresponds to strategies known to be effective in human language acquisition research (error correction → morphology; reordering → syntax; etc.).
- The approach requires no external annotation and no architectural modifications; purely data-level intervention makes it highly scalable.
- Even on identical source text (Shared scenario), multi-task transformations outperform repeated exposure — a finding with important implications for data efficiency research.

## Limitations & Future Work

- Validation is limited to 500M and 1B scales; the effect at 10B+ parameters remains unknown, and larger models may be more sensitive to the proportion of raw text.
- Results are based on single training runs, limiting statistical significance (though consistent improvements across two scales × two data scenarios provide indirect evidence).
- Task design focuses primarily at the sentence level and below; more complex discourse-level and cross-sentence tasks are lacking.
- General reasoning performance drops by 1.38 points in the 1B Shared scenario, suggesting that larger models require better balance between structural learning and knowledge consolidation.
- Evaluation is conducted only in English; multilingual generalization remains to be verified.

## Related Work & Insights

- **vs. Standard CLM Pre-training**: L2T improves linguistic competence by 2–3 points and accelerates acquisition, at the cost of a slight drop in general performance.
- **vs. Instruction Fine-tuning**: L2T introduces structured signals during pre-training and requires no external supervision data.
- **vs. Curriculum Learning / Architectural Modifications**: L2T is realized purely through data transformation, without modifying the model architecture or adopting complex training strategies.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The idea of mixing language learning tasks into pre-training is original, and the analogy to human language acquisition is substantive.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Multiple scales, multiple data scenarios, per-task ablations, and cognitive evaluations are included, though single-run results are a limitation.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — The logical chain from theoretical motivation to task design to experimental validation is exceptionally clear and complete.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Exploiting Vocabulary Frequency Imbalance in Language Model Pre-training](../../NeurIPS2025/llm_evaluation/exploiting_vocabulary_frequency_imbalance_in_language_model_pre-training.md)
- [\[ACL 2026\] Closing the Modality Reasoning Gap for Speech Large Language Models](closing_the_modality_reasoning_gap_for_speech_large_language_models.md)
- [\[ACL 2026\] AnchorMem: Anchored Facts with Associative Contexts for Building Memory in Large Language Models](anchormem_anchored_facts_with_associative_contexts_for_building_memory_in_large_.md)
- [\[ACL 2026\] E2EDev: Benchmarking Large Language Models in End-to-End Software Development Task](e2edev_benchmarking_large_language_models_in_end-to-end_software_development_tas.md)
- [\[ACL 2026\] ReTraceQA: Evaluating Reasoning Traces of Small Language Models in Commonsense Question Answering](retraceqa_evaluating_reasoning_traces_of_small_language_models_in_commonsense_qu.md)

</div>

<!-- RELATED:END -->
