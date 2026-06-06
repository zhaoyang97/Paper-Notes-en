---
title: >-
  [Paper Note] Enhancing Linguistic Competence of Language Models through Pre-training with Language Learning Tasks
description: >-
  [ACL 2026][LLM Evaluation][Linguistic competence] L2T proposes a pre-training framework that mixes 14 language learning tasks (character-level to discourse-level) with standard next-token prediction. It improves BLiMP li…
tags:
  - "ACL 2026"
  - "LLM Evaluation"
  - "Linguistic competence"
  - "pre-training"
  - "language learning tasks"
  - "language acquisition"
  - "structured stimuli"
date: 2026-05-08
content_hash: 6ef13873760ab433
---

# Enhancing Linguistic Competence of Language Models through Pre-training with Language Learning Tasks

**Conference**: ACL 2026  
**arXiv**: [2601.03448](https://arxiv.org/abs/2601.03448)  
**Code**: [https://github.com/gucci-j/l2t](https://github.com/gucci-j/l2t)  
**Area**: LLM Evaluation  
**Keywords**: Linguistic competence, pre-training, language learning tasks, language acquisition, structured stimuli

## TL;DR

L2T proposes a pre-training framework that mixes 14 language learning tasks (character-level to discourse-level) with standard next-token prediction. It improves BLiMP linguistic competence scores by 2-3 percentage points at 500M/1B scales and accelerates the acquisition process while maintaining general reasoning performance.

## Background & Motivation

**Background**: Language models are pre-trained on raw text via causal language modeling (CLM), allowing them to learn world knowledge and reasoning capabilities. However, they are not explicitly optimized for linguistic competence—the ability to understand morphological, syntactic, and semantic phenomena.

**Limitations of Prior Work**:
- LMs often act as "stochastic parrots," mimicking surface patterns without mastering underlying linguistic structures.
- This is analogous to human rote learning, where patterns are copied without understanding generative rules.
- Existing improvement methods typically rely on architectural modifications or complex curriculum designs, increasing engineering complexity.

**Key Challenge**: CLM is a single-task objective that prioritizes learning surface statistical features over structural understanding. Conversely, humans do not acquire language through a single objective but through multi-task learning.

**Goal**: To introduce structured language learning tasks during the pre-training phase to enhance the model's linguistic competence and accelerate its acquisition process without harming general reasoning performance.

**Key Insight**: Inspired by human language acquisition—where humans learn through error correction, reorganization, and completion—the authors automatically convert raw text into multi-granular structured input-output pairs to provide explicit linguistic structural stimuli during pre-training.

**Core Idea**: Pre-training should not only involve sequence reconstruction (CLM) but also diverse language learning tasks that require "extracting and reorganizing information," forming a structured scaffold to facilitate the development of linguistic competence.

## Method

### Overall Architecture

The L2T framework automatically converts raw text into structured input-output pairs for 14 language learning tasks across four linguistic granularity levels (character, word, sentence, and discourse). These are mixed with standard CLM data for pre-training from scratch. Task generation is entirely automatic and does not requires external annotation resources, inducing structure directly from the raw text.

### Key Designs

1.  **14 Language Learning Tasks Across Four Levels**:
    - **Function**: Provide multi-granular structured linguistic stimuli during pre-training.
    - **Mechanism**:
        - Character-level (4 types): Character counting, masked character reconstruction, whitespace recovery, typo correction—aiming to enhance morphological awareness.
        - Word-level (5 types): Final word prediction, masked word reconstruction, random word correction, word reordering, token type counting—breaking linear sequence dependency and promoting structural inference.
        - Sentence-level (2 types): Irrelevant sentence deletion, sentence reordering—requiring understanding of inter-sentence relationships.
        - Discourse-level (3 types): Infilling, second-half completion, word-to-text generation—supporting global coherence and disambiguation.
    - **Design Motivation**: Analogous to how humans improve morphological awareness through error correction, learn syntactic structures through reordering, and learn global coherence through completion tasks.

2.  **Two Data Scenario Designs**:
    - **Function**: Verify the robustness of L2T under different resource conditions.
    - **Mechanism**:
        - Disjoint (Sufficient data): 100B tokens are split into halves: one half for CLM and the other for generating L2T samples (~36B raw + ~64B L2T).
        - Shared (Data-constrained): 42B tokens are used for both CLM and generating L2T samples, totaling 100B tokens.
    - **Design Motivation**: Disjoint tests the effect of data diversity combined with structured tasks; Shared tests structured stimuli vs. repeated exposure under identical source data (analogy of "multi-task learning vs. rote learning").

3.  **Structure Induction without External Supervision**:
    - **Function**: Automatically generate training signals from raw text without human labeling.
    - **Mechanism**: Each task defines a deterministic or randomized transformation that automatically converts text segments into $(x, y)$ pairs, where $x$ is the perturbed/query input and $y$ is the recovery/analytical output.
    - **Design Motivation**: Unlike instruction tuning (which requires external supervision), L2T induces structure directly from raw text, making it low-cost and scalable.

### Loss & Training

- Loss is calculated on all tokens (including both input and output parts of the L2T tasks).
- Uses Qwen2.5 architecture + Mistral tokenizer (32K vocabulary) to pre-train 500M and 1B models from scratch.
- The total pre-training budget is fixed at 100B tokens, exceeding the Chinchilla optimal threshold to evaluate scenarios of full training.

## Key Experimental Results

### Main Results (Linguistic Competence - BLiMP)

| Scale | Data | Raw | L2T | Gain |
|-------|------|-----|-----|------|
| 500M | Disjoint | 78.6 | **80.2** | +1.6 |
| 500M | Shared | 78.1 | **80.9** | +2.8 |
| 1B | Disjoint | 79.0 | **80.8** | +1.8 |
| 1B | Shared | 78.9 | **81.2** | +2.3 |

### General Benchmarks

| Scale | Data | Raw avg | L2T avg | Gain |
|-------|------|---------|---------|------|
| 500M | Disjoint | - | - | -0.87 (Slight decrease) |
| 1B | Disjoint | - | - | -0.07 (Almost no difference) |
| 500M | Shared | - | - | +0.15 (Slight increase) |
| 1B | Shared | - | - | -1.38 (Decrease, mainly in ARC) |

### Ablation Study (Single Task Analysis)

| Task | Linguistic Competence | Description |
|------|-----------------------|-------------|
| 9/14 Tasks | Surpass Raw baseline | Char Count, Reordering, etc., provide key structural scaffolding |
| Space, Masked Char | Below Raw baseline | Training signals are unstable when used individually |
| Combined L2T | Surpass most single tasks | Multi-task complementarity leads to better robustness |

### Key Findings
- Improvement in Island effects is most significant (+6.9 to 11.3 points), indicating that multi-granular structured tasks help capture long-distance dependencies.
- L2T models surpass the Raw baseline from 5B tokens onwards and maintain this advantage—accelerating the acquisition of linguistic competence.
- Effects are more pronounced in the Shared scenario (+2.3~2.8 vs +1.6~1.8), suggesting "structured stimuli" are more effective than "repeated exposure."
- L2T also enhances broader cognitive intelligence, such as fluid reasoning (RPM +5.4%) and numerical abilities.

## Highlights & Insights
- The analogy of "Language Models = Rote Learning" is insightful; L2T uses multi-task structured stimuli to solve the surface pattern learning issue of single-task CLM.
- Task design is theoretically supported: each task category corresponds to strategies known to be effective in human language acquisition research (e.g., error correction $\rightarrow$ morphology, reordering $\rightarrow$ syntax).
- Requiring no external labeling or architectural modifications, the pure data-layer intervention makes the method highly scalable.
- Even on the same source text (Shared), multi-task transformation is more effective than repeated exposure, which is an important insight for data efficiency research.

## Limitations & Future Work
- Only validated at 500M and 1B scales; effectiveness at 10B+ scales is unknown, as larger models may be more sensitive to the ratio of raw text.
- Based on single training runs, statistical significance is limited (though consistency across two scales × two data scenarios provides indirect evidence).
- Task design focuses on the sentence level and below, lacking more complex discourse-level and cross-sentence tasks.
- In the 1B Shared scenario, general reasoning dropped by 1.38 points; larger models need a better balance between structural learning and knowledge consolidation.
- Only evaluated in English; multilingual generalization remains to be verified.

## Related Work & Insights
- **Ours vs. Standard CLM Pre-training**: L2T improves linguistic competence by 2-3 points and accelerates acquisition at the cost of a slight drop in general performance.
- **Ours vs. Instruction Tuning**: L2T introduces structural signals during the pre-training phase and does not require external supervision data.
- **Ours vs. Curriculum Learning/Architecture Modification**: L2T is implemented solely through data transformation without needing to modify the model architecture or use complex training strategies.

## Rating
- Novelty: ⭐⭐⭐⭐ The idea of mixing language learning tasks in pre-training is unique, and the analogy to human language acquisition is profound.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple scales, multiple data scenarios, single-task analysis, and cognitive evaluations, though the single run is a drawback.
- Writing Quality: ⭐⭐⭐⭐⭐ The logical chain from theoretical motivation to task design to experimental verification is very complete and clear.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Teaching Language Models to Forecast Research Success Through Comparative Idea Evaluation](teaching_language_models_to_forecast_research_success_through_comparative_idea_e.md)
- [\[ACL 2026\] How Hypocritical Is Your LLM Judge? Listener–Speaker Asymmetries in the Pragmatic Competence of Large Language Models](how_hypocritical_is_your_llm_judge_listener-speaker_asymmetries_in_the_pragmatic.md)
- [\[NeurIPS 2025\] Exploiting Vocabulary Frequency Imbalance in Language Model Pre-training](../../NeurIPS2025/llm_evaluation/exploiting_vocabulary_frequency_imbalance_in_language_model_pre-training.md)
- [\[ACL 2026\] Evaluating Temporal Consistency in Multi-Turn Language Models](evaluating_temporal_consistency_in_multi-turn_language_models.md)
- [\[ACL 2026\] CUB: Benchmarking Context Utilisation Techniques for Language Models](cub_benchmarking_context_utilisation_techniques_for_language_models.md)

</div>

<!-- RELATED:END -->
