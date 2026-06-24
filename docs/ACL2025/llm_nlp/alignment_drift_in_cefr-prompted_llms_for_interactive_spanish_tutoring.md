---
title: >-
  [Paper Note] Alignment Drift in CEFR-prompted LLMs for Interactive Spanish Tutoring
description: >-
  [ACL 2025][LLM (Other)][alignment drift] Through experiments simulating teacher-student conversations using LLMs, it was discovered that while system prompting based on CEFR levels can initially constrain the difficulty of the generated Spanish text, this constraint gradually decays as the conversation rounds progress. The authors term this phenomenon "alignment drift," indicating that prompt engineering alone is insufficient for sustaining long-term adaptive language educati…
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "alignment drift"
  - "CEFR"
  - "language pedagogy"
  - "system prompting"
  - "conversation simulation"
date: 2026-05-08
content_hash: fd923bcb2d041647
---

# Alignment Drift in CEFR-prompted LLMs for Interactive Spanish Tutoring

**Conference**: ACL 2025  
**arXiv**: [2505.08351](https://arxiv.org/abs/2505.08351)  
**Code**: [https://github.com/INTERACT-LLM/alignment-drift-llms](https://github.com/INTERACT-LLM/alignment-drift-llms)  
**Area**: LLM Applications / Educational NLP  
**Keywords**: alignment drift, CEFR, language pedagogy, system prompting, conversation simulation

## TL;DR
Through experiments simulating teacher-student conversations using LLMs, it was discovered that while system prompting based on CEFR levels can initially constrain the difficulty of the generated Spanish text, this constraint gradually decays as the conversation rounds progress. The authors term this phenomenon "alignment drift," indicating that prompt engineering alone is insufficient for sustaining long-term adaptive language educational systems.

## Background & Motivation
**Background**: LLMs are widely explored as interactive language learning tutors, particularly providing practice opportunities for L2 learners who lack access to target language communities. Existing studies primarily focus on English learning scenarios.

**Limitations of Prior Work**: Current language tutoring with LLMs heavily relies on general-purpose tools (e.g., ChatGPT), requiring learners to master prompt engineering themselves to obtain appropriately leveled outputs. No systematic research has yet demonstrated whether system prompting alone can reliably constrain LLM output to a specific language proficiency level.

**Key Challenge**: The Common European Framework of Reference for Languages (CEFR) provides explicit definitions of language proficiency levels (A1-C2). However, whether LLMs truly "understand" these level definitions, can execute them consistently, and whether they drift back to unconstrained behaviors over multi-turn conversations remains a critical challenge.

**Goal**: To systematically evaluate the effectiveness and persistence of CEFR-based system prompting in constraining the difficulty of LLM outputs across multi-turn Spanish tutoring dialogues.

**Key Insight**: Utilizing the LLM to simultaneously act as the "teacher" and "student" (managing independent chat histories) to simulate full multi-turn conversations, bypassing the high cost of human subject experiments and offering a scalable, low-cost evaluation method.

**Core Idea**: The constraining effect of CEFR prompting on LLMs decays as conversation turns increase (alignment drift). Prompt engineering alone is insufficient for building reliable, adaptive language tutoring systems.

## Method

### Overall Architecture
- **Input**: System prompts corresponding to three CEFR levels (A1/B1/C1) + a fixed initial message "Hola"
- **Process**: A single LLM instance maintaining two separate chat logs respectively acting as the teacher and the student, alternately generating a 9-turn dialogue.
- **Evaluation**: Extracting 6 metrics from the teacher LLM's outputs to measure whether the textual difficulty matches the target proficiency level.
- **Models**: Four 7B-12B open-source instruction-tuned models (Llama-3.1-8B, Gemma-3-12B, Mistral-7B, Qwen-2.5-7B).
- **Scale**: Each model × 3 levels × 30 simulations = 90 dialogues per model, totaling 360 dialogues.

### Key Designs

1. **Dialogue Simulation Method**:

    - **Function**: Utilizing the same LLM instance to simultaneously simulate the teacher and the student, alternating dialogue by switching active chat logs.
    - **Mechanism**: LLM architectures are stateless, processing the entire chat log during each generation. Thus, the system is implemented by maintaining two independent lists of chat histories and feeding them alternately.
    - **Design Motivation**: Avoiding the high cost of human participation to offer a reproducible and scalable evaluation methodology. Fixing the initial message to "Hola" eliminates variability on the student side.

2. **CEFR System Prompt Design**:

    - **Function**: Designing system prompts containing CEFR level descriptions for the teacher LLM, with only the level-related keywords modified.
    - **Mechanism**: The prompts include level keywords ("beginner/intermediate/advanced") alongside official descriptions from the CEFR Global Scale (summarizing the learner's competence in 3-4 sentences).
    - **Design Motivation**: Leveraging the standardized definitions of CEFR to ensure experimental replicability and provide objective benchmarks for text difficulty.

3. **Multidimensional Text Difficulty Evaluation**:

    - **Function**: Extracting 6 metrics to cover readability, structural complexity, and semantic naturalness.
    - **Traditional Readability** (3 metrics): Fernández Huerta, Szigriszt-Pazos (Spanish adaptation of Flesch), and Gutiérrez de Polini, which are computed based on syllables, characters, and sentence lengths.
    - **Structural Complexity** (2 metrics): Mean Dependency Distance (MDD) to evaluate syntactic complexity; text length (in tokens).
    - **LLM Surprisal** (1 metric): Utilizing EuroBERT to compute sentence-level surprisal scores, where lower scores imply higher "naturalness" (aligning closer to high-level texts).
    - **Design Motivation**: A single metric is insufficient to capture multidimensional text difficulty, so a comprehensive evaluation framework provides greater reliability.

### Statistical Analysis
For each metric per model, a linear mixed-effects model is fitted: $\text{metric}_{\text{model}} \sim \text{level} + (1|\text{chat}_{\text{id}})$, using A1 as the baseline to compare the statistical significance of B1 and C1. Bonferroni correction is applied for multiple comparisons.

## Key Experimental Results

### Main Results

| Metric | Model | A1 vs B1 Difference | A1 vs C1 Difference | p-value |
|------|------|-------------|-------------|------|
| Fernández Huerta | All models | β: -4 to -9 | β: -12 to -17 | p<0.001 |
| Text Length | All models | Significant increase | Significant increase | p<0.001 |
| MDD | Most models | Significant increase | Significant increase | Mostly p<0.001 |
| Surprisal | Except Qwen | Significant decrease | Significant decrease | p<0.001 |

### Ablation Study

| Phenomenon | Observation | Description |
|------|------|------|
| Alignment Drift | All models, all metrics | Metric values of different proficiency levels converge as conversation turns increase |
| A1 vs C1 Discriminability | Good | The readability gap peaks at ~17 Fernández Huerta points |
| B1 vs C1 Discriminability | Poor | Heavy distribution overlaps, particularly severe for Qwen |
| Cross-model Variance | Llama is the most stable | Gemma/Mistral exhibit higher fluctuations; Qwen shows no significant difference in surprisal |
| Language Shifting | Gemma/Llama -> English, Qwen -> Chinese | Particularly prominent at the A1 level |

### Key Findings
- **Alignment drift is a pervasive phenomenon**: All 4 models across all 6 metrics show a decay in constraints over time.
- The effect of CEFR prompting is most robust in the initial turn, but by turn 9, the variations across different levels narrow significantly.
- Differentiation between A1 and C1 is relatively successful, but B1 and C1 frequently overlap.
- Readability metrics differentiate levels best, whereas syntactic complexity and surprisal exhibit weaker discriminative power.
- Even in terms of readability metrics, the average score for C1 (~70 Fernández Huerta) corresponds merely to the level of Spanish primary school students, indicating that the models might not be generating true C1-level text.

## Highlights & Insights
- **Coining the concept of "Alignment Drift"**: Systematically names and quantifies the phenomenon where LLMs progressively deviate from system prompt constraints in multi-turn dialogues for the first time. This finding offers caution for all applications relying on system prompts to preserve long-term behavioral consistency, beyond just language pedagogy.
- **Low-cost LLM-LLM conversation simulation evaluation**: Simulating teachers and students using LLMs simultaneously bypasses human labor costs, presenting a scalable valuation approach adaptable to other dialogue system evaluation scenarios.
- **Multidimensional metric system**: The comprehensive evaluation framework combining traditional readability, syntactic complexity, and neural language model surprisal is pioneering in the L2 Spanish pedagogical research area.

## Limitations & Future Work
- Only one set of system prompts was tested; the optimization space for prompts (e.g., using Spanish-written prompts, more detailed CEFR descriptions) remains unexplored.
- The student LLM was unoptimized, which may affect the teacher LLM's drift behavior; specifically, the quality of student responses retroactively influences the teacher's outputs.
- Traditional readability metrics (from the Flesch family) are designed for long documents, and their applicability to short dialogue messages is questionable.
- Only 7B-12B models were evaluated; whether larger models (e.g., 70B+) perform better remains unknown.
- Lacks comparison with constraints using fine-tuning or decoding strategy-guided methods.
- Future work can explore incorporating a CEFR classifier for rejection sampling, or constraining output complexity during the decoding stage.

## Related Work & Insights
- **vs Tyen et al. (2022, 2024)**: They leverage classifiers for rejection sampling to constrain output difficulty, whereas Ours relies solely on prompting and reveals its unreliability; combining the two might prove to be a superior approach.
- **vs Malik et al. (2024)**: They observed that integrating CEFR details enhances the level alignment of GPT-4. In contrast, this paper finds that constraints still drift despite detailed CEFR descriptions. This indicates that the core issue lies not entirely in the prompt content but in the memory decay over multi-turn dialogues.
- **vs Qiu & Yang (2024)**: They observed that LLMs struggle to adhere to system prompts in multi-turn dialogues in other domains, verifying the pervasiveness of alignment drift.

## Rating
- Novelty: ⭐⭐⭐⭐ Coins the concept of alignment drift, features a novel experimental design (LLM-LLM simulation), and targets a highly practical research problem.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation using 4 models, 360 dialogues, 6 metrics, and linear mixed-effects statistical analysis, but lacks a comparative baseline with fine-tuning.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear structure, in-depth analysis, and an honest, comprehensive discussion of limitations.
- Value: ⭐⭐⭐⭐ The discovery of alignment drift holds broad significance for downstream LLM applications, extending well beyond language teaching.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Nudging: Inference-time Alignment of LLMs via Guided Decoding](nudging_inference_time_alignment.md)
- [\[ACL 2025\] Towards Enhanced Immersion and Agency for LLM-based Interactive Drama](towards_enhanced_immersion_and_agency_for_llm-based_interactive_drama.md)
- [\[ACL 2025\] Unlocking Recursive Thinking of LLMs: Alignment via Refinement](unlocking_recursive_thinking_of_llms_alignment_via_refinement.md)
- [\[ACL 2025\] Psycholinguistic Word Features: A New Approach for the Evaluation of LLMs Alignment with Humans](psycholinguistic_word_features_a_new_approach_for_the_evaluation_of_llms_alignme.md)
- [\[ACL 2025\] Interactive and Expressive Code-Augmented Planning with Large Language Models](interactive_and_expressive_code-augmented_planning_with_large_language_models.md)

</div>

<!-- RELATED:END -->
