---
title: >-
  [Paper Note] A Multi-Agent Framework for Feature-Constrained Difficulty Control in Reading Comprehension Item Generation
description: >-
  [ACL2026][Multi-Agent][Multi-agent generation] This paper proposes MAFIG, a framework that utilizes multi-agent collaboration, feature-level evaluators…
tags:
  - "ACL2026"
  - "Multi-Agent"
  - "Multi-agent generation"
  - "Reading comprehension item generation"
  - "Difficulty control"
  - "Constraint satisfaction"
  - "LLM evaluator"
date: 2026-05-08
content_hash: 226ca73293d9c57e
---

# A Multi-Agent Framework for Feature-Constrained Difficulty Control in Reading Comprehension Item Generation

**Conference**: ACL2026  
**arXiv**: [2605.19316](https://arxiv.org/abs/2605.19316)  
**Code**: https://github.com/SeonjeongHwang/mafig  
**Area**: LLM Agent / Educational Assessment / Controllable Generation  
**Keywords**: Multi-agent generation, Reading comprehension item generation, Difficulty control, Constraint satisfaction, LLM evaluator

## TL;DR
This paper proposes MAFIG, a framework that utilizes multi-agent collaboration, feature-level evaluators, and iterative revisions to generate multiple-choice reading comprehension questions. Compared to single-turn prompting, it significantly improves the success rate of satisfying constraints such as vocabulary, passage length, sentence length, reasoning complexity, factuality, and option neutrality, while providing a more stable increase in difficulty levels.

## Background & Motivation
**Background**: Reading comprehension (RC) items are central to language teaching, competency assessment, and computer-based testing systems. While LLMs can generate items with fluent language and complete structures in a zero-shot manner, existing controllable generation approaches generally follow two paths: one relies on psychometric models like IRT, which require learner response data to calibrate difficulty before training a generation model; the other directly uses prompts to control Bloom’s cognitive levels, vocabulary difficulty, or other interpretable features.

**Limitations of Prior Work**: IRT-style methods require large amounts of student response data, making them costly to scale across question types, languages, and exam scenarios. Prompting methods, while lightweight, typically rely on a single LLM to satisfy multiple constraints in one go, often leading to trade-offs. For example, the passage length might meet the requirement, but the vocabulary exceeds the specified CEFR level; or the options might be factually correct but contain entailment or contradiction relations among themselves, leading to unstable quality and difficulty control.

**Key Challenge**: Fine-grained difficulty control is inherently more complex than simply "telling the model to generate a Level 5 item." It requires the item to fall into target ranges across multiple interpretable features simultaneously. Abstract difficulty levels rely on internal model heuristics and lack verifiable constraints. Meanwhile, multi-dimensional feature constraints require the participation of external lexicons, rule-based tools, and semantic judgments, which are difficult to complete reliably in a single turn.

**Goal**: The authors aim to solve two sub-problems. First, how to generate RC multiple-choice items that strictly satisfy multi-dimensional feature constraints. Second, how to organize these feature constraints into a sequence of progressively increasing difficulty levels, ensuring a perceptible difficulty gap between adjacent levels.

**Key Insight**: The paper treats item generation as a constraint satisfaction problem rather than a one-time text generation task. The core observation is that if "generation, measurement, diagnosis, and revision" are decomposed into multiple roles, and each role iterates based on explicit feature feedback, the LLM does not need to hit all conditions at once. Instead, it can revise the item round-by-round like a human item generation expert.

**Core Idea**: Replace single-turn prompting with a multi-agent closed-loop revision and replace abstract difficulty labels with a difficulty-calibrated sequence of feature constraints. This grounds RC difficulty control in specific features that are checkable and iteratively optimizable.

## Method
MAFIG targets Multiple-Choice Factual Information (MCFI) reading comprehension items: given a source document and feature constraints corresponding to a target difficulty, the system generates a passage, a question stem, and several options. Items are generated in two stages: first, generating a passage that satisfies passage-level constraints, followed by generating options that satisfy option-level constraints based on that passage. Each stage executes a closed loop of "candidate generation → constraint evaluation → revision planning → local rewriting → re-evaluation" until all constraints are met or the maximum number of rounds is reached.

The paper uses six categories of feature variables: four categories to control cognitive load (vocabulary level, passage length, average sentence length, and reasoning complexity) and two categories to ensure item validity (factuality and option neutrality). Continuous features are discretized (e.g., passage length as short/medium/long). Vocabulary is mapped to CEFR levels (A, B, C), and reasoning complexity is categorized into single-sentence literal match, paraphrasing, single-sentence inference, multi-sentence inference, and insufficient information. This transforms difficulty from a black-box label into target states that can be checked by an evaluator.

### Overall Architecture
The input includes a source document and a set of feature constraints. The first stage, **Passage Generation**, handles constraints like vocabulary, passage length, and sentence length. The second stage, **Option Generation**, constructs options and processes constraints like factuality, vocabulary, reasoning complexity, and option neutrality. Both stages use the same loop: the **Evaluator** checks for violations; the **Planner** decides what to change based on reports and revision memory; the **Reworder** or **Editor** performs local revisions; finally, the **Refiner** applies minimal readability polishing.

In implementation, all MAFIG agents use Qwen3-32B (non-reasoning mode) with top-p=0.8 and temperature=0.7. Five initial candidates are sampled in parallel. Passage generation is capped at 20 rounds, and option generation at 100 rounds. If no candidate fully satisfies the constraints, a random candidate from the final pool is returned.

### Key Designs
1. **Feature-level Evaluator and Error Reporting**:
	- **Function**: Decomposes the item into measurable features, determines satisfaction of target constraints, and outputs specific error reports.
	- **Mechanism**: Surface features use rules or NLP tools (e.g., NLTK for length). Vocabulary level uses external CEFR word lists, using the highest-level word in the item as the check criterion. Semantic features like reasoning complexity, factuality, and neutrality are assessed using LLM judges with CoT and self-consistency.
	- **Design Motivation**: Difficulty control failures usually happen not because the model cannot write, but because it is unaware of which constraints were violated. Separating evaluation from generation provides actionable diagnostic signals.

2. **Planner + Reworder/Editor Role Division**:
	- **Function**: The Planner formulates revision strategies and selects executors; the Reworder handles vocabulary; the Editor handles all other constraints.
	- **Mechanism**: The Planner's input includes the current item state and repair history to avoid redundant edits. If a constraint fails for several rounds, "Creativity Enhancement Prompting" triggers more aggressive strategies like deleting and regenerating segments. The Reworder uses a RAG process to suggest contextually appropriate alternatives from word lists.
	- **Design Motivation**: Vocabulary constraints depend on external standards, while reasoning complexity requires semantic editing. Devolving errors to specific roles reduces the chaos of a single model attempting everything at once.

3. **Difficulty-aligned Feature Constraint Sequence**:
	- **Function**: Combines multiple features into 8 difficulty levels to ensure a monotonic increase theoretically and empirically.
	- **Mechanism**: The authors construct 16 candidate constraint sets and generate items for each using MAFIG. Adjacent candidate levels are compared in pairs by an LLM judge to calculate the Difficulty Alignment Score: $$DAS(Q_i,Q_j)=\frac{\sum_{n=1}^{N}x_f^{(n)}+\sum_{n=1}^{N}(-x_r^{(n)})}{2N}$$. Only pairs exceeding the threshold $\rho=0.4$ are retained, resulting in an 8-level monotonic sequence.
	- **Design Motivation**: Theoretical difficulty does not always guarantee perceived difficulty. Empirically validating the sequence ensures levels are distinguishable.

### Loss & Training
This work does not train a new model but proposes an inference-time multi-agent framework. There is no traditional supervised loss function. The optimization objective is embedded in the closed-loop: maximizing the probability that all target features are simultaneously satisfied, measured by Success Ratio (SR) and Achievement Ratio (AR). The difficulty sequence construction uses a filtering threshold of $\rho=0.4$ for pairwise LLM comparisons.

Key inference strategies: 1) Parallel generation of 5 candidates allows exploration of different revision paths. 2) Separation of passage and option stages prevents search space explosion. 3) The Planner maintains revision memory and triggers creative rewriting during persistent failures.

## Key Experimental Results

### Main Results
The experiment used 40 source documents from the NLTK Brown Corpus across 10 categories. 320 items were generated (40 documents × 8 levels). Baselines included abstract level-based control (Direct/Incremental Prompting) and feature-based Direct Prompting.

| Control Granularity | Method | SR(%) | AR(%) | DAS | Validity | Coherence | Fluency |
|--------------|------|-------|-------|-----|----------|-----------|---------|
| Level-based | Direct Qwen3-32B | - | - | 0.1037 | 2.6371 | 0.9355 | 0.9280 |
| Level-based | Direct GPT-5 | - | - | 0.2949 | 2.9816 | 0.9332 | 0.9408 |
| Level-based | Incremental Qwen3-32B | - | - | 0.1804 | 2.5605 | 0.9332 | 0.9408 |
| Level-based | Incremental GPT-5 | - | - | 0.2750 | 2.9637 | 0.9348 | 0.9309 |
| Feature-based | Direct Qwen3-32B | 0.00 | 59.10 | 0.2759 | 2.6094 | 0.9368 | 0.9393 |
| Feature-based | Direct GPT-5 | 2.50 | 77.81 | 0.4952 | 2.9105 | 0.9094 | 0.9241 |
| Feature-based | MAFIG Qwen3-32B | 92.29 | 99.32 | 0.5226 | 2.9242 | 0.9518 | 0.9429 |

**Key Findings**: Single-turn prompting struggles to satisfy all constraints simultaneously. Even with GPT-5, the SR was only 2.50% for feature-based direct prompting. MAFIG achieved an SR of 92.29% and AR of 99.32%, demonstrating that iterative revision successfully converts "partial instruction following" into "complete constraint satisfaction."

### Ablation Study
The ablation analysis shows the impact of various mechanisms on the convergence curve:
- **Planner’s Instruction**: Crucial for option generation to avoid inefficient edits.
- **Creativity Enhancement**: Necessary for resolving complex constraints like reasoning complexity.
- **Parallel Revision (n=5)**: All tested backbone models eventually reach 100% satisfaction with n=5; path diversity is more effective than repeated single-stream edits.

### Key Experimental Data (Human Evaluation)
Three experts with IELTS teaching or item generation experience evaluated 126 item pairs.

| Method | Human DAS | CAR(%) |
|------|-----------|--------|
| Direct Qwen3-32B | 0.2817 | 42.86 |
| Direct GPT-5 | 0.4722 | 57.14 |
| MAFIG Qwen3-32B | 0.6190 | 76.19 |

MAFIG's Human DAS (0.6190) and CAR (76.19%) are significantly higher than Direct GPT-5. This suggests that items generated by MAFIG are more likely to exhibit perceptible difficulty differences for human experts.

## Highlights & Insights
- Grounding difficulty control in "evaluable features" rather than "abstract labels" is the most critical design. It makes generation goals explicit and failures diagnostic.
- Multi-agent role division corresponds to the nature of different constraints: vocabulary needs retrieval, semantic constraints need judgment, and fluency needs polishing.
- The difficulty calibration process ensures the sequence is not just theoretically sound but empirically monotonic.
- LLM reasoning strength $\neq$ constraint satisfaction strength. Even powerful models like GPT-5 fail to hit multi-dimensional constraints in a single turn.

## Limitations & Future Work
- **Limitations**: The scope is limited to MCFI items. Absolute difficulty has not been validated with actual student performance data (IRT calibration). The computational cost is high due to the iterative nature (up to 100 rounds for options).
- **Future Work**: Future research could extend this to other item types (inference, main idea) and incorporate document topic and abstractness into the constraint sequence.

## Related Work & Insights
- **Comparison with IRT methods**: MAFIG does not require response data, making it lightweight and transferable, though it lacks real-world psychometric calibration.
- **Comparison with Bloom Taxonomy**: Abstract Bloom levels often show high variance internally; MAFIG's fine-grained features (sentence length, reasoning complexity) provide more direct control over cognitive load.
- **Insights**: The framework is highly applicable to any educational task requiring "verifiable generation," such as graded readers or automated formative assessments. The key is designing reliable evaluators and actionable revision steps for each control dimension.

## Rating
- **Novelty**: ⭐⭐⭐⭐☆
- **Experimental Thoroughness**: ⭐⭐⭐⭐☆
- **Writing Quality**: ⭐⭐⭐⭐☆
- **Value**: ⭐⭐⭐⭐☆

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Memory-Augmented LLM-based Multi-Agent System for Automated Feature Generation on Tabular Data](memory-augmented_llm-based_multi-agent_system_for_automated_feature_generation_o.md)
- [\[ACL 2026\] PosterForest: Hierarchical Multi-Agent Collaboration for Scientific Poster Generation](posterforest_hierarchical_multi-agent_collaboration_for_scientific_poster_genera.md)
- [\[ACL 2026\] RoadMapper: A Multi-Agent System for Roadmap Generation of Solving Complex Research Problems](roadmapper_a_multi-agent_system_for_roadmap_generation_of_solving_complex_resear.md)
- [\[AAAI 2026\] FinRpt: Dataset, Evaluation System and LLM-based Multi-agent Framework for Equity Research Report Generation](../../AAAI2026/multi_agent/finrpt_dataset_evaluation_system_and_llm-based_multi-agent_framework_for_equity_.md)
- [\[ACL 2026\] EvoSci: A Bio-Inspired Multi-Agent Framework for the Evolution of Scientific Discovery](evosci_a_bio-inspired_multi-agent_framework_for_the_evolution_of_scientific_disc.md)

</div>

<!-- RELATED:END -->
