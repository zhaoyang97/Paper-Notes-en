---
title: >-
  [Paper Note] HSKBenchmark: Modeling and Benchmarking Chinese Second Language Acquisition in Large Language Models through Curriculum Tuning
description: >-
  [AAAI 2026][Interpretability][Chinese second language acquisition] This paper introduces HSKBenchmark, the first benchmark for staged modeling and writing assessment of Chinese second language acquisition (SLA) in LLMs. It comprises HSK levels 3–6 textbooks (6.76M tokens), 16K synthetic instruction data, 30 test prompts, a linguistically-grounded evaluation system, and a curriculum tuning framework designed to simulate human acquisition trajectories.
tags:
  - "AAAI 2026"
  - "Interpretability"
  - "Chinese second language acquisition"
  - "curriculum tuning"
  - "HSK benchmark"
  - "writing assessment"
  - "linguistic complexity"
date: 2026-05-08
content_hash: 1095d785484e4eef
---

# HSKBenchmark: Modeling and Benchmarking Chinese Second Language Acquisition in Large Language Models through Curriculum Tuning

**Conference**: AAAI 2026
**arXiv**: [2511.15574](https://arxiv.org/abs/2511.15574)  
**Code**: [GitHub](https://github.com/CharlesYang030/HSKB)  
**Area**: Language Acquisition Modeling / LLM Evaluation
**Keywords**: Chinese second language acquisition, curriculum tuning, HSK benchmark, writing assessment, linguistic complexity

## TL;DR

This paper introduces HSKBenchmark, the first benchmark for staged modeling and writing assessment of Chinese second language acquisition (SLA) in LLMs. It comprises HSK levels 3–6 textbooks (6.76M tokens), 16K synthetic instruction data, 30 test prompts, a linguistically-grounded evaluation system, and a curriculum tuning framework designed to simulate human acquisition trajectories.

## Background & Motivation

**Background**: Language acquisition research is central to understanding human linguistic intelligence. LLMs, owing to their controllability and reproducibility, have emerged as viable tools for simulating language acquisition. Existing work has focused primarily on first language (L1) acquisition modeling, while second language acquisition (SLA) modeling — particularly for Chinese — remains nascent.

**Limitations of Prior Work**:
- Existing SLA modeling differentiates acquisition stages solely by controlling training data volume (e.g., one stage per 200K tokens), without accounting for graded linguistic difficulty.
- Systematic graded training data and evaluation frameworks are absent.
- Existing multilingual benchmarks (e.g., MMLU) assess static model capabilities rather than dynamic acquisition development.

**Key Challenge**: Controlling linguistic input in human learner experiments is ethically and practically infeasible, making LLMs an appealing controlled alternative. However, no systematic benchmark exists to support staged SLA modeling and evaluation in LLMs.

**Goal**: To provide a reusable, level-structured training-and-evaluation benchmark for Chinese SLA modeling in LLMs.

**Key Insight**: The HSK (Hanyu Shuiping Kaoshi) proficiency framework serves as the organizing structure, enabling multi-dimensional benchmark construction across textbook grading, grammar-item grading, and writing assessment. Writing is selected as the primary evaluation lens, as it most directly reflects the development of productive language ability.

**Core Idea**: Construct graded training data (textbook pre-training + grammar instruction tuning), design a curriculum tuning framework for progressive LLM training, establish a five-dimensional evaluation system covering grammar coverage, error count, lexical complexity, syntactic complexity, and holistic score, and train HSKAgent for automated evaluation.

## Method

### Overall Architecture

HSKBenchmark comprises four major components:
1. **Graded Training Data**: HSK 3–6 textbooks (6.76M tokens) + 16K synthetic instruction data derived from 591 grammar items.
2. **Curriculum Tuning Framework**: Progressive pre-training (textbooks) followed by progressive instruction tuning (writing exercises), advancing from HSK 3 to HSK 6.
3. **Linguistic Evaluation System**: Five dimensions — grammar item coverage rate, writing errors, lexical complexity (MATTR-50), syntactic complexity (MDD), and holistic score.
4. **HSKAgent**: An automated evaluation model fine-tuned on 10K compositions written by human L2 learners.

### Key Designs

**Module 1: Graded Training Data Construction**

- **Function**: Collect 79 mainstream international Chinese education textbooks (partitioned by HSK levels 3–6), clean image/pinyin/English auxiliary content; integrate 591 grammar items from the *Chinese Proficiency Grading Standards for International Chinese Language Education* (covering vocabulary, phrases, fixed expressions, sentence constituents, sentence patterns, and emphatic usages); generate graded instruction data using GPT-4.1-mini, DeepSeek-V3, and Gemini-2.5-Flash.
- **Mechanism**:
    - Textbooks are naturally stratified by HSK level, with token counts ranging from 895K at HSK 3 to 2.68M at HSK 6.
    - Each grammar item yields 10 instruction–input–output triples, validated by three graduate annotators (Fleiss's Kappa = 0.91; acceptance rate = 95%).
    - The final synthetic dataset contains 16,462 instruction samples.
- **Design Motivation**: Krashen's Input Hypothesis posits that language acquisition requires comprehensible, incrementally challenging input ($i+1$). Level-stratified textbooks naturally satisfy this requirement, constituting a principled alternative to the coarse practice of partitioning data purely by volume.

**Module 2: Curriculum Tuning**

- **Function**: LLMs are trained progressively through HSK levels 3 → 4 → 5 → 6; at each level, pre-training on textbooks (simulating self-study) precedes instruction tuning on grammar data (simulating writing practice).
- **Mechanism**:
    - Pre-training: standard next-token prediction loss: $\mathcal{L}_{PT}^{(l)} = -\sum_i \sum_t \log P_{\theta}(x_{i,t}|x_{i<t})$
    - Instruction tuning: instruction-following loss conditioned on writing prompts: $\mathcal{L}_{IT}^{(l)} = -\sum_i \sum_t \log P_{\theta_{PT}^{(l)}}(y_{i,t}|p_i, y_{i<t})$
    - Progressive update: $\theta_{PT}^{(l)} = \text{Pretraining}(\theta^{(l-1)}, \mathcal{T}^{(l)})$, $\theta_{IT}^{(l)} = \text{InstructionTuning}(\theta_{PT}^{(l)}, \mathcal{D}^{(l)})$
- **Design Motivation**: This design emulates the human learner's gradual progression from elementary to advanced proficiency, ensuring that at each stage the model acquires linguistic competencies commensurate with the target level, rather than being exposed to data of all difficulty levels simultaneously.

**Module 3: Linguistic Evaluation System + HSKAgent**

- **Function**: Evaluate LLM writing outputs across five dimensions; fine-tune HSKAgent on 10K human L2 compositions to enable automated assessment.
- **Mechanism**:
    - **Grammar Item Coverage Rate**: Measures the distributional usage of grammar items at each HSK level; advanced models are expected to exhibit higher coverage of high-level items.
    - **Writing Errors (Err)**: Detects grammatical, lexical, and collocational errors.
    - **Lexical Complexity (MATTR-50)**: Moving-average type-token ratio over a window of 50 tokens; higher values indicate greater lexical diversity.
    - **Syntactic Complexity (MDD)**: Mean dependency distance, reflecting structural elaboration of sentences.
    - **Holistic Score**: An integrated score referencing HSK examination rubrics.
- **Design Motivation**: Writing evaluation must be multidimensional and automated for large-scale deployment. Existing tools (CTAP, L2C-Rater) either lack automatic scoring or automatic error detection; HSKAgent addresses this gap.

### Loss & Training

- Both pre-training and instruction tuning adopt standard autoregressive language modeling losses.
- The curriculum strictly follows the ascending HSK level order (3 → 4 → 5 → 6).
- HSKAgent is domain-fine-tuned on the writing assessment dataset.

## Key Experimental Results

### Main Results

Comparison of human learners and LLMs on 30 HSK writing prompts (selected results):

| Model / Human | HSK3 Grammar Coverage | HSK6 Grammar Coverage | Errors | MATTR-50 | MDD | Score |
|---|---|---|---|---|---|---|
| Native Speaker | 0.341 | 0.126 | 1.40 | 0.806 | 2.98 | 88.3 |
| Learner (95-pt) | 0.356 | 0.139 | 2.87 | 0.817 | 2.84 | 85.0 |
| Learner (80-pt) | 0.386 | 0.133 | 3.50 | 0.793 | 2.65 | 74.8 |

(Fine-tuned LLMs achieve writing performance comparable to advanced human L2 learners.)

### Ablation Study

- Curriculum tuning vs. full one-pass training: curriculum tuning yields superior grammar coverage across levels and higher holistic scores.
- Importance of the pre-training stage: omitting pre-training and proceeding directly to instruction tuning leads to significant degradation in lexical and syntactic complexity.
- Multi-source instruction data (GPT + DeepSeek + Gemini) produces higher quality than data from any single source.

### Key Findings

- Fine-tuned LLMs exhibit acquisition patterns analogous to human learners: as training level advances, usage of higher-level grammar items increases, error counts decrease, and syntactic complexity rises.
- Curriculum tuning more faithfully captures staged developmental trajectories than non-curriculum approaches.
- Modeling Chinese SLA — a typologically isolated language markedly distinct from English — validates the cross-typological generalization capacity of LLMs.

## Highlights & Insights

- **First Chinese SLA modeling benchmark**: fills a critical gap in the field; the HSK proficiency system provides a natural, principled difficulty stratification framework.
- The curriculum tuning design operationalizes Krashen's Input Hypothesis, elegantly bridging language acquisition theory with LLM training paradigms.
- HSKAgent carries independent practical value — automated assessment of Chinese L2 writing has long been an open challenge, and an agent fine-tuned on 10K human compositions constitutes a standalone contribution.
- The five-dimensional evaluation system covers the core constructs of interest in SLA research.

## Limitations & Future Work

- HSK levels 1–2 are not covered due to insufficient textbook resources and difficulties in aligning multiple-choice items, limiting complete simulation of the elementary acquisition stage.
- The textbook corpus (6.76M tokens) is modest relative to typical LLM pre-training scale, constraining the upper bound of pre-training effectiveness.
- Evaluation is restricted to the writing modality; HSK listening, speaking, and reading abilities are not assessed.
- Cross-linguistic transfer effects (e.g., transfer from English L1 to Chinese L2) are not explored.

## Related Work & Insights

- **BabyLM**: A shared task that established the evaluation framework for child L1 acquisition modeling in LLMs; serves as the L1 counterpart to the present work.
- **Krashen's Input Hypothesis ($i+1$)**: The theoretical foundation for curriculum tuning.
- **CTAP for Chinese**: An automated extraction tool covering 196 linguistic complexity indices but lacking holistic scoring capability.
- Oba et al. 2023: A study of L1-to-L2 transfer on XLM, demonstrating that typological distance modulates transfer effectiveness.

## Rating

⭐⭐⭐⭐ (4/5)

**Rationale**: The problem formulation is distinctively valuable (Chinese SLA + LLMs); the benchmark construction is systematic and complete (data–training–evaluation); the curriculum tuning framework is theoretically grounded in language acquisition research. One point is deducted for the limited training data scale, the restriction to the writing modality, and the absence of lower HSK levels.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Towards Atoms of Large Language Models](../../ICML2026/interpretability/towards_atoms_of_large_language_models.md)
- [\[ICLR 2026\] Spilled Energy in Large Language Models](../../ICLR2026/interpretability/spilled_energy_in_large_language_models.md)
- [\[ICLR 2026\] Comparing the learning dynamics of in-context learning and fine-tuning in language models](../../ICLR2026/interpretability/comparing_the_learning_dynamics_of_in-context_learning_and_fine-tuning_in_langua.md)
- [\[ACL 2026\] Tracing Relational Knowledge Recall in Large Language Models](../../ACL2026/interpretability/tracing_relational_knowledge_recall_in_large_language_models.md)
- [\[ACL 2026\] Compositional Steering of Large Language Models with Steering Tokens](../../ACL2026/interpretability/compositional_steering_of_large_language_models_with_steering_tokens.md)

</div>

<!-- RELATED:END -->
