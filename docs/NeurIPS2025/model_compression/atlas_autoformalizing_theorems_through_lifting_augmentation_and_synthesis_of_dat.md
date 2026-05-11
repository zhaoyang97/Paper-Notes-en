---
title: >-
  [Paper Note] ATLAS: Autoformalizing Theorems through Lifting, Augmentation, and Synthesis of Data
description: >-
  [NeurIPS 2025][Model Compression][Autoformalization] ATLAS proposes a data generation framework based on a concept repository, expert iteration with knowledge distillation…
tags:
  - "NeurIPS 2025"
  - "Model Compression"
  - "Autoformalization"
  - "Lean4"
  - "Knowledge Distillation"
  - "Expert Iteration"
  - "Data Augmentation"
date: 2026-05-08
content_hash: 8e7b725301be6872
---

# ATLAS: Autoformalizing Theorems through Lifting, Augmentation, and Synthesis of Data

**Conference**: NeurIPS 2025
**arXiv**: [2502.05567](https://arxiv.org/abs/2502.05567)
**Code**: [GitHub](https://github.com/XiaoyangLiu-sjtu/ATLAS)
**Area**: Autoformalization / Theorem Proving
**Keywords**: Autoformalization, Lean4, Knowledge Distillation, Expert Iteration, Data Augmentation

## TL;DR

ATLAS proposes a data generation framework based on a concept repository, expert iteration with knowledge distillation, and two novel augmentation strategies. It constructs a parallel corpus of 117K theorem statements, and achieves SOTA on all autoformalization benchmarks after fine-tuning Llama3.1-8B-Instruct.

## Background & Motivation

Autoformalization refers to the task of translating natural language mathematical content into machine-verifiable formal languages (e.g., Lean4, Isabelle). While advances in large language models (LLMs) have driven progress in this area, the central bottleneck remains **the scarcity of high-quality parallel corpora**—training data that maps informal mathematical text to its formal counterpart is extremely limited.

Challenges faced by existing approaches:

**Prohibitively high annotation cost**: Mathematical formalization requires expert-level knowledge; a single high-quality sample may take hours to produce.

**Small scale of existing datasets**: For instance, ProofNet contains only a few hundred samples.

**Unstable quality of direct LLM translation**: Even GPT-4 frequently fails on complex theorems.

**Underutilization of formal language verifiability**: Formal languages allow automatic correctness checking via type checkers, a property that has not been sufficiently exploited.

## Method

### Overall Architecture

The core mechanism of the ATLAS framework is to "bootstrap large-scale, high-quality parallel corpora from a small set of seed data through iterative self-improvement":

1. **Concept Repository**: Mathematical concepts and definitions are extracted from Mathlib (the Lean4 mathematics library) as seeds.
2. **Expert Iteration**: A strong teacher model generates candidate translations, which are filtered by Lean4 verification.
3. **Knowledge Distillation**: A student model is trained on the filtered data and progressively replaces or supplements the teacher in subsequent rounds.
4. **Data Augmentation**: Two novel augmentation strategies that exploit the structural properties of formal languages.

The pipeline runs for 10 iterations; each round generates and verifies new data, progressively improving model capability.

### Key Designs

**Construction of the Concept Repository**:
- Undergraduate-level mathematical concepts are extracted from Mathlib4 (set theory, algebra, analysis, topology, etc.).
- Each concept includes its formal definition, related lemmas, and type information.
- The repository provides a structured starting point for data generation.

**Two Novel Augmentation Strategies**:

**Strategy 1: Structural Lifting**
- Exploits the type system of formal languages to "lift" simple theorems to more general settings.
- Example: a proposition about real numbers is lifted to a general metric space.
- This augmentation is difficult to automate in natural language but is achievable in formal languages via type inference.

**Strategy 2: Compositional Synthesis**
- Combines verified formal theorems to generate new composite theorems.
- Example: combining "$A \to B$" and "$B \to C$" to yield "$A \to C$".
- The Lean4 verifier ensures the composed theorem remains well-typed and valid.

**Expert Iteration + Knowledge Distillation Loop**:
- Round 0: GPT-4 serves as the teacher to generate initial translations.
- Rounds 1–10: the current best student model generates candidates → Lean4 verification → correct samples are added to the training set → a stronger student model is trained.
- Data quantity and quality improve in tandem across rounds, forming a positive feedback loop.

### Loss & Training

- **Base Model**: Llama3.1-8B-Instruct
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation), rank=16, alpha=32
- **Training Data**: 117K theorem statement pairs generated over 10 iterations
- **Verification Filtering**: All formalized outputs must pass the Lean4 type checker
- **Loss Function**: Standard cross-entropy loss; input is the natural language theorem statement, target is its Lean4 formalization

## Key Experimental Results

### Main Results

**Overall results on standard benchmarks** (Llama3.1-8B-Instruct + LoRA):

| Model / Method | ProofNet (Pass@1) | MiniF2F (Pass@1) | FIMO (Pass@1) | Avg. |
|---|---|---|---|---|
| GPT-4 (zero-shot) | 16.1 | 22.3 | 12.8 | 17.1 |
| Herald Translator | 25.4 | 30.1 | 18.6 | 24.7 |
| Kimina-Autoformalizer | 28.7 | 33.5 | 21.2 | 27.8 |
| **ATLAS Translator (LoRA)** | **34.2** | **38.9** | **26.4** | **33.2** |

All comparisons are statistically significant ($p < 0.05$, two-tailed t-test).

**Comparison across base models and fine-tuning strategies**:

| Base Model | Fine-tuning | ProofNet | MiniF2F | FIMO |
|---|---|---|---|---|
| Llama3.1-8B-Instruct | LoRA | 34.2 | 38.9 | 26.4 |
| Llama3.1-8B-Instruct | Full FT | 36.8 | 41.2 | 29.1 |
| Llama3.1-70B-Instruct | LoRA | 38.5 | 44.7 | 31.8 |
| Llama3.1-70B-Instruct | Full FT | **41.3** | **47.2** | **34.6** |
| DeepSeek-Prover-V1.5 | Full FT | 40.1 | 46.8 | 33.9 |

### Ablation Study

**Ablation analysis of individual component contributions** (Llama3.1-8B-Instruct + LoRA):

| Training Data Configuration | ProofNet | MiniF2F | FIMO |
|---|---|---|---|
| Seed data only (Round 0) | 22.1 | 26.4 | 15.3 |
| + Expert Iteration (no augmentation) | 29.5 | 33.7 | 22.1 |
| + Structural Lifting augmentation | 31.8 | 36.2 | 24.3 |
| + Compositional Synthesis augmentation | 30.9 | 35.4 | 23.7 |
| + Both augmentations (full ATLAS) | **34.2** | **38.9** | **26.4** |

**Relationship between number of iterations and performance**:
- Rounds 1–3: rapid performance improvement
- Rounds 4–7: growth gradually slows
- Rounds 8–10: near saturation with diminishing marginal returns
- Data volume grows from ~5K in Round 1 to 117K in Round 10

### Key Findings

1. **Expert iteration is the primary driver**: removing iteration leads to a ~30% relative performance drop.
2. **The two augmentation strategies are complementary**: Structural Lifting contributes ~10% improvement, Compositional Synthesis ~8%, and combining both yields ~16%.
3. **Stronger base model + Full FT achieves the best results**: the 70B model with full fine-tuning reaches the highest performance.
4. **General value of ATLAS data**: consistent gains are observed across different base models, confirming that data quality is the key factor.
5. **Lean4 verification as a quality gate**: approximately 30–40% of candidate translations are rejected by the verifier in each round.

## Highlights & Insights

- **Formal verification as a free annotator**: The Lean4 type checker is cleverly repurposed as an automatic sample quality evaluator, eliminating the need for human annotation.
- **A successful instance of bootstrapped learning**: Large-scale, high-quality data generation is achieved from a small seed set through iterative self-improvement.
- **Creative exploitation of formal language structure**: Both augmentation strategies are uniquely enabled by the properties of formal languages, fully leveraging the advantages of the target domain.
- **Open-source contribution**: The dataset, models, and code are all publicly released, facilitating reproducibility and community follow-up.

## Limitations & Future Work

1. **Restricted to theorem statements**: The current work addresses only the formalization of theorem statements, not of proofs.
2. **Undergraduate-level mathematics**: The dataset primarily covers undergraduate-level concepts; complex frontier research theorems remain unexplored.
3. **Lean4 dependency**: The technical stack is tied to a specific formal language; migration to Coq or Isabelle requires additional effort.
4. **Reliance on a teacher model**: The initial round depends on GPT-4 to generate seed data.
5. **Non-trivial iteration cost**: Ten rounds of expert iteration require substantial GPU resources and Lean4 verification time.

## Related Work & Insights

- **Herald Translator**: The previous SOTA autoformalization model; ATLAS builds upon and improves it.
- **Kimina-Autoformalizer**: A competing approach also based on LLM fine-tuning.
- **DeepSeek-Prover-V1.5**: An automated theorem proving model that is complementary to ATLAS.
- **Expert Iteration / Self-Play**: The iterative framework in ATLAS shares conceptual similarity with the self-play paradigm of AlphaGo.
- Takeaway: The combination of formal verification and large language models represents a promising research paradigm.

## Rating

- Theoretical Depth: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Novelty: ⭐⭐⭐⭐⭐
- Practicality: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Single-Teacher View Augmentation: Boosting Knowledge Distillation via Angular Diversity](single-teacher_view_augmentation_boosting_knowledge_distillation_via_angular_div.md)
- [\[NeurIPS 2025\] Skrull: Towards Efficient Long Context Fine-tuning through Dynamic Data Scheduling](skrull_towards_efficient_long_context_fine-tuning_through_dynamic_data_schedulin.md)
- [\[ICLR 2026\] Pedagogically-Inspired Data Synthesis for Language Model Knowledge Distillation](../../ICLR2026/model_compression/pedagogically-inspired_data_synthesis_for_language_model_knowledge_distillation.md)
- [\[NeurIPS 2025\] A Token is Worth over 1,000 Tokens: Efficient Knowledge Distillation through Low-Rank Clone](a_token_is_worth_over_1000_tokens_efficient_knowledge_distillation_through_low-r.md)
- [\[ACL 2026\] Find Your Optimal Teacher: Personalized Data Synthesis via Router-Guided Multi-Teacher Distillation](../../ACL2026/model_compression/find_your_optimal_teacher_personalized_data_synthesis_via_router-guided_multi-te.md)

</div>

<!-- RELATED:END -->
