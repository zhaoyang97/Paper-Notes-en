---
title: >-
  [Paper Note] HyGenar: An LLM-Driven Hybrid Genetic Algorithm for Few-Shot Grammar Generation
description: >-
  [ACL 2025][LLM (Other)][Grammar Generation] Constructs a grammar generation dataset comprising 540 challenges, designs 6 evaluation metrics, and proposes HyGenar, an LLM-driven hybrid genetic algorithm, significantly improving LLMs' ability to generate BNF grammars from few-shot examples.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Grammar Generation"
  - "BNF"
  - "Few-Shot"
  - "Genetic Algorithm"
  - "Context-Free Grammar"
date: 2026-05-08
content_hash: 1ea7d9dfd5a62ebd
---

# HyGenar: An LLM-Driven Hybrid Genetic Algorithm for Few-Shot Grammar Generation

**Conference**: ACL 2025  
**arXiv**: [2505.16978](https://arxiv.org/abs/2505.16978)  
**Code**: [https://github.com/RutaTang/HyGenar](https://github.com/RutaTang/HyGenar)  
**Area**: LLM NLP / Grammar Induction  
**Keywords**: Grammar Generation, BNF, Few-Shot, Genetic Algorithm, Context-Free Grammar

## TL;DR
Constructs a grammar generation dataset comprising 540 challenges, designs 6 evaluation metrics, and proposes HyGenar, an LLM-driven hybrid genetic algorithm, significantly improving LLMs' ability to generate BNF grammars from few-shot examples.

## Background & Motivation
**Background**: Grammar induction has broad applications in NLP and software engineering (e.g., parser generation, LLM structured output constraints).

**Limitations of Prior Work**: Although LLMs perform exceptionally in most tasks, their ability to induce and generate BNF grammars from a few positive and negative examples has not been fully explored.

**Key Challenge**: Traditional grammar induction requires a large number of characteristic examples. Can LLMs leverage pre-trained knowledge to induce grammar from extremely few examples?

**Goal**: To evaluate and enhance the few-shot grammar generation capabilities of LLMs.

**Key Insight**: Combining the search capability of genetic algorithms with the generative capability of LLMs.

**Core Idea**: Utilizing LLMs to initialize the population and drive mutation, while leveraging a BNF parser to provide precise fitness scoring.

## Method

### Overall Architecture
(1) LLM generates the initial candidate grammar population $\rightarrow$ (2) BNF parser computes fitness $\rightarrow$ (3) Selection + Crossover + LLM-driven mutation $\rightarrow$ Iterate until convergence.

### Key Designs
1. **Fitness Function (Fitness)**:

    - **Function**: Evaluates the quality of candidate grammars
    - **Mechanism**: Scores -1 for syntactic errors; if syntactically correct, computes the count of accepted positive examples plus rejected negative examples
    - **Design Motivation**: To provide precise feedback signals

2. **Crossover Operation (Crossover)**:

    - **Function**: Generates a new grammar from two parent grammars
    - **Mechanism**: Concatenates production rules from two grammars at a random position
    - **Design Motivation**: A core operation in traditional genetic algorithms to retain high-quality genetic segments

3. **LLM-Driven Mutation (Mutation)**:

    - **Function**: Modifies candidate grammars using an LLM
    - **Mechanism**: Feeds the current grammar, positive/negative examples, and fitness feedback to the LLM to guide it in modifying rules
    - **Design Motivation**: To leverage the semantic comprehension capability of LLMs for directed mutation

### 6 Evaluation Metrics
- **SX**: Syntax Correctness (valid BNF)
- **SE**: Semantic Correctness (accepting positives + rejecting negatives)
- **Diff◇**: Difference in the number of production rules
- **OF**: Overfitting rate
- **OG**: Overgeneralization rate
- **TU◇**: Rule utilization rate

## Key Experimental Results

### Main Results (Syntax/Semantic Correctness %)

| Model | SX(DP) | SX(HyGenar) | SE(DP) | SE(HyGenar) |
|------|-------|------------|-------|------------|
| GPT-4o | 93 | 96 (+3) | 84 | **93** (+9) |
| GPT-3.5-Turbo | 94 | 99 (+5) | 37 | **61** (+24) |
| Llama3:70b | 57 | 75 (+18) | 41 | **61** (+20) |
| Gemma2:27b | 91 | 98 (+7) | 56 | **79** (+23) |
| Codestral:22b | 53 | 80 (+27) | 44 | **67** (+23) |
| Qwen:72b | 47 | 76 (+29) | 20 | **38** (+18) |

### Ablation Study

| Method | Average SE Gain |
|------|-------------|
| Direct Prompting (DP) | Baseline |
| OPF (Parser Feedback) | +1~8% |
| HyGenar (Complete) | +9~24% |

### Key Findings
- Even for models with high SX (e.g., GPT-3.5 at 94%), SE can be very low (37%), indicating that syntax correctness does not guarantee semantic correctness.
- HyGenar achieves the most significant improvement on medium-capability models (e.g., Codestral +23%, Qwen +18%).
- As the number of non-terminals increases (higher grammar complexity), performance decreases across all models.
- LLMs are capable of identifying keywords (such as "if", "SELECT") as atomic terminals.

## Highlights & Insights
- Combining LLMs with genetic algorithms is a novel approach: LLMs provide semantic guidance, while genetic algorithms provide systematic search.
- The BNF parser functions as a precise oracle to offer noise-free feedback.
- The dataset design covers grammars with 1 to 9 non-terminals, presenting a reasonable difficulty gradient.

## Limitations & Future Work
- Each challenge has only 3 positive and 3 negative examples; more examples could substantially improve performance.
- Only context-free grammars (CFGs) are tested, leaving more complex grammar types unaddressed.
- The iterations of the genetic algorithm require multiple LLM calls, which is computationally expensive.

## Related Work & Insights
- **vs Reflexion (Shinn et al. 2023)**: Reflexion relies on self-feedback to improve code, whereas HyGenar employs genetic search to refine grammars.
- **vs ANTLR4**: ANTLR4 requires manual grammar writing, whereas HyGenar can automatically induce grammar from examples.

## Additional Details
- The dataset is constructed by generating 90 reference grammars via GPT-4o, with 6 challenges derived from each, totaling 540 challenges.
- Each challenge includes 3 positive and 3 negative examples.
- Genetic algorithm parameters: population size $k$, selecting the top $k/2$ individuals in each generation, propagating until convergence.
- Evaluated 8 LLMs: GPT-4o, GPT-3.5, Llama3, Qwen, Gemma2, Mistral, Codestral, and Starcoder2.
- Grammar complexity is graded by the number of non-terminals $k=1\sim9$.
- LLMs can recognize keywords like "if" and "SELECT" as atomic terminals.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The combination of LLM and genetic algorithm is highly novel, and this work represents the first systematic study in the field of few-shot grammar generation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive evaluation covering 8 LLMs × 540 challenges × 6 metrics.
- **Writing Quality**: ⭐⭐⭐⭐ Rigorous formal definitions, though with a somewhat high barrier to readability.
- **Value**: ⭐⭐⭐⭐ Offers direct inspiration for application scenarios such as structured LLM output constraints.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Genetic Instruct: Scaling up Synthetic Generation of Coding Instructions for Large Language Models](genetic_instruct_scaling_up_synthetic_generation_of_coding_instructions_for_larg.md)
- [\[ACL 2025\] BIPro: Zero-shot Chinese Poem Generation via Block Inverse Prompting Constrained Generation Framework](bipro_zero-shot_chinese_poem_generation_via_block_inverse_prompting_constrained_.md)
- [\[ICML 2025\] Random Registers for Cross-Domain Few-Shot Learning](../../ICML2025/llm_nlp/random_registers_for_cross-domain_few-shot_learning.md)
- [\[ACL 2025\] Zero-Shot Belief: A Hard Problem for LLMs](zero-shot_belief_a_hard_problem_for_llms.md)
- [\[ACL 2025\] Dynamic Knowledge Integration for Evidence-Driven Counter-Argument Generation with Large Language Models](dynamic_knowledge_integration_for_evidence-driven_counter-argument_generation_wi.md)

</div>

<!-- RELATED:END -->
