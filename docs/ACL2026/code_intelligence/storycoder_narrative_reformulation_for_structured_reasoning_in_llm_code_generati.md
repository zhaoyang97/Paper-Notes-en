---
title: >-
  [Paper Note] StoryCoder: Narrative Reformulation for Structured Reasoning in LLM Code Generation
description: >-
  [ACL 2026][Narrative reformulation] This paper proposes StoryCoder, a prompting framework that reformulates code generation problems into coherent natural language narratives. By guiding LLMs through three narrative components—task overview, constraints, and examples—the framework achieves an average zero-shot pass@10 improvement of 18.7% across 11 models.
tags:
  - ACL 2026
  - Narrative reformulation
  - code generation
  - prompt engineering
  - structured reasoning
  - algorithm selection
date: 2026-05-08
content_hash: 14fde6262f7fb06d
---

# StoryCoder: Narrative Reformulation for Structured Reasoning in LLM Code Generation

**Conference**: ACL 2026
**arXiv**: [2604.14631](https://arxiv.org/abs/2604.14631)
**Code**: Available
**Area**: Code Intelligence
**Keywords**: Narrative reformulation, code generation, prompt engineering, structured reasoning, algorithm selection

## TL;DR

This paper proposes StoryCoder, a prompting framework that reformulates code generation problems into coherent natural language narratives. By guiding LLMs through three narrative components—task overview, constraints, and examples—the framework achieves an average zero-shot pass@10 improvement of 18.7% across 11 models.

## Background & Motivation

**Background**: Code generation performance depends not only on model capability but also on problem representation. Existing methods primarily improve performance by adding reasoning steps (CoT, SCoT) or through repeated sampling, without altering the problem description itself—fragmented, instruction-style problem conditions remain unchanged.

**Limitations of Prior Work**: Programming task descriptions are often incomplete or ambiguous, requiring solvers to infer missing details from context. CoT introduces reasoning steps but does not change input representation; repeated sampling expands outputs but does not improve comprehension; SCoT introduces program structure, yet none of these approaches address the fundamental fragmentation of problem formulation.

**Key Challenge**: Cognitive science research shows that humans reason and comprehend more effectively when fragmented conditions are organized into coherent mental models. LLMs facing fragmented problem descriptions struggle to form a unified problem representation, leading to disordered reasoning paths.

**Goal**: Design a narrative reformulation framework that transforms coding problems into coherent natural language descriptions, providing richer contextual structure than simple paraphrasing.

**Key Insight**: Inspired by analogical reasoning and mental model theory in cognitive science—humans reason better by organizing information into coherent structures. Programming problems are reformulated as "stories," enabling the model to understand and solve them within a more natural linguistic structure.

**Core Idea**: The model first selects an appropriate algorithm category and narrative genre, then reformulates the programming problem into a three-part narrative comprising a task overview, constraints, and examples. This structured natural language representation replaces the original fragmented description to guide code generation.

## Method

### Overall Architecture

Given a programming problem $Q_i$: (i) the model first identifies the algorithm category $a_i$ (selected from 8 predefined categories) and chooses a narrative genre $g_i$; (ii) the problem is reformulated into a structured narrative $\mathcal{N}_i$ containing a task overview, constraints, and example inputs/outputs; (iii) the narrative is passed to a solver model to generate code, verified against test cases. Each problem generates $N=5$ narrative variants.

### Key Designs

1. **Three-Part Narrative Structure**:

    - Function: Organizes fragmented programming problems into coherent natural language descriptions.
    - Mechanism: The narrative $\mathcal{N}_i = \{\text{TO}_i, \text{C}_i, \text{E}_i\}$ consists of three components—Task Overview (presents the coding objective within a narrative frame, integrating scattered conditions into a coherent system), Constraints (reformulates input ranges, time limits, and rules as natural limitations within the story), and Example Inputs/Outputs (embeds test cases into contextual scenarios).
    - Design Motivation: Based on mental model theory, effective reasoning requires forming a coherent and complete problem representation before generating solutions. The three-part structure ensures all critical information from the original problem is preserved.

2. **Algorithm-Guided Genre Selection**:

    - Function: Aligns the narrative with the algorithmic nature of the problem.
    - Mechanism: The model first identifies the most appropriate category from 8 predefined algorithm classes (sorting, search, dynamic programming, etc.), then freely selects a narrative genre matching the problem and algorithm. Different narrative variants may adopt different algorithm–genre combinations, providing diverse perspectives on problem representation.
    - Design Motivation: Experiments demonstrate that genre alignment is critical—mismatched genres lead to significant performance degradation. Narrative genre helps the model infer the correct algorithmic strategy from fragmented descriptions.

3. **Narrative Diversity Sampling**:

    - Function: Expands the model's representation space through multiple narrative variants.
    - Mechanism: Each problem generates $N$ narrative variants, each potentially differing in algorithm identification, genre selection, and narrative elaboration. Experiments aggregate 5 pure narrative variants and 5 narrative-plus-original-problem concatenation variants, yielding 10 responses in total. This differs from simple repeated sampling—each narrative provides a distinct interpretive perspective on the problem.
    - Design Motivation: Diverse narrative variants explore the solution space more effectively than repeated sampling, as each variant modifies the input representation rather than merely sampling different outputs.

### Loss & Training

StoryCoder is a purely inference-time method requiring no training. Narrative generation uses a temperature of 1.0 (to encourage diversity), while code generation uses a temperature of 0.2 (to encourage accuracy).

## Key Experimental Results

### Main Results (pass@10, average over open-source models)

| Method | HumanEval | LiveCodeBench | CodeForces |
|--------|-----------|---------------|------------|
| Repeated Sampling (RS) | 81.31 | 26.36 | 18.96 |
| CoT | 82.26 | 27.57 | 19.32 |
| SCoT | 82.60 | 26.93 | 19.26 |
| **StoryCoder** | **89.76** | **32.22** | **28.58** |

Representative closed-source model results (pass@10):

| Model | Method | LiveCodeBench | CodeForces |
|-------|--------|---------------|------------|
| Claude-3.5-Haiku | RS | 33.71 | 47.17 |
| Claude-3.5-Haiku | **Narr.** | **38.29** | **50.95** |
| Gemini-2.5-Flash | RS | 53.14 | 60.00 |
| Gemini-2.5-Flash | **Narr.** | **57.14** | **67.55** |

### Ablation Study

| Analysis Dimension | Finding |
|-------------------|---------|
| Algorithm selection accuracy | StoryCoder significantly improves the likelihood of selecting the correct algorithm |
| Implementation error rate | Narrative reformulation reduces implementation errors |
| Code structure | Narrative guidance produces more modular code structure |
| Genre mismatch | Using mismatched genres leads to significant performance degradation |

### Key Findings
- Narrative reformulation consistently outperforms all baselines across all 11 models and 3 benchmarks, with an average pass@10 improvement of 18.7%.
- Gains are especially pronounced on the challenging CodeForces benchmark, where open-source model averages improve from 18.96% to 28.58% (+50.7% relative improvement).
- Narratives not only improve accuracy but also guide models toward correct algorithmic strategies, reduce implementation errors, and produce more modular code.
- Narrative generation quality correlates with a model's instruction-following capability—Gemma-2 27B achieves a 96% valid narrative rate, while Llama-3.1 8B achieves only 36.7%.

## Highlights & Insights
- The notion of **"changing problem representation rather than changing the reasoning process"** is highly novel: instead of adding reasoning steps as in CoT, understanding is improved by reformulating the input. The connection to mental model theory in cognitive science is compelling.
- The **three-part narrative design** balances narrative coherence with computational rigor; in particular, embedding constraints and test cases within the narrative rather than simply appending them ensures organic integration of information.
- **Cross-model setting findings** offer practical value: a model with strong instruction-following capability can generate narratives while a model with strong coding capability generates code, enabling complementary collaboration.

## Limitations & Future Work
- Narrative generation incurs additional token and inference overhead (although the authors characterize it as "free," the generation cost is real).
- Narrative quality is highly dependent on the instruction-following capability of the generating model; valid narrative rates are very low for smaller models (e.g., Llama-3.1 8B).
- The 8 predefined algorithm categories may not cover all programming problem types.
- The effectiveness of narrative reformulation on non-algorithmic programming tasks such as mathematical proofs or system design remains unknown.
- Combinations of narrative reformulation with other reasoning-enhancement methods (e.g., self-consistency, reflection) have not been explored.

## Related Work & Insights
- **vs. CoT/SCoT**: CoT adds reasoning steps without changing input representation; SCoT introduces code structure but still operates on the original problem description. StoryCoder improves problem comprehension at the input level, making it orthogonal to and composable with these methods.
- **vs. Repeated Sampling**: Repeated sampling increases diversity at the output level, whereas StoryCoder increases diversity at the input level through distinct narrative variants, exploring the solution space more effectively.

## Rating
- Novelty: ⭐⭐⭐⭐ The narrative reformulation idea is highly novel and grounded in cognitive science.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 11 models, 3 benchmarks, and rich analysis covering algorithm selection, error types, and code structure.
- Writing Quality: ⭐⭐⭐⭐ Motivation and methodology are clearly articulated; the cognitive science connection is persuasive.
- Value: ⭐⭐⭐⭐ Introduces a new dimension for improving code generation with significant practical gains.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] SolidCoder: Bridging the Mental-Reality Gap in LLM Code Generation through Concrete Execution](solidcoder_bridging_the_mental-reality_gap_in_llm_code_generation_through_concre.md)
- [\[ICLR 2026\] Breaking the SFT Plateau: Multimodal Structured Reinforcement Learning for Chart-to-Code Generation](../../ICLR2026/code_intelligence/breaking_the_sft_plateau_multimodal_structured_reinforcement_learning_for_chart-.md)
- [\[ACL 2026\] CollabCoder: Plan-Code Co-Evolution via Collaborative Decision-Making for Efficient Code Generation](collabcoder_plan-code_co-evolution_via_collaborative_decision-making_for_efficie.md)
- [\[ACL 2026\] CodeRL+: Improving Code Generation via Reinforcement with Execution Semantics Alignment](coderl_improving_code_generation_via_reinforcement_with_execution_semantics_alig.md)
- [\[AAAI 2026\] DiffBench Meets DiffAgent: End-to-End LLM-Driven Diffusion Acceleration Code Generation](../../AAAI2026/code_intelligence/diffbench_meets_diffagent_end-to-end_llm-driven_diffusion_ac.md)

<!-- RELATED:END -->
