---
title: >-
  [Paper Note] StoryCoder: Narrative Reformulation for Structured Reasoning in LLM Code Generation
description: >-
  [ACL 2026][Code Intelligence][Narrative Reformulation] This paper proposes StoryCoder, a prompting framework that reformulates code generation problems into coherent natural language narratives. By guiding LLMs through s…
tags:
  - "ACL 2026"
  - "Code Intelligence"
  - "Narrative Reformulation"
  - "Code Generation"
  - "Prompt Engineering"
  - "Structured Reasoning"
  - "Algorithm Selection"
date: 2026-05-08
content_hash: 73c39e2261c6435a
---

# StoryCoder: Narrative Reformulation for Structured Reasoning in LLM Code Generation

**Conference**: ACL 2026  
**arXiv**: [2604.14631](https://arxiv.org/abs/2604.14631)  
**Code**: Available  
**Area**: Code Intelligence  
**Keywords**: Narrative Reformulation, Code Generation, Prompt Engineering, Structured Reasoning, Algorithm Selection

## TL;DR

This paper proposes StoryCoder, a prompting framework that reformulates code generation problems into coherent natural language narratives. By guiding LLMs through structured reasoning with three narrative components—task overview, constraints, and examples—it achieves an average improvement of 18.7% in zero-shot pass@10 across 11 models.

## Background & Motivation

**Background**: The performance of code generation depends not only on model capability but also on how the problem is represented. Existing methods primarily enhance performance by adding reasoning steps (CoT, SCoT) or through repeated sampling, but they do not alter the problem description itself; fragmented and imperative problem conditions remain unchanged.

**Limitations of Prior Work**: Programming task descriptions are often incomplete or ambiguous, requiring the solver to infer missing details from context. CoT introduces reasoning steps without changing the input representation; repeated sampling expands the output space without improving understanding; while SCoT introduces program structure, none of these methods address the fundamental issue of fragmented problem statements.

**Key Challenge**: Cognitive science research indicates that humans are more effective at understanding and reasoning when fragmented conditions are organized into coherent mental models. However, LLMs directly face fragmented problem descriptions, making it difficult to form a unified problem representation, which leads to disorganized reasoning paths.

**Goal**: To design a narrative reformulation framework that transforms coding problems into coherent natural language descriptions, providing a richer contextual structure than simple paraphrasing.

**Key Insight**: Inspired by analogical reasoning and mental model theories in cognitive science—humans reason better by organizing information into coherent structures. By transforming programming problems into "stories," the model can understand and solve problems within more natural language structures.

**Core Idea**: The model first selects the appropriate algorithm category and narrative genre, then reformulates the programming problem into a three-part narrative consisting of a task overview, constraints, and examples. This structured natural language representation replaces the original fragmented description to guide code generation.

## Method

### Overall Architecture

Given a programming problem $Q_i$: (i) The model first identifies the algorithm category $a_i$ (selected from 8 predefined categories) and chooses a narrative genre $g_i$; (ii) The problem is reformulated into a structured narrative $\mathcal{N}_i$, containing a task overview, constraints, and example inputs/outputs; (iii) The narrative is passed to a solver model to generate code, which is verified via test cases. Each problem generates $N=5$ narrative variants.

### Key Designs

1.  **Three-part Narrative Structure**:
    - **Function**: Organizes fragmented programming problems into coherent natural language descriptions.
    - **Mechanism**: The narrative $\mathcal{N}_i = \{\text{TO}_i, \text{C}_i, \text{E}_i\}$ consists of three parts: Task Overview (presenting the coding goal within a narrative frame, integrating scattered conditions into a coherent system), Constraints (reformulating input ranges/time limits/rules into natural limitations within the story), and Example inputs/outputs (embedding test cases into contextual scenarios).
    - **Design Motivation**: Based on mental model theory, effective reasoning requires forming a coherent and complete problem representation before generating a solution. The three-part structure ensures all key information from the original problem is preserved.

2.  **Algorithm-guided Genre Selection**:
    - **Function**: Aligns the narrative with the algorithmic essence of the problem.
    - **Mechanism**: The model first determines the most suitable category from 8 predefined algorithm categories (sorting, search, dynamic programming, etc.), then freely chooses a narrative genre that matches the problem and algorithm. Different narrative variants may select different combinations of algorithms and genres, providing diverse perspectives on problem representation.
    - **Design Motivation**: Experiments prove that genre alignment is crucial—using mismatched genres leads to a significant performance drop. Narrative genres help the model infer the correct algorithmic strategy from fragmented descriptions.

3.  **Narrative Diversity Sampling**:
    - **Function**: Expands the model's representation space through multiple narrative variants.
    - **Mechanism**: For each problem, $N$ narrative variants are generated, each potentially having different algorithm judgments, genre selections, and narrative developments. In experiments, 5 pure narrative variants and 5 narrative+original question concatenated variants are aggregated, totaling 10 responses. This differs from simple repeated sampling, as each narrative provides a different perspective for understanding the problem.
    - **Design Motivation**: Diverse narrative variants explore the solution space more effectively than repeated sampling because each variant alters the input representation rather than merely sampling different outputs.

### Loss & Training

A pure inference-time method requiring no training. Narrative generation temperature is set to 1.0 (to encourage diversity), and code generation temperature is set to 0.2 (to encourage accuracy).

## Key Experimental Results

### Main Results (pass@10, Average of Open-source Models)

| Method | HumanEval | LiveCodeBench | CodeForces |
| :--- | :--- | :--- | :--- |
| Repeated Sampling (RS) | 81.31 | 26.36 | 18.96 |
| CoT | 82.26 | 27.57 | 19.32 |
| SCoT | 82.60 | 26.93 | 19.26 |
| **StoryCoder** | **89.76** | **32.22** | **28.58** |

Representative Closed-source Models (pass@10):

| Model | Method | LiveCodeBench | CodeForces |
| :--- | :--- | :--- | :--- |
| Claude-3.5-Haiku | RS | 33.71 | 47.17 |
| Claude-3.5-Haiku | **Narr.** | **38.29** | **50.95** |
| Gemini-2.5-Flash | RS | 53.14 | 60.00 |
| Gemini-2.5-Flash | **Narr.** | **57.14** | **67.55** |

### Ablation Study

| Analysis Dimension | Finding |
| :--- | :--- |
| Algorithm Selection Accuracy | StoryCoder makes it easier for the model to choose the correct algorithm (+Significant Improvement) |
| Implementation Error Rate | Narrative reformulation reduces implementation errors |
| Code Structure | Narrative guidance produces more modular code structures |
| Genre Mismatch | Using mismatched genres leads to a significant performance drop |

### Key Findings
- Narrative reformulation consistently outperforms all baselines across all 11 models and 3 benchmarks, with an average pass@10 gain of 18.7%.
- Improvements are particularly significant on difficult benchmarks (CodeForces), where open-source models improved from an average of 18.96% to 28.58% (+50.7% relative gain).
- Narratives not only improve accuracy but also guide the model to select correct algorithmic strategies, reduce implementation errors, and produce more modular code.
- The quality of narrative generation correlates with the model's instruction-following capability—Gemma-2 27B has a 96% valid narrative rate, while Llama-3.1 8B is only 36.7%.

## Highlights & Insights
- The idea of **"changing the problem representation instead of the reasoning process"** is highly novel: unlike CoT which adds reasoning steps, it improves understanding by reformulating the input. The link to mental model theory in cognitive science is compelling.
- The **three-part narrative design** balances narrative coherence with computational rigor, particularly by embedding constraints and test cases within the narrative rather than simply appending them, ensuring organic integration of information.
- Findings from the **cross-model setup** have practical value: narratives can be generated by models with strong instruction-following capabilities, while code can be generated by models with high coding proficiency, achieving complementarity.

## Limitations & Future Work
- Narrative generation itself consumes extra tokens and inference time (although the authors consider it "free," there is practical overhead).
- Narrative quality depends heavily on the instruction-following capability of the generating model; small models (e.g., Llama-3.1 8B) have very low valid narrative rates.
- The 8 predefined algorithm categories may not cover all types of programming problems.
- The effectiveness of narrative reformulation on non-algorithmic programming tasks, such as mathematical proofs or system design, remains unknown.
- The combination of narratives with other reasoning enhancement methods (e.g., self-consistency, reflection) has not been explored.

## Related Work & Insights
- **vs CoT/SCoT**: CoT adds reasoning steps without changing the input representation; SCoT introduces code structure but remains based on the original problem description. StoryCoder improves problem understanding from the input side, acting orthogonally and combinably with these methods.
- **vs Repeated Sampling**: Repeated sampling increases diversity at the output stage, whereas StoryCoder increases diversity at the input stage through different narrative variants, the latter exploring the solution space more effectively.

## Rating
- Novelty: ⭐⭐⭐⭐ The idea of narrative reformulation is highly novel and grounded in cognitive science.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 11 models, 3 benchmarks, and extensive analysis (algorithm selection, error types, code structure).
- Writing Quality: ⭐⭐⭐⭐ Motivation and methods are clearly articulated, and the cognitive science connection is persuasive.
- Value: ⭐⭐⭐⭐ Provides a new dimension for enhancing code generation with significant practical effects.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] ReCode: Reinforcing Code Generation with Reasoning-Process Rewards](recode_reinforcing_code_generation_with_reasoning-process_rewards.md)
- [\[ACL 2026\] SolidCoder: Bridging the Mental-Reality Gap in LLM Code Generation through Concrete Execution](solidcoder_bridging_the_mental-reality_gap_in_llm_code_generation_through_concre.md)
- [\[ICLR 2026\] Breaking the SFT Plateau: Multimodal Structured Reinforcement Learning for Chart-to-Code Generation](../../ICLR2026/code_intelligence/breaking_the_sft_plateau_multimodal_structured_reinforcement_learning_for_chart-.md)
- [\[ACL 2026\] CollabCoder: Plan-Code Co-Evolution via Collaborative Decision-Making for Efficient Code Generation](collabcoder_plan-code_co-evolution_via_collaborative_decision-making_for_efficie.md)
- [\[ACL 2026\] CodeRL+: Improving Code Generation via Reinforcement with Execution Semantics Alignment](coderl_improving_code_generation_via_reinforcement_with_execution_semantics_alig.md)

</div>

<!-- RELATED:END -->
