---
title: >-
  [Paper Note] Extracting Events Like Code: A Multi-Agent Programming Framework for Zero-Shot Event Extraction
description: >-
  [AAAI 2026][Code Intelligence][Zero-Shot Event Extraction] This paper proposes Agent-Event-Coder (AEC), which reformulates zero-shot event extraction as a software engineering workflow. Four specialized agents (Retrieval…
tags:
  - "AAAI 2026"
  - "Code Intelligence"
  - "Zero-Shot Event Extraction"
  - "Multi-Agent"
  - "Programmatic Framework"
  - "Python Class Template"
  - "Dual-Loop Refinement"
date: 2026-05-08
content_hash: cb31968a2a87c340
---

# Extracting Events Like Code: A Multi-Agent Programming Framework for Zero-Shot Event Extraction

**Conference**: AAAI 2026
**arXiv**: [2511.13118](https://arxiv.org/abs/2511.13118)
**Code**: [https://github.com/UESTC-GQJ/Agent-Event-Coder](https://github.com/UESTC-GQJ/Agent-Event-Coder)
**Area**: Code Intelligence
**Keywords**: Zero-Shot Event Extraction, Multi-Agent, Programmatic Framework, Python Class Template, Dual-Loop Refinement

## TL;DR

This paper proposes Agent-Event-Coder (AEC), which reformulates zero-shot event extraction as a software engineering workflow. Four specialized agents (Retrieval→Planning→Coding→Verification) collaborate to perform extraction, while event schemas are encoded as executable Python classes to enable compiler-style deterministic validation and dual-loop iterative correction. AEC comprehensively outperforms zero-shot baselines across 5 domains and 6 LLMs.

## Background & Motivation

**Importance of Event Extraction**: Event extraction (EE) identifies event triggers and their arguments from unstructured text, playing a central role in knowledge base construction, information retrieval, and question answering.

**Bottleneck of Supervised Methods**: Traditional EE relies on large amounts of annotated data; however, event types continue to grow and annotation costs are prohibitively high, making it infeasible to collect training data for every new type.

**Promise and Challenges of Zero-Shot EE**: ZSEE can extract unseen event types using only type names or definitions, offering strong scalability, yet practical performance remains far from satisfactory.

**Contextual Ambiguity**: Trigger words are often polysemous (e.g., "strike" may refer to a labor strike or an attack). Under the zero-shot setting, LLMs tend to over-rely on the trigger word itself while neglecting contextual cues, leading to type misclassification.

**Structural Fidelity**: EE requires outputs that strictly conform to predefined schemas (e.g., JSON), yet untuned LLMs frequently produce schema-violating outputs—including nonexistent roles, hallucinated arguments, and data type errors.

**Limitations of Existing Methods**: Single-agent direct prompting approaches (e.g., DirectEE, ChatIE) are highly sensitive to prompt design, lack explicit decomposition, and cannot handle complex trigger–argument interactions. Existing multi-agent IE work does not simultaneously address semantic disambiguation and structural validation.

## Method

### Overall Architecture

AEC redefines ZSEE as a "collaborative, verifiable code generation" process. The input consists of unstructured text $T$ and an unseen event schema $S_e = \langle e, R_e \rangle$ (where $e$ denotes the event type and $R_e$ denotes argument roles and their types), and the output is an event instance $y = \langle e, z, A \rangle$ (where $z$ is the trigger word and $A$ is the set of argument–role pairs). The entire pipeline chains four specialized agents in sequence, with inner and outer dual-loop feedback mechanisms for iterative reasoning and verification.

### Key Designs

#### 1. Retrieval Agent

- **Function**: Given schema $S_e$, automatically generates $k$ high-quality example sentences $D_{ex} = \{s_1, \dots, s_k\}$ to serve as an analogical bridge from schema to text.
- **Mechanism**: Drawing on the analogy prompting paradigm (Yasunaga et al. 2024), the LLM constructs examples semantically aligned with the schema, rather than relying on external retrieval.
- **Design Motivation**: Since no annotated samples are available in the zero-shot setting, self-generated examples map abstract schema constraints to concrete linguistic realizations, helping downstream agents reduce early commitment errors on ambiguous trigger words.

#### 2. Planning Agent

- **Function**: Given input text $T$, schema $S_e$, and retrieved examples $D_{ex}$, generates $k$ ranked trigger–type hypotheses: $P = \{((z_i, e), \beta_i, \rho_i)\}_{i=1}^k$, where $\beta_i \in [0,1]$ is the confidence score and $\rho_i$ is a natural-language rationale.
- **Mechanism**: Integrates lexical and semantic cues to provide explanatory reasoning for why each candidate trigger may activate the given event type.
- **Design Motivation**: Generating multiple ranked hypotheses rather than a single prediction provides a candidate pool for fallback upon verification failure; retaining rationales $\rho_i$ facilitates error analysis and interpretability.

#### 3. Coding Agent

- **Function**: Transforms the highest-confidence hypothesis $((z^\star, e), \beta^\star, \rho^\star)$ into executable Python code that instantiates a predefined event class (based on Pydantic BaseModel).
- **Mechanism**: Compiles the event schema into a Python class definition, such that schema compliance is equivalent to constructing a valid class instance—verifiable deterministically at runtime.
- **Design Motivation**: By leveraging the type system and runtime validation of a programming language, the structural fidelity problem is converted from a natural language constraint into a programmatic constraint, transforming validation from probabilistic to deterministic.

#### 4. Verification Agent

- **Function**: Executes a three-stage test suite on the generated code object $C_{obj}$, returning a binary verdict $V$ and diagnostic information $\varepsilon$.
- **Three-Stage Verification**:
    - **Semantic Check $\mathcal{T}_1$**: Verifies that the predicted trigger $z^\star$ appears in the input text and is semantically compatible with the event type.
    - **Type Check $\mathcal{T}_2$**: Verifies that each argument value conforms to the data type specified in the schema, with multi-value constraints validated via Pydantic.
    - **Structural Check $\mathcal{T}_3$**: Confirms that the code is compilable, contains exactly three fields (event_type, trigger, arguments), and can be serialized.
- **Design Motivation**: Compiler-style deterministic feedback replaces vague natural-language feedback; verification failures yield precise error localization to guide subsequent correction.

#### 5. Dual-Loop Refinement Algorithm

- **Function**: Upon verification failure, two levels of iteration are executed—the inner loop attempts up to $t$ patch corrections on the current hypothesis; if the inner loop is exhausted without success, the outer loop falls back to the next highest-confidence hypothesis for re-coding.
- **Mechanism**: Traverses $O(kt)$ candidate paths to ensure the final output satisfies both semantic correctness and structural consistency.
- **Design Motivation**: Single-pass generation cannot guarantee quality. The dual-loop architecture of multiple hypotheses and multiple attempts improves both recall and structural fidelity, with effects saturating at $k=t=3$ and computational overhead remaining manageable.

## Key Experimental Results

### Main Results (Llama3 Series, F1 %)

| Method | FewEvent TI | FewEvent TC | ACE TI | ACE TC | ACE AI | ACE AC | GENIA TC | CASIE TC |
|--------|:-----------:|:-----------:|:------:|:------:|:------:|:------:|:--------:|:--------:|
| DirectEE (8B) | 21.5 | 17.5 | 26.4 | 25.7 | - | - | 34.3 | 11.8 |
| ChatIE (8B) | - | 24.8 | - | 44.2 | 32.4 | 30.8 | - | - |
| **AEC (8B)** | **27.0** | **27.6** | **40.5** | **48.8** | **33.7** | **31.8** | **41.8** | **16.5** |
| DirectEE (70B) | 32.1 | 30.3 | 50.7 | 46.9 | - | - | 44.7 | 13.5 |
| ChatIE (70B) | - | 40.7 | - | 47.5 | 36.6 | 34.5 | - | - |
| **AEC (70B)** | **42.1** | **40.5** | **57.0** | **54.6** | **38.4** | **34.7** | **52.3** | **18.7** |

### Ablation Study (Llama3-70B, FewEvent & ACE, F1 %)

| Ablation Setting | FewEvent TI | FewEvent TC | ACE TI | ACE TC | ACE AI | ACE AC |
|------------------|:-----------:|:-----------:|:------:|:------:|:------:|:------:|
| AEC (Full) | 42.1 | 40.5 | 57.0 | 54.6 | 38.4 | 34.7 |
| w/o Retrieval Agent | 36.5 | 34.2 | 49.8 | 47.2 | 33.1 | 30.8 |
| w/o Planning Rationales | 38.2 | 36.0 | 52.6 | 50.7 | 35.6 | 32.8 |
| w/o Verification Loop | 35.0 | 32.5 | 47.1 | 44.7 | 30.7 | 28.5 |
| w/o Structural Check | 39.8 | 37.6 | 54.9 | 52.5 | 37.2 | 33.6 |

### Cross-LLM Generalization (ACE TC, F1 %)

| LLM | GuidelineEE | DecomposeEE | **AEC** |
|-----|:-----------:|:-----------:|:-------:|
| Qwen2.5-14B | 33.5 | 38.8 | **45.3** |
| Qwen2.5-72B | 50.2 | 53.7 | **58.0** |
| GPT-3.5-turbo | 37.9 | 45.8 | **50.1** |
| GPT-4o | 55.9 | 58.4 | **61.8** |

## Key Findings

1. **AEC achieves best performance across all 5 domains and 6 LLMs**, with particularly pronounced advantages on schema-complex datasets (ACE, CASIE); on ACE TC, it surpasses the strongest baseline ChatIE by +7.1% (70B).
2. **The verification loop contributes most**: removing it causes the sharpest performance drop (ACE TC: 54.6→44.7, a decline of −9.9), confirming that compiler-style validation is the core driver of AEC.
3. **The Retrieval Agent is critical for disambiguation**: its removal causes a −5.6 drop on FewEvent TI; self-generated examples effectively compensate for the lack of contextual reference in the zero-shot setting.
4. **Hyperparameters $k$ and $t$ saturate at $k=t=3$**: further increases yield negligible gains (ACE TC on GPT-4o: 61.8→62.3) while computational cost grows linearly.
5. **Model scale effects are prominent**: GPT-4o > Qwen2.5-72B > Llama3-70B > smaller models; AEC consistently benefits from stronger backbone models.

## Highlights & Insights

- The analogy of **"extracting events like writing code"** is highly elegant: representing schemas as Python classes, equating validation with unit testing, and equating correction with debugging establishes a bridge between NLP structured prediction and software engineering.
- **Deterministic validation vs. probabilistic assessment** is the central design philosophy of this work: natural language feedback is inherently ambiguous, whereas type systems and compiler diagnostics are definitive—the latter fundamentally addresses the structural fidelity problem of LLM outputs.
- The **dual-loop fallback mechanism** elegantly handles the case where the first hypothesis may be incorrect, preserving multi-hypothesis exploration without sacrificing efficiency ($O(kt)$ complexity).
- Qualitative analysis shows that each agent's focus is orthogonal and complementary: Planning handles trigger selection, Coding fills in argument slots, and Verification corrects type errors—together forming a progressive quality improvement pipeline.

## Limitations & Future Work

1. **Computational overhead**: Four agents executing sequentially with dual-loop iterations require multiple LLM calls per sample, making inference efficiency substantially lower than single-step methods and hindering large-scale deployment.
2. **Serial inter-agent communication**: The current architecture is a linear pipeline between agents, leaving potential for parallel or asynchronous execution unexplored.
3. **No quality guarantee for self-generated examples**: Examples produced by the Retrieval Agent are LLM-generated and may introduce bias or hallucination, particularly in highly specialized domains (e.g., biomedicine).
4. **Evaluation limited to English datasets**: Cross-lingual generalization ability remains unknown.
5. **Overlapping and nested event scenarios are not discussed**: The pipeline's behavior when a sentence contains multiple interrelated events is unclear.
6. **Fixed hyperparameter setting $k=t=3$**: Although experiments suggest saturation, optimal values may vary across domains and models, and no adaptive tuning mechanism is provided.

## Related Work & Insights

- **Code4Struct / GuidelineEE**: Pioneer work on representing event extraction as code generation; AEC builds upon this by introducing multi-agent collaboration and deterministic validation.
- **ChatIE**: A multi-turn dialogue-based extraction approach; AEC replaces unstructured dialogue with a structured agent pipeline.
- **CMAS (NER)**: The most closely related multi-agent IE work, but focused on NER and lacking a code verification mechanism.
- **Insights**: The "like code" paradigm is generalizable to other structured prediction tasks (e.g., relation extraction, slot filling), with the core idea being the use of programming language type systems as deterministic output constraints. The architectural pattern of multi-agent collaboration combined with code verification is a direction worth following for agent-based IE research.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First work to model ZSEE as a multi-agent code generation task; the schema-as-code validation idea is highly innovative
- Experimental Thoroughness: ⭐⭐⭐⭐ 5 domains + 6 LLMs + full ablation + hyperparameter analysis, with broad coverage
- Writing Quality: ⭐⭐⭐⭐ Architecture diagrams are clear, algorithm pseudocode is well-formatted, and each component's responsibilities are well-defined
- Value: ⭐⭐⭐⭐⭐ Introduces a new paradigm for zero-shot structured information extraction with strong generalization potential

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] MARS2: Scaling Multi-Agent Tree Search via Reinforcement Learning for Code Generation](../../ACL2026/code_intelligence/mars2_scaling_multi-agent_tree_search_via_reinforcement_learning_for_code_genera.md)
- [\[ICCV 2025\] TikZero: Zero-Shot Text-Guided Graphics Program Synthesis](../../ICCV2025/code_intelligence/tikzero_zero-shot_text-guided_graphics_program_synthesis.md)
- [\[ICLR 2026\] CARD: Towards Conditional Design of Multi-agent Topological Structures](../../ICLR2026/code_intelligence/card_towards_conditional_design_of_multi-agent_topological_structures.md)
- [\[AAAI 2026\] EquaCode: A Multi-Strategy Jailbreak Approach for Large Language Models via Equation Solving and Code Completion](equacode_a_multi-strategy_jailbreak_approach_for_large_language_models_via_equat.md)
- [\[NeurIPS 2025\] A Stochastic Differential Equation Framework for Multi-Objective LLM Interactions](../../NeurIPS2025/code_intelligence/a_stochastic_differential_equation_framework_for_multi-objective_llm_interaction.md)

</div>

<!-- RELATED:END -->
