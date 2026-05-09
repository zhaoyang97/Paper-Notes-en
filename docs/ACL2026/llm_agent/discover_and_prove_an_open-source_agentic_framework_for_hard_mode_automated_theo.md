---
title: >-
  [Paper Note] Discover and Prove: An Open-source Agentic Framework for Hard Mode Automated Theorem Proving in Lean 4
description: >-
  [ACL 2026][LLM Agent][To be supplemented] To be supplemented after thorough reading.
tags:
  - ACL 2026
  - LLM Agent
  - To be supplemented
date: 2026-05-08
content_hash: 4b3eb014ec829e12
---

# Discover and Prove: An Open-source Agentic Framework for Hard Mode Automated Theorem Proving in Lean 4

**Conference**: ACL 2026
**arXiv**: [2604.15839](https://arxiv.org/abs/2604.15839)
**Code**: [GitHub](https://github.com/liuchengwucn/discover-and-prove)
**Area**: LLM Agent
**Keywords**: Automated Theorem Proving, Hard Mode, Lean 4, Answer Discovery, Formal Verification

## TL;DR
DAP introduces the concept of Hard Mode ATP (where AI must independently discover answers before constructing proofs, as opposed to Easy Mode statements with embedded answers), releases the MiniF2F-Hard and FIMO-Hard benchmarks, and proposes a two-stage "Discover and Prove" framework — using LLM natural language reasoning to discover answers, then rewriting the statement into an Easy Mode declaration for a formal prover. The framework improves solved problems on CombiBench from 7 to 10 and, for the first time, proves 36 theorems on PutnamBench in Hard Mode.

## Background & Motivation

**Background**: Automated Theorem Proving (ATP) has advanced rapidly, with Seed-Prover approaching saturation on MiniF2F. However, existing benchmarks predominantly adopt "Easy Mode" — embedding the final answer directly into formal statements — which reduces task difficulty, since human competitors must discover answers on their own.

**Limitations of Prior Work**: (1) Easy Mode substantially reduces problem difficulty — for many competition problems, discovering the answer is the primary challenge, while the subsequent proof is comparatively straightforward once the answer is known; (2) Some formal statements are not fully aligned with the original problem semantics — e.g., only one direction of an implication is proved when the original requires a biconditional; (3) LLMs achieve over 80% answer accuracy in informal reasoning, yet formal provers succeed on fewer than 10% of problems, revealing a substantial capability gap.

**Key Challenge**: Easy Mode causes ATP benchmarks to overestimate AI mathematical capability by omitting the "discovery" phase, which is the most challenging aspect of human mathematical problem solving.

**Goal**: (1) Establish fairer Hard Mode ATP benchmarks; (2) Design a framework capable of handling Hard Mode problems.

**Key Insight**: Decompose Hard Mode problems into two steps — first use informal LLM reasoning to discover the answer (Discovery), then use a formal prover to construct the proof (Proving) — mirroring the workflow of human mathematicians.

**Core Idea**: Decouple "answer discovery" from "proof construction" — leveraging the strong informal reasoning capability of LLMs to compensate for the limitations of formal provers.

## Method

### Overall Architecture
A two-module pipeline: (1) Discovery Module — a reasoning LLM generates natural language solution steps, followed by self-verification and self-correction, after which the Hard Mode statement is rewritten into an Easy Mode declaration (with the discovered answer filled in); (2) Proving Module — the rewritten Easy Mode statement is passed to an off-the-shelf ATP prover (Goedel-Prover-V2) to generate a formal proof.

### Key Designs

1. **Discovery Module (Answer Discovery)**:

    - Function: Independently discover the answer from a mathematical problem.
    - Mechanism: A four-step pipeline — (a) Solution generation: a reasoning LLM (GPT-OSS-120B) generates a detailed chain-of-thought solution; (b) Self-verification: the LLM checks its own steps for potential errors and produces an error report; (c) Self-correction: a revised solution is generated based on the error report (executed only when errors are detected); (d) Rewriting: the discovered answer is inserted into the first `sorry` placeholder of the Hard Mode Lean 4 statement, yielding an Easy Mode declaration.
    - Design Motivation: Separating discovery from proving allows each stage to employ the strongest available tool — a powerful reasoning LLM for informal reasoning and a dedicated ATP system for formal verification.

2. **Hard Mode Benchmark Curation**:

    - Function: Provide a fairer evaluation standard for ATP.
    - Mechanism: Experts re-annotate MiniF2F and FIMO — encoding "answer-dependent" problems with two `sorry` placeholders (the first for the answer, the second for the proof), correcting known semantic alignment issues (e.g., Easy Mode statements that only prove one direction of a biconditional), and providing a Lean 4 version of FIMO.
    - Design Motivation: Easy Mode formalizations may be semantically weaker than the original problems (e.g., omitting reachability proofs), while Hard Mode ensures AI faces the same task as human competitors.

3. **Modular Decoupling and Extensibility**:

    - Function: Allow independent upgrading of each component.
    - Mechanism: The Discovery Module and Proving Module use different LLMs and are mutually independent. Advances in any reasoning model or ATP system directly improve DAP's overall performance.
    - Design Motivation: LLMs are progressing rapidly in informal reasoning while formal proving remains limited; the decoupled design maximally exploits advances in both directions.

### Loss & Training
No training is involved. The Discovery Module is driven by carefully designed prompts, and the Proving Module uses the pretrained Goedel-Prover-V2.

## Key Experimental Results

### Main Results

| Benchmark | Method | Solved | Note |
|-----------|--------|--------|------|
| CombiBench Hard | Prev. SOTA (Kimina) | 7–8 | Pass@16 |
| CombiBench Hard | **DAP** | **10** | New SOTA |
| PutnamBench Hard | Prior work | 0 (no published results) | First evaluation |
| PutnamBench Hard | **DAP** | **36** | First Hard Mode result |

### Ablation Study

| Configuration | Note |
|---------------|------|
| Discovery only (no Proving) | Answer accuracy >80%, but no formal guarantees |
| Proving only (Easy Mode) | Formal proof rate <10% |
| Discovery + Proving | Complementary; significantly more theorems proved |

### Key Findings
- LLM answer accuracy in informal reasoning (>80%) far exceeds the formal proof rate (<10%), revealing the unique measurement value of Hard Mode benchmarks.
- The Discovery Module contributes most to overall performance — incorrect answer discovery directly yields unprovable statements.
- Self-verification and self-correction steps substantially improve answer accuracy.
- The performance gap between Easy Mode and Hard Mode is more pronounced on harder problems.

## Highlights & Insights
- **The Easy Mode vs. Hard Mode distinction** is an important methodological contribution to ATP evaluation — exposing the potential over-optimism of existing benchmarks.
- The gap between 80% informal reasoning accuracy and 10% formal proof rate quantifies the divide between "knowing the answer" and "rigorously proving it."
- The DAP framework is simple yet effective — requiring no complex search or RL training, relying solely on prompt engineering and modular decoupling.

## Limitations & Future Work
- The Discovery Module's answer accuracy is not 100%; unprovable statements resulting from incorrect answers waste prover computation.
- Only a single reasoning LLM is used; integrating multiple reasoning models could improve answer discovery rates.
- The reliability of self-verification is limited — LLMs may fail to detect subtle reasoning errors in their own outputs.

## Related Work & Insights
- **vs. DSP/DSP+**: DSP uses natural language drafts to guide formal proofs, whereas DAP uses natural language reasoning to discover answers and then rewrites the statement — DAP directly addresses the answer discovery challenge in Hard Mode.
- **vs. Seed-Prover**: Seed-Prover is a lemma-style full-proof reasoning model; DAP decouples discovery and proving, offering greater flexibility.
- **vs. AlphaProof**: AlphaProof uses reinforcement learning; DAP is fully open-source and prompt-based, making it more reproducible.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ A unified contribution of Hard Mode ATP concept, benchmarks, and framework.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-benchmark evaluation with ablations, though dataset scale is limited.
- Writing Quality: ⭐⭐⭐⭐⭐ The motivation for the Easy/Hard Mode distinction is articulated with exceptional clarity.
- Value: ⭐⭐⭐⭐⭐ Significant implications for both ATP evaluation methodology and practice.

**Code**: To be confirmed
**Area**: human_understanding
**Keywords**: To be supplemented

## TL;DR
To be supplemented after thorough reading.

## Background & Motivation
To be supplemented after thorough reading.

## Method
To be supplemented after thorough reading.

## Key Experimental Results
To be supplemented after thorough reading.

## Highlights & Insights
To be supplemented after thorough reading.

## Limitations & Future Work
To be supplemented after thorough reading.

## Related Work & Insights
To be supplemented after thorough reading.

## Rating
- Novelty: Pending
- Experimental Thoroughness: Pending
- Writing Quality: Pending
- Value: Pending

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] HeLa-Mem: Hebbian Learning and Associative Memory for LLM Agents](hela-mem_hebbian_learning_and_associative_memory_for_llm_agents.md)
- [\[ACL 2026\] ToolOmni: Enabling Open-World Tool Use via Agentic Learning with Proactive Retrieval and Grounded Execution](toolomni_enabling_open-world_tool_use_via_agentic_learning_with_proactive_retrie.md)
- [\[NeurIPS 2025\] Benchmarking Agentic Systems in Automated Scientific Information Extraction with ChemX](../../NeurIPS2025/llm_agent/benchmarking_agentic_systems_in_automated_scientific_information_extraction_with.md)
- [\[NeurIPS 2025\] Automated Composition of Agents: A Knapsack Approach for Agentic Component Selection](../../NeurIPS2025/llm_agent/automated_composition_of_agents_a_knapsack_approach_for_agentic_component_select.md)
- [\[AAAI 2026\] Loss-Guided Auxiliary Agents for Overcoming Mode Collapse in GFlowNets](../../AAAI2026/llm_agent/loss-guided_auxiliary_agents_for_overcoming_mode_collapse_in_gflownets.md)

<!-- RELATED:END -->
