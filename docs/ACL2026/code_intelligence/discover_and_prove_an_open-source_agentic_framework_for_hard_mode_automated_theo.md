---
title: >-
  [Paper Note] Discover and Prove: An Open-source Agentic Framework for Hard Mode Automated Theorem Proving in Lean 4
description: >-
  [ACL 2026][Code Intelligence][Automated Theorem Proving] DAP proposes the concept of Hard Mode ATP (AI must discover the answer before constructing a proof…
tags:
  - "ACL 2026"
  - "Code Intelligence"
  - "Automated Theorem Proving"
  - "Hard Mode"
  - "Lean 4"
  - "Answer Discovery"
  - "Formal Verification"
date: 2026-05-08
content_hash: dff10ae2569b8b56
---

# Discover and Prove: An Open-source Agentic Framework for Hard Mode Automated Theorem Proving in Lean 4

**Conference**: ACL 2026  
**arXiv**: [2604.15839](https://arxiv.org/abs/2604.15839)  
**Code**: [GitHub](https://github.com/liuchengwucn/discover-and-prove)  
**Area**: LLM Agent  
**Keywords**: Automated Theorem Proving, Hard Mode, Lean 4, Answer Discovery, Formal Verification

## TL;DR
DAP proposes the concept of Hard Mode ATP (AI must discover the answer before constructing a proof, rather than using an Easy Mode statement with an embedded answer). It releases the MiniF2F-Hard and FIMO-Hard benchmarks and designs a "discovery + proving" two-stage framework. By using reasoning LLMs to discover answers via natural language and rewriting them into Easy Mode statements for a formal prover, DAP improves the number of solved problems on CombiBench from 7 to 10 and proves 36 theorems on PutnamBench Hard Mode for the first time.

## Background & Motivation

**Background**: Automated Theorem Proving (ATP) has achieved rapid progress, with Seed-Prover approaching saturation on MiniF2F. However, existing benchmarks commonly adopt "Easy Mode"—embedding the final answer into the formal statement—which reduces task difficulty, as human contestants must discover the answer themselves.

**Limitations of Prior Work**: (1) Easy Mode significantly reduces problem difficulty; for many competition problems, discovering the answer is the primary challenge, and proving it is relatively simple once the answer is known. (2) Some formal statements are not fully aligned with the original semantics (e.g., proving only one direction of an implication when the original problem requires a necessary and sufficient condition). (3) LLMs achieve over 80% answer accuracy in informal reasoning, but formal provers can handle less than 10%, exposing a massive capability gap.

**Key Challenge**: Easy Mode causes ATP benchmarks to overestimate AI's mathematical capabilities by omitting the most challenging "discovery" phase of human problem-solving.

**Goal**: (1) Establish a fairer Hard Mode ATP benchmark; (2) Design a framework capable of handling Hard Mode problems.

**Key Insight**: Decompose the Hard Mode problem into two steps—first using an informal LLM for answer discovery (Discovery) and then using a formal prover for proof construction (Proving), simulating the cognitive process of a human mathematician.

**Core Idea**: Decouple "finding the answer" and "constructing the proof"—using the strong informal reasoning capabilities of LLMs to compensate for the weaknesses of formal provers.

## Method

### Overall Architecture
A two-module pipeline: (1) Discovery Module—a reasoning LLM generates natural language solution steps, which, after self-verification and self-correction, rewrites the Hard Mode statement into Easy Mode (filling in the discovered answer); (2) Proving Module—passes the rewritten Easy Mode statement to an off-the-shelf ATP prover (Goedel-Prover-V2) to generate a formal proof.

### Key Designs

1. **Discovery Module (Answer Discovery)**:

    - **Function**: Independently discover answers from mathematical problems.
    - **Mechanism**: A four-step process—(a) Solution Generation: Reasoning LLM (GPT-OSS-120B) generates a detailed Chain-of-Thought solution; (b) Self-verification: The LLM checks its own steps for potential errors and generates an error report; (c) Self-correction: Generates a revised solution based on the error report (executed only if errors are found); (d) Rewriting: Fills the discovered answer into the first `sorry` placeholder of the Hard Mode Lean 4 statement to generate an Easy Mode statement.
    - **Design Motivation**: Separating discovery and proving allows each to use the strongest tools—informal reasoning uses strong reasoning LLMs, while formal proof uses specialized ATP systems.

2. **Hard Mode Benchmark Curation**:

    - **Function**: Provide a fairer standard for ATP evaluation.
    - **Mechanism**: Experts re-annotated MiniF2F and FIMO—encoding "answer-oriented" problems with two `sorry` placeholders (the first for the answer, the second for the proof), correcting known semantic alignment issues (e.g., where Easy Mode proves only one direction), and providing a Lean 4 version of FIMO.
    - **Design Motivation**: Easy Mode formalization may be semantically weaker than the original problem (e.g., omitting reachability proofs); Hard Mode ensures the tasks faced by AI are consistent with those faced by human contestants.

3. **Module Decoupling & Scalability**:

    - **Function**: Allow independent upgrades of components.
    - **Mechanism**: The Discovery Module and Proving Module use different LLMs and are not interdependent. Improvements in any reasoning model or ATP system directly enhance the overall performance of DAP.
    - **Design Motivation**: Currently, LLMs progress rapidly in informal reasoning while formal proof remains limited; the decoupled design maximizes the utilization of progress in both directions.

### Loss & Training
Training-free. The Discovery Module is driven by carefully designed prompts, and the Proving Module uses the pre-trained Goedel-Prover-V2.

## Key Experimental Results

### Main Results

| Benchmark | Method | Solved | Description |
|-----------|--------|--------|-------------|
| CombiBench Hard | Prev. SOTA (Kimina) | 7-8 | Pass@16 |
| CombiBench Hard | **Ours** | **10** | New SOTA |
| PutnamBench Hard | Previous | 0 (No public results) | First evaluation |
| PutnamBench Hard | **Ours** | **36** | First Hard Mode result |

### Ablation Study

| Configuration | Description |
|---------------|-------------|
| Discovery only (No Proving) | Answer accuracy >80%, but no formal guarantee |
| Proving only (Easy Mode) | Formal proof rate <10% |
| Discovery+Proving | Both are complementary, proof count increases significantly |

### Key Findings
- LLM answer accuracy in informal reasoning (>80%) far exceeds the formal proof rate (<10%), revealing the unique measurement value of the Hard Mode benchmark.
- DAP's Discovery Module contributes the most to final performance—incorrect answer discovery directly results in unprovable statements.
- Self-verification and self-correction steps significantly improve answer accuracy.
- The performance gap between Easy Mode and Hard Mode is more pronounced on difficult problems.

## Highlights & Insights
- **The distinction between Easy Mode and Hard Mode** is a significant contribution to ATP evaluation methodology—exposing how existing benchmarks might be overly optimistic.
- The gap between 80% informal reasoning and 10% formal proof quantifies the chasm between "knowing the answer" and "rigorous proof."
- The DAP framework design is simple yet effective—requiring no complex search or RL training, relying instead on prompt engineering and module decoupling.

## Limitations & Future Work
- The answer accuracy of the Discovery Module is not yet 100%; unprovable statements caused by incorrect answers waste prover computation.
- Only a single reasoning LLM is used; integrating multiple reasoning models might improve the answer discovery rate.
- The reliability of self-verification is limited—LLMs may fail to detect their own subtle reasoning errors.

## Related Work & Insights
- **vs DSP/DSP+**: DSP uses natural language drafts to guide formal proofs; DAP uses natural language to discover answers and then rewrites statements—DAP directly addresses answer discovery in Hard Mode.
- **vs Seed-Prover**: Seed-Prover is a lemma-style full-proof reasoning model; DAP decouples discovery and proving, offering greater flexibility.
- **vs AlphaProof**: AlphaProof uses reinforcement learning; DAP is fully open-source and prompt-based, making it more reproducible.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Triple contribution of Hard Mode ATP concept, benchmark, and framework.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluation on multiple benchmarks and ablation, though data scale is limited.
- Writing Quality: ⭐⭐⭐⭐⭐ The motivation for distinguishing Easy/Hard Mode is exceptionally clear.
- Value: ⭐⭐⭐⭐⭐ Significant impact on both ATP evaluation methodology and practice.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Benchmarking Testing in Automated Theorem Proving](benchmarking_testing_in_automated_theorem_proving.md)
- [\[ACL 2026\] FormalScience: Scalable Human-in-the-Loop Autoformalisation of Science with Agentic Code Generation in Lean](formalscience_scalable_human-in-the-loop_autoformalisation_of_science_with_agent.md)
- [\[ACL 2026\] LogicEval: A Systematic Framework for Evaluating Automated Repair Techniques for Logical Vulnerabilities in Real-World Software](logiceval_a_systematic_framework_for_evaluating_automated_repair_techniques_for_.md)
- [\[ACL 2026\] CuBridge: An LLM-Based Framework for Understanding and Reconstructing High-Performance Attention Kernels](cubridge_an_llm-based_framework_for_understanding_and_reconstructing_high-perfor.md)
- [\[AAAI 2026\] Why Do Open-Source LLMs Struggle with Data Analysis? A Systematic Empirical Study](../../AAAI2026/code_intelligence/why_do_open-source_llms_struggle_with_data_analysis_a_systematic_empirical_study.md)

</div>

<!-- RELATED:END -->
