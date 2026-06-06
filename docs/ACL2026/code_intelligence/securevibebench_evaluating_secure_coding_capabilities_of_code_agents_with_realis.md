---
title: >-
  [Paper Note] SecureVibeBench: Evaluating Secure Coding Capabilities of Code Agents with Realistic Vulnerability Scenarios
description: >-
  [ACL 2026][Code Intelligence][Secure coding] Ours proposes SecureVibeBench, the first repository-level multi-file editing benchmark for secure coding. It constructs 105 C/C++ secure coding tasks from 41 OSS-Fuzz projects…
tags:
  - "ACL 2026"
  - "Code Intelligence"
  - "Secure coding"
  - "code agents"
  - "vulnerability introduction"
  - "benchmark"
  - "repository-level code generation"
date: 2026-05-08
content_hash: d00149cb7d6e9a15
---

# SecureVibeBench: Evaluating Secure Coding Capabilities of Code Agents with Realistic Vulnerability Scenarios

**Conference**: ACL 2026  
**arXiv**: [2509.22097](https://arxiv.org/abs/2509.22097)  
**Code**: [GitHub](https://github.com/iCSawyer/SecureVibeBench)  
**Area**: LLM Agent  
**Keywords**: Secure coding, code agents, vulnerability introduction, benchmark, repository-level code generation

## TL;DR

Ours proposes SecureVibeBench, the first repository-level multi-file editing benchmark for secure coding. It constructs 105 C/C++ secure coding tasks from 41 OSS-Fuzz projects, accurately recreating scenarios where vulnerabilities were first introduced through cascaded static and dynamic analysis. Evaluation reveals that the best agent (SWE-agent + Claude Sonnet 4.5) achieves both functional correctness and security in only 23.8% of cases.

## Background & Motivation

**Background**: LLM-driven code agents (e.g., SWE-agent, Claude Code) are rapidly transforming software engineering. However, the security of generated code is concerning—approximately 40% of GitHub Copilot code completions contain exploitable vulnerabilities.

**Limitations of Prior Work**: Existing secure coding benchmarks suffer from three critical deficiencies: (1) Task format: most are function-level code completions, failing to reflect realistic repository-level multi-file editing scenarios; (2) Context alignment: scenarios synthesized from CWE catalogs do not align with the actual code versions and requirements present when human developers introduced vulnerabilities; (3) Evaluation: some benchmarks ignore functional correctness, and almost all overlook the potential for agents to introduce entirely new security risks.

**Key Challenge**: To fairly compare the secure coding capabilities of humans and agents, agents must be placed in the same scenarios where humans actually introduced vulnerabilities—but such a benchmark has been lacking.

**Goal**: Build a repository-level secure coding benchmark based on realistic vulnerability introduction scenarios to comprehensively evaluate both the functional correctness and security of agents.

**Key Insight**: Precisely trace back to the commit where a vulnerability was first introduced into the codebase via cascaded static and dynamic analysis to restore the original requirements and code version.

**Core Idea**: Shift secure coding evaluation from "whether agents can avoid known vulnerability patterns" to "whether agents repeat human mistakes or introduce new risks when placed in the same scenarios where humans introduced vulnerabilities."

## Method

### Overall Architecture

The construction workflow of SecureVibeBench involves: (1) Collecting 4,993 vulnerability instances from ARVO and OSS-Fuzz; (2) Tracing back to the vulnerability introduction commit through cascaded static and dynamic analysis; (3) Extracting requirement descriptions and code versions from that commit to build tasks; (4) Isolating project environments using Docker; (5) A four-dimensional evaluation: functional correctness (differential testing) + known vulnerabilities (PoV verification) + new security risks (SAST detection).

### Key Designs

1.  **Vulnerability Introduction Identification**:
    - **Function**: Accurately identifies the specific commit where human developers first introduced a vulnerability.
    - **Mechanism**: A cascaded two-stage analysis—first using SAST (CodeQL/Semgrep) to quickly locate a candidate range of commits, followed by dynamic verification with PoV programs. For cases not covered by SAST, binary search combined with dynamic verification is used.
    - **Design Motivation**: The commit immediately preceding a fix is often not the introduction point (vulnerabilities are usually introduced much earlier). Using the actual introduction point restores the exact coding scenario faced by humans.

2.  **Four-category Classification of Results**:
    - **Function**: Comprehensively classifies the quality of agent-generated code.
    - **Mechanism**: Agent outputs are categorized into four types: IC (Incorrent), C-VUL (Correct but contains known vulnerability), C-SUS (Correct but introduces new security risks), and C-SEC (Correct and Secure). Functional correctness is assessed via differential testing, while security is evaluated using PoV validation for known vulnerabilities and SAST for new risks.
    - **Design Motivation**: Simply detecting known vulnerabilities is insufficient—agents might avoid the original bug while introducing entirely new security issues.

3.  **Repository-level Multi-file Editing Task Format**:
    - **Function**: Reflects realistic software maintenance scenarios.
    - **Mechanism**: Given a repository and a natural language requirement, agents must perform edits across multiple files to implement functionality. The 105 tasks originate from 41 projects with large average repository sizes.
    - **Design Motivation**: Function-level completion is too far removed from real-world programming; repository-level editing captures the actual security challenges of AI-assisted programming.

## Key Experimental Results

### Main Results

| Agent + LLM | C-SEC (Correct & Secure) | C-VUL | C-SUS | IC |
| :--- | :--- | :--- | :--- | :--- |
| SWE-agent + Claude Sonnet 4.5 | **23.8%** | — | — | — |
| OpenHands + Claude Sonnet 4.5 | ~20% | — | — | — |
| Claude Code | ~18% | — | — | — |
| Codex | ~15% | — | — | — |

### Key Findings
- The best agent satisfies both functional and security standards in only 23.8% of cases, indicating that secure coding is a major weakness for current agents.
- Different agents and models exhibit different failure modes—some are functionally correct but insecure, while others are secure but functionally incorrect.
- Agents demonstrate some ability to avoid the original human vulnerability but frequently introduce brand-new security risks (the proportion of C-SUS is significant).
- Functional correctness is a prerequisite for security evaluation—a large amount of code fails at the functional level first.

## Highlights & Insights
- **Perspective Innovation**: By placing agents in the same scenarios where humans introduced vulnerabilities, it achieves the first truly fair human-vs-agent secure coding comparison.
- **Valuable Backtracking Method**: The cascaded static and dynamic analysis for locating introduction commits is highly effective and reusable for other security research.
- **Comprehensive Evaluation**: The four-category classification, PoV dynamic validation, and SAST-based new risk detection provide a more complete assessment than existing benchmarks.
- **Impactful 23.8% Result**: Clearly demonstrates the severe state of AI secure coding in practice.

## Limitations & Future Work
- **Language Coverage**: Currently only covers C/C++; security patterns in other languages may differ.
- **SAST False Positives**: The C-SUS category may include false positives from static analysis tools.
- **Task Scale**: With 105 tasks, the benchmark size could be further expanded.
- **Future Directions**: Expanding to more languages and vulnerability types, and investigating security-aware code generation strategies.

## Related Work & Insights
- **vs BaxBench**: While BaxBench constructs backend code from scratch to evaluate security, SecureVibeBench focuses on the evolution of existing codebases, making them complementary.
- **vs SusVibes**: A concurrent work with a similar task format, but it does not consider realistic vulnerability introduction scenarios or the detection of new security risks.
- **vs SecRepoBench**: Although expanded to the repository level, it remains limited to single-function completion formats.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ First repository-level secure coding benchmark with a unique perspective on vulnerability introduction backtracking.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers 5 agents and 5 LLMs with a complete evaluation framework, though the task count (105) is relatively small.
- **Writing Quality**: ⭐⭐⭐⭐ Clear problem definitions and thorough comparisons with prior work.
- **Value**: ⭐⭐⭐⭐⭐ Significantly advances AI secure coding research; the 23.8% result serves as a critical warning for the industry.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] CodeDistiller: Automatically Generating Code Libraries for Scientific Coding Agents](codedistiller_automatically_generating_code_libraries_for_scientific_coding_agen.md)
- [\[ACL 2026\] DeepGuard: Secure Code Generation via Multi-Layer Semantic Aggregation](deepguard_secure_code_generation_via_multi-layer_semantic_aggregation.md)
- [\[ACL 2026\] RExBench: Can coding agents autonomously implement AI research extensions?](rexbench_can_coding_agents_autonomously_implement_ai_research_extensions.md)
- [\[ICML 2026\] NEMO: Execution-Aware Optimization Modeling via Autonomous Coding Agents](../../ICML2026/code_intelligence/nemo_execution-aware_optimization_modeling_via_autonomous_coding_agents.md)
- [\[ACL 2026\] Can LLMs Compress (and Decompress)? Evaluating Code Understanding and Execution via Invertibility](can_llms_compress_and_decompress_evaluating_code_understanding_and_execution_via.md)

</div>

<!-- RELATED:END -->
