---
title: >-
  [Paper Note] SecureVibeBench: Evaluating Secure Coding Capabilities of Code Agents with Realistic Vulnerability Scenarios
description: >-
  [ACL 2026][LLM Agent][Secure coding] This paper presents SecureVibeBench, the first repository-level, multi-file-editing secure coding benchmark. It constructs 105 C/C++ secure coding tasks from 41 OSS-Fuzz projects, precisely reconstructing the scenarios in which vulnerabilities were first introduced via cascaded static and dynamic analysis. Evaluation results reveal that the best-performing agent (SWE-agent + Claude Sonnet 4.5) produces code that is simultaneously functionally correct and secure in only 23.8% of cases.
tags:
  - ACL 2026
  - LLM Agent
  - Secure coding
  - code agents
  - vulnerability introduction
  - benchmark
  - repository-level code generation
date: 2026-05-08
content_hash: ea8be9775489c7d0
---

# SecureVibeBench: Evaluating Secure Coding Capabilities of Code Agents with Realistic Vulnerability Scenarios

**Conference**: ACL 2026
**arXiv**: [2509.22097](https://arxiv.org/abs/2509.22097)
**Code**: [GitHub](https://github.com/iCSawyer/SecureVibeBench)
**Area**: LLM Agent
**Keywords**: Secure coding, code agents, vulnerability introduction, benchmark, repository-level code generation

## TL;DR

This paper presents SecureVibeBench, the first repository-level, multi-file-editing secure coding benchmark. It constructs 105 C/C++ secure coding tasks from 41 OSS-Fuzz projects, precisely reconstructing the scenarios in which vulnerabilities were first introduced via cascaded static and dynamic analysis. Evaluation results reveal that the best-performing agent (SWE-agent + Claude Sonnet 4.5) produces code that is simultaneously functionally correct and secure in only 23.8% of cases.

## Background & Motivation

**Background**: LLM-driven code agents (e.g., SWE-agent, Claude Code) are rapidly transforming software engineering, yet the security of generated code remains a serious concern—approximately 40% of GitHub Copilot completions contain exploitable vulnerabilities.

**Limitations of Prior Work**: Existing secure coding benchmarks suffer from three critical shortcomings: (1) *Task format*: most tasks are function-level code completions that do not reflect real-world repository-level, multi-file editing scenarios; (2) *Context alignment*: tasks are often synthesized from CWE catalogs and thus fail to align with the actual code versions and requirements under which human developers introduced vulnerabilities; (3) *Evaluation*: some benchmarks disregard functional correctness, and nearly all overlook the possibility that agents may introduce entirely new security risks.

**Key Challenge**: A fair comparison of human and agent secure coding capabilities requires placing agents in the exact same scenarios where humans historically introduced vulnerabilities—yet no such benchmark previously existed.

**Goal**: To construct a repository-level secure coding benchmark grounded in realistic vulnerability introduction scenarios, enabling comprehensive evaluation of both functional correctness and security.

**Key Insight**: Cascaded static and dynamic analysis is employed to precisely trace back to the commit at which a vulnerability was first introduced into the codebase, thereby reconstructing the requirements and code state at that point in time.

**Core Idea**: Shift the secure coding evaluation paradigm from "can the agent avoid known vulnerability patterns?" to "when placed in the same scenario where a human introduced a vulnerability, does the agent repeat the mistake or introduce new risks?"

## Method

### Overall Architecture

The construction pipeline of SecureVibeBench proceeds as follows: (1) collect 4,993 vulnerability instances from ARVO and OSS-Fuzz; (2) trace vulnerability-introducing commits via cascaded static and dynamic analysis; (3) extract the corresponding requirement descriptions and code states to formulate tasks; (4) isolate project environments using Docker; (5) perform four-dimensional evaluation: functional correctness (differential testing) + known vulnerabilities (PoV validation) + new security risks (SAST detection).

### Key Designs

1. **Vulnerability Introduction Identification**

    - *Function*: Precisely locate the commit at which a human developer first introduced a vulnerability.
    - *Mechanism*: A cascaded two-stage approach is applied—SAST tools (CodeQL/Semgrep) first perform static analysis to rapidly narrow down candidate commits, followed by dynamic validation using PoV programs for confirmation. Binary search combined with dynamic validation is used for cases beyond static analysis coverage.
    - *Design Motivation*: The commit immediately preceding a fix is not necessarily the vulnerability-introducing commit, as vulnerabilities are typically introduced much earlier. Using the true introduction point is essential to reconstruct the same coding scenario faced by the human developer.

2. **Four-Category Evaluation Schema**

    - *Function*: Comprehensively classify the quality of agent-generated code.
    - *Mechanism*: Agent outputs are classified into four categories—IC (functionally incorrect), C-VUL (correct but containing the known vulnerability), C-SUS (correct but introducing new security risks), and C-SEC (correct and secure). Functional correctness is assessed via differential testing; security is evaluated by PoV validation for known vulnerabilities and SAST for newly introduced risks.
    - *Design Motivation*: Detecting only known vulnerabilities is insufficient—an agent may avoid the original vulnerability while simultaneously introducing entirely new security issues.

3. **Repository-Level Multi-File Editing Task Format**

    - *Function*: Reflect realistic software maintenance scenarios.
    - *Mechanism*: Given a repository and a natural language requirement description, the agent must perform edits across multiple files to implement the desired functionality. The 105 tasks span 41 projects, with repositories of substantial average scale.
    - *Design Motivation*: Function-level completion diverges greatly from real programming practice; repository-level multi-file editing better captures the security challenges of actual AI-assisted software development.

## Key Experimental Results

### Main Results

| Agent + LLM | C-SEC (Correct & Secure) | C-VUL | C-SUS | IC |
|---|---|---|---|---|
| SWE-agent + Claude Sonnet 4.5 | **23.8%** | — | — | — |
| OpenHands + Claude Sonnet 4.5 | ~20% | — | — | — |
| Claude Code | ~18% | — | — | — |
| Codex | ~15% | — | — | — |

### Key Findings

- The best-performing agent achieves only 23.8% C-SEC, indicating that secure coding remains a critical weakness of current agents.
- Different agents and models exhibit distinct failure modes—some produce functionally correct but insecure code, while others are secure but functionally incorrect.
- Agents demonstrate some capacity to avoid the original vulnerability, yet frequently introduce entirely new security risks (the proportion of C-SUS is non-negligible).
- Functional correctness is a prerequisite for security evaluation—a substantial portion of generated code fails at the functional level.

## Highlights & Insights

- **Perspective novelty**: Placing agents in the exact scenarios where humans introduced vulnerabilities enables, for the first time, a genuinely fair human-agent comparison in secure coding.
- **Value of vulnerability introduction tracing**: The cascaded static and dynamic analysis method for precisely locating vulnerability-introducing commits is reusable for other security research endeavors.
- **Comprehensive evaluation**: The four-category classification scheme, combined with PoV dynamic validation and SAST-based new risk detection, is more complete than existing benchmarks.
- **Impact of the 23.8% result**: This figure compellingly illustrates the severity of the AI secure coding challenge.

## Limitations & Future Work

- **C/C++ only**: Security patterns in other programming languages may differ substantially.
- **SAST false positives**: The C-SUS category may include false positives.
- **Limited task count**: 105 tasks is a relatively small scale; a larger benchmark would be preferable.
- Future directions include extending coverage to more languages and vulnerability types, and investigating security-aware code generation strategies.

## Related Work & Insights

- **vs. BaxBench**: BaxBench evaluates security by constructing backend code from scratch, complementing SecureVibeBench's focus on the evolution of existing codebases.
- **vs. SusVibes**: A concurrent work with a similar task format, but without consideration of realistic vulnerability introduction scenarios or detection of newly introduced security risks.
- **vs. SecRepoBench**: Although SecRepoBench extends evaluation to the repository level, it remains limited to single-function completion tasks.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First repository-level secure coding benchmark; the vulnerability introduction tracing perspective is highly original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers 5 agents and 5 LLMs with a complete evaluation framework, though 105 tasks is a relatively small scale.
- **Writing Quality**: ⭐⭐⭐⭐ — Problem formulation is clear and comparisons with prior work are thorough.
- **Value**: ⭐⭐⭐⭐⭐ — Provides an important impetus for AI secure coding research; the 23.8% result serves as a significant warning for industry.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] LiveNewsBench: Evaluating LLM Web Search Capabilities with Freshly Curated News](../../ICLR2026/llm_agent/livenewsbench_evaluating_llm_web_search_capabilities_with_freshly_curated_news.md)
- [\[ACL 2026\] CodeStruct: Code Agents over Structured Action Spaces](codestruct_code_agents_over_structured_action_spaces.md)
- [\[AAAI 2026\] LieCraft: A Multi-Agent Framework for Evaluating Deceptive Capabilities in Language Models](../../AAAI2026/llm_agent/liecraft_a_multi-agent_framework_for_evaluating_deceptive_capabilities_in_langua.md)
- [\[ICLR 2026\] FeatureBench: Benchmarking Agentic Coding for Complex Feature Development](../../ICLR2026/llm_agent/featurebench_benchmarking_agentic_coding_for_complex_feature_development.md)
- [\[AAAI 2026\] SoMe: A Realistic Benchmark for LLM-based Social Media Agents](../../AAAI2026/llm_agent/some_a_realistic_benchmark_for_llm-based_social_media_agents.md)

<!-- RELATED:END -->
