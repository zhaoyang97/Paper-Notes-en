---
title: >-
  [Paper Note] LogicEval: A Systematic Framework for Evaluating Automated Repair Techniques for Logical Vulnerabilities in Real-World Software
description: >-
  [ACL 2026][Logical vulnerabilities] This paper presents LogicEval, the first systematic evaluation framework for logical vulnerability repair, along with the LogicDS dataset (61 real-world logical vulnerabilities + 61 synthetic Java samples). It systematically evaluates both traditional AVR tools and LLMs on logical vulnerability repair, finding that LLMs perform best when provided with auxiliary information yet overall repair rates remain low (only 5 out of 61 real-world samples correctly repaired). Key bottlenecks identified include prompt sensitivity, context loss, and patch localization difficulty.
tags:
  - ACL 2026
  - Logical vulnerabilities
  - automated repair evaluation
  - LLM-based code repair
  - patch generation
  - benchmark dataset
date: 2026-05-08
content_hash: cb2f3a6149577ca9
---

# LogicEval: A Systematic Framework for Evaluating Automated Repair Techniques for Logical Vulnerabilities in Real-World Software

**Conference**: ACL 2026
**arXiv**: [2604.12994](https://arxiv.org/abs/2604.12994)
**Code**: [GitHub](https://github.com/LogicEval)
**Area**: Code Intelligence / Vulnerability Repair Evaluation
**Keywords**: Logical vulnerabilities, automated repair evaluation, LLM-based code repair, patch generation, benchmark dataset

## TL;DR

This paper presents LogicEval, the first systematic evaluation framework for logical vulnerability repair, along with the LogicDS dataset (61 real-world logical vulnerabilities + 61 synthetic Java samples). It systematically evaluates both traditional AVR tools and LLMs on logical vulnerability repair, finding that LLMs perform best when provided with auxiliary information yet overall repair rates remain low (only 5 out of 61 real-world samples correctly repaired). Key bottlenecks identified include prompt sensitivity, context loss, and patch localization difficulty.

## Background & Motivation

**Background**: Logical vulnerabilities stem from incorrect implementations of program logic or functionality, rather than memory safety violations. They can be exploited for authentication bypass, sensitive data leakage, or system disruption, and do not trigger conventional security defenses such as address sanitizers. Existing automated vulnerability repair (AVR) techniques predominantly target memory corruption vulnerabilities.

**Limitations of Prior Work**: (1) Logical vulnerabilities lack consistent, reusable repair templates or patterns; each repair requires deep understanding of program semantics and intended behavior. (2) Logical vulnerabilities do not necessarily cause crashes or illegal memory accesses, making traditional signals (compilation logs, runtime logs, memory sanitizers) of limited utility for localization. (3) Existing datasets focus primarily on memory safety bugs, lacking logical vulnerability samples with demonstrable security impact.

**Key Challenge**: While LLMs have demonstrated strong capabilities in code understanding and generation, no systematic framework exists to analyze their ability and limitations in repairing logical vulnerabilities — a gap that impedes the extension of AVR from memory safety to the more subtle domain of logical vulnerabilities.

**Goal**: To construct the first systematic evaluation framework for analyzing the capabilities, limitations, and failure modes of both traditional and LLM-based methods when repairing real-world logical vulnerabilities.

**Key Insight**: Logical vulnerability repair is highly context-dependent — relying on vulnerability descriptions, behavioral specifications, and repair steps — making it appropriate to evaluate the impact of different auxiliary information dimensions systematically.

**Core Idea**: Construct the LogicDS dataset and LogicEval evaluation framework to systematically assess performance across three dimensions: LLM configuration, source code granularity, and auxiliary information. Introduce reasoning-based automatic evaluation metrics (cosine similarity + LLM judgment) to complement conventional compilation/test-based evaluation.

## Method

### Overall Architecture

LogicEval is an end-to-end evaluation pipeline: (1) **Input** — vulnerable source code $S$, fixed code $F$, vulnerability description $D$, behavioral specification $V_S$ (optional), context $V_{ctx}$ (optional), and compilation/test scripts; (2) **Patch Localization** — assuming perfect localization, the core repair region (single hunk) is manually identified; (3) **Patch Generation** — prompts are constructed along different dimensions to drive LLM patch generation, with tagged code extracted to replace the vulnerable region; (4) **Patch Evaluation** — compilation and testing, supplemented by reasoning-based automatic evaluation (semantic similarity between generated patch explanations and ground-truth repair explanations).

### Key Designs

1. **LogicDS Dataset Construction**:

    - Function: Provides the first logical vulnerability benchmark with demonstrated real-world security impact.
    - Mechanism: 61 real-world logical vulnerabilities are curated from CVEs across 28 popular open-source projects. Each sample includes vulnerable/fixed code, CVE description, manually localized core repair region, compilation scripts, and test cases. An additional 61 synthetic Java samples are constructed for compatibility with Java-specific repair tools.
    - Design Motivation: Existing datasets (Defects4J, Vul4J) primarily contain memory safety bugs with few security-impactful logical defects. Each data point requires approximately 10 person-hours to construct.

2. **Multi-Dimensional LLM Evaluation System**:

    - Function: Systematically decouples the effect of different factors on repair performance.
    - Mechanism: Prompts are varied along three dimensions: (a) LLM configuration — temperature (0.2/0.5/0.9), orientation (role/task), strategy (zero-shot/few-shot/CoT); (b) source code — vulnerable block $V_b$ vs. full function $V_f$, with or without context $V_{ctx}$; (c) auxiliary information — combinations of none / vulnerability description $D$ / specification $V_S$ / repair steps $R$.
    - Design Motivation: Logical vulnerability repair is highly context-dependent, necessitating precise identification of which information types are most beneficial to LLMs.

3. **Reasoning-Based Patch Quality Evaluation**:

    - Function: Assesses the reasoning soundness of patches beyond compilation and testing.
    - Mechanism: An LLM generates natural language explanations $E$ and $E_g$ for the generated patch and the ground-truth fix respectively. Cosine similarity $CS$ and LLM judgment $J$ are used to measure semantic alignment. High similarity indicates that the patch's repair logic is consistent with the ground-truth fix.
    - Design Motivation: Logical vulnerabilities lack uniform repair patterns; traditional static analysis and testing cannot reliably assess patch correctness. Reasoning-based analysis captures whether a patch "understands the problem."

### Loss & Training

This paper presents an evaluation framework rather than a training methodology. Evaluation is conducted using three LLMs — Llama 3.1, Qwen 2.5, and OpenAI o3-mini — and three baseline AVR tools: SimFix, KNOD, and VRPilot.

## Key Experimental Results

### Main Results

**Baseline AVR Tools (Synthetic Java Samples)**

| Tool | Compilation Rate | Test Pass Rate | Cosine Similarity | LLM Judgment Agreement |
|------|-----------------|----------------|-------------------|------------------------|
| SimFix | 0.01 | 0.00 | 0.62–0.64 | 0.00–0.01 |
| KNOD | 0.35 | 0.00 | 0.64–0.65 | 0.00–0.02 |
| VRPilot | 0.56 | 0.09 | 0.65 | 0.03–0.15 |

**LLM Zero-Shot Repair (Real-World Vulnerabilities, $V_b$ + $D$ Provided)**

| LLM | Compilation Rate | Test Pass Rate | Reasoning Similarity (CS) |
|-----|-----------------|----------------|--------------------------|
| Llama 3.1 | 0.50 | 0.06 | 0.76–0.81 |
| Qwen 2.5 | 0.66 | 0.04 | 0.73–0.81 |
| o3-mini | 0.58 | 0.07 | 0.77 |

### Ablation Study

**Effect of Auxiliary Information (Real-World Vulnerabilities, Llama 3.1)**

| Auxiliary Information | Compilation Rate | Test Pass Rate | LLM Judgment Agreement |
|-----------------------|-----------------|----------------|------------------------|
| No auxiliary info | 0.66 | 0.04 | 0.02–0.10 |
| + Vulnerability description $D$ | 0.55 | 0.03 | 0.13–0.41 |
| + Description + specification $V_S$ | 0.49 | 0.00 | 0.18–0.51 |
| + Description + repair steps $R$ | 0.62 | 0.07 | 0.46–0.72 |

### Key Findings

- Without auxiliary information, LLMs achieve the highest compilation rates but lowest reasoning scores — LLMs tend to treat logical vulnerabilities as memory vulnerabilities, generating patches that "compile but are logically incorrect."
- Providing repair steps $R$ yields the highest reasoning scores (LLM judgment agreement 0.46–0.72), but may cause compilation failures as LLMs introduce undeclared variables.
- Zero-shot generally outperforms CoT — CoT reasoning steps introduce additional undefined variables that cause compilation errors.
- Temperature and orientation (role/task) have negligible impact on performance.
- In the real-world setting, LLMs correctly repair only 5 out of 61 samples, demonstrating that logical vulnerability repair remains an extreme challenge.

## Highlights & Insights

- The distinction between "logical vulnerabilities" and "memory vulnerabilities" highlights an important yet underexplored direction in AVR research.
- The introduction of reasoning-based evaluation metrics addresses the inadequacy of conventional compilation/testing for assessing logical vulnerability patches.
- The finding that "no auxiliary information → high compilation rate but low reasoning score" is particularly revealing — it demonstrates that superficially passing compilation can mask fundamental repair failure.

## Limitations & Future Work

- Perfect patch localization is assumed; in practice, localizing logical vulnerabilities is itself a significant challenge.
- The dataset is relatively small (61 real-world samples), limiting statistical significance.
- Only single-hunk repairs are evaluated; multi-location repair scenarios are not covered.
- Reasoning-based evaluation relies on an LLM as judge, whose reliability warrants further validation.

## Related Work & Insights

- **vs. VRPilot**: VRPilot is the strongest existing LLM-based repair method, yet its CoT strategy yields lower reasoning scores on logical vulnerabilities than zero-shot.
- **vs. SimFix/KNOD**: Traditional template-based and learning-based methods almost entirely fail on logical vulnerabilities, validating the unique challenges they pose.
- **vs. Pearce et al.**: Prior LLM repair evaluations do not consider auxiliary information, lack reasoning-based evaluation, and employ CodeQL testing unsuitable for logical vulnerabilities.

## Rating

- Novelty: ⭐⭐⭐⭐ First systematic evaluation framework and dataset for logical vulnerability repair.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 21 prompt configurations × 3 LLMs × 2 datasets; analysis is exceptionally comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Well-structured with coherent analysis.
- Value: ⭐⭐⭐⭐ Reveals critical bottlenecks of LLMs in logical vulnerability repair and points the way for future AVR research.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] ReFEree: Reference-Free and Fine-Grained Method for Evaluating Factual Consistency in Real-World Code Summarization](referee_reference-free_and_fine-grained_method_for_evaluating_factual_consistenc.md)
- [\[ACL 2026\] QiMeng-PRepair: Precise Code Repair via Edit-Aware Reward Optimization](qimeng-prepair_precise_code_repair_via_edit-aware_reward_optimization.md)
- [\[ACL 2026\] EET: Experience-Driven Early Termination for Cost-Efficient Software Engineering Agents](eet_experience-driven_early_termination_for_cost-efficient_software_engineering_.md)
- [\[ACL 2026\] CodeWiki: Evaluating AI's Ability to Generate Holistic Documentation for Large-Scale Codebases](codewiki_evaluating_ai39s_ability_to_generate_holistic_documentation_for_large-s.md)
- [\[ACL 2026\] SOCIA-EVO: Automated Simulator Construction via Dual-Anchored Bi-Level Optimization](socia-evo_automated_simulator_construction_via_dual-anchored_bi-level_optimizati.md)

<!-- RELATED:END -->
