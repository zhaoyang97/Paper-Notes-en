---
title: >-
  [Paper Note] LogicEval: A Systematic Framework for Evaluating Automated Repair Techniques for Logical Vulnerabilities in Real-World Software
description: >-
  [ACL 2026][Code Intelligence][Logical Vulnerabilities] Ours constructs the first repair evaluation framework LogicEval and dataset LogicDS (61 real-world logical vulnerabilities + 61 synthetic Java samples) specifically…
tags:
  - "ACL 2026"
  - "Code Intelligence"
  - "Logical Vulnerabilities"
  - "Automated Repair Evaluation"
  - "LLM Code Repair"
  - "Patch Generation"
  - "Benchmark Dataset"
date: 2026-05-08
content_hash: f9e4e45c21713a5d
---

# LogicEval: A Systematic Framework for Evaluating Automated Repair Techniques for Logical Vulnerabilities in Real-World Software

**Conference**: ACL 2026  
**arXiv**: [2604.12994](https://arxiv.org/abs/2604.12994)  
**Code**: [GitHub](https://github.com/LogicEval)  
**Area**: Code Intelligence / Vulnerability Repair Evaluation  
**Keywords**: Logical Vulnerabilities, Automated Repair Evaluation, LLM Code Repair, Patch Generation, Benchmark Dataset

## TL;DR

Ours constructs the first repair evaluation framework LogicEval and dataset LogicDS (61 real-world logical vulnerabilities + 61 synthetic Java samples) specifically for logical vulnerabilities. It systematically evaluates the capabilities of traditional AVR tools and LLMs in repairing logical vulnerabilities, finding that LLMs perform best when provided with auxiliary information but the overall repair rate remains very low (only 5 out of 61 real-world samples correctly repaired), while identifying key bottlenecks such as prompt sensitivity, context loss, and patch localization difficulties.

## Background & Motivation

**Background**: Logical vulnerabilities stem from incorrect implementations of program logic/functionality rather than memory safety violations. They can be exploited for authentication bypass, sensitive data leakage, or system operation disruption without triggering traditional security defenses (e.g., address sanitizers). Existing automated vulnerability repair (AVR) techniques primarily target memory corruption vulnerabilities.

**Limitations of Prior Work**: (1) Logical vulnerabilities lack consistent, reusable repair templates/patterns, requiring deep understanding of program semantics and expected behavior for each fix; (2) Logical vulnerabilities do not necessarily cause crashes or illegal memory access, meaning traditional signals (compilation logs, runtime logs, memory sanitizers) provide limited help for localization; (3) Existing datasets mainly focus on memory safety bugs and lack logical vulnerability samples with actual security impact.

**Key Challenge**: LLMs have demonstrated powerful capabilities in code understanding and generation, but there is no systematic framework to analyze their capabilities and limitations in repairing logical vulnerabilities—this hinders the expansion of AVR from memory safety to the more subtle domain of logical vulnerabilities.

**Goal**: Construct the first systematic evaluation framework to analyze the capabilities, limitations, and failure modes of traditional and LLM-based methods in repairing real-world logical vulnerabilities.

**Key Insight**: The specificity of logical vulnerability repair lies in its repair logic being highly dependent on context (vulnerability descriptions, behavior specifications, repair steps). Therefore, the impact of different dimensions is evaluated by systematically varying auxiliary information.

**Core Idea**: Construct the LogicDS dataset + LogicEval evaluation framework to systematically evaluate from three dimensions: LLM configuration, source code granularity, and auxiliary information. Introduce reasoning-based automated evaluation metrics (Cosine Similarity + LLM Judgment) to supplement traditional compilation/testing evaluation.

## Method

### Overall Architecture

LogicEval is an end-to-end evaluation pipeline: (1) **Input**—Vulnerable source code $S$, repaired code $F$, vulnerability description $D$, behavior specification $V_S$ (optional), context $V_{ctx}$ (optional), compilation/test scripts; (2) **Patch Localization**—Assuming perfect localization, manually identifying the core repair area (single hunk); (3) **Patch Generation**—Constructing prompts across different dimensions to drive LLMs to generate patches, extracting marked code to replace the vulnerable area; (4) **Patch Evaluation**—Compilation testing + reasoning-based automated evaluation (comparing the semantic alignment between patch explanations and ground-truth repair explanations).

### Key Designs

1. **LogicDS Dataset Construction**:

    - **Function**: Provides the first benchmark of logical vulnerabilities with actual security impact.
    - **Mechanism**: 61 real-world logical vulnerabilities were filtered from CVEs of 28 popular open-source projects. Each sample includes vulnerable/repaired code, CVE description, manually localized core repair area, compilation scripts, and test cases. An additional 61 synthetic Java samples were constructed for compatibility with Java-specific repair tools.
    - **Design Motivation**: Existing datasets (Defects4J, Vul4J) primarily contain memory safety bugs with few security-impacting logical defects. Each data point takes approximately 10 person-hours to construct.

2. **Multi-dimensional LLM Evaluation System**:

    - **Function**: Systematically decouples the impact of different factors on repair performance.
    - **Mechanism**: Prompts are varied along three dimensions: (a) LLM configuration—temperature (0.2/0.5/0.9), orientation (Role/Task), strategy (zero-shot/few-shot/CoT); (b) Source code—vulnerable block $V_b$ vs. complete function $V_f$, with or without context $V_{ctx}$; (c) Auxiliary information—different combinations of None/vulnerability description $D$/specification $V_S$/repair steps $R$.
    - **Design Motivation**: Logical vulnerability repair is highly dependent on contextual information, requiring precise understanding of which information is most helpful to LLMs.

3. **Reasoning-based Patch Quality Evaluation**:

    - **Function**: Evaluates the reasoning rationality of patches beyond compilation/testing.
    - **Mechanism**: LLMs generate natural language explanations $E$ and $E_g$ for the generated patch and the ground-truth repair, respectively. Semantic alignment is evaluated using Cosine Similarity $CS$ and LLM Judgment $J$. High similarity indicates that the patch's repair logic is consistent with the ground truth.
    - **Design Motivation**: Logical vulnerabilities lack unified repair patterns, and traditional static analysis or testing cannot reliably evaluate them; reasoning analysis captures whether the patch "understands the problem."

### Loss & Training

Ours is an evaluation framework rather than a training method. Evaluation uses three LLMs—Llama 3.1, Qwen 2.5, and OpenAI o3-mini—alongside three baseline AVR tools: SimFix, KNOD, and VRPilot.

## Key Experimental Results

### Main Results

**Baseline AVR Tools (Synthetic Java Samples)**

| Tool | Compilation Pass Rate | Test Pass Rate | Cosine Similarity | LLM Judgment Consistency Rate |
|------|----------|----------|----------|-------------|
| SimFix | 0.01 | 0.00 | 0.62-0.64 | 0.00-0.01 |
| KNOD | 0.35 | 0.00 | 0.64-0.65 | 0.00-0.02 |
| VRPilot | 0.56 | 0.09 | 0.65 | 0.03-0.15 |

**LLM Zero-shot Repair (Real-world Vulnerabilities, providing $V_b$ + $D$)**

| LLM | Compilation Pass Rate | Test Pass Rate | Reasoning Similarity (CS) |
|-----|----------|----------|--------------|
| Llama 3.1 | 0.50 | 0.06 | 0.76-0.81 |
| Qwen 2.5 | 0.66 | 0.04 | 0.73-0.81 |
| o3-mini | 0.58 | 0.07 | 0.77 |

### Ablation Study

**Impact of Auxiliary Information (Real-world Vulnerabilities, Llama 3.1)**

| Auxiliary Information | Compilation Rate | Test Rate | LLM Judgment Consistency Rate |
|---------|-------|-------|-------------|
| No auxiliary information | 0.66 | 0.04 | 0.02-0.10 |
| + Description $D$ | 0.55 | 0.03 | 0.13-0.41 |
| + Description + Spec $V_S$ | 0.49 | 0.00 | 0.18-0.51 |
| + Description + Repair Steps $R$ | 0.62 | 0.07 | 0.46-0.72 |

### Key Findings

- LLMs achieve the highest compilation rates but lowest reasoning scores without auxiliary information—LLMs tend to treat logical vulnerabilities as memory vulnerabilities, generating patches that "pass compilation but have incorrect logic."
- Providing repair steps $R$ yields the highest reasoning scores (LLM judgment consistency 0.46-0.72) but may lead to compilation failures (e.g., LLMs creating undeclared variables).
- Zero-shot generally outperforms CoT—reasoning steps in CoT introduce additional undefined variables, leading to compilation errors.
- Temperature and orientation (Role/Task) do not significantly impact performance.
- In real-world scenarios, LLMs correctly repair only 5 out of 61 samples, indicating that logical vulnerability repair remains a significant challenge.

## Highlights & Insights

- The distinction between "logical vulnerabilities vs. memory vulnerabilities" reveals an important overlooked direction in the AVR field.
- The introduction of reasoning evaluation metrics compensates for the shortcomings of traditional compilation/testing in evaluating logical vulnerabilities.
- The finding of "No auxiliary information → high compilation rate but low reasoning score" is profound—it indicates that superficial compilation success masks fundamental repair failures.

## Limitations & Future Work

- Perfect patch localization is assumed, whereas localization of logical vulnerabilities is a major challenge in real-world scenarios.
- The dataset scale is small (61 real-world samples), limiting statistical significance.
- Only single-hunk repairs are evaluated; multi-location repair evaluation is not covered.
- Reasoning evaluation depends on LLMs as judges, and its reliability requires further verification.

## Related Work & Insights

- **vs. VRPilot**: VRPilot is the strongest existing LLM repair method, yet its CoT strategy actually yields lower reasoning scores on logical vulnerabilities than zero-shot.
- **vs. SimFix/KNOD**: Traditional template/learning methods fail almost completely on logical vulnerabilities, validating their unique challenges.
- **vs. Pearce et al.**: Previous LLM repair evaluations did not consider auxiliary information, lacked reasoning evaluation, and used CodeQL testing which is unsuitable for logical vulnerabilities.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First systematic evaluation framework and dataset for logical vulnerability repair.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Extremely detailed analysis with 21 prompt configurations × 3 LLMs × 2 datasets.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure and organized analysis.
- **Value**: ⭐⭐⭐⭐ Identifies key bottlenecks for LLMs in logical vulnerability repair, pointing the way for future AVR research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] ReFEree: Reference-Free and Fine-Grained Method for Evaluating Factual Consistency in Real-World Code Summarization](referee_reference-free_and_fine-grained_method_for_evaluating_factual_consistenc.md)
- [\[ACL 2026\] Discover and Prove: An Open-source Agentic Framework for Hard Mode Automated Theorem Proving in Lean 4](discover_and_prove_an_open-source_agentic_framework_for_hard_mode_automated_theo.md)
- [\[ACL 2026\] ChatHLS: Towards Systematic Design Automation and Optimization for High-Level Synthesis](chathls_towards_systematic_design_automation_and_optimization_for_high-level_syn.md)
- [\[ACL 2026\] QiMeng-PRepair: Precise Code Repair via Edit-Aware Reward Optimization](qimeng-prepair_precise_code_repair_via_edit-aware_reward_optimization.md)
- [\[ACL 2026\] Benchmarking Testing in Automated Theorem Proving](benchmarking_testing_in_automated_theorem_proving.md)

</div>

<!-- RELATED:END -->
