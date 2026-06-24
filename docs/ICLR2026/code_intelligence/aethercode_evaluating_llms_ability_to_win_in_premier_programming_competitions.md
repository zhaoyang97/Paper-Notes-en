---
title: >-
  [Paper Note] AetherCode: Evaluating LLMs' Ability to Win In Premier Programming Competitions
description: >-
  [ICLR2026][Code Intelligence][Competitive Programming] AetherCode is the first code reasoning benchmark to systematically collect 456 high-difficulty problems from premier programming competitions such as IOI and ICPC. It utilizes a hybrid approach of "automated generation + manual annotation by 67 experts" to achieve 100% TPR / 100% TNR for test cases. Results indicate that even the strongest model, o4-mini-high, achieves only a 35.5% Pass@1…
tags:
  - "ICLR2026"
  - "Code Intelligence"
  - "Competitive Programming"
  - "Code Reasoning"
  - "Test Case Quality"
  - "IOI/ICPC"
  - "LLM Benchmark"
date: 2026-05-08
content_hash: 317819a30d626b5d
---

# AetherCode: Evaluating LLMs' Ability to Win In Premier Programming Competitions

**Conference**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=lSM6MtjQcM](https://openreview.net/forum?id=lSM6MtjQcM)  
**Code**: To be confirmed (the paper mentions an online leaderboard)  
**Area**: Code Intelligence / LLM Evaluation / Competitive Programming Benchmark  
**Keywords**: Competitive Programming, Code Reasoning, Test Case Quality, IOI/ICPC, LLM Benchmark

## TL;DR
AetherCode is the first code reasoning benchmark to systematically collect 456 high-difficulty problems from premier programming competitions such as IOI and ICPC. It utilizes a hybrid approach of "automated generation + manual annotation by 67 experts" to achieve 100% TPR / 100% TNR for test cases. Results indicate that even the strongest model, o4-mini-high, achieves only a 35.5% Pass@1, debunking the illusion that "LLMs have conquered competitive programming."

## Background & Motivation
**Background**: Competitive programming is widely regarded as a key metric for measuring the reasoning and coding capabilities of LLMs. In the past one to two years, Pass@1 scores on benchmarks like MBPP and HumanEval have exceeded 90%, and over 80% on LiveCodeBench, leading to a general industry consensus that "competitive programming has been conquered."

**Limitations of Prior Work**: The authors point out that this optimistic assessment primarily stems from two critical flaws in existing benchmarks. First is **insufficient difficulty and coverage**—HumanEval/MBPP consist of basic tasks like sorting or reversing lists, while "competition-level" benchmarks like LiveCodeBench, CodeELO, and LiveCodeBench Pro are almost exclusively sourced from LeetCode, AtCoder, and CodeForces. LeetCode problems are relatively simple and often require implementing only a single function; CodeForces contests (5-7 problems in 2-3 hours) limit the design space and lack problems requiring large-scale, complex implementations. Second is **evaluation bias caused by test case quality**—incomplete test sets frequently fail to identify incorrect submissions (especially those with boundary handling errors or timeouts on extreme data). HumanEval/MBPP feature only a small number of handwritten cases; CodeContests and EvalPlus use naive methods like random mutation; some studies have even found that many cases in CodeContests violate problem constraints, incorrectly penalizing correct solutions.

**Key Challenge**: To create a highly discriminative benchmark, researchers must simultaneously solve "sufficient difficulty/breadth" and "accurate/comprehensive judging." Both tasks rely heavily on expert experience in the competitive programming domain, which has been a bottleneck for previous automated pipelines. Some works (CodeELO, LiveCodeBench Pro) attempt to "leverage" high-quality cases by directly calling official CodeForces judging services, but this poses compliance risks (as CodeForces prohibits scraping their judging API) and is limited by submission frequency, hindering flexible experimentation.

**Goal**: Construct an **open-source, self-contained, high-quality** competitive programming benchmark to re-expose the true gap between LLMs and top-tier human contestants.

**Key Insight**: Instead of scraping online problem sets, directly target premier global offline competitions like IOI and ICPC, which offer higher difficulty, larger design spaces, and comprehensive algorithm coverage. Treat judging quality as a first-class citizen and quantify it through the lens of a "test set as a binary classifier."

**Core Idea**: Collect problems from top-tier competitions and refine test cases to zero false positives and zero false negatives using a hybrid pipeline of "Generator-Validator Agents + 67 experts," creating a truly discriminative scale for code reasoning.

## Method

### Overall Architecture
AetherCode functions essentially as a **dataset curation pipeline** rather than a specific model, aimed at producing a benchmark with high-difficulty problems and high-quality judging. The pipeline consists of three main stages (corresponding to Fig. 1 in the paper): (a) **Problem Processing**—converting PDF problem statements from top competitions into Markdown+LaTeX with manual proofreading; (b) **Problem Classification**—experts tag each problem with algorithm types, difficulty, time/organizer, and other multi-dimensional labels; (c) **Test Case Construction**—automated generation via Generator-Validator (G-V) Agents, followed by expert manual annotation and auditing, with quality verified against collected human submissions. The final AetherCode v1 contains 456 problems (76 OI, 380 ICPC) with an average of 47.15 test cases per problem, covering 10 primary categories and 144 sub-categories.

### Key Designs

**1. Premier Competition Selection: Using IOI/ICPC to Expand Difficulty and Coverage**

Addressing the "simple problems, narrow sources" issue, AetherCode is the first to systematically collect problems from global premier competitions. Sources include two series: the OI series for secondary students (centered on IOI, including national events like China's NOI and the USA's USACO, where individuals solve 3 problems in 5 hours, primarily in C++) and the ICPC series for university students (teams of 3 with one computer solve 10-13 problems in 5 hours, including major events like CCPC). These offline competitions provide ample design space, naturally yielding problems that require large-scale complex implementation and deep algorithmic reasoning. Difficulty is rated as ★★★, level with APPS and CodeContests but more current and focused on top-tier events. For each problem, the authors collected over **30,000 human solutions** (at least 5 correct and 20 incorrect per problem) to serve as critical material for measuring test case quality.

**2. Multi-dimensional Classification: Enabling Fine-grained Analysis by Difficulty, Algorithm, and Time**

Beyond collection, the authors applied multi-dimensional tags to allow for fine-grained diagnostics. **Difficulty** is categorized into Easy/Medium/Hard/Extreme. Crucially, this is **determined from a human perspective** rather than model performance—specifically by the number of successful human solves within a contest or through expert assessment across contests. Problems **not solved by any human contestant during the competition** are classified as Extreme. This design enables the study of disparities between "LLM difficulty" and "human difficulty." The benchmark also includes **temporal and contextual dimensions** like date, organizer, and scope (regional/national/world) for decontamination, while excluding image-dependent problems and explicitly marking those requiring special judges. The **Algorithm Classification** utilizes a hierarchical system of 10 primary categories (Basic Algorithms, Search, DP, String, Math, Data Structures, Graph Theory, Computational Geometry, Common Techniques, Tree Problems) and 144 sub-labels, allowing a problem to belong to multiple categories to reflect its interdisciplinary nature.

**3. Test Set as a Binary Classifier: Measuring Quality via TPR/TNR**

This represents the most significant methodological contribution. Previously, test case quality was measured by quantity (assuming more is better), but quantity does not equal quality—old datasets contained cases violating constraints, and random data often misses edge cases. The authors treat **a problem's entire test set as a binary classifier** intended to distinguish correct solutions from incorrect ones. They evaluate this classifier using the collected human submissions, defining two core metrics:

$$\text{TPR} = \frac{\text{Number of Correct Solutions Passed}}{\text{Total Number of Correct Solutions}}, \qquad \text{TNR} = \frac{\text{Number of Incorrect Solutions Rejected}}{\text{Total Number of Incorrect Solutions}}$$

Where TPR measures **correctness** (ensuring correct solutions are not falsely rejected) and TNR measures **comprehensiveness/coverage** (ensuring incorrect solutions are "hacked"). This perspective transforms the vague notion of "case quality" into two calculable figures and defines the pipeline's ultimate goal: 100% TPR + 100% TNR.

**4. Hybrid G-V Agent + Expert Annotation: Reaching 100% TNR**

With quantitative goals set, the authors use a **hybrid pipeline**. The first step is **automated construction via Generator-Validator (G-V) Agents**: a dual-agent system where a generator writes a program to produce random and boundary data, and a validator ensures the data adheres to problem constraints. Human-in-the-loop review of the validator programs is included to prevent errors. G-V agents alone reach 89.9% TNR, and with manual verification of the validator, TPR reaches 100%. To reach 100% TNR, **expert annotation** is introduced: 67 competition experts (mostly Codeforces rating 2000+, some International Grandmasters over 2600) were recruited to construct targeted cases to "hack" the collected incorrect solutions. For problems with fewer than 50 incorrect solutions where TNR alone cannot guarantee robustness, an **elite audit team** (each with at least 3 ICPC gold medals and 2+ years of problem-setting experience) performed manual quality checks, added missing boundary conditions, and wrote additional incorrect/inefficient solutions for reverse verification. This achieved **100% TPR + 100% TNR** on the collected solution set—the first benchmark to reach this standard, according to the authors.

## Key Experimental Results

The evaluation covers 11 reasoning models and 6 non-reasoning models, with a maximum output of 32,768 tokens and an average taken from 4 samples per problem.

### Main Results: Overall Model Performance (Pass@1, %)

| Model | Easy | Medium | Hard | Extreme | Pass@1 | Pass@4 |
|------|------|--------|------|---------|--------|--------|
| o4-mini-high (Reasoning) | 65.3 | 32.1 | 8.0 | 3.8 | 35.5 | 46.6 |
| Gemini-2.5-Pro (Reasoning) | 60.1 | 28.6 | 8.5 | 2.5 | 32.7 | 46.0 |
| Seed-1.6-Thinking | 53.9 | 20.2 | 4.7 | 0 | 26.6 | 38.5 |
| DeepSeek-R1-0528 | 46.2 | 16.0 | 3.8 | 0 | 22.3 | 32.4 |
| Claude-4-Opus-thinking | 30.0 | 5.2 | 1.0 | 0 | 12.4 | 18.2 |
| GPT-4.1 (Non-Reasoning) | 23.9 | 5.7 | 1.1 | 0 | 10.5 | 15.3 |
| GPT-4o (Non-Reasoning) | 11.6 | 1.0 | 0.2 | 0 | 4.4 | 7.0 |

### Specific Algorithm Performance (Pass@1, Selected)

| Model | Basic | String | DP | Math | Geometry | Tree |
|------|------|--------|----------|------|----------|----------|
| o4-mini-high | 38.1 | 35.6 | 27.7 | 31.8 | 27.1 | 7.3 |
| Gemini-2.5-Pro | 36.1 | 29.8 | 24.6 | 31.5 | 18.1 | 7.3 |
| Qwen3-32B | 19.7 | 18.3 | 10.9 | 14.1 | 6.9 | 0 |

### Key Findings
- **Clear stratification and high benchmark discrimination**: o4-mini-high and Gemini-2.5-Pro form a top tier and are the only models capable of solving "Extreme" problems. They maintain a significant lead across all difficulties, confirming AetherCode's discriminative power. Even the strongest model has a total Pass@1 of only 35.5%, dropping to 8% for Hard and nearly zero for Extreme, supporting the argument that "LLMs have not yet conquered competitive programming."
- **Reasoning models dominate non-reasoning models**: Reasoning models like Qwen3-32B outperform multiple non-reasoning models despite having fewer parameters. Notably, non-reasoning models even at Pass@4 fail to catch up to reasoning models, indicating that for complex tasks, non-reasoning models have limited search space exploration capabilities that cannot be compensated for by increased sampling.
- **Top models possess greater exploration potential**: Moving from Pass@1 to Pass@4, o4-mini-high improves by 11.1% (35.5→46.6) and Gemini-2.5-Pro by 13.3% (32.7→46.0), while the weaker Qwen3-32B only increases by 7.6%. This suggests that multiple sampling provides greater gains for stronger models capable of generating diverse, high-quality candidates.
- **Abstract problem types are a common weakness**: All models perform relatively well on boilerplate "Basic Algorithm" and "String" tasks but fail significantly on abstract types like "Computational Geometry" and "Tree Problems" (where most models score single digits or zero). For non-reasoning models, this bottleneck extends to DP and Math, which require deep logical reasoning. While GPT-4.1 is the top non-reasoning model, it is notably weak in Math.

## Highlights & Insights
- **"Test set as a binary classifier + TPR/TNR" is a transferable methodology**: Shifting the metric from "quantity of cases" to "classification performance on real correct/incorrect solutions" provides a quantifiable, reproducible audit framework for the code evaluation field.
- **Defining difficulty through a human lens rather than model performance**, and separating the Extreme category, allows the benchmark to directly compare the mismatch between "human-perceived difficulty" and "model-perceived difficulty," providing more information than automated model-score-based grading.
- **30,000+ real human solutions serve as both evaluation material and quality assurance**: Incorrect solutions act as "adversarial samples" to test coverage, injecting the adversarial experience of the human community into benchmark construction.
- **Honest debunking of inflated scores**: In an era where most leaderboards are saturated (90%+), this paper uses a new scale to pull SOTA back to 35%, reminding the community that current ceilings for "competition-level" evaluation are no longer sufficient.

## Limitations & Future Work
- **Limited scale and temporal coverage**: v1 contains only 450 problems, concentrated in 2024 (400 problems) and 2025 (56 problems), with only 76 OI problems, skewing toward ICPC. Continuous updates are needed to combat data contamination and expand volume.
- **100% TPR/TNR is relative to the "collected solution set"**: Zero false positives/negatives are measured against existing correct/incorrect solutions. Coverage might not remain 100% against future, more sophisticated incorrect solutions, necessitating continuous addition of adversarial samples.
- **Heavy reliance on human experts**: The hybrid pipeline of 67 high-rating experts + gold-medal auditors is high quality, but cost and scalability are concerns, making it difficult to quickly expand to other domains or low-resource languages.
- **Narrow evaluation setting**: The setup is limited to a 32,768 token limit, pure text input/output, and excludes image-based problems, failing to cover interactive problems or agentic problem-solving (e.g., tool/compilation feedback) closer to real contests.

## Related Work & Insights
- **vs HumanEval / MBPP**: These are basic coding tasks with small handwritten test sets; SOTA is saturated at 90%+. AetherCode draws from top-tier competitions (★★★ difficulty) with refined hybrid test cases, pulling model scores down to the 35% range with vastly superior discrimination.
- **vs LiveCodeBench / CodeELO / LiveCodeBench Pro**: These are limited to LeetCode/AtCoder/CodeForces. CodeELO and LCB Pro rely on CodeForces' judging service (compliance risks, frequency limits); AetherCode uses broader sources (IOI/ICPC/CCPC) and is **self-contained** with high-quality open-source cases.
- **vs CodeContests / EvalPlus**: These use random mutation for case generation and face correctness issues like constraint violations; AetherCode uses the TPR/TNR framework to quantify and achieve 100% quality, addressing judging bias at the source.

## Rating
- Novelty: ⭐⭐⭐⭐ "Test set as a binary classifier" framework + first systematic IOI/ICPC benchmark. Solid methodology but limited single-point innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive matrix evaluation of 17 models × Difficulty × Algorithm Category × Pass@N with rich diagnostic dimensions.
- Writing Quality: ⭐⭐⭐⭐ Clear chain of reasoning, accurate problem positioning, and fully explained benchmark construction details.
- Value: ⭐⭐⭐⭐⭐ Provides a high-discrimination scale for saturated code evaluations; the TPR/TNR framework is widely reusable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Can LLMs Compress (and Decompress)? Evaluating Code Understanding and Execution via Invertibility](../../ACL2026/code_intelligence/can_llms_compress_and_decompress_evaluating_code_understanding_and_execution_via.md)
- [\[ACL 2026\] CodeWiki: Evaluating AI's Ability to Generate Holistic Documentation for Large-Scale Codebases](../../ACL2026/code_intelligence/codewiki_evaluating_ai39s_ability_to_generate_holistic_documentation_for_large-s.md)
- [\[ACL 2025\] TeXpert: A Multi-Level Benchmark for Evaluating LaTeX Code Generation by LLMs](../../ACL2025/code_intelligence/texpert_a_multi-level_benchmark_for_evaluating_latex_code_generation_by_llms.md)
- [\[ICLR 2026\] SpotIt: Evaluating Text-to-SQL Evaluation with Formal Verification](spotit_evaluating_text-to-sql_evaluation_with_formal_verification.md)
- [\[ICLR 2026\] Multi-LCB: Extending LiveCodeBench to Multiple Programming Languages](multi-lcb_extending_livecodebench_to_multiple_programming_languages.md)

</div>

<!-- RELATED:END -->
