---
title: >-
  [Paper Note] Can External Validation Tools Improve Annotation Quality for LLM-as-a-Judge?
description: >-
  [ACL 2025][LLM Evaluation][LLM-as-a-Judge] Proposes Evaluation Agent, a tool-augmented LLM-as-a-Judge framework that integrates web search (fact-checking), code execution, and mathematical verification tools. It improves human agreement from 63% to 81% on long-text fact-checking, and from 31% to 71% on coding evaluation, with virtually no degradation in out-of-domain areas.
tags:
  - "ACL 2025"
  - "LLM Evaluation"
  - "LLM-as-a-Judge"
  - "tool augmentation"
  - "fact-checking"
  - "code execution"
  - "annotation quality"
  - "agentic evaluation"
date: 2026-05-08
content_hash: 5ad87ac14c82b561
---

# Can External Validation Tools Improve Annotation Quality for LLM-as-a-Judge?

**Conference**: ACL 2025  
**arXiv**: [2507.17015](https://arxiv.org/abs/2507.17015)  
**Code**: [GitHub](https://github.com/apple/ml-agent-evaluator)  
**Area**: LLM / Evaluation  
**Keywords**: LLM-as-a-Judge, tool augmentation, fact-checking, code execution, annotation quality, agentic evaluation

## TL;DR
Proposes Evaluation Agent, a tool-augmented LLM-as-a-Judge framework that integrates web search (fact-checking), code execution, and mathematical verification tools. It improves human agreement from 63% to 81% on long-text fact-checking, and from 31% to 71% on coding evaluation, with virtually no degradation in out-of-domain areas.

## Background & Motivation
**Background**: Pairwise preference annotation (given a prompt + two responses, choosing the better one) is the standard method for LLM evaluation and RLHF feedback collection. AI annotators (LLM-as-a-Judge) are replacing expensive human annotators.

**Limitations of Prior Work**: AI annotation suffers from known biases—length bias (preferring verbose responses), positional bias (influenced by order), and self-enhancement bias (preferring self-generated content). More critically, LLM evaluation remains unreliable with respect to factuality, code correctness, and mathematical accuracy—"using an unreliable model to evaluate other models" is the fundamental issue.

**Key Challenge**: LLM judgments lack external validation mechanisms. When evaluating code quality, they do not execute the code; when evaluating facts, they do not check sources—rendering them like examiners without calculators or search engines.

**Goal**: Equip LLM-as-a-Judge with external validation tools to improve annotation quality while preserving out-of-domain performance.

**Key Insight**: Design an agentic framework where the LLM first assesses the response domain, selects appropriate tools, and integrates the tool feedback to make the final determination. If no suitable tools are found, the system falls back to the baseline annotator.

**Core Idea**: LLM judgment + Web search fact-checking + Code execution validation + Mathematical computation = More reliable annotation.

## Method

### Overall Architecture
A three-step pipeline: (1) Initial domain assessment—the LLM determines whether the responses belong to a domain where tools can assist; (2) Tool execution—running the selected tools to externally validate both responses; (3) Final judgment—the LLM synthesizes all tool outputs to make a pairwise preference decision. If no tools are applicable, the method directly falls back to the baseline annotator (e.g., AlpacaEval 2.0).

### Key Designs

1. **Initial Domain Assessment (Step 1)**:

    - **Function**: Assesses which tools are helpful for each response.
    - **Mechanism**: Designs a set of questions regarding response characteristics for each tool (e.g., "Does the text contain executable code?"), and uses the LLM to answer these questions to decide whether to activate the tool.
    - **Design Motivation**: Avoids running tools in non-applicable domains (e.g., running fact-checking on creative writing), thereby reducing out-of-domain degradation. Structured JSON output is used to minimize parsing errors.

2. **Tool A — Fact-checking (Based on SAFE)**:

    - **Function**: Performs web-search verification of factual claims in long-text responses.
    - **Mechanism**: (1) Extracts atomic facts $\rightarrow$ (2) Makes facts self-contained $\rightarrow$ (3) Verifies each fact using web search. This is based on the SAFE algorithm by Wei et al. (2024), but omits the relevance check, since the truthfulness of all facts is relevant in a pairwise preference configuration.

3. **Tool B — Code Execution**:

    - **Function**: Executes code from programming responses and captures feedback.
    - **Mechanism**: Based on the OpenAI Code Interpreter API, which can generate additional unit tests, execute multiple steps, and draw conclusions.

4. **Tool C — Mathematical Verification**:

    - **Function**: Verifies mathematical derivations and arithmetic calculations with code execution.
    - **Design Motivation**: Kept independent from the general code execution tool—testing revealed that general code interpreters perform poorly in mathematical scenarios, requiring specialized prompt constraints.

5. **Out-of-Domain Safety Mechanism**:

    - When the domain assessment indicates that no tools are useful, the system falls back to the baseline annotator instead of forcing tool utilization.
    - Tie-breaking: When only one response is suitable for tools, there is a 50% probability of using the agent and a 50% probability of falling back to the baseline.

## Key Experimental Results

### Main Results: Performance Gains in Target Domains

| Domain | Best Baseline | Best Agent | Gain |
|------|---------|----------|------|
| Long-form Facts (LongFact) | 78% (ArenaHard) | **81%** | +3% |
| Advanced Programming (APPS) | 42% (ArenaHard) | **72%** | +30% |
| Mathematics (GSM8k-hard) | ~54% (ArenaHard) | ~56% | +2% |
| Simple Facts (pick-best GPT-4o) | 63% | **81%** | +18% |

### Out-of-Domain Safety Testing

| Domain | Baseline | Agent | Degradation |
|------|------|-------|------|
| RewardBench Out-of-Domain (Chat/Safety) | ~85% | ~83% | <2% |

### Key Findings
- **Programming domain achieves the largest gain (+30%)**: Code execution is the most reliable validation method—the baseline annotator even performed worse than random on APPS (preferring incorrect code generated by GPT-4), which the agent completely reverses.
- **Anomalous baseline behavior on APPS**: All baseline preference rates were only 26-42% (below the 50% random benchmark), displaying self-enhancement bias toward GPT-4-style outputs.
- **Minimal out-of-domain degradation (<2%)**: The agent fell back to the baseline for 73.9% of the out-of-domain data points—domain assessment accurately identified non-applicable scenarios.
- **Simple configuration changes have a huge impact**: For the same GPT-4o, the difference between pick-best and ArenaHard prompts resulted in 63% vs 78% (solely due to variations in prompt and parsing methods).
- **Agent outperforms non-expert humans**: On long-form fact-checking, the agent's agreement (81%) was higher than that of human annotators (76.8%).

## Highlights & Insights
- The intuition of **"providing tools to the evaluator"** is simple yet remarkably effective—likened to equipping an examiner with a calculator and a search engine, leading to dramatic improvements particularly in the code execution domain.
- The **31% $\rightarrow$ 71%** increase in the programming domain demonstrates that LLMs rely heavily on style rather than correctness when judging code quality—running code is an irreplaceable validation step.
- **A warning on prompt sensitivity**: The same LLM varied from 63% to 78% purely due to different prompt configurations—highlighting the need to tune prompts as carefully as hyperparameters when deploying AI annotators.

## Limitations & Future Work
- **Tool invocation increases latency and cost**: Each sample requires multiple LLM calls coupled with web searches/code execution, resulting in high inference overhead.
- **Web search may return incorrect information**: The methodology relies on the LLM to judge the reliability of search results, allowing search engine hallucinations to propagate.
- **Only pairwise preference evaluation was tested**: Absolute grading scenarios remain unverified.
- **Limited gains in the mathematical domain**: On GSM8k-hard, the ArenaHard baseline even outperformed the agent—indicating a need for improved code execution strategies in mathematical contexts.
- **Occasional tool misclassification out-of-domain**: In 30 analyzed failure cases, 9 selected the incorrect tool (e.g., applying fact-checking to a safety refusal scenario).

## Related Work & Insights
- **vs Vanilla LLM-as-a-Judge (AlpacaEval/ArenaHard)**: Substantially improves reliability in factual and coding domains at the expense of complexity and latency.
- **vs Themis (Li et al.)**: Unlike Themis, which requires customized architectures and fine-tuning, the proposed method can directly leverage proprietary SOTA models.
- **vs SAFE (Wei et al.)**: While SAFE performs single-text factual evaluation, this work extends it to a pairwise preference setting.
- Direct implications for RLHF data quality—more reliable preference annotations yield superior model alignment.

## Rating
- Novelty: ⭐⭐⭐⭐ The agentic framework for tool-augmented LLM evaluation is practical and novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers multiple domains + out-of-domain safety + new dataset construction + comparisons with human annotators + failure analysis.
- Writing Quality: ⭐⭐⭐⭐ Clearly structured with persuasive results and open-source code.
- Value: ⭐⭐⭐⭐⭐ Possesses direct practical utility for LLM evaluation and RLHF workflows.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] YESciEval: Robust LLM-as-a-Judge for Scientific Question Answering](yescieval_llm_judge_science.md)
- [\[ACL 2025\] From Tools to Teammates: Evaluating LLMs in Multi-Session Coding Interactions](from_tools_to_teammates_evaluating_llms_in_multi-session_coding_interactions.md)
- [\[NeurIPS 2025\] Beyond the Surface: Enhancing LLM-as-a-Judge Alignment with Human via Internal Representations](../../NeurIPS2025/llm_evaluation/beyond_the_surface_enhancing_llm-as-a-judge_alignment_with_human_via_internal_re.md)
- [\[ICLR 2026\] Preference Leakage: A Contamination Problem in LLM-as-a-judge](../../ICLR2026/llm_evaluation/preference_leakage_a_contamination_problem_in_llm-as-a-judge.md)
- [\[ICML 2026\] On Cost-Effective LLM-as-a-Judge Improvement Techniques](../../ICML2026/llm_evaluation/on_cost-effective_llm-as-a-judge_improvement_techniques.md)

</div>

<!-- RELATED:END -->
