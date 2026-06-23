---
title: >-
  [Paper Note] Do Large Language Models Know What They Are Capable Of?
description: >-
  [ICLR 2026][LLM Agent][Paper Note] The authors systematically measure the ability of LLMs to "predict their success before starting a task" through three experimental setups. The study reveals that all models are systematically overconfident, though most possess discriminative power better than random. Furthermore, this self-awareness does not improve c
tags:
  - ICLR 2026
  - LLM Agent
date: 2026-05-08
content_hash: 8e532961cec97b4f
---
# Do Large Language Models Know What They Are Capable Of?

**Conference**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=EO6WtJ0q6G](https://openreview.net/forum?id=EO6WtJ0q6G)  
**Code**: https://github.com/cbarkan1/do-llms-know-what-theyre-capable-of  
**Area**: LLM Evaluation / Confidence Calibration / Agent Decision Making  
**Keywords**: In-advance confidence, overconfidence, discriminative power, risk aversion, misuse risk

## TL;DR
The authors systematically measure the ability of LLMs to "predict their success before starting a task" through three experimental setups. The study reveals that all models are systematically overconfident, though most possess discriminative power better than random. Furthermore, this self-awareness does not improve consistently as models grow stronger—current LLM agents are fundamentally limited by their inadequate understanding of their own capabilities.

## Background & Motivation

**Background**: Significant research exists concerning LLM confidence calibration, yet the vast majority focuses on "after-the-fact confidence"—where a model evaluates its own correctness after providing an answer (e.g., works by Lin, Tian, Xiong, Kapoor, etc.).

**Limitations of Prior Work**: In high-risk scenarios, "in-advance confidence" is the truly valuable metric—judging "can I do this" before committing. Knowing "when not to act" is critical in contexts where failure is costly. However, research in this area is sparse, and existing studies (Xu, Cash, Kadavath, Wei) are limited to single-step tasks, neglecting how LLMs update confidence with experience or how in-advance confidence translates into actual decision-making.

**Key Challenge**: When an agent executes multi-step tasks (e.g., solving a GitHub issue or launching a cyberattack), every attempt carries opportunity costs and explicit penalties. If an agent cannot accurately predict success/failure before or during execution, it will continue to invest resources when it should stop, causing losses or even exposure (in misuse scenarios). In other words, **decision quality is bottlenecked by the quality of confidence calibration** rather than the decision logic itself.

**Goal**: To decompose the problem into three progressive sub-questions: (i) Can LLMs accurately predict success on single-step tasks? (ii) Can LLMs learn from in-context experiences of success and failure to improve decisions when failure is costly? (iii) Does LLM confidence estimation improve or degrade as a multi-step agentic task progresses?

**Key Insight**: The authors operationalize "self-capability awareness" into two quantifiable metrics: **overconfidence** (predicted success rate minus actual success rate) and **discriminative power** (AUROC measuring the ability to distinguish "can do" vs. "cannot do"). By placing decision-making within an expected utility framework for "accepting/rejecting contracts," the authors decouple "suboptimal decisions" from whether the fault lies in the decision logic or the inflated confidence estimation itself.

**Core Idea**: No new models or methods are proposed. Instead, the authors design three experiments to map the previously unexplored chain of "in-advance confidence + experience-based learning + mid-task updates," linking these findings to risk assessments of AI misuse and loss of control.

## Method

### Overall Architecture
The paper is an evaluation and analysis study centered on three complementary experiments that cover a difficulty ladder: "Single-step prediction → Sequential decision-making with costs → Multi-step mid-task updates." All experiments share a quantitative language: for each task $i$, the model provides a success probability estimate $\hat{p}_i$, then separately performs the task to determine actual success/failure. Evaluation is conducted using overconfidence ($\frac{1}{N}\sum_i \hat{p}_i$ minus actual success rate) and AUROC (discriminative power). Tested models span the Llama, GPT, and Claude families, ranked by a composite capability score (average of MBPP, GPQA, MMLU-Pro, BigCodeBench) to observe trends between capability and self-awareness.

### Key Designs

**1. Exp 1—In-advance Confidence on Single-step Tasks: Establishing the Baseline**

Addressing the gap in in-advance confidence evaluation, the authors used BigCodeBench (1,140 Python problems). For each problem $i$, the model first estimates a success probability $\hat{p}_i$, and in a separate session, actually attempts the problem. Since reasoning models might "solve" the problem in their hidden chain-of-thought, this experiment only evaluates non-reasoning models or forces reasoning budgets to 0. Findings: First, all models exhibit **systematic overconfidence**. Only the Claude family showed overconfidence decreasing as capability increased; Llama and GPT showed no such trend. Second, most models significantly outperformed the random baseline (AUROC > 0.5), meaning they **possess some ability to distinguish soluble tasks from insoluble ones**, albeit with an optimistic bias.

**2. Exp 2—Sequential "Job Contract" Scenario: Decision-making with Cost and In-context Learning**

Exp 1 only measures estimation accuracy. Exp 2 places the model in a resource-acquisition scenario: it is offered 9 "job contracts" (BigCodeBench problems). Success yields +\$1, while failure costs −\$1. Contracts are curated so the model has a 50% success probability; thus, a policy of "always accept" or "always reject" yields zero expected return. Profitability requires discriminative power. Previous proposals, decisions, and results remain in the context, allowing "reflection." 

A key finding: models differ significantly. GPT-4o-2024-05-13 (4.1) learns little from experience, remaining overconfident with near-zero profit. However, Claude 3.5 Sonnet, while showing little AUROC improvement, learns to **accept fewer contracts**. By the 5th contract, its acceptance rate approaches the 50% ideal baseline, increasing profit. The mechanism is **risk aversion** rather than improved discrimination—the acceptance rate drops much faster than the predicted success rate. Further utility function fitting confirms models are rational decision-makers; the failure to optimize profit stems from **distorted probability inputs**.

**3. Exp 3—Mid-task Updates in Multi-step Agentic Tasks: Does Progress Bring Accuracy?**

This experiment uses SWE-Bench Verified (~500 agentic tasks requiring tool calls). After each tool call $s$, the model estimates the ultimate success probability $\hat{p}_{i,s}$. The author hypothesized that models would become more accurate as they gained task familiarity. This was **largely disproven**: most Claude Sonnet models became more overconfident as they progressed. Only the weakest GPT-4o significantly reduced overconfidence. Only 4 out of 10 models (GPT 4o, 4o-mini, 5.1(none), o1) showed significantly higher post-hoc discriminative power than at step 1. Sonnet's AUROC followed an "increase-then-decrease" curve; it gained confidence quickly on tasks eventually solved but also gained confidence slowly on tasks destined to fail. Notably, **reasoning models are not significantly better at self-prediction** than non-reasoning models.

## Key Experimental Results

### Main Results

| Experiment | Setup | Key Finding |
| :--- | :--- | :--- |
| Exp 1: Single-step | BigCodeBench (1140), estimate $\hat{p}_i$ vs actual | All overconfident; most AUROC > 0.5; Claude improves with capability. |
| Exp 2: Sequential | 512×9 contracts, +\$1/−\$1, 50% success rate | Sonnet 3.5/4.5 improve profit via "accepting less"; decisions are rational but inputs are high. |
| Exp 3: Multi-step | SWE-Bench Verified, $\hat{p}_{i,s}$ updates per step | Overconfidence increases with progress for most; reasoning models do not outperform. |

### Key Findings
- **Capability $\neq$ Self-awareness**: Newer/larger models are not universally more discriminative. Only Claude showed improvement with capability in Exp 1.
- **Profit via Risk Aversion, Not Discrimination**: In Exp 2, profit increases were driven by a drop in acceptance rates (positive risk aversion) rather than a significant rise in AUROC.
- **Rational Logic, Distorted Estimates**: All models roughly follow expected utility maximization, but sub-optimal decisions are driven by overestimations of success.
- **Reasoning Does Not Aid Calibration**: Reasoning models perform similarly or worse than non-reasoning models in overconfidence and discriminative power.

## Highlights & Insights
- **Systematic Mapping of the "In-advance Confidence" Chain**: The three experiments cover the complexity scale essential for real-world agentic decision-making.
- **Decoupling Causes via Expected Utility**: By proving the decision logic is rational, the paper cleanly attributes "bad decisions" to confidence calibration distortion.
- **Risk Aversion Explains Profit Gains**: A counter-intuitive insight showing that models "learning to profit" may just be getting more conservative rather than smarter at judging their own abilities.
- **Direct Implications for AI Risk**: Overconfidence implies that "sandbagging" (intentionally underperforming on evals) is likely detectable and would consistently undershoot, providing a lever for misuse and control assessments.

## Limitations & Future Work
- **Constraint on Reasoning Models**: Exp 1 and 2 excluded hidden chain-of-thought to obtain genuine pre-task estimates, potentially limiting the evaluation of specific model architectures.
- **Lack of Human Baseline**: There is no comparison to human calibration on long-form programming tasks due to the high cost of data collection.
- **Domain Specificity**: The results rely heavily on programming tasks (BigCodeBench/SWE-Bench); it remains to be seen if these generalize to other high-risk domains.
- **Future Directions**: Extending experiments to "dangerous capability" tasks (e.g., evading monitors) could provide quantitative estimates for loss-of-control risks.

## Related Work & Insights
- **vs. Post-hoc Calibration**: Unlike prior work (Lin 2022, Tian 2023) focused on evaluating answers after they are generated, this paper focuses on "pre-task" evaluation which is critical for agent failure mitigation.
- **vs. Single-step Pre-task Studies**: This paper expands the scope from single-step estimations (Xu 2025, Cash 2025) to sequential decision-making and mid-task updates.
- **vs. Self-Knowledge Studies**: Unlike studies on internal model properties, this paper focuses on the operational ability to predict task success.

## Rating
- Novelty: ⭐⭐⭐⭐ (First systematic study of in-advance confidence in the agentic chain)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Extensive multi-family models and realistic task benchmarks)
- Writing Quality: ⭐⭐⭐⭐ (Clear logic with deep causal decomposition)
- Value: ⭐⭐⭐⭐ (Critical for agent reliability and AI safety assessments)

<!-- RELATED:START -->
<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] K²-Agent: Co-Evolving Know-What and Know-How for Hierarchical Mobile Device Control](k²-agent_co-evolving_know-what_and_know-how_for_hierarchical_mobile_device_contr.md)
- [\[ICLR 2026\] GPS: Graph-guided Proactive Information Seeking in Large Language Models](gps_graph-guided_proactive_information_seeking_in_large_language_models.md)
- [\[ICLR 2026\] ToolWeaver: Weaving Collaborative Semantics for Scalable Tool Use in Large Language Models](toolweaver_weaving_collaborative_semantics_for_scalable_tool_use_in_large_langua.md)
- [\[ICML 2026\] Towards Diverse Scientific Hypothesis Search with Large Language Models](../../ICML2026/llm_agent/towards_diverse_scientific_hypothesis_search_with_large_language_models.md)
- [\[ACL 2026\] ImplicitMemBench: Measuring Unconscious Behavioral Adaptation in Large Language Models](../../ACL2026/llm_agent/implicitmembench_measuring_unconscious_behavioral_adaptation_in_large_language_m.md)

</div>

<!-- RELATED:END -->
