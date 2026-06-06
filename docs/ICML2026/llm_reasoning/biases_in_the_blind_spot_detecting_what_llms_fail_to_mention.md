---
title: >-
  [Paper Note] Biases in the Blind Spot: Detecting What LLMs Fail to Mention
description: >-
  [ICML2026][LLM Reasoning][Bias Detection] This paper proposes a fully automated black-box pipeline to detect "unverbalized biases" in LLMs—hidden factors that systematically influence model decisions but are never mentio…
tags:
  - "ICML2026"
  - "LLM Reasoning"
  - "Bias Detection"
  - "Chain-of-Thought Faithfulness"
  - "Counterfactual Testing"
  - "Black-box Auditing"
  - "LLM Fairness"
date: 2026-05-08
content_hash: a560cfeecb1f5392
---

# Biases in the Blind Spot: Detecting What LLMs Fail to Mention

**Conference**: ICML2026  
**arXiv**: [2602.10117](https://arxiv.org/abs/2602.10117)  
**Code**: https://github.com/FlyingPumba/biases-in-the-blind-spot/  
**Area**: AI Safety  
**Keywords**: Bias Detection, Chain-of-Thought Faithfulness, Counterfactual Testing, Black-box Auditing, LLM Fairness  

## TL;DR
This paper proposes a fully automated black-box pipeline to detect "unverbalized biases" in LLMs—hidden factors that systematically influence model decisions but are never mentioned in CoT reasoning. By automatically generating concept hypotheses, counterfactual input variants, and staged statistical tests via LLMs, the pipeline discovered known biases (gender, race) and novel ones (Spanish fluency, English proficiency, writing formality) across three decision-making tasks.

## Background & Motivation

**Background**: Chain-of-Thought (CoT) is widely used to monitor LLM decision processes—researchers often trust the reasons the model states. However, increasing evidence suggests that CoT may not faithfully reflect actual decision drivers: models may be influenced by hidden factors without ever mentioning them in the reasoning chain.

**Limitations of Prior Work**: Existing bias evaluation methods typically rely on pre-defined categories (e.g., gender, race) and manually constructed datasets, resulting in limited coverage and high costs. Biases not pre-convisaged by researchers remain undetected. Furthermore, monitoring CoT alone misses factors that affect decisions but remain unverbalized.

**Key Challenge**: There is a systematic disconnect between what LLMs "say" and "do"—models may utilize information not mentioned in CoT (e.g., race implied by a name), yet external auditors cannot identify these hidden factors by reading CoT alone.

**Goal**: Design a fully automated, hypothesis-free pipeline that, given any decision task dataset, automatically discovers which input attributes systematically influence model decisions without being mentioned in CoT.

**Key Insight**: The authors formalize the problem as counterfactual testing—constructing "positive" and "negative" variants of the same input to observe model decision flips. If a concept causes significant decision flipping but is not cited in the CoT, it is identified as an unverbalized bias.

**Core Idea**: Use LLMs to automatically generate candidate bias concepts + counterfactual input variants + McNemar paired tests + staged statistical early stopping to discover hidden LLM biases under black-box conditions.

## Method

### Overall Architecture
Input consists of a decision task dataset $\mathcal{D}$ (e.g., resume screening) and a target model $M$. The pipeline proceeds in five steps: (1) Perform k-means clustering on inputs and sample representative cases; (2) Use an LLM to automatically generate candidate bias concepts $\mathcal{C}$; (3) Baseline verbalization filtering—collect CoT on original inputs and remove concepts already frequently mentioned; (4) Generate positive/negative counterfactual variants for each concept, expanding sample size in stages with statistical testing; (5) Check verbalization rates on inconsistent pairs, finally outputting concepts that pass significance tests with verbalization rates below threshold $\tau$.

### Key Designs

1.  **LLM-driven concept hypothesis generation**:
    - **Function**: Automatically discover candidate concepts from task data without manual pre-definition.
    - **Mechanism**: Performs k-means ($k=10$) on input embeddings, samples 3 representative inputs per cluster, and tasks an LLM to hypothesize concepts that might affect decisions based solely on input content (without seeing model answers). For each concept, it generates a verbalization check guide, a positive "add" action, and a negative "remove" action. An LLM judge filters variants that introduce confounding variables (filtering out 42% of candidates).
    - **Design Motivation**: Traditional methods are limited by researcher imagination; automated generation discovers unforeseen biases (e.g., writing formality).

2.  **Staged statistical testing and early stopping**:
    - **Function**: Substantially reduces computational overhead while controlling the Family-Wise Error Rate (FWER).
    - **Mechanism**: Uses McNemar's test to compare the proportion of inconsistent pairs, with Bonferroni correction setting the per-concept threshold to $\alpha' = \alpha / |\mathcal{C}|$. Sample sizes double each stage, with thresholds allocated via the O'Brien-Fleming alpha spending function $\alpha_s = 2(1 - \Phi(z_{\alpha'/2} / \sqrt{t_s}))$ (strict early on, relaxed later). Includes efficacy early stopping (flag if significant) and futility early stopping (discard if conditional power $<\gamma=0.01$). Saves approximately 1/3 of API calls.
    - **Design Motivation**: Exhaustive testing of all concepts across all inputs is infeasible; the staged design maintains statistical rigor while efficiently screening.

3.  **Two-layer verbalization filter**:
    - **Function**: Distinguishes "factors affecting decisions" from "factors affecting decisions *without* being mentioned in CoT."
    - **Mechanism**: **Baseline filtering**—collect CoT on original inputs; if a concept is cited as a reason in $>\tau$ ($\tau=0.3$) of cases, it is filtered. **Variant filtering**—checks verbalization rates specifically on inconsistent pairs (where the decision flipped); a concept is retained only if the verbalization rate in flipped samples is $\leq\tau$. Note: Simply repeating the concept from the input does not count as verbalization; it must be cited as a decision rationale.
    - **Design Motivation**: Inconsistent pairs provide direct evidence of bias; the goal is to confirm the model does not admit to the basis of the flip. LLM judge validation shows Cohen's $\kappa > 0.67$ with human labels.

## Key Experimental Results

### Main Results

Tested 7 LLMs across three decision tasks: Resume Screening (1,336 inputs), Loan Approval (2,500 inputs), and University Admissions (1,500 inputs).

| Bias Category | Models Detected (/7) | Effect Size Range | Direction |
| :--- | :--- | :--- | :--- |
| Gender Bias (pro-female) | 5-6 | 0.017–0.060 | 22 pro-female vs 0 pro-male |
| Race/Ethnicity Bias (pro-minority) | 5 | 0.026–0.060 | 21 pro-minority vs 0 pro-majority |
| English Proficiency Bias | 2-3 | 0.021–0.048 | Pro-fluent English |
| Writing Formality Bias | 2 | 0.033–0.044 | Pro-formal style |
| Spanish Ability Bias | 1 (QwQ-32B) | 0.040 | Pro-Spanish ability |
| Religious Bias | 1 (Claude Sonnet 4) | 0.037 | Pro-mainstream religion |

### Ablation Study

| Validation Dimension | Results | Description |
| :--- | :--- | :--- |
| Injected Bias Detection (80 test cases) | 92.5% Accuracy | 85% of secret biases detected, 100% of public biases filtered |
| Verbalization Detection Reliability (100 samples) | $\kappa = 0.79$ (best) | GPT-4o-mini and o1 approach human levels |
| Random Seed Consistency (5 runs) | Semantically similar concepts | No contradictory directions detected |
| Early Stopping Savings | ~1/3 API calls | Combined effect of O'Brien-Fleming + Futility stopping |
| Concept Quality Filtering | 42% filtered | 80% agreement between LLM judge and human labels |

### Key Findings
- **Cross-task Consistency**: Gender bias (pro-female) and racial bias (pro-minority) consistently appeared across all three tasks, suggesting these reflect intrinsic model behaviors rather than task-specific artifacts.
- **Grok 4.1 Fast is Most Transparent**: Out of 30 concepts labeled as unverbalized by other models, 27 were filtered for Grok—it actively mentions and discusses demographic factors in CoT (e.g., "Demographics: Shanice (likely underrepresented minority based on name)"), even if not explicitly citing them as a rationale.
- **RLVR vs SFT Comparison**: QwQ-32B (RLVR) and Qwen2.5-32B-Instruct (SFT) showed nearly identical verbalization filter rates (97.0% vs 97.2%) in the loan task; reasoning training changed which biases appeared rather than improving overall faithfulness.
- **Novel Biases Identified**: Biases regarding Spanish fluency, English proficiency, and writing formality—previously overlooked in manual analyses—were automatically discovered by the pipeline.

## Highlights & Insights
- **Hypothesis-free Automated Discovery**: A core differentiation from prior work (e.g., Karvonen & Marks 2025) is that LLMs automatically generate and verify hypotheses, discovering biases in researcher "blind spots." This paradigm is transferable to any decision-making scenario requiring systematic auditing.
- **Quantitative "Say vs Do" Metrics**: Converts the nebulous problem of CoT faithfulness into actionable quantitative indicators—Verbalization Rate + McNemar Effect Size—providing a replicable protocol for LLM deployment audits.
- **Balancing Statistical Rigor and Efficiency**: The combination of O'Brien-Fleming alpha spending and futility early stopping is highly practical for industrial-scale auditing, ensuring FWER control while reducing costs by 1/3.

## Limitations & Future Work
- **Variant Quality and Confounders**: Counterfactual variants generated by LLMs may introduce changes beyond the target concept (e.g., female names associated with gender-stereotypical occupations). While 42% of low-quality concepts were filtered, confounders cannot be entirely eliminated.
- **Single Occupation Focus**: The resume screening task was fixed to software engineering; whether gender bias interacts with different occupational stereotypes was not explored.
- **Conservative Design and False Negatives**: Bonferroni correction plus conservative early stopping may miss real biases with small effect sizes. Seed consistency experiments also show that not every run detects all possible biases.
- **Generalization to Open-ended Tasks**: Current tasks are binary decisions. Extending this to open-ended generation tasks requires substituting the decision metric.

## Related Work & Insights
- Karvonen & Marks (2025): Identifies gender/racial bias in resume screening manually; this work automates, replicates, and expands those findings.
- Arcuschin et al. (2025): Reveals "implicit post-hoc rationalization" in CoT, which serves as a direct motivation for this study.
- Atanasova et al. (2023): Framework for counterfactual faithfulness testing; this work extends it to LLM-driven concept variant generation.
- Lai et al. (2026): Concurrent work using seed bias expansion to discover LLM-as-Judge biases, which is complementary to this pipeline.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] From Abstract to Contextual: What LLMs Still Cannot Do in Mathematics](../../ICLR2026/llm_reasoning/from_abstract_to_contextual_what_llms_still_cannot_do_in_math_word_problem_solvi.md)
- [\[NeurIPS 2025\] Lost in Transmission: When and Why LLMs Fail to Reason Globally](../../NeurIPS2025/llm_reasoning/lost_in_transmission_when_and_why_llms_fail_to_reason_globally.md)
- [\[ICML 2026\] What Really Improves Mathematical Reasoning: Structured Reasoning Signals Beyond Pure Code](what_really_improves_mathematical_reasoning_structured_reasoning_signals_beyond_.md)
- [\[ICLR 2026\] Is It Thinking or Cheating? Detecting Implicit Reward Hacking by Measuring Reasoning Effort](../../ICLR2026/llm_reasoning/is_it_thinking_or_cheating_detecting_implicit_reward_hacking_by_measuring_reason.md)
- [\[ICML 2026\] Diagnosing Multi-step Reasoning Failures in Black-box LLMs via Stepwise Confidence Attribution](diagnosing_multi-step_reasoning_failures_in_black-box_llms_via_stepwise_confiden.md)

</div>

<!-- RELATED:END -->
