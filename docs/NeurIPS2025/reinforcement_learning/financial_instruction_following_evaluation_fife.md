---
title: >-
  [Paper Note] Financial Instruction Following Evaluation (FIFE)
description: >-
  [NeurIPS 2025 (GenAI Finance Workshop)][Reinforcement Learning][Instruction Following] FIFE is a challenging instruction-following benchmark for financial analysis tasks…
tags:
  - "NeurIPS 2025 (GenAI Finance Workshop)"
  - "Reinforcement Learning"
  - "Instruction Following"
  - "Finance Domain"
  - "Benchmark"
  - "Chainable Constraints"
  - "LLM Evaluation"
date: 2026-05-08
content_hash: 139e48b19f7820cb
---

# Financial Instruction Following Evaluation (FIFE)

**Conference**: NeurIPS 2025 (GenAI Finance Workshop)  
**arXiv**: [2512.08965](https://arxiv.org/abs/2512.08965)  
**Code**: [https://github.com/gtfintechlab/FIFE](https://github.com/gtfintechlab/FIFE)  
**Area**: Reinforcement Learning  
**Keywords**: Instruction Following, Finance Domain, Benchmark, Chainable Constraints, LLM Evaluation

## TL;DR

FIFE is a challenging instruction-following benchmark for financial analysis tasks, comprising 88 manually authored complex prompts and 40+ chainable, domain-specific verifiable constraints. It evaluates 53 models under both strict and loose modes, revealing that even the strongest open-weight model (76.1% strict) fails to perfectly follow complex financial instruction requirements.

## Background & Motivation

**Background**: Language models have achieved considerable maturity in general-purpose instruction following, and benchmarks such as IFEval are widely used to assess this capability. However, the financial domain demands extremely high precision—incorrect numeric formatting, omitted risk disclosures, and non-compliant regulatory statements can all carry serious consequences.

**Limitations of Prior Work**: Existing instruction-following benchmarks (e.g., IFEval) are primarily designed for general tasks and lack finance-specific constraints. Financial analysis involves a large number of domain-specific requirements: LaTeX formulations for Black-76 option pricing, specific numeric formats for VaR calculations, particular numbering conventions for Rule 10b-5 compliance disclosures, and so forth. These constraints are often mutually dependent, forming chain-like structures—yet no existing benchmark systematically evaluates model performance under such complex constraint regimes.

**Key Challenge**: The requirements for instruction following in finance far exceed those of general domains, while existing evaluation tools fall well short of meeting this need. General-purpose benchmarks cannot capture finance-specific constraint types, leading to distorted assessments of model capability in financial settings.

**Goal**: To construct a challenging financial instruction-following benchmark that (1) incorporates domain-specific constraints spanning multiple financial sub-domains; (2) provides a chainable, automated verification system; and (3) supplies fine-grained reward signals to support RL training.

**Key Insight**: Deep customization of the IFEval framework for the financial domain—designing 40+ finance-specific instruction checkers covering a broad range of scenarios from equity analysis and derivatives pricing to compliance reporting and ESG assessment. Each checker can be applied independently or chained together to form complex, multi-constraint prompts.

**Core Idea**: Construct a high-difficulty benchmark using 40+ chainable, finance-specific verifiable constraints, providing precise reward signals for RL training in the financial domain.

## Method

### Overall Architecture

The FIFE pipeline proceeds as follows: (1) 88 manually authored prompts, each containing 1–5 financial domain constraints; (2) model response generation; (3) automated verification that checks each constraint individually, outputting prompt-level and instruction-level accuracy; (4) support for both strict (exact match) and loose (tolerating minor formatting deviations) evaluation modes.

### Key Designs

1. **Finance-Specific Instruction Checker System**:

    - Function: Covers 40+ constraint types across 10+ financial sub-domains.
    - Mechanism: Each checker inherits from the `InstructionChecker` base class and implements two methods: `build_description` (generating the constraint description) and `check_following` (verifying compliance). Covered domains include equity analysis (bold opening + italic risk), credit spread analysis (table formatting), FX calculation (code block restrictions), compliance reporting (numbering format), derivatives pricing (Black-76 LaTeX), VaR calculation (bold dollar signs), ESG reporting, private equity, and more.
    - Design Motivation: General-purpose instruction checkers (e.g., "limit word count," "use a numbered list") cannot capture the specialized constraints of the financial domain; domain-expert-designed verification logic is required.

2. **Chainable Constraint Composition System**:

    - Function: Combines multiple simple constraints into complex multi-constraint prompts, providing fine-grained reward signals.
    - Mechanism: Each prompt contains a list of constraint IDs (`instruction_id_list`), which are verified sequentially. A single prompt may simultaneously require "describe risks with bold headings," "present spread analysis in a table," and "include a compliance disclaimer." Each constraint independently produces a binary signal, forming a vectorized reward.
    - Design Motivation: Chaining constraints enables controllable difficulty (1 constraint vs. 5 constraints), and vectorized rewards provide richer gradient information for RL training compared to simple pass/fail signals.

3. **Dual Evaluation Modes (Strict / Loose)**:

    - Function: Distinguishes between formatting precision and content correctness.
    - Mechanism: Strict mode requires exact matching and tolerates no formatting deviation; Loose mode tests multiple response variants—e.g., removing Markdown markup, truncated versions, etc. The gap between the two modes reflects the difference in a model's ability to "understand constraints" versus "precisely execute formatting."
    - Design Motivation: A model may understand the intent of a constraint but fail strict evaluation due to minor formatting details (e.g., an extra blank line); loose mode helps disentangle these two capabilities.

### Loss & Training

FIFE is an evaluation benchmark rather than a training methodology. All models are evaluated in a zero-shot setting without few-shot prompting or fine-tuning. Constraint verification results can serve as external reward signals for RL.

## Key Experimental Results

### Main Results

| Model Category | Best Model | Strict Accuracy | Loose Accuracy |
|---------------|-----------|----------------|---------------|
| Open-weight | (Top open-weight) | 76.1% | 79.5% |
| Proprietary | (Top proprietary) | 65.9% | 70.5% |
| Open-source | (Top open-source) | 45.5% | 48.9% |

### Ablation Study

| Evaluation Detail | Metric Type | Typical Result |
|------------------|------------|---------------|
| Prompt-level accuracy | Proportion with all constraints satisfied | Drops sharply as number of constraints increases |
| Instruction-level accuracy | Proportion of individual constraints satisfied | 2–9 percentage points higher than prompt-level |
| Strict vs. Loose gap | Formatting precision | 3–5 percentage point difference |

### Key Findings

- **Open-weight models outperform proprietary models**: This result challenges the conventional assumption that proprietary models are superior; open-weight models demonstrate stronger instruction-following capability on finance-specific constraints.
- **Even the strongest models fall short of perfect compliance**: A strict accuracy of 76.1% indicates that complex, chained financial constraints pose a genuine challenge to all current LLMs.
- **Open-source models lag significantly** (45.5%), trailing open-weight and proprietary models by approximately 20–30 percentage points.
- The 3–5% gap between strict and loose modes suggests that a portion of failures stem from formatting details rather than a failure to understand the constraints.

## Highlights & Insights

- **The design of fine-grained reward signals is notably forward-looking.** Vectorized feedback produced by independently verifying each constraint is better suited for RL training than simple binary pass/fail signals—making FIFE not merely an evaluation tool but foundational infrastructure for RL in finance.
- **The domain specificity of financial constraints is practically grounded.** For instance, `fin:deriv:black76_latex_sigma` requires the Black-76 pricing formula to be expressed in LaTeX—a constraint unobtainable from general-purpose benchmarks, underscoring the irreplaceable role of domain-specific financial evaluation.
- The methodology is transferable to other high-stakes domains (e.g., healthcare, law) by designing domain-specific chainable constraint checkers accordingly.

## Limitations & Future Work

- The benchmark scale of 88 prompts is relatively small and may not adequately cover the full range of constraint types and difficulty distributions in the financial domain.
- Only zero-shot capability is evaluated; differences under few-shot prompting, chain-of-thought reasoning, or fine-tuning are not explored.
- The verification system relies on rule-based matching, which may produce false positives or negatives—for example, when a model uses an equivalent but differently formatted LaTeX expression.
- The paper does not report which specific constraint types are most difficult to follow, and fine-grained error analysis is absent.
- Model names are somewhat vague in the paper (e.g., "top open-weight model"), without fully specifying which models are being referenced.
- As a workshop paper, the experimental scope and depth are limited.

## Related Work & Insights

- **vs. IFEval (Google)**: FIFE builds upon IFEval's framework and evaluation logic (`evaluation_lib.py` is adapted from IFEval), but replaces general-purpose constraints with 40+ finance-specific ones, substantially increasing difficulty.
- **vs. FinBen / FinGPT**: Existing financial NLP benchmarks focus on tasks such as knowledge QA and sentiment analysis, whereas FIFE targets the orthogonal dimension of instruction following—a model may "know" financial knowledge yet fail to produce output in the required format.
- FIFE has direct implications for constructing financial RLHF datasets—its constraint checkers can serve directly as the rule-based component of reward models.

## Rating

- Novelty: ⭐⭐⭐⭐ The framework extends IFEval; the core innovation lies in the design of financial constraints rather than the methodology itself.
- Experimental Thoroughness: ⭐⭐⭐⭐ The evaluation scale of 53 models is commendable, but analytical depth is limited.
- Writing Quality: ⭐⭐⭐⭐ Standard workshop-paper writing—concise but lacking in detail.
- Value: ⭐⭐⭐⭐⭐ Fills a gap in instruction-following evaluation for the financial domain and holds practical significance for RL in finance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Generalizing Verifiable Instruction Following](generalizing_verifiable_instruction_following.md)
- [\[NeurIPS 2025\] Incentivizing Reasoning for Advanced Instruction-Following of Large Language Models](incentivizing_reasoning_for_advanced_instruction-following_of_large_language_mod.md)
- [\[ACL 2026\] ImpRIF: Stronger Implicit Reasoning Leads to Better Complex Instruction Following](../../ACL2026/reinforcement_learning/imprif_stronger_implicit_reasoning_leads_to_better_complex_instruction_following.md)
- [\[NeurIPS 2025\] Finite-Sample Analysis of Policy Evaluation for Robust Average Reward Reinforcement Learning](finite-sample_analysis_of_policy_evaluation_for_robust_average_reward_reinforcem.md)
- [\[NeurIPS 2025\] Quantifying Generalisation in Imitation Learning](quantifying_generalisation_in_imitation_learning.md)

</div>

<!-- RELATED:END -->
