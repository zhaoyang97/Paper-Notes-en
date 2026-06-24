---
title: >-
  [Paper Note] Recommendations and Reporting Checklist for Rigorous & Transparent Human Baselines in Model Evaluations
description: >-
  [ICML 2025 Spotlight][Recommender Systems][Human Baseline] This paper systematically reviews the methodology of "human baselines" in AI evaluation. It reveals critical deficiencies in rigor and transparency across 115 existing human baseline studies, and proposes methodological recommendations and a reporting checklist covering the entire baseline lifecycle.
tags:
  - "ICML 2025 Spotlight"
  - "Recommender Systems"
  - "Human Baseline"
  - "AI Evaluation Methodology"
  - "Evaluation Transparency"
  - "Measurement Theory"
  - "Reporting Standards"
date: 2026-05-08
content_hash: ebd3acfedec2efbc
---

# Recommendations and Reporting Checklist for Rigorous & Transparent Human Baselines in Model Evaluations

**Conference**: ICML 2025 Spotlight  
**arXiv**: [2506.13776](https://arxiv.org/abs/2506.13776)  
**Code**: [https://github.com/kevinlwei/human-baselines](https://github.com/kevinlwei/human-baselines)  
**Area**: Recommender Systems  
**Keywords**: Human Baseline, AI Evaluation Methodology, Evaluation Transparency, Measurement Theory, Reporting Standards

## TL;DR

This paper systematically reviews the methodology of "human baselines" in AI evaluation. It reveals critical deficiencies in rigor and transparency across 115 existing human baseline studies, and proposes methodological recommendations and a reporting checklist covering the entire baseline lifecycle.

## Background & Motivation

Comparing models with human performance is a widely adopted practice in AI evaluation. From ImageNet to MMLU and various reasoning benchmarks, "surpassing human performance" is often regarded as a milestone achievement. However, these comparisons presuppose the existence of a **reliable human baseline** as a reference.

In reality, however, the construction of human baselines is often plagued by severe issues: inconsistent test sets (AI and humans evaluated on different subsets), insufficient sample sizes, lack of quality control, uncontrolled method effects across different groups, and unquantified uncertainty. Consequently, claims of "AI outperforming humans" can be highly misleading, as comparisons are not conducted under fair conditions.

The **Key Challenge** of this issue lies in the contrast where the AI evaluation community increasingly prioritizes the rigor of model evaluation (e.g., standardized benchmarks, leaderboards), yet **lacks an equivalent level of methodological attention on constructing human baselines**. A non-rigorous human baseline introduces three risks: (1) overestimating AI capabilities (as the human baseline is underestimated), (2) underestimating AI capabilities, and (3) misleading future research directions and policy-making.

The **Key Insight** of this paper stems from measurement theory—a well-established methodology in the social sciences encompassing questionnaire design, sampling, and reliability/validity testing—which is systematically transferred to the context of AI evaluation.

## Method

### Overall Architecture

The paper organizes its core framework around the "human baseline lifecycle," dividing the entire workflow into five distinct stages and offering specific methodological recommendations for each:

1. **Baseline Design & Implementation**
2. **Baseliner Recruitment**
3. **Baseline Execution**
4. **Baseline Analysis**
5. **Baseline Documentation**

### Key Designs

1. **Consistent and Representative Test Sets**: It is required that the human baseline and the AI be evaluated on **the exact same test set**. If budget constraints necessitate the use of a subset, the AI evaluation must also be computed on this subset, which should be **randomly sampled or stratified by difficulty/topic**. The **Design Motivation** is that many existing studies test humans on a subset but report the AI's score on the entire set, leading to incomparability.

2. **Sufficient Sample Sizes**: For generalist (non-expert) baselines, a **statistical power analysis** is recommended, with a rule of thumb requiring around 1,000 participants to represent the US adult population. For expert baselines, convenience samples can be utilized, but expert qualification criteria must be explicitly defined. The **Design Motivation** is that many studies use only 3-5 annotators, which is far from sufficient to represent "human level."

3. **Controlling Method Effects and Using Consistent Tasks**: Method effects refer to outcome biases caused by the data collection method rather than actual differences in capability. For instance, an AI can process an entire article in a single inference step, whereas humans might face time pressure or interface limitations. It is recommended to use **identical tasks** (same instructions, exemplars, and context) and randomize the order of questions and options. The **Design Motivation** is that when humans and AI face different presentations of the same task, performance discrepancies may stem from the format rather than capability.

4. **Controlling Level of Effort**: Fair comparisons should be conducted under similar resource investments, such as identical time limits or comparable financial costs. While AI "effort" can be measured by inference cost or sampling strategies, human "effort" is influenced by training, compensation, and time constraints.

5. **Uncertainty Quantification**: Statistical tests, confidence intervals, or performance distributions should be reported instead of mere point estimates. Consistent evaluation metrics and scoring methods (such as pass@k or majority vote applied uniformly across humans and AI) should be utilized.

### Reporting Checklist

The authors provide a detailed reporting checklist in Appendix B, covering:
- Demographic characteristics of the participants
- Sampling strategy and inclusion/exclusion criteria
- Informed consent and ethical review
- Task instructions and training materials
- Quality control measures
- Scoring methods and metric definitions
- Measures of uncertainty

## Key Experimental Results

### Main Results

The authors conducted a systematic review of 115 existing human baseline studies, yielding the following key findings:

| Aspect | Findings / Issues | Prevalence |
|------|---------|---------|
| Test Set Consistency | Humans and AI evaluated on different test sets | Significant portion of studies |
| Sample Size | Insufficient number of participants is widespread | Majority of studies |
| Sampling Strategy | Dominated by convenience sampling, lacking representativeness | Majority of studies |
| Quality Control | Lack of attention checks and exclusion criteria | Common |
| Uncertainty Reporting | Only report point estimates | Extremely common |
| Ethical Review | Fail to report IRB approval or informed consent | Significant proportion |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| Same test set vs. Different test sets | Comparison conclusions can be reversed | Subset selection bias can severely impact results |
| 3-5 participants vs. 1000 participants | Estimation errors differ significantly | Confidence intervals are extremely wide under small samples |
| With/Without control for method effects | Unknown impact but potentially significant | Differences in format can be mistaken for capabilities |

### Key Findings

- Existing human baselines are **systematically inadequate** in both rigor and transparency, failing to provide a reliable reference for human-AI comparisons.
- Many claims of "AI outperforming humans" may rest upon inadequate human baselines.
- Mature tools from measurement theory and survey methodology (such as stratified sampling, power analysis, and attention checks) remain severely underutilized in AI evaluation.
- Even when maximum rigor cannot be achieved (e.g., due to budget constraints), researchers should **explicitly discuss methodological limitations and narrow the scope of their claims**.

## Highlights & Insights

- **Interdisciplinary Perspective Transfer**: Systematically introduces decades of measurement theory accumulated in the social sciences into AI evaluation, addressing a long-neglected methodological gap.
- **Lifecycle Framework**: Completely covers everything from design and recruitment to analysis and documentation. It not only prescribes "what to do" but also explains "why to do it."
- **Practical Actionability**: Provides a ready-to-use reporting checklist (Appendix B), lowering the barrier to adoption.
- **Significant Implications for AI Policy and Governance**: If human baselines are unreliable, policy decisions regarding AI capabilities made based on these baselines may be biased.

## Limitations & Future Work

- The review scope of 115 studies may not be exhaustive; human baselines in certain domains (e.g., robotics, autonomous driving) may exhibit different characteristics.
- Although the recommendations are comprehensive, their feasibility remains challenging in high-cost scenarios (such as expert evaluations or long-duration tasks).
- The work does not deeply discuss "whether human performance itself is a meaningful reference"—on certain tasks, human performance may not represent a reasonable upper bound or baseline.
- The asymmetry of method effects between humans and AI (e.g., AI does not experience fatigue and is unaffected by order effects) poses a fundamental challenge, which is mentioned but not fully resolved in this paper.

## Related Work & Insights

- Echoes the reflections on NLU benchmark design raised by Bowman & Dahl (2021).
- Complements the pursuit of standardized evaluation in the HELM framework by Liang et al. (2023)—while HELM standardizes model evaluation, this study standardizes the construction of human baselines.
- Inspires reflection: In the evaluation of recommender systems, offline experiments often use implicit feedback as a "human baseline." However, such feedback likewise suffers from issues of representativeness, noise, and method effects.

## Rating

- Novelty: ⭐⭐⭐⭐ Offers a fresh perspective by systematically introducing measurement theory into AI evaluation.
- Experimental Thoroughness: ⭐⭐⭐⭐ The systematic review of 115 studies provides a solid empirical foundation.
- Writing Quality: ⭐⭐⭐⭐⭐ Extremely clear structure, with a thoughtful layered reading guide and high-density information tables.
- Value: ⭐⭐⭐⭐⭐ Provides a foundational methodological contribution to the broader AI evaluation community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Recommendations with Sparse Comparison Data: Provably Fast Convergence for Nonconvex Matrix Factorization](recommendations_with_sparse_comparison_data_provably_fast_convergence_for_noncon.md)
- [\[ICML 2025\] RLTHF: Targeted Human Feedback for LLM Alignment](rlthf_targeted_human_feedback_for_llm_alignment.md)
- [\[NeurIPS 2025\] Position: Towards Bidirectional Human-AI Alignment](../../NeurIPS2025/recommender/position_towards_bidirectional_human-ai_alignment.md)
- [\[ICML 2025\] MATCHA: Toward Safe and Human-Aligned Game Conversational Recommendation via Multi-Agent Decomposition](toward_safe_and_human-aligned_game_conversational_recommendation_via_multi-agent.md)
- [\[ICML 2025\] How to Set AdamW's Weight Decay as You Scale Model and Dataset Size](how_to_set_adamws_weight_decay_as_you_scale_model_and_dataset_size.md)

</div>

<!-- RELATED:END -->
