---
title: >-
  [Paper Note] Experiments or Outcomes? Probing Scientific Feasibility in Large Language Models
description: >-
  [ACL 2026][Interpretability][Scientific feasibility assessment] A controlled knowledge framework was constructed to systematically investigate how LLMs utilize experimental descriptions and outcome evidence in scientific feasibility assessment. The study found that outcome evidence is more reliable than experimental descriptions, and partial experimental information often leads to performance below the baseline using only parametric knowledge…
tags:
  - "ACL 2026"
  - "Interpretability"
  - "Scientific feasibility assessment"
  - "controlled knowledge framework"
  - "evidence robustness"
  - "experiments vs. outcomes"
  - "LLM reasoning"
date: 2026-05-08
content_hash: 85be623a944ca4fd
---

# Experiments or Outcomes? Probing Scientific Feasibility in Large Language Models

**Conference**: ACL 2026  
**arXiv**: [2604.18786](https://arxiv.org/abs/2604.18786)  
**Code**: [https://github.com/mohammadi-ali/scify](https://github.com/mohammadi-ali/scify)  
**Area**: Interpretability  
**Keywords**: Scientific feasibility assessment, controlled knowledge framework, evidence robustness, experiments vs. outcomes, LLM reasoning

## TL;DR

A controlled knowledge framework was constructed to systematically investigate how LLMs utilize experimental descriptions and outcome evidence in scientific feasibility assessment. The study found that outcome evidence is more reliable than experimental descriptions, and partial experimental information often leads to performance below the baseline using only parametric knowledge, revealing the fragility of LLM reasoning.

## Background & Motivation

**Background**: LLMs are increasingly utilized in scientific workflows (literature review, hypothesis generation, experimental planning), but their ability to execute the fundamental task of scientific feasibility assessment remains unclear. Feasibility assessment requires determining whether a claim aligns with existing knowledge and whether experimental evidence supports or refutes it.

**Limitations of Prior Work**: Existing research either focuses on hypothesis generation rather than evaluation, mixes model parametric knowledge with retrieval without isolating their respective contributions, or examines adherence to external knowledge in non-scientific scenarios. Three key questions remain unanswered: (RQ1) Can LLMs evaluate feasibility using only parametric knowledge? (RQ2) How does providing experimental/outcome context change their judgments? (RQ3) How robust are these judgments when information is incomplete?

**Key Challenge**: Intuitively, more evidence should assist in judgment, but partial or noisy evidence might misguide the model—can LLMs handle incomplete information gracefully?

**Goal**: To understand the impact of evidence types on LLM feasibility judgments through systematic control of experiment and outcome visibility.

**Key Insight**: Design 4 knowledge conditions (Hypothesis only / +Experiments / +Outcomes / +Both) and stability analysis (progressive removal of evidence).

**Core Idea**: Outcome evidence is generally more reliable than experimental descriptions, and partial evidence often leads to brittle collapse rather than graceful degradation.

## Method

### Overall Architecture

This is a probe-based analysis study: given a scientific hypothesis $h$, LLMs are required to output "feasible/infeasible + reasoning" under strictly controlled evidence visibility. By comparing changes in judgments across different evidence conditions, the study infers which types of information the model relies on. Specifically, four knowledge conditions are established—H (Hypothesis only), H+E (plus Experimental descriptions), H+O (plus Outcome summaries), and H+E+O (both). Proportional parameters $k_1, k_2 \in \{0, 0.5, 1.0\}$ control the visibility of experiments and outcomes respectively. Each configuration is averaged over 5 random samples to ensure evidence type is the sole variable.

### Key Designs

**1. Controlled Knowledge Framework: Fixing the prediction task while varying evidence**

Prior work often conflated internal model knowledge, retrieved content, and experimental information, failing to distinguish which evidence type was active. This study maintains an identical prediction task (always outputting feasibility judgment plus reasoning) while only varying the context $x \in \{\emptyset, \mathcal{E}^*, \mathcal{O}^*, (\mathcal{E}^*, \mathcal{O}^*)\}$ accompanying the hypothesis. Furthermore, experimental descriptions and outcomes are extracted directly from source papers rather than retrieved, eliminating quality fluctuations in the evidence itself. Thus, prediction variances can be cleanly attributed to evidence types rather than task difficulty or evidence noise.

**2. Stability Analysis: Assessing graceful degradation vs. brittle collapse**

Real-world scientific reasoning often relies on incomplete evidence. A robust reasoner should utilize available information such that performance scales smoothly with evidence volume. This study decreases the visibility ratio of experiments and outcomes from 1.0 to 0.5 and then to 0 (adjusting $k_1, k_2$), observing whether the performance curve exhibits monotonic degradation or non-monotonic collapse. A "below-baseline rate" is defined—the frequency at which performance under partial evidence conditions falls below the zero-evidence baseline (H). If half the evidence yields worse results than no evidence, it indicates the model performs surface alignment rather than deep reasoning.

**3. Multi-dimensional Evaluation: Selecting credible metrics under class imbalance**

Feasibility assessment is an imbalanced binary classification task where accuracy can be misleadingly inflated by the majority class. Consequently, the study reports Accuracy, macro-F1, and MCC (the latter being more informative for imbalanced classification). ROUGE lexical overlap between generated explanations and reference explanations serves as a secondary diagnostic signal. The evaluation spans five leading models—GPT-5.1, GPT-4o, Gemini-2.5-Pro/Flash, and Grok-4.1-fast—across two datasets to ensure findings are consistent across platforms.

### Loss & Training

A pure evaluation study using zero-shot prompting. All models use identical task instructions; no training or fine-tuning is involved.

## Key Experimental Results

### Main Results

Performance of GPT-5.1 on the MoF dataset:

| Condition | Accuracy | F1_macro | MCC |
|-----------|----------|----------|-----|
| H (Hypothesis only) | 0.68 | 0.67 | 0.42 |
| H+E (100% Experiment) | 0.70 | 0.69 | 0.44 |
| H+O (100% Outcome) | 0.66 | 0.66 | 0.33 |
| H+E+O (All) | 0.66 | 0.66 | 0.33 |

### Ablation Study

On the Reasons dataset (GPT-5.1):

| Condition | Accuracy | Description |
|-----------|----------|-------------|
| H | 0.84 | Parametric knowledge baseline |
| H+E (50%) | 0.85 | Slight improvement |
| H+O (100%) | 0.92 | Strong outcome evidence |
| H+E+O (100%) | 0.93 | Optimal |
| H+E (50%) + H+O (50%) | 0.90 | Partial evidence utility |

### Key Findings

- Outcome evidence (outcomes) generally improves feasibility judgments more effectively than experimental descriptions (experiments)—on the Reasons dataset, H+O consistently outperforms H+E.
- Experimental descriptions can be "brittle": partial experimental information ($k_1=0.5$) led to performance below the hypothesis-only baseline across multiple models, suggesting models perform surface feature matching rather than true understanding of experimental design.
- Degradation is often non-monotonic—performance at $k_1=0.5$ can be worse than at $k_1=0$—indicating models do not follow a "use what information is available" reasoning logic.
- Gemini-2.5-Pro showed the most instability under experimental description conditions (dropping from 0.67 to 0.48), exposing significant surface alignment issues.
- Even for the strongest model, GPT-5.1, providing complete experiments and outcomes does not necessarily outperform providing outcomes alone (MCC is the same or lower on the MoF dataset).

## Highlights & Insights

- The finding that "partial evidence is harmful" is a profound and cautionary insight: it reveals a fundamental fragility in LLM scientific reasoning—models act more like pattern matchers than logical reasoners understanding experimental structures. This serves as a significant warning for using LLMs in scientific review and decision-making.
- The experimental design of the controlled knowledge framework is elegant: by keeping the task constant and varying only the context, it achieves clean causal inference. This methodology is transferable to other studies evaluating LLM knowledge utilization.
- The "Outcomes > Experiments" finding implies that LLMs are more proficient at processing declarative knowledge ("what happened") than procedural knowledge ("how it was done")—consistent with the nature of LLM training data.

## Limitations & Future Work

- Only zero-shot evaluation was used; fine-tuning or few-shot settings might yield different results.
- Feasibility judgment was simplified into binary classification, whereas real scientific feasibility is often a spectrum.
- The extraction quality of experiments and outcomes might influence conclusions—incomplete extraction itself could cause "fragility."
- Explanation quality was assessed only via ROUGE lexical overlap, which cannot truly measure the logical correctness of scientific reasoning.
- Only commercial API models were tested; the performance of open-source models may differ.

## Related Work & Insights

- **vs Qi et al. (2023) / Yang et al. (2024)**: Focused on hypothesis generation rather than evaluation; this paper fills the gap in feasibility judgment.
- **vs Jansen et al. (2025)**: Mixed internal knowledge and retrieval without isolating contributions; this controlled framework achieves clean separation.
- **vs Mohammadi et al. (2025)**: Studied LLM adherence to external knowledge but in non-scientific scenarios; this work focuses on evidence utilization in scientific reasoning.

## Rating
- Novelty: ⭐⭐⭐⭐ Innovative experimental design with controlled knowledge framework and stability analysis.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 5 models × 2 datasets × 9 evidence conditions × 5 random seeds.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear problem formalization and rigorous experimental design.
- Value: ⭐⭐⭐⭐ Significantly advances the understanding of LLM scientific reasoning capabilities.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] MINED: Probing and Updating with Multimodal Time-Sensitive Knowledge for Large Multimodal Models](mined_probing_and_updating_with_multimodal_time-sensitive_knowledge_for_large_mu.md)
- [\[ACL 2026\] Knowledge Vector of Logical Reasoning in Large Language Models](knowledge_vector_of_logical_reasoning_in_large_language_models.md)
- [\[ACL 2026\] Compositional Steering of Large Language Models with Steering Tokens](compositional_steering_of_large_language_models_with_steering_tokens.md)
- [\[ACL 2026\] Towards Intrinsic Interpretability of Large Language Models: A Survey of Design Principles and Architectures](towards_intrinsic_interpretability_of_large_language_modelsa_survey_of_design_pr.md)
- [\[ICML 2026\] Towards Atoms of Large Language Models](../../ICML2026/interpretability/towards_atoms_of_large_language_models.md)

</div>

<!-- RELATED:END -->
