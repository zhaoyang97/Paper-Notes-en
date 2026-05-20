---
title: >-
  [Paper Note] What Makes LLMs Effective Sequential Recommenders? A Study on Preference Intensity and Temporal Context
description: >-
  [ACL 2026][Recommender Systems][Sequential Recommendation] This paper identifies that binary preference modeling in existing LLM-based recommender systems discards two critical signals—preference intensity and temporal c…
tags:
  - "ACL 2026"
  - "Recommender Systems"
  - "Sequential Recommendation"
  - "Preference Alignment"
  - "Preference Intensity"
  - "Temporal Context"
  - "DPO"
date: 2026-05-08
content_hash: da0fa92ac472b864
---

# What Makes LLMs Effective Sequential Recommenders? A Study on Preference Intensity and Temporal Context

**Conference**: ACL 2026
**arXiv**: [2506.02261](https://arxiv.org/abs/2506.02261)  
**Code**: [https://github.com/zyouyang/RecPO](https://github.com/zyouyang/RecPO)  
**Area**: Recommender Systems
**Keywords**: Sequential Recommendation, Preference Alignment, Preference Intensity, Temporal Context, DPO

## TL;DR

This paper identifies that binary preference modeling in existing LLM-based recommender systems discards two critical signals—preference intensity and temporal context—and proposes RecPO, a framework that incorporates both factors into preference optimization via adaptive reward margins, substantially outperforming S-DPO and other baselines across five datasets.

## Background & Motivation

**State of the Field**: Large language models are increasingly applied to sequential recommendation tasks, leveraging textualized interaction histories to predict users' next items of interest. Preference alignment techniques such as DPO and S-DPO have become the dominant training paradigm.

**Limitations of Prior Work**: Existing preference alignment methods (DPO, S-DPO) reduce all preferences to binary pairwise comparisons—distinguishing only "liked" from "disliked"—thereby discarding substantial information. In practice, user ratings ranging from 1 to 5 encode structured preference intensity (strongly liked vs. mildly liked), and more recent interactions are more reflective of users' current intent.

**Root Cause**: A fundamental mismatch exists between binary preference modeling and human decision-making behavior. Humans exhibit structured preferences (varying intensities) and temporally sensitive preferences (recency matters), both of which are entirely neglected by existing methods.

**Paper Goals**: (1) Systematically validate the importance of preference intensity and temporal context for LLM-based recommendation; (2) Design a preference optimization framework that exploits both factors.

**Starting Point**: Drawing on established characteristics of human decision-making from behavioral economics and cognitive science, the paper uses controlled experiments to demonstrate that retaining negative feedback alongside structured ratings yields substantial performance gains, providing an empirical foundation for method design.

**Core Idea**: Encode preference intensity and interaction recency into the DPO objective via adaptive reward margins, enabling the model to learn preference representations that better align with human decision-making patterns.

## Method

### Overall Architecture

RecPO adopts a two-stage training paradigm: an initial SFT stage adapts a general-purpose LLM to the recommendation task, followed by preference optimization with adaptive margins for further alignment. The input consists of the user's complete interaction history (including both positive and negative feedback with ratings), and the output is the recommended next item selected from a candidate set. Unlike S-DPO, RecPO retains negative interaction records and treats ratings as structured preference signals.

### Key Designs

1. **Complete and Structured Feedback Input**:

    - Function: Provides the model with rich preference signals.
    - Mechanism: Rather than filtering out negative interactions as in S-DPO, the full interaction sequence is retained. Each historical item is annotated with a preference signal (explicit rating or a structured score derived from implicit feedback) in the format "[ItemTitle] | Rating: [ItemRating]". For datasets lacking explicit ratings, proxy signals such as play duration or play count are used.
    - Design Motivation: Proof-of-concept experiments show that recommendation performance is optimal only when both complete feedback and structured ratings are retained simultaneously. Retaining negative interactions without ratings introduces noise and degrades performance, demonstrating that the two components are mutually necessary.

2. **Adaptive Reward Margin**:

    - Function: Dynamically adjusts the optimization strength between preference pairs based on preference intensity and interaction recency.
    - Mechanism: For each preference pair $(y_p, y_d)$, the margin is defined as $\gamma_r = \lambda \cdot \phi(s_p, \Delta t_p) / \phi(s_d, \Delta t_d)$, where $\phi(s, \Delta t) = s / (\Delta t)^{0.5}$ is a utility function, $s$ denotes the preference score, and $\Delta t$ denotes the temporal distance from the current decision point. Larger preference gaps and greater recency yield larger margins and stronger optimization signals.
    - Design Motivation: A uniform margin cannot distinguish between fundamentally different preference contrasts such as "5 vs. 1" and "4 vs. 3". The ratio-based margin amplifies training gradients in scenarios where users exhibit low rating variance.

3. **Plackett-Luce Listwise Ranking Extension**:

    - Function: Generalizes pairwise comparisons to listwise ranking with multiple negative samples.
    - Mechanism: Based on the PL model, adaptive margins are embedded into a listwise preference distribution, pairing each positive sample with multiple negatives. Setting $\lambda=0$ recovers standard S-DPO, ensuring generality.
    - Design Motivation: A single negative sample is insufficient to cover the user's "dislike" space; listwise ranking enables the model to simultaneously learn relative ordering among multiple negative samples.

### Loss & Training

The final loss function extends S-DPO by incorporating the adaptive margin term $\gamma_r$, with $\lambda$ controlling the margin influence (default $\lambda=2$). Training follows an SFT-then-alignment pipeline, with preference optimization initialized from the SFT checkpoint. Default preference scores and temporal delays are assigned to negatively sampled items and history interactions lacking explicit feedback.

## Key Experimental Results

### Main Results

| Dataset | Metric | RecPO (LLaMA3-8B) | S-DPO | Gain |
|--------|------|------|----------|------|
| MovieLens | HR@1 | 0.3451 | 0.2902 | +18.9% |
| Amazon-Books | HR@1 | 0.5802 | 0.5065 | +14.6% |
| BeerAdvocate | HR@1 | 0.5771 | 0.4698 | +22.8% |
| Steam | HR@1 | 0.4672 | 0.3588 | +30.2% |
| LastFM | HR@1 | 0.6830 | 0.5719 | +19.4% |

RecPO also substantially outperforms all baselines on Qwen-7B, with HR@1 improvements ranging from 10% to 30%.

### Ablation Study

| Configuration | MovieLens | Amazon-Books | BeerAdvocate | Steam | LastFM |
|------|---------|------|------|------|------|
| –I –T (= S-DPO) | 0.2902 | 0.5065 | 0.4698 | 0.3588 | 0.5719 |
| –T (intensity only) | 0.3343 | 0.5661 | 0.6143 | 0.4202 | 0.6544 |
| RecPO (full) | 0.3451 | 0.5802 | 0.5771 | 0.4672 | 0.6830 |

### Key Findings

- **Preference intensity contributes most**: Introducing preference intensity alone (–T) yields significant gains, indicating that structured preference signals are the most critical factor.
- **Temporal context provides complementary gains**: Adding temporal context on top of preference intensity further improves performance on 4 out of 5 datasets, with the largest gain on Steam (0.4202 → 0.4672).
- **Margin function form**: The ratio-based form (default) outperforms both Log Diff and Log Ratio alternatives.
- **Human-aligned behavior**: RecPO learns four human decision-making patterns—immediate gratification prioritization, temptation resistance, implicit aversion modeling, and robustness across context lengths (HR@1 variance 8.7% vs. 17.8% for S-DPO).

## Highlights & Insights

- **Empiricism-first methodology**: The paper first demonstrates the importance of preference intensity and temporal context through controlled experiments before designing the method accordingly. This hypothesis-driven research paradigm is worth emulating.
- **Concise and effective margin design**: The utility function $\phi(s, \Delta t) = s / (\Delta t)^{0.5}$ is elegantly simple, with a single hyperparameter $\lambda$ controlling its influence, facilitating reproducibility.
- **Emergence of implicit aversion modeling**: The model learns to identify users' least-preferred items without explicit aversion labels, suggesting that structured preference signals can implicitly encode negative preferences.

## Limitations & Future Work

- Only simplified sequential preference structures and satisfaction delay are considered as contextual factors; real-world human decision-making involves more complex preference hierarchies.
- Gains on implicit feedback datasets are relatively modest, as the homogeneity of proxy signals limits the advantages of the approach.
- Future work may explore cognitively grounded preference modeling beyond recommendation, in broader preference-sensitive tasks.

## Related Work & Insights

- **vs. S-DPO**: S-DPO employs listwise optimization with multiple negatives but uses a uniform margin. RecPO is a natural extension of S-DPO (recovering S-DPO when $\lambda=0$), introducing preference intensity and temporal information via adaptive margins.
- **vs. SimPO**: SimPO uses a fixed margin with length normalization, but fixed margins cannot capture differences across preference pairs, and a lower Valid Ratio constrains deployment.

## Rating

- Novelty: ⭐⭐⭐⭐ The cognitive science perspective on preference alignment in recommendation is highly inspiring.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Five datasets, two backbones, extensive ablations, and behavioral analyses make for a very comprehensive evaluation.
- Writing Quality: ⭐⭐⭐⭐⭐ The empiricism-first narrative structure is clear and well-organized.
- Value: ⭐⭐⭐⭐ Provides a practical and principled improvement direction for preference alignment in LLM-based recommender systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] What Makes an Ideal Quote? Recommending "Unexpected yet Rational" Quotations via Novelty](what_makes_an_ideal_quote_recommending_34unexpected_yet_rational34_quotations_vi.md)
- [\[ACL 2026\] Personalized Benchmarking: Evaluating LLMs by Individual Preferences](personalized_benchmarking_evaluating_llms_by_individual_preferences.md)
- [\[ACL 2026\] Decisive: Guiding User Decisions with Optimal Preference Elicitation from Unstructured Documents](decisive_guiding_user_decisions_with_optimal_preference_elicitation_from_unstruc.md)
- [\[ACL 2026\] Where and What: Reasoning Dynamic and Implicit Preferences in Situated Conversational Recommendation](where_and_what_reasoning_dynamic_and_implicit_preferences_in_situated_conversati.md)
- [\[AAAI 2026\] HyMoERec: Hybrid Mixture-of-Experts for Sequential Recommendation](../../AAAI2026/recommender/hymoerec_hybrid_mixture-of-experts_for_sequential_recommendation.md)

</div>

<!-- RELATED:END -->
