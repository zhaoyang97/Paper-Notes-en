---
title: >-
  [Paper Note] What Makes LLMs Effective Sequential Recommenders? A Study on Preference Intensity and Temporal Context
description: >-
  [ACL 2026][Recommender Systems][Sequential Recommendation] This paper reveals that binary preference modeling in existing LLM-based recommender systems neglects two critical pieces of information: preference intensity an…
tags:
  - "ACL 2026"
  - "Recommender Systems"
  - "Sequential Recommendation"
  - "Preference Alignment"
  - "Preference Intensity"
  - "Temporal Context"
  - "DPO"
date: 2026-05-08
content_hash: fb10db2a25eb737e
---

# What Makes LLMs Effective Sequential Recommenders? A Study on Preference Intensity and Temporal Context

**Conference**: ACL 2026  
**arXiv**: [2506.02261](https://arxiv.org/abs/2506.02261)  
**Code**: [https://github.com/zyouyang/RecPO](https://github.com/zyouyang/RecPO)  
**Area**: Recommender Systems  
**Keywords**: Sequential Recommendation, Preference Alignment, Preference Intensity, Temporal Context, DPO

## TL;DR

This paper reveals that binary preference modeling in existing LLM-based recommender systems neglects two critical pieces of information: preference intensity and temporal context. It proposes the RecPO framework to incorporate these factors into preference optimization through an adaptive reward margin, significantly outperforming baselines like S-DPO across five datasets.

## Background & Motivation

**Background**: Large Language Models (LLMs) are being widely applied to sequential recommendation tasks to predict a user's next likely interaction via text-based interaction histories. Current mainstream methods employ preference alignment techniques such as DPO/S-DPO for training.

**Limitations of Prior Work**: Existing preference alignment methods (DPO, S-DPO) treat all preferences as binary pairwise comparisons—distinguishing only between "liked" and "disliked"—thus discarding substantial valuable information. In real-world user behavior, structured preference intensity differences exist (e.g., ratings from 1 to 5), and more recent interactions better reflect current user intent.

**Key Challenge**: A fundamental mismatch exists between binary preference modeling and human decision-making behavior, as humans exhibit structured preferences (varying intensities) and time-sensitive preferences (recency importance), which are entirely ignored by current methods.

**Goal**: (1) Systematically verify the importance of preference intensity and temporal context for LLM recommendations; (2) design a preference optimization framework capable of leveraging these factors.

**Key Insight**: Starting from known characteristics of human decision-making in behavioral economics and cognitive science, this study provides an empirical foundation by demonstrating through controlled experiments that preserving negative feedback and structured ratings significantly enhances recommendation performance.

**Core Idea**: Encode preference intensity and interaction recency into the DPO objective function using an adaptive reward margin, enabling the model to learn preference representations that better align with human decision patterns.

## Method

### Overall Architecture

RecPO follows a two-stage training paradigm: first, it adapts a general LLM into a recommendation model using SFT; then, it performs further alignment through preference optimization with an adaptive margin. The input consists of the user's complete interaction history (including positive/negative feedback and ratings), and the output is the next recommended item selected from a candidate set. Unlike S-DPO, RecPO retains user records of negative interactions and utilizes ratings as structured preference signals.

### Key Designs

1. **Complete and Structured Feedback Input**:

    - **Function**: Provides rich preference signals to the model.
    - **Mechanism**: Instead of filtering out negative interactions as in S-DPO, the complete interaction sequence is preserved. Each historical item is accompanied by a preference signal (explicit ratings or structured scores converted from implicit feedback), formatted as "[ItemTitle] | Rating: [ItemRating]". For datasets without explicit ratings, proxies such as gameplay duration or play counts are used.
    - **Design Motivation**: Proof-of-concept experiments show that recommendation performance is optimal only when both complete feedback and structured ratings are preserved simultaneously. Retaining negative interactions without ratings can introduce noise and degrade performance, indicating both are indispensable.

2. **Adaptive Reward Margin**:

    - **Function**: Dynamically adjusts the optimization strength between preference pairs based on preference intensity and temporal recency.
    - **Mechanism**: For each preference pair $(y_p, y_d)$, a margin $\gamma_r = \lambda \cdot \phi(s_p, \Delta t_p) / \phi(s_d, \Delta t_d)$ is defined, where $\phi(s, \Delta t) = s / (\Delta t)^{0.5}$ is a utility function, $s$ is the preference score, and $\Delta t$ is the temporal distance from the current decision point. Larger preference differences and shorter temporal distances lead to a larger margin and stronger optimization signal.
    - **Design Motivation**: A uniform margin cannot distinguish between essentially different preference comparisons such as "5 stars vs. 1 star" and "4 stars vs. 3 stars." A ratio-based margin amplifies training gradients in scenarios with low user rating volatility.

3. **Plackett-Luce List-wise Ranking Extension**:

    - **Function**: Generalizes pairwise comparisons to list-wise ranking with multiple negative samples.
    - **Mechanism**: Based on the PL model, the adaptive margin is embedded into the list-wise preference distribution, with each positive sample paired with multiple negative samples. When $\lambda=0$, it degrades to standard S-DPO, ensuring generality.
    - **Design Motivation**: A single negative sample struggles to sufficiently cover a user's "dislike" space; list-wise ranking allows the model to learn relative ranking relationships across multiple negative samples simultaneously.

### Loss & Training

The final loss function incorporates the adaptive margin term $\gamma_r$ into the S-DPO framework, with $\lambda$ controlling the influence of the margin (default $\lambda=2$). Training proceeds with SFT followed by preference alignment, initializing the latter from the SFT checkpoint. For negative sampling and historical interactions without explicit feedback, default preference scores and time delays are assigned.

## Key Experimental Results

### Main Results

| Dataset | Metric | RecPO (LLaMA3-8B) | S-DPO | Gain |
|--------|------|------|----------|------|
| MovieLens | HR@1 | 0.3451 | 0.2902 | +18.9% |
| Amazon-Books | HR@1 | 0.5802 | 0.5065 | +14.6% |
| BeerAdvocate | HR@1 | 0.5771 | 0.4698 | +22.8% |
| Steam | HR@1 | 0.4672 | 0.3588 | +30.2% |
| LastFM | HR@1 | 0.6830 | 0.5719 | +19.4% |

RecPO significantly outperforms all baselines on Qwen-7B as well, with HR@1 improvements ranging between 10% and 30%.

### Ablation Study

| Configuration | MovieLens | Amazon-Books | BeerAdvocate | Steam | LastFM |
|------|---------|------|------|------|------|
| –I –T (=S-DPO) | 0.2902 | 0.5065 | 0.4698 | 0.3588 | 0.5719 |
| –T (Intensity only) | 0.3343 | 0.5661 | 0.6143 | 0.4202 | 0.6544 |
| RecPO (Full) | 0.3451 | 0.5802 | 0.5771 | 0.4672 | 0.6830 |

### Key Findings

- **Preference intensity contributes most**: Simply adding preference intensity (–T) yields significant improvements, showing that structured preference signals are the most critical factor.
- **Temporal context provide complementary gains**: Adding temporal context on top of preference intensity further improves results for 4 out of 5 datasets (Steam showed the largest gain, from 0.4202 to 0.4672).
- **Margin function form**: The ratio-based form (default) outperforms alternatives like Log Diff and Log Ratio.
- **Human alignment behavior**: RecPO learns four human decision patterns: immediate gratification prioritization, resisting temptation, implicit aversion modeling, and robustness across context lengths (HR@1 variance of 8.7% vs. 17.8% for S-DPO).

## Highlights & Insights

- **Empirical-first methodology**: Proving the importance of preference intensity and temporal context through controlled experiments before designing the method. This hypothesis-driven research paradigm is highly effective.
- **Simple yet effective margin design**: The form $\phi(s, \Delta t) = s / (\Delta t)^{0.5}$ is concise, controlling influence via a single hyperparameter $\lambda$, making it easy to reproduce.
- **Emergence of implicit aversion modeling**: The ability to identify users' most disliked items despite lacking explicit aversion labels suggests that structured preference signals can implicitly encode negative preferences.

## Limitations & Future Work

- Only considers simplified sequential preference structures and satisfaction delay as contextual factors; real-world human decision-making involves more complex preference hierarchies.
- Smaller improvements on implicit feedback datasets, where the homogeneity of proxy signals limits advantages.
- Future work could explore the application of cognitively plausible preference modeling in non-recommendation preference tasks.

## Related Work & Insights

- **vs S-DPO**: S-DPO uses list-wise optimization with multiple negative samples but a uniform margin. RecPO is a natural extension of S-DPO (degenerating to S-DPO when $\lambda=0$), introducing preference intensity and temporal information through an adaptive margin.
- **vs SimPO**: SimPO uses a fixed margin and length regularization, but a fixed margin fails to capture differences between various preference pairs, and its lower Valid Ratio affects deployment.

## Rating

- Novelty: ⭐⭐⭐⭐ Approaching recommender system preference alignment from a cognitive science perspective is highly inspiring.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely comprehensive across five datasets, two backbones, multiple ablations, and behavioral analyses.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear narrative structure with an empirical-first approach.
- Value: ⭐⭐⭐⭐ Provides a practical direction for improving preference alignment in LLM-based recommender systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] What Makes an Ideal Quote? Recommending "Unexpected yet Rational" Quotations via Novelty](what_makes_an_ideal_quote_recommending_34unexpected_yet_rational34_quotations_vi.md)
- [\[ACL 2026\] Personalizing LLMs with Binary Feedback: A Preference-Corrected Optimization Framework](personalizing_llms_with_binary_feedback_a_preference-corrected_optimization_fram.md)
- [\[ACL 2026\] Where and What: Reasoning Dynamic and Implicit Preferences in Situated Conversational Recommendation](where_and_what_reasoning_dynamic_and_implicit_preferences_in_situated_conversati.md)
- [\[ACL 2026\] SenseJudge: Human-Centric Preference-Driven Judgment Framework](sensejudge_human-centric_preference-driven_judgment_framework.md)
- [\[ACL 2026\] Bridging Language and Items for Retrieval and Recommendation: Benchmarking LLMs as Semantic Encoders](bridging_language_and_items_for_retrieval_and_recommendation_benchmarking_llms_a.md)

</div>

<!-- RELATED:END -->
