---
title: >-
  [Paper Note] Easy Samples Are All You Need: Self-Evolving LLMs via Data-Efficient Reinforcement Learning
description: >-
  [ACL 2026][Reinforcement Learning][To be supplemented] To be supplemented after thorough reading.
tags:
  - ACL 2026
  - Reinforcement Learning
  - To be supplemented
date: 2026-05-08
content_hash: 2dc8c59083d3f1c2
---

# Easy Samples Are All You Need: Self-Evolving LLMs via Data-Efficient Reinforcement Learning

**Conference**: ACL 2026
**arXiv**: [2604.18639](https://arxiv.org/abs/2604.18639)
**Code**: [https://github.com/YuZhiyin/EasyRL](https://github.com/YuZhiyin/EasyRL)
**Area**: Reinforcement Learning / Data-Efficient Training
**Keywords**: Data-efficient RL, self-evolving LLM, pseudo-labeling, easy-to-hard curriculum, cognitive learning theory

## TL;DR

This paper proposes EasyRL, a cognitively inspired framework that uses only 10% easy labeled data for warmup initialization via knowledge transfer, then progressively masters hard unlabeled data through divide-and-conquer pseudo-labeling and difficulty-progressive self-training, consistently outperforming supervised GRPO trained on the full dataset.

## Background & Motivation

**Background**: RLVR has emerged as a key post-training paradigm for enhancing LLM reasoning. Existing approaches fall into two categories: supervised (relying on annotated answers or reward models at high annotation cost) and unsupervised (constructing rewards via voting or entropy estimation, prone to collapse or reward hacking).

**Limitations of Prior Work**: (1) Supervised methods require large amounts of high-quality labeled data, with annotation costs particularly prohibitive for hard problems; (2) unsupervised methods yield limited and unstable performance gains, susceptible to model collapse or reward hacking; (3) neither category accounts for the difficulty distribution of data — in practice, easy problems are far cheaper to annotate than hard ones.

**Key Challenge**: Hard problems are costly to annotate yet highly informative, while easy problems are cheap to annotate but insufficient on their own. The central question is how to leverage a small amount of easy labeled data to progressively master a large corpus of hard unlabeled data.

**Goal**: To design a cognitively inspired RL framework that, starting from limited easy labeled data, self-evolves to handle increasingly difficult reasoning tasks.

**Key Insight**: Vygotsky's Zone of Proximal Development (ZPD) theory — learners first internalize knowledge from simple, accessible examples, then progressively extend to harder challenges with minimal external guidance.

**Core Idea**: (1) Train a warmup model via GRPO on a small set of easy labeled data; (2) apply a divide-and-conquer strategy to pseudo-label unlabeled data — consistency selection for low-uncertainty samples and reflective resolution for medium-uncertainty samples; (3) iteratively expand the model's capability boundary through difficulty-progressive self-training.

## Method

### Overall Architecture

EasyRL proceeds in three stages: (1) **Knowledge Transfer** — train a warmup model on easy labeled data using GRPO; (2) **Divide-and-Conquer Pseudo-Labeling** — the warmup model performs multiple inference passes on unlabeled data, partitioning samples into low/medium/high uncertainty groups, with high-quality pseudo-labels constructed via consistency selection and reflective resolution respectively; (3) **Difficulty-Progressive Self-Training** — iterative RL training on a mixture of labeled and pseudo-labeled data, with high-uncertainty samples from each round re-pseudo-labeled in the next.

### Key Designs

1. **Divide-and-Conquer Pseudo-Labeling Strategy**:

    - **Function**: Construct high-quality pseudo-labels from unlabeled data.
    - **Mechanism**: Generate $N$ independent inference passes for each unlabeled sample. *Consistency selection* — if all $N$ outputs agree on the same answer, it is directly adopted as the pseudo-label (low uncertainty). *Reflective resolution* — when outputs disagree, the predictive entropy $H(x)$ is computed; if $H(x) \leq \tau_t$, a reflection mechanism re-evaluates candidate answers to produce a pseudo-label (medium uncertainty). High-uncertainty samples ($H(x) > \tau_t$) are deferred to the next iteration.
    - **Design Motivation**: Naïve majority voting yields insufficient pseudo-label quality. The three-tier treatment — confidently adopt fully consistent outputs, correct partially consistent outputs via reflection, and defer highly uncertain ones — maximizes overall pseudo-label quality.

2. **Difficulty-Progressive Self-Training**:

    - **Function**: Gradually extend the model's capability boundary to harder problems.
    - **Mechanism**: In each iteration, the current model $\pi_i$ re-pseudo-labels the high-uncertainty samples left from the previous round, filters them, and mixes them with labeled data for RL training to obtain $\pi_{i+1}$. As model capability improves, previously unconfident hard samples are progressively incorporated. The sequence $\pi_1, \pi_2, \ldots, \pi_n$ forms a chain of models with monotonically increasing capability.
    - **Design Motivation**: This directly instantiates Vygotsky's ZPD theory — the pseudo-labeled dataset at each round corresponds precisely to the model's current "zone of proximal development," i.e., tasks that slightly exceed current capability yet remain learnable.

3. **Cognitive Learning Curve Simulation**:

    - **Function**: Theoretical foundation unifying the framework.
    - **Mechanism**: EasyRL simulates the human cognitive acquisition curve: first learning basic rules from simple examples (knowledge transfer), then transferring knowledge to novel, harder problems via analogy and self-reflection (divide-and-conquer + self-training). The consistency rate (ConsRate) of pseudo-labels increases across iterations, empirically confirming progressive capability growth.
    - **Design Motivation**: Importing learning theory from cognitive science into RL framework design provides principled methodological guidance.

### Loss & Training

Standard GRPO objective. Correctness rewards: $r=1$ (correct), $r=-0.5$ (format error), $r=0$ (otherwise). Evaluated on Qwen2.5-Math-1.5B/7B and Llama-3.2-3B. Easy labeled data constitutes 10% of the full dataset (easy subsets selected by AoPS difficulty ratings).

## Key Experimental Results

### Main Results

| Model / Method | Math Avg. | Science Avg. | Labeled Data |
|---|---|---|---|
| Qwen2.5-Math-1.5B Base | 32.6 | 1.5 | 0 |
| w/ Supervised GRPO (10%) | 35.7 | 7.9 | 10% |
| w/ Unsupervised EMPO | 38.5 | 15.6 | 0 |
| w/ EasyRL Iter3 | **40.3** | **19.4** | 10% |
| Qwen2.5-Math-7B Base | 38.5 | 24.1 | 0 |
| w/ Supervised GRPO (10%) | 43.3 | 27.4 | 10% |
| w/ EasyRL Iter3 | **50.6** | **30.6** | 10% |

### Ablation Study

| Configuration | Effect | Note |
|---|---|---|
| Knowledge transfer only (warmup) | Baseline | Easy data only |
| + Divide-and-conquer pseudo-labeling Iter1 | Improvement | Incorporates easy unlabeled data |
| + Iter2 | Further improvement | Incorporates medium-difficulty data |
| + Iter3 | Best | Incorporates more hard data |

### Key Findings

- EasyRL with only 10% easy labeled data surpasses Supervised GRPO trained on the full dataset.
- Iterative self-training yields consistent gains: Iter1→Iter2→Iter3 shows steady improvement across both math and science benchmarks.
- Pseudo-label consistency rate increases across iterations, confirming progressive capability self-evolution.
- EasyRL generalizes to out-of-domain tasks (science reasoning), indicating acquisition of general reasoning ability rather than domain-specific patterns.
- The reflection mechanism significantly improves pseudo-label quality for medium-uncertainty samples.

## Highlights & Insights

- **The "easy samples are all you need" finding has practical value**: Annotating hard problems is prohibitively expensive; EasyRL demonstrates that labeling only easy problems is sufficient to cover hard ones through self-evolution — a finding of significant relevance for annotation-constrained real-world scenarios.
- **The divide-and-conquer strategy offers a natural uncertainty-stratified data treatment**: The three-tier pipeline — fully consistent → reflective correction → deferred processing — provides an optimal handling strategy at each level.
- **Grounding in cognitive science theory provides principled methodological guidance**: ZPD theory is not merely an analogy here, but actively informs concrete design choices (easy-to-hard ordering, progressive expansion).

## Limitations & Future Work

- The pseudo-labeling process with multiple inference passes incurs non-trivial computational overhead.
- The convergence speed and performance ceiling of iterative self-training depend on the quality of the initial warmup model.
- The definition of "easy" and "hard" relies on AoPS difficulty ratings, which are not available in all domains.
- Behavior at larger model scales (>7B) remains unexplored.

## Related Work & Insights

- **vs. Standard GRPO (full supervision)**: EasyRL surpasses full-data supervised GRPO using only 10% easy labeled data via self-evolution, demonstrating that data quality and learning strategy matter more than data quantity.
- **vs. Unsupervised EMPO**: EMPO requires no labeled data at all, but yields limited and unstable gains. EasyRL uses a small amount of easy labeled data as an anchor, enabling more stable and effective self-evolution.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The integration of cognitive learning theory is inspiring; the combination of divide-and-conquer pseudo-labeling and progressive self-training is elegantly designed.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Three model backbones, math and science benchmarks, multi-round iteration ablations, and pseudo-label quality analysis; very comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ — Motivation is clearly articulated; the correspondence between theoretical motivation and method design is well presented.
**Code**: To be confirmed
**Area**: reinforcement_learning
**Keywords**: To be supplemented

## TL;DR
To be supplemented after thorough reading.

## Background & Motivation
To be supplemented after thorough reading.

## Method
To be supplemented after thorough reading.

## Key Experimental Results
To be supplemented after thorough reading.

## Highlights & Insights
To be supplemented after thorough reading.

## Limitations & Future Work
To be supplemented after thorough reading.

## Related Work & Insights
To be supplemented after thorough reading.

## Rating
- Novelty: Pending
- Experimental Thoroughness: Pending
- Writing Quality: Pending
- Value: Pending

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] Reverse Constitutional AI: A Framework for Controllable Toxic Data Generation via Probability-Clamped RLAIF](reverse_constitutional_ai_a_framework_for_controllable_toxic_data_generation_via.md)
- [\[ACL 2026\] The Stackelberg Speaker: Optimizing Persuasive Communication in Social Deduction Games](the_stackelberg_speaker_optimizing_persuasive_communication_in_social_deduction_.md)
- [\[ACL 2026\] HEALing Entropy Collapse: Enhancing Exploration in Few-Shot RLVR via Hybrid-Domain Entropy Dynamics Alignment](healing_entropy_collapse_enhancing_exploration_in_few-shot_rlvr_via_hybrid-domai.md)
- [\[ICLR 2026\] SPELL: Self-Play Reinforcement Learning for Evolving Long-Context Language Models](../../ICLR2026/reinforcement_learning/spell_self-play_reinforcement_learning_for_evolving_long-context_language_models.md)
- [\[ACL 2026\] A Survey of Reinforcement Learning for Large Language Models under Data Scarcity: Challenges and Solutions](a_survey_of_reinforcement_learning_for_large_language_models_under_data_scarcity.md)

<!-- RELATED:END -->
