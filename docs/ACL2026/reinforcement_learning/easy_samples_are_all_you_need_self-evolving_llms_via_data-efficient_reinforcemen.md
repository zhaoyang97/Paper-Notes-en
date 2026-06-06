---
title: >-
  [Paper Note] Easy Samples Are All You Need: Self-Evolving LLMs via Data-Efficient Reinforcement Learning
description: >-
  [ACL 2026][Reinforcement Learning][Data-Efficient RL] The EasyRL framework is proposed, inspired by cognitive development theory. It initializes the model using only 10% simple labeled data via knowledge transfer…
tags:
  - "ACL 2026"
  - "Reinforcement Learning"
  - "Data-Efficient RL"
  - "Self-Evolving LLM"
  - "Pseudo-labeling"
  - "Simple-to-Hard"
  - "Cognitive Learning Theory"
date: 2026-05-08
content_hash: a835ec7da8afc76e
---

# Easy Samples Are All You Need: Self-Evolving LLMs via Data-Efficient Reinforcement Learning

**Conference**: ACL 2026  
**arXiv**: [2604.18639](https://arxiv.org/abs/2604.18639)  
**Code**: [https://github.com/YuZhiyin/EasyRL](https://github.com/YuZhiyin/EasyRL)  
**Area**: Reinforcement Learning / Data-Efficient Training  
**Keywords**: Data-Efficient RL, Self-Evolving LLM, Pseudo-labeling, Simple-to-Hard, Cognitive Learning Theory

## TL;DR

The EasyRL framework is proposed, inspired by cognitive development theory. It initializes the model using only 10% simple labeled data via knowledge transfer, then progressively masters difficult unlabeled data through divide-and-conquer pseudo-labeling and difficulty-based self-training, consistently outperforming GRPO supervised with full data.

## Background & Motivation

**Background**: RLVR has become a key post-training paradigm for enhancing the reasoning capabilities of LLMs. Existing methods are categorized into supervised (relying on ground-truth answers or reward models, with high labeling costs) and unsupervised (constructing rewards via voting or entropy estimation, prone to collapse or reward hacking).

**Limitations of Prior Work**: (1) Supervised methods require large amounts of high-quality labeled data, with extremely high costs for labeling difficult problems; (2) Unsupervised methods offer limited and unstable performance gains, often leading to model collapse or reward hacking; (3) Neither considers the distribution of data difficulty—practically, labeling costs for simple problems are much lower than for hard problems.

**Key Challenge**: Labeling costs for hard problems are high but their value is significant, while simple problems are cheap to label but insufficient for training alone. How can a model gradually master a large amount of hard unlabeled data using only a small amount of simple labeled data?

**Goal**: Design a cognitive-inspired RL framework to self-evolve and learn increasingly difficult reasoning tasks starting from limited simple labeled data.

**Key Insight**: Vygotsky's "Zone of Proximal Development" (ZPD) theory—learners first internalize knowledge from simple, reachable cases and then gradually expand to more difficult challenges with minimal external guidance.

**Core Idea**: (1) Use a small amount of simple labeled data to train a warmup model via GRPO; (2) Apply a divide-and-conquer strategy for pseudo-labeling unlabeled data—consistency selection for low-uncertainty samples and reflection for medium-uncertainty samples; (3) Use difficulty-based self-training iterations to expand the model's capability boundaries.

## Method

### Overall Architecture

EasyRL consists of three stages: (1) Knowledge Transfer—training a warmup model using GRPO on simple labeled data; (2) Divide-and-Conquer Pseudo-labeling—the warmup model performs multiple inferences on unlabeled data, categorizing them into low/medium/high uncertainty groups to construct high-quality pseudo-labels via consistency selection and reflection; (3) Progressive Difficulty Self-Training—iterative RL training using a mix of labeled and pseudo-labeled data, with high-uncertainty samples from the previous round being re-labeled in each new iteration.

### Key Designs

1.  **Divide-and-Conquer Pseudo-labeling Strategy**:
    - **Function**: Constructs high-quality pseudo-labels from unlabeled data.
    - **Mechanism**: Generates $N$ independent reasoning paths for each unlabeled sample. Consistency Selection—if all $N$ outputs yield the same answer, it is directly used as a pseudo-label (low uncertainty). Reflection-based Solution—if answers are inconsistent, the predictive entropy $H(x)$ is calculated; if $H(x) \leq \tau_t$, the candidate answers are re-evaluated through a reflection mechanism to obtain a pseudo-label (medium uncertainty). High-uncertainty samples ($H(x) > \tau_t$) are deferred to the next iteration.
    - **Design Motivation**: Simple majority voting is insufficient for high-quality pseudo-labels. A three-tier strategy is used: high-confidence adoption for full consistency, reflection-based correction for partial consistency, and deferred processing for high uncertainty to maximize pseudo-label quality.

2.  **Progressive Difficulty Self-Training**:
    - **Function**: Enables the model to gradually extend its capability boundaries to harder problems.
    - **Mechanism**: In each iteration, the current model $\pi_i$ re-labels and filters the high-uncertainty samples left over from the previous round. These are mixed with labeled data for RL training to obtain $\pi_{i+1}$. As model capability improves, difficult samples that could not be labeled confidently before are gradually incorporated. $\pi_1, \pi_2, ..., \pi_n$ form a sequence of models with increasing capabilities.
    - **Design Motivation**: Corresponds to Vygotsky’s ZPD theory—the pseudo-labeled dataset in each round specifically targets the model's current "Zone of Proximal Development," i.e., tasks slightly beyond current capabilities but learnable.

3.  **Cognitive Learning Curve Simulation**:
    - **Function**: Provides a unified theoretical foundation for the framework.
    - **Mechanism**: EasyRL simulates the human cognitive acquisition curve: learning basic rules from simple cases (Knowledge Transfer), then transferring knowledge to new, harder problems through analogy and self-reflection (Divide-and-Conquer + Self-Training). The consistency rate (ConsRate) of pseudo-labels increases across iterations, confirming progressive growth in capability.
    - **Design Motivation**: Introduces learning theories from cognitive science into RL framework design to provide methodological guidance.

### Loss & Training

Standard GRPO objective function is used. Correctness reward $r=1$ (match), $r=-0.5$ (format error), $r=0$ (otherwise). Evaluated on Qwen2.5-Math-1.5B/7B and Llama-3.2-3B. Simple labeled data accounts for 10% of the total (selected as a simple subset according to AoPS difficulty levels).

## Key Experimental Results

### Main Results

| Model / Method | Math Avg | Science Avg | Labeled Data |
|----------|---------|---------|----------|
| Qwen2.5-Math-1.5B Base | 32.6 | 1.5 | 0 |
| w/ Supervised GRPO (10%) | 35.7 | 7.9 | 10% |
| w/ Unsupervised EMPO | 38.5 | 15.6 | 0 |
| w/ EasyRL Iter3 | **40.3** | **19.4** | 10% |
| Qwen2.5-Math-7B Base | 38.5 | 24.1 | 0 |
| w/ Supervised GRPO (10%) | 43.3 | 27.4 | 10% |
| w/ EasyRL Iter3 | **50.6** | **30.6** | 10% |

### Ablation Study

| Configuration | Gain | Description |
|------|------|------|
| Knowledge Transfer only (warmup) | Baseline | Simple data only |
| + Divide-and-Conquer Iter1 | Improvement | Includes easy unlabeled data |
| + Iter2 | Further Improvement | Includes medium difficult data |
| + Iter3 | Optimal | Includes more difficult data |

### Key Findings

- EasyRL using only 10% simple labeled data outperforms Supervised GRPO using 100% of the data.
- Iterative self-training leads to continuous improvement: Iter1 → Iter2 → Iter3 shows steady growth on both math and science benchmarks.
- The consistency rate of pseudo-labels increases with iterations, confirming the self-evolution of model capability.
- EasyRL is also effective on out-of-distribution tasks (scientific reasoning), indicating the acquisition of general reasoning abilities.
- The reflection mechanism significantly helps improve pseudo-label quality for medium-uncertainty samples.

## Highlights & Insights

- **The finding that "easy samples are enough" has practical value**: Labeling hard problems is expensive; EasyRL proves that labeling only simple problems allows for covering hard problems through self-evolution. This is highly meaningful for practical scenarios with limited labeling resources.
- **The divide-and-conquer strategy for uncertainty-graded data is natural**: A three-tier strategy (full consistency → reflection correction → deferred processing) applies the optimal handling method to each level.
- **The introduction of cognitive science provides methodological guidance**: ZPD theory is not a simple analogy but guides specific design choices (simple-to-hard, progressive expansion).

## Limitations & Future Work

- Multiple inferences for pseudo-label evaluation involve certain computational costs.
- The convergence speed and final upper bound of iterative self-training depend on the quality of the initial warmup model.
- The definition of "simple" and "hard" depends on AoPS difficulty labels, which may not be available in all domains.
- Behavior on larger models (>7B) has not been explored.

## Related Work & Insights

- **vs Standard GRPO (Full Supervision)**: EasyRL outperforms full-supervision GRPO using only 10% simple data via self-evolution, proving that data quality and learning strategy are more important than data quantity.
- **vs Unsupervised EMPO**: EMPO uses no labels, but its performance gains are limited and unstable. EasyRL uses a small amount of simple labels as anchors to achieve more stable self-evolution.

## Rating

- Novelty: ⭐⭐⭐⭐ The introduction of cognitive learning theory is inspiring; the combination of divide-and-conquer pseudo-labeling and progressive self-training is cleverly designed.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Very thorough, covering 3 models plus math and science benchmarks, multi-round iteration ablations, and pseudo-label quality analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation; the correspondence between theoretical motivation and method design is well-explained.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] D$^2$Evo: Dual Difficulty-Aware Self-Evolution for Data-Efficient Reinforcement Learning](../../ICML2026/reinforcement_learning/d2evo_dual_difficulty-aware_self-evolution_for_data-efficient_reinforcement_lear.md)
- [\[ICML 2026\] Metis: Learning to Jailbreak LLMs via Self-Evolving Metacognitive Policy Optimization](../../ICML2026/reinforcement_learning/metis_learning_to_jailbreak_llms_via_self-evolving_metacognitive_policy_optimiza.md)
- [\[ICLR 2026\] SPELL: Self-Play Reinforcement Learning for Evolving Long-Context Language Models](../../ICLR2026/reinforcement_learning/spell_self-play_reinforcement_learning_for_evolving_long-context_language_models.md)
- [\[ACL 2026\] EvoCoT: Overcoming the Exploration Bottleneck in Reinforcement Learning for LLMs](evocot_overcoming_the_exploration_bottleneck_in_reinforcement_learning.md)
- [\[ACL 2026\] ARGUS: Policy-Adaptive Ad Governance via Evolving Reinforcement with Adversarial Umpiring](argus_policy-adaptive_ad_governance_via_evolving_reinforcement_with_adversarial_.md)

</div>

<!-- RELATED:END -->
