---
title: >-
  [Paper Note] A Survey of Reinforcement Learning for Large Language Models under Data Scarcity: Challenges and Solutions
description: >-
  [ACL 2026][Reinforcement Learning][Data Scarcity] The first systematic survey of Reinforcement Learning (RL) for LLMs under data scarcity, proposing a three-layer taxonomy: data-centric, training-centric, and framework-centric. It covers directions such as data pruning/synthesis/compression, trajectory generation/reward engineering/policy optimization, and self-evolution/co-evolution/multi-agent evolution.
tags:
  - "ACL 2026"
  - "Reinforcement Learning"
  - "Data Scarcity"
  - "LLM Post-training"
  - "Data Efficiency"
  - "Survey"
date: 2026-05-08
content_hash: f21b66dc58fc214c
---

# A Survey of Reinforcement Learning for Large Language Models under Data Scarcity: Challenges and Solutions

**Conference**: ACL 2026  
**arXiv**: [2604.17312](https://arxiv.org/abs/2604.17312)  
**Code**: [https://github.com/YuZhiyin/Data-Efficient-RL](https://github.com/YuZhiyin/Data-Efficient-RL)  
**Area**: Reinforcement Learning / LLM Training  
**Keywords**: Reinforcement Learning, Data Scarcity, LLM Post-training, Data Efficiency, Survey

## TL;DR

The first systematic survey of Reinforcement Learning (RL) for LLMs under data scarcity, proposing a three-layer taxonomy: data-centric, training-centric, and framework-centric. It covers directions such as data pruning/synthesis/compression, trajectory generation/reward engineering/policy optimization, and self-evolution/co-evolution/multi-agent evolution.

## Background & Motivation

**Background**: Reinforcement Learning (RL) has become a crucial paradigm for LLM post-training. Models such as DeepSeek-R1 and OpenAI-o1 demonstrate that RL post-training can stimulate emergent capabilities like self-reflection, significantly enhancing performance in complex reasoning.

**Limitations of Prior Work**: RL training faces severe data scarcity challenges in two dimensions: (1) External data scarcity—high-quality human feedback, preference annotations, and expert step-level reasoning data are expensive to acquire; (2) Internal data scarcity—model-generated interaction data is limited by rollout quantities, trajectory lengths, and exploration budgets. Jones (2024) noted that "the AI revolution is running out of data."

**Key Challenge**: Simply increasing data scale or compute often yields diminishing returns. Although existing studies explore various directions, there is a lack of a systematic and unified framework to organize these fragmented efforts.

**Goal**: To provide the first systematic survey and construct a unified taxonomy to review the landscape of RL for LLMs under data scarcity.

**Key Insight**: Starting from a bottom-up hierarchical structure, solutions are categorized into three complementary perspectives: data, training, and framework.

**Core Idea**: Propose a three-level taxonomy—data-centric for optimizing the data itself, training-centric for improving the RL process, and framework-centric for building self-evolving systems.

## Method

### Overall Architecture

This survey constructs a three-level hierarchical taxonomy: Level 1 (Data-Centric) → Level 2 (Training-Centric) → Level 3 (Framework-Centric). It ranges from optimizing available data to improving training efficiency, and finally to building frameworks capable of self-evolution to reduce reliance on external data.

### Key Designs

**1. Data-Centric Perspective: Directly optimizing the data itself before/during/after training to maximize the value of each sample.**

When external data is inherently scarce, the most direct response is to start from the data end. The survey divides this perspective into three sub-paths. First is **data pruning**, which retains high-information-density samples via offline/online/fine-grained filtering; for example, LIMR uses the alignment between reward trajectories and average learning curves to select samples, while RORL selects medium-difficulty problems based on online pass rate estimates. Second is **data synthesis**, including static synthesis (e.g., Constitutional AI synthesizing preference data), dynamic synthesis with continuous enhancement within the training loop, and hard data synthesis targeting model weaknesses. Third is **data compression**, spanning token-level (updating only high-entropy tokens), step-level (pruning redundant reasoning steps), trajectory-level (filtering zero-gradient trajectories), to dataset-level (requiring only a single sample in extreme cases). The focus of this perspective is maximizing the information carried by each sample when data is limited.

**2. Training-Centric Perspective: Improving the utilization rate of each trajectory at the algorithmic level when the data volume is fixed.**

If data cannot be increased, the focus shifts to improving the RL pipeline: "trajectory generation—reward evaluation—policy update." Regarding **trajectory generation**, methods include guided exploration (e.g., integrating MCTS into LLM decoding) and selective rollouts (pre-filtering low-information prompts, calculating gradients only for high-entropy tokens). **Reward engineering** covers process rewards (utilizing consistency and volatility patterns for self-rewarding), intrinsic motivation (debates around entropy minimization vs. maximization), and consensus mechanisms like self-consistency/majority voting. **Policy optimization** focuses on experience replay and sample-efficient objective functions. The goal is to ensure every hard-won trajectory is more fully utilized.

**3. Framework-Centric Perspective: Fundamentally reducing dependence on external annotations by building self-evolving systems.**

The highest level no longer focuses on optimizing existing data but rather on allowing the system to generate its own data and close the learning loop. The survey summarizes three paradigms: **Self-evolution frameworks** allow a single model to act as both generator and evaluator, achieving a closed loop through self-training and adaptive learning; **Asymmetric co-evolution** uses dual-agent collaboration (proposer-solver) or competition (generator-discriminator); **Multi-agent evolution** follows competitive self-play (e.g., poker games) and multi-role cooperation (Proposer-Solver-Verifier triad). The value of this layer lies in pursuing the ultimate goal of "continuous self-improvement without external data."

### Loss & Training

As a survey paper, this work does not propose new loss functions but systematically organizes various training strategies: from data-side curriculum learning (easy-to-hard), training-side entropy regularization strategies (the debate between minimization vs. maximization), to framework-side self-play and multi-agent interaction.

## Key Experimental Results

### Main Results

This is a survey paper and does not contain original experiments. However, it systematically compiles key findings from representative methods in each direction:

| Direction | Representative Method | Key Finding |
|------|---------|---------|
| Dataset Compression | One-shot RLVR | Performance equal to 7.5K samples achieved with only 1 sample |
| Token Compression | High-Entropy Minority Tokens | Performance maintained or improved by updating only high-entropy tokens |
| Intrinsic Reward | Intuitor | Unsupervised learning achieved using only model confidence as reward signals |
| Self-play | SPIRAL | Systematic reasoning capabilities triggered through multi-round games |

### Ablation Study

| Category Dimension | Number of Methods | Core Trend |
|---------|---------|---------|
| Data Pruning | 15+ | Medium-difficulty samples are the most valuable |
| Data Synthesis | 10+ | Dynamic synthesis outperforms static synthesis |
| Reward Engineering | 15+ | Intrinsic motivation can replace external rewards |
| Framework Evolution | 10+ | Multi-agent outperforms single-agent |

### Key Findings

- Data efficiency limits can be extreme: One-shot RLVR proves that a single sample is sufficient to activate RL training effects.
- Fundamental divergence exists regarding the role of entropy in RL: one school advocates minimization (reducing uncertainty), while the other advocates maximization (encouraging exploration); both have merits.
- Self-play and multi-agent frameworks show strong potential for continuous improvement with zero external data.

## Highlights & Insights

- The three-level taxonomy (Data → Training → Framework) provides a clear research roadmap, helping researchers position their work. The framework progresses from "optimizing existing data" to "improving data utilization" and finally to "reducing data dependency."
- The summary of the entropy minimization vs. maximization debate is highly insightful: two seemingly contradictory strategies apply to different scenarios and may require adaptive switching in the future.
- The timing of the survey is appropriate, appearing as RL post-training becomes a mainstream paradigm (post-DeepSeek-R1, o1); this systematic organization offers significant reference value.

## Limitations & Future Work

- Restricted by the submission deadline, the field is evolving rapidly (new methods weekly); the taxonomy requires continuous updates.
- Lack of in-depth discussion on the combinatorial effects between methods—can data-centric and training-centric methods be used synergistically?
- Discussion on safety risks (e.g., bias amplification in self-play, reward hacking) is relatively brief.
- Three future directions proposed by the authors are noteworthy: reliability of internal rewards, generalization to non-verifiable tasks, and safety risks in self-play.

## Related Work & Insights

- **vs. LLM+Agentic RL Survey (Zhang et al., 2025)**: Focuses on LLMs acting as RL agents, whereas this paper focuses on the specific challenge of data scarcity.
- **vs. Self-evolving Agents Survey (Tao et al., 2024)**: Focuses on self-evolution capabilities, while this paper treats self-evolution as one of three perspectives for more systematic induction.
- **vs. Data-efficient Post-training Survey (Luo et al., 2025)**: Covers broader post-training topics, whereas this paper focuses on data efficiency specifically within the RL paradigm.

## Rating
- Novelty: ⭐⭐⭐⭐ First survey focusing on RL for LLMs from a data scarcity perspective with a novel taxonomy.
- Experimental Thoroughness: ⭐⭐⭐ No original experiments as a survey, but comprehensive literature coverage.
- Writing Quality: ⭐⭐⭐⭐⭐ Well-organized with an excellent three-level progressive structure.
- Value: ⭐⭐⭐⭐ Provides a much-needed systematic overview of the fast-evolving RL post-training field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Why Does Reinforcement Learning Generalize? A Feature-Level Mechanistic Study of Post-Training in Large Language Models](why_does_reinforcement_learning_generalize_a_feature-level_mechanistic_study_of_.md)
- [\[ICLR 2026\] On Predictability of Reinforcement Learning Dynamics for Large Language Models](../../ICLR2026/reinforcement_learning/on_predictability_of_reinforcement_learning_dynamics_for_large_language_models.md)
- [\[ICLR 2026\] Revolutionizing Reinforcement Learning Framework for Diffusion Large Language Models](../../ICLR2026/reinforcement_learning/revolutionizing_reinforcement_learning_framework_for_diffusion_large_language_mo.md)
- [\[ICLR 2026\] Using Reinforcement Learning to Train Large Language Models to Explain Human Decisions](../../ICLR2026/reinforcement_learning/using_reinforcement_learning_to_train_large_language_models_to_explain_human_dec.md)
- [\[ICLR 2026\] TROLL: Trust Regions improve Reinforcement Learning for Large Language Models](../../ICLR2026/reinforcement_learning/troll_trust_regions_improve_reinforcement_learning_for_large_language_models.md)

</div>

<!-- RELATED:END -->
