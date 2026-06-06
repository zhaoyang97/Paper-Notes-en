---
title: >-
  [Paper Note] A Survey of Reinforcement Learning for Large Language Models under Data Scarcity: Challenges and Solutions
description: >-
  [ACL 2026][Reinforcement Learning][Data Scarcity] The first systematic survey of Reinforcement Learning for LLMs under data scarcity. It proposes a three-tier taxonomy: data-centric, training-centric…
tags:
  - "ACL 2026"
  - "Reinforcement Learning"
  - "Data Scarcity"
  - "LLM Post-training"
  - "Data Efficiency"
  - "Survey"
date: 2026-05-08
content_hash: df40fbbeb55317e0
---

# A Survey of Reinforcement Learning for Large Language Models under Data Scarcity: Challenges and Solutions

**Conference**: ACL 2026  
**arXiv**: [2604.17312](https://arxiv.org/abs/2604.17312)  
**Code**: [https://github.com/YuZhiyin/Data-Efficient-RL](https://github.com/YuZhiyin/Data-Efficient-RL)  
**Area**: Reinforcement Learning / LLM Training  
**Keywords**: Reinforcement Learning, Data Scarcity, LLM Post-training, Data Efficiency, Survey

## TL;DR

The first systematic survey of Reinforcement Learning for LLMs under data scarcity. It proposes a three-tier taxonomy: data-centric, training-centric, and framework-centric, covering data pruning/synthesis/compression, trajectory generation/reward engineering/policy optimization, and self-evolution/co-evolution/multi-agent evolution.

## Background & Motivation

**Background**: Reinforcement Learning (RL) has become a critical paradigm for LLM post-training. Models such as DeepSeek-R1 and OpenAI-o1 demonstrate that RL post-training can trigger emergent capabilities like self-reflection, significantly enhancing performance in complex reasoning.

**Limitations of Prior Work**: RL training faces severe data scarcity challenges manifesting in two aspects: (1) External data scarcity—high costs associated with obtaining high-quality human feedback, preference annotations, and expert step-level reasoning data; (2) Internal data scarcity—model-generated interaction data is limited by the number of rollouts, trajectory lengths, and exploration budgets. Jones (2024) noted that "the AI revolution is running out of data."

**Key Challenge**: Simply increasing data scale or computational resources often yields diminishing returns. While existing studies have explored various directions, there is a lack of a systematic and unified framework to organize these fragmented works.

**Goal**: To provide the first systematic survey and construct a unified taxonomy to overview the research landscape of LLM RL under data scarcity.

**Key Insight**: Starting from a bottom-up hierarchical structure, the solutions are categorized into three complementary perspectives: data, training, and framework.

**Core Idea**: Propose a three-level taxonomy—a data-centric perspective to optimize the data itself, a training-centric perspective to improve the RL process, and a framework-centric perspective to build self-evolving systems.

## Method

### Overall Architecture

This survey constructs a three-level hierarchical taxonomy: Level 1 (Data-Centric) → Level 2 (Training-Centric) → Level 3 (Framework-Centric), moving from optimizing available data to improving training efficiency, and finally to constructing frameworks that can evolve autonomously to reduce reliance on external data.

### Key Designs

1.  **Data-Centric Perspective**:

    - **Function**: Optimizes the data itself before, during, or after RL training to maximize available information.
    - **Mechanism**: Divided into three sub-directions—(a) Data pruning: Retaining high-information density samples through offline/online/fine-grained screening (e.g., LIMR filters based on the alignment between reward trajectories and average learning curves; RORL selects medium-difficulty samples based on online pass rate estimation); (b) Data synthesis: Static synthesis (e.g., Constitutional AI for preference data), dynamic synthesis (continuous enhancement within the training loop), and hard data synthesis (generating new problems for weak areas); (c) Data compression: Ranging from token-level (updating only high-entropy tokens), step-level (pruning redundant reasoning steps), trajectory-level (filtering zero-gradient trajectories) to dataset-level (requiring only one sample in extreme cases).
    - **Design Motivation**: Directly address the scarcity problem from the data side to maximize the value of each sample when data is limited.

2.  **Training-Centric Perspective**:

    - **Function**: Improves how RL generates trajectories, evaluates rewards, and updates policies.
    - **Mechanism**: Three sub-directions—(a) Trajectory generation: Guided exploration (e.g., integrating MCTS into LLM decoding) and selective rollouts (pre-filtering low-information prompts, calculating gradients only for high-entropy tokens); (b) Reward engineering: Process rewards (self-rewarding using consistency and volatility patterns), intrinsic motivation (the debate between entropy minimization vs. entropy maximization), and consensus mechanisms (self-consistency, majority voting); (c) Policy optimization: Experience replay and sample-efficient objective functions.
    - **Design Motivation**: Improve the utilization efficiency of each trajectory at the training algorithm level when data volume is limited.

3.  **Framework-Centric Perspective**:

    - **Function**: Constructs RL frameworks capable of self-evolution to reduce dependence on external data.
    - **Mechanism**: Three paradigms—(a) Self-evolution frameworks: A single model acts as both generator and evaluator, closing the learning loop through self-training and adaptive learning; (b) Asymmetric co-evolution: Dual-agent collaboration (proposer-solver) or competition (generator-discriminator); (c) Multi-agent evolution: Competitive self-play (e.g., poker games) and multi-role cooperation (Proposer-Solver-Verifier triad).
    - **Design Motivation**: Fundamentally reduce the reliance on external annotated data and achieve continuous self-improvement.

### Loss & Training

As a survey paper, this work does not propose a new loss function but systematically organizes various training strategies: from curriculum learning (easy-to-hard) at the data end, to entropy regularization strategies (the conflict between minimization and maximization) at the training end, to self-play and multi-agent interactions at the framework end.

## Key Experimental Results

### Main Results

As this is a survey, it contains no primary experiments. However, it systematically compiles key findings from representative methods in various directions:

| Direction | Representative Methods | Key Findings |
| :--- | :--- | :--- |
| Dataset Compression | One-shot RLVR | Performance comparable to 7.5K samples achieved with only 1 sample |
| Token Compression | High-Entropy Minority Tokens | Updating only high-entropy tokens maintains or even improves performance |
| Intrinsic Reward | Intuitor | Unsupervised learning achieved using only model confidence as a reward signal |
| Self-play | SPIRAL | Emergence of systematic reasoning capabilities through multi-round games |

### Ablation Study

| Category Dimension | Number of Methods | Core Trend |
| :--- | :--- | :--- |
| Data Pruning | 15+ | Medium-difficulty samples are most valuable |
| Data Synthesis | 10+ | Dynamic synthesis outperforms static synthesis |
| Reward Engineering | 15+ | Intrinsic motivation can substitute for external rewards |
| Framework Evolution | 10+ | Multi-agent systems outperform single-agent systems |

### Key Findings

- Limits of data efficiency can be extreme: One-shot RLVR proves that a single sample is sufficient to activate the effects of RL training.
- There is a fundamental divergence regarding the role of entropy in RL: one school advocates for minimization (reducing uncertainty), while another advocates for maximization (encouraging exploration), both with valid justifications.
- Self-play and multi-agent frameworks demonstrate significant potential for continuous improvement under zero-external-data conditions.

## Highlights & Insights

- The three-tier taxonomy (Data → Training → Framework) provides a clear research roadmap, helping researchers situate their work. The framework moves from "optimizing existing data" to "improving data utilization" and finally to "reducing data dependence," forming a progressive relationship.
- The organized debate between entropy minimization and maximization is highly insightful: the two seemingly contradictory strategies have different applicable scenarios and may require adaptive switching in the future.
- The timing of this survey is optimal, as RL post-training has become a mainstream paradigm (post-DeepSeek-R1 and o1), making this systematic organization highly valuable for the field.

## Limitations & Future Work

- Due to the cutoff time of the survey and the rapid development of the field (new methods appearing weekly), the taxonomy requires continuous updates.
- The study does not deeply discuss the combination effects between different methods—can data-centric and training-centric methods be used synergistically?
- Discussions on safety risks (e.g., bias amplification in self-play, reward hacking) are relatively brief.
- Three future directions proposed by the authors deserve attention: reliability of internal rewards, generalization to non-verifiable tasks, and safety risks in self-play.

## Related Work & Insights

- **vs. LLM+Agentic RL Survey (Zhang et al., 2025)**: While the former focuses on LLMs as RL agents, this paper focuses specifically on the challenge of data scarcity.
- **vs. Self-evolving Agents Survey (Tao et al., 2024)**: While focusing on self-evolution, this paper incorporates self-evolution as one of three systematic perspectives.
- **vs. Data-efficient Post-training Survey (Luo et al., 2025)**: Covering broader post-training topics, this paper focuses specifically on data efficiency within the RL paradigm.

## Rating
- Novelty: ⭐⭐⭐⭐ First survey focusing on the data scarcity perspective of RL for LLM, with a novel taxonomy.
- Experimental Thoroughness: ⭐⭐⭐ No primary experiments as it is a survey, but literature coverage is comprehensive.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear organization with an excellent three-tier progressive structure.
- Value: ⭐⭐⭐⭐ Provides a much-needed systematic organization for the rapidly evolving field of RL post-training.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Why Does Reinforcement Learning Generalize? A Feature-Level Mechanistic Study of Post-Training in Large Language Models](why_does_reinforcement_learning_generalize_a_feature-level_mechanistic_study_of_.md)
- [\[ICLR 2026\] TROLL: Trust Regions improve Reinforcement Learning for Large Language Models](../../ICLR2026/reinforcement_learning/troll_trust_regions_improve_reinforcement_learning_for_large_language_models.md)
- [\[ICLR 2026\] Robust Multi-Objective Controlled Decoding of Large Language Models](../../ICLR2026/reinforcement_learning/robust_multi-objective_controlled_decoding_of_large_language_models.md)
- [\[ICLR 2026\] VerifyBench: Benchmarking Reference-based Reward Systems for Large Language Models](../../ICLR2026/reinforcement_learning/verifybench_benchmarking_reference-based_reward_systems_for_large_language_model.md)
- [\[ICML 2026\] Can Large Language Models Generalize Procedures Across Representations?](../../ICML2026/reinforcement_learning/can_large_language_models_generalize_procedures_across_representations.md)

</div>

<!-- RELATED:END -->
