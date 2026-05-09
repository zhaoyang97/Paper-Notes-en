---
title: >-
  [Paper Note] A Survey of Reinforcement Learning for Large Language Models under Data Scarcity: Challenges and Solutions
description: >-
  [ACL 2026][Reinforcement Learning] The first systematic survey of reinforcement learning for LLMs under data scarcity, proposing a three-level taxonomy organized around data-centric, training-centric, and framework-centric perspectives, covering data pruning/synthesis/compression, trajectory generation/reward engineering/policy optimization, and self-evolution/co-evolution/multi-agent evolution paradigms.
tags:
  - ACL 2026
  - Reinforcement Learning
  - data scarcity
  - LLM post-training
  - data efficiency
  - survey
date: 2026-05-08
content_hash: 99e8a32706f8aeea
---

# A Survey of Reinforcement Learning for Large Language Models under Data Scarcity: Challenges and Solutions

**Conference**: ACL 2026
**arXiv**: [2604.17312](https://arxiv.org/abs/2604.17312)
**Code**: [https://github.com/YuZhiyin/Data-Efficient-RL](https://github.com/YuZhiyin/Data-Efficient-RL)
**Area**: Reinforcement Learning / LLM Training
**Keywords**: reinforcement learning, data scarcity, LLM post-training, data efficiency, survey

## TL;DR

The first systematic survey of reinforcement learning for LLMs under data scarcity, proposing a three-level taxonomy organized around data-centric, training-centric, and framework-centric perspectives, covering data pruning/synthesis/compression, trajectory generation/reward engineering/policy optimization, and self-evolution/co-evolution/multi-agent evolution paradigms.

## Background & Motivation

**Background**: Reinforcement learning (RL) has become an important paradigm for LLM post-training. Models such as DeepSeek-R1 and OpenAI-o1 demonstrate that RL post-training can elicit emergent capabilities such as self-reflection and significantly improve performance on complex reasoning tasks.

**Limitations of Prior Work**: RL training faces severe data scarcity challenges along two dimensions: (1) *external data scarcity*—high-quality human feedback, preference annotations, and expert-level step-by-step reasoning data are costly to obtain; and (2) *internal data scarcity*—model-generated interaction data is constrained by the number of rollouts, trajectory length, and exploration budget. As Jones (2024) notes, "the AI revolution is running out of data."

**Key Challenge**: Simply scaling data volume or compute often yields diminishing returns. While prior work has explored individual directions, a systematic and unified framework for organizing these scattered efforts is lacking.

**Goal**: To provide the first systematic survey and construct a unified taxonomy that maps the research landscape of LLM RL under data scarcity.

**Key Insight**: A bottom-up hierarchical structure is adopted, decomposing solutions into three complementary perspectives: data, training, and framework.

**Core Idea**: A three-level taxonomy is proposed—the data-centric perspective optimizes the data itself, the training-centric perspective improves the RL process, and the framework-centric perspective constructs self-evolving systems.

## Method

### Overall Architecture

This survey constructs a three-level hierarchical taxonomy: Level 1 (Data-Centric) → Level 2 (Training-Centric) → Level 3 (Framework-Centric), progressing from optimizing available data to improving training efficiency, and ultimately to building frameworks capable of self-evolution that reduce dependence on external data.

### Key Designs

1. **Data-Centric Perspective**:

    - **Function**: Optimizes data before, during, or after RL training to maximize the information extracted from available samples.
    - **Mechanism**: Three sub-directions are identified— (a) *Data Pruning*: retaining high-information-density samples via offline, online, or fine-grained filtering; e.g., LIMR selects samples by alignment between reward trajectories and the mean learning curve, while RORL selects moderately difficult samples based on online pass-rate estimates; (b) *Data Synthesis*: including static synthesis (e.g., Constitutional AI for preference data), dynamic synthesis (continuous augmentation within the training loop), and hard-data synthesis (generating new problems targeting model weaknesses); (c) *Data Compression*: operating at the token level (updating only high-entropy tokens), step level (pruning redundant reasoning steps), trajectory level (filtering zero-gradient trajectories), and dataset level (in extreme cases, a single sample may suffice).
    - **Design Motivation**: Addresses scarcity directly at the data level, maximizing the value of each sample when data is limited.

2. **Training-Centric Perspective**:

    - **Function**: Improves how RL generates trajectories, evaluates rewards, and updates policies.
    - **Mechanism**: Three sub-directions are identified— (a) *Trajectory Generation*: guided exploration (e.g., integrating MCTS into LLM decoding) and selective rollout (pre-filtering low-information prompts and computing gradients only for high-entropy tokens); (b) *Reward Engineering*: process rewards (self-rewarding using consistency and volatility patterns), intrinsic motivation (debate between entropy minimization and entropy maximization), and consensus mechanisms (self-consistency, majority voting); (c) *Policy Optimization*: experience replay and sample-efficient objective functions.
    - **Design Motivation**: Improves the utilization efficiency of each trajectory at the algorithmic level when data volume is limited.

3. **Framework-Centric Perspective**:

    - **Function**: Constructs RL frameworks capable of self-evolution, reducing dependence on external annotated data.
    - **Mechanism**: Three paradigms are identified— (a) *Self-evolution*: a single model simultaneously acts as generator and evaluator, closing the learning loop through self-training and adaptive learning; (b) *Asymmetric Co-evolution*: dual-agent collaboration (proposer–solver) or adversarial interaction (generator–discriminator); (c) *Multi-agent Evolution*: competitive self-play (e.g., poker-style games) and multi-role cooperation (a Proposer–Solver–Verifier triadic structure).
    - **Design Motivation**: Fundamentally reduces dependence on external annotated data, enabling continuous self-improvement.

### Loss & Training

As a survey paper, no novel loss functions are proposed. Instead, the paper systematically reviews various training strategies: curriculum learning (easy-to-hard) from the data side, entropy regularization strategies (the minimization vs. maximization debate) from the training side, and self-play and multi-agent interaction from the framework side.

## Key Experimental Results

### Main Results

This work is a survey and contains no original experiments. However, key findings from representative methods across each direction are systematically documented:

| Direction | Representative Method | Key Finding |
|---|---|---|
| Dataset Compression | One-shot RLVR | A single sample suffices to match the performance of RLVR trained on 7.5K samples |
| Token Compression | High-Entropy Minority Tokens | Updating only high-entropy tokens maintains or even improves performance |
| Intrinsic Reward | Intuitor | Unsupervised learning achieved using only the model's own confidence as the reward signal |
| Self-play | SPIRAL | Systematic reasoning capabilities emerge through multi-round game-playing |

### Ablation Study

| Taxonomy Dimension | Number of Methods | Core Trend |
|---|---|---|
| Data Pruning | 15+ | Moderately difficult samples are most valuable |
| Data Synthesis | 10+ | Dynamic synthesis outperforms static synthesis |
| Reward Engineering | 15+ | Intrinsic motivation can substitute for external rewards |
| Framework Evolution | 10+ | Multi-agent frameworks outperform single-agent ones |

### Key Findings

- The limits of data efficiency can be remarkably extreme: One-shot RLVR demonstrates that a single sample is sufficient to activate effective RL training.
- There is a fundamental divergence regarding the role of entropy in RL: one camp advocates minimization (reducing uncertainty) while another advocates maximization (encouraging exploration), both with valid justifications.
- Self-play and multi-agent frameworks demonstrate strong potential for continuous improvement under zero external data conditions.

## Highlights & Insights

- The three-level taxonomy (data → training → framework) provides a clear research roadmap, helping researchers situate their own contributions. The framework forms a natural progression from "optimizing existing data" to "improving data utilization" to "reducing data dependence."
- The synthesis of the entropy minimization vs. maximization debate is particularly illuminating: these seemingly contradictory strategies each have appropriate use cases, and adaptive switching between them may be a productive direction for future work.
- The timing of this survey is apt, arriving precisely as RL post-training has become a mainstream paradigm (following DeepSeek-R1 and o1), making this systematic synthesis highly valuable to the field.

## Limitations & Future Work

- The cutoff date of the survey is a limitation, as the field evolves rapidly (with new methods appearing weekly) and the taxonomy requires continuous updating.
- The combinatorial effects between methods are not thoroughly discussed—can data-centric and training-centric approaches be used synergistically?
- Discussion of safety risks (e.g., bias amplification in self-play, reward hacking) is relatively brief.
- Three future directions proposed by the authors merit attention: reliability of intrinsic rewards, generalization to non-verifiable tasks, and safety risks in self-play.

## Related Work & Insights

- **vs. LLM + Agentic RL Survey (Zhang et al., 2025)**: That survey focuses on LLMs as RL agents, whereas this work focuses on the specific challenge of data scarcity.
- **vs. Self-evolving Agents Survey (Tao et al., 2024)**: That survey focuses on self-evolution capabilities, whereas this work treats self-evolution as one of three perspectives within a more systematic framework.
- **vs. Data-efficient Post-training Survey (Luo et al., 2025)**: That survey covers post-training more broadly, whereas this work focuses specifically on data efficiency within the RL paradigm.

## Rating
- Novelty: ⭐⭐⭐⭐ — The first RL-for-LLM survey focused on the data scarcity perspective, with a novel taxonomy.
- Experimental Thoroughness: ⭐⭐⭐ — No original experiments, but comprehensive literature coverage.
- Writing Quality: ⭐⭐⭐⭐⭐ — Clearly organized; the three-level progressive structure is excellent.
- Value: ⭐⭐⭐⭐ — Provides a much-needed systematic synthesis for the rapidly evolving RL post-training field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Table Question Answering in the Era of Large Language Models: A Comprehensive Survey](table_question_answering_in_the_era_of_large_language_models_a_comprehensive_sur.md)
- [\[ACL 2026\] Feedback-Driven Tool-Use Improvements in Large Language Models via Automated Build Environments](feedback-driven_tool-use_improvements_in_large_language_models_via_automated_bui.md)
- [\[ACL 2026\] From Passive Metric to Active Signal: The Evolving Role of Uncertainty Quantification in Large Language Models](from_passive_metric_to_active_signal_the_evolving_role_of_uncertainty_quantifica.md)
- [\[ACL 2026\] Easy Samples Are All You Need: Self-Evolving LLMs via Data-Efficient Reinforcement Learning](easy_samples_are_all_you_need_self-evolving_llms_via_data-efficient_reinforcemen.md)
- [\[ICLR 2026\] TROLL: Trust Regions improve Reinforcement Learning for Large Language Models](../../ICLR2026/reinforcement_learning/troll_trust_regions_improve_reinforcement_learning_for_large_language_models.md)

</div>

<!-- RELATED:END -->
