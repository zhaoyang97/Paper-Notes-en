---
title: >-
  [Paper Note] HEALing Entropy Collapse: Enhancing Exploration in Few-Shot RLVR via Hybrid-Domain Entropy Dynamics Alignment
description: >-
  [ACL 2026][Reinforcement Learning][To be supplemented] To be supplemented after thorough reading.
tags:
  - "ACL 2026"
  - "Reinforcement Learning"
  - "To be supplemented"
date: 2026-05-08
content_hash: 5fb6e99c934eeb48
---

# HEALing Entropy Collapse: Enhancing Exploration in Few-Shot RLVR via Hybrid-Domain Entropy Dynamics Alignment

**Conference**: ACL 2026
**arXiv**: [2604.17928](https://arxiv.org/abs/2604.17928)  
**Code**: [https://github.com/XMUDeepLIT/HEAL](https://github.com/XMUDeepLIT/HEAL)  
**Area**: Reinforcement Learning / LLM Reasoning
**Keywords**: RLVR, entropy collapse, few-shot reinforcement learning, cross-domain alignment, exploration diversity

## TL;DR

This work proposes the HEAL framework, which addresses severe entropy collapse in few-shot RLVR by mixing general-domain data with an Entropy Dynamics Alignment (EDA) reward mechanism. Using only 32 target-domain samples, HEAL matches or surpasses full-shot RLVR performance trained on 1K samples.

## Background & Motivation

**Background**: RLVR (Reinforcement Learning with Verifiable Rewards) has emerged as a key technique for training reasoning-oriented LLMs, using binary accuracy rewards to guide the learning of reasoning chains. However, existing research has primarily focused on data-abundant settings.

**Limitations of Prior Work**: In low-resource domains such as medical reasoning and specialized knowledge, RLVR training data is scarce. Few-shot RLVR tends to rapidly overfit to a small set of generated trajectories, leading to premature convergence of exploration and more severe entropy collapse. Existing entropy regularization methods do not account for the effect of data scale, and their direct application to low-resource settings yields suboptimal results.

**Key Challenge**: In few-shot RLVR, the policy entropy is significantly lower than in full-data training, resulting in severely insufficient exploration diversity. Naively increasing entropy (e.g., adding an entropy regularization term) lacks proper constraints and may limit policy exploitation or even destabilize training.

**Goal**: To design a framework specifically tailored for few-shot RLVR that mitigates entropy collapse and improves exploration diversity while maintaining training stability.

**Key Insight**: Drawing an analogy to human learning — when encountering a new domain, humans leverage general skills to compensate for insufficient domain-specific knowledge. Mixing in general-domain data provides basic reasoning patterns and prevents the policy from prematurely narrowing its search space.

**Core Idea**: (1) Selectively introduce high-value general-domain data to alleviate entropy collapse in the target domain; (2) Use EDA rewards to guide the policy in aligning the trajectory-level entropy dynamics (magnitude and fine-grained variation) of the target domain to the level observed in the general domain, achieving controlled entropy improvement.

## Method

### Overall Architecture

HEAL consists of two core components: (1) **Hybrid Training** — high-value samples are selected from the general domain based on two criteria, reasoning uncertainty and exploration diversity, and mixed with a small number of target-domain samples for training; (2) **EDA Reward** — an entropy dynamics alignment reward is added on top of the standard accuracy reward to guide the policy in learning the exploration behavior patterns of the general domain.

### Key Designs

1. **High-Value General-Domain Data Selection**:

    - *Function*: Maximize the effectiveness of hybrid training with minimal data volume.
    - *Mechanism*: For each general-domain sample, $N$ trajectories are generated and two metrics are computed: reasoning uncertainty $\text{Uncertainty}(x) = 1 - 2|\text{Acc}(x) - 0.5|$ (favoring hard samples with accuracy near 50%) and exploration diversity (the average entropy of the top-20% highest-entropy tokens across each trajectory). The product of the two yields a composite score $c(x)$, and the top-$K$ samples are selected. As few as 384 general-domain samples suffice to yield significant improvement.
    - *Design Motivation*: Indiscriminately mixing large volumes of general-domain data incurs excessive computational overhead, and low-quality samples may introduce noise.

2. **Entropy Dynamics Alignment (EDA) Reward**:

    - *Function*: Guide the exploration patterns of the target-domain policy to align with those of the general domain.
    - *Mechanism*: The token-level entropy sequence of each trajectory is defined as the "entropy dynamics" $\tau_y = (\mathcal{H}_1, \mathcal{H}_2, ..., \mathcal{H}_{|y|})$, capturing both the magnitude and fine-grained variation of entropy. For each target-domain trajectory, intra-domain similarity $\mathcal{S}_{\text{intra}}$ (maximum similarity to other target-domain trajectories) and cross-domain similarity $\mathcal{S}_{\text{inter}}$ (maximum similarity to general-domain trajectories) are computed. When cross-domain similarity exceeds intra-domain similarity, an additional reward $r_{\text{EDA}} = 1$ is granted; otherwise $r_{\text{EDA}} = 0$. The final reward is $r = r_{\text{Acc}} + r_{\text{EDA}}$.
    - *Design Motivation*: Even after hybrid training, target-domain entropy remains significantly lower than that of the general domain, necessitating explicit alignment. EDA does not naively increase entropy; instead, it uses the natural entropy patterns of the general domain as a reference to achieve controlled entropy improvement.

3. **Trajectory-Level Entropy Dynamics Representation**:

    - *Function*: Provide richer entropy features than scalar aggregation.
    - *Mechanism*: Rather than simply averaging entropy, the full token-level sequence is preserved. Trajectories of different lengths are aligned via interpolation before similarity computation. This captures not only the absolute magnitude of entropy but also its variation patterns (e.g., entropy peaks at intermediate reasoning steps).
    - *Design Motivation*: Direct aggregation (e.g., mean entropy) discards fine-grained information about the generation process; two trajectories with identical mean entropy may exhibit completely different exploration patterns.

### Loss & Training

Standard PPO/GRPO optimization from RLVR is employed, with the reward defined as the sum of the accuracy reward and the EDA reward. Experiments are conducted on the Qwen3 and LLaMA-3.2 model families. The target domain uses 32 samples, and 384 high-value samples are selected from the general domain.

## Key Experimental Results

### Main Results

| Model / Method | Target Samples | General Samples | Medicine Avg | Physics Avg | Code Avg |
|---|---|---|---|---|---|
| Qwen3-1.7B Few-shot | 32 | 0 | 41.70 | — | — |
| Qwen3-1.7B Full-shot | 1K | 0 | 44.73 | 36.24 | 39.18 |
| Qwen3-1.7B HEAL | 32 | 384 | **44.67** | **37.20** | **41.28** |
| Qwen3-4B Few-shot | 32 | 0 | 49.22 | 42.57 | 47.35 |
| Qwen3-4B Full-shot | 1K | 0 | 56.70 | 44.57 | 51.00 |
| Qwen3-4B HEAL | 32 | 384 | 50.82 | **46.99** | **53.42** |

### Ablation Study

| Configuration | Description | Performance |
|---|---|---|
| Few-shot (32 samples) | Target domain only | Baseline; severe entropy collapse |
| Only-General (10K) | General domain only | Does not provide target-domain knowledge |
| Hybrid (32+384) | Hybrid training without EDA | Alleviates entropy collapse but insufficiently |
| HEAL (32+384+EDA) | Full framework | Best; matches Full-shot |

### Key Findings

- With only 32 target-domain samples and 384 general-domain samples, HEAL matches or surpasses full-shot RLVR trained on 1K samples across multiple settings.
- Hybrid training alone significantly alleviates entropy collapse, but target-domain entropy remains below that of the general domain; EDA further closes this gap.
- The data selection strategy outperforms random sampling of general-domain data.
- Consistent improvements are observed across four domains: Medicine, Physics, Code, and Math.
- HEAL also outperforms existing entropy regularization methods (e.g., DAPO-style entropy rewards) in low-resource settings.

## Highlights & Insights

- **The entropy collapse problem in few-shot RLVR is systematically characterized for the first time**: Prior work reported training collapse but did not identify the root cause; this paper attributes it to entropy collapse and proposes a targeted solution.
- **Trajectory-level entropy dynamics is a richer representation than scalar entropy**: It preserves fine-grained exploration patterns throughout the generation process, and this representation is generalizable to other settings requiring analysis of generation behavior.
- **The idea of using the general domain as an "exploration teacher" is intuitive**: The general domain is not expected to supply domain-specific knowledge, but merely to serve as a behavioral reference for "how to explore."

## Limitations & Future Work

- General-domain data selection requires generating multiple trajectories in advance, incurring non-trivial upfront cost.
- The binary reward design of EDA may lack sufficient granularity; a continuous reward formulation could be more effective.
- Validation is limited to small-to-medium-scale models (1.7B–4B); the behavior of few-shot RLVR on larger models may differ.
- The choice of similarity function (e.g., DTW vs. cosine similarity) is not thoroughly explored.

## Related Work & Insights

- **vs. Standard Entropy Regularization Methods**: These methods naively increase entropy without constraints, potentially destabilizing training. HEAL uses the natural entropy patterns of the general domain as a reference to achieve controlled entropy improvement.
- **vs. Data Augmentation-Based Few-Shot RLVR**: Data augmentation relies on the model's internal knowledge and is thus bounded by its knowledge frontier. HEAL introduces external general-domain data as an exploration guide.

## Rating

- Novelty: ⭐⭐⭐⭐ The entropy collapse problem in few-shot RLVR is clearly defined, and the EDA reward design is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Four domains, two model families, and extensive ablations — very comprehensive.
- Writing Quality: ⭐⭐⭐⭐ The problem motivation is convincing, the method is clearly described, and figures aid understanding.

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

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Easy Samples Are All You Need: Self-Evolving LLMs via Data-Efficient Reinforcement Learning](easy_samples_are_all_you_need_self-evolving_llms_via_data-efficient_reinforcemen.md)
- [\[ACL 2026\] The Stackelberg Speaker: Optimizing Persuasive Communication in Social Deduction Games](the_stackelberg_speaker_optimizing_persuasive_communication_in_social_deduction_.md)
- [\[ACL 2026\] Reverse Constitutional AI: A Framework for Controllable Toxic Data Generation via Probability-Clamped RLAIF](reverse_constitutional_ai_a_framework_for_controllable_toxic_data_generation_via.md)
- [\[AAAI 2026\] Reasoning with Exploration: An Entropy Perspective](../../AAAI2026/reinforcement_learning/reasoning_with_exploration_an_entropy_perspective.md)
- [\[ICLR 2026\] Exploration vs Exploitation: Rethinking RLVR through Clipping, Entropy, and Spurious Reward](../../ICLR2026/reinforcement_learning/exploration_vs_exploitation_rethinking_rlvr_through_clipping_entropy_and_spuriou.md)

</div>

<!-- RELATED:END -->
