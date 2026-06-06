---
title: >-
  [Paper Note] HEALing Entropy Collapse: Enhancing Exploration in Few-Shot RLVR via Hybrid-Domain Entropy Dynamics Alignment
description: >-
  [ACL 2026][Reinforcement Learning][RLVR] The HEAL framework is proposed to address the severe entropy collapse in few-shot Reinforcement Learning from Verifiable Rewards (RLVR) by mixing general-domain data and employing…
tags:
  - "ACL 2026"
  - "Reinforcement Learning"
  - "RLVR"
  - "Entropy Collapse"
  - "Few-Shot RL"
  - "Cross-Domain Alignment"
  - "Exploration Diversity"
date: 2026-05-08
content_hash: aa41b35cd8abba5b
---

# HEALing Entropy Collapse: Enhancing Exploration in Few-Shot RLVR via Hybrid-Domain Entropy Dynamics Alignment

**Conference**: ACL 2026  
**arXiv**: [2604.17928](https://arxiv.org/abs/2604.17928)  
**Code**: [https://github.com/XMUDeepLIT/HEAL](https://github.com/XMUDeepLIT/HEAL)  
**Area**: Reinforcement Learning / LLM Reasoning  
**Keywords**: RLVR, Entropy Collapse, Few-Shot RL, Cross-Domain Alignment, Exploration Diversity

## TL;DR

The HEAL framework is proposed to address the severe entropy collapse in few-shot Reinforcement Learning from Verifiable Rewards (RLVR) by mixing general-domain data and employing an Entropy Dynamics Alignment (EDA) reward mechanism. It enables matching or exceeding the performance of full-shot RLVR (using 1K samples) with only 32 target-domain samples.

## Background & Motivation

**Background**: RLVR has become a critical technology for training reasoning-oriented LLMs, using binary accuracy rewards to guide the learning of reasoning chains. However, existing research focuses primarily on data-abundant scenarios.

**Limitations of Prior Work**: In low-resource domains such as medical reasoning or specialized expertise, RLVR training data is scarce. Few-shot RLVR tends to overfit rapidly to a small number of generated trajectories, leading to premature convergence of exploration and severe entropy collapse. Existing entropy regularization methods do not consider the impact of data scale and yield sub-optimal results in low-resource settings.

**Key Challenge**: The entropy of policies in few-shot RLVR is significantly lower than in full-shot training, resulting in a severe lack of exploration diversity. Naive methods of increasing entropy (e.g., adding entropy regularization terms) lack constraints and may limit policy exploitation or undermine training stability.

**Goal**: Design a framework specifically for few-shot RLVR to mitigate entropy collapse and enhance exploration diversity while maintaining training stability.

**Key Insight**: Drawing an analogy to human learning, when facing a new domain, humans leverage general skills to compensate for the lack of domain-specific knowledge. Mixing general-domain data provides basic reasoning patterns and prevents the policy from narrowing the search space prematurely.

**Core Idea**: (1) Selectively introduce high-value general-domain data to mitigate entropy collapse in the target domain; (2) Use EDA rewards to guide the policy to align target-domain trajectory-level entropy dynamics (including magnitude and fine-grained variations) with general-domain levels, achieving controllable entropy enhancement.

## Method

### Overall Architecture

HEAL consists of two core components: (1) Hybrid Training, which selects high-value samples from the general domain based on reasoning uncertainty and exploration diversity to mix with a few target-domain samples; (2) EDA Reward, which adds an entropy dynamics alignment reward to the standard accuracy reward to guide the policy in learning exploratory behavior patterns from the general domain.

### Key Designs

1.  **High-Value General-Domain Data Selection**:
    - **Function**: Maximize the effectiveness of hybrid training with a minimal amount of data.
    - **Mechanism**: For each general-domain sample, $N$ trajectories are generated to calculate two metrics: $\text{Reasoning Uncertainty}(x) = 1 - 2|\text{Acc}(x) - 0.5|$ (preferring difficult samples with accuracy near 50%) and Exploration Diversity (average entropy of the top 20% highest-entropy tokens in each trajectory). The product of these metrics yields a comprehensive score $c(x)$, used to select top-$K$ samples. Only 384 general-domain samples are needed for significant improvement.
    - **Design Motivation**: Indiscriminately mixing large amounts of general data is computationally expensive, and low-quality samples may introduce noise.

2.  **Entropy Dynamics Alignment (EDA) Reward**:
    - **Function**: Guide the target-domain exploration patterns to align with the general domain.
    - **Mechanism**: The token-level entropy sequence of each trajectory is defined as the "Entropy Dynamics" $\tau_y = (\mathcal{H}_1, \mathcal{H}_2, ..., \mathcal{H}_{|y|})$, capturing magnitude and fine-grained variations. For target-domain trajectories, intra-domain similarity $\mathcal{S}_{\text{intra}}$ (maximum similarity with other target trajectories) and inter-domain similarity $\mathcal{S}_{\text{inter}}$ (maximum similarity with general-domain trajectories) are calculated. An extra reward $r_{\text{EDA}} = 1$ is given if $\mathcal{S}_{\text{inter}} > \mathcal{S}_{\text{intra}}$, otherwise $0$. The final reward is $r = r_{\text{Acc}} + r_{\text{EDA}}$.
    - **Design Motivation**: Even after mixing data, target-domain entropy remains lower than general-domain entropy; explicit guidance is required. EDA is not a naive entropy increase but a controllable enhancement referencing natural general-domain patterns.

3.  **Trajectory-Level Entropy Dynamics Representation**:
    - **Function**: Provide richer entropy features than scalar aggregation.
    - **Mechanism**: Entropy is not simply averaged; the complete token-level sequence is preserved. Trajectories of different lengths are aligned via interpolation before calculating similarity. This captures both absolute entropy and its variation patterns (e.g., entropy peaks during reasoning steps).
    - **Design Motivation**: Direct aggregation (e.g., mean entropy) loses fine-grained information; two trajectories with the same mean entropy might exhibit entirely different exploration patterns.

### Loss & Training

Standard RLVR optimization using PPO/GRPO is employed, with the reward being the sum of the accuracy reward and the EDA reward. Experiments were conducted on Qwen3 and LLaMA-3.2 series models, utilizing 32 target-domain samples and 384 selected high-value general-domain samples.

## Key Experimental Results

### Main Results

| Model / Method | Target Samples | General Samples | Medicine Avg | Physics Avg | Code Avg |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Qwen3-1.7B Few-shot | 32 | 0 | 41.70 | - | - |
| Qwen3-1.7B Full-shot | 1K | 0 | 44.73 | 36.24 | 39.18 |
| Qwen3-1.7B **Ours** (HEAL) | 32 | 384 | **44.67** | **37.20** | **41.28** |
| Qwen3-4B Few-shot | 32 | 0 | 49.22 | 42.57 | 47.35 |
| Qwen3-4B Full-shot | 1K | 0 | 56.70 | 44.57 | 51.00 |
| Qwen3-4B **Ours** (HEAL) | 32 | 384 | 50.82 | **46.99** | **53.42** |

### Ablation Study

| Configuration | Description | Effect |
| :--- | :--- | :--- |
| Few-shot (32 samples) | Target domain only | Baseline, severe entropy collapse |
| Only-General (10K) | General domain only | No target domain knowledge provided |
| Hybrid (32+384) | Hybrid without EDA | Mitigates collapse but insufficient |
| **Ours** (HEAL 32+384+EDA) | Full framework | Optimal, matches Full-shot performance |

### Key Findings

- With only 32 target samples and 384 general samples, HEAL matches or exceeds the performance of full-shot RLVR (1K samples) across multiple settings.
- Hybrid training significantly mitigates entropy collapse, but target-domain entropy still lags behind the general domain; EDA further closes this gap.
- The data selection strategy outperforms random sampling of general-domain data.
- Consistent **Gain** is observed across Medicine, Physics, Code, and Math domains.
- HEAL outperforms existing entropy regularization methods (e.g., DAPO-style entropy rewards) in low-resource scenarios.

## Highlights & Insights

- **Systematic revelation of entropy collapse in few-shot RLVR**: Prior work reported training collapse without identifying the root cause; this paper attributes it to entropy collapse and provides a solution.
- **Trajectory-level entropy dynamics are superior features**: By preserving fine-grained exploration patterns during generation, this representation can be generalized to other scenarios requiring analysis of generative behavior.
- **The concept of general domain as an "Exploration Teacher" is intuitive**: The general domain is not expected to provide domain knowledge but rather a behavioral reference for "how to explore."

## Limitations & Future Work

- General-domain data selection relies on pre-generating multiple trajectories, incurring some upfront costs.
- The binary reward design for EDA might be insufficiently granular; continuous rewards might perform better.
- Validated only on small-to-mid-scale models (1.7B-4B); few-shot RLVR behavior might differ in larger models.
- The choice of similarity functions (e.g., DTW vs. Cosine) has not been fully explored.

## Related Work & Insights

- **vs. Standard Entropy Regularization**: These methods increase entropy naively without constraints, potentially undermining stability. HEAL achieves controllable enhancement by referencing natural patterns.
- **vs. Data-Augmentation Few-Shot RLVR**: Data augmentation relies on the model's internal knowledge and is limited by its boundaries. HEAL introduces external general data as guidance for exploration.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Clearly defines the entropy collapse problem in few-shot RLVR with an ingenious EDA reward design.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Very comprehensive across four domains, two model series, and multiple ablations.
- **Writing Quality**: ⭐⭐⭐⭐ Convincing problem introduction, clear methodology, and helpful diagrams.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Reasoning with Exploration: An Entropy Perspective](../../AAAI2026/reinforcement_learning/reasoning_with_exploration_an_entropy_perspective.md)
- [\[ICLR 2026\] Controllable Exploration in Hybrid-Policy RLVR for Multi-Modal Reasoning](../../ICLR2026/reinforcement_learning/controllable_exploration_in_hybrid-policy_rlvr_for_multi-modal_reasoning.md)
- [\[ICLR 2026\] Exploration vs Exploitation: Rethinking RLVR through Clipping, Entropy, and Spurious Reward](../../ICLR2026/reinforcement_learning/exploration_vs_exploitation_rethinking_rlvr_through_clipping_entropy_and_spuriou.md)
- [\[ACL 2026\] Semantic-Space Exploration and Exploitation in RLVR for LLM Reasoning](semantic-space_exploration_and_exploitation_in_rlvr_for_llm_reasoning.md)
- [\[ACL 2026\] Targeted Exploration via Unified Entropy Control for Reinforcement Learning](targeted_exploration_via_unified_entropy_control_for_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
