---
title: >-
  [Paper Note] FOUNDER: Grounding Foundation Models in World Models for Open-Ended Embodied Decision Making
description: >-
  [ICML 2025][Robotics][foundation model] The FOUNDER framework is proposed to align the multimodal task representations of Foundation Models (FMs) to the state space of World Models (WMs) by learning a mapping function. In combination with a temporal distance predictor, it generates reward signals to achieve open-ended multi-task embodied decision-making without environment rewards.
tags:
  - "ICML 2025"
  - "Robotics"
  - "foundation model"
  - "world model"
  - "Goal-Conditioned RL"
  - "Embodied Decision Making"
  - "Temporal Distance"
date: 2026-05-08
content_hash: 58c232d3aa66aaed
---

# FOUNDER: Grounding Foundation Models in World Models for Open-Ended Embodied Decision Making

**Conference**: ICML 2025  
**arXiv**: [2507.12496](https://arxiv.org/abs/2507.12496)  
**Code**: [Project Homepage](https://sites.google.com/view/founder-rl)  
**Area**: Robotics  
**Keywords**: foundation model, world model, Goal-Conditioned RL, Embodied Decision Making, Temporal Distance

## TL;DR

The FOUNDER framework is proposed to align the multimodal task representations of Foundation Models (FMs) to the state space of World Models (WMs) by learning a mapping function. In combination with a temporal distance predictor, it generates reward signals to achieve open-ended multi-task embodied decision-making without environment rewards.

## Background & Motivation

**Background**: Foundation Models (FMs) possess strong vision-language generalization capabilities, while World Models (WMs) excel at modeling environment dynamics and generating trajectories through imagination to improve sample efficiency. The two offer complementary advantages in task generalization—FMs provide high-level semantic knowledge, and WMs provide low-level dynamics modeling.

**Limitations of Prior Work**: WMs rely on manually designed task reward functions to complete specific tasks, restricting their applicability under open-ended task instructions such as text or video. Although FMs can understand multimodal task instructions, their generic representations are not aligned with target RL environments, preventing direct application to policy learning within WMs.

**Key Challenge**: Existing methods such as GenRL employ step-by-step visual alignment to bridge FMs and WMs. However, this approach is essentially "style transfer"—aligning only visual appearance rather than deep semantics. When tasks involve complex observations, inherent static attributes, or cross-domain viewpoint variations, it is prone to producing misleading task interpretations and reward signals, leading to reward hacking (e.g., an agent merely waving its hand near a light switch instead of actually turning it on).

**Key Insight**: Mapping FM representations to goal states in WMs, transforming task solving into Goal-Conditioned RL (GCRL), and utilizing temporal distance as a reward signal to capture deep task semantics, rather than relying on visual similarity.

**Core Idea**: Learn an FM$\rightarrow$WM mapping function to "infer" the physical state corresponding to multimodal task prompts, and then utilize a temporal distance predictor to generate informative rewards for goal-conditioned policy learning through imagination in the WM.

## Method

### Overall Architecture

FOUNDER is divided into two phases:

1. **Pre-training Phase**: (a) Train a DreamerV3-style World Model using offline data; (b) Learn a temporal distance predictor between states; (c) Learn a mapping function from VLM representations to WM states.
2. **Behavior Learning Phase**: Given a text/video task prompt $\rightarrow$ VLM encoding $\rightarrow$ Mapping to WM goal state $\rightarrow$ Performing GCRL policy learning in the WM using temporal distance as rewards.

Three pre-training components cooperatively support behavior learning, and all are task-agnostic, allowing reusability across a variety of downstream tasks.

### Key Designs

1. **World Model Learning**: The RSSM architecture from DreamerV3 is adopted to optimize the ELBO on offline observation-action data. The state $z_t = (h_t, s_t)$ contains deterministic and stochastic components, preserving full historical information. Unlike GenRL, FOUNDER retains the original DreamerV3 structure (with both the encoder and decoder conditioned on $h_t$), ensuring that WM states contain richer environmental information. No reward model is included (as the data lacks reward annotations).

2. **Mapping Function Learning (FM$\rightarrow$WM Grounding)**: The core mechanism is to treat the VLM embedding $e_t$ of a short video as the external observation of some physical state $z_t$ in the environment, and learn an inference function $Q_\psi(z|e)$. Specifically:

    - Construct paired data using offline trajectories: $e_t = \text{VLM}(o_{t-k:t})$ and $z_t \sim \text{WM}(\cdot|o_{\leq t}, a_{<t})$
    - Optimize two objectives: (a) A KL constraint to align the mapped state distribution with the WM encoding distribution; (b) An autoencoder reconstruction loss ensuring the mapped state retains full VLM semantic information
    - This is equivalent to variational inference, but replaces the KL constraint of a stochastic prior with WM state alignment
    - During inference, the task prompt is encoded by the VLM and directly mapped to the goal state: $z_g \sim Q_\psi(\cdot|e_g)$

3. **Temporal Distance Predictor**: Learn a model $D_\theta$ to predict the time-step distance between two WM states. Positive sample pairs $(z_t, z_{t+c})$ are sampled from the same trajectory to predict the normalized distance $c/T$, while negative sample pairs from different trajectories predict the maximum distance of 1. Core formula: $\min_{D_\theta} \text{MSE}(D_\theta(z_t, z_{t+c}), c/T)$. Temporal distance is more robust than cosine similarity because it extracts environmental dynamics, capturing deep task semantics beyond visual details.

### Loss & Training

- **WM Training**: Standard ELBO objective (reconstruction + KL divergence), 500K gradient steps
- **Mapping Function**: $\mathcal{L}_{map} = \mathbb{D}_{KL}[Q_\psi(\cdot|e_t) \| \text{WM}(\cdot|o_{\leq t}, a_{<t})] + \mathbb{E}[-\ln P_\psi(e_t|\hat{z}_t)]$
- **Temporal Distance**: MSE loss (on positive and negative sample pairs)
- **Policy Learning**: Reward $r_D(z_t, z_g) = -D_\theta(z_t, z_g)$, DreamerV3-style Actor-Critic, behavior learning up to 50K steps
- VLM uses InternVideo2, with all methods sharing the same VLM

## Key Experimental Results

### Main Results (DMC + Kitchen Multi-Task Text Instructions)

| Task | GenRL | WM-CLIP | FOUNDER w/o TempD | FOUNDER | Gain |
|------|-------|---------|-------------------|---------|------|
| Cheetah Flip | -0.04 | -0.11 | -0.26 | **0.97** | From failure to success |
| Cheetah Run | 0.68 | 0.51 | 0.21 | **0.81** | +19% |
| Kitchen Light | 0.00 | 0.35 | 1.00 | **0.97** | From 0 to near-perfect score |
| Kitchen Burner | 0.35 | 0.10 | 1.00 | **0.60** | +71% vs GenRL |
| Overall (19 tasks) | 0.60 | 0.57 | 0.52 | **0.81** | +35% |

FOUNDER ranks highest in 14 out of 19 tasks, with the Overall score improving from 0.60 to 0.81.

### Ablation Study

| Configuration | Overall (19 tasks) | Description |
|------|-------------------|------|
| FOUNDER (Full) | **0.81** | Mapping + Temporal Distance + GCRL |
| FOUNDER w/o TempD | 0.52 | Substituting temporal distance with cosine similarity, severe degradation |
| GenRL-TempD | 0.59 | Only adding temporal distance to GenRL yields poor results |
| GenRL | 0.60 | Step-by-step visual alignment |
| WM-CLIP | 0.57 | Alignment in VLM space |

**Key Findings**: Adding temporal distance to GenRL alone is ineffective (GenRL-TempD 0.59 vs. GenRL 0.60); it only becomes effective within the GCRL framework of FOUNDER. FOUNDER w/o TempD degrades significantly to 0.52, indicating that cosine similarity is prone to triggering reward hacking.

### Reward Consistency Evaluation

| Method | Corr↑ | Regret↓ | Precision↑ | F1↑ |
|------|-------|---------|-----------|-----|
| GenRL | 0.12 | 0.37 | 0.47 | 0.44 |
| FOUNDER | **0.54** | **0.07** | **1.00** | **0.59** |

The pseudo-rewards learned by FOUNDER exhibit the highest consistency with the ground-truth rewards. A Precision of 1.0 implies that low rewards are never misclassified as high rewards, thereby effectively preventing reward hacking.

### Key Findings

- **Cross-Embodiment Transfer**: FOUNDER wins in 11 out of 12 cross-embodiment tasks. It is the only method that successfully completes tasks in the Cheetah domain using Walker/Stickman video prompts.
- **Minecraft**: FOUNDER significantly outperforms GenRL in 3 out of 5 tasks, matching or exceeding the MineCLIP oracle pre-trained on large-scale internet data.
- **Inherent Limitations of GenRL**: GenRL completely fails on static Kitchen tasks (Light=0.00) because it lacks temporal awareness and only performs visual-level matching.

## Highlights & Insights

- **Elegant Problem Reformulation**: Formulating "understanding multimodal task prompts" as "inferring physical states in an environmental simulator" offers a deeper perspective than step-by-step alignment.
- **Temporal Distance vs. Cosine Similarity**: Unveiling the vulnerability of cosine similarity in the WM state space—it is easily deceived by static "visual mimicry", whereas temporal distance effectively encodes dynamic information.
- **Precision over Recall**: In reward design for policy learning, avoiding False Positives (misclassifying low rewards as high) is more critical than identifying all high-reward behaviors, analogous to cost-sensitive learning.

## Limitations & Future Work

- Dependence on offline data quality—if elements in the prompts do not appear in the dataset, the mapping may fail.
- Current experiments primarily focus on short-horizon tasks; long-horizon tasks require task decomposition by incorporating the reasoning capabilities of FMs.
- The mapping function is trained only on interaction data from the target environment. Although the generalization ability of the VLM extends to other domains, robustness to out-of-distribution scenarios requires further validation.
- Incorporating real-world videos into WM learning can be explored to expand capability boundaries.

## Related Work & Insights

- **GenRL** (Mazzaglia et al., 2024): The most relevant work, modeling the FM$\rightarrow$WM mapping in a seq2seq manner, yet essentially performing visual-level style transfer.
- **LEXA** / **Director**: Pioneering works on GCRL within WMs, but utilized for exploration or sub-task decomposition rather than direct behavior learning.
- **MineDojo** (Fan et al., 2022): Provides an open-ended Minecraft task benchmark, where the effectiveness of FOUNDER is validated.
- **Insights**: The integration of FMs and WMs is an important direction for embodied AI, where the key lies in establishing bridges within an appropriate representation space.

## Rating

- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Hi Robot: Open-Ended Instruction Following with Hierarchical Vision-Language-Action Models](hi_robot_open-ended_instruction_following_with_hierarchical_vision-language-acti.md)
- [\[ICML 2025\] SENSEI: Semantic Exploration Guided by Foundation Models to Learn Versatile World Models](sensei_semantic_exploration_guided_by_foundation_models_to_learn_versatile_world.md)
- [\[NeurIPS 2025\] Self-Improving Embodied Foundation Models](../../NeurIPS2025/robotics/self-improving_embodied_foundation_models.md)
- [\[CVPR 2025\] Decision SpikeFormer: Spike-Driven Transformer for Decision Making](../../CVPR2025/robotics/decision_spikeformer_spike-driven_transformer_for_decision_making.md)
- [\[CVPR 2025\] UniAct: Universal Actions for Enhanced Embodied Foundation Models](../../CVPR2025/robotics/universal_actions_for_enhanced_embodied_foundation_models.md)

</div>

<!-- RELATED:END -->
