---
title: >-
  [Paper Note] PAR: Poly-Autoregressive Prediction for Modeling Interactions
description: >-
  [CVPR 2025][Autonomous Driving][Multi-agent Interaction] PAR (Poly-Autoregressive) proposes a simple and unified multi-agent behavior prediction framework. By conditioning on the state sequences of other agents during interactions, paired with the next-timestep prediction of the same agent and learned agent ID embeddings, it outperforms single-agent autoregressive baselines across three distinct tasks: social behavior prediction, autonomous driving trajectory prediction…
tags:
  - "CVPR 2025"
  - "Autonomous Driving"
  - "Multi-agent Interaction"
  - "Autoregressive Prediction"
  - "Transformer"
  - "Trajectory Prediction"
  - "Behavior Prediction"
date: 2026-05-08
content_hash: 5de92e9d1132daeb
---

# PAR: Poly-Autoregressive Prediction for Modeling Interactions

**Conference**: CVPR 2025  
**arXiv**: [2502.08646](https://arxiv.org/abs/2502.08646)  
**Code**: Open-sourced  
**Area**: Autonomous Driving (Trajectory Prediction)  
**Keywords**: Multi-agent Interaction, Autoregressive Prediction, Transformer, Trajectory Prediction, Behavior Prediction

## TL;DR

PAR (Poly-Autoregressive) proposes a simple and unified multi-agent behavior prediction framework. By conditioning on the state sequences of other agents during interactions, paired with the next-timestep prediction of the same agent and learned agent ID embeddings, it outperforms single-agent autoregressive baselines across three distinct tasks: social behavior prediction, autonomous driving trajectory prediction, and hand-object interaction.

## Background & Motivation

Predicting the future behavior of an agent in multi-agent interaction scenarios is a core problem. Unlike autoregressive modeling in language, physical real-world interactions are jointly constrained by physical laws and the internal states of the agents, with the states of multiple agents changing *simultaneously*.

Limitations of Prior Work:
- **Standard Autoregression (AR) is Insufficient**: It only focuses on the historical state sequence of a single agent, ignoring the influence of other agents. For example, predicting that a person will continue talking, when in fact the interlocutor has already started speaking and they should transition to listening.
- **Fragmented Multi-Agent Methods**: Different interaction scenarios (social behavior, driving, hand-object interaction) design specialized solutions individually, lacking a unified framework.
- **Naive Multi-Agent AR Backfires**: Simply arranging multi-agent tokens into a sequence for next-token prediction confuses the model—because the next token represents the state of another agent at the same timestep, rather than the state of the same agent at the next timestep.

Key Insight: In interaction scenarios, the future of the ego agent depends on its own history *and* the current/past states of other agents. There is a need for a prediction paradigm of "same-agent next-timestep" instead of "sequence next-token".

## Method

### Overall Architecture

The PAR framework represents the states of $N$ agents over $T$ timesteps as a flattened sequence of $N \times T$ tokens. A Transformer decoder learns to predict the next-timestep state of the ego agent given the historical states of all agents. The framework can be applied to different tasks without modifying the architecture, requiring only adjustments to data preprocessing and tokenization.

### Key Designs

**Design 1: Same-Agent Next-Timestep Prediction — Alternative to standard next-token prediction**

- **Function**: Ensures that during each prediction, the model leverages state information from all agents at the same timestep.
- **Mechanism**: In the flattened $N \times T$ sequence, standard AR's next-token prediction predicts agent $k+1$ at time $t$ from agent $k$ at time $t$ (different agents at the same timestep). PAR changes this to predict the state of agent $k$ at time $t+1$ (the same agent at the next timestep). The loss is joint-calculated over all $N$ agents during training.
- **Design Motivation**: Next-token prediction violates causality—predicting the state of another agent at the same moment using one agent's state has no physical meaning. Same-agent next-timestep prediction is the correct temporal causal relationship.

**Design 2: Learned Agent ID Embeddings — Distinguishing multi-agent identities**

- **Function**: Informs the model which agent each token belongs to.
- **Mechanism**: Maps integer agent IDs to hidden dimension-sized vectors, which are added to the token embeddings. This allows the model to distinguish states of different agents when processing mixed sequences.
- **Design Motivation**: Ablation experiments show that multi-agent models lacking agent ID embeddings perform worse than single-agent AR, indicating that the model confuses the states of different agents.

**Design 3: Unified Framework — Supporting discrete/continuous tokens and multiple tasks**

- **Function**: Handles different types of multi-agent interaction predictions without changing the architecture.
- **Mechanism**: Discrete tokens (e.g., action categories) use standard embedding + cross-entropy loss; continuous tokens (e.g., position coordinates) use learned projection layers + regression loss. Data is sourced from video, with state sequences of each agent extracted via dataset annotations or computer vision techniques. Optional positional encodings (such as LPE in trajectory prediction) are superimposed to provide spatial information.
- **Design Motivation**: State representations in different interaction tasks vary greatly (60-dimensional action probabilities vs. 2D positions vs. 6DoF poses), but the core framework for interaction modeling should be general. A unified framework reduces the cost of migrating to new domains.

### Loss & Training

- Social behavior prediction: MSE regression loss on 60-dimensional action tokens.
- Vehicle trajectory prediction: Cross-entropy classification loss on discrete velocity/acceleration tokens.
- Hand-object interaction: Regression loss on 6DoF poses.

## Key Experimental Results

### Main Results

| Task | Metric | AR | PAR | Gain |
|------|------|-----|-----|------|
| AVA Social Behavior Prediction | mAP ↑ | 40.7 | **42.6** | +1.9 |
| AVA 2-person Interaction Categories | mAP ↑ | 36.3 | **39.8** | +3.5 |
| nuScenes Trajectory Prediction | ADE ↓ | Baseline | **-6.3%** | Relative |
| nuScenes Trajectory Prediction | FDE ↓ | Baseline | **-6.4%** | Relative |
| DexYCB Object Rotation | Error ↓ | Baseline | **-8.9%** | Relative |
| DexYCB Object Translation | Error ↓ | Baseline | **-41.0%** | Relative |

### Ablation Study

| Method | Timestep Prediction | ID Embedding | mAP ↑ |
|------|-----------|--------|-------|
| 1-agent AR | N/A | N/A | 40.7 |
| 2-agent AR | ✗ | ✗ | 38.0 |
| 2-agent PAR* | ✗ | ✓ | 40.2 |
| 2-agent PAR* | ✓ | ✗ | 40.0 |
| **2-agent PAR** | **✓** | **✓** | **42.6** |

### Key Findings

- Naive multi-agent AR (row 2) actually performs 2.7 mAP worse than single-agent AR, proving the failure of next-token prediction in multi-agent scenarios.
- Same-agent next-timestep prediction and agent ID embeddings **are both indispensable**—both are necessary conditions.
- On 2-person interaction categories (kiss +8.3, listen +7.0, hug +5.7, fight/hit +5.7), PAR's gains are particularly significant.
- The 41% relative improvement in translation prediction in DexYCB indicates that object motion is highly dependent on hand state.
- A small Transformer with only 4.4M parameters can demonstrate the advantages of PAR.

## Highlights & Insights

1. **Extreme Simplicity**: The same 4M-parameter Transformer handles three vastly different tasks without architecture modifications, only adjusting data preprocessing and tokenization.
2. **Insightful Failure Analysis**: Demonstrates why naive multi-agent AR fails; the distinction between next-token vs. next-timestep is highly pedagogical and insightful.
3. **Vivid Qualitative Analysis**: The example of talk-listen turn-taking prediction intuitively demonstrates PAR's capability to capture interaction dynamics.

## Limitations & Future Work

- Currently, the concept is validated only with a small 4M Transformer; large-scale experiments are yet to be conducted.
- Only 2-agent interactions are considered in the three tasks (ego + 1 other); scalability to more agents needs to be verified.
- Inference requires ground-truth future states (or accurate predictions) of other agents, which is an additional constraint in practical applications.
- Future work can combine PAR with larger scale Transformers and datasets.

## Related Work & Insights

- **MotionLM**: Multi-agent trajectory prediction using learned agent ID embeddings inspired PAR's design.
- **SinGAN/GPT Series**: The success of single-instance/autoregressive modeling, which PAR extends to multi-agent scenarios.
- **Flamingo/LLaVa**: Multimodal autoregressive models; PAR focuses on physical interaction rather than vision-language.
- Insight: **The key to the problem lies not in model complexity, but in correct serialization and causal modeling**—the difference between next-token and next-timestep seems subtle, but yields fundamental improvements.

## Rating

⭐⭐⭐⭐ — The simple and unified nature of the framework is commendable, and the "one framework, three tasks" validation is powerful. The logic chain analyzing the failure of naive AR and PAR's correction is clear and complete. The limitation of small-scale validation is a drawback, but the proof of concept is thorough.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] DriveGPT: Scaling Autoregressive Behavior Models for Driving](../../ICML2025/autonomous_driving/drivegpt_scaling_autoregressive_behavior_models_for_driving.md)
- [\[CVPR 2025\] ModeSeq: Taming Sparse Multimodal Motion Prediction with Sequential Mode Modeling](modeseq_taming_sparse_multimodal_motion_prediction_with_sequential_mode_modeling.md)
- [\[CVPR 2025\] Certified Human Trajectory Prediction](certified_human_trajectory_prediction.md)
- [\[ICCV 2025\] Epona: Autoregressive Diffusion World Model for Autonomous Driving](../../ICCV2025/autonomous_driving/epona_autoregressive_diffusion_world_model_for_autonomous_driving.md)
- [\[CVPR 2025\] Physical Plausibility-aware Trajectory Prediction via Locomotion Embodiment](physical_plausibility-aware_trajectory_prediction_via_locomotion_embodiment.md)

</div>

<!-- RELATED:END -->
