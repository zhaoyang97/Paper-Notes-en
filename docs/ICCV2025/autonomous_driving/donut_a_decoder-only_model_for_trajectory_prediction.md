---
title: >-
  [Paper Note] DONUT: A Decoder-Only Model for Trajectory Prediction
description: >-
  [ICCV 2025][Autonomous Driving][trajectory prediction] DONUT draws inspiration from the decoder-only architecture of LLMs and proposes a unified autoregressive model for processing both historical and future trajectories, coupled with an *overprediction* strategy to improve anticipation of the distant future. It achieves state-of-the-art performance on the Argoverse 2 benchmark.
tags:
  - ICCV 2025
  - Autonomous Driving
  - trajectory prediction
  - decoder-only
  - autoregressive model
  - motion prediction
date: 2026-05-08
content_hash: 9c85ae8be65789a1
---

# DONUT: A Decoder-Only Model for Trajectory Prediction

**Conference**: ICCV 2025
**arXiv**: [2506.06854](https://arxiv.org/abs/2506.06854)
**Code**: [https://vision.rwth-aachen.de/donut](https://vision.rwth-aachen.de/donut)
**Area**: Autonomous Driving
**Keywords**: trajectory prediction, decoder-only, autoregressive model, motion prediction, autonomous driving

## TL;DR
DONUT draws inspiration from the decoder-only architecture of LLMs and proposes a unified autoregressive model for processing both historical and future trajectories, coupled with an *overprediction* strategy to improve anticipation of the distant future. It achieves state-of-the-art performance on the Argoverse 2 benchmark.

## Background & Motivation
Motion prediction is a core task in autonomous driving—by forecasting the future trajectories of surrounding agents, autonomous vehicles can plan ahead. Dominant approaches adopt an encoder-decoder architecture: the encoder embeds historical trajectories, and the decoder predicts future ones.

**Limitations of Prior Work**:

1. **Single-shot prediction of the entire future** lacks awareness of scene elements near distant time steps, leading to inaccurate long-horizon predictions.
2. **Recurrent decoders**, while iterative, suffer from inconsistency—inputs alternate between learned embeddings and previous-step outputs, complicating the decoder's task.
3. Recurrent decoders can only access the "stale" historical information provided by the encoder; at distant time steps, the states of other agents are already significantly out of sync.
4. Encoder and decoder employ different modules, creating a structural disconnect in how historical and future trajectories are processed.

**Key Insight**: Analogous to the successful decoder-only paradigm in LLMs, the paper unifies the processing of historical and future trajectory sequences within a single autoregressive model to ensure consistency and up-to-date information. Inspired by multi-token prediction in LLMs, an overprediction strategy is introduced to help the model anticipate further into the future.

## Method

### Overall Architecture
All agent trajectories are segmented into sub-trajectories of $T_{\text{sub}}=10$ time steps (1 second). A single decoder network processes historical sub-trajectories and autoregressively predicts future sub-trajectories step by step. Each step consists of: a proposer generating initial predictions and overpredictions → updating the reference point → a refiner correcting the predictions.

### Key Designs

1. **Unified Decoder-Only Architecture**:

    - **Function**: A single model simultaneously encodes historical trajectories and predicts future ones.
    - **Mechanism**:
        - Historical trajectories are processed through the proposer and refiner to produce historical tokens.
        - Future sub-trajectories are predicted step by step using the same network structure.
        - After each prediction, the reference point is updated to the predicted endpoint, and relative positional encodings with respect to surrounding scene elements are recomputed.
        - A query-centric scheme (each scene element has its own local reference frame) enables reuse of features from the previous step.
    - **Design Motivation**: Eliminates the inconsistencies of encoder-decoder designs (heterogeneous input types, stale information of other agents) and allows the model to naturally transfer knowledge from historical encoding to future prediction.

2. **Overprediction Strategy**:

    - **Function**: When predicting the current sub-trajectory, the model simultaneously predicts the *next* sub-trajectory as an auxiliary task.
    - **Mechanism**: The proposer outputs a prediction $\hat{Y}'_{\{0;T_{\text{sub}}\}}$ and an overprediction $\hat{Y}'^{\text{over}}_{\{T_{\text{sub}};2T_{\text{sub}}\}}$. The overprediction is supervised by ground truth during training and discarded at inference.
    - **Design Motivation**: Inspired by multi-token prediction in LLMs. It forces the model to consider a longer temporal horizon, providing better future awareness for the current prediction step. Experiments show that overprediction promotes more stable training convergence and unlocks the potential of the refiner.

3. **Proposer and Refiner**:

    - **Function**: Responsible for initial prediction and subsequent correction, respectively.
    - **Core Structure**: Both share the same architecture.
        - *Tokenizer*: Extracts Fourier features of position, heading, motion vectors, and velocity for each sub-trajectory; an MLP fuses these into a single token.
        - *Four attention types*: (1) temporal self-attention (historical tokens of the same agent), (2) map attention (road tokens within radius $r=50$ m), (3) social attention (tokens of other agents), (4) modal attention (across different modes of the same agent).
        - *Detokenizer*: An MLP predicting the next sub-trajectory and the overprediction.
    - **Design Motivation**: After the reference point is updated, the refiner can access the latest predicted trajectories of other agents and more accurate scene information.

### Loss & Training
- Positions are parameterized with a Laplace mixture distribution: $p(\hat{Y}_n^{\text{pos}}) = \sum_k P_{n,k} \prod_t \text{Laplace}(\hat{Y}_{n,t}^{\text{pos}} | \mu, b)$
- Headings are parameterized with a von Mises distribution.
- Loss is computed only for the closest mode (minimum endpoint distance).
- Losses are applied independently to proposed and refined trajectories, as well as to main predictions and overpredictions.
- Training: AdamW optimizer, 4×H100 GPUs, batch size 64 (with gradient accumulation), 60 epochs.

## Key Experimental Results

### Main Results (Argoverse 2 Test Leaderboard, Non-Ensemble Methods)

| Method | b-minFDE₆↓ | minFDE₆↓ | minADE₆↓ | MR₆↓ |
|------|-----------|---------|---------|------|
| QCNet | 1.91 | 1.29 | 0.65 | 0.16 |
| DeMo | 1.84 | 1.17 | 0.61 | 0.13 |
| SmartRefine | 1.86 | 1.23 | 0.63 | 0.15 |
| SEPT* | 1.74 | 1.15 | 0.61 | 0.14 |
| QCNet* (ensemble) | 1.78 | 1.19 | 0.62 | 0.14 |
| DeMo* (ensemble) | 1.73 | 1.11 | 0.60 | 0.12 |
| **DONUT** | **1.79** | **1.16** | 0.63 | 0.14 |

DONUT achieves state-of-the-art among non-ensemble methods on the primary metric b-minFDE₆, and achieves overall state-of-the-art on MR₁ (0.54).

### Ablation Study

| Configuration | b-minFDE₆↓ | minFDE₆↓ | minADE₆↓ | MR₆↓ | Note |
|------|-----------|---------|---------|------|------|
| Encoder-decoder baseline | 1.874 | 1.253 | 0.720 | 0.157 | QCNet |
| Decoder-only | 1.838 | 1.198 | 0.745 | 0.145 | Decoder-only alone improves FDE |
| + Overprediction | 1.838 | 1.193 | 0.728 | 0.146 | Marginal improvement |
| + Refinement | 1.835 | 1.218 | 0.751 | 0.150 | Unstable training when used alone |
| + Both | **1.807** | **1.176** | **0.722** | **0.144** | Significant synergy |

### Key Findings
- The decoder-only architecture outperforms the encoder-decoder baseline on both b-minFDE₆ and minFDE₆, with a particularly pronounced advantage in long-horizon prediction (6 seconds).
- Overprediction and refinement yield limited gains individually, but their combination produces significant synergistic improvement.
- The largest accuracy gains occur at distant horizons: the decoder-only model achieves substantially lower FDE than the encoder-decoder in the 3–6 second range.
- DONUT achieves state-of-the-art on MR₁ (0.54), indicating highly accurate best-mode predictions.
- The encoder-decoder baseline slightly outperforms DONUT on minADE₆, possibly because its encoder generates a dedicated token for each individual time step, offering finer granularity.

## Highlights & Insights
- The core insight draws on the analogy to the success of LLMs: motion prediction is fundamentally a sequence prediction task, for which the decoder-only paradigm is most fitting.
- The overprediction strategy cleverly adapts multi-token prediction from LLMs to the trajectory prediction domain.
- Real-time reference point updates ensure the model always has access to scene information in the vicinity of the current position.
- Qualitative analyses are highly intuitive: in complex intersection scenarios, DONUT demonstrably outperforms QCNet.

## Limitations & Future Work
- DONUT slightly underperforms the encoder-decoder baseline on minADE₆; the coarse-grained sub-trajectory tokenization may sacrifice precision at intermediate time steps.
- Validation is currently limited to single-agent prediction; extension to joint multi-agent prediction warrants future exploration.
- Inference speed analysis is absent—it remains unclear whether the autoregressive approach is slower than single-shot prediction.
- Sensitivity analysis with respect to different values of $T_{\text{sub}}$ is insufficient.

## Related Work & Insights
- Builds upon QCNet's query-centric scene encoding, replacing the decoder structure on top of this foundation.
- Closely mirrors the decoder-only trend in the LLM community (GPT series).
- Differs from GPT-style models in motion simulation (e.g., MotionLM): motion prediction requires a fixed number of multi-modal predictions to cover the future distribution.
- Takeaway: successful paradigms from NLP can be transferred to other sequence prediction tasks through thoughtful adaptation.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The combination of decoder-only architecture and overprediction is a novel contribution to the trajectory prediction field.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Ablations are comprehensive, leaderboard validation is rigorous, and qualitative analyses are intuitive.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Motivation is articulated clearly; the analogy to LLMs is apt and well-calibrated.
- **Value**: ⭐⭐⭐⭐ Achieves state-of-the-art on the highly competitive Argoverse 2 benchmark, validating the effectiveness of the decoder-only paradigm for trajectory prediction.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Generative Active Learning for Long-tail Trajectory Prediction via Controllable Diffusion Model](generative_active_learning_for_long-tail_trajectory_prediction_via_controllable_.md)
- [\[ICCV 2025\] Foresight in Motion: Reinforcing Trajectory Prediction with Reward Heuristics](foresight_in_motion_reinforcing_trajectory_prediction_with_reward_heuristics.md)
- [\[ICCV 2025\] LangTraj: Diffusion Model and Dataset for Language-Conditioned Trajectory Simulation](langtraj_diffusion_model_and_dataset_for_language-conditioned_trajectory_simulat.md)
- [\[ICCV 2025\] GS-Occ3D: Scaling Vision-only Occupancy Reconstruction with Gaussian Splatting](gs-occ3d_scaling_vision-only_occupancy_reconstruction_with_gaussian_splatting.md)
- [\[ICCV 2025\] SRefiner: Soft-Braid Attention for Multi-Agent Trajectory Refinement](srefiner_soft-braid_attention_for_multi-agent_trajectory_refinement.md)

<!-- RELATED:END -->
