---
title: >-
  [Paper Note] Exploration-Driven Generative Interactive Environments
description: >-
  [CVPR 2025][LLM Pretraining][World Models] This work provides an open-source implementation of the Genie world model (GenieRedux), which is enhanced to GenieRedux-G by incorporating ground-truth action conditioning, Token Distance Cross-Entropy (TDCE) loss, and token skip connections. Additionally, the AutoExplore agent is proposed to utilize the world model's token prediction uncertainty as an intrinsic reward to drive diverse data collection…
tags:
  - "CVPR 2025"
  - "LLM Pretraining"
  - "World Models"
  - "Interactive Environments"
  - "Exploration Agents"
  - "Video Generation"
  - "Game Simulation"
date: 2026-05-08
content_hash: 52f3e1414b465256
---

# Exploration-Driven Generative Interactive Environments

**Conference**: CVPR 2025  
**arXiv**: [2504.02515](https://arxiv.org/abs/2504.02515)  
**Code**: [https://github.com/insait-institute/GenieRedux](https://github.com/insait-institute/GenieRedux)  
**Area**: LLM Pre-training  
**Keywords**: World Models, Interactive Environments, Exploration Agents, Video Generation, Game Simulation

## TL;DR
This work provides an open-source implementation of the Genie world model (GenieRedux), which is enhanced to GenieRedux-G by incorporating ground-truth action conditioning, Token Distance Cross-Entropy (TDCE) loss, and token skip connections. Additionally, the AutoExplore agent is proposed to utilize the world model's token prediction uncertainty as an intrinsic reward to drive diverse data collection, improving simulation quality by up to 7.4 PSNR.

## Background & Motivation

**Background**: World models (e.g., Genie) learn to simulate interactive environments—generating corresponding video frames given an action sequence. However, Genie remains closed-source, and open-source implementations are lacking. Training data is typically collected by random agents, resulting in low exploration efficiency.

**Limitations of Prior Work**: (1) Random agents tend to repeatedly visit the same areas, leading to a lack of diversity in the collected videos. (2) The Latent Action Model (LAM) has limited accuracy in inferring actions from videos, which introduces noise. (3) The standard cross-entropy loss does not account for the semantic distance between tokens in the codebook—predicting a semantically close token is penalized just as severely as predicting a completely incorrect one.

**Key Challenge**: World models require diverse training videos, but collecting diverse data requires efficient exploration by agents. Traditional exploration rewards (e.g., curiosity) require manual design and are environment-dependent.

**Goal**: (1) Provide and improve an open-source implementation of Genie. (2) Design an environment-agnostic exploration agent to collect diverse training data for the world model.

**Key Insight**: The prediction of world models in uncertain regions exhibits high uncertainty (high entropy). This intrinsic "uncertainty signal" can be leveraged as an exploration reward—regions where the world model is uncertain are precisely those requiring more training data.

**Core Idea**: Use the world model's own token prediction uncertainty as an intrinsic exploration reward to drive the agent to collect the data most needed by the world model, forming an "explore-to-learn" self-improvement loop.

## Method

### Overall Architecture
GenieRedux (Video Tokenizer + LAM + Dynamics) $\rightarrow$ GenieRedux-G (incorporating ground-truth action conditioning + TDCE loss + token skip connections) $\rightarrow$ AutoExplore Agent (utilizing the average entropy of the top-25% most uncertain tokens from the Dynamics module as rewards) $\rightarrow$ diverse data collection $\rightarrow$ world-model fine-tuning.

### Key Designs

1. **Token Distance Cross-Entropy (TDCE)**:

    - **Function**: Weights the prediction loss based on the semantic distance between codebook tokens.
    - **Mechanism**: $TDCE(x,y) = (y^T K) \cdot \text{softmax}(x) + CE(x,y)$, where $K$ is the cosine distance matrix between codebook tokens. Predicting semantically close tokens (visually similar) is penalized less severely, while completely incorrect tokens receive heavier penalties.
    - **Design Motivation**: Ablation displays that TDCE contributes +0.41 PSNR (from 26.65 to 27.06) because it allows the model to allocate more reasonable probabilities to semantically close alternative tokens.

2. **AutoExplore Agent (Uncertainty-Driven Exploration)**:

    - **Function**: Collects diverse training data that the world model needs the most.
    - **Mechanism**: Reward = average entropy of the top-25% most uncertain tokens in the current frame prediction from the world model's Dynamics module. High-uncertainty regions = regions where the world model is poor at predicting = regions requiring more training data. Actor-Critic (CNN+LSTM) maximizes the cumulative exploration reward.
    - **Design Motivation**: Random exploration yields FID 42.34 / PSNR 27.04 $\rightarrow$ AutoExplore yields FID 11.33 / PSNR 33.61 (Adventure Island II), demonstrating massive gains.

3. **RetroAct Dataset**:

    - **Function**: A standardized benchmark with retro game environments and action annotations.
    - **Mechanism**: 974 annotated retro game environments, including behavioral and control labels, supporting standardized training and evaluation of world models.
    - **Design Motivation**: Prior to this, there was no unified world-model benchmark for interactive environments.

### Loss & Training
GenieRedux-G: TDCE + token skip connections. AutoExplore: Actor-Critic + uncertainty rewards. Pretrained on Platformers-200 (4.6M images) $\rightarrow$ fine-tuned on Platformers-50 (4.8M images) $\rightarrow$ fine-tuned further on each environment using data collected by AutoExplore.

## Key Experimental Results

### Main Results

| Environment | Policy | FID↓ | PSNR↑ | ΔPSNR↑ |
|------|------|------|-------|--------|
| Adventure Island II | Random | 42.34 | 27.04 | 1.19 |
| Adventure Island II | **AutoExplore AR** | **11.33** | **33.61** | **2.09** |
| Super Mario Bros | Random | 29.83 | 34.24 | 0.56 |
| Super Mario Bros | **AutoExplore AR** | **9.33** | **37.77** | **0.76** |

### Ablation Study

| Component | PSNR |
|------|------|
| GenieRedux-G Base | 26.36 |
| +Token Input | 26.65 |
| +TDCE Loss | 27.06 |
| +Autoregressive | **28.07** |

### Key Findings
- **AutoExplore improves PSNR by up to 7.4** (Adventure Island II): demonstrating the massive value of smart exploration for data collection.
- **TDCE exploits the codebook structure**: semantically close tokens should not be penalized equally.
- **Random exploration is sufficient for training base models**: However, intelligent exploration triggers a leap in fine-tuning quality.
- **User study validation**: Models trained with exploration are favored as better with a preference rate of 0.75.

## Highlights & Insights
- **"Using world model uncertainty as an exploration reward"** forms an elegant self-improvement loop—where the model performs poorly is exactly where more data is required.
- **TDCE loss is generalizable to any VQ-VAE-based model**—leveraging the semantic structural information of the codebook is a universal improvement.
- **The open-sourcing of GenieRedux** provides a reproducible foundation for world model research.

## Limitations & Future Work
- Only validated on 2D retro games; its applicability to 3D environments remains unknown.
- The Actor-Critic in AutoExplore needs to be trained on each new environment.
- The PSNR improvement mainly stems from exploration covering new areas, with limited quality improvements in already-seen areas.

## Related Work & Insights
- **vs. Genie (DeepMind)**: Genie is closed-source. GenieRedux provides an open-source implementation and achieves significant improvements via TDCE, ground-truth actions, and exploration agents.
- **vs. GameNGen**: These are game-specific models. GenieRedux is a generalist world model across various environments.

## Rating
- Novelty: ⭐⭐⭐⭐ Uncertainty-driven exploration and the TDCE loss are clever contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-environment comparisons, component ablations, and a user study.
- Writing Quality: ⭐⭐⭐⭐ The narrative of the exploration-learning loop is clear.
- Value: ⭐⭐⭐⭐ The open-source world model implementation makes a significant contribution to the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Heterogeneous Adversarial Play in Interactive Environments](../../NeurIPS2025/llm_pretraining/heterogeneous_adversarial_play_in_interactive_environments.md)
- [\[ICML 2025\] Provable Maximum Entropy Manifold Exploration via Diffusion Models](../../ICML2025/llm_pretraining/provable_maximum_entropy_manifold_exploration_via_diffusion_models.md)
- [\[NeurIPS 2025\] Learning to Flow from Generative Pretext Tasks for Neural Architecture Encoding](../../NeurIPS2025/llm_pretraining/learning_to_flow_from_generative_pretext_tasks_for_neural_architecture_encoding.md)
- [\[ACL 2025\] AutoDS: Autonomous Data Selection with Zero-shot Generative Classifiers for Mathematical Texts](../../ACL2025/llm_pretraining/autonomous_data_selection_with_zero-shot_generative_classifiers_for_mathematical.md)
- [\[CVPR 2025\] ScaMo: Exploring the Scaling Law in Autoregressive Motion Generation Model](scamo_exploring_the_scaling_law_in_autoregressive_motion_generation_model.md)

</div>

<!-- RELATED:END -->
