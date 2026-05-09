---
title: >-
  [Paper Note] JointDiff: Bridging Continuous and Discrete in Multi-Agent Trajectory Generation
description: >-
  [ICLR 2026][Image Generation][joint diffusion] This paper proposes JointDiff, a joint continuous-discrete diffusion framework that, for the first time, unifies Gaussian diffusion (for trajectories) and multinomial diffusion (for ball-possession events) in a single model. It further introduces a CrossGuid module to support weak possession guidance and text-guided semantic controllable generation, achieving state-of-the-art performance on multi-agent trajectory generation in sports scenarios.
tags:
  - ICLR 2026
  - Image Generation
  - joint diffusion
  - continuous-discrete unification
  - multi-agent
  - trajectory generation
  - controllable generation
date: 2026-05-08
content_hash: 8fd1d598694f6574
---

# JointDiff: Bridging Continuous and Discrete in Multi-Agent Trajectory Generation

**Conference**: ICLR 2026
**arXiv**: [2509.22522](https://arxiv.org/abs/2509.22522)
**Code**: [GitHub](https://github.com/kognia/JointDiff) (mentioned on project page)
**Area**: Diffusion Models / Multi-Agent Trajectory Generation
**Keywords**: joint diffusion, continuous-discrete unification, multi-agent, trajectory generation, controllable generation

## TL;DR

This paper proposes JointDiff, a joint continuous-discrete diffusion framework that, for the first time, unifies Gaussian diffusion (for trajectories) and multinomial diffusion (for ball-possession events) in a single model. It further introduces a CrossGuid module to support weak possession guidance and text-guided semantic controllable generation, achieving state-of-the-art performance on multi-agent trajectory generation in sports scenarios.

## Background & Motivation

In multi-agent systems such as team sports, continuous motion trajectories and discrete state-change events (e.g., passes, ball possession) are tightly coupled and occur simultaneously. Existing generative models face the following challenges:

**Continuous-discrete disconnect**: Most methods model only continuous trajectories and ignore discrete events (e.g., ball possession), leading to unrealistic behaviors such as implausible passing paths and distorted player-ball interactions.

**Lack of semantic controllability**: Existing trajectory diffusion models primarily control individual-level attributes (waypoints, velocities) and lack the ability to condition on scene-level semantics (e.g., "who possesses the ball," "game momentum").

**Inadequate evaluation metrics**: Individual-level ADE/FDE metrics inherited from pedestrian trajectory prediction fail to capture scene-level consistency and are insufficient for evaluating sports scenarios.

Core insight: Only by jointly modeling continuous trajectories and discrete events can realistic, consistent, and controllable multi-agent scenes be generated.

## Method

### Overall Architecture

JointDiff represents a scene state as the tuple $\mathbf{X} = (\mathbf{Y}, \mathbf{E})$, where $\mathbf{Y} \in \mathbb{R}^{T \times N \times 2}$ denotes continuous trajectory coordinates and $\mathbf{E} \in \{0,1\}^{T \times N}$ denotes discrete ball-possession events (one-hot). In the forward process, the two modalities are noised independently: trajectories via Gaussian diffusion, and events via multinomial diffusion (converging toward a uniform distribution). In the reverse process, a single neural network models both modalities simultaneously, learning cross-modal dependencies through a shared state representation.

### Key Designs

1. **Joint Continuous-Discrete Diffusion**: The forward process decomposes independently with a shared variance schedule $\{\beta_s\}$:

$$q(\mathbf{Y}_s | \mathbf{Y}_0) = \mathcal{N}(\mathbf{Y}_s; \sqrt{\bar{\alpha}_s} \mathbf{Y}_0, (1-\bar{\alpha}_s)\mathbf{I})$$
$$q(\mathbf{E}_s | \mathbf{E}_0) = \mathrm{Cat}(\mathbf{E}_s; \bar{\alpha}_s \mathbf{E}_0 + (1-\bar{\alpha}_s)/N)$$

   The reverse network $p_\theta$ conditions on the full state $(\mathbf{Y}_s, \mathbf{E}_s)$ and outputs two heads: a regression head predicting trajectory noise $\epsilon_\theta$, and a classification head predicting original event probabilities $\hat{\mathbf{E}}_0$. This allows the reverse denoising process to learn cross-modal dependencies even though the forward process is modality-independent. Multinomial diffusion is preferred over absorbing-state diffusion because multinomial allows discrete variables to be continuously revised throughout the process, whereas absorbing-state diffusion freezes tokens once unmasked, precluding subsequent corrections.

2. **CrossGuid Conditioning Module**: Inserted within the Social-Temporal Block between Temporal Mamba and Social Transformer, CrossGuid injects external guidance signals. Two variants are implemented:

    - **Weak Possession Guidance (WPG)**: Takes a player index sequence $[n_1, n_2, ..., n_L]$, encodes it via learnable agent embeddings as K/V, and performs MHA with the ball's intermediate representation as Q. Only the ball's trajectory representation is updated, and agent embeddings are added to each player to support social reasoning.
    - **Text Guidance**: Processes natural language descriptions with a frozen T5-Base encoder, projects the output, and performs MHA over all agents. Each agent prepends an agent embedding to its Query for differentiation.

3. **Hybrid Sampling Strategy**: At inference time, DDIM with step interval $\zeta=5$ is used to accelerate continuous trajectory sampling, while standard stochastic sampling is applied to discrete events. The discrete step count is $S^d = 10$ (vs. continuous $S = 50$), aligned via $s^d = \lceil s \cdot S^d / S \rceil$.

### Loss & Training

The joint training objective is a weighted sum of the simplified continuous loss and the exact variational discrete loss:

$$\mathcal{L}_{\mathrm{joint}} = \mathcal{L}_{\mathrm{simple}}^{\mathbf{Y}} + \lambda \mathcal{L}_{\mathrm{vb}}^{\mathbf{E}}$$

where $\lambda = 0.1$ balances the contribution of the two modalities. Importance sampling rather than uniform timestep sampling is employed. For controllable generation, Classifier-Free Guidance training is applied by randomly dropping conditioning signals with probability 25%.

## Key Experimental Results

### Main Results: Future Trajectory Generation (min / avg, 20 modes)

| Dataset | Metric | JointDiff | U2Diff (Prev. SOTA) | Gain |
|--------|------|-----------|----------|------|
| NFL | SADE↓ | **2.36/3.40** | 2.59/3.74 | -0.23/-0.34 |
| NFL | SFDE↓ | **5.53/8.40** | 5.97/9.02 | -0.44/-0.62 |
| Bundesliga | SADE↓ | **2.47/3.66** | 2.69/4.21 | -0.22/-0.55 |
| NBA | SADE↓ | **1.39/2.01** | 1.48/2.12 | -0.09/-0.11 |
| NBA | SFDE↓ | **2.53/3.95** | 2.68/4.14 | -0.15/-0.19 |

### Ablation Study: Effect of Joint Modeling (Controllable Generation Task)

| Configuration | NFL SADE↓ | NFL Acc↑ | Bundesliga SADE↓ | Bundesliga Acc↑ |
|------|-----------|----------|------------------|-----------------|
| w/o joint + w/o $\mathcal{G}$ | 2.42/3.57 | .76/.52 | 2.60/3.99 | .67/.44 |
| w/o joint + w $\mathcal{G}_{\text{WPG}}$ | 2.37/3.49 | .80/.59 | 2.20/3.07 | .73/.50 |
| JointDiff + w/o $\mathcal{G}$ | 2.36/3.40 | .78/.54 | 2.47/3.66 | .68/.39 |
| **JointDiff + w $\mathcal{G}_{\text{text}}$** | **2.19/3.09** | **.86/.74** | **2.08/2.72** | **.80/.59** |

### Key Findings

- Joint modeling (JointDiff) outperforms continuous-only variants on both controllable and uncontrollable tasks.
- Text guidance > weak possession guidance > no guidance; finer-grained guidance yields larger improvements.
- Multinomial diffusion achieves substantially better event-trajectory consistency than absorbing-state diffusion (e.g., Bundesliga avg Acc: 0.80 vs. 0.70).
- In human evaluation, JointDiff achieves an 80% win rate over MoFlow, with 24% of cases rated on par with ground-truth trajectories.
- Even under IID sampling, JointDiff remains competitive with non-IID methods on min metrics.

## Highlights & Insights

- This is the first work to apply joint continuous-discrete diffusion to temporally dynamic systems, filling a gap previously limited to static tasks (layout design, CAD).
- The WPG mode of CrossGuid is elegantly designed — providing only a list of player indices is sufficient to control game momentum, offering low entry barrier with high semantic expressiveness.
- The comparative analysis of multinomial diffusion vs. absorbing-state diffusion has broad reference value, demonstrating that a continuous correction mechanism outperforms one-shot decisions in temporal modeling.
- A unified sports benchmark (NFL + Bundesliga with text descriptions) is provided, facilitating future community research.

## Limitations & Future Work

- The model assumes a ball-possession event exists at every timestep (dense event regime); extension to sparse events (e.g., fouls, shots) is a direction for future work.
- Validation is currently limited to sports scenarios; broader adaptation to multi-agent systems such as autonomous driving and robotic collaboration remains to be explored.
- Discrete event categories are limited to ball possession ($N$ classes); extending to hierarchical discrete spaces with multiple event types requires further investigation.
- Text guidance relies on the T5 encoder, limiting comprehension of non-English descriptions or complex tactical language.

## Related Work & Insights

- U2Diff serves as the primary continuous trajectory baseline; JointDiff extends its Social-Temporal Block architecture with joint modeling capability.
- Levi et al. (2023) and Li et al. (2025) apply joint diffusion to static layout and vision-language settings; JointDiff generalizes this to dynamic temporal scenarios.
- The CrossGuid design can be adapted to other tasks requiring conditional injection into structured multi-agent embeddings.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First joint continuous-discrete diffusion for dynamic multi-agent systems; the WPG task formulation is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three datasets, multiple tasks, human evaluation, and consistency analysis — comprehensive and thorough.
- Writing Quality: ⭐⭐⭐⭐ Method is clearly presented with complete mathematical derivations and intuitive figures.
- Value: ⭐⭐⭐⭐ Significant contribution to multi-agent generation and sports analytics; the joint diffusion paradigm is broadly generalizable.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Multi-agent Coordination via Flow Matching](multi-agent_coordination_via_flow_matching.md)
- [\[ICLR 2026\] MAC-AMP: A Closed-Loop Multi-Agent Collaboration System for Multi-Objective Antimicrobial Peptide Design](mac-amp_a_closed-loop_multi-agent_collaboration_system_for_multi-objective_antim.md)
- [\[CVPR 2026\] coDrawAgents: A Multi-Agent Dialogue Framework for Compositional Image Generation](../../CVPR2026/image_generation/codrawagents_a_multi-agent_dialogue_framework_for_compositional_image_generation.md)
- [\[ICLR 2026\] Bridging Degradation Discrimination and Generation for Universal Image Restoration](bridging_degradation_discrimination_and_generation_for_universal_image_restorati.md)
- [\[ICLR 2026\] Discrete Adjoint Matching](discrete_adjoint_matching.md)

<!-- RELATED:END -->
