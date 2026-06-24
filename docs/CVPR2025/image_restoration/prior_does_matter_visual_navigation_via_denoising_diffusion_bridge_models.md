---
title: >-
  [Paper Note] Prior Does Matter: Visual Navigation via Denoising Diffusion Bridge Models
description: >-
  [CVPR 2025][Image Restoration][Denoising Diffusion Bridge Models] NaviBridger introduces Denoising Diffusion Bridge Models (DDBM) to visual navigation tasks, replacing Gaussian noise with information-rich prior actions as the denoising starting point. It theoretically proves that a source distribution closer to the target distribution yields a lower error upper bound, and designs three prior strategies (Gaussian, rule-based, and learning-based) to accelerate inference and sur…
tags:
  - "CVPR 2025"
  - "Image Restoration"
  - "Denoising Diffusion Bridge Models"
  - "Visual Navigation"
  - "Imitation Learning"
  - "Prior Action"
  - "Denoising"
date: 2026-05-08
content_hash: 281a8f241cfe2ae9
---

# Prior Does Matter: Visual Navigation via Denoising Diffusion Bridge Models

**Conference**: CVPR 2025  
**arXiv**: [2504.10041](https://arxiv.org/abs/2504.10041)  
**Code**: [https://github.com/hren20/NaiviBridger](https://github.com/hren20/NaiviBridger)  
**Area**: Image Restoration  
**Keywords**: Denoising Diffusion Bridge Models, Visual Navigation, Imitation Learning, Prior Action, Denoising

## TL;DR

NaviBridger introduces Denoising Diffusion Bridge Models (DDBM) to visual navigation tasks, replacing Gaussian noise with information-rich prior actions as the denoising starting point. It theoretically proves that a source distribution closer to the target distribution yields a lower error upper bound, and designs three prior strategies (Gaussian, rule-based, and learning-based) to accelerate inference and surpass baselines in both indoor/outdoor simulations and real-world scenarios.

## Background & Motivation

**Background**: Local path planning in visual navigation has begun utilizing diffusion models for imitation learning (such as NoMaD) to learn generated action sequences from expert demonstration data. These methods excel in modeling multi-modal distributions and capturing sequential correlations.

**Limitations of Prior Work**: Standard diffusion models (such as DDPM) initiate denoising from pure Gaussian noise, but the target action distribution diverges significantly from Gaussian noise, leading to: (1) a large number of denoising steps being wasted on constraining the noise within the action space, while truly task-relevant refinement only occurs in the final few steps; (2) the distribution of effective actions being sparse, making it difficult to generate accurate actions from chaotic random noise, especially when lacking guidance. Together, these two issues increase computed costs and degrade performance.

**Key Challenge**: The contradiction between the powerful generation capability of diffusion models and their fixed Gaussian initialization—the powerful model is dragged down by inefficient initialization.

**Goal**: Construct an appropriate initial distribution to transform the diffusion model from "generation from noise" into "distribution transfer starting from a meaningful prior", thereby reducing denoising steps and improving action quality.

**Key Insight**: Inspired by optimal transport theory and the success of diffusion bridge methods (e.g., DDBM) in image translation/restoration, the authors believe this idea can be transferred to robotic learning—with the key challenge being that visual navigation has no naturally paired source distributions.

**Core Idea**: Replace standard diffusion models with DDBM and design three prior action generation strategies (Gaussian, rule-based, and learning-based) to start the denoising process from a distribution close to the target, reducing required denoising steps and boosting accuracy.

## Method

### Overall Architecture

The overall architecture of NaviBridger consists of three components: (1) Feature Extraction Module—uses a Transformer encoder to process current and historical visual observation sequences and target images, generating a context vector $c_t$; (2) Prior Action Generation Module—generates source distribution actions $a_s$ based on the context vector; (3) Denoising Diffusion Bridge Module—uses a 1D temporal CNN with FiLM-modulated condition vectors to transform prior actions into target actions $a_0$ via the reverse process of DDBM.

### Key Designs

1. **Adaptation of Denoising Diffusion Bridge (DDBM) in Imitation Learning**:

    - **Function**: Achieves distribution transfer from an arbitrary source distribution to the target action distribution, rather than starting from a fixed Gaussian distribution.
    - **Mechanism**: The reverse SDE of DDBM contains both a score function $s$ and Doob's h-transform $h$; the former guides the denoising direction, while the latter ensures the trajectory reaches the target distribution at terminal time $T$. The sampling distribution $q(a_t|a_0, a_T)$ is conditional Gaussian, where its mean mixes the target and source actions, and the variance scales proportionally to the SNR.
    - **Design Motivation**: Standard diffusion models can only map complex data distributions to Gaussian distributions. DDBM breaks this limitation through Doob's h-transform, allowing a bridge to be established between any two distributions. This enables a drastic reduction in denoising steps when a good prior is available.

2. **Three Prior Action Generation Strategies**:

    - **Function**: Provides prior options ranging from uninformative to informative based on different scenarios and available information.
    - **Mechanism**:
        - **Gaussian Prior**: White noise identical to standard diffusion, serving as an uninformative baseline to ensure backward compatibility.
        - **Rule-based Prior**: Uses an FC layer to predict path length and motion behavior classification (straight, left turn, right turn, U-turn), and generates a parabolic prior path accordingly. The prior actions are parameterized curves handcrafted based on the predicted action category and distance.
        - **Learning-based Prior**: Uses a lightweight CVAE to generate prior actions directly from observations. The CVAE encoder takes observations + actions to learn the posterior distribution, and the decoder samples from the prior distribution to generate actions.
    - **Design Motivation**: Theoretical analysis shows that the error upper bound $E[||a_t - a_0||^2] \leq C \cdot D_{KL}(\pi_s || \pi)$, meaning that the closer the source distribution is to the target, the lower the error upper bound. The learning-based prior is closest to the target distribution but requires an extra model; the rule-based prior does not require learning but calls for domain knowledge; the Gaussian prior is the fallback.

3. **FiLM Conditional Modulation Denoising Network**:

    - **Function**: Injects visual observation information into the denoising process to condition action generation on the current scene.
    - **Mechanism**: Uses a 1D temporal CNN as the denoising network, applying the context vector $c_t$ to intermediate feature layers via Feature-wise Linear Modulation (FiLM) for $k$ iteration denoising.
    - **Design Motivation**: A 1D CNN is suitable for processing sequential action series, while FiLM is a lightweight and effective way of injecting conditions, and it shares the same structure as baselines like NoMaD to facilitate fair comparison.

### Loss & Training

The total loss is $\mathcal{L} = \lambda_b \mathcal{L}_b + \lambda_p \mathcal{L}_p + \lambda_d \mathcal{L}_d$:

- **Diffusion Bridge Loss** $\mathcal{L}_b$: Weighted MSE, predicting the distance between target action $a_0$ and the ground truth.
- **Prior Action Loss** $\mathcal{L}_p$: Use cross-entropy (classification) + MSE (regression) for rule-based priors; use MSE + KL divergence (CVAE) for learning-based priors.
- **Temporal Distance Loss** $\mathcal{L}_d$: Predicts the temporal interval between the target image and the current image, which is used for high-level planning.

Train for 30 epochs, with a learning rate of 0.0001 and a batch size of 256, taking about 30 hours on a single RTX TITAN card. DDBM uses the VE model, with $\sigma_0 = \sigma_T = 0.5$, and defaults to $k=10$ sampling steps.

## Key Experimental Results

### Main Results

| Scenario | Method | Base Task Success Rate | Adaptation Task Success Rate | Collisions↓ |
|------|------|---------------|---------------|----------|
| Indoor | ViNT | 68% | 28% | 1.02/1.58 |
| Indoor | NoMaD (DDPM) | 86% | 32% | 0.74/1.32 |
| Indoor | NaviBridger-Gaussian | 82% | 64% | 0.72/0.98 |
| Indoor | NaviBridger-Learning | **92%** | **88%** | **0.61/0.41** |
| Outdoor | NoMaD (DDPM) | 22% | 52% | 0.58/0.34 |
| Outdoor | NaviBridger-Learning | **44%** | **64%** | **0.51/0.30** |

### Ablation Study

| Configuration (k=denoising steps) | DDPM Avg Rank | NaviBridger-Gaussian Avg Rank |
|-------------------|---------------|------------------------------|
| k=10 | 7.0 | 2.4 |
| k=7 | 6.0 | 2.6 |
| k=4 | 4.4 | 2.4 |
| k=1 | 7.8 | 3.2 |

### Key Findings

- The learning-based prior performs best across all scenarios and tasks, with a particularly remarkable boost in adaptation tasks (environmental changes) (indoor 32% $\rightarrow$ 88%).
- The rule-based prior is effective in specific scenarios but generalizes poorly, performing even lower than the baseline under certain settings (outdoor base task 14%)—indicating that an unsuitable prior is worse than no prior.
- NaviBridger under the Gaussian prior still outperforms DDPM-based NoMaD—due to the higher denoising efficiency of the DDBM framework itself.
- When reducing denoising steps, the performance of NaviBridger barely degrades (rank remains stable around 2.4-3.2), whereas DDPM deteriorates drastically at $k=1$—proving that convergence occurs in fewer steps.
- Visualization demonstrates that the DDBM method reaches stable actions within 2-4 steps, whereas most steps of DDPM are spent constraining noise into the action space.

## Highlights & Insights

- **Design Driven by Theoretical Analysis**: Instead of arbitrary prior selection, the necessity of a good prior is derived from the KL divergence error upper bound. The bound $E[||a_t - a_0||^2] \leq C \cdot D_{KL}(\pi_s || \pi)$ is concise yet powerful, providing theoretical guidance for future prior designs.
- **Generality of the Framework**: NaviBridger is applicable not only to visual navigation but can also be directly transferred to other imitation learning tasks (manipulation, motion planning) by simply replacing the prior strategy.
- **Practical Value of Few-step Inference**: In real-time robotic systems (deployed on Jetson Orin AGX), few-step denoising translates to lower latency, which is crucial for safety-critical navigation.

## Limitations & Future Work

- The learning-based prior requires additional CVAE training, increasing system complexity; if the prior model itself is poor, it may introduce bias.
- The rule-based prior classifies navigation behaviors too coarsely (5 classes), yielding limited applicability in complex environments.
- Validated only on point-goal and image-goal navigation, lacking testing in semantic navigation or manipulation tasks.
- Integration with newer vision foundation models (such as VLA) remains unexplored.

## Related Work & Insights

- **vs NoMaD**: NoMaD is the first method to apply policy diffusion to visual navigation, but it uses standard DDPM to denoise from Gaussian noise. The key improvement of NaviBridger is replacing the denoising framework with DDBM, allowing a start from a better baseline. Under the identical training data and network structure, merely changing the denoising framework yields significant improvements.
- **vs ViNT**: ViNT is a regression-based method (CNN + self-attention) that does not use diffusion models. NaviBridger offers advantages in generation quality and multi-modal handling.
- **vs DiffusionPolicy**: Diffusion Policy performs excellently in manipulation tasks but is also limited by starting from Gaussian noise. The DDBM concept in NaviBridger can be directly transferred to Diffusion Policy.

## Rating

- Novelty: ⭐⭐⭐⭐ Applies diffusion bridge models to visual navigation for the first time, with insightful theoretical analysis.
- Experimental Thoroughness: ⭐⭐⭐⭐ Indoor and outdoor simulation + real robot validation, comparing three priors along with ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Formal derivations are clear, though the paper is somewhat lengthy.
- Value: ⭐⭐⭐⭐ Inspiring for all works using diffusion models for decision-making.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] ε-VAE: Denoising as Visual Decoding](../../ICML2025/image_restoration/epsilon-vae_denoising_as_visual_decoding.md)
- [\[CVPR 2025\] Visual-Instructed Degradation Diffusion for All-in-One Image Restoration](visual-instructed_degradation_diffusion_for_all-in-one_image_restoration.md)
- [\[NeurIPS 2025\] Audio Super-Resolution with Latent Bridge Models](../../NeurIPS2025/image_restoration/audio_super-resolution_with_latent_bridge_models.md)
- [\[ICLR 2026\] Energy-oriented Diffusion Bridge for Image Restoration with Foundational Diffusion Models](../../ICLR2026/image_restoration/energy-oriented_diffusion_bridge_for_image_restoration_with_foundational_diffusi.md)
- [\[CVPR 2025\] Efficient Visual State Space Model for Image Deblurring](efficient_visual_state_space_model_for_image_deblurring.md)

</div>

<!-- RELATED:END -->
