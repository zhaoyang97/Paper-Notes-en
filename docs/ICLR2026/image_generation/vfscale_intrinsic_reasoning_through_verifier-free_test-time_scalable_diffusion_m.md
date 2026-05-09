---
title: >-
  [Paper Note] VFScale: Intrinsic Reasoning through Verifier-Free Test-time Scalable Diffusion Model
description: >-
  [ICLR 2026][Image Generation][Test-time scaling] VFScale proposes a test-time scalable diffusion model that requires no external verifier. By introducing an MRNCL loss and KL regularization to improve the energy landscape, the model's intrinsic energy function serves as a verifier. Combined with hybrid MCTS denoising for efficient search, a model trained on 6×6 mazes achieves 88% success on 15×15 mazes, where standard diffusion models fail entirely.
tags:
  - ICLR 2026
  - Image Generation
  - Test-time scaling
  - verifier-free
  - energy function
  - Monte Carlo tree search
  - diffusion model reasoning
date: 2026-05-08
content_hash: ace4f1a42c694dbb
---

# VFScale: Intrinsic Reasoning through Verifier-Free Test-time Scalable Diffusion Model

**Conference**: ICLR 2026
**arXiv**: [2502.01989](https://arxiv.org/abs/2502.01989)
**Code**: [https://github.com/AI4Science-WestlakeU/VFScale](https://github.com/AI4Science-WestlakeU/VFScale)
**Area**: Diffusion Models / Reasoning
**Keywords**: Test-time scaling, verifier-free, energy function, Monte Carlo tree search, diffusion model reasoning

## TL;DR
VFScale proposes a test-time scalable diffusion model that requires no external verifier. By introducing an MRNCL loss and KL regularization to improve the energy landscape, the model's intrinsic energy function serves as a verifier. Combined with hybrid MCTS denoising for efficient search, a model trained on 6×6 mazes achieves 88% success on 15×15 mazes, where standard diffusion models fail entirely.

## Background & Motivation

**State of the Field**: Inspired by human System 2 thinking, LLMs achieve strong performance on complex reasoning tasks via Chain-of-Thought. Diffusion models, with their iterative refinement process, are also well-suited for reasoning tasks, but their performance degrades sharply when problem difficulty exceeds the training distribution.

**Limitations of Prior Work**: (1) Simply increasing the number of sampling steps quickly saturates (Du et al. 2024); (2) test-time scaling via increased sample count relies on external verifiers for dense scoring signals, which are difficult to obtain for reasoning tasks; (3) humans can perform introspective reasoning without external feedback, a capability that existing methods lack.

**Root Cause**: A diffusion model's energy function can inherently serve as a verifier (since the score function is the negative gradient of the energy), but existing energy landscapes are of insufficient quality — low energy does not necessarily correspond to high-quality solutions (poor performance–energy consistency).

**Paper Goals**: How can the diffusion model's intrinsic energy function replace an external verifier to enable verifier-free test-time scaling?

**Starting Point**: A two-pronged approach — improving the energy landscape on the training side, and improving search efficiency on the inference side.

**Core Idea**: Aligning the monotonic relationship between energy values and sample quality via the MRNCL loss, and balancing exploration and exploitation during denoising via hMCTS.

## Method

### Overall Architecture
**Training side**: On top of the standard MSE and contrastive losses, an MRNCL loss (aligning the monotonic relationship between energy and quality) and KL regularization (smoothing the energy landscape) are added. **Inference side**: Hybrid MCTS denoising — broad exploration via BoN in early steps and deep exploitation via MCTS in later steps.

### Key Designs

1. **MRNCL Loss (Monotonic-Regression Negative Contrastive Learning)**:

    - **Function**: Ensures that samples farther from the ground truth have higher energy (performance–energy consistency).
    - **Mechanism**: For each positive sample $x_0$, two negative samples $x_0^-$ and $x_0^{--}$ are generated (the latter being farther from the positive sample). After adding noise, three energy-distance pairs $(0, E_t^+)$, $(l_{2,0}^-, E_t^-)$, $(l_{2,0}^{--}, E_t^{--})$ are obtained and used in a linear regression to estimate slope $k_t$ and intercept $b_t$.
    - **Loss**: $\mathcal{L}_{\text{MRNCL}} = \mathbb{E}[\max(0, \gamma - k_t) + \sum \|E - \hat{E}\|_2^2]$
    - **Design Motivation**: The original contrastive loss only requires positive samples to be local energy minima, without constraining the energy ordering among negative samples.

2. **KL Regularization**:

    - $\mathcal{L}_{\text{KL}} = \mathbb{E}_{t, p_{\theta,t}}[E_{\text{stop-grad}(\theta)}(x)] + \mathbb{E}_{t, p_{\theta,t}}[\log p_{\theta,t}(x)]$
    - The first term encourages samples to have low energy; the second maximizes sampling diversity (entropy maximization).
    - Applied at every denoising step $t$, unlike Du et al. 2021 who apply it only at the terminal step.

3. **Hybrid MCTS Denoising (hMCTS)**:

    - **Early stages** (high noise): BoN is used — $L$ initial noise samples are denoised in parallel to prevent premature elimination of promising trajectories.
    - **Later stages** (low noise): MCTS is used:
        - **Selection**: UCB formula $\text{UCB}(x_t, a_t) = Q(x_t, a_t) + c\sqrt{\frac{\ln N_i}{n_i}}$
        - **Expansion**: Single-step denoising with different Gaussian noise samples → $K$ branches.
        - **Simulation**: Fast sampling to $x_0$ via DDIM; $E_\theta(\hat{x}_0)$ is used as the reward (no external verifier required).
        - **Backpropagation**: Values of all nodes along the path are updated.
    - DDIM's subsequence sampling property makes simulation efficient.

### Complete Training Objective
$$\mathcal{L} = \mathcal{L}_{\text{MSE}} + \mathcal{L}_{\text{Contrast}} + \mathcal{L}_{\text{MRNCL}} + \mathcal{L}_{\text{KL}}$$

## Key Experimental Results

### Base Generalization (N=1 Inference)

| Method | Maze 6×6 | Maze 10×10 | Maze 15×15 | Sudoku D=33 | Sudoku D=25 |
|--------|----------|------------|------------|-------------|-------------|
| Original | 1.000 | 0.578 | 0.063 | 0.320 | 0.023 |
| VFScale tr. | 1.000 | 0.775 | 0.281 | 0.195 | 0.008 |

### Test-time Scaling (Maze 15×15)

| Method | N=1 | N=11 | N=41 | N=161 |
|--------|-----|------|------|-------|
| Original BoN (Energy) | 0.063 | 0.047 | 0.078 | 0.109 |
| Original BoN (GT) | 0.063 | 0.125 | 0.133 | 0.172 |
| VFScale tr. BoN (GT) | 0.250 | 0.508 | 0.656 | 0.742 |
| **VFScale tr. hMCTS** | **0.281** | — | — | **0.880** |

### Key Findings
1. **Test-time scaling completely fails with the original training**: Even with a ground-truth verifier guiding BoN, the success rate on Maze 15×15 only increases from 6% to 17%.
2. **Energy landscape quality is the bottleneck**: The original model achieves only ~70% performance–energy consistency.
3. **VFScale training substantially improves scalability**: Under the same BoN budget, GT-guided success rate improves from 17% to 74%.
4. **hMCTS further unlocks scaling potential**: Final success rate reaches 88% (trained on 6×6, tested on 15×15).
5. **MRNCL and KL regularization are complementary**: Removing either degrades performance.

## Highlights & Insights
- **Paradigm innovation**: The diffusion model's intrinsic energy function is used as a verifier, genuinely realizing "introspective reasoning without external feedback."
- **Deep insight behind MRNCL**: Contrastive learning constrains positive–negative relationships but ignores the ordering among negative samples — this is the root cause of poor energy landscape quality.
- **Elegant design of hMCTS**: Broad search with BoN in early stages and deep search with MCTS in later stages perfectly matches the denoising process as noise decreases over time.
- **Remarkable generalization**: Training on 6×6 mazes and achieving 88% success on 15×15 mazes demonstrates the true potential of test-time scaling.

## Limitations & Future Work
- The computational overhead of MCTS grows with the number of branches $K$ and rollout count $N_r$, requiring careful balancing.
- Validation is currently limited to structured reasoning tasks such as mazes and Sudoku; more complex settings such as language reasoning remain to be explored.
- The choice of linear regression in MRNCL may not be the optimal monotonic constraint formulation.
- Adaptive switching points between the BoN and MCTS phases are worth investigating.

## Related Work & Insights
- **vs. Du et al. 2024**: Their energy-based diffusion model saturates during test-time scaling; VFScale addresses the root cause.
- **vs. Ma et al. 2025**: Their approach depends on external verifiers for sample-count scaling, whereas VFScale fully internalizes this process.
- **vs. AlphaGo/AlphaZero**: The core ideas of MCTS are borrowed and adapted to the diffusion denoising process.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — The concept of verifier-free test-time scaling, MRNCL, and hMCTS are all novel contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Maze and Sudoku tasks provide solid validation, though task diversity is limited.
- Writing Quality: ⭐⭐⭐⭐⭐ — The logical flow from motivation to analysis to solution is clear and well-structured.
- Value: ⭐⭐⭐⭐⭐ — Opens a new direction for reasoning capability and test-time scaling in diffusion models.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Test-Time Iterative Error Correction for Efficient Diffusion Models](test-time_iterative_error_correction_for_efficient_diffusion_models.md)
- [\[ICLR 2026\] Compose Your Policies! Improving Diffusion-based or Flow-based Robot Policies via Test-time Distribution-level Composition](compose_your_policies_improving_diffusion-based_or_flow-based_robot_policies_via.md)
- [\[ICLR 2026\] Verifier-Constrained Flow Expansion for Discovery Beyond the Data](verifier-constrained_flow_expansion_for_discovery_beyond_the_data.md)
- [\[ICLR 2026\] SPEED: Scalable, Precise, and Efficient Concept Erasure for Diffusion Models](speed_scalable_precise_and_efficient_concept_erasure_for_diffusion_models.md)
- [\[ICLR 2026\] Beyond Confidence: The Rhythms of Reasoning in Generative Models](beyond_confidence_the_rhythms_of_reasoning_in_generative_models.md)

<!-- RELATED:END -->
