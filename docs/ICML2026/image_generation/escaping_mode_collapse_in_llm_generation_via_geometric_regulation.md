---
title: >-
  [Paper Note] Escaping Mode Collapse in LLM Generation via Geometric Regulation
description: >-
  [ICML 2026][Image Generation][Mode Collapse] This paper reinterprets "mode collapse" (repetition, cycling, and monotony) in long-form LLM generation as "geometric collapse" of hidden state trajectories within the represe…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "Mode Collapse"
  - "Geometric Collapse"
  - "Correlation Dimension"
  - "KV Cache Intervention"
  - "Low-rank Damping"
date: 2026-05-08
content_hash: ad9d72967be7a701
---

# Escaping Mode Collapse in LLM Generation via Geometric Regulation

**Conference**: ICML 2026  
**arXiv**: [2605.00435](https://arxiv.org/abs/2605.00435)  
**Code**: None  
**Area**: LLM Generation / Dynamical Systems / Decoding Control  
**Keywords**: Mode Collapse, Geometric Collapse, Correlation Dimension, KV Cache Intervention, Low-rank Damping

## TL;DR
This paper reinterprets "mode collapse" (repetition, cycling, and monotony) in long-form LLM generation as "geometric collapse" of hidden state trajectories within the representation space from a dynamical systems perspective. It proposes RMR—a method that applies lightweight low-rank damping to the Transformer value cache to suppress the most persistent self-reinforcing directions, maintaining stable, high-quality generation even in extremely low-entropy decoding regimes (0.8 nats/step).

## Background & Motivation

**Background**: Failures in long-form decoding (repetition, loops, and monotonization) are chronic challenges for LLM deployment. Current mitigation methods are primarily "token-level," such as top-k/top-p, temperature sampling, repetition penalties, and locally typical sampling, all of which modify the probability distribution of the next token.

**Limitations of Prior Work**: These approaches are essentially "local, symbolic-level" patches. In low-temperature or low-entropy scenarios (e.g., temperature 0.5, entropy target 1.0), models still tend to get trapped in loops with high probability. Token-level heuristics only suppress symptoms without explaining why cycles emerge systematically or providing controllable knobs for long-range dynamics.

**Key Challenge**: Mode collapse is not caused by "incorrect probability for a single token" but rather by "the entire generation process sliding down a narrow path." Using "token-wise/local" tools to solve an inherently "trajectory-based/long-range" problem is naturally inefficient.

**Goal**: (1) Establish a geometric metric capable of directly characterizing long-range collapse; (2) design a lightweight method to intervene directly in internal states without altering the probability distribution.

**Key Insight**: Autoregressive decoding is viewed as a random trajectory in a high-dimensional state space (where states are KV caches or next-token log-prob vectors). Mode collapse corresponds to the trajectory being trapped in a low-dimensional "quasi-attractor," i.e., "reachability collapse in the state space."

**Core Idea**: Quantify "reachability" using the correlation dimension $d_t$. When a strong self-reinforcing low-rank direction is detected (analogous to the order parameter in Ising model phase transitions), low-rank damping is applied to the value cache to slightly attenuate these directions, thereby restoring the full-space exploration capability of the trajectory.

## Method

### Overall Architecture
The framework consists of two layers. The first layer is **Diagnosis**: The authors use a 2D state-dependent IFS (Iterated Function System) as a minimal dynamical model to prove that once the "inverse temperature $\beta$" crosses a critical threshold $\beta_0$, the system splits from a single ergodic invariant measure into two stable basins of attraction—the geometric equivalent of mode collapse. Then, the "finite-time correlation dimension" $d_t$ (based on the scaling law $C_t(\varepsilon) \propto \varepsilon^d$) is used for online measurement in real LLM decoding, using the sequence of next-token log-prob vectors as input. Experiments show that $d_t$ drops significantly before cycles appear and is more robust than token-level entropy or Distinct-n.

The second layer is **Intervention** (RMR - Reinforced Mode Regulation): During decoding forward pass intervals, a generalized eigenvalue problem with a bounded spectrum is solved on the recent segment of the value cache to identify "abnormally persistent" low-rank subspaces. A low-rank damping update is then applied to the value cache, which is a high-dimensional generalization of the $(1-\eta)$ contraction applied to the historical mean $m_t$ in the minimal model. This process does not alter softmax probabilities or logits; it is a pure state-space intervention.

### Key Designs

1.  **Correlation Dimension as a "Geometric Collapse" Probe**:
    - **Function**: Real-time estimation of the effective dimension of internal trajectories during decoding, acting as an early warning and evaluation metric for mode collapse.
    - **Mechanism**: Calculates the correlation sum $C_t(\varepsilon) = \frac{2}{t(t-1)} \sum_{i<j} \mathbf{1}(\|x_i - x_j\| < \varepsilon)$ for the trajectory $\{x_t\}$, and derives $d_t$ from the slope in a log-log plot against $\varepsilon$. The authors optimized the $O(t^2)$ naive algorithm into an $O(t)$ online update: $C_{t+1}(\varepsilon) = \frac{t-1}{t+1} C_t(\varepsilon) + \frac{2}{t(t+1)} \sum_i \mathbf{1}(\|x_i - x_{t+1}\| < \varepsilon)$.
    - **Design Motivation**: Traditional entropy or Distinct-n are "token-level" stochastic variables with high variance across single trajectories, making thresholds difficult to define. Correlation dimension is a "trajectory-level" geometric invariant that directly captures the "trapped trajectory" phenomenon and aligns naturally with intervention goals.

2.  **Persistent Direction Detection (Bounded-Spectrum Generalized Eigenvalue Problem)**:
    - **Function**: Locates the few "most self-reinforcing and slowest-decaying" low-rank directions within high-dimensional value caches.
    - **Mechanism**: Constructs two covariance-like matrices (instantaneous vs. historical average) on a sliding window of the value cache matrix and solves for generalized eigenvectors. To avoid numerical instability, a bounded spectrum form $\lambda \in [0,1]$ is used to represent "persistence intensity." Principled thresholding is applied to select only the most significant directions far from the background spectrum, avoiding damage to normal semantic directions.
    - **Design Motivation**: Applying damping to all dimensions would degrade language quality. Suppressing only the "most persistent" directions breaks loop traps with minimal disruption, corresponding to the insight from the minimal model where a "weak damping" of $\eta = 10^{-4}$ is sufficient to restore reachability.

3.  **Value Cache Low-Rank Damping Update (RMR)**:
    - **Function**: Subtracts a small portion of the selected directions in low-rank form from the value cache as an inference-time intervention.
    - **Mechanism**: Constructs a low-rank projection $P = \sum_i u_i u_i^\top$ and performs a low-rank update on the value cache $V \leftarrow V - \eta \, V P$, which is equivalent to $m_t \leftarrow (1-\eta)m_t$ in the high-dimensional minimal model. The operation introduces only one additional small matrix multiplication, with overhead comparable to or lower than a single attention operation.
    - **Design Motivation**: As a state intervention on the value cache, it does not affect the analytical form of the token probability distribution. It is orthogonal to and can be used with any sampler (top-p, temperature, contrastive decoding), making it deployment-friendly.

### Loss & Training
RMR is an inference-time method that requires **no training**, no fine-tuning, and no reward model. The only two hyperparameters are $\eta$ and the target low-rank $r$. The authors suggest $\eta \in [10^{-3}, 10^{-2}]$ and $r \in \{2, 4, 8\}$ work for most models.

## Key Experimental Results

### Main Results
The authors tested multiple open-source LLMs (including Qwen3-4B-Base) using "temperature-locked" and "entropy-locked" decoding protocols. The core metric is the "non-collapse rate" (the proportion of samples that do not trigger explicit loops during long generation).

| Decoding Setting | Baseline non-collapse | RMR non-collapse | Remarks |
| :--- | :--- | :--- | :--- |
| Temperature = 0.7 | 8% | **56%** | Substantial improvement |
| Entropy target = 1.0 nats/step | 5% | **33%** | Baseline nearly collapses in low-entropy zones |
| Entropy target ≈ 2.0 nats/step | Near saturation | Near saturation | Gap narrows at high entropy |
| Entropy target = 0.8 nats/step | Near 0 | **Still usable** | RMR opens a new usable low-entropy regime |

### Ablation Study

| Configuration | Non-collapse Performance | Explanation |
| :--- | :--- | :--- |
| RMR (Full) | Significant recovery | Detection + Low-rank damping |
| Detection only (No damping) | Equivalent to baseline | Verifies that intervention is essential |
| Full-dimension damping (Non-low-rank) | Quality degradation | Highlights the value of the "minimal necessary intervention" principle |
| Token-level repetition penalty | Limited improvement | Verifies the failure of symbolic methods in low-temperature zones |

### Key Findings
- The correlation dimension $d_t$ drops significantly **before** explicit loops appear, serving as an early warning signal that is much more sensitive than entropy or Distinct-n.
- "Persistent directions" are extremely sparse (usually $< 8$ dimensions), confirming the intuition that the "order parameter is low-dimensional" and explaining why low-rank damping is sufficient.
- RMR extends the usable decoding regime from ~2.0 nats/step down to ~0.8 nats/step, effectively unlocking a "high-certainty + high-diversity" operational zone previously unusable due to cycling.

## Highlights & Insights
- **Interdisciplinary Analogy**: Connecting LLM decoding with non-equilibrium statistical physics (Ising phase transitions, slow variables, self-organization). The correspondence between correlation dimension and the order parameter is elegant; this "trajectory geometry" perspective is closer to the essence of the problem than token probabilities.
- **Diagnosis-Intervention Loop**: The system qualitatively identifies "reachability collapse" via correlation dimension and then solves it directionally with low-rank damping. The entire pipeline is self-consistent and derived from theory rather than trial-and-error.
- **Transferable Trick**: The path of "low-cost low-rank intervention on the value cache" may be applicable to other long-range issues such as hallucination drift, Chain-of-Thought collapse, or agent tool-call loops—all of which involve "trajectory traps" in high-dimensional latent space.

## Limitations & Future Work
- Experiments focus on open-ended text generation and the Qwen3 series; coverage of structured tasks like reasoning, agents, or code is limited. The assumption that "most persistent direction = unwanted direction" might not hold in structured tasks.
- Correlation dimension estimation is sensitive to window length. The provided online algorithm still relies on empirical thresholds $\varepsilon_0, \varepsilon_1$; automated threshold selection is a potential improvement.
- RMR is currently a "post-hoc intervention." Feeding the persistent direction detection signal back into training objectives (e.g., adding a geometric term to RLHF rewards) is an obvious next step.

## Related Work & Insights
- **vs. Locally Typical Sampling / top-p**: These modify probabilities, while Ours modifies states; they are orthogonal and can be used together.
- **vs. Activation Steering (Zou 2023 / Turner 2023)**: Both intervene in the cache, but RMR's directions are derived from "temporal persistence" rather than task vectors, with the goal of stabilizing dynamics rather than controlling semantics.
- **vs. Existing Repetition Penalties**: Fundamentally avoids the need for engineering patches like "N-gram history windows," providing a more universal mechanism.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Redefines mode collapse using dynamical systems/phase transition language; provides computable geometric quantities and corresponding interventions.
- Experimental Thoroughness: ⭐⭐⭐⭐ Complete comparisons across multiple models and protocols, though reasoning/agent long-range tasks are not explored.
- Writing Quality: ⭐⭐⭐⭐ Clear theoretical narrative, transitioning smoothly from minimal models to real LLM interventions.
- Value: ⭐⭐⭐⭐ Provides a nearly cost-free new interval for low-entropy decoding with minimal deployment friction; significant engineering value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Taming Preference Mode Collapse via Directional Decoupling Alignment in Diffusion Reinforcement Learning](../../CVPR2026/image_generation/taming_preference_mode_collapse_via_directional_decoupling_alignment_in_diffusio.md)
- [\[ICML 2026\] Information-Geometric Adaptive Sampling for Graph Diffusion](information-geometric_adaptive_sampling_for_graph_diffusion.md)
- [\[ICML 2026\] Quantifying Error Propagation and Model Collapse in Diffusion Models](quantifying_error_propagation_and_model_collapse_in_diffusion_models.md)
- [\[ICML 2026\] Restoring Initial Noise Sensitivity in Text-to-Image Distillation via Geometric Alignment](restoring_initial_noise_sensitivity_in_text-to-image_distillation_via_geometric_.md)
- [\[ICLR 2026\] Steer Away From Mode Collisions: Improving Composition In Diffusion Models](../../ICLR2026/image_generation/steer_away_from_mode_collisions_improving_composition_in_diffusion_models.md)

</div>

<!-- RELATED:END -->
