---
title: >-
  [Paper Note] SLAP: The Semantic Least Action Principle for Variational Video-Language Modeling
description: >-
  [ICML 2026][Video Understanding][Video-Language Models] SLAP applies the "Principle of Least Action" from classical mechanics to the semantic video manifold. It models the completion of missing frames in sparsely sampled…
tags:
  - "ICML 2026"
  - "Video Understanding"
  - "Video-Language Models"
  - "temporal interpolation"
  - "least action principle"
  - "variational methods"
  - "object permanence"
date: 2026-05-08
content_hash: ce1f0420fea01858
---

# SLAP: The Semantic Least Action Principle for Variational Video-Language Modeling

**Conference**: ICML 2026  
**arXiv**: [2605.30750](https://arxiv.org/abs/2605.30750)  
**Code**: To be confirmed  
**Area**: Multi-modal VLM / Long Video Understanding  
**Keywords**: Video-Language Models, temporal interpolation, least action principle, variational methods, object permanence

## TL;DR
SLAP applies the "Principle of Least Action" from classical mechanics to the semantic video manifold. It models the completion of missing frames in sparsely sampled videos as a two-point boundary value problem on a Riemannian manifold—replacing probabilistic generation with semantic dynamics to enforce object permanence. It achieves 83.9% accuracy on the tunnel occlusion test (surpassing diffusion models by 12 points) with a 177× inference speedup.

## Background & Motivation

**Background**: Contemporary Large Video-Language Models (LVLMs) such as Video-LLaMA and LLaVA-Video are proficient in static scene question answering. However, due to the $O(n^2)$ complexity of self-attention in long videos, they require aggressive sparse sampling (typically < 0.5 fps), which leaves substantial temporal "blind spots" for the model.

**Limitations of Prior Work**: These blind spots lead to two primary failure modes:
- **Implicit Pooling** (mean-pool / Q-Former): Directly compressing frame sequences into a single token destroys the temporal causal structure.
- **Generative Hallucinations** (e.g., frame interpolation via Stable Video Diffusion): While visually realistic, these rely on statistical texture priors and violate object permanence. For instance, when a car enters a tunnel, diffusion models may cause the car to vanish because "empty tunnels" are more common in the training set.

**Key Challenge**: Current LVLMs are "kinematically naive," treating video frames as bags of independent tokens. They lack a "semantic entity conservation" constraint and cannot spontaneously reject physically impossible trajectories, such as object teleportation or disappearance.

**Goal**: To shift "frame completion" from a probabilistic framework (maximizing $P(x_t \mid x_{t-1})$) to a physical framework (minimizing action), replacing statistical learning with the elegant constraints of classical mechanics.

**Key Insight**: The Principle of Least Action governs phenomena ranging from planetary orbits to quantum field theory, naturally ensuring path smoothness and energy optimality. By analogy, this principle is applied to the semantic manifold, introducing "semantic inertia" (kinetic) and "semantic force fields" (potential) to constrain embedding trajectories.

**Core Idea**: Replace probabilistic generation with variational mechanics; model missing intervals as a two-point boundary value problem (BVP) solved via discrete Euler-Lagrange equations. Object permanence is obtained "for free" without the need for pixel-by-pixel rendering.

## Method

### Overall Architecture
Given start and end frames $t_{\text{start}}, t_{\text{end}}$, the goal is to find the missing embedding sequence $\{z_t\}$ that minimizes the total action. The process involves three steps:

1. **Encoding**: The visual encoder $f_\phi$ and text encoder $g_\psi$ map both frames and queries into the same $d$-dimensional latent space, inducing a Riemannian geometry on this space (Assumption 3.1: semantic isometry).
2. **Learning the Potential Field**: A lightweight MLP $P_\theta$ fits the "energy landscape excited by the text query in the latent space," trained using Noise Contrastive Estimation (NCE).
3. **Action Minimization at Inference**: The discrete sequence is substituted into the action functional, and gradient descent is used to find the optimal trajectory. This avoids frame-by-frame auto-regressive prediction and prevents error accumulation.

### Key Designs

1. **Semantic Lagrangian (Kinetic + Potential Energy Terms)**:

    - **Function**: Defines the "physics" that drives trajectory optimization. The total action is $S[z] = \int (T(z) - \lambda V(z, q)) dt$, where kinetic energy $T = \frac{1}{2}\|\dot{z}\|^2$ represents the inertial cost of semantic velocity, and potential energy $V(z, q) = 1 - \text{sim}(z, g_\psi(q))$ represents the attraction of the text query.
    - **Mechanism**: The kinetic term automatically penalizes "semantic jumps"—instantaneous object disappearance would require infinite semantic velocity, which is prohibited under the least action principle. The potential term allows the query to guide the trajectory like a "gravitational field." The discrete form is $S_{\text{disc}} = \sum_t [\frac{1}{2}\|\frac{z_{t+1} - z_t}{\Delta t}\|^2 - \lambda P_\theta(z_t, q)]$.
    - **Design Motivation**: Conservation laws (energy minimization) automatically encode object permanence more elegantly than explicit supervision. The coupling coefficient $\lambda$ balances smoothness and semantic alignment; experiments identify an optimal "resonance point" at $\lambda \approx 0.5$.

2. **Potential Field Network + NCE + Sobolev Regularization**:

    - **Function**: Uses a differentiable, lightweight MLP $P_\theta$ as a surrogate for the "true semantic potential defined by a frozen LLM," avoiding the astronomical cost of backpropagating through a large LLM at each optimization step.
    - **Mechanism**: The problem is framed as a density ratio estimation for an energy-based model using the InfoNCE loss: $\mathcal{L}_{NCE} = -\mathbb{E} \log \frac{\exp(P_\theta(z, q) / \tau)}{\exp(P_\theta(z, q)/\tau) + \sum_j \exp(P_\theta(z_j, q)/\tau)}$. As $K \to \infty$, the optimal surrogate satisfies $P_\theta^*(z, q) = \log \frac{p(z\mid q)}{p(z)} + C(q)$. Maximizing the surrogate is theoretically equivalent to minimizing the true potential. Sobolev regularization $\mathcal{L}_{\text{reg}} = \mathbb{E}\|\nabla_z P_\theta\|^2$ and spectral normalization are applied to ensure "gentle semantic gravity" and stable discrete Euler-Lagrange solving.
    - **Design Motivation**: Euler-Lagrange solvers require smooth potential function gradients to prevent discrete steps from diverging. Theorem 3.7 provides an explicit upper bound for trajectory deviation $\frac{T^2}{\mu}\epsilon$ relative to gradient error $\epsilon$.

3. **Boundary Value Problem (BVP) Solving**:

    - **Function**: Rather than performing frame-by-frame auto-regression (which is prone to drift), the entire missing interval is optimized simultaneously, with the start and end frames fixed as hard constraints.
    - **Mechanism**: Initialization is performed via SLERP, followed by gradient descent to minimize the discrete action. Theorem 3.6 proves that the action functional is strictly convex with a unique global optimum when $\lambda \cdot \max \|\nabla^2 P_\theta\| < \frac{2}{\lambda \Delta t^2}$.
    - **Design Motivation**: Using the start and end frames as "future constraints" pulls the intermediate trajectory back toward the correct semantic path, preventing the "context drift" common in Transformers over long sequences (e.g., forgetting a car entered a tunnel and hallucinating streetlights instead).

### Loss & Training
The potential network $P_\theta$ is pre-trained on WebVid-10M with the objective $\mathcal{L}_{\text{total}} = \mathcal{L}_{NCE} + \gamma \mathcal{L}_{\text{reg}}$. The encoders remain frozen.

## Key Experimental Results

### Main Results: Tunnel Test (Object Permanence)

| Method | Accuracy ↑ | Permanence Score (1-5) ↑ | Semantic Drift ↓ |
|:---|:---:|:---:|:---:|
| ZOH (Zero-Order Hold) | 24.3 | 1.2 | 0.45 |
| SLERP (Linear) | 41.5 | 2.1 | 0.38 |
| Latent ODE | 58.2 | 3.4 | 0.29 |
| Video-LLaMA 3 (Auto-regressive) | 68.1 | 3.9 | 0.25 |
| Stable Video Diffusion | 71.4 | 3.5 | 0.28 |
| **SLAP (Ours)** | **83.9** | **4.7** | **0.14** |

### Ablation Study (Tunnel Test)

| Configuration | Accuracy | Description |
|:---|:---:|:---|
| Full SLAP | 83.9 | $\lambda \approx 0.5$ |
| $\mu \to 0$ (Pure Potential) | 62.0 | Loss of inertia; objects frequently appear/disappear |
| $\mu \to \infty$ (Pure Inertia) | 41.5 | Degenerates to SLERP; ignores text |
| Static Potential (Fixed Cosine) | 70.5 | Learned $P_\theta$ is indispensable |

### MSR-VTT Video QA (Robustness vs. Sampling Rate)

| Method | 50% Frames ↑ | 25% Frames ↑ | 10% Frames ↑ | Gain (Drop) ↓ |
|:---|:---:|:---:|:---:|:---:|
| ZOH | 38.4 | 31.2 | 22.5 | -15.9 |
| Linear | 40.1 | 35.8 | 30.1 | -10.0 |
| Video-LLaMA 3 | 44.5 | 41.2 | 34.7 | -9.8 |
| SVD | 43.8 | 39.5 | 35.2 | -8.6 |
| **SLAP** | **45.2** | **43.9** | **41.8** | **-3.4** |

### Computational Efficiency

| Method | TFLOPs ↓ | Latency (s) ↓ | Memory (GB) ↓ | Speedup |
|:---|:---:|:---:|:---:|:---:|
| Stable Video Diffusion | 185.0 | 14.20 | 22.5 | 1.0× |
| Video-LLaMA 3 | 45.2 | 3.80 | 16.0 | 3.7× |
| Neural ODE | 12.5 | 1.10 | 8.4 | 12.9× |
| **SLAP** | **0.15** | **0.08** | **0.8** | **177.5×** |

### Key Findings
- SLAP significantly outperforms SVD in object permanence (+12.5 points) and halves semantic drift. This is because replacing an object with an empty tunnel requires a massive semantic velocity in the kinetic term, which is automatically rejected by the least action principle.
- Under extreme sparsity (10% frames), performance drops by only 3.4%, compared to 9.8% for Video-LLaMA 3, indicating that the semantic action defined by boundary frames and text queries is often sufficient for QA tasks.
- On "action-centric" questions, SLAP outperforms Video-LLaMA 3 by 12 points, as the least action principle naturally recovers geodesics in the verb space (e.g., standing → falling → lying down).
- At 0.15 TFLOPs per inference (~0.5 Joules) compared to ~150 Joules for SVD, carbon emissions are reduced by three orders of magnitude.
- A $\lambda$ scan reveals a clear "resonance regime": Ballistic ($\lambda \to 0$, 41.5%) → Weak Coupling ($\lambda = 0.1$, 65.2%) → Resonance ($\lambda = 0.5$, 83.9%) → Strong Coupling ($\lambda = 1.0$, 79.1%) → Chaos ($\lambda > 5$, 31%).

## Highlights & Insights
- **Elegant Transfer of Physical Intuition**: Transitioning conservation laws and the least action principle from classical mechanics to the semantic manifold is a philosophical innovation. It suggests that future architectures could be designed based on "symmetry and conservation laws."
- **BVP vs. Auto-regression**: Using dual constraints to "globally pull" intermediate trajectories back to the correct path is a universal solution for handling drift in long sequences, applicable to long documents or trajectory prediction.
- **Lightweight Surrogates with Theoretical Guarantees**: Training the potential field via InfoNCE and Sobolev regularization provides explicit upper bounds for trajectory deviation, a strategy applicable to RL reward models and EBM training.
- **Discovery of Resonance Regimes**: The $\lambda$ scan reveals distinct physical phase transitions (ballistic—resonance—chaos), a phenomenological result rarely seen so clearly in deep learning.

## Limitations & Future Work
- When the missing interval is too long, the trajectory error upper bound $\frac{T^2}{\mu}\epsilon$ grows quadratically, requiring multi-stage solvers or denser anchor points.
- The method relies heavily on Assumption 3.1 (that the Riemannian geometry induced by the encoder is proportional to pixel distance); changing the encoder may invalidate this.
- Experiments focused primarily on short videos (10–20 seconds) like MSR-VTT and ActivityNet; generalization to minute-long videos or audio-visual multi-modality is unknown.
- The potential network is pre-trained on WebVid-10M, potentially limiting robustness to domain shifts in fields like medicine or science.

## Related Work & Insights
- **vs. Implicit Pooling**: Pooling destroys causal structure; SLAP preserves the continuous evolution of the entire trajectory via the kinetic term.
- **vs. Diffusion (SVD)**: Diffusion tends to generate statistically likely pixels, leading to hallucinations of disappearing objects in rare scenes; SLAP uses least action to favor the "most economical trajectory," naturally conserving object identity.
- **vs. Auto-regressive Transformers**: Long sequences prone to context drift; SLAP's dual-endpoint constraints mitigate this issue.
- **Insight**: For tasks requiring "physical common sense," seeking inspiration from classical physical dualities and conservation laws is more effective than relying solely on statistical learning.

## Rating
- Novelty: ⭐⭐⭐⭐⭐  Adapting the principle of least action to the semantic manifold breaks the monopoly of generative models.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐  Covers tunnel tests, MSR-VTT, and ActivityNet, including detailed ablations and power/efficiency comparisons.
- Writing Quality: ⭐⭐⭐⭐  Clear physical analogies and mathematical rigor; discussion on Assumption 3.1 could be more detailed.
- Value: ⭐⭐⭐⭐⭐  177× speedup and 3-order magnitude reduction in carbon emissions; provides a direct solution for object permanence and is transferable to various domains like RL energy models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Frequency-Semantic Enhanced Variational Autoencoder for Zero-Shot Skeleton-based Action Recognition](../../ICCV2025/video_understanding/frequency-semantic_enhanced_variational_autoencoder_for_zero-shot_skeleton-based.md)
- [\[AAAI 2026\] Explicit Temporal-Semantic Modeling for Dense Video Captioning via Context-Aware Cross-Modal Interaction](../../AAAI2026/video_understanding/explicit_temporal-semantic_modeling_for_dense_video_captioning_via_context-aware.md)
- [\[ICML 2026\] SkelHCC: A Hyperbolic CLIP-Driven Cache Adaptation Framework for Skeleton-based One-Shot Action Recognition](skelhcc_a_hyperbolic_clip-driven_cache_adaptation_framework_for_skeleton-based_o.md)
- [\[ICML 2026\] OmniSIFT: Modality-Asymmetric Token Compression for Efficient Omni-modal Large Language Models](omnisift_modality-asymmetric_token_compression_for_efficient_omni-modal_large_la.md)
- [\[ACL 2026\] Response-G1: Explicit Scene Graph Modeling for Proactive Streaming Video Understanding](../../ACL2026/video_understanding/response-g1_explicit_scene_graph_modeling_for_proactive_streaming_video_understa.md)

</div>

<!-- RELATED:END -->
