---
title: >-
  [Paper Note] Offline Reinforcement Learning with Generative Trajectory Policies
description: >-
  [ICLR 2026][Image Generation][Offline Reinforcement Learning] This paper proposes Generative Trajectory Policies (GTP), which adopts a unified perspective treating diffusion models, flow matching, and consistency models as special cases of ODE solution mappings. GTP learns a complete continuous-time trajectory solution mapping and introduces two adaptation techniques—score approximation and advantage weighting—achieving state-of-the-art performance on the D4RL benchmark.
tags:
  - ICLR 2026
  - Image Generation
  - Offline Reinforcement Learning
  - Generative Policy
  - ODE Trajectory
  - Consistency Models
  - Flow Matching
date: 2026-05-08
content_hash: d7ae37a860c54608
---

# Offline Reinforcement Learning with Generative Trajectory Policies

**Conference**: ICLR 2026
**arXiv**: [2510.11499](https://arxiv.org/abs/2510.11499)
**Code**: None (planned open-source)
**Area**: Image Generation
**Keywords**: Offline Reinforcement Learning, Generative Policy, ODE Trajectory, Consistency Models, Flow Matching

## TL;DR

This paper proposes Generative Trajectory Policies (GTP), which adopts a unified perspective treating diffusion models, flow matching, and consistency models as special cases of ODE solution mappings. GTP learns a complete continuous-time trajectory solution mapping and introduces two adaptation techniques—score approximation and advantage weighting—achieving state-of-the-art performance on the D4RL benchmark.

## Background & Motivation

1. **Background**: In offline RL, generative models have emerged as powerful policy classes due to their ability to capture complex multimodal behavior distributions. Diffusion-based policies excel in expressiveness, while consistency-based policies offer inference efficiency.

2. **Limitations of Prior Work**: Diffusion policies require iterative denoising, resulting in high inference cost. Consistency policies enable 1–2 step generation but often suffer performance degradation. The two approaches present a fundamental trade-off between expressiveness and efficiency.

3. **Key Challenge**: Existing methods have developed independently, lacking a unified perspective to understand and transcend their respective limitations. It remains an open question whether a policy class can be both highly expressive and computationally efficient.

4. **Goal**: To break the expressiveness-efficiency trade-off in generative policies by designing a flexible multi-step generative policy that achieves high performance even with a small number of sampling steps.

5. **Key Insight**: The authors observe that diffusion models, flow matching, and consistency models can all be viewed as learning solution mappings $\Phi(\bm{x}_t, t, s)$ of a continuous-time ODE $\frac{d\bm{x}_t}{dt} = f(\bm{x}_t, t)$, and exploit this unified perspective to design a novel policy class.

6. **Core Idea**: Represent the policy as an ODE solution mapping (flow map), combined with efficient score approximation and advantage-weighted objectives, yielding an offline RL policy that is both expressive and efficient.

## Method

### Overall Architecture

GTP is an actor-critic framework. The actor is a generative trajectory policy $\Phi_\theta(s, a_t, t, \tau)$ that learns a solution mapping from noisy actions to clean actions. The critic is a standard double-Q network $Q_\varphi$. The actor is optimized through two complementary objectives (instantaneous flow loss + trajectory consistency loss) with advantage weighting for policy improvement. At inference time, actions are produced by iteratively stepping from Gaussian noise.

### Key Designs

**1. Unified ODE Trajectory Framework**

- **Function**: Provides the theoretical foundation for a unified understanding of diffusion, flow matching, and consistency models.
- **Mechanism**: Defines the ideal flow map $\Phi(\bm{x}_t, t, s) = \bm{x}_t + \int_t^s f(\bm{x}_\tau,\tau)d\tau$ and its reparameterized form $\phi(\bm{x}_t, t, s)$. Training employs two complementary objectives: an instantaneous flow loss (local correctness, corresponding to diffusion denoisers and flow matching velocity fields) and a trajectory consistency loss (global coherence, $\Phi(\bm{x}_t,t,s) \approx \Phi(\Phi(\bm{x}_t,t,u),u,s)$).
- **Design Motivation**: Existing models are each special cases of ODE learning; the unified perspective reveals a clearer policy design space.

**2. Efficient and Stable Score Approximation**

- **Function**: Addresses the computational cost of ODE integration and the training instability caused by self-referential supervision.
- **Mechanism**: Replaces the true score $f^*(\bm{x}_t,t)$, which requires multi-step integration, with a closed-form surrogate $\tilde{f}(\bm{x}_t,t) = (\bm{x}_t - \bm{x})/t$. Theorem 1 proves that the resulting objective error is $O(h^p)$ (where $p$ is the solver order) and vanishes as the step size approaches zero.
- **Design Motivation**: Using inaccurate early estimates as supervision creates a vicious cycle analogous to bootstrapping in TD learning. Anchoring to an analytic signal derived from offline data eliminates error propagation.

**3. Advantage-Weighted Value-Guided Policy Improvement**

- **Function**: Elevates the generative model from behavior cloning to genuine policy improvement.
- **Mechanism**: Theorem 2 proves that the optimal solution to KL-regularized policy optimization satisfies $\pi^*(a|s) \propto \pi_{BC}(a|s)\exp(\eta A(s,a))$. An exponential advantage weight $w(s,a) = \exp(\eta \cdot \max(0, A(s,a))/(\text{std}(A)+\epsilon))$ is incorporated into the generative loss to prioritize imitation of high-advantage actions.
- **Design Motivation**: A purely generative objective reduces to behavior cloning and cannot achieve policy improvement. Advantage weighting provides a theoretically principled form of value guidance.

### Loss & Training

Total actor loss: $\mathcal{L}_{\text{actor}} = \mathcal{L}_{\text{Consistency}} + \lambda_{\text{Flow}} \cdot \mathcal{L}_{\text{Flow}}$

- Trajectory consistency loss: $\mathcal{L}_{\text{Consistency}} = \mathbb{E}[w(s,a)\|\Phi_\theta(s,a_t,t,\tau) - \Phi_{\theta^-}(s,\tilde{a}_u,u,\tau)\|_2^2]$
- Instantaneous flow loss: $\mathcal{L}_{\text{Flow}} = \mathbb{E}[w(s,a)\|a - \phi_\theta^{\text{inst}}(s,a_t,t)\|_2^2]$

The critic uses a standard double-Q network trained with TD error loss, with target networks updated via EMA.

## Key Experimental Results

### Main Results

D4RL behavior cloning (BC) performance comparison; GTP-BC uses 5-step sampling:

| Task | Diffusion-BC | Consistency-BC | GTP-BC (Ours) |
|------|-------------|----------------|---------------|
| Gym Average | 76.3 | 69.7 | **82.3** |
| AntMaze Average | 41.7 | 44.1 | **66.3** |
| halfcheetah-mr | 41.7 | 34.4 | **46.3** |
| hopper-mr | 67.3 | 99.7 | **100.5** |

D4RL offline RL performance comparison (full actor-critic framework):

| Task | IDQL | DIPO | D-QL | C-QL | GTP (Ours) |
|------|------|------|------|------|------------|
| AntMaze-large-diverse | 47.5 | — | 47.3 | 51.0 | **100.0** |
| AntMaze-medium-diverse | — | — | — | — | **100.0** |

### Ablation Study

| Configuration | Gym Avg. | AntMaze Avg. | Notes |
|------|---------|-------------|------|
| Full GTP | Best | 100.0 | Both losses + advantage weighting + score approximation |
| w/o trajectory consistency loss | Drops | Drops | Global consistency critical for long-horizon tasks |
| w/o instantaneous flow loss | Drops | Drops | Local dynamics anchoring is indispensable |
| True score (ODE integration) | Unstable | Poor | Validates necessity of score approximation |
| w/o advantage weighting | Degrades to BC | Degrades | No policy improvement capability |

### Key Findings

- GTP **achieves perfect scores (100.0) for the first time** on several challenging AntMaze tasks, significantly outperforming all prior methods.
- Under the BC setting, GTP-BC already substantially outperforms Diffusion-BC and Consistency-BC, demonstrating the intrinsic expressive power of the trajectory policy.
- Score approximation accelerates training and improves stability; the theoretical error bound $O(h^p)$ is empirically validated.
- Strong performance is achieved with as few as 5 sampling steps, demonstrating a favorable balance between efficiency and quality.

## Highlights & Insights

- **Theoretical contribution of the unified perspective**: Subsuming diffusion, flow matching, and consistency models into a single ODE framework provides a clear design space for policy development.
- **Complementary dual-objective design**: The instantaneous flow loss ensures local accuracy while the trajectory consistency loss ensures global coherence.
- **Elegance of score approximation**: A simple closed-form expression replaces complex ODE integration, with theoretical guarantees and superior empirical performance.
- **Perfect AntMaze scores**: A landmark result with significant implications for the offline RL community.

## Limitations & Future Work

- Evaluation is primarily conducted on standard D4RL benchmarks; more complex real-world tasks remain untested.
- Despite the elegance of the unified ODE framework, theoretical guidance for selecting the optimal number of sampling steps is lacking.
- Extending GTP to online RL and model-based RL settings is a promising direction.
- Integration with recent token-level generative policy methods warrants exploration.

## Related Work & Insights

- Consistency Trajectory Models (CTM) provide the foundation for learning ODE solution mappings; GTP extends this to the RL domain.
- The advantage weighting scheme is conceptually aligned with AWR/AWAC, but admits a novel theoretical interpretation within the generative model framework.
- Insight: A unified theoretical perspective across different generative models can yield approaches that surpass each individual method.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Unified ODE trajectory perspective combined with two theoretically principled adaptations.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive D4RL evaluation with thorough ablations; perfect AntMaze scores are compelling.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous theoretical derivations and clear architectural diagrams.
- Value: ⭐⭐⭐⭐⭐ Substantial and lasting impact on generative policy research in offline RL.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Flow Matching with Injected Noise for Offline-to-Online Reinforcement Learning](flow_matching_with_injected_noise_for_offline-to-online_reinforcement_learning.md)
- [\[ICLR 2026\] Hierarchical Entity-centric Reinforcement Learning with Factored Subgoal Diffusion](hierarchical_entity-centric_reinforcement_learning_with_factored_subgoal_diffusi.md)
- [\[ICLR 2026\] DiffusionNFT: Online Diffusion Reinforcement with Forward Process](diffusionnft_online_diffusion_reinforcement_with_forward_process.md)
- [\[ICLR 2026\] Improving Discrete Diffusion Unmasking Policies Beyond Explicit Reference Policies (UPO)](improving_discrete_diffusion_unmasking_policies_beyond_explicit_reference_polici.md)
- [\[ICLR 2026\] RIDER: 3D RNA Inverse Design with Reinforcement Learning-Guided Diffusion](rider_3d_rna_inverse_design_with_reinforcement_learning-guided_diffusion.md)

</div>

<!-- RELATED:END -->
