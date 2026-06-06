---
title: >-
  [Paper Note] Act to See, See to Act: Diffusion-Driven Perception-Action Interplay for Adaptive Policies
description: >-
  [NeurIPS 2025][Reinforcement Learning][diffusion policy] This paper proposes DP-AG (Action-Guided Diffusion Policy), which uses the Vector-Jacobian Product (VJP) of a diffusion policy's noise prediction as a structured s…
tags:
  - "NeurIPS 2025"
  - "Reinforcement Learning"
  - "diffusion policy"
  - "perception-action loop"
  - "VJP"
  - "variational inference"
  - "imitation learning"
date: 2026-05-08
content_hash: 9bbf6c8ab697a55a
---

# Act to See, See to Act: Diffusion-Driven Perception-Action Interplay for Adaptive Policies

**Conference**: NeurIPS 2025
**arXiv**: [2509.25822](https://arxiv.org/abs/2509.25822)  
**Code**: [Project Page](https://jingwang18.github.io/dp-ag.github.io/)  
**Area**: Diffusion Models / Robot Learning / Imitation Learning
**Keywords**: diffusion policy, perception-action loop, VJP, variational inference, imitation learning

## TL;DR
This paper proposes DP-AG (Action-Guided Diffusion Policy), which uses the Vector-Jacobian Product (VJP) of a diffusion policy's noise prediction as a structured stochastic force to drive dynamic evolution of latent observation features across diffusion steps, and closes the perception-action loop via a cycle-consistent contrastive loss. DP-AG achieves +6% on Push-T, +13% on Dynamic Push-T, and +23%+ success rate on a real UR5 robot.

## Background & Motivation

**Background**: In imitation learning (IL), Diffusion Policy (DP) achieves state-of-the-art performance in robotic manipulation by modeling action distributions via diffusion denoising. However, DP and related methods **decouple** perception from action — observation features remain static throughout a single action sequence generation.

**Limitations of Prior Work**: Static feature encoding is "frozen" during the entire action sequence generation, ignoring intermediate feedback during action production. Unlike humans, such methods cannot dynamically adjust their understanding of the environment based on the actions being executed. This results in less coherent and smooth action sequences, especially in dynamic environments.

**Key Challenge**: The denoising process of diffusion policies already contains rich intermediate action information (per-step noise predictions), yet this information is never used to refine perceptual representations. The perception-action relationship is unidirectional (perception → action), lacking the action → perception feedback loop.

**Goal**
- How to establish a bidirectional closed loop between perception and action?
- How to leverage intermediate noise predictions within the diffusion process to dynamically update observation features?

**Key Insight**: Inspired by human perception-action coupling ("Act to See, See to Act"), the VJP of the diffusion noise prediction with respect to latent features is used as a "structured stochastic force" to drive latent feature evolution, enabling perceptual representations to co-evolve with action refinement.

**Core Idea**: Drive the SDE-based evolution of latent observation features using the VJP of the diffusion policy's noise prediction, thereby closing the perception-action loop.

## Method

### Overall Architecture
DP-AG augments standard Diffusion Policy with three components: (1) **Variational Inference**: encoding observation features as a Gaussian posterior $q_\phi(z_t|o_t) = \mathcal{N}(\mu_\phi, \sigma_\phi^2)$; (2) **Action-Guided SDE**: evolving latent features across diffusion step $k$ as $d\tilde{z}_t^k = \text{VJP}(\hat{a}_t^k, z_t) dt + \sigma_\phi dW_t$; (3) **Cycle-Consistent Contrastive Loss**: aligning noise predictions conditioned on static vs. evolved latent features. During action generation, observation features are no longer static but are updated in sync with action refinement.

### Key Designs

1. **VJP-Driven Latent Feature Evolution**

    - Function: Dynamically update latent observation features at each diffusion denoising step based on action feedback.
    - Mechanism: The VJP computes the "backward propagation" direction of the noise prediction with respect to latent features: $\text{VJP}(\hat{a}_t^k, z_t) = (\frac{\partial \epsilon_\theta}{\partial z_t})^\top \epsilon_\theta$. This direction points toward feature adjustments that most reduce action uncertainty. In discretized form: $\tilde{z}_t^k = \mu_\phi(z_t) + \gamma \sigma_\phi(z_t) \odot \text{VJP}(\hat{a}_t^k, z_t)$.
    - Design Motivation: The VJP provides "task-driven attention" — guiding latent features to focus on the parts of the observation most relevant to the current action. Analogously, while driving, the scenery outside is unchanged, but attention shifts to the curve edge when turning and to the vehicle ahead when accelerating. The VJP is the computational realization of this dynamic attention.

2. **Cycle-Consistent Contrastive Loss (Cycle-Consistent InfoNCE)**

    - Function: Ensure that latent feature evolution remains consistent with action diffusion, preventing excessive drift.
    - Mechanism: At each step $k$, two noise predictions are computed: $\varepsilon_k = \epsilon_\theta(\hat{a}_t^k, z_t, k)$ (static latent) and $\tilde{\varepsilon}_k = \epsilon_\theta(\hat{a}_t^k, \tilde{z}_t^k, k)$ (evolved latent). An InfoNCE loss pulls matched pairs together and pushes non-matched pairs apart: $\mathcal{L}_{\text{cont}} = -\frac{1}{B}\sum_i \log \frac{\exp(\text{sim}(\varepsilon_k^i, \tilde{\varepsilon}_k^i)/\tau)}{\sum_{j \neq i} \exp(\text{sim}(\varepsilon_k^i, \tilde{\varepsilon}_k^j)/\tau)}$.
    - Design Motivation: VJP-guided evolution may cause latent features to drift excessively. The contrastive loss "anchors" evolved features near static features, preserving semantic consistency. Theorem 1 theoretically establishes that under Lipschitz conditions, minimizing the contrastive loss provides an upper bound on latent feature drift: $\|\tilde{z}_t^{k+1} - \tilde{z}_t^k\|^2 \leq 2L^2(2 - \tau\ln(B-1) + \tau\alpha)$.

3. **Variational Inference Foundation**

    - Function: Provide a probabilistic framework for latent feature evolution.
    - Mechanism: An action-guided ELBO is derived (Eq. 12): $\log p(\varepsilon_k|z_t) \geq \mathbb{E}_{q_\phi}[\log p(\tilde{\varepsilon}_k|\tilde{z}_t^k)] - \text{KL}(q_\phi \| p)$. The KL term encourages evolved latent features to remain close to the prior $p(\tilde{z}_t^k|z_t) = \mathcal{N}(z_t, I)$.

### Loss & Training
- Total loss: $\mathcal{L}_{\text{DP-AG}} = \mathcal{L}_{\text{DP}} + \lambda_{\text{cont}} \mathcal{L}_{\text{cont}} + \lambda_{\text{KL}} \mathcal{L}_{\text{KL}}$
- The noise matching term preserves action prediction capability; the contrastive term closes the perception-action loop; the KL term prevents latent feature drift.

## Key Experimental Results

### Main Results: Push-T and Dynamic Push-T

| Method | Push-T (img) | Push-T (kp) | Dynamic Push-T |
|--------|-------------|-------------|----------------|
| IBC | 0.75 | 0.90 | 0.52 |
| AdaFlow | 0.87 | 0.91 | 0.67 |
| DP (baseline) | 0.87 | 0.95 | 0.65 |
| **DP-AG (ours)** | **0.93** | **0.99** | **0.80** |

Push-T image modality: 0.87 → **0.93** (+6%); Dynamic Push-T: 0.65 → **0.80** (+13%).

### Real Robot UR5 Experiments

| Task | DP Success Rate | DP-AG Success Rate | Gain |
|------|-----------------|--------------------|------|
| Manipulation | ~60% | ~83% | **+23%** |
| Action Smoothness | Baseline | **↓60%** smoother | Significant |

### Ablation Study: Contribution of Each Component

| Configuration | Irregular Spiral MSE |
|---------------|----------------------|
| Base Flow (no VJP) | 0.0095 |
| **VJP-Guided Flow** | **0.0052** (↓45.3%) |

### Key Findings
- **VJP-guided latent feature evolution forms a structured manifold**: Visualization shows that Base Flow latent states are scattered, whereas VJP guidance produces a structured manifold aligned with the output.
- **Greater advantage in dynamic environments**: Improvement on Dynamic Push-T (+13%) exceeds that on static Push-T (+6%), as dynamic scenarios demand more adaptive perception.
- **Contrastive loss outperforms MSE**: Replacing MSE (absolute matching) with contrastive loss (relative similarity) avoids overly rigid constraints while allowing bounded adaptation.
- **VJP computation overhead is negligible**: Leveraging modern automatic differentiation frameworks, VJP computation introduces minimal additional cost.

## Highlights & Insights
- The **biological inspiration of "Act to See, See to Act"** is highly intuitive and substantive: it introduces the cyclic nature of human active perception into policy learning — not by acquiring new observations, but by **reinterpreting** the same observation in light of actions. This is a computational realization of "enactive perception" in cognitive science.
- The interpretive framework of **VJP as "task-driven attention"** is elegant: the VJP direction points toward feature adjustments that most reduce action uncertainty, and its magnitude reflects the degree of uncertainty — large adjustments under high uncertainty, fine-tuning under low uncertainty.
- **Theory and experiment are tightly coupled**: Lemma 1 and Theorem 1 are not abstract guarantees but directly predict the causal chain — contrastive loss minimization → bounded latent drift → improved trajectory continuity — which is validated experimentally.

## Limitations & Future Work
- **VJP computation requires backpropagation through the noise predictor**: Although the overhead is claimed to be small, it may become non-negligible for large models.
- **Validated only on DDPM**: Extension to flow matching and similar frameworks is claimed but not experimentally verified.
- **Hyperparameter $\gamma$ for latent feature evolution**: The VJP intensity requires tuning and may need task-specific configuration.
- **Single-view observations**: Effectiveness with multi-view inputs or other modalities such as tactile sensing has not been evaluated.

## Related Work & Insights
- **vs. Diffusion Policy (Chi et al.)**: DP maintains static features; DP-AG evolves features dynamically — this is the core distinction. DP-AG extends action continuity from DP to encompass perceptual continuity.
- **vs. PlaNet/Dreamer (latent state models)**: These methods use VAEs to predict future states for planning, but do not dynamically update features during action generation — a fundamentally different dimension.
- **vs. VLAs (OpenVLA, π₀)**: VLAs enhance perception through vision-language models but remain static; DP-AG dynamically enhances perception through action feedback — the two approaches may be complementary.
- **Transferable Insight**: The concept of using VJP as a driving force for latent feature evolution is transferable to any conditional generative process — e.g., adjusting context representations in text generation based on feedback from generated content.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The perception-action closed-loop design is conceptually distinctive; VJP-driven latent feature evolution represents a genuinely novel technical pathway.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Theoretical validation (spiral) + simulation (Push-T/Robomimic/Kitchen) + real robot (UR5), with a progressively rigorous evaluation structure.
- Writing Quality: ⭐⭐⭐⭐⭐ The narrative chain from biological inspiration → mathematical framework → theoretical guarantees → experimental validation is exceptionally coherent.
- Value: ⭐⭐⭐⭐⭐ Introducing dynamic perceptual mechanisms into diffusion policies yields significant practical improvements in robotic manipulation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Recovering Hidden Reward in Diffusion-Based Policies](../../ICML2026/reinforcement_learning/recovering_hidden_reward_in_diffusion-based_policies.md)
- [\[NeurIPS 2025\] Learning Intractable Multimodal Policies with Reparameterization and Diversity Regularization](learning_intractable_multimodal_policies_with_reparameterization_and_diversity_r.md)
- [\[NeurIPS 2025\] Real-World Reinforcement Learning of Active Perception Behaviors](real-world_reinforcement_learning_of_active_perception_behaviors.md)
- [\[NeurIPS 2025\] Reinforcement Learning with Action Chunking](reinforcement_learning_with_action_chunking.md)
- [\[AAAI 2026\] CHDP: Cooperative Hybrid Diffusion Policies for RL in Parametric Environments](../../AAAI2026/reinforcement_learning/chdp_cooperative_hybrid_diffusion_policies_for_reinforcement_learning_in_paramet.md)

</div>

<!-- RELATED:END -->
