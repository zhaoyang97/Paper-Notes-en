---
title: >-
  [Paper Note] Flow Matching with Injected Noise for Offline-to-Online Reinforcement Learning
description: >-
  [ICLR 2026][Image Generation][Flow Matching] By injecting controllable noise into flow matching training to broaden policy coverage, and combining an entropy-guided sampling mechanism to dynamically balance exploration a…
tags:
  - "ICLR 2026"
  - "Image Generation"
  - "Flow Matching"
  - "Offline-to-Online RL"
  - "Noise Injection"
  - "Exploration-Exploitation Balance"
  - "Entropy-Guided Sampling"
date: 2026-05-08
content_hash: aab945fec3eedc07
---

# Flow Matching with Injected Noise for Offline-to-Online Reinforcement Learning

**Conference**: ICLR 2026
**arXiv**: [2602.18117](https://arxiv.org/abs/2602.18117)  
**Code**: [GitHub](https://github.com/CTID282/FINO)  
**Area**: Flow Matching / Reinforcement Learning
**Keywords**: Flow Matching, Offline-to-Online RL, Noise Injection, Exploration-Exploitation Balance, Entropy-Guided Sampling

## TL;DR
By injecting controllable noise into flow matching training to broaden policy coverage, and combining an entropy-guided sampling mechanism to dynamically balance exploration and exploitation during online fine-tuning, FINO significantly improves sample efficiency in offline-to-online RL under limited interaction budgets.

## Background & Motivation

**Background**: Generative models (diffusion / flow matching) excel as policy representations in offline RL due to their ability to model multimodal distributions. Flow Q-Learning (FQL) has demonstrated the effectiveness of flow-matching policies in offline settings.

**Limitations of Prior Work**: Policies pre-trained offline are over-constrained to the dataset distribution, resulting in insufficient exploration during online fine-tuning. Existing methods such as FQL treat online fine-tuning as a straightforward continuation of offline pre-training without dedicated exploration mechanisms. In the antmaze-giant task, the FQL agent almost exclusively follows the upper path present in the dataset to reach the goal, completely ignoring other viable routes.

**Key Challenge**: Offline RL requires conservative constraints to avoid out-of-distribution actions, whereas the online phase demands broad exploration beyond data coverage. These two phases impose fundamentally opposing requirements on the policy distribution.

**Goal**: How can a policy learn action coverage broader than the dataset during offline pre-training—without augmenting the dataset—and effectively leverage this diversity during online fine-tuning?

**Key Insight**: Standard flow matching with $\sigma_{\min}=0$ collapses the conditional probability path onto individual data points, limiting coverage. Injecting controlled noise into flow matching can expand the variance of the conditional probability path.

**Core Idea**: Inject noise into the flow matching objective to enlarge the policy support set, and employ entropy-guided sampling to adaptively balance exploration and exploitation during the online phase.

## Method

### Overall Architecture
FINO (Flow matching with Injected Noise for Offline-to-online RL) builds on the FQL framework and comprises two core components: (1) noise-injected flow matching training during offline pre-training, and (2) an entropy-guided sampling mechanism during online fine-tuning. The inputs are state-action pair datasets; the output is a policy capable of efficient exploration in online interaction.

### Key Designs

1. **Noise Injection for Flow Matching**:

    - **Function**: Injects time-dependent Gaussian noise $\epsilon_t \sim \mathcal{N}(0, \alpha_t^2 I)$ along the interpolation path during flow matching training.
    - **Mechanism**: The training objective is modified to $\mathcal{L}_{\text{FINO}}(\theta) = \mathbb{E}[\|v_\theta(t, s, x_t + \epsilon_t) - (x_1 - (1-\eta)x_0)\|^2_2]$, where $\alpha_t^2 = (\eta^2 - 2\eta)t^2 + 2\eta t$ and $\eta \in [0,1]$ controls noise magnitude. Setting $\eta=0$ recovers standard flow matching.
    - **Design Motivation**: Theorem 2 proves that the marginal probability path variance induced by FINO is no less than that of standard FM, i.e., $\text{Var}(X_t^{\text{FINO}}) \geq \text{Var}(X_t^{\text{FM}})$, enabling the policy to cover a broader action space. Theorem 1 guarantees that the noise-injected model still constitutes a valid continuous normalizing flow.

2. **Entropy-Guided Sampling**:

    - **Function**: Samples multiple candidate actions from the policy during online fine-tuning, constructs a softmax sampling distribution based on Q-values, and adaptively adjusts the temperature parameter.
    - **Mechanism**: The sampling probability is $p(i) = \frac{\exp(\xi \cdot Q_\phi(s, a_i))}{\sum_j \exp(\xi \cdot Q_\phi(s, a_j))}$, with temperature $\xi$ updated adaptively according to policy entropy: $\xi_{\text{new}} = \xi - \alpha_\xi[\mathcal{H} - \bar{\mathcal{H}}]$.
    - **Design Motivation**: A fixed temperature cannot adapt to the dynamic changes during learning. When policy entropy is high, $\xi$ is increased to favor exploitation; when entropy is low, $\xi$ is decreased to encourage exploration, achieving automatic balance.

3. **Implementation Details**:

    - **Function**: Addresses the intractable entropy estimation problem for one-step policy distributions.
    - **Mechanism**: Multiple actions are sampled from the same state and a Gaussian Mixture Model (GMM) is fitted to estimate entropy. $\eta=0.1$ (calibrated to the action range $[-1,1]$); the number of candidate actions is set to half the action dimensionality.
    - **Design Motivation**: The one-step policy is obtained via distillation and Q-value optimization, so its distribution cannot be used to compute entropy directly.

### Loss & Training
- **Offline phase**: Jointly trains the flow policy (Eq. 7), the one-step policy (Eq. 5), and the Q-network (TD loss).
- **Online phase**: Continues optimizing all three networks; temperature $\xi$ is updated every $N_\xi$ steps.
- **Inference**: Deterministically selects the action with the highest Q-value.

## Key Experimental Results

### Main Results
Evaluated on 45 tasks across OGBench and D4RL, averaged over 10 random seeds:

| Task Category | Metric | FINO | FQL | Cal-QL | Gain |
|---|---|---|---|---|---|
| OGBench humanoidmaze-medium | Score after online fine-tuning | Best | 3±3 | 0±0 | Significant |
| D4RL antmaze (avg. 6 tasks) | Score after online fine-tuning | Best | 2nd | — | Consistent |
| D4RL adroit (avg. 4 tasks) | Score after online fine-tuning | Best | 2nd | — | Consistent |
| OGBench (avg. 5 tasks) | Score after online fine-tuning | Best | 2nd | — | Significant |

### Ablation Study

| Configuration | Key Performance | Note |
|---|---|---|
| Full FINO | Best | Noise injection + entropy-guided sampling |
| w/o noise injection ($\eta=0$) | Notable drop | Degenerates to standard FQL |
| w/o entropy guidance | Drop | Fixed temperature cannot adapt |
| Noise injection only | Moderate | Lacks exploration-exploitation balance in online phase |

### Key Findings
- In the antmaze-giant task, FINO discovers multiple routes to the goal (including the lower path), whereas FQL exclusively follows the upper path.
- Noise injection causes the policy to cover a substantially broader region around data points in toy experiments, while remaining data-centered.
- The advantage is more pronounced in high-dimensional action space tasks (e.g., humanoidmaze), where the exploration provided by noise injection is more critical.

## Highlights & Insights
- **Theoretical guarantees for noise injection**: Three theorems collectively justify the noise injection design—preserving a valid flow (Theorem 1), expanding coverage (Theorem 2), and constituting a proper distribution (Proposition 1). The theoretical foundation is rigorous and empirically effective.
- **Offline-serves-online design philosophy**: Rather than treating offline training in isolation, the method prepares for subsequent online fine-tuning from the outset by injecting noise to retain diversity. This forward-looking design philosophy is worth emulating.
- **Adaptive temperature** scheme is concise and elegant, using policy entropy as a closed-loop signal to regulate the exploration-exploitation trade-off without manual tuning.

## Limitations & Future Work
- $\eta$ is uniformly set to 0.1, without accounting for potentially different optimal noise magnitudes across tasks or states.
- Entropy estimation relies on GMM fitting, whose accuracy may degrade in extremely high-dimensional action spaces.
- Setting the number of candidate actions to half the action dimensionality is a heuristic rule lacking theoretical justification.
- Experiments are conducted primarily on locomotion and navigation tasks; effectiveness on fine-grained manipulation tasks remains unverified.

## Related Work & Insights
- **vs. FQL**: FQL uses standard flow matching directly as the policy; FINO injects noise into the training objective to expand coverage. FINO significantly outperforms FQL after online fine-tuning, particularly on tasks requiring exploration.
- **vs. Cal-QL / RLPD**: These methods do not employ generative models as policies. FINO leverages the expressive power of flow matching to achieve greater competitiveness on complex tasks.
- **Transferable idea**: The concept of enlarging distributional coverage via noise injection is transferable to other generative modeling scenarios, such as diffusion model fine-tuning and diversity enhancement in conditional generation.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — The idea of noise-injected flow matching is original and the theoretical analysis is rigorous.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 45 tasks, 10 seeds, multiple baselines, detailed ablations.
- **Writing Quality**: ⭐⭐⭐⭐ — Motivation is clear, theoretical derivations are complete, and extensive appendices provide additional support.
- **Value**: ⭐⭐⭐⭐ — Provides a practical, theoretically grounded solution for offline-to-online RL.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Offline Reinforcement Learning with Generative Trajectory Policies](offline_reinforcement_learning_with_generative_trajectory_policies.md)
- [\[ICLR 2026\] DiffusionNFT: Online Diffusion Reinforcement with Forward Process](diffusionnft_online_diffusion_reinforcement_with_forward_process.md)
- [\[ICLR 2026\] Multi-agent Coordination via Flow Matching](multi-agent_coordination_via_flow_matching.md)
- [\[NeurIPS 2025\] Composite Flow Matching for Reinforcement Learning with Shifted-Dynamics Data](../../NeurIPS2025/image_generation/composite_flow_matching_for_reinforcement_learning_with_shifted-dynamics_data.md)
- [\[ICLR 2026\] Laplacian Multi-scale Flow Matching for Generative Modeling](laplacian_multi-scale_flow_matching_for_generative_modeling.md)

</div>

<!-- RELATED:END -->
