---
title: >-
  [Paper Note] Flattening Hierarchies with Policy Bootstrapping
description: >-
  [NeurIPS 2025][Image Generation][Offline goal-conditioned reinforcement learning] This paper proposes SAW (Subgoal Advantage-Weighted Policy Bootstrapping)…
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "Offline goal-conditioned reinforcement learning"
  - "hierarchical reinforcement learning"
  - "policy bootstrapping"
  - "subgoals"
  - "long-horizon tasks"
date: 2026-05-08
content_hash: af4b3987b4a252c6
---

# Flattening Hierarchies with Policy Bootstrapping

**Conference**: NeurIPS 2025
**arXiv**: [2505.14975](https://arxiv.org/abs/2505.14975)  
**Code**: [https://johnlyzhou.github.io/saw/](https://johnlyzhou.github.io/saw/)  
**Area**: Image Generation
**Keywords**: Offline goal-conditioned reinforcement learning, hierarchical reinforcement learning, policy bootstrapping, subgoals, long-horizon tasks

## TL;DR

This paper proposes SAW (Subgoal Advantage-Weighted Policy Bootstrapping), which distills the long-horizon reasoning advantages of hierarchical RL into a single flat policy by sampling subgoals from in-dataset trajectories and performing policy bootstrapping via advantage-weighted importance sampling. The approach requires no learned subgoal generative model, and matches or surpasses state-of-the-art performance across 20 offline GCRL datasets.

## Background & Motivation

**Background**: Offline goal-conditioned reinforcement learning (offline GCRL) trains general-purpose policies capable of reaching arbitrary goal states from reward-free trajectory data, and is regarded as a pretraining paradigm analogous to self-supervised learning. Hierarchical RL (HRL) methods such as HIQL achieve state-of-the-art performance on long-horizon tasks by learning a high-level policy that generates subgoals alongside a low-level policy that executes primitive actions—particularly in settings where one-step methods fail entirely on distant-goal tasks.

**Limitations of Prior Work**: The modular architecture of HRL introduces three fundamental challenges: (1) it requires learning a generative model over the subgoal space, which becomes a severe bottleneck when the state space is high-dimensional (e.g., pixel observations or the 69-dimensional state of a humanoid); (2) compact subgoal representations, while improving generative model tractability, limit policy expressiveness and lead to notable performance degradation in large-scale environments; (3) the choice of representation for subgoal learning (autoregressive prediction, metric learning, value function intermediate layers, etc.) remains an open question, introducing substantial additional design complexity.

**Key Challenge**: The core advantage of HRL lies in exploiting the fact that "closer goals yield better policies," without necessarily requiring a hierarchical structure. The key challenge is how to transfer the benefits of subpolicies to a flat policy without introducing a generative model.

**Goal**: To isolate the fundamental reasons for HRL's success in offline GCRL and distill those advantages into a simple single-level policy, eliminating the complexity of the hierarchical architecture.

**Key Insight**: Through empirical analysis, the authors identify two key factors behind HIQL's success: (1) near-horizon goals provide a better signal-to-noise ratio for advantage estimates, and (2) high-quality actions toward near-horizon goals are more readily sampled from the dataset. HRL is then reinterpreted as implicit policy bootstrapping at test time, from which a training-time bootstrapping objective that requires no generative model is derived.

**Core Idea**: Use true future states from dataset trajectories as subgoals, and bootstrap a flat goal-conditioned policy via subgoal advantage-weighted importance sampling, thereby avoiding the need to learn a subgoal generative model.

## Method

### Overall Architecture

SAW consists of three sequential training stages that share a single value function: (1) train a GCIVL value function $V(s,g)$; (2) train a goal-conditioned subpolicy $\pi^{sub}(a|s,w)$ via AWR, using only near-horizon subgoals; (3) train the final flat goal-conditioned policy $\pi_\theta(a|s,g)$ using the SAW objective, where this policy acquires long-horizon behavior by bootstrapping from the subpolicy. At inference time, only the flat policy is used; no high-level policy is required.

### Key Designs

1. **Unified Probabilistic Inference Perspective on Hierarchical RL**:

    - **Function**: Derives HIQL, RIS, and SAW from a single unified theoretical framework.
    - **Mechanism**: Hierarchical policy optimization is formulated as maximizing the likelihood of a subgoal optimality variable $U$, introducing an ELBO lower bound and variational inference. By choosing different decompositions of the posterior (hierarchical vs. flat) and different approaches to estimating the subgoal distribution (parametric generative model vs. dataset sampling with importance weighting), all three methods are derived from the same unified objective. HIQL corresponds to a hierarchical posterior with hierarchical policy optimization; RIS corresponds to a flat posterior with a parametric subgoal generator; SAW corresponds to a flat posterior with advantage-weighted dataset sampling.
    - **Design Motivation**: The unified derivation reveals the essential connections and distinctions among the three methods, and establishes a clear theoretical foundation for SAW—demonstrating that it is not an ad-hoc approximation, but a specific design choice within the same inference framework.

2. **Subgoal Advantage-Weighted Policy Bootstrapping (SAW Objective)**:

    - **Function**: Enables the flat policy to acquire long-horizon reasoning signals from the near-horizon subpolicy, without requiring a generative model.
    - **Mechanism**: The core objective is $\mathcal{J}(\theta) = \mathbb{E}[e^{\alpha A(s,a,g)} \log \pi_\theta(a|s,g) - e^{\beta A(s,w,g)} D_{KL}(\pi_\theta(a|s,g) \| \pi^{sub}(a|s,w))]$. The first term is standard one-step AWR (using direct value function signals); the second is the policy bootstrapping term (using the subpolicy as a regression target). Crucially, the subgoal $w$ is sampled directly from future states in dataset trajectories ($w \sim p^\mathcal{D}(w|s)$), and is weighted by $e^{\beta A(s,w,g)}$ so that only high-advantage subgoals exert substantial influence. As the goal becomes more distant, the one-step term provides weaker signal, and the bootstrapping term automatically dominates, furnishing a more stable learning target.
    - **Design Motivation**: RIS requires learning a subgoal generative model $\pi^h(w|s,g)$ to approximate the optimal subgoal distribution, which is difficult in high-dimensional spaces. By applying Bayes' rule to convert the expectation from the generative distribution to an importance-weighted form under the dataset distribution $p^\mathcal{D}(w|s)$, the approach entirely circumvents subgoal generation.

3. **The Insight That "Closer Goals Yield More Readily Sampled High-Quality Actions"**:

    - **Function**: Explains why hierarchical training (using near-horizon subgoals to train the low-level policy) is more effective than direct training with distant goals.
    - **Mechanism**: In offline datasets, actions corresponding to near-horizon subgoals (states $k$ steps ahead in the trajectory) exhibit substantially higher advantage values, since those actions were originally executed in the direction of those subgoals. For distant goals, optimal actions are extremely rare, and advantage-weighted regression struggles to extract useful learning signal from low-advantage samples.
    - **Design Motivation**: This observation accounts for the second half of HIQL's performance advantage over simple AWR—beyond improved advantage signal-to-noise ratio, near-horizon subgoals also provide more high-quality samples, improving sampling efficiency.

### Loss & Training

The value function is trained using GCIVL (an action-free IQL variant) with expectile regression to avoid overestimation of out-of-distribution action values. The subpolicy is trained with standard AWR on near-horizon goals. The final policy is trained with the SAW objective (Eq. 9). All three stages are executed sequentially. The subgoal sampling horizon $k$ follows the setting used in HIQL.

## Key Experimental Results

### Main Results

| Environment | Dataset | GCIVL | HIQL | SAW |
|---|---|---|---|---|
| antmaze | medium | 72 | 96 | **97** |
| antmaze | large | 16 | 91 | **90** |
| antmaze | giant | 0 | 65 | **73** |
| humanoidmaze | medium | 24 | 89 | **88** |
| humanoidmaze | large | 2 | 49 | 46 |
| humanoidmaze | giant | 0 | 12 | **35** |
| cube-single | play | 53 | 44 | **72** |
| cube-double | play | 36 | 6 | **40** |
| scene | play | 42 | 38 | **63** |

### Ablation Study

| Configuration | antmaze-giant | humanoidmaze-giant | cube-single |
|---|---|---|---|
| GCIVL + AWR | 0% | 0% | 53% |
| GCWAE (improved SNR) | ~16% | — | — |
| HIQL (hierarchical) | 65% | 12% | 44% |
| HIQL w/o subgoal representation | 65% | — | — |
| SAW w/ subgoal representation | degraded | — | — |
| SAW (dataset sampling) | **73%** | **35%** | **72%** |

### Key Findings

- **SAW substantially outperforms HIQL on the most challenging long-horizon tasks**: on humanoidmaze-giant, SAW achieves 35% vs. HIQL's 12% (nearly 3×), making it the only method to attain a non-trivial success rate in this environment.
- **Subgoal representations are a double-edged sword**: HIQL's subgoal representations are beneficial in medium-scale environments but become a bottleneck in high-dimensional large-scale ones. SAW operates directly in raw observation space, avoiding this issue entirely.
- **Policy bootstrapping is more effective than SNR improvement alone**: GCWAE improves advantage signal-to-noise ratio but still falls far short of HIQL/SAW, demonstrating that the stable training signal provided by the subpolicy as a regression target is equally critical.
- On pixel-based tasks, SAW substantially outperforms HIQL on visual-antmaze-large (82% vs. 53%), but performance degrades on the giant variant, suggesting that value function learning under extreme horizon length combined with high-dimensional visual inputs remains an open problem.

## Highlights & Insights

- **Theoretical unification**: Deriving HIQL, RIS, and SAW from a single probabilistic inference framework establishes the essential connections among all three methods, demonstrates that hierarchical RL is equivalent to test-time policy bootstrapping, and positions SAW as a natural training-time bootstrapping variant. This offers a new theoretical lens through which to understand the advantages of HRL.
- **Key trick for eliminating the generative model**: Applying Bayes' rule to convert the subgoal expectation from a generative distribution to an importance-weighted form under the dataset distribution is conceptually simple yet highly effective—completely bypassing the challenge of subgoal generation in high-dimensional spaces.
- **Analogy to chunking theory**: In the discussion, the authors draw an analogy between policy bootstrapping and the neuroscientific "chunking theory," in which complex skills are first decomposed into simpler segments and then integrated into fluent wholes—an intriguing cross-disciplinary parallel.

## Limitations & Future Work

- SAW relies on subgoal sampling from dataset trajectories, and performance degrades on datasets that require extensive trajectory stitching.
- Value function training instability under pixel observations in extremely long-horizon tasks (visual-antmaze-giant, visual-humanoidmaze) remains an issue, though this is not specific to SAW and limits its applicability.
- The subpolicy's target sampling horizon $k$ is a fixed hyperparameter; adaptive selection may yield further improvements.
- The current analysis is limited to sparse-reward goal-reaching tasks; applicability to dense-reward or continuous control settings has not been validated.

## Related Work & Insights

- **vs. HIQL**: HIQL is the strongest hierarchical baseline and shares the same value function, but requires an additional high-level generative policy and subgoal representation. SAW eliminates both, with clear advantages in simplicity and scalability to high-dimensional settings. HIQL achieves only 12% on humanoidmaze-giant (with subgoal representation) or worse (without), while SAW reaches 35%.
- **vs. RIS**: RIS similarly performs policy bootstrapping but requires learning a subgoal generative model; its offline variant RISoff exhibits high variance across environments. SAW replaces the generative model with dataset sampling and importance weighting, yielding greater stability.
- **vs. QRL/CRL**: These methods use contrastive learning or quasi-metric structures to improve the value function, achieving strong results in certain environments but lacking long-horizon reasoning capacity. SAW supplements long-horizon signals via subpolicy bootstrapping.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The theoretical derivation unifies three methods; the core idea of eliminating the generative model is elegant and concise.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 20 datasets / 100 evaluation tasks / 8 seeds, both state-based and pixel-based, with exceptionally comprehensive coverage.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — The logical chain is clear throughout: problem analysis → insight extraction → theoretical derivation → method design → comprehensive experiments.
- **Value**: ⭐⭐⭐⭐⭐ — Makes important contributions to both offline GCRL and hierarchical RL; the method is simple and practically useful.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] FreqPolicy: Efficient Flow-based Visuomotor Policy via Frequency Consistency](freqpolicy_efficient_flow-based_visuomotor_policy_via_frequency_consistency.md)
- [\[NeurIPS 2025\] RLZero: Direct Policy Inference from Language Without In-Domain Supervision](rlzero_direct_policy_inference_from_language_without_in-domain_supervision.md)
- [\[NeurIPS 2025\] Two-Steps Diffusion Policy for Robotic Manipulation via Genetic Denoising](two-steps_diffusion_policy_for_robotic_manipulation_via_genetic_denoising.md)
- [\[ICLR 2026\] Unsupervised Conformal Inference: Bootstrapping and Alignment to Control LLM Uncertainty](../../ICLR2026/image_generation/unsupervised_conformal_inference_bootstrapping_and_alignment_to_control_llm_unce.md)
- [\[ICCV 2025\] Dense Policy: Bidirectional Autoregressive Learning of Actions](../../ICCV2025/image_generation/dense_policy_bidirectional_autoregressive_learning_of_actions.md)

</div>

<!-- RELATED:END -->
