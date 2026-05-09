---
title: >-
  [Paper Note] From Parameters to Behaviors: Unsupervised Compression of the Policy Space
description: >-
  [ICLR 2026][Image Generation][policy space compression] Based on the manifold hypothesis, this paper proposes unsupervised compression of the policy space—training an autoencoder with a behavioral reconstruction loss (rather than a parameter reconstruction loss) to compress the high-dimensional policy parameter space $\Theta \subseteq \mathbb{R}^P$ into a low-dimensional latent behavioral space $\mathcal{Z} \subseteq \mathbb{R}^k$ (up to a 121801:1 compression ratio). Experiments on Mountain Car, Reacher, Hopper, and HalfCheetah demonstrate that the intrinsic dimensionality of the behavioral manifold is determined by environment complexity rather than network size, and that PGPE optimization in the latent space converges faster than PPO, SAC, and other SOTA baselines on 7 out of 8 tasks.
tags:
  - ICLR 2026
  - Image Generation
  - policy space compression
  - behavioral manifold
  - autoencoder
  - latent space optimization
  - unsupervised pretraining
date: 2026-05-08
content_hash: caa311ad47e7ae1b
---

# From Parameters to Behaviors: Unsupervised Compression of the Policy Space

**Conference**: ICLR 2026
**arXiv**: [2509.22566](https://arxiv.org/abs/2509.22566)
**Code**: [GitHub](https://github.com/DavideTenediniPoliMi/from-parameters-to-behaviors-unsupervised-compression-of-the-policy-space)
**Area**: Image Generation
**Keywords**: policy space compression, behavioral manifold, autoencoder, latent space optimization, unsupervised pretraining

## TL;DR

Based on the manifold hypothesis, this paper proposes unsupervised compression of the policy space—training an autoencoder with a behavioral reconstruction loss (rather than a parameter reconstruction loss) to compress the high-dimensional policy parameter space $\Theta \subseteq \mathbb{R}^P$ into a low-dimensional latent behavioral space $\mathcal{Z} \subseteq \mathbb{R}^k$ (up to a 121801:1 compression ratio). Experiments on Mountain Car, Reacher, Hopper, and HalfCheetah demonstrate that the intrinsic dimensionality of the behavioral manifold is determined by environment complexity rather than network size, and that PGPE optimization in the latent space converges faster than PPO, SAC, and other SOTA baselines on 7 out of 8 tasks.

## Background & Motivation

**Background**: The success of deep reinforcement learning relies heavily on high-dimensional parameterization of policies via deep neural networks. However, this high-dimensional parameterization incurs severe sample inefficiency—policy network parameter spaces can span tens of thousands to millions of dimensions, yet many distinct parameter configurations produce identical or highly similar behaviors (state-action distributions).

**Limitations of Prior Work**: (1) The high redundancy of the parameter space leads to extremely low search efficiency—agents struggle to navigate vast parameter spaces where many directions have no effect on behavior. (2) This problem is exacerbated in multi-task settings, where each new task typically requires learning from scratch (tabula rasa), failing to exploit shared structure across environments. (3) Existing approaches (e.g., diverse behavior discovery via DIAYN, asymmetric actor-critic) only partially alleviate these issues without fundamentally addressing parameter space redundancy.

**Key Challenge**: The parameter dimensionality $P$ of a policy network can be very large (e.g., $10^5$), while the intrinsic dimensionality of effective behaviors may be extremely small (e.g., $1 \sim 16$). Searching for solutions on a low-dimensional manifold within a high-dimensional space is fundamentally inefficient.

**Goal**: Learn a generative mapping $g: \mathcal{Z} \to \Theta$ from a low-dimensional latent space to the high-dimensional parameter space such that: (1) the latent space is organized by behavioral similarity rather than parameter proximity, (2) the compression is task-agnostic (unsupervised), and (3) the compressed space supports efficient task-specific optimization.

**Key Insight**: The authors ground their approach in the Manifold Hypothesis—a widely accepted assumption in machine learning that high-dimensional data concentrates on low-dimensional manifolds. Applying this to RL: the behavioral distributions of effective policies lie on a low-dimensional manifold within the parameter space, whose dimensionality is determined by environment complexity rather than network size.

**Core Idea**: Train an autoencoder with a behavioral reconstruction loss (rather than a parameter reconstruction loss) to compress the policy parameter space, then perform policy optimization in the learned low-dimensional latent space.

## Method

### Overall Architecture

The method follows the two-phase paradigm of Unsupervised RL (URL): a **pretraining phase** (unsupervised, task-agnostic) learns a low-dimensional representation of the policy parameter space, and a **fine-tuning phase** (supervised, task-specific) performs optimization in the learned low-dimensional space. The pretraining phase is further divided into three steps: (1) policy dataset generation—collecting a behaviorally diverse set of policies; (2) behavioral manifold learning—compressing policy parameters to a low-dimensional space via an autoencoder; and (3) latent behavior optimization—freezing the decoder and performing policy gradient optimization in the low-dimensional space.

### Key Designs

1. **Novelty Search-Based Policy Dataset Generation**:

    - Function: Generate a set of behaviorally diverse policies to serve as training data for the autoencoder.
    - Mechanism: Naive random sampling in parameter space yields poor coverage of the behavioral space—many parameter configurations correspond to similar or identical behaviors. The authors therefore define a behavioral dissimilarity measure based on L2 distance in action space: $D(\pi_\theta \| \pi_{\theta'}) = \sqrt{\sum_{i=1}^{M}(\pi_\theta(\cdot|s_i) - \pi_{\theta'}(\cdot|s_i))^2}$, evaluated on a sampled subset of states. Novelty Search is then applied, scoring each policy by its average behavioral dissimilarity to its $k_n$ nearest neighbors, retaining only the top 10% most novel policies.
    - Design Motivation: Measuring diversity directly in behavioral space (rather than parameter space) ensures that training data covers distinct regions of the behavioral manifold—a prerequisite for learning meaningful compressed representations.

2. **Autoencoder with Behavioral Reconstruction Loss**:

    - Function: Compress the high-dimensional policy parameter space $\Theta \subseteq \mathbb{R}^P$ into a low-dimensional latent space $\mathcal{Z} \subseteq \mathbb{R}^k$ ($k \ll P$).
    - Mechanism: A symmetric autoencoder is used, with encoder $f_\xi: \Theta \to \mathcal{Z}$ and decoder $g_\zeta: \mathcal{Z} \to \Theta$. The key innovation lies in the training loss—rather than minimizing parameter reconstruction error $\|\theta - (g \circ f)(\theta)\|^2$ (which would require exact parameter recovery), the model minimizes the behavioral reconstruction loss $\mathcal{L}_B = \mathbb{E}_{\theta \sim \mathcal{D}_\Theta}[D(\pi_\theta \| \pi_{(g \circ f)(\theta)})]$. In practice, this is approximated via MSE over sampled states: $\hat{\mathcal{L}}_B = \frac{1}{NM}\sum_{i=1}^N \sum_{j=1}^M \|\pi_{\theta_i}(s_j) - \pi_{(g \circ f)(\theta_i)}(s_j)\|_2^2$.
    - Design Motivation: The behavioral reconstruction loss liberates the decoder—it need not recover exact parameter values, but may discover any parameterization that produces the same behavior. This allows the latent space to be organized purely by functional similarity rather than parameter proximity, and constitutes the technical core of the paradigm shift "from parameters to behaviors."

3. **Policy Gradient Optimization in Latent Space**:

    - Function: Leverage the learned low-dimensional manifold representation for efficient task-specific policy optimization.
    - Mechanism: The decoder parameters $\zeta^*$ are frozen, making $g_{\zeta^*}$ a deterministic differentiable function. Standard policy gradients can then be backpropagated through the decoder to the latent space via the chain rule: $\nabla_z J^R(z) = \nabla_z g_{\zeta^*}(z)^\top \nabla_\theta J^R(\theta)$. This is particularly well-suited for parameter-exploring policy gradient methods such as PGPE, which are otherwise inefficient in high-dimensional parameter spaces but regain their effectiveness in low-dimensional latent spaces. When PGPE operates in the latent space, computation of the decoder Jacobian is not required.
    - Design Motivation: Reducing policy optimization from a search over a $10^5$-dimensional parameter space to a search over a $1 \sim 16$-dimensional latent space directly exploits the manifold structure and constitutes the core efficiency gain.

### Loss & Training

The autoencoder is trained with the behavioral reconstruction loss $\hat{\mathcal{L}}_B$, sampling a random batch of states at each gradient step to compute action-space MSE. Training hyperparameters (network architecture, learning rate, etc.) are held constant across all environments and configurations, demonstrating the generality of the approach.

## Key Experimental Results

### Main Results: Latent Behavioral Compression Quality on Mountain Car (Performance Recovery Rate)

Performance recovery rate = performance of the decoded policy in latent space / performance of policies in the training dataset. Values $\geq 1$ indicate recovery that matches or exceeds the original performance.

| Policy Size | Dataset Size | 1D | 2D | 3D |
|------------|-------------|------|------|------|
| Small (~$10^1$ params) | 50k | 0.64 | 0.93 | **0.94** |
| Medium (~$10^3$ params) | 50k | 0.66 | 1.01 | **1.02** |
| Large (~$10^5$ params) | 50k | **1.02** | 1.01 | 1.01 |
| Medium | 100k | 0.51 | **1.02** | **1.02** |
| Large | 100k | **1.01** | **1.01** | **1.01** |

Large policies achieve a recovery rate of 1.01 even in a 1D latent space ($10^5:1$ compression ratio → near-perfect behavioral preservation). Medium policies exceed 1.0 starting from 2D. Small policies exhibit partial collapse at 1D (recovery rates of 0.50–0.64).

### Ablation Study: Generalization Performance on HalfCheetah and Hopper

| Environment | Task | 5D Recovery | 8D Recovery | 16D Recovery |
|-------------|------|-------------|-------------|--------------|
| Hopper | forward | 1.33 | **1.59** | 1.48 |
| Hopper | backward | **2.66** | 1.29 | 1.20 |
| Hopper | jump | **3.83** | 1.54 | 2.42 |
| HalfCheetah | forward | 1.63 | 1.80 | **1.84** |
| HalfCheetah | backward | 1.27 | 1.52 | **1.72** |
| HalfCheetah | frontflip | 0.54 | 0.74 | **1.20** |
| HalfCheetah | backflip | 0.55 | 0.75 | **1.23** |

On HalfCheetah, increasing the latent dimensionality consistently improves performance recovery (5D→16D), suggesting a higher intrinsic dimensionality of the behavioral manifold in this environment. Challenging tasks (frontflip/backflip) have recovery rates below 1.0 at low dimensions but exceed 1.0 at 16D. Trends on Hopper are obscured by high variance.

### Key Findings

- **Extreme compression ratios**: Up to 121801:1 (Large policy → 1D latent space) with near-perfect behavioral preservation, strongly validating the low-dimensionality of the behavioral manifold.
- **Behavioral manifold dimensionality vs. environment complexity**: Mountain Car achieves high recovery rates at 1–2D; Reacher requires 3–5D; HalfCheetah requires 8–16D—validating the hypothesis that intrinsic dimensionality is determined by the environment rather than network size.
- **Convergence speed advantage**: Latent PGPE converges faster than PPO, SAC, TD3, and DDPG on 7 out of 8 tasks, though it does not always converge to the optimal solution.
- **Dataset coverage constrains fine-tuning performance**: Latent PGPE fails on the height task because high-performing height policies are underrepresented in the training dataset, leading to insufficient behavioral manifold learning in that region.

## Highlights & Insights

- **Behavioral reconstruction loss as the core innovation**: Standard autoencoders minimize parameter reconstruction error, but the high redundancy of policy parameters makes exact parameter recovery both unnecessary and infeasible under aggressive compression. The behavioral reconstruction loss frees the decoder, allowing it to discover any parameterization that produces equivalent behavior, enabling the latent space to naturally organize by functional similarity.
- **Empirical evidence that environment complexity determines intrinsic dimensionality**: Mountain Car (simple environment) → 1D suffices; HalfCheetah (complex environment) → 16D required. This not only validates the manifold hypothesis but also offers a novel perspective for "measuring environment complexity."
- **Paradigmatic significance of the modular design**: Each of the three stages (data collection, compression, optimization) is independently replaceable, forming a blueprint for a new RL algorithm design paradigm.

## Limitations & Future Work

- **Coverage bottleneck in the pretraining dataset**: Fine-tuning performance is bounded by the behavioral coverage of the pretraining dataset—behaviors absent from the training set are not encoded in the latent space (as illustrated by the failure on the height task).
- **Computational cost of policy dataset generation**: Generating large numbers (10k–100k) of policies and evaluating their pairwise behavioral dissimilarity incurs non-trivial computational cost in environments such as MuJoCo.
- **Autoencoder architecture not optimized**: The paper employs a fixed symmetric autoencoder architecture with untuned hyperparameters across all settings; more advanced generative models (e.g., VAE, Diffusion Models) may further improve compression quality.
- **Restricted to deterministic policies**: The current approach focuses on deterministic policies $\pi: \mathcal{S} \to \mathcal{A}$; behavioral manifold learning for stochastic policies remains an open problem.

## Related Work & Insights

- **vs. diverse behavior discovery methods (e.g., DIAYN)**: Methods such as DIAYN aim to discover diverse skills or options but do not explicitly compress the policy space. This paper directly learns the low-dimensional manifold structure of the policy space; the two approaches are complementary—skills discovered by DIAYN could be used to improve behavioral coverage in the policy dataset.
- **vs. policy distillation / network pruning**: These methods compress individual policy networks in a task-specific manner. This paper performs task-agnostic compression—learning the low-dimensional structure of the entire policy space once, serving multiple downstream tasks.
- **vs. policy space reduction** (Mutti et al., 2022): That work reduces the cardinality of the policy space (discretization), whereas this paper reduces its dimensionality (continuous representation). The constraints here are more relaxed, avoiding NP-hard optimization.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Behavioral reconstruction loss + empirical validation of the manifold hypothesis in RL + modular two-phase paradigm; the approach is both novel and conceptually deep.
- Experimental Thoroughness: ⭐⭐⭐⭐ Four environments, multiple policy sizes and latent dimensionalities, comparison against four SOTA baselines, 10 random seeds; however, validation on more complex environments (e.g., Atari) is lacking.
- Writing Quality: ⭐⭐⭐⭐ Problem formulation is clear, figures are intuitive, and mathematical notation is rigorous; some content is repetitive.
- Value: ⭐⭐⭐⭐⭐ Introduces a fundamentally new paradigm for improving RL efficiency; the concept of a behavioral manifold opens broad avenues for future research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Generalization of Diffusion Models Arises with a Balanced Representation Space](generalization_of_diffusion_models_arises_with_a_balanced_representation_space.md)
- [\[ICLR 2026\] Unsupervised Conformal Inference: Bootstrapping and Alignment to Control LLM Uncertainty](unsupervised_conformal_inference_bootstrapping_and_alignment_to_control_llm_unce.md)
- [\[ICLR 2026\] Diffusion Fine-Tuning via Reparameterized Policy Gradient of the Soft Q-Function](diffusion_fine-tuning_via_reparameterized_policy_gradient_of_the_soft_q-function.md)
- [\[ICLR 2026\] PCPO: Proportionate Credit Policy Optimization for Aligning Image Generation Models](pcpo_proportionate_credit_policy_optimization_for_aligning_image_generation_mode.md)
- [\[ICLR 2026\] Translate Policy to Language: Flow Matching Generated Rewards for LLM Explanations](translate_policy_to_language_flow_matching_generated_rewards_for_llm_explanation.md)

</div>

<!-- RELATED:END -->
