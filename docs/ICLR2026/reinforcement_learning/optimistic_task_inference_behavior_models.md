---
title: >-
  [Paper Note] Optimistic Task Inference for Behavior Foundation Models
description: >-
  [ICLR 2026 Oral][Reinforcement Learning][behavior foundation models] This paper proposes OpTI-BFM — a test-time task inference method for Behavior Foundation Models that requires neither a complete reward function nor an…
tags:
  - "ICLR 2026 Oral"
  - "Reinforcement Learning"
  - "behavior foundation models"
  - "task inference"
  - "zero-shot RL"
  - "successor features"
  - "UCB"
  - "linear bandits"
date: 2026-05-08
content_hash: 5b78696e0a71f1af
---

# Optimistic Task Inference for Behavior Foundation Models

**Conference**: ICLR 2026 Oral
**arXiv**: [2510.20264](https://arxiv.org/abs/2510.20264)  
**Code**: [GitHub](https://github.com/ThomasRupf/opti-bfm)  
**Area**: Reinforcement Learning / Foundation Models / Zero-shot RL
**Keywords**: behavior foundation models, task inference, zero-shot RL, successor features, UCB, linear bandits

## TL;DR
This paper proposes OpTI-BFM — a test-time task inference method for Behavior Foundation Models that requires neither a complete reward function nor an annotated dataset, and recovers oracle performance within approximately 5 episodes of environment interaction. The core insight is to exploit the linear structure of successor features to reduce task inference to a linear bandit problem, employing a UCB strategy for optimistic exploration in task embedding space, with formal regret guarantees.

## Background & Motivation
**Background**: Behavior Foundation Models (BFMs) leverage Universal Successor Features (USFs) for zero-shot RL — during pretraining, a family of policies $(\pi_z)_{z \in \mathcal{Z}}$ is learned, and at test time the optimal policy $\pi_{z_r}$ can be retrieved immediately given a reward function $r$.

**Limitations of Prior Work**:
   - Standard task inference (Eq. 4) requires computing $z_r = \text{Cov}(\phi)^{-1} \mathbb{E}[\phi(s) r(s)]$ over an inference dataset $\mathcal{D}$, which demands (i) access to an inference dataset and (ii) reward labels for each state.
   - Pretraining data may be unavailable or proprietary; annotating rewards from pixels is costly.
   - Existing methods (e.g., FB) use 50K annotated samples for task inference by default.

**Key Challenge**: BFM zero-shot policy retrieval is computationally efficient (requiring only a single linear regression) but data-annotation-inefficient (requiring large amounts of labeled data for task inference).

**Core Idea**: Transform task inference from offline regression to online interaction — exploiting the linear structure $V^{\pi_z}(s) = z^\top \psi^{\pi_z}(s)$ to reduce policy selection to a linear bandit, and performing optimistic exploration in task embedding space via UCB.

## Method

### Overall Architecture
A pretrained BFM (providing policy family $\pi_z$, successor features $\psi$, and features $\phi$) → At test time: maintain a posterior estimate $\hat{z}_t$ and confidence set $\mathcal{C}_t$ for $z_r$ → Select the optimistic $z_t$ at each step (optimal under the upper confidence bound over $\mathcal{C}_t$) → Execute $\pi_{z_t}$ to collect states and rewards → Update $\hat{z}_t$ → Repeat until convergence.

### Key Designs

1. **Online Task Embedding Estimation**

    - Function: Maintain a regularized least-squares estimate of $z_r$.
    - Mechanism: $\hat{z}_t = V_t^{-1} \sum_{i=0}^t \phi(s_i) r_i$, where $V_t = \lambda I_d + \sum_{i=0}^t \phi(s_i) \phi(s_i)^\top$. Updated after each new interaction $(s_t, r_t)$.
    - Confidence set: $\mathcal{C}_t = \{z : \|z - \hat{z}_t\|_{V_t} \leq \beta_t\}$, an ellipsoidal confidence region.

2. **Optimistic Task Selection (UCB)**

    - Function: Select the optimistic task embedding among all plausible candidates.
    - Mechanism: $z_t = \arg\max_{z \in \mathcal{C}_t} \psi^{\pi_z}(s_t)^\top z$ — selects the task embedding within the confidence set that maximizes predicted return at the current state.
    - Key distinction from linear bandits: Two distinct "contexts" are involved — $\phi$ (for regression and confidence interval estimation) and $\psi$ (for the acquisition function). Regressing on $\phi$ yields tighter estimates.
    - Design Motivation: Optimism naturally balances exploration and exploitation — exploring (selecting informative policies) under high uncertainty, and exploiting (selecting the estimated best policy) under low uncertainty.

3. **Regret Bound**

    - Function: Formally bound the regret of OpTI-BFM.
    - Core Result: For a perfectly trained BFM (satisfying the linear structure), the regret after $n$ episodes is $R_n = \tilde{O}(\sqrt{n \cdot d})$, where $d$ is the task embedding dimension.
    - Key Assumptions: Perfectly trained BFM (accurate SFs, optimal policies) + reward lies within the linear span of the features.
    - Design Motivation: By formalizing the connection to the classical LinUCB algorithm, the method directly inherits theoretical guarantees from the linear bandit literature.

4. **Thompson Sampling Variant (OpTI-BFM-TS)**

    - Avoids the optimization problem by directly sampling from the posterior: $z_t \sim \mathcal{N}(\hat{z}_t, \beta_t V_t^{-1})$.
    - Simpler but slightly lower performance (a notable gap from UCB on Cheetah).

### Loss & Training
- Pretraining: Standard Forward-Backward (FB) framework.
- Test-time inference: Purely online; no additional training required. Each step only updates $V_t$ and $\hat{z}_t$ (matrix-vector operations).
- Computational overhead: Approximately 5× slower than pure policy inference (Table 1), but the absolute time cost remains small.

## Key Experimental Results

### Main Results (DMC Walker / Cheetah / Quadruped, 4 tasks each)

| Method | Episodes to Recover Oracle Performance | Annotation Requirement |
|--------|---------------------------------------|------------------------|
| Oracle (offline inference) | — | 50K annotated samples |
| **OpTI-BFM** | **~5 episodes (5K steps)** | **5K online rewards** |
| OpTI-BFM-TS | ~8 episodes | 8K online rewards |
| LoLA (black-box policy search) | ~30+ episodes | 30K+ |
| Random | Does not converge | — |

OpTI-BFM matches offline inference requiring 50K annotated samples using 10× fewer annotations.

### Ablation Study

| Configuration | Result | Notes |
|---------------|--------|-------|
| Per-step update vs. per-episode update | Per-step update converges faster | Per-episode update has theoretical guarantees |
| OpTI-BFM vs. RND (task-agnostic exploration) | OpTI-BFM is more data-efficient | Demonstrates value of task-aware exploration |
| OpTI-BFM vs. random baseline | Active data collection greatly outperforms passive | Validates the importance of active collection |
| Non-stationary rewards (decay $\rho < 1$) | Can track changing rewards | Requires an appropriate forgetting factor |
| Warm-start ($n$ offline annotated samples) | Initial performance improves rapidly with $n$ | Compatible with offline data |

### Key Findings
- Oracle performance is recovered within 5 episodes across all environments and tasks, a direct consequence of fully exploiting the BFM linear structure.
- LoLA (black-box search) ignores the linear structure and thus converges more than 6× slower, demonstrating the value of structure-aware exploration.
- OpTI-BFM naturally extends to non-stationary rewards (via a forgetting factor $\rho$) and warm-start settings (via offline data pre-initialization).
- Computational overhead is manageable (~5×); the primary bottleneck is environment interaction rather than inference itself.

## Highlights & Insights
- **Eliminates the most critical practical bottleneck of BFMs**: Prior "zero-shot" BFMs required 50K annotated samples for task inference — a substantial data demand that severely limits real-world deployment. OpTI-BFM reduces this requirement from 50K to ~5K (online, without pre-stored datasets).
- **Elegant theoretical connection**: Task inference in BFMs is precisely reduced to the classical linear bandit problem, inheriting a rich set of theoretical tools (regret bounds, ellipsoidal confidence sets, the elliptical potential lemma).
- **Practical unified interface**: OpTI-BFM supports three modes — (1) purely online with no prior knowledge, (2) warm-start with a small amount of offline data, and (3) non-stationary reward tracking — making it a unified framework for BFM deployment.

## Limitations & Future Work
- Relies on the linear SF structure of BFMs — nonlinear reward functions fall outside the scope of the theoretical guarantees.
- Theoretical analysis assumes a perfectly trained BFM (accurate SFs and optimal policies); how approximation errors propagate in practice requires further analysis.
- The regret bound covers only episode-level updates, whereas experiments show that step-level updates perform better — a theory-practice gap remains.
- Evaluation is limited to DMC environments (Walker/Cheetah/Quadruped); more complex high-dimensional visual settings are not tested.
- The dimension $d$ of the task embedding space $\mathcal{Z}$ affects regret — high-dimensional embedding settings may require substantially more exploration.

## Related Work & Insights
- **vs. Standard BFM task inference (FB)**: Requires 50K annotated samples and a pre-stored dataset; OpTI-BFM requires only ~5 episodes of online interaction.
- **vs. LoLA (Sikchi 2025)**: Black-box policy search that ignores the linear structure, converging more than 6× slower.
- **vs. LinUCB (Abbasi-Yadkori 2011)**: The theoretical foundation of OpTI-BFM, extended to the novel BFM setting where regression and acquisition use distinct contexts.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The connection between BFM task inference and linear bandits is remarkably elegant, directly addressing the core practical bottleneck of BFMs.
- Experimental Thoroughness: ⭐⭐⭐⭐ Standard benchmarks with diverse ablations (non-stationarity, warm-start, data efficiency comparisons).
- Writing Quality: ⭐⭐⭐⭐⭐ Theoretical motivation is clear, with a smooth progression from intuition to formal derivation.
- Value: ⭐⭐⭐⭐⭐ The ICLR Oral distinction is well deserved; the work substantially advances the practical deployment of BFMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Efficient Estimation of Kernel Surrogate Models for Task Attribution](efficient_estimation_of_kernel_surrogate_models_for_task_attribution.md)
- [\[ICLR 2026\] ARM-FM: Automated Reward Machines via Foundation Models for Compositional Reinforcement Learning](arm-fm_automated_reward_machines_via_foundation_models_for_compositional_reinfor.md)
- [\[NeurIPS 2025\] Self-Improving Embodied Foundation Models](../../NeurIPS2025/reinforcement_learning/self-improving_embodied_foundation_models.md)
- [\[NeurIPS 2025\] Foundation Models as World Models: A Foundational Study in Text-Based GridWorlds](../../NeurIPS2025/reinforcement_learning/foundation_models_as_world_models_a_foundational_study_in_text-based_gridworlds.md)
- [\[ICLR 2026\] One Model for All Tasks: Leveraging Efficient World Models in Multi-Task Planning](one_model_for_all_tasks_leveraging_efficient_world_models_in_multi-task_planning.md)

</div>

<!-- RELATED:END -->
