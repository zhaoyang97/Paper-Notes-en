---
title: >-
  [Paper Note] DR.Q: Debiased Model-based Representations for Sample-efficient Continuous Control
description: >-
  [ICML 2026][Reinforcement Learning][Model-based representation] DR.Q builds upon the MR.Q framework (model-based representation + actor-critic) by introducing two key components: explicit maximization of mutual informati…
tags:
  - "ICML 2026"
  - "Reinforcement Learning"
  - "Model-based representation"
  - "mutual information"
  - "InfoNCE"
  - "faded PER"
  - "primacy bias"
date: 2026-05-08
content_hash: 51c05e567ba18119
---

# DR.Q: Debiased Model-based Representations for Sample-efficient Continuous Control

**Conference**: ICML 2026  
**arXiv**: [2605.11711](https://arxiv.org/abs/2605.11711)  
**Code**: <https://github.com/dmksjfl/DR.Q>  
**Area**: Reinforcement Learning / off-policy actor-critic / Representation Learning  
**Keywords**: Model-based representation, mutual information, InfoNCE, faded PER, primacy bias

## TL;DR
DR.Q builds upon the MR.Q framework (model-based representation + actor-critic) by introducing two key components: explicit maximization of mutual information (MI) between $z_{sa}$ and next-state representation $z_{s'}$ using InfoNCE, and a "faded prioritized replay" that fuses PER with a forgetting mechanism to alleviate over-fitting on early experiences. It outperforms strong baselines like SimBaV2, MR.Q, and TDMPC2 across 73 continuous control tasks using a single set of hyperparameters.

## Background & Motivation

**Background**: To improve sample efficiency, the community currently follows two main paths: (a) model-free methods focusing on value-overestimation mitigation, replay reuse, and architecture improvements; (b) world models for planning (TDMPC2) or data augmentation (MBPO). Recently, "model-based representation" has emerged as a third path: training state/state-action encoders with model-based objectives to embed latent dynamics into representations for standard actor-critic algorithms (TD7, MR.Q).

**Limitations of Prior Work**: Methods like MR.Q use $\min \mathbb E[(z_{sa}-z_{s'})^2]$ for latent space consistency. However, **minimizing Euclidean distance does not necessarily increase mutual information** (Theorem 4.1)—it might simply align redundant dimensions while critical ones are ignored. Furthermore, uniform sampling or standard PER are prone to primacy bias, causing representations to overfit early experiences.

**Key Challenge**: Objectives for representation learning ("geometric proximity vs. informational alignment") and sampling strategies ("priority by TD-error vs. recency") have evolved separately, yet both introduce biases that ultimately hinder actor-critic performance.

**Goal**: (1) Upgrade "latent dynamics consistency" from pure geometry to "geometry + mutual information" with theoretical justification; (2) Integrate "importance" and "recency" prioritization signals into a single sampling formula; (3) Maintain performance across 73 tasks with unified hyperparameters.

**Key Insight**: MR.Q's implicit assumption that small MSE equals high MI is refuted by Theorem 4.1. By combining the forgetting mechanism of Wang/Kang with Schaul’s PER into a unified formula, both biases can be addressed simultaneously.

**Core Idea**: Replace MR.Q's implicit MI assumption with an explicit InfoNCE loss and use faded PER $P(i)\propto |\delta(i)|^\alpha (1-\epsilon)^i$ to suppress both "old" and "unimportant" negative signals.

## Method

### Overall Architecture
The framework follows the two-stage approach of MR.Q: (a) Train encoder $f_\omega:s\to z_s$, $g_\omega:(z_s,a)\to z_{sa}$, and a linear MDP predictor $M(z_{sa})\to (\hat r,\hat z_{s'})$; (b) Train a deterministic policy $\pi_\phi$ and clipped double Q $Q_{\theta_{1,2}}$ on $z_s$ and $z_{sa}$. The encoder is optimized using reward CE loss, dynamics MSE, and InfoNCE over rollouts of length $H$. Sampling is performed via faded PER.

### Key Designs

1. **InfoNCE Mutual Information Loss (Equation 8)**:
    - **Function**: Explicitly raises the lower bound of MI between $z_{sa}$ and the target network's $\tilde z_{s'}$, ensuring both numerical proximity and informational alignment.
    - **Mechanism**: Treats $N$ samples in a batch as mutual negative samples and performs contrastive learning via cosine similarity: $\mathcal L_I=-\frac1N\sum_i\log\frac{\exp(\cos(\hat z_{s'_i},\tilde z_{s'_i})/\tau)}{\sum_k\exp(\cos(\hat z_{s'_i},\tilde z_{s'_k})/\tau)}$. This implies $I(\hat Z_{s'};\tilde Z_{s'})\ge \log N - \mathcal L_I$. Theorem 4.1 proves that minimizing $\|Z_{sa}-Z_{s'}\|^2$ alone does not guarantee increased MI; Lemma 4.2 further proves that increasing $I$ leads to a decrease in $H(Z_{s'}|Z_{sa})$, making latent dynamics more deterministic.
    - **Design Motivation**: Eliminates "spurious alignment" in MR.Q representations, ensuring more accurate latent dynamics models and tighter value-error bounds.

2. **Faded Prioritized Experience Replay (Equation 4)**:
    - **Function**: Simultaneously considers the "TD-error importance" and "temporal recency" of transitions to avoid primacy bias from old experiences while preventing the neglect of high TD-error older samples.
    - **Mechanism**: $P(i)=\frac{|\delta(i)|^\alpha (1-\epsilon)^i}{\sum_j |\delta(j)|^\alpha (1-\epsilon)^j}$, where $i=0$ is the latest transition. Implementation-wise, it improves upon LAP and uses $\epsilon_\mathrm{low}$ to truncate the forget weight, preventing valuable old experiences from being zeroed out. Theorem 4.3 proves that given equal TD-errors, older samples have strictly lower probabilities, and the total sampling count is bounded by a constant.
    - **Design Motivation**: Standard PER can trap the policy on early high-TD-error transitions (primacy bias), while pure forgetting ignores informative low-frequency transitions. The product provides a composite metric of "important and fresh."

3. **Integrated Encoder Loss and Actor-Critic Configuration**:
    - **Function**: Bridges three losses (reward CE, dynamics MSE, InfoNCE) with actor-critic CDQ and multi-step Q, ensuring unified hyperparameters across 73 tasks.
    - **Mechanism**: $\mathcal L^\mathrm{DR.Q}_\mathrm{enc}=\sum_{t=1}^H \lambda_r \mathcal L_\mathrm{reward} + \lambda_d \mathcal L_\mathrm{dynamics} + \lambda_m \mathcal L_I$. The critic uses Huber loss with multi-step returns (horizon $H_Q$) and clipped double Q; the actor employs Gaussian noise and clipping for exploration. Target networks are updated periodically.
    - **Design Motivation**: Avoids common tricks like normalization, parameter reset, or hidden regularization to prove that "good representation + good sampling" is sufficient, making the method concise and reusable.

### Loss & Training
- Reward loss: Two-hot encoding + symexp spacing + CE.
- Dynamics loss: $\mathcal L_\mathrm{dynamics}=\mathbb E[(\hat z_{s'}-\mathrm{SG}(\tilde z_{s'}))^2]$, using stop-gradient to prevent target encoder drift.
- InfoNCE: Equation 8 as defined above with temperature $\tau$.
- Total loss is a weighted sum with $\lambda_r, \lambda_d, \lambda_m$ unified across all tasks.
- Replay Ratio (UTD) = 1, which is more efficient than high-UTD methods like SimBaV2 or FoG.

## Key Experimental Results

### Main Results (73 tasks, 10 seeds, unified hyperparameters; summarized from Figure 1)

| Benchmark | No. of Tasks | Key Comparison | Gain |
|---|---|---|---|
| MuJoCo | — | DR.Q vs. MR.Q / SimBaV2 / TDMPC2 | Matches or exceeds |
| DMC-Hard (7 dog/humanoid) | 7 | DR.Q vs. SimBaV2 | +15.5% |
| DMC-Visual | — | DR.Q vs. MR.Q | +26.8% |
| HumanoidBench (w/ hand) | 14 | DR.Q vs. FoG | +58.9% |
| All 73 | 73 | DR.Q comprehensively matches or beats MR.Q | Consistently leading |

DR.Q is the first algorithm to push the average return of the dog-run task past 700 within 1M environment steps.

### Ablation Study (Figure 4, 4 representative tasks, 10 seeds)

| Configuration | Observation | Explanation |
|---|---|---|
| Full DR.Q | Optimal sample efficiency and asymptotic performance | Synergy of InfoNCE + faded PER |
| w/o InfoNCE ($\lambda_m=0$) | Significant drop on high-dimensional HumanoidBench tasks | MI constraints are vital when state spaces are redundant |
| DR.Q (only forget) | Curve collapses without PER | Important samples are buried without TD-error priority |
| DR.Q (only LAP) | Curve collapses without forget | Primacy bias emerges as early experiences are overfitted |
| w/o InfoNCE | Performance at least matches MR.Q | DR.Q gracefully degrades to MR.Q |

### Key Findings
- InfoNCE gains are particularly significant in **high-dimensional redundant states** (e.g., HumanoidBench with dexterous hands) because it forces the representation to encode task-relevant signals and suppress redundancy.
- PER and forgetting must be **combined**: either alone performs worse than the full version, proving that "importance × recency" are complementary axes.
- Achieving high performance across 73 tasks with unified hyperparameters is rare, demonstrating DR.Q’s robustness and serving as a correction to task-specific tuning cultures in RL benchmarking.

## Highlights & Insights
- **Theoretical-Empirical Closed Loop**: Theorem 4.1 refutes MR.Q's implicit assumption, Lemma 4.2 links MI to conditional entropy and value error bounds, and InfoNCE provides the practical implementation.
- Faded PER uses a simple formula to reflect two priors simultaneously. Theorem 4.3 provides a strict "strictly decreasing sampling probability for old samples" property—a rare example of a "simple engineering trick with a theoretical proof."
- The deliberate exclusion of common tricks (normalization, resets, hidden reg) suggests that in representation-oriented RL, "less is more" remains valid.

## Limitations & Future Work
- Performance on Hopper-v4 is slightly worse than baselines—the cost of unified hyperparameters; simple dynamics might be hindered by high-dimensional representers.
- DR.Q fails on visual-humanoid-run along with all other methods within the 1M step budget; representations cannot be learned in such a short window.
- Not yet validated on discrete actions (Atari) or non-Markovian (POMDP) tasks; hard exploration tasks was not addressed.
- InfoNCE introduces intra-batch contrast, implying an implicit dependence on batch size and negative sample quality not fully ablated in the paper.

## Related Work & Insights
- **vs. MR.Q (Fujimoto et al. 2025)**: Shares the skeleton, but MR.Q only minimizes MSE, whereas DR.Q adds explicit InfoNCE. MR.Q uses uniform/PER, DR.Q uses faded PER.
- **vs. TDMPC2 (Hansen et al. 2024)**: TDMPC2 performs planning in the latent space; DR.Q focuses on actor-critic learning, which is more lightweight without losing performance.
- **vs. SimBaV2 (Lee et al. 2025)**: SimBaV2 focuses on "architecture + high UTD"; DR.Q matches or exceeds it with UTD=1 and better representations, suggesting "information density > computation density."
- **vs. FoG (Kang et al. 2025)**: FoG's forget mechanism is used in isolation; DR.Q fuses it with PER into faded PER, proving more stable theoretically and empirically.

## Rating
- Novelty: ⭐⭐⭐⭐ While individual components aren't entirely new, refuting prior assumptions with Theorem 4.1 and systematically fusing PER + forget constitutes significant combinatorial innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 73 tasks × 10 seeds × unified hyperparameters across HumanoidBench/DMC/MuJoCo. Ablations are solid.
- Writing Quality: ⭐⭐⭐⭐ Tight connection between theory and experiments, though formulas are somewhat dense. Figures 1/3/4 are intuitive.
- Value: ⭐⭐⭐⭐ Provides a clear upgrade to the "model-based representation" school; open-source code and unified hyperparameters are directly usable for industrial RL teams.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] WIMLE: Uncertainty-Aware World Models with IMLE for Sample-Efficient Continuous Control](../../ICLR2026/reinforcement_learning/wimle_uncertainty-aware_world_models_with_imle_for_sample-efficient_continuous_c.md)
- [\[ICLR 2026\] Sample-efficient and Scalable Exploration in Continuous-Time RL](../../ICLR2026/reinforcement_learning/sample-efficient_and_scalable_exploration_in_continuous-time_rl.md)
- [\[ICML 2026\] Dr. Tulu: Reinforcement Learning with Evolving Rubrics for Deep Research](dr_tulu_reinforcement_learning_with_evolving_rubrics_for_deep_research.md)
- [\[ICML 2026\] Laplacian Representations for Decision-Time Planning](laplacian_representations_for_decision-time_planning.md)
- [\[ICML 2026\] From Reward-Free Representations to Preferences: Rethinking Offline Preference-Based Reinforcement Learning](from_reward-free_representations_to_preferences_rethinking_offline_preference-ba.md)

</div>

<!-- RELATED:END -->
