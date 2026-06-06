---
title: >-
  [Paper Note] DR.Q: Debiased Model-based Representations for Sample-efficient Continuous Control
description: >-
  [ICML 2026][Reinforcement Learning][Model-based representation] DR.Q builds upon the MR.Q "model-based representation + actor-critic" framework with two key additions: (1) explicitly maximizes the mutual information betw…
tags:
  - "ICML 2026"
  - "Reinforcement Learning"
  - "Model-based representation"
  - "Mutual Information"
  - "InfoNCE"
  - "faded PER"
  - "primacy bias"
date: 2026-05-08
content_hash: 0942a5cc10f11c4f
---

# DR.Q: Debiased Model-based Representations for Sample-efficient Continuous Control

**Conference**: ICML 2026  
**arXiv**: [2605.11711](https://arxiv.org/abs/2605.11711)  
**Code**: <https://github.com/dmksjfl/DR.Q>  
**Area**: Reinforcement Learning / off-policy actor-critic / Representation Learning  
**Keywords**: Model-based representation, Mutual Information, InfoNCE, faded PER, primacy bias

## TL;DR
DR.Q builds upon the MR.Q "model-based representation + actor-critic" framework with two key additions: (1) explicitly maximizes the mutual information between $z_{sa}$ and the next state representation $z_{s'}$ using InfoNCE; (2) introduces "faded prioritized replay," a fusion of "PER × forget," to mitigate overfitting to early experiences. With a single hyperparameter set, DR.Q outperforms strong baselines such as SimBaV2, MR.Q, and TDMPC2 across 73 continuous control tasks.

## Background & Motivation

**Background**: To improve sample efficiency, the community has mainly pursued two directions: (a) model-free approaches addressing value overestimation, replay reuse, and network architecture improvements; (b) learning world models for planning (TDMPC2) or data augmentation (MBPO). Recently, "model-based representation" has emerged as a third path: training state/state-action encoders with model-based objectives to embed latent dynamics, then feeding these to standard actor-critic methods (e.g., TD7, MR.Q).

**Limitations of Prior Work**: Methods like MR.Q enforce latent space consistency via $\min \mathbb E[(z_{sa}-z_{s'})^2]$, but **minimizing Euclidean distance does not necessarily increase mutual information** (see Theorem 4.1)—it may merely align redundant dimensions while neglecting critical ones. Additionally, uniform sampling or PER can induce primacy bias, causing representations to overfit early experiences.

**Key Challenge**: The objectives for representation learning ("geometric proximity vs. informational alignment") and sampling strategies ("TD error importance vs. recency") have evolved independently, each introducing bias that ultimately hampers actor-critic performance.

**Goal**: (1) Upgrade "latent dynamics consistency" from purely geometric to "geometric + mutual information," with supporting lemmas; (2) Fuse "importance" and "recency" prioritization signals into a unified sampling probability formula; (3) Cover 73 tasks with a single hyperparameter set.

**Key Insight**: MR.Q implicitly assumes that small MSE implies high mutual information, which is refuted by Theorem 4.1; this work unifies Wang/Kang's forget mechanism and Schaul's PER into a single formula.

**Core Idea**: Replace MR.Q's implicit mutual information assumption with explicit InfoNCE; use faded PER $P(i)\propto |\delta(i)|^\alpha (1-\epsilon)^i$ to simultaneously suppress the negative effects of "old + unimportant" samples.

## Method

### Overall Architecture
Follows MR.Q's two-stage approach: (a) train encoder $f_\omega:s\to z_s$, $g_\omega:(z_s,a)\to z_{sa}$, and a linear MDP predictor $M(z_{sa})\to (\hat r,\hat z_{s'})$; (b) train a deterministic policy $\pi_\phi$ and clipped double Q $Q_{\theta_{1,2}}$ on $z_s, z_{sa}$. The encoder is trained on rollout sequences of length $H$ with reward CE loss, dynamics MSE, and InfoNCE; sampling uses faded PER.

### Key Designs

1. **InfoNCE Mutual Information Loss (Equation 8)**:

    - **Function**: Explicitly increases the lower bound of mutual information between $z_{sa}$ and the target network's $\tilde z_{s'}$, ensuring not only "numerical proximity" but also "informational alignment."
    - **Mechanism**: Treats $N$ samples in a batch as negatives for each other, using cosine similarity for contrastive learning: $\mathcal L_I=-\frac1N\sum_i\log\frac{\exp(\cos(\hat z_{s'_i},\tilde z_{s'_i})/\tau)}{\sum_k\exp(\cos(\hat z_{s'_i},\tilde z_{s'_k})/\tau)}$. This is equivalent to $I(\hat Z_{s'};\tilde Z_{s'})\ge \log N - \mathcal L_I$. Theorem 4.1 shows that minimizing $\|Z_{sa}-Z_{s'}\|^2$ does not necessarily increase mutual information; Lemma 4.2 further proves that increasing $I$ reduces $H(Z_{s'}|Z_{sa})$, making latent dynamics more deterministic.
    - **Design Motivation**: Directly eliminates "spurious alignment" in MR.Q representations—ensuring more accurate latent dynamics models and tighter value-error bounds (connecting to DeepMDP/MR.Q theory).

2. **Faded Prioritized Experience Replay (Equation 4)**:

    - **Function**: Considers both "TD-error importance" and "recency" of transitions, avoiding overuse of old experiences (primacy bias) and neglect of high TD-error old samples.
    - **Mechanism**: $P(i)=\frac{|\delta(i)|^\alpha (1-\epsilon)^i}{\sum_j |\delta(j)|^\alpha (1-\epsilon)^j}$, where $i=0$ is the newest transition; implemented with LAP-improved PER and a lower truncation $\epsilon_\mathrm{low}$ to prevent valuable old experiences from being discarded. Theorem 4.3 proves that, for equal TD-error, older samples have strictly lower probability, and total sampling count is bounded.
    - **Design Motivation**: PER alone can anchor the policy to early high-TD-error transitions (primacy bias), while forget alone ignores infrequent but informative transitions; their product yields a composite metric of "importance × freshness."

3. **Full Encoder Loss and Actor-Critic Configuration**:

    - **Function**: Integrates three losses (reward CE, dynamics MSE, InfoNCE) with actor-critic's CDQ and multi-step Q, enabling a unified hyperparameter across 73 tasks.
    - **Mechanism**: $\mathcal L^\mathrm{DR.Q}_\mathrm{enc}=\sum_{t=1}^H \lambda_r \mathcal L_\mathrm{reward} + \lambda_d \mathcal L_\mathrm{dynamics} + \lambda_m \mathcal L_I$; critic uses Huber loss, multi-step return (horizon $H_Q$), and clipped double Q; actor adds Gaussian noise and clipped exploration. Target networks are updated periodically.
    - **Design Motivation**: The authors deliberately avoid common tricks such as normalization, parameter reset, or hidden regularization, demonstrating that "good representation + good sampling" suffices, making the method simpler and more reusable.

### Loss & Training

- Reward loss: two-hot encoding + symexp intervals + CE.
- Dynamics loss: $\mathcal L_\mathrm{dynamics}=\mathbb E[(\hat z_{s'}-\mathrm{SG}(\tilde z_{s'}))^2]$, with stop-gradient to prevent target encoder drift.
- InfoNCE: as in Equation 8, with temperature $\tau$.
- Total loss is a weighted sum of the three, with weights $\lambda_r,\lambda_d,\lambda_m$ unified across all tasks.
- Replay Ratio (UTD) = 1, more efficient than high-UTD methods like SimBaV2/FoG.

## Key Experimental Results

### Main Results (73 tasks, 10 seeds, single hyperparameter; summarized from Figure 1)

| Baseline | #Tasks | Key Comparison | Gain |
|---|---|---|---|
| MuJoCo | — | DR.Q vs MR.Q / SimBaV2 / TDMPC2 | Matches or surpasses |
| DMC-Hard (7 dog/humanoid) | 7 | DR.Q vs SimBaV2 | +15.5% |
| DMC-Visual | — | DR.Q vs MR.Q | +26.8% |
| HumanoidBench (w/ hand) | 14 | DR.Q vs FoG | +58.9% |
| All 73 | 73 | DR.Q matches or outperforms MR.Q | Consistent lead |

DR.Q is the first algorithm to push the average return of the dog-run task above 700 within 1M environment steps.

### Ablation Study (Figure 4, 4 representative tasks, 10 seeds)

| Configuration | Phenomenon | Explanation |
|---|---|---|
| Full DR.Q | Best sample efficiency and asymptotic performance | InfoNCE + faded PER synergy |
| w/o InfoNCE ($\lambda_m=0$) | Significant drop on high-dimensional HumanoidBench tasks | Mutual information constraint is crucial for redundant state spaces |
| DR.Q (only forget) | Curve collapses without PER | Important samples are drowned out without TD-error prioritization |
| DR.Q (only LAP) | Curve collapses without forget | Overfitting to early experiences, primacy bias emerges |
| Without InfoNCE | At least matches MR.Q | DR.Q degrades gracefully to MR.Q |

### Key Findings
- The benefit of InfoNCE is especially pronounced in **high-dimensional redundant states** (e.g., HumanoidBench with dexterous hand), as it forces representations to encode task-relevant signals and suppress redundant dimensions.
- PER and forget **must be combined**: either alone underperforms the full method, confirming that "importance × freshness" is a genuinely complementary axis.
- Achieving cross-task robustness with a single hyperparameter is rare, demonstrating DR.Q's robustness and correcting the RL benchmarking culture (many SOTA results rely on task-specific tuning).

## Highlights & Insights
- **Theory + Experiment + Engineering Closed Loop**: Theorem 4.1 exposes MR.Q's implicit assumption, Lemma 4.2 links mutual information to conditional entropy and value error bounds, and InfoNCE operationalizes this, forming a tightly connected chain.
- The faded PER formula is extremely simple yet captures both priors, and Theorem 4.3 rigorously proves that "old sample probability strictly decreases"—a rare example of an "engineering trick" with a formal theorem.
- Deliberately avoiding tricks (no normalization, reset, or hidden regularization) yet outperforming others suggests that "less is more" still holds for representation learning-driven RL.

## Limitations & Future Work
- Underperforms baseline on Hopper-v4—an inherent cost of unified hyperparameters; simple dynamics tasks may be hindered by high-dimensional encoders.
- On visual-humanoid-run, DR.Q and all methods fail, limited by the 1M step budget; the encoder cannot learn meaningful representations with such a small budget.
- Not validated on discrete action (Atari) or non-Markovian (POMDP) tasks; hard exploration tasks remain untested.
- InfoNCE introduces implicit dependencies on batch size and negative sample quality; the paper does not provide ablation for these.

## Related Work & Insights
- **vs MR.Q (Fujimoto et al. 2025)**: Shares the same framework, but MR.Q only minimizes MSE, while DR.Q explicitly adds InfoNCE; MR.Q uses uniform/PER sampling, DR.Q uses faded PER.
- **vs TDMPC2 (Hansen et al. 2024)**: TDMPC2 performs planning in the latent space, while DR.Q focuses on actor-critic learning; the latter is lighter without sacrificing performance.
- **vs SimBaV2 (Lee et al. 2025)**: SimBaV2 relies on "network architecture + high UTD," while DR.Q achieves parity or better with UTD=1 and better representations, suggesting "information density > compute density."
- **vs FoG (Kang et al. 2025)**: FoG uses the forget mechanism alone; DR.Q fuses it with PER into faded PER, achieving greater theoretical and empirical stability.

## Rating
- Novelty: ⭐⭐⭐⭐ Each component alone is not new, but Theorem 4.1 exposes a previously unchallenged assumption and the systematic fusion of PER + forget constitutes a meaningful combinatorial innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 73 tasks × 10 seeds × single hyperparameter, covering HumanoidBench/DMC/MuJoCo, with ablations for both components—evidence is very solid.
- Writing Quality: ⭐⭐⭐⭐ Tight integration of theory and experiment, though formulas are dense; Figures 1/3/4 are intuitive.
- Value: ⭐⭐⭐⭐ Provides a clear upgrade for the "model-based representation" paradigm; open-source code and single hyperparameter make it directly usable for industrial RL teams.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] WIMLE: Uncertainty-Aware World Models with IMLE for Sample-Efficient Continuous Control](../../ICLR2026/reinforcement_learning/wimle_uncertainty-aware_world_models_with_imle_for_sample-efficient_continuous_c.md)
- [\[ICLR 2026\] Sample-efficient and Scalable Exploration in Continuous-Time RL](../../ICLR2026/reinforcement_learning/sample-efficient_and_scalable_exploration_in_continuous-time_rl.md)
- [\[AAAI 2026\] Test-driven Reinforcement Learning in Continuous Control](../../AAAI2026/reinforcement_learning/test-driven_reinforcement_learning_in_continuous_control.md)
- [\[ICLR 2026\] The Sample Complexity of Online Reinforcement Learning: A Multi-Model Perspective](../../ICLR2026/reinforcement_learning/the_sample_complexity_of_online_reinforcement_learning_a_multi-model_perspective.md)
- [\[NeurIPS 2025\] Succeed or Learn Slowly: Sample Efficient Off-Policy Reinforcement Learning for Mobile App Control](../../NeurIPS2025/reinforcement_learning/succeed_or_learn_slowly_sample_efficient_off-policy_reinforcement_learning_for_m.md)

</div>

<!-- RELATED:END -->
