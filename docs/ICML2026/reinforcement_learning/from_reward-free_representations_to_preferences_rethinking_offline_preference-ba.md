---
title: >-
  [Paper Note] From Reward-Free Representations to Preferences: Rethinking Offline Preference-Based Reinforcement Learning
description: >-
  [ICML 2026][Reinforcement Learning][PbRL] This paper reformulates offline Preference-based Reinforcement Learning (PbRL) within the Forward-Backward (FB) representation space. It demonstrates that the standard Bradley-Te…
tags:
  - "ICML 2026"
  - "Reinforcement Learning"
  - "PbRL"
  - "Forward-Backward Representation"
  - "Contrastive Learning"
  - "Zero-Shot RL"
  - "Successor Measure"
date: 2026-05-08
content_hash: 5b2aae10926425bf
---

# From Reward-Free Representations to Preferences: Rethinking Offline Preference-Based Reinforcement Learning

**Conference**: ICML 2026  
**arXiv**: [2606.01123](https://arxiv.org/abs/2606.01123)  
**Code**: https://github.com/rl-bandits-lab/FB-PbRL (Available)  
**Area**: Reinforcement Learning / Preference Learning  
**Keywords**: PbRL, Forward-Backward Representation, Contrastive Learning, Zero-Shot RL, Successor Measure

## TL;DR
This paper reformulates offline Preference-based Reinforcement Learning (PbRL) within the Forward-Backward (FB) representation space. It demonstrates that the standard Bradley-Terry (BT) preference loss is equivalent to the SimCLR contrastive loss under the FB framework. Consequently, the authors propose FB-PbRL: pre-training FB representations on reward-free offline data, followed by searching for a task vector $\boldsymbol{z}^\star$ and fine-tuning the representation using a contrastive objective on preference data. This pipeline eliminates the need for training any explicit reward or preference models.

## Background & Motivation
**Background**: Standard practice in offline PbRL involves two stages: first, training a reward model $r_{\boldsymbol{\psi}}$ from paired preference data $(\sigma^{(1)},\sigma^{(2)},y)$ using a BT model (minimizing $\mathcal{L}(\boldsymbol{\psi})=-\mathbb{E}[\mathbb{I}(y=1)\log P_{\boldsymbol{\psi}}(\sigma^{(1)}\succ\sigma^{(2)})+\ldots]$), then applying off-the-shelf offline RL algorithms; alternatively, learning a preference model directly.

**Limitations of Prior Work**: Human preferences are expensive, with typical budgets limited to a few thousand pairs. This leads to failures in both standard paths: reward models suffer from over-optimization and poor generalization (Fig. 2 shows BT rewards collapsing to the mean), while direct preference models often underfit. On low-quality datasets like ExORL, these methods struggle to learn effectively.

**Key Challenge**: PbRL suffers from overfitting under scarce supervision, whether learning rewards or preferences. Conversely, Reward-Free Representation Learning (RFRL) frameworks (FB, Laplacian, HILP, PSM) can learn general-purpose representations from reward-free data to provide near-optimal policies for any reward function $r(s,a)$ at test-time by assembling a task vector $\boldsymbol{z}_r=\mathbb{E}[\mathbf{B}_\omega(s,a)r(s,a)]$. However, PbRL scenarios lack reward functions and only provide preferences.

**Goal**: How can RFRL representations be utilized for PbRL without reward supervision? This is decomposed into: (a) How to derive the task vector $\boldsymbol{z}$ directly from preference data? (b) How to adapt pre-trained, task-agnostic representations to specific preference-defined tasks?

**Key Insight**: The authors discovered that in the FB framework, if the reward is linearly representable by the backward representation ($r_{\psi}(s,a)=\mathbf{B}_{\bar\omega}(s,a)^\top\boldsymbol{\psi}$) and the backward representation satisfies orthonormality ($\mathbf{H}_\mathbf{B}\approx\mathbf{I}_d$), the BT preference loss can be analytically rewritten as a SimCLR-style loss for $\boldsymbol{z}$. This replaces "reward learning" with "contrastive retrieval in the FB latent space."

**Core Idea**: Instead of learning reward or preference models, the approach transforms preference optimization into contrastive learning over frozen FB backward representations, followed by a fine-tuning step to align the pre-trained FB geometry with the specific preference task, thereby bypassing reward over-optimization.

## Method

### Overall Architecture
FB-PbRL consists of two stages, taking reward-free offline data $\mathcal{D}$ and paired preference data $\mathcal{D}_{\text{pref}}$ as input:

1.  **RFRL Pre-training**: Use the FB framework to decompose the successor measure into $\mathcal{M}^{\pi_r^*}(s,a,\{(s',a')\})=\mathbf{F}_\theta(s,a,\boldsymbol{z}_r)^\top\mathbf{B}_\omega(s',a')$. Learn $\mathbf{F}, \mathbf{B}$ and a conditional policy $\pi(\cdot\mid s,\boldsymbol{z})$ on $\mathcal{D}$ using measure and orthonormality losses (fully unsupervised).
2.  **Preference-guided search + fine-tune**: Alternatingly perform: (i) Contrastive Preference Task Search (CPTS) for the anchor vector $\boldsymbol{z}^\star$, and (ii) Preference-Guided Fine-Tuning (PG-FT) of $\mathbf{F}, \mathbf{B}$ using $\boldsymbol{z}^\star$ as an anchor to align the latent geometry. Evaluation is performed using $\pi(\cdot\mid s,\boldsymbol{z}^\star)$.

The process never explicitly constructs rewards. $\boldsymbol{z}^\star$ is a low-dimensional vector (typically $d \sim$ hundreds), making optimization significantly cheaper than training high-capacity models.

### Key Designs

1.  **CPTS: Redefining BT Loss as SimCLR in FB Space**:
    *   **Function**: Searches for the task vector $\boldsymbol{z}_{\text{CPTS}}^\star$ directly from preference data on frozen FB representations.
    *   **Mechanism**: Define the latent representation of a segment $\sigma$ as $\mathbf{B}_{\bar\omega}(\sigma):=\tfrac{1}{k}\sum_i \mathbf{B}_{\bar\omega}(s_i,a_i)$. Given linear reward realizability and orthonormality, $\boldsymbol{\psi}=\mathbf{H}_\mathbf{B}^{-1}\boldsymbol{z}_{\boldsymbol{\psi}}$. Substituting this into the BT loss yields $\mathcal{L}_{\text{pref}}(\boldsymbol{z};\bar\omega)=-\mathbb{E}[\log\frac{\exp(\boldsymbol{z}^\top\boldsymbol{z}_\sigma^+)}{\exp(\boldsymbol{z}^\top\boldsymbol{z}_\sigma^+)+\exp(\boldsymbol{z}^\top\boldsymbol{z}_\sigma^-)}]$, which is exactly the SimCLR loss.
    *   **Design Motivation**: Reward over-optimization is the primary failure mode of PbRL under sparse feedback. Moving the objective to the FB latent space allows searching for a minimizer of a low-dimensional convex objective, avoiding the overfitting of high-capacity networks.

2.  **PG-FT: Fine-tuning FB Latent Space with $\boldsymbol{z}^\star$**:
    *   **Function**: Overcomes the task-agnostic nature of the pre-trained $\boldsymbol{z}$ prior, as the retrieved $\boldsymbol{z}_{\text{CPTS}}^\star$ might reside far from the $\boldsymbol{z}_\sigma$ clusters induced by preference data.
    *   **Mechanism**: Alternatingly update $\boldsymbol{z}^\star$ via $\nabla_{\boldsymbol{z}}\mathcal{L}_{\text{pref}}$ and fine-tune $\mathbf{F}_\theta, \mathbf{B}_\omega$ using $\mathcal{L}_m(\theta,\omega;\boldsymbol{z}^\star)+\lambda\mathcal{L}_{\text{ortho}}(\omega)+\alpha\mathcal{L}_{\text{pref}}(\omega;\boldsymbol{z}^\star)$.
    *   **Design Motivation**: General RFRL representations are jack-of-all-trades but master-of-none; PG-FT uses preference signals as "instructions" to reshape the latent geometry to be reward-aligned.

3.  **Collaborative Alternating Optimization**:
    *   **Function**: Merges standard FB objectives with contrastive preference objectives to preserve geometric constraints while ensuring alignment.
    *   **Mechanism**: The measure loss $\mathcal{L}_m$ handles the Bellman residual of the successor measure; $\mathcal{L}_{\text{ortho}}$ ensures $\mathbf{H}_\mathbf{B}\approx\mathbf{I}_d$. The algorithm cycles through updating the measure/orthonormality with transitions, updating $\mathbf{B}_\omega$ and $\boldsymbol{z}^\star$ with preferences, and synchronizing the policy.

### Loss & Training
- **Total Loss**: $\mathcal{L}_m(\theta,\omega;\boldsymbol{z}^\star)+\lambda\mathcal{L}_{\text{ortho}}(\omega)+\alpha\mathcal{L}_{\text{pref}}(\boldsymbol{z}^\star,\omega)$, default $\alpha=100$.
- **Protocols**: Standard PbRL Protocol (2000 preference pairs); Zero-Shot RL Protocol (preferences from 400 trajectories/10k transitions) for fair comparison with RFRL baselines.

## Key Experimental Results

### Main Results
Evaluated on 16 DMC tasks using ExORL RND unsupervised data (low-quality, no rewards). Ours-T refers to CPTS only; Ours-FT is the full method.

**vs Offline PbRL Baselines (PbRL Protocol, average return per domain)**:

| Domain | DPPO | OPPO | OPRL | CLARIFY | LIRE | Ours-T | Ours-FT |
|---|---|---|---|---|---|---|---|
| Cheetah | 202.3 | 200.9 | 276.4 | 271.5 | 313.4 | 344.7 | **621.7** |
| Walker | 242.3 | 247.5 | 253.8 | 248.9 | 232.5 | 533.4 | **762.9** |
| Quadruped | 309.1 | 569.3 | 631.1 | 602.9 | 378.7 | 663.4 | **846.9** |
| Pointmass| 16.3 | 24.1 | 337.5 | 317.8 | 102.3 | 69.1 | **570.8** |

Ours-FT is the best performer across nearly all tasks. Even Ours-T (search only) outperforms most PbRL baselines, indicating that FB representations are inherently more robust to distribution shift.

**vs Zero-Shot RFRL Baselines (Zero-Shot Protocol, average return; Ours uses preferences only)**:

| Domain | Laplace | FB | HILP | PSM | RLDP | Ours-FT |
|---|---|---|---|---|---|---|
| Cheetah | 316.5 | 385.6 | 193.5 | 626.0 | 609.6 | **645.4** |
| Walker | 136.7 | **719.9** | 348.1 | 689.1 | 621.6 | 699.4 |
| Quadruped | 601.2 | 561.7 | 289.8 | 618.7 | 612.8 | **826.3** |

Ours-FT, using only preferences, outperforms RFRL baselines that utilize ground-truth rewards.

### Ablation Study
| Configuration | Cheetah | Walker | Quadruped | Description |
|---|---|---|---|---|
| FB-BT-FT | 536.6 | 600.6 | 714.1 | Integrating BT rewards + FB fine-tuning |
| Ours-FT | **621.7** | **794.5** | **846.9** | Contrastive fine-tuning (Ours) |

### Key Findings
- **Contrastive FT > Reward FT**: Ours-FT consistently outperforms FB-BT-FT, confirming that treating preferences as contrastive signals is superior to learning BT rewards for fine-tuning.
- **Sample Efficiency**: Performance remains stable even when preferences are reduced to 200 pairs.
- **Bypassing Reward Collapse**: Contrastive loss avoids the collapsing modes typical of reward networks under sparse supervision.

## Highlights & Insights
- **Analytical Equivalence**: Establishing the link between BT loss and SimCLR loss within the FB framework is a significant conceptual bridge, connecting PbRL and RFRL.
- **Search + Fine-tune Paradigm**: CPTS provides coarse alignment through low-dimensional search, while PG-FT provides fine alignment through high-dimensional fine-tuning.
- **Engineering Significance**: Providing a viable path that bypasses reward model training is highly relevant for RLHF in LLMs, where reward hacking is a major issue.

## Limitations & Future Work
- The equivalence relies on the specific structural assumptions of the FB framework (linear rewards + orthonormality), which may not hold for other RFRL architectures.
- The pre-training phase is computationally expensive.
- Performance is sensitive to data coverage; tasks with poor coverage and sparse preferences (e.g., Pointmass) show higher variance.

## Related Work & Insights
- **vs DPPO/OPPO**: Unlike these methods that contrast trajectory embeddings directly, FB-PbRL utilizes pre-trained RFRL representations, leading to significantly better performance (3-5x higher).
- **vs OPRL/CLARIFY**: While these use active query selection for efficiency, FB-PbRL achieves high efficiency through representational contrast.
- **vs RFRL (FB/HILP)**: These require rewards at test-time, whereas Ours uses preferences and aligns the representation.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The BT-SimCLR bridge is a strong, previously undiscovered insight.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive coverage across 16 tasks and multiple protocols.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Logical flow from motivation to analytical discovery.
- **Value**: ⭐⭐⭐⭐⭐ High potential impact on both RL and LLM alignment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Safe Reinforcement Learning with Preference-Based Constraint Inference](safe_reinforcement_learning_with_preference-based_constraint_inference.md)
- [\[NeurIPS 2025\] Reward-Aware Proto-Representations in Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/reward-aware_proto-representations_in_reinforcement_learning.md)
- [\[ICML 2026\] Offline Reinforcement Learning with Universal Horizon Models](offline_reinforcement_learning_with_universal_horizon_models.md)
- [\[ICML 2026\] Laplacian Representations for Decision-Time Planning](laplacian_representations_for_decision-time_planning.md)
- [\[ICML 2026\] CPMöbius: Iterative Coach–Player Reasoning for Data-Free Reinforcement Learning](cpmobius_iterative_coach-player_reasoning_for_data-free_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
