---
title: >-
  [Paper Note] Implicit Safety Alignment from Crowd Preferences
description: >-
  [ICML 2026][LLM Alignment][Crowdsourced Preferences] Addressing the structure of "diverse user goals but shared safety principles" in crowdsourced preference data…
tags:
  - "ICML 2026"
  - "LLM Alignment"
  - "Crowdsourced Preferences"
  - "Implicit Safety Alignment"
  - "Skill Discovery"
  - "VAE"
  - "Hierarchical Reinforcement Learning"
date: 2026-05-08
content_hash: 9fd405b4a948174d
---

# Implicit Safety Alignment from Crowd Preferences

**Conference**: ICML 2026  
**arXiv**: [2605.21822](https://arxiv.org/abs/2605.21822)  
**Code**: Not publicly available  
**Area**: Alignment RLHF / Safe RL / Preference Learning  
**Keywords**: Crowdsourced Preferences, Implicit Safety Alignment, Skill Discovery, VAE, Hierarchical Reinforcement Learning

## TL;DR
Addressing the structure of "diverse user goals but shared safety principles" in crowdsourced preference data, the authors demonstrate that traditional reward combination is susceptible to majority preference pollution and weight sensitivity. Instead, they propose Safe Crowd Preference-based RL: using a VAE to encode crowdsourced preferences into latent-conditioned low-level skills, followed by training a high-level policy to compose these skills. This reduces downstream costs to near-Oracle levels without explicit safety rewards while maintaining task returns.

## Background & Motivation

**Background**: RLHF has expanded from single-annotator settings to crowdsourced preference scenarios. Most existing works (VPL, MaxMin-RLHF, Personalized Soups) focus on respecting user **differences**—learning distinct rewards or policies for various users. Safe RLHF typically treats safety as a separate category of explicit preference labels.

**Limitations of Prior Work**: In reality, preference data often contains both **individual differences and shared principles** (e.g., "I may not like this trajectory, but no one wants a collision"), yet these signals are rarely labeled separately. Applying vanilla RLHF to learn a global reward $\hat r(s,a)$ and weighting it with a downstream task reward $r_{\text{new}}$ ($r' = (1-\omega)r_{\text{new}} + \omega \hat r$) leads to two issues: (i) $\hat r$ couples shared safety principles with majority personal preferences; (ii) the weight $\omega$ is extremely sensitive and difficult to tune due to scale differences.

**Key Challenge**: Shared safety principles and user-specific goals in crowdsourced preferences are coupled at the reward level without natural decoupling signals. Downstream tasks often prioritize $r_{\text{new}}$ and seek to avoid being skewed by majority preferences.

**Goal**: (1) Formalize the structure of shared safety principles in crowdsourced preferences and characterize the failure modes of vanilla RLHF; (2) Transfer shared safety signals to arbitrary downstream tasks without explicit safety rewards, oracle labels $z$, or balanced preference data.

**Key Insight**: Rather than combining at the **reward** level, composition should occur at the **policy** level. If each user preference is encoded into a latent-conditioned skill, these skills naturally inherit safety principles by being "on the preference distribution." A high-level policy trained in the skill space will stay within the "universally safe" behavior manifold, allowing optimization of $r_{\text{new}}$ without violating safety boundaries.

**Core Idea**: Replace *reward combination* with *policy composition*. A VAE extracts latent skills from crowdsourced preferences, and the high-level policy makes decisions solely over skill indices. Safety is an emergent property of the skill space structure rather than a result of reward weight tuning.

## Method

### Overall Architecture

The crowdsourced preference reward is decomposed as $r(s,a,z) = r_{\text{user}}(s,a,z) + r_{\text{share}}(s,a)$, where $z$ is the unobservable user context and $r_{\text{share}}$ is the shared safety penalty ($-K$ for $X_{\text{unsafe}}$, else 0). The pipeline consists of two stages:

1.  **Offline Skill Discovery**: Utilizing the preference set $\mathcal D_{\text{pref}} = \{S_z\}$, a VAE encoder $q_\psi(z'|S_z)$ maps each user's preference set to a latent $z'$. The decoder is a latent-conditioned reward $r_\phi(s,a,z')$ or policy $\pi_\theta(a|s,z')$, producing a set of *preference-aligned* low-level skills $\pi_l(a|s,z')$.
2.  **Online/Offline Downstream Training**: Low-level skills are frozen while a high-level policy $\pi_h(z'|s)$ is trained. Actions are generated via $a \sim \pi_l(a|s, z'=\pi_h(s))$. The high-level policy optimizes Q-values using $r_{\text{new}}$ and includes a prior regularization term to keep $z'$ within the prior learned by the VAE, preventing OOD skills.

Input: $\mathcal D_{\text{pref}}$ (crowdsourced preferences) + $\mathcal D_{\tau}$ (arbitrary offline trajectories) + downstream $r_{\text{new}}$. Output: Downstream policy $\pi = \pi_h \circ \pi_l$.

### Key Designs

1.  **Formal Characterization of Vanilla RLHF Failure Modes (Motivation)**:
    *   Function: Explains the inadequacy of learning a global $\hat r$ for reward combination.
    *   Mechanism: Theorem 4.2 proves that when safety penalty $K > 2L\max|r_{\text{user}}|$, all "safe vs. unsafe" trajectory pairs are consistent, allowing $\hat r$ to learn safety preferences in the infinite data limit. However, Theorem 4.3 characterizes the imbalanced scenario: if a user $z_k$ has a proportion $p(z_k) > \frac{|\mathcal T|-1}{\min_{(\tau,\tau') \in X_{\text{ics}}} N(\tau,\tau',z_k) + |\mathcal T|}$, the ranking of $\hat u$ on all inconsistent pairs becomes identical to $u(\cdot, z_k)$. Consequently, $\hat r$ forces majority personal preferences into downstream optimization.
    *   Design Motivation: To justify alternative routes by formalizing baseline pitfalls (weight sensitivity and imbalance bias).

2.  **VAE-based Latent Skill Discovery (VPL + New CPL Variant)**:
    *   Function: Learns preference-aligned skills indexed by latent $z'$ without true labels $z$.
    *   Mechanism: The encoder $q_\psi(z'|S_z)$ maps a user's preference set to a latent space. The decoder predicts preferences via Bradley–Terry: $P(y=1|\tau^1,\tau^2,z') = \frac{\exp \hat u(\tau^1,z')}{\exp \hat u(\tau^1,z') + \exp \hat u(\tau^2,z')}$, trained with KL regularization $D_{KL}(q_\psi \| p(z'))$. For partial-return models, low-level policies are trained via offline RL on $\mathcal D_\tau$: $\max_{\pi_\theta(a|s,z')} \mathbb E_{\tau \sim \mathcal D_\tau}[\sum_t r_\phi(s_t,a_t,z')]$. A **Safe-CPL** variant is introduced, integrating VPL with regret-based models using $P(y=1|\tau^1,\tau^2,z') = \frac{\exp f(\tau^1|z')}{\exp f(\tau^1|z') + \exp \lambda f(\tau^2|z')}$, where $f(\tau^i|z') = \sum_t \gamma^t \alpha \log \pi_\theta(a_t^i|s_t^i,z')$, learning the policy directly to avoid optimization instability.
    *   Design Motivation: Uses latent $z'$ as a proxy for unobservable $z$; using a preference set $S_z$ as input to the encoder is crucial for user differentiation.

3.  **Hierarchical Policy Composition + Prior Regularization**:
    *   Function: Constrains downstream search to a "pre-aligned" skill space, turning safety into a structural property.
    *   Mechanism: Actions are generated as $a \sim \pi_l(a|s, z'=\pi_h(s))$, with the high-level policy switching skills at each step. Training uses TD3 with loss $L_{\pi_h} = -\mathbb E_{a \sim \pi_h \cdot \pi_l}[Q(s,a) + \beta_{\text{reg}} L_{\text{reg}}]$, where $L_{\text{reg}} = \log p(z' = \pi_h(s))$ pulls $z'$ toward the VAE prior. In offline settings, $\beta_{\text{BC}} \|a - a_D\|_2^2$ is added. $\pi_l$ remains frozen.
    *   Design Motivation: (i) Skills inherent safety, allowing $r_{\text{new}}$ optimization without a trade-off $\omega$; (ii) $\beta_{\text{reg}}$ is easier to tune than $\omega$; (iii) Theorem A.7 provides a cost upper bound proportional to low-level skill suboptimality.

### Loss & Training

VAE ELBO for skill discovery (Eq. 7):
$$\mathbb E_{S_z \sim \mathcal D_{\text{pref}}}\big[\mathbb E_{z' \sim q_\psi(z'|S_z)}[\sum_{(\tau^1,\tau^2,y) \in S_z} \log P(y|\tau^1,\tau^2,z')] - D_{KL}(q_\psi(z'|S_z) \| p(z'))\big]$$

Offline downstream training (Eq. 12):
$$L_{\pi_h}^{\text{offline}} = -\mathbb E[Q(s_D,a) + \beta_{\text{reg}} L_{\text{reg}} + \beta_{\text{BC}} L_{\text{BC}}]$$

## Key Experimental Results

### Main Results

Evaluated on 6 safe-RL environments (Bullet-Safety-Gym + Safety-Gymnasium) with simulated crowdsourced preferences:

| Env | Metric | Oracle | Task-Only | SOPL | RC($\omega$=0.5) | Safe-VPL | Safe-CPL |
|------|------|--------|-----------|------|------------------|----------|----------|
| Reach | Rew / Cost | 1.00 / .038 | 1.04 / 1.000 | 0.98 / .024 | 0.83 / .101 | 0.98 / .166 | 0.98 / .069 |
| Run | Rew / Cost | 1.00 / 0 | 1.00 / 1.000 | 0.99 / 0 | 1.00 / 0 | 0.95 / 0 | 0.97 / 0 |
| HalfCheetah-vel | Rew / Cost | 1.00 / 0 | 1.85 / 1.000 | 0.93 / .014 | 0.44 / .107 | 0.96 / .004 | 0.92 / .018 |
| **Average** | Rew / Cost | 1.00 / .01 | **1.46 / 1.00** | 1.04 / .01 | 0.82 / .05 | **0.93 / .03** | **0.92 / .02** |

The Task-Only baseline yields high rewards but extreme costs. Safe-VPL/CPL suppress costs to 0.02-0.03 (near Oracle's 0.01) while maintaining 92-93% of Oracle rewards.

### Ablation Study

| Configuration | Key Observation | Description |
|------|---------|------|
| Varied $\beta_{\text{reg}}$ | Rewards stable; cost improves monotonically | Easier to tune than Reward Combination (RC) $\omega$ |
| Preference noise | Reward stable; cost degrades | Safety signals are susceptible to noise, but skill diversity persists |
| Crowd size | Moderate degradation | Latent capacity covers increasing user counts |
| Balanced vs Imbalanced (10:1)| Ours: degradation < 0.02; RC: degradation ≥ 0.10 | Validates Theorem 4.3 regarding RC bias |

### Key Findings
- The Pareto frontier for RC shifts significantly toward high cost/low reward in imbalanced settings, whereas the proposed method remains near-Oracle, demonstrating robustness to preference imbalance.
- Higher $\beta_{\text{reg}}$ yields more conservative skill selection, improving safety with minor reward trade-offs.
- Preference noise primarily impacts safety rather than task performance, suggesting "shared principles" are more sensitive than "user diversity" signals.

## Highlights & Insights
- **Safety as a Spatial Constraint**: Unlike Reward Combination which treats safety as an additive term, policy composition treats safety as a property of the skill space itself. This "manifesting constraints as manifolds" approach is applicable to any shared-principle preference scenario.
- **Safe-CPL Variant**: Extending VPL to regret-based CPL models allows for reward-free supervised skill discovery, avoiding RL optimization issues.
- **Theoretical Justification**: Theorem 4.3 provides a closed-form upper bound for imbalance thresholds, mathematically proving why reward-based baselines fail when a single user type dominates.

## Limitations & Future Work
- The assumption of "shared, consistent safety principles" may not hold with adversarial users; experiments show safety is sensitive to label noise.
- LLM validation is limited to a 3-class bandit toy, requiring further proof for real-world conversational scenarios.
- Potential improvements include replacing latent priors with learnable mixture distributions, introducing robust aggregation to resist adversarial users, and extending the model to continuous cost signals.

## Related Work & Insights
- **vs VPL**: Shares the VAE backbone for latent context but differs in goal (diversity vs. safety alignment via composition).
- **vs Safe RLHF**: Safe RLHF assumes explicit task vs. safety labels; this work assumes they are coupled.
- **vs ICRL**: Instead of explicitly modeling constraints from preferences, this work embeds them implicitly into the skill space.
- **vs Skill Prior (SPiRL/OPAL)**: While traditional skill priors are extracted from demonstrations, this work extracts them from preferences, providing a path for demonstration-free scenarios like LLM annotation.

## Rating
- Novelty: ⭐⭐⭐⭐ New formulation (shared safety in crowd preferences) and approach (policy composition).
- Experimental Thoroughness: ⭐⭐⭐ Comprehensive in safe-RL envs, though LLM evaluation is limited.
- Writing Quality: ⭐⭐⭐⭐ Strong logical link between theorems and experimental design.
- Value: ⭐⭐⭐⭐ High conceptual value in treating constraints as space structures.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Implicit Preference Alignment for Human Image Animation](implicit_preference_alignment_for_human_image_animation.md)
- [\[ICML 2026\] Curriculum Learning for Safety Alignment](curriculum_learning_for_safety_alignment.md)
- [\[ICML 2026\] MESA: Improving MoE Safety Alignment via Decentralized Expertise](mesa_improving_moe_safety_alignment_via_decentralized_expertise.md)
- [\[ICML 2026\] Quantifying the Salience of Geo-Cultural Values for Pluralistic Safety Alignment](quantifying_the_salience_of_geo-cultural_values_for_pluralistic_safety_alignment.md)
- [\[ICML 2026\] Towards Context-Invariant Safety Alignment for Large Language Models](towards_context-invariant_safety_alignment_for_large_language_models.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[ICML 2026\] Implicit Preference Alignment for Human Image Animation](implicit_preference_alignment_for_human_image_animation.md)
- [\[ICML 2026\] Curriculum Learning for Safety Alignment](curriculum_learning_for_safety_alignment.md)
- [\[ICML 2026\] MESA: Improving MoE Safety Alignment via Decentralized Expertise](mesa_improving_moe_safety_alignment_via_decentralized_expertise.md)
- [\[ICML 2026\] Towards Context-Invariant Safety Alignment for Large Language Models](towards_context-invariant_safety_alignment_for_large_language_models.md)
- [\[ICML 2026\] Quantifying the Salience of Geo-Cultural Values for Pluralistic Safety Alignment](quantifying_the_salience_of_geo-cultural_values_for_pluralistic_safety_alignment.md)

</div>

<!-- RELATED:END -->
