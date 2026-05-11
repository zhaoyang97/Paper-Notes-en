---
title: >-
  [Paper Note] The Burden of Interactive Alignment with Inconsistent Preferences
description: >-
  [NeurIPS 2025][Reinforcement Learning][interactive alignment] This paper models user interactions with engagement-driven algorithms as a multi-leader single-follower Stackelberg game…
tags:
  - "NeurIPS 2025"
  - "Reinforcement Learning"
  - "interactive alignment"
  - "inconsistent preferences"
  - "Stackelberg game"
  - "engagement optimization"
  - "costly signaling"
date: 2026-05-08
content_hash: 2a8f62bb325b027b
---

# The Burden of Interactive Alignment with Inconsistent Preferences

**Conference**: NeurIPS 2025
**arXiv**: [2510.16368](https://arxiv.org/abs/2510.16368)
**Authors**: Ali Shirali (UC Berkeley)
**Code**: None
**Area**: Reinforcement Learning / Alignment Theory / Game Theory
**Keywords**: interactive alignment, inconsistent preferences, Stackelberg game, engagement optimization, costly signaling

## TL;DR

This paper models user interactions with engagement-driven algorithms as a multi-leader single-follower Stackelberg game, establishing a critical planning-horizon threshold: users whose effective horizon exceeds this threshold can align the algorithm to their interests, while those below it are instead aligned to the algorithm's objectives. The paper further demonstrates that introducing low-cost signals (e.g., an extra click) can substantially reduce the burden of alignment.

## Background & Motivation

From recommendation systems to chatbots, algorithms profoundly shape how users access information and engage with content. These systems typically optimize for user engagement, yet user engagement behavior does not always reflect genuine user preferences.

The central tension is **inconsistent preferences**: users may spend substantial time on enticing but low-value content (e.g., scrolling short videos), thereby sending misleading signals to the algorithm and reinforcing recommendations of such content. This phenomenon is grounded in the dual-system theory of human decision-making—the deliberative "System 2" governs whether to engage, while the impulsive "System 1" governs engagement duration.

Most prior work assumes consistent user preferences (e.g., Bradley-Terry models, RLHF), or designs alignment strategies from the platform or algorithm side. This paper takes a distinctive **user-side** perspective, asking: when users have inconsistent preferences, what cost must they bear to steer the algorithm toward their true interests? This perspective fills a theoretical gap in user-driven alignment.

## Method

### User–Algorithm Interaction Model

Interactions are organized into sessions. In each interaction, the algorithm recommends an item $s$, and the user makes two decisions:

1. **Whether to engage**: governed by the deliberative System 2, with engagement probability $f_\theta(s)$
2. **Duration of engagement**: governed by the impulsive System 1, with expected duration $1/\alpha_\theta(s)$

The user type $\theta \in \Theta$ encodes her intent (a single user may have different types across sessions). Engaging yields reward $r_\theta(s)$ (independent of duration), discounted by factor $\gamma_H < 1$. The algorithm seeks to maximize total engagement duration $\sum_t y_t$, discounted by $\gamma_A$.

### Stackelberg Game Formulation

The alignment problem is modeled as an **extended multi-leader single-follower** Stackelberg game:

- **Leaders** (users / System 2): commit to engagement strategies $\mathbf{f} = (f_\theta)_{\theta \in \Theta}$
- **Follower** (algorithm): best-responds based on observed interaction history

Two entry scenarios are defined:
- **Random Entry (RE)**: users encounter initial content by chance
- **Algorithmic Entry (AE)**: the algorithm actively selects the first item

From the user side, equilibrium additionally requires a mixed-strategy Nash condition—no individual user has a unilateral incentive to deviate from the equilibrium strategy.

### Core Special Case: Inconsistent Actions and Rewards

Consider two items $S = \{a, b\}$ and two user types:

| User Type | Engagement Duration | Reward |
|-----------|--------------------|----|
| $\Theta_1$ (Type 1) | $1/\alpha_\theta(a) > 1/\alpha_\theta(b)$ ($a$ more enticing) | $r_\theta(a) < r_\theta(b)$ ($b$ more valuable) |
| $\Theta_2$ (Type 2) | $1/\alpha_\theta(a) > 1/\alpha_\theta(b)$ | $r_\theta(a) > r_\theta(b)$ (aligned with algorithm) |

Type-1 users face the core conflict: item $a$ is more enticing (System 1 engages longer), but item $b$ is more valuable. Since longer engagement on $a$ causes the algorithm to keep recommending it, users must strategically **withhold engagement** to steer the algorithm.

Two illustrative examples:
- **Music recommendation**: a user wants ambient music while working ($b$), but is also a fan of artist X ($a$), whose songs are captivating yet distracting
- **Chatbot**: an engineer needs quick answers (Type 1), but the conversational interface incentivizes the algorithm to prolong interactions

### Algorithm Best Response (Theorem 4.1)

The algorithm's optimal strategy is equivalent to a **linear classifier** over the posterior $\lambda$: recommend $a$ if and only if $\sum_\theta h_\theta \lambda_\theta \geq 0$. Crucially, $h_\theta$ depends only on type $\theta$'s own strategy, with no cross-type terms. This structural result substantially simplifies equilibrium analysis.

A Type-1 user reduces $f_\theta(a)$ (lowering engagement probability on $a$) to make $h_\theta$ negative, pushing the classifier toward recommending $b$. However, withholding engagement also foregoes the immediate reward $(1 - f_\theta(a)) \cdot r_\theta(a)$, creating a trade-off between short-term gains and long-term signaling.

### User Best Response and Burden of Alignment (Theorem 5.1)

For each user type $\theta$, there exists a **steerable set** $F_\theta$. If $F_\theta$ is nonempty, the user can choose any strategy within it to achieve alignment.

Whether a user can steer the algorithm depends on the **classifier margin** $m_\theta = \sum_{\theta' \neq \theta} h_{\theta'} \lambda_{\theta'}$, which captures the influence of other users' strategies on the classification boundary. A larger margin makes it harder for Type-1 users to steer the algorithm.

Defining the user's effective horizon as $\tau_H = 1/(1 - \gamma_H)$, a Type-1 user achieves constant regret if and only if:

$$\tau_H > \frac{r_\theta(b)}{r_\theta(b) - r_\theta(a)}$$

If the effective horizon is insufficient, the user will fully engage with the enticing content and is instead aligned to the algorithm's objective. **This is the core meaning of the "burden of alignment."**

### Low-Cost Signals Reduce the Alignment Burden (Section 6)

Introducing an observable signal with cost $c$ (e.g., clicking "Not interested") decouples type communication from content consumption.

Key changes:
- The user's strategy expands from $f_\theta(s)$ to $(f_\theta(s), u_\theta(s))$, where $u_\theta(s)$ is the probability of sending the signal
- The algorithm updates its posterior based on the joint history of engagement and signals
- The steerable set transitions from a linear constraint to a bilinear constraint, which is at least as permissive when projected onto the $f_\theta$ dimension

The alignment threshold under signals becomes $\gamma_H^c$, requiring a shorter effective horizon. Even a small signal cost can substantially reduce the planning horizon required for alignment.

Practical implication: a sufficiently forward-looking user facing unwanted content optimally **partially engages and incurs the signal cost** (rather than fully abstaining), a strategy more flexible than the no-signal case.

## Experiments and Theoretical Validation

This is a purely theoretical work with no empirical experiments; complete equilibrium characterizations are established through rigorous mathematical derivation.

### Table 1: Core Special Case Setup

| User Type | Engagement Duration Order | Reward Order | Preference Consistency |
|-----------|--------------------------|-------------|----------------------|
| $\theta \in \Theta_1$ | $1/\alpha_\theta(a) > 1/\alpha_\theta(b)$ | $r_\theta(a) < r_\theta(b)$ | Inconsistent |
| $\theta \in \Theta_2$ | $1/\alpha_\theta(a) > 1/\alpha_\theta(b)$ | $r_\theta(a) > r_\theta(b)$ | Consistent |

Type-1 users are the central object of study: they spend more time on enticing content despite deriving less genuine value from it.

### Table 2: Alignment Burden Comparison With and Without Signals

| Condition | No Signal | With Signal (cost $c$) |
|-----------|-----------|------------------------|
| Effective horizon required for alignment | $\tau_H > r_\theta(b)/(r_\theta(b)-r_\theta(a))$ | Shorter (lower threshold) |
| User strategy (when non-steerable) | Full abstention or full engagement | Partial engagement + signal cost |
| Steerable set constraint | Linear | Bilinear (more permissive) |
| Regret property | Constant iff horizon sufficient | Same, but with lower threshold |

Core theoretical results:
- **Theorem 4.1**: Algorithm best response is a linear classifier over the posterior
- **Theorem 5.1**: Fully characterizes the Stackelberg equilibrium under algorithmic entry
- **Corollary 5.2/5.3**: Structure of the steerable set and strategies when non-steerable
- **Corollary 5.4**: Necessary and sufficient conditions for constant regret
- **Theorem 6.1/6.2**: Extend the above results to the signal setting
- **Corollary 6.3/6.4**: Prove that signals reduce the burden of alignment

## Highlights & Insights

- **Quantifying the alignment burden**: The paper introduces the concept of the "burden of alignment" for the first time, measured by the minimum planning horizon required for the user to steer the algorithm, translating a vague notion of alignment difficulty into a precise mathematical bound.
- **Linear classifier structure**: The algorithm's optimal strategy is shown to be equivalent to a linear classifier over the type posterior, with different types decoupled, making equilibrium analysis tractable; this structural finding has independent theoretical value.
- **Outsized impact of signals**: Even a negligibly small signal cost (e.g., an extra click) can significantly reduce the alignment burden, providing direct practical guidance for platform design—offering simple feedback mechanisms is far more cost-effective than complex algorithmic redesign.
- **Formalization of dual-system decision-making**: The System 1/System 2 framework from behavioral economics is naturally integrated into the Stackelberg game model, yielding a rigorous operational definition of inconsistent preferences and bridging behavioral science with algorithmic game theory.

## Limitations & Future Work

- **Strong information assumptions**: The framework assumes the algorithm has full knowledge of user strategies and users have full knowledge of their own rewards; in practice, information is always incomplete, and the existence and structure of equilibria under relaxed assumptions remain open.
- **Lack of empirical validation**: As a purely theoretical framework, its conclusions have not been validated on real recommendation systems or LLM interaction data.
- **Two-party game only**: The strategic behavior of content creators is ignored; in reality, the platform–user–creator relationship constitutes a three-party game.
- **Restricted item set**: Core analysis is limited to the special case $|S| = 2$; theoretical guarantees for scaling to large content spaces remain unclear.
- **Static type assumption**: User types are assumed fixed within a session, whereas real preferences may evolve dynamically during interaction.

## Related Work & Insights

- **RLHF / DPO series**: Bradley-Terry models, RLHF (Christiano et al., 2017; Ouyang et al., 2022), and DPO (Rafailov et al., 2024) assume consistent preferences; this paper relaxes that core assumption.
- **Strategic users and recommendation systems**: Haupt et al. (2023) and Cen et al. (2024) study strategic user behavior in recommendations, but with the platform as the Stackelberg leader; this paper inverts the role, positioning the user as leader.
- **Modeling inconsistent preferences**: Kleinberg et al. (2024) introduce the dual-system framework for inconsistent preferences; this paper builds on that foundation by incorporating game-theoretic equilibrium analysis.
- **Risks of engagement optimization**: Besbes et al. (2024) analyze the risks of optimizing measurable metrics; Milli et al. (2021) distinguish engagement from value.
- **Money burning in mechanism design**: The costly signaling idea from Hartline & Roughgarden (2008) is elegantly applied to reduce the alignment burden, demonstrating new relevance of classical economic tools for AI alignment.

## Rating ⭐

| Dimension | Rating |
|-----------|--------|
| Novelty | ⭐⭐⭐⭐⭐ |
| Theoretical Depth | ⭐⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| Overall Recommendation | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Learning Interactive World Model for Object-Centric Reinforcement Learning](learning_interactive_world_model_for_object-centric_reinforcement_learning.md)
- [\[NeurIPS 2025\] Interactive and Hybrid Imitation Learning: Provably Beating Behavior Cloning](interactive_and_hybrid_imitation_learning_provably_beating_behavior_cloning.md)
- [\[NeurIPS 2025\] Teaching Language Models to Evolve with Users: Dynamic Profile Modeling for Personalized Alignment](teaching_language_models_to_evolve_with_users_dynamic_profile_modeling_for_perso.md)
- [\[ICLR 2026\] Reasoning Boosts Opinion Alignment in LLMs](../../ICLR2026/reinforcement_learning/reasoning_boosts_opinion_alignment_in_llms.md)
- [\[ICLR 2026\] Menlo: From Preferences to Proficiency – Evaluating and Modeling Native-like Quality Across 47 Languages](../../ICLR2026/reinforcement_learning/menlo_from_preferences_to_proficiency_--_evaluating_and_modeling_native-like_qua.md)

</div>

<!-- RELATED:END -->
