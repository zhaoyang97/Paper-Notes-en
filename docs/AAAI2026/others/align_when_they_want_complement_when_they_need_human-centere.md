---
title: >-
  [Paper Note] Align When They Want, Complement When They Need! Human-Centered Ensembles for Adaptive Human-AI Collaboration
description: >-
  [AAAI 2026][human-AI collaboration] This paper reveals a fundamental trade-off between *complementarity* and *alignment* in human-AI collaboration—no single model can simultaneously optimize both objectives. It proposes…
tags:
  - "AAAI 2026"
  - "human-AI collaboration"
  - "complementarity-alignment trade-off"
  - "adaptive ensemble"
  - "trust modeling"
  - "behavior-aware AI"
date: 2026-05-08
content_hash: f529e70e508043e8
---

# Align When They Want, Complement When They Need! Human-Centered Ensembles for Adaptive Human-AI Collaboration

**Conference**: AAAI 2026
**arXiv**: [2602.20104v1](https://arxiv.org/abs/2602.20104v1)
**Code**: [GitHub](https://github.com/shasanamin/aaai26-adaptive-ai)
**Area**: Other
**Keywords**: human-AI collaboration, complementarity-alignment trade-off, adaptive ensemble, trust modeling, behavior-aware AI

## TL;DR

This paper reveals a fundamental trade-off between *complementarity* and *alignment* in human-AI collaboration—no single model can simultaneously optimize both objectives. It proposes an adaptive AI ensemble framework that dynamically switches between an alignment model and a complementarity model via a Rational Routing Shortcut (RRS) mechanism, achieving up to 9% improvement in team accuracy over standard AI.

## Background & Motivation

A long-overlooked tension exists in AI-assisted human decision-making:
- **Complementarity AI**: Optimizes AI correctness on instances where human judgment is weak, theoretically improving team performance, but diverges from human judgment in high-confidence regions, causing **trust degradation**—users tend to ignore AI suggestions precisely when they need them most.
- **Alignment AI**: Maintaining consistency with human judgment builds trust, but risks **reinforcing suboptimal human decisions**, wasting the potential performance gains of AI.
- Existing behavior-aware AI approaches have begun to account for human-AI interaction dynamics, yet remain limited to training a single model that attempts to balance both objectives.

## Core Problem

**The fundamental limitation of the single-model paradigm**: Can it be formally proven that a single AI model cannot simultaneously optimize complementarity and alignment? If not, how can a practical framework be designed to overcome this trade-off?

## Method

### Overall Architecture

Two specialist models are trained—an alignment expert $m_a$ and a complementarity expert $m_c$. At inference time, the RRS mechanism dynamically selects which expert's recommendation to use based on instance-level context.

### Key Designs

1. **Confidence-Gated Probabilistic Reliance (CGPR) Model**: Models human decision-making behavior. The decision space is partitioned into an *alignment region* $\mathcal{D}_a$ (high human confidence) and a *complementarity region* $\mathcal{D}_c$ (low human confidence). In $\mathcal{D}_a$, humans follow their own judgment; in $\mathcal{D}_c$, humans accept AI recommendations with probability $r = 1 - L_h(\mathcal{D}_a, m)$, which is determined by AI-human agreement in the high-confidence region. Key insight: trust is primarily driven by the degree to which AI aligns with humans in regions where humans are confident.

2. **Theoretical Proof of the Complementarity-Alignment Trade-off (Theorem 2)**: Under logistic loss with $\ell_2$ regularization, when model parameters are updated by an infinitesimal step toward alignment, the ratio of the increase in complementarity loss to the decrease in alignment loss—the unit trade-off $\mathcal{T}(\theta)$—satisfies the lower bound $\mathcal{T}(\theta) \geq \frac{\lambda_r}{\kappa} \frac{d_c}{d_a}(-\cos\phi(\theta))$, and $\mathcal{T} \to +\infty$ as the model approaches the alignment optimum. This mathematically establishes the fundamental limitation of the single-model paradigm.

3. **Rational Routing Shortcut (RRS) Mechanism**: The core practical contribution. Rather than requiring unobservable internal states such as human confidence levels or confidence thresholds, RRS routes solely based on the prediction confidence of the two expert models—assigning each instance to the more confident expert: $m_{\text{RRS}}(\mathbf{x}) = m_a(\mathbf{x})$ if $\mathcal{C}^a(\mathbf{x}) \geq \mathcal{C}^c(\mathbf{x})$, else $m_c(\mathbf{x})$. Intuition: each expert exhibits higher confidence on instances drawn from its own region of specialization, so confidence serves as an implicit proxy for region membership.

### Loss & Training

- **Complementarity expert** $m_c$: Minimizes prediction loss over the complementarity region $\min L(\mathcal{D}_c, m_c)$, with sample weights $w_i^c = 1 - F_T(\mathcal{C}_i^h)$, where $F_T$ is the CDF of the confidence threshold.
- **Alignment expert** $m_a$: Minimizes disagreement with human judgment over the alignment region $\min L_h(\mathcal{D}_a, m_a)$, using human decisions $h(\mathbf{x})$ as pseudo-labels with weights $w_i^a = F_T(\mathcal{C}_i^h)$.
- Uncertainty handling: The confidence threshold $\tau$ is modeled as a random variable; robust training is achieved via probabilistic region membership and expected region-level loss.

## Key Experimental Results

| Dataset | Metric | Ours (Adaptive AI) | Prev. SOTA | Gain |
|--------|------|------|----------|------|
| WoofNette (real data) | Team accuracy | Highest | Standard AI | +9% |
| WoofNette (real data) | Team accuracy | Highest | Behavior-aware AI | +6% |
| College Admissions (simulation) | Team accuracy | Highest | Single AI | Consistent gains across multiple settings |

### Ablation Study

- **Expert divergence** $D$: Greater divergence between experts yields larger gains from adaptive ensembling (validating the $D^2$ dependence in Theorem 4).
- **Human accuracy** $\alpha$: Higher human accuracy in the alignment region produces larger gains ($\kappa = 2\alpha - 1$).
- **Task mixing ratio** $p$: Gains are maximized when alignment and complementarity regions are roughly balanced ($p \approx 0.5$), yielding an inverted-U pattern.
- **Region assignment certainty**: Even under noisy region assignments, ensemble gains degrade gracefully, demonstrating strong robustness.
- Even when each specialist model's independent accuracy is lower than the standard AI model, team accuracy remains higher.

## Highlights & Insights

1. **Solid theoretical contributions**: The paper provides the first rigorous proof of the mathematical impossibility of simultaneously optimizing complementarity and alignment (Theorem 2), and precisely quantifies the lower bound on the performance gain of adaptive ensembles over single models (Theorem 4), including a generalization under uncertainty (Corollary 6).
2. **RRS is elegantly simple and practical**: No observation of human internal states is required; routing relies solely on the confidence scores of the two models, and is provably near-optimal (Theorem 3).
3. **"Weak models, strong combination" phenomenon**: Individual specialist models may have lower standalone accuracy than a standard AI model, yet adaptive routing yields higher overall team accuracy—a counter-intuitive finding.
4. **Innovation in behavior modeling**: CGPR unifies human trust, confidence, and probabilistic reliance behavior in a single framework, offering a more realistic characterization than prior deterministic threshold models (CGR).

## Limitations & Future Work

1. **Simplified human behavior assumptions**: Although CGPR is more realistic than CGR, it still assumes a specific linear trust-alignment relationship ($r = 1 - L_h$); real human trust dynamics may be considerably more complex (e.g., temporal decay, anchoring effects).
2. **Binary region partition is overly coarse**: Dichotomizing the decision space into alignment and complementarity regions may be too simplistic, as human confidence is a continuous spectrum in practice.
3. **Validation limited to binary classification**: Theoretical analysis and main experiments are grounded in binary classification scenarios (College Admissions, WoofNette); generalization to multi-class or regression settings requires further investigation.
4. **Static human model**: Human behavior is assumed to remain constant throughout the interaction, without accounting for learning effects or the dynamic evolution of trust.
5. **Fixed number of two expert models**: Future work could explore larger ensembles (e.g., dedicated handling of mid-confidence regions) or continuous-spectrum routing strategies.

## Related Work & Insights

| Method | Focus | Limitation |
|------|--------|------|
| Standard AI | Maximize standalone AI accuracy | Ignores human-AI interaction |
| Complementarity AI | Optimize AI on human weak instances | Undermines trust |
| Alignment AI | Match human judgment | Reinforces suboptimal behavior |
| Learning to Defer | Human-AI division of labor | Assumes AI can make final decisions |
| Behavior-aware AI (Bansal 2021, Mahmood 2024) | Optimize team objective | Single model; cannot escape the trade-off |
| **Ours (Adaptive AI Ensemble)** | **Adaptively switch between alignment/complementarity** | **Overcomes single-model limitations** |

**Connection to model routing**: The RRS strategy of "select the more confident expert" bears resemblance to the gating mechanism in Mixture-of-Experts, but here the routing objective is human-AI team performance rather than pure model performance.

**The central role of trust in collaborative systems**: This work reaffirms that the value of an AI system depends not only on its objective performance but also on whether users trust and act upon its recommendations—a design implication relevant to any AI system requiring human-in-the-loop participation.

**Generalizability**: The core idea of RRS—using model confidence as a proxy signal for region membership—may transfer to other settings requiring dynamic strategy switching, such as task routing in multi-task learning.

## Rating

- Novelty: ⭐⭐⭐⭐ (The formalization of the complementarity-alignment trade-off and the RRS mechanism are original contributions, though the dual-expert ensemble framework itself is not entirely novel.)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Simulation experiments systematically validate theoretical predictions; real-data experiments on WoofNette are convincing, though validation across more domains is lacking.)
- Writing Quality: ⭐⭐⭐⭐⭐ (The paper is well-structured, with tight correspondence between theory and experiments and clearly articulated motivation.)
- Value: ⭐⭐⭐⭐ (Provides a new theoretical foundation and practical framework for human-AI collaborative system design, with meaningful implications for HCI and trustworthy AI.)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Intrinsic Barriers and Practical Pathways for Human-AI Alignment: An Agreement-Based Complexity Analysis](intrinsic_barriers_and_practical_pathways_for_human-ai_alignment_an_agreement-ba.md)
- [\[AAAI 2026\] How Hard Is It to Rig a Tournament When Few Players Can Beat or Be Beaten by the Favorite?](how_hard_is_it_to_rig_a_tournament_when_few_players_can_beat_or_be_beaten_by_the.md)
- [\[ICLR 2026\] When to Retrain after Drift: A Data-Only Test of Post-Drift Data Size Sufficiency](../../ICLR2026/others/when_to_retrain_after_drift_a_data-only_test_of_post-drift_data_size_sufficiency.md)
- [\[CVPR 2026\] LoViF 2026 Challenge on Human-oriented Semantic Image Quality Assessment](../../CVPR2026/others/lovif_2026_semantic_quality_assessment_challenge.md)
- [\[CVPR 2026\] TeamHOI: Learning a Unified Policy for Cooperative Human-Object Interactions with Any Team Size](../../CVPR2026/others/teamhoi_learning_a_unified_policy_for_cooperative_human-object_interactions_with.md)

</div>

<!-- RELATED:END -->
