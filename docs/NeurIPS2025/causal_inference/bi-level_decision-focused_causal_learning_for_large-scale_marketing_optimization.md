---
title: >-
  [Paper Note] Bi-Level Decision-Focused Causal Learning for Large-Scale Marketing Optimization
description: >-
  [NeurIPS 2025][Causal Inference][decision-focused learning] This paper proposes Bi-DFCL, a bilevel optimization framework that jointly leverages observational (OBS) data and randomized controlled trial (RCT) data to trai…
tags:
  - "NeurIPS 2025"
  - "Causal Inference"
  - "decision-focused learning"
  - "bilevel optimization"
  - "marketing"
  - "uplift modeling"
date: 2026-05-08
content_hash: 7c38a58b8adefc63
---

# Bi-Level Decision-Focused Causal Learning for Large-Scale Marketing Optimization

**Conference**: NeurIPS 2025  
**arXiv**: [2510.19517](https://arxiv.org/abs/2510.19517)  
**Code**: None (deployed at Meituan)  
**Area**: Causal Inference / Operations Research Optimization  
**Keywords**: decision-focused learning, causal inference, bilevel optimization, marketing, uplift modeling

## TL;DR
This paper proposes Bi-DFCL, a bilevel optimization framework that jointly leverages observational (OBS) data and randomized controlled trial (RCT) data to train marketing resource allocation models. The upper level trains a Bridge Network with unbiased decision loss on RCT data to dynamically correct the bias of the lower level trained on OBS data. The framework further introduces differentiable surrogate decision losses (PPL/PIFD) grounded in the primal problem and an implicit differentiation algorithm, addressing the predict-then-optimize inconsistency and the bias-variance dilemma of conventional two-stage methods. The system has been deployed at scale on Meituan.

## Background & Motivation

**Background**: Marketing optimization on online platforms (e.g., coupon allocation) is a classic resource allocation problem. The dominant paradigm is the two-stage method (TSM): the first stage uses ML to predict individual treatment effects, and the second stage uses operations research (OR) to make allocation decisions.

**Limitations of Prior Work**: (1) **Predict-then-optimize inconsistency**: ML optimizes predictive accuracy while OR optimizes decision quality, but better predictions do not necessarily yield better decisions—prediction errors are amplified in non-convex NP-hard resource allocation problems; (2) **Bias-variance dilemma**: OBS data is abundant but biased (selection bias, position bias), while RCT data is unbiased but scarce and high-variance.

**Key Challenge**: Decision-Focused Learning (DFL) can narrow the predict-then-optimize gap, but due to the counterfactual problem, the decision loss can only be computed on scarce RCT data, which exacerbates the bias-variance dilemma.

**Goal**: Simultaneously address both challenges of TSM and existing DFL methods—predict-then-optimize alignment and bias-variance balance.

**Key Insight**: Exploit the unbiasedness of RCT data to guide the learning direction on OBS data by assigning the decision loss and the prediction loss to the upper and lower levels of a bilevel optimization framework, respectively.

**Core Idea**: In the bilevel framework, the upper level trains a Bridge Network with an unbiased decision loss on RCT data, while the lower level trains a Target Network with a (Bridge-corrected) prediction loss on OBS data, achieving data complementarity and objective alignment.

## Method

### Overall Architecture
The Target Network $\mathcal{F}_\theta$ is trained on large-scale OBS data with a prediction loss (lower level), while the Bridge Network $\mathcal{G}_\phi$ is trained on RCT data with a decision loss (upper level). The Bridge outputs gating coefficients $w$ to mix predictions from a Teacher Network and the Target Network to generate counterfactual pseudo-labels, dynamically correcting the bias in OBS data.

### Key Designs

1. **Unbiased Decision Loss Estimation**:

    - Function: Derives an unbiased estimator of decision quality by exploiting the randomization property of RCT data.
    - Mechanism: Under the strong ignorability assumption, the decision loss is rewritten as $\mathcal{L}_\text{DL} = -\mathbb{E}[\frac{N}{N_{t_i}} \cdot z^*_{it_i} \cdot r_{it_i}]$, where the weight $N/N_{t_i}$ corrects for imbalance across treatment groups in the RCT.
    - Design Motivation: The true decision loss is not directly computable due to unobservable counterfactuals; the unbiased estimator enables reliable evaluation of decision quality on RCT data.

2. **Primal Policy Learning (PPL) Surrogate Loss**:

    - Function: Relaxes the discrete argmax decision into a continuous differentiable loss.
    - Mechanism: Replaces the indicator function with a Softmax relaxation: $z'_{it_i} = \frac{\exp[\hat{r}_{it_i} - \lambda^* \hat{c}_{it_i}]}{\sum_j \exp[\hat{r}_{ij} - \lambda^* \hat{c}_{ij}]}$. This operates directly under the primal problem constraint (a specific budget $B$), providing closer alignment with the actual scenario than DFCL's dual decision loss (which considers all budgets).
    - Design Motivation: The indicator function in discrete optimization is non-differentiable; a differentiable surrogate is required to allow gradient flow through the decision layer.

3. **Bilevel Optimization + Implicit Differentiation**:

    - Function: Resolves the problem that the Jacobian $\partial\theta^*/\partial\phi$ in the upper-level gradient computation is not analytically tractable.
    - Mechanism: Applies the implicit function theorem starting from the lower-level optimality condition $\partial\mathcal{L}_\text{PL}/\partial\theta|_{\theta=\theta^*} = 0$ to derive the Jacobian by solving a Hessian equation. The conjugate gradient algorithm is used to avoid explicitly computing the Hessian inverse, requiring only Hessian-vector products.
    - Design Motivation: Explicit differentiation (gradient unrolling) depends on the optimization path and is prone to vanishing gradients; implicit differentiation is path-independent and more stable.

4. **Bridge Network Gating Mechanism**:

    - Function: Adaptively mixes predictions from the Teacher and the Target to generate counterfactual pseudo-labels for OBS data.
    - Mechanism: The gating coefficient $w = \text{sigmoid}(\mathcal{G}_\phi(i,j))$ controls the degree of reliance on the Teacher (unbiased but high-variance) versus the Target (biased but low-variance), and is dynamically adjusted via the gradient signal from the upper-level decision loss.
    - Design Motivation: Using Teacher predictions directly as pseudo-labels yields high variance (few RCT samples), while using the Target's own predictions provides no corrective effect. The Bridge adaptively balances between the two.

### Loss & Training
Lower level: MSE prediction loss on OBS data. Upper level: PPL or PIFD decision surrogate loss on RCT data. The Bridge Network is updated every $k=5$ batches.

## Key Experimental Results

### Main Results

| Method | CRITEO-UPLIFT | Meituan Marketing I | Meituan Marketing II |
|--------|--------------|--------------------|--------------------|
| TSM (best uplift) | baseline | baseline | baseline |
| DHCL | improves but overfits | improves | improves |
| DFCL | improves | improves | improves |
| **Bi-DFCL** | **significantly outperforms all** | **significantly outperforms all** | **significantly outperforms all** |

### Online A/B Testing
- Large-scale online A/B tests at Meituan demonstrate statistically significant revenue improvements with Bi-DFCL.
- Deployed across multiple real-world marketing scenarios.

### Ablation Study

| Configuration | Description |
|--------------|-------------|
| w/o bilevel optimization (train on RCT only) | High variance, overfitting |
| w/o Bridge Network | Cannot correct OBS bias |
| PPL vs. PIFD | PIFD preserves the original optimization landscape and is sometimes superior |
| Explicit vs. implicit differentiation | Implicit differentiation is more stable |

### Key Findings
- **The bilevel optimization framework effectively resolves the bias-variance dilemma**: OBS data provides low-variance generalization signal, while RCT data provides unbiased decision guidance.
- **Primal decision loss outperforms the dual formulation**: PPL operates directly under a specific budget constraint, yielding better practical alignment than DFCL's dual loss.
- **Implicit differentiation is superior to explicit differentiation**: More stable and independent of the optimization path.
- **Adaptive balancing eliminates manual tuning**: The Bridge automatically learns the mixing ratio between OBS and RCT, removing the need for manual search of $\alpha$ in DFCL.

## Highlights & Insights
- **The bilevel framework precisely aligns data type with loss type**: decision loss–RCT (guarantees unbiasedness), prediction loss–OBS (guarantees generalization)—a design philosophy considerably deeper than naively combining both losses on the same data.
- **The Bridge Network's gating mechanism** effectively realizes "using the unbiased signal from RCT to correct the learning direction on OBS data," representing a design pattern transferable to other causal inference settings.
- **Large-scale industrial deployment validation**: Online A/B testing on a large-scale platform such as Meituan provides strong real-world empirical evidence.

## Limitations & Future Work
- **Assumes availability of RCT data**: In many settings, collecting RCT data is prohibitively costly or infeasible.
- **NP-hard nature of MCKP**: While Lagrangian relaxation provides an approximate solution, approximation quality may degrade in extreme scenarios.
- **Limited treatment dimensionality**: The current formulation only considers a finite set of discrete treatment options and does not extend to continuous treatment spaces.

## Related Work & Insights
- **vs. DFCL (Zhou et al.)**: DFCL uses only RCT data and requires manual tuning of $\alpha$; Bi-DFCL jointly leverages OBS and RCT data with adaptive balancing.
- **vs. standard DFL**: General DFL methods do not address counterfactuals or marketing-specific constraints; Bi-DFCL introduces tailored unbiased estimators and surrogate losses.
- **vs. meta-learning**: The bilevel optimization is formally analogous to MAML but serves a fundamentally different objective.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The bilevel optimization + Bridge Network design is novel and principled
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Public benchmarks + industrial datasets + online A/B testing
- Writing Quality: ⭐⭐⭐⭐ Technically complete but notation-heavy
- Value: ⭐⭐⭐⭐⭐ Deployed at scale on a large platform, demonstrating high practical impact

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Cyclic Counterfactuals under Shift–Scale Interventions](cyclic_counterfactuals_under_shift-scale_interventions.md)
- [\[NeurIPS 2025\] Do-PFN: In-Context Learning for Causal Effect Estimation](do-pfn_in-context_learning_for_causal_effect_estimation.md)
- [\[NeurIPS 2025\] Revealing Multimodal Causality with Large Language Models](revealing_multimodal_causality_with_large_language_models.md)
- [\[NeurIPS 2025\] Differentiable Structure Learning and Causal Discovery for General Binary Data](differentiable_structure_learning_and_causal_discovery_for_general_binary_data.md)
- [\[NeurIPS 2025\] Counterfactual Reasoning for Steerable Pluralistic Value Alignment of Large Language Models](counterfactual_reasoning_for_steerable_pluralistic_value_alignment_of_large_lang.md)

</div>

<!-- RELATED:END -->
