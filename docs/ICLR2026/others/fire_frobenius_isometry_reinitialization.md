---
title: >-
  [Paper Note] FIRE: Frobenius-Isometry Reinitialization for Balancing the Stability-Plasticity Tradeoff
description: >-
  [ICLR 2026 Oral][stability-plasticity] This paper formalizes the stability-plasticity tradeoff in continual learning as a constrained optimization problem—minimizing weight deviation (stability) subject to an orthogonali…
tags:
  - "ICLR 2026 Oral"
  - "stability-plasticity"
  - "reinitialization"
  - "orthogonal Procrustes"
  - "continual learning"
  - "plasticity loss"
date: 2026-05-08
content_hash: 82d907bbc95380cf
---

# FIRE: Frobenius-Isometry Reinitialization for Balancing the Stability-Plasticity Tradeoff

**Conference**: ICLR 2026 Oral
**arXiv**: [2602.08040](https://arxiv.org/abs/2602.08040)
**Code**: Available
**Area**: Continual Learning / Reinforcement Learning
**Keywords**: stability-plasticity, reinitialization, orthogonal Procrustes, continual learning, plasticity loss

## TL;DR
This paper formalizes the stability-plasticity tradeoff in continual learning as a constrained optimization problem—minimizing weight deviation (stability) subject to an orthogonality constraint (plasticity)—yielding a closed-form solution to the orthogonal Procrustes problem, $\tilde{W}^* = W(W^\top W)^{-1/2}$ (polar decomposition), implemented efficiently via Newton-Schulz iteration (<1% additional time). FIRE comprehensively outperforms baselines such as S&P across visual continual learning, LLM continual pre-training, and RL.

## Background & Motivation

**Background**: Neural networks trained on non-stationary data face the **stability-plasticity dilemma**: high stability leads to rigidity that impedes learning new knowledge, while high plasticity leads to catastrophic forgetting of old knowledge. Existing approaches include Shrink & Perturb (S&P), DASH, and various reinitialization methods.

**Limitations of Prior Work**: (a) S&P requires careful tuning of shrinkage and perturbation ratios; (b) DASH incurs high computational cost (69s vs. FIRE's 0.06s); (c) full reinitialization destroys useful knowledge and causes instability; (d) existing plasticity measures (loss landscape curvature, dormant neurons, feature rank) are non-differentiable and data-dependent, making direct optimization intractable.

**Key Challenge**: Stability requires weights to remain unchanged, while plasticity requires weights to be "well-conditioned" (orthogonal, low curvature). How can both objectives be unified in a single formulation?

**Goal**: To propose a principled reinitialization method with a closed-form solution that automatically finds the optimal balance between stability and plasticity without hyperparameter tuning.

**Key Insight**: The paper proposes **Deviation from Isometry (DfI)** as a differentiable, data-free plasticity measure: $\text{DfI}(W) = \|W^\top W - I\|_F^2$. It is shown that DfI simultaneously captures loss landscape curvature (Theorem 2), feature rank (Theorem 3), and dormant neurons (Theorem 4).

**Core Idea**: Reinitialization is formulated as "minimize weight deviation subject to an orthogonality constraint," yielding a closed-form polar decomposition solution that resolves the stability-plasticity tradeoff in a single step.

## Method

### Overall Architecture
Between two learning phases (e.g., task transitions, midpoint of RL training), each layer's weight matrix undergoes a single orthogonal reinitialization: $\tilde{W}^* = W(W^\top W)^{-1/2}$. This operation minimizes $\|W - \tilde{W}\|_F^2$ (stability) while enforcing $\tilde{W}^\top \tilde{W} = I$ (plasticity).

### Key Designs

1. **Stability Measure: Squared Frobenius Error (SFE)**

    - Function: Quantifies weight deviation before and after reinitialization.
    - Mechanism: $\text{SFE}(W, \tilde{W}) = \|W - \tilde{W}\|_F^2$. Theorem 1 proves that SFE bounds the discrepancy between the normalized feature covariances of the two networks.
    - Design Motivation: Directly measures "how much has changed," ensuring useful knowledge is preserved.

2. **Plasticity Measure: Deviation from Isometry (DfI)**

    - Function: Quantifies the degree to which a weight matrix deviates from orthogonality.
    - Mechanism: $\text{DfI}(W) = \|W^\top W - I\|_F^2$. Three theorems establish its connections to:
        - **Theorem 2**: The Hessian spectral norm is bounded by a function of layerwise DfI (loss landscape curvature).
        - **Theorem 3**: Low DfI implies high feature rank (effective utilization of all dimensions).
        - **Theorem 4**: Low DfI implies a tighter lower bound on activation fraction (absence of dormant neurons).
    - Design Motivation: Unifies multiple seemingly distinct symptoms of plasticity loss under a single optimizable metric.

3. **Closed-Form Solution and Efficient Implementation**

    - Function: Exactly solves the constrained optimization problem.
    - Mechanism: $\min_{\tilde{W}} \|W - \tilde{W}\|_F^2 \text{ s.t. } \tilde{W}^\top \tilde{W} = I$ is an orthogonal Procrustes problem, with solution given by the polar decomposition $\tilde{W}^* = W(W^\top W)^{-1/2}$. Approximated efficiently via 5-step Newton-Schulz iteration: `X = X/||X||; for _ in range(5): A = X.T @ X; X = 1.5*X - 0.5*X@A`
    - Design Motivation: SVD has complexity $O(d^3)$, whereas Newton-Schulz requires only matrix multiplications, adding <1% overhead; convergence is achieved in 5 iterations with no sensitivity to this hyperparameter.

### Application Strategy
- **Continual Learning**: A single orthogonalization is applied to all layers at task boundaries.
- **RL**: A single reinitialization is applied at the midpoint of training.
- **Layer-Specific Handling**: Convolutional layers are processed by spatial slicing; in ViT, only Q/K projections are orthogonalized.

## Key Experimental Results

### Main Results

| Benchmark | Task | FIRE vs. Best Baseline |
|-----------|------|------------------------|
| CIFAR-10 (ResNet-18) | Continual classification | Consistently outperforms S&P/DASH |
| CIFAR-100 (ViT-Tiny) | Continual classification | Consistently outperforms all baselines |
| Tiny-ImageNet (VGG-16) | Continual classification | Consistently outperforms all baselines |
| GPT-0.1B (WikiText→OWT) | LLM continual pre-training | Outperforms S&P (which requires tuning) |
| Atari (DQN, 3 games) | Discrete control | Outperforms S&P |
| HumanoidBench (SAC) | Continuous control | Competitive/superior |

### Ablation Study

| Analysis | Key Finding |
|----------|-------------|
| DfI comparison | FIRE achieves the lowest DfI and lowest SFE simultaneously |
| Loss landscape smoothness | FIRE produces smoother loss landscapes than S&P |
| Computational overhead | FIRE: 0.06s, 55MB vs. DASH: 69s, 2834MB |
| Newton-Schulz iterations | 5 iterations suffice; results are insensitive to this parameter |
| Full reinitialization | Severe degradation—erasing knowledge causes instability |

### Key Findings
- **No hyperparameter tuning required**: The constrained optimization automatically finds the optimal balance, whereas S&P/DASH require careful tuning.
- **Negligible computational cost**: 0.06s and 55MB, approximately 1000× faster than DASH.
- **DfI unifies multiple symptoms**: A single measure simultaneously captures curvature, rank, and dormant neurons—theoretically elegant and practically useful.
- **Effective for LLM continual pre-training**: Applicability to large models is validated on GPT-0.1B.

## Highlights & Insights
- **Principled over heuristic**: Modeling the stability-plasticity tradeoff as a constrained optimization problem rather than an ad hoc trick yields clear theoretical guarantees. The polar decomposition emerges naturally as the optimal solution—a manifestation of mathematical elegance.
- **DfI as a "unified theory" of plasticity**: Three theorems unify loss landscape curvature, feature rank, and dormant neurons under a single differentiable measure—a contribution that may prove more enduring than the method itself.
- **No tuning required**: S&P requires balancing shrinkage and noise; FIRE automatically identifies the optimal solution, which is critical for practical deployment.

## Limitations & Future Work
- **Validated only on small LLMs**: GPT-0.1B is too small; validation on models with 7B+ parameters is needed.
- **Assumes access to past data**: This assumption may not hold in certain continual learning scenarios.
- **Timing of orthogonalization**: The paper applies the operation once at the training midpoint or task boundary; automatic selection of the optimal timing remains unexplored.
- **Limited RL experiment scale**: Only 3 Atari games and HumanoidBench are evaluated; broader RL benchmarks (e.g., full MuJoCo suite) are not covered.

## Related Work & Insights
- **vs. S&P (Shrink & Perturb)**: S&P heuristically balances stability and plasticity by shrinking weights and adding random noise. FIRE demonstrates that orthogonal projection is the theoretically optimal solution, of which S&P is a suboptimal approximation.
- **Analogy with Neon**: Neon applies negative extrapolation in weight space to improve generative models; FIRE applies orthogonal projection in weight space to improve continual learning—both exemplify the paradigm of "simple parameter-space transformations yielding substantial gains."
- **Connection to LoongRL**: Plasticity loss during RL training is a practical concern; FIRE can potentially improve the stability of RL training algorithms such as GRPO.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — DfI as a unified measure and the orthogonal Procrustes closed-form solution represent outstanding theoretical contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Validated across vision, NLP, and RL, though the scale of experiments in each domain is limited.
- Writing Quality: ⭐⭐⭐⭐⭐ — Theoretical derivations are clear, the theorem chain is complete, and experimental organization is coherent.
- Value: ⭐⭐⭐⭐⭐ — Minimal and practical—a single line of code addresses a core challenge in continual learning; the DfI measure is broadly reusable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Plasticity as the Mirror of Empowerment](../../NeurIPS2025/others/plasticity_as_the_mirror_of_empowerment.md)
- [\[NeurIPS 2025\] Sample-Adaptivity Tradeoff in On-Demand Sampling](../../NeurIPS2025/others/sample-adaptivity_tradeoff_in_on-demand_sampling.md)
- [\[CVPR 2026\] Your Classifier Can Do More: Towards Balancing the Gaps in Classification, Robustness, and Generation](../../CVPR2026/others/your_classifier_can_do_more_towards_balancing_the.md)
- [\[AAAI 2026\] Guided Perturbation Sensitivity (GPS): Detecting Adversarial Text via Embedding Stability and Word Importance](../../AAAI2026/others/guided_perturbation_sensitivity_gps_detecting_adversarial_text_via_embedding_sta.md)
- [\[NeurIPS 2025\] Meta-learning three-factor plasticity rules for structured credit assignment with sparse feedback](../../NeurIPS2025/others/meta-learning_three-factor_plasticity_rules_for_structured_credit_assignment_wit.md)

</div>

<!-- RELATED:END -->
