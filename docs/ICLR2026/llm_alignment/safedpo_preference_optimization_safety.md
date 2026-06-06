---
title: >-
  [Paper Note] SafeDPO: A Simple Approach to Direct Preference Optimization with Enhanced Safety
description: >-
  [ICLR 2026 Oral][LLM Alignment][safety alignment] This work revisits the safety-constrained RLHF objective, proves the existence of a closed-form optimal policy, and derives an equivalent tractable objective…
tags:
  - "ICLR 2026 Oral"
  - "LLM Alignment"
  - "safety alignment"
  - "DPO"
  - "constrained optimization"
  - "safety margin"
  - "PKU-SafeRLHF"
date: 2026-05-08
content_hash: a4b7d46cfae1ad22
---

# SafeDPO: A Simple Approach to Direct Preference Optimization with Enhanced Safety

**Conference**: ICLR 2026 Oral
**arXiv**: [2505.20065](https://arxiv.org/abs/2505.20065)  
**Code**: None  
**Area**: AI Safety / LLM Alignment
**Keywords**: safety alignment, DPO, constrained optimization, safety margin, PKU-SafeRLHF

## TL;DR
This work revisits the safety-constrained RLHF objective, proves the existence of a closed-form optimal policy, and derives an equivalent tractable objective, SafeDPO. The method requires only a safety-aware data transformation and a safety margin term (one additional hyperparameter) on top of standard DPO, without reward or cost models. It achieves a 96.87% harmlessness rate on PKU-SafeRLHF-30K while maintaining competitive helpfulness, and trains 25× faster than SafeRLHF.

## Background & Motivation

**Background**: Deploying LLMs requires simultaneously ensuring helpfulness and safety. The mainstream approach introduces safety constraints into RLHF; for example, SafeRLHF employs a Lagrangian method to constrain a cost function.

**Limitations of Prior Work**: (a) SafeRLHF requires training a reward model, a cost model, two value networks, and online sampling — six networks in total — resulting in extreme complexity (35,200s training vs. 1,388s for DPO); (b) methods such as SACPO rely on approximate relaxations and cannot guarantee convergence to the solution of the original safety-constrained problem; (c) training separately on helpfulness or harmlessness data with DPO yields poor results — DPO-HELPFUL is useful but unsafe, while DPO-HARMLESS is safe but unhelpful.

**Key Challenge**: A natural tension exists between helpfulness and safety — the more a model complies with user requests, the more helpful it is, yet the more likely it is to generate harmful content. How can both be addressed within a single training stage?

**Goal**: Can a closed-form solution to the safety-constrained optimization problem be found, enabling direct supervised training analogous to DPO, thereby avoiding complex multi-stage pipelines?

**Key Insight**: The safety constraint is reformulated as a cost-augmented reward $r_c(x,y) = r(x,y)$ if safe, $-\infty$ if unsafe. This ensures that unsafe responses receive zero probability under the optimal policy, and the resulting problem admits a closed-form optimal solution.

**Core Idea**: Safety constraints are precisely encoded into the DPO loss via a safety-aware data transformation (swapping unsafe winners) and a safety margin term, without requiring any additional models.

## Method

### Overall Architecture
SafeDPO introduces two modifications to the standard DPO pipeline: (1) a safety-aware data transformation $T$ that swaps winner and loser when the winner is unsafe and the loser is safe, and discards pairs where both responses are unsafe; (2) a safety margin $\Delta$ added to the DPO loss to increase the margin between safe and unsafe response pairs.

### Key Designs

1. **Cost-Augmented Reward and Closed-Form Optimal Policy**

    - Function: Converts hard safety constraints into explicit reward modifications.
    - Mechanism: $r_c(x,y) = r(x,y)$ if $c(x,y) \leq 0$ (safe), $-\infty$ otherwise (unsafe). The resulting closed-form optimal policy is $\pi^*(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp(\frac{1}{\beta} r_c(x,y))$, under which unsafe responses automatically receive zero probability.
    - Design Motivation: Transforms an apparently intractable hard-constraint problem into a standard KL-regularized framework, enabling DPO-style reparameterization.

2. **Safety-Aware Data Transformation $T$**

    - Function: Reorganizes preference pairs according to safety labels.
    - Mechanism: Given $(x, y_w, y_l, h_w, h_l)$ where $h=1$ denotes unsafe: (a) $h_w=0$: keep unchanged; (b) $h_w=1, h_l=0$: **swap** winner and loser; (c) $h_w=1, h_l=1$: **discard**.
    - Design Motivation: Ablation studies confirm this step is the most critical — adding a margin alone to other DPO variants yields limited improvement, whereas the safety-aware transformation substantially enhances safety.

3. **SafeDPO Loss and Safety Margin**

    - Function: Augments the standard DPO loss with a safety margin term.
    - Mechanism: $\mathcal{L}(\theta; \Delta) = -\mathbb{E}[\log \sigma(\beta \log \frac{\pi_\theta(\tilde{y}_w|x)}{\pi_{\text{ref}}(\tilde{y}_w|x)} - \beta \log \frac{\pi_\theta(\tilde{y}_l|x)}{\pi_{\text{ref}}(\tilde{y}_l|x)} - (\tilde{h}_l - \tilde{h}_w)\Delta)]$. The margin $\Delta \geq 0$ is activated only on safe–unsafe pairs ($\tilde{h}_l - \tilde{h}_w = 1$), amplifying the advantage of safe responses.
    - Design Motivation: **Proposition 4.4** proves that $\Delta$ does not alter the optimal solution set (optimality invariance) but improves optimization dynamics by accelerating divergence from the unsafe region.

### Loss & Training
Training follows the standard DPO framework with $\beta=0.1$, $\Delta=10$ (default), 3 epochs, lr=1e-6, and a cosine schedule. Only preference data and binary safety labels ($h \in \{0,1\}$) are required; fine-grained harmlessness preference labels are unnecessary.

## Key Experimental Results

### Main Results

| Method | Helpfulness | Harmless Rate (%) | Harmlessness | Training Time |
|--------|-------------|-------------------|--------------|---------------|
| SFT | 0.00 | 45.49 | -0.77 | — |
| DPO-HELPFUL | 10.00 | 37.59 | -2.23 | — |
| DPO-HARMLESS | 0.52 | 75.69 | 3.14 | — |
| SafeRLHF | 4.23 | 88.97 | 3.63 | 35,200s |
| SACPO | 2.80 | 89.60 | 4.34 | — |
| **SafeDPO** | **4.61** | **96.87** | **5.97** | **1,388s** |

### Ablation Study

| Configuration | Key Findings |
|---------------|--------------|
| $\Delta=0$ (no margin) | Still achieves high harmlessness rate, confirming that transformation $T$ is the core component |
| $\Delta=10$ (default) | Best helpfulness–safety trade-off |
| $\Delta=50$ (too large) | Helpfulness degrades due to gradient saturation |
| DPO + margin only | Limited safety improvement, far inferior to SafeDPO |
| 1.5B → 13B | Both safety and helpfulness improve with model scale |

### Key Findings
- **Data transformation is the core component**: Ablations show that the safety-aware transformation $T$ accounts for the majority of safety gains; $\Delta$ primarily improves optimization dynamics.
- **25× training speedup**: SafeDPO completes in 1,388s vs. 35,200s for SafeRLHF, requiring only 2 networks (policy + reference) instead of 6.
- **100% harmlessness rate under GPT-4 evaluation**: SafeDPO achieves a 100% harmlessness rate as judged by GPT-4.
- **XSTest over-refusal**: SafeDPO exhibits a 12.4% over-refusal rate (vs. 3.2% for SafeRLHF), indicating that the safety gains come with a degree of over-refusal.

## Highlights & Insights
- **Theory-driven simplicity**: The method is naturally derived from the closed-form solution to the safety-constrained problem rather than being an ad-hoc design. Proposition 4.4's proof that the margin does not alter the optimal solution set provides an elegant theoretical guarantee.
- **Broadly reusable data transformation**: The safety-aware winner/loser swapping strategy can be applied to any preference learning method (IPO, KTO, etc.), not limited to DPO.
- **Binary safety labels suffice**: No cost model training or fine-grained safety scoring is required; a simple binary label $h \in \{0,1\}$ is sufficient, substantially reducing data annotation costs.

## Limitations & Future Work
- **Over-refusal**: The 12.4% over-refusal rate exceeds that of SafeRLHF (3.2%) and may be unacceptable in certain application scenarios.
- **Limited benchmark coverage**: Experiments are confined to the PKU-SafeRLHF dataset; validation on broader safety benchmarks (e.g., Anthropic HH, BeaverTails) is absent.
- **Coarse safety labels**: Real-world safety exists on a continuum, and the binary $h \in \{0,1\}$ granularity may discard useful information.
- **Complementarity with AuxDPO**: SafeDPO addresses safety while AuxDPO addresses misspecification; whether the two can be combined warrants investigation.

## Related Work & Insights
- **vs. SafeRLHF**: SafeRLHF employs a full constrained RL pipeline (reward model + cost model + PPO); SafeDPO demonstrates that a closed-form solution can entirely eliminate these complex components.
- **vs. Why DPO is Misspecified**: SafeDPO addresses safety within the DPO framework but does not resolve the parameterized policy misspecification issue identified by AuxDPO. The two approaches are orthogonal and complementary.
- **vs. SACPO**: SACPO relaxes the original objective via a surrogate; SafeDPO proves that the original problem can be solved directly.

## Rating
- Novelty: ⭐⭐⭐⭐ Closed-form derivation is novel, though the data transformation idea is relatively intuitive.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-scale validation and comprehensive ablations, but limited to a single dataset.
- Writing Quality: ⭐⭐⭐⭐⭐ Theoretical derivations are clear with a complete chain of propositions.
- Value: ⭐⭐⭐⭐⭐ Highly practical and minimal; significantly lowers the barrier to safety alignment, making it well-suited for industrial deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] $\varphi$-DPO: Fairness Direct Preference Optimization Approach to Continual Learning in Large Multimodal Models](../../CVPR2026/llm_alignment/φ-dpo_fairness_direct_preference_optimization_approach_to_continual_learning_in_.md)
- [\[AAAI 2026\] Rethinking Direct Preference Optimization in Diffusion Models](../../AAAI2026/llm_alignment/rethinking_direct_preference_optimization_in_diffusion_models.md)
- [\[ACL 2026\] Topology-Enhanced Alignment for Large Language Models: Trajectory Topology Loss and Topological Preference Optimization](../../ACL2026/llm_alignment/topology-enhanced_alignment_for_large_language_models_trajectory_topology_loss_a.md)
- [\[ICLR 2026\] Token-Importance Guided Direct Preference Optimization (TI-DPO)](token-importance_guided_direct_preference_optimization.md)
- [\[ICML 2026\] TUR-DPO: Topology- and Uncertainty-Aware Direct Preference Optimization](../../ICML2026/llm_alignment/tur-dpo_topology-_and_uncertainty-aware_direct_preference_optimization.md)

</div>

<!-- RELATED:END -->
