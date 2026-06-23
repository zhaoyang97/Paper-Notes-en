---
title: >-
  [Paper Note] SafeDPO: A Simple Approach to Direct Preference Optimization with Enhanced Safety
description: >-
  [ICLR 2026][Alignment & RLHF][safety alignment] By revisiting the safety-constrained RLHF objective and proving it possesses a closed-form optimal policy, this work derives an equivalent tractable objective, SafeDPO. It requires only safety-aware data transformation and a safety margin term (one additional hyperparameter) on top of standard DPO. Without needing rewa
tags:
  - ICLR 2026
  - Alignment & RLHF
  - safety alignment
  - DPO
  - constrained optimization
  - safety margin
  - PKU-SafeRLHF
date: 2026-05-08
content_hash: e204af0f40d051d0
---
# SafeDPO: A Simple Approach to Direct Preference Optimization with Enhanced Safety

**Conference**: ICLR 2026 Oral  
**arXiv**: [2505.20065](https://arxiv.org/abs/2505.20065)  
**Code**: None  
**Area**: AI Safety / LLM Alignment  
**Keywords**: safety alignment, DPO, constrained optimization, safety margin, PKU-SafeRLHF

## TL;DR
By revisiting the safety-constrained RLHF objective and proving it possesses a closed-form optimal policy, this work derives an equivalent tractable objective, SafeDPO. It requires only safety-aware data transformation and a safety margin term (one additional hyperparameter) on top of standard DPO. Without needing reward/cost models, it achieves a 96.87% harmless rate on PKU-SafeRLHF-30K while maintaining competitive helpfulness, with training speeds 25x faster than SafeRLHF.

## Background & Motivation

**Background**: LLM deployment requires simultaneously ensuring helpfulness and safety. The mainstream approach is to introduce safety constraints into RLHF, such as SafeRLHF which utilizes Lagrangian methods to constrain cost functions.

**Limitations of Prior Work**: (a) SafeRLHF requires training a reward model + cost model + two value networks + online sampling, totaling 6 networks, which involves extreme complexity (35,200s vs 1,388s for DPO); (b) Methods like SACPO rely on approximate relaxations and cannot guarantee convergence to the optimal solution of the original safety-constrained problem; (c) Direct training on DPO with helpful/harmless data separately yields poor results—DPO-HELPFUL is helpful but unsafe, while DPO-HARMLESS is safe but unhelpful.

**Key Challenge**: A natural tension exists between helpfulness and safety—the more a model "cooperates" with user requests, the more helpful it is, but it also becomes more prone to generating harmful content. How can both be balanced in a single-stage training process?

**Goal**: Whether a closed-form solution to the safety-constrained optimization problem can be found, thereby allowing direct training via supervised learning like DPO and avoiding complex multi-stage pipelines.

**Key Insight**: Transforming safety constraints into a cost-augmented reward $r_c(x,y) = r(x,y)$ if safe, $-\infty$ if unsafe, ensures that the probability of unsafe responses in the optimal policy is zero, and this problem has a closed-form optimal policy.

**Core Idea**: Precisely encode safety constraints into the DPO loss through safety-aware data transformation (swapping unsafe winners) and a safety margin term, without requiring additional models.

## Method

### Overall Architecture
SafeDPO aims to resolve the problem of making a model simultaneously helpful and safe without incurring the costs of SafeRLHF's 6-network, online-sampling pipeline. The approach begins from a theoretical basis—first proving that the safety-constrained RLHF objective has a closed-form optimal policy, then back-substituting this optimal solution as in DPO to derive a pure supervised learning loss. Implementation-wise, the pipeline is nearly identical to standard DPO, modifying only two components: first, a safety-aware transformation is applied to preference data (replacing or discarding unsafe winners), and then a safety margin term $\Delta$ is inserted into the DPO loss. Starting from input $(x, y_w, y_l)$ preference pairs with binary safety labels, the transformed and margin-augmented loss outputs a policy that retains helpfulness while automatically avoiding unsafe regions.

### Key Designs

**1. Cost-augmented reward and closed-form optimal policy: Integrating hard constraints into the KL regularization framework**

Safety-constrained RLHF is originally a hard-constrained optimization problem (responses must satisfy $c(x,y) \leq 0$), which typically precludes a closed-form solution like DPO. SafeDPO's approach is to write the constraint directly into the reward: defining the cost-augmented reward $r_c(x,y) = r(x,y)$ when $c(x,y) \leq 0$ (safe), otherwise $r_c(x,y) = -\infty$ (unsafe). This modification converts the constrained problem back into a standard KL-regularized objective, allowing the application of DPO-style reparameterization to obtain the closed-form optimal policy:

$$\pi^*(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp\Big(\frac{1}{\beta} r_c(x,y)\Big).$$

The critical effect is that unsafe responses, due to $r_c = -\infty$, have their probabilities reduced to zero in the optimal policy, meaning safety constraints are encoded precisely rather than satisfied approximately. This distinguishes it from relaxation methods like SACPO, which use surrogate objectives that cannot guarantee convergence to the original constrained problem's solution.

**2. Safety-aware data transformation $T$: Reordering preference pairs with labels to align "safety" with the preference direction**

Given the optimal policy form above, preference signals must be aligned with "safety." The transformation $T$ reorganizes each preference pair $(x, y_w, y_l, h_w, h_l)$ based on binary safety labels $h$ ($h=1$ indicates unsafe): if the winner is safe ($h_w=0$), it remains unchanged; if the winner is unsafe while the loser is safe ($h_w=1, h_l=0$), the winner and loser are **swapped**, making the safe response the preferred item; if both are unsafe ($h_w=1, h_l=1$), the pair is **discarded**. This step is the most critical part of the method—ablation experiments indicate that adding a margin to other DPO variants without this transformation yields negligible gains, whereas safety increases significantly once this safety-aware transformation is introduced.

**3. SafeDPO loss and safety margin: Accelerating departure from danger zones without altering the optimal solution**

Finally, the transformed data is fed into the DPO loss with a margin:

$$\mathcal{L}(\theta; \Delta) = -\mathbb{E}\Big[\log \sigma\Big(\beta \log \frac{\pi_\theta(\tilde{y}_w|x)}{\pi_{\text{ref}}(\tilde{y}_w|x)} - \beta \log \frac{\pi_\theta(\tilde{y}_l|x)}{\pi_{\text{ref}}(\tilde{y}_l|x)} - (\tilde{h}_l - \tilde{h}_w)\Delta\Big)\Big].$$

The safety margin $\Delta \geq 0$ is activated only on "safe-unsafe" pairs (when $\tilde{h}_l - \tilde{h}_w = 1$), effectively increasing the advantage of safe responses relative to unsafe ones. Its elegance lies in **Proposition 4.4**, which proves that $\Delta$ does not change the optimal solution set (optimality invariance)—adding the margin does not push the model toward a different optimum but merely improves the optimization dynamics, allowing the training to move probability mass away from unsafe regions more quickly. Thus, $\Delta$ is purely an "accelerator" hyperparameter, which is why SafeDPO introduces only one extra hyperparameter.

### Loss & Training
Based on the standard DPO training framework, $\beta=0.1$, $\Delta=10$ (default), 3 epochs, lr=1e-6, and a cosine schedule. It requires only preference data + binary safety labels ($h \in \{0,1\}$), without the need for harmlessness preference labels or separate cost models.

## Key Experimental Results

### Main Results

| Method | Helpfulness | Harmless Rate (%) | Harmlessness | Training Time |
|------|--------|-----------|-------|---------|
| SFT | 0.00 | 45.49 | -0.77 | — |
| DPO-HELPFUL | 10.00 | 37.59 | -2.23 | — |
| DPO-HARMLESS | 0.52 | 75.69 | 3.14 | — |
| SafeRLHF | 4.23 | 88.97 | 3.63 | 35,200s |
| SACPO | 2.80 | 89.60 | 4.34 | — |
| **SafeDPO** | **4.61** | **96.87** | **5.97** | **1,388s** |

### Ablation Study

| Config | Key Findings |
|------|---------|
| $\Delta=0$ (no margin) | Still achieves high harmless rate, proving transformation $T$ is core |
| $\Delta=10$ (default) | Best trade-off |
| $\Delta=50$ (too large) | Helpfulness degradation (gradient saturation) |
| DPO+margin only | Limited safety improvement, far inferior to SafeDPO |
| 1.5B → 13B | Both safety and helpfulness improve with model scale |

### Key Findings
- **Data transformation is core**: Ablations prove that the safety-aware transformation $T$ contributes most of the safety gains, while $\Delta$ primarily improves optimization dynamics.
- **25× training speedup**: SafeDPO (1,388s) vs SafeRLHF (35,200s), requiring only 2 networks (policy + reference) vs 6.
- **GPT-4 evaluation reaches 100% harmless rate**: Under GPT-4 judgment, SafeDPO achieves a 100% harmless rate.
- **XSTest over-refusal**: SafeDPO’s over-refusal rate is 12.4% (SafeRLHF 3.2%), indicating that safety improvements come with a degree of over-refusal.

## Highlights & Insights
- **Theory-driven simple method**: The method is naturally derived from the closed-form solution of the safety-constrained problem rather than being an ad-hoc design. Proposition 4.4 proving that the margin does not change the optimal solution provides an elegant theoretical guarantee.
- **Broadly reusable data transformation**: The safety-aware winner/loser swap strategy can be applied to any preference learning method (IPO, KTO, etc.), not just DPO.
- **Only binary safety labels required**: There is no need to train cost models or fine-grained safety scores; a simple $h \in \{0,1\}$ label is sufficient, significantly reducing data annotation costs.

## Limitations & Future Work
- **Over-refusal problem**: The 12.4% over-refusal rate is higher than SafeRLHF (3.2%), which may be unacceptable for certain scenarios.
- **Experiments limited to PKU-SafeRLHF dataset**: Not yet validated on more safety benchmarks (e.g., Anthropic HH, BeaverTails).
- **Limitations of binary safety labels**: In reality, safety is a continuum; the coarse granularity of $h \in \{0,1\}$ may lose information.
- **Complementarity with AuxDPO**: SafeDPO addresses safety while AuxDPO addresses misspecification—could the two be combined?

## Related Work & Insights
- **vs SafeRLHF**: SafeRLHF uses a full constrained RL pipeline (reward model + cost model + PPO); SafeDPO proves that a closed-form solution can entirely avoid these complex components.
- **vs Why DPO is Misspecified**: SafeDPO addresses safety within the DPO framework but does not resolve the policy parameterization misspecification pointed out by AuxDPO. The two are orthogonal and complementary.
- **vs SACPO**: SACPO uses surrogate objectives for relaxation; SafeDPO proves the original problem can be solved directly.

## Rating
- Novelty: ⭐⭐⭐⭐ Closed-form derivation is novel, though the data transformation idea is relatively intuitive.
- Experimental Thoroughness: ⭐⭐⭐⭐ Sufficient multi-scale validation and ablation, despite the single dataset.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear theoretical derivation with a complete chain of propositions.
- Value: ⭐⭐⭐⭐⭐ Highly simple and usable, significantly lowering the threshold for safety alignment, suitable for industrial deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Autoregressive Direct Preference Optimization](../../ICML2026/llm_alignment/autoregressive_direct_preference_optimization.md)
- [\[ICML 2026\] Boosting Direct Preference Optimization with Penalization](../../ICML2026/llm_alignment/boosting_direct_preference_optimization_with_penalization.md)
- [\[ICLR 2026\] Token-Importance Guided Direct Preference Optimization (TI-DPO)](token-importance_guided_direct_preference_optimization.md)
- [\[ICLR 2026\] ActiveDPO: Active Direct Preference Optimization for Sample-Efficient Alignment](activedpo_active_direct_preference_optimization_for_sample-efficient_alignment.md)
- [\[ICLR 2026\] Alignment-Weighted DPO: A Principled Reasoning Approach to Improve Safety Alignment](alignment-weighted_dpo_a_principled_reasoning_approach_to_improve_safety_alignme.md)

</div>

<!-- RELATED:END -->
