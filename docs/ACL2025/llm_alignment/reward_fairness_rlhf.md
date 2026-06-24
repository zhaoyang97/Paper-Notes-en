---
title: >-
  [Paper Note] Towards Reward Fairness in RLHF: From a Resource Allocation Perspective
description: >-
  [ACL 2025][LLM Alignment][RLHF] This paper unifies various reward biases in RLHF (such as length bias, category bias, and social bias) under the concept of "reward unfairness". Drawing on resource allocation theory, the authors propose two bias-agnostic methods, Fairness Regularization and Fairness Coefficient, applied to reward model training and policy model training respectively. These methods simultaneously mitigate multiple biases and improve alignment quality without be…
tags:
  - "ACL 2025"
  - "LLM Alignment"
  - "RLHF"
  - "Reward Fairness"
  - "Resource Allocation"
  - "Bias Mitigation"
  - "Preference Learning"
date: 2026-05-08
content_hash: 0e07ff88d10328c1
---

# Towards Reward Fairness in RLHF: From a Resource Allocation Perspective

**Conference**: ACL 2025  
**arXiv**: [2505.23349](https://arxiv.org/abs/2505.23349)  
**Code**: [https://github.com/shoyua/Towards-Reward-Fairness](https://github.com/shoyua/Towards-Reward-Fairness)  
**Area**: LLM Alignment  
**Keywords**: RLHF, Reward Fairness, Resource Allocation, Bias Mitigation, Preference Learning

## TL;DR

This paper unifies various reward biases in RLHF (such as length bias, category bias, and social bias) under the concept of "reward unfairness". Drawing on resource allocation theory, the authors propose two bias-agnostic methods, Fairness Regularization and Fairness Coefficient, applied to reward model training and policy model training respectively. These methods simultaneously mitigate multiple biases and improve alignment quality without being tailored to any specific bias.

## Background & Motivation

- **Background**: The core assumption of RLHF is that the Reward Model (RM) can accurately proxy human preferences. The RM is trained under the Bradley-Terry model, ensuring $r(y^w) > r(y^l)$ for each preference pair $(x, y^w, y^l)$. However, the correctness of the relative order within a pair does not imply a reasonable absolute reward distribution across pairs — data from different categories or lengths may receive systematically unequal rewards, thereby misleading the optimization direction of the policy model.
- **Limitations of Prior Work**: Existing works propose customized solutions for different biases: length regularization (such as R-DPO) for length bias, and model ensembling or diversified preference learning for category bias. This fragmented strategy of "one method per bias" is not scalable and cannot handle unknown types of biases.
- **Key Challenge**: The BT model only optimizes the within-pair margin $r(y^w) - r(y^l)$ without constraining the absolute reward distribution across pairs. Consequently, the reward of an undesired response $y_1^l$ might be higher than that of a desired response $y_2^w$ from another pair, leading the policy model to optimize in the wrong direction when maximizing absolute rewards.
- **Goal**: To view various reward biases as different manifestations of "reward unfairness", leverage a unified fairness metric from resource allocation theory to balance utility and fairness, and propose a bias-agnostic generalized solution.

## Method

### Overall Architecture

The preference learning is modeled as a resource allocation problem: rewards are the "resources" to be allocated to different data groups (divided by length, category, demographics, etc.). The optimization goal is extended from sole utility maximization ($\max U(\mathbf{a})$, i.e., standard BT loss) to joint utility-fairness optimization. Two combination methodologies are proposed: Fairness Regularization (additive form) and Fairness Coefficient (multiplicative form). The framework is instantiated in two scenarios: (1) training a fair reward model (Fairness RM), and (2) training a fair policy model (Fairness DPO).

### Key Designs

1. **Unified Fairness Metric Function**
    - **Function**: To provide a family of fairness metrics that satisfy three major properties: continuity, homogeneity, and monotonicity.
    - **Mechanism**: Adopting the unified fairness metric proposed by Lan et al. (2010): $f_\tau(\mathbf{a}) = \text{sign}(1-\tau) \cdot \left[\sum_{i=1}^{n}\left(\frac{a_i}{\sum_j a_j}\right)^{1-\tau}\right]^{1/\tau}$, where $a_i = r_\phi(y^w_i) - r_\phi(y^l_i)$ is the reward margin of the $i$-th preference pair, and $\tau$ controls the shape of the family of fairness functions ($\tau = -1$ degenerates to Jain's Index).
    - **Design Motivation**: Homogeneity ensures that the fairness metric is independent of the reward scale, bypassing the constraint on total resources; the unified parameter $\tau$ allows flexible selection from a family of fairness functions; experiments show insensitivity to $\tau$, with all values superior to the baseline.

2. **Fairness Regularization (FR, Additive Fairness Regularization)**
    - **Function**: To introduce fairness constraints as a regularization term in the training objective.
    - **Mechanism**: The loss function is $\mathcal{L}_{\text{FR}} = -\mathbb{E}[\log\sigma(a_i)] - \alpha \cdot f_\tau(\mathbf{a})$, where the first term is the standard BT utility loss and the second term is the fairness regularization. The hyperparameter $\alpha$ controls the fairness-utility trade-off. For the DPO scenario, $a_i$ is replaced with the implicit reward margin $\beta\log\frac{\pi_\theta(y^w|x)}{\pi_{\text{ref}}(y^w|x)} - \beta\log\frac{\pi_\theta(y^l|x)}{\pi_{\text{ref}}(y^l|x)}$.
    - **Design Motivation**: The additive form is simple to implement, and the gradient of the fairness term is independent of the utility term, making the trade-off easy to adjust; setting $\alpha=0.1$ achieves the best balance across multiple benchmarks.

3. **Fairness Coefficient (FC, Multiplicative Fairness Coefficient)**
    - **Function**: To dynamically scale the utility loss as a multiplicative coefficient.
    - **Mechanism**: The loss function is $\mathcal{L}_{\text{FC}} = -\mathbb{E}[\log\sigma(a_i)] \cdot f_\tau(\mathbf{a})^\gamma$, where the fairness metric serves as the weight coefficient for the utility loss. When the reward distribution is less fair (smaller $f_\tau$), the overall loss is reduced — the gradient automatically guides the optimization towards a fairer distribution.
    - **Design Motivation**: The multiplicative form allows adaptive adjustment of fairness — automatically increasing the regularization effect when unfair, while maintaining normal training when fair. It eliminates the need to manually tune $\alpha$, requiring only the setting of $\gamma$.

### Two Application Scenarios

| Scenario | Goal | Input | Definition of Allocation Vector $a_i$ | Output |
|------|------|------|-------|------|
| Fairness RM (Validation) | Train a fair reward model | Preference pairs $(x, y^w, y^l)$ | $r_\phi(y^w) - r_\phi(y^l)$ | Fair reward scorer, used for BoN data selection |
| Fairness DPO (RL) | Train a fair policy model | Preference pairs + reference model $\pi_{\text{ref}}$ | Implicit reward margin | Preference-aligned and fair generative model |

## Key Experimental Results

### Reward Model Fairness Validation (LLaMA3-SFT Base)

| Evaluator | Reward Bench Avg. | Chat | Chat Hard | Reasoning | Safety | HH-RLHF Avg. |
|--------|-------------------|------|-----------|-----------|--------|--------------|
| BT RM (Baseline) | 78.11 | 93.02 | 57.02 | 84.98 | 77.43 | 73.81 |
| FR RM | **78.38** | **94.41** | 57.02 | 83.86 | **78.24** | 73.55 |
| FC RM | 77.50 | **94.41** | 53.29 | **85.53** | 76.76 | 73.96 |

The fair RM significantly improves the consistency of reward distributions between the Helpful and Harmless categories without degrading accuracy.

### Policy Model Alignment Quality

| Base Model | Method | AlpacaEval2 LC WR | AlpacaEval2 WR | MT-Bench |
|------|------|-------------------|----------------|----------|
| LLaMA3 | DPO | 16.71 | 14.23 | 6.46 |
| LLaMA3 | R-DPO (Length Regularization) | 20.87 | 11.16 | 6.48 |
| LLaMA3 | KTO | 19.44 | 16.64 | 6.64 |
| LLaMA3 | **FR DPO** | 20.48 | 15.74 | **6.70** |
| LLaMA3 | **FC DPO** | **21.10** | **16.96** | 6.58 |
| Qwen2.5 | DPO | 18.93 | 13.18 | 6.59 |
| Qwen2.5 | **FR DPO** | **21.05** | **15.25** | **7.24** |
| Qwen2.5 | **FC DPO** | 19.72 | 14.53 | 7.00 |

### Key Findings

- FR/FC DPO consistently outperforms standard DPO and customized R-DPO across both base models, validating the effectiveness of the bias-agnostic approach.
- Fairness constraints do not compromise alignment quality but actually enhance it — as mitigating bias allows the model to learn more accurate preference signals.
- The fair RM achieves equivalent performance with fewer samples in BoN data selection, demonstrating higher sampling efficiency.
- On the CrowS-Pairs social bias dataset, the gap in reward distributions between stereotypical and non-stereotypical outputs is significantly reduced for FR/FC RM.
- Ablation studies show that all fairness functions within the range of $\tau \in [-5, 10]$ outperform the baseline, indicating that the method is robust to $\tau$; $\alpha = 0.1$ represents the optimal utility-fairness trade-off.

## Highlights & Insights

- The **resource allocation perspective** is the core theoretical innovation — bridging RLHF bias with economic fair division theory, and providing a unified analytical framework.
- A **bias-agnostic** approach is more practically meaningful than customized solutions — because bias types in real-world scenarios are often unknown or blended.
- It reveals a structural flaw in the BT model: within-pair correctness does not equal cross-pair fairness, which is the root cause of policy model misguidance.
- The "plug-and-play" characteristic of the FC method holds direct application value for deployed RLHF systems — bypassing the need to retrain a reward model.

## Limitations & Future Work

- The fairness metric requires pre-defining data grouping methods (by length, category, or demographics), and the quality of the grouping affects performance.
- Hyperparameters $\alpha$ and $\gamma$ still require tuning on a validation set.
- The method has only been validated on BT RM + DPO and hasn't been extended to other RL algorithms such as PPO or GRPO.
- The relationship between reward unfairness and reward hacking has not been explored in depth.

## Related Work & Insights

- **vs R-DPO (Length Regularization)**: Only mitigates length bias; Ours generalizes to mitigate multiple biases with superior performance (LC WR 21.10 vs 20.87).
- **vs ODIN (Disentangled Rewards)**: Mitigates hacking through multi-dimensional reward disentanglement; Ours constrains distribution fairness from a resource allocation perspective — these perspectives are complementary.
- **vs Original DPO**: DPO bypasses explicit RM but fits rewards implicitly; Ours proves its implicit rewards likewise suffer from unfairness, making FR/FC directly applicable.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Unifying the understanding of reward biases through the resource allocation perspective is novel, and the design of the fairness metric family is elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Evaluated across RM validation, DPO policy, social bias, data selection, and ablation studies, with validation on two base models.
- **Writing Quality**: ⭐⭐⭐⭐ Rigorous theoretical derivation, progressing logically from resource allocation to fairness metrics and specific methodologies.
- **Value**: ⭐⭐⭐⭐ FC is plug-and-play, robust to $\tau$, and does not sacrifice utility, holding direct value for actual RLHF deployments.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Reward Generalization in RLHF: A Topological Perspective](reward_generalization_in_rlhf_a_topological_perspective.md)
- [\[ACL 2025\] Aligning to What? Limits to RLHF Based Alignment](aligning_to_what_limits_to_rlhf_based_alignment.md)
- [\[ICML 2025\] Can RLHF be More Efficient with Imperfect Reward Models? A Policy Coverage Perspective](../../ICML2025/llm_alignment/can_rlhf_be_more_efficient_with_imperfect_reward_models_a_policy_coverage_perspe.md)
- [\[ACL 2025\] Rethinking Reward Model Evaluation Through the Lens of Reward Overoptimization](rethinking_reward_model_evaluation_through_the_lens_of_reward_overoptimization.md)
- [\[NeurIPS 2025\] What Makes a Reward Model a Good Teacher? An Optimization Perspective](../../NeurIPS2025/llm_alignment/what_makes_a_reward_model_a_good_teacher_an_optimization_perspective.md)

</div>

<!-- RELATED:END -->
