---
title: >-
  [Paper Note] Bone Soups: A Seek-and-Soup Model Merging Approach for Controllable Multi-Objective Generation
description: >-
  [ACL 2025][Model Compression][Model Merging] This paper proposes the Bone Soup model merging approach, which addresses the suboptimality of single-objective model merging in Rewarded Soup by first constructing "backbone rewards" (combinations of multi-objective rewards) to train backbone models, and then using a symmetric circulant matrix mapping to determine merging coefficients. This achieves superior Pareto frontiers and better controllability across three multi-objective…
tags:
  - "ACL 2025"
  - "Model Compression"
  - "Model Merging"
  - "Multi-objective Generation"
  - "Controllable Generation"
  - "Pareto Optimality"
  - "Reinforcement Learning"
date: 2026-05-08
content_hash: bc3aace5acc3997a
---

# Bone Soups: A Seek-and-Soup Model Merging Approach for Controllable Multi-Objective Generation

**Conference**: ACL 2025  
**arXiv**: [2502.10762](https://arxiv.org/abs/2502.10762)  
**Code**: [https://github.com/andyclsr/BoneSoups](https://github.com/andyclsr/BoneSoups)  
**Area**: Others  
**Keywords**: Model Merging, Multi-objective Generation, Controllable Generation, Pareto Optimality, Reinforcement Learning  

## TL;DR

This paper proposes the Bone Soup model merging approach, which addresses the suboptimality of single-objective model merging in Rewarded Soup by first constructing "backbone rewards" (combinations of multi-objective rewards) to train backbone models, and then using a symmetric circulant matrix mapping to determine merging coefficients. This achieves superior Pareto frontiers and better controllability across three multi-objective generation tasks.

## Background & Motivation

**Background**: Controllable Multi-Objective Generation (CMOG) requires LLMs to dynamically adjust their generation strategies based on user preferences at inference time (e.g., balancing factuality vs. completeness). Model merging (or model soup) represents a mainstream paradigm—separately training specialized models for each objective, and linearly interpolating parameters using user preference weights during inference.

**Limitations of Prior Work**: Existing methods like Rewarded Soup train a specialized model for each individual objective and directly use preference weights as merging coefficients. However, this neglects the interactions between competing objectives—when different objective reward functions have different curvatures, simple weighted merging leads to results that significantly deviate from the optimal solution.

**Key Challenge**: Specialized models trained with single-objective rewards are overly specialized to their respective goals, leading to an uneven distribution of optimal points in the model parameter space. Linear interpolation is restricted to points on the straight line connecting these two specialized points, whereas true Pareto-optimal solutions may lie along a curve.

**Goal**: How to select superior "base models" (backbone models) so that their linear combinations can better approximate the Pareto frontier.

**Key Insight**: Instead of directly optimizing individual rewards, multiple rewards are first combined into "backbone rewards" to allow backbone models to simultaneously consider multiple objectives. Mathematical proof is provided to demonstrate that this approach outperforms Rewarded Soup over most preference intervals.

**Core Idea**: First "select high-quality ingredients" (construct backbone rewards to train backbone models), then "brew the soup" (merge backbone models according to user preferences).

## Method

### Overall Architecture
Given $n$ objectives and their corresponding reward functions $\{r_i\}_{i=1}^n$, as well as user preference weights $\bm{\mu}$: (1) Generate backbone rewards $h_j = \sum_i B_{ji} r_i$ via a basis vector construction method (where $B$ is the combination weight matrix); (2) Train backbone models $\{\bm{\theta}_j^{\text{bone}}\}$ using the backbone rewards through Multi-Objective Reinforcement Learning (MORL); (3) At inference time, compute merging coefficients $\bm{\lambda}$ from user preferences $\bm{\mu}$ using a symmetric circulant matrix mapping, and merge the backbone models.

### Key Designs

1. **Backbone Reward Construction**:

    - Function: Combines original single-objective rewards into backbone rewards that account for multi-objective interactions.
    - Mechanism: Constructs a transformation matrix $B$ such that $[h_1,...,h_n]^T = B[r_1,...,r_n]^T$. $B$ takes the form of a symmetric circulant matrix, parameterized by $\beta \in (1/2, 1)$. For example, in a two-objective setting, $h_1 = \beta r_1 + (1-\beta)r_2$ and $h_2 = (1-\beta)r_1 + \beta r_2$.
    - Design Motivation: Mathematical proofs (Theorem 1) show that for distinguishable quadratic reward functions, Bone Soup strictly outperforms Rewarded Soup within the preference interval $\mu \in (\frac{1-\sqrt{2\beta^2-2\beta+1}}{2}, \frac{1+\sqrt{2\beta^2-2\beta+1}}{2})$, which has a length of at least $\frac{\sqrt{2}}{2} \approx 0.71$.

2. **Backbone Model Training**:

    - Function: Trains backbone models through multi-objective reinforcement learning using backbone rewards.
    - Mechanism: Uses the backbone reward $h_j$ as the reward signal for PPO training to obtain the backbone model $\bm{\theta}_j^{\text{bone}}$.
    - Design Motivation: The backbone models have already accounted for trade-offs during training, placing them in areas of the parameter space closer to the true Pareto frontier.

3. **Merging Coefficient Mapping**:

    - Function: Automatically computes merging coefficients based on user preference weights.
    - Mechanism: Leverages the inverse mapping of the symmetric circulant matrix $B$ to compute merging coefficients $\bm{\lambda} = B^{-1}\bm{\mu}$ from preference weights $\bm{\mu}$. The final merged model is obtained as $\bar{\bm{\theta}} = \sum_j \lambda_j \bm{\theta}_j^{\text{bone}}$.
    - Design Motivation: Ensures alignment between merging coefficients and the construction of backbone rewards. When the user preferences exactly match the combination weights of a specific backbone reward, the corresponding backbone model can be directly output as the optimal solution.

### Loss & Training
- Backbone models are trained using PPO with backbone rewards.
- $\beta$ is selected from $\{0.6, 0.7, 0.8\}$, requiring only 20% of the training steps to evaluate using hypervolume.
- Optional extrapolation step: $\hat{\bm{\theta}}^b = (1+\alpha)\hat{\bm{\theta}} - \alpha\bm{\theta}_{\text{sft}}$ is used for further enhancement.

## Key Experimental Results

### Main Results

| Task | Method | Pareto Frontier | Controllability |
|------|------|-----------|--------|
| Long Form QA (Factuality vs. Completeness) | Rewarded Soup | Suboptimal Frontier | 10–11 points |
| Long Form QA (Factuality vs. Completeness) | **Bone Soup** | **Close to or Even Exceeds Oracle MORLHF** | 10–11 points |
| Helpful Assistant (Helpful vs. Harmless) | Rewarded Soup | Standard Frontier | - |
| Helpful Assistant (Helpful vs. Harmless) | **Bone Soup** | **Significantly Outperforms RS and MOD** | - |
| Reddit Summary (Faithful vs. Preference) | Rewarded Soup | Standard Frontier | - |
| Reddit Summary (Faithful vs. Preference) | **Bone Soup** | **Significantly Outperforms RS** | - |

### Ablation Study

| Configuration | Effect | Description |
|------|------|------|
| ABA (Backbone-only training, direct preference merging) | Unstable | Some trade-offs are worse than RS, validating the necessity of a two-stage process |
| β=0.6 | Weak | Backbone rewards are overly close to the original rewards |
| β=0.7 | Best | Moderate mixture |  
| β=0.8 | Decent | Stronger mixture |
| Extrapolation (α=0.1-0.5) | Further Improvement | Reduces the bias of SFT initialization models |

### Key Findings
- Bone Soup is equally effective in three-objective settings, with its Pareto frontier even dominating the frontier of Oracle MORLHF.
- GPT-4 evaluation yields consistent conclusions with reward model evaluations, confirming the robustness of the improvement.
- Merging backbone models directly (ABA, without mapped coefficients) is unstable, illustrating that correct mapping of merging coefficients is crucial.
- Training overhead is comparable to Rewarded Soup (requiring the training of $n$ models likewise), and inference overhead is identical.

## Highlights & Insights

- **Mathematical proof of the suboptimality of model merging** along with a proposed solution—Theorem 1 strictly proves that Bone Soup outperforms Rewarded Soup over at least 71% of the preference intervals, providing a theoretical guarantee rather than mere empirical observation.
- **The metaphor of "selecting ingredients before brewing soup"** precisely conveys the core idea—one cannot just merge arbitrary models; base models must be carefully crafted.
- **The design of symmetric circulant matrices** is simple and elegant—introducing only a single hyperparameter $\beta$, which guarantees theoretical properties while remaining easy to implement.
- The methodology can be directly transferred to other fields of model merging, such as computer vision and multimodal scenarios.

## Limitations & Future Work

- Relies solely on automatic evaluators (reward models + GPT-4) without human evaluation.
- Experiments are only conducted on 2–3 objectives; scalability to a higher number of objectives remains unverified.
- $\beta$ still requires searching, lacking an adaptive selection mechanism.
- Theoretical analysis is limited to quadratic reward functions, whereas real-world reward functions are more complex.

## Related Work & Insights

- **vs. Rewarded Soup**: RS performs direct model merging using single-objective models and ignores objective interactions. Bone Soup incorporates objective interactions during training via backbone rewards, yielding superior merged results.
- **vs. MORLHF**: MORLHF trains a separate model for each preference point (enumeration), which is highly costly and uncontrollable at inference time. Bone Soup only requires training $n$ backbone models.
- **vs. MOD (Decoding-time method)**: MOD merges multi-model outputs at the logit level, which is flexible but introduces inference overhead. Bone Soup merges at the parameter level, resulting in zero inference overhead.

## Rating

- Novelty: ⭐⭐⭐⭐ Theory-driven approach—Theorem 1 strictly proves that Bone Soup outperforms Rewarded Soup on at least 71% of the preference intervals, and the parameterization of the symmetric circulant matrix is simple and elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated across three tasks (Long Form QA, Helpful Assistant, and Reddit Summary) with eight reward models and two base models. Ablations are comprehensive, though human evaluation is missing.
- Writing Quality: ⭐⭐⭐⭐ The motivation is highly persuasive, demonstrating the suboptimality of Rewarded Soup through Example 1, alongside an intuitive oracle comparison in Figure 3.
- Value: ⭐⭐⭐⭐ Offers substantial contributions to the model merging field; the "selecting ingredients before brewing soup" mindset is generalizable and transferrable to vision, multimodal, and other model merging settings.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] From Parameter to Representation: A Closed-Form Approach for Controllable Model Merging](../../AAAI2026/model_compression/from_parameter_to_representation_a_closed-form_approach_for_controllable_model_m.md)
- [\[ACL 2025\] Unraveling LoRA Interference: Orthogonal Subspaces for Robust Model Merging](osrm_lora_merging_orthogonal.md)
- [\[NeurIPS 2025\] LT-Soups: Bridging Head and Tail Classes via Subsampled Model Soups](../../NeurIPS2025/model_compression/lt-soups_bridging_head_and_tail_classes_via_subsampled_model_soups.md)
- [\[ICCV 2025\] FREE-Merging: Fourier Transform for Efficient Model Merging](../../ICCV2025/model_compression/free-merging_fourier_transform_for_efficient_model_merging.md)
- [\[ACL 2025\] DAC: A Dynamic Attention-aware Approach for Task-Agnostic Prompt Compression](dac_prompt_compression.md)

</div>

<!-- RELATED:END -->
