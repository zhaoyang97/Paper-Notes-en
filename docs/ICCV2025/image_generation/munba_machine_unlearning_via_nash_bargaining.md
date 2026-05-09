---
title: >-
  [Paper Note] MUNBa: Machine Unlearning via Nash Bargaining
description: >-
  [ICCV 2025][Image Generation][Machine Unlearning] This work formulates Machine Unlearning (MU) as a two-player cooperative bargaining game and derives a closed-form solution via Nash bargaining theory to simultaneously address gradient conflict and gradient dominance between the forgetting and retention objectives, achieving an optimal balance between unlearning and preservation across both classification and generation tasks.
tags:
  - ICCV 2025
  - Image Generation
  - Machine Unlearning
  - Nash Bargaining
  - Gradient Conflict
  - Pareto Optimality
  - Diffusion Models
  - CLIP
  - Multi-Objective Optimization
date: 2026-05-08
content_hash: d3f0569b1dfeb64e
---

# MUNBa: Machine Unlearning via Nash Bargaining

**Conference**: ICCV 2025
**arXiv**: [2411.15537](https://arxiv.org/abs/2411.15537)
**Code**: Not released
**Area**: Machine Unlearning / Image Generation Safety
**Keywords**: Machine Unlearning, Nash Bargaining, Gradient Conflict, Pareto Optimality, Diffusion Models, CLIP, Multi-Objective Optimization

## TL;DR

This work formulates Machine Unlearning (MU) as a two-player cooperative bargaining game and derives a closed-form solution via Nash bargaining theory to simultaneously address gradient conflict and gradient dominance between the forgetting and retention objectives, achieving an optimal balance between unlearning and preservation across both classification and generation tasks.

## Background & Motivation

Machine Unlearning (MU) aims to selectively erase the influence of specific data or concepts from a trained model while preserving its performance on the remaining data. This requirement is driven by several factors:

**Privacy Regulations**: GDPR and CCPA grant users the "right to be forgotten," obligating organizations to erase the influence of user data upon request.

**Safety and Copyright Protection**: Text-to-image diffusion models (e.g., Stable Diffusion) may have been trained on NSFW content or copyrighted material, necessitating the removal of such undesirable concepts.

**Retraining from Scratch Is Infeasible**: The ideal unlearning approach would retrain the model from scratch on data excluding the forget set, but the computational cost of retraining large-scale models (e.g., diffusion models, CLIP) is prohibitive, motivating approximate unlearning algorithms.

**Core Problems in Existing Methods — Gradient Conflict and Gradient Dominance**:

Current MU methods typically formulate forgetting and retention as a weighted sum of two sub-objectives: a retention loss that fine-tunes the model on remaining data, and a forgetting loss that maximizes loss on the forget set. Through empirical analysis, the authors identify two critical failure modes:

- **Gradient Conflict**: The cosine similarity between the forgetting gradient and the retention gradient is frequently negative, indicating that the two objectives often pull in opposing directions.
- **Gradient Dominance**: The norms of the two gradients differ substantially, causing the joint update direction to be dominated by one objective while the other is effectively ignored.

Although both phenomena have been extensively studied in the multi-objective optimization (MOO) literature, they have been largely overlooked in the MU community.

## Method

### Core Idea: Formulating MU as a Cooperative Game

The central contribution of MUNBa is to recast the MU problem as a **two-player cooperative bargaining game**:

- **Forgetting Player**: proposes gradient $g_f$, preferring that the joint update direction favors forgetting.
- **Preservation Player**: proposes gradient $g_r$, preferring that the joint update direction favors retention.

The two players negotiate to find a mutually beneficial joint update direction that maximizes their collective payoff.

### Utility Function Definition

The utility of each player is defined as the inner product between its gradient and the final joint update direction. The preservation player's utility $u_r$ equals the inner product of $g_r$ with the joint direction, and the forgetting player's utility $u_f$ equals the inner product of $g_f$ with the joint direction. The utility function measures the alignment between the final update direction and each player's individual objective. If the joint update direction deviates from a player's gradient direction, that player's payoff decreases.

### Nash Bargaining Objective

Inspired by Nash bargaining theory, the optimization objective is reformulated as maximizing $\log(u_r) + \log(u_f)$, subject to the constraint that the joint update vector lies within a ball of radius $\epsilon$ centered at the origin. The logarithmic form enforces diminishing marginal utility — larger gains yield smaller incremental benefits — thereby naturally balancing the contributions of both objectives.

### Closed-Form Solution

This is the key technical contribution of the paper. Through a series of theorem derivations, the joint update direction is obtained as $\alpha_r g_r + \alpha_f g_f$, where the coefficients $\alpha$ admit a **closed-form solution**:

$$\alpha_r = \frac{1}{\|g_r\|} \sqrt{\frac{1 - \cos(\phi)}{\sin^2(\phi) + \xi}}, \quad \alpha_f = \frac{1}{\|g_f\|} \sqrt{\frac{1 - \cos(\phi)}{\sin^2(\phi) + \xi}}$$

where $\phi$ is the angle between $g_r$ and $g_f$, and $\xi$ is a small constant to prevent division by zero.

**Key Properties of the Closed-Form Solution**:

1. **Coefficients are inversely proportional to gradient norms** (proportional to $1/\|g\|$): this automatically suppresses the objective with the larger gradient norm, resolving gradient dominance.
2. **Conflict-adaptive**: when gradient conflict is severe ($\cos(\phi) \approx -1$), the coefficients increase to amplify each objective's contribution; when gradients are aligned ($\cos(\phi) \approx 1$), coefficients decrease.
3. **Exact and efficient**: unlike prior work that requires approximate solutions for $\alpha$, this paper exploits the fact that MU has exactly two objectives to derive an exact closed-form solution with negligible computational overhead.

### Handling Degenerate Cases

When the two gradients are linearly dependent ($g_r = \zeta g_f$), the Gram matrix becomes singular. This is handled as follows:
- If $\zeta < 0$ (opposite directions), noise is added to the gradient with the smaller norm to break linear dependence.
- If $\zeta \geq 0$ (same direction), $\alpha$ is set directly to $[0.5, 0.5]$.

### Algorithm Pipeline

1. Sample mini-batches from the forget set and the retain set.
2. Compute the forgetting gradient $g_f$ and the retention gradient $g_r$ separately.
3. Construct the Gram matrix $K = G^\top G$ and compute the coefficient $\alpha$ via the closed-form solution.
4. Update model parameters using the joint gradient $(\alpha_r g_r + \alpha_f g_f)$.

### Theoretical Guarantees

- **Pareto Improvement** (Theorem 2.9): Under Lipschitz smoothness conditions, an appropriate learning rate ensures the monotonic decrease of both players' losses.
- **Convergence** (Theorem 2.10): The joint loss converges to a Pareto stationary point, where any deviation from the final state degrades at least one objective.
- **Lower Bound Guarantee** (Lemma 2.8): Each player's coefficient $\alpha_i$ is bounded below by $\frac{1}{\sqrt{2}M}$, ensuring that neither objective is entirely ignored.

## Key Experimental Results

### 1. Classification Tasks (ResNet)

Forgetting 10% of identities on Celeb-HQ-307 and 10% of data on CIFAR-10:

| Method | Celeb-HQ Avg. Gap ↓ | CIFAR-10 Avg. Gap ↓ |
|--------|----------------------|----------------------|
| SalUn | 0.60 | 1.24 |
| SHs | 0.39 | 1.62 |
| **MUNBa** | **0.10** | **0.97** |

MUNBa achieves the smallest Avg. Gap on both datasets, most closely approximating retraining from scratch. On Celeb-HQ-307, it attains 0% accuracy on forget data and 87.24% on the test set.

### 2. CLIP Unlearning (Oxford Pets Category Forgetting)

| Method | Forget 1 Class $\text{Acc}_{D_f}$ ↓ | $\text{Acc}_{D_t}$ ↑ | ImageNet Acc ↑ |
|--------|---------------------------------------|----------------------|----------------|
| SHs | 0.00% | 91.41% | 37.97% |
| SalUn | 4.69% | 82.93% | 59.94% |
| **MUNBa** | 2.50% | **94.99%** | 59.36% |

MUNBa achieves effective unlearning while preserving CLIP's generalization capability (ImageNet accuracy 59.36% vs. original 60.09%), whereas SHs, despite achieving complete forgetting, suffers a catastrophic drop to 37.97% on ImageNet.

### 3. Diffusion Model Class Forgetting (Imagenette)

MUNBa achieves an average FID of 1.20 and UA of 99.94%, outperforming ESD (FID=1.49, UA=99.40%) and SalUn (FID=1.22, UA=99.82%) across 10 categories.

### 4. NSFW Concept Erasure (Stable Diffusion v1.4)

| Method | FID ↓ | CLIP Score ↑ | ASR ↓ |
|--------|-------|--------------|-------|
| ESD | 15.76 | 30.33 | 73.24% |
| SA | 25.58 | 31.03 | 48.59% |
| SalUn | 25.06 | 28.91 | 11.27% |
| **MUNBa** | **15.92** | 30.43 | **3.52%** |

MUNBa is particularly effective against UnlearnDiffAtk, achieving an attack success rate of only 3.52% (vs. 11.27% for SalUn) while maintaining high generation quality (FID=15.92 vs. SalUn's 25.06).

## Highlights & Insights

1. **Precise Problem Formulation**: This is the first work to systematically analyze and empirically demonstrate gradient conflict and gradient dominance in MU, using cosine similarity histograms and gradient norm ratio visualizations to make the problems immediately apparent.
2. **Elegance of the Closed-Form Solution**: By exploiting the structural property that MU has exactly two objectives (forgetting and retention), the paper derives an exact closed-form solution from Nash bargaining theory, avoiding the approximate solvers required by general multi-objective optimization methods. The form of the coefficients is intuitively clear: inversely proportional to gradient norms, automatically balancing the contributions of both objectives.
3. **General-Purpose Framework**: MUNBa is not tailored to any specific model or unlearning scenario; it can be seamlessly applied to ResNet classification, CLIP vision-language models, and Stable Diffusion generative models alike.
4. **Strong Adversarial Robustness**: The attack success rate under UnlearnDiffAtk is only 3.52%, far below competing methods, demonstrating that the Nash bargaining solution is not only effective in standard settings but also more robust under adversarial conditions. This is a meaningful finding — the theoretically optimal balance point is also harder to break under adversarial attack.
5. **Transferability After CLIP Unlearning**: Plugging the unlearned CLIP text encoder into Stable Diffusion preserves generation quality while preventing the generation of images from the forgotten category, demonstrating practical applicability.

## Limitations & Future Work

1. **Computational Overhead**: The authors acknowledge that MUNBa is slower than some baselines, as each step requires computing two separate gradients before solving for the coefficients. Although the closed-form solution itself incurs negligible cost, the doubled gradient computation increases overall runtime.
2. **Imperfect Forgetting**: In certain scenarios MUNBa may still fail (noted in Appendix 8.4), and all MU methods, including MUNBa, continue to exert some influence on the retained concepts or classes.
3. **No Discussion of Data-Free Settings**: The current method assumes access to a retain dataset $D_r$, but in practice training data may no longer be available after deployment.
4. **Potential for Misuse**: Unlearning techniques could be maliciously exploited to erase critical information, bias decision-making processes, or conceal important data.
5. **Extension to Multiple Objectives**: The current framework handles only two players (forgetting + retention). Simultaneously forgetting multiple concepts or incorporating additional constraints (e.g., fairness) would require extension to a multi-player bargaining formulation.

## Related Work & Insights

- **SalUn (Fan et al., ICLR 2024)**: Uses gradient saliency maps to identify parameters most relevant to the forget data for selective unlearning; a strong baseline on both classification and generation tasks.
- **Scissorhands (Wu & Harandi, ECCV 2024)**: Erases data influence via network connectivity sensitivity; a prior work from the same research group.
- **ESD (Gandikota et al., ICCV 2023)**: An energy-based method for concept erasure targeting the classifier-free guidance mechanism.
- **Nash-MTL (Navon et al., ICML 2022)**: Formulates multi-task learning as a bargaining game; the direct inspiration for MUNBa. MUNBa, however, exploits the two-objective structure of MU to derive a closed-form solution.
- **CAGrad (Liu et al., NeurIPS 2021)**: Conflict-averse gradient descent that addresses gradient conflict but does not handle gradient dominance.

**Insights**: The Nash bargaining framework offers a theoretically grounded and general solution to the balancing problem in multi-objective optimization, and can be extended to other settings with competing objectives, such as balancing forgetting and new knowledge acquisition in continual learning, or coordinating heterogeneous client objectives in federated learning.

## Rating
- Novelty: 4/5 — Introducing game theory to address gradient conflict/dominance in MU; the closed-form derivation is elegant
- Experimental Thoroughness: 5/5 — Covers three major model families: classification (ResNet), VLM (CLIP), and generative models (SD), including adversarial robustness evaluation
- Writing Quality: 4/5 — Problem motivation is clear, theoretical derivations are complete, and visualizations are intuitive
- Value: 4/5 — Strong generality, solid theory and experiments, with meaningful practical contributions to the MU field

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Invisible Watermarks, Visible Gains: Steering Machine Unlearning with Bi-Level Watermarking Design](invisible_watermarks_visible_gains_steering_machine_unlearning_with_bi-level_wat.md)
- [\[ICCV 2025\] Meta-Unlearning on Diffusion Models: Preventing Relearning Unlearned Concepts](meta-unlearning_on_diffusion_models_preventing_relearning_unlearned_concepts.md)
- [\[ICCV 2025\] Holistic Unlearning Benchmark: A Multi-Faceted Evaluation for Text-to-Image Diffusion Model Unlearning](holistic_unlearning_benchmark_a_multi-faceted_evaluation_for_text-to-image_diffu.md)
- [\[ICLR 2026\] Model Collapse Is Not a Bug but a Feature in Machine Unlearning for LLMs](../../ICLR2026/image_generation/model_collapse_is_not_a_bug_but_a_feature_in_machine_unlearning_for_llms.md)
- [\[NeurIPS 2025\] Large-Scale Training Data Attribution for Music Generative Models via Unlearning](../../NeurIPS2025/image_generation/large-scale_training_data_attribution_for_music_generative_models_via_unlearning.md)

</div>

<!-- RELATED:END -->
