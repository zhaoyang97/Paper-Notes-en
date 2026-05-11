---
title: >-
  [Paper Note] Compose Your Policies! Improving Diffusion-based or Flow-based Robot Policies via Test-time Distribution-level Composition
description: >-
  [ICLR 2026][Image Generation][policy composition] This paper proposes General Policy Composition (GPC), which at test time convexly combines the distribution scores of multiple pretrained diffusion/flow policies without…
tags:
  - "ICLR 2026"
  - "Image Generation"
  - "policy composition"
  - "diffusion policy"
  - "distribution-level composition"
  - "test-time search"
  - "robot manipulation"
date: 2026-05-08
content_hash: db9461a05ab20321
---

# Compose Your Policies! Improving Diffusion-based or Flow-based Robot Policies via Test-time Distribution-level Composition

**Conference**: ICLR 2026
**arXiv**: [2510.01068](https://arxiv.org/abs/2510.01068)
**Code**: [https://sagecao1125.github.io/GPC-Site/](https://sagecao1125.github.io/GPC-Site/)
**Area**: Image Generation
**Keywords**: policy composition, diffusion policy, distribution-level composition, test-time search, robot manipulation

## TL;DR
This paper proposes General Policy Composition (GPC), which at test time convexly combines the distribution scores of multiple pretrained diffusion/flow policies without additional training, yielding a composite policy that surpasses any individual parent policy. Theoretical analysis proves that convex combination improves single-step score error, and this improvement propagates to the full sampling trajectory via a Grönwall bound.

## Background & Motivation

**Background**: Diffusion Policy has emerged as a powerful policy parameterization in robot learning, capable of representing complex multimodal action distributions. However, its progress is constrained by the high cost of acquiring large-scale interaction datasets.

**Limitations of Prior Work**: (a) Scaling model capacity requires more data; (b) supervised fine-tuning demands expensive data collection; (c) reinforcement learning requires reward engineering and extensive online interaction; (d) existing policy composition methods (e.g., PoCo) use fixed weights without exploring task-dependent weight search.

**Key Challenge**: The performance of a single policy is bounded by its training data and model capacity, yet combining multiple policies requires theoretical guarantees—naïve averaging does not necessarily yield improvement.

**Goal**: Obtain stronger policies by composing existing ones without any additional training.

**Key Insight**: Drawing an analogy to compositional generative modeling—in diffusion models, the convex combination of multiple score functions is equivalent to a product of probability density functions, biasing sampling toward regions of consensus.

**Core Idea**: Convex combination of score functions from multiple diffusion policies + test-time weight search = training-free policy enhancement.

## Method

### Overall Architecture
Given two pretrained diffusion/flow policies $\pi_1, \pi_2$, GPC convexly combines their score estimates at each denoising step during inference: $\hat{s}_{\text{comp}} = w_1 s_1 + w_2 s_2$, then searches for the optimal weight over $w_1 \in \{0.0, 0.1, \dots, 1.0\}$. The framework supports heterogeneous policy composition (VA+VLA, different visual modalities, diffusion+flow-matching).

### Key Designs

1. **Theoretical Guarantees for Convex Score Combination**:

    - Function: Proves that convex combination improves upon any single model at both the functional and system levels.
    - Mechanism: Proposition 4.1 proves that for two score estimators with different biases/noise, their convex combination yields an MSE $Q(w)$ that is a convex quadratic in $w$—the minimizer strictly outperforms either endpoint (unless the two estimators have identical errors). Proposition 4.2 shows via a Grönwall bound that this single-step improvement propagates to the entire sampling trajectory.
    - Design Motivation: To provide mathematical guarantees for "composition outperforms any individual policy," rather than relying solely on empirical observation.

2. **General Policy Composition Framework (GPC)**:

    - Function: Applies score combination to arbitrary diffusion/flow policies.
    - Mechanism: Generalizes classifier-free guidance (CFG) to a multi-policy composition: $\hat{\epsilon}(\tau_t, t, \mathbf{c}) = \epsilon_\theta(\tau_t, t) + \sum_i w_i(\epsilon_\theta(\tau_t, t, \mathbf{c}_i) - \epsilon_\theta(\tau_t, t))$. Heterogeneous models (e.g., with different denoising steps or noise schedules) are unified in score space before combination.
    - Design Motivation: To maximize flexibility—parent policies need not share architecture, input modality, or training data.

3. **Test-Time Weight Search**:

    - Function: Identifies the optimal composition weights for each task.
    - Mechanism: Searches over $w \in [0, 1]$ with a step size of 0.1, selecting the weight that achieves the best performance on validation data.
    - Design Motivation: Optimal weights are task-dependent—even the same pair of parent policies requires different combination ratios across tasks.

4. **Alternative Composition Operators (AND/OR)**:

    - AND composition (product of distributions): $\nabla \log p(\tau) = \nabla \log p_1(\tau) + \nabla \log p_2(\tau)$, equivalent to sampling only from regions endorsed by both policies.
    - OR composition (mixture of distributions): $p(\tau) \propto w_1 p_1(\tau) + w_2 p_2(\tau)$, preserving the multimodality of both policies.

### Loss & Training

**No training required.** GPC operates entirely at inference time without modifying any pretrained model parameters. The only search overhead consists of evaluating 11 candidate weight values.

## Key Experimental Results

### Main Results

Evaluated on Robomimic (6 tasks), PushT, and RoboTwin benchmarks:

| Setting | Method | Avg. Success Rate |
|------|------|----------|
| DP alone | Single policy | baseline |
| DP3 alone | Single policy | baseline |
| **GPC (DP + DP3)** | Convex combination | **Surpasses both** |

- GPC outperforms the better of the two parent policies on most tasks.
- Consistent performance gains are also validated in real-robot experiments.

### Ablation Study

| Analysis Dimension | Key Findings |
|---------|---------|
| Convex combination vs. AND vs. OR | Convex combination is generally optimal; AND performs better on precision-demanding tasks; OR performs better on multimodal tasks. |
| Optimal weight analysis | The optimal $w$ varies substantially across tasks (0.2–0.8), confirming task dependence. |
| Heterogeneous composition (VA+VLA) | Policies with entirely different architectures and even different visual inputs can be successfully composed. |
| Search granularity | A step size of 0.1 is sufficient; finer granularity yields diminishing returns. |

### Key Findings
- A composed policy can genuinely surpass any individual parent policy—the most surprising and central finding.
- Optimal weights are highly task-dependent: no fixed weight generalizes across tasks.
- Convex score combination steers sampling toward high-density regions of consensus between the two policies, naturally reducing boundary errors of individual models.
- Even when one parent policy completely fails on a given task, the composed policy may still succeed.

## Highlights & Insights
- **Theoretical guarantee of 1+1>2**: The proof that convex combination reduces score error (Proposition 4.1) is elegant and compelling—the key insight is that different models typically exhibit biases in different directions, which can cancel upon mixing.
- **Zero training cost**: Operating entirely at inference time, GPC can plug-and-play enhance any existing policy, which is of high practical value in real-world deployment.
- **Flexibility of heterogeneous composition**: VA and VLA models, RGB and point-cloud inputs, diffusion and flow-matching policies can all be composed—any policy from which a score function can be extracted is a valid candidate.
- **Unified view with CFG**: Interpreting GPC as a multi-policy generalization of classifier-free guidance provides a clear probabilistic interpretation.

## Limitations & Future Work
- Test-time weight search requires per-task tuning, making search overhead non-trivial when the number of tasks is large.
- The theoretical guarantees assume the two score estimators have different biases; if both models are trained on the same data, their biases may be highly correlated.
- Only two-policy composition is validated; the weight space grows exponentially for $N>2$.
- Safety guarantees for the composed policy are not discussed—consensus regions are not necessarily safe.
- Experiments focus primarily on success rate, lacking analysis of action quality (e.g., smoothness, efficiency) of the composed policy.

## Related Work & Insights
- **vs. PoCo (Wang et al., 2024c)**: PoCo performs constraint-, task-, and modality-level composition but uses fixed weights. GPC introduces test-time search for task-optimal weights and provides more rigorous theoretical analysis of the composition mechanism.
- **vs. Model Ensembling**: Traditional ensembling averages predictions; GPC composes at the score/distribution level—the former averages behaviors, the latter focuses on consensus regions.
- **Transfer Potential**: The theoretical framework and methodology for score composition can be directly transferred to model composition in image/video generation (e.g., compositional generation from multiple style-conditioned diffusion models).

## Rating
- Novelty: ⭐⭐⭐⭐ The theoretical guarantees are elegant, though the idea of score composition is not entirely new in the generative modeling literature.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers simulation and real-robot settings, multiple benchmarks and policy types, with comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ The logical chain from theoretical motivation to method to experiments is very clear.
- Value: ⭐⭐⭐⭐ A highly practical "free lunch" method directly applicable to existing robotic systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Improving Discrete Diffusion Unmasking Policies Beyond Explicit Reference Policies (UPO)](improving_discrete_diffusion_unmasking_policies_beyond_explicit_reference_polici.md)
- [\[ICLR 2026\] Offline Reinforcement Learning with Generative Trajectory Policies](offline_reinforcement_learning_with_generative_trajectory_policies.md)
- [\[ICLR 2026\] Steer Away From Mode Collisions: Improving Composition In Diffusion Models](steer_away_from_mode_collisions_improving_composition_in_diffusion_models.md)
- [\[ICLR 2026\] Test-Time Iterative Error Correction for Efficient Diffusion Models](test-time_iterative_error_correction_for_efficient_diffusion_models.md)
- [\[NeurIPS 2025\] Failure Prediction at Runtime for Generative Robot Policies](../../NeurIPS2025/image_generation/failure_prediction_at_runtime_for_generative_robot_policies.md)

</div>

<!-- RELATED:END -->
