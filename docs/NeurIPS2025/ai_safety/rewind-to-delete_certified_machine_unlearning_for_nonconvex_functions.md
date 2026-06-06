---
title: >-
  [Paper Note] Rewind-to-Delete: Certified Machine Unlearning for Nonconvex Functions
description: >-
  [NeurIPS 2025][AI Safety][machine unlearning] This paper proposes R2D (Rewind-to-Delete), the first first-order, black-box certified machine unlearning algorithm for general nonconvex loss functions. It achieves data del…
tags:
  - "NeurIPS 2025"
  - "AI Safety"
  - "machine unlearning"
  - "differential privacy"
  - "nonconvex optimization"
  - "certified unlearning"
  - "privacy protection"
date: 2026-05-08
content_hash: 611a055b63cdb815
---

# Rewind-to-Delete: Certified Machine Unlearning for Nonconvex Functions

**Conference**: NeurIPS 2025
**arXiv**: [2409.09778](https://arxiv.org/abs/2409.09778)  
**Code**: None  
**Area**: AI Safety / Machine Unlearning
**Keywords**: machine unlearning, differential privacy, nonconvex optimization, certified unlearning, privacy protection

## TL;DR
This paper proposes R2D (Rewind-to-Delete), the first first-order, black-box certified machine unlearning algorithm for general nonconvex loss functions. It achieves data deletion by rewinding to an earlier checkpoint in the training trajectory and then performing gradient descent on the retained data, while providing $(ε, δ)$-certified unlearning guarantees and theoretical trade-offs among privacy, utility, and efficiency.

## Background & Motivation
Machine unlearning aims to efficiently remove the influence of specific data from a trained model without retraining from scratch. This is particularly important in the context of privacy regulations such as GDPR's "right to be forgotten," as well as for the practical removal of erroneous, outdated, or harmful data.

**Limitations of Prior Work**:
- Most certified unlearning algorithms (e.g., Newton Step, Descent-to-Delete) are only applicable to convex or strongly convex functions and cannot handle the nonconvex losses arising in deep learning.
- Existing nonconvex unlearning methods have notable drawbacks: second-order methods (CNS) incur high computational costs and require $O(d^2)$ storage; quasi-second-order methods (HF) require per-sample statistics to be recorded at each training step, making them white-box methods; first-order methods (Langevin Unlearning) require noise injection at every step and are also white-box.

**Key Challenge**: How can one simultaneously achieve first-order efficiency, black-box compatibility (no modification to the training process), and certified unlearning guarantees in the nonconvex setting?

**Key Insight**: Leverage rewinding to an earlier checkpoint along the training trajectory, followed by gradient descent on the retained data. Checkpoints can be recovered post hoc via the proximal point algorithm, enabling a fully black-box approach.

## Method

### Overall Architecture
R2D consists of two phases: learning and unlearning. In the learning phase, standard gradient descent is run for $T$ steps on the full dataset; the checkpoint at step $T-K$ is saved, and the final model is released with added Gaussian noise for inference. In the unlearning phase, the $T-K$ checkpoint is loaded, and $K$ steps of gradient descent are performed on the retained dataset, again with Gaussian noise added to the output. No special operations are required during training.

### Key Designs

1. **Rewinding Mechanism**:

    - Core insight: The strategy in Descent-to-Delete of directly "descending" from the terminal parameter $\theta_T$ fails in the nonconvex setting, because nonconvex functions lack a unique global minimum to attract trajectory convergence.
    - R2D rewinds to an earlier checkpoint $\theta_{T-K}$, bringing the unlearning trajectory closer to that of retraining from scratch, thereby reducing trajectory divergence caused by "unlearning bias."
    - Larger $K$ → more rewinding → smaller required noise $\sigma$ → better model utility, but at higher computational cost.
    - When $K = T$, the method degenerates to noiseless full retraining.

2. **Post Hoc Checkpoint Recovery via Proximal Point Algorithm**:

    - For models whose intermediate checkpoints were not saved during training, the proximal point method can be used to recover $\theta_{T-K}$ by inverting from $\theta_T$.
    - Key lemma: When the step size $\eta < 1/L$, one step of gradient descent is equivalent to applying the proximal operator of $-f$; due to strong convexity, the proximal subproblem can be solved via standard gradient descent or Newton's method.
    - Experiments show that checkpoints with $K < T/2$ can be recovered faithfully, but earlier checkpoints become unstable due to accumulated approximation error.

3. **Gaussian Noise Mechanism**:

    - Based on the differential privacy Gaussian mechanism, independent Gaussian noise is added once to the final output of both the learning and unlearning phases.
    - The noise standard deviation $\sigma$ is jointly determined by the unlearning guarantee parameters $(ε, δ)$, dataset size $n$, deletion size $m$, and the number of rewind steps $K$.
    - $\sigma$ is inversely proportional to $n$—larger datasets require less noise.
    - $\sigma$ decreases monotonically with $K$ via the function $h(K)$—more rewinding permits smaller noise.

### Loss & Training
- The learning phase minimizes the empirical risk $f_\mathcal{D}(\theta)$ over the full dataset using standard gradient descent.
- The unlearning phase minimizes $f_{\mathcal{D}'}(\theta)$ over the retained dataset.
- For nonconvex functions satisfying the Polyak-Łojasiewicz (PL) inequality, linear convergence rates and generalization guarantees can be obtained.
- The PL condition is closely related to over-parameterized neural networks satisfying it locally near initialization.
- In practice, the recommended procedure is: first select $\sigma$ to preserve model utility (e.g., 0.01), then select $K$ within the computational budget (e.g., $10\% \times T$), and finally compute the achieved privacy level.

## Key Experimental Results

### Main Results

| Dataset | Algorithm | D_retain AUC | D_unlearn AUC (Δ) | D_test AUC | MIA AUC↓ |
|--------|------|-------------|-------------------|-----------|---------|
| eICU | R2D 74% | 0.7333 | 0.7439 (-0.0012) | 0.7326 | **0.481** |
| eICU | Hessian-Free | 0.7476 | 0.7587 (-0.0027) | 0.7465 | 0.524 |
| eICU | CNS | 0.7622 | 0.7848 (-0.0012) | 0.7601 | 0.543 |
| Lacuna-100 | R2D 74% | 0.9712 | 0.8998 (-0.0995) | 0.9534 | **Best** |
| Lacuna-100 | Hessian-Free | 0.9757 | 0.9555 (-0.0191) | 0.9652 | Inferior |
| Lacuna-100 | Finetune | 0.9997 | 0.9975 (-0.0018) | 0.9854 | Inferior |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| R2D 7–10% | Moderate unlearn AUC drop | Even modest rewinding yields measurable effect |
| R2D 36–37% | Continued unlearn AUC decline | Rewind ratio positively correlates with unlearning effectiveness |
| R2D 74% | Largest unlearn AUC drop | More rewinding = stronger unlearning, with slight utility degradation on retained data |
| $\sigma=0$ vs $\sigma=0.01$ | Small noise significantly improves MIA defense | Certified unlearning methods inherently defend against MIA better than uncertified ones |
| Training speed | R2D up to 88× faster than HF | Major advantage of the black-box approach |

### Key Findings
- R2D exhibits significantly larger performance degradation on $D_\text{unlearn}$, indicating more thorough forgetting.
- R2D achieves the best performance under both classical MIA and MIA-U membership inference attacks.
- Even with minimal noise perturbation, certified unlearning methods defend against MIA more effectively than uncertified methods (Finetune, SCRUB, Fisher Forgetting).
- As a black-box method, R2D is up to 88× faster than HF in the training phase.

## Highlights & Insights
- **Black-box + First-order + Nonconvex**: The first approach to simultaneously satisfy all three practical requirements in the nonconvex setting, directly applicable to any pretrained model trained with standard gradient descent.
- **Rewinding outperforms descending**: The core insight that "rewinding > descending" provides a more powerful nonconvex unlearning baseline than Descent-to-Delete.
- **Post hoc checkpoint recovery via proximal point**: Even without saved checkpoints at training time, they can be recovered post hoc via the proximal point algorithm, enabling full black-box operation.
- **Theory-practice unification**: An explicit three-way trade-off formula among privacy, utility, and efficiency (the relationship among $\sigma$, $K$, and $\varepsilon$) allows practitioners to configure the method flexibly according to specific requirements.
- Novel experimental framework: Data deletion is performed at the user level, more closely reflecting real-world unlearning scenarios (as opposed to random deletion or class-level removal).

## Limitations & Future Work
- Theoretical guarantees rely on uniformly bounded gradients and Lipschitz smoothness assumptions, which may be overly strong for all practical models.
- The proximal point algorithm for post hoc checkpoint recovery becomes unstable when $K > T/2$, limiting the rewind depth in black-box mode.
- The analysis assumes full-batch gradient descent; theoretical guarantees become looser when mini-batch SGD is used in practice.
- Generalization guarantees for PL functions require strong assumptions, and whether practical deep networks satisfy the PL condition locally remains debated.
- Validation on large-scale models such as large language models has not been conducted.

## Related Work & Insights
- **vs. Descent-to-Delete (D2D)**: D2D fine-tunes directly from $\theta_T$ and is only applicable to strongly convex functions; R2D circumvents the fundamental difficulty that nonconvex functions lack a unique global minimum by rewinding.
- **vs. CNS**: Second-order methods require $O(d^2)$ storage and do not scale; R2D requires only $O(d)$ storage.
- **vs. Hessian-Free**: This white-box method requires per-sample statistics to be recorded at every training step, incurring substantial overhead; R2D requires no modification to the training process whatsoever.
- **Insights**: The rewind-then-fine-tune paradigm is highly general and may inspire other scenarios requiring "undoing" model behavior (e.g., bias removal, copyright data deletion).

## Rating
- Novelty: ⭐⭐⭐⭐ The "rewind to checkpoint" idea is intuitively clean, and the proximal point recovery is a notable contribution, though there is some overlap with existing rewinding ideas.
- Experimental Thoroughness: ⭐⭐⭐⭐ The novel user-level unlearning experimental framework, comprehensive baseline comparisons, and MIA evaluation are thorough, though dataset scales are relatively small.
- Writing Quality: ⭐⭐⭐⭐⭐ Theoretical derivations are clear, problem formulation and motivation are well-articulated, and comparison with related work is objective and balanced.
- Value: ⭐⭐⭐⭐ Fills an important gap in certified unlearning for the nonconvex setting, though applicability to large-scale practical models remains to be validated.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Efficient Verified Machine Unlearning for Distillation](efficient_verified_machine_unlearning_for_distillation.md)
- [\[NeurIPS 2025\] Position: Bridge the Gaps between Machine Unlearning and AI Regulation](position_bridge_the_gaps_between_machine_unlearning_and_ai_regulation.md)
- [\[NeurIPS 2025\] The Unseen Threat: Residual Knowledge in Machine Unlearning under Perturbed Samples](the_unseen_threat_residual_knowledge_in_machine_unlearning_under_perturbed_sampl.md)
- [\[NeurIPS 2025\] Machine Unlearning Doesn't Do What You Think: Lessons for Generative AI Policy and Research](machine_unlearning_doesnt_do_what_you_think_lessons_for_generative_ai_policy_and.md)
- [\[ICML 2026\] Two Blind Spots of Machine Unlearning: Over-unlearning and Prototype Relearning Attacks](../../ICML2026/ai_safety/unlearnings_blind_spots_over-unlearning_and_prototypical_relearning_attack.md)

</div>

<!-- RELATED:END -->
