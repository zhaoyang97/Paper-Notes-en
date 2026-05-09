---
title: >-
  [Paper Note] Mitigating the Safety Alignment Tax with Null-Space Constrained Policy Optimization
description: >-
  [ICLR 2026][LLM Alignment][Safety Alignment] This paper proposes NSPO, which projects safety alignment policy gradients onto the null space of general-task representations, geometrically ensuring that safety optimization does not degrade general capabilities. Using only 40% of the safety training data, NSPO achieves state-of-the-art results across 7 safety benchmarks while incurring virtually no performance loss on mathematics, code generation, and instruction following.
tags:
  - ICLR 2026
  - LLM Alignment
  - Safety Alignment
  - Null Space
  - Policy Optimization
  - Alignment Tax
  - Gradient Projection
date: 2026-05-08
content_hash: 8990dcfe2d02e1ea
---

# Mitigating the Safety Alignment Tax with Null-Space Constrained Policy Optimization

**Conference**: ICLR 2026
**arXiv**: [2512.11391](https://arxiv.org/abs/2512.11391)
**Code**: [https://github.com/ivanniu/NSPO](https://github.com/ivanniu/NSPO)
**Area**: Alignment / RLHF
**Keywords**: Safety Alignment, Null Space, Policy Optimization, Alignment Tax, Gradient Projection

## TL;DR
This paper proposes NSPO, which projects safety alignment policy gradients onto the null space of general-task representations, geometrically ensuring that safety optimization does not degrade general capabilities. Using only 40% of the safety training data, NSPO achieves state-of-the-art results across 7 safety benchmarks while incurring virtually no performance loss on mathematics, code generation, and instruction following.

## Background & Motivation

**State of the Field**: LLM safety alignment (refusing harmful requests, adhering to ethical norms) is typically achieved by training on safety data via RL methods such as PPO, GRPO, or DPO.

**Limitations of Prior Work**: Safety alignment induces an **alignment tax** — the model becomes overly conservative, leading to performance degradation on general tasks such as mathematical reasoning and code generation. Existing approaches (SafeRLHF, W-DOOR, BFPO) model safety and general capability as a bi-objective optimization problem, mitigating the trade-off through balancing weights or mixing in large amounts of general data, but **none of them explicitly resolve gradient conflicts between the two objectives during training**.

**Root Cause**: The gradients of safety objectives and general-capability objectives conflict in direction — updating parameters along the safety gradient disrupts the general-task representations already learned by the model.

**Paper Goals**: How can safety alignment be performed without fundamentally harming general capabilities?

**Starting Point**: If a parameter update $\Delta$ lies in the null space of the general-task input representations $K$ (i.e., $\Delta K = 0$), the model's outputs on general inputs remain unchanged after the update.

**Core Idea**: Project safety policy gradients onto the null space of the general-task representation matrix, geometrically guaranteeing that safety updates are orthogonal to the general capability subspace.

## Method

### Overall Architecture
NSPO is built upon the GRPO framework, with the key modification being the introduction of null-space projection during gradient updates. The pipeline is as follows: (1) collect intermediate representations $K$ from the model on general-task data; (2) perform SVD on $KK^T$ to obtain the null-space projection matrix $\hat{U}\hat{U}^T$; (3) compute the GRPO gradient on safety data, project it onto the null space, and then update the parameters.

### Key Designs

1. **Null-Space Projection Matrix Construction**:

    - **Function**: Extract input representations $K$ from each linear transformation layer of the model using sampled general-task data (commonsense, mathematics, and code), and construct the projection matrix.
    - **Mechanism**: $\{U, \Lambda, U^T\} = \text{SVD}(KK^T)$; retain the eigenvectors $\hat{U}$ corresponding to near-zero eigenvalues ($< 5 \times 10^{-4}$); the projection matrix is $\hat{U}\hat{U}^T$.
    - **Design Motivation**: Directly computing the null space of $K \in \mathbb{R}^{d \times N}$ is computationally prohibitive when $N \gg d$; using the null space of $KK^T \in \mathbb{R}^{d \times d}$ is equivalent and far more efficient.

2. **Gradient Projection**:

    - **Function**: Project the safety GRPO gradient $\nabla_W \mathcal{J}$ onto the null space to obtain $\nabla_W \mathcal{J}_{\text{NSPO}} = (\nabla_W \mathcal{J}) \cdot \hat{U}\hat{U}^T$.
    - **Mechanism**: After projection, $\nabla_W \mathcal{J}_{\text{NSPO}} \cdot K = 0$, meaning the parameter update does not alter the model's outputs on general inputs: $(W - \eta \nabla_W \mathcal{J}_{\text{NSPO}})K = WK = V$.
    - **Design Motivation**: This imposes a hard geometric constraint preventing safety updates from intruding into the general capability subspace, which is more reliable than soft constraints such as KL regularization.

3. **Removal of KL Divergence Regularization**:

    - **Function**: Remove the $D_{\text{KL}}[\pi_\theta \| \pi_{\text{ref}}]$ term from GRPO.
    - **Design Motivation**: KL regularization pulls the policy toward the reference model, which may itself be unsafe, thereby conflicting with the safety objective. Null-space projection serves as a superior regularizer — it prevents over-optimization while ensuring descent on the safety objective.

4. **Theoretical Guarantees**:

    - **Gradient Stability**: Projection is a non-expansive mapping, so $\|\nabla_W \mathcal{J}_{\text{NSPO}}\|_2 \leq \|\nabla_W \mathcal{J}\|_2$.
    - **Valid Descent Direction**: The projected gradient remains a valid descent direction for the safety objective: $\exists \eta > 0: \mathcal{J}(W - \eta \nabla_W \mathcal{J}_{\text{NSPO}}) \leq \mathcal{J}(W)$.

### Loss & Training
The training objective is the GRPO objective with null-space projection applied and the KL term removed. Only 40% of the PKU-SafeRLHF dataset (~11K samples) is used for training, with no need to mix in general-task data. The projection matrix is constructed from only 1,000 general-task samples, computed via a one-time SVD and then cached.

## Key Experimental Results

### Main Results

**Safety Performance (Llama3-8B-Instruct, ASR% ↓ lower is better)**:

| Method | AdvBench | HarmBench | SORRY-Bench | ALERT |
|--------|----------|-----------|-------------|-------|
| Base | 1.36 | 7.50 | 34.15 | 3.81 |
| SafeRLHF | 14.39 | 41.34 | 44.77 | 13.66 |
| W-DOOR | 0.75 | 2.81 | 30.46 | 1.86 |
| BFPO | 0.64 | 4.16 | 29.73 | 7.22 |
| **NSPO** | **0.06** | **0.18** | **16.81** | **1.00** |

**General Capability Preservation (Qwen2.5-7B-Instruct)**:

| Method | MATH | HumanEval | IFEval | MMLU |
|--------|------|-----------|--------|------|
| Base | High baseline | High baseline | High baseline | High baseline |
| NSPO | ~No loss | ~No loss | ~No loss | ~No loss |
| Other methods | Notable drop | Notable drop | Partial drop | Partial drop |

### Ablation Study

| Configuration | Safety | General Capability | Notes |
|---------------|--------|--------------------|-------|
| NSPO (full) | Best | Preserved | Null-space projection + KL removed |
| No projection (standard GRPO) | Improved | Degraded | Safety gradients harm general capability |
| Random projection | Poor | Poor | Random directions cannot satisfy both objectives |
| KL divergence retained | Slightly worse | Better preserved | KL pulls policy toward unsafe reference model |

### Key Findings
- NSPO using only 40% of the safety data (~11K samples) outperforms baselines trained on the full dataset.
- Consistent effectiveness on both Llama3-8B and Qwen2.5-7B demonstrates that the method is not architecture-specific.
- The additional computational cost of null-space projection is negligible: SVD is computed once at $O(d^3)$, and the per-step projection cost $O(d^3)$ is far smaller than the forward/backward pass cost $O(n^2 d + nd^2)$.
- Only 1,000 general-task samples are needed to construct the projection matrix, reflecting high data efficiency.

## Highlights & Insights
- **Geometric perspective on the alignment tax**: Rather than mitigating the conflict through soft constraints (KL regularization, data mixing), NSPO applies a hard constraint (null-space projection) that fundamentally eliminates the interference of safety gradients with general capability. This paradigm is transferable to any scenario requiring "learning new capabilities without forgetting old ones" (e.g., continual learning, multi-task fine-tuning).
- **Theoretical proof of descent direction**: The proof that the projected gradient remains a valid descent direction is critical — it rules out the concern that the projection is too restrictive to enable learning.
- **Removing KL regularization improves performance**: While KL regularization is standard practice for preventing policy drift, in the safety alignment setting the reference model may itself be unsafe, making KL regularization counterproductive. Replacing it with null-space projection is a more principled choice.

## Limitations & Future Work
- The dimensionality of the null space depends on the rank of the general-task representations — if these representations span most of the parameter space, the null space may be too small to accommodate sufficient safety learning.
- The method has only been validated in the safety alignment setting; generalization to helpfulness alignment, multi-task learning, and other scenarios requires further investigation.
- The eigenvalue threshold of $5 \times 10^{-4}$ is set manually, lacking an adaptive selection mechanism.
- Whether 1,000 sampled general-task examples are sufficient to represent the full distribution of general capabilities remains an open question.

## Related Work & Insights
- **vs. SafeRLHF**: SafeRLHF employs a constrained MDP framework but still relies on soft constraints; NSPO enforces hard constraints, yielding substantially better safety and general capability preservation.
- **vs. W-DOOR / BFPO**: These methods balance safety and general capability through preference ranking and bi-objective optimization, but require more data and still incur an alignment tax.
- **vs. continual learning methods (e.g., LoRA, EWC)**: NSPO's null-space approach is conceptually related to Elastic Weight Consolidation (EWC) — both seek to protect important parameter directions. However, NSPO is more direct: rather than penalizing deviation, it applies a geometric projection.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — Applying null-space projection to LLM safety alignment is a genuinely novel perspective, supported by rigorous theoretical analysis.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Evaluated on 7 safety and 7 general benchmarks across two model families, with extensive ablations.
- Writing Quality: ⭐⭐⭐⭐ — Theoretical derivations are clear and experiments are comprehensive.
- Value: ⭐⭐⭐⭐⭐ — Addresses a core pain point in safety alignment research with a simple, efficient, and deployable method.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Superficial Safety Alignment Hypothesis](superficial_safety_alignment_hypothesis.md)
- [\[ICLR 2026\] AlphaSteer: Learning Refusal Steering with Principled Null-Space Constraint](alphasteer_learning_refusal_steering_with_principled_null-space_constraint.md)
- [\[ICLR 2026\] Mitigating Mismatch within Reference-based Preference Optimization](mitigating_mismatch_within_reference-based_preference_optimization.md)
- [\[ICLR 2026\] Is On-Policy Data always the Best Choice for Direct Preference Optimization-based LM Alignment?](is_on-policy_data_always_the_best_choice_for_direct_preference_optimization-base.md)
- [\[ICLR 2026\] A2D: Any-Order, Any-Step Safety Alignment for Diffusion Language Models](a2d_any-order_any-step_safety_alignment_for_diffusion_language_models.md)

<!-- RELATED:END -->
