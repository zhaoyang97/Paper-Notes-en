---
title: >-
  [Paper Note] Taught Well, Learned Ill: Towards Distillation-Conditional Backdoor Attack
description: >-
  [NeurIPS 2025][AI Safety][knowledge distillation] This paper proposes the Distillation-Conditional Backdoor Attack (DCBA) paradigm and its instantiation SCAR, which embeds a "dormant" backdoor into a teacher model via bi-level optimization. The backdoor remains undetectable on the teacher model but is activated and transferred to the student model during knowledge distillation, even when the distillation dataset is entirely clean.
tags:
  - NeurIPS 2025
  - AI Safety
  - knowledge distillation
  - backdoor attack
  - bi-level optimization
  - implicit differentiation
  - model security
date: 2026-05-08
content_hash: 0a18375d40272c08
---

# Taught Well, Learned Ill: Towards Distillation-Conditional Backdoor Attack

**Conference**: NeurIPS 2025
**arXiv**: [2509.23871](https://arxiv.org/abs/2509.23871)
**Code**: [GitHub](https://github.com/WhitolfChen/SCAR)
**Area**: AI Security / Backdoor Attack
**Keywords**: knowledge distillation, backdoor attack, bi-level optimization, implicit differentiation, model security

## TL;DR
This paper proposes the Distillation-Conditional Backdoor Attack (DCBA) paradigm and its instantiation SCAR, which embeds a "dormant" backdoor into a teacher model via bi-level optimization. The backdoor remains undetectable on the teacher model but is activated and transferred to the student model during knowledge distillation, even when the distillation dataset is entirely clean.

## Background & Motivation
Knowledge distillation (KD) is a core technique for transferring knowledge from large teacher models to lightweight student models, and is widely adopted for model deployment on resource-constrained devices. In practice, developers frequently obtain pre-trained models from third-party platforms (e.g., Hugging Face, GitHub) to serve as teacher models. Such platforms typically apply security checks, including backdoor detection, to uploaded models.

**Limitations of Prior Work**: The prevailing security assumption holds that "a teacher model passing backdoor detection plus a clean distillation dataset guarantees a safe student model." However, whether this assumption truly holds remains an open question.

**Key Challenge**: Existing distillation-resistant backdoor attacks (e.g., ADBA) aim to preserve the backdoor through distillation, but the backdoor remains active in the teacher model and is thus susceptible to detection. Simply fine-tuning to conceal the backdoor (ADBA-FT) causes the attack to fail, as it lacks dynamic guidance from the distillation process.

**Key Insight**: The authors introduce a fundamentally new paradigm—Distillation-Conditional Backdoor Attack (DCBA)—in which the backdoor lies dormant and undetectable in the teacher model, yet is automatically activated during distillation. The attack is formulated as a bi-level optimization problem, using a surrogate student model to simulate the distillation process and guide the teacher model's optimization.

## Method

### Overall Architecture
The core mechanism of SCAR is to jointly train the teacher model and a surrogate student model through bi-level optimization. The inner-level optimization simulates the distillation process to train the surrogate student, while the outer-level optimization leverages the surrogate student's output signals to optimize the teacher model such that three conditions are satisfied: (1) the teacher behaves normally on both clean and poisoned samples (backdoor dormant); (2) the surrogate student behaves normally on clean samples but triggers the backdoor on poisoned samples. In addition, SCAR pre-optimizes the trigger injection function to simplify the bi-level optimization.

### Key Designs

1. **Bi-Level Optimization Formulation**:

    - **Outer-level optimization** (teacher model parameters λ): comprises four loss terms — cross-entropy of the teacher on clean samples, cross-entropy of the teacher on poisoned samples (enforcing correct classification to conceal the backdoor), cross-entropy of the surrogate student on clean samples, and the attack loss of the surrogate student on poisoned samples (toward the target label).
    - **Inner-level optimization** (surrogate student parameters ω): standard distillation loss, including the student's cross-entropy loss and a KL divergence distillation loss between teacher and student.
    - The key insight of this design is that the outer-level optimization perceives the distillation dynamics through feedback from the surrogate student, rather than relying solely on the teacher model's own behavior.

2. **Implicit Differentiation Algorithm**:

    - Since the outer-level loss implicitly depends on λ through ω(λ), gradients cannot be computed directly via backpropagation.
    - The implicit function theorem is applied to derive the implicit gradient: differentiating through the inner-level optimality condition yields the Jacobian matrix expression.
    - A fixed-point iteration based on the Neumann series expansion is used to approximate the vector-inverse-Hessian product: $\mathbf{v}_{n+1} = \mathbf{J}_{\Phi,\omega}\mathbf{v}_n + \mathbf{g}_\omega$
    - The approximate gradient after K truncated steps is efficiently computed via vector-Jacobian products.

3. **Pre-Optimized Trigger Injection Function**:

    - Prior to the bi-level optimization, an additive trigger pattern μ is optimized: $G(\mathbf{x};\mu) = \Pi(\mathbf{x} + \mu)$
    - Using pre-trained clean teacher and student models, the trigger is optimized to drive both models to output the target label.
    - An L∞ norm constraint is imposed on the trigger to maintain visual imperceptibility.
    - This step provides a favorable initialization for the subsequent bi-level optimization, reducing its difficulty.

### Loss & Training
- The outer-level loss consists of four weighted terms, with coefficients α, β, and γ controlling the relative importance of each objective.
- At each outer-level epoch, the surrogate student parameters are re-initialized and T steps of gradient descent are performed to simulate distillation.
- The outer-level gradient estimate uses a subset of data (M batches) to improve computational efficiency.
- The number of fixed-point iteration steps K controls the approximation quality of the gradient.

## Key Experimental Results

### Main Results

| Dataset | Distillation Method | Student Model | SCAR Student ASR | ADBA-FT Student ASR | Teacher ASR |
|--------|---------|---------|-------------|---------------|---------|
| CIFAR-10 | Response | MobileNet-V2 | **99.94%** | 92.87% | 1.50% |
| CIFAR-10 | Response | ShuffleNet-V2 | **99.02%** | 81.02% | 1.50% |
| CIFAR-10 | Response | EfficientViT | **86.31%** | 30.58% | 1.50% |
| ImageNet | Response | MobileNet-V2 | **81.69%** | 45.39% | 2.12% |
| ImageNet | Relation | MobileNet-V2 | **91.96%** | 42.61% | 2.12% |

### Ablation Study

| Configuration | MobileNet-V2 ASR | ShuffleNet-V2 ASR | EfficientViT ASR | Note |
|------|-----------------|------------------|-----------------|------|
| SCAR (full) | 99.94% | 99.02% | 86.31% | Best performance |
| w/o surrogate model | 82.92% | 51.58% | 31.42% | Without distillation dynamics guidance, attack degrades significantly |
| w/o pre-optimized trigger | 1.03% | 1.06% | 2.09% | Attack completely fails; initialization of bi-level optimization is critical |

### Key Findings
- Both the surrogate model and the pre-optimized trigger are necessary conditions for SCAR's success; neither can be omitted.
- The absence of the pre-optimized trigger has a greater impact, indicating that initialization is paramount for the bi-level optimization.
- SCAR is effective across all three distillation paradigms (Response/Feature/Relation-based KD), demonstrating generalization to unknown distillation strategies.
- Performance degrades on ImageNet, likely because larger image resolutions make convergence of the bi-level optimization more difficult.

## Highlights & Insights
- **Novel Threat Paradigm**: This work is the first to propose the concept of a "distillation-conditional backdoor"—dormant in the original model, activated only after distillation—thereby breaking the assumption that "a safe teacher plus clean data equals a safe student."
- **Bi-Level Optimization Modeling**: The inner and outer levels elegantly simulate the distillation process and the attack objective, respectively, with implicit differentiation bridging the two levels.
- **Detection Evasion**: Both Neural Cleanse and SCALE-UP fail to detect the dormant backdoor in the teacher model, validating the attack's stealthiness.
- The pre-optimized trigger design is practically effective, substantially reducing the search space of the bi-level optimization.

## Limitations & Future Work
- Attack effectiveness degrades on large-scale datasets such as ImageNet, where convergence of bi-level optimization in high-dimensional spaces is more challenging.
- Attack success rates are relatively lower against student models with architectures substantially different from the teacher, such as EfficientViT.
- The assumption that the attacker has no knowledge of the distillation process is strong; in practice, more robust surrogate model selection strategies may be required.
- The evaluation is limited to classification tasks; applicability to generative models and large language models remains unexplored.
- The paper primarily focuses on the attack perspective and lacks discussion of effective defense strategies.

## Related Work & Insights
- **vs. ADBA**: ADBA directly injects distillation-resistant backdoors that remain active in the teacher model and are thus detectable; SCAR keeps the backdoor dormant, and ADBA-FT's attempt at concealment fails due to the absence of distillation dynamics guidance.
- **vs. Conventional Backdoor Attacks (BadNets, etc.)**: Conventional backdoors do not survive distillation; SCAR is specifically designed for the distillation scenario.
- **Insight**: This work serves as a warning for MLaaS platform security—inspecting the teacher model alone is insufficient; backdoor detection must also be applied to the distilled student model.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to propose the distillation-conditional backdoor attack paradigm; the concept is original and the threat is realistic.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers multiple datasets, architectures, distillation methods, ablation studies, and detection evasion, though ImageNet results are moderate.
- Writing Quality: ⭐⭐⭐⭐ Logically clear, with well-articulated problem motivation and complete technical derivations.
- Value: ⭐⭐⭐⭐⭐ Reveals a critical security blind spot in the knowledge distillation supply chain, carrying significant implications for the model security community.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Efficient Verified Machine Unlearning for Distillation](efficient_verified_machine_unlearning_for_distillation.md)
- [\[NeurIPS 2025\] On the Hardness of Conditional Independence Testing In Practice](on_the_hardness_of_conditional_independence_testing_in_practice.md)
- [\[ICCV 2025\] Mind the Cost of Scaffold! Benign Clients May Even Become Accomplices of Backdoor Attack](../../ICCV2025/ai_safety/mind_the_cost_of_scaffold_benign_clients_may_even_become_accomplices_of_backdoor.md)
- [\[NeurIPS 2025\] Deceptron: Learned Local Inverses for Fast and Stable Physics Inversion](deceptron_learned_local_inverses_for_fast_and_stable_physics_inversion.md)
- [\[NeurIPS 2025\] Stealthy Yet Effective: Distribution-Preserving Backdoor Attacks on Graph Classification](stealthy_yet_effective_distribution-preserving_backdoor_attacks_on_graph_classif.md)

<!-- RELATED:END -->
