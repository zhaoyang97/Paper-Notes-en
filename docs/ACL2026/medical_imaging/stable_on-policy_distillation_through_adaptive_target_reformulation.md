---
title: >-
  [Paper Note] Stable On-Policy Distillation through Adaptive Target Reformulation
description: >-
  [ACL 2026][Medical Imaging][Knowledge Distillation] This paper proposes Veto, a target-level reformulation method that stabilizes on-policy knowledge distillation by constructing a geometric bridging distribution between teacher and student in logit space. A single parameter $\beta$ simultaneously serves as an adaptive gradient veto in forward KL (suppressing harmful gradients from low-confidence tokens) and a decisiveness knob in reverse KL (balancing reward-driven optimization and output diversity), achieving a 9.2% improvement over SFT on GSM8K.
tags:
  - ACL 2026
  - Medical Imaging
  - Knowledge Distillation
  - On-Policy Distillation
  - Gradient Stability
  - KL Divergence
  - Target Reformulation
date: 2026-05-08
content_hash: 7fef08fc9deb90fe
---

# Stable On-Policy Distillation through Adaptive Target Reformulation

**Conference**: ACL 2026
**arXiv**: [2601.07155](https://arxiv.org/abs/2601.07155)
**Code**: N/A
**Area**: Medical Imaging
**Keywords**: Knowledge Distillation, On-Policy Distillation, Gradient Stability, KL Divergence, Target Reformulation

## TL;DR

This paper proposes Veto, a target-level reformulation method that stabilizes on-policy knowledge distillation by constructing a geometric bridging distribution between teacher and student in logit space. A single parameter $\beta$ simultaneously serves as an adaptive gradient veto in forward KL (suppressing harmful gradients from low-confidence tokens) and a decisiveness knob in reverse KL (balancing reward-driven optimization and output diversity), achieving a 9.2% improvement over SFT on GSM8K.

## Background & Motivation

**Background**: Knowledge distillation (KD) is a widely adopted technique for transferring capabilities from large language models to smaller student models. Conventional supervised KD trains on fixed teacher-generated trajectories, but suffers from exposure bias—training uses teacher data while inference relies on self-generated data, leading to performance degradation on autoregressive tasks. On-policy KD mitigates this by training on student-generated outputs.

**Limitations of Prior Work**: On-policy KD suffers from severe training instability due to the large distributional gap between a novice student and an expert teacher: (1) The forward KL objective produces gradient explosion when the student assigns near-zero probability to tokens preferred by the teacher ($P_T(y)/P_S(y) \to \infty$); (2) The reverse KL objective, while numerically stable, lacks explicit control over the intensity of mode-seeking behavior, making it prone to mode collapse and diversity loss.

**Key Challenge**: Existing methods primarily bridge the gap at the data level by mixing teacher and student tokens, while neglecting the stability of the optimization objective itself. Even with mixed data, forcing a novice student to immediately match the sharp distribution of an expert teacher creates steep optimization cliffs. The root cause lies in the geometric properties of the divergence objective.

**Goal**: To propose a target-level reformulation that constructs a distributional bridge between teacher and student in logit space, simultaneously addressing gradient explosion in forward KL and mode collapse in reverse KL.

**Key Insight**: Rather than mixing samples at the data level, Veto performs mixing at the distributional level—constructing an intermediate target distribution in logit space that emphasizes consensus regions between teacher and student, effectively "vetoing" harmful updates on low-confidence tokens.

**Core Idea**: Construct a geometric bridging distribution $Q \propto P_T \cdot P_S^\beta$ as a Product-of-Experts consensus filter, where only tokens supported by both the teacher (quality) and the student (confidence) receive high target probability. A single parameter $\beta$ uniformly controls gradient suppression in forward KL and the decisiveness–diversity trade-off in reverse KL.

## Method

### Overall Architecture

Veto modifies the target distribution in standard on-policy KD: instead of directly using the teacher distribution $P_T$ as the target, it constructs an intermediate target $Q$ via geometric interpolation in logit space. For each token position, the teacher and student logits $z_T$ and $z_S$ are used to construct $Q \propto \exp(z_T + \beta \cdot z_S)$, and then either $D_{KL}(Q \| P_S)$ (forward KL) or $D_{KL}(P_S \| Q)$ (reverse KL) is minimized. $\beta$ is linearly decayed from its initial value to 0 over the course of training.

### Key Designs

1. **Adaptive Gradient Veto (Forward KL)**:

    - Function: Suppresses gradient explosion on low-confidence tokens in forward KL.
    - Mechanism: In standard forward KL, when $P_S(y) \to 0$ while $P_T(y) > 0$, the ratio $P_T(y)/P_S(y)$ diverges, causing gradient explosion (gradients exceeding $10^7$ in experiments). After introducing $Q = P_T \cdot P_S^\beta$, the loss term becomes $\mathcal{L}(y) \approx P_S(y)^\beta \log P_S(y)$. By L'Hôpital's rule, the polynomial term $P_S^\beta(y)$ decays to zero faster than the logarithmic term $\log P_S(y)$ diverges, effectively gating out harmful updates on ignorant tokens.
    - Design Motivation: Fundamentally addresses optimization instability during the early stages of on-policy training, when student outputs are highly noisy, without requiring any modification to the model architecture or data generation strategy.

2. **Decisiveness Knob (Reverse KL)**:

    - Function: Provides explicit control over the mode-seeking versus diversity trade-off in reverse KL.
    - Mechanism: The gradient of the reverse KL objective is equivalent to a policy gradient update: $\nabla_\theta \mathcal{L}_{\text{REV}} = \mathbb{E}_{y \sim P_S}[\nabla_\theta \log P_S(y) \cdot A(y)]$, where the advantage function is $A(y) = -\log P_T(y) + (1-\beta) \log P_S(y)$. Setting $\beta=0$ recovers standard reverse KD (full teacher matching); $0<\beta<1$ corresponds to geometric KD (mode-seeking in high-reward regions while maintaining a diversity budget); $\beta \to 1$ approximates pure REINFORCE (zero entropy regularization, collapsing to the single highest-reward mode).
    - Design Motivation: Standard reverse KL lacks an explicit mechanism to control the intensity of mode-seeking behavior. Veto provides a continuous spectrum from KD to RL via $\beta$, allowing users to tune behavior according to task requirements.

3. **Sharpening Effect and Linear Decay Schedule**:

    - Function: Theoretically guarantees that the student converges to a sharpened version of the teacher, while progressively closing the gap during training.
    - Mechanism: At the optimal fixed point $P_S^* = Q$, it holds that $P_S^*(y|x) \propto P_T(y|x)^{1/(1-\beta)}$. Since $0 \leq \beta < 1$, the exponent $1/(1-\beta) > 1$, making the student naturally more decisive (sharper distribution) than the teacher. A linear decay schedule is applied: $\beta \leftarrow \beta \cdot (1 - i/N)$, providing strong protection early in training and gradually recovering standard KD as training progresses.
    - Design Motivation: Greater protection is needed when the student policy is highly noisy early in training; as the student improves, it should progressively narrow the gap with the teacher.

### Loss & Training

Qwen2-0.5B-IT is used as the student and Qwen2-7B-IT as the teacher. The teacher is first fine-tuned on task data via supervised learning, after which 1K instances are sampled from the training set for on-policy student training. Training hyperparameters: learning rate 1e-5, warmup ratio 0.1, dropout 0.1, 3 epochs, 2 H100 GPUs. $\beta$ is selected via grid search and linearly decayed. Task-specific $\beta$ values are: $\beta=0.8$ for reasoning, $\beta=1.0$ for code generation, and $\beta=0.3$ for summarization.

## Key Experimental Results

### Main Results

**Performance Comparison Across Three Domains**

| Method | GSM8K (Accuracy) | HumanEval (Pass@1) | HumanEval (Pass@10) | DialogSum (Win-rate) |
|------|------|------|------|------|
| Teacher SFT | 74.7 | 64.7 | 72.2 | 65.0 |
| Student SFT | 30.7 | 26.9 | 34.6 | 54.0 |
| Supervised KD | 33.4 | 26.8 | 34.5 | 54.3 |
| SKD | 33.6 | 24.8 | 34.8 | 53.6 |
| On-policy KD | 35.1 | 22.9 | 35.3 | 54.3 |
| **Veto (Ours)** | **39.9** | **29.0** | **37.7** | **56.5** |

### Ablation Study

| Configuration | GSM8K Accuracy | Note |
|------|------|------|
| Student SFT | 30.7 | Baseline |
| Supervised KD | 33.4 | +2.7 |
| On-policy KD | 35.1 | +4.4 |
| Veto ($\beta=0.8$) | **39.9** | **+9.2**, best |
| Veto (no decay) | — | Decay schedule beneficial |

**Effect of Different $\beta$ Values**:
- $\beta=0$ degenerates to standard on-policy KD
- $\beta=0.8$ is optimal on GSM8K
- $\beta=1.0$ is optimal for code generation
- $\beta=0.3$ is optimal for summarization
- Excessively large $\beta$ leads to over-sharpening; excessively small $\beta$ provides insufficient protection

### Key Findings

- Veto improves GSM8K accuracy by 9.2 percentage points over Student SFT (30.7%→39.9%) and by 4.8 points over on-policy KD.
- Standard forward KL produces gradients exceeding $10^7$ on ignorant tokens; Veto effectively suppresses these to a stable range.
- HumanEval Pass@1 improves from 22.9 to 29.0 (+6.1); DialogSum Win-rate improves from 54.3 to 56.5 (+2.2).
- The optimal $\beta$ varies across tasks, reflecting the fundamental difference between reasoning tasks (requiring high decisiveness) and generation tasks (requiring diversity).
- Linear $\beta$ decay outperforms constant $\beta$, validating the strategy of strong early protection followed by gradual relaxation.

## Highlights & Insights

- Addressing on-policy KD stability from the geometric properties of divergence objectives is more principled than data-level mixing.
- A single parameter $\beta$ elegantly unifies solutions to both gradient explosion in forward KL and mode collapse in reverse KL.
- Theorem 3 reveals that Veto under reverse KL is equivalent to REINFORCE with scaled entropy regularization, establishing a formal bridge between KD and RL.
- The Product-of-Experts "consensus filter" intuition is clear: only tokens supported by both teacher and student receive high weight.

## Limitations & Future Work

- Experiments are limited to Qwen2-0.5B as student and Qwen2-7B as teacher; validation at larger scales (e.g., 7B→70B) is absent.
- Different tasks require different $\beta$ values, necessitating grid search for optimal hyperparameters.
- Theoretical analysis is primarily at the token level; sequence-level dynamics are not thoroughly examined.
- The relationship with and potential for combination with advanced on-policy methods (e.g., RLHF/DPO) remains underexplored.

## Related Work & Insights

- **vs. GKD (On-policy KD)**: GKD introduced the on-policy distillation framework but did not address objective stability; Veto provides stability guarantees at the target level.
- **vs. SKD (Interleaved Sampling)**: SKD improves feedback quality through interleaved sampling but operates at the data level; Veto operates at the distributional level, making the two approaches orthogonal.
- **vs. MiniLLM/f-distill (Reverse KL)**: These methods use reverse KL to encourage mode-seeking but lack diversity control; Veto provides an explicit decisiveness–diversity trade-off via $\beta$.

## Rating

- Novelty: ⭐⭐⭐⭐ The approach of unifying solutions to two problems from the geometric properties of the objective function is elegant, and the KD–RL bridge has theoretical depth.
- Experimental Thoroughness: ⭐⭐⭐ Validation across three tasks is effective, but the model scale is limited (0.5B–7B) and additional baseline comparisons are lacking.
- Writing Quality: ⭐⭐⭐⭐ Theoretical derivations are clear, intuitive explanations are well-presented, and figures are of high quality.
- Value: ⭐⭐⭐⭐ Provides a concise and effective stabilization solution for on-policy KD with a good balance of theory and practice.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] ATPO: Adaptive Tree Policy Optimization for Multi-Turn Medical Dialogue](../../ICLR2026/medical_imaging/atpo_adaptive_tree_policy_optimization_for_multi-turn_medical_dialogue.md)
- [\[AAAI 2026\] A Principle-Driven Adaptive Policy for Group Cognitive Stimulation Dialogue for Elderly with Cognitive Impairment](../../AAAI2026/medical_imaging/a_principle-driven_adaptive_policy_for_group_cognitive_stimu.md)
- [\[ACL 2026\] Cognitive Policy-Driven LLM for Diagnosis and Intervention of Cognitive Distortions in Emotional Support Conversation](cognitive_policy-driven_llm_for_diagnosis_and_intervention_of_cognitive_distorti.md)
- [\[CVPR 2026\] Accelerating Stroke MRI with Diffusion Probabilistic Models through Large-Scale Pre-training and Target-Specific Fine-Tuning](../../CVPR2026/medical_imaging/accelerating_stroke_mri_with_diffusion_probabilist.md)
- [\[CVPR 2026\] Momentum Memory for Knowledge Distillation in Computational Pathology](../../CVPR2026/medical_imaging/momentum_memory_for_knowledge_distillation_in_computational_pathology.md)

<!-- RELATED:END -->
