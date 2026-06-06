---
title: >-
  [Paper Note] Stable On-Policy Distillation through Adaptive Target Reformulation
description: >-
  [ACL 2026][Model Compression][Knowledge Distillation] This paper proposes Veto, a target-level reformulation method that stabilizes on-policy knowledge distillation by constructing a teacher-student geometric bridge dist…
tags:
  - "ACL 2026"
  - "Model Compression"
  - "Knowledge Distillation"
  - "On-policy Distillation"
  - "Gradient Stability"
  - "KL Divergence"
  - "Target Reformulation"
date: 2026-05-08
content_hash: f7f9998f76c48185
---

# Stable On-Policy Distillation through Adaptive Target Reformulation

**Conference**: ACL 2026  
**arXiv**: [2601.07155](https://arxiv.org/abs/2601.07155)  
**Code**: None  
**Area**: Medical Imaging  
**Keywords**: Knowledge Distillation, On-policy Distillation, Gradient Stability, KL Divergence, Target Reformulation

## TL;DR

This paper proposes Veto, a target-level reformulation method that stabilizes on-policy knowledge distillation by constructing a teacher-student geometric bridge distribution in the logit space. A single parameter $\beta$ simultaneously acts as an adaptive gradient vetoer in the forward KL (suppressing harmful gradients from low-confidence tokens) and a decisiveness knob in the reverse KL (balancing reward-driven behavior and output diversity), achieving a 9.2% improvement over SFT on GSM8K.

## Background & Motivation

**Background**: Knowledge Distillation (KD) is a widespread technique for transferring large language model capabilities to smaller student models. Traditional supervised KD trains on fixed teacher-generated trajectories but suffers from exposure bias—using teacher data during training and self-generated data during inference, leading to performance degradation in autoregressive tasks. On-policy KD mitigates this by learning from the student's self-generated outputs.

**Limitations of Prior Work**: On-policy KD faces severe training instability due to the massive distribution gap between the novice student and the expert teacher: (1) The forward KL objective produces gradient explosion ($P_T(y)/P_S(y) \to \infty$) when the student assigns near-zero probability to tokens preferred by the teacher; (2) The reverse KL objective, while numerically stable, lacks explicit control over mode-seeking intensity, leading to mode collapse and loss of diversity.

**Key Challenge**: Existing methods primarily bridge the gap by mixing teacher and student tokens at the data level but overlook the stability of the optimization objective itself. Even with mixed data, forcing a novice student to immediately match the teacher's sharp distribution creates steep optimization cliffs. The root cause lies in the geometric properties of the divergence objectives.

**Goal**: To propose a target-level reformulation that constructs a distribution bridge between the teacher and student in the logit space, simultaneously resolving gradient explosion in forward KL and mode collapse in reverse KL.

**Key Insight**: Instead of mixing samples at the data level, mix them at the distribution level—create an intermediate target distribution in the logit space that emphasizes teacher-student consensus, effectively "vetoing" harmful updates on low-confidence tokens.

**Core Idea**: Construct a geometric bridge distribution $Q \propto P_T \cdot P_S^\beta$ as a consensus filter in a Product of Experts form. Only tokens supported by both the teacher (quality) and student (confidence) receive high target probabilities. The single parameter $\beta$ uniformly controls gradient suppression in forward KL and the decisiveness-diversity trade-off in reverse KL.

## Method

### Overall Architecture

Veto modifies the target distribution in standard on-policy KD: instead of using the teacher distribution $P_T$ directly, it constructs an intermediate target $Q$ via geometric interpolation in the logit space. For each token position, the teacher and student logits $z_T$ and $z_S$ are computed to form $Q \propto \exp(z_T + \beta \cdot z_S)$, followed by minimizing $D_{KL}(Q \| P_S)$ (forward KL) or $D_{KL}(P_S \| Q)$ (reverse KL). $\beta$ follows a linear decay schedule from an initial value down to 0.

### Key Designs

1.  **Adaptive Gradient Veto (Forward KL)**:
    - **Function**: Suppresses gradient explosion on low-confidence tokens during forward KL.
    - **Mechanism**: In standard forward KL, when $P_S(y) \to 0$ while $P_T(y) > 0$, the $P_T(y)/P_S(y)$ term diverges, causing gradient explosion (gradients exceeding $10^7$ in experiments). By introducing $Q = P_T \cdot P_S^\beta$, the loss term becomes $\mathcal{L}(y) \approx P_S(y)^\beta \log P_S(y)$. By L'Hôpital's rule, the polynomial term $P_S^\beta(y)$ decays to zero faster than the logarithmic term $\log P_S(y)$ diverges, acting as a gate to suppress updates on ignorant tokens.
    - **Design Motivation**: Fundamentally solve optimization instability in the early stages of on-policy training when student outputs are highly noisy, without modifying model architecture or data generation strategies.

2.  **Decisiveness Knob (Reverse KL)**:
    - **Function**: Provides explicit control over mode-seeking vs. diversity-preservation in reverse KL.
    - **Mechanism**: The gradient of reverse KL is equivalent to a policy gradient update: $\nabla_\theta \mathcal{L}_{\text{REV}} = \mathbb{E}_{y \sim P_S}[\nabla_\theta \log P_S(y) \cdot A(y)]$, where the advantage function $A(y) = -\log P_T(y) + (1-\beta) \log P_S(y)$. $\beta=0$ is standard reverse KD (perfect teacher matching); $0<\beta<1$ is geometric KD (finding high-reward regions while maintaining a diversity budget); $\beta \to 1$ is equivalent to pure REINFORCE (zero entropy regularization, collapsing to the single highest-reward mode).
    - **Design Motivation**: Standard reverse KL lacks an explicit mechanism to control the intensity of mode-seeking behavior. Veto provides a continuum from KD to RL via $\beta$, adjustable based on task requirements.

3.  **Sharpening Effect and Linear Decay Schedule**:
    - **Function**: Theoretically ensures the student converges to a sharpened version of the teacher and gradually approaches the teacher during training.
    - **Mechanism**: At the optimal fixed point $P_S^* = Q$, we have $P_S^*(y|x) \propto P_T(y|x)^{1/(1-\beta)}$. Since $0 \leq \beta < 1$, the exponent $1/(1-\beta) > 1$, making the student naturally more decisive (sharper distribution) than the teacher. A linear decay schedule is used: $\beta \leftarrow \beta \cdot (1 - i/N)$, where large $\beta$ provides strong protection early on, gradually reverting to standard KD as training progresses.
    - **Design Motivation**: Students require more protection when the policy is highly noisy early on; as the student improves, the gap with the teacher should be gradually closed.

### Loss & Training

Qwen2-0.5B-IT is used as the student and Qwen2-7B-IT as the teacher. The teacher is first supervised fine-tuned on task data, followed by 1K instances sampled from the training set for student on-policy training. Learning rate is 1e-5, warmup ratio 0.1, dropout 0.1, trained for 3 epochs on 2 H100 GPUs. $\beta$ is chosen via grid search and linearly decayed. Different $\beta$ values are used for different tasks: reasoning $\beta=0.8$, code $\beta=1.0$, and summarization $\beta=0.3$.

## Key Experimental Results

### Main Results

**Performance Comparison Across Three Domains**

| Method | GSM8K (Accuracy) | HumanEval (Pass@1) | HumanEval (Pass@10) | DialogSum (Win-rate) |
| :--- | :--- | :--- | :--- | :--- |
| Teacher SFT | 74.7 | 64.7 | 72.2 | 65.0 |
| Student SFT | 30.7 | 26.9 | 34.6 | 54.0 |
| Supervised KD | 33.4 | 26.8 | 34.5 | 54.3 |
| SKD | 33.6 | 24.8 | 34.8 | 53.6 |
| On-policy KD | 35.1 | 22.9 | 35.3 | 54.3 |
| **Veto (Ours)** | **39.9** | **29.0** | **37.7** | **56.5** |

### Ablation Study

| Configuration | GSM8K Accuracy | Description |
| :--- | :--- | :--- |
| Student SFT | 30.7 | Baseline |
| Supervised KD | 33.4 | +2.7 |
| On-policy KD | 35.1 | +4.4 |
| Veto ($\beta=0.8$) | **39.9** | **+9.2**, Best |
| Veto (No Decay) | — | Decay schedule is beneficial |

**Impact of different $\beta$ values**:
- $\beta=0$ degrades to standard on-policy KD.
- $\beta=0.8$ is optimal for GSM8K.
- $\beta=1.0$ is optimal for code generation.
- $\beta=0.3$ is optimal for summarization.
- Excessively large $\beta$ leads to over-sharpening, while small values provide insufficient protection.

### Key Findings

- Veto improves GSM8K by 9.2 percentage points over Student SFT (30.7%→39.9%) and by 4.8 percentage points over standard on-policy KD.
- Standard Forward KL exhibits gradients exceeding $10^7$ on ignorant tokens; Veto effectively suppresses these within a stable range.
- HumanEval Pass@1 increased from 22.9 to 29.0 (+6.1), and DialogSum Win-rate increased from 54.3 to 56.5 (+2.2).
- Optimal $\beta$ varies by task, reflecting the intrinsic differences between reasoning (high decisiveness needed) and generation (diversity needed).
- Linear $\beta$ decay outperforms constant $\beta$, validating the strategy of "strong protection early, gradual relaxation later."

## Highlights & Insights

- Addresses the stability of on-policy KD from the geometric properties of the divergence objective, which is more fundamental than data-level mixing.
- A single parameter $\beta$ unifiedly solves both forward KL gradient explosion and reverse KL mode collapse with theoretical elegance.
- Theorem 3 reveals that Veto under reverse KL is equivalent to REINFORCE with scaled entropy regularization, establishing a bridge between KD and RL.
- The "consensus filter" intuition in Product of Experts form is clear: only tokens supported by both the teacher and the student receive high weights.

## Limitations & Future Work

- Experiments only utilized Qwen2-0.5B as the student and Qwen2-7B as the teacher; validation on larger scales (e.g., 7B→70B) is missing.
- Different tasks require different $\beta$ values, and optimal hyperparameters must be determined via grid search.
- Theoretical analysis is primary conducted at the token level; the dynamic characteristics at the sequence level were not explored in depth.
- The relationship and combination potential with other advanced on-policy methods (e.g., RLHF/DPO) have not been sufficiently explored.

## Related Work & Insights

- **vs GKD (On-policy KD)**: GKD proposed the on-policy distillation framework but did not resolve objective instability; Veto provides stability guarantees at the objective level.
- **vs SKD (Interleaved Sampling)**: SKD improves feedback quality via interleaved sampling but still operates at the data level; Veto operates at the distribution level, making the two orthogonal.
- **vs MiniLLM/f-distill (Reverse KL)**: Uses reverse KL to encourage mode-seeking but lacks diversity control; Veto provides an explicit decisiveness-diversity trade-off via $\beta$.

## Rating

- Novelty: ⭐⭐⭐⭐ The idea of solving two problems unifiedly from the geometric properties of the loss function is elegant; the KD-RL bridge has theoretical depth.
- Experimental Thoroughness: ⭐⭐⭐ Validated effectively across three tasks, but model sizes are limited (0.5B-7B) and more baselines are needed.
- Writing Quality: ⭐⭐⭐⭐ Theoretical derivations are clear, intuitive explanations are well-placed, and visualization quality is high.
- Value: ⭐⭐⭐⭐ Provides a simple and effective stabilization scheme for on-policy KD, with a good balance of theory and practice.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Entropy-Aware On-Policy Distillation of Language Models](../../ICML2026/model_compression/entropy-aware_on-policy_distillation_of_language_models.md)
- [\[ICLR 2026\] π-Flow: Policy-Based Few-Step Generation via Imitation Distillation](../../ICLR2026/model_compression/pi-flow_policy-based_few-step_generation_via_imitation_distillation.md)
- [\[CVPR 2026\] WPT: World-to-Policy Transfer via Online World Model Distillation](../../CVPR2026/model_compression/wpt_world-to-policy_transfer_via_online_world_model_distillation.md)
- [\[NeurIPS 2025\] A Token is Worth over 1,000 Tokens: Efficient Knowledge Distillation through Low-Rank Clone](../../NeurIPS2025/model_compression/a_token_is_worth_over_1000_tokens_efficient_knowledge_distillation_through_low-r.md)
- [\[ACL 2026\] Find Your Optimal Teacher: Personalized Data Synthesis via Router-Guided Multi-Teacher Distillation](find_your_optimal_teacher_personalized_data_synthesis_via_router-guided_multi-te.md)

</div>

<!-- RELATED:END -->
