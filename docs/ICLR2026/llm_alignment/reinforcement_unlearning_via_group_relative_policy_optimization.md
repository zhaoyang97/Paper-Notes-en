---
title: >-
  [Paper Note] PURGE: Reinforcement Unlearning via Group Relative Policy Optimization
description: >-
  [ICLR 2026][LLM Alignment][Machine Unlearning] PURGE reformulates LLM unlearning as a verifiable RL task, employing the GRPO framework with intrinsic reward signals (penalizing mentions of forbidden concepts) to achieve safe and consistent knowledge removal. It consumes 46× fewer tokens than the SOTA while improving fluency by +5.48% and adversarial robustness by +12.02%.
tags:
  - ICLR 2026
  - LLM Alignment
  - Machine Unlearning
  - GRPO
  - Verifiable Rewards
  - LLM Compliance
  - Privacy Protection
date: 2026-05-08
content_hash: 4dbe8d2547d75402
---

# PURGE: Reinforcement Unlearning via Group Relative Policy Optimization

**Conference**: ICLR 2026
**arXiv**: [2601.20568](https://arxiv.org/abs/2601.20568)
**Code**: N/A
**Area**: LLM Alignment
**Keywords**: Machine Unlearning, GRPO, Verifiable Rewards, LLM Compliance, Privacy Protection

## TL;DR
PURGE reformulates LLM unlearning as a verifiable RL task, employing the GRPO framework with intrinsic reward signals (penalizing mentions of forbidden concepts) to achieve safe and consistent knowledge removal. It consumes 46× fewer tokens than the SOTA while improving fluency by +5.48% and adversarial robustness by +12.02%.

## Background & Motivation

**Background**: The GDPR "right to be forgotten" and the EU AI Act require AI systems to delete specific data on demand. LLMs inadvertently memorize sensitive or copyrighted data during pretraining. Conventional unlearning methods include gradient ascent, DPO/NPO preference optimization, and rejection fine-tuning.

**Limitations of Prior Work**:
   - Gradient Ascent: Overly aggressive updates cause model collapse (loss of fluency and utility).
   - Preference Optimization (DPO/NPO): Relies on external reward models, increasing system complexity.
   - Rejection Fine-Tuning: Creates shortcut behaviors; latent traces may resurface under specific conditions.
   - Context-Based Methods: Risk data leakage and consume limited context windows.

**Key Challenge**: Effective unlearning, utility preservation, and adversarial robustness are difficult to achieve simultaneously.

**Key Insight**: DeepSeek's RLVR (RL with Verifiable Rewards) succeeds on reasoning tasks → unlearning is likewise a verifiable task (whether data has been removed is objectively measurable) → GRPO can be applied to optimize unlearning.

**Core Idea**: LLM unlearning is inherently a verifiable task — the GRPO intrinsic reward function penalizes mentions of forbidden entities, training an unlearning model in the same manner as a reasoning model.

## Method

### Overall Architecture
A three-stage pipeline: (1) construct a synthetic unlearning corpus (generate from the model itself + extract forbidden entity sets via NER) → (2) design an intrinsic reward function (detecting whether forbidden concepts appear in outputs) → (3) optimize the policy with GRPO to minimize the probability of forbidden concept occurrence.

### Key Designs

1. **Synthetic Unlearning Corpus Construction**: Reuses the query set from the RWKU benchmark; performs inference on the target model to capture its current knowledge; applies GPT-4-conditioned NER to extract the entity set $\mathcal{X}_0$ for each unlearning target.
2. **Intrinsic Reward Function**: $\varphi(y) \in \{0, 1\}$ — reward 1 if the output contains no forbidden entities, 0 otherwise. No external reward model is required; rewards are entirely rule-defined (verifiable).
3. **GRPO Optimization**: Standard GRPO framework with KL regularization to preserve general capabilities. Theoretical analysis proves that the probability of forbidden tokens decays geometrically: $P(\text{forbidden token at step } t) \leq (1-\epsilon)^t$.
4. **Theoretical Guarantees**: Utility preservation is guaranteed via a high-probability bound on KL divergence.

### Loss & Training
- GRPO clipped surrogate objective + KL penalty.
- Reward: binary (1 = no forbidden entities, 0 = forbidden entities present).
- No external reward model or human annotation required.

## Key Experimental Results

### Main Results (RWKU Benchmark)

| Method | Forget Efficacy↑ | Utility Retention↑ | Fluency↑ | Adversarial Robustness↑ | Tokens/Target↓ |
|---|---|---|---|---|---|
| Gradient Ascent | High | 60% | −15% | Low | High |
| DPO | Medium | 85% | +2% | Medium | Medium |
| Rejection Tuning | Medium | 90% | 0% | Low | Low |
| **PURGE** | **11%** | **98%** | **+5.48%** | **+12.02%** | **46× fewer** |

### Key Findings
- **High Token Efficiency**: Requires 46× fewer tokens per unlearning target than the SOTA.
- **Near-Lossless Utility**: Retains 98% of original utility — far surpassing gradient ascent methods.
- **Improved Fluency**: +5.48% gain, likely attributable to the alignment effect of GRPO's KL regularization.
- **Substantially Enhanced Adversarial Robustness**: +12.02% — the unlearned model is significantly less susceptible to adversarial attacks that attempt to reactivate memorized knowledge.
- **Theoretical Guarantees**: Geometric decay of forbidden token probability and a KL-divergence-based utility preservation bound.

## Highlights & Insights
- **Reformulating unlearning as a verifiable RL task** is the central innovation — GRPO was originally designed for reasoning, yet "whether a forbidden concept is mentioned" is equally verifiable, and this insight bridges RL with privacy compliance.
- **Eliminating the external reward model** substantially reduces engineering complexity — intrinsic rule-based rewards are far simpler than training a preference model and support arbitrary granularity in defining unlearning targets.
- **Practical value of theoretical guarantees**: The geometric decay bound provides a quantitative prediction of unlearning convergence speed, while the KL bound offers an upper-bound control on utility loss.

## Limitations & Future Work
- The absolute forget efficacy of 11% is relatively low — while utility is well preserved, the unlearning is insufficiently thorough.
- Validation is limited to the single RWKU benchmark — additional unlearning scenarios are needed.
- Synthetic corpus construction relies on GPT-4 for NER, introducing a dependency on an external large model.
- The binary reward may be too coarse-grained — it does not distinguish between partial and complete information leakage.
- Performance on models larger than 7B parameters has not been evaluated.

## Related Work & Insights
- **vs. Gradient Ascent**: GA achieves high forget efficacy but carries significant collapse risk; PURGE avoids collapse via GRPO + KL constraints.
- **vs. DPO/NPO**: Preference optimization requires external reward models; PURGE uses intrinsic verifiable rewards with zero additional overhead.
- **vs. Rejection Tuning**: RT creates shortcut behaviors and latent traces may resurface; PURGE directly optimizes the probability distribution.

## Rating
- Novelty: ⭐⭐⭐⭐ Applying GRPO to unlearning is an interesting new direction, though the technical contribution is relatively straightforward.
- Experimental Thoroughness: ⭐⭐⭐ Limited to a single benchmark (RWKU); broader validation is needed.
- Writing Quality: ⭐⭐⭐⭐ Theoretical sections are rigorous and the method is clearly described.
- Value: ⭐⭐⭐⭐ The paradigm of treating unlearning as a verifiable task is inspiring, but the 11% forget efficacy requires improvement.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Group-Relative REINFORCE Is Secretly an Off-Policy Algorithm: Demystifying Some Myths About GRPO and Its Friends](group-relative_reinforce_is_secretly_an_off-policy_algorithm_demystifying_some_m.md)
- [\[ICLR 2026\] Slow-Fast Policy Optimization: Reposition-Before-Update for LLM Reasoning](slow-fast_policy_optimization_reposition-before-update_for_llm_reasoning.md)
- [\[NeurIPS 2025\] GVPO: Group Variance Policy Optimization for Large Language Model Post-Training](../../NeurIPS2025/llm_alignment/gvpo_group_variance_policy_optimization_for_large_language_model_post-training.md)
- [\[ICLR 2026\] Learning More with Less: A Dynamic Dual-Level Down-Sampling Framework for Efficient Policy Optimization](learning_more_with_less_a_dynamic_dual-level_down-sampling_framework_for_efficie.md)
- [\[ICLR 2026\] Hierarchy-of-Groups Policy Optimization for Long-Horizon Agentic Tasks](hierarchy-of-groups_policy_optimization_for_long-horizon_agentic_tasks.md)

<!-- RELATED:END -->
