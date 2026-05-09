---
title: >-
  [Paper Note] Reason Only When Needed: Efficient Generative Reward Modeling via Model-Internal Uncertainty
description: >-
  [ACL 2026][Model Compression][Generative reward model] This paper proposes E-GRM, a framework that estimates uncertainty from the convergence behavior of parallel decoding, triggers CoT reasoning only when necessary, and employs a discriminative scorer trained with a hybrid loss to evaluate reasoning path quality. E-GRM achieves state-of-the-art performance across multiple reward modeling benchmarks while reducing inference latency by 62%.
tags:
  - ACL 2026
  - Model Compression
  - Generative reward model
  - dynamic CoT triggering
  - model-internal uncertainty
  - discriminative scoring
  - inference efficiency
date: 2026-05-08
content_hash: db358f7fa143c9d5
---

# Reason Only When Needed: Efficient Generative Reward Modeling via Model-Internal Uncertainty

**Conference**: ACL 2026
**arXiv**: [2604.10072](https://arxiv.org/abs/2604.10072)
**Code**: None
**Area**: Model Compression / LLM Efficiency
**Keywords**: Generative reward model, dynamic CoT triggering, model-internal uncertainty, discriminative scoring, inference efficiency

## TL;DR

This paper proposes E-GRM, a framework that estimates uncertainty from the convergence behavior of parallel decoding, triggers CoT reasoning only when necessary, and employs a discriminative scorer trained with a hybrid loss to evaluate reasoning path quality. E-GRM achieves state-of-the-art performance across multiple reward modeling benchmarks while reducing inference latency by 62%.

## Background & Motivation

**Background**: Generative reward models (GRMs) augment LLMs with chain-of-thought (CoT) prompting to enhance reasoning-based evaluation, achieving strong results on complex tasks such as mathematical problem solving and multi-step decision making.

**Limitations of Prior Work**: Existing GRMs suffer from two fundamental issues. First, CoT reasoning is applied indiscriminately to all inputs regardless of difficulty, forcing simple queries through the full CoT pipeline and incurring substantial unnecessary computation. Second, prevailing methods rely primarily on voting mechanisms to aggregate CoT outputs, a coarse-grained approach that fails to capture fine-grained quality differences among reasoning paths.

**Key Challenge**: A dual bottleneck of efficiency and quality—on one hand, reasoning resources must be allocated adaptively according to problem complexity; on the other, a more fine-grained scoring mechanism is needed to differentiate reasoning quality. Existing adaptive CoT methods (e.g., AdaCoT) rely on task-specific heuristics or hand-crafted features, limiting their generalizability.

**Goal**: (1) Identify a task-agnostic signal to determine whether CoT is required; (2) design a more fine-grained evaluation method for reasoning paths beyond voting.

**Key Insight**: The authors observe that, when the same prompt is decoded multiple times in parallel, outputs for easy problems converge rapidly, while outputs for hard problems diverge substantially—this convergence behavior serves as a natural indicator of problem complexity.

**Core Idea**: Use the consensus among parallel generations as an uncertainty estimation signal to dynamically decide whether to trigger CoT, while training a lightweight discriminative scorer with a hybrid regression–ranking loss to provide fine-grained quality scores.

## Method

### Overall Architecture

E-GRM comprises two core modules: (1) a dynamic CoT triggering mechanism based on model-internal uncertainty, and (2) a discriminative scoring module trained with a hybrid loss. Training proceeds in two stages: supervised fine-tuning (SFT) to teach the model both short- and long-reasoning modes, followed by preference optimization via an extended GRPO. At inference time, the system first rapidly determines whether CoT is needed; if so, multiple reasoning paths are generated and the scorer selects the best one.

### Key Designs

1. **Dynamic CoT Triggering**:

    - *Function*: Automatically determines whether to enable CoT reasoning based on problem complexity.
    - *Mechanism*: Given input $x$, $M$ parallel decodes are performed (with varying temperature/sampling parameters). The answer consensus is computed as $\text{Consensus}(x) = \max_y \text{Count}(y) / M$. If consensus $\geq \tau$ (default 0.8), the consensus answer is returned directly; otherwise, full CoT generation is triggered. Approximately 58% of samples are identified as "short-reasoning" instances that bypass CoT entirely.
    - *Design Motivation*: Exploiting the model's own generation behavior as a complexity probe requires no external features or task-specific heuristics, achieving truly task-agnostic adaptive reasoning.

2. **Discriminative Scoring Module**:

    - *Function*: Assigns fine-grained quality scores to generated reasoning paths.
    - *Mechanism*: A lightweight scorer $\mathcal{S}_\phi$ is trained to output quality scores in $[0,1]$. The loss combines Huber Loss (for regression robustness, smoothly transitioning from L2 to L1 for outliers) and Hinge Loss (for ranking discriminability, enforcing a margin $m$ between high- and low-quality paths): $\mathcal{L} = \alpha \cdot \ell_{\text{Huber}} + (1-\alpha) \cdot \ell_{\text{Hinge}}$.
    - *Design Motivation*: Pure voting considers only answer consistency while ignoring reasoning process quality. The hybrid loss enables the scorer to simultaneously calibrate absolute quality and reliably distinguish subtle differences.

3. **Coupled-GRPO**:

    - *Function*: Leverages paired preference data to optimize the policy during reinforcement learning.
    - *Mechanism*: Standard GRPO is extended with a paired reward signal $R_{\text{pair}} = \mathcal{S}_\phi(x, r^+) - \mathcal{S}_\phi(x, r^-) + \beta \cdot \mathbb{I}(\text{Ans}(r^+) = y)$, directly contrasting scorer outputs between positive and negative samples to provide stronger learning signals.
    - *Design Motivation*: Standard GRPO computes relative rewards within independently sampled groups, whereas paired data inherently encodes positive–negative contrasts; directly exploiting this structure yields more targeted gradients.

### Loss & Training

Training proceeds in two stages: (1) In the SFT stage, the model is jointly trained on short-reasoning samples (direct answer prediction) and long-reasoning samples (full CoT sequences), with uncertainty estimation used to automatically partition the dataset. (2) In the GRPO stage, paired preference data and the discriminative scorer are used for alignment optimization, with KL regularization to prevent excessive deviation from the reference policy.

## Key Experimental Results

### Main Results

| Benchmark | Metric | E-GRM (32B) | Prev. SOTA | Gain |
|-----------|--------|-------------|------------|------|
| RM-Bench | Avg | 79.2% | 76.4% (14B) | +2.8% |
| RMB | Overall | 0.743 | 0.738 (GPT-4o) | +0.005 |
| RewardBench | Overall | 91.5% | 90.0% (Self-taught-70B) | +1.5% |
| RewardBench | Reasoning | 95.4% | 88.4% (Self-taught-70B) | +7.0% |

### Ablation Study

| Configuration | Acc (%) | FLOPs (T) | Latency (s) |
|---------------|---------|-----------|-------------|
| Full E-GRM | 78.4 | 15.7 | 2.2 |
| w/o Dynamic CoT | 75.2 | 23.4 | 3.4 |
| w/o Discrim. Scoring | 72.8 | 15.9 | 2.2 |
| Base CoT-GRM | 69.1 | 23.7 | 3.6 |

### Key Findings

- The discriminative scoring module contributes most significantly: its removal reduces accuracy by 5.6%, underscoring the importance of fine-grained scoring for reasoning quality.
- Dynamic CoT triggering yields a 49% reduction in FLOPs and 55% reduction in latency, while accuracy actually improves by 3.2%, demonstrating that unnecessary CoT introduces error propagation.
- Compared to heuristic methods such as AdaCoT, E-GRM achieves higher accuracy (78.4% vs. 76.8%) and lower latency (2.2s vs. 2.9s) without any task-specific prior knowledge.
- Coupled-GRPO provides consistent but moderate gains over standard GRPO (MATH: 78.4% vs. 76.9%).

## Highlights & Insights

- **Parallel decoding consensus as a complexity probe**: This is an elegant design—leveraging the model's own behavioral characteristics rather than external signals to assess reasoning demand, achieving task-agnosticism at zero additional parameter cost. This idea is transferable to any scenario requiring adaptive computation (e.g., early exit, dynamic depth).
- **Hybrid regression–ranking loss**: The combination of Huber and Hinge losses elegantly addresses the dual objectives of "absolute calibration" and "relative ranking," proving more robust than either pure MSE or pure ranking loss alone.
- The finding that 58% of samples are identified as simple problems not requiring CoT is itself a significant observation, indicating severe computational waste in current GRMs on a large portion of straightforward tasks.

## Limitations & Future Work

- Uncertainty estimation via parallel decoding requires $M$ forward passes (M=5). While less costly than full CoT, this is not zero-overhead; its acceptability in extreme low-latency settings warrants further investigation.
- The threshold $\tau$ and the number of parallel samples $M$ are currently set manually and may require domain-specific tuning.
- The discriminative scorer requires annotated quality data for training, and data acquisition costs may hinder rapid deployment in new domains.
- Future directions include combining uncertainty estimation with speculative decoding, or replacing multi-sample inference with internal representations (e.g., attention entropy) from a single forward pass.

## Related Work & Insights

- **vs. DeepSeek-GRM**: DeepSeek-GRM is also a generative reward model but lacks adaptive reasoning; it applies CoT uniformly to all inputs. E-GRM achieves comparable or superior accuracy at substantially reduced computational cost via dynamic triggering.
- **vs. AdaCoT**: AdaCoT employs task-specific heuristics based on estimated solution length to decide whether CoT is needed. E-GRM's parallel consensus approach is entirely task-agnostic and is experimentally demonstrated to be superior in both accuracy and latency.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Parallel decoding consensus as an uncertainty signal is a novel angle; the hybrid loss and GRPO extension are more incremental.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive evaluation on three mainstream benchmarks with complete ablations and well-designed comparisons against AdaCoT.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure and detailed method descriptions, though some formulations are verbose.
- **Value**: ⭐⭐⭐⭐ Addresses a key efficiency bottleneck in GRMs; a 62% latency reduction has substantial practical value for deployment.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] Latent-Condensed Transformer for Efficient Long Context Modeling](latent-condensed_transformer_for_efficient_long_context_modeling.md)
- [\[ACL 2026\] Model Internal Sleuthing: Finding Lexical Identity and Inflectional Features in Modern Language Models](model_internal_sleuthing_finding_lexical_identity_and_inflectional_features_in_m.md)
- [\[ACL 2026\] MAESTRO: Meta-learning Adaptive Estimation of Scalarization Trade-offs for Reward Optimization](maestro_meta-learning_adaptive_estimation_of_scalarization_trade-offs_for_reward.md)
- [\[ACL 2026\] Which Reasoning Trajectories Teach Students to Reason Better? A Simple Metric of Informative Alignment](which_reasoning_trajectories_teach_students_to_reason_better_a_simple_metric_of_.md)
- [\[AAAI 2026\] Credal Ensemble Distillation for Uncertainty Quantification](../../AAAI2026/model_compression/credal_ensemble_distillation_for_uncertainty_quantification.md)

<!-- RELATED:END -->
