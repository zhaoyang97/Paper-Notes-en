---
title: >-
  [Paper Note] Probing RLVR Training Instability through the Lens of Objective-Level Hacking
description: >-
  [ICML 2026][Reinforcement Learning][RLVR] The authors propose the "objective-level hacking" framework, attributing the increasing training-inference discrepancy in MoE large models under RLVR to biased pseudo-signals int…
tags:
  - "ICML 2026"
  - "Reinforcement Learning"
  - "RLVR"
  - "GRPO"
  - "MoE"
  - "Training-Inference Discrepancy"
  - "Objective-Level Hacking"
date: 2026-05-08
content_hash: 29fd92a38efebe55
---

# Probing RLVR Training Instability through the Lens of Objective-Level Hacking

**Conference**: ICML 2026  
**arXiv**: [2602.01103](https://arxiv.org/abs/2602.01103)  
**Code**: None  
**Area**: LLM Alignment / RLHF / Reinforcement Learning Training Stability / MoE  
**Keywords**: RLVR, GRPO, MoE, Training-Inference Discrepancy, Objective-Level Hacking

## TL;DR
The authors propose the "objective-level hacking" framework, attributing the increasing training-inference discrepancy in MoE large models under RLVR to biased pseudo-signals introduced by token-level weight distortion in the optimization objective. Through four sets of experiments on a 30B MoE, they verify that *bias (not variance) is the root cause*.

## Background & Motivation

**Background**: RLVR (Reward-Verifiable Reinforcement Learning, with representative algorithms GRPO/DAPO/GSPO) has become the core post-training paradigm behind inference models such as OpenAI o1 and DeepSeek-R1, demonstrating stronger generalization and long-term gains than SFT in mathematics, code, and agent tasks.

**Limitations of Prior Work**: Especially under the MoE architecture, RLVR training frequently suffers from "sudden collapse"—validation drops, token entropy collapses, and gradient norm anomalies. The most puzzling accompanying phenomenon is the *training-inference discrepancy*: the output token probabilities from the same weights become increasingly inconsistent between vLLM inference and Megatron training, even with parameter synchronization at every step.

**Key Challenge**: This should have been a transient noise due to "infrastructure-level numerical precision differences." Why does it monotonically increase during training, eventually leading to irreversible collapse? Existing patches (TIS, various clip variants, GSPO sequence-level clip) can alleviate the issue, but the underlying mechanism remains unclear.

**Goal**: To answer two specific questions—(1) Why does the training-inference discrepancy accumulate rather than remain constant? (2) Which common techniques (initial discrepancy, token-level clipping, custom token weighting) inadvertently introduce biased signals into the optimization objective?

**Key Insight**: Elevate the concept of "reward hacking" from the verifier to the *objective level*—any token-level weight adjustment is equivalent to adding a $\Delta\mathcal J(\theta)$ term to the original GRPO objective. If this term is positively correlated with a pseudo-signal (such as $\rho_{i,t}^{-1}$), optimization will reinforce the discrepancy, forming a positive feedback loop.

**Core Idea**: Use a unified formula $\mathcal J_{\text{dist}}=\mathcal J + \Delta_{\text{dist}}\mathcal J$ to describe the implicit bias of various "token-level weight distortions" on the optimization objective. Through active injection experiments, it is demonstrated that *bias is the key*, while variance noise does not trigger collapse.

## Method

### Overall Architecture
The framework consists of two parts: (i) Theoretical: decompose the GRPO/GSPO objective into $\mathcal J(\theta)+\Delta\mathcal J(\theta)$, expressing both initial discrepancy and token-level clipping in the unified form $\sum_{i,t} X_{i,t}(\theta)(\phi_{i,t}-1)$ as "token weight perturbations"; (ii) Experimental: on Qwen3-30B-A3B MoE, using verl + vLLM + Megatron to run DAPO-Math-17k, conduct four experiments—(a) comparing GRPO with TIS correction, (b) varying clip range, (c) actively injecting low-probability token weight distortion, (d) injecting unbiased variance noise—to progressively isolate the causal chain "bias ⇒ discrepancy growth ⇒ collapse".

### Key Designs

1. **Unified Formalization of Objective-Level Hacking**:

    - **Function**: Map all "seemingly harmless" token-level modifications to implicit bias in the optimization objective.
    - **Mechanism**: Starting from the ideal objective $\mathcal J(\theta)=\mathbb E_{\text{train}}[\sum_{i,t} X_{i,t}(\theta)]$ ($X_{i,t}=r_{i,t}\hat A_{i,t}/(G|o_i|)$), after shifting rollout from $\pi_{\text{train}}$ to $\pi_{\text{infer}}$ and performing first-order derivation, obtain $\Delta\mathcal J(\theta)\simeq \sum_{i,t}\text{Cov}_{\text{train}}(X_{i,t},\rho_{i,t}^{-1})$, where $\rho_{i,t}=\pi_{\text{train}}/\pi_{\text{infer}}$. Similarly, token-level clip is equivalent to a multiplicative weight $\phi_{i,t}\in\{0,1\}$, yielding $\Delta_{\text{clip}}\mathcal J=\mathbb E_{\text{train}}[\sum X_{i,t}(\phi_{i,t}-1)]$.
    - **Design Motivation**: Seemingly disparate "training accidents" (numerical errors, clipping, custom weighting) are essentially doing the same thing—secretly changing weights for certain tokens. A unified representation enables a consistent experimental approach.

2. **Active Injection Experiments as Causal Probes**:

    - **Function**: Directly verify "which perturbations trigger discrepancy growth" without relying on specific patches.
    - **Mechanism**: Using GSPO (sequence-level clip itself does not induce growth) as a stable baseline, artificially weight low-probability tokens $\varphi_{i,t}=\delta$ if $\pi_{\text{train}}<\pi_{\text{low}}=0.1$, otherwise $=1$, sweeping $\delta\in\{1.2,2,3\}$. This explicitly constructs a biased $\Delta_{\text{inj}}\mathcal J$. As a control, inject unbiased variance noise $\xi_{i,t}\sim\mathcal N(1,\sigma^2)$, with derivation showing $\Delta\mathcal J_{\text{var}}\simeq 0$ (since $\xi-1$ is independent of $Y_{i,t}$).
    - **Design Motivation**: Observing correlation cannot prove causality; active injection allows "switching" the pseudo-signal. If $\delta=1.2$ can immediately trigger collapse, while variance noise does not, it proves the problem is *bias*, not noise itself.

3. **Objective-Level Signal Monitoring + Positive Feedback Loop Description**:

    - **Function**: Transform the abstract hacking concept into a scalar that can be tracked in training logs, and explain why collapse is irreversible.
    - **Mechanism**: Define a proxy $J=\sum_{i,t}\hat A_{i,t}(\varphi_{i,t}-1)$ and plot it in real time during training. A significantly positive Pearson correlation between $J$ and step indicates the pseudo-signal is being "continuously optimized". Simultaneously, statistics of $\rho_{i,t}$ in different probability intervals show that low-probability tokens' $\rho$ persistently deviates below 1 during training; survival bias makes recovery harder, further amplifying hacking and forming a "discrepancy ⇄ hacking" positive feedback loop. Even reverting to early checkpoints cannot recover.
    - **Design Motivation**: Provide industrial RLVR with an actionable early warning indicator—once $J$ increases monotonically, training can be stopped before collapse, rather than discovering the issue only after validation drops.

### Loss & Training
No new loss is introduced; only a "weighted objective" reformulation of existing GRPO/GSPO is used. All experiments are run on 4 nodes × 8 A100s, with each step processing 128 problems × 16 responses, 4 parameter updates, and response length of 8K; GRPO clip defaults to 0.2, GSPO sequence-level clip to 3e-4/4e-4.

## Key Experimental Results

### Main Results

Discrepancy and validation behaviors under different stabilization strategies (qualitative summary based on Figures 2, 4, 5):

| Configuration | Discrepancy Growth | Validation | Notes |
|---|---|---|---|
| GRPO + token clip (vanilla) | Significant increase | Drops | Standard RLVR, poor stability |
| + TIS correction | Noticeably slowed | Improved | Only changes objective, not infra |
| Token clip strong ($\epsilon=0.2$) | Fastest | Earliest collapse | Strong clip = strong bias |
| Token clip weak ($\epsilon=0.28$) | Slower | Better | Weaker clip is more stable |
| GSPO (sequence clip) | No increase | Stable | No token-level bias |
| GSPO + inject $\delta=1.2$ | Triggers increase | Degrades | Only 20% weighting triggers collapse |
| GSPO + variance injection $\xi\sim\mathcal N(1,\sigma^2)$ | No increase | Stable | Bias is the culprit, not variance |

### Ablation Study

| Question | Design | Key Observation |
|---|---|---|
| Clip strength vs. discrepancy | Right clip $\in\{0.2, 0.24, 0.28\}$ | Stronger clip, faster discrepancy growth |
| Bias vs. variance | Biased vs. unbiased token weighting | Unbiased variance does not induce growth (Eq. (22) explanation) |
| Lowering low-probability token weights | Symmetric perturbation as increasing | Also triggers discrepancy growth, indicating the root cause is "distortion" not "weighting direction" |

### Key Findings
- Token-level clip, which "intuitively should stabilize training," actually accelerates discrepancy growth on MoE, as it is essentially a form of token weight distortion, fully equivalent to manually injecting pseudo-signals under the unified framework.
- Discrepancy growth is a *positive feedback* process: hacking causes low-probability tokens' $\rho$ to deviate further from 1, and this deviation further amplifies the effective strength of hacking. Once the model collapses, changing batch or rolling back checkpoints cannot recover it.
- Sequence-level algorithms (GSPO) are stable on MoE not because "clip is looser," but because they structurally avoid token-level weight distortion. This provides concrete guidelines for future MoE-specific RL algorithm design: *never introduce biased weights dependent on token probability into the objective*.

## Highlights & Insights
- Horizontally transfers the concept of "reward hacking" to "objective hacking," for the first time explaining training-inference discrepancy—a "system bug"—as a product of "algorithm-level biased objectives," providing a truly reproducible mechanistic explanation.
- The active injection experiment design is elegant: using GSPO as a stable base, explicitly toggling $\delta$ and $\sigma$, effectively conducting a clean bidirectional causal experiment.
- Provides industrial RLVR with a *monitorable* early signal $J=\sum \hat A_{i,t}(\varphi_{i,t}-1)$, rather than only rolling back after validation collapse.

## Limitations & Future Work
- Experiments are only conducted on Qwen3-30B-A3B MoE; the strength of conclusions for dense models and other MoE routing strategies is unknown.
- The framework explains "what triggers collapse," but does not yet provide "how to automatically debias"; a potential direction is to unify importance sampling correction and token weight distortion into a single objective repair.
- A100 numerical precision is lower than H100; the authors acknowledge this amplifies initial discrepancy. The hacking trigger threshold may differ on higher-precision hardware.

## Related Work & Insights
- **vs DAPO / Dr.GRPO and other GRPO variants**: These modify clip/advantage/length normalization empirically; this paper provides a unified formula $\mathcal J=\mathcal J+\Delta\mathcal J$ explaining "why these modifications are effective."
- **vs GSPO (Zheng 2025)**: GSPO was empirically found to be more stable with sequence clip; this paper uses objective-level hacking to provide a mechanistic explanation, turning its advantage from "empirically discovered" to "theoretically inevitable."
- **vs TIS (Yao 2025)**: TIS directly applies importance sampling correction to the objective; this paper positions it as "a targeted remedy at the $\Delta\mathcal J$ level," and shows that TIS cannot fully eliminate discrepancy under MoE.

## Rating
- Novelty: ⭐⭐⭐⭐ Proposes the concept of objective-level hacking and a unified formula, explaining MoE RLVR collapse mechanism for the first time.
- Experimental Thoroughness: ⭐⭐⭐⭐ Four sets of experiments on 30B MoE: control, intensity sweep, active injection, bias/variance disentanglement.
- Writing Quality: ⭐⭐⭐⭐ Clear derivations, intuitive positive feedback loop description.
- Value: ⭐⭐⭐⭐ Provides concrete guidelines for RLVR algorithm design and a monitorable engineering signal.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Accelerating Stroke MRI with Diffusion Probabilistic Models through Large-Scale Pre-training and Target-Specific Fine-Tuning](../../CVPR2026/medical_imaging/accelerating_stroke_mri_with_diffusion_probabilist.md)
- [\[CVPR 2026\] Event-Level Detection of Surgical Instrument Handovers in Videos](../../CVPR2026/medical_imaging/event_level_detection_of_surgical_instrument_handovers_in_videos.md)
- [\[ICML 2026\] MEG-XL: Data-Efficient Brain-to-Text via Long-Context Pre-Training](meg-xl_data-efficient_brain-to-text_via_long-context_pre-training.md)
- [\[ACL 2026\] Semi-Supervised Diseased Detection from Speech Dialogues with Multi-Level Data Modeling](../../ACL2026/medical_imaging/semi-supervised_diseased_detection_from_speech_dialogues_with_multi-level_data_m.md)
- [\[ACL 2026\] Stable On-Policy Distillation through Adaptive Target Reformulation](../../ACL2026/medical_imaging/stable_on-policy_distillation_through_adaptive_target_reformulation.md)

</div>

<!-- RELATED:END -->
