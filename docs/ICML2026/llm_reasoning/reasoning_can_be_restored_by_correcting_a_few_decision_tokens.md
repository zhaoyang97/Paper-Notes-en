---
title: >-
  [Paper Note] Reasoning Can Be Restored by Correcting a Few Decision Tokens
description: >-
  [ICML 2026][LLM Reasoning][Reasoning model gap] The authors quantify the gap between a base LLM and a reasoning LRM using token-level distribution divergence. They find the gap is highly concentrated in a small fraction…
tags:
  - "ICML 2026"
  - "LLM Reasoning"
  - "Reasoning model gap"
  - "token-level intervention"
  - "sparse control"
  - "planning tokens"
  - "inference-time collaborative decoding"
date: 2026-05-08
content_hash: b565be2c0a22a97e
---

# Reasoning Can Be Restored by Correcting a Few Decision Tokens

**Conference**: ICML 2026  
**arXiv**: [2605.16874](https://arxiv.org/abs/2605.16874)  
**Code**: https://github.com/AlphaLab-USTC/RRTokenIntervention  
**Area**: LLM Reasoning  
**Keywords**: Reasoning model gap, token-level intervention, sparse control, planning tokens, inference-time collaborative decoding

## TL;DR
The authors quantify the gap between a base LLM and a reasoning LRM using token-level distribution divergence. They find the gap is highly concentrated in a small fraction of early, planning-related tokens where the base model is uncertain (~8%). Based on this, they propose "Divergence-Gated One-Token Takeover"—allowing the LRM to generate a single token only at divergence spikes before immediately returning control to the base model. This method restores or exceeds the performance of "Thinking" models of the same size with only a ~4-13% intervention budget.

## Background & Motivation

**Background**: The mainstream paradigm for enhancing reasoning involves large-scale RLVR post-training (e.g., OpenAI o1, DeepSeek-R1, Qwen3-Thinking), making LRMs significantly better than their base versions on math and competitive programming. Simultaneously, a "potentiality perspective" (activation steering, self-rewarding, etc.) suggests base models already possess reasoning machinery, which post-training merely activates or amplifies.

**Limitations of Prior Work**: Explanations from the training side are macro-level and fail to answer **generation-level** questions: where exactly does the base model fail during generation? Is it uniform drift or a few key decisions? Without a token-level causal ledger, hypotheses about why "reasoning mode works" remain speculative.

**Key Challenge**: One must either admit reasoning is distributed across every step (requiring expensive full takeover) or believe key decisions are sparse but lack actionable signals to locate them. The former contradicts observations that "small parameter subspaces induce reasoning," while the latter has lacked explicit token-level metrics.

**Goal**: (1) Define and measure the token-level behavior gap between base and LRM; (2) Characterize the quantity, position, and semantics of these gap tokens; (3) Verify the hypothesis that correcting a few key tokens can restore reasoning capacity.

**Key Insight**: Between a base model $\mathcal{M}_b$ and a strong reasoning model $\mathcal{M}_r$ sharing the same vocabulary, one can compute the next-token distribution divergence $s_t = \mathcal{D}_f(p_b(\cdot|x_t), p_r(\cdot|x_t))$ (defaulting to cross-entropy) along the base model's rollout. This metric exposes where the two models diverge token-by-token without requiring training.

**Core Idea**: The reasoning gap is **sparse, early-stage, planning-related, and aligned with base uncertainty**. Thus, a globally calibrated divergence threshold can act as a gate, letting the LRM take over for just one token at spikes and immediately returning control to the base, leveraging a minimal budget to steer the entire trajectory.

## Method

### Overall Architecture
The method consists of two phases: **Analysis Phase**—using $s_t$ along base rollouts to describe the statistical form of divergence (sparsity, position, semantics, predictive power for failure); **Intervention Phase**—designing an inference-time gate $g_t$ based on these statistics to decide whether the base or LRM generates the next token. Input is the prompt $x_0$, and output is the mixed-decoded sequence $y_{1:T}$. Model parameters and hidden states remain unchanged.

### Key Designs

1.  **Token-level Divergence Metric and Empirical Patterns**:
    - **Function**: Defines a lightweight, step-wise metric for the behavioral gap and characterizes its distribution.
    - **Mechanism**: Calculates $s_t = -\sum_{y} p_b(y|x_t)\log p_r(y|x_t)$ (CE) for each step $t$ on base rollouts. The authors also discuss reverse KL ($D_{\text{rKL}}=\mathcal{D}_{\text{CE}}-H_b$). Four properties emerge: (i) The Lorenz curve deviates significantly from the diagonal with a Gini coefficient $G\!\approx\!0.936$, showing divergence is concentrated in ~1-8% of tokens. (ii) Top-1% divergence tokens are strongly left-skewed in normalized position $u=t/T$, peaking at $u\!\approx\!0.05$. (iii) These tokens have high IoU with the top-p% of the base model's Shannon entropy $H_b(t)$, and the proportion of planning-related words increases from a global 1.89% to 15.75% (entropy) / 14.13% (divergence), an enrichment of ~$7.5$–$8.3\times$. (iv) The mean of top-100 divergence scores within a sample predicts eventual failure (GSM8K AUROC 0.851 vs. entropy baseline 0.817).
    - **Design Motivation**: Establishes the empirical basis for "sparse control"—since the gap is concentrated on planning points and aligns with base uncertainty, intervention has clear target locations and low-cost proxy signals.

2.  **Globally Calibrated Dual Gating**:
    - **Function**: Converts "divergence spikes" into an online binary gate $g_t$ that controls takeover frequency via a budget $r$.
    - **Mechanism**: Divergence scores $\mathcal{S}=\{s_t\}$ are collected on a calibration set to determine a global threshold $\tau = Q_{1-r}(\mathcal{S})$ at the $(1-r)$ quantile. Additionally, a tail-to-global mean ratio $\lambda = \mathbb{E}[s|s>\tau]/\mathbb{E}[s]$ is calculated. The runtime gate is the conjunction of two parts: $g_t = \mathbb{I}[s_t>\tau \land s_t>\lambda\cdot\bar{s}_{t,W}]$, where $\bar{s}_{t,W}=\frac{1}{W}\sum_{i=1}^W s_{t-i}$ is the local moving average. The global threshold ensures tail events, while the local ratio suppresses continuous triggers in high-divergence regions, discretizing takeovers into "spikes."
    - **Design Motivation**: A fixed threshold causes budget fluctuations across prompts; local spikes alone might trigger in flat but noisy regions. The conjunction ensures budget controllability (via $\tau$) and spike sparsity (via $\lambda$). The calibrated $\tau$ is stable across benchmarks.

3.  **One-Token Takeover Sparse Delegation Decoding**:
    - **Function**: Unifies base decoding and LRM intervention under a single sampling rule.
    - **Mechanism**: Samples $y_t \sim p_r(\cdot|x_t)$ if $g_t=1$, otherwise $p_b(\cdot|x_t)$. Each intervention generates exactly one token before returning control to the base. A cheaper version using only $H_b(t)$ (no online $p_r$ query) also restores most of Pass@8 at low budgets.
    - **Design Motivation**: Minimizes the causal mechanism of "decision point correction." The authors argue that planning tokens redirect the trajectory, rather than the LRM needing sustained control. This explains why a ~4% budget intervention outperforms 25% random or early injection—**the location of correction matters more than the amount**.

### Loss & Training
This is an inference-time intervention with **no parameter updates**. The only "training" is calibration: running base rollouts on a small prompt set to determine $\tau$ and $\lambda$. Window size $W$ and spike ratio $r$ are the only hyperparameters.

## Key Experimental Results

### Main Results
Evaluation Setup: base $p_b$ = Qwen3-0.6B/1.7B-Base, guide $p_r$ = Qwen3-8B (Thinking), six math benchmarks. Reports Accuracy / Pass@8 and Recovery = $(P_{\text{Guided}}-P_{\text{Base}})/(P_{\text{Thinking}}-P_{\text{Base}})$.

| Model | Setting | Avg Acc / Pass@8 | Recovery |
| :--- | :--- | :--- | :--- |
| Qwen3-0.6B-Base | — | 13.0 / 36.0 | — |
| Qwen3-0.6B-Base | +Guided $\bar{\rho}\!\approx\!0.04$ | 29.1 / 61.4 | 91% |
| Qwen3-0.6B-Base | +Guided $\bar{\rho}\!\approx\!0.13$ | 52.4 / 80.0 | **157%** |
| Qwen3-0.6B (Thinking) | — | 43.4 / 64.1 | 100% baseline |
| Qwen3-1.7B-Base | +Guided $\bar{\rho}\!\approx\!0.16$ | 62.1 / 83.8 | 112% |
| Qwen3-1.7B (Thinking) | — | 64.1 / 80.3 | — |
| Qwen3-8B (Thinking) | — | 78.1 / 87.3 | Upper Bound |

With only 13% budget, 0.6B-Base + Guided **exceeds** the same-sized thinking model (Pass@8 80.0 vs. 64.1) and achieves 157% recovery relative to the teacher baseline, indicating a synergy in mixed decoding.

### Ablation Study

| Config | Budget $\rho$ | Avg Acc | Avg Pass@8 | Note |
| :--- | :--- | :--- | :--- | :--- |
| Base | 0.00 | 13.0 | 36.0 | Baseline |
| +Random | 0.25 | 26.4 | 55.2 | 6× Budget |
| +Early-only | 0.25 | 25.7 | 58.4 | 6× Budget |
| +Guided (**Ours**) | **0.04** | **29.1** | **61.4** | 1× Budget |

| Category | Global | Takeover Set % | Enrichment |
| :--- | :--- | :--- | :--- |
| Planning | 1.9% | **33.3%** | **17.6×** |
| Execution | 98.1% | 66.7% | 0.7× |

### Key Findings
- **Position Selection > Injection Amount**: Guided intervention at 1x budget outperforms random/early baselines at 6x budget. The key is **which token** is corrected, and "early stage" alone is insufficient—it must hit the divergence spikes.
- **Pass@8 vs. $\rho$ is a Non-linear "Knee Curve"**: The first few percent of takeovers yield the highest gains (41%→61% at $\rho\!\approx\!3\%$). Marginal returns diminish after ~8%.
- **Takeover Semantics Focused on Planning**: Planning tokens are enriched 17.6x in the intervention set, while execution tokens are suppressed (0.7x). Qualitative examples show the LRM often inserts "stop-and-check" tokens before returning control for calculation.
- **Cross-family Generalization**: Using LLaMA-3.1-8B as base and DeepSeek-R1-Distill-Llama-8B as guide restores 91% of the gap with ~20% budget.

## Highlights & Insights
- **Translating RL Efficacy into Token-level Facts**: Moves beyond macro training-side explanations to show that ~8% of early planning tokens account for ~94% of the reasoning gap.
- **Self-closing Diagnostic and Intervention Signal**: The top-K divergence mean acts as both a failure predictor and an intervention trigger, unifying "what went wrong" and "where to fix it."
- **Entropy-based Deployment**: When $p_r$ queries are expensive, base entropy $H_b(t)$ serves as a strong proxy, suggesting base models already know when they are likely to fail.
- **Transferable Principles**: The "sparse delegation + one-token takeover" template is applicable to speculative decoding, agent routing, and data filtering for on-policy distillation.

## Limitations & Future Work
- Primarily focused on Qwen3 and math domains. While LLaMA and GPQA-Diamond show promise, general performance on code, multi-hop QA, and larger models needs systematic verification.
- Runtime requires querying both $p_b$ and $p_r$, which is not inherently faster than running the full thinking model. It serves more as a "diagnostic sparse control" than an optimized system.
- Calibration depends on a hold-out set; the robustness of $\tau$ and $\lambda$ against distribution shifts remains to be explored.
- The source of the >100% recovery (surpassing the teacher) is not fully decomposed; it likely stems from a beneficial ensemble effect in mixed decoding.

## Related Work & Insights
- **vs. High-entropy token policy update (Wang et al., 2025a)**: That work uses high-entropy tokens to constrain gradients in RL; this work uses divergence for inference-time intervention and shows cross-model divergence is a more precise signal than self-entropy.
- **vs. Speculative decoding / RouteLLM**: Unlike speculative decoding (which targets speed using a drafter), this reverse-uses the base model as a default drafter to restore LRM-level performance.
- **vs. RelayLLM (Huang et al., 2026)**: While RelayLLM hands over difficult steps to large models, this method operates at a finer token level and takes over only a single token, which is more sparse and controllable.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Quantifying the "reasoning gap" via CE divergence on rollouts is a clear and insightful perspective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive across math benchmarks and model sizes, though limited in non-math domain coverage.
- **Writing Quality**: ⭐⭐⭐⭐ Clean narrative with a direct mapping between findings and designs.
- **Value**: ⭐⭐⭐⭐ Provides rare token-level causal evidence for the efficacy of reasoning models and offers a deployable sparse intervention scheme.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Can Reasoning Path still be Effective as Input? Bridging Post-Reasoning to Chain-of-Thought Compression](../../ACL2026/llm_reasoning/can_reasoning_path_still_be_effective_as_input_bridging_post-reasoning_to_chain-.md)
- [\[NeurIPS 2025\] Adaptive Dual Reasoner: Large Reasoning Models Can Think Efficiently by Hybrid Reasoning](../../NeurIPS2025/llm_reasoning/adaptive_dual_reasoner_large_reasoning_models_can_think_efficiently_by_hybrid_re.md)
- [\[NeurIPS 2025\] Unlabeled Data Can Provably Enhance In-Context Learning of Transformers](../../NeurIPS2025/llm_reasoning/unlabeled_data_can_provably_enhance_in-context_learning_of_transformers.md)
- [\[ACL 2026\] Is Chain-of-Thought Really Not Explainability? Chain-of-Thought Can Be Faithful without Hint Verbalization](../../ACL2026/llm_reasoning/is_chain-of-thought_really_not_explainability_chain-of-thought_can_be_faithful_w.md)
- [\[ICLR 2026\] I Can't Believe It's Not Robust: Catastrophic Collapse of Safety Classifiers under Embedding Drift](../../ICLR2026/llm_reasoning/i_cant_believe_its_not_robust_catastrophic_collapse_of_safety_classifiers_under_.md)

</div>

<!-- RELATED:END -->
