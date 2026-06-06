---
title: >-
  [Paper Note] Furina: Fragmented Uncertainty-Driven Refusal Instability Attack
description: >-
  [ICML 2026][Multimodal VLM][Refusal Instability Band] This paper employs multi-metric diagnostics to demonstrate that "LLM safety decision-making is not a binary threshold but instead contains a refusal instability band…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "Refusal Instability Band"
  - "Semantic Entropy"
  - "Internal Security Signal Decoupling"
  - "Fragmented Prompts"
  - "Cross-model Transfer"
date: 2026-05-08
content_hash: eae9ebe5662d64fc
---

# Furina: Fragmented Uncertainty-Driven Refusal Instability Attack

**Conference**: ICML 2026  
**arXiv**: [2605.26158](https://arxiv.org/abs/2605.26158)  
**Code**: https://github.com/0xCavaliers/Furina_Jailbreak  
**Area**: LLM Security / Jailbreak Attacks / Uncertainty Quantization / Multimodal Security  
**Keywords**: Refusal Instability Band, Semantic Entropy, Internal Security Signal Decoupling, Fragmented Prompts, Cross-model Transfer  

## TL;DR
This paper employs multi-metric diagnostics to demonstrate that "LLM safety decision-making is not a binary threshold but instead contains a refusal instability band," characterized by "rising external uncertainty while internal safety signals decrease." Consequently, the authors propose Furina—a jailbreak attack requiring no model-specific optimization that forces inputs into the instability band by shattering malicious intent into contextual narratives, outperforming strong baselines on HarmBench.

## Background & Motivation
**Background**: The industry generally perceives LLM/MLLM safety alignment as a clean binary decision boundary—one side for refusal and the other for compliance—assuming this boundary is relatively sharp.

**Limitations of Prior Work**: The authors identify three empirical phenomena contradicting the "binary boundary": repeated sampling of the same input drifts between refusal/compliance, minor paraphrasing flips decisions, and adversarial prompts successful on one model often fail on another. This suggests the "boundary" is far more blurred and context-dependent than assumed.

**Key Challenge**: Most existing attacks find sharp adversarial samples for specific models via GCG/AutoDAN, resulting in poor transferability. Meanwhile, most defenses assume "unsafe prompts elicit separable internal representations," a hypothesis the authors suspect fails near the boundary.

**Goal**: (1) Formalize and quantify this instability band; (2) Identify diagnostic metrics strongly correlated with instability that appear consistently across methods; (3) Reverse these metrics as "targets" to construct a universal jailbreak method.

**Key Insight**: Jailbreaking is viewed as a unified process of "pushing the model state into high-uncertainty regions"—role-playing, identity blurring, accumulating contextual entropy in multi-turn dialogues, and injecting cross-modal noise via adversarial suffixes are essentially entropy amplifiers.

**Core Idea**: Instead of searching for specific adversarial token sequences, the method constructs "fragmented, scene-anchored" prompts. Malicious intent is disassembled into semantically shifted sub-problems embedded within a metaphorical scenario, forcing the model into incorrect compliance under high uncertainty.

## Method

### Overall Architecture
The paper consists of two parts: diagnosis (Section 3), characterizing the "instability band" with a set of metrics, and the attack (Section 4 + Fig. 2), which reverses diagnostic signals into attack objectives. The pipeline:

1.  **Instability Band Formalization**: Define compliance probability $\pi_\theta(x) := \mathbb{E}_{Y\sim p_\theta(\cdot|x)}[C(Y)]$ and partition the input space using thresholds $\tau_-, \tau_+$ into stable refusal $\mathcal{S}$, stable compliance $\mathcal{U}$, and the instability band $\mathcal{I}$.
2.  **Multi-metric Diagnosis**: Sample each prompt $M$ times to calculate five-dimensional features: ASR, token entropy $H_\mathrm{tok}$, semantic entropy $H_\mathrm{sem}$, HiddenDetect signal $HD_{\max}$, and Refusal Direction signal $RD_{\max}$.
3.  **Semantic Rewriting Ladder Experiments**: Rewrite malicious queries across five levels (Original→Minor→Moderate→High→Semantic) to trace metrics along the path of contextual diffusion.
4.  **Furina Attack Construction**: Decompose original malicious intent into "intent-preserving" semantically shifted sub-problems and generate a metaphorical scene description as a contextual anchor. Text-only models receive sub-problems directly, while MLLMs receive the scene description rendered as typographic or diffusion-generated images alongside the text.

### Key Designs

1.  **Compliance Probability Characterization of the Refusal Instability Band**:
    - **Function**: Transforms the qualitative question of whether safety behavior is binary into a measurable probabilistic problem.
    - **Mechanism**: Repeats sampling $M$ times for a fixed input $x$ to obtain binary compliance judgments $C(Y^{(m)})$. $\pi_\theta(x)$ is partitioned into three intervals: $\mathcal{S}=\{x:\pi_\theta(x)\le\tau_-\}$, $\mathcal{U}=\{x:\pi_\theta(x)\ge\tau_+\}$, and $\mathcal{I}=\{x:\tau_-<\pi_\theta(x)<\tau_+\}$. Dataset-level ASR is defined as the frequency where at least one of $M$ samples is judged UNSAFE: $\mathrm{ASR}=\tfrac{1}{N}\sum_i \mathbb{I}[\max_m C(Y_i^{(m)})=1]$, making the diagnosis robust to nucleus sampling randomness.
    - **Design Motivation**: Single greedy sampling "collapses" the instability band into a deterministic output, masking the issue. At $M=8$, ASR transitions smoothly from $\mathcal{S}$ to $\mathcal{U}$ (e.g., 0.02→0.04→0.11→0.56→0.77 on Qwen3-8B), empirically proving the non-binary nature.

2.  **External-Internal Decoupled Diagnostic Signature**:
    - **Function**: Identifies recognizable fingerprints of the "instability band" to explain why probe-based defenses fail.
    - **Mechanism**: Externally measures two types of entropy—token-level entropy $H_\mathrm{tok}(x) = \frac{1}{M}\sum_m \frac{1}{T^{(m)}}\sum_t \mathcal{H}(p_\theta(v|x,y^{(m)}_{<t}))$ and semantic entropy $H_\mathrm{sem}(x) = \frac{2}{M(M-1)}\sum_{i<j} d(\phi(Y^{(i)}),\phi(Y^{(j)}))$. Internally uses HiddenDetect's $HD_{\max} = \max_l \mathrm{proj}(\mathbf{h}_l)\cdot \mathbf{r}/(\|\mathrm{proj}(\mathbf{h}_l)\|\|\mathbf{r}\|)$ and Refusal Direction's $RD_{\max}=\max_l \mathbf{a}^{(l)}\cdot \mathbf{r}^{(l)}/\|\mathbf{r}^{(l)}\|$. Along the rewrite ladder, ASR and $H_\mathrm{tok}$ increase, $H_\mathrm{sem}$ peaks mid-way, while $HD_{\max}$ and $RD_{\max}$ monotonically decrease.
    - **Design Motivation**: This decoupling (rising external noise vs. falling internal safety signals) provides a mechanistic explanation for why hidden-state probes fail against sophisticated jailbreaks—the model is pushed to a representational state that looks harmless but behaves compliantly.

3.  **Furina: Semantic Drift Sub-problems + Metaphorical Scene Anchors**:
    - **Function**: Proactively pushes inputs into the instability band $\mathcal{I}$ without per-model searching.
    - **Mechanism**: A scheduler LLM splits the original malicious query into multiple "intent-preserving + semantically shifted" sub-problems and generates a metaphorical scene as cohesive context. For text-only models, sub-problems are sent directly; for MLLMs, the scene is used as a synthetic anchor or rendered into typographic/diffusion images. This mechanism directionally amplifies $H_\mathrm{tok}$ and context complexity to trigger the decoupled signature identified in Section 3.
    - **Design Motivation**: Unlike AmpleGCG or AutoDAN which require gradients or iterative search, Furina generates instability signals purely through prompt engineering, independent of target model weights, enabling cross-model transfer. In Table 2, Furina yields higher $H_{tok}$ (0.396) than all baselines and achieves an ASR of 0.86.

## Key Experimental Results

### Main Results: Semantic Rewriting Ladder Diagnosis (Excerpt from Table 1)

| Model / Dataset | Rewrite Level | ASR | $H_\mathrm{tok}$ | $RD_{\max}$ |
|---|---|---|---|---|
| LLaMA-2-7B / AdvBench | Original | 0.01 | 0.345 | 0.677 |
| LLaMA-2-7B / AdvBench | Semantic | 0.42 | 0.435 | 0.083 |
| Qwen3-8B / AdvBench | Original | 0.02 | 0.235 | – |
| Qwen3-8B / AdvBench | High | 0.56 | 0.320 | – |
| Qwen3-8B / AdvBench | Semantic | 0.77 | 0.334 | – |
| LLaMA-2-7B / HarmBench | Original | 0.08 | 0.346 | 0.548 |
| LLaMA-2-7B / HarmBench | Semantic | 0.72 | 0.428 | 0.070 |

Observation: Across all models, ASR and $H_\mathrm{tok}$ increase monotonically, $RD_{\max}$ decreases monotonically, and $H_\mathrm{sem}$ peaks at Moderate/High levels before falling—tracing the $\mathcal{I}$→$\mathcal{U}$ transition.

### Cross-Method Comparison (Table 2, Average of LLaMA-2-7B-Chat and Qwen3-8B)

| Method | Category | $H_\mathrm{tok}$ | $H_\mathrm{sem}$ | $HD_{\max}$ | ASR |
|---|---|---|---|---|---|
| Original prompt | — | 0.289 | 0.091 | 0.023 | 0.08 |
| AmpleGCG | Suffix Optimization | 0.306 | 0.138 | 0.019 | 0.24 |
| PAIR | Auto Prompt Search | 0.316 | 0.104 | 0.021 | 0.18 |
| AutoDAN | Auto Prompt Search | 0.360 | 0.132 | 0.012 | 0.39 |
| ActorBreaker | Multi-turn Context | 0.378 | 0.112 | – | 0.81 |
| **Furina (Ours)** | Fragmented + Scene Anchor | **0.396** | 0.101 | – | **0.86** |

### Key Findings
- All jailbreak methods exhibit the same signature: "$H_\mathrm{tok}$ rises, $HD_{\max}$ falls," proving uncertainty amplification is the root cause of successful jailbreaks, while specific forms (gradients, role-play, etc.) are merely different implementation paths.
- $H_\mathrm{sem}$ is method-dependent: AmpleGCG and AutoDAN cause semantic dispersion, whereas PAIR and Furina maintain surface semantic consistency, indicating "semantically stable compliance" is the most dangerous jailbreak outcome.
- When feeding typographic images and diffusion-generated scenes to MLLMs, cross-modal mismatch further amplifies $H_\mathrm{tok}$, making the diagnostic signature equally valid for MLLMs.

## Highlights & Insights
- Elevating the "binary boundary hypothesis" to a falsifiable hypothesis and empirically refuting it with intermediate values of $\pi_\theta(x)$ is the paper's cleanest methodological contribution.
- Revealing the decoupling of "rising external uncertainty vs. falling internal safety signals" provides a mechanistic explanation for why hidden-state probes fail against mature jailbreaks, serving as a warning for defense research.
- The reversal strategy of treating "diagnostic metrics" as "attack objective functions" bypasses reliance on model weights, yielding a naturally transferable attack paradigm.

## Limitations & Future Work
- The ASR calculation counts a success if *any* of $M$ samples succeed, which may be aggressive for real-world deployment scenarios using single samples.
- Internal signals only considered HiddenDetect and Refusal Direction probes; decoupling from stronger multi-vector probes remains unverified.
- Furina relies on a scheduler LLM for semantic drift and scene generation; attack cost and stealth are influenced by the scheduler's own alignment.
- Future defensive work could leverage the multi-metric signatures to construct "instability-aware" refusal enhancement methods.

## Related Work & Insights
- **vs AmpleGCG / AutoDAN (Gradient/Search)**: These find sharp adversarial points for specific models with low transferability; Furina achieves robust cross-model attacks by directionally exciting the instability band.
- **vs ActorBreaker (Multi-turn)**: Multi-turn attacks rely on accumulating contextual entropy to enter $\mathcal{I}$; Furina achieves this in a single turn, but the underlying mechanism (entropy amplification) is consistent.
- **vs HiddenDetect / Refusal Direction (Internal Probe Defenses)**: These assume harmful samples activate separable representations; this work proves they fail in the instability band, suggesting defenses need to incorporate external uncertainty signals.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Formalizing the "instability band," proving "external-internal decoupling," and reversing diagnostics into attack targets is a cohesive tripartite contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 4 open-source/commercial LLMs and MLLMs, two benchmarks, 5 rewrite levels, and 5 metric categories.
- Writing Quality: ⭐⭐⭐⭐ The transition from diagnosis to attack is smooth; notation for metrics is slightly dense and requires the appendix for full clarity.
- Value: ⭐⭐⭐⭐⭐ Provides actionable tools for both attackers (transferable templates) and defenders (diagnostic signatures and evidence of failure).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] TUR-DPO: Topology- and Uncertainty-Aware Direct Preference Optimization](tur-dpo_topology-_and_uncertainty-aware_direct_preference_optimization.md)
- [\[CVPR 2026\] FlowHijack: A Dynamics-Aware Backdoor Attack on Flow-Matching VLA Models](../../CVPR2026/multimodal_vlm/flowhijack_dynamics_aware_backdoor_attack_on_flow_matching_vla_models.md)
- [\[ICLR 2026\] Reasoning-Driven Multimodal LLM for Domain Generalization](../../ICLR2026/multimodal_vlm/reasoning-driven_multimodal_llm_for_domain_generalization.md)
- [\[ICML 2026\] LIMSSR: LLM-Driven Sequence-to-Score Reasoning under Training-Time Incomplete Multimodal Observations](limssr_llm-driven_sequence-to-score_reasoning_under_training-time_incomplete_mul.md)
- [\[CVPR 2026\] When to Think and When to Look: Uncertainty-Guided Lookback](../../CVPR2026/multimodal_vlm/when_to_think_and_when_to_look_uncertainty-guided_lookback.md)

</div>

<!-- RELATED:END -->
