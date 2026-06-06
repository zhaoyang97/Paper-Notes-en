---
title: >-
  [Paper Note] The Realignment Problem: When Right becomes Wrong in LLMs
description: >-
  [ICML 2026][LLM Alignment][realignment] This paper formalizes the "what if the policy changes after model deployment" scenario as the Realignment problem…
tags:
  - "ICML 2026"
  - "LLM Alignment"
  - "realignment"
  - "alignment-reality gap"
  - "triage"
  - "IPO"
  - "NPO"
  - "bi-level optimization"
date: 2026-05-08
content_hash: be8494f359d9b9b1
---

# The Realignment Problem: When Right becomes Wrong in LLMs

**Conference**: ICML 2026  
**arXiv**: [2511.02623](https://arxiv.org/abs/2511.02623)  
**Code**: Available (release mentioned in the paper)  
**Area**: LLM Alignment / Preference Learning / Policy Realignment  
**Keywords**: realignment, alignment-reality gap, triage, IPO, NPO, bi-level optimization

## TL;DR
This paper formalizes the "what if the policy changes after model deployment" scenario as the Realignment problem, and proposes the TRACE framework: using a stronger proxy model to triage existing preference pairs into three categories (Invert / Punish / Retain), then performing surgical realignment with a hybrid IPO+NPO+KL objective, enabling adaptation to policy drift without a new round of human annotation.

## Background & Motivation

**Background**: In industrial LLM deployment, mainstream alignment methods are RLHF / DPO—using a BPO-labeled binary preference dataset $\mathcal{D}=\{(x, y_w, y_l)\}$ to train a model $\mathcal{M}_\theta$. This alignment is guideline-dependent: once data is embedded in parameters, the original policy guideline becomes invisible and unchangeable.

**Limitations of Prior Work**: Regulations (EU AI Act, NIST RMF), cultural, and organizational risk preferences are constantly evolving; what was compliant yesterday may be non-compliant today. Redoing full-scale human annotation is prohibitively expensive; machine unlearning can only delete, not "modify rules"; simply using NPO to penalize old behaviors leads to over-conservatism and over-refusal; influence function-based methods are hypersensitive to minor policy changes and are often closed-source.

**Key Challenge**: Policies are dynamic, but parameterized alignment is immutable—this creates an **Alignment-Reality Gap**. Existing methods are either "too costly (re-annotation)" or "inadequate tools (unlearning/NPO lacks positive signal)".

**Goal**: Without new human annotation, treat "policy update" as a **dataset reinterpretation** problem—given a new policy $\pi_{\text{new}}$ and an existing preference dataset, automatically decide how to use each data point (invert / suppress / retain), then surgically optimize the model towards the new policy without destroying general capabilities.

**Key Insight**: The authors introduce a simplified but practical "non-blind" assumption—we have access to the original preference dataset (though not to $\pi_{\text{old}}$ itself), thus avoiding the instability of blind settings that require sampling thousands of responses to infer the implicit policy.

**Core Idea**: Use a stronger proxy LLM as the oracle for $\pi_{\text{new}}$, triage each $(y_w, y_l)$ into three categories, and then apply a hybrid loss: "invert with IPO + suppress with NPO + retain with KL", combined with bi-level optimization for impact weighting to achieve fine-grained alignment.

## Method

### Overall Architecture
Starting from a model $\mathcal{M}_{\text{ref}}$ aligned to $\pi_{\text{old}}$ and the original preference data $\mathcal{D}$, given a new policy $\pi_{\text{new}}$ (a function returning compliant/non-compliant), TRACE proceeds in three stages: **Stage 1 Triage** uses a proxy LLM to assess the compliance of each $(x, y_w, y_l)$ under $\pi_{\text{new}}$, categorizing them into $\mathcal{D}_I$ (Invert), $\mathcal{D}_{II}$ (Punish), and $\mathcal{D}_R$ (Retain); **Stage 2 Hybrid Objectives** applies different losses to each category; **Stage 3 Alignment Impact Weighting** uses bi-level optimization to compute a weight $w_i$ for each sample, then optimizes the model $\mathcal{M}_\theta$ with weighted summation.

### Key Designs

1. **Triage: Using the new policy as oracle to split data into three categories**:

    - **Function**: Addresses the "False Dichotomy" error of naive realignment—cannot assume "$y_w$ is non-compliant implies $y_l$ is compliant", since $\pi_{\text{new}}$ may render both non-compliant.
    - **Mechanism**: The proxy LLM evaluates both $\pi_{\text{new}}(y_w|x)$ and $\pi_{\text{new}}(y_l|x)$, assigning each pair to one of three buckets: $\mathcal{D}_I$ (old winner non-compliant, old loser compliant, needs inversion), $\mathcal{D}_{II}$ (both non-compliant, needs suppression), $\mathcal{D}_R$ (old winner still compliant, retain). The theoretical case where both are compliant is merged into $\mathcal{D}_R$ as it provides no discriminative signal for optimization.
    - **Design Motivation**: The authors highlight that the Triage stage contributes most of the alignment gain—ablation shows removing Triage and using a uniform punitive loss drops Target Policy Agreement from 70.7% to 58.1%, a 12.6-point decrease.

2. **Hybrid Objectives: Tailored hybrid loss functions**:

    - **Function**: Different conflict types receive different optimization signals, maximizing data utility and avoiding over-refusal.
    - **Mechanism**: For $\mathcal{D}_I$, use inverted DPO/IPO loss $\mathcal{L}_I=-\log\sigma\big(\beta(\log\frac{p_\theta(y_l|x)}{p_{\text{ref}}(y_l|x)} - \log\frac{p_\theta(y_w|x)}{p_{\text{ref}}(y_w|x)})\big)$; for $\mathcal{D}_{II}$, default to NPO to suppress both $y_w$ and $y_l$, or optionally use oracle-LLM to generate a corrective response $y_c$ and apply DPO loss on $(y_c, y_w)$; for $\mathcal{D}_R$, use forward KL divergence $\mathcal{L}_{KL}=D_{KL}(\text{Logits}_{\mathcal{M}_{\text{ref}}} \| \text{Logits}_{\mathcal{M}_\theta})$ to anchor general capability.
    - **Design Motivation**: NPO provides only negative signals, risking the model becoming an "overly safe machine" that refuses everything; adding oracle correction to $\mathcal{D}_{II}$ teaches the model "what to say" rather than just "what not to say"; the KL term preserves the original distribution on the retain set, preventing catastrophic forgetting.

3. **Alignment Impact Weighting: Bi-level optimization for sample weighting**:

    - **Function**: Allocates scarce gradient budget to samples that truly drive policy compliance, filtering out updates orthogonal or even antagonistic to the global objective.
    - **Mechanism**: Inspired by U2A, the global objective $\mathcal{J}$ (e.g., $\pi_{\text{new}}$ compliance) has gradient $g_\mathcal{J}=\nabla_\theta \mathcal{J}(\theta_{\text{ref}})$ as the "gold standard direction". For each conflict sample, compute its task gradient $g_{\mathcal{L}_i}=\nabla_\theta \mathcal{L}_i(\theta_{\text{ref}})$, and define weight $w_i=\langle g_\mathcal{J}, g_{\mathcal{L}_i}\rangle$. The final objective is $\mathcal{L}_{\text{TRACE}}(\theta)=\sum_{i\in\mathcal{D}_I\cup\mathcal{D}_{II}} w_i \mathcal{L}_i(\theta) + \alpha_{KL}\sum_{j\in\mathcal{D}_R}\mathcal{L}_{KL}(\theta;j)$.
    - **Design Motivation**: This is a marginal gain approximation derived from the implicit function theorem (simplified to a dot product when $H_{\mathcal{L}_i}\approx \gamma I$), acting as a "gradient filter"—orthogonal samples get near-zero weight, antagonistic samples get negative weight, automatically avoiding harmful updates. Ablation shows removing impact weighting drops Target Policy Agreement by 7.4 points, with concurrent degradation on GPQA and HellaSwag.

### Loss & Training
The final objective $\mathcal{L}_{\text{TRACE}}$ is as above. $\beta$ is the DPO temperature, $\alpha_{KL}$ is the fixed coefficient for the KL term on the retain set. Training is validated on Qwen2.5-7B / Gemma-2-9B / Llama-3.1-8B backbones.

## Key Experimental Results

### Main Results (Pairwise Win Rate %, averaged over three backbones)

| Comparison | PKU-SafeRLHF | SynthValueBench | Annotation Consistency α |
|------------|--------------|-----------------|-------------------------|
| DPO-Gold vs TRACE | 68.2 | 74.6 | 0.80-0.82 |
| **TRACE vs U2A** | **81.8** | **85.3** | 0.75-0.79 |
| U2A vs TRACE | 18.2 | 14.7 | — |

TRACE significantly outperforms the U2A baseline (~82-85% win rate), and the gap with the "fully re-annotated gold standard" DPO-Gold is reasonable (DPO-Gold only beats TRACE at 68-75%, indicating TRACE bridges much of the gap between NPO-like methods and full re-annotation).

### Ablation & General Capability (PKU-SafeRLHF)

| Model | GPQA | MMLU | HellaSwag | GSM8K |
|-------|------|------|-----------|-------|
| Base (pre-alignment) | 31.6 | 70.6 | 81.4 | 70.4 |
| DPO-Gold (full re-annotation) | 32.1 | 70.5 | 81.3 | 70.8 |
| **TRACE (Ours)** | 30.1 | 70.2 | 78.2 | 70.6 |
| U2A (Baseline) | 29.5 | 70.2 | 80.8 | 69.9 |

| Ablation (Llama-3.1-8B) | Target Policy Agree. | ASR | MMLU |
|-------------------------|----------------------|-----|------|
| Full TRACE | 70.7 | 27.3 | ~70 |
| – Triage (uniform punitive) | 58.1 (-12.6) | — | — |
| – Impact Weighting | 62.8 (-7.9) | 32.1 (+4.8) | — |
| – KL on Retain | ~70 | — | ~64 (-6.1) |

### Key Findings
- **Triage is the main contributor**: Removing it results in a 12.6-point drop, indicating that "triaging data by new policy" itself provides the main signal—this suggests to the community that the bottleneck in realignment is not loss design, but data reinterpretation.
- **Impact weighting boosts performance and prevents degradation**: Removing it not only reduces alignment, but also increases ASR and degrades HellaSwag/GPQA, confirming its role in filtering gradient conflicts.
- **KL term is a utility anchor**: Removing it leaves alignment unchanged but drops MMLU by 6 points, showing its role is purely to "prevent forgetting old knowledge".
- **Helpfulness comes at a cost**: TRACE drops 3 points on HellaSwag compared to base; the authors candidly describe this as a "Helpfulness-Utility trade-off" rather than claiming lossless performance—this cost is acceptable in deployment scenarios where alignment is the primary goal.

## Highlights & Insights
- **Clear distinction between realignment and unlearning**: Methods like U2A assume a known forget set; TRACE provides an upstream solution for "how to derive the forget set from policy changes"—this reframing, though simple, is the true core contribution.
- **Three-class loss + weighted hybrid design is a reusable trick**: Applicable to any "policy-driven behavior modification" scenario—safety redirection, brand voice switching, regional compliance adaptation, not just RLHF.
- **Engineering pragmatism of the non-blind assumption**: Rather than pretending to solve blind realignment (which requires sampling thousands of responses to estimate implicit policy and is unstable in practice), the authors directly state "we have the original preference data"—this is entirely reasonable in industrial BPO pipelines. They openly acknowledge the information-theoretic ceiling of data-reuse realignment, rather than overselling it as a panacea.

## Limitations & Future Work
- The authors acknowledge a robustness gap with DPO-Gold (adversarial ASR, win rate), reflecting the information ceiling of data-reuse methods—when $\pi_{\text{new}}$ introduces dimensions not covered by the original data, nothing can be done without new data.
- Relies on the quality of proxy LLM judgments; biases in the proxy will propagate downstream. For certain value judgments (e.g., cultural, political), the neutrality of the proxy is questionable.
- Impact weighting uses isotropic Hessian approximation, which may be inaccurate in highly anisotropic loss landscapes.
- The drop in helpfulness on HellaSwag is real; in helpfulness-critical scenarios (creative writing, customer service), trade-offs need to be reconsidered.
- Future directions: make Triage continuous/fuzzy for non-binary preferences; introduce small-scale active labeling to bridge the information gap between new policy and old data.

## Related Work & Insights
- **vs DPO (Rafailov et al. 2023)**: DPO addresses initial alignment, assuming new full-scale human preference data; TRACE addresses post-deployment policy updates, recycling old data.
- **vs NPO (Zhang et al. 2024)**: NPO only suppresses bad responses, easily collapsing into over-conservatism; TRACE uses IPO for the Invert class to provide positive signals, and oracle correction for the Punish class to avoid this failure mode.
- **vs U2A (Feng et al. 2025)**: U2A proposes forget set weighting but assumes the forget set is known; TRACE fills the gap of "how to identify the forget set", serving as an upstream solution to U2A.
- **vs value evaluation benchmarks (ValueBench, WorldValuesBench)**: These only diagnose value drift; TRACE provides therapeutic intervention.

## Rating
- Novelty: ⭐⭐⭐⭐ The Triage stage is a clean new contribution; the hybrid loss and impact weighting are clever combinations of existing components.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three backbones × two datasets × human evaluation + adversarial testing + general capability + three ablations—very solid.
- Writing Quality: ⭐⭐⭐⭐⭐ Concepts like Realignment, Alignment-Reality Gap, and False Dichotomy are clearly framed; assumptions (non-blind) and trade-offs (HellaSwag drop) are explicitly stated.
- Value: ⭐⭐⭐⭐⭐ Directly addresses industrial deployment pain points, code is open source, highly practical, and immediately usable by LLM service providers.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] When Human Preferences Flip: An Instance-Dependent Robust Loss for RLHF](../../AAAI2026/llm_alignment/when_human_preferences_flip_an_instance-dependent_robust_loss_for_rlhf.md)
- [\[NeurIPS 2025\] Ask a Strong LLM Judge when Your Reward Model is Uncertain](../../NeurIPS2025/llm_alignment/ask_a_strong_llm_judge_when_your_reward_model_is_uncertain.md)
- [\[ICLR 2026\] Uni-DPO: A Unified Paradigm for Dynamic Preference Optimization of LLMs](../../ICLR2026/llm_alignment/uni-dpo_a_unified_paradigm_for_dynamic_preference_optimization_of_llms.md)
- [\[AAAI 2026\] Enhancing Uncertainty Estimation in LLMs with Expectation of Aggregated Internal States](../../AAAI2026/llm_alignment/enhancing_uncertainty_estimation_in_llms_with_expectation_of_aggregated_internal.md)
- [\[CVPR 2026\] MoD-DPO: Towards Mitigating Cross-modal Hallucinations in Omni LLMs using Modality Decoupled Preference Optimization](../../CVPR2026/llm_alignment/mod-dpo_towards_mitigating_cross-modal_hallucinations_in_omni_llms_using_modalit.md)

</div>

<!-- RELATED:END -->
