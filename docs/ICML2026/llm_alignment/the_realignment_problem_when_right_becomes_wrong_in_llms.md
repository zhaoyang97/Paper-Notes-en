---
title: >-
  [Paper Note] The Realignment Problem: When Right becomes Wrong in LLMs
description: >-
  [ICML 2026][LLM Alignment][realignment] This paper formalizes the problem of "what to do when policies change post-deployment" as the Realignment problem and proposes the TRACE framework. TRACE uses a stronger proxy mode…
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
content_hash: 1b86171d0e6c88ce
---

# The Realignment Problem: When Right becomes Wrong in LLMs

**Conference**: ICML 2026  
**arXiv**: [2511.02623](https://arxiv.org/abs/2511.02623)  
**Code**: Available (Released as mentioned in the paper)  
**Area**: LLM Alignment / Preference Learning / Policy Realignment  
**Keywords**: realignment, alignment-reality gap, triage, IPO, NPO, bi-level optimization

## TL;DR
This paper formalizes the problem of "what to do when policies change post-deployment" as the Realignment problem and proposes the TRACE framework. TRACE uses a stronger proxy model to classify existing preference pairs into three categories (Invert / Punish / Retain) and performs surgical realignment using a hybrid IPO+NPO+KL objective, keeping pace with policy drift without requiring new rounds of human annotation.

## Background & Motivation

**Background**: In industrial LLM deployment, alignment primarily relies on RLHF / DPO, where a model $\mathcal{M}_\theta$ is trained on a binary preference dataset $\mathcal{D}=\{(x, y_w, y_l)\}$ generated from a BPO annotation pipeline. This alignment is guideline-dependent: once data is internalized into parameters, the original policy guidelines become neither visible nor modifiable.

**Limitations of Prior Work**: Regulations (EU AI Act, NIST RMF), cultures, and organizational risk appetites are constantly evolving; a compliant behavior yesterday might be a violation today. Redoing full-scale human annotation is prohibitively expensive; machine unlearning can only delete but not "modify rules"; simply using NPO to punish old behaviors leads to over-conservatism and over-refusal; influence function-based methods are hypersensitive to minor policy changes and difficult to implement for closed-source models.

**Key Challenge**: Policies are dynamic, while parametric alignment is immutable — creating an **Alignment-Reality Gap**. Existing methods are either "too costly (re-annotation)" or "the wrong tool (unlearning/NPO lacks positive signals)."

**Goal**: Without new human annotation, this work aims to treat "policy updates" as a **dataset re-interpretation** problem. Given a new policy $\pi_{\text{new}}$ and an existing preference dataset, the system automatically decides how to use each data point (Invert / Punish / Retain) and employs surgical optimization to shift the model toward the new policy without destroying general capabilities.

**Key Insight**: The authors introduce a simplified yet practical "non-blind" assumption — having access to the original preference dataset (though $\pi_{\text{old}}$ itself remains unknown). This avoids the instability of sampling thousands of responses to infer an implicit policy under blind settings.

**Core Idea**: Use a stronger proxy LLM as an oracle for $\pi_{\text{new}}$ to categorize each $(y_w, y_l)$ pair into three types, then use a refined alignment strategy involving a hybrid loss (Invert with IPO + Punish with NPO + Retain with KL) combined with impact weighting via bi-level optimization.

## Method

### Overall Architecture
Starting with a model $\mathcal{M}_{\text{ref}}$ aligned to $\pi_{\text{old}}$ and the original preference data $\mathcal{D}$, and given a new policy $\pi_{\text{new}}$ (a function returning compliant/non-compliant), TRACE follows three stages: **Stage 1 Triage** uses a proxy LLM to evaluate the compliance of each $(x, y_w, y_l)$ under $\pi_{\text{new}}$, assigning them to $\mathcal{D}_I$ (Invert), $\mathcal{D}_{II}$ (Punish), or $\mathcal{D}_R$ (Retain); **Stage 2 Hybrid Objectives** applies specific losses to each category; **Stage 3 Alignment Impact Weighting** derives weights $w_i$ for each sample via bi-level optimization for a weighted sum optimization of $\mathcal{M}_\theta$.

### Key Designs

1.  **Triage: Tri-partitioning Data Using New Policy as Oracle**:
    - **Function**: Solves the "False Dichotomy" error made by naive realignment — one cannot assume that "if $y_w$ is non-compliant, $y_l$ must be compliant," as $\pi_{\text{new}}$ could easily render both non-compliant.
    - **Mechanism**: The proxy LLM evaluates both $\pi_{\text{new}}(y_w|x)$ and $\pi_{\text{new}}(y_l|x)$ simultaneously. Combinations fall into three buckets: $\mathcal{D}_I$ (old winner non-compliant, old loser compliant, requires inversion), $\mathcal{D}_{II}$ (both non-compliant, requires suppression), and $\mathcal{D}_R$ (old winner remains compliant, retain). A theoretical fourth case where both are compliant is merged into $\mathcal{D}_R$ as it provides no discriminative signal for optimization.
    - **Design Motivation**: The authors highlight that the Triage stage contributes the most significant alignment gains. Removing Triage and using a uniform punitive approach on all data drops the Target Policy Agreement from 70.7% to 58.1%, a 12.6 percentage point difference.

2.  **Hybrid Objectives: Targeted Hybrid Losses**:
    - **Function**: Applies different optimization signals for different conflict types to avoid wasting data or inducing over-refusal.
    - **Mechanism**: For $\mathcal{D}_I$, an inverted DPO/IPO loss is used: $\mathcal{L}_I=-\log\sigma\big(\beta(\log\frac{p_\theta(y_l|x)}{p_{\text{ref}}(y_l|x)} - \log\frac{p_\theta(y_w|x)}{p_{\text{ref}}(y_w|x)})\big)$. For $\mathcal{D}_{II}$, NPO is used by default to suppress both $y_w$ and $y_l$; optionally, an oracle-LLM can generate a corrective response $y_c$ to use a DPO loss on $(y_c, y_w)$. For $\mathcal{D}_R$, forward KL divergence $\mathcal{L}_{KL}=D_{KL}(\text{Logits}_{\mathcal{M}_{\text{ref}}} \| \text{Logits}_{\mathcal{M}_\theta})$ anchors general capabilities.
    - **Design Motivation**: Providing only negative signals (NPO) can turn models into "safe machines that answer nothing." Adding oracle corrections for $\mathcal{D}_{II}$ allows the model to learn "what to say in this situation" rather than just "what not to say." The KL term preserves the original distribution on the retain set to prevent catastrophic forgetting.

3.  **Alignment Impact Weighting: Bi-level Optimization Weights**:
    - **Function**: Ensures the limited gradient budget is spent on samples that truly drive policy compliance, filtering out local updates that are orthogonal or conflicting with global goals.
    - **Mechanism**: Based on the U2A concept, the gradient of the global objective $\mathcal{J}$ (e.g., $\pi_{\text{new}}$ compliance), $g_\mathcal{J}=\nabla_\theta \mathcal{J}(\theta_{\text{ref}})$, is used as the "gold standard direction." For each conflicting sample, its task gradient $g_{\mathcal{L}_i}=\nabla_\theta \mathcal{L}_i(\theta_{\text{ref}})$ is calculated to define weight $w_i=\langle g_\mathcal{J}, g_{\mathcal{L}_i}\rangle$. The final objective is $\mathcal{L}_{\text{TRACE}}(\theta)=\sum_{i\in\mathcal{D}_I\cup\ mechanism \mathcal{D}_{II}} w_i \mathcal{L}_i(\theta) + \alpha_{KL}\sum_{j\in\mathcal{D}_R}\mathcal{L}_{KL}(\theta;j)$.
    - **Design Motivation**: This is an approximation of marginal gain derived via the implicit function theorem (simplified to a dot product assuming $H_{\mathcal{L}_i}\approx \gamma I$). It acts as a "gradient filter" — weighting orthogonal samples near 0 and negative samples negatively, automatically avoiding harmful updates. Ablations show that removing impact weighting causes a 7.4 point drop in Target Policy Agreement along with degradation in GPQA and HellaSwag.

### Loss & Training
The final objective $\mathcal{L}_{\text{TRACE}}$ is provided above. $\beta$ is the DPO temperature, and $\alpha_{KL}$ is the fixed coefficient for the KL term on the retain set. Training was validated on three backbones: Qwen2.5-7B, Gemma-2-9B, and Llama-3.1-8B.

## Key Experimental Results

### Main Results (Pairwise Win Rate %, Average of 3 Backbones)

| Comparison | PKU-SafeRLHF | SynthValueBench | Annotation Consistency α |
|------|--------------|-----------------|--------------|
| DPO-Gold vs TRACE | 68.2 | 74.6 | 0.80-0.82 |
| **TRACE vs U2A** | **81.8** | **85.3** | 0.75-0.79 |
| U2A vs TRACE | 18.2 | 14.7 | — |

TRACE significantly outperforms the U2A baseline (~82-85% win rate). The gap between TRACE and the "fully re-annotated gold standard" (DPO-Gold) is reasonable (DPO-Gold wins against TRACE by only 68-75%), indicating that TRACE bridges a large portion of the gap between NPO-style methods and full re-annotation.

### Ablation Study & General Capabilities (PKU-SafeRLHF)

| Model | GPQA | MMLU | HellaSwag | GSM8K |
|------|------|------|-----------|-------|
| Base (Pre-alignment) | 31.6 | 70.6 | 81.4 | 70.4 |
| DPO-Gold (Full Re-annotation) | 32.1 | 70.5 | 81.3 | 70.8 |
| **TRACE (Ours)** | 30.1 | 70.2 | 78.2 | 70.6 |
| U2A (Baseline) | 29.5 | 70.2 | 80.8 | 69.9 |

| Ablation (Llama-3.1-8B) | Target Policy Agree. | ASR | MMLU |
|----------------------|----------------------|-----|------|
| Full TRACE | 70.7 | 27.3 | ~70 |
| – Triage (Uniform punitive) | 58.1 (-12.6) | — | — |
| – Impact Weighting | 62.8 (-7.9) | 32.1 (+4.8) | — |
| – KL on Retain | ~70 | — | ~64 (-6.1) |

### Key Findings
- **Triage is the primary contributor**: Removing it causes a -12.6 point drop, indicating that "tri-partitioning data according to the new policy" provides the majority of the signal. This suggests that the bottleneck of realignment lies in data re-interpretation rather than loss design.
- **Impact weighting improves performance and prevents degradation**: Removing it not only hurts alignment but also increases ASR and degrades HellaSwag/GPQA, confirming its role in filtering gradient conflicts.
- **KL term acts as the utility anchor**: Removing it doesn't affect alignment but causes a 6-point drop in MMLU, showing its role is purely to "prevent forgetting old knowledge while learning new."
- **Cost for Helpfulness**: TRACE drops 3 points on HellaSwag compared to the base. The authors candidly describe this as a "Helpfulness-Utility trade-off" rather than claiming it as lossless. This cost is considered acceptable in deployment scenarios where alignment is the priority.

## Highlights & Insights
- **Clear separation of realignment from unlearning**: Methods like U2A assume a forget set is given; TRACE provides the upstream solution for "how to derive a forget set from policy changes." This reframe is a fundamental contribution.
- **Reusable trick of tri-type loss + weighting**: This design can be applied to any "policy-driven behavior modification" scenario, such as safety redirection, brand voice switching, or regional compliance adaptation, beyond just RLHF.
- **Pragmatic "non-blind" assumption**: Instead of pretending to solve blind realignment (which is unstable due to sampling thousands of responses), the authors directly assume access to the original preference data. This is perfectly reasonable in industrial BPO pipelines. The authors acknowledge the information-theoretic ceiling of data-reuse realignment rather than claiming it as a panacea.

## Limitations & Future Work
- A robustness gap (Adversarial ASR, win rate) remains between TRACE and DPO-Gold, reflecting the information limit of data-reuse — when $\pi_{\text{new}}$ introduces dimensions entirely absent in the original data, new data is inevitable.
- Dependence on proxy LLM judgment quality; proxy biases may propagate downstream. Proxy neutrality in subjective value judgments (e.g., culture, politics) may be questionable.
- Impact weighting uses an isotropic Hessian approximation, which may be inaccurate under high loss landscape anisotropy.
- The decline in helpfulness on HellaSwag is real; re-weighting is necessary for helpfulness-critical scenarios like creative writing or customer service.
- Future work: Developing continuous/fuzzy triage for non-binary preferences and introducing small-scale active labeling for information gaps between new policies and old data.

## Related Work & Insights
- **vs DPO (Rafailov et al. 2023)**: DPO handles initial alignment assuming new full human preference data; TRACE handles post-deployment policy updates by recycling old data.
- **vs NPO (Zhang et al. 2024)**: NPO only suppresses bad responses, risking a collapse into over-conservatism; TRACE's Invert category provides positive signals via IPO, and the Punish category is enhanced by oracle correction to avoid this failure mode.
- **vs U2A (Feng et al. 2025)**: U2A proposes forget set weighting but assumes the forget set is known; TRACE fills the gap of "how to identify the forget set," acting as an upstream component to U2A.
- **vs value evaluation benchmarks (ValueBench, WorldValuesBench)**: These only diagnose value drift; TRACE provides a therapeutic intervention.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The Triage stage is a clean new contribution; the mixed loss and impact weighting are clever combinations of existing components.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 3 backbones × 2 datasets × Human eval + Adversarial tests + General capabilities + 3 types of ablations; very solid.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear framing of concepts like Realignment, Alignment-Reality Gap, and False Dichotomy. Assumptions (non-blind) and costs (HellaSwag drop) are explicitly stated.
- **Value**: ⭐⭐⭐⭐⭐ Directly addresses industrial deployment pain points with open-source code; high deployability for LLM service providers.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] When Human Preferences Flip: An Instance-Dependent Robust Loss for RLHF](../../AAAI2026/llm_alignment/when_human_preferences_flip_an_instance-dependent_robust_loss_for_rlhf.md)
- [\[ICML 2026\] PICACO: Pluralistic In-Context Value Alignment of LLMs via Total Correlation Optimization](picaco_pluralistic_in-context_value_alignment_of_llms_via_total_correlation_opti.md)
- [\[NeurIPS 2025\] Ask a Strong LLM Judge when Your Reward Model is Uncertain](../../NeurIPS2025/llm_alignment/ask_a_strong_llm_judge_when_your_reward_model_is_uncertain.md)
- [\[ACL 2026\] WildFeedback: Aligning LLMs With In-situ User Interactions And Feedback](../../ACL2026/llm_alignment/wildfeedback_aligning_llms_with_in-situ_user_interactions_and_feedback.md)
- [\[ACL 2026\] PERSA: Reinforcement Learning for Professor-Style Personalized Feedback with LLMs](../../ACL2026/llm_alignment/persa_reinforcement_learning_for_professor-style_personalized_feedback_with_llms.md)

</div>

<!-- RELATED:END -->
