---
title: >-
  [Paper Note] On the Rejection Criterion for Proxy-Based Test-Time Alignment
description: >-
  [ACL2026][LLM Alignment][test-time alignment] This paper unifies proxy-based test-time alignment methods, such as implicit rewards, Nudging, and KAD, into a "sample-then-decide" probabilistic graphical model. It proposes a **conservative confidence bet** that uses the best confidence of a small alignment model as a reference, improving hybrid decoding accuracy across multiple mathematical and commonsense reasoning datasets.
tags:
  - "ACL2026"
  - "LLM Alignment"
  - "test-time alignment"
  - "proxy models"
  - "rejection criterion"
  - "guided decoding"
  - "LLM inference"
date: 2026-05-08
content_hash: d6e088c5ed709f81
---

# On the Rejection Criterion for Proxy-Based Test-Time Alignment

**Conference**: ACL2026  
**arXiv**: [2604.16146](https://arxiv.org/abs/2604.16146)  
**Code**: https://github.com/ayoubhammal/knapsack-approximation-deferral  
**Area**: LLM Alignment / Test-time Alignment  
**Keywords**: test-time alignment, proxy models, rejection criterion, guided decoding, LLM inference

## TL;DR
This paper unifies proxy-based test-time alignment methods, such as implicit rewards, Nudging, and KAD, into a "sample-then-decide" probabilistic graphical model. It proposes a **conservative confidence bet** that uses the best confidence of a small alignment model as a reference, improving hybrid decoding accuracy across multiple mathematical and commonsense reasoning datasets.

## Background & Motivation
**Background**: LLM alignment typically depends on training phases such as SFT, RLHF, DPO, and RLVR to shift the output distribution of base models toward human preferences, specific formats, or reasoning requirements. While training-based alignment is effective, its cost rises rapidly with model size; re-aligning a massive target model without full post-training resources is often impractical.

**Limitations of Prior Work**: Test-time alignment seeks to directly modify the base LLM’s distribution during the generation phase to avoid re-training. Explicit reward-guided decoding can filter candidates token-by-token but often requires additional forward passes for every candidate or handles only local rewards with limited expressiveness. Methods based on full-answer reranking or MCMC require sampling many outputs, which is slow.

**Key Challenge**: Proxy-based test-time alignment uses a smaller aligned model $q^\ast$ to guide a larger base model $p$, appearing as an effective compromise: it preserves the large model's capabilities while borrowing the small model's alignment preferences. However, existing methods lack a unified understanding of "when to trust the large model and when to defer to the small one." Implicit reward methods use $q^\ast/q$ as an alignment signal to warp $p$; Nudging and KAD perform deferral based on $p$'s confidence. Despite different surface mechanisms, they all revolve around how to define the rejection criterion for large model samples.

**Goal**: The authors first aim to provide a unified probabilistic interpretation of these methods, demonstrating that they are not disjoint tricks but different parameterizations of the same rejection-based generation process. Second, the paper points out that the intuition of "rejecting upon low confidence" is not robust, as multiple equally correct tokens in natural language often share probability mass. Finally, the authors aim to design a more conservative rejection criterion: switching generation control from the large base model only when the small aligned model is likely to provide a stronger alternative.

**Key Insight**: The paper identifies the critical action in proxy methods: the large model first provides a draft token, and the small model takes over only if the draft is rejected. Therefore, rather than viewing different methods as reward shaping, guided decoding, or cascades, they are reformulated as a single probabilistic graphical model with a latent rejection variable. Consequently, methodological differences are localized to the rejection distribution $\pi(r=1\mid \bar{x}=v)$.

**Core Idea**: Unify proxy-based test-time alignment using a "rejection criterion" and shift the criterion from solely observing the large model's internal confidence to comparing the probability of the large model's draft token $p_v$ against the probability of the small aligned model's strongest candidate $\max_w q^\ast_w$.

## Method

### Overall Architecture
The paper does not propose a new training pipeline but reformulates the probabilistic structure of test-time decoding: in the absence of a large aligned model $p^\ast$, it utilizes information from a small aligned model $q^\ast$ to improve the large base model $p$. For each generated token, a draft token $w$ is first sampled from $p$ as a candidate; a rejection variable $r$ then determines its fate—if $r=0$, $w$ is accepted; if $r=1$, $w$ is rejected, and a new token is sampled from $q^\ast$. The final output distribution is thus decomposed into two parts: "retaining the large model sample" and "taking over by the small model after rejection." Most of the internal structure is fixed (latent draft distribution $\pi(\bar{x}=w)=p_w$, fallback distribution as $q^\ast$), leaving the rejection probability $\mu_v=\pi(r=1\mid\bar{x}=v)$—the "rejection criterion"—as the primary design component. In this formulation, many seemingly different test-time alignment methods are condensed into a single problem: how to set $\mu_v$ to preserve large model capabilities while switching to the small model when it is more reliable.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400, 'subGraphTitleMargin': {'top': 8, 'bottom': 16}}}}%%
flowchart TD
    IN["Input: Next-token distributions of large base model p and small aligned model q*"]
    DRAFT["Rejection-based Probabilistic Graphical Model<br/>Sample draft token w from p, then sample rejection variable r"]
    subgraph BET["Conservative confidence bet (Rejection criterion μ_v)"]
        direction TB
        CMP["Compare draft probability p_v with q* strongest candidate max q*_w"]
        DEC["Decision: p_v < max q*_w − λ ?"]
        CMP --> DEC
    end
    IN --> DRAFT --> BET
    DEC -->|No: draft strong enough, r=0| ACC["Accept token w (sampled from p)"]
    DEC -->|Yes: fallback stronger, r=1| REJ["Reject w, resample from q*"]
    ACC --> OUT["Output token x"]
    REJ --> OUT
```

### Key Designs

**1. Rejection-based Probabilistic Graphical Model: Decomposing proxy alignment into "drafting" and "rejection"**

Existing methods are often obscured by implementation details. The authors unify them using a generative model with a latent draft token and rejection variable: first sample $\bar{x}=w$ from $p$, then sample rejection variable $r$; if $r=0$, the final token is $w$, and if $r=1$, sampling occurs from $q^\ast$. The overall output distribution is summarized as $\pi(x=v)=p_v(1-\mu_v)+q^\ast_v\sum_w p_w\mu_w$.  
This framework explicitly separates the use of a small aligned model to guide a large base model into two interpretable steps—proposing a candidate and defining when to reject it—allowing Nudging, KAD, and implicit rewards to be compared directly.

**2. Reducing existing proxy methods to different rejection criteria**

Within the unified framework, prior methods are merely different choices for $\mu_v$. Nudging is a distribution-level criterion: when $\max_w p_w<\lambda$, the entire step is handed over to $q^\ast$, rejecting the whole distribution of $p$ rather than a specific token. Dual KAD is a token-level criterion: the sampled token is rejected if its probability $p_v<\lambda$. Implicit reward methods construct $s_v=p_v(q^\ast_v/q_v)/Z$; Proposition 1 proves that under enclosing conditions, this can also be generated by a specific rejection distribution.  
This reduction exposes a common weakness: both Nudging and KAD rely only on the absolute confidence of $p$ without asking if the fallback $q^\ast$ can do better. Implicit reward requires access to both the base and aligned versions of the small model, which increases deployment overhead.

**3. Conservative confidence bet: Rejection depends on whether the fallback has a stronger candidate**

The new criterion shifts the decision from "whether the large model is confident" to "whether the small aligned model has a more confident alternative." For token $v$ sampled from $p$, it compares $p_v$ with the probability of the most confident token in the small aligned model $\max_w q^\ast_w$: if $p_v<\max_w q^\ast_w-\lambda$, it indicates that $q^\ast$ has at least one candidate more reliable than the current draft, triggering rejection; otherwise, it is kept. A larger margin $\lambda$ makes the policy more conservative and less likely to trigger deferral.  
This design accounts for the fact that "low confidence" does not equate to being "wrong." For example, when "frameworks like Pytorch" and "frameworks such as Pytorch" are both valid, probability mass is split between "like" and "such," resulting in low individual token probabilities. By introducing the best candidate of $q^\ast$ as a baseline, rejection occurs only when the fallback truly offers a stronger option, avoiding excessive switching caused by natural language ambiguity.

### Loss & Training
The paper does not train new models or introduce additional loss functions. It focuses on decoding-time distribution combination strategies: at each generation step, the next-token distributions of $p$ and $q^\ast$ are computed, and the final token source is determined by the rejection criterion. Experiments use a temperature of 0.7 to isolate the gains of the decoding rule itself; the margin $\lambda$ is selected from $\{0, 0.1, 0.2\}$ on a small development subset. The authors emphasize that performance remains competitive even at $\lambda=0$.

## Key Experimental Results

### Main Results
Experiments follow the setup of Hammal et al. (2026), evaluating three mathematical reasoning datasets (GSM8K, MATH500, SVAMP) and two commonsense reasoning datasets (ARC-Challenge, CommonsenseQA). Model families include OLMo 2 and Qwen 3, with a proxy setting consisting of a small aligned model and a large base model for each. The metric is accuracy after answer extraction.

| Model Family | Method | GSM8K | MATH | SVAMP | ARC | CSQA | Avg. |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| OLMo 2 | Large base model $p$ | 54.5 | 9.4 | 57.6 | 29.6 | 19.4 | 34.1 |
| OLMo 2 | Small aligned model $q^\ast$ | 62.5 | 16.4 | 70.3 | 43.8 | 48.4 | 48.2 |
| OLMo 2 | Implicit reward | 58.4 | 18.2 | 73.0 | 63.3 | 55.8 | 53.7 |
| OLMo 2 | Dual KAD, $\lambda=0.4$ | 72.3 | 23.4 | 75.3 | 61.9 | 55.6 | 57.7 |
| OLMo 2 | Confidence bet, $\lambda=0.2$ | 71.7 | 26.4 | 79.0 | 62.6 | 54.9 | 58.9 |
| OLMo 2 | Target large aligned $p^\ast$ | 84.3 | 39.6 | 87.6 | 82.5 | 76.9 | 74.1 |
| Qwen 3 | Large base model $p$ | 75.5 | 51.8 | 80.0 | 86.6 | 76.9 | 74.1 |
| Qwen 3 | Small aligned model $q^\ast$ | 75.3 | 53.0 | 86.6 | 82.9 | 68.7 | 73.3 |
| Qwen 3 | Implicit reward | 80.7 | 60.6 | 89.0 | 88.9 | 78.1 | 79.4 |
| Qwen 3 | Dual KAD, $\lambda=0.4$ | 81.7 | 60.6 | 87.3 | 91.5 | 80.7 | 80.3 |
| Qwen 3 | Confidence bet, $\lambda=0.2$ | 82.1 | 61.6 | 89.3 | 90.5 | 79.3 | 80.5 |
| Qwen 3 | Target large aligned $p^\ast$ | 82.4 | 64.0 | 88.3 | 93.8 | 83.1 | 82.3 |

For OLMo 2, the confidence bet's average accuracy reaches 58.9, outperforming dual KAD (57.7) and implicit reward (53.7); MATH accuracy specifically improves from 23.4 (dual KAD) to 26.4, showing the new criterion prevents incorrect deferral on hard math tasks. For Qwen 3, the confidence bet average of 80.5 is very close to dual KAD (80.3) and slightly leads implicit reward (79.4); it performs better on math tasks but not consistently on commonsense tasks.

### Ablation Study
The paper analyzes the sensitivity of the margin $\lambda$ as a proxy for ablation. Average accuracy is listed below to show how conservatism impacts performance.

| Configuration | OLMo 2 Avg. | Qwen 3 Avg. | Description |
| :--- | ---: | ---: | :--- |
| Confidence bet, $\lambda=0$ | 56.0 | 78.4 | Most aggressive; switches whenever $p_v$ is lower than the best $q^\ast$ candidate |
| Confidence bet, $\lambda=0.1$ | 58.4 | 79.9 | Moderate margin; more stable than $\lambda=0$ for both families |
| Confidence bet, $\lambda=0.2$ | 58.9 | 80.5 | Best average setting; indicates conservative rejection reduces unnecessary deferral |
| Nudging, $\lambda=0.4$ | 50.2 | 78.7 | Distribution-level threshold; significantly weaker than token-level rules on OLMo 2 |
| Dual KAD, $\lambda=0.4$ | 57.7 | 80.3 | Strong baseline; looks at $p_v$ but without comparing to $q^\ast$'s best candidate |

### Key Findings
- The most stable gains occur in OLMo 2: the gap between base model $p$ and target aligned model $p^\ast$ is 37.4 accuracy points, indicating a large alignment gap where proxy signals are more valuable.
- Gains on Qwen 3 are smaller because the base model and aligned model are already close: $p$ averages 71.4 while $p^\ast$ averages 80.2. This compressed gap limits the improvement room for any deferral method.
- $\lambda=0.2$ is the best average across both families, suggesting "being conservative" is reasonable: instead of switching upon any low probability, a margin for $p$ prevents over-rejection caused by ambiguity.
- The new criterion is particularly attractive for mathematical reasoning. Qwen 3's MATH score reaches 61.6 (higher than dual KAD's 60.6), and OLMo 2's MATH reaches 26.4 (higher than dual KAD's 23.4).
- It does not win across all commonsense tasks. On Qwen 3's ARC/CSQA, dual KAD scores 91.5/80.7 compared to 90.5/79.3 for the confidence bet, indicating that the calibration quality of the small model's baseline affects decisions across task types.

## Highlights & Insights
- The primary highlight is the **unified perspective**. Rather than treating Nudging, KAD, and implicit reward as isolated approaches, the paper identifies them as instances of a rejection distribution, allowing subsequent designs to focus on $\mu_v$.
- The **conservative confidence bet** is clever because it stops asking "Is the large model confident?" and asks "Does the small model have a more confident alternative?" This is closer to the essence of deferral, which is a choice between two generation sources.
- The critique of **linguistic ambiguity** is insightful. It is normal for multiple tokens to be correct in natural language; a drop in single-token probability doesn't necessarily mean a model error. Misinterpreting this as uncertainty leads to frequent switching to smaller models, losing the large model's capabilities.
- These ideas can migrate to other **cascade or speculative decoding** scenarios. In any system where a primary model proposes candidates and an auxiliary model can take over, the decision can shift from fixed thresholds to relative comparisons between sources.
- The paper serves as a reminder that **proxy models are not natural experts**. Traditional reject-option intuitions assume fallback errors are negligible, but $q^\ast$ may be weaker than $p$. Rejection criteria must explicitly consider fallback quality.

## Limitations & Future Work
- The margin $\lambda$ still needs selection on a development set. While $\lambda=0$ is competitive, the best results require tuning, which might vary with model family, task, and temperature.
- Experiments focus on math and commonsense QA with accuracy as the metric. Whether this criterion works stably for open-ended writing, safety refusal, or multi-turn dialogue requires further validation.
- The method requires viewing both $p$ and $q^\ast$ distributions at each step, making inference overhead higher than a single model. Analysis of throughput, latency, and VRAM usage is limited.
- Confidence bet depends on **probability calibration**. If $q^\ast$’s maximum probability is consistently over or underestimated, the baseline $\max_w q^\ast_w$ will mislead the deferral. Future work could consider temperature calibration or adaptive margins.
- The criterion currently only compares single-token confidence, without modelling long-term rewards. Tokens with lower current probability might lead to better overall reasoning chains; future work could extend the criterion to short horizons or value estimates.

## Related Work & Insights
- **vs Implicit reward / proxy tuning**: Implicit reward uses $q^\ast/q$ to extract the alignment offset and multiplies it with $p$. This paper explains this as a special case of rejection sampling while removing the requirement to access the small base model $q$.
- **vs Nudging**: Nudging performs distribution-level deferral based on $\max_w p_w$. This paper argues this confuses ambiguity with uncertainty and shifts to token-level decisions with a relative proxy reference.
- **vs Dual KAD**: Dual KAD moved from distribution-level to token-level decisions, but still uses absolute thresholds like $p_v<\lambda$. This work improves it by making the threshold relative to $q^\ast$'s current best candidate, aligning better with the question of whether a proxy is worth using.
- **vs Cascade / speculative decoding**: Related cascade methods perform handover decisions. The insight for NLP alignment is that handover rules should utilize relative evidence between models rather than treating the small model as a perfect fallback expert.

## Rating
- Novelty: ⭐⭐⭐⭐ The unified PGM perspective is very clear, and the new criterion captures the relative nature of deferral; it is simple in form but targets the right problem.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers two model families and five reasoning datasets with margin analysis; however, tasks are biased toward accuracy, lacking open-ended alignment and efficiency analysis.
- Writing Quality: ⭐⭐⭐⭐ Concise and direct; intuition matches the formulas well. Table 1 has high information density, though failure cases and throughput are not explored in depth.
- Value: ⭐⭐⭐⭐ Highly relevant for researchers in test-time alignment, model cascading, and guided decoding, providing a solid framework for designing rejection/deferral rules.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Test-Time Alignment for Large Language Models via Textual Model Predictive Control](../../ICLR2026/llm_alignment/test-time_alignment_for_large_language_models_via_textual_model_predictive_contr.md)
- [\[ICLR 2026\] GuardAlign: Test-time Safety Alignment in Multimodal Large Language Models](../../ICLR2026/llm_alignment/guardalign_test-time_safety_alignment_in_multimodal_large_language_models.md)
- [\[CVPR 2025\] Jailbreaking the Non-Transferable Barrier via Test-Time Data Disguising](../../CVPR2025/llm_alignment/jailbreaking_the_non-transferable_barrier_via_test-time_data_disguising.md)
- [\[ACL 2026\] To Intervene or Not: Guiding Inference-time Alignment with Probabilistic Model Blending](to_intervene_or_not_guiding_inference-time_alignment_with_probabilistic_model_bl.md)
- [\[ACL 2026\] Pref-CTRL: Preference Driven LLM Alignment using Representation Editing](pref-ctrl_preference_driven_llm_alignment_using_representation_editing.md)

</div>

<!-- RELATED:END -->
