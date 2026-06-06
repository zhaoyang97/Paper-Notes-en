---
title: >-
  [Paper Note] On the Rejection Criterion for Proxy-Based Test-Time Alignment
description: >-
  [ACL2026][LLM Alignment][Test-time alignment] This paper unifies proxy-based test-time alignment methods such as implicit reward, Nudging…
tags:
  - "ACL2026"
  - "LLM Alignment"
  - "Test-time alignment"
  - "proxy models"
  - "rejection criterion"
  - "guided decoding"
  - "LLM inference"
date: 2026-05-08
content_hash: ebf8541b7ca1822f
---

# On the Rejection Criterion for Proxy-Based Test-Time Alignment

**Conference**: ACL2026  
**arXiv**: [2604.16146](https://arxiv.org/abs/2604.16146)  
**Code**: https://github.com/ayoubhammal/knapsack-approximation-deferral  
**Area**: LLM Alignment / Test-time Alignment  
**Keywords**: Test-time alignment, proxy models, rejection criterion, guided decoding, LLM inference

## TL;DR
This paper unifies proxy-based test-time alignment methods such as implicit reward, Nudging, and KAD into a probabilistic graphical model centered on "sampling then deciding whether to reject." It proposes a "conservative confidence bet" that uses the best confidence of a small aligned model as a reference, improving hybrid decoding accuracy across several mathematical and commonsense reasoning datasets.

## Background & Motivation
**Background**: Large Language Model (LLM) alignment typically relies on training phases such as SFT, RLHF, DPO, or RLVR to shift the output distribution of the base model toward human preferences, task formats, or reasoning requirements. While effective, the cost of training-based alignment scales rapidly with model size; re-aligning becomes impractical when the target model is already massive but lacks full post-training resources.

**Limitations of Prior Work**: Test-time alignment attempts to modify the distribution of the base LLM during the generation phase to avoid retraining. Explicit reward-guided decoding can filter candidates token-by-token but often requires extra forward passes for each candidate or handles only local rewards with limited expressiveness. Methods based on full-answer reranking or MCMC require sampling many outputs, which is slow.

**Key Challenge**: Proxy-based test-time alignment uses a smaller aligned model $q^\ast$ to guide a larger base model $p$, offering a middle-ground solution that retains the large model's capabilities while leveraging the small model's alignment preferences. However, existing methods lack a unified understanding of when to trust the large model versus the small model. Implicit reward methods treat $q^\ast/q$ as an alignment signal to warp $p$; Nudging and KAD defer based on the confidence of $p$. Despite different surface mechanisms, they all revolve around one question: how to define the rejection criterion for large model samples.

**Goal**: The authors aim to provide a shared probabilistic interpretation for these methods, showing they are different parameterizations of the same rejection-based generation process. Furthermore, they point out that the intuition of "rejecting upon low confidence" is not robust, as natural language often has multiple equally correct tokens sharing probability mass. Finally, the authors aim to design a more conservative rejection criterion: only shifting generation authority from the base model when the small aligned model is likely to provide a better choice.

**Key Insight**: The paper identifies the critical action in proxy methods: the large model first provides a draft token, and the small model takes over only if the draft is rejected. Therefore, rather than viewing different methods as reward shaping, guided decoding, or cascades, it is better to formulate them as a probabilistic graphical model with a latent rejection variable. Consequently, research differences focus on the rejection distribution $\pi(r=1\mid \bar{x}=v)$.

**Core Idea**: Use a "rejection criterion" to unify proxy-based test-time alignment and change the criterion from solely looking at the base model's self-confidence to comparing the probability of the draft token $p_v$ against the probability of the best candidate from the small aligned model $\max_w q^\ast_w$.

## Method
The method does not propose a new training pipeline but rewrites the probabilistic structure of test-time decoding. The authors place the large base model $p$, the small base model $q$, and the small aligned model $q^\ast$ into the same framework, aiming to improve the generation of $p$ using information from $q^\ast$ without access to a large aligned model $p^\ast$.

### Overall Architecture
The overall pipeline consists of three steps:

First, a draft token $w$ is sampled from the large base model $p$. This token is a candidate rather than the final answer.

Second, the rejection distribution determines whether to keep this draft. A binary variable $r$ is introduced: if $r=0$, $w$ is accepted; if $r=1$, $w$ is rejected, triggering the small aligned model.

Third, the final token is generated. If the draft is accepted, the final token is $w$. If rejected, a new token is sampled from the small aligned model $q^\ast$. Thus, the total output distribution can be written as the sum of a "retained large model sample" part and a "fallback to small model after rejection" part.

In this framework, most components are fixed: the latent draft distribution is $\pi(\bar{x}=w)=p_w$, and the fallback distribution after rejection is $q^\ast$. The only designable part is the rejection probability for each token $\mu_v=\pi(r=1\mid \bar{x}=v)$, which is the rejection criterion.

This formulation condenses various test-time alignment methods into a single problem: given the next-token distributions of $p$ and $q^\ast$, how should $\mu_v$ be set to leverage the large model's capabilities while deferring when the small aligned model is more reliable?

### Key Designs
1.  **Rejection-based Probabilistic Graphical Model**:
    - **Function**: Expresses proxy-based test-time alignment as a generative model with a latent draft token and a rejection variable.
    - **Mechanism**: The model samples $\bar{x}=w$ from $p$, then samples the rejection variable $r$. If $r=0$, the final token copies $w$. If $r=1$, the final token is sampled from $q^\ast$. The final distribution is $\pi(x=v)=p_v(1-\mu_v)+q^\ast_v\sum_w p_w\mu_w$.
    - **Design Motivation**: Implementation details often obscure the essence of existing methods. This PGM decomposes "guiding a large base model with a small aligned model" into two interpretable steps: proposing candidates and defining rejection. This allows Nudging, KAD, and implicit reward methods to be compared under a single framework.

2.  **Reducing Existing Proxy Methods to Different Rejection Criteria**:
    - **Function**: Explains the correspondence between Nudging, dual KAD, implicit reward, and the PGM.
    - **Mechanism**: Nudging uses a distribution-level criterion: if $\max_w p_w < \lambda$, the entire step is handed over to $q^\ast$. Dual KAD uses a token-level criterion: reject if the sampled token probability $p_v < \lambda$. Implicit reward constructs $s_v=p_v(q^\ast_v/q_v)/Z$; Proposition 1 shows that under enclosing conditions, this can also be generated by a rejection distribution.
    - **Design Motivation**: This reduction explains why older methods share common weaknesses. Nudging and KAD rely primarily on the absolute confidence of $p$ without asking if the fallback model $q^\ast$ can do better. Implicit reward utilizes $q^\ast/q$ but requires access to both the base and aligned versions of the small model, increasing deployment requirements.

3.  **Conservative Confidence Bet Rejection Criterion**:
    - **Function**: Decides at the token level whether to transfer generation authority from the large model $p$ to the small aligned model $q^\ast$.
    - **Mechanism**: For a token $v$ sampled from $p$, it compares $p_v$ with the probability of the most certain token in the small aligned model $\max_w q^\ast_w$. If $p_v < \max_w q^\ast_w - \lambda$, it indicates the small model has a candidate significantly more confident than the current draft, leading to rejection. Otherwise, the draft is kept. Here, $\lambda$ is a margin; larger values are more conservative.
    - **Design Motivation**: The authors argue that "low confidence" is not equivalent to being "wrong." For example, when "frameworks like Pytorch" and "frameworks such as Pytorch" are both valid, probability mass is split between "like" and "such," resulting in low individual token probabilities. The new criterion uses the best candidate of $q^\ast$ as a baseline, rejecting only when the fallback has a clearly stronger choice, thereby reducing over-switching caused by natural language ambiguity.

### Loss & Training
The paper does not train new models or introduce extra loss functions. It focuses on the distribution combination strategy during decoding.

In practice, both $p$ and $q^\ast$ next-token distributions are required at each generation step. The final token source is determined by the rejection criterion. Experiments use a temperature of 0.7 to isolate the gains of the decoding rules. The margin $\lambda$ is selected on a small dev set, with final tests on $\{0, 0.1, 0.2\}$. The authors emphasize that performance remains competitive even with $\lambda=0$.

## Key Experimental Results

### Main Results
Following the setup of Hammal et al. (2026), the evaluation uses three mathematical reasoning datasets (GSM8K, MATH500, SVAMP) and two commonsense reasoning datasets (ARC-Challenge, CommonsenseQA). Model families include OLMo 2 and Qwen 3, with each proxy setting consisting of one small aligned model and one large base model. The metric is accuracy after answer extraction.

| Model Family | Method | GSM8K | MATH | SVAMP | ARC | CSQA | Avg. |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| OLMo 2 | Large Base Model $p$ | 54.5 | 9.4 | 57.6 | 29.6 | 19.4 | 34.1 |
| OLMo 2 | Small Aligned Model $q^\ast$ | 62.5 | 16.4 | 70.3 | 43.8 | 48.4 | 48.2 |
| OLMo 2 | Implicit reward | 58.4 | 18.2 | 73.0 | 63.3 | 55.8 | 53.7 |
| OLMo 2 | Dual KAD, $\lambda=0.4$ | 72.3 | 23.4 | 75.3 | 61.9 | 55.6 | 57.7 |
| OLMo 2 | Confidence bet, $\lambda=0.2$ | 71.7 | 26.4 | 79.0 | 62.6 | 54.9 | 58.9 |
| OLMo 2 | Target Large Aligned $p^\ast$ | 84.3 | 39.6 | 87.6 | 82.5 | 76.9 | 74.1 |
| Qwen 3 | Large Base Model $p$ | 75.5 | 51.8 | 80.0 | 86.6 | 76.9 | 74.1 |
| Qwen 3 | Small Aligned Model $q^\ast$ | 75.3 | 53.0 | 86.6 | 82.9 | 68.7 | 73.3 |
| Qwen 3 | Implicit reward | 80.7 | 60.6 | 89.0 | 88.9 | 78.1 | 79.4 |
| Qwen 3 | Dual KAD, $\lambda=0.4$ | 81.7 | 60.6 | 87.3 | 91.5 | 80.7 | 80.3 |
| Qwen 3 | Confidence bet, $\lambda=0.2$ | 82.1 | 61.6 | 89.3 | 90.5 | 79.3 | 80.5 |
| Qwen 3 | Target Large Aligned $p^\ast$ | 82.4 | 64.0 | 88.3 | 93.8 | 83.1 | 82.3 |

For OLMo 2, "confidence bet" reaches an average accuracy of 58.9, higher than "dual KAD" (57.7) and "implicit reward" (53.7). Notably, MATH accuracy increases from 23.4 (dual KAD) to 26.4, showing the new criterion avoids erroneous deferral in difficult math tasks. For Qwen 3, "confidence bet" (80.5) is very close to "dual KAD" (80.3) and slightly higher than "implicit reward" (79.4), showing strength in math while not always leading in commonsense.

### Ablation Study
The paper lacks a module-deletion ablation study as the method is itself a rejection criterion; the closest equivalent is a sensitivity analysis of the margin $\lambda$.

| Configuration | OLMo 2 Avg. | Qwen 3 Avg. | Description |
| :--- | :---: | :---: | :--- |
| Confidence bet, $\lambda=0$ | 56.0 | 78.4 | Most aggressive; switches whenever $p_v$ is lower than $q^\ast$'s best. |
| Confidence bet, $\lambda=0.1$ | 58.4 | 79.9 | Medium margin; more stable than $\lambda=0$ for both models. |
| Confidence bet, $\lambda=0.2$ | 58.9 | 80.5 | Best average setting; shows conservative rejection reduces unnecessary deferral. |
| Nudging, $\lambda=0.4$ | 50.2 | 78.7 | Distribution-level threshold; significantly weaker than token-level rules on OLMo 2. |
| Dual KAD, $\lambda=0.4$ | 57.7 | 80.3 | Strong baseline; considers only $p_v$ without explicit comparison to $q^\ast$. |

### Key Findings
- The most stable gains come from OLMo 2, where the base model $p$ has a 37.4 percentage point gap behind the large aligned model $p^\ast$, indicating a significant alignment deficit and higher value for the small model's proxy signal.
- Qwen 3 shows smaller gains because the base and target aligned models are already close (average $p=71.4$ vs $p^\ast=80.2$).
- $\lambda=0.2$ is the best average setting across both model families, suggesting that being "a bit conservative" is reasonable to avoid over-rejection due to linguistic ambiguity.
- The new criterion is particularly attractive for mathematical reasoning.
- It does not win across all commonsense tasks, suggesting that the quality of the small model's confidence baseline calibration still impacts decisions across different task types.

## Highlights & Insights
- The primary highlight is the "unified perspective." The paper doesn't treat Nudging, KAD, and implicit reward as isolated paths but identifies them as different instances of a rejection distribution.
- The cleverness of "conservative confidence bet" lies in asking "does the small aligned model have a better alternative?" rather than just "is the large model confident?" This is closer to the essence of deferral—choosing the more trustworthy source.
- The critique of linguistic ambiguity is insightful. Token probability drops don't always mean an error. Over-reacting to this uncertainty can sacrifice the superior capabilities of the large model.
- These ideas are transferable to other cascade or speculative decoding scenarios by replacing fixed thresholds with relative comparisons between the primary and fallback sources.
- The paper reminds us that proxy models are not natural experts. Traditional reject options assume small models have negligible error, but $q^\ast$ can be weaker than $p$. Rejection criteria must explicitly account for fallback quality.

## Limitations & Future Work
- The margin $\lambda$ still needs selection on a development set.
- Experiments focus on math and commonsense QA (accuracy). Performance on open-ended writing, safety, and multi-turn dialogue needs further verification.
- Running both $p$ and $q^\ast$ distributions at each step increases inference overhead. Analysis of throughput and memory relative to accuracy gains is limited.
- Confidence bet depends on probability calibration. If $q^\ast$ is consistently over- or under-confident, the $\max_w q^\ast_w$ baseline will be misleading.
- The current criterion only compares single-token confidence without modeling long-term returns. Future work could extend the rejection criterion to short horizons or value estimates.

## Related Work & Insights
- **vs Implicit reward / proxy tuning**: Implicit reward uses $q^\ast/q$ to extract the offset between small models and applies it to $p$. This paper explains such distribution transformations as a special case of rejection-based sampling while removing the requirement to access the small base version $q$.
- **vs Nudging**: Nudging performs distribution-level deferral based on $\max_w p_w$. This paper argues this confuses ambiguity with uncertainty, moving instead to token-level relative comparisons with $q^\ast$.
- **vs Dual KAD**: Dual KAD already uses token-level decisions but relies on absolute thresholds like $p_v < \lambda$. The advantage of this work is that the threshold is not determined by $p$ alone but by a comparison with the best candidate of $q^\ast$.
- **vs Cascade / speculative decoding**: Inspired by cascade methods, this work suggests that handover rules should use relative evidence between models rather than treating the small model as an infallible expert fallback.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The unified PGM perspective is clear, and the new criterion captures the relative nature of deferral well.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers two model families and five datasets with margin analysis. Lacks open-ended alignment evaluation and efficiency analysis.
- **Writing Quality**: ⭐⭐⭐⭐ Direct and concise, with good mapping between formulas and intuition.
- **Value**: ⭐⭐⭐⭐ Highly relevant for those working on test-time alignment, model cascades, and guided decoding.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] GuardAlign: Test-time Safety Alignment in Multimodal Large Language Models](../../ICLR2026/llm_alignment/guardalign_test-time_safety_alignment_in_multimodal_large_language_models.md)
- [\[ACL 2026\] Pref-CTRL: Preference Driven LLM Alignment using Representation Editing](pref-ctrl_preference_driven_llm_alignment_using_representation_editing.md)
- [\[ACL 2026\] Debiasing Reward Models via Causally Motivated Inference-Time Intervention](debiasing_reward_models_via_causally_motivated_inference-time_intervention.md)
- [\[NeurIPS 2025\] Inference-time Alignment in Continuous Space](../../NeurIPS2025/llm_alignment/inference-time_alignment_in_continuous_space.md)
- [\[AAAI 2026\] W2S-AlignTree: Weak-to-Strong Inference-Time Alignment for Large Language Models via Monte Carlo Tree Search](../../AAAI2026/llm_alignment/w2s-aligntree_weak-to-strong_inference-time_alignment_for_large_language_models_.md)

</div>

<!-- RELATED:END -->
