---
title: >-
  [Paper Note] Preference Heads in Large Language Models: A Mechanistic Framework for Interpretable Personalization
description: >-
  [ACL2026][Interpretability][Mechanistic Interpretability] This paper proposes Preference Heads and Differential Preference Steering (DPS), using causal ablation to identify a small number of attention heads carrying user…
tags:
  - "ACL2026"
  - "Interpretability"
  - "Mechanistic Interpretability"
  - "Attention Heads"
  - "Personalized Generation"
  - "Contrastive Decoding"
  - "User Preference"
date: 2026-05-08
content_hash: 8fb088ba22ce6d6c
---

# Preference Heads in Large Language Models: A Mechanistic Framework for Interpretable Personalization

**Conference**: ACL2026  
**arXiv**: [2604.22345](https://arxiv.org/abs/2604.22345)  
**Code**: https://github.com/weixuzhang/DPS  
**Area**: Interpretability / Personalized Generation / LLM  
**Keywords**: Mechanistic Interpretability, Attention Heads, Personalized Generation, Contrastive Decoding, User Preference

## TL;DR
This paper proposes Preference Heads and Differential Preference Steering (DPS), using causal ablation to identify a small number of attention heads carrying user preferences. It then amplifies these preference signals during decoding to enhance personalized generation and prediction without modifying model parameters.

## Background & Motivation
**Background**: LLMs have demonstrated strong implicit personalization capabilities. Given user history, profiles, or brief preference descriptions, models often align with the user in terms of tone, topic selection, headline style, and recommendation reasoning. Existing personalized LLM methods primarily follow three paths: incorporating user information into prompts or RAG contexts, performing fine-tuning or preference learning on user data, or altering output distributions via contrastive methods during decoding.

**Limitations of Prior Work**: Most of these methods treat the model as a black box. While they successfully align outputs with user preferences, they struggle to answer a mechanistic question: where exactly are user preferences represented within the model? Do all layers contribute uniformly, or do specific modules play critical roles? Relying solely on final text metrics makes interpretability analysis difficult and complicates diagnosis when personalization fails—whether due to inaccurate user profiles, poor retrieval, or inactive internal preference pathways.

**Key Challenge**: Personalization requires enhancing user-related signals, but LLM logits simultaneously mix general language capabilities, task constraints, and user preference signals. Pure prompting or fine-tuning can change overall behavior but fails to distinguish which internal components transmit preferences. If preference signals are indeed sparse and localized, global interventions remain non-transparent and may introduce redundant noise.

**Goal**: The authors aim to deconstruct personalization from a black-box behavior into a locatable mechanical problem: first identify which attention heads contribute causally to user-aligned outputs, and then utilize these heads to perform controllable preference steering during inference while maintaining content relevance and generation fluency.

**Key Insight**: The paper draws on mechanistic interpretability research concerning induction heads, retrieval heads, and factuality heads. Since certain transformer behaviors can be explained by a few specialized attention heads, user style and thematic preferences might also be localized in similar "Preference Heads." The key is not just correlation or activation strength, but validation through causal ablation: checking if the likelihood of a user reference output decreases after removing a specific head.

**Core Idea**: Identify Preference Heads using causal head ablation, then leverage the differential signal—the original model logits minus the generic logits from the model with masked Preference Heads—to reinforce the user preference direction during decoding.

## Method
The methodology consists of three stages: defining and discovering Preference Heads; performing Differential Preference Steering using these heads; and scaling to heterogeneous preferences via user clustering and weighted routing.

### Overall Architecture
The input consists of user-conditioned data, including task input $x$, user profile or history $u$, and reference output $y^*$. The authors first perform an offline traversal of the attention heads, applying targeted masking to each head and measuring the change in negative log-likelihood (NLL) of the user reference output. Heads that cause the greatest degradation when masked are identified as Preference Heads.

During inference, DPS performs two forward passes on the same context: one through the original model to obtain preference-conditioned logits, and another through the model with masked Preference Heads to obtain generic logits. The difference between the two is amplified to produce the final decoding logits. The intuition is that this difference primarily represents the personalization signal contributed by the preference heads.

To handle diverse user populations, the paper encodes user profiles into embeddings for clustering. Preference Heads are rediscovered within each cluster. During inference, either hard or soft routing based on user-to-cluster similarity is used to avoid roughly merging inconsistent preferences into a single global set.

### Key Designs
1.  **Preference Contribution Score**:
    - **Function**: Assigns a causal contribution score to each attention head, measuring its actual impact on user-aligned outputs.
    - **Mechanism**: Masking the $k$-th head at layer $l$ ($h_{l,k}$) and comparing the average NLL of the masked model $M_{\theta \setminus h_{l,k}}$ versus the original model $M_\theta$ on user reference outputs. The score is defined as $PCS(h_{l,k}) = E[L(M_{\theta \setminus h_{l,k}}, x, u, y^*) - L(M_\theta, x, u, y^*)]$. A high positive PCS indicates that removing the head significantly lowers the probability of user-aligned outputs, suggesting a causal role in personalization.
    - **Design Motivation**: Interpretability analyses often stop at activation visualizations or correlation statistics, which do not prove a component's necessity. PCS directly evaluates head utility via intervention, tying it closely to the objective of personalized output likelihood.

2.  **Differential Preference Steering**:
    - **Function**: Translates discovered Preference Heads into a decoding-time steering signal without updating model parameters.
    - **Mechanism**: Separately calculates original model logits $l_t^{pref}$ and generic logits $l_t^{gen}$ (with masked Preference Heads), then combines them as $\tilde{l}_t = (1 + \gamma) l_t^{pref} - \gamma l_t^{gen}$. For $\gamma = 0$, it reduces to the original model; as $\gamma$ increases, the model emphasizes the preference directions present in the original model relative to the generic one.
    - **Design Motivation**: Forcing an output style can damage content consistency. DPS amplifies differences already expressed by the model's internal pathways, acting as an enhancement of internal mechanisms rather than an external control constraint.

3.  **Cluster-aware Preference Steering**:
    - **Function**: Addresses the issue that Preference Heads are not shared across all users, making head discovery more stable.
    - **Mechanism**: User profile embeddings are clustered using k-means. PCS discovery is run within each cluster to find cluster-specific head sets. During inference, users are either hard-assigned to the nearest cluster or soft-routed using similarity weights.
    - **Design Motivation**: Jaccard overlap analysis shows low overlap between top-$K$ Preference Heads for different users. Averaging across all users dilutes signals; sharing heads among similar users balances individualization and statistical stability.

### Loss & Training
DPS does not involve training model parameters or additional fine-tuning. The offline phase uses reference outputs to calculate NLL changes for head selection. The inference phase requires an extra forward pass with masked Preference Heads to control generation intensity via differential logits. The paper analyzes $K$ and routing: small $K$ misses signals while large $K$ introduces noise; hard routing suits classification, while soft routing is more stable for generation.

## Key Experimental Results

### Main Results
Evaluation was conducted using LaMP benchmarks on LLaMA-3-8B-Instruct, Qwen2-7B-Instruct, and Mistral-7B-Instruct. Tasks included news and academic headline generation, tweet paraphrasing, citation identification, movie tagging, and product rating. Metrics used were ROUGE-1/L and METEOR for generation, Accuracy/F1 for classification, and MAE/RMSE for regression. Baselines included CAD, DeCoRe, and DoLa.

| Model | Method | News Headline R-1 / R-L / METEOR | Academic Headline R-1 / R-L / METEOR | Tweet Paraphrasing R-1 / R-L / METEOR |
|------|------|------------------------------|-------------------------------|-------------------------------|
| LLaMA-3-8B | CAD | 0.1681 / 0.1498 / 0.1568 | 0.3530 / 0.3068 / 0.3925 | 0.3368 / 0.2893 / 0.2813 |
| LLaMA-3-8B | DeCoRe | 0.1768 / 0.1572 / 0.1626 | 0.4010 / 0.3527 / 0.4004 | 0.3231 / 0.2764 / 0.2729 |
| LLaMA-3-8B | DoLa | 0.1694 / 0.1508 / 0.1592 | 0.3636 / 0.3117 / 0.4079 | 0.3365 / 0.2877 / 0.2795 |
| LLaMA-3-8B | DPS | 0.1787 / 0.1596 / 0.1650 | 0.3243 / 0.2787 / 0.3826 | 0.3389 / 0.2898 / 0.2884 |
| Qwen2-7B | CAD | 0.1580 / 0.1392 / 0.1255 | 0.4197 / 0.3780 / 0.4381 | 0.3590 / 0.3106 / 0.3384 |
| Qwen2-7B | DeCoRe | 0.1581 / 0.1305 / 0.1232 | 0.4311 / 0.3729 / 0.4565 | 0.3470 / 0.3065 / 0.3173 |
| Qwen2-7B | DoLa | 0.1642 / 0.1473 / 0.1272 | 0.4277 / 0.3746 / 0.4596 | 0.3524 / 0.3046 / 0.3246 |
| Qwen2-7B | DPS | 0.1627 / 0.1450 / 0.1318 | 0.4071 / 0.3421 / 0.4230 | 0.3533 / 0.2981 / 0.3269 |
| Mistral-7B | CAD | 0.1361 / 0.1132 / 0.0980 | 0.4375 / 0.3712 / 0.4561 | 0.3342 / 0.2916 / 0.3097 |
| Mistral-7B | DeCoRe | 0.1299 / 0.1085 / 0.0908 | 0.4135 / 0.3648 / 0.4419 | 0.3407 / 0.2927 / 0.2978 |
| Mistral-7B | DoLa | 0.1362 / 0.1136 / 0.0962 | 0.4364 / 0.3733 / 0.4605 | 0.3291 / 0.2852 / 0.3026 |
| Mistral-7B | DPS | 0.1536 / 0.1366 / 0.1399 | 0.3983 / 0.3350 / 0.4162 | 0.3441 / 0.2998 / 0.2990 |

DPS is notably stable in news and tweet tasks, particularly with Mistral-7B. While DeCoRe or DoLa sometimes perform better on academic headlines, DPS shows superior cross-task consistency for maintaining user style in short text generation.

| Model | Method | Citation ID Acc / F1 | Movie Tagging Acc / F1 | Product Rating MAE / RMSE |
|------|------|-------------------|-------------------|---------------------|
| LLaMA-3-8B | CAD | 0.6240 / 0.6070 | 0.4552 / 0.3839 | 0.4426 / 0.9300 |
| LLaMA-3-8B | DeCoRe | 0.6232 / 0.6200 | 0.4639 / 0.4034 | 0.4442 / 0.9458 |
| LLaMA-3-8B | DoLa | 0.6156 / 0.5961 | 0.2800 / 0.1643 | 0.4200 / 0.8718 |
| LLaMA-3-8B | DPS | 0.6356 / 0.6288 | 0.4610 / 0.3910 | 0.4236 / 0.9278 |
| Qwen2-7B | CAD | 0.6230 / 0.6250 | 0.1850 / 0.1181 | 0.3180 / 0.6240 |
| Qwen2-7B | DeCoRe | 0.5400 / 0.5891 | 0.2320 / 0.1217 | 0.3250 / 0.6325 |
| Qwen2-7B | DoLa | 0.6790 / 0.6795 | 0.2412 / 0.0958 | 0.3200 / 0.6300 |
| Qwen2-7B | DPS | 0.6932 / 0.7078 | 0.3902 / 0.3202 | 0.3276 / 0.6719 |

DPS significantly outperforms baselines in movie tagging for Qwen2-7B. For regression tasks like product ratings, performance is mixed, suggesting that amplifying preference signals does not always minimize numerical error.

### Ablation Study

| Analysis Item | Result | Note |
|--------|------|------|
| Preference Head Sparsity | High PCS heads cluster locally in heatmaps | Personalization is driven by sparse internal components |
| Cross-user Overlap | Jaccard overlap of top-K heads is near 0 | Distinct preference pathways support cluster-aware design |
| Random Head Control | Replacing Preference Heads with random ones causes degradation | Gains come from causally meaningful components |
| K-value Sensitivity | Performance saturates as K increases; excess K introduces noise | Preference signals are concentrated in limited head sets |
| Routing Strategy | Hard routing helps classification; Soft routing stabilizes generation | Discrete tasks benefit from specialization; generation from smooth mixing |

Inference overhead analysis shows that while DPS requires two forward passes during decoding, prefill is shared, resulting in a low 1.02x overhead for 2048 token prompts.

| Prompt Length | Standard Decoding TFlop | DPS TFlop | Relative Overhead |
|-------------|----------------|-----------|----------|
| 512 | 6.57 | 6.96 | 1.06x |
| 1024 | 13.04 | 13.43 | 1.03x |
| 2048 | 26.80 | 27.21 | 1.02x |

### Key Findings
- **Sparsity**: Preference Heads are sparse and user-specific, supporting the idea that personalization can be explained via local circuits.
- **Low Overlap**: Different users rely on different heads, justifying cluster-aware routing.
- **Efficiency**: Overhead is minimal for long-context tasks as prefill is shared.
- **Human Evaluation**: GPT-5 and human judges prefer DPS outputs for style and alignment, even when automatic metrics show smaller gains.

## Highlights & Insights
- The transition from "external condition control" to "internal component localization" is a major highlight, enabling researchers to identify specific heads carrying preferences.
- PCS provides a clean causal evaluation via interventions rather than just correlation or weight magnitudes.
- DPS bridges mechanistic interpretability and decoding-time steering.
- Cluster-aware logic captures the essence of personalization: user preferences are not a global property but require hierarchical modeling.

## Limitations & Future Work
- **Access Requirements**: Requires internal access to heads/activations, making it inapplicable to black-box APIs.
- **Inference Latency**: Double forward passes might be a burden for real-time, low-latency applications.
- **Offline Cost**: Updating Preference Heads for millions of users or evolving profiles remains an efficiency challenge.
- **Bias**: Explicitly amplifying preferences may reinforce biases present in user history; safety constraints are needed.

## Related Work & Insights
- **vs Prompting/RAG**: While others provide context externally, Ours investigates internal absorption of signals.
- **vs Fine-tuning**: Ours offers zero-shot control of internal pathways without parameter updates.
- **vs CAD/DoLa**: Unlike generic contrastive methods, DPS is task-mechanism driven by identifying specific personalization heads.
- **Future Directions**: PCS can potentially discover "Safety Heads," "Politeness Heads," or "Terminological Heads" for fine-grained behavioral control.

## Rating
- **Novelty**: ⭐⭐⭐⭐☆ Introduces mechanistic interpretability to personalization with a well-defined discovery-steering link.
- **Experimental Thoroughness**: ⭐⭐⭐⭐☆ Extensive multi-model and multi-task evaluation, including human and efficiency analyses.
- **Writing Quality**: ⭐⭐⭐⭐☆ Logic is clear, and the PCS-to-DPS chain is well-supported by data.
- **Value**: ⭐⭐⭐⭐☆ A strong reference for interpretable personalization and lightweight inference-time control.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Embracing Anisotropy: Turning Massive Activations into Interpretable Control Knobs for Large Language Models](embracing_anisotropy_turning_massive_activations_into_interpretable_control_knob.md)
- [\[ACL 2026\] From Interpretability to Performance: Optimizing Retrieval Heads for Long-Context Language Models](from_interpretability_to_performance_optimizing_retrieval_heads_for_long-context.md)
- [\[ACL 2026\] FineSteer: A Unified Framework for Fine-Grained Inference-Time Steering in Large Language Models](finesteer_a_unified_framework_for_fine-grained_inference-time_steering_in_large_.md)
- [\[ACL 2026\] Retrieval Heads are Dynamic](retrieval_heads_are_dynamic.md)
- [\[ICML 2026\] Towards Atoms of Large Language Models](../../ICML2026/interpretability/towards_atoms_of_large_language_models.md)

</div>

<!-- RELATED:END -->
