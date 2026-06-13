---
title: >-
  [Paper Note] IF-GUIDE: Influence Function-Guided Detoxification of LLMs
description: >-
  [NeurIPS 2025][Social Computing][LLM detoxification] This paper proposes IF-Guide, which leverages influence functions to identify toxic content in training data at the token granularity and actively suppresses the model…
tags:
  - "NeurIPS 2025"
  - "Social Computing"
  - "LLM detoxification"
  - "influence functions"
  - "training data attribution"
  - "token-level suppression"
  - "proactive safety"
date: 2026-05-08
content_hash: e465b9129c69854d
---

# IF-GUIDE: Influence Function-Guided Detoxification of LLMs

**Conference**: NeurIPS 2025
**arXiv**: [2506.01790](https://arxiv.org/abs/2506.01790)  
**Code**: [GitHub](https://github.com/ztcoalson/IF-Guide)  
**Area**: Social Computing
**Keywords**: LLM detoxification, influence functions, training data attribution, token-level suppression, proactive safety

## TL;DR

This paper proposes IF-Guide, which leverages influence functions to identify toxic content in training data at the token granularity and actively suppresses the model from learning toxic behaviors during pre-training or fine-tuning via a penalty-based training objective, substantially outperforming passive alignment methods such as DPO and RAD.

## Background & Motivation

Current LLM detoxification predominantly follows a passive "learn-then-fix" paradigm: models are first pre-trained on large-scale corpora that may contain toxic content, then corrected post-hoc via alignment methods such as RLHF or DPO. This approach suffers from several critical issues:

**Dependence on human preference annotations**: RLHF/DPO require large volumes of high-quality human preference data, which is costly to annotate and difficult to scale.

**Fundamentally reactive**: Alignment strategies suppress toxic outputs rather than preventing the model from acquiring toxic knowledge; under adversarial attacks, suppressed toxic associations may resurface.

**Coarse data filtering**: Existing keyword filtering or heuristic approaches fail to capture context-dependent implicit toxicity and may inadvertently remove benign content.

This paper fundamentally reframes the problem: **Can toxic content be identified and its influence suppressed during training itself?** This constitutes a proactive safety approach that addresses the problem from the perspective of training data attribution.

## Method

### Overall Architecture

IF-Guide proceeds in three stages: (1) computing token-level toxicity attribution scores using an improved influence function; (2) fine-grained selection of toxic training tokens; and (3) suppressing the model from learning those tokens via a penalty-based training objective.

### Key Design 1: Differential Influence Function Attribution

The standard influence function approximates the effect of a training sample on model outputs via the inverse Hessian:

$$\mathcal{I}_\theta(x_i, q) = -\nabla_\theta[\log \mathbf{Pr}(c|p;\theta)]^\top \mathbf{H}^{-1} \nabla_\theta \mathcal{L}(x_i;\theta)$$

However, directly applying standard influence functions to identify toxic training data is ineffective (removing 50% of high-influence data reduces toxicity by only 33% while severely degrading fluency). The reason is that high-influence documents frequently contain common benign tokens such as "the," which interfere with toxicity attribution.

To address this, the paper introduces **differential attribution**: toxic query set $Q_{\text{tox}}$ and safe query set $Q_{\text{safe}}$ are sampled simultaneously, and their difference is computed:

$$\Delta\mathcal{I}_\theta(x_i) = \mathcal{I}_\theta(x_i, Q_{\text{tox}}) - \mathcal{I}_\theta(x_i, Q_{\text{safe}}) \approx -(\bar{g}_{\text{tox}} - \bar{g}_{\text{safe}})^\top \tilde{\mathbf{H}}^{-1} \nabla_\theta \mathcal{L}(x_i;\theta)$$

This filters out general-purpose tokens that exhibit high influence on both toxic and safe queries, precisely localizing training content specific to toxicity.

### Key Design 2: Token-Level Attribution

Training documents in modern LLMs typically contain thousands of tokens; even when a document includes a small amount of toxic content, the majority remains benign. Assigning a single influence score to an entire document leads to: (1) missing documents with sparse toxicity; and (2) treating benign portions as toxic.

The paper decomposes document-level attribution into token-level scores:

$$\mathcal{S}_{ij} = -(\bar{g}_{\text{tox}} - \bar{g}_{\text{safe}})^\top \tilde{\mathbf{H}}^{-1} \nabla_\theta \mathcal{L}(x_{ij};\theta)$$

where $\mathcal{L}(x_{ij};\theta) = -\log \mathbf{Pr}(x_{ij}|x_{i,<j};\theta)$ is the loss for an individual token.

### Key Design 3: High-Fidelity Toxic Token Selection

1. **Document importance ranking**: For each document, the count of tokens exceeding threshold $\tau_{\text{tox}}$ (99th percentile) and the sum of their scores are computed; the harmonic mean of the two normalized quantities is used as the document rank, prioritizing toxicity-dense documents.
2. **Context expansion**: Each toxic token is expanded by a window of $w=1$, incorporating neighboring context into the suppression scope.
3. **Budget control**: Toxic tokens are selected in descending document rank order, with the total capped at 2% of all training tokens.

### Loss & Training

For training sample $x_i$ with toxic token index set $T_i$, the final training objective is:

$$\mathcal{L}_{\text{tox}}(x_i, T_i;\theta) = -\sum_{j \notin T_i} \log \mathbf{Pr}(x_{ij}|x_{i,<j};\theta) + \lambda \sum_{j \in T_i} \log \mathbf{Pr}(x_{ij}|x_{i,<j};\theta)$$

The first term trains the model normally on benign tokens (standard cross-entropy); the second term penalizes the model's predicted probability on toxic tokens (the sign flip causes high probability to incur a higher penalty). The default setting is $\lambda=1$, which can be tuned to control the toxicity–fluency trade-off.

### Computational Efficiency

- EK-FAC is used to approximate the inverse Hessian, avoiding $O(d^3)$ direct computation.
- Gradient batching and half-precision arithmetic achieve approximately $2.5\times$ speedup.
- A small proxy model (e.g., Pythia-160M) can substitute the target model for influence score computation, reducing parameter count by $7.5\times$ and requiring only 7.5 hours (vs. 145 hours originally), yielding an overall speedup of up to $19\times$.

## Key Experimental Results

### Main Results: RealToxicityPrompts Detoxification

| Model | Method | EMT(Full)↓ | TP(Full)↓ | EMT(Toxic)↓ | TP(Toxic)↓ | PPL↓ | Acc↑ |
|------|------|-----------|----------|------------|-----------|------|------|
| Pythia-160M | None | 0.557 | 0.560 | 0.764 | 0.801 | 25.84 | 0.450 |
| Pythia-160M | DPO | 0.348 | 0.330 | 0.517 | 0.525 | 26.47 | 0.474 |
| Pythia-160M | RAD | 0.118 | 0.094 | 0.202 | 0.176 | – | 0.457 |
| Pythia-160M | **IF-Guide** | **0.101** | **0.054** | **0.136** | **0.085** | 26.77 | 0.433 |
| Pythia-160M | IF-Guide+RAD | **0.031** | **0.017** | **0.047** | **0.030** | – | 0.438 |
| Pythia-1B | None | 0.585 | 0.591 | 0.811 | 0.848 | 18.74 | 0.509 |
| Pythia-1B | DPO | 0.437 | 0.433 | 0.660 | 0.692 | 19.14 | 0.544 |
| Pythia-1B | RAD | 0.162 | 0.138 | 0.275 | 0.254 | – | 0.522 |
| Pythia-1B | **IF-Guide** | **0.118** | **0.065** | **0.160** | **0.101** | 22.22 | 0.464 |
| Llama-3.2-1B | IF-Guide | **0.127** | **0.085** | **0.172** | **0.131** | 23.01 | 0.445 |
| Llama-3.2-1B | IF-Guide+RAD | **0.042** | **0.028** | **0.063** | **0.046** | – | 0.449 |

IF-Guide achieves $4.2$–$5.5\times$ EMT reduction and $6.8$–$10.4\times$ TP reduction across all models; combined with RAD, it yields $14$–$18\times$ EMT and $21$–$33\times$ TP reduction.

### Implicit Toxicity Experiment (ToxiGen-RoBERTa Detector, Pythia-1B)

| Method | EMT(Full)↓ | TP(Full)↓ | EMT(Toxic)↓ | TP(Toxic)↓ |
|------|-----------|----------|------------|-----------|
| None | 0.548 | 0.563 | 0.742 | 0.775 |
| DPO | 0.401 | 0.406 | 0.573 | 0.595 |
| RAD | 0.286 | 0.278 | 0.397 | 0.398 |
| **IF-Guide** | **0.245** | **0.230** | **0.317** | **0.305** |

IF-Guide also outperforms RAD on implicit toxicity, achieving $2.2\times$ EMT and $2.4\times$ TP reduction.

### Key Findings

1. **Effective in fine-tuning settings**: Fine-tuning a pre-trained uncensored model requires only ~400M tokens (10% of pre-training compute) to achieve $3.0$–$5.7\times$ EMT reduction.
2. **Good proxy model generalizability**: Using Pythia-160M as a proxy to compute influence scores for Llama-3.2-1B results in a maximum performance gap of only 0.044 EMT.
3. **Adversarial robustness**: Under GCG attacks, IF-Guide achieves an ASR of only 0.22, compared to 0.39–0.43 for the base model and DPO.
4. **Mechanistic analysis**: Logit Lens analysis reveals that IF-Guide models entirely cease to encode toxic representations in intermediate layers (probability < 0.004), whereas DPO models suppress toxicity only in the final 3 layers.

## Highlights & Insights

1. **Paradigm shift**: Moving from "learn-then-fix" to proactive prevention — blocking toxic learning at the source via training data attribution represents a new direction in alignment research.
2. **Token-granularity operation**: Differential attribution combined with token-level scores enables precise localization of toxic segments within documents, rather than coarsely discarding entire documents.
3. **Orthogonality to existing methods**: IF-Guide can be stacked with DPO/RAD, further reducing toxicity by an order of magnitude.
4. **Computational practicality**: Only ~10k toxic reference samples (0.0005% of the corpus) are required; a small proxy model enables efficient attribution, and the identified toxic tokens can be reused across model training runs.
5. **Deep mechanistic insight**: Through Logit Lens and activation space analysis, the paper demonstrates that IF-Guide learns a direction that actively suppresses toxicity, rather than performing surface-level correction in the final few layers as DPO does.

## Limitations & Future Work

1. **Fluency cost**: PPL increases by approximately 1–4 points, particularly when training data is limited (academic-scale corpora of ~1B tokens).
2. **Toxicity classifier dependence**: The method relies on Detoxify for pseudo-labels; biases inherent in the classifier may propagate.
3. **Restricted to next-token prediction**: The current formulation applies only to autoregressive language models; applicability to encoder-only or multimodal models has not been verified.
4. **Influence function approximation error**: The degree to which EK-FAC approximation error affects larger models remains unclear.

## Related Work & Insights

- Recent advances in influence functions for data attribution (Grosse et al., 2023) have made large-scale LLM attribution practically feasible.
- IF-Guide is complementary to activation-space editing methods such as representation engineering: the former operates at training time, while the latter operates at inference time.
- Potential extensions: applying IF-Guide's token-level attribution to identify bias, privacy risks, and other undesirable properties in training data.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First to combine influence functions with gradient suppression for proactive detoxification; paradigm is highly original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 6 models × multiple baselines × pre-training/fine-tuning × explicit/implicit toxicity × adversarial testing × mechanistic analysis.
- **Value**: ⭐⭐⭐⭐ — Proxy model and incremental computation make the method practical, though substantial computational resources are still required.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Clear logical flow, well-motivated, with experiments structured in a progressive manner.
- **Overall**: ⭐⭐⭐⭐⭐ — Opens a new direction of proactive training data intervention in LLM safety, with comprehensive and in-depth experimentation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Position Paper: If Innovation in AI Systematically Violates Fundamental Rights, Is It Innovation at All?](position_paper_if_innovation_in_ai_systematically_violates_fundamental_rights_is.md)
- [\[NeurIPS 2025\] Auto-Search and Refinement: An Automated Framework for Gender Bias Mitigation in LLMs](auto-search_and_refinement_an_automated_framework_for_gender_bias_mitigation_in_.md)
- [\[AAAI 2026\] FactGuard: Event-Centric and Commonsense-Guided Fake News Detection](../../AAAI2026/social_computing/factguard_event-centric_and_commonsense-guided_fake_news_detection.md)
- [\[ICLR 2026\] Tracing and Reversing Edits in LLMs](../../ICLR2026/social_computing/tracing_and_reversing_edits_in_llms.md)
- [\[ICML 2026\] FLIPS: Instance-Fingerprinting for LLMs via Pseudo-Random Sequences](../../ICML2026/social_computing/flips_instance-fingerprinting_for_llms_via_pseudo-random_sequences.md)

</div>

<!-- RELATED:END -->
