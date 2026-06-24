---
title: >-
  [Paper Note] Cape: Context-Aware Prompt Perturbation Mechanism with Differential Privacy
description: >-
  [ICML 2025][LLM Safety][Differential Privacy] Cape is proposed—a context-aware prompt perturbation mechanism that combines a hybrid utility function (integrating token embedding distance and contextual logits) with a bucketized exponential sampling mechanism to achieve a superior privacy-utility trade-off under local DP guarantees compared to existing methods.
tags:
  - "ICML 2025"
  - "LLM Safety"
  - "Differential Privacy"
  - "Prompt Perturbation"
  - "LLM Inference Privacy"
  - "Exponential Mechanism"
  - "Context-Awareness"
date: 2026-05-08
content_hash: 86f1e4714d4d719c
---

# Cape: Context-Aware Prompt Perturbation Mechanism with Differential Privacy

**Conference**: ICML 2025  
**arXiv**: [2505.05922](https://arxiv.org/abs/2505.05922)  
**Code**: None  
**Area**: AI Safety  
**Keywords**: Differential Privacy, Prompt Perturbation, LLM Inference Privacy, Exponential Mechanism, Context-Awareness

## TL;DR
Cape is proposed—a context-aware prompt perturbation mechanism that combines a hybrid utility function (integrating token embedding distance and contextual logits) with a bucketized exponential sampling mechanism to achieve a superior privacy-utility trade-off under local DP guarantees compared to existing methods.

## Background & Motivation

**Background**: LLM inference services (e.g., ChatGPT) require users to send plaintext prompts to servers. These prompts may contain sensitive information (such as emails or trade secrets), posing privacy leakage risks.

**Limitations of Prior Work**:
   - Cryptographic schemes (MPC/HE) provide provable security but incur immense overhead (~30s per token), making them impractical.
   - White-box schemes (e.g., DP-Forward) require deploying shallow layers of the model on the client side, demanding model modifications.
   - Existing DP schemes (SANTEXT, CUSTEXT, InferDPT) only measure semantic similarity using token embedding distances and ignore contextual information, leading to poor semantic consistency (e.g., "enjoyable" and "unenjoyable" are close in embedding space but have opposite meanings).

**Key Challenge**: The NLP vocabulary is extremely large (30K+). Under such a large sampling space, the standard exponential mechanism suffers from the long-tail phenomenon, where the cumulative probability of a huge number of low-utility tokens is high, resulting in frequent sampling of irrelevant tokens.

**Goal**: How to perturb prompts using an efficient DP mechanism under a black-box inference scenario while preserving both semantic consistency and privacy?

**Key Insight**: (a) Introduce contextual logit information to enhance the utility function; (b) design a bucketized sampling mechanism to suppress the long-tail effect.

**Core Idea**: Contextual relevance (logits provided by a small client-side model) and token embedding distance are fused into a hybrid utility function. Combined with an equal-width bucketized exponential mechanism, this enables context-aware prompt perturbation.

## Method

### Overall Architecture
The client holds the original prompt $x = \{t_1, t_2, \dots, t_n\}$ and a small on-device model $\mathcal{M}_c$ (such as BERT/GPT-2). For each sensitive token $t_i$:
1. Compute the hybrid utility function $u(t_i, t_r)$ to score each candidate token $t_r$ in the vocabulary.
2. Sample the replacement token $\hat{t}_i$ using the bucketized exponential mechanism.
3. Send the perturbed prompt $\hat{x}$ to the server.

### Key Designs

1. **Hybrid Utility Function**:

    - **Function**: Integrates token embedding distance and context logits to score candidate tokens.
    - **Mechanism**: $u(t_i, t_r) = L_r^{\lambda_L} \cdot D(t_i, t_r)^{\lambda_D}$, where $L_r = \mathcal{M}_c(t_r | \text{Ctx})$ is the contextual logit, and $D(t_i, t_r) = \exp(-d_{\text{euc}}^{\text{norm}}(t_i, t_r))$ represents the exponential decay of the normalized Euclidean distance.
    - **Design Motivation**: Relying solely on embedding distance can confuse antonyms (e.g., enjoyable vs. unenjoyable); incorporating context logits suppresses contextually inappropriate replacements with low logits.
    - Boundedness Guarantee: The distance component is $D \in (0, 1]$, and logits are clipped to $[-B, B]$ to ensure controllable sensitivity.

2. **Bucketized Exponential Mechanism**:

    - **Function**: Resolves the long-tail issue of the standard exponential mechanism over large vocabularies.
    - **Mechanism**: Candidate tokens are sorted by utility score and grouped into $N_b$ equal-width buckets. Each bucket is represented by its mean utility. First, the exponential mechanism (EM) is used to sample a bucket, and then a token is sampled uniformly within the selected bucket.
    - Sampling Probability: $\mathbb{P}[\mathcal{R}(t) = t_r] \propto \frac{\exp(\frac{\epsilon}{2\Delta} \text{mean}(b_i))}{|b_i|}$
    - **Design Motivation**: In the standard EM, the cumulative probability of the top-10 high-utility tokens is less than 1% (when $\epsilon=6, N=50000$). After bucketization, the impact of low-utility tokens is compressed by bucket-level probabilities.
    - Privacy Guarantee: Satisfies $(\epsilon + \epsilon')$-DP, where $\epsilon' = \ln(\max_{i,j} \frac{|b_i|}{|b_j|})$.

3. **Non-sensitive Token Retention**:

    - **Function**: Predefines 179 stop words + 32 punctuation marks as non-sensitive tokens, leaving them unperturbed.
    - **Design Motivation**: These tokens do not contain private information but are crucial for textual coherence.

### Loss & Training
- No training required, pure inference-time mechanism.
- The client-side small model (BERT/GPT-2) provides logit information without needing fine-tuning.
- Default hyperparameter configuration: $\lambda_L = 0.5$, $\lambda_D = 1.0$, $N_b = 50$.

## Key Experimental Results

### Main Results
Sentence-level similarity (Rouge-L F1) for zero-shot classification using Qwen2-1.5B-Instruct on SST-2:

| Method | ε=1 | ε=6 | ε=14 | ε=20 |
|------|-----|-----|------|------|
| SANTEXT | 0.87 | 99.36 | 99.45 | 99.45 |
| CUSTEXT | 14.50 | 47.27 | 95.54 | 99.48 |
| InferDPT | 13.00 | 16.48 | 38.68 | 68.11 |
| **Cape (BERT)**| **38.38** | **46.85** | **76.49** | **92.03** |
| Cape (GPT2) | 37.60 | 44.55 | 73.46 | 90.65 |

### Ablation Study

| Configuration | Effect | Explanation |
|------|------|------|
| Only distance ($\lambda_L=0$) | Utility drop | Loss of contextual information |
| Only logit ($\lambda_D=0$) | Utility drop | Loss of semantic similarity constraints |
| No bucketization ($N_b=1$) | Severe long-tail | Extremely low sampling probability for high-utility tokens |
| $N_b=50$ (Default) | Optimal | Balances the number of buckets and in-bucket granularity |
| $N_b=500$ | Slight drop | Too many buckets, some are empty |

### Key Findings
- After aligning the actual privacy budget ($\epsilon' \sim 14$), Cape's Rouge-L is significantly superior to SANTEXT ($\epsilon \sim 1$) and InferDPT ($\epsilon \sim 6$).
- Although CUSTEXT achieves high utility, its privacy is weak—even at $\epsilon=1$, the KNN attack success rate still exceeds 60%.
- The BERT-based context model outperforms the GPT-2 version because BERT captures bidirectional context.

## Highlights & Insights
- **Hybrid Utility Function** is simple yet effective: Multiplying the context logit and the embedding distance allows a single formula to handle both semantic similarity and context consistency.
- **Bucketization Strategy** addresses the overlooked long-tail issue in DP-NLP: By sampling buckets first and then sampling uniformly within the bucket, the sampling space is compressed from 30K+ to 50 bucket-level decisions.
- A pure inference-time scheme requiring no modifications to the back-end model, demonstrating high practicality (~0.1s per input).

## Limitations & Future Work
- Bucketization introduces an additional privacy overhead $\epsilon'$, which increases as the variation in bucket sizes grows.
- The client needs to deploy a small model (BERT/GPT-2), which remains a burden for extremely resource-constrained devices.
- The retention strategy for stop words/punctuation is overly simple, potentially leaking sentence structure information.
- Privacy composition issues under multi-turn dialogue scenarios are not discussed.

## Related Work & Insights
- **vs. SANTEXT**: Sampling is performed over the entire vocabulary, which is essentially random when $\epsilon=1$. Cape introduces significant improvements through bucketization and the hybrid utility.
- **vs. CUSTEXT**: Achieves high utility through a fixed small adjacency list but sacrifices privacy, essentially acting as truncated DP.
- **vs. InferDPT**: Both are black-box schemes, but Cape's utility is substantially enhanced after introducing contextual information.
- **vs. DP-Forward/TextObfuscator**: White-box schemes require model modifications, making Cape more practical.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of the hybrid utility function and bucketized sampling is novel and resolves practical pain points.
- Experimental Thoroughness: ⭐⭐⭐⭐ Thorough evaluations across multiple datasets, attacks, and complete ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with intuitive motivational examples.
- Value: ⭐⭐⭐⭐ A practical privacy-preserving scheme for black-box inference.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] InvisibleInk: High-Utility and Low-Cost Text Generation with Differential Privacy](../../NeurIPS2025/llm_safety/invisibleink_high-utility_and_low-cost_text_generation_with_differential_privacy.md)
- [\[ICML 2025\] The Canary's Echo: Auditing Privacy Risks of LLM-Generated Synthetic Text](the_canarys_echo_auditing_privacy_risks_of_llm-generated_synthetic_text.md)
- [\[ICML 2025\] ICLShield: Exploring and Mitigating In-Context Learning Backdoor Attacks](iclshield_exploring_and_mitigating_in-context_learning_backdoor_attacks.md)
- [\[ICML 2025\] Empirical Privacy Variance](empirical_privacy_variance.md)
- [\[ICML 2025\] Federated In-Context Learning: Iterative Refinement for Improved Answer Quality](federated_in-context_learning_iterative_refinement_for_improved_answer_quality.md)

</div>

<!-- RELATED:END -->
