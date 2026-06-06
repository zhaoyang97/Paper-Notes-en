---
title: >-
  [Paper Note] Context-Fidelity Boosting: Enhancing Faithful Generation through Watermark-Inspired Decoding
description: >-
  [ACL 2026][LLM Safety][faithfulness hallucination] CFB adapts the additive logit bias technique used in text watermarking in reverse—applying a bonus to tokens "supported by the input context" at each decoding step. It p…
tags:
  - "ACL 2026"
  - "LLM Safety"
  - "faithfulness hallucination"
  - "logit shaping"
  - "watermark"
  - "context-aware decoding"
  - "RAG"
date: 2026-05-08
content_hash: a5f3069b132b6c2e
---

# Context-Fidelity Boosting: Enhancing Faithful Generation through Watermark-Inspired Decoding

**Conference**: ACL 2026  
**arXiv**: [2604.22335](https://arxiv.org/abs/2604.22335)  
**Code**: <https://github.com/weixuzhang/CFB>  
**Area**: LLM Security / Faithful Generation / Decoding Strategy  
**Keywords**: faithfulness hallucination, logit shaping, watermark, context-aware decoding, RAG

## TL;DR
CFB adapts the additive logit bias technique used in text watermarking in reverse—applying a bonus to tokens "supported by the input context" at each decoding step. It proposes three progressive strategies: static, context-aware (adaptive scaling via JSD), and token-aware (redistribution via attention and semantic relevance). The method consistently improves faithfulness metrics in summarization and QA tasks across various models with negligible decoding overhead.

## Background & Motivation

**Background**: LLMs frequently output plausible-sounding content that contradicts the input context in context-driven tasks like RAG, summarization, and conversational IR—a phenomenon known as faithfulness hallucination (distinct from factuality hallucination, which concerns inconsistencies with world facts).

**Limitations of Prior Work**: (1) Training-time methods (faithful fine-tuning) require retraining and suffer from poor cross-domain generalization; (2) prompting methods (chain-of-thought, self-consistency) are unstable across models; (3) existing decoding-time methods (CAD / ADACAD / COIECD) rely on contrasting entire distributions from two forward passes or impose hard constraints, often leading to severe fluctuations between faithfulness and fluency and sensitivity to hyperparameters.

**Key Challenge**: To make the LLM adhere to external context without rendering the output rigid or incoherent—requiring a strong bias toward context-related words while maintaining the natural language distribution.

**Goal**: To introduce a lightweight, model-agnostic, and virtually overhead-free decoding intervention that biases the model toward source-supported tokens without retraining, while allowing the bias intensity to adapt to sample difficulty and token importance.

**Key Insight**: Literature on text watermarking has demonstrated that lightweight additive logit biases can stably modify generation without destroying fluency (using green/red token sets). While watermarking aims to embed detectable signals, the same logit-shaping mechanism can be reversed—replacing "green tokens" with "context-supported tokens."

**Core Idea**: At each decoding step, an additive bias $\Delta_t(w)$ is applied to the logits of tokens appearing in the source span. The method designs three strategies of increasing granularity: fixed values, sample-level adaptation based on the divergence between distributions with and without context, and token-level redistribution based on attention and semantic relevance.

## Method

### Overall Architecture
Given a context $C$ and query $Q$, the source span $S \subseteq C$ is parsed, and its token set $V_S$ is extracted as the "supported vocabulary." In each decoding step: (1) obtain the original logits $l_t$; (2) calculate $\Delta_t(w)$ based on the selected strategy; (3) rewrite logits as $\tilde l_t(w) = l_t(w) + \Delta_t(w)$ if $w \in V_S$ else $l_t(w)$; (4) sample the next token using softmax; (5) append and continue. The three modes progressively enhance controllability from static to sample-level and finally token-level adaptation.

### Key Designs

1.  **Static Boosting (Fixed Bias)**:
    - **Function**: Applies a uniform bias $\Delta_t(w) = \delta$ to all source-supported tokens.
    - **Mechanism**: A fixed value such as $\delta = 5$ is used to forcefully elevate the log-likelihood of context words. Tokens not in $V_S$ remain unaffected, ensuring that "natural tokens" from the base distribution can still be selected to prevent extreme rigidity.
    - **Design Motivation**: Served as a baseline to validate whether the "logit shaping" approach is effective. It is the most computationally efficient solution for deployment, adding only a single tensor addition per step (Table 5 shows this accounts for 0.003% of base model FLOPS).

2.  **Context-Aware Boosting (Adaptive via JSD)**:
    - **Function**: Dynamically adjusts the bias magnitude based on how much the context shifts the model's prediction.
    - **Mechanism**: The Jensen-Shannon divergence $D = \mathrm{JSD}(P_w \| P_{wo})$ is calculated between the next-token distributions with and without context ($D \in [0,1]$). Then, $\Delta_t(w) = \delta_{\min} + (\delta_{\max} - \delta_{\min}) \cdot D$. This implies minimal bias when the context does not significantly change the model's preference (low $D$) and strong bias during high conflict (high $D$).
    - **Design Motivation**: Sample-level adaptation avoids excessive intervention and distortion in cases where the context does not conflict with parametric knowledge, while maintaining strong steering for high-conflict scenarios. It serves as a lightweight alternative to ADACAD.

3.  **Token-Aware Boosting (Attention + Semantic Relevance Redistribution)**:
    - **Function**: Distributes the boost across different tokens based on the local relevance of each source token, building upon the sample-level adaptation $\delta(D)$.
    - **Mechanism**: For each candidate $w \in V_S$, the local relevance $r_t(w)$ is estimated by aggregating its attention weights in the source $\alpha_t(w) = \mathrm{Agg}\{a_t(p): p \in \mathcal{P}(w,C)\}$ and calculating its source-scoped semantic similarity $s(w) = \frac{1}{|S|} \sum_{c \in S} \cos(e_w, e_c)$. Relevance is defined as $r_t(w) = \lambda_1 \alpha_t(w) + \lambda_2 s(w)$ (with $\lambda_1=0.6, \lambda_2=0.4$); after normalization, $\hat r_t(w) = r_t(w) / \frac{1}{|V_S|}\sum_u r_t(u)$, and the final bias is $\Delta_t(w) = \delta(D) \cdot \hat r_t(w)$.
    - **Design Motivation**: While sample-level bias treats all source tokens equally, some are more relevant to the current decoding state. By combining dynamic attention and static embedding similarity, the "boost budget" is concentrated on the most useful tokens, enhancing precision without increasing the total intervention.

### Loss & Training
No training is required; this is a pure decoding-time intervention. Semantic similarity is pre-computed once per sample, while attention is recalculated at each step to reflect the current decoding state. All experiments use top-$p$ sampling in a zero-shot setting with fixed $\lambda_1 = 0.6, \lambda_2 = 0.4$.

## Key Experimental Results

### Main Results (Summarization: CNN/DM + XSum, QA: NQ-Synth + NQ-Swap, Models: Mistral-7B / Llama2-13B / Llama3-8B)

| Task + Model | Method | ROUGE-L | FactKB | BERT-P | Acc |
|-------------|------|---------|--------|--------|-----|
| CNN/DM + Llama2-13B | CAD | 35.63 | 97.26 | 89.38 | – |
| CNN/DM + Llama2-13B | **Static CFB** | 37.40 | **98.85** | 89.61 | – |
| CNN/DM + Llama2-13B | **Context-aware CFB** | **37.52** | 98.69 | 89.62 | – |
| CNN/DM + Llama2-13B | **Token-aware CFB** | 36.16 | 97.24 | **89.83** | – |
| XSum + Llama3-8B | CAD | 12.92 | 45.77 | 87.05 | – |
| XSum + Llama3-8B | **Context-aware CFB** | 12.59 | **66.85** | 88.67 | – |
| XSum + Llama3-8B | **Token-aware CFB** | **13.23** | 55.29 | 88.45 | – |
| NQ-Synth + Llama3-8B | CAD | 28.19 | 32.26 | 86.50 | 66.80 |
| NQ-Synth + Llama3-8B | **Token-aware CFB** | **32.90** | **45.94** | 88.13 | **73.40** |
| NQ-Swap + Llama3-8B | ADACAD | 12.52 | 39.14 | 85.82 | **86.50** |
| NQ-Swap + Llama3-8B | Token-aware CFB | 14.54 | 40.92 | 87.99 | 32.43 |

> ADACAD outperforms on NQ-Swap: When context explicitly conflicts with parametric knowledge, the "contrastive suppression" strategy is more effective than "additive boosting." Since CFB's philosophy is to boost rather than suppress, it is stronger in complementary-context scenarios but weaker in direct-conflict scenarios—a clear design trade-off.

### Ablation Study (Token-aware CFB on Llama3-8B / CNN-DM)

| Configuration | ROUGE-L | FactKB | BERT-P |
|------|---------|--------|--------|
| Full Token-aware CFB | 35.81 | 94.31 | 89.38 |
| w/o attention | 35.60 | 93.74 | 88.48 |
| **w/o semantic** | **4.45** | **66.84** | **67.68** |
| w/o JSD | 35.24 | 93.60 | 88.43 |

Human + GPT-4o judge evaluation (100 cases each for CNN-DM and NQ-Swap):

| Method | Faith. | Flu. | Info. | Consistency | Hallucinations | Contradiction |
|------|--------|------|-------|-------------|----------------|---------------|
| CAD | 3.82 | 4.15 | 3.76 | 0.83 | 1.24 | 0.12 |
| ADACAD | 4.03 | 4.21 | 3.89 | 0.87 | 0.95 | 0.09 |
| **Token-aware CFB** | **4.31** | 4.18 | **4.12** | **0.91** | **0.67** | **0.05** |

### Key Findings
- All three boosting variants outperform CAD / ADACAD / COIECD in faithfulness metrics on CNN/DM, with negligible loss in fluency (BERT-P) or lexical overlap (ROUGE-L).
- Ablation reveals that **semantic similarity is the crux of token-aware boosting**—removing it causes ROUGE-L to plummet from 35.81 to 4.45, indicating that semantic relevance provides a critical stability signal that attention alone cannot sustain.
- CFB underperforms relative to ADACAD on NQ-Swap (high knowledge conflict). Simple context boosting is insufficient when context contradicts parametric knowledge; suppression of parametric preferences is also required—highlighting the fundamental difference between the "boosting" and "suppression" paradigms.
- Computational Overhead: Static and Context-aware variants add only 0.003% to base FLOPS. Token-aware requires $2.86 \times 10^8$ FLOPS for attention and cosine calculations, which remains negligible.

## Highlights & Insights
- Reversing logit shaping from watermarking for anti-hallucination is a simple yet elegant instance of idea cross-pollination—using the same mathematical mechanism for opposite goals (adding detectable signals vs. adding context signals).
- The three-tier progressive design (Static → Sample Adaptive → Token-level Granularity) allows users to choose based on their precision and compute requirements, representing a model for tiered research and engineering.
- Combining dynamic attention, static embedding similarity, and sample-level JSD scaling allows multiple signals to be fused with clear physical interpretations for each component.
- The candid admission of underperformance on NQ-Swap, attributed to the "boost vs suppress" paradigm, is more informative than standard "blanket SOTA" claims.

## Limitations & Future Work
- Dependency on logit and attention access makes it inapplicable to black-box APIs (GPT-4 / Gemini); the paper lists black-box approximation as future work.
- Performance is lower in high-conflict scenarios (NQ-Swap), suggesting a need for integration with suppression strategies (e.g., combining with ADACAD) to cover the full spectrum of scenarios.
- The dominance of semantic similarity suggests that the "fine-grained" contribution of the token-aware approach may be limited—a simplified version using only sample-level and semantic signals might achieve similar results.
- Sensitivity to the $\delta$ hyperparameter: Moderate values are optimal for CNN-DM, while performance collapses at high values. NQ-Synth has a wider tolerance. Grid search may be necessary for new datasets.

## Related Work & Insights
- **vs CAD (Shi et al. 2024)**: CAD subtracts "without context" from "with context" distributions. CFB uses a single forward pass with additive bias, reducing overhead and maintaining better fluency.
- **vs ADACAD (Wang et al. 2024)**: ADACAD uses JSD to adjust contrastive intensity; CFB uses it to adjust boost intensity. Their philosophies are opposite (suppress vs. boost), making ADACAD stronger in high-conflict scenarios and CFB stronger in low-conflict ones.
- **vs COIECD (Yuan et al. 2024)**: COIECD uses entropy constraints to distinguish conflicting vs. non-conflicting tokens. CFB applies a uniform boost.
- **vs watermarking (Kirchenbauer / Liu et al.)**: Uses the same logit-shaping mechanism. Watermarking selects green tokens as random seeds, while CFB treats context-supported tokens as the target set—different goals, same mathematical origin.

## Rating
- Novelty: ⭐⭐⭐⭐ Reversing watermarking with a tiered boost design is a clear contribution, though individual components are straightforward.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive evaluation across 3 models, 4 datasets, and 6 methods, including ablation and human/LLM judging.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear algorithms, intuitive case studies, and honest analysis of failure cases on NQ-Swap.
- Value: ⭐⭐⭐⭐ Directly actionable for RAG and summarization deployment with minimal overhead; however, the requirement for white-box access limits usage in some scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Enhancing Hallucination Detection via Future Context](enhancing_hallucination_detection_via_future_context.md)
- [\[ACL 2026\] Differentially Private Synthetic Text Generation for Retrieval-Augmented Generation (RAG)](differentially_private_synthetic_text_generation_for_retrieval-augmented_generat.md)
- [\[ICLR 2026\] Enhancing Hallucination Detection through Noise Injection](../../ICLR2026/llm_safety/enhancing_hallucination_detection_through_noise_injection.md)
- [\[ACL 2026\] Membership Inference Attacks on In-Context Learning Recommendation](membership_inference_attacks_on_llm-based_recommender_systems.md)
- [\[ACL 2026\] LeakDojo: Decoding the Leakage Threats of RAG Systems](leakdojo_decoding_the_leakage_threats_of_rag_systems.md)

</div>

<!-- RELATED:END -->
