---
title: >-
  [Paper Note] FAITH: Factuality Alignment through Integrating Trustworthiness and Honestness
description: >-
  [ACL 2026][Factuality alignment] This paper proposes FAITH, a framework that maps LLM uncertainty signals (consistency + semantic entropy) to natural-language descriptions of knowledge state quadrants (trustworthiness × honestness), designs uncertainty-aware fine-grained reward functions for PPO training, and applies a RAG module to correct potentially erroneous outputs, systematically improving the factual accuracy of LLMs.
tags:
  - ACL 2026
  - Factuality alignment
  - knowledge state quadrant
  - uncertainty estimation
  - PPO
  - retrieval-augmented generation
date: 2026-05-08
content_hash: ef2c75ac1b520136
---

# FAITH: Factuality Alignment through Integrating Trustworthiness and Honestness

**Conference**: ACL 2026
**arXiv**: [2604.10189](https://arxiv.org/abs/2604.10189)
**Code**: [https://github.com/xndong/FAITH](https://github.com/xndong/FAITH)
**Area**: Information Retrieval
**Keywords**: Factuality alignment, knowledge state quadrant, uncertainty estimation, PPO, retrieval-augmented generation

## TL;DR
This paper proposes FAITH, a framework that maps LLM uncertainty signals (consistency + semantic entropy) to natural-language descriptions of knowledge state quadrants (trustworthiness × honestness), designs uncertainty-aware fine-grained reward functions for PPO training, and applies a RAG module to correct potentially erroneous outputs, systematically improving the factual accuracy of LLMs.

## Background & Motivation

**Background**: LLMs may generate fluent yet factually incorrect content (hallucinations), even when the model internally possesses the correct knowledge. This know-tell gap — where the model "knows but fails to say correctly" — severely undermines reliability. Recent work has attempted to incorporate uncertainty signals during training to align factuality.

**Limitations of Prior Work**: (1) Existing methods directly embed numerical uncertainty scores into QA prompts (e.g., "Conf: 0.833"), which lack semantic richness and are difficult for LLMs to interpret and utilize; (2) Binary reward functions (correct/incorrect) ignore the confidence of responses, potentially encouraging "guessing" behavior; (3) External knowledge is not leveraged to correct potentially erroneous answers.

**Key Challenge**: Factual inconsistency in LLMs manifests as the same question receiving correct answers under some phrasings and incorrect answers under others. The fundamental cause is a misalignment between the model's actual knowledge possession (whether it truly knows) and its response behavior (whether it expresses that knowledge honestly). Numerical uncertainty signals fail to help the model understand its own knowledge boundaries.

**Goal**: To design a post-training framework that transforms uncertainty signals into semantically rich knowledge state descriptions, and improves LLM factual accuracy and truthfulness through fine-grained rewards and external knowledge retrieval.

**Key Insight**: The paper categorizes each LLM's knowledge state for a given question into four quadrants — KH (knowledgeable and honest), K¬H (knowledgeable but dishonest), ¬KH (not knowledgeable but honest), and ¬K¬H (not knowledgeable and dishonest) — and embeds natural-language descriptions of these states into the training prompts.

**Core Idea**: Replace numerical uncertainty with natural-language knowledge states → design fine-grained rewards that jointly consider correctness and uncertainty → apply a RAG module to correct weakly grounded responses.

## Method

### Overall Architecture
FAITH consists of three modules: (1) **Data Augmentation** — sampling multiple responses, estimating uncertainty, and mapping to knowledge state quadrants to augment training data; (2) **Model Training** — reference model SFT → reward model training → PPO policy optimization → RAG model training; (3) **Inference** — knowledge state estimation → policy model response → RAG correction.

### Key Designs

1. **Knowledge State Quadrant Mapping**:

    - **Function**: Transforms numerical uncertainty signals into semantically rich natural-language knowledge state descriptions.
    - **Mechanism**: For each question $x_i$, $K=6$ responses are sampled to compute consistency $\text{Consistency}(x_i) = \frac{1}{K}\sum \mathbb{1}\{y_i^k = \hat{y_i}\}$ and semantic entropy $SE(x_i) = -\sum_c p(c|x_i) \log p(c|x_i)$. The mapping rules are: consistency $> 0$ and $SE = 0$ → KH; consistency $> 0$ and $SE \neq 0$ → K¬H; consistency $= 0$ and $SE = 0$ → ¬KH; otherwise → ¬K¬H. The resulting state description is embedded in natural language within the training prompt.
    - **Design Motivation**: Raw numerical values (e.g., "Conf: 0.833") are opaque to LLMs — the model cannot infer from them whether it "knows" the answer or "should respond." Natural-language descriptions provide semantically rich and interpretable guidance, enabling the model to better recognize its own knowledge boundaries.

2. **Fine-Grained Reward Function**:

    - **Function**: Jointly considers response correctness and model uncertainty, providing more informative feedback than binary rewards.
    - **Mechanism**: $R_{\text{FAITH}} = R_{\text{correctness}} + R_{\text{uncertainty}}$, where the correctness reward is standard exact match, and the uncertainty reward is assigned according to knowledge state: KH → $+2$, K¬H → $+1$, ¬KH → $-1$, ¬K¬H → $-2$. The total reward ranges from $-2$ to $+3$.
    - **Design Motivation**: Binary rewards focus solely on correctness, failing to distinguish between "confidently correct" and "luckily correct." The fine-grained reward encourages the model to answer confidently when it knows (KH → $+2$), to honestly abstain when uncertain (¬KH), and penalizes uninformed guessing (¬K¬H → $-2$).

3. **RAG Correction Module**:

    - **Function**: Uses external knowledge to correct weakly grounded responses produced by the policy model.
    - **Mechanism**: A vector database is constructed from Wikipedia, and a RAG model is trained to retrieve relevant passages as contextual input to correct potentially incorrect policy model responses. The RAG model operates after the policy model as a final factuality verification layer.
    - **Design Motivation**: Even after PPO training, the model may still produce incorrect answers for questions in the ¬K state (insufficient knowledge). RAG provides an external knowledge source to compensate for gaps in the model's internal knowledge.

### Loss & Training
Four-stage training pipeline: (1) SFT to train the reference model $\pi_\mu$; (2) reward model training; (3) PPO to optimize the policy model; (4) RAG model training. Experiments are conducted on Llama3-8B and Mistral-7B-v0.1. NQ-Open, SciQ, and TriviaQA are used for in-domain training; PopQA is used for out-of-domain evaluation.

## Key Experimental Results

### Main Results (Llama3-8B)

| Method | In-Domain Accuracy | In-Domain Truthfulness | Out-of-Domain Accuracy | Out-of-Domain Truthfulness |
|---|---|---|---|---|
| Baseline Average | Low | Low | Low | Low |
| UAlign | Medium | Medium | Medium | Medium |
| FAITH | **74.26%** | **45.73%** | **67.99%** | **34.03%** |

### Ablation Study

| Configuration | Accuracy | Truthfulness | Note |
|---|---|---|---|
| Full FAITH | Best | Best | Complete framework |
| w/o Knowledge State (numerical) | Degraded | Degraded | Advantage of natural-language descriptions |
| w/o Uncertainty Reward (binary) | Degraded | Degraded | Necessity of fine-grained rewards |
| w/o RAG | Degraded | Degraded | Contribution of external knowledge correction |

### Key Findings
- Natural-language knowledge state descriptions consistently outperform numerical uncertainty signals across both models and all four datasets.
- Fine-grained rewards provide better learning signals than binary rewards, reducing inflated performance caused by lucky guesses.
- The RAG module contributes most substantially to out-of-domain generalization, compensating for gaps in the model's internal knowledge coverage.
- FAITH outperforms five strong baselines on both Llama3-8B and Mistral-7B, demonstrating cross-model generalizability.

## Highlights & Insights
- **Semantic Knowledge State Representation**: Transforming opaque uncertainty values into natural-language descriptions such as "knowledgeable and honest / dishonest" enables LLMs to understand their own knowledge boundaries during training — a more natural fit for LLM capabilities than learning what "0.833" implies.
- **Distinguishing "Confidently Correct" from "Luckily Correct"**: By penalizing low-confidence correct responses through uncertainty rewards, FAITH reduces inflated performance metrics — particularly important in high-stakes domains such as medicine and law.
- **Three-Layer Factuality Assurance**: Knowledge state guidance (self-awareness) → fine-grained rewards (behavioral alignment) → RAG correction (external verification), forming a complete factuality assurance pipeline.

## Limitations & Future Work
- Knowledge state estimation requires sampling multiple responses per question ($K=6$), incurring additional inference overhead.
- The four-quadrant discretization may oversimplify reality — actual knowledge states may be continuous rather than discrete.
- The RAG module relies on a Wikipedia corpus, which may provide insufficient coverage for recent knowledge or specialized domains.
- PPO training entails high computational costs and may not scale readily to larger models.

## Related Work & Insights
- **vs. UAlign**: UAlign directly embeds numerical uncertainty scores into prompts and uses binary rewards. FAITH employs natural-language states and fine-grained rewards, providing richer learning signals.
- **vs. R-Tuning**: R-Tuning trains models to abstain when uncertain but does not capture the fine-grained distinction between knowledge possession and response behavior.
- **vs. Self-Consistency Methods**: Self-consistency detects uncertainty through repeated sampling; FAITH converts this signal into a training signal rather than using it solely at inference time.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of knowledge state quadrants and fine-grained rewards represents an insightful design.
- Experimental Thoroughness: ⭐⭐⭐⭐ Four datasets, two models, five baselines, and detailed ablation studies.
- Writing Quality: ⭐⭐⭐⭐ The framework is described clearly with complete formal derivations.
- Value: ⭐⭐⭐⭐ Provides a practical and interpretable approach to factuality alignment in LLMs.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] Multi-Faceted Self-Consistent Preference Alignment for Query Rewriting in Conversational Search](multi-faceted_self-consistent_preference_alignment_for_query_rewriting_in_conver.md)
- [\[AAAI 2026\] RRRA: Resampling and Reranking through a Retriever Adapter](../../AAAI2026/information_retrieval/rrra_resampling_and_reranking_through_a_retriever_adapter.md)
- [\[AAAI 2026\] HiMo-CLIP: Modeling Semantic Hierarchy and Monotonicity in Vision-Language Alignment](../../AAAI2026/information_retrieval/himo-clip_modeling_semantic_hierarchy_and_monotonicity_in_vi.md)
- [\[CVPR 2026\] Explaining CLIP Zero-shot Predictions Through Concepts](../../CVPR2026/information_retrieval/explaining_clip_zero-shot_predictions_through_concepts.md)
- [\[AAAI 2026\] Cog-RAG: Cognitive-Inspired Dual-Hypergraph with Theme Alignment Retrieval-Augmented Generation](../../AAAI2026/information_retrieval/cog-rag_cognitive-inspired_dual-hypergraph_with_theme_alignment_retrieval-augmen.md)

<!-- RELATED:END -->
