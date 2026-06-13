---
title: >-
  [Paper Note] MIND: Multi-Rationale Integrated Discriminative Reasoning Framework for Multi-Modal Fake News
description: >-
  [ICML 2026][Social Computing][Multi-view reasoning] MIND provides an explainable and robust discriminative framework for fake news detection through **multi-view rationale generation + cross-rationale discriminative reas…
tags:
  - "ICML 2026"
  - "Social Computing"
  - "Multi-view reasoning"
  - "Fake news detection"
  - "Explainable reasoning"
  - "LLM integration"
date: 2026-05-08
content_hash: 483de7130c25bde4
---

# MIND: Multi-Rationale Integrated Discriminative Reasoning Framework for Multi-Modal Fake News

**Conference**: ICML 2026  
**arXiv**: [2605.29117](https://arxiv.org/abs/2605.29117)  
**Code**: To be confirmed  
**Area**: Social Computing / Multi-modal Learning / Explainable Fake News Detection  
**Keywords**: Multi-view reasoning, Fake news detection, Explainable reasoning, LLM integration

## TL;DR
MIND provides an explainable and robust discriminative framework for fake news detection through **multi-view rationale generation + cross-rationale discriminative reasoning**. By simultaneously utilizing three types of LLM-generated rationales—fact-checking, modal consistency, and semantic plausibility—it achieves a 4-8% F1 improvement over SOTA on Weibo, Twitter, and Fakeddit.

## Background & Motivation

**Background**: Multi-modal fake news detection faces two major challenges: **discriminative accuracy** (requiring fusion of text, images, and external knowledge) and **explainability** (requiring justification for the judgment). Most existing methods rely on end-to-end binary classification with poor interpretability.

**Limitations of Prior Work**: (1) End-to-end methods are black boxes, unable to explain the decision rationale; (2) Single reasoning perspectives (e.g., fact-checking or visual consistency) are easily deceived by adversarial samples; (3) While LLMs have strong reasoning capabilities, they are prone to "hallucinations" when used alone; (4) Existing explainable methods only provide attention visualization, lacking structured reasoning.

**Key Challenge**: Fake news detection requires **multi-view integrated judgment + structured reasoning evidence**, but existing methods either rely on a single perspective prone to deception or lack structured explanations.

**Goal**: Build a multi-view reasoning framework to simultaneously improve discriminative accuracy and explainability.

**Key Insight**: Human experts identify fake news by synthesizing three types of information: fact-checking (consistency with known facts), modal consistency (image-text alignment), and semantic plausibility (rationality of the narrative). This work simulates this process using LLMs and integrates it for discrimination.

**Core Idea**: Use LLMs to generate "rationales" from three independent perspectives as discriminative evidence; perform discriminative reasoning via cross-rationale attention; classify based on weighted multi-rationale evidence.

## Method

### Overall Architecture
(1) **Multi-view Rationale Generation**: Use pretrained LLMs (e.g., GPT-4 or Qwen-2.5) to generate three types of rationales $r_{\text{fact}}, r_{\text{cons}}, r_{\text{plau}}$ for each news item; (2) **Rationale Encoding**: Use a text encoder (e.g., BERT) to encode rationales into vectors $\mathbf{e}_{\text{fact}}, \mathbf{e}_{\text{cons}}, \mathbf{e}_{\text{plau}}$; (3) **Cross-Rationale Discriminative Reasoning**: Interact rationales through Transformer blocks; (4) **Multi-Rationale Fusion Classification**: Perform binary classification based on weighted rationale features + original multi-modal features.

### Key Designs

1. **Multi-view Rationale Generation Prompt Templates**:

    - **Function**: Generate three types of structured reasoning via designed prompts.
    - **Mechanism**: Each rationale type has an independent prompt—**Fact-checking** prompt ("Judge if this news is real based on known facts, provide 3 pieces of evidence"), **Modal Consistency** prompt ("Analyze if text and images are consistent, describe specific discrepancies"), and **Semantic Plausibility** prompt ("Evaluate if the news narrative conforms to common sense, point out suspicious points"). The LLM outputs in a fixed format (conclusion + evidence), which is stored as text rationale $r$.
    - **Design Motivation**: A single prompt asking an LLM to synthesize all perspectives is prone to bias; independent multi-view prompts force the LLM to reason from different angles, preserving detailed information.

2. **Cross-Rationale Discriminative Reasoning Module**:

    - **Function**: Capture interactions and conflicts between the three types of rationales.
    - **Mechanism**: Concatenate the three rationale embeddings $[\mathbf{e}_{\text{fact}}, \mathbf{e}_{\text{cons}}, \mathbf{e}_{\text{plau}}]$ into a sequence. Encode via $L$ layers of Transformer: self-attention $\mathbf{Z} = \text{softmax}(QK^T / \sqrt{d}) V$ captures correlations between rationales, and FFN enhances non-linear expression. The output provides updated embeddings $\tilde{\mathbf{e}}_{\text{fact}}, \tilde{\mathbf{e}}_{\text{cons}}, \tilde{\mathbf{e}}_{\text{plau}}$.
    - **Design Motivation**: Rationales may conflict—e.g., fact-checking may suggest the news is real, but modal consistency suggests it is fake. The cross-rationale reasoning module identifies and resolves these conflicts.

3. **Multi-Rationale Weighted Fusion Classifier**:

    - **Function**: Make final judgment based on rationale evidence + original multi-modal features.
    - **Mechanism**: Compute rationale weights via a gating network $\alpha_i = \text{softmax}(W_g [\tilde{\mathbf{e}}_i; \mathbf{e}_{\text{orig}}])$. The aggregated rationale feature is $\mathbf{e}_{\text{aggr}} = \sum_i \alpha_i \tilde{\mathbf{e}}_i$. The classifier takes $[\mathbf{e}_{\text{aggr}}; \mathbf{t}; \mathbf{v}]$ as input and is trained with cross-entropy loss.
    - **Design Motivation**: Different news items may rely on different perspectives (e.g., text-only fake news relies on fact-checking, while deepfakes rely on modal consistency). The weighting mechanism adaptively selects key perspectives.

## Key Experimental Results

### Main Results

| Dataset | Method | Acc | F1 | AUC |
|--------|------|-----|-----|-----|
| Weibo | EANN | 78.2 | 76.5 | 84.3 |
| Weibo | MVAE | 81.7 | 80.4 | 87.6 |
| Weibo | MCAN | 84.5 | 83.7 | 90.2 |
| Weibo | CAFE | 85.8 | 85.1 | 91.7 |
| Weibo | **MIND** | **90.3** | **89.5** | **95.2** |
| Twitter | MCAN | 79.3 | 78.4 | 85.6 |
| Twitter | CAFE | 82.1 | 81.5 | 88.3 |
| Twitter | **MIND** | **88.9** | **88.2** | **94.1** |
| Fakeddit | CAFE | 79.7 | 78.9 | 86.5 |
| Fakeddit | **MIND** | **86.7** | **86.0** | **92.4** |

### Ablation Study

| Configuration | Weibo F1 | Twitter F1 |
|------|---------|-----------|
| Only Fact-checking Rationale | 86.3 | 84.7 |
| Only Consistency Rationale | 84.7 | 83.5 |
| Only Plausibility Rationale | 85.1 | 83.9 |
| Three Rationales (w/o Cross-reasoning) | 87.9 | 86.5 |
| Three Rationales + Cross-reasoning (w/o Gating) | 88.4 | 87.1 |
| **Full MIND** | **89.5** | **88.2** |

### LLM Backend Comparison

| LLM Backend | Weibo F1 | Inference Cost |
|---------|---------|---------|
| GPT-4 | 89.5 | High |
| GPT-3.5 | 87.2 | Medium |
| Qwen-2.5-72B | 88.7 | Medium |
| Qwen-2.5-7B | 86.8 | Low |
| Llama-3-8B | 86.1 | Low |

### Interpretability Evaluation (Human Scoring, 1-5)

| Method | Explanation Quality | Reasoning Trustworthiness | Overall Satisfaction |
|------|---------|-----------|----------|
| Attention Visualization (Baseline) | 2.3 | 2.5 | 2.4 |
| Single LLM Explanation | 3.7 | 3.5 | 3.6 |
| **MIND** | **4.5** | **4.4** | **4.5** |

### Key Findings
- **Multi-view fusion significantly outperforms single perspectives**: 3-view vs. single view improves F1 by 3-5 percentage points.
- **Cross-rationale reasoning handles conflicts**: The gating network adaptively weighs contradicting rationales when inconsistencies occur.
- **Flexibility in LLM backend selection**: Using a 7B small model still maintains an 88% F1.
- **Substantial improvement in interpretability**: Human scores reach 4.5 vs. 2.4 for attention visualization.

## Highlights & Insights
- **Elegant design of the multi-view reasoning framework**: Simulates the cognitive process of human experts judging fake news from multiple angles.
- **Conflict resolution via cross-rationale discriminative reasoning**: Avoids blind reliance on a single rationale.
- **Achieving both interpretability and accuracy**: Breaks the "accuracy vs. black box" dichotomy.
- **Flexibility of LLM backends**: Effective across models from GPT-4 to Qwen-7B, making deployment costs controllable.

## Limitations & Future Work
- LLM inference cost: Requires 3 LLM calls per news item.
- Rationale quality depends on LLM capability: Rationales may be incorrect if the LLM undergoes hallucinations.
- Perspective coverage: Three perspectives might not be exhaustive.
- Future Work: Explore dynamic perspective selection; introduce active learning for rationale updates; multi-language adaptation.

## Related Work & Insights
- **vs EANN/MVAE**: These use single fusion classification without explicit reasoning.
- **vs MCAN/CAFE**: These focus on cross-modal attention or contrastive learning but remain black boxes.
- **vs IDO**: While IDO models inconsistent distributions explicitly, MIND performs explicit multi-view reasoning; the two are complementary and could potentially be integrated.
- **Insight**: Multi-view rationale generation + cross-view reasoning can be extended to other scenarios requiring explainable discrimination.

## Rating
- Novelty: ⭐⭐⭐⭐ The multi-view rationale generation + cross-rationale reasoning framework is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 3 datasets + 5 baselines + LLM backend comparison + human interpretability scoring + detailed ablation.
- Writing Quality: ⭐⭐⭐⭐⭐ Logical and clear, with complete prompt templates provided, ensuring high reproducibility.
- Value: ⭐⭐⭐⭐⭐ Simultaneously improves detection accuracy and interpretability, holding significant value for practical deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] IDO: Incongruity-Aware Distribution Optimization for Multimodal Fake News Detection](ido_incongruity-aware_distribution_optimization_for_multimodal_fake_news_detecti.md)
- [\[ACL 2026\] MM-StanceDet: Retrieval-Augmented Multi-modal Multi-agent Stance Detection](../../ACL2026/social_computing/mm-stancedet_retrieval-augmented_multi-modal_multi-agent_stance_detection.md)
- [\[AAAI 2026\] Cross-modal Prompting for Balanced Incomplete Multi-modal Emotion Recognition](../../AAAI2026/social_computing/cross-modal_prompting_for_balanced_incomplete_multi-modal_emotion_recognition.md)
- [\[AAAI 2026\] FactGuard: Event-Centric and Commonsense-Guided Fake News Detection](../../AAAI2026/social_computing/factguard_event-centric_and_commonsense-guided_fake_news_detection.md)
- [\[AAAI 2026\] Multi-modal Dynamic Proxy Learning for Personalized Multiple Clustering](../../AAAI2026/social_computing/multi-modal_dynamic_proxy_learning_for_personalized_multiple_clustering.md)

</div>

<!-- RELATED:END -->
