---
title: >-
  [Paper Note] FactGuard: Event-Centric and Commonsense-Guided Fake News Detection
description: >-
  [AAAI2026][Social Computing][fake news detection] This paper proposes FactGuard, a framework that leverages LLMs to extract event-centric content (with style removed) and generate commonsense rationales. A Rationale Usability Evaluator dynamically assesses the reliability of LLM suggestions. Knowledge distillation yields a lightweight variant, FactGuard-D, that operates without LLM inference, achieving both robustness and efficiency in fake news detection.
tags:
  - "AAAI2026"
  - "Social Computing"
  - "fake news detection"
  - "LLM reasoning"
  - "knowledge distillation"
  - "commonsense reasoning"
  - "style debiasing"
date: 2026-05-08
content_hash: 99d283e14dea50c5
---

# FactGuard: Event-Centric and Commonsense-Guided Fake News Detection

**Conference**: AAAI2026
**arXiv**: [2511.10281](https://arxiv.org/abs/2511.10281)  
**Code**: [ryliu68/FACTGUARD](https://github.com/ryliu68/FACTGUARD)  
**Area**: Social Computing
**Keywords**: fake news detection, LLM reasoning, knowledge distillation, commonsense reasoning, style debiasing

## TL;DR
This paper proposes FactGuard, a framework that leverages LLMs to extract event-centric content (with style removed) and generate commonsense rationales. A Rationale Usability Evaluator dynamically assesses the reliability of LLM suggestions. Knowledge distillation yields a lightweight variant, FactGuard-D, that operates without LLM inference, achieving both robustness and efficiency in fake news detection.

## Background & Motivation
- **Vulnerability to writing style**: Style-based fake news detection methods have advanced, yet adversaries can mimic legitimate news styles to evade detection.
- **Shallow exploitation of LLMs**: Existing LLM-augmented approaches suffer from several issues:
    - Style-augmented data cannot fully eliminate style interference (SheepDog, LLM-Fake)
    - LLM few-shot/CoT reasoning has limited accuracy and is prone to hallucination
    - Multi-agent debate frameworks (TED) incur high inference costs, making them unsuitable for cold-start or resource-constrained settings
    - No reliable mechanism exists to evaluate the usability of LLM suggestions (ARG knows correctness during training but not at inference)

Core Mechanism: News originates from real-world events (news communication theory); extracting event-centric content removes style noise, while LLMs' commonsense reasoning capacity supplements factual consistency judgment.

## Method

### Feature Extraction
For each news article $n$, carefully designed prompts are used to extract via LLM:
- **Topic-Content $c$**: Core topic and main content (de-stylized), with textual similarity constraints and information density evaluation
- **Commonsense Rationale $r$**: Commonsense reasoning analysis identifying whether content violates common sense

All three representations are encoded separately via an SLM (BERT/RoBERTa).

### Topic-Content & Rationale Interactor
Bidirectional cross-attention enables deep feature interaction between topic-content and commonsense rationale:
- $f_{C \to R}$: LLM suggestion feature vector
- $f_{R \to C}$: Interaction features used for weight estimation

### Rationale Usability Evaluator
A dual-branch MLP structure dynamically evaluates the reliability of LLM suggestions:
- **Branch 1**: Reduces contribution when LLM's direct detection capability is limited (supervision signal = 0)
- **Branch 2**: Increases contribution when commonsense reasoning detects contradictions or uncertainty (supervision signal = $y_{llm}$)
- Final LLM feature: $f_{llm} = [w_1 \cdot f_{C \to R1}; w_2 \cdot f_{C \to R2}]$

### Loss & Training
$$\mathcal{L}_{total} = \mathcal{L}_{cls} + \alpha \frac{\mathcal{L}_{usability}}{2} + \beta \frac{\mathcal{L}_{text}}{2}$$

### FactGuard-D (Distilled Variant)
- A four-layer Transformer encoder with linear attention simulates the teacher model's reasoning capacity
- Takes only raw news text as input; no LLM calls required
- Training loss includes an additional MSE feature distillation term $\mathcal{L}_{distill}$

## Key Experimental Results

Evaluated on Weibo21 (Chinese) and GossipCop (English).

### Main Results (macF1)

| Method | Weibo21 macF1 | GossipCop macF1 |
|---|---|---|
| BERT | 0.753 | 0.765 |
| ARG (LLM+SLM) | 0.784 | 0.790 |
| TED (multi-agent debate) | 0.795 | 0.803 |
| **FactGuard** | **0.801** | **0.805** |
| ARG-D (distilled) | 0.771 | 0.778 |
| **FactGuard-D** | **0.788** | **0.790** |

- FactGuard achieves Acc. 0.804 on Weibo21, surpassing TED by 0.8%
- FactGuard-D requires no LLM inference yet outperforms the full ARG model

### Ablation Study
- Removing the raw news representation causes the largest performance drop (irreplaceable foundation)
- Removing topic-content extraction reduces macF1 by approximately 3% (event-centric information is critical)
- Removing the usability module degrades performance, validating dynamic reliability estimation
- Topic-content and commonsense rationale must be used jointly to maximize gains

## Highlights & Insights
- **Event-centric de-stylization**: LLM semantic capacity is used to extract event-centric content, reducing writing style interference at the source—a more fundamental approach than style-augmented data methods
- **Dynamic usability evaluation of LLM suggestions**: The dual-branch structure distinguishes LLM direct judgment capability from commonsense reasoning contribution, avoiding blind trust in LLM outputs
- **Full-scenario deployment**: FactGuard suits resource-rich settings; FactGuard-D adapts to cold-start and resource-constrained scenarios via distillation, surpassing multi-agent debate frameworks with only two simple prompts
- **Cross-lingual validation**: Consistently effective across both Chinese and English datasets

## Limitations & Future Work
- macF1 gains are relatively modest (approximately 0.6–0.8% over TED), with smaller improvement on GossipCop
- Quality of LLM-extracted topic-content depends heavily on prompt design and LLM capability, potentially varying across different LLMs
- Validation is limited to the text modality; multimodal fake news detection is not addressed
- The feature simulator in FactGuard-D increases student model complexity; its advantage over direct SLM usage warrants further evaluation across more settings
- Benchmark data contamination of the employed LLMs is not discussed

## Rating
- **Novelty**: ⭐⭐⭐ — The event extraction de-stylization idea is reasonable, but the overall framework is largely an engineering combination of existing components
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Bilingual datasets, 14 baselines, detailed ablation and parameter analysis
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure, well-motivated arguments, intuitive illustrations
- **Value**: ⭐⭐⭐⭐ — Provides a complete fake news detection solution covering both resource-rich and resource-constrained scenarios

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Detection of Human and Machine-Authored Fake News in Urdu](../../ACL2025/social_computing/detection_of_human_and_machine-authored_fake_news_in_urdu.md)
- [\[ICML 2026\] IDO: Incongruity-Aware Distribution Optimization for Multimodal Fake News Detection](../../ICML2026/social_computing/ido_incongruity-aware_distribution_optimization_for_multimodal_fake_news_detecti.md)
- [\[ACL 2025\] Synergizing LLMs with Global Label Propagation for Multimodal Fake News Detection](../../ACL2025/social_computing/llm_label_propagation.md)
- [\[ACL 2026\] LiveFact: A Dynamic, Time-Aware Benchmark for LLM-Driven Fake News Detection](../../ACL2026/social_computing/livefact_a_dynamic_time-aware_benchmark_for_llm-driven_fake_news_detection.md)
- [\[ICML 2026\] MIND: Multi-Rationale Integrated Discriminative Reasoning Framework for Multi-Modal Fake News](../../ICML2026/social_computing/mind_multi-rationale_integrated_discriminative_reasoning_framework_for_multi-mod.md)

</div>

<!-- RELATED:END -->
