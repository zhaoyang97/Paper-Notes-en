---
title: >-
  [Paper Note] Interpreting Fedspeak with Confidence: A LLM-Based Uncertainty-Aware Framework Guided by Monetary Policy Transmission Paths
description: >-
  [AAAI2026 Oral][Time Series][Fedspeak] This paper proposes an LLM-based uncertainty-aware framework for interpreting Fedspeak (Federal Reserve language). The framework enhances inputs through domain reasoning along monetary policy transmission paths, and introduces a dynamic uncertainty decoding module to quantify prediction confidence (Perceptual Uncertainty = Environmental Ambiguity × Cognitive Risk), achieving SOTA performance on FOMC monetary policy stance analysis.
tags:
  - "AAAI2026 Oral"
  - "Time Series"
  - "Fedspeak"
  - "monetary policy stance"
  - "LLM"
  - "uncertainty quantification"
  - "financial sentiment analysis"
date: 2026-05-08
content_hash: 9de70883433ad850
---

# Interpreting Fedspeak with Confidence: A LLM-Based Uncertainty-Aware Framework Guided by Monetary Policy Transmission Paths

**Conference**: AAAI2026 Oral  
**arXiv**: [2508.08001](https://arxiv.org/abs/2508.08001)  
**Code**: [yuuki20001/FOMC-sentiment-path](https://github.com/yuuki20001/FOMC-sentiment-path)  
**Area**: Time Series
**Keywords**: Fedspeak, monetary policy stance, LLM, uncertainty quantification, financial sentiment analysis

## TL;DR
This paper proposes an LLM-based uncertainty-aware framework for interpreting Fedspeak (Federal Reserve language). The framework enhances inputs through domain reasoning along monetary policy transmission paths, and introduces a dynamic uncertainty decoding module to quantify prediction confidence (Perceptual Uncertainty = Environmental Ambiguity × Cognitive Risk), achieving SOTA performance on FOMC monetary policy stance analysis.

## Background & Motivation
Fedspeak is the specialized language used by the Federal Reserve to convey policy signals, characterized by strong context dependence — the same word may indicate opposite stances under different economic conditions (e.g., a "strong" labor market is dovish in a weak economy but hawkish in an overheating one).

Limitations of prior work:
- **Dictionary-based methods**: Simple and interpretable but unable to handle complex context.
- **Fine-tuned models (e.g., FinBERT)**: Strong performance but black-box with limited transparency.
- **Zero-shot large models (e.g., GPT-4)**: Capable but neglect reliability, bias, and hallucination concerns.
- Existing LLM work focuses predominantly on performance metrics while overlooking the evaluation of prediction reliability.

Core idea: The LLM is analogized to a policy analyst, with two uncertainty dimensions — Cognitive Risk (CR) and Environmental Ambiguity (EA) — introduced to quantify prediction confidence.

## Method

### Data Augmentation: Domain Reasoning
1. **Financial Entity Relation Extraction**: Atomic relations $r(e_i, e_j) \in \mathcal{R}$ are decomposed from Fedspeak, covering six types: CAUSE, COND, EVID, PURP, ACT, and COMP.
2. **Monetary Policy Transmission Path Reasoning**: A quadruple $\Gamma = (\mathbf{X}, \mathbf{Y}, \mathbf{Z}, \mathbf{M})$ is constructed, where:
    - $\mathbf{X}$: economic shock vector
    - $\mathbf{Y}$: transmission channels (credit channel, asset price channel, aggregate demand channel, etc.)
    - $\mathbf{Z}$: transmission paths (state transition sequences)
    - $\mathbf{M}$: final policy recommendations
3. Structured templates combined with human-AI collaboration are used to construct the SFT dataset.

### Dynamic Uncertainty Decoding
The top-$k$ logits from the LLM output are used to construct a Dirichlet distribution, from which three uncertainty measures are derived:

- **Environmental Ambiguity (EA)**: Expected entropy of the predictive distribution
$$EA(a_t) = -\sum_{k=1}^{K} \frac{\alpha_k}{\alpha_0}(\psi(\alpha_k+1) - \psi(\alpha_0+1))$$

- **Cognitive Risk (CR)**: Inversely proportional to the total evidence mass
$$CR(a_t) = \frac{K}{\sum_{k=1}^{K}(\alpha_k + 1)}$$

- **Perceptual Uncertainty (PU)**: $PU = EA \times CR$

The decoding strategy switches dynamically based on the PU threshold:
- Low PU → aggressive (select top-1 token directly)
- High PU → conservative (sample from top-2)

## Key Experimental Results

### Experimental Setup
- **Dataset**: Trillion Dollar Words FOMC dataset (1996–2022), comprising three document types: meeting minutes, press conferences, and speeches.
- **Baselines**: 10+ models including GPT-4.1, Gemini-2.5-Pro, DeepSeek-R1, Phi-4, FinBERT, and AICBC.
- **Backbone**: Qwen3-14B fine-tuned with LoRA.

### Main Results (All Categories)

| Method | Macro F1 | Weighted F1 |
|---|---|---|
| GPT-4.1 (zero-shot) | 0.6662 | 0.6763 |
| AICBC (zero-shot) | 0.6637 | 0.6802 |
| Qwen3-8B (fine-tuned) | 0.6586 | 0.6745 |
| **Ours** | **0.7327** | **0.7426** |

- Outperforms the strongest baseline by **+6.6%** in Macro F1 and **+6.2%** in Weighted F1.
- Best performance on meeting minutes: Macro F1 = 0.7449 (+7.4%).
- Speeches: Macro F1 = 0.7291 (+6.7%).

### Ablation Study

| Configuration | Macro F1 | Weighted F1 |
|---|---|---|
| Full model | 0.7327 | 0.7426 |
| w/o PU | 0.7291 | 0.7378 |
| w/o Transmission Path | 0.6538 | 0.6699 |
| w/o Entity Relations | 0.6397 | 0.6551 |

The transmission path contributes most (−7.9% when removed), followed by entity relations, while the PU module contributes more modestly but consistently.

### Uncertainty Validation
- Low-PU predictions: Macro F1 = 0.7791; high-PU predictions: Macro F1 = 0.2473.
- p-values from t-test, Mann-Whitney U test, and logistic regression are all well below 0.001, indicating strong statistical significance.

## Highlights & Insights
- **Domain reasoning innovation**: The first work to formalize monetary policy transmission mechanisms as structured reasoning templates, simulating the analytical workflow of human domain experts.
- **Practical PU measure**: The decomposition of EA × CR aligns with the classical risk/ambiguity distinction in economics, making it intuitively natural for financial applications.
- **High-PU warning mechanism**: Enables identification of unreliable predictions, supporting human-in-the-loop decision making.
- **Comprehensive superiority over GPT-4.1**: Substantially outperforms closed-source large models on both meeting minutes and speeches.

## Limitations & Future Work
- Performance on press conferences falls below GPT-4.1 (−1.3%), suggesting insufficient capture of dynamic context dependencies in real-time Q&A settings.
- Transmission path construction relies on manually designed templates, limiting automation.
- Validation is limited to FOMC English data; generalization to other central banks (ECB, BoE) or multilingual scenarios remains unexplored.
- The PU threshold requires search on a validation set and must be re-tuned for different datasets.
- The "abstain from answering" strategy in practical deployment has not been explored.

## Rating
- Novelty: ⭐⭐⭐⭐ — The combination of monetary policy transmission path reasoning and PU quantification constitutes a clear methodological contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive coverage with 10+ baselines, three document types, ablation studies, and statistical testing.
- Writing Quality: ⭐⭐⭐⭐ — Well-structured with smooth integration of economics and NLP concepts.
- Value: ⭐⭐⭐⭐ — Meaningfully advances reliability research in financial NLP.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] HiVid: LLM-Guided Video Saliency For Content-Aware VOD And Live Streaming](../../ICLR2026/time_series/hivid_llm-guided_video_saliency_for_content-aware_vod_and_live_streaming.md)
- [\[ICML 2025\] Event-Aware Sentiment Factors from LLM-Augmented Financial Tweets: A Transparent Framework for Interpretable Quant Trading](../../ICML2025/time_series/event-aware_sentiment_factors_from_llm-augmented_financial_tweets_a_transparent_.md)
- [\[AAAI 2026\] ProbFM: Probabilistic Time Series Foundation Model with Uncertainty Decomposition](probfm_probabilistic_time_series_foundation_model_with_uncertainty_decomposition.md)
- [\[ICLR 2026\] EVEREST: A Transformer for Probabilistic Rare-Event Anomaly Detection with Evidential and Tail-Aware Uncertainty](../../ICLR2026/time_series/everest_a_transformer_for_probabilistic_rare-event_anomaly_detection_with_eviden.md)
- [\[CVPR 2026\] Towards Uncertainty-aware Unsupervised Domain Adaptation for Videos and Time-Series with Causal Optimal Transport](../../CVPR2026/time_series/towards_uncertainty-aware_unsupervised_domain_adaptation_for_videos_and_time-ser.md)

</div>

<!-- RELATED:END -->
