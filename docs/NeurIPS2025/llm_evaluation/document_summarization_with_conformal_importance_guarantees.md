---
title: >-
  [Paper Note] Document Summarization with Conformal Importance Guarantees
description: >-
  [NeurIPS 2025][LLM Evaluation][Document Summarization] This work presents the first application of Conformal Prediction to document summarization. By calibrating a threshold on sentence importance scores…
tags:
  - "NeurIPS 2025"
  - "LLM Evaluation"
  - "Document Summarization"
  - "Conformal Prediction"
  - "Importance Coverage Guarantee"
  - "Extractive Summarization"
  - "Distribution-Free"
date: 2026-05-08
content_hash: b7930cf24d68c4c8
---

# Document Summarization with Conformal Importance Guarantees

**Conference**: NeurIPS 2025
**arXiv**: [2509.20461](https://arxiv.org/abs/2509.20461)
**Code**: [https://github.com/layer6ai-labs/conformal-importance-summarization](https://github.com/layer6ai-labs/conformal-importance-summarization)
**Area**: LLM Evaluation
**Keywords**: Document Summarization, Conformal Prediction, Importance Coverage Guarantee, Extractive Summarization, Distribution-Free

## TL;DR
This work presents the first application of Conformal Prediction to document summarization. By calibrating a threshold on sentence importance scores, it provides rigorous statistical guarantees on user-controllable coverage ($1-\alpha$) and recall ($\beta$) for extractive summaries. The method is model-agnostic and requires only a small calibration set.

## Background & Motivation

**Background**: LLMs have substantially improved summarization quality; however, in high-stakes domains such as healthcare, law, and finance, the omission of critical information from summaries can have severe consequences. Existing summarization approaches—whether extractive or abstractive—offer no guarantees on the coverage of key content.

**Limitations of Prior Work**: (a) Abstractive LLM summarization is prone to hallucination and provides no control over information coverage; (b) extractive methods are more faithful but lack theoretical guarantees; (c) users cannot express requirements such as "I want at least 80% of important information to be retained."

**Key Challenge**: Summarization inherently demands compression (shorter is better), whereas safety-critical scenarios require completeness (important information must not be omitted)—a controllable trade-off between conciseness and completeness is needed.

**Goal**: To provide formal statistical guarantees for summarization—retaining $\geq \beta$ proportion of important sentences with probability $\geq 1-\alpha$.

**Key Insight**: Conformal Prediction has delivered distribution-free guarantees in classification, regression, and QA. This paper extends it from a *precision guarantee* (conformal factuality ensuring retained claims are factual) to a *recall guarantee* (ensuring important sentences are retained).

**Core Idea**: Find an importance score threshold $\hat{q}$ on a calibration set such that summaries filtered by this threshold retain $\geq \beta$ of important sentences with probability $\geq 1-\alpha$.

## Method

### Overall Architecture
Given a long document $x = \{c_1, \ldots, c_p\}$ segmented into sentences, an importance scoring function $R(c;x)$ assigns a score to each sentence. Conformal Prediction then calibrates a threshold $\hat{q}$, and sentences with scores $\geq \hat{q}$ are retained to form the summary $y = F_{\hat{q}}(x)$. The output is an extractive summary satisfying $\mathbb{P}[B(y;y^*) \geq \beta] \geq 1-\alpha$.

### Key Designs

1. **Generalized Coverage Guarantee**

    - **Function**: Relaxes the "full coverage" requirement of classical Conformal Prediction, allowing users to specify an acceptable recall level $\beta$.
    - **Mechanism**: Recall is defined as $B(y;y^*) = |y \cap y^*| / |y^*|$, with the objective $\mathbb{P}[B(y;y^*) \geq \beta] \geq 1-\alpha$. Setting $\beta=1$ recovers full coverage. For each calibration sample, the conformal score is computed as $S_\beta(x_i, y_i^*) = \max\{q \in \mathbb{R}^+ \mid B(F_q(x_i); y_i^*) \geq \beta\}$, i.e., the maximum threshold that maintains $\beta$ recall. The $\lfloor\alpha(n+1)\rfloor/n$ quantile of all scores is used as $\hat{q}$.
    - **Design Motivation**: This is symmetric to the precision guarantee of conformal factuality ($y \subseteq T(x,y^*)$), but oriented toward recall. The parameter $\beta$ affords flexible user control—medical scenarios may require $\beta=1$ (no omissions), while news summarization may tolerate $\beta=0.8$.

2. **Importance Scoring Function $R(c;x)$**

    - **Function**: Estimates an importance score for each sentence in the document.
    - **Mechanism**: Two families of scoring approaches are provided—(a) LLM scoring: prompting GPT-4o mini, Gemini, Llama, etc., to assign scores in [0,1]; (b) embedding-based similarity: using SBERT sentence embeddings aggregated via graph algorithms such as Cosine Centrality, Sentence Centrality, GUSUM, and LexRank.
    - **Design Motivation**: The framework is model-agnostic—any method that produces scores can serve as $R$. LLM-based scoring generally achieves higher AUPRC, while graph-based methods require no API calls. Scoring quality directly determines the conciseness of the summary at a fixed coverage level.

3. **Hybrid Extractive–Abstractive Pipeline**

    - **Function**: First extracts important sentences via Conformal Importance (with coverage guarantees), then rewrites them with an LLM for fluency and conciseness.
    - **Mechanism**: Summarization is decomposed into two sub-tasks—information selection (extractive, with guarantees) and fluent synthesis (abstractive, without guarantees but empirically preserving most information). This is analogous to the retrieval–generation separation in RAG.
    - **Design Motivation**: Pure extractive summaries may lack coherence, while pure abstractive summaries cannot control coverage. The two-stage pipeline achieves higher empirical information retention than direct LLM summarization.

### Theoretical Guarantee (Theorem 1)
Under the exchangeability assumption, for $\alpha \in [1/(n+1), 1]$:
$$1 - \alpha \leq \mathbb{P}[B(F_{\hat{q}}(x_{n+1}); y^*_{n+1}) \geq \beta] < 1 - \alpha + \frac{1}{n+1}$$
The guarantee is tight, and only $n=100$ calibration samples are required to achieve a coverage error of approximately 1%.

## Key Experimental Results

### Main Results (Importance Scoring Quality AUPRC + Summary Conciseness)

| Scoring Method | ECT AUPRC | CSDS | CNN/DM | Avg. Conciseness (α=0.2, β=0.8) |
|---|---|---|---|---|
| Random (positive rate) | 0.10 | 0.27 | 0.10 | 0% compression |
| Cos. Sim. Centrality | 0.22 | 0.34 | 0.34 | 22%/11%/18% |
| GUSUM | 0.21 | 0.44 | 0.33 | 11%/24%/27% |
| LexRank | 0.22 | 0.43 | 0.32 | 16%/12%/20% |
| GPT-4o mini | **0.30** | **0.49** | 0.34 | 24%/25%/30% |
| Gemini 2.5 Flash | **0.31** | **0.55** | **0.44** | **26%/37%/33%** |
| Llama3-8B | 0.18 | 0.39 | 0.22 | 13%/11%/14% |

### Coverage Guarantee Verification

| Setting | Theoretical Lower Bound | Empirical Coverage (400 random splits) | Theoretical Upper Bound |
|---|---|---|---|
| α=0.1, β=1.0 | 90% | 90.2% | 91% |
| α=0.2, β=0.8 | 80% | 80.4% | 81% |
| α=0.3, β=0.6 | 70% | 70.1% | 71% |

Empirical coverage strictly falls within the theoretical bounds across all experiments, confirming Theorem 1.

### Key Findings
- Gemini 2.5 Flash achieves the best importance scoring across all datasets; GPT-4o mini ranks second; smaller models (Llama3-8B, Qwen3-8B) underperform graph-based methods.
- Only 100 calibration samples are needed for stable guarantees ($1/(n+1) \approx 1\%$).
- The hybrid pipeline achieves higher empirical information retention than direct LLM summarization (86% vs. 79% recall on ECT) while producing more concise outputs.
- The $\alpha$ and $\beta$ parameters provide continuous control over the conciseness–completeness trade-off, whereas direct LLM summarization offers only a single fixed operating point.

## Highlights & Insights
- **Novel extension of Conformal Prediction from precision to recall**: Conformal factuality guarantees precision (retained claims are correct); this work guarantees recall (important sentences are retained)—an elegant symmetric inversion that adapts CP to the summarization setting.
- **Extremely lightweight framework**: The core requires only a scoring function, a calibrated threshold, and a filtering step, and can be seamlessly integrated into any existing method. Any sentence-level scoring approach is compatible.
- **Dual $\alpha$-$\beta$ parameter control**: Users can precisely specify "how much risk I can accept ($\alpha$)" and "what minimum proportion of important information must be retained ($\beta$)"—highly practical for high-stakes applications.

## Limitations & Future Work
- A calibration set with ground-truth annotations is required. Although 100 samples is modest, obtaining annotations in new domains still incurs cost.
- Importance is entirely determined by the scoring function $R$—if $R$ quality is poor, coverage guarantees may be satisfied but summaries will be excessively long (insufficiently concise).
- Sentence-level granularity may not suit all scenarios (e.g., conversational or tabular data).
- The abstractive generation step in the hybrid pipeline cannot formally guarantee the preservation of coverage (though empirical results are favorable).
- The exchangeability assumption may not hold under distribution shift (e.g., across different versions of the same LLM).

## Related Work & Insights
- **vs. Conformal Factuality (Mohri & Hashimoto)**: That work guarantees that retained claims in QA are factual; this work guarantees that important sentences are retained in summaries—opposite directions, but similar frameworks.
- **vs. BERTSum/TextRank and other extractive methods**: Traditional methods only perform ranking without coverage guarantees; adding a calibration layer on top is sufficient to obtain guarantees.
- **vs. Direct LLM summarization**: LLMs produce an uncontrollable fixed recall; this work enables continuous and controllable recall.

## Rating
- **Novelty**: ⭐⭐⭐⭐ First application of CP to summarization; the $\alpha$-$\beta$ generalization is a meaningful contribution
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Five datasets, nine scoring functions, 400 random-split validations
- **Writing Quality**: ⭐⭐⭐⭐⭐ Theoretical derivations are clear; experimental design is systematic
- **Value**: ⭐⭐⭐⭐ Directly applicable to high-stakes summarization scenarios

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Conformal Online Learning of Deep Koopman Linear Embeddings](conformal_online_learning_of_deep_koopman_linear_embeddings.md)
- [\[AAAI 2026\] Axis-Aligned Document Dewarping](../../AAAI2026/llm_evaluation/axis-aligned_document_dewarping.md)
- [\[ICLR 2026\] Conformal Prediction Adaptive to Unknown Subpopulation Shifts](../../ICLR2026/llm_evaluation/conformal_prediction_adaptive_to_unknown_subpopulation_shifts.md)
- [\[ICCV 2025\] ForCenNet: Foreground-Centric Network for Document Image Rectification](../../ICCV2025/llm_evaluation/forcennet_foreground-centric_network_for_document_image_rectification.md)
- [\[AAAI 2026\] Improved Runtime Guarantees for the SPEA2 Multi-Objective Optimizer](../../AAAI2026/llm_evaluation/improved_runtime_guarantees_for_the_spea2_multi-objective_optimizer.md)

</div>

<!-- RELATED:END -->
