---
title: >-
  [Paper Note] Reliable Decision Making via Calibration Oriented Retrieval Augmented Generation
description: >-
  [NeurIPS 2025][Information Retrieval & RAG][RAG] This paper proposes CalibRAG, a framework that trains a temperature-conditioned forecasting function to ensure confidence calibration in RAG-assisted decision-making…
tags:
  - "NeurIPS 2025"
  - "Information Retrieval & RAG"
  - "RAG"
  - "calibration"
  - "decision-making"
  - "retrieval augmented generation"
  - "confidence estimation"
date: 2026-05-08
content_hash: d2be05a89b354f9d
---

# Reliable Decision Making via Calibration Oriented Retrieval Augmented Generation

**Conference**: NeurIPS 2025
**arXiv**: [2411.08891](https://arxiv.org/abs/2411.08891)
**Code**: To be confirmed
**Area**: Information Retrieval
**Keywords**: RAG, calibration, decision-making, retrieval augmented generation, confidence estimation

## TL;DR

This paper proposes CalibRAG, a framework that trains a temperature-conditioned forecasting function to ensure confidence calibration in RAG-assisted decision-making, achieving improvements in both calibration quality and accuracy.

## Background & Motivation

- LLMs are increasingly used to assist human decision-making, yet they frequently produce incorrect information with high confidence (hallucination), leading users to make suboptimal decisions.
- Studies show that users tend to over-rely on LLM outputs, and the degree of reliance is proportional to the model's expressed confidence.
- RAG mitigates hallucination by incorporating external documents, but **RAG retrievers may return irrelevant documents**, and LLMs tend to be overconfident about retrieved content.
- Existing RAG methods focus solely on retrieval relevance, **without considering whether the resulting user decisions are well-calibrated**.
- Traditional temperature scaling is not applicable to calibration in long-form text generation.
- Prior decision calibration methods require fine-tuning 3 LLMs with PPO training, which is costly and unstable.

## Method

### Overall Architecture

The core idea of CalibRAG is to train a forecasting function $f(t, q, d)$ that predicts "the probability that a user makes a correct decision given temperature $t$, query $q$, and retrieved document $d$." At inference time, this function re-ranks retrieved documents and selects those most likely to lead to a correct decision.

The four-stage inference pipeline is as follows:
1. **Stage 1 – Initial Retrieval**: Given query $q^*$, retrieve Top-K candidate documents.
2. **Stage 2 – Scoring and Selection**: Score and re-rank each document $d_i^*$ using $f(t, q^*, d_i^*)$.
3. **Stage 3 – Query Reformulation (Optional)**: If the highest confidence score falls below threshold $\epsilon = 0.5$, reformulate the query and retrieve again.
4. **Stage 4 – Final Decision**: Generate guidance and a confidence score for the user to make a decision.

### Key Designs

**Forecasting Function Modeling**:

A frozen LLM $\mathcal{M}$ serves as the feature extractor $f_{\text{feat}}$. The temperature parameter is encoded using Fourier positional encoding:

$$\text{PE}(t) = [\sin(\omega_1 t), \cos(\omega_1 t), \ldots, \sin(\omega_N t), \cos(\omega_N t)]$$

where $\omega_n = 2^n \cdot \frac{2\pi}{t_{\max} - t_{\min}}$. The final model is:

$$f(t, q, d) = \sigma\left(W_{\text{head}}^\top \left(f_{\text{feat}}(\text{concat}[q, d]; W_{\text{LoRA}}) + W_p \cdot \text{PE}(t)\right) + b_{\text{head}}\right)$$

Only the LoRA adapters and a lightweight prediction head are trained; the LLM backbone remains frozen.

**Synthetic Supervision Data Generation**:

- $(x, y)$ pairs are extracted from TriviaQA, SQuAD2.0, and WikiQA.
- Top-20 documents (rather than only Top-1) are retrieved per query, for two reasons: (1) lower-ranked documents may also contribute to correct decisions; (2) this avoids biasing the training data toward negative samples.
- A proxy user model $U$ is used to sample $R=10$ responses at varying temperatures $t$.
- Soft labels $b \in [0,1]$ represent the proportion of correct responses.

### Loss & Training

A log-likelihood loss (strictly proper scoring rule) is used:

$$\mathcal{L} = -\frac{1}{|\mathcal{S}|} \sum_{(t,q,d,b) \in \mathcal{S}} \left[b \log f(t,q,d) + (1-b)\log(1-f(t,q,d))\right]$$

This loss is a strictly proper scoring rule (logarithmic score), whose unique maximizer is the true probability $p$, thereby guaranteeing calibration convergence.

A multi-class variant, CalibRAG-multi, is also explored, which discretizes the correctness distribution into histogram bins (0–10).

## Key Experimental Results

### Main Results: General Domain (NQ, WebQA)

Llama-3.1-8B is used as both the RAG model and the decision model, with BM25 and Contriever as retrievers.

| Method | Metric | NQ (BM25) | WebQA (BM25) |
|--------|--------|-----------|--------------|
| CT-probe | ECE↓ | ~0.35 | ~0.38 |
| Number-LoRA | ECE↓ | ~0.30 | ~0.33 |
| CalibRAG | ECE↓ | **~0.15** | **~0.18** |
| CalibRAG-multi | ECE↓ | **~0.14** | **~0.17** |

CalibRAG outperforms all baselines across all metrics (1-AUROC, 1-ACC, ECE, BS).

### Medical Domain (MedCPT Retriever)

| Metric | BioASQ-Y/N | MMLU-Med | PubMedQA |
|--------|-----------|---------|---------|
| CalibRAG ECE↓ | Best | Best | Best |
| CalibRAG ACC↑ | Best | Best | Best |

CalibRAG is trained on general-domain data yet achieves state-of-the-art performance on medical benchmarks with unseen retrievers and out-of-distribution datasets.

### Comparison with Re-ranking / Robust RAG Methods

| Dataset | Method | AUROC↑ | ACC↑ | ECE↓ | BS↓ |
|---------|--------|--------|------|------|-----|
| HotpotQA | Cross-encoder | 60.74 | 34.98 | 0.477 | 0.477 |
| HotpotQA | LLM-rerank | 60.57 | 38.52 | 0.248 | 0.297 |
| HotpotQA | **CalibRAG** | **72.47** | **42.37** | **0.106** | **0.206** |
| NQ | SelfRAG | 48.4 | 36.2 | 0.522 | 0.545 |
| NQ | **CalibRAG** | **63.5** | **37.4** | **0.258** | **0.287** |

### Ablation Study

- **Temperature conditioning**: Removing temperature conditioning leads to a significant increase in ECE, particularly under high-temperature sampling, validating the necessity of temperature modeling.
- **Number of retrieved documents**: Performance is optimal at $K=20$; increasing to 40 yields diminishing returns.
- **Query reformulation**: Stage 3 consistently improves performance across all settings, albeit at additional computational cost.

### Key Findings

1. The Top-1 retrieved document is not always optimal—lower-ranked documents can sometimes yield better decisions.
2. While adding retrieved documents improves accuracy, it also increases ECE (overconfidence), necessitating additional calibration.
3. Although CalibRAG is primarily designed for calibration, it also improves accuracy by selecting documents more likely to lead to correct decisions.

## Highlights & Insights

1. **Novel problem formulation**: The objective of RAG is extended from "retrieving relevant documents" to "ensuring well-calibrated decisions," representing a distinctly new perspective.
2. **Elegant temperature-conditioned design**: Fourier encoding is used to model variability in user behavior, enabling a single model to accommodate users with different risk preferences.
3. **Strong cross-domain generalization**: A model trained on general-domain data can be directly applied to unseen medical retrievers and datasets.
4. **Lightweight solution**: Only LoRA adapters and a classification head require training, avoiding unstable procedures such as PPO.
5. **Rigorous theoretical guarantee**: The use of a strictly proper scoring rule as the loss function provides convergence guarantees for calibration.

## Limitations & Future Work

- Synthetic data generation and forecasting function training introduce additional overhead.
- The evaluation model $\mathcal{G}$ relies on GPT-4o-mini, which may introduce evaluation bias.
- The proxy user model $U$ may not fully capture real human decision-making behavior.
- A gap remains between the temperature parameter $t$ and actual user behavior.
- The effectiveness of the forecasting function across more LLM backbones has not been explored.
- The triggering mechanism for query reformulation in Stage 3 could be made more fine-grained.

## Related Work & Insights

- Extending confidence calibration from classification tasks to long-form RAG generation is broadly instructive.
- Document re-ranking is not equivalent to calibration—re-ranking optimizes ranking metrics, whereas CalibRAG optimizes decision correctness.
- CalibRAG can be used in a complementary fashion alongside other robust RAG methods such as SelfRAG.
- The framework has direct practical value for the reliable deployment of LLMs in high-stakes decision-making scenarios (e.g., healthcare, law).

## Rating

- Novelty: ⭐⭐⭐⭐ Introducing calibration into RAG-based decision-making is a novel perspective, though the technical approach is relatively straightforward.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers multiple datasets, retrievers, and domains with comprehensive ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Problem motivation is clearly articulated and mathematical formalization is rigorous, though notation is occasionally dense.
- Value: ⭐⭐⭐⭐ Offers practical value for improving the reliability of RAG systems, particularly in high-stakes decision-making contexts.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Retrieval-Augmented Generation for Reliable Interpretation of Radio Regulations](retrieval-augmented_generation_for_reliable_interpretation_of_radio_regulations.md)
- [\[NeurIPS 2025\] Chain-of-Retrieval Augmented Generation (CoRAG)](chain-of-retrieval_augmented_generation.md)
- [\[NeurIPS 2025\] Cooperative Retrieval-Augmented Generation for Question Answering: Mutual Information Exchange and Ranking by Contrasting Layers](cooperative_retrieval-augmented_generation_for_question_answering_mutual_informa.md)
- [\[NeurIPS 2025\] Windsock is Dancing: Adaptive Multimodal Retrieval-Augmented Generation](windsock_is_dancing_adaptive_multimodal_retrieval-augmented_generation.md)
- [\[NeurIPS 2025\] Benchmarking Retrieval-Augmented Multimodal Generation for Document Question Answering](benchmarking_retrievalaugmented_multimodal_generation_for_do.md)

</div>

<!-- RELATED:END -->
