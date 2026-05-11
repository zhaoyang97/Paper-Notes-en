---
title: >-
  [Paper Note] Efficient and Effective Internal Memory Retrieval for LLM-Based Healthcare Prediction
description: >-
  [ACL 2026][Medical Imaging][Internal memory retrieval] This paper proposes the K2K framework, which treats the FFN parameter space of LLMs as a retrievable knowledge base. Clinical knowledge is injected via LoRA…
tags:
  - "ACL 2026"
  - "Medical Imaging"
  - "Internal memory retrieval"
  - "FFN key-value memory"
  - "healthcare prediction"
  - "knowledge injection"
  - "cross-attention reranking"
date: 2026-05-08
content_hash: 78861497ee410ea3
---

# Efficient and Effective Internal Memory Retrieval for LLM-Based Healthcare Prediction

**Conference**: ACL 2026
**arXiv**: [2604.07659](https://arxiv.org/abs/2604.07659)
**Code**: [https://anonymous.4open.science/r/K2K-2390/](https://anonymous.4open.science/r/K2K-2390/)
**Area**: Medical Imaging
**Keywords**: Internal memory retrieval, FFN key-value memory, healthcare prediction, knowledge injection, cross-attention reranking

## TL;DR
This paper proposes the K2K framework, which treats the FFN parameter space of LLMs as a retrievable knowledge base. Clinical knowledge is injected via LoRA, activation-guided probes enable precise retrieval, and cross-attention reranking adaptively integrates multi-source internal knowledge — achieving state-of-the-art healthcare prediction without external retrieval latency.

## Background & Motivation

**Background**: LLMs have demonstrated significant potential in the medical domain, yet deployment is hindered by hallucinations and the lack of fine-grained medical context. Retrieval-Augmented Generation (RAG) is the dominant knowledge grounding strategy, with existing methods retrieving from knowledge graphs, unstructured documents, or self-generated knowledge.

**Limitations of Prior Work**: Conventional RAG pipelines exhibit two critical bottlenecks: (1) injecting external knowledge through input prompts expands context length, increasing inference cost and limiting scalability; (2) constructing high-quality retrievers remains challenging — supervised retrieval requires large quantities of annotated query–context pairs, while structured retrieval relies on expensive graph search or overly simplified heuristics. These are unacceptable in time-sensitive clinical environments.

**Key Challenge**: Relevant medical knowledge must be accessed both accurately and rapidly, yet the latency and complexity introduced by external retrieval conflict with the demands of real-time clinical decision-making. Prior work has shown that FFN layers implicitly store factual knowledge (the key-value memory interpretation), but directly retrieving internal keys using raw queries is insufficiently accurate — keys retrieved by different queries are highly similar, and probe representations lack discriminability.

**Goal**: To design a framework that retrieves knowledge directly from the internal parameter space of LLMs, eliminating the latency and complexity of external retrieval.

**Key Insight**: This work builds on the FFN key-value memory interpretation of Geva et al. — columns of the FFN weight matrix $W_1$ serve as "keys" encoding semantic patterns, while rows of $W_2$ serve as "values" storing corresponding knowledge. After injecting domain knowledge via LoRA, these keys constitute a searchable internal knowledge base.

**Core Idea**: Medical knowledge is "written" into the LLM parameter space via LoRA, then activation-guided probes precisely retrieve relevant internal keys, which are subsequently integrated through cross-attention.

## Method

### Overall Architecture
K2K operates in three stages: (1) **Internal memory construction** — building document-level and graph-level memories via a domain-adapted model and LoRA injection, respectively; (2) **Activation-guided probe construction** — identifying critical tokens and rare anomalous features to enhance the discriminability of queries; (3) **Cross-attention reranking** — dynamically integrating and reweighting multi-source internal knowledge. The input is a longitudinal EHR diagnostic code sequence; the output is a binary classification prediction (mortality/readmission).

### Key Designs

1. **Internal Memory Construction**

    - **Function**: Encodes external clinical knowledge into the FFN parameter space of the LLM to form a retrievable internal memory.
    - **Mechanism**: Document-level memory uses the FFN $W_1$ matrix of a domain-adapted LLM (e.g., BioMistral) as keys $K_{\text{doc}}^l$. Graph-level memory linearizes medical knowledge graph triples into text (e.g., "The relationship between [head] and [tail] is [relation]"), injects them via LoRA fine-tuning, and uses the LoRA $A_1 B_1$ matrix as graph keys $K_{\text{graph}}^l$. The two memory sources provide complementary unstructured and structured knowledge.
    - **Design Motivation**: External retrieval requires maintaining separate retrieval systems and processing long contexts, whereas encoding knowledge into parameters eliminates retrieval latency at inference time. The low-rank nature of LoRA makes knowledge injection efficient without compromising the model's pre-existing capabilities.

2. **Activation-Guided Probe Construction**

    - **Function**: Constructs highly discriminative query probes to enable precise retrieval of relevant knowledge from internal memory.
    - **Mechanism**: Given the hidden states $H_w$ of the input sequence, rather than applying simple mean pooling (which diffuses attention), the framework computes a diagonal approximation of the Mahalanobis distance for each token, $\phi_j^w \approx \sqrt{\sum_d \frac{(h_{j,d}^w - \bar{z}_d^w)^2}{\sigma_d^2}}$, as contextual activation weights. After normalization, these serve as a soft attention distribution; the enhanced probe $Q_w = \sum_j \alpha_j^w \cdot h_j^w$ is obtained by weighted aggregation of token vectors, emphasizing semantic anchor tokens.
    - **Design Motivation**: Standard mean pooling causes different queries to produce highly similar probes, lacking discriminability. The Mahalanobis distance accounts for per-dimension variance and is more sensitive to deviations along low-variance directions, enabling the identification of genuinely important anomalous features.

3. **Cross-Attention Reranking**

    - **Function**: Dynamically integrates and reweights multi-source knowledge retrieved from internal memory, enabling task-aware knowledge selection.
    - **Mechanism**: The input representation is divided into multiple windows; for each window, the enhanced probe $Q_w^+$ retrieves the top-$k$ keys from both document and graph memories. A cross-attention (CA) mechanism reranks these keys to produce document knowledge $H_{\text{doc}}^w$ and graph knowledge $H_{\text{graph}}^w$. Both are pooled and concatenated, then merged with the original input representation for final prediction via an MLP.
    - **Design Motivation**: Retrieved internal keys are latent and ungrounded, lacking explicit provenance. Cross-attention provides an adaptive, task-aware mechanism for dynamically selecting and weighting the most relevant knowledge.

### Loss & Training
Standard cross-entropy loss is used for classification. LoRA is applied during the knowledge injection phase. Evaluation is conducted on the MIMIC-III and MIMIC-IV datasets, with patient-ID-based splits to prevent data leakage.

## Key Experimental Results

### Main Results

| Method | MIMIC-III Mort Avg | MIMIC-III Read Avg | MIMIC-IV Mort Avg | MIMIC-IV Read Avg |
|--------|-------------------|-------------------|-------------------|-------------------|
| KARE (Prev. SOTA) | Lower | Lower | Lower | Lower |
| Standard RAG | Moderate | Moderate | Moderate | Moderate |
| K2K (BioMistral) | Higher | Higher | Higher | Higher |
| K2K (Meditron3) | **Highest** | **Highest** | **Highest** | **Highest** |

### Ablation Study

| Configuration | Performance | Notes |
|---------------|-------------|-------|
| Full K2K | Best | Complete framework |
| w/o graph memory | Degraded | Contribution of structured knowledge |
| w/o activation guidance | Degraded | Importance of probe discriminability |
| w/o cross-attention reranking | Degraded | Necessity of dynamic integration |
| Mean-pooling probe | Significantly degraded | Validates the advantage of Mahalanobis weighting |

### Key Findings
- K2K achieves state-of-the-art performance across four benchmark datasets with substantially higher retrieval efficiency than KARE and prompt-based methods.
- Meditron3-Qwen2.5-7B outperforms BioMistral-7B, indicating that base model capability plays an important role in internal memory quality.
- Activation-guided probes significantly improve retrieval precision over standard mean pooling, validating the effectiveness of Mahalanobis distance weighting.
- Combining document-level and graph-level memories outperforms either source alone, confirming their complementarity.

## Highlights & Insights
- **Parameters as a Knowledge Base**: The FFN key-value memory interpretation is transformed from a theoretical insight into a practical system that retrieves knowledge directly from the parameter space, eliminating external retrieval latency. This paradigm generalizes to any scenario requiring rapid knowledge access.
- **Mahalanobis Distance-Enhanced Probes**: Weighting token importance by per-dimension variance more effectively identifies semantic anchor tokens than simple Euclidean distance or mean pooling — a broadly applicable representation enhancement technique.
- **LoRA as a Knowledge Injection Tool**: Rather than using LoRA to fine-tune task performance, this work employs it to "write" new knowledge into the parameter space, opening a novel application direction for LoRA.

## Limitations & Future Work
- Retrieved internal keys are latent and ungrounded, limiting interpretability — it is unclear what specific knowledge each retrieved key corresponds to.
- The framework depends on domain-adapted base models (BioMistral/Meditron3); performance on general-purpose LLMs is unknown.
- Validation is limited to classification tasks over ICD code sequences; generative tasks have not been explored.
- The quality of knowledge graph linearization depends on the formulation of the triples.

## Related Work & Insights
- **vs. KARE**: KARE combines document retrieval with shortest-path graph search, but incurs high graph traversal costs. K2K encodes all knowledge into parameters, achieving zero retrieval latency at inference time.
- **vs. Standard RAG**: RAG expands context length and increases inference cost; K2K avoids context bloat through internal retrieval.
- **vs. RETRO**: RETRO also employs window-based retrieval and cross-attention, but retrieves from an external database. K2K adapts this architecture to internal parameter retrieval.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ — Translating the FFN key-value memory interpretation into a practical internal retrieval system is a highly original contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Four datasets, multiple baselines, and comprehensive ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ — Method description is clear with a good balance between theory and practice.
- **Value**: ⭐⭐⭐⭐ — Provides a low-latency knowledge grounding solution for time-sensitive clinical settings.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Benchmarking and Enabling Efficient Chinese Medical Retrieval via Asymmetric Encoders](benchmarking_and_enabling_efficient_chinese_medical_retrieval_via_asymmetric_enc.md)
- [\[AAAI 2026\] BiCA: Effective Biomedical Dense Retrieval with Citation-Aware Hard Negatives](../../AAAI2026/medical_imaging/bica_effective_biomedical_dense_retrieval_with_citation-aware_hard_negatives.md)
- [\[ACL 2026\] HCFD: A Benchmark for Audio Deepfake Detection in Healthcare](hcfd_a_benchmark_for_audio_deepfake_detection_in_healthcare.md)
- [\[AAAI 2026\] Towards Effective and Efficient Context-aware Nucleus Detection in Histopathology Whole Slide Images](../../AAAI2026/medical_imaging/towards_effective_and_efficient_context-aware_nucleus_detection_in_histopatholog.md)
- [\[ICLR 2026\] Scaling with Collapse: Efficient and Predictable Training of LLM Families](../../ICLR2026/medical_imaging/scaling_with_collapse_efficient_and_predictable_training_of_llm_families.md)

</div>

<!-- RELATED:END -->
