---
title: >-
  [Paper Note] Efficient and Effective Internal Memory Retrieval for LLM-Based Healthcare Prediction
description: >-
  [ACL 2026][Medical NLP][Internal Memory Retrieval] This paper proposes the K2K framework, which treats the FFN parameter space of LLMs as a retrievable knowledge base. By injecting clinical knowledge via LoRA…
tags:
  - "ACL 2026"
  - "Medical NLP"
  - "Internal Memory Retrieval"
  - "FFN Key-Value Memory"
  - "Healthcare Prediction"
  - "Knowledge Injection"
  - "Cross-Attention Reranking"
date: 2026-05-08
content_hash: 842c5eb8f265726c
---

# Efficient and Effective Internal Memory Retrieval for LLM-Based Healthcare Prediction

**Conference**: ACL 2026  
**arXiv**: [2604.07659](https://arxiv.org/abs/2604.07659)  
**Code**: [https://anonymous.4open.science/r/K2K-2390/](https://anonymous.4open.science/r/K2K-2390/)  
**Area**: Medical NLP  
**Keywords**: Internal Memory Retrieval, FFN Key-Value Memory, Healthcare Prediction, Knowledge Injection, Cross-Attention Reranking

## TL;DR
This paper proposes the K2K framework, which treats the FFN parameter space of LLMs as a retrievable knowledge base. By injecting clinical knowledge via LoRA, constructing precise retrieval probes through activation guidance, and adaptively integrating knowledge via cross-attention reranking, it achieves SOTA in healthcare prediction without external retrieval latency.

## Background & Motivation

**Background**: LLMs have demonstrated significant potential in the medical domain, but deployment faces challenges such as hallucinations and a lack of fine-grained medical context. RAG is a mainstream strategy for knowledge grounding, where existing methods retrieve from knowledge graphs, unstructured documents, or self-generated knowledge.

**Limitations of Prior Work**: Traditional RAG pipelines suffer from two critical bottlenecks: (1) Injecting external knowledge through input prompts extends the context length, increasing inference costs and limiting scalability; (2) Building high-quality retrievers remains difficult, as supervised retrieval requires a large number of annotated query-context pairs, and structured retrieval relies on expensive graph searches or overly simplified heuristics. These are unacceptable in time-sensitive clinical environments.

**Key Challenge**: The need for accurate and rapid access to relevant medical knowledge conflicts with the latency and complexity introduced by external retrieval. Previous studies indicate that FFN layers implicitly store factual knowledge (key-value memory interpretation), but directly retrieving internal keys using raw queries is inaccurate—different queries retrieve highly similar keys, and probe representations lack discriminative power.

**Goal**: Design a framework to retrieve knowledge directly from the internal parameter space of LLMs, avoiding the latency and complexity of external retrieval.

**Key Insight**: Utilize the FFN key-value memory interpretation by Geva et al.—the columns of the FFN weight matrix $W_1$ act as "keys" storing semantic patterns, while rows of $W_2$ act as "values" storing corresponding knowledge. After injecting domain knowledge via LoRA, these keys become a searchable internal knowledge base.

**Core Idea**: "Write" medical knowledge into the LLM parameter space via LoRA, then precisely retrieve relevant internal keys using activation-guided probes, and finally integrate them dynamically through cross-attention.

## Method

### Overall Architecture
K2K consists of three steps: (1) Internal memory construction—building document-level and graph-level memories through domain-adapted models and LoRA injection respectively; (2) Activation-guided probe construction—identifying key tokens and scarce abnormal features to enhance query discriminatability; (3) Cross-attention reranking—dynamically integrating and re-weighting multi-source internal knowledge. The input is a longitudinal EHR diagnostic code sequence, and the output is a binary classification prediction (mortality/readmission).

### Key Designs

1. **Internal Memory Construction**:

    - **Function**: Encode external clinical knowledge into the LLM's FFN parameter space to form retrievable internal memory.
    - **Mechanism**: Document-level memory utilizes the FFN $W_1$ matrix of domain-adapted LLMs (e.g., BioMistral) as keys $K_{\text{doc}}^l$; graph-level memory linearizes Medical Knowledge Graph triplets into text (e.g., "The relationship between [head] and [tail] is [relation]") and injects them via LoRA fine-tuning, where the LoRA $A_1 B_1$ matrix serves as graph keys $K_{\text{graph}}^l$. These memories provide complementary unstructured and structured knowledge.
    - **Design Motivation**: External retrieval requires maintaining independent systems and processing long contexts, whereas encoding knowledge into parameters eliminates retrieval latency during inference. The low-rank nature of LoRA makes knowledge injection efficient without compromising the model's original capabilities.

2. **Activation-Guided Probe Construction**:

    - **Function**: Build highly discriminative query probes to ensure precise retrieval of relevant knowledge from internal memory.
    - **Mechanism**: For the hidden states $H_w$ of the input sequence, instead of using simple mean pooling (which disperses attention), the diagonal approximation of the Mahalanobis distance for each token is calculated as $\phi_j^w \approx \sqrt{\sum_d \frac{(h_{j,d}^w - \bar{z}_d^w)^2}{\sigma_d^2}}$, serving as context activation weights. After normalization, these act as a soft attention distribution to aggregate token vectors into an enhanced probe $Q_w = \sum_j \alpha_j^w \cdot h_j^w$, emphasizing semantic anchor tokens.
    - **Design Motivation**: Standard mean pooling leads to highly similar probes for different queries (lacking discriminatability). Mahalanobis distance considers the variance of each dimension and is more sensitive to deviations in low-variance directions, identifying truly important abnormal features.

3. **Cross-Attention Reranking**:

    - **Function**: Dynamically integrate and re-weight multi-source knowledge retrieved from internal memory to achieve task-aware knowledge selection.
    - **Mechanism**: The input representation is divided into multiple windows, and the enhanced probe $Q_w^+$ of each window retrieves top-k keys from document and graph memories. These keys are reranked via a cross-attention (CA) mechanism to obtain document knowledge $H_{\text{doc}^w}$ and graph knowledge $H_{\text{graph}}^w$. Both are pooled, concatenated, and merged with the original input representation for the final prediction via an MLP.
    - **Design Motivation**: Retrieved internal keys are latent and ungrounded, lacking explicit sources. Cross-attention provides an adaptive, task-aware way to dynamically select and weight the most relevant knowledge.

### Loss & Training
Standard cross-entropy loss is used for classification. LoRA is employed during the knowledge injection phase. Evaluation is conducted on MIMIC-III and MIMIC-IV datasets, partitioned by patient ID to prevent data leakage.

## Key Experimental Results

### Main Results

| Method | MIMIC-III Mort Avg | MIMIC-III Read Avg | MIMIC-IV Mort Avg | MIMIC-IV Read Avg |
|------|-------------------|-------------------|-------------------|-------------------|
| KARE (Prev. SOTA) | Lower | Lower | Lower | Lower |
| Standard RAG | Medium | Medium | Medium | Medium |
| K2K (BioMistral) | Higher | Higher | Higher | Higher |
| K2K (Meditron3) | **Highest** | **Highest** | **Highest** | **Highest** |

### Ablation Study

| Config | Effect | Description |
|------|------|------|
| Full K2K | Optimal | Full framework |
| w/o Graph Memory | Decrease | Contribution of structured knowledge |
| w/o Activation Guidance | Decrease | Importance of probe discriminatability |
| w/o CA Reranking | Decrease | Necessity of dynamic integration |
| Using Mean Pooling Probes | Significant Decrease | Validates advantage of Mahalanobis weighting |

### Key Findings
- K2K achieves SOTA on four benchmarks, with retrieval efficiency far exceeding KARE and prompt-based methods.
- Meditron3-Qwen2.5-7B outperforms BioMistral-7B, indicating that base model capability significantly impacts internal memory quality.
- Activation-guided probes significantly improve retrieval precision compared to standard mean pooling, validating the effectiveness of Mahalanobis distance.
- Using both document-level and graph-level memories outperforms single sources, as they provide complementary information.

## Highlights & Insights
- **Parameters as Knowledge Base**: Transforms the theoretically insightful FFN key-value memory interpretation into a practical system, retrieving knowledge directly in parameter space and eliminating external retrieval latency. This idea can be generalized to any scenario requiring rapid knowledge access.
- **Mahalanobis Distance Enhanced Probes**: Weighting token importance by considering per-dimension variance identifies semantic anchors better than simple Euclidean distance or mean pooling. This is a generic representation enhancement technique.
- **LoRA as a Knowledge Injection Tool**: Rather than using LoRA to fine-tune task performance, it is used to "write" new knowledge into the parameter space. This usage opens new application directions for LoRA.

## Limitations & Future Work
- Internal keys are latent and ungrounded, lacking interpretability—it is unclear what specific knowledge retrieved keys correspond to.
- Reliance on domain-adapted base models (BioMistral/Meditron3); performance on general LLMs is unknown.
- Validated only on classification tasks for ICD code sequences; generative tasks remain untested.
- The quality of knowledge graph linearization depends on the phrasing of triplets.

## Related Work & Insights
- **vs KARE**: KARE combines document retrieval and KG shortest paths but incurs high graph search costs. K2K encodes all knowledge into parameters, resulting in zero latency during inference.
- **vs Standard RAG**: RAG increases inference cost by extending context length, whereas K2K avoids context bloat through internal retrieval.
- **vs RETRO**: RETRO also uses window-based retrieval and cross-attention but retrieves from external databases. K2K adapts this architecture for internal parameter retrieval.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Transforming FFN key-value interpretation into a practical internal retrieval system is a highly novel idea.
- Experimental Thoroughness: ⭐⭐⭐⭐ Four datasets, multiple baselines, and comprehensive ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clear method description, good integration of theory and practice.
- Value: ⭐⭐⭐⭐ Provides a low-latency knowledge grounding solution for time-sensitive clinical scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] BiCA: Effective Biomedical Dense Retrieval with Citation-Aware Hard Negatives](../../AAAI2026/medical_nlp/bica_effective_biomedical_dense_retrieval_with_citation-aware_hard_negatives.md)
- [\[ACL 2026\] ReMedi: Reasoner for Medical Clinical Prediction](remedi_reasoner_for_medical_clinical_prediction.md)
- [\[ACL 2026\] IndicMedDialog: A Parallel Multi-Turn Medical Dialogue Dataset for Accessible Healthcare in Indic Languages](indicmeddialog_a_parallel_multi-turn_medical_dialogue_dataset_for_accessible_hea.md)
- [\[ACL 2026\] CURA: Clinical Uncertainty Risk Alignment for Language Model-Based Risk Prediction](cura_clinical_uncertainty_risk_alignment_for_language_model-based_risk_predictio.md)
- [\[ACL 2026\] HypEHR: Hyperbolic Modeling of Electronic Health Records for Efficient Question Answering](hypehr_hyperbolic_modeling_of_electronic_health_records_for_efficient_question_a.md)

</div>

<!-- RELATED:END -->
