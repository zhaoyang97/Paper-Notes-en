---
title: >-
  [Paper Note] ComplianceNLP: Knowledge-Graph-Augmented RAG for Multi-Framework Regulatory Gap Detection
description: >-
  [ACL 2026][Graph Learning][Regulatory Compliance] ComplianceNLP is an end-to-end financial regulatory compliance system that constructs a Knowledge Graph from 12…
tags:
  - "ACL 2026"
  - "Graph Learning"
  - "Regulatory Compliance"
  - "Knowledge-Graph-Augmented RAG"
  - "Multi-task Obligation Extraction"
  - "Medusa Speculative Decoding"
  - "Production Deployment"
date: 2026-05-08
content_hash: 9702ab60e94ec40d
---

# ComplianceNLP: Knowledge-Graph-Augmented RAG for Multi-Framework Regulatory Gap Detection

**Conference**: ACL 2026  
**arXiv**: [2604.23585](https://arxiv.org/abs/2604.23585)  
**Code**: Not provided by the paper (N/A)  
**Area**: Graph Learning / RAG / Compliance NLP  
**Keywords**: Regulatory Compliance, Knowledge-Graph-Augmented RAG, Multi-task Obligation Extraction, Medusa Speculative Decoding, Production Deployment

## TL;DR
ComplianceNLP is an end-to-end financial regulatory compliance system that constructs a Knowledge Graph from 12,847 SEC / MiFID II / Basel III regulations to enhance RAG retrieval. Combined with multi-task obligation extraction using LEGAL-BERT and threshold-based gap analysis, it outperforms GPT-4o+RAG by 3.5 points with an 87.7 F1 on RegObligation / GapBench. It achieves $2.8\times$ inference acceleration through domain knowledge distillation and Medusa speculative decoding. During 4 months of parallel operation, it processed 9,847 updates, achieving 96.0% recall and a 3.1× improvement in analyst efficiency.

## Background & Motivation
**Background**: Financial institutions track 60,000+ regulatory events annually across dozens of jurisdictions. Since the 2008 financial crisis, global banks have paid over $300B in fines and settlements. Existing commercial GRC platforms (Ascent RegTech / Wolters Kluwer OneSumX) still rely on rule-based systems and manual curation, while academic Legal NLP focuses primarily on benchmarks (LegalBench / LexGLUE / CUAD) and single-framework QA (ObliQA / DERECHA), lacking end-to-end production-ready compliance systems.

**Limitations of Prior Work**: (1) LLMs suffer from severe hallucinations on long regulatory texts, requiring trustworthy grounding; (2) existing obligation extraction systems target single frameworks (e.g., GDPR) and cannot handle multiple regulatory regimes simultaneously; (3) deontic modality (shall/must/may not) and nested cross-references in regulations are difficult to process uniformly; (4) real-time compliance monitoring requires sub-second p50 latency, but 70B teacher models are too slow for inference.

**Key Challenge**: The four requirements for compliance tasks—high precision, explainable grounding, multi-framework unified extraction, and production-grade latency—conflict with each other. Deeper models are more accurate but slower; unified frameworks are more general but prone to over-generalization; and stricter grounding compromises creativity.

**Goal**: (1) Construct a Regulatory Knowledge Graph (RKG) covering SEC, MiFID II, and Basel III; (2) jointly train a multi-task obligation extractor for NER, deontic modality classification, and cross-reference resolution; (3) design an end-to-end compliance gap analysis pipeline; (4) compress a 70B teacher model into an 8B student using domain-specific distillation and Medusa while maintaining accuracy.

**Key Insight**: The authors observe that regulatory text has extremely low entropy ($H=2.31$ bit vs. 3.87 for general text). This provides optimal conditions for a high "draft token acceptance rate" in Medusa speculative decoding, giving the combination of small-model distillation and speculative decoding an inherent advantage in this domain.

**Core Idea**: Use a Knowledge Graph for "structural reranking" to overcome RAG's weaknesses in multi-hop reasoning; use multi-task joint training with shared LEGAL-BERT representations to overcome the limitations of single extraction heads; and exploit domain entropy characteristics to maximize inference efficiency with the distillation + Medusa combination.

## Method

### Overall Architecture
A three-stage pipeline: (1) **Regulatory Ingestion and Indexing**—Three format parsers (SEC EDGAR XML / EUR-Lex HTML / BIS PDF) extract provisions to construct an RKG with 12,847 nodes and 34,219 edges, stored in a vector database; (2) **Multi-task Obligation Extraction**—A shared LEGAL-BERT encoder simultaneously outputs NER (23 financial entity types), deontic classification (Obligation / Permission / Prohibition / Recommendation), and cross-reference resolution; (3) **Compliance Gap Analysis**—Extracted structured obligations $\langle$entity, action, modality, condition, source_provision$\rangle$ are aligned and scored against internal policy clauses, then classified as Compliant / Partial Gap / Full Gap based on a threshold $\delta$ to generate gap reports.

### Key Designs

1. **KG-Augmented Hybrid Retrieval + Reranking (KG-Augmented RAG)**:

    - **Function**: Superimposes KG distance reranking on top of traditional dense + sparse retrieval to improve the hit rate for queries involving multi-hop cross-references.
    - **Mechanism**: Stage-one hybrid retrieval utilizes $s(q, d) = \alpha \cdot \text{sim}_{\text{dense}}(q, d) + (1-\alpha) \cdot \text{BM25}(q, d)$, where $\alpha = 0.7$, using a legal bi-encoder fine-tuned from MiniLM-L6-v2 on 50K regulatory paragraph pairs. Stage-two applies KG reranking to top-5 paragraphs: $s_{KG}(q, d) = \beta \cdot \text{KGScore}(q, d, \mathcal{G}) + (1-\beta) \cdot s(q, d)$, where $\beta = 0.3$. KGScore measures the graph distance in the RKG between the query source provision and the referenced provisions in the retrieved segment.
    - **Design Motivation**: Regulations contain numerous multi-hop dependencies (Provision X depends on Y, which references Z). Pure embeddings cannot capture these structural relationships. Using KG distance for soft reranking introduces structural priors without hurting recall. Ablation shows KG reranking is the single largest contributor to performance (-4.6 gap F1 when removed).

2. **Multi-task Joint Obligation Extraction**:

    - **Function**: Simultaneously outputs entity boundaries, obligation modality, and cross-references to avoid cascading errors from independently trained models.
    - **Mechanism**: A shared LEGAL-BERT encoder (further pre-trained on Pile of Law) is followed by three heads: (a) a CRF layer for 23 financial NER categories (e.g., Regulated_Entity, Capital_Requirement, Compliance_Period); (b) sentence-level deontic classification; (c) a span-pair bilinear classifier for cross-reference resolution. The joint loss is $\mathcal{L} = 0.4 \mathcal{L}_{NER} + 0.3 \mathcal{L}_{deontic} + 0.3 \mathcal{L}_{xref}$. Training involved 8,742 sentences with an inter-annotator agreement of $\kappa = 0.84$ (Fleiss).
    - **Design Motivation**: The three attributes of a regulatory obligation (who does what, mandatory level, and reference relations) are naturally coupled. Separate training wastes representation capacity. Banking NER types must exceed general PER/ORG/LOC to distinguish domain semantics like "Investment Firm" (Regulated_Entity) vs. "Reporting Entity."

3. **Domain-Specific Distillation + Medusa Speculative Decoding (Production Optimization)**:

    - **Function**: Compresses a LLaMA-3-70B teacher into an 8B student while maintaining accuracy, further accelerating to sub-second p50 latency using Medusa.
    - **Mechanism**: First, reverse KL distillation via MiniLLM is applied: $\mathcal{L}_{KD} = \text{KL}(p_{student} \| p_{teacher}) + 0.5 \mathcal{L}_{SFT}$, trained on 15K compliance instruction pairs, yielding $2.2\times$ speedup. Then, $M=3$ Medusa heads are added to the student and trained on 2.1M regulatory tokens. A key finding is that the regulatory text entropy $H = 2.31$ bit (much lower than 3.87 for C4) increases the Medusa token acceptance rate from 82.7% to 91.3%, achieving a total acceleration of $2.8\times$ (659ms p50).
    - **Design Motivation**: Real-time compliance requires sub-second latency; distillation alone is insufficient. Medusa's "draft heads" naturally have higher acceptance rates in low-entropy domains (e.g., code, regulations). Exploiting this domain characteristic doubles the benefits compared to general Medusa.

### Loss & Training
Multi-task extraction loss as defined above; distillation stage uses $\gamma = 0.5$ to balance KL and SFT; MiniCheck is used for post-processing fact-checking, improving grounding accuracy from 86.7% to 94.2%; evaluation threshold $\delta = 0.6$, deployment threshold $\delta = 0.45$ (for higher recall).

## Key Experimental Results

### Main Results (RegObligation + GapBench)

| System | NER F1 | Deon F1 | Gap Det F1 | RegQA EM | RegQA F1 |
|------|--------|---------|-----------|----------|----------|
| GPT-4o (5-shot) | 85.9 | 88.1 | 81.4 | 43.7 | 61.3 |
| GPT-4o + RAG | 88.6 | 90.5 | 84.2 | 48.1 | 66.8 |
| LLaMA-3-8B + RAG | 87.9 | 89.8 | 83.5 | 47.4 | 65.9 |
| LLaMA-3-70B (teacher) | 90.2 | 91.8 | 86.3 | 49.1 | 67.4 |
| **ComplianceNLP** | **91.3**†‡ | **92.7**†‡ | **87.7**†‡ | **52.8**†‡ | **71.9**†‡ |
| RIRAG (regulatory QA SOTA) | — | — | — | 38.9 | 54.2 |
| LEGAL-BERT (domain SOTA) | 82.1 | 84.6 | 71.3 | — | — |

ComplianceNLP improves over GPT-4o+RAG by +2.7 NER / +2.2 Deontic / **+3.5 Gap F1** / +5.1 QA F1, all statistically significant (p < 0.05). Grounding accuracy is 94.2% (vs. 85.1% for GPT-4o+RAG), correlating with human judgment at $r = 0.83$.

### Ablation Study and Latency Analysis

| Configuration | NER F1 | Gap F1 | RegQA F1 | Description |
|------|--------|--------|----------|------|
| ComplianceNLP (Full) | **91.3** | **87.7** | **71.9** | — |
| w/o KG reranking | 88.4 | 83.1 (−4.6) | 66.2 | **KG reranking removal causes the largest drop** |
| w/o multi-task | 89.1 (−2.2) | 84.9 | 69.1 | NER most affected |
| w/o MiniCheck | 91.0 | 87.2 | 71.0 | F1 stable but grounding drops from 94.2% to 86.7% |
| End-to-end (inc. error propagation) | — | 83.4 | — | 12.3% samples affected by extraction errors |

| Inference Configuration | p50 (ms) | Gain | NER Retention | Gap Retention |
|---------|---------|------|-----------|-----------|
| 70B Teacher | 1847 | $1.0\times$ | 100 | 100 |
| 8B SFT only | 897 | $2.1\times$ | 95.1 | 95.4 |
| 8B KD only | 824 | $2.2\times$ | 96.8 | 97.0 |
| 8B + Medusa (general heads) | 793 | $2.3\times$ | 96.4 | 96.7 |
| **8B + Medusa (domain heads)** | **659** | **$2.8\times$** | **98.6** | **98.1** |

### Key Findings
- **KG Reranking = Most Impactful Module**: Removing it drops gap detection F1 by 4.6 points, far exceeding the impact of removing multi-tasking (-2.8) or MiniCheck (-0.5). This proves that structural reference relationships are the most informative priors in regulations.
- **Domain Medusa Head Acceptance Rate 91.3% vs. General 82.7%**: The authors attribute this gap to regulatory text entropy ($H=2.31$ vs. 3.87). This validates the natural synergy between "low-entropy domains + speculative decoding." This approach of using domain statistical properties to guide inference optimization is highly commendable.
- **End-to-End Error Propagation**: F1 only dropped from 87.7 to 83.4. Approximately 2.1 obligations were missed per 100 pages, with 1.3 false positive alerts daily, which analysts found acceptable. This suggests that multi-stage pipelines do not need zero error at every stage, provided errors do not cascade or amplify.
- **4-Month Production Run**: Processed 9,847 updates with an estimated 96.0% recall, 90.7% precision, and $3.1\times$ analyst efficiency. This is a rare academic paper providing real-world production evidence.
- **Framework Performance Variance**: SEC (NER F1 93.1) > MiFID II (91.4) > Basel III. This reflects that SEC EDGAR standardized XML parsing is the cleanest, while Basel III’s nested conditional obligations remain the most challenging.

## Highlights & Insights
- **"Low-Entropy Domain + Speculative Decoding + Distillation" Suite**: Directly linking domain statistical property (entropy) to engineering optimization (Medusa acceptance rate) is a rare example of "domain-inference joint optimization." This approach is transferable to code generation, medical reports, and legal contracts, where higher speedups than general Medusa are expected.
- **Multi-task Shared LEGAL-BERT over Stacked Models**: A single encoder with three heads shares semantics while reducing cascading errors, offering an inherent advantage in extraction consistency over traditional pipelines (NER $\rightarrow$ Modality $\rightarrow$ Reference).
- **KG Distance as a RAG Soft Reranking Signal**: Compared to hard rule filtering or pure embeddings, using graph hop distance for weighting preserves recall while introducing structural priors. This is applicable to any RAG scenario with strong structural relationships (e.g., medical guidelines, patent citations, API dependencies).
- **MiniCheck's Role**: It doesn't change F1 but improves grounding by 8 points. This highlights the importance of evaluating "task correctness" and "output trustworthiness" separately.
- **Production Integration and Evidence**: Retrospective analysis of trust calibration, GRC integration, and drift monitoring provides valuable references for industrial NLP systems.

## Limitations & Future Work
- Current coverage is limited to SEC / MiFID II / Basel III. Extending to other jurisdictions (e.g., CBIRC in China / MAS in Singapore) requires new format parsers and NER annotations.
- The schema for 23 NER classes and 4 deontic classes is manually designed and may require restructuring for other frameworks. The cross-ref agreement ($\kappa = 0.78$) is lower than NER/deontic, indicating ambiguity in nested reference boundaries.
- KG construction relies on regex + learned linkers (91.8% accuracy). The 87.3% recall means ~13% of real references are missed, acting as a potential ceiling for deep multi-hop reasoning.
- A 18-hour "blind spot" exists for real-time updates before nightly syncs, during which the system falls back to embedding-only mode (-4.6 F1). Business-wise, this requires supplemental manual review.
- While the distilled student retains 98% performance on NER/Gap, deep reasoning tasks (e.g., multi-step clause interpretation) might suffer more significant losses, which were not fully evaluated.

## Related Work & Insights
- **vs. DERECHA (Cejas et al. 2023)**: DERECHA handles single-framework GDPR compliance and assumes pre-structured policy clauses (precision 89.1%). ComplianceNLP handles three frameworks end-to-end from raw text with 90.7% production precision, offering superior deployability.
- **vs. RIRAG / ObliQA (Bayer et al. 2025)**: These focus solely on regulatory QA without obligation extraction or gap analysis. ComplianceNLP achieves RegQA F1 71.9 vs. 54.2 (+17.7) and provides production evidence.
- **vs. Sun et al. (2025)**: Sun's work uses an eventic graph but targets single corpora, uses pure embedding retrieval without typed KGs, and assumes structured input. ComplianceNLP extends all four dimensions.
- **vs. Zagyva et al. (2025)**: Adapts the Medusa+KD pattern and uncovers the "low entropy $\rightarrow$ high acceptance" domain trait, raising the rate from 82.7% to 91.3%.

## Rating
- Novelty: ⭐⭐⭐⭐ Individual techniques (KG-RAG, Multi-tasking, Medusa-KD) are not entirely new, but the end-to-end integration for three frameworks with production evidence and the "low entropy $\rightarrow$ Medusa" insight is significant.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Academic benchmarks + error propagation + 4 months of production data + framework-wise analysis + full ablation + significance testing; exceptionally solid.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and technical details; however, it is appendix-heavy (schemas and pseudocode are relegated to the appendix).
- Value: ⭐⭐⭐⭐⭐ A rare combination of academic SOTA and industrial deployment evidence. Highly insightful for RegTech, low-entropy LLM deployment, and KG-RAG design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] M3KG-RAG: Multi-hop Multimodal Knowledge Graph-enhanced Retrieval-Augmented Generation](../../CVPR2026/graph_learning/m3kg_rag_multi_hop_multimodal_knowledge_graph_enhanced_retrieval_augmented_genera.md)
- [\[ACL 2026\] MegaRAG: Multimodal Knowledge Graph-Based Retrieval Augmented Generation](megarag_multimodal_knowledge_graph-based_retrieval_augmented_generation.md)
- [\[ACL 2026\] TagRAG: Tag-guided Hierarchical Knowledge Graph Retrieval-Augmented Generation](tagrag_tag-guided_hierarchical_knowledge_graph_retrieval-augmented_generation.md)
- [\[ACL 2026\] AutoPKG: An Automated Framework for Dynamic E-commerce Product-Attribute Knowledge Graph Construction](autopkg_an_automated_framework_for_dynamic_e-commerce_product-attribute_knowledg.md)
- [\[ACL 2026\] LegalGraphRAG: Multi-Agent Graph Retrieval-Augmented Generation for Reliable Legal Reasoning](legalgraphrag_multi-agent_graph_retrieval-augmented_generation_for_reliable_lega.md)

</div>

<!-- RELATED:END -->
