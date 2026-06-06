---
title: >-
  [Paper Note] HeteroRAG: A Heterogeneous Retrieval-Augmented Generation Framework for Medical Vision Language Tasks
description: >-
  [ACL 2026][Medical Imaging][Heterogeneous RAG] HeteroRAG constructs the MedAtlas knowledge base with 2.7 million image-text pairs and five types of corpora. It decomposes medical multimodal RAG into three components—ModC…
tags:
  - "ACL 2026"
  - "Medical Imaging"
  - "Heterogeneous RAG"
  - "Modality-specific CLIP"
  - "Multi-Corpora Query Generation"
  - "Preference Fine-tuning"
  - "Med-LVLM"
date: 2026-05-08
content_hash: f3f933e4a1004c70
---

# HeteroRAG: A Heterogeneous Retrieval-Augmented Generation Framework for Medical Vision Language Tasks

**Conference**: ACL 2026  
**arXiv**: [2508.12778](https://arxiv.org/abs/2508.12778)  
**Code**: https://github.com/Jack-ZC8/HeteroRAG-Med  
**Area**: Medical Imaging / RAG / Multimodal  
**Keywords**: Heterogeneous RAG, Modality-specific CLIP, Multi-Corpora Query Generation, Preference Fine-tuning, Med-LVLM

## TL;DR
HeteroRAG constructs the MedAtlas knowledge base with 2.7 million image-text pairs and five types of corpora. It decomposes medical multimodal RAG into three components—ModCLIPs trained by modality for report retrieval, MQG for corpus-customized query generation, and HKPT preference fine-tuning to align cross-modal and multi-source knowledge. This allows a 7B model to consistently outperform open-source Med-LVLMs with 4-5× the parameters across 11 datasets.

## Background & Motivation

**Background**: Med-LVLMs (Medical Large Vision-Language Models) are widely used for multimodal diagnosis and clinical decision support. However, poor factuality and reliability severely restrict real-world deployment. Mainstream mitigation strategies involve multimodal RAG: using medical CLIPs (FactMM-RAG, RULE, MMed-RAG) to retrieve relevant reports as context, or using zero-shot query rewriting (ChatCAD+, MIRA) to retrieve documents.

**Limitations of Prior Work**: (1) **Poor report retrieval**—existing medical CLIPs are only fine-tuned on a few public training sets, resulting in low retrieval recall (Radiology recall@5 ~30). Retrieved reports are often irrelevant, causing generation "contamination"; (2) **Worse document retrieval**—medical corpora (research papers, textbooks, clinical guidelines, knowledge graphs) vary in linguistic style. Using a unified multimodal query fails to retrieve the correct information, and MIRA's zero-shot rewriting cannot adapt to the characteristics of each corpus; (3) **Knowledge misalignment**—even when retrieval is successful, models often ignore the image to copy the report, or conversely, stick to internal knowledge while ignoring external evidence.

**Key Challenge**: The two types of knowledge sources in medical RAG (image-text reports vs. purely textual heterogeneous corpora) require fundamentally different retrieval strategies, yet existing works force them into the same retriever. Furthermore, even with accurate retrieval, neither cross-modality nor multi-source alignment has been explicitly addressed.

**Goal**: (1) Construct a truly "large and broad" medical multimodal knowledge base, expanding the report database to the million-level and the corpora to 5 types; (2) Design "targeted" retrievers for reports and documents respectively; (3) Use a unified preference learning framework to solve both "modality neglect" and "knowledge utilization/robustness" alignment issues.

**Key Insight**: (a) Alignment distributions of different modalities (radiology, ophthalmology, pathology) vary significantly → train a ModCLIP for each modality; (b) Different corpora (PubMed, Wikipedia, textbooks, guidelines, knowledge graphs) require different query styles → train a Multi-Corpora Query Generator (MQG) to explicitly generate customized queries per corpus; (c) Both cross-modality and multi-source alignment can be addressed by constructing counterfactual preference pairs of "should use vs. should not use."

**Core Idea**: Treat "heterogeneity" as a first-class citizen—heterogeneous knowledge base + heterogeneous retrievers (ModCLIPs for images + MQG for text) + heterogeneous knowledge preference tuning (HKPT managing both modality and multi-source alignment).

## Method

### Overall Architecture
HeteroRAG connects three modules: (1) **MedAtlas Knowledge Base**—1.10M Radiology / 0.11M Ophthalmology / 1.51M Pathology image-text pairs + Research/Wiki/Book/Guideline/Graph text corpora; (2) **Heterogeneous Retrieval Module (HRM)**—ModCLIPs (BiomedCLIPs trained per modality) for reports, and MQG generating customized query sets $Q=\{(i,j,q_j^i)\}$ for each multimodal query $(v,t)$ for documents; (3) **Heterogeneous Knowledge Preference Tuning (HKPT)**—constructs cross-modality alignment $\mathcal{D}_{cm}$ and multi-source alignment $\mathcal{D}_{mk}$ preference pairs, optimized via a DPO-style loss $\mathcal{L}_{\text{HKPT}}$.

### Key Designs

1. **Modality-specific CLIPs (ModCLIPs)**:
    - **Function**: Provide each medical modality with a dedicated, well-aligned image-text retriever instead of using a "universal medical CLIP" for all modalities.
    - **Mechanism**: Starting from BiomedCLIP initialization, each modality is independently trained using all remaining samples (Radiology 1.10M, Oph 0.11M, Pathology 1.51M) after setting aside 2000 dev/test samples for contrastive learning. The training volume is an order of magnitude larger than previous works. The three ModCLIPs achieve image→text recall@5 of 79.40 / 47.55 / 77.35, more than doubling FactMM-RAG (44.25 Rad) and MMed-RAG (19.25 Oph / 30.20 Pat).
    - **Design Motivation**: Visual statistics of medical images differ drastically across modalities (gray-scale X-ray vs. color fundus vs. pathology H&E). Feature spaces of a single CLIP are insufficiently differentiated. Partitioning by modality allows for more accurate training on larger intra-modality data while avoiding cross-modal noise interference.

2. **Multi-Corpora Query Generator (MQG)**:
    - **Function**: Generate "targeted" retrieval queries for each corpus based on the multimodal question, matching the linguistic characteristics of the corpus.
    - **Mechanism**: For each query $(v,t)$, an expert MLLM (Lingshu-32B AWQ) exploratory generates 6 candidate queries for each source. The same expert then judges if the documents retrieved by each query "support the reference answer." This VLM-as-a-judge system achieves acc=0.836 and F1=0.855 on 500 human-verified samples. Queries judged as supporting form $Q_w$, while others form $Q_l$. MQG is trained in two stages: first SFT $\mathcal{L}_{\text{SFT}}=-\mathbb{E}\log\mathcal{M}_\theta(Q_w\mid v,t)$, then DPO $\mathcal{L}_{\text{DPO}}=-\log\sigma(\beta\log\frac{\mathcal{M}_\theta(Q_w\mid v,t)}{\mathcal{M}_{\text{ref}}}-\beta\log\frac{\mathcal{M}_\theta(Q_l\mid v,t)}{\mathcal{M}_{\text{ref}}})$.
    - **Design Motivation**: PubMed papers require scientific terminology; Wikipedia requires general encyclopedic phrasing; clinical guidelines require disease/procedure names; and knowledge graphs require "term, relation" structures. Using a single query for all corpora is ineffective. MQG explicitly parameterizes and trains corpus-specific query generation, outperforming zero-shot rewriting.

3. **Heterogeneous Knowledge Preference Tuning (HKPT)**:
    - **Function**: Use a unified DPO loss to simultaneously address three types of "knowledge misalignment": ignoring images, inability to use external knowledge, and insufficient robustness to noisy knowledge.
    - **Mechanism**: Construct two sets of preference pairs—**Cross-Modality $\mathcal{D}_{cm}$**: Retrieve the most dissimilar image from the same modality training set as $v^*$. If $\mathcal{M}(v,t,K)=y$, $\mathcal{M}(v^*,t)\ne y$, and $\mathcal{M}(v^*,t,K)=y$, it indicates the model answers correctly even with the wrong image, proving it "copies K without looking at the image." $(v,t,K)$ is set as chosen and $(v^*,t,K)$ as rejected to force visual attention. **Multi-Source $\mathcal{D}_{mk}$**: For $k\in\{\{k_r\},\{k_d\},\{k_r,k_d\}\}$ in two steps: (a) **Utilization**—if $\mathcal{M}(v,t,K)=y$ but $\mathcal{M}(v,t,K\setminus k)\ne y$, $k$ is critical; chosen=correct answer with $K$. (b) **Robustness**—if $\mathcal{M}(v,t,K\setminus k)=y$ but $\mathcal{M}(v,t,K)\ne y$, $k$ is noise; chosen="correct answer ignoring $k$" as $y_w$. All are optimized via $\mathcal{L}_{\text{HKPT}}=-\mathbb{E}\log\sigma(\beta\log\frac{\mathcal{M}_{\theta'}(y_w\mid x_w)}{\mathcal{M}_{\text{ref}'}}-\beta\log\frac{\mathcal{M}_{\theta'}(y_l\mid x_l)}{\mathcal{M}_{\text{ref}'}})$.
    - **Design Motivation**: Prior methods like MMed-RAG only perform modality alignment, while K-LLaVA types only perform document utilization. HKPT unified these failure modes into chosen/rejected pairs using counterfactual data construction, solved in a single DPO pass.

### Loss & Training
ModCLIPs: Single-modality image-text contrastive learning. MQG: SFT + DPO stages. Med-LVLM: Unified HKPT fine-tuning with $\mathcal{L}_{\text{HKPT}}$. Base model: Lingshu-7B.

## Key Experimental Results

### Main Results: Medical VQA (Lingshu-7B base, Accuracy ↑)

| Method | Retrieval | VQA-RAD | SLAKE | OMVQA-Rad† | DME-VQA | OMVQA-Oph† | PathMMU | PathVQA | Quilt-VQA† |
|------|------|---------|-------|------------|---------|------------|---------|---------|------------|
| Original | — | 72.79 | 83.65 | 74.92 | 81.92 | 80.83 | 57.36 | 77.38 | 49.27 |
| FactMM-RAG (Report) | R | 76.84 | 83.89 | 75.58 | 81.92 | 81.50 | 73.58 | 91.98 | 69.68 |
| MMed-RAG (Report) | R | 75.74 | 86.06 | 76.33 | 80.70 | 79.08 | 68.06 | 85.67 | 67.35 |
| K-LLaVA (Doc) | D | 77.21 | 84.62 | 76.00 | 88.48 | 83.75 | 73.75 | 87.76 | 61.81 |
| MIRA (R+D) | R+D | 76.84 | 84.38 | 76.58 | 87.95 | 82.50 | 74.25 | **92.10** | 68.80 |
| **HeteroRAG (R+D)** | R+D | **81.99** | **87.50** | **80.42** | **88.56** | **86.00** | **75.59** | 90.83 | **72.89** |

(†=OOD datasets, no training split.)

For report generation, HeteroRAG achieves SOTA on most metrics for MIMIC-CXR / IU-Xray / Harvard-FairVLMed. Figure 3 shows HeteroRAG-7B surpasses 4-5× larger models such as HuatuoGPT-V-34B and Lingshu-32B.

### Ablation Study: Knowledge Sources and SFT vs. HKPT

| Configuration | OMVQA-Rad | OMVQA-Oph | Quilt-VQA |
|------|------|------|------|
| Original | 74.92 | 80.83 | 49.27 |
| SFT | 75.00 | 82.17 | 63.85 |
| **HeteroRAG** | **80.42** | **86.00** | **72.89** |
| w/o Reports | 75.08 | 81.33 | 62.97 |
| w/o Doc | 74.17 | 79.08 | 53.94 |
| w/o Research | 79.42 | 84.33 | 68.80 |
| w/o Wiki | 77.75 | 84.00 | 67.64 |
| w/o Book | 75.17 | 80.92 | 67.06 |
| w/o Guideline | 79.33 | 84.58 | 69.68 |
| w/o Graph | 78.58 | 83.25 | 66.47 |

Retriever Comparison (image→text Recall@5): ModCLIPs in Rad/Oph/Pat reach 79.40/47.55/77.35, significantly outperforming Prev. SOTA FactMM-RAG (44.25) and MMed-RAG (19.25 / 30.20).

## Key Findings
- **Reports and Documents are Both Essential**: "w/o Doc" drops performance from 72.89 to 53.94 on Quilt-VQA, proving report-only RAG is insufficient for OOD pathology. "w/o Reports" drops OMVQA-Rad performance, as reports are crucial for tasks tightly coupled with images.
- **HKPT Outperforms SFT by 5–9%**: On three OOD datasets, SFT offers minor gains, but HKPT shows significant improvement, proving preference tuning forces models to "look at images and use knowledge correctly" better than pure supervision.
- **Varying Contributions from Corpora**: "w/o Book" nearly returns OMVQA-Oph to baseline, indicating textbooks are vital for ophthalmology. "w/o Research" has a smaller impact, likely due to PubMed's broad but less specific coverage. "w/o Graph" shows knowledge graphs are useful for specialized terminology.
- **Small Model + Quality RAG > Large Model**: HeteroRAG-7B exceeds the 32B Lingshu on most tasks, suggesting that investing in retrieval quality and alignment is more efficient than increasing parameters.

## Highlights & Insights
- **"Modality-specific Retrievers" Taken to the Extreme**: While most medical RAG assumes a universal CLIP, HeteroRAG uses three independent ModCLIPs with million-level training data to double recall. This suggests "modality splitting" is highly effective in vertical domains.
- **MQG Formalizes Corpus Adaptation**: Instead of simple prompt tricks, HeteroRAG uses VLM-as-a-judge to generate DPO data and train a specialized generator, parameterizing the relationship between query style and corpus type.
- **Counterfactual Preference Construction in HKPT**: Constructing pairs based on "answering correctly with an unrelated image" effectively identifies knowledge copying. This paradigm for generating supervision signals from model behavior is an elegant application of RLAIF for RAG alignment.

## Limitations & Future Work
- **MedAtlas Bias**: Primarily English-based and limited to three imaging modalities. CT, MRI, Ultrasound, and ECG are not yet covered, and low-resource language corpora are missing.
- **Dependency on Lingshu-32B**: The pipeline's performance ceiling is constrained by the 32B teacher model; judge accuracy (0.836) still leaves room for error in clinical settings.
- **Self-Correction Bias**: HKPT relies on the model's own predictions, which might amplify existing base model biases.
- **Future Directions**: Expanding to more modalities and Chinese corpora; implementing ensemble judges; and end-to-end joint optimization of MQG and the retriever.

## Related Work & Insights
- **vs. FactMM-RAG / RULE / MMed-RAG**: These focused on report retrieval and modality alignment without document support. HeteroRAG adds the document side and multi-source alignment with significantly larger training data.
- **vs. K-LLaVA / MKGF**: These used document retrieval without reports and relied on single Queries. HeteroRAG's MQG provides finer granularity by customizing queries per corpus.
- **vs. MIRA**: MIRA uses both reports and documents but relies on zero-shot rewriting. HeteroRAG upgrades to an SFT+DPO trained MQG and uses HKPT for alignment, exceeding MIRA by 4–6 points on most benchmarks.

## Rating
- Novelty: ⭐⭐⭐⭐ MQG customized queries + HKPT heterogeneous alignment is a novel combination in medical RAG.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 11 datasets across 3 modalities with robust baselines and comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear system diagrams and complete HKPT algorithm pseudocode.
- Value: ⭐⭐⭐⭐ Open-sourcing MedAtlas and ModCLIPs provides foundational infrastructure for medical MM-RAG.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] RA-RRG: Multimodal Retrieval-Augmented Radiology Report Generation with Key Phrase Extraction](ra-rrg_multimodal_retrieval-augmented_radiology_report_generation_with_key_phras.md)
- [\[AAAI 2026\] Expert-Guided Prompting and Retrieval-Augmented Generation for Emergency Medical Service Question Answering](../../AAAI2026/medical_imaging/expert-guided_prompting_and_retrieval-augmented_generation_for_emergency_medical.md)
- [\[ACL 2026\] Region-Grounded Report Generation for 3D Medical Imaging: A Fine-Grained Dataset and Graph-Enhanced Framework](region-grounded_report_generation_for_3d_medical_imaging_a_fine-grained_dataset_.md)
- [\[ACL 2026\] Beyond the Leaderboard: Rethinking Medical Benchmarks for Large Language Models](beyond_the_leaderboard_rethinking_medical_benchmarks_for_large_language_models.md)
- [\[NeurIPS 2025\] RAxSS: Retrieval-Augmented Sparse Sampling for Explainable Variable-Length Medical Time Series Classification](../../NeurIPS2025/medical_imaging/raxss_retrieval-augmented_sparse_sampling_for_explainable_variable-length_medica.md)

</div>

<!-- RELATED:END -->
