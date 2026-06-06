---
title: >-
  [Paper Note] RA-RRG: Multimodal Retrieval-Augmented Radiology Report Generation with Key Phrase Extraction
description: >-
  [ACL 2026][Medical Imaging][Radiology report generation] The RA-RRG framework is proposed to extract clinical key phrases from radiology reports using LLMs to construct a retrieval database. Given a chest X-ray image…
tags:
  - "ACL 2026"
  - "Medical Imaging"
  - "Radiology report generation"
  - "retrieval-augmented generation"
  - "key phrase extraction"
  - "hallucination suppression"
  - "multi-view"
date: 2026-05-08
content_hash: 8655eb92051d59ae
---

# RA-RRG: Multimodal Retrieval-Augmented Radiology Report Generation with Key Phrase Extraction

**Conference**: ACL 2026  
**arXiv**: [2504.07415](https://arxiv.org/abs/2504.07415)  
**Code**: [GitHub](https://github.com/deepnoid-ai/RA-RRG)  
**Area**: Medical Imaging / Radiology Report Generation  
**Keywords**: Radiology report generation, retrieval-augmented generation, key phrase extraction, hallucination suppression, multi-view

## TL;DR
The RA-RRG framework is proposed to extract clinical key phrases from radiology reports using LLMs to construct a retrieval database. Given a chest X-ray image, relevant phrases are retrieved and input into an LLM to generate reports. This suppresses hallucinations without LLM fine-tuning, requires only 18 GPU hours for training, and achieves SOTA results on CheXbert metrics.

## Background & Motivation

**Background**: Automated Radiology Report Generation (RRG) is a crucial direction for alleviating the workload of radiologists. Multimodal LLMs (e.g., LLaVA-Rad, MAIRA) have demonstrated the ability to generate reports directly from chest X-rays but require substantial computational resources and large-scale fine-tuning data.

**Limitations of Prior Work**: (1) High training costs for MLLMs (>200 GPU hours) limit clinical deployment; (2) Retrieval-augmented methods (e.g., CXR-RePaiR) retrieve complete sentences or reports, but since multiple findings often co-occur in the same sentence, naive retrieval may introduce information irrelevant or even contradictory to the current image; (3) Reports often contain comparative statements regarding previous examinations (e.g., "unchanged", "improved"), which constitute "comparative hallucinations" in a single-image setting.

**Key Challenge**: Retrieval-augmented methods require retrieval units with sufficient granularity to avoid co-occurrence contamination, yet excessive segmentation may lose clinical context. A balance between granularity and informational integrity is essential.

**Goal**: To design a retrieval-augmented RRG framework that does not require LLM fine-tuning, can retrieve fine-grained, hallucination-free clinical key phrases, and generates accurate radiology reports.

**Key Insight**: Utilize RadGraph to extract the knowledge graph structure of reports, then use an LLM to refine it into minimal clinically meaningful phrases while explicitly excluding comparative statements.

**Core Idea**: Refine RadGraph outputs into hallucination-free key phrases via LLM → train a multimodal retriever to match images with phrases → use an LLM to expand retrieved phrases into coherent reports, without fine-tuning the LLM throughout the process.

## Method

### Overall Architecture
RA-RRG consists of three stages: (1) Key phrase extraction—RadGraph parses the report structure, and an LLM (Llama 70B) refines it into key phrases by removing comparative hallucinations; (2) Multimodal retriever training—dual visual encoders (XrayDINOv2 + XrayCLIP) extract visual features, and a DETR decoder outputs semantic embeddings aligned with MPNet text embeddings; (3) Report generation—retrieved phrases are input into GPT-4o to generate coherent reports without LLM fine-tuning.

### Key Designs

1.  **LLM-assisted key phrase extraction**:

    - **Function**: Decomposes radiology reports into minimal clinically meaningful phrases while removing hallucination-inducing content.
    - **Mechanism**: First, entities and relations are extracted from the FINDINGS section using RadGraph to construct RadGraph phrases. Then, Llama 70B jointly processes the RadGraph output and original report to refine them into key phrases while excluding comparative statements (e.g., unchanged, improved). The training set averages 7.16 key phrases per image, totaling 243,064 unique phrases.
    - **Design Motivation**: Pure RadGraph outputs may produce fragmented graph structures and do not handle comparative hallucinations; pure LLM processing of raw text might miss domain-specific clinical details. The combination of both is complementary.

2.  **Multimodal retriever with dual encoders + DETR decoder**:

    - **Function**: Predicts semantic embeddings from images to match a key phrase vector database.
    - **Mechanism**: The visual side fuses XrayDINOv2 (self-supervised features) and XrayCLIP (vision-language aligned features) via channel concatenation to obtain complementary visual representations. A DETR decoder parallelly decodes $N=50$ query embeddings, where each embedding determines activation via a selection classifier, and semantic embeddings are generated through a three-layer FFN. The text side uses a frozen MPNet to encode phrases with NEFTune-style noise to prevent overfitting. Training involves Hungarian matching + phrase matching loss + in-batch semantic contrastive loss.
    - **Design Motivation**: A single visual encoder cannot simultaneously capture self-supervised fine-grained features and cross-modal alignment features; DETR-style set prediction is naturally suited for "one image to many phrases" retrieval.

3.  **Zero-training LLM report generation**:

    - **Function**: Consolidates the retrieved phrase list into a coherent radiology report.
    - **Mechanism**: Retrieved key phrases and task instructions are fed into GPT-4o to generate a complete report. Since phrases are already filtered for hallucinations, the LLM only performs linguistic organization rather than clinical judgment. The framework naturally extends to multi-view (frontal + lateral) by merging phrases retrieved from each image.
    - **Design Motivation**: Avoids the high cost of LLM fine-tuning while leveraging strong language generation capabilities to organize fragmented phrases into coherent text.

### Loss & Training
The total loss is $\mathcal{L} = \sum_b \mathcal{L}_{PM}(y^b, \hat{y}^b) + \lambda \mathcal{L}_{SC}(E)$, where the phrase matching loss $\mathcal{L}_{PM}$ uses Hungarian assignment + distribution-balanced classification loss + cosine similarity loss. The in-batch semantic contrastive loss $\mathcal{L}_{SC}$ adopts a CLIP-style symmetric cross-entropy with soft targets to avoid penalizing semantically similar non-matching pairs. $\lambda = 0.1$. Visual and text encoders are frozen; only the DETR decoder is trained.

## Key Experimental Results

### Main Results
MIMIC-CXR Single-view RRG (FINDINGS section):

| Type | Model | CheXbert micro-F1 | RadGraph F1 | ROUGE-L |
|------|------|--------------------|-------------|---------|
| Generation | LLaVA-Rad | 57.3 | - | 30.6 |
| Generation | M4CXR | 58.1 | 21.7 | 28.4 |
| Retrieval | MCA-RG | - | - | 30.0 |
| **Retrieval** | **Ours** | **62.3** | **24.3** | **30.7** |

### Ablation Study

| Configuration | CheXbert micro-F1 | RadGraph F1 |
|------|--------------------|-------------|
| RadGraph phrases only | 59.1 | 22.8 |
| LLM key phrases (no comparative filtering) | 60.5 | 23.4 |
| LLM key phrases (with comparative filtering) | **62.3** | **24.3** |
| Single encoder (CLIP only) | 58.7 | 22.1 |
| Dual encoder (CLIP + DINOv2) | **62.3** | **24.3** |

### Key Findings
- Comparative hallucination filtering contributes significantly (micro-F1: 60.5 → 62.3), proving the necessity of excluding expressions like "unchanged/improved".
- Dual encoder fusion provides a 3.6% micro-F1 improvement over single encoders, indicating that DINOv2 and CLIP features are complementary.
- RA-RRG requires only 18 GPU hours for training (vs. MLLM >200 GPU hours) and outperforms all MLLMs on CheXbert metrics.
- The framework extends naturally to multi-view RRG, where multi-view results show further improvements.

## Highlights & Insights
- The design of key phrases as retrieval units finds an excellent balance in granularity—finer than sentences to avoid co-occurrence contamination, yet coarser than entities to preserve clinical context. This design can be generalized to other domains requiring fine-grained retrieval.
- LLMs assume different roles in two stages: knowledge refinement in the extraction stage (Llama 70B) and linguistic organization in the generation stage (GPT-4o). Neither stage requires fine-tuning, maximizing the off-the-shelf value of LLMs.
- The explicit definition and handling of comparative hallucinations is a highly practical contribution, as such hallucinations are prevalent in radiology but were ignored by previous methods.

## Limitations & Future Work
- Dependency on commercial APIs (GPT-4o) for report generation poses cost and privacy concerns that limit clinical deployment.
- RadGraph itself may produce incomplete graph structures on complex reports.
- Recall of key phrase retrieval is limited by the phrase coverage of the training set—rare findings may lack matching phrases.
- Future work could replace GPT-4o with open-source LLMs or train the retriever and a small generative model end-to-end.

## Related Work & Insights
- **vs CXR-RePaiR**: Retrieves full reports/sentences, leading to co-occurrence information contamination; RA-RRG retrieves minimal clinical phrases for higher precision.
- **vs MAIRA-1/LLaVA-Rad**: These MLLMs require large-scale fine-tuning, whereas RA-RRG achieves lower costs through retrieval and frozen LLMs.

## Rating
- Novelty: ⭐⭐⭐⭐ Combination of key phrase extraction, dual encoder retrieval, and zero-training LLM generation is innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation on two datasets with thorough ablation and hallucination analysis.
- Writing Quality: ⭐⭐⭐⭐ Methodology is clearly described with intuitive architecture diagrams.
- Value: ⭐⭐⭐⭐ Provides a practical solution for radiology report generation in resource-constrained scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] MARCH: Multi-Agent Radiology Clinical Hierarchy for CT Report Generation](march_multi-agent_radiology_clinical_hierarchy_for_ct_report_generation.md)
- [\[ACL 2026\] HeteroRAG: A Heterogeneous Retrieval-Augmented Generation Framework for Medical Vision Language Tasks](heterorag_a_heterogeneous_retrieval-augmented_generation_framework_for_medical_v.md)
- [\[CVPR 2026\] OraPO: Oracle-educated Reinforcement Learning for Data-efficient and Factual Radiology Report Generation](../../CVPR2026/medical_imaging/orapo_oracle-educated_reinforcement_learning_for_data-efficient_and_factual_radi.md)
- [\[ICML 2026\] CAME-Grad: The Double Dilemma in Multi-Task Radiology Report Generation — A Gradient Dynamics Analysis and Solution](../../ICML2026/medical_imaging/the_double_dilemma_in_multi-task_radiology_report_generation_a_gradient_dynamics.md)
- [\[AAAI 2026\] Expert-Guided Prompting and Retrieval-Augmented Generation for Emergency Medical Service Question Answering](../../AAAI2026/medical_imaging/expert-guided_prompting_and_retrieval-augmented_generation_for_emergency_medical.md)

</div>

<!-- RELATED:END -->
