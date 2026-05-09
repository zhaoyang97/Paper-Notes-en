---
title: >-
  [Paper Note] RA-RRG: Multimodal Retrieval-Augmented Radiology Report Generation with Key Phrase Extraction
description: >-
  [ACL 2026][Medical Imaging][radiology report generation] This paper proposes RA-RRG, a framework that leverages an LLM to extract clinically relevant key phrases from radiology reports and construct a retrieval database. Given a chest X-ray image, relevant phrases are retrieved and fed to an LLM for report generation—without any LLM fine-tuning—effectively suppressing hallucinations. The approach requires only 18 GPU hours of training and achieves state-of-the-art performance on CheXbert metrics.
tags:
  - ACL 2026
  - Medical Imaging
  - radiology report generation
  - retrieval-augmented generation
  - key phrase extraction
  - hallucination suppression
  - multi-view
date: 2026-05-08
content_hash: 87a0b84741007cca
---

# RA-RRG: Multimodal Retrieval-Augmented Radiology Report Generation with Key Phrase Extraction

**Conference**: ACL 2026
**arXiv**: [2504.07415](https://arxiv.org/abs/2504.07415)
**Code**: [GitHub](https://github.com/deepnoid-ai/RA-RRG)
**Area**: Medical Imaging / Radiology Report Generation
**Keywords**: radiology report generation, retrieval-augmented generation, key phrase extraction, hallucination suppression, multi-view

## TL;DR
This paper proposes RA-RRG, a framework that leverages an LLM to extract clinically relevant key phrases from radiology reports and construct a retrieval database. Given a chest X-ray image, relevant phrases are retrieved and fed to an LLM for report generation—without any LLM fine-tuning—effectively suppressing hallucinations. The approach requires only 18 GPU hours of training and achieves state-of-the-art performance on CheXbert metrics.

## Background & Motivation

**Background**: Automated radiology report generation (RRG) is a promising direction for alleviating radiologists' workload. Multimodal LLMs (e.g., LLaVA-Rad, MAIRA) have demonstrated the capability to directly generate reports from chest X-rays, but demand substantial computational resources and large-scale fine-tuning data.

**Limitations of Prior Work**: (1) MLLM-based methods incur high training costs (>200 GPU hours), limiting clinical deployment; (2) retrieval-augmented methods (e.g., CXR-RePaiR) retrieve complete sentences or reports, yet multiple findings often co-occur within a single sentence, so naive retrieval may introduce information irrelevant or even contradictory to the current image; (3) reports frequently contain comparative statements referencing prior studies (e.g., "unchanged," "improved"), which constitute "comparative hallucinations" in single-image settings.

**Key Challenge**: Retrieval-augmented methods require sufficiently fine-grained retrieval units to avoid co-occurrence contamination, yet overly granular segmentation risks losing clinical context. A balance between granularity and informational completeness is needed.

**Goal**: Design a retrieval-augmented RRG framework that requires no LLM fine-tuning, retrieves fine-grained, hallucination-free clinical key phrases, and generates accurate radiology reports.

**Key Insight**: RadGraph is used to extract the knowledge graph structure of reports, which an LLM then refines into minimally clinically meaningful phrases while explicitly excluding comparative expressions.

**Core Idea**: An LLM refines RadGraph outputs into hallucination-free key phrases → a multimodal retriever is trained to match images with phrases → an LLM expands the retrieved phrases into coherent reports, with no LLM fine-tuning required throughout.

## Method

### Overall Architecture
RA-RRG consists of three stages: (1) **Key Phrase Extraction**—RadGraph parses the report structure, and an LLM (Llama 70B) refines the output into key phrases free of comparative hallucinations; (2) **Multimodal Retriever Training**—a dual visual encoder (XrayDINOv2 + XrayCLIP) extracts visual features, a DETR decoder produces semantic embeddings aligned with MPNet text embeddings; (3) **Report Generation**—retrieved phrases are fed to GPT-4o to generate coherent reports without any LLM fine-tuning.

### Key Designs

1. **LLM-Assisted Key Phrase Extraction**:

    - **Function**: Decompose radiology reports into minimally clinically meaningful phrases while removing hallucination-inducing content.
    - **Mechanism**: RadGraph first extracts entities and relations from the FINDINGS section to construct RadGraph phrases; Llama 70B then takes both the RadGraph output and the original report as joint input, refining them into key phrases while explicitly excluding comparative statements (e.g., "unchanged," "improved"). On average, 7.16 key phrases are associated with each training image, yielding 243,064 unique phrases in total.
    - **Design Motivation**: Raw RadGraph output may produce fragmented graph structures and does not address comparative hallucinations; processing raw text with an LLM alone may miss domain-specific clinical details. The joint input is complementary.

2. **Dual-Encoder + DETR Decoder Multimodal Retriever**:

    - **Function**: Predict semantic embeddings from images to match a key phrase vector database.
    - **Mechanism**: On the visual side, XrayDINOv2 (self-supervised features) and XrayCLIP (vision-language aligned features) are fused via channel concatenation to yield complementary visual representations. A DETR decoder decodes $N=50$ query embeddings in parallel; each embedding passes through a selection classifier to determine activation, and semantic embeddings are generated via a three-layer FFN. On the text side, a frozen MPNet encodes key phrases with NEFTune-style noise added to prevent overfitting. Training employs Hungarian matching + phrase matching loss + in-batch semantic contrastive loss.
    - **Design Motivation**: A single visual encoder cannot simultaneously capture fine-grained self-supervised features and cross-modal alignment features; DETR-style set prediction naturally suits the one-image-to-many-phrases retrieval scenario.

3. **Training-Free LLM Report Generation**:

    - **Function**: Integrate a list of retrieved phrases into a coherent radiology report.
    - **Mechanism**: Retrieved key phrases and task instructions are fed together to GPT-4o to generate a complete report. Since phrases have already undergone hallucination filtering, the LLM performs linguistic organization rather than clinical judgment. The same framework naturally extends to multi-view settings (frontal + lateral): phrases are retrieved separately from each image and then merged as input.
    - **Design Motivation**: Avoid the high cost of LLM fine-tuning while leveraging the LLM's powerful language generation capability to organize fragmented phrases into coherent text.

### Loss & Training
The total loss is $\mathcal{L} = \sum_b \mathcal{L}_{PM}(y^b, \hat{y}^b) + \lambda \mathcal{L}_{SC}(E)$, where the phrase matching loss $\mathcal{L}_{PM}$ employs Hungarian assignment + distribution-balanced classification loss + cosine similarity loss. The in-batch semantic contrastive loss $\mathcal{L}_{SC}$ adopts CLIP-style symmetric cross-entropy with soft targets to avoid penalizing semantically similar non-matched pairs. $\lambda = 0.1$. Visual and text encoder parameters are frozen; only the DETR decoder is trained.

## Key Experimental Results

### Main Results
MIMIC-CXR single-view RRG (FINDINGS section):

| Type | Model | CheXbert micro-F1 | RadGraph F1 | ROUGE-L |
|------|-------|-------------------|-------------|---------|
| Generative | LLaVA-Rad | 57.3 | - | 30.6 |
| Generative | M4CXR | 58.1 | 21.7 | 28.4 |
| Retrieval | MCA-RG | - | - | 30.0 |
| **Retrieval** | **RA-RRG** | **62.3** | **24.3** | **30.7** |

### Ablation Study

| Configuration | CheXbert micro-F1 | RadGraph F1 |
|--------------|-------------------|-------------|
| RadGraph phrases only | 59.1 | 22.8 |
| LLM key phrases (w/o comparative filtering) | 60.5 | 23.4 |
| LLM key phrases (w/ comparative filtering) | **62.3** | **24.3** |
| Single encoder (CLIP only) | 58.7 | 22.1 |
| Dual encoder (CLIP + DINOv2) | **62.3** | **24.3** |

### Key Findings
- Comparative hallucination filtering contributes substantially (micro-F1: 60.5 → 62.3), demonstrating the necessity of excluding expressions such as "unchanged/improved."
- The dual-encoder fusion improves micro-F1 by 3.6% over the single encoder, confirming the complementarity of DINOv2 and CLIP features.
- RA-RRG requires only 18 GPU hours of training (vs. >200 GPU hours for MLLMs) while surpassing all MLLMs on CheXbert metrics.
- The framework extends naturally to multi-view RRG, with multi-view results yielding further improvements.

## Highlights & Insights
- Using key phrases as retrieval units strikes a favorable balance in granularity—finer than sentences to avoid co-occurrence contamination, yet coarser than individual entities to preserve clinical context. This design is generalizable to any domain requiring fine-grained retrieval.
- The LLM plays distinct roles in two stages: knowledge refinement during extraction (Llama 70B) and linguistic organization during generation (GPT-4o), neither of which requires fine-tuning, maximizing the plug-and-play value of LLMs.
- The explicit definition and handling of comparative hallucinations is a practically valuable contribution—such hallucinations are pervasive in radiology yet have been overlooked by prior methods.

## Limitations & Future Work
- Reliance on a commercial API (GPT-4o) for report generation raises cost and privacy concerns that limit clinical deployment.
- RadGraph itself may produce incomplete graph structures for complex reports.
- The recall of key phrase retrieval is bounded by phrase coverage in the training set—rare findings may have no matching phrases.
- Future work could replace GPT-4o with open-source LLMs, or jointly train the retriever with a compact generative model end-to-end.

## Related Work & Insights
- **vs. CXR-RePaiR**: Retrieves complete reports or sentences, suffering from co-occurrence information contamination; RA-RRG retrieves minimal clinical phrases, achieving greater precision.
- **vs. MAIRA-1/LLaVA-Rad**: These MLLMs require large-scale fine-tuning; RA-RRG achieves lower cost via retrieval combined with frozen LLMs.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The combination of key phrase extraction, dual-encoder retrieval, and training-free LLM generation is innovative.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive evaluation on two datasets with thorough ablations, including hallucination analysis.
- **Writing Quality**: ⭐⭐⭐⭐ Method description is clear and the architecture diagram is intuitive.
- **Value**: ⭐⭐⭐⭐ Provides a practical solution for radiology report generation under resource-constrained settings.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] MARCH: Multi-Agent Radiology Clinical Hierarchy for CT Report Generation](march_multi-agent_radiology_clinical_hierarchy_for_ct_report_generation.md)
- [\[CVPR 2026\] OraPO: Oracle-educated Reinforcement Learning for Data-efficient and Factual Radiology Report Generation](../../CVPR2026/medical_imaging/orapo_oracle_rl_radiology_report_generation.md)
- [\[AAAI 2026\] Expert-Guided Prompting and Retrieval-Augmented Generation for Emergency Medical Service Question Answering](../../AAAI2026/medical_imaging/expert-guided_prompting_and_retrieval-augmented_generation_for_emergency_medical.md)
- [\[ACL 2026\] AROMA: Augmented Reasoning Over a Multimodal Architecture for Virtual Cell Genetic Perturbation Modeling](aroma_augmented_reasoning_over_a_multimodal_architecture_for_virtual_cell_geneti.md)
- [\[ACL 2026\] Region-Grounded Report Generation for 3D Medical Imaging: A Fine-Grained Dataset and Graph-Enhanced Framework](region-grounded_report_generation_for_3d_medical_imaging_a_fine-grained_dataset_.md)

</div>

<!-- RELATED:END -->
