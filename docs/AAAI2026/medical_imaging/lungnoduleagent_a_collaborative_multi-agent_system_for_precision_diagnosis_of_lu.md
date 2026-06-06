---
title: >-
  [Paper Note] LungNoduleAgent: A Collaborative Multi-Agent System for Precision Diagnosis of Lung Nodules
description: >-
  [AAAI 2026][Medical Imaging][lung nodule diagnosis] This paper proposes LungNoduleAgent, the first collaborative multi-agent system for lung nodule analysis. It simulates the clinical workflow through a three-stage pipel…
tags:
  - "AAAI 2026"
  - "Medical Imaging"
  - "lung nodule diagnosis"
  - "multi-agent collaboration"
  - "vision-language model"
  - "CT report generation"
  - "malignancy grading"
date: 2026-05-08
content_hash: 4608c48a623a9905
---

# LungNoduleAgent: A Collaborative Multi-Agent System for Precision Diagnosis of Lung Nodules

**Conference**: AAAI 2026
**arXiv**: [2511.21042](https://arxiv.org/abs/2511.21042)  
**Code**: [GitHub](https://github.com/ImYangC7/LungNoduleAgent)  
**Area**: Medical Imaging / Multi-Agent Systems
**Keywords**: lung nodule diagnosis, multi-agent collaboration, vision-language model, CT report generation, malignancy grading

## TL;DR

This paper proposes LungNoduleAgent, the first collaborative multi-agent system for lung nodule analysis. It simulates the clinical workflow through a three-stage pipeline—Nodule Spotter, Simulated Radiologist, and Doctor Agent System—and substantially outperforms mainstream VLMs (GPT-4o, Claude 3.7 Sonnet) and medical agents (MedAgent-Pro) on CT report generation and malignancy grading tasks.

## Background & Motivation

Lung cancer is the leading cause of cancer-related mortality worldwide; early detection and accurate diagnosis are critical for improving patient outcomes. CT scanning is the primary modality for identifying lung nodules, yet radiologists must examine CT images slice by slice and synthesize professional knowledge into diagnostic reports—a time-consuming process subject to inter-observer variability due to subjective judgment.

Deep learning has achieved notable progress in lung nodule detection, classification, and grading, but faces three major bottlenecks: (1) **poor interpretability**—models yield high performance metrics but opaque reasoning, limiting clinical adoption; (2) **limited generalizability**—performance degrades on out-of-distribution data; (3) **task specificity**—most methods address only detection or classification, lacking comprehensive diagnostic capability.

General-purpose VLMs (e.g., GPT-4o, Claude 3.7 Sonnet) possess strong multimodal understanding and generalization, but struggle in specialized medical settings due to insufficient domain-specific training. Medical VLMs (e.g., MedGemma, Med-R1) improve reasoning via fine-tuning on medical data, yet suffer from **insufficient fine-grained visual perception**, making quantitative lung nodule analysis difficult; moreover, they rely primarily on parametric knowledge rather than evidence-based reasoning. Existing medical agent systems (e.g., MedAgent-Pro, MDAgent) achieve 75–80% accuracy on general medical tasks but only **40–50% on lung-cancer-specific tasks**, lacking fine-grained nodule analysis and adequate pathological knowledge.

**Core Idea**: Simulate the authentic clinical workflow of a radiologist by decomposing the diagnostic process into three specialized stages—**detection → description → reasoning**—each handled by a dedicated module, with evidence-based reasoning supported by a medical knowledge graph and multi-agent discussion.

## Method

### Overall Architecture

LungNoduleAgent processes a pulmonary CT volume $V$ sequentially through three modules:

1. **Nodule Spotter**: Localizes lung nodule regions and outputs the final mask $M$.
2. **Simulated Radiologist**: Generates a localized CT report based on the detected regions.
3. **Doctor Agent System (DAS)**: Performs malignancy reasoning using the report, images, and medical knowledge, and outputs the final diagnosis $\mathcal{FD}$.

### Key Designs

1. **Nodule Spotter**:

    - **Function**: Precisely localizes lung nodule regions in CT slices.
    - **Mechanism**: A three-tier cascade design—
        - **Mixture of Experts (MoE)**: Multiple specialized detection base models process each CT slice in parallel; each expert is proficient in different nodule characteristics and independently generates a mask $m$.
        - **Mask Clustering**: Inter-mask distance is defined as $d(m_i, m_j) = 1 - \text{IoU}(m_i, m_j)$; DBSCAN clusters spatially overlapping masks into the same group, removes outliers, and binarizes the averaged mask per cluster at a threshold of 0.5.
        - **Judging Panel**: $N_{VLM}$ independent VLMs simultaneously assess the validity of each candidate nodule, producing a binary decision $\text{Sign}(\mathcal{V}_j)$ and confidence $C_j$; weighted voting $\text{Score}(M_g) = \sum_j \text{Sign}(\mathcal{V}_j) \times C_j$ determines acceptance.
    - **Design Motivation**: Single detection models inevitably produce false positives; MoE improves robustness, DBSCAN clustering denoises candidates, and VLM voting simulates peer review for further filtering.

2. **Simulated Radiologist**:

    - **Function**: Generates detailed CT reports for the localized nodule regions.
    - **Mechanism**:
        - **Focal Prompting Mechanism**: Focal crops of the image and mask are generated while retaining surrounding context. Both the full image and the focal crop are encoded by a local visual backbone, then fused via gated cross-attention to incorporate global context. Sequence concatenation captures cross-slice nodule dynamics.
        - **MedPrompt**: A medically tailored prompt that directs the model to focus on the annotated region, use anatomically accurate terminology, produce formatted clinical reports, and avoid speculative content. The VLM ultimately processes the fused features: $\mathcal{O}_{vlm} = \text{VLM}(\text{MedPrompt}, \Theta_{\text{volume}})$.
    - **Design Motivation**: General-purpose VLMs lack fine-grained regional perception; focal prompting amplifies nodule details while preserving global context, and MedPrompt constrains output toward clinical professionalism.

3. **Doctor Agent System (DAS)**:

    - **Function**: Performs malignancy reasoning based on CT reports and nodule images.
    - **Mechanism**:
        - **Medical Graph RAG**: A knowledge graph $\mathcal{G} = \text{GraphConstruct}(\mathcal{D})$ is constructed from authoritative pathology literature; community-level summaries $\mathcal{S}$ are extracted; given query $Q$, the VLM generates answers $\mathcal{A} = \text{VLM}(\mathcal{S}, Q, \mathcal{N})$ by combining summaries with nodule images, enabling evidence-based diagnostic reasoning.
        - **Multi-Agent Roundtable**: $K$ reasoning agents independently analyze the inputs and produce initial diagnoses $O_i^{(1)} = \text{Agent}_i(I, \text{Report})$; upon disagreement, each agent revises its position based on others' views $O_i^{(t)} = \text{Revise}(O_i^{(t-1)}, \{O_j^{(t-1)}\}_{j \neq i})$; a summarizer agent consolidates intermediate results and iterates until consensus is reached.
    - **Design Motivation**: Single-model diagnoses are prone to bias; multi-agent discussion emulates clinical case conferencing, and the knowledge graph supplements the specialized pathological knowledge absent from VLM parameters.

4. **Memory Module**:

    - **Function**: Serves as the system's central storage component, managing nodule images, size measurements, CT reports, and multi-agent dialogues and summaries.
    - **Design Motivation**: Reduces information processing complexity and supports inter-agent information sharing.

### Loss & Training

The system does not involve end-to-end training but instead coordinates multiple pretrained models. For evaluation, the paper introduces the **LungDLC-score**: clinically relevant yes/no questions are constructed for each nodule (positive QA verifying feature presence; negative QA detecting hallucinated details), and the mean accuracy serves as a metric for CT report quality assessment.

## Key Experimental Results

### Main Results — CT Report Generation

| Method | PrivateA LungDLC | PrivateB LungDLC | LIDC-IDRI LungDLC |
|--------|-----------------|-----------------|-------------------|
| GPT-4o | 71.8 | 68.2 | 73.2 |
| Claude 3.7 Sonnet | 65.6 | 64.3 | 65.9 |
| MedGemma-27B | 75.6 | 76.2 | 75.2 |
| MedAgent-Pro | 70.4 | 68.8 | 70.0 |
| **LungNoduleAgent** | **81.9** | **80.3** | **83.5** |

LungNoduleAgent surpasses the strongest competitor by 6.3, 4.1, and 8.3 points on the three datasets, respectively.

### Main Results — Malignancy Grading

| Method | PrivateA Acc(%) | PrivateB Acc(%) | LIDC-IDRI Acc(%) |
|--------|----------------|----------------|------------------|
| GPT-4o | 46.2 | 41.2 | 64.1 |
| MedGemma-27B | 62.3 | 60.2 | 73.2 |
| MedAgent-Pro | 60.2 | 60.1 | 72.6 |
| **LungNoduleAgent** | **86.7** | **81.2** | **89.1** |

Compared to MedGemma, accuracy improves by 15.9–24.4%. When using Qwen-2.5-VL-7B as the backbone, the improvement on malignancy grading reaches up to +54.4% accuracy.

### Ablation Study

Module combination ablation (PrivateA dataset):

| NS | SR | DAS | Acc(%) | LungDLC |
|----|----|----|--------|---------|
| - | - | ✓ | 62.1 | 57.9 |
| ✓ | ✓ | - | 66.7 | 88.9 |
| ✓ | - | ✓ | 75.1 | 67.3 |
| ✓ | ✓ | ✓ | **86.7** | **88.9** |

Nodule Spotter internal ablation:

| MoE | Clustering | Judge Panel | mAP(%) | F1 |
|-----|-----------|-------------|--------|-----|
| ✓ | - | - | 67.1 | 0.643 |
| ✓ | ✓ | - | 71.6 | 0.713 |
| ✓ | ✓ | ✓ | **79.3** | **0.835** |

### Key Findings

- The optimal number of agents is five; beyond this, performance becomes unstable, likely due to redundancy or inconsistency.
- Higher detection accuracy (greater IoU with ground truth) consistently leads to better malignancy grading, demonstrating the advantage of evidence-based quantitative analysis over empirical qualitative judgment.
- GPT-4o achieves higher LLM-scores but lower LungDLC-scores, indicating that general-purpose metrics favor fluency over medical accuracy.
- General-purpose VLMs frequently generate incorrect anatomical locations or density descriptions for nodules, whereas LungNoduleAgent accurately describes lobar location, density, margins, and morphological features.

## Highlights & Insights

- The stage-wise design that mirrors the clinical workflow is highly pragmatic; each module has clearly defined capability boundaries, facilitating localization and improvement of failure cases.
- The weighted voting mechanism of the Judging Panel simulates peer review and represents an elegant quality control solution for multi-model ensemble methods.
- The combination of a medical knowledge graph with multi-agent discussion enables evidence-based diagnosis rather than relying purely on parametric memory, which is key to improving trustworthiness.
- The LungDLC-score metric is a noteworthy contribution; its attribute-level QA formulation avoids dependence on ground-truth reference reports.

## Limitations & Future Work

- The system relies on cascaded calls to multiple large VLMs, resulting in high inference cost and latency that fall short of real-time clinical deployment requirements.
- The three modules are arranged in a sequential pipeline, so errors propagate downstream—a missed detection by the Nodule Spotter cannot be recovered by subsequent modules.
- Validation is limited to lung nodules; adapting the system to other lesion types would require redesigning MedPrompt and the knowledge graph.
- The convergence criterion ("consensus") for the multi-agent discussion is not rigorously defined, leaving open the possibility of spurious agreement.
- The private datasets are relatively small in scale (1,616 and 386 slices), providing insufficient evidence for generalizability.

## Related Work & Insights

- MedAgent-Pro's tool-augmented agent paradigm (integrating specialized DL modules such as nnUNet) is a precursor to the Nodule Spotter design in this work.
- GraphRAG provides the foundational framework for medical knowledge retrieval; the community-level summary retrieval adopted here is adapted to handle the broad query demands of medical scenarios.
- Multi-agent discussion (e.g., ChatEval, CAMEL) has been validated in general NLP for improving reasoning quality; this work is the first to systematically introduce this paradigm into lung nodule diagnosis.
- The focal prompting concept from Describe Anything Model is effectively transferred to region-level description in medical imaging.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Resp-Agent: An Agent-Based System for Multimodal Respiratory Sound Generation and Disease Diagnosis](../../ICLR2026/medical_imaging/resp-agent_an_agent-based_system_for_multimodal_respiratory_sound_generation_and.md)
- [\[AAAI 2026\] MAMA-Memeia! Multi-Aspect Multi-Agent Collaboration for Depressive Symptoms Identification in Memes](mama-memeia_multi-aspect_multi-agent_collaboration_for_depressive_symptoms_ident.md)
- [\[AAAI 2026\] Refine and Align: Confidence Calibration through Multi-Agent Interaction in VQA](refine_and_align_confidence_calibration_through_multi-agent_interaction_in_vqa.md)
- [\[AAAI 2026\] GIIM: Graph-based Learning of Inter- and Intra-view Dependencies for Multi-view Medical Image Diagnosis](giim_graph-based_learning_of_inter-_and_intra-view_dependencies_for_multi-view_m.md)
- [\[AAAI 2026\] MAPI-GNN: Multi-Activation Plane Interaction Graph Neural Network for Multimodal Medical Diagnosis](mapi-gnn_multi-activation_plane_interaction_graph_neural_network_for_multimodal_.md)

</div>

<!-- RELATED:END -->
