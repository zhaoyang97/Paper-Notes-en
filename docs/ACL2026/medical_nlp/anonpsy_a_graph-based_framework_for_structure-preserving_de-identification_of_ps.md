---
title: >-
  [Paper Note] Anonpsy: A Graph-Based Framework for Structure-Preserving De-identification of Psychiatric Narratives
description: >-
  [ACL 2026][Medical NLP][De-identification] The Anonpsy framework is proposed, redefining de-identification of psychiatric narratives as a graph-guided semantic rewriting problem—first converting narratives into semantic…
tags:
  - "ACL 2026"
  - "Medical NLP"
  - "De-identification"
  - "Psychiatric Narratives"
  - "Semantic Graph"
  - "Structure-Preserving"
  - "LLM Generation"
date: 2026-05-08
content_hash: 057fe46baa68e0d3
---

# Anonpsy: A Graph-Based Framework for Structure-Preserving De-identification of Psychiatric Narratives

**Conference**: ACL 2026 Findings  
**arXiv**: [2601.13503](https://arxiv.org/abs/2601.13503)  
**Code**: None  
**Area**: Medical Imaging  
**Keywords**: De-identification, Psychiatric Narratives, Semantic Graph, Structure-Preserving, LLM Generation

## TL;DR

The Anonpsy framework is proposed, redefining de-identification of psychiatric narratives as a graph-guided semantic rewriting problem—first converting narratives into semantic graphs, then performing constrained perturbations on the graph to modify identity information while maintaining clinical structure, and finally reconstructing narratives through graph-conditional generation.

## Background & Motivation

**Background**: Psychiatric narratives contain rich clinical information (symptom timelines, causal relationships, diagnostic logic), which is crucial for downstream tasks such as diagnostic prediction, but also embeds significant patient identity information.

**Limitations of Prior Work**: (1) Token-level PHI masking preserves clinical structure but results in excessively high semantic similarity, leading to a high residual re-identification risk; (2) LLM-based synthetic data creation (SDC) reduces identifiability but uncontrollably distorts clinical structures—e.g., changing persecutory delusions to grandiose delusions; (3) Both methods treat text as unstructured sequences, ignoring relationships and temporal dependencies in psychiatric narratives.

**Key Challenge**: In psychiatric narratives, identifiability stems from the narrative structure itself (specific life events, timelines) rather than just explicit identifiers. There is a need to simultaneously modify identity information and preserve clinical structure—a fundamental contradiction for text-level methods.

**Goal**: Redefine de-identification as a structure-preserving generation problem, achieving fine-grained control over intermediate graph representations.

**Key Insight**: Convert narratives into semantic graphs containing clinical entities, temporal anchors, and typed relationships, and perform constrained perturbations on the graph.

**Core Idea**: By decoupling event structure and surface text, it is possible to precisely control what is preserved and what is modified at the graph level, then regenerate a coherent narrative from the modified graph.

## Method

### Overall Architecture

The process consists of three steps: (1) Semantic graph transformation $G = \mathcal{E}(X)$—narratives are converted into semantic graphs using LLM-assisted schema-constrained transformation; (2) Graph-constrained perturbation $\tilde{G} = \mathcal{P}(G)$—modifying contextual attributes while maintaining temporal, causal, and diagnostic structures; (3) Graph-conditional text generation $\hat{X} = \mathcal{D}(\tilde{G})$—generating de-identified narratives from the modified graph.

### Key Designs

1.  **Semantic Graph Representation**:
    - **Function**: Provides an editable intermediate representation, decoupling structure from content.
    - **Mechanism**: Nodes $V$ are clinical entities (symptoms, treatments, diagnoses), and edges $E$ are typed relationships (diagnostic dependencies, causal links, temporal sequences). Extracted using schema-constrained LLMs.
    - **Design Motivation**: Graph representation makes "what to preserve and what to modify" controllable—modifying demographic attributes while maintaining symptom-diagnosis relationships.

2.  **Graph-Constrained Perturbation**:
    - **Function**: Modifies identifiable information while maintaining clinical logic.
    - **Mechanism**: Selectively modifies contextual attributes (e.g., age, profession, specific life events) while keeping temporal offsets and causal/diagnostic edges unchanged.
    - **Design Motivation**: Psychiatric diagnosis relies on the temporal development of symptoms and causal relationships, which must not be perturbed.

3.  **Graph-Conditional Text Generation**:
    - **Function**: Generates coherent de-identified narratives from the modified graph.
    - **Mechanism**: Conditions the generation of a new narrative on the modified semantic graph using a locally deployed LLM. Lower temperatures are used for schema extraction and narrative generation (stability), while higher temperatures are used for perturbation (diversity).
    - **Design Motivation**: All processes use a local LLM (gpt-oss:120b), as clinical privacy environments typically prohibit cloud APIs.

### Loss & Training

Training-free; the three operators (transformation, perturbation, generation) are implemented through prompt engineering and deterministic control flows. All LLM processing runs locally on 4 RTX A6000 GPUs.

## Key Experimental Results

### Main Results

| Method | Diagnostic Fidelity (F1) | Identifiability (cosine sim) | Explanation |
| :--- | :--- | :--- | :--- |
| PHI Masking | High | High (Dangerous) | Structure intact but traceable |
| LLM-SDC | Low (Semantic Drift) | Low | Safe but clinically distorted |
| Ours (Anonpsy) | High | Low | Balanced performance |

### Ablation Study

| Configuration | Key Metrics | Explanation |
| :--- | :--- | :--- |
| No Graph Perturbation | High Identifiability | Highly traceable if structure is unchanged |
| No Structural Constraints | Lower Diagnostic F1 | Free rewriting harms clinical meaning |
| Expert Evaluation | Low Re-identification Risk | Psychiatrists cannot trace original cases |
| GPT-5 Evaluation | Low Semantic Similarity | Automated evaluation aligns with humans |

### Key Findings
- Anonpsy achieves the optimal trade-off in the privacy protection-clinical fidelity balance.
- The intermediate graph representation makes "what is modified" transparent and controllable.
- Expert evaluation confirms that de-identified narratives maintain the original diagnostic logic.

## Highlights & Insights
- Paradigm shift from "text processing" to "structure-aware generation" for de-identification.
- Semantic graph representation allows clinical personnel to inspect and intervene in the modification process.
- Full local deployment ensures usability in real clinical environments.

## Limitations & Future Work
- Tested on only 90 psychiatric cases, representing a small scale.
- Extraction quality of semantic graphs depends on LLM capabilities.
- Currently targeted at psychiatric narratives; applicability to other clinical specialties is not yet verified.
- Future work can extend to multilingual and larger-scale clinical data.

## Related Work & Insights
- **vs PHI Masking**: Operates at the semantic level rather than the token level, more thoroughly eliminating identifiable information.
- **vs LLM-SDC**: Controls the scope of rewriting through graph constraints, avoiding uncontrolled semantic drift.
- **vs Knowledge Graph Methods**: Used for controlling generation rather than retrieval or reasoning, representing a new use for KGs.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Graph-guided de-identification is a completely new paradigm.
- Experimental Thoroughness: ⭐⭐⭐ Data scale is small but evaluation dimensions are comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Problem definition is clear and the methodology is rigorously formalized.
- Value: ⭐⭐⭐⭐⭐ Holds significant practical importance for clinical NLP privacy protection.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Region-Grounded Report Generation for 3D Medical Imaging: A Fine-Grained Dataset and Graph-Enhanced Framework](region-grounded_report_generation_for_3d_medical_imaging_a_fine-grained_dataset_.md)
- [\[ACL 2026\] Reliable Automated Triage in Spanish Clinical Notes: A Hybrid Framework for Risk-Aware HIV Suspicion Identification](reliable_automated_triage_in_spanish_clinical_notes_a_hybrid_framework_for_risk-.md)
- [\[ACL 2026\] Text-Attributed Knowledge Graph Enrichment with Large Language Models for Medical Concept Representation](text-attributed_knowledge_graph_enrichment_with_large_language_models_for_medica.md)
- [\[ACL 2026\] MultiDx: A Multi-Source Knowledge Integration Framework towards Diagnostic Reasoning](multidx_a_multi-source_knowledge_integration_framework_towards_diagnostic_reason.md)
- [\[ACL 2026\] HeteroRAG: A Heterogeneous Retrieval-Augmented Generation Framework for Medical Vision Language Tasks](heterorag_a_heterogeneous_retrieval-augmented_generation_framework_for_medical_v.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[ACL 2025\] RedactX: An LLM-Powered Framework for Automatic Clinical Data De-Identification](../../ACL2025/medical_imaging/redactor_an_llm-powered_framework_for_automatic_clinical_data_de-identification.md)
- [\[ACL 2026\] Region-Grounded Report Generation for 3D Medical Imaging: A Fine-Grained Dataset and Graph-Enhanced Framework](region-grounded_report_generation_for_3d_medical_imaging_a_fine-grained_dataset_.md)
- [\[ACL 2026\] Reliable Automated Triage in Spanish Clinical Notes: A Hybrid Framework for Risk-Aware HIV Suspicion Identification](reliable_automated_triage_in_spanish_clinical_notes_a_hybrid_framework_for_risk-.md)
- [\[ICLR 2026\] NeuroCircuitry-Inspired Hierarchical Graph Causal Attention Networks for Explainable Depression Identification](../../ICLR2026/medical_imaging/neurocircuitry-inspired_hierarchical_graph_causal_attention_networks_for_explain.md)
- [\[ACL 2026\] MultiDx: A Multi-Source Knowledge Integration Framework towards Diagnostic Reasoning](multidx_a_multi-source_knowledge_integration_framework_towards_diagnostic_reason.md)

</div>

<!-- RELATED:END -->
