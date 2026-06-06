---
title: >-
  [Paper Note] Anonpsy: A Graph-Based Framework for Structure-Preserving De-identification of Psychiatric Narratives
description: >-
  [ACL 2026][Medical Imaging][De-identification] This paper proposes Anonpsy, a framework that reformulates the de-identification of psychiatric narratives as a graph-guided semantic rewriting problem. The approach first c…
tags:
  - "ACL 2026"
  - "Medical Imaging"
  - "De-identification"
  - "Psychiatric Narratives"
  - "Semantic Graph"
  - "Structure Preservation"
  - "LLM Generation"
date: 2026-05-08
content_hash: 7af7b279184c94bc
---

# Anonpsy: A Graph-Based Framework for Structure-Preserving De-identification of Psychiatric Narratives

**Conference**: ACL 2026
**arXiv**: [2601.13503](https://arxiv.org/abs/2601.13503)  
**Code**: None  
**Area**: Medical Imaging
**Keywords**: De-identification, Psychiatric Narratives, Semantic Graph, Structure Preservation, LLM Generation

## TL;DR

This paper proposes Anonpsy, a framework that reformulates the de-identification of psychiatric narratives as a graph-guided semantic rewriting problem. The approach first converts narratives into semantic graphs, applies constrained perturbations on the graph to modify identity-related information while preserving clinical structure, and finally reconstructs the narrative via graph-conditioned generation.

## Background & Motivation

**Background**: Psychiatric narratives contain rich clinical information—symptom timelines, causal relationships, and diagnostic reasoning—that is critical for downstream tasks such as diagnostic prediction, yet also embeds substantial patient-identifying information.

**Limitations of Prior Work**: (1) Token-level PHI masking preserves clinical structure but retains high semantic similarity to the original, leaving significant residual re-identification risk. (2) LLM-based synthetic data creation (SDC) reduces identifiability but uncontrollably distorts clinical structure—e.g., converting persecutory delusions into grandiose delusions. (3) Both approaches treat text as unstructured sequences, ignoring the relational and temporal dependencies inherent in psychiatric narratives.

**Key Challenge**: In psychiatric narratives, identifiability derives from the narrative structure itself—specific life events and timelines—rather than from explicit identifiers alone. Simultaneously modifying identity-related information while preserving clinical structure represents a fundamental tension for text-level methods.

**Goal**: To reformulate de-identification as a structure-preserving generation problem, enabling fine-grained control through an intermediate graph representation.

**Key Insight**: Converting narratives into semantic graphs encoding clinical entities, temporal anchors, and typed relations enables constrained perturbation at the graph level.

**Core Idea**: By decoupling event structure from surface text, the framework can precisely control what is preserved and what is modified at the graph level, then regenerate a coherent narrative from the modified graph.

## Method

### Overall Architecture

The framework consists of three steps: (1) Semantic graph conversion $G = \mathcal{E}(X)$ — narratives are transformed into semantic graphs via LLM-assisted, schema-constrained extraction; (2) Graph-constrained perturbation $\tilde{G} = \mathcal{P}(G)$ — contextual attributes are modified while temporal, causal, and diagnostic structure is preserved; (3) Graph-conditioned text generation $\hat{X} = \mathcal{D}(\tilde{G})$ — de-identified narratives are reconstructed from the modified graph.

### Key Designs

1. **Semantic Graph Representation**:

    - **Function**: Provides an editable intermediate representation that decouples structure from content.
    - **Mechanism**: Nodes $V$ represent clinical entities (symptoms, treatments, diagnoses); edges $E$ encode typed relations (diagnostic dependency, causality, temporal ordering). Extracted using a schema-constrained LLM.
    - **Design Motivation**: The graph representation makes "what to preserve and what to modify" explicitly controllable—demographic attributes can be altered while symptom–diagnosis relations remain intact.

2. **Graph-Constrained Perturbation**:

    - **Function**: Modifies identifiable information while preserving clinical logic.
    - **Mechanism**: Selectively alters contextual attributes (e.g., age, occupation, specific life events) while keeping temporal offset relations and causal/diagnostic edges unchanged.
    - **Design Motivation**: Psychiatric diagnosis depends critically on the temporal progression and causal relationships among symptoms, which must not be perturbed.

3. **Graph-Conditioned Text Generation**:

    - **Function**: Generates coherent de-identified narratives from the modified graph.
    - **Mechanism**: A locally deployed LLM generates new narratives conditioned on the modified semantic graph. Lower temperature is used for schema extraction and narrative generation (stability), while higher temperature is applied during perturbation (diversity).
    - **Design Motivation**: All processing uses a local LLM (gpt-oss:120b), as clinical privacy environments typically prohibit the use of cloud-based APIs.

### Loss & Training

No training is required. All three operators—conversion, perturbation, and generation—are implemented via prompt engineering and deterministic control flow. All LLM inference is performed locally on four RTX A6000 GPUs.

## Key Experimental Results

### Main Results

| Method | Diagnostic Fidelity (F1) | Identifiability (Cosine Sim.) | Note |
|---|---|---|---|
| PHI Masking | High | High (Risky) | Structure intact but traceable |
| LLM-SDC | Low (Semantic Drift) | Low | Safe but clinically distorted |
| Anonpsy | High | Low | Best of both |

### Ablation Study

| Configuration | Key Metric | Note |
|---|---|---|
| w/o Graph Perturbation | High identifiability | Unmodified structure remains highly traceable |
| w/o Structural Constraints | Lower diagnostic F1 | Unconstrained rewriting harms clinical meaning |
| Expert Evaluation | Low re-identification risk | Psychiatrists unable to trace back to original cases |
| GPT-5 Evaluation | Low semantic similarity | Automated assessment consistent with human judgment |

### Key Findings
- Anonpsy achieves the optimal trade-off between privacy protection and clinical fidelity.
- The graph intermediate representation makes the modification process transparent and controllable.
- Expert evaluation confirms that de-identified narratives preserve the original diagnostic reasoning.

## Highlights & Insights
- Represents a paradigm shift in de-identification from "text processing" to "structure-aware generation."
- The semantic graph representation allows clinicians to inspect and intervene in the modification process.
- Full local deployment ensures practical applicability in real clinical environments.

## Limitations & Future Work
- Evaluated on only 90 psychiatric cases, limiting scale.
- The quality of semantic graph extraction is dependent on LLM capability.
- Currently validated only for psychiatric narratives; generalizability to other clinical specialties remains unverified.
- Future work may extend to multilingual settings and larger-scale clinical datasets.

## Related Work & Insights
- **vs. PHI Masking**: Operates at the semantic level rather than the token level, more thoroughly eliminating identifiable information.
- **vs. LLM-SDC**: Graph constraints bound the rewriting scope, preventing uncontrolled semantic drift.
- **vs. Knowledge Graph Methods**: The semantic graph is used to control generation rather than for retrieval or reasoning, representing a novel application of knowledge graph techniques.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Graph-guided de-identification constitutes an entirely new paradigm.
- Experimental Thoroughness: ⭐⭐⭐ Dataset scale is small, but evaluation dimensions are comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Problem formulation is clear and method formalization is rigorous.
- Value: ⭐⭐⭐⭐⭐ Significant practical implications for privacy preservation in clinical NLP.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] NeuroCircuitry-Inspired Hierarchical Graph Causal Attention Networks for Explainable Depression Identification](../../ICLR2026/medical_imaging/neurocircuitry-inspired_hierarchical_graph_causal_attention_networks_for_explain.md)
- [\[ACL 2026\] Region-Grounded Report Generation for 3D Medical Imaging: A Fine-Grained Dataset and Graph-Enhanced Framework](region-grounded_report_generation_for_3d_medical_imaging_a_fine-grained_dataset_.md)
- [\[AAAI 2026\] Learning with Preserving for Continual Multitask Learning](../../AAAI2026/medical_imaging/learning_with_preserving_for_continual_multitask_learning.md)
- [\[CVPR 2026\] Multiscale Structure-Guided Latent Diffusion for Multimodal MRI Translation](../../CVPR2026/medical_imaging/multiscale_structure-guided_latent_diffusion_for_multimodal_mri_translation.md)
- [\[AAAI 2026\] Radiation-Preserving Selective Imaging for Pediatric Hip Dysplasia: A Cross-Modal Approach](../../AAAI2026/medical_imaging/radiation-preserving_selective_imaging_for_pediatric_hip_dysplasia_a_cross-modal.md)

</div>

<!-- RELATED:END -->
