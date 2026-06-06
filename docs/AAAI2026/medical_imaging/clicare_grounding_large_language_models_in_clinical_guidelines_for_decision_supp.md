---
title: >-
  [Paper Note] CliCARE: Grounding Large Language Models in Clinical Guidelines for Decision Support over Longitudinal Cancer Electronic Health Records
description: >-
  [AAAI 2026][Medical Imaging][LLM clinical decision support] This paper proposes CliCARE, a framework that transforms unstructured longitudinal cancer EHRs into temporal knowledge graphs (TKGs)…
tags:
  - "AAAI 2026"
  - "Medical Imaging"
  - "LLM clinical decision support"
  - "electronic health records"
  - "temporal knowledge graph"
  - "clinical guideline alignment"
  - "RAG"
date: 2026-05-08
content_hash: 65dc1c231b3d83b6
---

# CliCARE: Grounding Large Language Models in Clinical Guidelines for Decision Support over Longitudinal Cancer Electronic Health Records

**Conference**: AAAI 2026  
**arXiv**: [2507.22533](https://arxiv.org/abs/2507.22533)  
**Code**: [Available](https://github.com/sakurakawa1/CliCARE)  
**Area**: Medical Imaging  
**Keywords**: LLM clinical decision support, electronic health records, temporal knowledge graph, clinical guideline alignment, RAG

## TL;DR

This paper proposes CliCARE, a framework that transforms unstructured longitudinal cancer EHRs into temporal knowledge graphs (TKGs), aligns them with clinical practice guideline (CPG) knowledge graphs, and provides evidence-grounded clinical decision support for LLMs. An LLM-as-a-Judge evaluation protocol highly correlated with expert assessments is also introduced.

## Background & Motivation

LLMs show promise in clinical decision support for reducing physician cognitive burden, yet deployment in high-stakes domains such as oncology faces three core challenges:

**Insufficient long-context temporal reasoning**: Cancer EHRs span multiple years and exceed 20,000 tokens, sometimes containing multilingual entries. Existing models suffer from the "lost-in-the-middle" problem and cannot effectively process fragmented longitudinal records.

**Clinical hallucination risk**: Fragmented text retrieved by standard RAG fails to capture sequential dependencies in patient trajectories or align with process-oriented CPGs—factually incorrect recommendations may compromise patient safety.

**Unreliable evaluation**: Traditional metrics (ROUGE, BLEU) focus on lexical overlap and cannot measure clinical validity, factual accuracy, or safety. LLM-as-a-Judge approaches also exhibit systematic biases such as position bias and verbosity preference.

Core gap: **Existing work treats long-context processing, knowledge grounding, and reliable evaluation as separate problems, lacking a unified framework that addresses all three simultaneously.**

## Method

### Overall Architecture

CliCARE comprises three core stages:

1. **EHR → TKG conversion**: Transforming unstructured clinical records into patient-centric temporal knowledge graphs.
2. **Trajectory–guideline alignment**: Deep alignment between real patient trajectories and normative CPG graphs.
3. **Expert-validated LLM evaluation**: Ensuring evaluation outcomes are highly correlated with clinical expert judgments.

### Key Designs

#### 1. EHR-to-TKG Conversion

**Event extraction pipeline** $E_p = f_{pipeline}(D_p)$:

- **Historical record compression**: Patient records are sorted chronologically; Longformer (pretrained on clinical text) is applied to perform extractive summarization of the historical records $D_p^{hist}$, yielding a past medical history summary $S_p^{hist}$.
- **Recent record retention**: The most recent clinical note $d_{\tau_n}$ is preserved in full as the present illness history.
- **Information extraction**: BERT is used to extract key clinical facts (diagnosis confirmation, staging updates, treatment regimens, biomarker trends, imaging assessments, etc.) from the concatenated text.

**TKG instantiation** $G_t = (E_t, R_t, T)$:

- A general static biomedical knowledge graph $G_B$ is constructed containing standardized medical concepts and relations.
- An entity linking function $\phi: \mathcal{E}_p \rightarrow \mathcal{E}_B$ maps clinical entities to standard ontology entries.
- Each entity is represented as $e = (e_B, \tau, A)$: standard entity + timestamp + event attributes.
- Hierarchical temporal granularity is adopted: precise timestamps for macro-level clinical events, and relative temporal relations within events.

#### 2. Trajectory–Guideline Alignment

**Knowledge formalization**:

- Patient temporal trajectories $Tr_p = \langle e_1, e_2, \ldots, e_m \rangle$ are extracted from the TKG.
- All normative treatment pathways $\{Pa_k\}_{k=1}^K$ are enumerated from the guideline knowledge graph $G_g$.

**Similarity matching**: Semantic similarity is computed using biomedical BERT:

$$\text{Score}(Tr_p, Pa_k) = \sum_{j=1}^l \max_{e_i \in Tr_p} \text{cos\_sim}(f_{BERT}(\text{desc}(s_j)), f_{BERT}(\text{desc}(e_i)))$$

**LLM re-ranking**: The top-$N$ candidate pathways and their matching scores are provided as context, and an LLM performs zero-shot re-ranking in the role of a clinical reasoner, compensating for clinical logic that purely algorithmic matching cannot capture.

**Alignment extension** (bootstrap-inspired):

- The LLM re-ranked optimal alignment serves as the seed set $A'$.
- For each unaligned event, the most consistent guideline node with existing high-confidence alignments is identified:

$$\hat{s} = \arg\max_{s_v \in Pa'_1} \sum_{(e_i, s_j) \in A'} \text{sim}((e_u, s_v), (e_i, s_j))$$

- The alignment set is iteratively expanded, leveraging established strong associations to infer new ones.

#### 3. Expert-Validated LLM-as-a-Judge Evaluation

**Evaluation dimensions** (co-designed with senior oncologists):

- Factual Accuracy
- Completeness & Thoroughness
- Clinical Soundness
- Actionability & Relevance

**Bias mitigation**:

- Ensemble judging: GPT-4.1 + Claude 4.0 Sonnet + Gemini 2.5 Pro scores are averaged.
- Evaluation item order is randomly shuffled to eliminate position bias.
- Validation: LLM judge achieves Spearman rank correlation $\rho \approx 0.7$ with assessments from three oncologists.

### Loss & Training

The training phase of CliCARE primarily involves fine-tuning expert models:

- 2,000 samples split into 1,800 training + 200 test; 10% of training data used for validation.
- Batch size = 1, maximum context 20,000 tokens, learning rate 5e-5 with cosine scheduler.
- BF16 mixed-precision training, maximum output 4,096 tokens, 3 epochs.
- Hardware: 4× NVIDIA A800 GPUs.

The framework supports both generative large models (Gemini 2.5 Pro, GPT-4.1, etc.) and fine-tuned expert models (Qwen-3-8B, etc.); TKG + guideline-aligned structured inputs yield significant gains for both categories.

## Key Experimental Results

### Main Results

Multiple RAG baselines are compared on two large-scale clinical datasets:

**CancerEHR dataset (2,000 Chinese cancer patients, spanning 20+ years):**

| Method | Clinical Summary ↑ | Clinical Recommendation ↑ |
|--------|--------------------|---------------------------|
| StandardRAG (Qwen-3-8B) | 1.485 | 1.527 |
| BriefContext (Gemini 2.5 Pro) | 4.527 | 4.468 |
| MedRAG* (Gemini 2.5 Pro) | 4.470 | 4.576 |
| **CliCARE (Qwen-3-8B)** | **3.173** | **3.215** |
| **CliCARE (Gemini 2.5 Pro)** | **4.976** | **4.965** |

**CliCARE + Gemini 2.5 Pro approaches a perfect score on CancerEHR (4.976 on a 5-point scale), substantially outperforming all baselines.**

**Framework improvement over StandardRAG:**

| Model | CancerEHR Summary Gain | MIMIC-Cancer Summary Gain |
|-------|------------------------|---------------------------|
| Qwen-3-8B | +1.688 | +0.100 |
| Deepseek-R1 | +2.279 | +0.393 |
| Gemini 2.5 Pro | +2.241 | +0.835 |

### Ablation Study

| Configuration | CancerEHR Summary | CancerEHR Recommendation |
|---------------|-------------------|--------------------------|
| CliCARE (Qwen) | 3.173 | 3.215 |
| w/o Extension | 3.012 (−) | 3.035 (−) |
| w/o Re-ranking | 2.857 (−) | 2.866 (−) |
| w/o TKG Compression | 1.485 (−) | 1.527 (−) |
| CliCARE (Gemini) | 4.976 | 4.965 |
| w/o TKG Compression | 2.735 (−) | 2.818 (−) |

TKG compression is the most critical component; its removal causes a substantial performance drop (Gemini: 4.976 → 2.735).

### Key Findings

- **Structured knowledge is essential for complex EHRs**: The largest gains are observed on the long-record CancerEHR dataset, demonstrating that even powerful models require structured representations for effective reasoning.
- **Larger models benefit more from longer records**: Gemini 2.5 Pro achieves the highest scores on the longest record segment (66–100%), indicating that CliCARE helps advanced models exploit richer context.
- **Smaller models degrade on long records**: Qwen-3-8B performance decreases on the longest record segment, yet remains far superior to StandardRAG.
- **TKG compression may be counterproductive on simpler datasets**: On the simpler MIMIC-Cancer dataset, removing TKG compression slightly improves performance for Qwen (2.575 → 2.475), suggesting that overly aggressive compression may discard useful information.

## Highlights & Insights

1. **End-to-end knowledge-grounded pipeline**: The complete loop from unstructured EHR → TKG → guideline alignment → LLM generation is theoretically motivated at every stage.
2. **Bootstrap alignment extension**: The approach of inferring new alignments from high-confidence seeds is elegant, analogous to label propagation on graphs.
3. **Methodological contribution to evaluation**: The $\rho \approx 0.7$ correlation between LLM judges and expert assessments provides a credible evaluation scheme for clinical NLP.
4. **Model-agnostic generality**: The framework improves both 7B-scale and top-tier commercial models, demonstrating that structured knowledge representation is more effective than simply scaling model size.

## Limitations & Future Work

- **Limited domain coverage**: Validation is restricted to oncology; generalization to cardiology, neurology, and other clinical specialties remains to be demonstrated.
- **Guideline graph construction cost**: Domain expert involvement is required to build CPG knowledge graphs, imposing non-trivial manual effort when extending to new diseases.
- **Privacy considerations**: The framework relies on complete EHR data processing; deployment must strictly comply with data privacy regulations.
- **Evaluation imperfection**: Although $\rho \approx 0.7$ is relatively high, it is not perfect, and edge cases may yield judgments inconsistent with expert opinion.
- **Fixed alignment threshold**: The BERT semantic similarity threshold is fixed at 0.7, which may require adaptive tuning for different guidelines.

## Related Work & Insights

- **ColaCare**: A multi-agent EHR modeling architecture representing a recent trend in EHR research.
- **MedRAG / KG2RAG / GNN-RAG**: Graph-augmented RAG methods; CliCARE surpasses these via TKG alignment.
- **Longformer**: An efficient Transformer for long document processing, used in CliCARE to compress historical records.
- The knowledge graph + LLM integration paradigm holds similar promise in other high-stakes decision-making domains such as law and finance.

## Rating

- **Novelty**: ★★★★★ — Triple innovations in TKG construction, guideline alignment, and evaluation protocol.
- **Experimental Thoroughness**: ★★★★★ — Dual datasets (Chinese and English), multiple baselines and models, detailed ablation and record-length analysis.
- **Writing Quality**: ★★★★★ — Problem formulation is clear; framework design is logically rigorous.
- **Value**: ★★★★☆ — Code is open-sourced, but guideline graph construction and large-scale EHR processing still present engineering challenges.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Personalization of Large Foundation Models for Health Interventions](personalization_of_large_foundation_models_for_health_interventions.md)
- [\[AAAI 2026\] G2L: From Giga-Scale to Cancer-Specific Large-Scale Pathology Foundation Models via Efficient Fine-Tuning](g2lfrom_giga-scale_to_cancer-specific_large-scale_pathology_foundation_models_vi.md)
- [\[NeurIPS 2025\] Position: Thematic Analysis of Unstructured Clinical Transcripts with Large Language Models](../../NeurIPS2025/medical_imaging/position_thematic_analysis_of_unstructured_clinical_transcripts_with_large_langu.md)
- [\[AAAI 2026\] Unleashing the Potential of Large Language Models for Text-to-Image Generation through Autoregressive Representation Alignment](unleashing_the_potential_of_large_language_models_for_text-to-image_generation_t.md)
- [\[AAAI 2026\] Do Large Language Models Think Like the Brain? Sentence-Level Evidences from Layer-Wise Embeddings and fMRI](do_large_language_models_think_like_the_brain_sentence-level_evidences_from_laye.md)

</div>

<!-- RELATED:END -->
