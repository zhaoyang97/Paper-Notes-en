---
title: >-
  [Paper Note] TumorChain: Interleaved Multimodal Chain-of-Thought Reasoning for Traceable Clinical Tumor Analysis
description: >-
  [ICLR 2026][LLM Reasoning][Tumor Analysis] This paper proposes TumorChain, an interleaved multimodal chain-of-thought reasoning framework for tumor analysis across five major digestive organs. It integrates a knowledge g…
tags:
  - "ICLR 2026"
  - "LLM Reasoning"
  - "Tumor Analysis"
  - "Multimodal CoT Reasoning"
  - "Interleaved Reasoning"
  - "3D CT"
  - "TNM Staging"
date: 2026-05-08
content_hash: b045d83b241e415c
---

# TumorChain: Interleaved Multimodal Chain-of-Thought Reasoning for Traceable Clinical Tumor Analysis

**Conference**: ICLR 2026
**arXiv**: [2603.05867](https://arxiv.org/abs/2603.05867)  
**Code**: [GitHub](https://github.com/ZJU4HealthCare/TumorChain)  
**Area**: LLM Reasoning
**Keywords**: Tumor Analysis, Multimodal CoT Reasoning, Interleaved Reasoning, 3D CT, TNM Staging

## TL;DR

This paper proposes TumorChain, an interleaved multimodal chain-of-thought reasoning framework for tumor analysis across five major digestive organs. It integrates a knowledge graph-driven 1.5M CoT-VQA data engine, organ-guided iterative interleaved reasoning (IIR), and joint optimization of segmentation, classification, and LLM models to realize a complete reasoning chain from imaging findings → clinical impressions → pathological predictions, achieving an average accuracy of 84.41% and substantially outperforming GPT-5-Mini (51.59%).

## Background & Motivation

- **Background**: Medical VLMs have made progress in general report generation, but remain critically insufficient for the high-stakes domain of clinical oncology. Tumor analysis requires a complete reasoning chain connecting imaging findings, clinical impressions, and pathological endpoints (TNM staging).
- **Three Key Limitations**: (1) Existing Med-VLMs lack tumor-specific capabilities and cannot reliably map radiological findings to pathology-level endpoints; (2) Large-scale, multi-granularity tumor-specific datasets are absent — existing benchmarks such as CT-RATE focus on multiple-choice or short-text QA and do not support CoT reasoning; (3) Most Med-VLMs are restricted to 2D images and single-step reasoning, whereas the structural complexity of 3D CT demands multi-step clinical reasoning.
- **Key Challenge**: Clinical tumor diagnosis is an inherently multi-step reasoning process (detecting anomalies → synthesizing judgments → pathological staging), yet existing models cannot produce traceable reasoning chains, leaving the internal reasoning process opaque.
- **Key Insight**: This work constructs a complete findings → impressions → pathology reasoning pipeline and introduces a dedicated CoT evaluation protocol (TumorChain-Eval) to assess the quality of each step in the reasoning chain.

## Method

### Overall Architecture

TumorChain comprises five modules: a 3D visual encoder $\mathcal{E}_v$, an organ segmentation expert $\mathcal{S}eg$, an auxiliary classification model $\mathcal{C}ls$, an MLP projector $\mathcal{P}$, and an LLM $\mathcal{LLM}$. These modules collectively enable end-to-end tumor analysis through global-local visual alignment and interleaved multimodal reasoning.

### Key Designs

**1. Knowledge Graph-Driven CoT Data Engine (TumorCoT-1.5M)**:
- Raw data: 41,059 3D CT scans + 10,708 radiology reports + partial pathology reports, covering five major digestive organs: liver, pancreas, stomach, colon, and esophagus.
- Six collaborative agents: a segmentation expert (TotalSegmentator), a structured feature extractor (Qwen3-235B), a CoT reasoner (GPT-4o-mini), a logic calibrator (Claude3.5-Haiku), and a summarizer (GPT-5-mini).
- Diagnostic knowledge graph (KG) constraints: Five organ-specific KGs co-constructed with radiologists and pathologists to guide reasoning chains toward clinical standards.
- Cross-validation mechanism: When the logic calibrator detects issues in a reasoning chain, two repair strategies are triggered (expanding the organ region / providing suspected causes) to guide re-reasoning.
- Final output: 1,497,818 CoT-VQA pairs covering four task types: localization, lesion attributes, TNM staging, and CoT reports.

**2. Organ-Guided Iterative Interleaved Reasoning (IIR)**:
- Step I: The LLM receives global CT tokens and a task prompt to produce an initial diagnosis $\mathcal{R}^1_{cot}$.
- Step II: The target organ is identified from the initial output → ROI is extracted via segmentation → an enhanced prompt ("more attention should be paid to [organ name]") is generated → local organ tokens are obtained.
- Step III: Global tokens + task prompt + initial answer + local tokens are jointly fed into the LLM for iterative reasoning; if additional relevant organs are identified, the loop continues.
- Effect: Simulates the clinical radiologist workflow — global overview first, then focused inspection of suspicious regions with repeated confirmation.

**3. Hybrid Collaborative Optimization (HCO)**:
- Segmentation model: Continuously provides accurate ROI localization.
- Classification model: Trained on local organ features for normal/abnormal binary classification, enhancing the visual encoder's discriminative power for subtle anomalies.
- LLM: Integrates reasoning results and leverages the segmentation model for iterative decision-making.
- Joint loss: $L_{total} = L_{LLM} + \lambda L_{cls}$

**4. TumorChain-Eval Evaluation Protocol**:
- Subject-predicate-object triplets are extracted from CoT reasoning chains (e.g., "pancreatic tail – finding – malignancy").
- Three-level scoring: finding chain $S_{FC}$ (individual facts) → impression chain $S_{IC}$ (synthesis of multiple findings) → long reasoning chain $S_{LRC}$ (high-level inference).
- GPT-4 is used to score against defined rubrics; $CoT_e$ is the weighted sum of all three levels.

## Key Experimental Results

### Main Results

| Method | Avg. Accuracy | TNM-T | TNM-N | TNM-M | CoTe Score |
|--------|:------------:|:-----:|:-----:|:-----:|:----------:|
| GPT-5-Mini | 51.59% | — | — | — | 61.23 |
| Gemini2.0 | 41.29% | — | — | — | 54.28 |
| **TumorChain-7B** | **84.41%** | **88.83%** | **61.63%** | **71.07%** | **58.33** |

### Ablation Study

| Configuration | Avg. Accuracy | Note |
|---------------|:------------:|------|
| Full TumorChain | **84.41%** | Complete framework |
| w/o IIR | 80.34% (−4.07%) | IIR is the largest contributor |
| w/o CoT | 82.45% (−1.96%) | CoT data also contributes significantly |
| w/o Classification Model | 82.93% (−1.48%) | Auxiliary classification enhances discrimination |

### Key Findings

- Localization accuracy is near-perfect: organ-level 99.97%, position-level 97.57%, substantially surpassing all baselines.
- IIR contributes the most (−4.07% when removed) — iterative refinement is the core mechanism, mirroring the "scan → focus → re-examine" radiologist workflow.
- Zero-shot generalization on the public DeepTumorVQA benchmark: 73.30% vs. MedVLM-R1 56.41%, demonstrating strong domain transferability.
- TNM-N (lymph node metastasis) yields the lowest accuracy (61.63%), consistent with its difficulty in clinical practice.

## Highlights & Insights

- **Complete clinical reasoning pipeline**: The three-level reasoning chain design of findings → impressions → pathology ensures traceability and interpretability.
- **Knowledge graph-driven data engine**: Automatic generation of 1.5M high-quality CoT samples addresses the scarcity of tumor-specific annotated data.
- **Iterative Interleaved Reasoning (IIR)**: Elegantly fuses global context with local evidence through multi-round self-verification, reducing hallucination risk.
- **Triplet-based evaluation protocol**: Extracts structured knowledge from CoT chains for scoring, providing finer granularity than end-to-end metrics.

## Limitations & Future Work

- Iterative reasoning introduces a latency of 2.51 seconds per sample, requiring acceleration for real-time clinical deployment.
- CoT evaluation relies on GPT-4 scoring, which may introduce systematic bias.
- Coverage is currently limited to five digestive organs; generalizability to other domains (e.g., lung, breast) remains to be validated.
- TNM-N staging accuracy of only 61.63% indicates that lymph node metastasis assessment remains a critical challenge.
- Training data originate from multi-center Chinese hospitals; cross-regional and cross-device generalization requires further verification.
- The absence of comparison experiments with specialist physicians limits conclusions regarding clinical deployment value.

## Related Work & Insights

- Compared to general medical VLM datasets such as CT-RATE and 3D-RAD, TumorCoT-1.5M is the first large-scale dataset to provide tumor-specific CoT annotations.
- Compared to medical reasoning models such as MedVLM-R1, TumorChain achieves deeper multi-step reasoning through iterative interleaved reasoning.
- The IIR design philosophy — LLM → ROI identification → segmentation → local feature injection → re-reasoning — is generalizable to other medical imaging tasks requiring spatial refinement.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (First multimodal CoT reasoning framework specifically targeting oncology)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (1.5M data / multi-task evaluation / generalization / ablation)
- Writing Quality: ⭐⭐⭐⭐ (Deep clinical motivation, complete technical details)
- Value: ⭐⭐⭐⭐⭐ (Important tool for precision oncology with strong clinical translation potential)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] AIMCoT: Active Information-driven Multimodal Chain-of-Thought for Vision-Language Reasoning](aimcot_active_information-driven_multimodal_chain-of-thought_for_vision-language.md)
- [\[CVPR 2026\] Understanding and Mitigating Hallucinations in Multimodal Chain-of-Thought Models](../../CVPR2026/llm_reasoning/understanding_and_mitigating_hallucinations_in_multimodal_chain-of-thought_model.md)
- [\[AAAI 2026\] BLM-Guard: Explainable Multimodal Ad Moderation with Chain-of-Thought and Policy-Aligned Rewards](../../AAAI2026/llm_reasoning/blm-guard_explainable_multimodal_ad_moderation_with_chain-of.md)
- [\[ICLR 2026\] Compositional Generalization from Learned Skills via CoT Training: A Theoretical and Structural Analysis for Reasoning](compositional_generalization_from_learned_skills_via_cot_training_a_theoretical_.md)
- [\[ICLR 2026\] Continuous Chain of Thought Enables Parallel Exploration and Reasoning](continuous_chain_of_thought_enables_parallel_exploration_and_reasoning.md)

</div>

<!-- RELATED:END -->
