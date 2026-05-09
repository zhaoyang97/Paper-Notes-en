---
title: >-
  [Paper Note] Passing the Driving Knowledge Test
description: >-
  [ICCV 2025][Autonomous Driving][Driving Knowledge Test] This paper introduces DriveQA — the first large-scale text-and-visual dual-modality driving knowledge test benchmark (26K text QA + 448K image QA) — to systematically evaluate LLMs/MLLMs on driving knowledge including traffic regulations, sign recognition, and right-of-way judgment. The benchmark reveals significant deficiencies in numerical reasoning and complex right-of-way scenarios, and demonstrates that DriveQA pretraining yields generalization gains on downstream driving tasks.
tags:
  - ICCV 2025
  - Autonomous Driving
  - Driving Knowledge Test
  - LLM
  - MLLM
  - VQA
  - Traffic Rules
  - benchmark
  - Fine-tuning
date: 2026-05-08
content_hash: 821b7d263eac46fa
---

# Passing the Driving Knowledge Test

**Conference**: ICCV 2025
**arXiv**: [2508.21824](https://arxiv.org/abs/2508.21824)
**Code**: [driveqaiccv.github.io](https://driveqaiccv.github.io)
**Area**: Autonomous Driving / Driving Knowledge QA Evaluation
**Keywords**: Driving Knowledge Test, LLM, MLLM, VQA, Traffic Rules, benchmark, Fine-tuning

## TL;DR

This paper introduces DriveQA — the first large-scale text-and-visual dual-modality driving knowledge test benchmark (26K text QA + 448K image QA) — to systematically evaluate LLMs/MLLMs on driving knowledge including traffic regulations, sign recognition, and right-of-way judgment. The benchmark reveals significant deficiencies in numerical reasoning and complex right-of-way scenarios, and demonstrates that DriveQA pretraining yields generalization gains on downstream driving tasks.

## Background & Motivation

Safe driving requires not only visual perception but also reasoning over traffic rules and informed decision-making. Human drivers must pass a written knowledge test to obtain a license. However, critical gaps exist in current autonomous driving benchmarks and multimodal LLM evaluations:

**Existing benchmarks focus on perception and basic planning**: nuScenes-QA, DriveLM, and similar works target spatial understanding and collision avoidance, rarely assessing comprehension of traffic regulations — such as speed limit signs, right-of-turn priority, or rare markings.

**Insufficient coverage of long-tail rules**: Real driving tests cover a large number of edge cases (e.g., special construction signs, complex intersection yielding rules) that appear extremely infrequently in real-world driving data.

**Limited traffic knowledge in MLLMs**: Although MLLMs may inherit partial traffic knowledge from pretraining data, experiments demonstrate that such knowledge and the associated reasoning capacity remain limited.

**Practical failures in commercial systems**: Evidence of misinterpretations of traffic rules has been documented in systems such as Tesla FSD.

The central question posed by DriveQA is: *if an LLM were to take a driving knowledge test today, would it pass?*

## Method

### Dataset Construction

**DriveQA-T (Text QA)**:
- 26K text QA pairs covering 5 major categories and 19 subcategories (traffic lights, traffic signs, parking, regulations, symbols, etc.)
- Data source: official driver's manuals collected from all 50 U.S. states plus D.C. (51 manuals total)
- Construction pipeline: GPT-4o automatically generates questions from manual content → human quality validation → multi-round review to remove ambiguous or inconsistent entries
- Each QA pair is accompanied by an answer explanation for assessing reasoning capability

**DriveQA-V (Visual QA)**:
- 68K images and 448K VQA pairs
- **Traffic signs**: 220 3D models of U.S. traffic signs are inserted into the CARLA simulator, with controlled variation in viewpoint (frontal/oblique/overhead), weather, time of day, and distance
- **Right-of-way judgment**: Intersections are identified in CARLA maps, vehicles of different colors are randomly generated, and right-of-way judgment scenarios are constructed
- **Real-world data**: Annotated real-world data collected from Mapillary as a supplementary source

### Evaluation Methodology

1. **Question type classification**: BERT embeddings combined with hierarchical clustering categorize questions into 19 semantic classes; KeyBERT is used to extract keywords.
2. **Prompting strategies**: Four prompt designs are employed — baseline, CoT (chain-of-thought), RAG (retrieval-augmented generation from driving manuals), and CoT+RAG.
3. **Fine-tuning**: LoRA low-rank adaptation is used for parameter-efficient fine-tuning.
4. **Loss function**: Standard cross-entropy for multiple-choice classification.

### Key Evaluation Dimensions

- Text QA accuracy (across 19 categories)
- CoT reasoning quality (BLEU-4, ROUGE-L)
- Visual QA accuracy (traffic sign recognition, right-of-way judgment)
- Sensitivity to environmental factors (viewpoint, weather, time of day)
- Downstream task transfer (nuScenes, BDD trajectory prediction)

## Key Experimental Results

### Main Results: LLM Performance on DriveQA-T

| Model | Size | CoT | RAG | FT | Speed Limit | Parking | Intersection | Avg. |
|-------|------|-----|-----|----|----|------|------|------|
| Gemma-2 | 2B | | | | 42.2 | 35.6 | 27.9 | 44.2 |
| Gemma-2 | 9B | ✓ | ✓ | | 64.9 | 68.3 | 77.9 | 76.9 |
| Llama-3.1 | 8B | ✓ | ✓ | ✓ | 72.7 | 86.1 | **91.6** | **87.6** |
| Phi-3.5-mini | 3.8B | | | | 49.2 | 48.5 | 79.7 | 69.8 |
| Phi-3.5-mini | 3.8B | ✓ | ✓ | ✓ | 66.9 | 65.4 | 87.2 | 81.1 |
| **GPT-4o** | — | ✓ | ✓ | | **76.7** | **93.8** | **97.3** | **92.0** |

**Key Findings**:
- Open-source models perform reasonably on basic traffic rules but exhibit pronounced weaknesses in **numerical reasoning** (speed/distance limits) and complex **right-of-way scenarios**.
- CoT+RAG consistently improves performance, underscoring the importance of retrieval-augmented knowledge for traffic rule understanding.
- Fine-tuning substantially improves open-source models; Llama-3.1 reaches 87.6%, approaching GPT-4o (92.0%).

### Ablation Study: MLLM Performance on DriveQA-V

| Model | Size | T-intersection (frontal) | 4-way intersection (frontal) | Regulatory Signs | Warning Signs | Avg. |
|-------|------|--------------------------|------------------------------|-----------------|--------------|------|
| Mini-InternVL | 2B (original) | 27.8 | 26.0 | 64.1 | 55.3 | 41.8 |
| Mini-InternVL | 2B (fine-tuned) | **86.7** | **74.3** | **93.8** | **92.2** | **86.6** |
| LLaVA-1.6 | 7B (original) | 18.8 | 31.0 | 42.6 | 43.0 | 34.5 |
| LLaVA-1.6 | 7B (fine-tuned) | 86.1 | 74.4 | 82.1 | 84.1 | 83.7 |
| **GPT-4o** | — (zero-shot) | 55.1 | 50.5 | **93.8** | **94.0** | **75.3** |

**Key Findings**:
- Without fine-tuning, MLLM right-of-way judgment accuracy is near random chance (~25%); sign recognition is slightly better but still far from sufficient.
- Fine-tuning yields substantial gains: Mini-InternVL improves from 41.8% to 86.6%, approaching GPT-4o's 75.3%.
- The 10 most difficult sign categories are predominantly regulatory and warning signs (e.g., Playground, Trauma Center).

### CoT Reasoning Quality Evaluation

| Model | BLEU-4 (w/o RAG) | ROUGE-L (w/o RAG) | BLEU-4 (w/ RAG) | ROUGE-L (w/ RAG) |
|-------|-----------------|------------------|-----------------|------------------|
| Gemma-2 (9B, FT) | **0.4112** | 0.5420 | 0.4105 | **0.5528** |
| GPT-4o | 0.3905 | 0.5354 | 0.3989 | 0.5393 |

The fine-tuned Gemma-2 (9B) surpasses GPT-4o on reasoning quality metrics.

## Highlights & Insights

1. **First comprehensive driving knowledge benchmark**: DriveQA is the only multimodal benchmark that simultaneously covers traffic regulations, signs, and right-of-way, filling a critical gap in LLM/MLLM driving reasoning evaluation.
2. **Value of controllable synthetic data**: By leveraging CARLA to procedurally generate large-scale controlled variations (viewpoint, weather, sign type), DriveQA-V pretraining transfers to improved real-world downstream task performance.
3. **Gap between knowledge and reasoning**: Even the strongest model, GPT-4o, falls well below human-level performance on right-of-way judgment (~60%) and speed limit reasoning (~77%), demonstrating that rule-based reasoning is considerably harder than simple pattern recognition.
4. **Effectiveness of fine-tuning**: LoRA fine-tuning alone substantially compensates for inadequate pretraining knowledge, suggesting that insufficient coverage of traffic rules in pretraining corpora is the primary bottleneck.

## Limitations & Future Work

1. Coverage is limited to U.S. traffic regulations; traffic laws across other countries and regions are not addressed.
2. Image data is predominantly synthetic (CARLA), and a domain gap with real-world scenes remains.
3. Evaluation is primarily multiple-choice; open-ended driving decision reasoning is not assessed.
4. Only a limited number of open-source MLLMs are evaluated; larger-scale models (e.g., Gemini Pro, Claude) are not included.

## Related Work & Insights

- **MLLM driving agents**: DriveGPT4, DriveLM, EMMA, and similar works apply LLMs to driving decision-making but focus on planning rather than rule comprehension.
- **Driving VQA datasets**: NuScenes-QA, DriveBench, and LingoQA target spatial perception rather than regulatory reasoning.
- **Traffic sign recognition**: Traditional benchmarks such as GTSRB lack any reasoning requirement.
- **CoT and RAG**: Chain-of-thought and retrieval-augmented generation have proven effective for complex reasoning tasks.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (First systematic evaluation of LLMs on "driving exam" capabilities; a novel and practically motivated perspective)
- Technical Depth: ⭐⭐⭐ (Primarily a benchmark contribution; methodology is relatively standard)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Covers text & visual QA, multiple models, multiple strategies, and downstream transfer validation)
- Practical Value: ⭐⭐⭐⭐⭐ (Directly exposes knowledge blind spots of current MLLMs in safety-critical scenarios)
- Overall Recommendation: ⭐⭐⭐⭐ (Excellent benchmark contribution, though methodological innovation is limited)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] DriveX: Omni Scene Modeling for Learning Generalizable World Knowledge in Autonomous Driving](drivex_omni_scene_modeling_for_learning_generalizable_world_knowledge_in_autonom.md)
- [\[ICCV 2025\] ACAM-KD: Adaptive and Cooperative Attention Masking for Knowledge Distillation](acam-kd_adaptive_and_cooperative_attention_masking_for_knowledge_distillation.md)
- [\[ICCV 2025\] SDKD: Frequency-Aligned Knowledge Distillation for Lightweight Spatiotemporal Forecasting](frequency-aligned_knowledge_distillation_for_lightweight_spatiotemporal_forecast.md)
- [\[CVPR 2026\] KnowVal: A Knowledge-Augmented and Value-Guided Autonomous Driving System](../../CVPR2026/autonomous_driving/knowval_a_knowledge-augmented_and_value-guided_autonomous_driving_system.md)
- [\[ICCV 2025\] Adaptive Dual Uncertainty Optimization: Boosting Monocular 3D Object Detection under Test-Time Shifts](adaptive_dual_uncertainty_optimization_boosting_monocular_3d_object_detection_un.md)

</div>

<!-- RELATED:END -->
