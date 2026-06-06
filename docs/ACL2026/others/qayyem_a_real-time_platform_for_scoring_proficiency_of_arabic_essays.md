---
title: >-
  [Paper Note] Qayyem: A Real-time Platform for Scoring Proficiency of Arabic Essays
description: >-
  [ACL 2026][Automated Essay Scoring] Qayyem is the first web platform that supports cross-prompt multi-trait Automated Essay Scoring (AES) for Arabic. It integrates various scoring schemes ranging from feature engineering…
tags:
  - "ACL 2026"
  - "Automated Essay Scoring"
  - "Arabic NLP"
  - "Multi-trait scoring"
  - "Cross-prompt generalization"
  - "Educational Technology"
date: 2026-05-08
content_hash: 8b6a13cfe74aac81
---

# Qayyem: A Real-time Platform for Scoring Proficiency of Arabic Essays

**Conference**: ACL 2026  
**arXiv**: [2603.01009](https://arxiv.org/abs/2603.01009)  
**Code**: [https://qayyem.qu.edu.qa/](https://qayyem.qu.edu.qa/)  
**Area**: others  
**Keywords**: Automated Essay Scoring, Arabic NLP, Multi-trait scoring, Cross-prompt generalization, Educational Technology

## TL;DR
Qayyem is the first web platform that supports cross-prompt multi-trait Automated Essay Scoring (AES) for Arabic. It integrates various scoring schemes ranging from feature engineering to SOTA neural models, supporting an end-to-end workflow for academic writing assessment.

## Background & Motivation
**Background**: Automated Essay Scoring (AES) systems have seen significant progress in English scenarios, but Arabic AES lags behind due to linguistic complexity and the scarcity of large-scale annotated datasets.

**Limitations of Prior Work**: Existing Arabic writing tools (such as Qalam) primarily focus on writing assistance rather than scoring; the only known deployed system, ARWI, only supports prompt-specific modes and lacks multi-trait scoring support.

**Key Challenge**: Educational settings require a deployable system that can generalize to new prompts and score multiple writing traits (grammar, organization, vocabulary, etc.) separately. However, existing tools cannot simultaneously satisfy cross-prompt generalization and multi-trait scoring.

**Goal**: To build the first online platform for cross-prompt multi-trait Arabic AES.

**Key Insight**: Utilizing a web platform as a carrier to encapsulate various SOTA scoring models into a unified API service, while providing a complete workflow for assignment creation, uploading, scoring, and auditing.

**Core Idea**: By using cross-prompt models trained on the large-scale LAILA dataset combined with a modular architectural design, teachers can utilize advanced AES models without needing a technical background.

## Method

### Overall Architecture
Qayyem adopts a decoupled front-end and back-end architecture: the Assignment Manager (a Next.js full-stack application) handles web interactions and assignment management, while the Scoring Server (FastAPI) handles model inference and API services. The complete workflow consists of four steps: (1) design of prompts and rubrics → (2) assignment configuration (selecting scoring traits and models) → (3) batch scoring → (4) result auditing and manual correction.

### Key Designs
1. **Cross-prompt Multi-trait Scoring Architecture**:

    - **Function**: Supports selecting different models for scoring across 7 writing traits (Relevance, Organization, Vocabulary, Style, Development, Mechanics, Grammar).
    - **Mechanism**: Each trait is configured with an independent model, all of which are trained on cross-prompt partitions of the LAILA dataset.
    - **Design Motivation**: The evaluation difficulty and optimal model may vary across different writing traits; trait-specific model selection allows for flexible combinations.

2. **Modular Scoring Server**:

    - **Function**: Manages the loading, enablement, and metadata of all models via configuration files, exposing a unified REST + SSE API.
    - **Mechanism**: All models follow a shared interface, protected by API Key authentication and rate limiting, supporting real-time streaming progress updates.
    - **Design Motivation**: Allows system administrators to add or disable models without modifying the core code, supporting hot updates.

3. **Multi-level Model Deployment (Feature-based + SOTA Models)**:

    - **Function**: Simultaneously deploys lightweight feature-based models (RF, NN, XGB) and SOTA models (TRATES, MOOSE), providing a choice between performance and efficiency.
    - **Mechanism**: Feature-based models utilize $816$ manual features; TRATES leverages LLMs to generate trait-specific features + linguistic features; MOOSE combines MoE + pairwise ranking + relevance modeling.
    - **Design Motivation**: In educational scenarios, the requirements for real-time performance and accuracy vary by use case; providing multiple levels of models meets different needs.

### Loss & Training
All models are trained in a cross-prompt setting on the LAILA dataset ($7,859$ Arabic essays, $8$ prompts). The evaluation metric is Quadratic Weighted Kappa ($QWK$), which measures the consistency between predicted scores and human scores. TRATES uses the Fanar LLM to generate features, and MOOSE uses the AraBERT encoder.

## Key Experimental Results

### Main Results
Average $QWK$ of each model across 8 prompts in the LAILA dataset:

| Model | REL | ORG | VOC | STY | DEV | MEC | GRM | HOL | AVG |
|------|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| RF | 0.331 | 0.609 | 0.644 | 0.637 | 0.573 | 0.559 | 0.609 | 0.682 | 0.581 |
| NN | 0.353 | 0.609 | 0.621 | 0.631 | 0.566 | 0.565 | 0.597 | 0.651 | 0.574 |
| XGB | 0.360 | 0.645 | 0.641 | 0.641 | 0.583 | 0.577 | 0.619 | 0.679 | 0.593 |
| MOOSE | 0.411 | 0.627 | 0.642 | 0.649 | 0.585 | 0.586 | 0.623 | 0.649 | 0.597 |
| **TRATES** | **0.557** | **0.696** | **0.657** | **0.664** | **0.652** | **0.608** | **0.643** | **0.744** | **0.653** |

### Efficiency-Effectiveness Trade-off
| Model | Average $QWK$ | Inference Time per Essay |
|------|----------|-------------|
| NN | 0.574 | 0.2 s |
| RF | 0.581 | 0.3 s |
| XGB | 0.593 | 0.2 s |
| MOOSE | 0.597 | 1 s |
| TRATES | 0.653 | 30 s |

### Key Findings
- TRATES achieves the best performance across all dimensions, with an overall $QWK$ of $0.653$, which is $5.6$ percentage points higher than the runner-up MOOSE.
- TRATES shows the greatest advantage in the Relevance (REL) dimension, leading MOOSE by $15$ percentage points.
- Differences between models in Vocabulary and Style dimensions are relatively small (approx. $3.5$ percentage points), while gaps in Holistic, Development, and Organization can reach $9$ percentage points.
- The high performance of TRATES comes with a $150 \times$ inference overhead ($30$ s vs $0.2$ s), requiring high-end GPUs or LLM APIs.

## Highlights & Insights
- The first cross-prompt multi-trait Arabic AES system, filling a gap in Arabic educational technology.
- The platform design balances fully automated scoring and decision-support scenarios, supporting manual auditing and score correction.
- Provides public API access, facilitating researchers to directly call models for secondary development.
- The efficiency-effectiveness gradient design allows teachers to flexibly choose scoring schemes based on the scenario.

## Limitations & Future Work
- Currently only supports Arabic, though the bilingual UI design reserves interfaces for integrating English models.
- TRATES inference is slow ($30$ s/essay); large-scale batch scoring scenarios may require asynchronous queue optimization.
- The LAILA dataset only covers persuasive and expository essays for grades 10-12; generalization to other genres and grade levels remains to be verified.
- The system relies on manual features ($816$ dimensions); future work could consider end-to-end multi-trait scoring models.

## Related Work & Insights
- **ARWI**: The only known deployed Arabic AES system, but it only supports prompt-specific modes and lacks multi-trait scoring.
- **CriterionSM / IFlyEA**: Mature AES systems for English/Chinese; Qayyem draws inspiration from their cross-prompt + multi-trait design philosophies.
- **LAILA Dataset**: The first large-scale multi-trait Arabic AES benchmark, serving as the foundation for the model training in this system.
- Insight: Engineering NLP systems for low-resource languages requires addressing data, models, and deployment simultaneously.

## Rating
- Novelty: ⭐⭐⭐ The core contribution lies in system integration rather than model innovation, but it fills the gap in Arabic AES platforms.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive cross-prompt evaluation on LAILA with clear efficiency-effectiveness analysis.
- Writing Quality: ⭐⭐⭐⭐ Detailed description of system design with clear workflow visualization.
- Value: ⭐⭐⭐⭐ Significant practical deployment value for the field of Arabic educational technology.

## Rating
- Novelty: Pending
- Experimental Thoroughness: Pending
- Writing Quality: Pending
- Value: Pending

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] I2E: Real-Time Image-to-Event Conversion for High-Performance Spiking Neural Networks](../../AAAI2026/others/i2e_real-time_image-to-event_conversion_for_high-performance_spiking_neural_netw.md)
- [\[CVPR 2026\] MyoVision: A Mobile Research Tool and NEATBoost-Attention Ensemble Framework for Real Time Chicken Breast Myopathy Detection](../../CVPR2026/others/myovision_a_mobile_research_tool_and_neatboost_attention_ensemble_framework.md)
- [\[CVPR 2026\] Crowdsourcing of Real-world Image Annotation via Visual Properties](../../CVPR2026/others/crowdsourcing_of_real_world_image_annotation_via_visual_properties.md)
- [\[CVPR 2026\] SimRecon: SimReady Compositional Scene Reconstruction from Real Videos](../../CVPR2026/others/simrecon_simready_compositional_scene_reconstruction_from_real_videos.md)
- [\[ICML 2026\] TEMPORA: Characterising the Time-Contingent Utility of Online Test-Time Adaptation](../../ICML2026/others/tempora_characterising_the_time-contingent_utility_of_online_test-time_adaptatio.md)

</div>

<!-- RELATED:END -->
