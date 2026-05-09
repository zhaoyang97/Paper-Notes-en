---
title: >-
  [Paper Note] Demo: Generative AI helps Radiotherapy Planning with User Preference
description: >-
  [NeurIPS 2025 (GenAI for Health Workshop)][Medical Imaging][radiotherapy planning] This paper proposes the Flexible Dose Proposer (FDP), a two-stage training framework (VQ-VAE pretraining + multi-condition encoding) that enables slider-based interactive 3D dose distribution prediction incorporating user preferences. The system is integrated into the Eclipse clinical treatment planning system and outperforms Varian RapidPlan in head-and-neck cancer radiotherapy scenarios.
tags:
  - NeurIPS 2025 (GenAI for Health Workshop)
  - Medical Imaging
  - radiotherapy planning
  - dose prediction
  - user preference interaction
  - VQ-VAE
  - generative AI
date: 2026-05-08
content_hash: 6ae967bd2792d759
---

# Demo: Generative AI helps Radiotherapy Planning with User Preference

**Conference**: NeurIPS 2025 (GenAI for Health Workshop)  
**arXiv**: [2512.08996](https://arxiv.org/abs/2512.08996)  
**Code**: [Demo Video](https://huggingface.co/Jungle15/DoseProposerDemo)  
**Area**: Medical Imaging  
**Keywords**: radiotherapy planning, dose prediction, user preference interaction, VQ-VAE, generative AI

## TL;DR
This paper proposes the Flexible Dose Proposer (FDP), a two-stage training framework (VQ-VAE pretraining + multi-condition encoding) that enables slider-based interactive 3D dose distribution prediction incorporating user preferences. The system is integrated into the Eclipse clinical treatment planning system and outperforms Varian RapidPlan in head-and-neck cancer radiotherapy scenarios.

## Background & Motivation

**State of the Field**: Radiotherapy planning is a complex clinical workflow with significant variability across institutions and planners. Deep learning methods have made progress in dose prediction, fluence map generation, and MLC leaf sequencing.

**Limitations of Prior Work**:
   - Knowledge-driven systems such as RapidPlan rely on DVH prediction and cannot capture spatial dose details; PCA regression-based pipelines are trained on only tens of plans with limited generalizability.
   - Existing deep learning dose prediction models neglect user preference interaction — different planners have varying requirements for the trade-off between OAR sparing and PTV coverage.
   - Dose prediction itself does not constitute a deliverable plan, and integration with clinical TPS systems remains insufficiently studied.

**Root Cause**: A single model cannot accommodate diverse planning styles, and training is prone to bias toward the specific style of the reference plans.

**Starting Point**: Drawing on the conditional control paradigm from generative AI, "preference flavors" sliders allow users to customize the OAR–PTV trade-off in real time.

**Core Idea**: A VQ-VAE pretrained dose decoder provides a stable foundation, and user preference encodings serve as conditional inputs, enabling interactive and personalized dose prediction.

## Method

### Overall Architecture
Inputs consist of CT images, RT structures (PTV/OAR contours), beam/angle plates, and user preference slider values. The pipeline proceeds in two stages: Stage I pretrains a VQ-VAE dose decoder, and Stage II trains a flexible dose prediction model with multi-condition inputs. The output is a 3D dose distribution, which is subsequently converted into a deliverable treatment plan in Eclipse via objective function extraction.

### Key Designs

1. **Stage I: VQ-VAE Base Dose Decoder Pretraining**

    - Function: Pretrains a VQ-VAE on 31K dose samples to learn latent representations of realistic dose distributions.
    - Mechanism: Unlike latent diffusion models that use a VAE for compression and acceleration, this work uses pretraining to **stabilize Stage II training**. The loss function is:
      $\mathcal{L}_{\text{stage1}} = \underbrace{\mathbb{E}_i[\|x_i - \hat{x}_i\|]}_{\text{Reconstruction}} + \beta L_{vq} + L_{adv}(x, \hat{x}) + \underbrace{\lambda \cdot \log(\mathbb{E}_{i<j}[\exp(-t\|\hat{z}_i - \hat{z}_j\|^2)])}_{\text{Uniformity}}$
    - Design Motivation: A uniformity loss regularizes the latent space to prevent mode collapse; the adversarial loss ensures the realism of generated doses. Without pretraining, Stage II training becomes unstable under complex conditioning and produces artifacts at PTV/OAR boundaries.

2. **Stage II: Multi-Condition Flexible Dose Prediction**

    - Function: Takes CT and RT structures as image inputs and user preferences along with beam plates as conditional inputs to predict personalized 3D doses.
    - Mechanism: A MedNeXt-based image encoder processes multi-channel inputs; user preferences and beam/angle information are injected via AdaIN (Adaptive Instance Normalization). The loss function is:
      $\mathcal{L}_{\text{stage2}}^{(i)} = \|x_i - \hat{x}_i\| + \|z_i - \hat{z}_i\| + L_{adv}(x_i, \hat{x}_i) + \mathcal{L}_{\text{obj}}^{(i)}$
    - The objective consistency loss $\mathcal{L}_{\text{obj}}$ ensures alignment between predictions and user preferences:
      $\mathcal{L}_{\text{obj}}^{(i)} = \|\tilde{h} - \hat{h}\| + \|p - \hat{p}\| + \|\tilde{w} \cdot u_{\text{oar}} - \hat{u}_{\text{oar}}\|$
      where $\tilde{h}$ denotes the user-specified HI value and $\tilde{w}$ is the OAR sparing preference weight.

3. **Random Preference Sampling during Training**

    - Function: Slider values $\{\tilde{h}, \tilde{w}\}$ are randomly sampled within predefined ranges during training.
    - Design Motivation: This enables the model to learn responses to diverse preference combinations rather than overfitting to a single planning style.

4. **Clinical Integration**

    - The predicted 3D dose is converted into optimization objectives for the Eclipse treatment planning system via objective function extraction, yielding a deliverable plan.

### Loss & Training
- Stage I: reconstruction + VQ quantization + adversarial + uniformity losses, trained on 31K dose samples.
- Stage II: image-space reconstruction + latent-space reconstruction + adversarial + objective consistency losses, trained on 6 cohorts (~820 training samples in total).
- Single-step generation via GAN; inference takes approximately 30 ms, avoiding the iterative sampling of diffusion models.

## Key Experimental Results

### Main Results: Intra-Patient DVH Variability (VMAT Plans)

| OAR | RapidPlan std | FDP std | RapidPlan mean | FDP mean |
|-----|--------------|---------|----------------|----------|
| SpinalCord05 | 3.02 | **1.18** | 3.25 | **1.69** |
| Larynx-PTV | 3.49 | **1.78** | 6.29 | **2.94** |
| ParotidCon-PTV | 3.38 | **1.22** | 4.51 | **1.92** |
| PosteriorNeck | 4.30 | **0.92** | 8.15 | **1.19** |
| Trachea | 3.40 | **1.68** | 3.42 | **2.01** |
| **Better count** | 0 | **15** | 1 | **14** |

FDP outperforms RapidPlan on the std metric for 15/15 OARs and on the mean metric for 14/15 OARs.

### Ablation Study

| Configuration | MAE (↓) | Notes |
|------|---------|------|
| w/o Stage I pretrain | 2.63 | boundary artifacts observed |
| w/ Stage I pretrain | **2.56** | more realistic dose distributions |

### Key Findings
- Stage I pretraining not only reduces MAE but, more importantly, eliminates artifacts at PTV/OAR boundaries, indicating that pretraining provides prior constraints on the dose distribution.
- The user preference sliders effectively control the trade-off between OAR sparing and PTV homogeneity: P1 (prioritizing OAR protection) and P2 (prioritizing PTV) produce distinct dose distributions in actual Eclipse plans.
- Model inference takes only 30 ms; with visualization the total latency is approximately 1.5 s, approaching real-time interaction.

## Highlights & Insights
- **First dose prediction model with interactive sliders**: The paper introduces the conditional control paradigm of generative AI into radiotherapy planning, enabling real-time adjustment of user preferences — a concept transferable to other medical imaging tasks requiring personalized outputs.
- **Elegant two-stage training strategy**: Large-scale (31K) but lower-quality data are used to pretrain the decoder, followed by conditional fine-tuning on small-scale high-quality data, addressing training instability under medical data scarcity.
- **End-to-end clinical validation**: The work goes beyond dose prediction by validating deliverable plan quality through Eclipse integration, which is relatively rare in academic research.

## Limitations & Future Work
- Evaluation is limited to head-and-neck cancer; generalizability to other treatment sites (lung, abdomen, etc.) has not been verified.
- RapidPlan is trained with a small dataset using conventional methods, so the comparison with a deep learning model involves differing training paradigms and warrants further discussion of fairness.
- User preferences currently span only two dimensions — HI and OAR weight — with no control over CI or finer-grained DVH objectives.
- GAN-based single-step generation may limit output diversity; diffusion models could potentially offer finer-grained conditional control.

## Related Work & Insights
- **vs. RapidPlan**: RapidPlan relies on PCA and small-data conventional ML, predicting only DVH; FDP directly predicts 3D dose with voxel-level spatial awareness.
- **vs. DoseDiff**: Diffusion-model-based dose prediction achieves high accuracy but slow inference due to iterative sampling; FDP employs single-step GAN generation for real-time interaction.
- **vs. OpenKBP series**: Prior work focuses solely on prediction accuracy without considering user preference interaction or clinical TPS integration.

## Rating
- Novelty: ⭐⭐⭐⭐ First interactive preference-driven dose prediction system, though individual technical components (VQ-VAE, AdaIN) are not novel contributions.
- Experimental Thoroughness: ⭐⭐⭐ Limited dataset scale, single treatment site, and insufficient comparison with other deep learning methods.
- Writing Quality: ⭐⭐⭐⭐ Well-structured, clinical motivation clearly articulated, and demo presentation is intuitive.
- Value: ⭐⭐⭐⭐ Strong clinical applicability; addresses genuine pain points in planner workflows.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] GEM: Generative Entropy-Guided Preference Modeling for Few-shot Alignment of LLMs](../../AAAI2026/medical_imaging/gem_generative_entropy-guided_preference_modeling_for_few-shot_alignment_of_llms.md)
- [\[NeurIPS 2025\] Demo: Guide-RAG: Evidence-Driven Corpus Curation for Retrieval-Augmented Generation in Long COVID](demo_guide-rag_evidence-driven_corpus_curation_for_retrieval-augmented_generatio.md)
- [\[AAAI 2026\] Rethinking Bias in Generative Data Augmentation for Medical AI: a Frequency Recalibration Approach](../../AAAI2026/medical_imaging/rethinking_bias_in_generative_data_augmentation_for_medical_ai_a_frequency_recal.md)
- [\[NeurIPS 2025\] The Boundaries of Fair AI in Medical Image Prognosis: A Causal Perspective](the_boundaries_of_fair_ai_in_medical_image_prognosis_a_causal_perspective.md)
- [\[NeurIPS 2025\] Mind the Gap: Aligning Knowledge Bases with User Needs to Enhance Mental Health Retrieval](mind_the_gap_aligning_knowledge_bases_with_user_needs_to_enhance_mental_health_r.md)

<!-- RELATED:END -->
