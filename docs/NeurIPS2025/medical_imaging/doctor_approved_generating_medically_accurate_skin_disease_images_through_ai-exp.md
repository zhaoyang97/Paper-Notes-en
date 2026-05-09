---
title: >-
  [Paper Note] Doctor Approved: Generating Medically Accurate Skin Disease Images through AI-Expert Feedback
description: >-
  [NeurIPS 2025][Medical Imaging][Medical image generation] This paper proposes MAGIC, a framework that encodes dermatologist-defined clinical checklists into structured evaluation prompts executable by MLLMs (e.g., GPT-4o), and uses the resulting feedback to fine-tune diffusion models via DPO or reward-based fine-tuning (RFT), generating clinically accurate skin disease images for data augmentation. MAGIC achieves +9.02% improvement on a 20-class skin disease classification task and +13.89% in few-shot settings.
tags:
  - NeurIPS 2025
  - Medical Imaging
  - Medical image generation
  - diffusion models
  - DPO
  - AI feedback
  - skin disease diagnosis
date: 2026-05-08
content_hash: 669c2106612917d6
---

# Doctor Approved: Generating Medically Accurate Skin Disease Images through AI-Expert Feedback

**Conference**: NeurIPS 2025
**arXiv**: [2506.12323](https://arxiv.org/abs/2506.12323)
**Code**: [https://github.com/janet-sw/MAGIC.git](https://github.com/janet-sw/MAGIC.git)
**Area**: Medical Imaging
**Keywords**: Medical image generation, diffusion models, DPO, AI feedback, skin disease diagnosis

## TL;DR
This paper proposes MAGIC, a framework that encodes dermatologist-defined clinical checklists into structured evaluation prompts executable by MLLMs (e.g., GPT-4o), and uses the resulting feedback to fine-tune diffusion models via DPO or reward-based fine-tuning (RFT), generating clinically accurate skin disease images for data augmentation. MAGIC achieves +9.02% improvement on a 20-class skin disease classification task and +13.89% in few-shot settings.

## Background & Motivation

**Background**: Deep learning holds great promise for skin disease diagnosis, but privacy constraints and data scarcity—particularly for rare conditions and underrepresented skin tones—severely limit model generalization. Diffusion models (DMs) have been explored for synthesizing medical images to augment training data.

**Limitations of Prior Work**: Existing DM-based augmentation methods typically follow an end-to-end generation pipeline, where expert involvement is limited to post-hoc evaluation or filtering rather than actively guiding the generation process. As a result, synthetic images frequently lack clinical accuracy (e.g., incorrect lesion characteristics) and may even degrade diagnostic model performance.

**Key Challenge**: Reinforcement learning from human feedback (RLHF) requires extensive expert annotation and robust reward model training. While DPO bypasses the explicit reward model, it remains underexplored in the medical imaging domain. Moreover, having medical experts manually evaluate large volumes of synthetic images is prohibitively expensive.

**Goal**: How can high-quality clinical feedback be obtained with minimal expert effort to guide diffusion model generation?

**Key Insight**: MLLMs (e.g., GPT-4o) are employed as automated evaluators. Experts only need to design structured clinical checklists (5 visual criteria per disease); the MLLM then evaluates synthetic images against the checklist and returns binary scores, substantially reducing manual workload. This constitutes a "task-centric" alignment paradigm: rather than adapting MLLMs to medical tasks, complex medical judgments are decomposed into simple visual verification subtasks that MLLMs can reliably perform.

**Core Idea**: Clinical expert knowledge is encoded as attribute-level checklists, enabling general-purpose MLLMs to automatically evaluate synthetic images. DPO is then applied to guide diffusion models toward generating clinically accurate medical images.

## Method

### Overall Architecture
The MAGIC pipeline consists of four stages: (1) pre-fine-tuning Stable Diffusion via Textual Inversion and LoRA to learn skin disease concepts; (2) generating image pairs using the fine-tuned DM through an image-to-image (I2I) pipeline; (3) submitting image pairs to GPT-4o for clinical checklist evaluation (5-dimensional binary scoring); and (4) leveraging the evaluation feedback to further fine-tune the DM via DPO or RFT. The augmented data is then used to train a downstream classifier.

### Key Designs

1. **Pre-Fine-Tuning (Textual Inversion + LoRA)**:

    - **Function**: Enables the pre-trained Stable Diffusion model to understand specific skin disease concepts.
    - **Mechanism**: Textual Inversion first learns a unique token embedding $v_*$ for each disease; LoRA (low-rank matrices $A \in \mathbb{R}^{n \times r}, B \in \mathbb{R}^{r \times n}$) then fine-tunes the UNet attention layers to capture fine-grained lesion visual features.
    - **Design Motivation**: Off-the-shelf diffusion models lack domain knowledge of skin lesions, resulting in poor direct generation quality.

2. **MLLM-Driven Expert Feedback Collection**:

    - **Function**: GPT-4o automatically evaluates synthetic images against a 5-dimensional clinical checklist (location, lesion type, shape/size, color, texture) designed by dermatologists.
    - **Mechanism**: Each evaluation processes one image pair; the MLLM returns a 5-dimensional binary vector (e.g., [1,0,0,1,0]) per image, which is aggregated into an overall binary score via a predefined algorithm. This simultaneously produces single-sample feedback for RFT and preference pairs for DPO.
    - **Design Motivation**: Decomposing complex medical diagnostic reasoning into closed-form visual verification tasks substantially reduces MLLM hallucination risk. Evaluating only synthetic images also preserves patient privacy.

3. **Feedback-Based Fine-Tuning — DPO Path**:

    - **Function**: Directly optimizes diffusion model parameters using preference data, without training an explicit reward model.
    - **Mechanism**: The denoising process is modeled as a multi-step MDP; each state-action pair along the generation trajectory of winner/loser images receives a +1/−1 reward. $t' = \gamma T$ sub-segments are constructed to maximize learning efficiency:
    $$\mathcal{L}_{\text{DPO}}^i(\theta) = -\mathbb{E}[\log \sigma(\beta \log \frac{\pi_\theta(a_i^w|s_i^w)}{\pi_{\text{ref}}(a_i^w|s_i^w)} - \beta \log \frac{\pi_\theta(a_i^l|s_i^l)}{\pi_{\text{ref}}(a_i^l|s_i^l)})]$$
    - **Design Motivation**: DPO bypasses reward model training, yielding greater robustness in specialized domains with limited feedback data.

4. **Image-to-Image (I2I) Generation Strategy**:

    - **Function**: Generates target disease images by partially noising real skin disease images and applying text-guided denoising.
    - **Mechanism**: Source image body-site information is preserved while only the lesion semantics are transformed, realizing factorized translation.
    - **Design Motivation**: Reduces semantic distortion and prevents classifiers from learning spurious correlations (e.g., associating specific lesions with specific body locations).

### Loss & Training
- **RFT path**: A reward model is trained via $\mathcal{L}_{\text{RM}}(\phi) = \sum (y - \mathcal{R}_\phi(x,c))^2$, followed by reward-weighted likelihood maximization to fine-tune the DM.
- **DPO path**: A multi-segment loss with an additional fidelity constraint on real data.
- During classifier training, the synthetic data ratio is set to $\rho = 0.2$ to prevent overfitting to the synthetic distribution.

## Key Experimental Results

### Main Results (Fitzpatrick17k 20-class Subset)

| Method | ResNet18 Acc | ResNet18 F1 | DINOv2 Acc | DINOv2 F1 |
|--------|-------------|-------------|-----------|----------|
| Real only | 29.31% | 28.73% | 49.89% | 49.43% |
| + T2I | 25.57% (−3.74) | 24.63% | 47.73% (−2.16) | 47.26% |
| + I2I | 31.45% (+2.14) | 31.09% | 50.71% (+0.82) | 50.17% |
| + MAGIC-RFT | 33.49% (+4.18) | 30.40% | 51.16% (+1.27) | 52.66% |
| + **MAGIC-DPO** | **38.33% (+9.02)** | **37.01%** | **55.01% (+5.12)** | **54.05%** |

### Ablation Study

| Ablation | ResNet18 Acc | DINOv2 Acc | Note |
|----------|-------------|-----------|------|
| MAGIC-DPO (Structured checklist) | 38.33% | 55.01% | Full model |
| MAGIC-DPO (Coarse checklist) | 32.83% (−5.50) | 51.16% (−3.85) | Coarse-grained checklist yields substantial drop |
| Few-shot (310 samples) + MAGIC-DPO | 37.39% (+10.94) | — | Larger gains under extreme data scarcity |
| Few-shot + MAGIC-A (with unlabeled data) | **40.34% (+13.89)** | — | Further gains by leveraging unlabeled data |
| GPT-4o evaluation | 38.33% | 55.01% | Default MLLM |
| MedGemma-4B evaluation | 36.97% | 54.19% | Open-source alternative with comparable performance |

### Key Findings
- MAGIC-DPO substantially outperforms RFT, likely because DPO directly optimizes preference alignment without an intermediate reward model, yielding greater robustness under limited feedback.
- Checklist quality has a large impact: the structured checklist outperforms the coarse-grained version by 5.5% on ResNet18.
- Performance is stable for synthetic data ratios $\rho \in [0.1, 0.3]$, with $\rho = 0.2$ being optimal.
- DPO performance saturates at approximately 512 feedback pairs, indicating that large-scale feedback is not required.
- T2I generation without guidance degrades performance (−3.74%), demonstrating that unguided synthetic images introduce noise.
- Dermatologist evaluation shows that 55.5% of MAGIC-DPO generated images satisfy 3+ clinical criteria, far exceeding the baseline.

## Highlights & Insights
- **Task-centric alignment paradigm**: Rather than turning MLLMs into medical experts, complex medical judgments are decomposed into simple visual verification subtasks that MLLMs can reliably execute—a generalizable approach applicable to AI feedback in any specialized domain.
- **Factorized translation via I2I**: Preserving body-site information while altering only lesion features simultaneously improves medical plausibility and prevents classifiers from learning spurious correlations.
- **Open-source MLLM viability**: MedGemma-4B achieves performance comparable to GPT-4o, reducing dependence on closed-source APIs.
- **Anti-hallucination by design**: Converting open-ended medical reasoning into closed-form checklist verification reduces MLLM hallucination at the architectural level.

## Limitations & Future Work
- Framework performance is bounded by MLLM capabilities, particularly in fine-grained visual understanding.
- Validation is limited to a 20-class subset of Fitzpatrick17k; generalization to more disease types and datasets remains to be demonstrated.
- Clinical checklists require dermatologist expertise to design; extending the framework to new diseases still involves manual effort.
- Domain shift between synthetic and real images persists, necessitating careful tuning of the synthetic-to-real ratio.
- A finer-grained 9-criterion checklist (mentioned in the paper as yielding marginal further gains) warrants further investigation.

## Related Work & Insights
- **vs. Denoising diffusion probabilistic model augmentation (Sagers et al.)**: Prior work fine-tunes models via DreamBooth/Textual Inversion and generates images directly without expert feedback, leaving synthesis quality uncontrolled.
- **vs. RLHF fine-tuning for DMs (Reward-weighted MDP)**: Conventional approaches require extensive human evaluation to train reward models; MAGIC substantially reduces manual cost via MLLM-based checklist evaluation.
- **vs. DPO for DMs (Yang et al.)**: Prior DPO work focuses on natural images; this paper is among the first to systematically apply DPO to medical image generation.
- The proposed framework has direct implications for synthetic augmentation in other medical imaging modalities, including radiology and pathology.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — Using MLLMs as evaluators combined with checklist-guided DPO fine-tuning is an elegant combination, though each individual component is based on existing techniques.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Multiple backbone networks, comprehensive baselines, thorough ablations, and three-pronged validation (FID, expert evaluation, classification), though the dataset and disease type coverage are limited.
- **Writing Quality**: ⭐⭐⭐⭐ — The method is clearly described with well-motivated design choices, though the notation is dense and requires careful reading.
- **Value**: ⭐⭐⭐⭐⭐ — Provides a scalable AI-expert collaboration framework for data-scarce medical imaging; the +13.89% improvement in few-shot settings is highly practical.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Online Feedback Efficient Active Target Discovery in Partially Observable Environments](online_feedback_efficient_active_target_discovery_in_partially_observable_enviro.md)
- [\[NeurIPS 2025\] Dynamic Causal Discovery in Alzheimer's Disease through Latent Pseudotime Modelling](dynamic_causal_discovery_in_alzheimers_disease_through_latent_pseudotime_modelli.md)
- [\[NeurIPS 2025\] PatientSim: A Persona-Driven Simulator for Realistic Doctor-Patient Interactions](patientsim_a_persona-driven_simulator_for_realistic_doctor-patient_interactions.md)
- [\[CVPR 2026\] Robust Fair Disease Diagnosis in CT Images](../../CVPR2026/medical_imaging/robust_fair_disease_diagnosis_in_ct_images.md)
- [\[NeurIPS 2025\] The Boundaries of Fair AI in Medical Image Prognosis: A Causal Perspective](the_boundaries_of_fair_ai_in_medical_image_prognosis_a_causal_perspective.md)

</div>

<!-- RELATED:END -->
