---
title: >-
  [Paper Note] PatientVLM Meets DocVLM: Pre-Consultation Dialogue Between Vision-Language Models for Efficient Diagnosis
description: >-
  [AAAI 2026][Multimodal VLM][Medical Diagnosis] This paper proposes the Pre-Consultation Dialogue Framework (PCDF), which simulates multi-turn doctor–patient dialogues using two VLMs (DocVLM and PatientVLM) to generate image–dialogue–diagnosis triplets for fine-tuning DocVLM, achieving an average F1 improvement of 11.48 across four medical imaging benchmarks.
tags:
  - AAAI 2026
  - Multimodal VLM
  - Medical Diagnosis
  - Multi-turn Dialogue
  - VLM Interaction
  - Data Synthesis
  - Dialogue-driven Fine-tuning
date: 2026-05-08
content_hash: 1dfb44efd7a7a55e
---

# PatientVLM Meets DocVLM: Pre-Consultation Dialogue Between Vision-Language Models for Efficient Diagnosis

**Conference**: AAAI 2026
**arXiv**: [2601.10945](https://arxiv.org/abs/2601.10945)
**Code**: [https://vl2g.github.io/projects/pcdf](https://vl2g.github.io/projects/pcdf)
**Area**: Multimodal VLM
**Keywords**: Medical Diagnosis, Multi-turn Dialogue, VLM Interaction, Data Synthesis, Dialogue-driven Fine-tuning

## TL;DR
This paper proposes the Pre-Consultation Dialogue Framework (PCDF), which simulates multi-turn doctor–patient dialogues using two VLMs (DocVLM and PatientVLM) to generate image–dialogue–diagnosis triplets for fine-tuning DocVLM, achieving an average F1 improvement of 11.48 across four medical imaging benchmarks.

## Background & Motivation

### State of the Field
Medical imaging AI research has long revolved around the "image → diagnosis" paradigm, progressing from early CNN-based classification to medical adaptations of CLIP (MedCLIP, BioMedCLIP), and then to large VLMs (MedPaLM2, MedGemma, LLaVA-Med), with continuously improving visual understanding capabilities.

### Limitations of Prior Work
In clinical practice, however, **diagnosis rarely relies on imaging alone**. Physicians engage in multi-turn interactions with patients, progressively eliciting symptoms and medical history to narrow the differential diagnosis. This dialogue-driven reasoning process is central to clinical diagnosis, yet existing models entirely neglect this aspect, resulting in fragile predictions.

### Data Collection Challenges
Collecting real doctor–patient dialogue data is extremely difficult: it requires IRB ethical approval, patient informed consent, and faces physician concerns over legal liability and workflow disruption, making large-scale data collection practically infeasible.

### Root Cause
Existing efforts (Yang et al. 2024; Chen et al. 2023) have attempted to generate synthetic dialogues using a **single LLM** playing both doctor and patient, but suffer from two critical limitations: (1) they operate in text-only settings without medical images; and (2) having a single model generate both roles undermines role separation and interactional authenticity.

### Starting Point
This paper proposes PCDF — employing **two independent VLMs** to respectively play the doctor and patient roles, conducting joint visual-dialogue reasoning over medical images. PatientVLM generates symptom-based responses conditioned on ground-truth diagnostic labels (while being explicitly instructed not to reveal the diagnosis), and DocVLM generates follow-up questions based on the image and dialogue history. This design preserves the information asymmetry inherent in real clinical consultations.

## Method

### Overall Architecture
PCDF consists of two stages: a **dialogue simulation stage** and a **dialogue-conditioned fine-tuning stage**.

**Stage 1**: Given a medical dataset $\mathcal{D}=\{(I_i, C_i)\}_{i=1}^N$, T-turn DocVLM–PatientVLM dialogues are simulated for each sample to produce an augmented dataset $\hat{\mathcal{D}}=\{(I_i, H_i, C_i)\}$.

**Stage 2**: DocVLM is fine-tuned on the augmented dataset to learn $P(C|I,H)$, i.e., diagnosis conditioned on both image and dialogue history.

### Key Designs

1. **DocVLM (Doctor Model)**:

    - Generates follow-up questions based on the medical image $I_i$, dialogue history $H_{i,<t}$, and the full set of candidate diagnoses $\mathcal{C}$
    - Core formulation: $Q_{i,t} = \text{DocVLM}(P_{doc}(I_i, H_{i,<t}, \mathcal{C}))$
    - Design Motivation: Including all candidate diagnoses in the prompt encourages the generation of discriminative questions that help distinguish between similar conditions

2. **PatientVLM (Patient Model)**:

    - Generates responses based on the image $I_i$, ground-truth diagnosis $C_i$, and DocVLM's question $Q_{i,t}$
    - Core formulation: $A_{i,t} = \text{PatientVLM}(P_{pat}(I_i, C_i, Q_{i,t}))$
    - Design Motivation: The ground-truth diagnosis guides symptom expression, while the model is **explicitly instructed not to disclose the diagnosis itself**, preserving information asymmetry
    - PatientVLM parameters remain frozen throughout dialogue simulation

3. **Iterative Dialogue Generation**:

    - DocVLM and PatientVLM interact for up to T rounds (T=8 in experiments)
    - Each round consists of one question from DocVLM and one response from PatientVLM
    - The process ultimately yields image–dialogue–diagnosis triplets

### Loss & Training
- Diagnosis classification is formulated as a **text generation problem**, with diagnoses generated autoregressively
- Standard generation loss: $\mathcal{L}_{gen}(\theta) = -\mathbb{E}_{(I,H,C)}\left[\sum_m \log P_\theta(C_m|C_{<m}, I, H)\right]$
- DocVLM is fine-tuned with LoRA: rank=16, alpha=32, dropout=0.05
- Training for 10 epochs with batch size=8
- mPLUG-Owl3 is used as the default PatientVLM in experiments

## Key Experimental Results

### Main Results

Evaluated on four datasets from MedMNIST v2:

| Model | Setting | DermaMNIST F1 | PneumoniaMNIST F1 | RetinaMNIST F1 | PathMNIST F1 |
|------|------|---------------|-------------------|----------------|--------------|
| InternVL3-2B | Image-only SFT | 36.5 | 88.4 | 31.5 | 70.9 |
| InternVL3-2B | +PCDF | **73.7(+37.2)** | **98.6(+10.2)** | **54.9(+23.4)** | **85.5(+14.6)** |
| Qwen2.5-VL-7B | Image-only SFT | 56.5 | 83.3 | 33.8 | 73.5 |
| Qwen2.5-VL-7B | +PCDF | **81.0(+24.5)** | **94.5(+11.2)** | **39.7(+5.9)** | **77.9(+4.4)** |
| Gemma3-4B | Image-only SFT | 78.3 | 95.7 | 47.7 | 86.0 |
| Gemma3-4B | +PCDF | **81.9(+3.6)** | **99.0(+3.3)** | **67.7(+20.0)** | **90.2(+4.2)** |
| MedGemma3-4B | Image-only SFT | 81.5 | 99.1 | 71.2 | 90.9 |
| MedGemma3-4B | +PCDF | **86.4(+4.9)** | **99.3(+0.2)** | **81.3(+10.1)** | **96.9(+6.0)** |

Key Findings: PCDF-augmented VLMs achieve an average F1 gain of 11.48, with general-purpose VLMs benefiting most (InternVL3 F1 +37.2).

### Ablation Study

**Dialogue Length Analysis (Gemma3 + mPLUG-Owl3)**:

| Turns T | DermaMNIST F1 | PneumoniaMNIST F1 | RetinaMNIST F1 | PathMNIST F1 |
|-----------|---------------|-------------------|----------------|--------------|
| 2 | 63.5 | 78.8 | 27.8 | 59.1 |
| 4 | 70.3 | 80.3 | 36.6 | 49.5 |
| 6 | 71.9 | 91.7 | 44.1 | 71.8 |
| 8 | **81.9** | **99.0** | **67.7** | **90.2** |

**PatientVLM Selection Analysis (DocVLM = Qwen2.5-VL-7B)**:

| PatientVLM | Avg. F1 | Notes |
|------------|--------|------|
| Image-only SFT | 61.8 | Baseline |
| mPLUG-Owl3 | **73.3** | Best PatientVLM |
| InternVL3 | 70.1 | Second best |
| Qwen2.5-VL | 72.7 | Same architecture, different role |
| MedGemma | 70.5 | Medical domain model |

### Key Findings

1. **General-purpose VLMs benefit more**: InternVL3 achieves F1 +37.2 on DermaMNIST due to the lack of medical domain pre-training
2. **Longer dialogues yield better performance**: T from 2 to 8 increases RetinaMNIST F1 from 27.8 to 67.7 (+39.9 absolute)
3. **PCDF outperforms CoT reasoning**: MedGemma with PCDF zero-shot F1 exceeds CoT by an average of 23.6
4. **Clinical validation**: 96.9% of simulated dialogues are rated as clinically relevant, with no cases of diagnostic leakage

## Highlights & Insights

- **Dual-VLM role separation** is an elegant design — it preserves the information asymmetry between doctor and patient inherent in real consultations, yielding greater authenticity than single-model generation
- **Model-agnostic**: PCDF is applicable to arbitrary VLMs without architectural modification, requiring only LoRA fine-tuning
- **Even medically specialized models such as MedGemma benefit**, indicating that dialogue-based supervisory signals are complementary to conventional domain adaptation
- **Zero-cost clinical dialogue data**: No real doctor–patient dialogues are required, entirely circumventing the ethical and financial barriers to data collection

## Limitations & Future Work

- Clinical validation is limited in scale (210 cases), necessitating larger and more diverse evaluations
- DocVLM-generated questions tend toward professional terminology, potentially difficult for lay patients to understand
- Current support is limited to English, constraining applicability in multilingual healthcare settings
- MedMNIST datasets are relatively straightforward; validation in more complex clinical scenarios (e.g., multi-morbidity) is lacking
- The quality of symptom generation by PatientVLM depends on the underlying VLM's medical knowledge

## Related Work & Insights

- Unlike evaluation-oriented works such as MedIQ and 3MDBench, PCDF is a **training framework**
- Inspired by real clinical consultation workflows: physicians do not only read images but also elicit symptoms
- Future work could extend PCDF to **multimodal** settings (incorporating laboratory test data) or **multilingual** scenarios

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (Dual-VLM dialogue simulation for medical consultation is an entirely novel framework design)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Four datasets + four VLMs + multi-dimensional ablation)
- Writing Quality: ⭐⭐⭐⭐⭐ (Clear problem motivation and naturally presented methodology)
- Value: ⭐⭐⭐⭐ (Demonstrates practical application potential in medical AI)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] SafeR-CLIP: Mitigating NSFW Content in Vision-Language Models While Preserving Pre-Trained Knowledge](safer-clip_mitigating_nsfw_content_in_vision-language_models_while_preserving_pr.md)
- [\[AAAI 2026\] TinyChemVL: Advancing Chemical Vision-Language Models via Efficient Visual Token Reduction and Complex Reaction Tasks](tinychemvl_advancing_chemical_vision-language_models_via_efficient_visual_token_.md)
- [\[AAAI 2026\] EM-KD: Distilling Efficient Multimodal Large Language Model with Unbalanced Vision Tokens](em-kd_distilling_efficient_multimodal_large_language_model_w.md)
- [\[AAAI 2026\] Difference Vector Equalization for Robust Fine-tuning of Vision-Language Models](difference_vector_equalization_for_robust_fine-tuning_of_vis.md)
- [\[AAAI 2026\] RMAdapter: Reconstruction-based Multi-Modal Adapter for Vision-Language Models (Oral)](rmadapter_reconstructionbased_multimodal_adapter_for_visionlanguage.md)

</div>

<!-- RELATED:END -->
