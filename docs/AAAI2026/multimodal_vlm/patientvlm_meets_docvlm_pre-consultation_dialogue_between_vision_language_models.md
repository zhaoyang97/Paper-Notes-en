---
title: >-
  [Paper Note] PatientVLM Meets DocVLM: Pre-Consultation Dialogue Between Vision-Language Models for Efficient Diagnosis
description: >-
  [AAAI 2026][Multimodal VLM][Medical Diagnosis] This paper proposes PCDF (Pre-Consultation Dialogue Framework), which simulates realistic doctor-patient dialogue through two VLMs in role-play — DocVLM asks questions and P…
tags:
  - "AAAI 2026"
  - "Multimodal VLM"
  - "Medical Diagnosis"
  - "Vision-Language Models"
  - "Doctor-Patient Dialogue Simulation"
  - "Multi-turn Dialogue"
  - "Data Augmentation"
date: 2026-05-08
content_hash: 0634e006db90d991
---

# PatientVLM Meets DocVLM: Pre-Consultation Dialogue Between Vision-Language Models for Efficient Diagnosis

**Conference**: AAAI 2026
**arXiv**: [2601.10945](https://arxiv.org/abs/2601.10945)  
**Code**: [https://vl2g.github.io/projects/pcdf](https://vl2g.github.io/projects/pcdf)  
**Area**: Multimodal VLM
**Keywords**: Medical Diagnosis, Vision-Language Models, Doctor-Patient Dialogue Simulation, Multi-turn Dialogue, Data Augmentation

## TL;DR
This paper proposes PCDF (Pre-Consultation Dialogue Framework), which simulates realistic doctor-patient dialogue through two VLMs in role-play — DocVLM asks questions and PatientVLM answers — to generate image-dialogue-diagnosis triplets for fine-tuning DocVLM. The framework achieves an average F1 improvement of 11.48 percentage points across four medical imaging benchmarks without relying on real clinical dialogue data.

## Background & Motivation

**Background**: AI-assisted medical diagnosis has been a long-standing research direction. Early approaches relied on CNNs for image classification; subsequently, CLIP and its medical adaptations (MedCLIP, BioMedCLIP) introduced vision-text alignment; more recently, VLMs (e.g., LLaVA-Med, MedPaLM2, MedGemma) have demonstrated strong zero-shot generalization.

**Limitations of Prior Work**: Existing methods reduce diagnosis to a direct "image → diagnosis" mapping, overlooking the importance of clinical context. In real clinical practice, physicians rarely diagnose from images alone — they conduct multi-turn dialogues with patients to inquire about symptoms and medical history, progressively narrowing down possibilities. This dialogue-driven diagnostic reasoning is central to accurate diagnosis, yet current models are entirely disconnected from this process.

**Key Challenge**: How can VLMs acquire dialogue-aware diagnostic capabilities? The ideal solution is to collect real doctor-patient dialogue data for training, but this faces substantial barriers:
- Real medical dialogues involve sensitive privacy, requiring IRB approval and informed patient consent
- Clinicians are reluctant to participate due to concerns about workflow disruption, medicolegal risks, and patient trust
- Large-scale data collection is practically infeasible

**Limitations of Prior Work**: Previous work used a single LLM to simultaneously play both doctor and patient roles to generate synthetic dialogues, but suffered from two fundamental flaws: (i) operation was limited to text-only settings without incorporating medical images; (ii) a single model playing dual roles resulted in dialogues lacking genuine role separation and authenticity.

**Key Insight**: Two independent VLMs are assigned to play the doctor and patient roles respectively, conducting natural multi-turn interactions conditioned on images and dialogue history to generate visual-dialogue-diagnosis triplets for training. The key innovation is that PatientVLM generates symptom responses based on ground-truth diagnoses while being explicitly instructed not to reveal the diagnosis itself, thereby preserving the information asymmetry characteristic of real clinical consultations.

## Method

### Overall Architecture

The PCDF framework consists of two stages:
1. **Dialogue Simulation Stage**: DocVLM and PatientVLM interact over $T$ turns to generate image-dialogue-diagnosis triplets.
2. **Dialogue-Conditioned Fine-tuning Stage**: DocVLM is fine-tuned on the generated triplets to learn diagnosis conditioned on both images and dialogue history.

### Key Designs

1. **DocVLM (Doctor Model)**
    - **Function**: Generates clinically relevant follow-up questions based on the medical image and dialogue history.
    - **Core Formula**: $Q_{i,t} = \text{DocVLM}(P_{doc}(I_i, H_{i,<t}, \mathcal{C}))$
    - Inputs include: image $I_i$, dialogue history up to the current turn $H_{i,<t}$, and the set of all possible diagnostic categories $\mathcal{C}$.
    - **Design Motivation**: Including all possible diagnoses in the prompt encourages DocVLM to pose discriminative questions that help distinguish among different candidate diagnoses (adapted from the strategy of Kurz et al. 2025).

2. **PatientVLM (Patient Model)**
    - **Function**: Acts as a pseudo-patient, generating responses to the doctor's questions based on ground-truth diagnoses.
    - **Core Formula**: $A_{i,t} = \text{PatientVLM}(P_{pat}(I_i, C_i, Q_{i,t}))$
    - **Key Constraint**: Although diagnostic information is used internally to guide symptom expression, the model is explicitly instructed not to disclose the diagnosis.
    - **Design Motivation**: This preserves the information asymmetry of real clinical consultations — the doctor does not know the diagnosis, and the patient can describe symptoms but cannot directly reveal the answer.

3. **Dialogue-Conditioned Fine-tuning**
    - **Function**: Fine-tunes DocVLM on the generated augmented dataset $\hat{D} = \{I_i, H_i, C_i\}$.
    - **Core Loss**: Diagnosis classification is modeled as a text generation task using standard generation loss:
    $$\mathcal{L}_{gen}(\theta) = -\mathbb{E}_{(I,H,C)}\left[\sum_m \log P_\theta(C_m \mid C_{<m}, I, H)\right]$$
    - **Design Motivation**: Enables DocVLM to learn $P(C|I,H)$ — joint diagnostic reasoning conditioned on both image and dialogue history — rather than relying solely on the image.

### Loss & Training

- mPLUG-Owl3 is used as the default PatientVLM.
- Number of dialogue turns: $T = 8$.
- DocVLM is fine-tuned with LoRA (rank=16, alpha=32, dropout=0.05).
- Training for 10 epochs with batch size=8.
- Both VLMs remain frozen during the dialogue simulation stage.

## Key Experimental Results

### Main Results

| Model / Dataset | DermaMNIST F1 | PneumoniaMNIST F1 | RetinaMNIST F1 | PathMNIST F1 |
|----------------|--------------|------------------|---------------|-------------|
| **InternVL3-2B** | | | | |
| Image-only SFT | 36.5 | 88.4 | 31.5 | 70.9 |
| +PCDF | **73.7** (+37.2) | **98.6** (+10.2) | **54.9** (+23.4) | **85.5** (+14.6) |
| **Gemma3-4B** | | | | |
| Image-only SFT | 78.3 | 95.7 | 47.7 | 86.0 |
| +PCDF | **81.9** (+3.6) | **99.0** (+3.3) | **67.7** (+20.0) | **90.2** (+4.2) |
| **MedGemma3-4B** | | | | |
| Image-only SFT | 81.5 | 99.1 | 71.2 | 90.9 |
| +PCDF | **86.4** (+4.9) | **99.3** (+0.2) | **81.3** (+10.1) | **96.9** (+6.0) |
| **Qwen2.5-VL-7B** | | | | |
| Image-only SFT | 56.5 | 83.3 | 33.8 | 73.5 |
| +PCDF | **81.0** (+24.5) | **94.5** (+11.2) | **39.7** (+5.9) | **77.9** (+4.4) |

### Ablation Study

**Effect of Dialogue Turns (Gemma3 as DocVLM)**:

| Turns $T$ | DermaMNIST F1 | PneumoniaMNIST F1 | RetinaMNIST F1 | PathMNIST F1 |
|----------|--------------|------------------|---------------|-------------|
| 2 | 63.5 | 78.8 | 27.8 | 59.1 |
| 4 | 70.3 | 80.3 | 36.6 | 49.5 |
| 6 | 71.9 | 91.7 | 44.1 | 71.8 |
| 8 | **81.9** | **99.0** | **67.7** | **90.2** |

**Effect of PatientVLM Selection (Qwen2.5-VL-7B as DocVLM)**:

| PatientVLM | Avg. F1 | Note |
|-----------|---------|------|
| Image-only SFT | 61.8 | No-dialogue baseline |
| InternVL3 | 70.1 | +8.3 |
| MedGemma | 70.5 | +8.7 |
| Qwen2.5-VL | 72.7 | +10.9 |
| mPLUG-Owl3 | **73.3** | **+11.5**, best performance |

### Key Findings

1. **General VLMs benefit more**: The F1 gains for InternVL3 and Qwen2.5-VL substantially exceed those of MedGemma, which already benefits from medical pre-training, as general-purpose models lack medical supervision.
2. **Domain models also improve significantly**: Even MedGemma achieves notable gains (RetinaMNIST F1: 71.2 → 81.3), indicating that dialogue supervision complements pre-trained knowledge.
3. **Longer dialogues are consistently better**: As $T$ increases from 2 to 8, F1 improves monotonically, with the longest dialogue yielding a 39.9 percentage-point gain on RetinaMNIST.
4. **Zero-shot PCDF outperforms CoT**: Applying PCDF dialogue without fine-tuning still outperforms Chain-of-Thought prompting.
5. **Clinical validation passed**: 96.9% of 1,680 QA pairs were rated as clinically relevant by clinical experts, with no diagnosis leakage observed.

## Highlights & Insights

1. **Elegant dual-VLM role separation**: Compared to single-model dialogue generation, assigning distinct roles to two VLMs produces more authentic interactions; the information asymmetry constraint ensures PatientVLM does not leak the diagnosis.
2. **Model-agnostic general framework**: PCDF can be combined with any VLM and demonstrates effectiveness on both general-purpose and medical VLMs.
3. **Scalable solution without real data**: The approach entirely circumvents the ethical and cost barriers of collecting real doctor-patient dialogue data.
4. **Clinical validation enhances credibility**: Evaluation by licensed clinicians (albeit at limited scale) confirms the clinical relevance of the synthesized symptoms.
5. **Elegant problem reformulation**: The question of "how to make VLMs diagnose better" is reframed as "how to make VLMs ask questions like a physician."

## Limitations & Future Work

1. **Limited clinical validation scale**: Only 210 cases were evaluated by medical professionals; larger-scale and more demographically diverse assessment is needed.
2. **Overly technical questions**: Some follow-up questions generated by DocVLM are excessively specialized and may be difficult for lay patients to understand.
3. **English-only support**: This limits applicability in multilingual healthcare settings.
4. **Dataset constraints**: Validation is conducted only on MedMNIST v2 (low-resolution images) and has not been tested on large-scale real clinical data.
5. **Symptom generation quality of PatientVLM**: Generating symptoms conditioned on ground-truth diagnoses may produce "idealized" symptom descriptions that diverge from authentic patient experiences.

## Related Work & Insights

- **MedIQ** focuses on the quality of medical question generation but only provides evaluation, not a training methodology.
- **3MDBench** assesses diagnostic capability through text-driven persona-based dialogues.
- PCDF shares conceptual similarities with self-play in LLMs, but is applied to generate training data rather than directly optimizing a policy.
- Insight: The dual-VLM dialogue paradigm may be extensible to other scenarios requiring interactive reasoning, such as legal consultation or educational tutoring.
- Data augmentation perspective: Enriching training data through synthetic dialogue is a paradigm with potential for generalization to other data-scarce domains.

## Rating
- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] SafeR-CLIP: Mitigating NSFW Content in Vision-Language Models While Preserving Pre-Trained Knowledge](safer-clip_mitigating_nsfw_content_in_vision-language_models_while_preserving_pr.md)
- [\[ACL 2026\] More Than Meets the Eye: Measuring the Semiotic Gap in Vision-Language Models via Semantic Anchorage](../../ACL2026/multimodal_vlm/more_than_meets_the_eye_measuring_the_semiotic_gap_in_vision-language_models_via.md)
- [\[AAAI 2026\] TinyChemVL: Advancing Chemical Vision-Language Models via Efficient Visual Token Reduction and Complex Reaction Tasks](tinychemvl_advancing_chemical_vision-language_models_via_efficient_visual_token_.md)
- [\[NeurIPS 2025\] Are Vision Language Models Ready for Clinical Diagnosis? A 3D Medical Benchmark for Tumor-centric Visual Question Answering](../../NeurIPS2025/multimodal_vlm/are_vision_language_models_ready_for_clinical_diagnosis_a_3d_medical_benchmark_f.md)
- [\[ACL 2026\] Efficient Inference for Large Vision-Language Models: Bottlenecks, Techniques, and Prospects](../../ACL2026/multimodal_vlm/efficient_inference_for_large_vision-language_models_bottlenecks_techniques_and_.md)

</div>

<!-- RELATED:END -->
