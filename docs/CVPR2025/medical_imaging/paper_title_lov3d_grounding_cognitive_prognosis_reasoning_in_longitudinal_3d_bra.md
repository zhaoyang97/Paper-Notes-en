---
title: >-
  [Paper Note] LoV3D: Grounding Cognitive Prognosis Reasoning in Longitudinal 3D Brain MRI via Regional Volume Assessments
description: >-
  [CVPR2025][Medical Imaging][Alzheimer's disease] LoV3D proposes an end-to-end longitudinal 3D brain MRI vision-language model pipeline. Through a structured verifiable output design, the framework achieves concurrent anatomical region assessment, longitudinal comparison, and three-class diagnostic reasoning. Fueled by a clinically-weighted verifier to drive Direct Preference Optimization (DPO) training without human annotation, it achieves a 93.7% three-class classification a…
tags:
  - "CVPR2025"
  - "Medical Imaging"
  - "Alzheimer's disease"
  - "3D VLM"
  - "longitudinal MRI"
  - "structured reasoning"
  - "DPO"
  - "normative Z-score"
date: 2026-05-08
content_hash: f6715f66119b59d1
---

# LoV3D: Grounding Cognitive Prognosis Reasoning in Longitudinal 3D Brain MRI via Regional Volume Assessments

**Conference**: CVPR2025  
**arXiv**: [2603.12071](https://arxiv.org/abs/2603.12071)  
**Code**: [GitHub](https://github.com/Anonymous-TEVC/LoV-3D)  
**Area**: Medical Imaging  
**Keywords**: Alzheimer's disease, 3D VLM, longitudinal MRI, structured reasoning, DPO, normative Z-score

## TL;DR

LoV3D proposes an end-to-end longitudinal 3D brain MRI vision-language model pipeline. Through a structured verifiable output design, the framework achieves concurrent anatomical region assessment, longitudinal comparison, and three-class diagnostic reasoning. Fueled by a clinically-weighted verifier to drive Direct Preference Optimization (DPO) training without human annotation, it achieves a 93.7% three-class classification accuracy on ADNI and zero non-adjacent diagnostic errors.

## Background & Motivation

- Alzheimer's disease (AD) is the leading cause of dementia, and longitudinal brain MRI is the core tool for tracking its progression.
- Existing tools operate in silos: classifiers only output labels (losing anatomical details), volumetric analysis pipelines (such as FreeSurfer) only provide numerical values without reasoning, and vision-language models (VLMs) may generate fluent but hallucinated statements.
- Clinically, neuroradiologists' reports are multi-tiered: anatomical observation $\rightarrow$ clinical context integration $\rightarrow$ comparison with prior scans $\rightarrow$ comprehensive impression. Automation requires structured reasoning rather than a single label.
- Existing 3D medical VLMs (e.g., RadFM, M3D-LaMed) exhibit a 0% JSON validity rate under zero-shot settings, failing to complete structured clinical reporting tasks.
- Key Insight: If the model output is designed as a verifiable structured JSON, hallucinations can be detected programmatically, and this exact structure can also drive automated training.

## Method

### Overall Architecture
3D Vision Encoder $\rightarrow$ Learnable Projector $\rightarrow$ Large Language Model (Qwen-2.5-14B + LoRA)

- **3D Encoder**: MONAI ResNet-50 with layer3 output extracted, generating a feature map of $1024 \times 16 \times 16 \times 16$, pooled into 512 visual tokens.
- **Projector**: A two-layer MLP (GELU) that maps 1024 dimensions to 5120 dimensions (matching Qwen's embedding space).
- **Text Input**: Demographics, APOE $\varepsilon_4$ status, cognitive scores (MMSE, CDR-SB), and FreeSurfer anatomical labels from the prior scan (the current scan's FreeSurfer results only serve as the verifier's ground truth and are invisible to the model).

### Structured Verifiable Output
The model outputs a JSON object containing qualitative fields (free-text reasoning) and verifiable fields, following a "reasoning-first, diagnosis-second" order:

- **C1 Region Selection Constraint**: Regions flagged as abnormal must appear in the reasoning text.
- **C2 Region Classification Constraint**: Neurodegeneration is irreversible; current labels cannot be more than two levels milder than the prior labels.
- **C3 Longitudinal Progression Constraint**: The direction of change (stable/progressive atrophy/progressive enlargement) must be consistent with the threshold crossing flags.

### Normative Z-Score Model
- Fits age- and sex-adjusted normative models to AD feature regions (using only CN subjects from the training set).
- Discretizes Z-scores into three levels: normal ($z > -0.5$), mild atrophy ($-1.5 < z \le -0.5$), and severe atrophy ($z \le -1.5$).
- Introduces a soft tolerance zone of $\pm 0.25$ Z at the boundaries to prevent boundary noise from affecting DPO signals.

### Clinically-Weighted Verifier
Comprehensive scoring function: 
$$S_{\text{verifier}} = M(\hat{d}, d^*) \cdot \sum_{c} \lambda_c S_c$$

- Global clinical multiplier $M$: non-adjacent diagnostic errors $\times 2.0$, adjacent errors $\times 1.5$.
- Five subcheck components: anatomy (0.25), diagnosis (0.25), longitudinal (0.20), reasoning (0.15), and summary (0.15).
- Hippocampus weight of 1.2, entorhinal cortex weight of 1.1 (reflecting priority in AD diagnosis).

### Four-Stage Training
1. **Stage 0**: Encoder warmup (regional volume regression on baseline scans), followed by freezing.
2. **Stage 1a**: Projector alignment (LLM frozen).
3. **Stage 1b**: Joint projector + LoRA training (differential learning rates).
4. **Stage 2**: Verifier-driven DPO ($K=4$ candidates, temperature 0.7, $\beta=0.1$).

## Key Experimental Results

### ADNI Test Set (479 scans, 258 subjects)

| Metric | LoV3D | LoV3D (no-grounding) | ResNet-50 | RadFM | M3D-LaMed |
|------|-------|---------------------|-----------|-------|-----------|
| 3-Class Accuracy | **93.7%** | 92.5% | 58.9% | 17.5% | 38.2% |
| Binary Accuracy AD/CN | **97.2%** | 96.4% | 87.8% | — | — |
| Regional Accuracy | **82.6%** | 80.7% | — | 41.4% | 49.5% |
| Cohen's $\kappa$ | **0.911** | 0.891 | 0.461 | — | — |
| Non-adjacent Errors | **0** | 1 | — | — | — |

### Ablation Study

| Stage | Accuracy | BLEU-4 | ROUGE-L | False Severe Rate $\downarrow$ |
|------|--------|--------|---------|----------|
| 1a (Projector) | 89.1% | .431 | .635 | 6.3% |
| 1b (+LoRA) | 93.3% | .354 | .558 | 4.1% |
| 2 (+DPO) | **93.7%** | **.584** | **.763** | **2.2%** |

- DPO improves BLEU-4 by 65%, ROUGE-L by 37%, and reduces the false severe rate by 46%.

### Cross-Site Zero-Shot Transfer

| Dataset | Accuracy | Characteristics |
|--------|--------|------|
| MIRIAD | **95.4%** | 100% dementia recall rate |
| AIBL (3-class) | **82.9%** | Exceeds the strongest published baseline by 6+ pp |

## Highlights & Insights

1. **Structured Verifiable Output**: Transforming VLM hallucination detection from an intractable problem into programmatically checkable constraint verification, offering a universal design paradigm.
2. **Fully Automated DPO Training**: Preference pairs are constructed via automated verifier scoring, shifting human annotation costs to zero and breaking the labelling bottleneck of RLHF.
3. **Crucial Role of Anatomical Grounding**: Removing the Stage 0 regional volume regression pre-training leads to 1 non-adjacent error, demonstrating that anatomical priors are vital for clinical safety.
4. **Non-monotonic Quality Trajectory**: SFT improves classification but degrades report quality (ROUGE-L .635 $\rightarrow$ .558), while DPO simultaneously recovers and exceeds both, revealing meaningful training dynamics.
5. **Cross-Site Zero-Shot Generalization**: High precision is maintained across scanners, cohorts, and datasets without domain adaptation, proving that the encoder learns scanner-independent anatomical representations.
6. **Zero confusion between CN and Dementia across all 479 test scans**: Ensuring outstanding clinical safety.

## Limitations & Future Work

1. **Dependency on FreeSurfer**: Ground truth is derived from FreeSurfer volumetric segmentation, which itself has inherent uncertainties on atrophied tissues.
2. **Limited to T1-weighted MRI**: Fails to exploit complementary information from other modalities (e.g., FLAIR, DWI, PET).
3. **No Distinction Between Amnestic vs Non-amnestic MCI**: Clinically, this subdivision has a significant impact on treatment strategies.
4. **Mild Atrophy Detection Remains a Bottleneck**: The accuracy is only 67.1%, showing limited progress in detecting the earliest and most clinically actionable stage.
5. **Single GPU (A100-80GB) Training**: The computational demand for Qwen-2.5-14B + LoRA remains high, hindering replication by resource-constrained institutions.
6. **Two Non-adjacent Errors Still Persist on AIBL**: Safety is slightly compromised under zero-shot transfer.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (Structured verifiability + automated DPO forms a highly elegant closed-loop design)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (ADNI 3-class classification + ablation + zero-shot transfer on two external datasets)
- Writing Quality: ⭐⭐⭐⭐⭐ (Rigorous logic, tight alignment between methodology and evaluation, and clear clinical motivations)
- Value: ⭐⭐⭐⭐⭐ (Provides broad insights for verifiable reasoning paradigms in medical VLMs)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Unified Brain Surface and Volume Registration](../../ICLR2026/medical_imaging/unified_brain_surface_and_volume_registration.md)
- [\[CVPR 2026\] PGR-Net: Prior-Guided ROI Reasoning Network for Brain Tumor MRI Segmentation](../../CVPR2026/medical_imaging/pgr-net_prior-guided_roi_reasoning_network_for_brain_tumor_mri_segmentation.md)
- [\[ICLR 2026\] A Cognitive Process-Inspired Architecture for Subject-Agnostic Brain Visual Decoding](../../ICLR2026/medical_imaging/a_cognitive_process-inspired_architecture_for_subject-agnostic_brain_visual_deco.md)
- [\[CVPR 2026\] Sketch2CT: Multimodal Diffusion for Structure-Aware 3D Medical Volume Generation](../../CVPR2026/medical_imaging/sketch2ct_multimodal_diffusion_for_structure-aware_3d_medical_volume_generation.md)
- [\[ICCV 2025\] M-Net: MRI Brain Tumor Sequential Segmentation Network via Mesh-Cast](../../ICCV2025/medical_imaging/m-net_mri_brain_tumor_sequential_segmentation_network_via_mesh-cast.md)

</div>

<!-- RELATED:END -->
