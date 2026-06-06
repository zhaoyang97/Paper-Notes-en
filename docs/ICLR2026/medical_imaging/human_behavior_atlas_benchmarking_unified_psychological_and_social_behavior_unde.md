---
title: >-
  [Paper Note] Human Behavior Atlas: Benchmarking Unified Psychological and Social Behavior Understanding
description: >-
  [ICLR 2026][Medical Imaging][behavior understanding benchmark] This work introduces Human Behavior Atlas—the first large-scale multimodal unified benchmark for behavior understanding spanning four dimensions (affective…
tags:
  - "ICLR 2026"
  - "Medical Imaging"
  - "behavior understanding benchmark"
  - "psychological and social behavior"
  - "multimodal learning"
  - "unified model"
  - "affective computing"
date: 2026-05-08
content_hash: 94b0e7164adc8213
---

# Human Behavior Atlas: Benchmarking Unified Psychological and Social Behavior Understanding

**Conference**: ICLR 2026  
**arXiv**: [2510.04899](https://arxiv.org/abs/2510.04899)  
**Code**: To be released after review  
**Area**: Medical Imaging  
**Keywords**: behavior understanding benchmark, psychological and social behavior, multimodal learning, unified model, affective computing

## TL;DR

This work introduces Human Behavior Atlas—the first large-scale multimodal unified benchmark for behavior understanding spanning four dimensions (affective, cognitive, pathological, and social processes) with 101K+ samples—and trains three OmniSapiens-7B model variants to validate its effectiveness in multi-task training and transfer learning.

## Background & Motivation

Leveraging intelligent systems to perceive psychological and social behavior—i.e., the emotional, cognitive, and pathological states manifested through observable actions and social interactions—has long been a core challenge in AI. Key problems with existing work:

**Fragmentation**: Each task (sentiment analysis, depression detection, action recognition, etc.) has dedicated datasets and single-task systems, lacking cross-task scalability and transfer capability.

**Format inconsistency**: Datasets are highly heterogeneous in input representations (pre-extracted features vs. raw signals), output formats (subjective annotations vs. categorical labels), and evaluation protocols.

**Redundant effort**: Each task requires independent architecture design, data collection, and training pipelines, resulting in substantial resource waste.

**Lack of unified models**: The community has made limited progress in training models capable of simultaneously understanding affective, cognitive, pathological, and social behavior.

Human Behavior Atlas aims to fill this gap by standardizing data formats and evaluation metrics to advance the development of general-purpose behavior understanding models.

## Method

### Overall Architecture

Human Behavior Atlas is constructed following a five-step pipeline: (1) define a behavioral taxonomy; (2) collect aligned multimodal datasets; (3) unify data format into prompt-target pairs; (4) standardize evaluation metrics; (5) extract behavioral descriptors to enrich the benchmark.

### Key Designs

1. **Behavioral Taxonomy (Four Dimensions)**:

    - **Affective States**: Emotions and moods—ranging from transient feelings (anger, joy) to sustained affect.
    - **Cognitive States**: Internal mental processes such as attention, reasoning, surprise, and decision-making.
    - **Pathology**: Psychological and psychiatric conditions—depression, anxiety, etc.
    - **Social Processes**: Social interaction and communicative behavior—humor, intent, cooperation, etc.
    - A single task may correspond to multiple dimensions (e.g., emotion recognition involves both affective and cognitive dimensions).

2. **Dataset Collection and Unification (13 Public Datasets)**:

    - Covers 10 behavioral tasks: sentiment polarity (SEN), emotion recognition (EMO), social reasoning (SOC), intent recognition (INT), nonverbal communication (NVC), humor detection (HUM), sarcasm detection (SAR), anxiety detection (ANX), depression detection (DEP), and PTSD detection (PTSD).
    - Totaling 101,964 samples across text, audio, and video modalities.
    - Unified into prompt-target format: prompts reference available modalities; targets are free-form text or discrete label sets.
    - Continuous outputs (e.g., PHQ-9 scores) are discretized following original paper guidelines.

3. **Standardized Evaluation Framework**:

    - SEN: binary weighted F1 (positive/negative sentiment).
    - EMO: mean of per-class weighted accuracy.
    - HUM/SAR/ANX/DEP/PTSD: weighted F1.
    - SOC/INT/NVC: LLM-Judge accuracy (GPT-5-nano judges whether generated responses match reference answers).
    - Emotion labels are unified (merging joy/happiness; distinguishing positive/negative surprise).

4. **Behavioral Descriptor Extraction**:

    - **Visual**: MediaPipe extracts facial landmarks and body pose keypoints.
    - **Audio**: OpenSMILE (ComParE 2016) extracts prosodic features (pitch, energy, spectral properties).
    - **Text**: Whisper v3 Large transcribes missing textual content.

5. **Three OmniSapiens-7B Model Variants**:

    - **OmniSapiens-7B SFT**: Supervised fine-tuning based on Qwen2.5-Omni-7B, using penultimate-layer representations processed through classification and decoding heads for different task types.
    - **OmniSapiens-7B BAM**: A Behavioral Adapter Module (residual-style adapter) appended after SFT backbone freezing to integrate behavioral descriptors. Formula: $h_{\text{adapt}} = h_{\text{penult}} + \alpha \cdot z_f$, where $z_f$ is obtained by processing behavioral descriptors through an FFN.
    - **OmniSapiens-7B RL**: Trained with GRPO (Group Relative Policy Optimization), uniformly using the decoding head and chain-of-thought format `<think>...</think>\boxed{answer}`.

### Loss & Training

- SFT: 5 epochs, LoRA ($r=32, \alpha=64$), learning rate $1 \times 10^{-4}$, batch size 512.
- BAM: 4 epochs, backbone frozen, only adapter and heads trained, hidden dimension 256.
- RL: 10 epochs, learning rate $1 \times 10^{-6}$, group sampling $n=5$, composite reward function (accuracy + format + semantic similarity).

## Key Experimental Results

### Main Results (Multi-Task Training)

| Model | EMO | HUM | INT | PTSD | ANX | DEP | SEN | SAR | SOC | NVC |
|------|-----|-----|-----|------|-----|-----|-----|-----|-----|-----|
| Gemma-3-4B | .550 | .597 | .227 | .499 | .601 | .463 | .738 | .529 | .191 | .023 |
| Qwen2.5-Omni-7B | .583 | .543 | .254 | .760 | .793 | .714 | .672 | .656 | .254 | .069 |
| HumanOmniV2-7B | .597 | .638 | .263 | .824 | .527 | .654 | .742 | .395 | .282 | .093 |
| **OmniSapiens-7B SFT** | **.631** | .532 | .256 | **1.00** | .909 | .733 | **.768** | .624 | .257 | .121 |
| **OmniSapiens-7B BAM** | **.645** | **.644** | .177 | **1.00** | **.909** | **.789** | **.786** | **.795** | .201 | **.162** |
| **OmniSapiens-7B RL** | .573 | **.639** | **.486** | .968 | .919 | .772 | .396 | .647 | **.304** | .133 |

SFT and BAM outperform general-purpose multimodal LLMs on 8 out of 10 tasks.

### Transfer Learning Results

| Dataset | OmniSapiens-7B SFT | Qwen2.5-Omni-7B | Gain |
|--------|-------------------|----------------|------|
| MOSEI (SEN) | 0.724 | 0.612 | +18.3% |
| MELD (EMO) | 0.711 | 0.684 | +3.95% |
| DAIC-WOZ (DEP) | 0.749 | 0.579 | +29.4% |
| MUStARD (SAR) — novel task | 0.658 | 0.473 | **+39.1%** |

### Ablation Study: Effect of Behavioral Descriptors (BAM vs. SFT)

| Task | SFT | BAM | Change |
|------|-----|-----|------|
| NVC | 0.12 | 0.16 | +33.0% |
| SAR | 0.62 | 0.80 | +29.0% |
| HUM | 0.53 | 0.64 | +21.0% |
| DEP | 0.73 | 0.79 | +8.2% |
| SOC | 0.26 | 0.20 | -23.1% |
| INT | 0.26 | 0.18 | -30.8% |

### Key Findings

- **Complementarity of SFT and RL**: SFT is stronger on structured classification tasks, while RL excels on open-ended generative tasks (INT, SOC), demonstrating the complementary nature of the two training strategies.
- **Selective benefits of behavioral descriptors**: BAM yields substantial gains on tasks relying on subtle facial/vocal cues (NVC, SAR, HUM), but degrades performance on reasoning-intensive tasks (SOC, INT), suggesting descriptors should be applied selectively rather than globally.
- **Pragmatic recognition supported by pretraining**: On sarcasm detection, OmniSapiens-7B recognizes pragmatic cues (e.g., Chandler's irony), whereas Qwen2.5-Omni-7B defaults to predicting "no sarcasm" (93.2% prediction rate).
- **Cross-task transfer**: Even for tasks not seen during pretraining (SAR), pretraining on Human Behavior Atlas yields a 39.1% transfer improvement.

## Highlights & Insights

1. **Systematic benchmark construction methodology**: Beyond providing datasets, the paper proposes a methodological framework for constructing a "behavior atlas"—from taxonomy definition, data standardization, and metric unification to model evaluation—generalizable to specific domains such as autism.
2. **Integration of end-to-end learning and feature engineering**: The residual BAM adapter enables non-invasive integration of behavioral descriptors, preserving backbone representations while selectively enhancing task-specific performance.
3. **Potential of RL for behavior understanding**: OmniSapiens-7B RL demonstrates the unique advantages of reinforcement learning on reasoning-intensive social understanding tasks, suggesting directions for future hybrid training strategies.
4. **Dataset diversity**: Datasets are sourced from multiple regions across North America, Europe, and Asia, providing a degree of cultural diversity.

## Limitations & Future Work

1. **Sample imbalance**: Data volume varies substantially across tasks (CMU-MOSEI 31K vs. DAIC-WOZ 189), potentially affecting multi-task training balance.
2. **Reliance on LLM Judge**: SOC/INT/NVC evaluation uses GPT-5-nano as judge; its consistency and bias are not thoroughly analyzed.
3. **Lack of real-world validation**: All data originate from laboratory or scripted media scenarios, creating a gap with naturalistic interactions.
4. **Subjectivity in emotion label merging**: The decisions to merge joy/happiness and to split surprise lack rigorous theoretical justification.
5. **Limited model scale**: Only 7B-parameter models are evaluated; scaling effects remain unexplored.
6. **Privacy and ethics**: The use of real human behavioral data raises privacy and informed consent concerns that are insufficiently addressed in the paper.

## Related Work & Insights

- **eMotions (Wu et al., 2025)**: A short-video sentiment analysis dataset, but limited to the single task of emotion recognition.
- **HumanOmni (Zhao et al., 2025)**: A human-centric understanding dataset, primarily targeting human scene understanding rather than psychological behavior.
- **PaLI / BLIP / Kosmos**: Exemplars of large-scale multimodal pretraining, demonstrating the generalization capability of multi-task pretraining.
- **Affective Computing (Picard, 2000)**: Foundational work in affective computing; this paper extends its scope to cognitive, pathological, and social dimensions.

## Rating

- Novelty: ⭐⭐⭐⭐ — First unified benchmark covering four behavioral dimensions; the methodological framework has broad applicability.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Three model variants with comprehensive multi-task, transfer, and descriptor ablation analyses.
- Writing Quality: ⭐⭐⭐⭐ — Well-structured with intuitive data presentation; some details require consulting the appendix.
- Value: ⭐⭐⭐⭐ — Fills the gap in unified behavior understanding benchmarks and provides important research infrastructure for the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Faithfulness vs. Safety: Evaluating LLM Behavior Under Counterfactual Medical Evidence](../../ACL2026/medical_imaging/faithfulness_vs_safety_evaluating_llm_behavior_under_counterfactual_medical_evid.md)
- [\[CVPR 2026\] Unlocking Multi-Site Clinical Data: A Federated Approach to Privacy-First Child Autism Behavior Analysis](../../CVPR2026/medical_imaging/unlocking_multi-site_clinical_data_a_federated_approach_to_privacy-first_child_a.md)
- [\[NeurIPS 2025\] A Unified Solution to Video Fusion: From Multi-Frame Learning to Benchmarking](../../NeurIPS2025/medical_imaging/a_unified_solution_to_video_fusion_from_multi-frame_learning_to_benchmarking.md)
- [\[ICLR 2026\] DM4CT: Benchmarking Diffusion Models for Computed Tomography Reconstruction](dm4ct_benchmarking_diffusion_models_for_computed_tomography_reconstruction.md)
- [\[ICLR 2026\] Benchmarking ECG FMs: A Reality Check Across Clinical Tasks](benchmarking_ecg_fms_a_reality_check_across_clinical_tasks.md)

</div>

<!-- RELATED:END -->
