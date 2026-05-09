---
title: >-
  [Paper Note] Resp-Agent: An Agent-Based System for Multimodal Respiratory Sound Generation and Disease Diagnosis
description: >-
  [ICLR2026][Medical Imaging][Respiratory sound analysis] This paper proposes Resp-Agent, a closed-loop multi-agent framework that coordinates a controllable respiratory sound generator and a multimodal diagnoser via an active adversarial curriculum planner (Thinker-A2CA). Built upon a 229k-scale benchmark, the system achieves co-design of generation and diagnosis, substantially improving diagnostic performance on long-tail categories.
tags:
  - ICLR2026
  - Medical Imaging
  - Respiratory sound analysis
  - multimodal fusion
  - controllable audio generation
  - active adversarial curriculum learning
  - flow matching
  - data augmentation
date: 2026-05-08
content_hash: 9d7aa8c301d8d432
---

# Resp-Agent: An Agent-Based System for Multimodal Respiratory Sound Generation and Disease Diagnosis

**Conference**: ICLR2026
**arXiv**: [2602.15909](https://arxiv.org/abs/2602.15909)
**Code**: [github.com/zpforlove/Resp-Agent](https://github.com/zpforlove/Resp-Agent)
**Area**: Medical Imaging
**Keywords**: Respiratory sound analysis, multimodal fusion, controllable audio generation, active adversarial curriculum learning, flow matching, data augmentation

## TL;DR

This paper proposes Resp-Agent, a closed-loop multi-agent framework that coordinates a controllable respiratory sound generator and a multimodal diagnoser via an active adversarial curriculum planner (Thinker-A2CA). Built upon a 229k-scale benchmark, the system achieves co-design of generation and diagnosis, substantially improving diagnostic performance on long-tail categories.

## Background & Motivation

1. **Single-modality representation bottleneck**: Existing methods convert respiratory sounds into mel spectrograms for CNN processing, discarding phase information and transient events (e.g., crackles), and failing to capture millisecond-level clinically critical acoustic features.
2. **Lack of large-scale multimodal datasets**: Public respiratory sound datasets are small in scale, cover few diseases, and lack systematic text–audio paired supervision, severely limiting multimodal model development.
3. **Disconnect between analysis and generation**: Existing research focuses on diagnostic tasks such as classification and detection, while generative modeling remains largely unexplored, precluding the use of synthetic data to mitigate class imbalance and data scarcity.
4. **Insufficient shallow fusion**: Even when auxiliary metadata (demographics, symptoms, etc.) is available, existing methods employ only basic fusion techniques (e.g., concatenation followed by full attention), failing to enable deep cross-modal interaction.
5. **Non-targeted data augmentation**: Conventional augmentation strategies (e.g., SpecAugment) apply unconditional and untargeted general perturbations, incapable of precisely generating adversarial samples for model failure modes.
6. **Weak cross-domain generalization**: Most systems are evaluated only on in-distribution data, lacking rigorous cross-institution and cross-device evaluation protocols, which limits clinical deployability.

## Method

### Overall Architecture: Closed-Loop Multi-Agent System

Resp-Agent consists of three interacting modules:

- **Thinker-A2CA (Active Adversarial Curriculum Planner)**: Built upon DeepSeek-V3.2-Exp as the central controller, it parses diagnostic objectives and schedules tasks. Its core capability is to reuse model error profiles and calibrated confidence scores to identify diagnostic weaknesses, then direct the generator to synthesize hard-to-classify samples, forming a closed loop of "analyze → identify weaknesses → targeted synthesis → retraining."
- **Generator**: Responsible for controllable respiratory sound synthesis, operating in two stages.
- **Diagnoser**: Responsible for multimodal disease classification.

### Generator: Discrete Unit Planning + Conditional Flow Matching Reconstruction

**Stage 1: Resp-MLLM (Style-Conditioned Unit Generation)**

Qwen3-0.6B-Base is repurposed as a multimodal unit generator:

1. BEATs extracts frame-level features $Z \in \mathbb{R}^{T \times D}$ from reference audio, which are compressed into $K$ style descriptors via temporal pooling, then mapped to the LLM hidden space via a two-layer MLP to obtain $E^{\text{style}}$.
2. The input sequence consists of "diagnostic text $d$ (controlling *what* disease)" and "$K$ [AUDIO] placeholders (controlling *what* style)," achieving content–style disentanglement.
3. The LLM autoregressively predicts a discrete acoustic unit sequence from the BEATs codebook. During training, ~10% random masking is applied to prevent teacher-forcing leakage.

**Stage 2: Conditional Flow Matching (CFM) Decoding**

A Diffusion Transformer (DiT)-parameterized CFM decoder reconstructs discrete units into mel spectrograms, followed by the Vocos vocoder to produce waveforms. CFM learns the velocity field along the linear path $x_t = (1-t)x_0 + tx_1$ with a dual-path conditioning design: (i) content stream—unit indices are embedded and temporally interpolated; (ii) timbre stream—BEATs features are temporally averaged and broadcast.

### Diagnoser: Modality Weaving + Strategic Global Attention

**Input-level modality weaving**: EHR clinical text tokens and 496 audio placeholders are arranged into a single sequence; at the Longformer embedding layer, audio placeholders are replaced with projected BEATs features, enabling cross-modal interaction between text and audio from the very first layer.

**Strategic global attention**: On top of Longformer's sliding-window attention, three types of global tokens are assigned: (i) [CLS] classifier; (ii) [DESCRIPTION] EHR sentinel; (iii) audio anchors with stride $s=4$. The anchor spacing of ~80.6 ms provides sub-100 ms temporal resolution, enabling textual symptoms (e.g., "nocturnal dry cough") to directly query distant transient acoustic events, while maintaining linear time complexity.

### Resp-229k Benchmark Dataset

Five public databases are aggregated (UK COVID-19, ICBHI, SPRSound, COUGHVID, KAUH), yielding 229,101 recordings, 408 hours, and 16 diagnostic categories. Each recording is paired with a standardized clinical narrative distilled by an LLM. A strict cross-domain split is applied: the first three datasets are used for training/validation, and only COUGHVID and KAUH are used for testing.

## Key Experimental Results

### Table 1: Respiratory Sound Classification on ICBHI Official 60-40 Split

| Method | Backbone | Sp (%) | Se (%) | Score (%) |
|---|---|---|---|---|
| Dong et al. (2025) | AST | 85.99 | 49.11 | 67.55 |
| MVST (He et al., 2024) | AST | 81.99 | 51.10 | 66.55 |
| BTS (Kim et al., 2024c) | CLAP | 81.40 | 45.67 | 63.54 |
| **Resp-Agent [Ours]** | **LLM+Longformer** | **79.29** | **66.10** | **72.70** |

Resp-Agent achieves a Score of 72.7, surpassing the previous best by over 5 absolute points. In particular, sensitivity (Se) reaches 66.1%, far exceeding other methods (maximum 51.1%), demonstrating that multimodal fusion effectively improves recognition of minority classes.

### Table 2: Diagnostic Performance under Different Planning Strategies on Test-CD (Budget $B=50$k)

| Planning Strategy | Acc | Macro-F1 | Macro-F1_tail |
|---|---|---|---|
| No synthesis (CE baseline) | 0.849 | 0.212 | 0.074 |
| Random sampling | 0.869 | 0.442 | 0.291 |
| Class-prior rebalancing | 0.876 | 0.512 | 0.349 |
| Static uncertainty sampling | 0.881 | 0.546 | 0.376 |
| **Thinker-A2CA** | **0.887** | **0.598** | **0.421** |

Thinker-A2CA achieves the best performance across all metrics, improving Macro-F1 from the baseline of 0.212 to 0.598 (+182%) and tail-class Macro-F1_tail from 0.074 to 0.421 (+469%), demonstrating that the active adversarial curriculum substantially outperforms all passive strategies.

### Generator Content–Style Disentanglement Verification

In style-swapping experiments with fixed pathology labels and varying style references, Style-Sim = 0.91, Pathology-Acc = 97.9%, and FAD = 1.18, confirming that the generator can independently control acoustic style without altering disease semantics.

## Highlights & Insights

1. **Closed-loop co-design**: For the first time, respiratory sound analysis and generation are unified within a single agent framework, realizing an active learning loop of "diagnose to find weaknesses → generate targeted synthesis → retrain to improve," as opposed to conventional passive data augmentation.
2. **Resp-MLLM**: To the authors' knowledge, this is the first respiratory sound multimodal large language model trained under aligned text–audio supervision, enabling controllable disentangled generation of pathological content and acoustic style.
3. **Strategic audio anchors**: Sparse global attention anchors at ~80 ms intervals enable long-range routing from text to transient acoustic events at linear complexity, addressing the modeling challenge of millisecond-level events such as crackles in respiratory sounds.
4. **Resp-229k benchmark**: At 229k scale, 16 categories, cross-domain splits, and LLM-distilled clinical narratives, this dataset fills the gap for large-scale multimodal benchmarks in the respiratory sound domain.
5. **High sample efficiency**: Thinker-A2CA achieves ~52% of the total gain with only 10k synthetic samples, substantially outperforming class-prior and random sampling strategies.

## Limitations & Future Work

1. **Planner dependency on proprietary LLM**: Thinker-A2CA uses DeepSeek-V3.2-Exp as the planner, incurring high deployment costs and limiting full reproducibility; lighter-weight planning strategies warrant exploration.
2. **Generation quality ceiling**: CFM decoding still relies on mel spectrogram as an intermediate representation, which may be insufficiently precise for certain extremely fine-grained acoustic transients (e.g., faint crackles).
3. **Text supervision reliance on LLM distillation**: Clinical narratives are generated by an LLM rather than real EHRs, potentially introducing systematic bias; despite an auditing pipeline, hallucination risks cannot be fully eliminated.
4. **Limitations of the 16-class taxonomy**: Real clinical scenarios involve more complex respiratory diseases and multi-morbidity, which the current framework does not address with multi-label classification.
5. **Non-certified medical system**: The paper explicitly states that the system is not intended for clinical decision-making; actual deployment requires additional regulatory approval and clinical validation.

## Related Work & Insights

- **vs. OPERA/RespLLM**: OPERA provides domain pretraining but remains unimodal; RespLLM fuses text but employs dense full-attention modality concatenation, which is computationally expensive and lacks fine-grained cross-modal routing. Resp-Agent's modality weaving and sparse anchors achieve deeper cross-modal interaction at sub-quadratic complexity.
- **vs. SpecAugment/unconditional generation**: Conventional augmentation applies untargeted general perturbations, whereas Resp-Agent uses the Thinker to identify failure modes and synthesize targeted adversarial samples, transforming augmentation into a precise curriculum learning tool.
- **vs. AudioLM/SoundStorm**: General-purpose audio generation models do not account for clinical controllability. Resp-MLLM achieves pathology–timbre disentanglement via dual conditioning on diagnostic text and style reference, constituting a generation solution specifically designed for medical audio.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (Closed-loop multi-agent + respiratory sound multimodal LLM + active curriculum learning — multiple pioneering contributions)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (7 experimental groups covering diagnosis, generation, ablation, cross-domain, sample efficiency, disentanglement verification, and LoSO)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure and well-formatted equations, though the high system complexity requires repeated reference to the appendix for certain details)
- Value: ⭐⭐⭐⭐⭐ (Dataset, framework, and models are fully open-sourced, making a significant contribution to the medical audio AI field)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] LungNoduleAgent: A Collaborative Multi-Agent System for Precision Diagnosis of Lung Nodules](../../AAAI2026/medical_imaging/lungnoduleagent_a_collaborative_multi-agent_system_for_precision_diagnosis_of_lu.md)
- [\[ICLR 2026\] MMedAgent-RL: Optimizing Multi-Agent Collaboration for Multimodal Medical Reasoning](mmedagent-rl_optimizing_multi-agent_collaboration_for_multimodal_medical_reasoni.md)
- [\[ICLR 2026\] EMR-AGENT: Automating Cohort and Feature Extraction from EMR Databases](emr-agent_automating_cohort_and_feature_extraction_from_emr_databases.md)
- [\[ACL 2026\] MARCH: Multi-Agent Radiology Clinical Hierarchy for CT Report Generation](../../ACL2026/medical_imaging/march_multi-agent_radiology_clinical_hierarchy_for_ct_report_generation.md)
- [\[ICLR 2026\] Learning Patient-Specific Disease Dynamics with Latent Flow Matching for Longitudinal Imaging Generation](learning_patient-specific_disease_dynamics_with_latent_flow_matching_for_longitu.md)

</div>

<!-- RELATED:END -->
