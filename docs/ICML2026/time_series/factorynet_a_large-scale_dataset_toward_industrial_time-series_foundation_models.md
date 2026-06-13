---
title: >-
  [Paper Note] FactoryNet: A Large-Scale Dataset toward Industrial Time-Series Foundation Models
description: >-
  [ICML 2026][Time Series][Industrial Time-Series] FactoryNet is the first large-scale industrial time-series dataset with a unified control-loop structure—51 million data points / 23,000 end-to-end task executions (13…
tags:
  - "ICML 2026"
  - "Time Series"
  - "Industrial Time-Series"
  - "Anomaly Detection"
  - "Cross-Entity Transfer"
  - "S-E-F-C Schema"
  - "Predictive Maintenance"
date: 2026-05-08
content_hash: 11499b6903a71e53
---

# FactoryNet: A Large-Scale Dataset toward Industrial Time-Series Foundation Models

**Conference**: ICML 2026  
**arXiv**: [2605.09081](https://arxiv.org/abs/2605.09081)  
**Code**: https://github.com/Forgis-Labs/FactoryNet  
**Area**: Time-Series Anomaly Detection / Industrial Time-Series Foundation Models / Datasets  
**Keywords**: Industrial Time-Series, Anomaly Detection, Cross-Entity Transfer, S-E-F-C Schema, Predictive Maintenance

## TL;DR
FactoryNet is the first large-scale industrial time-series dataset with a unified control-loop structure—51 million data points / 23,000 end-to-end task executions (13,300 real + 9,800 simulated) across 6 machine entities. All signals are aligned according to the Setpoint-Effort-Feedback-Context (S-E-F-C) cybernetic classification. With 27 types of annotated anomalies, healthy baselines, and counterfactual pairs, it enables zero-shot cross-entity transfer and parameter-efficient anomaly detection.

## Background & Motivation

**Background**: Manufacturing accounts for approximately 15% of global GDP and relies on the continuous operation of complex machinery. Foundation models in vision and language have been revolutionary, but industrial time-series foundation models do not yet exist—Industrial AI remains largely customized for single-machine deployments.

**Limitations of Prior Work**: (1) Existing anomaly detection and prediction datasets (NASA C-MAPSS, CWRU, PHM 2010, etc.) only record sensor results without separating "command intent" from "measurement response." To learn transfer dynamics for actuated systems, the complete control loop (target trajectory $\to$ execution effort $\to$ physical state) must be observed. (2) Dataset scales are small and restricted to single machines—voraus-AD (2,122 episodes) and AURSAD (2,045 episodes) are insufficient for training foundation models. (3) Heterogeneous datasets lack a unified schema, making cross-machine alignment difficult. (4) General time-series anomaly detection benchmarks (SKAB / MetroPT / TSB-AD, etc.) record overall system states without command-measurement decomposition.

**Key Challenge**: Training industrial foundation models requires: (a) large scale (millions of data points); (b) cross-entity coverage (multiple machine types); (c) control-loop structure (distinguishing intent from result). No existing dataset satisfies all three simultaneously.

**Goal**: (1) Release the first multi-entity, large-scale industrial time-series dataset with a unified schema; (2) Propose the S-E-F-C cybernetic schema to map any actuated system to a common representation; (3) Demonstrate the feasibility of zero-shot cross-entity transfer and efficient anomaly detection; (4) Serve as a growing dataset to drive community progress.

**Key Insight**: Starting from control theory—signals are classified into four categories: Setpoint (command intent), Effort (execution), Feedback (measurement response), and Context (boundary conditions). This is a natural extension of IEC 81346 functional classification. S-E-F-C enables direct comparative analysis such as "sim-to-real mismatch = forward-model error under matched inputs."

**Core Idea**: The S-E-F-C schema unifies the encoding of all actuated systems $\to$ transforming cross-entity transfer and anomaly detection into schema-aligned operations $\to$ providing an "ImageNet" level pre-training corpus for industrial foundation models.

## Method

### Overall Architecture

Dataset composition:
- 51M data points / 23k end-to-end task executions
- 13.3k real (recorded in lab) + 9.8k simulated (Isaac Sim)
- 6 machine entities (including UR3e, collaborative robots, CNC, etc.)
- 27 types of annotated anomalies + healthy baselines + counterfactual pairs
- 3 manipulation tasks (different setups)

Each signal is mapped to one of the 4 S-E-F-C categories:
- **Setpoint**: Commanded position/velocity/torque/...
- **Effort**: Current/torque output/PWM
- **Feedback**: Encoder position/acceleration/vibration/temperature
- **Context**: Workpiece information/environment/load

Sim-to-real pairing quantifies the sim2real gap (forward model error under identical inputs).

### Key Designs

1. **S-E-F-C Cybernetic Schema**:
    - **Function**: Maps signals from any actuated system to four common category representations.
    - **Mechanism**: References IEC 81346 functional classification, dividing signals into Setpoint (intent), Effort (actuation), Feedback (measurement), and Context (boundary). Schema-aware annotations are applied to raw signal channels of each machine entity. Model input consists of these 4 category combinations rather than raw channels.
    - **Design Motivation**: Previous datasets only provided raw channel numbers or machine-specific naming, preventing cross-machine alignment. S-E-F-C extracts a layer of physical meaning, allowing UR3e and CNC signals to reside in the same representation space—a prerequisite for foundation models.

2. **Multi-Entity + Simulation Pairing**:
    - **Function**: Leverages real data for fidelity and simulation for scale; sim-real pairs enable analysis of the sim2real gap.
    - **Mechanism**: 13.3k real + 9.8k simulated data points share the same schema, aligning (real, sim) pairs for the same task execution. Models can learn "what effort and feedback should be given a setpoint+context"; the sim-real difference represents forward-model error.
    - **Design Motivation**: Purely real data lacks sufficient scale (hundreds), while purely simulated data suffers from sim2real gaps. Pairing allows them to complement each other—simulation provides coverage, while real data provides calibration and quantification of the gap.

3. **27 Anomaly Types + Counterfactual Pairs**:
    - **Function**: Rich annotations cover the spectrum of industrial anomalies, and counterfactual pairs support causal learning.
    - **Mechanism**: Covers classic mechanical faults (bearing wear, misalignment, unbalance) to electrical (power supply fault), control (PID instability), and process faults (tool wear, collision). Each anomaly has a paired healthy baseline, allowing the model to learn "what healthy behavior looks like."
    - **Design Motivation**: Previous datasets mostly featured single-fault types. These 27 categories enable the model to learn as a "general anomaly detector." Counterfactual pairs support contrastive learning and causal attribution.

## Key Experimental Results

### Cross-Dataset Scale Comparison

| Dataset | Year | Machine Type | Episodes | Setpoint? | Effort? |
|------|-----|-----|------|------|------|
| CWRU | 2000 | Bearings | 480 | ✗ | ✗ |
| PHM 2010 | 2010 | CNC | 315 | Partial | ✓ |
| AURSAD | 2021 | UR3e robot | 2,045 | ✓ | ✓ |
| voraus-AD | 2023 | Collaborative robot | 2,122 | ✓ | ✓ |
| **FactoryNet** | 2026 | Multi-machine | **23,000** | ✓ Required | ✓ Required |

FactoryNet is 10× larger than the previous largest (voraus-AD) and is the only one to mandate both setpoint and effort.

### Cross-Entity Transfer (24 schema-aligned signals)

| Source → Target | Bias-aware accuracy | High-dim baseline (all channels)|
|----------|------------------|---------------|
| UR3e → CNC | 0.84 | 0.71 (Poor) |
| UR3e → Collaborative | 0.81 | 0.74 |
| CNC → UR3e | 0.79 | 0.65 |
| Real → Sim (forward model) | 0.92 | – |

The 24 schema-aligned signals perform significantly better across entities than 130+ raw channels (high-dim baseline), proving the value of S-E-F-C abstraction.

### Anomaly Detection (Parameter Efficient)

| Model | Parameters | F1 (27 Anomaly Classes)|
|------|------|----------|
| Anomaly-Transformer (high-dim) | 7M | 0.71 |
| TimesFM Pre-trained + Fine-tuned | 200M | 0.74 |
| Chronos | 60M | 0.73 |
| **FactoryNet pretrained + 24 signals** | **2M** | **0.76** |

A 2M parameter model outperforms a 200M general time-series foundation model—schema alignment results in far fewer but more precise effective signals.

### Key Findings
- **S-E-F-C schema improves transfer**: Transferring with a 24-channel schema is 10+ points higher than with 130+ raw channels across entities.
- **Small model + good schema > Large model + raw**: 2M parameters outperform 200M general time-series models.
- **Real + Sim pairing**: The sim2real gap can be interpreted as forward-model error, allowing for quantitative diagnosis.
- **Wide coverage of 27 anomaly classes**: Sufficient for training a general anomaly detector rather than just a single-fault classifier.

## Highlights & Insights
- **First industrial time-series dataset with a true "control-loop structure"**: Previous datasets lost the command-measurement decomposition; this paper reorganizes from control theory first principles—a paradigm-shifting contribution.
- **S-E-F-C as a unified representation like RGB for images**: Makes an "Industrial ImageNet" possible; different machines and tasks can be mapped into the same framework to learn cross-task representations.
- **Instrumentalization of sim2real through real + sim pairing**: Sim2real is typically a mystery; the schema-aligned pairing in this paper allows the gap to be directly measured and analyzed.
- **2M parameters beating 200M**: Proves that "correct inductive bias + clean schema" is more effective than "brute force scaling"—significant for industrial deployment on resource-constrained edge devices.

## Limitations & Future Work
- 6 entities are still relatively few; the schema for industrial foundation models with multi-modality (vision + time-series fusion) remains unexplored.
- 27 anomaly classes are manually annotated; future scaling requires automated anomaly generation.
- S-E-F-C assumes all signals can be cleanly categorized, but real signals sometimes overlap (e.g., hybrid effort-feedback).
- Simulation uses Isaac Sim; simulation fidelity limits the ceiling for sim2real.
- Lack of long-term operation (months) data, limiting support for predictive maintenance (gradual degradation).

## Related Work & Insights
- **vs CWRU / PHM 2010 / AURSAD / voraus-AD**: These are single-machine / single-fault; FactoryNet is multi-entity + multi-anomaly + mandatory control-loop structure.
- **vs Open X-Embodiment / DROID**: Those focus on cross-entity policies for robots; FactoryNet focuses on cross-entity anomaly detection for industrial actuated systems.
- **vs Chronos / TimesFM / Moirai**: Those are general time-series foundation models; FactoryNet aims for an equivalent status in the industrial vertical.
- **Insights**: The design of S-E-F-C style schema-aligned datasets can be applied to any "actuated system + heterogeneous machine" domain (automotive, aerospace, chemical, energy); this approach is also applicable to robotics policy learning.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First multi-entity industrial time-series dataset with mandatory control-loop structure; S-E-F-C schema is a genuine contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Includes cross-entity transfer + 27 anomaly detection + model scale comparison; lacks experiments on long-term degradation.
- Writing Quality: ⭐⭐⭐⭐ Cybernetic framing is clear; the Table 1 dataset comparison is intuitive.
- Value: ⭐⭐⭐⭐⭐ High impact in a sector accounting for 15% of GDP; the first "Industrial ImageNet" paves the way for future foundation models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Omni-iEEG: A Large-Scale, Comprehensive iEEG Dataset and Benchmark for Epilepsy Research](../../ICLR2026/time_series/omni-ieeg_a_large-scale_comprehensive_ieeg_dataset_and_benchmark_for_epilepsy_re.md)
- [\[ICLR 2026\] FeDaL: Federated Dataset Learning for General Time Series Foundation Models](../../ICLR2026/time_series/fedal_federated_dataset_learning_for_general_time_series_foundation_models.md)
- [\[ICML 2026\] OLIVIA: Harmonizing Time Series Foundation Models with Power Spectral Density](olivia_harmonizing_time_series_foundation_models_with_power_spectral_density.md)
- [\[NeurIPS 2025\] Multi-Scale Finetuning for Encoder-based Time Series Foundation Models](../../NeurIPS2025/time_series/multi-scale_finetuning_for_encoder-based_time_series_foundation_models.md)
- [\[ICML 2026\] Generalizing Multi-scale Time-Series Modeling with a Single Operator](generalizing_multi-scale_time-series_modeling_with_a_single_operator.md)

</div>

<!-- RELATED:END -->
