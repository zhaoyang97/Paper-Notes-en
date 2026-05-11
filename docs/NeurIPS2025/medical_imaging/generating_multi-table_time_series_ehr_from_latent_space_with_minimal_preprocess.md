---
title: >-
  [Paper Note] Generating Multi-Table Time Series EHR from Latent Space with Minimal Preprocessing
description: >-
  [NeurIPS 2025][Medical Imaging][EHR synthesis] This paper proposes RawMed — the first framework to synthesize multi-table time series EHR data from raw records with minimal lossy preprocessing: events are textualized → c…
tags:
  - "NeurIPS 2025"
  - "Medical Imaging"
  - "EHR synthesis"
  - "time series generation"
  - "Residual Quantization"
  - "privacy preservation"
  - "multi-table relational database"
  - "autoregressive Transformer"
date: 2026-05-08
content_hash: 4e971093ef0b15cb
---

# Generating Multi-Table Time Series EHR from Latent Space with Minimal Preprocessing

**Conference**: NeurIPS 2025
**arXiv**: [2507.06996](https://arxiv.org/abs/2507.06996)
**Code**: [eunbyeol-cho/RawMed](https://github.com/eunbyeol-cho/RawMed)
**Authors**: Eunbyeol Cho, Jiyoun Kim, Minjae Lee, Sungjin Park, Edward Choi (KAIST, FuriosaAI)
**Area**: Medical Informatics
**Keywords**: EHR synthesis, time series generation, Residual Quantization, privacy preservation, multi-table relational database, autoregressive Transformer

---

## TL;DR

This paper proposes RawMed — the first framework to synthesize multi-table time series EHR data from raw records with minimal lossy preprocessing: events are textualized → compressed into a discrete latent space via Residual Quantization → temporal dynamics are modeled with an autoregressive Transformer. RawMed comprehensively outperforms existing baselines in fidelity, clinical utility, and privacy protection.

---

## Background & Motivation

### Problem Background

Electronic Health Records (EHRs) are inherently **multi-table relational time series databases**, capturing sequences of clinical events (laboratory tests, prescriptions, input events, etc.) after patient admission, and containing categorical, numerical, and textual data. Due to sensitive privacy concerns, EHR data are strictly regulated by frameworks such as HIPAA, making inter-institutional sharing difficult. Synthetic EHR generation has thus become an essential need.

### Two Major Bottlenecks of Existing Methods

**Heavy reliance on feature selection**: Methods such as EMR-M-GAN, EHR-Safe, FLEXGEN-EHR, and TIMEDIFF all require domain experts to pre-select a small subset of tables and columns (at most ~5,373 features). If downstream research requires excluded features, the synthetic data cannot be used. This means the "usable scope" of synthetic data is locked in before generation begins.

**Complex lossy preprocessing**: Operations such as numerical binning, terminology normalization, and temporal aggregation inevitably distort original distributions. For example, temporal aggregation of laboratory values may mask sudden anomalies, while binning oversimplifies subtle trends and reduces the reliability of downstream predictive models.

### Scale Gap

| Method | Max Features | Max Time Steps | Uses All Columns | Preserves Raw Values |
|--------|-------------|---------------|-----------------|---------------------|
| EMR-M-GAN | 98 | 24 | ✗ | ✗ |
| EHR-Safe | 90 | 50 | ✗ | ✗ |
| FLEXGEN-EHR | 5,373 | 48 | ✗ | ✗ |
| TIMEDIFF | 15 | 276 | ✗ | ✗ |
| **RawMed** | **333,524** | **243** | **✓** | **✓** |

RawMed handles approximately **62×** more features than the closest competitor FLEXGEN-EHR, and is the first to simultaneously retain all columns and raw numerical values.

---

## Method

RawMed's core pipeline consists of three stages: **textualization** → **event-level compression** → **temporal modeling and sampling**.

### 1. Data Representation: Textualized Sequences

Each row event in the EHR is serialized into a plain-text string. For example, a laboratory event is represented as:

> `"lab item Glucose value 95 uom mg/dL"`

This representation naturally supports heterogeneous data types without requiring binning or normalization. Each patient $p$'s clinical trajectory is represented as an ordered event sequence $S^p = [e_1^p, e_2^p, \ldots, e_{n^p}^p]$, where $e_i^p = (t_i^p, x_i^p)$, $t_i^p$ is the timestamp after admission, and $x_i^p$ is the serialized text. Text is tokenized and padded/truncated to fixed length $L=128$, then embedded as $\mathbf{x}_i^p \in \mathbb{R}^{L \times F}$.

### 2. Event-Level Compression: RQ-VAE

Textualization causes the sequence length to expand dramatically, making direct modeling computationally infeasible. RawMed addresses this bottleneck with **Residual Quantization (RQ)**:

- An **encoder** (1D CNN) compresses $\mathbf{x}_i^p \in \mathbb{R}^{L \times F}$ into $\hat{\mathbf{z}}_i^p \in \mathbb{R}^{L_z \times F_z}$
- **RQ quantization** decomposes each latent vector into $D$ layers of residual quantization codes: $\text{RQ}(\hat{\mathbf{z}}; C, D) = (k_1, \ldots, k_D) \in [K]^D$, where each layer performs nearest-neighbor codebook lookup on the residual from the previous layer
- A **decoder** reconstructs $\hat{\mathbf{x}}_i^p$ from $\mathbf{z}_i^p = \sum_{d=1}^D \text{lut}(k_d)$

Compared to standard VQ-VAE, RQ progressively approximates the target through multi-layer residuals, yielding significantly higher fidelity for independently distributed columns (e.g., patientweight) — the KS statistic drops from 0.28 (VQ) to 0.09 (RQ).

**Compression results**: On MIMIC-IV, sequence length is compressed from 11.2k to 1.8k (**84% reduction**); on eICU, from 3.1k to 0.8k (**74% reduction**).

### 3. Temporal Modeling: TempoTransformer

The compressed patient trajectories are arranged as interleaved sequences:

$$S_{\text{quantized}}^p = [\tau_1^p, k_1^p, \tau_2^p, k_2^p, \ldots, \tau_{n^p}^p, k_{n^p}^p]$$

Two key design choices:

- **Temporal tokenization**: Timestamps are decomposed into decimal digit sequences at 10-minute resolution (e.g., 720 minutes → $[7, 2]$), using a compact vocabulary of digits 0–9
- **Time Separation**: Temporal tokens and event content tokens are explicitly interleaved with separate vocabularies, ensuring the model does not conflate the two types of information

The autoregressive Transformer is trained with standard NLL loss:

$$\mathcal{L}_{\text{AR}} = -\sum_{p \in \mathcal{P}} \sum_{i=1}^{|S_{\text{quantized}}^p|} \log P(s_i^p \mid s_1^p, \ldots, s_{i-1}^p)$$

### 4. Sampling and Post-Processing

- Top-$k$ sampling is used to generate new patient sequences; vocabulary masking enforces structural integrity
- **Event-level validation**: Levenshtein distance is used to correct misspelled column names; extraneous characters in numerical fields are removed
- **Patient-level validation**: Sequences containing invalid events are discarded, timestamp monotonicity is verified, and results are converted into relational tables

### 5. Evaluation Framework (Newly Proposed)

RawMed also introduces the first comprehensive evaluation framework for multi-table time series synthetic EHR, covering:

- **Single-table fidelity**: CDE (column distribution), I-CDE (item-level granular distribution), PCC/I-PCC (inter-column correlations), Predictive Similarity (higher-order dependencies)
- **Multi-table temporal fidelity**: Time Gap (KS distance of inter-event interval distributions), Event Count (per-patient event count distribution), Next Event Prediction (LSTM F1 for next-event prediction)
- **Clinical utility**: AUROC across 11 clinical prediction tasks
- **Privacy protection**: Membership Inference Attack accuracy

---

## Key Experimental Results

### Table 1: Single-Table Evaluation (MIMIC-IV & eICU, lower is better)

| Model | CDE (MIMIC) | I-CDE (MIMIC) | PCC (MIMIC) | I-PCC (MIMIC) | ER (MIMIC) | SMAPE (MIMIC) |
|-------|------------|--------------|------------|--------------|-----------|--------------|
| Real | - | - | - | - | 15.35 | 60.85 |
| SDV | 0.11 | 0.54 | 0.26 | 0.26 | 49.32 | 103.85 |
| RC-TGAN | 0.26 | 0.54 | 0.18 | 0.28 | 38.21 | 97.26 |
| ClavaDDPM | 0.06 | 0.22 | 0.08 | 0.27 | 27.91 | 80.02 |
| **RawMed** | **0.04** | **0.05** | **0.04** | **0.10** | **19.69** | **57.31** |

RawMed's I-CDE is approximately 1/4 that of ClavaDDPM, demonstrating substantially superior item-level fidelity. Its SMAPE closely approaches real data (57.31 vs. 60.85), indicating strong higher-order semantic consistency.

### Table 2: Clinical Utility and Temporal Fidelity (closer to Real / lower is better)

| Model | AUROC MEDS-TAB (MIMIC) | AUROC GenHPF (MIMIC) | Time Gap (MIMIC) | Event Count (MIMIC) | MIA (MIMIC) |
|-------|----------------------|---------------------|-----------------|-------------------|------------|
| Real | 0.90±0.06 | 0.82±0.09 | - | - | - |
| SDV | 0.46±0.13 | 0.47±0.13 | 0.76 | 0.46 | 0.499 |
| ClavaDDPM | 0.68±0.19 | 0.64±0.17 | 0.48 | 0.11 | 0.500 |
| **RawMed** | **0.87±0.08** | **0.80±0.09** | **0.01** | **0.02** | **0.498** |

RawMed's AUROC across 11 clinical prediction tasks falls only 0.02–0.03 below real data, far exceeding the second-best ClavaDDPM (gap of 0.16–0.19). The Time Gap metric of 0.01 is **48× lower** than ClavaDDPM's 0.48, demonstrating a decisive advantage in temporal fidelity. MIA accuracy near 0.5 (random chance) confirms adequate privacy protection.

---

## Highlights & Insights

1. **"Full-retention" paradigm**: RawMed is the first to demonstrate high-quality synthesis while preserving all columns and raw values of an EHR database, breaking the prevailing assumption that feature selection is mandatory. Its 333k-feature scale exceeds state-of-the-art methods by two orders of magnitude.

2. **Key insight on RQ vs. VQ**: For numerical columns weakly correlated with others, such as patientweight, VQ's single-layer codebook fails to sufficiently encode the distribution, leading to severe reconstruction distortion. RQ's multi-layer residual mechanism precisely compensates for this deficiency — a finding relevant to all medical data generation tasks employing VQ.

3. **Time Separation is the most critical component**: Ablation studies show that removing Time Separation causes Time Gap to surge from 0.01 to 0.51 and Event Count to rise from 0.02 to 0.40 — the most severe performance degradation observed. This demonstrates that explicitly separating temporal and content information is central to successful temporal modeling.

4. **Value of the evaluation framework**: The proposed I-CDE/I-PCC metrics elegantly address the challenge of evaluating EHR data where a single column stores values from different clinical items, filling a gap in synthetic EHR assessment.

5. **Compression and quality are compatible**: The 84% sequence length compression not only preserves quality but improves generation fidelity by reducing the difficulty of autoregressive modeling — RawMed uniformly outperforms the uncompressed baseline REaLTabFormer.

---

## Limitations & Future Work

1. **Limited number of tables**: Validation is currently restricted to 3 primary time series tables (laboratory, prescription, input events); scalability to dozens of tables remains unverified, and the complexity of inter-table dependency modeling may increase substantially.

2. **Unconditional generation only**: The framework supports only unconditional synthesis and cannot generate patient data conditioned on specified attributes (e.g., sex, age group, disease type), limiting applicability to scenarios such as clinical trial simulation.

3. **Static attributes not integrated**: Static features such as sex and birth year are excluded from the generation pipeline, leaving synthesized data without demographic consistency.

4. **Reliance on post-processing**: Generated data still exhibits issues such as misspelled column names and anomalous characters in numerical fields, requiring rule-based post-processing corrections — indicating room for improvement in structural fidelity within the latent space.

5. **Scalability to longer time windows**: Although experiments with a 24-hour window show that most metrics remain stable, the Event Count metric degrades, suggesting that longer hospitalization records may require more advanced compression or hierarchical sampling strategies.

---

## Related Work & Insights

- **Evolution of EHR synthesis**: From single-type GAN/VAE (generating diagnosis codes) → mixed-type time series (EVA, HALO) → recent works jointly modeling time series and heterogeneous data (EMR-M-GAN, EHR-Safe, FLEXGEN-EHR, TIMEDIFF), all of which rely on feature selection and heavy preprocessing
- **Textualized tabular generation**: GReaT, REaLTabFormer, and related methods convert table rows to text for language model-based generation, but are limited to single-table settings and do not address the temporal dimension
- **Vector quantization**: VQ-VAE and Residual Quantization have been widely applied in image and speech domains; RawMed is the first to introduce RQ for EHR event compression
- **Synthetic data evaluation**: SDMetrics and Synthcity focus on single-table settings and provide insufficient coverage for multi-table time series scenarios

---

## Rating

| Dimension | Score (1–10) | Notes |
|-----------|-------------|-------|
| Novelty | 8 | First multi-table time series raw EHR synthesis framework; scale improved by two orders of magnitude |
| Technical Depth | 7 | RQ-VAE + autoregressive Transformer combination is solid but not fundamentally disruptive |
| Experimental Thoroughness | 9 | Two public datasets, 11 downstream tasks, comprehensive ablations, multi-dimensional evaluation |
| Practical Value | 8 | Directly addresses data scarcity in medical AI; code is open-sourced |
| Writing Quality | 8 | Clear structure, precise problem formulation; evaluation framework contribution stands independently |
| **Overall** | **8.0** | Strong work with accurate problem framing, complete methodology, and convincing experiments |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Towards Unified and Lossless Latent Space for 3D Molecular Latent Diffusion Modeling](towards_unified_and_lossless_latent_space_for_3d_molecular_latent_diffusion_mode.md)
- [\[NeurIPS 2025\] Towards Self-Supervised Foundation Models for Critical Care Time Series](towards_self-supervised_foundation_models_for_critical_care_time_series.md)
- [\[NeurIPS 2025\] Manipulating 3D Molecules in a Fixed-Dimensional E(3)-Equivariant Latent Space](manipulating_3d_molecules_in_a_fixed-dimensional_e3-equivariant_latent_space.md)
- [\[NeurIPS 2025\] MIRA: Medical Time Series Foundation Model for Real-World Health Data](mira_medical_time_series_foundation_model_for_real-world_health_data.md)
- [\[NeurIPS 2025\] Self-Supervised Learning via Flow-Guided Neural Operator on Time-Series Data](self-supervised_learning_via_flow-guided_neural_operator_on_time-series_data.md)

</div>

<!-- RELATED:END -->
