---
title: >-
  [Paper Note] Controllable Sequence Editing for Biological and Clinical Trajectories
description: >-
  [ICLR 2026][Medical Imaging][Counterfactual generation] This paper proposes Clef, a controllable sequence editing model based on *temporal concepts* that performs immediate and delayed editing of biological/clinical multivariate trajectories under given conditions (e.g., drugs, surgery). On cell reprogramming and patient laboratory test data, Clef achieves 16.28% MAE improvement for immediate editing, 26.73% for delayed editing, and up to 62.84% improvement for zero-shot counterfactual generation.
tags:
  - ICLR 2026
  - Medical Imaging
  - Counterfactual generation
  - sequence editing
  - temporal concepts
  - patient trajectories
  - cell reprogramming
date: 2026-05-08
content_hash: 66e8ccfff72fa203
---

# Controllable Sequence Editing for Biological and Clinical Trajectories

**Conference**: ICLR 2026
**arXiv**: [2502.03569](https://arxiv.org/abs/2502.03569)
**Code**: [https://github.com/mims-harvard/CLEF](https://github.com/mims-harvard/CLEF)
**Area**: Medical Imaging / Bioinformatics
**Keywords**: Counterfactual generation, sequence editing, temporal concepts, patient trajectories, cell reprogramming

## TL;DR

This paper proposes Clef, a controllable sequence editing model based on *temporal concepts* that performs immediate and delayed editing of biological/clinical multivariate trajectories under given conditions (e.g., drugs, surgery). On cell reprogramming and patient laboratory test data, Clef achieves 16.28% MAE improvement for immediate editing, 26.73% for delayed editing, and up to 62.84% improvement for zero-shot counterfactual generation.

## Background & Motivation

Counterfactual reasoning ("What if the patient were switched to a different drug?" or "What if cell perturbation had been applied ten days earlier?") is a central problem in biology and medicine. Existing methods exhibit the following limitations:

**Controllable text generation (CTG) methods** support only *immediate editing* (predicting the next token) and cannot perform *delayed editing* (jumping to a future time step to predict counterfactual trajectories). CTG models must advance step by step to fill temporal gaps and cannot guarantee that the final output satisfies the target condition.

**Time-series diffusion models** support conditional generation but are restricted to univariate sequences and assume that the condition affects the entire sequence, precluding precise local edits.

In practice, an intervention (e.g., drug administration, surgery) should take effect only after a specific time point and should affect only a subset of variables (e.g., certain lab results), while the remaining variables and historical data must remain unchanged to preserve temporal causal consistency.

**Root Cause**: How can precise, condition-guided local edits be performed on multivariate sequences while maintaining global causal consistency?

**Starting Point**: Inspired by condition guidance in controllable text generation and spatial context in image inpainting, Clef introduces *temporal concepts*—vectors that encode the rate of change (trajectory) of a sequence—to capture how and when a condition affects the sequence, enabling precise, temporally localized controllable sequence editing.

## Method

### Overall Architecture

**Input**: Multivariate sequence $\mathbf{x}_{:,t_0:t_i}$ (with $V$ variables), condition $s$ (e.g., transcription factor or medical code), target prediction time $t_j > t_i$
**Output**: Counterfactual sequence $\hat{\mathbf{x}}_{:,t_j}^s$

Clef comprises four core components: a sequence encoder $F$, a condition adapter $H$, a concept encoder $E$, and a concept decoder $G$.

### Core Definitions

**Sequence Editing (Definition 3.1)** is divided into two types:
- **Immediate Editing**: Given $\mathbf{x}_{:,t_0:t_i}$ and condition $s$ occurring at $t_{i+1}$, predict $\hat{\mathbf{x}}_{:,t_{i+1}}$.
- **Delayed Editing**: Given $\mathbf{x}_{:,t_0:t_i}$ and condition $s$ occurring at $t_j \geq t_{i+1}$, directly predict $\hat{\mathbf{x}}_{:,t_j}$ in a single step.

**Temporal Concept (Definition 3.2)**: $\mathbf{c} = \mathbf{x}_{:,t_k} / \mathbf{x}_{:,t_j}$, i.e., the rate of change of the sequence between two time steps. This can be interpreted as the growth/decay factor for each variable between two time points.

### Key Designs

1. **Sequence Encoder $F$**: Extracts temporal features $\mathbf{h}_x = F(\mathbf{x}_{:,t_0:t_i})$ from the historical sequence. Any encoder (Transformer, xLSTM, MOMENT, etc.) can be used. A temporal encoder generates positional encodings $\mathbf{h}_t$ via sinusoidal embeddings of year/month/day/hour, and computes a time-gap embedding $\Delta_{t_i,t_j} = \mathbf{h}_{t_j} - \mathbf{h}_{t_i}$.

2. **Condition Adapter $H$**: Obtains condition embeddings $\mathbf{z}_s$ from a frozen pretrained embedding model (ESM-2 protein language model for cell experiments; clinical knowledge graph embeddings for patient data), then projects them to a hidden representation $\mathbf{h}_s = H(\mathbf{z}_s)$ via a linear layer.

3. **Concept Encoder $E$**: This is the core innovation of Clef. The time-gap embedding and condition embedding are summed to form a joint embedding $\mathbf{h}_s^{t_j} = \Delta_{t_i,t_j} \oplus \mathbf{h}_s$, which then interacts with sequence features via element-wise multiplication. An optional FFN layer with GELU activation produces the temporal concept:
$$\mathbf{c} = \text{GELU}(\text{FFN}(\mathbf{h}_x \odot \mathbf{h}_s^{t_j}))$$
This design enables the temporal concept to jointly encode historical sequence information, condition information, and temporal span information.

4. **Concept Decoder $G$**: Applies the learned temporal concept to the last time step of the input sequence via element-wise multiplication to generate predictions:
$$\hat{\mathbf{x}}_{:,t_j}^s = \mathbf{c} \odot \mathbf{x}_{:,t_i}$$
This design is remarkably concise—the temporal concept is essentially a "change-rate" vector that, when multiplied directly by the current state, yields the future state.

### Loss & Training

Huber Loss is adopted as the objective, balancing the sensitivity of MSE to outliers with the robustness of MAE:

$$\mathcal{L}(\mathbf{x}, \hat{\mathbf{x}}) = \begin{cases} 0.5\mathbf{a}^2, & \text{if } |\mathbf{a}| \leq \delta \\ \delta(|\mathbf{a}| - 0.5\delta), & \text{otherwise} \end{cases}$$

where $\mathbf{a} = \mathbf{x}_{:,t_j}^s - \hat{\mathbf{x}}_{:,t_j}^s$.

Training is conducted on a single NVIDIA A100 or H100 GPU. Hyperparameter search covers dropout rate $\in [0.3, 0.6]$, learning rate $\in [10^{-3}, 10^{-5}]$, and number of layers $\in [4, 8]$.

## Key Experimental Results

### Datasets

Four core datasets are constructed (extended to eight in later versions):
- **WOT**: Single-cell transcriptomic developmental trajectories simulated using the Waddington-OT model, with 1,479 highly variable genes.
- **WOT-CF**: Paired counterfactual cell trajectories for zero-shot evaluation.
- **eICU**: Patient routine laboratory test trajectories from the eICU database, covering 18 lab tests.
- **MIMIC-IV**: Patient test trajectories from MIMIC-IV.

Condition embeddings are sourced from ESM-2 (5,120-dimensional) for cell data and clinical knowledge graph embeddings (128-dimensional) for patient data.

### Main Results

Baselines include VAR (classical time series), Transformer, xLSTM, MOMENT (time-series foundation model), and their respective +Clef variants.

**Immediate Editing**:
- Clef consistently outperforms all baselines across datasets, with an average MAE improvement of 16.28%.
- Even the SimpleLinear ablation (concepts fixed to all ones, no learning) is competitive in some settings, but Clef is superior on datasets with complex short-term dynamics.

**Delayed Editing**:
- Clef outperforms or matches SimpleLinear and VAR on eICU and MIMIC-IV.
- Clef-Transformer and Clef-xLSTM achieve the lowest MAE.
- Average MAE improvement: 26.73%.
- On WOT, linear models (SimpleLinear, VAR) perform best, as cell developmental trajectories exhibit small per-step changes and may be noisy.

### Ablation Study

| Configuration | Key Metric | Notes |
|---|---|---|
| SimpleLinear (concepts all ones) | Competitive in some settings | Linear approximation is effective when $x_{t_j} \approx x_{t_i}$ |
| Clef-FFN=0 (no FFN) | Best on WOT | Cell data is relatively simple; additional nonlinearity is unnecessary |
| Clef-FFN=1 (with FFN) | Best on eICU/MIMIC | Patient data requires greater expressive capacity |
| Different sequence encoders | MOMENT performs worst | 1,024-dim embeddings from foundation models are less effective than training from scratch |

### Generalization Experiments

Using the SPECTRA method to create train/test splits with varying distributional similarity:
- Clef models remain more stable as the train/test distribution gap increases.
- Non-Clef models degrade substantially in performance.
- Clef-xLSTM achieves similar performance to the xLSTM baseline on delayed editing but generalizes significantly better.

### Zero-Shot Counterfactual Generation

Evaluated on WOT-CF paired counterfactual trajectories:
- Models are trained on "original" trajectories and evaluated zero-shot on "counterfactual" trajectories.
- Clef models consistently outperform non-Clef models on both immediate and delayed editing.
- Immediate editing improves by up to 14.45% MAE; delayed editing by up to 63.19% MAE.
- After the divergence time point ($t=10$), Clef substantially outperforms baselines.

### Key Findings

- The introduction of temporal concepts allows direct concept-level interventions (e.g., halving the glucose-related concept value) without requiring condition tokens.
- In a T1D case study, reducing the glucose concept causes generated counterfactual trajectories to more closely resemble those of healthy individuals.
- Intervening on the glucose concept also indirectly reduces white blood cell counts, consistent with clinical knowledge of T1D as an autoimmune disease.
- Reverse experiment: intervening on the white blood cell concept also induces changes in glucose levels, validating intrinsic inter-variable associations.
- Simultaneously intervening on multiple concepts (glucose + white blood cells) produces cumulative effects, generating trajectories that more closely approximate healthy individuals.

## Highlights & Insights

- **Minimal yet effective design**: Temporal concepts are essentially "rate-of-change vectors," and the decoder reduces to simple element-wise multiplication. This simplicity yields strong interpretability—each concept dimension corresponds to the rate of change of one variable.
- **Single-step delayed prediction**: Unlike CTG methods that require step-by-step advancement, Clef directly predicts counterfactual sequences at arbitrary future time points in a single step.
- **Encoder-agnostic**: Clef can be combined with any sequence encoder (Transformer, xLSTM, MOMENT, etc.), functioning as a plug-in "controllable editing enhancement module."
- **Concept interventions**: Users can directly edit specific dimensions of the temporal concept to generate counterfactual sequences, providing a unique interactive capability.
- **Regularization effect**: Even in settings where linear models are superior (WOT), Clef significantly reduces the MAE of neural network models, acting as a regularizer.

## Limitations & Future Work

- Each element of the temporal concept corresponds to one variable, which does not capture higher-order inter-variable relationships. Hierarchical abstract concept learning could be explored.
- The model is entirely data-driven and does not incorporate prior knowledge from domain causal models. Future work could refine causal relationships through user intervention feedback.
- Condition embeddings depend on pretrained models (ESM-2, clinical knowledge graphs); embedding quality directly affects generation performance.
- Validation is currently limited to biological and medical domains; extension to other sequence editing scenarios (e.g., finance, climate) remains unexplored.
- MOMENT as a sequence encoder performs worst, suggesting that existing time-series foundation models may be ill-suited for fine-grained counterfactual generation tasks.

## Related Work & Insights

- **Controllable text generation**: CTG methods guide sequence generation via condition tokens but support only immediate editing. Clef extends this paradigm to the temporal domain, enabling delayed editing.
- **Concept Bottleneck Models**: Provide interpretability and interventionability through intermediate concept layers. Clef's temporal concepts are the first application of this idea to conditional generation.
- **Optimal Transport**: The WOT model uses OT to infer cell trajectories; Clef performs conditional editing on top of this framework.
- **Trajectories as inductive bias**: Interpreting temporal data as trajectory patterns is more natural than reasoning over individual values; Clef's temporal concepts embody this principle.

## Rating

- Novelty: ⭐⭐⭐⭐ — The definition of temporal concepts and the design of interventionability are novel, though the core operation (element-wise multiplication) is relatively simple.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 4+ datasets, 9 baselines, generalization tests, zero-shot experiments, and real-world case studies; highly comprehensive.
- Writing Quality: ⭐⭐⭐⭐⭐ — Problem formulation is clear, formalization is rigorous, and experimental design is well-motivated.
- Value: ⭐⭐⭐⭐ — Significant application value in computational biology and clinical decision support.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] CrossNovo: Bidirectional Representations Augmented Autoregressive Biological Sequence Generation](../../NeurIPS2025/medical_imaging/bidirectional_representations_augmented_autoregressive_biological_sequence_gener.md)
- [\[ICLR 2026\] Characterizing Human Semantic Navigation in Concept Production as Trajectories in Embedding Space](characterizing_human_semantic_navigation_in_concept_production_as_trajectories_i.md)
- [\[ICLR 2026\] Benchmarking ECG FMs: A Reality Check Across Clinical Tasks](benchmarking_ecg_fms_a_reality_check_across_clinical_tasks.md)
- [\[ICLR 2026\] CARE: Towards Clinical Accountability in Multi-Modal Medical Reasoning with an Evidence-Grounded Agentic Framework](care_towards_clinical_accountability_in_multi-modal_medical_reasoning_with_an_ev.md)
- [\[ICCV 2025\] Controllable Latent Space Augmentation for Digital Pathology](../../ICCV2025/medical_imaging/controllable_latent_space_augmentation_for_digital_pathology.md)

<!-- RELATED:END -->
