---
title: >-
  [Paper Note] Generalizable, Real-Time Neural Decoding with Hybrid State-Space Models
description: >-
  [NeurIPS 2025][Medical Imaging][State-Space Models] POSSM proposes a hybrid SSM-attention architecture that combines spike-level tokenization with a recurrent state-space model backbone…
tags:
  - "NeurIPS 2025"
  - "Medical Imaging"
  - "State-Space Models"
  - "Brain-Computer Interface"
  - "Neural Decoding"
  - "Cross-Species Transfer"
  - "Real-Time Inference"
date: 2026-05-08
content_hash: 74cbf68b1f157ab8
---

# Generalizable, Real-Time Neural Decoding with Hybrid State-Space Models

**Conference**: NeurIPS 2025
**arXiv**: [2506.05320](https://arxiv.org/abs/2506.05320)  
**Code**: Available  
**Area**: Medical Imaging / Brain-Computer Interface / Neural Decoding
**Keywords**: State-Space Models, Brain-Computer Interface, Neural Decoding, Cross-Species Transfer, Real-Time Inference

## TL;DR

POSSM proposes a hybrid SSM-attention architecture that combines spike-level tokenization with a recurrent state-space model backbone, achieving generalizable real-time neural decoding with inference speeds up to 9× faster than Transformers while maintaining comparable accuracy.

## Background & Motivation

Neural decoding—mapping neural activity to behavioral or cognitive variables—is central to modern neuroscience and brain-computer interfaces (BCIs). An ideal neural decoder must simultaneously satisfy three requirements:

**Accuracy**: Robust and precise predictions

**Real-Time Capability**: Causal, low-latency inference suitable for online scenarios (≤10ms latency)

**Generalizability**: Flexible transfer to new subjects, tasks, and experimental settings

However, existing methods exhibit a fundamental tension among these three criteria:

| Method | Accuracy | Real-Time | Generalizability |
|--------|----------|-----------|-----------------|
| RNN (GRU) | Medium | ✓ High | ✗ Poor (fixed input format) |
| Transformer (POYO/NDT-2) | High | ✗ Poor ($O(n^2)$ complexity) | ✓ Good (flexible tokenization) |
| **POSSM** | **High** | **✓ High** | **✓ Good** |

RNNs are fast but struggle to generalize across sessions due to their dependence on fixed-size temporal binning; Transformers such as POYO achieve strong generalization through flexible spike tokenization but their quadratic complexity renders them infeasible for real-time use. POSSM resolves this tension through a hybrid architecture.

## Method

### Overall Architecture

The POSSM architecture, illustrated in Figure 2, consists of three modules:

1. **Input Cross-Attention**: Inherited from POYO, compresses variable-length spike token sequences into a fixed-size latent representation.
2. **Recurrent Backbone**: An SSM (or GRU/Mamba) updates hidden states across temporal chunks, preserving long-range context.
3. **Output Cross-Attention & Readout**: Queries the most recent $k=3$ hidden states to produce behavioral predictions.

**Key insight:** Input processing operates only on 50ms temporal chunks; the recurrent backbone updates hidden states in a streaming fashion without reprocessing historical data.

### Key Designs

**Spike Tokenization**:
Following POYO, each spike is represented by two pieces of information:
- Neural unit identity: a learnable unit embedding
- Spike timestamp: relative timing encoded via RoPE (Rotary Position Embedding)

$$\mathbf{x} = (\text{UnitEmb}(i), t_{\text{spike}})$$

This tokenization scheme supports variable numbers of neurons and different sampling rates.

**Input Cross-Attention**:
A PerceiverIO-style cross-attention mechanism compresses $N$ spike tokens using a learnable query $\mathbf{q} \in \mathbb{R}^{1 \times M}$:

$$\mathbf{z}^{(t)} = \text{softmax}\left(\frac{\mathbf{q}\mathbf{K}_t^\top}{\sqrt{d_k}}\right)\mathbf{V}_t$$

**Recurrent Backbone**:
The output $\mathbf{z}^{(t)}$ is passed to the SSM to update the hidden state:

$$\mathbf{h}^{(t)} = f_{\text{SSM}}(\mathbf{z}^{(t)}, \mathbf{h}^{(t-1)})$$

Cross-attention captures local (within-50ms) temporal structure, while the SSM integrates global context across chunks. Three backbone variants are evaluated: S4D, GRU, and Mamba.

**Generalization Strategies**: Two fine-tuning approaches are employed:
- **Unit Identification (UI)**: Freezes model weights and trains only new unit/session embeddings (updating <1% of parameters).
- **Full Finetuning (FT)**: Applies UI for several epochs, then unfreezes all parameters for end-to-end training.

### Loss & Training

- **NHP Motor Decoding**: Mean Squared Error (MSE) loss predicting 2D hand velocity time series.
- **Human Handwriting Decoding**: Classification loss predicting intended characters/strokes.
- **Human Speech Decoding**: CTC (Connectionist Temporal Classification) loss predicting phoneme sequences; a two-stage training procedure is used—first reconstructing spike counts, then training the CTC decoder.
- **Training Details**: Batch size 128–256, cosine learning rate scheduler, 500 epochs (NHP), LoRA-style data augmentation (unit dropout).

## Key Experimental Results

### Main Results

**NHP Motor Decoding** ($R^2$, higher is better):

| Method | C-CO 2016 (2) | C-CO 2010 (5) | T-CO (6) | T-RT (6) | H-CO (1) |
|--------|--------------|--------------|---------|---------|---------|
| MLP | 0.921 | 0.695 | 0.798 | 0.701 | 0.418 |
| GRU | 0.938 | 0.731 | 0.819 | 0.717 | 0.693 |
| POYO (SS) | 0.929 | 0.753 | 0.831 | 0.729 | — |
| POSSM-GRU (SS) | 0.944 | 0.738 | 0.836 | 0.738 | 0.690 |
| o-POSSM-S4D (FT) | **0.951** | **0.781** | **0.856** | **0.769** | **0.760** |

**Human Handwriting Decoding** (Accuracy %):

| Method | Accuracy |
|--------|----------|
| PCA-KNN (baseline) | 81.36 ± 7.53 |
| POYO | 94.86 ± 3.53 |
| POSSM-GRU (from scratch) | 95.82 ± 3.41 |
| o-POSSM-S4D (NHP pretrain + finetune) | **97.73 ± 2.56** |
| o-POSSM-GRU (NHP pretrain + finetune) | 97.25 ± 2.30 |

**Human Speech Decoding** (Phoneme Error Rate PER %, lower is better):

| Method | PER (%) |
|--------|---------|
| GRU (no noise) | 39.16 |
| GRU (standard) | 30.06 |
| S4D | 35.99 |
| Mamba | 32.19 |
| POSSM-GRU (no noise) | 29.70 |
| POSSM-GRU (standard) | **27.32** |

### Ablation Study

**Inference Efficiency Comparison** (Figure 4c):

| Model | Parameters | GPU Inference Time/Chunk |
|-------|-----------|--------------------------|
| MLP | Smallest | ~0.3ms |
| GRU | Small | ~0.3ms |
| POSSM-SS | Smallest (0.41–0.68M) | ~0.5ms |
| o-POSSM | ~8M | ~1.5ms |
| POYO | Medium | ~8ms |
| NDT-2 | Larger | ~12ms |

POSSM also achieves ~2.44ms/chunk (single-session) and ~5.65ms/chunk (pretrained) on CPU, both within the real-time BCI requirement (≤10ms).

**Cross-Species Transfer Experiments**:
- o-POSSM pretrained on macaque motor cortex data achieves 2–5% accuracy gains on the human handwriting task after fine-tuning.
- This represents the **first successful cross-species neural decoding transfer using deep learning**.

### Key Findings

1. **Hybrid architecture is optimal**: POSSM matches or surpasses pure Transformers (POYO) and pure RNNs across all tasks while achieving up to 9× faster inference.
2. **Pretraining is highly effective**: o-POSSM substantially outperforms models trained from scratch in low-data regimes, supporting efficient cross-session and cross-subject transfer.
3. **Cross-species transfer succeeds**: Monkey-to-human transfer learning yields ~16% accuracy improvement on the handwriting task compared to PCA-KNN.
4. **Long-sequence advantage**: In long-context tasks such as human speech decoding, Transformers are limited by quadratic complexity, whereas POSSM efficiently handles variable-length sequences of 2–18 seconds.
5. **Noise robustness**: POSSM maintains near-comparable performance without noise augmentation (PER 29.7 vs. 27.3), while the baseline GRU degrades substantially (39.2 vs. 30.1).

## Highlights & Insights

1. **Elegant architectural design**: Organically combines Perceiver's flexible input processing with SSM's efficient sequence modeling, resolving the core tension in neural decoding.
2. **Pioneering significance of cross-species transfer**: Demonstrates that motor cortical neural dynamics in monkeys and humans share transferable common features, providing a solution for data-scarce human BCI scenarios.
3. **Clinical practicality**: Satisfies real-time BCI latency requirements (<10ms) on both GPU and CPU, enabling deployment.
4. **Modular design**: The backbone is interchangeable (S4D/GRU/Mamba), accommodating diverse application requirements.
5. **Comprehensive experimental coverage**: Spans monkey motor decoding to human handwriting and speech, covering the major application domains of neural decoding.

## Limitations & Future Work

1. **Restricted to invasive recordings**: Validation is limited to intracortical electrode array data; non-invasive modalities such as EEG have not been explored.
2. **Offline evaluation**: Despite being designed for real-time scenarios, all experiments are conducted offline.
3. **No language model for speech decoding**: The absence of a language model for post-processing constrains further reduction of PER.
4. **Motor cortex only**: Decoding from other brain regions (e.g., visual cortex, hippocampus) has not been explored.
5. **Limited scope of cross-species transfer**: Only monkey-to-human transfer on the handwriting task has been validated.
6. **No self-supervised pretraining**: The current approach relies on supervised behavioral labels; self-supervised alternatives have not been explored.

## Related Work & Insights

- **POYO**: The direct foundation of POSSM, providing spike tokenization and PerceiverIO encoding.
- **NDT-2**: A Transformer decoder using spatiotemporal patch tokenization, but with high computational cost.
- **Mamba/S4**: Modern SSM architectures; POSSM demonstrates their potential as neural decoding backbones.
- **BRAND**: A real-time BCI deployment platform; POSSM's inference efficiency makes it well-suited for integration.
- **Insight**: Hybrid attention-recurrent architectures, beyond their success in NLP (e.g., Jamba), demonstrate unique advantages in neural signal processing—enabling flexible local encoding combined with efficient global state propagation.

## Rating

- **Novelty**: ★★★★★ — First hybrid SSM-attention architecture for neural decoding; cross-species transfer is pioneering.
- **Technical Depth**: ★★★★★ — Theoretically grounded architectural design with systematic comparison of multiple backbones.
- **Experimental Thoroughness**: ★★★★★ — Comprehensive evaluation across three task categories, multiple datasets, inference efficiency, sample efficiency, and cross-species transfer.
- **Practicality**: ★★★★★ — Meets real-time BCI latency requirements with direct clinical value.
- **Overall Recommendation**: ★★★★★

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] BarcodeMamba+: Advancing State-Space Models for Fungal Biodiversity Research](barcodemamba_advancing_state-space_models_for_fungal_biodiversity_research.md)
- [\[NeurIPS 2025\] GeoDynamics: A Geometric State-Space Neural Network for Understanding Brain Dynamics on Riemannian Manifolds](geodynamics_a_geometric_state-space_neural_network_for_understanding_brain_dynam.md)
- [\[NeurIPS 2025\] DyG-Mamba: Continuous State Space Modeling on Dynamic Graphs](dyg-mamba_continuous_state_space_modeling_on_dynamic_graphs.md)
- [\[NeurIPS 2025\] Bridging Graph and State-Space Modeling for Intensive Care Unit Length of Stay Prediction](bridging_graph_and_state-space_modeling_for_intensive_care_unit_length_of_stay_p.md)
- [\[NeurIPS 2025\] MIRA: Medical Time Series Foundation Model for Real-World Health Data](mira_medical_time_series_foundation_model_for_real-world_health_data.md)

</div>

<!-- RELATED:END -->
