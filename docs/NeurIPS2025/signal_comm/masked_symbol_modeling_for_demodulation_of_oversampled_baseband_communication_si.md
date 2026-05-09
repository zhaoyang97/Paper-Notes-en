---
title: >-
  [Paper Note] Masked Symbol Modeling for Demodulation of Oversampled Baseband Communication Signals
description: >-
  [NeurIPS 2025][Signal & Communication][Masked Symbol Modeling] This paper proposes Masked Symbol Modeling (MSM), transplanting BERT's masked prediction paradigm to the communication physical layer. It reframes inter-symbol contributions from pulse shaping as "contextual information," training a Transformer on clean oversampled baseband signals to learn waveform structure, and leveraging the learned context at inference time to recover symbols corrupted by impulsive noise.
tags:
  - NeurIPS 2025
  - "Signal & Communication"
  - Masked Symbol Modeling
  - Communication Physical Layer
  - Transformer
  - Pulse Shaping
  - Impulsive Noise
date: 2026-05-08
content_hash: a58c4df3b27c94c7
---

# Masked Symbol Modeling for Demodulation of Oversampled Baseband Communication Signals

**Conference**: NeurIPS 2025
**arXiv**: [2512.01428](https://arxiv.org/abs/2512.01428)
**Code**: [https://github.com/OguzBedir/Masked_Symbol_Modeling](https://github.com/OguzBedir/Masked_Symbol_Modeling)
**Area**: Signal & Communications
**Keywords**: Masked Symbol Modeling, Communication Physical Layer, Transformer, Pulse Shaping, Impulsive Noise

## TL;DR

This paper proposes Masked Symbol Modeling (MSM), transplanting BERT's masked prediction paradigm to the communication physical layer. It reframes inter-symbol contributions from pulse shaping as "contextual information," training a Transformer on clean oversampled baseband signals to learn waveform structure, and leveraging the learned context at inference time to recover symbols corrupted by impulsive noise.

## Background & Motivation

**State of the Field**: Transformer architectures are increasingly applied to communication systems, primarily for channel estimation, equalization, and other classical signal processing tasks. However, most existing work treats Transformers as black-box tools without deeply exploring what "context" means within physical waveforms.

**Limitations of Prior Work**: In pulse-shaped oversampled systems, pulses from adjacent symbols overlap in the time domain, producing inter-symbol contributions. Conventional methods treat this overlap as interference to be eliminated (e.g., via equalizers) rather than as an exploitable information source. Meanwhile, impulsive noise (e.g., Middleton Class-A noise) poses severe challenges to traditional detectors due to its bursty, high-amplitude nature—detectors designed under the Gaussian noise assumption degrade drastically in impulsive noise environments.

**Root Cause**: The inter-symbol overlap induced by pulse shaping actually encodes rich deterministic structural information—each sample contains contributions from multiple neighboring symbols. Nevertheless, existing methods do not systematically exploit this structure for more robust signal recovery.

**Paper Goals**: How can the "contextual understanding" capability from NLP be transferred to the communication physical layer? Specifically, can a model be trained to understand the "grammar" of a pulse-shaped waveform and use context to recover noise-corrupted symbols?

**Starting Point**: The authors draw an analogy between "inter-symbol contributions from pulse shaping" and "word context in natural language"—just as the meaning of a word can be inferred from its surrounding words, the identity of a masked symbol can be inferred from surrounding unmasked samples.

**Core Idea**: Transplant BERT's masked prediction paradigm to the communication physical layer, treating pulse-shaping inter-symbol overlap as contextual information rather than interference, and self-supervisedly learning the "latent grammar" of the waveform.

## Method

### Overall Architecture

The input is an oversampled complex baseband signal (8 samples per symbol), in which a random 15% of symbols are masked (the corresponding time-domain sample intervals are zeroed out). A Transformer model predicts the symbol identifier at each masked position from the surrounding unmasked samples. Training uses only clean (noise-free) signals; at inference, positions corrupted by impulsive noise are first identified, masked, and then fed to the model, which leverages learned context to recover the symbols.

### Key Designs

1. **Masked Symbol Modeling (MSM)**:

    - Function: Self-supervisedly learn structural representations of oversampled baseband waveforms.
    - Mechanism: A discrete vocabulary is defined by assigning a unique identifier to each constellation point across all considered modulation schemes (BPSK to QAM256), yielding 272 identifiers in total. During training, 15% of symbols are randomly masked (the corresponding sample intervals are zeroed), and the model predicts the identifier of each masked symbol. Loss is computed only at masked positions using cross-entropy with inverse-frequency weighting to handle class imbalance.
    - Design Motivation: The 15% masking ratio follows BERT; training exclusively on clean signals directs the model to learn waveform structure rather than noise patterns. The pulse-overlap inter-symbol contributions naturally supply the fill-in-the-blank context required by the task.

2. **Reformer Transformer Architecture**:

    - Function: Efficiently process long-sequence waveforms (1024 samples).
    - Mechanism: A 2-channel input (I/Q components) is projected to 512-dimensional embeddings via a 1D learnable linear projection, supplemented with sinusoidal positional encodings, and processed through 6 Reformer blocks. The Reformer employs locality-sensitive hashing (LSH) attention (bucket size 64, 4 hashes), shared weights, and reversible layers to reduce memory overhead. For each masked symbol, mean pooling is applied over the corresponding 8-sample span, followed by a linear classification head ($\mathbb{R}^{512} \to \mathbb{R}^{272}$).
    - Design Motivation: The Reformer is adopted over a standard Transformer for computational efficiency when handling sequences of length 1024. LSH attention reduces the attention complexity from $O(n^2)$ to $O(n\log n)$.

3. **Semi-Synthetic Impulsive Noise Inference Strategy**:

    - Function: Exploit learned context at inference time to recover symbols corrupted by impulsive noise.
    - Mechanism: Rather than feeding the entire noisy waveform directly to the model, the inference procedure first identifies symbol positions affected by impulsive noise, masks only those positions, and leaves the unaffected portions intact. The model then infers the symbol identifiers at masked positions from the surrounding clean samples. The impulsive index $A$ is calibrated according to a target symbol hit rate of 15%.
    - Design Motivation: This selective masking strategy converts the problem from "global noise robustness" to "local missing-data recovery," fully exploiting the contextual understanding the model has learned from clean signals.

### Loss & Training

Cross-entropy loss is used exclusively, computed only at masked symbol positions, with inverse-frequency weighting to handle class imbalance. Training is entirely self-supervised, relying on an online data-generating `IterableDataset` with no external dataset required. Optimizer: Adam ($lr=10^{-3}$); trained on a single A100 GPU for 24 hours (37,551 steps); batch size 64.

## Key Experimental Results

### Main Results

| Modulation | SER (no noise) | SER ($\Gamma=10^{-6}$, strong impulsive) | SER ($\Gamma=10^{-3}$, moderate impulsive, high SNR) |
|-----------|---------------|------------------------------------------|------------------------------------------------------|
| BPSK | ~0.001 | ~0.001 | ~0.001 |
| QPSK | ~0.02 | ~0.02 | ~0.02 |
| QAM16 | ~0.05 | ~0.05 | ~0.05 |
| QAM64 | ~0.15 | ~0.15 | ~0.15 |
| QAM256 | ~0.35 | ~0.35 | ~0.35 |

### Ablation Study

| Configuration | Observation | Notes |
|--------------|-------------|-------|
| $\Gamma=10^{-6}$ (negligible Gaussian component) | SER independent of SNR | Masking eliminates the impulsive component; residual Gaussian noise is negligible |
| $\Gamma=10^{-3}$ (non-negligible Gaussian component) | SER rises significantly at low SNR | Gaussian noise degrades context quality at unmasked positions |
| Simple vs. complex modulation | BPSK best, QAM256 worst | More constellation points make masked prediction harder |
| Varying filter span and roll-off factor | Stable performance | Model generalizes across pulse-shaping parameters |

### Key Findings

- Under strong impulsive noise ($\Gamma=10^{-6}$), the model's SER is nearly independent of SNR, since masking effectively eliminates the impulsive component and the Gaussian background noise is virtually zero. This validates the natural immunity of the "mask-and-context-recovery" strategy against impulsive noise.
- Under moderate impulsive noise ($\Gamma=10^{-3}$), performance degrades noticeably at low SNR, as Gaussian noise simultaneously affects the signal quality at both masked and unmasked positions.
- The model maintains stable performance across multiple modulation schemes (BPSK to QAM256) and pulse-shaping configurations (4 filter spans × 6 roll-off factors), demonstrating cross-configuration generalization.
- Statistical analysis of symbol hit rates shows that $A^\star = -\ln(0.85)/L$ precisely controls an average of 15% of symbols being affected by impulsive noise.

## Highlights & Insights

- **Transferring the concept of context from NLP to the communication physical layer** is the paper's most fundamental insight. Redefining pulse-shaping inter-symbol overlap—traditionally regarded as ISI to be eliminated—as exploitable contextual information represents a genuinely clever shift in perspective. It implies a new receiver design paradigm: not detecting signals but interpreting them.
- **Decoupled training and inference design**: Training on clean signals prevents the model from learning noise patterns, while selective masking at inference converts noise recovery into a cloze task. This design grants the model natural adaptability to noise type, provided the corrupted positions can be identified.
- The analogy from BERT to communication signals opens a new door: waveforms have "grammar," modulation alphabets define a "vocabulary," and masked prediction learns a "language model" of the waveform.

## Limitations & Future Work

- The current approach relies on prior identification of symbol positions affected by impulsive noise (the "semi-synthetic" setting); in real systems, noise localization is itself a non-trivial problem. Future work should enable the model to directly process noisy waveforms without explicit masking.
- The input representation is relatively simple (raw I/Q channels + linear projection); the paper itself suggests replacing this with quantization followed by embedding, more faithfully mirroring BERT's tokenization design.
- Comparisons with traditional communication methods (e.g., optimal nonlinear demodulators, deep learning baselines) are absent.
- Validation is limited to a simplified single-antenna, single-user, multipath-free scenario; applicability to real communication environments remains unknown.
- No systematic ablation study analyzes the effect of architectural choices (depth, number of heads, embedding dimension) on performance.

## Related Work & Insights

- **vs. Traditional equalizers (e.g., MMSE, ZF)**: Classical methods treat inter-symbol contributions as ISI and attempt to eliminate them. This paper takes the opposite approach, treating ISI as an information source. The two paradigms may be complementary—equalizers remove multipath ISI while MSM exploits pulse-shaping ISI.
- **vs. DL-based signal detection (DeepSIG, etc.)**: Most DL-based communication methods perform end-to-end supervised learning. This paper adopts a self-supervised pretraining approach that requires no noisy signal/label pairs, making training data generation considerably simpler.
- **vs. Original BERT**: MSM is a direct transplant of BERT to the continuous signal domain, but a key distinction is that "tokens" are not discrete symbols but continuous waveform segments, and "masking" is time-domain zeroing rather than special-token substitution.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The conceptual innovation of transplanting the NLP masked modeling paradigm to the communication physical layer is highly compelling.
- Experimental Thoroughness: ⭐⭐⭐ Demonstrates basic feasibility, but experiments are preliminary; comparisons with traditional methods and systematic ablations are lacking.
- Writing Quality: ⭐⭐⭐⭐ Analogies are clear and physical motivation is well articulated, though some descriptions could be more concise.
- Value: ⭐⭐⭐⭐ Opens a new direction for representation learning at the communication physical layer, though substantial work remains before practical deployment.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Multi-Modal Masked Autoencoders for Learning Image-Spectrum Associations for Galaxy Evolution and Cosmology](multi-modal_masked_autoencoders_for_learning_image-spectrum_associations_for_gal.md)
- [\[NeurIPS 2025\] The Last Vote: A Multi-Stakeholder Framework for Language Model Governance](the_last_vote_a_multi-stakeholder_framework_for_language_model_governance.md)
- [\[NeurIPS 2025\] Memory-Integrated Reconfigurable Adapters: A Unified Framework for Settings with Multiple Tasks](memory-integrated_reconfigurable_adapters_a_unified_framework_for_settings_with_.md)
- [\[NeurIPS 2025\] Perturbation Bounds for Low-Rank Inverse Approximations under Noise](perturbation_bounds_for_low-rank_inverse_approximations_under_noise.md)
- [\[NeurIPS 2025\] Bispectral OT: Dataset Comparison using Symmetry-Aware Optimal Transport](bispectral_ot_dataset_comparison_using_symmetry-aware_optimal_transport.md)

<!-- RELATED:END -->
