---
title: >-
  [Paper Note] From Black Box to Biomarker: Sparse Autoencoders for Interpreting Speech Models of Parkinson's Disease
description: >-
  [NeurIPS 2025][Medical Imaging][Sparse Autoencoders] This work adapts sparse autoencoder (SAE) techniques from large language model interpretability research to speech-based Parkinson's disease (PD) detection…
tags:
  - "NeurIPS 2025"
  - "Medical Imaging"
  - "Sparse Autoencoders"
  - "Parkinson's Disease"
  - "Speech Detection"
  - "Mechanistic Interpretability"
  - "Whisper"
  - "Spectral Flux"
  - "Putamen"
date: 2026-05-08
content_hash: 0e2b89409d974a87
---

# From Black Box to Biomarker: Sparse Autoencoders for Interpreting Speech Models of Parkinson's Disease

**Conference**: NeurIPS 2025
**arXiv**: [2507.16836](https://arxiv.org/abs/2507.16836)  
**Code**: Not provided  
**Area**: Medical Imaging
**Keywords**: Sparse Autoencoders, Parkinson's Disease, Speech Detection, Mechanistic Interpretability, Whisper, Spectral Flux, Putamen

## TL;DR

This work adapts sparse autoencoder (SAE) techniques from large language model interpretability research to speech-based Parkinson's disease (PD) detection, proposes a Mask-based SAE to address small-dataset limitations, discovers that model predictions rely primarily on spectral flux and spectral flatness in low-energy regions, and further reveals that these features correlate significantly with MRI putamen volume—establishing a bridge from internal model representations to clinical biomarkers.

## Background & Motivation

**Speech as a biomarker**: Audio recordings are low-cost and non-invasive, and can reflect motor and cognitive function, making them particularly suitable for monitoring neurodegenerative diseases such as Parkinson's disease (PD) that affect articulation, prosody, and fluency.

**The black-box barrier in deep learning**: Deep learning systems operating on raw audio achieve strong performance in detecting disease-related changes (F1 > 80%), yet offer no explanation for their predictions, severely impeding clinical adoption.

**Limitations of existing interpretability methods**:
   - **Intrinsic interpretability** approaches (prototype networks, concept bottlenecks) require prior knowledge of relevant concepts and are unsuitable when such concepts are unknown.
   - **Post-hoc explanation** methods (attention localization) produce noisy, diffuse attributions in audio and perform poorly for global properties such as breathiness or weak voice.

**Success of SAEs in LLM research**: Sparse autoencoders have successfully decomposed interpretable dictionary entries in the mechanistic interpretability of LLMs, but direct application to speech faces two major challenges: biomedical datasets are small, and the continuous nature of audio makes entry interpretation considerably more complex than in text.

## Method

### Stage 1: PD Detection System

A frozen speech foundation model (Whisper Small) is used to extract sequential embeddings, followed by an attention pooling layer and a linear classifier trained with binary cross-entropy. Architecture: Linear → Attention Pooling → Linear → Output.

- Optimizer: Adam 8-bit, lr = 1e-4, linear warmup for 2 epochs + cosine annealing for 18 epochs
- 1,024 thirty-second segments sampled per epoch, batch size 32
- Data augmentation: random SNR (0–15 dB) background noise added to 90% of batches, along with 2–5 notch filters at random frequencies

### Stage 2: Mask-based SAE

After freezing the detection parameters, an SAE is inserted following the attention pooling layer. Let $x \in \mathbb{R}^N$ denote the pooling output. The SAE operates through a dual-encoder projection:

**Mask computation** (separating activation decision from value computation):

$$\text{mask}(x) = \sigma(\tau \cdot (W_m x + b_m))$$

where the temperature $\tau$ is annealed from 1.0 to 0.2 over 200 steps.

**Dictionary representation**:

$$f(x) = (W_e x + b_e) \odot \text{mask}(x) \in \mathbb{R}^K$$

**Reconstruction**: $\hat{x} = W_d f(x) + b_d$

### Loss & Training

**Fidelity loss**: $\mathcal{L}_{\text{fidelity}}(x, \hat{x}) = \frac{1}{N}\sum_i^N (\hat{x}_i - x_i)^2$

**Sparsity loss** (applied only to the mask, avoiding penalization of activation magnitude and supporting negative activations):

$$\mathcal{L}_{\text{sparsity}}(x) = \frac{1}{K}\sum_i^K \text{mask}(x)_i \|W_{d,i}\|_2$$

**Total loss**: $\mathcal{L}_{\text{SAE}} = \mathcal{L}_{\text{fidelity}} + \lambda \cdot \mathcal{L}_{\text{sparsity}}$, with $\lambda = 0.001$

### Key Designs

- **Dictionary size**: Only 64 entries (far fewer than the tens of thousands used in LLM SAEs), justified by the small dataset size and sample-level (rather than frame-level) application.
- **One dictionary vector per sample**: The SAE is applied after attention pooling rather than at each temporal frame.
- **Mask vs. ReLU**: The mask formulation supports negative activations and separates activation decision from scale computation, achieving a superior sparsity–fidelity trade-off across multiple values of $\lambda$.

## Key Experimental Results

### Dataset

Quebec Parkinson Network (QPN): 208 patients and 52 controls performing a Cookie Theft picture description task. The test set comprises 32 demographically balanced subjects (equal patients/controls, French/English, male/female), with a mean of 3.1 years post-diagnosis, UPDRS II ≤ 12, and UPDRS III ≤ 32 (early, mild symptoms).

### Association Between SAE Dictionary Entries and Interpretable Features

| Entry # | Activations | Corresponding Feature | Activation $\rho$ | Prediction $\rho$ |
|---------|------------|----------------------|-------------------|-------------------|
| #61 | 229 | Spectral flux (articulatory clarity) | 0.851 | **-0.943** |
| #26 | 229 | Spectral flatness (phonation noisiness) | 0.816 | **0.975** |
| #06 | 229 | Harmonics-to-noise ratio (voice quality) | -0.682 | 0.794 |
| #13 | 129 | Speaker language (English/French) | 0.738 | -0.239 |

> After Bonferroni correction (33 features × 64 entries = 2,112 tests), the correlation between spectral flux and entry #61 remains highly significant ($p < 10^{-20}$).

### Attention Pattern Findings

Model attention is **negatively correlated** with signal energy—the network tends to focus on low-energy regions (silence or weak/breathy speech segments). This is consistent with prior findings that respiratory pauses and breath sounds themselves carry diagnostic signals for PD.

### Brain Structure Correlations

Spectral flux and entry #61 activation strength correlate significantly with MRI putamen volume, with no significant associations observed for other basal ganglia subregions (caudate nucleus, globus pallidus, nucleus accumbens). The putamen is involved in speech motor control and undergoes atrophy in the early stages of PD.

### SAE Fidelity

- Test set fidelity score: 0.0068 (near-perfect reconstruction)
- Replacing the original pooling output with SAE reconstructions results in an F1 drop of less than 2%

## Highlights & Insights

1. ⭐⭐⭐⭐ **A complete chain from model to biomarker**: SAE dictionary entries → spectral flux → putamen volume, establishing a three-level correspondence from deep learning representations to acoustic features to brain structure, with genuine clinical translational value.
2. ⭐⭐⭐ **Mask-based SAE innovation**: A dual-encoder masking mechanism designed for small biomedical datasets, separating activation decision from value computation and offering greater flexibility than standard ReLU SAEs.
3. ⭐⭐⭐ **Insights from negative findings**: Spectral flux extracted directly from raw acoustic features does not constitute a strong predictive basis on its own (weak baseline), yet Whisper's attention mechanism enhances the discriminative power of these features through precise focus on low-energy regions.
4. ⭐⭐ **Rigorous statistical testing**: One sample per speaker is used to avoid inflated correlations, with Bonferroni correction applied over 2,112 tests.

## Limitations & Future Work

1. **Limited dataset**: Results are based on a single dataset (QPN) with 208 patients and 52 controls; generalization to other languages, cultures, or datasets remains unvalidated.
2. **Single task**: Only the Cookie Theft picture description task is evaluated; performance on other speech tasks (e.g., sustained vowels, read speech) is unknown.
3. **Parkinson's disease only**: The applicability of the SAE framework to other neurodegenerative diseases (Alzheimer's disease, ALS) is unclear.
4. **Indirect interpretation**: SAE interpretations are inherently indirect—correlations between dictionary entries and handcrafted features do not imply causality.
5. **Sample size for brain structure correlations**: The number of subjects with available MRI data used for the brain structure correlation analysis is not explicitly reported in the paper.

## Rating

| Dimension | Score |
|-----------|-------|
| Novelty | ⭐⭐⭐⭐ |
| Technical Depth | ⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| Overall Recommendation | ⭐⭐⭐⭐ |

## Related Work & Insights

## Rating

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] ProtSAE: Disentangling and Interpreting Protein Language Models via Semantically-Guided Sparse Autoencoders](../../AAAI2026/medical_imaging/protsae_disentangling_and_interpreting_protein_language_models_via_semantically-.md)
- [\[NeurIPS 2025\] DesignX: Human-Competitive Algorithm Designer for Black-Box Optimization](designx_human-competitive_algorithm_designer_for_black-box_optimization.md)
- [\[NeurIPS 2025\] Care-PD: A Multi-Site Anonymized Clinical Dataset for Parkinson's Disease Gait Assessment](care-pd_a_multi-site_anonymized_clinical_dataset_for_parkinsons_disease_gait_ass.md)
- [\[ICLR 2026\] Knowledgeable Language Models as Black-Box Optimizers for Personalized Medicine](../../ICLR2026/medical_imaging/knowledgeable_language_models_as_black-box_optimizers_for_personalized_medicine.md)
- [\[NeurIPS 2025\] Interpreting GFlowNets for Drug Discovery: Extracting Actionable Insights for Medicinal Chemistry](interpreting_gflownets_for_drug_discovery_extracting_actionable_insights_for_med.md)

</div>

<!-- RELATED:END -->
