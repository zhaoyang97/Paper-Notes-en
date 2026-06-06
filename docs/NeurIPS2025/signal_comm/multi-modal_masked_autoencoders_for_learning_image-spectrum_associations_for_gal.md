---
title: >-
  [Paper Note] Multi-Modal Masked Autoencoders for Learning Image-Spectrum Associations for Galaxy Evolution and Cosmology
description: >-
  [NeurIPS 2025][Signal & Communication][Multimodal masked autoencoders] This paper applies a Multimodal Masked Autoencoder (MMAE) to jointly model galaxy images (HSC-PDR2, five bands) and spectra (DESI-DR1)…
tags:
  - "NeurIPS 2025"
  - "Signal & Communication"
  - "Multimodal masked autoencoders"
  - "galaxy images"
  - "spectral reconstruction"
  - "redshift regression"
  - "missing modality"
date: 2026-05-08
content_hash: cb486ac8b91cdf51
---

# Multi-Modal Masked Autoencoders for Learning Image-Spectrum Associations for Galaxy Evolution and Cosmology

**Conference**: NeurIPS 2025
**arXiv**: [2510.22527](https://arxiv.org/abs/2510.22527)  
**Code**: Available (GitHub + Zenodo dataset GalaxiesML-Spectra)  
**Area**: Astrophysics / Multimodal Learning
**Keywords**: Multimodal masked autoencoders, galaxy images, spectral reconstruction, redshift regression, missing modality

## TL;DR

This paper applies a Multimodal Masked Autoencoder (MMAE) to jointly model galaxy images (HSC-PDR2, five bands) and spectra (DESI-DR1), constructing a cross-modal dataset GalaxiesML-Spectra of 134,533 galaxies. Under a 75% masking ratio, the model reconstructs major spectral emission lines and image morphology. When spectra are entirely absent at inference, the model achieves $\sigma_{\text{NMAD}}=0.016$ for redshift prediction using images alone, outperforming AstroCLIP while extending the redshift range to $z \sim 4$ for the first time.

## Background & Motivation

**Background**: Next-generation astronomical surveys will produce images of billions of galaxies, yet acquiring a single spectrum requires roughly 100 times more observing time than imaging. Spectra encode critical physical information including redshift, chemical composition, and star formation rates, but obtaining spectra at survey scale is impractical. Astronomers have long relied on photometric redshifts—redshift estimates derived from images—as a surrogate for spectroscopic redshifts.

**Limitations of Prior Work**: (1) Conventional photometric redshift methods (MLP, CNN, BCNN) rely solely on image information without learning deep cross-modal representations from image–spectrum associations. (2) AstroMAE applies MAE exclusively to galaxy images without incorporating the spectral modality. (3) AstroCLIP aligns images and spectra via contrastive learning but does not optimize for reconstruction. (4) Existing methods are constrained to redshift ranges of $z \lesssim 0.5$.

**Key Challenge**: At survey scale, images are abundant while spectra are scarce, requiring models to leverage image–spectrum associations even when spectra are unavailable. Existing approaches either operate on a single modality or rely on contrastive rather than generative objectives.

**Key Insight**: The paper transfers the MultiMAE architecture to astronomy, using high masking ratios to force the model to learn complementary cross-modal relationships. During training, spectra are entirely zeroed out with 50% probability to simulate the practical scenario in which the vast majority of survey galaxies lack spectra.

**Core Idea**: An MMAE is trained to jointly perform cross-modal reconstruction of galaxy images and spectra alongside redshift regression, enabling accurate redshift prediction from images alone by leveraging representations informed by image–spectrum associations.

## Method

### Overall Architecture

Both modalities undergo patch tokenization independently → 75% random masking → independent Transformer encoding per modality → cross-attention fusion → attention pooling to produce a global embedding → three parallel task heads: image decoding, spectral decoding, and redshift regression. Redshift regression is directly integrated into MAE training—a first for multimodal MAE.

### Key Designs

1. **Dual-Modality Patch Tokenization**:

    - Images ($64 \times 64 \times 5$ bands) are divided into $8 \times 8 \times 5$ patches via 2D convolution and projected to 256-dimensional embeddings with 2D learnable positional encodings.
    - Spectra (downsampled from 7,783 to 259 pixels) are tokenized with 1D patches of length 8 followed by linear projection.
    - **Design Motivation**: Images have a 2D spatial structure (morphology) while spectra have a 1D spectral structure (emission lines/continuum); 2D and 1D patch tokenization respectively preserve the native structure of each modality.

2. **Independent Encoding + Cross-Attention Fusion**:

    - Each modality is independently encoded by a 1D Transformer encoder (depth 4, 8-head attention, dropout 0.1).
    - Four cross-attention blocks are applied: image features query spectra (to acquire physical information) and spectral features query images (to acquire morphological information).
    - Attention pooling aggregates the outputs into a global embedding, which is concatenated into a joint representation.
    - **Design Motivation**: Independent encoding preserves modality integrity before cross-attention learns inter-modal associations.

3. **50% Spectral Zeroing During Training + Joint Redshift Loss**:

    - Spectra are entirely zeroed out with 50% probability to simulate real survey conditions in which most galaxies lack spectra.
    - Three task heads are trained jointly: image MSE ×0.1 + spectral MSE ×0.01 + redshift loss ×1.0.
    - Redshift loss: $\mathcal{L}_z = 1 - \frac{1}{1+(dz/0.15)^2}$, where $dz = (z_{\text{pred}} - z_{\text{spec}})/(1+z_{\text{spec}})$.
    - **Design Motivation**: Multitask joint training encodes both reconstruction and physical quantity prediction in the learned representations. Integrating redshift regression directly into MAE training—rather than as a downstream fine-tuning step—is a novel contribution.

### Loss & Training

AdamW optimizer (weight decay 0.01, lr 0.0001) with gradient clipping. Dataset split 70/15/15.

## Key Experimental Results

### Main Results (test set of 20,181 galaxies)

**Redshift Regression** (25% image masking + 100% spectral masking):

| Model | $\sigma_{\text{NMAD}}$ | Condition | Redshift Range |
|-------|----------------------|-----------|---------------|
| **MMAE (Ours)** | **0.016** | 25% image mask + no spectrum | $z \lesssim 0.4$ |
| AstroCLIP | 0.020 | Contrastive learning | $z \lesssim 0.4$ |
| BCNN (fine-tuned) | 0.012 | CNN specifically optimized | $z \lesssim 0.4$ |
| MMAE | 0.026 | 0% mask + no spectrum | $z \lesssim 0.4$ |

**Spectral Reconstruction**: The model recovers the positions of common emission lines (H-α at low redshift; Lyα and CIV at high redshift), but systematically overestimates line widths by a factor of 10–15 and underestimates line strengths.

### Ablation Study

| Configuration | Key Finding |
|---------------|-------------|
| 25% vs. 0% image masking | 25% masking ($\sigma=0.016$) outperforms full images ($\sigma=0.026$)—moderate masking acts as regularization |
| Low vs. high redshift | High accuracy at low redshift; performance degrades at high redshift due to training data bias toward low redshifts |
| Emission line reconstruction | Line positions are approximately correct but widths are too broad and strengths too weak for physical diagnostics |

### Key Findings

- **Masking as regularization**: A 25% image masking rate yields better redshift predictions than using full images, preventing overfitting to small-scale features and noise.
- MMAE achieves lower scatter than AstroCLIP (0.016 vs. 0.020), suggesting that generative pre-training confers advantages for downstream regression.
- BCNN still performs better (0.012); Transformers have not yet surpassed Inception-style CNNs for redshift prediction in low-data astronomical regimes.
- Redshift predictions exhibit a step-like artifact near $z \sim 2$, corresponding to strong lines such as Lyα entering and exiting the spectrograph's wavelength coverage.
- The model learns emission line positions but fails to learn physical parameters (line width, strength, and ratios).

## Highlights & Insights

- **Masking regularization effect**: The finding that 25% image masking improves redshift prediction implies that moderate information dropout prevents overfitting to noise, a principle transferable to other multimodal regression tasks.
- **Cross-modal inference under missing modalities**: Training with 50% complete spectral zeroing and evaluating with no spectra at test time still yields reasonable redshift predictions. Cross-attention effectively allows the image encoder to internalize spectral information. This approach is applicable to any scenario in which one modality is substantially more expensive to acquire than another.
- **Dataset contribution**: GalaxiesML-Spectra (134K galaxies, $z_{\max}=4.119$) is currently the largest publicly available paired image–spectrum dataset and provides lasting value to the astronomical machine learning community.

## Limitations & Future Work

- Emission line widths are severely overestimated (10–15×); physically informed losses (constraints on line centers, widths, and ratios) should be explored.
- Performance degrades significantly at high redshift; supplementing with more high-redshift samples is necessary.
- Image resolution is limited to $64\times64$, causing loss of fine morphological detail.
- Comparison with additional baselines (e.g., image-only MAE, CLIP + linear head) is absent.
- Future work could explore physics-driven masking strategies (simulating bandpass gaps and instrumental noise) and extension to text modalities.

## Related Work & Insights

- **vs. AstroCLIP**: Contrastive learning aligns images and spectra but does not optimize for reconstruction; the MMAE's generative objective yields richer cross-modal representations.
- **vs. AstroMAE**: AstroMAE is a single-modality image MAE; this work is the first to apply joint image–spectrum MAE in astronomy.
- **vs. BCNN**: A specifically optimized CNN still leads in redshift accuracy, and the advantage of Transformers in low-sample astronomical settings has not been established.

## Rating

- **Novelty**: ⭐⭐⭐ Transferring MultiMAE to astronomy is a reasonable application-level contribution; no major architectural innovations.
- **Experimental Thoroughness**: ⭐⭐⭐ The dataset is substantial, but baseline comparisons are insufficient and quantitative metrics for spectral reconstruction are lacking.
- **Writing Quality**: ⭐⭐⭐⭐ Astronomical context is clearly presented; limitations are discussed honestly.
- **Value**: ⭐⭐⭐⭐ The dataset contribution is significant and lays groundwork for future astronomical foundation models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Multi-modal Data Spectrum: Multi-modal Datasets are Multi-dimensional](../../ICLR2026/signal_comm/multi-modal_data_spectrum_multi-modal_datasets_are_multi-dimensional.md)
- [\[NeurIPS 2025\] Masked Symbol Modeling for Demodulation of Oversampled Baseband Communication Signals](masked_symbol_modeling_for_demodulation_of_oversampled_baseband_communication_si.md)
- [\[NeurIPS 2025\] The Last Vote: A Multi-Stakeholder Framework for Language Model Governance](the_last_vote_a_multi-stakeholder_framework_for_language_model_governance.md)
- [\[NeurIPS 2025\] Feature-aware Modulation for Learning from Temporal Tabular Data](feature-aware_modulation_for_learning_from_temporal_tabular_data.md)
- [\[NeurIPS 2025\] Contrastive Consolidation of Top-Down Modulations Achieves Sparsely Supervised Continual Learning](contrastive_consolidation_of_top-down_modulations_achieves_sparsely_supervised_c.md)

</div>

<!-- RELATED:END -->
