---
title: >-
  [Paper Note] MEIcoder: Decoding Visual Stimuli from Neural Activity by Leveraging Most Exciting Inputs
description: >-
  [NeurIPS 2025][LLM Evaluation][Visual decoding] MEIcoder is proposed to leverage neuron-specific Most Exciting Inputs (MEIs) as biological priors, combined with SSIM loss and adversarial training, to achieve state-of-the-art visual stimulus reconstruction from neural population activity in the primary visual cortex (V1), with particular strengths in small-dataset and low-neuron-count regimes.
tags:
  - NeurIPS 2025
  - LLM Evaluation
  - Visual decoding
  - Most Exciting Input (MEI)
  - primary visual cortex
  - neural population activity
  - adversarial training
date: 2026-05-08
content_hash: 45a2fb7160ebc150
---

# MEIcoder: Decoding Visual Stimuli from Neural Activity by Leveraging Most Exciting Inputs

**Conference**: NeurIPS 2025
**arXiv**: [2510.20762](https://arxiv.org/abs/2510.20762)
**Code**: [https://github.com/Johnny1188/meicoder](https://github.com/Johnny1188/meicoder)
**Area**: Computational Neuroscience / Brain-Computer Interface
**Keywords**: Visual decoding, Most Exciting Input (MEI), primary visual cortex, neural population activity, adversarial training

## TL;DR

MEIcoder is proposed to leverage neuron-specific Most Exciting Inputs (MEIs) as biological priors, combined with SSIM loss and adversarial training, to achieve state-of-the-art visual stimulus reconstruction from neural population activity in the primary visual cortex (V1), with particular strengths in small-dataset and low-neuron-count regimes.

## Background & Motivation

Decoding visual stimuli from brain activity is a fundamental challenge in understanding the brain and a critical application in brain-computer interfaces. Several inherent difficulties arise:

**Data scarcity**: Per-subject data are limited, and training machine learning models from scratch often yields low-fidelity reconstructions.

**Generative AI hallucination**: While leveraging pretrained generative models (e.g., diffusion models) can produce high-resolution images, these outputs often contain fabricated content. Prior work has formally demonstrated that diffusion-based decoding exhibits "output dimensionality collapse," limiting the range of decodable features.

**Ill-posedness of the inverse problem**: Recovering high-information images from a small number of neurons—which provide highly compressed and noisy signals—is inherently difficult.

Existing methods suffer from either low reconstruction fidelity or unreliability (hallucination), and most do not directly optimize reconstruction quality in pixel space. A new approach is therefore needed to balance biological priors with data-driven signals.

## Method

### Overall Architecture

MEIcoder consists of two components: (1) a **Readin module**—trained independently per subject/dataset, embedding neural activity into the latent space of the core module; and (2) a **Core module**—a six-layer CNN shared across subjects, mapping from the latent space back to image space. This decoupled design enables reuse of learned representations across heterogeneous multi-subject data.

### Key Designs

1. **MEI-driven Readin module**: This is the central innovation. Each neuron's response $r_i$ and its learnable embedding $\mathbf{e}_i$ are mapped through a single-layer network $g_\psi$ to a context representation $\mathbf{C}_i \in \mathbb{R}^{h \cdot w}$. The precomputed MEI (the image maximally exciting the neuron) $\mathbf{M}_i$ is then element-wise multiplied with the context representation to yield a neural map $\mathbf{H} = \mathbf{M} \odot \mathbf{C}$. A pointwise convolution then compresses the result to a fixed number of channels $d_c$.

    - **Intuition**: MEIs encode receptive field information. A simple linear decoding baseline would reconstruct an image by summing all neurons' MEIs weighted by their response magnitudes. MEIcoder delegates this combination to the nonlinear CNN core.
    - MEIs need only be generated once from training data (approximately 70 minutes, compared to 11 hours for decoder training).

2. **SSIM reconstruction loss**: The negative log SSIM loss replaces conventional MSE: $\mathcal{L}_{SSIM} = -\log(\frac{SSIM(\mathbf{y}, \hat{\mathbf{y}}) + 1}{2} + \epsilon)$. SSIM captures luminance, contrast, and structural dimensions of perception, providing better guidance toward perceptually accurate reconstructions than MSE. VGG perceptual loss was found to be unstable during training and to introduce high-frequency artifacts.

3. **Adversarial training**: An auxiliary CNN discriminator distinguishes reconstructed images from real ones via an LS-GAN loss, pushing reconstructions toward the natural image manifold. Unlike standard GANs, the decoder is conditioned on neural responses and directly optimizes for spatially accurate reconstruction. Target-label noise (analogous to one-sided label smoothing) is introduced during discriminator training to stabilize optimization. Loss weights are $\lambda_{SSIM}=0.9$ and $\lambda_{ADV}=0.1$.

### Loss & Training

- Total loss: $\mathcal{L} = 0.9 \cdot \mathcal{L}_{SSIM} + 0.1 \cdot \mathcal{L}_{ADV}$
- AdamW optimizer, 300 epochs, best checkpoint selected based on validation-set Alex(5) score
- Core module is parameter-efficient: 6-layer CNN with channel progression 480→256→256→128→64→1, dropout 0.35
- Multi-subject training: shared core with independent readins per subject, reducing parameter count by approximately threefold

## Key Experimental Results

### Main Results

Evaluated on three datasets (Brainreader/SENSORIUM 2022: real mouse V1; Synthetic Cat V1: high-fidelity cat V1 computational model).

| Method | Brainreader SSIM | Brainreader Alex(5) | SENSORIUM SSIM | SENSORIUM Alex(5) |
|------|-----------------|--------------------|-----------------|--------------------|
| InvEnc | .321 | .896 | .288 | .720 |
| EGG | .256 | .659 | .256 | .755 |
| MonkeySee | .232 | .826 | .185 | .523 |
| MindEye2 | .277 | .878 | .210 | .762 |
| **MEIcoder** | **.400** | **.990** | **.331** | **.896** |
| **MEIcoder (FT)** | **.424** | .977 | .318 | **.908** |

| Method | Synthetic Cat V1 SSIM | PixCorr | Alex(2) | Alex(5) |
|------|----------------------|---------|---------|---------|
| InvEnc | **.771** | **.833** | .986 | .978 |
| MEIcoder | **.774** | .777 | **.994** | **.987** |

### Ablation Study

| Ablation Setting | Relative Impact (vs. full model) | Notes |
|---------|--------------------|----|
| Remove MEI | **Largest impact** (significant drop across all metrics) | MEI is the primary performance driver |
| Remove neuron embeddings | Moderate impact | Loss of additionally encoded attributes |
| Replace SSIM with MSE | Moderate impact | Degraded perceptual quality |

### Key Findings from Scaling Experiments

| Condition | Key Observation |
|------|---------|
| Training samples → 1000 | Already surpasses the second-best method trained on full data in Alex(2) |
| Neuron count → 1000–2500 | Alex(2) reaches 95%+; handwritten digits become distinguishable |
| 46,875 neurons | PixCorr has not saturated, indicating continued benefit from more neurons |

### Key Findings

- MEI is the single largest driver of performance—even with substantial Gaussian noise added to MEIs (std=1), performance remains on par with or superior to the baseline.
- MEIcoder exhibits significantly lower variance than MindEye2, indicating more reliable results and avoiding the hallucination issues of generative AI approaches.
- Only 1,000 training samples are needed to surpass all competing methods trained on full data.
- Between 1,000 and 2,500 V1 neurons suffice for fine-grained reconstruction.
- Multi-subject pretraining followed by single-subject fine-tuning further improves performance on Brainreader, but yields no benefit on SENSORIUM.
- Concept analysis reveals: (1) decoding proceeds from coarse to fine across layers; (2) many neurons encode global luminance conditions; (3) response patterns consistent with "black-dominant" OFF neurons are present.

## Highlights & Insights

- **MEIs as decoding priors constitute the core innovation**: Unlike generative AI methods that employ image priors unrelated to brain signals, MEIs are derived directly from encoding models fitted to actual neurons, providing biologically aligned priors.
- Exceptional data efficiency: 1,000 samples and 1,000 neurons suffice for meaningful decoding, which is critical for practical BCI applications.
- The Core–Readin separation architecture is elegant and practical, addressing the challenge of reusing learning signals across heterogeneous multi-subject data.
- Diffusion-based methods such as MindEye2 may appear visually sharp but are spatially inaccurate and prone to hallucination—this paper provides formal supporting evidence.
- The unified benchmark (160,000+ samples) constitutes a lasting contribution to the community.
- Concept analysis demonstrates MEIcoder's potential as a tool for scientific discovery.

## Limitations & Future Work

- Validation is limited to V1, which encodes low-level features; applicability to higher visual areas (e.g., V4) has not been tested.
- Cross-subject transfer is inconsistent (effective for Brainreader, ineffective for SENSORIUM).
- The synthetic cat dataset, while validating the method, leaves open the gap relative to real primate data.
- MEI quality depends on the accuracy of the underlying encoding model; limitations of the encoding model may propagate to decoding.
- Reconstruction resolution remains low (e.g., 36×64, 22×36), and high-resolution reconstruction remains unexplored.
- Validation on human data is absent; cross-species generalization from mouse/cat to human is an open problem.

## Related Work & Insights

- InvEnc (Cobos et al., 2022) influenced this work through its encoder-inversion approach to decoding.
- Energy Guided Diffusion (Pierzchlewicz et al., 2023) guides decoding with diffusion priors, but optimizes in neural activity space rather than pixel space.
- MonkeySee (Le et al., 2024) and its homeomorphic decoder and U-Net architecture serve as direct points of comparison.
- The formal analysis of "spurious reconstructions" by Shirakawa et al. (2025) supports the paper's decision to avoid generative AI priors.
- The methodology carries reference value for decoding research in other modalities (fMRI, EEG).

## Rating
- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Leveraging Robust Optimization for LLM Alignment under Distribution Shifts](leveraging_robust_optimization_for_llm_alignment_under_distribution_shifts.md)
- [\[NeurIPS 2025\] BLINK-Twice: You See But Do You Observe? A Reasoning Benchmark on Visual Perception](blink-twice_you_see_but_do_you_observe_a_reasoning_benchmark_on_visual_perceptio.md)
- [\[CVPR 2026\] Free-Grained Hierarchical Visual Recognition](../../CVPR2026/llm_evaluation/free-grained_hierarchical_visual_recognition.md)
- [\[AAAI 2026\] Structured Language Generation Model: Loss Calibration and Formatted Decoding for Efficient Text](../../AAAI2026/llm_evaluation/structured_language_generation_model_loss_calibration_and_formatted_decoding_for.md)
- [\[ACL 2026\] Contrastive Decoding Mitigates Score Range Bias in LLM-as-a-Judge](../../ACL2026/llm_evaluation/contrastive_decoding_mitigates_score_range_bias_in_llm-as-a-judge.md)

</div>

<!-- RELATED:END -->
