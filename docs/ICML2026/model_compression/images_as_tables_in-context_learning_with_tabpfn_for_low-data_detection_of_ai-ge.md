---
title: >-
  [Paper Note] Images as Tables: In-Context Learning with TabPFN for Low-Data Detection of AI-Generated Images
description: >-
  [ICML 2026][Model Compression][AIGC Detection] The authors reformulate AI-generated image detection into a three-stage pipeline: first, each image is compressed into a 768-dimensional CLS vector using a frozen DINOv3…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "AIGC Detection"
  - "TabPFN"
  - "DINOv3"
  - "In-Context Learning"
  - "Cross-Generator Transfer"
date: 2026-05-08
content_hash: 54e4fffbaa99d92d
---

# Images as Tables: In-Context Learning with TabPFN for Low-Data Detection of AI-Generated Images

**Conference**: ICML 2026  
**arXiv**: [2606.00872](https://arxiv.org/abs/2606.00872)  
**Code**: https://github.com/jpwalter30/Towards-Generalizable-Detection-of-AI-Generated-Images  
**Area**: AI-Generated Image Detection / Tabular Foundation Models / In-Context Learning  
**Keywords**: AIGC Detection, TabPFN, DINOv3, In-Context Learning, Cross-Generator Transfer  

## TL;DR
The authors reformulate AI-generated image detection into a three-stage pipeline: first, each image is compressed into a 768-dimensional CLS vector using a frozen DINOv3; then, it is reduced to 500 dimensions via PCA to serve as a tabular row; finally, it is fed into TabPFN for in-context inference. This transforms the problem from "retraining the classification head for every new generator" to "swapping the context samples in TabPFN." In low-data and cross-generator scenarios on GenImage, this method leads the strong baseline LATTE by up to 8.2% and excels in 54 out of 64 generator transfer pairs.

## Background & Motivation

**Background**: AI-generated image detection has been repeatedly proven to be a "moving target" problem: detectors trained on one generator (GAN or Diffusion) often fail when encountering new versions like Midjourney, Stable Diffusion, or Wukong. Current mainstream solutions remain image-domain classifiers—training a real/fake head on a CNN/ViT backbone, or using specialized detection based on diffusion denoising trajectories (LATTE) and CLIP representations (Cozzolino et al.), sometimes augmented with frequency or fingerprint features. These methods perform strongly in large-scale i.i.d. scenarios and have made cross-generator evaluation a standard benchmark.

**Limitations of Prior Work**: (i) Adapting to new generators still requires "swapping the head or retraining," either by training a classifier from scratch or fine-tuning the backbone on new data, which is operationally inefficient; (ii) The number of labels available for new generators in real-world forensic scenarios is often limited to dozens or hundreds, far below the scale required to train a ViT classification head; (iii) Almost all existing detectors couple "representation learning" with "discriminative learning," requiring gradients for the entire network whenever new data arrives, which is slow and prone to overfitting to generator-specific fingerprints.

**Key Challenge**: Detection capability primarily stems from strong visual representations, whereas adaptation speed is limited by the training paradigm of the discriminative head. Binding them together implies that even small updates to labels require a gradient pass through the entire deep model.

**Goal**: (1) Establish "Image → Table → In-Context Inference" as a viable new paradigm for AIGC detection; (2) Systematically compare low-data and cross-generator performance under four types of generator evaluation protocols on GenImage; (3) Benchmarking against the current SOTA diffusion detector LATTE to identify the trade-off boundaries between TabPFN-context and image-classification heads across different data scales.

**Key Insight**: The authors observe that TabPFN, a Prior-Data Fitted Network, can perform Bayesian-like inference on small tabular datasets without training, relying solely on in-context samples. By compressing each image into a row of structured features, AIGC detection is reduced to a standard small-data tabular classification problem, where TabPFN’s low-data advantage aligns perfectly with the "few labels" constraint in forensic scenarios.

**Core Idea**: DINOv3 ViT-B/16 is frozen as a visual encoder, and PCA is used to reduce the 768-dimensional CLS token to 500 dimensions (matching the 500-feature limit of the current TabPFN). TabPFN then performs in-context binary classification for each test image using a set of labeled "context rows." Adapting to a new generator only requires updating a few labeled samples in the context, without modifying the encoder or the classifier.

## Method

### Overall Architecture
DINOv3-PCA-TabPFN is a three-stage pipeline where the Image → Table → Decision chain is interpreted as "converting an image into a labeled tabular record and letting a general-purpose tabular foundation model judge real vs. fake."

The first stage is visual encoding: images are loaded in RGB, resized to 256 pixels on the shorter side, center-cropped to $224\times 224$, and normalized using ImageNet statistics. DINOv3 ViT-B/16 in eval mode produces an $N\times 768$ CLS feature matrix, with no image-domain fine-tuning of the backbone. The second stage is Incremental PCA dimensionality reduction: a PCA is fitted on training set features to map the 768 dimensions to 500-dimensional structured rows $z(h(x))\in\mathbb{R}^{500}$, with the same principal components applied to the test set. The third stage is TabPFN in-context inference: the reduced training rows (with 0/1 labels) and the test row are fed into TabPFN, which outputs real/fake predictions via its Prior-Data Fitted Network without gradient updates. This version of TabPFN is limited to 10,000 rows and 500 dimensions.

The brilliance of this pipeline lies not in any single advanced module, but in the realization of "adaptation = swapping context rows": when a new generator is encountered, one only needs to re-calculate PCA coefficients on a small sample set and insert them into the TabPFN context. Neither the encoder nor the discriminator requires gradient updates.

### Key Designs

1. **Frozen DINOv3 as a General Visual Encoder**:
    - **Function**: Provides high-quality, generator-agnostic visual representations, decoupling AIGC detection from "end-to-end training of an image classifier" into "forensic features + general discriminator."
    - **Mechanism**: The CLS token of DINOv3 ViT-B/16 is used directly as the global representation. The backbone remains in eval mode without any fine-tuning on real/fake signals; representation learning is entirely handled by self-supervised DINOv3 pre-training. DINOv3 proved more robust across accuracy, precision, recall, F1, and AUC compared to DINOv2, DCT, or FFT features.
    - **Design Motivation**: Forensic literature indicates classification heads easily overfit to generator fingerprints. By locking this risk within the TabPFN context, the semantic/texture representations from the encoder remain valid even if the new generator differs significantly from the training set; one only needs to define the new decision boundary via context samples.

2. **PCA 500-D Adaptation for TabPFN**:
    - **Function**: Compresses DINOv3’s 768-dimensional features to 500 to fit the input limits of TabPFN while removing irrelevant variance.
    - **Mechanism**: IncrementalPCA is fitted on training features to obtain $z(h(x))\in\mathbb{R}^{500}$. Evaluation scripts construct balanced sets under four generator-aware protocols (Multi-Multi, Multi-Single, Single-Multi, Single-Single), with PCA fitting always occurring on the training side to avoid leakage.
    - **Design Motivation**: Updating PCA coefficients is considered part of the adaptation—a step orders of magnitude faster than training a head, ensuring the input dimension always meets constraints and enabling "truly training-free" in-context inference.

3. **TabPFN In-Context Inference instead of Gradient Training**:
    - **Function**: Shifts real/fake discrimination from "gradient-based head training" to "Bayesian inference based on labeled context rows."
    - **Mechanism**: TabPFN (Hollmann 2023) takes a "context set" and a "test row" to produce a classification posterior in a single forward pass. This work uses tiny training scales of $k\in\{25,30,75,150,300,625\}$ samples per generator, with test sets unified at 10,000 images.
    - **Design Motivation**: In the low-data regime, "swapping context vs. training a new head" offers an order-of-magnitude advantage. In practice, this pipeline turns "adding a new generator" into a lightweight feature extraction and context assembly operation, reducing the total time from minutes to seconds.

### Loss & Training
No training occurs on the vision side: DINOv3 and TabPFN are frozen, and PCA is a closed-form fit. "Training" is equivalent to "preparing a new context table," which is the core advantage of the paradigm. Baselines like LATTE are trained normally under their respective protocols for fair comparison at $k\in\{150,300,625\}$.

## Key Experimental Results

### Main Results
Using GenImage (ImageNet real images + 8 generators: ADM, BigGAN, GLIDE, Midjourney, SDv1.4, SDv1.5, VQDM, Wukong), evaluations followed Multi-Multi (pooled), Multi-Single (pooled-to-single), Single-Multi (single-to-pooled), and Single-Single (pairwise) protocols.

| Protocol | Training Scale $k$ | LATTE | DINOv3-PCA-TabPFN | Gain |
|------|--------------|-------|-------------------|------|
| Multi-Multi (Low Data) | $k=25$ | — | **78%** | Start at 78% |
| Multi-Multi (Medium) | Small shared $k$ | Behind | Leads up to +8.2% | Strongest interval |
| Multi-Multi (High Data) | $k=625$ | Leads +7.4% | Behind | LATTE recovers at scale |
| Single-Single (Transfer) | $k=625$ | — | Wins 54/64 pairs | Max lead +31.5% |

### Ablation Study
| Configuration | Key Metrics | Description |
|------|---------|------|
| Full: DINOv3 + PCA-500 + TabPFN | Best across five metrics | Complete pipeline |
| Encoder: DINOv2 + TabPFN | Weaker performance | DINOv3 contributes significantly |
| Encoder: DCT/FFT + TabPFN | Significant drop | Semantic features > Handcrafted frequency |
| Classifier: MLP on DINOv3 | Matches at scale, fails at low-data | Validates TabPFN's small-context edge |
| Low Data $k=25$ (Pooled) | 78% | Higher than any MLP-based approach |

### Key Findings
- The combination of strong representations and in-context discrimination excels in low-data and cross-generator intervals. Both DINOv3 and TabPFN are necessary components—replacing either leads to performance degradation.
- In the 64-pair Single-Single protocol, DINOv3-PCA-TabPFN outperformed LATTE in 54 pairs. However, as $k$ increased, some cross-pair accuracies dropped, suggesting TabPFN might slightly specialize to the generator fingerprint as the context set grows.
- Generator difficulty varies: BigGAN and GLIDE are highly separable in PCA space, requiring minimal context. ADM, Midjourney, and Wukong distributions overlap significantly with real images in DINOv3 space, requiring more context for slower gains.
- LATTE's 7.4% lead at $k=625$ in pooled settings is mainly due to TabPFN's 10,000-row limit—once the "context is full," TabPFN lacks the headroom to improve, unlike gradient-based detectors.

## Highlights & Insights
- Proposes the "Images as Tables" paradigm by merging general-purpose vision foundation models with tabular foundation models. Once images are rows, any tool for small-data tables can be applied to AIGC detection.
- Adaptation costs are reduced to "re-sampling PCA + reassembling context." No gradients are involve, reducing the engineering burden of "training new heads" to nearly zero.
- The paper clearly defines the boundary: choose TabPFN for low-data/cross-generator tasks and LATTE for large-scale pooled scenarios, providing a roadmap for hybrid strategies.
- The two-stage abstraction (frozen-encode then tabular-ICL) is portable to other low-data vision problems beyond forensics, such as medical or remote sensing few-shot classification.

## Limitations & Future Work
- The 10,000-row $\times$ 500-feature limit of the current TabPFN prevents it from consuming the full GenImage pooled set, which is the primary reason LATTE takes the lead in high-data regimes. Future versions (like TabPFN-2.5) might shift this boundary.
- The pipeline relies heavily on DINOv3 representation quality. If future generators target DINOv3 specifically, the pipeline might lack the additional robustness of image-domain defenses (e.g., frequency-domain defense).
- Experiments were limited to binary classification on GenImage without assessing robustness against degradations (compression, blur, crop). PCA is also a linear reduction; stronger non-linear projections could be explored.

## Related Work & Insights
- **vs. LATTE (2025)**: LATTE uses diffusion denoising trajectories. This method, without diffusion-specific signals, outperforms it in cross-generator scenarios without training discriminative heads.
- **vs. CLIP-based Detection (Cozzolino 2024)**: Both use large pre-trained models, but CLIP-based detectors still require i.i.d. head training. This method eliminates training via in-context inference.
- **vs. Native TabPFN**: While TabPFN was designed for structured datasets, this work extends it as a "universal low-data discriminator" for visual features, serving as a template for cross-modal structured inference.
- **vs. Frequency/Fingerprint Detection (Yu 2019)**: Fingerprint methods fail across generators. DINOv3's self-supervised representations bypass this, delegating the transfer pressure to TabPFN’s context set, resulting in 54/64 transfer wins.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Towards Generalizable AI-Generated Image Detection via Image-Adaptive Prompt Learning](../../CVPR2026/model_compression/towards_generalizable_ai-generated_image_detection_via_image-adaptive_prompt_lea.md)
- [\[NeurIPS 2025\] AI-Generated Video Detection via Perceptual Straightening](../../NeurIPS2025/model_compression/ai-generated_video_detection_via_perceptual_straightening.md)
- [\[ICML 2026\] Easier to Judge Than to Find: Predicting In-Context Learning Success for Demonstration Selection](easier_to_judge_than_to_find_predicting_in-context_learning_success_for_demonstr.md)
- [\[ICML 2026\] Energy-Structured Low-Rank Adaptation for Continual Learning](energy-structured_low-rank_adaptation_for_continual_learning.md)
- [\[ICML 2026\] Procedural Pretraining: Warming Up Language Models with Abstract Data](procedural_pretraining_warming_up_language_models_with_abstract_data.md)

</div>

<!-- RELATED:END -->
