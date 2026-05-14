---
title: >-
  [Paper Note] Attribution as Retrieval: Model-Agnostic AI-Generated Image Attribution
description: >-
  [CVPR 2026][Image Generation][Deepfake attribution] This paper proposes LIDA, which reformulates AI-generated image attribution from a classification problem into a retrieval problem. By leveraging low-bit-plane fingerpr…
tags:
  - "CVPR 2026"
  - "Image Generation"
  - "Deepfake attribution"
  - "image forensics"
  - "retrieval-based"
  - "bit-plane"
  - "model-agnostic"
date: 2026-05-08
content_hash: 356310698327c931
---

# Attribution as Retrieval: Model-Agnostic AI-Generated Image Attribution

**Conference**: CVPR 2026
**arXiv**: [2603.10583](https://arxiv.org/abs/2603.10583)
**Code**: [GitHub](https://github.com/hongsong-wang/LIDA)
**Area**: Image Forensics / AI Safety
**Keywords**: Deepfake attribution, image forensics, retrieval-based, bit-plane, model-agnostic

## TL;DR
This paper proposes LIDA, which reformulates AI-generated image attribution from a classification problem into a retrieval problem. By leveraging low-bit-plane fingerprints to capture generator-specific artifacts, combined with unsupervised pre-training and few-shot adaptation, LIDA achieves state-of-the-art Deepfake detection and image attribution under zero-shot and few-shot settings.

## Background & Motivation
**Background**: With the rapid advancement of AIGC technologies, Deepfake detection has seen considerable progress; however, attribution of AI-generated images to specific generative models remains an open problem. Existing approaches fall into two categories: generative watermarking (requiring access to the generative model) and classification-based attribution.

**Limitations of Prior Work**: (1) Generative watermarking requires full access to the generative model and modification of its architecture, lacking flexibility and generality; (2) closed-set attribution assumes all generators are known at training time, making it unable to handle emerging models; (3) open-set attribution, while accounting for unknown generators, still follows a classification paradigm and requires large amounts of unlabeled generated images for retraining, resulting in slow adaptation to new models.

**Key Challenge**: New generative models continue to emerge (e.g., Midjourney, DALL-E, Stable Diffusion), and the classification paradigm requires retraining each time to extend categories and collecting large amounts of data from new models—which is impractical in real-world scenarios.

**Goal**: To design a model-agnostic, scalable attribution framework that generalizes to unseen generators and requires only a small number of examples to rapidly adapt to new models.

**Key Insight**: Attribution is redefined as an instance retrieval problem (rather than classification). A registration database is maintained, and new models can be added with just a few example images without retraining. Low-bit-plane fingerprints are used in place of raw RGB as input to explicitly capture generator-specific noise.

**Core Idea**: Low-bit-plane fingerprints + retrieval paradigm = model-agnostic, scalable, few-shot-friendly AI image attribution.

## Method

### Overall Architecture
LIDA consists of three modules: (1) Low-Bit Fingerprint Generation—extracting low-bit-plane fingerprints from RGB images as input; (2) Unsupervised Pre-Training—pre-training on real image fingerprints from ImageNet to learn general noise structure representations; (3) Few-Shot Attribution Adaptation—fine-tuning the encoder with a small set of registered database samples using center loss and real-prototype contrastive loss. At inference, cosine similarity is used to retrieve the nearest neighbor from the registration database.

### Key Designs

1. **Low-Bit Fingerprint Generation**:

    - Function: Removes semantic content from images while retaining generator-specific noise patterns as attribution cues.
    - Mechanism: Bit-plane decomposition is applied to each channel of the RGB image: $\mathbf{x}_c = \sum_{k=0}^{7} 2^k \cdot \mathbf{b}_c^k$. The lowest 3 bit-planes are extracted and thresholded: $\tilde{\mathbf{x}}_c = 255 \cdot \text{sgn}(\sum_{k=0}^{2} 2^k \cdot \mathbf{b}_c^k)$. The resulting fingerprint image strips away nearly all semantic information while preserving the distinct noise fingerprints of individual generators.
    - Design Motivation: PCA visualization shows that images from different generators are intermixed in RGB space, whereas in the low-bit-plane fingerprint space, images from the same generator cluster distinctly, with clear separation between real and generated images.

2. **Unsupervised Pre-Training**:

    - Function: Learns general representations of the fingerprint space on large-scale real images.
    - Mechanism: A modified ResNet-50 (with low-level downsampling removed to preserve spatial information) is trained on ImageNet low-bit-plane fingerprints using image classification as the pretext task. The loss is standard cross-entropy: $\mathcal{L}_P = -\sum_{b=1}^{B} \sum_{c=1}^{C} s_b^c \log q_b^c$.
    - Design Motivation: Unsupervised pre-training provides robust weight initialization, enabling the model to learn transferable noise structural features and enhancing generalization to unseen generators.

3. **Few-Shot Attribution Adaptation**:

    - Function: Rapidly adapts to new generators using a minimal number (1–10 images per generator) of registered samples.
    - Mechanism: Rather than using cross-entropy (which disrupts the pre-trained feature space structure), center loss is adopted as the attribution loss $\mathcal{L}_A = \sum_{i=1}^{m} \|x_i - c_{y_i}\|_2^2$ to encourage intra-class compactness. The detection loss employs a real-prototype contrastive loss $\mathcal{L}_D$ that pulls real images toward the real prototype and pushes generated images away. The overall loss is $\mathcal{L} = \mathcal{L}_A + \lambda \mathcal{L}_D$ with $\lambda = 0.9$. Inference follows a two-stage process: real/fake detection first, followed by retrieval-based attribution.
    - Design Motivation: Center loss acts as a regularizer to preserve the structure of the pre-trained feature space, preventing feature drift under few-shot fine-tuning; contrastive loss enhances real/fake separation.

### Loss & Training
The detection loss $\mathcal{L}_D$ is based on a real-prototype contrastive loss: the average feature of all ImageNet images from the pre-training phase serves as the real class prototype $p_r$. Sigmoid and cosine similarity are used to attract real images and repel generated images, with temperature parameter $\tau$ controlling distribution sharpness. Fine-tuning uses batch size 32, learning rate $1 \times 10^{-4}$, and trains for 100 epochs.

## Key Experimental Results

### Main Results
Cross-architecture attribution on the GenImage dataset (8 generator categories, Rank-1 / mAP):

| Shot | Method | Avg Rank-1 | Avg mAP |
|------|--------|-----------|---------|
| 1-shot | ResNet | 17.4 | 37.5 |
| 1-shot | DIRE | 14.3 | 34.8 |
| 1-shot | ESSP | 17.0 | 36.0 |
| 1-shot | **LIDA** | **40.4** | **61.5** |
| 10-shot | ResNet | 21.4 | 22.4 |
| 10-shot | DIRE | 17.2 | 28.8 |
| 10-shot | ESSP | 22.4 | 23.0 |
| 10-shot | **LIDA** | **54.0** | **51.6** |

Zero-shot Deepfake detection on GenImage (Accuracy):

| Method | BigGAN | Mid | WuK | SDv4 | SDv5 | ADM | GLIDE | VQ | Avg |
|--------|--------|-----|-----|------|------|-----|-------|-----|-----|
| RIGID | 53.0 | 94.1 | 87.8 | 87.0 | 87.2 | 51.4 | 45.9 | 52.2 | 69.8 |
| FSD | 62.1 | 75.1 | 88.0 | 88.0 | 88.0 | 74.1 | 93.9 | 69.1 | 77.1 |
| **LIDA** | **91.0** | 85.9 | 86.2 | 86.3 | 86.8 | **85.5** | 83.9 | **84.5** | **86.3** |

### Ablation Study

| BF | $\mathcal{L}_P$ | $\mathcal{L}_A$ | $\mathcal{L}_D$ | Avg mAP Change |
|----|------|------|------|----------------|
| ✗ | ✗ | ✗ | ✗ | baseline (RGB + ImageNet) |
| ✓ | ✗ | ✗ | ✗ | +10.6% |
| ✓ | ✓ | ✗ | ✗ | +12.1% (additional +1.5%) |
| ✓ | ✓ | ✓ | ✗ | +15.8% (additional +3.7%) |
| ✓ | ✓ | ✓ | ✓ | +24.0% (additional +8.2%) |

Replacing both center loss and contrastive loss with cross-entropy reduces mAP by 3.9%.

### Key Findings
- Low-bit-plane fingerprints are the single largest contributing factor to attribution performance (+10.6% mAP).
- The detection loss $\mathcal{L}_D$ contributes the most among training components (+8.2%), effectively increasing the feature distance between real and generated images.
- Zero-shot accuracy on BigGAN reaches 91%, validating the strong discriminability of bit-plane fingerprints for GAN-generated content.
- The method exhibits strong robustness under JPEG compression; even when Gaussian blur disrupts the low-bit-plane distribution, the fingerprint features still outperform raw RGB.

## Highlights & Insights
- The paradigm shift of "attribution as retrieval" is elegant and principled—new models require only a few images added to the registration database, with zero retraining overhead.
- The physical intuition behind low-bit planes is compelling: high-order bits carry semantic content, while low-order bits carry noise and artifacts; bit-level decomposition cleanly isolates generator fingerprints.
- The design choice to replace cross-entropy with center loss is critical—it preserves the structure of the pre-trained feature space under few-shot fine-tuning.
- The method is extremely lightweight: a ResNet-50 backbone with millisecond-level inference.

## Limitations & Future Work
- Robustness to Gaussian blur is limited, as it directly disrupts the low-bit-plane distribution.
- Single-threshold zero-shot detection may lack flexibility in practical deployment.
- Evaluation is limited to GenImage and WildFake; assessment on the latest video generative models is absent.
- The robustness of the bit-plane approach to real-world image post-processing pipelines (e.g., social media compression chains) requires further validation.

## Related Work & Insights
- Generative watermarking methods (Tree-Ring, Gaussian Shading) require model modification; the proposed method is entirely model-agnostic.
- Closed-set method RepMix is restricted to known GANs; the proposed framework naturally supports open-set scenarios.
- The retrieval paradigm is extensible to video generator attribution by expanding 2D fingerprints into the temporal dimension.
- The technique of preserving feature space structure via center loss during few-shot fine-tuning is a transferable insight for related tasks.

## Rating
- Novelty: ⭐⭐⭐⭐ Redefining attribution as retrieval is a clear paradigm innovation; the low-bit-plane fingerprint is concise and effective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive coverage across two datasets, multiple shot settings, and detection + attribution + ablation experiments.
- Writing Quality: ⭐⭐⭐⭐ Motivation is clearly articulated and the method is presented fluently.
- Value: ⭐⭐⭐⭐ Provides a highly practical new paradigm for AI-generated image attribution, particularly well-suited to the rapidly evolving generative model ecosystem.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] AEDR: Training-Free AI-Generated Image Attribution via Autoencoder Double-Reconstruction](../../AAAI2026/image_generation/aedr_training-free_ai-generated_image_attribution_via_autoen.md)
- [\[CVPR 2026\] Diversity over Uniformity: Rethinking Representation in Generated Image Detection](diversity_over_uniformity_rethinking_representation_in_generated_image_detection.md)
- [\[CVPR 2026\] Diffusion Probe: Generated Image Result Prediction Using CNN Probes](diffusion_probe_generated_image_result_prediction_using_cnn_probes.md)
- [\[NeurIPS 2025\] Fast Data Attribution for Text-to-Image Models](../../NeurIPS2025/image_generation/fast_data_attribution_for_text-to-image_models.md)
- [\[AAAI 2026\] Aggregating Diverse Cue Experts for AI-Generated Image Detection](../../AAAI2026/image_generation/aggregating_diverse_cue_experts_for_ai-generated_image_detec.md)

</div>

<!-- RELATED:END -->
