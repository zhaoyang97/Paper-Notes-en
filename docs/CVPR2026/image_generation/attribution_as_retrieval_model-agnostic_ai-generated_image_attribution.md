---
title: >-
  [Paper Note] Attribution as Retrieval: Model-Agnostic AI-Generated Image Attribution
description: >-
  [CVPR 2026][Image Generation][AIGC Attribution] This paper reframes AI-generated image attribution from a classification paradigm to an instance retrieval paradigm, proposing the LIDA framework. It extracts generator-specific fingerprints from RGB low-bit planes as input, and achieves open-set attribution via unsupervised pre-training on real images followed by few-shot adaptation. LIDA achieves average Rank-1 accuracies of 40.4%/77.5% on GenImage and WildFake under the 1-shot setting, substantially outperforming existing methods.
tags:
  - CVPR 2026
  - Image Generation
  - AIGC Attribution
  - Instance Retrieval
  - Low-Bit Plane Fingerprint
  - Few-Shot Learning
  - Deepfake Detection
date: 2026-05-08
content_hash: 2aaef3449df6716a
---

# Attribution as Retrieval: Model-Agnostic AI-Generated Image Attribution

**Conference**: CVPR 2026
**arXiv**: [2603.10583](https://arxiv.org/abs/2603.10583)
**Code**: [Available](https://github.com/hongsong-wang/LIDA)
**Area**: Diffusion Models / Image Generation
**Keywords**: AIGC Attribution, Instance Retrieval, Low-Bit Plane Fingerprint, Few-Shot Learning, Deepfake Detection

## TL;DR
This paper reframes AI-generated image attribution from a classification paradigm to an instance retrieval paradigm, proposing the LIDA framework. It extracts generator-specific fingerprints from RGB low-bit planes as input, and achieves open-set attribution via unsupervised pre-training on real images followed by few-shot adaptation. LIDA achieves average Rank-1 accuracies of 40.4%/77.5% on GenImage and WildFake under the 1-shot setting, substantially outperforming existing methods.

## Background & Motivation
The rapid development of AIGC technology has led to a proliferation of image generators (SD, FLUX, Midjourney, etc.), posing serious challenges to the authenticity and provenance of digital media.

Existing attribution methods fall into two categories: (1) **Generative watermarking**—embedding invisible watermarks during image generation (e.g., Tree-Ring, Gaussian Shading), which achieves high accuracy but requires full access to and modification of the generative model and cannot generalize across different generators; (2) **AI-generated image attribution**—operating independently of the generation process, but nearly all existing methods treat attribution as a classification problem. Closed-set methods assume all generators are known at training time and cannot adapt to newly emerging models; open-set methods attempt to handle unknown generators but still rely on large amounts of labeled or unlabeled AI-generated images for training, limiting their flexibility.

- **Background**: Rapid iteration of generative models vs. the heavy retraining required by attribution methods.
- **Key Challenge**: A general framework is needed that is **independent of generative models and can adapt to new generators with only a handful of registration samples**. The key insight of this paper is to redefine attribution as a retrieval problem: only a strong feature encoder needs to be trained, and a new generator can be registered by simply adding a few exemplar images to the database.

## Method

### Overall Architecture
LIDA consists of three modules: (1) low-bit fingerprint generation → (2) unsupervised pre-training → (3) few-shot attribution adaptation. A registration database $\mathcal{D}$ of AI-generated images is maintained; at inference time, attribution is performed by retrieving the nearest neighbor via feature similarity.

### Key Designs
1. **Low-Bit Fingerprint Generation**:

   - **Function**: Extracts generator-specific structured noise fingerprints from RGB images.
   - **Mechanism**: Bit-plane decomposition is applied to each channel $\mathbf{x}_c = \sum_{k=0}^{7} 2^k \cdot \mathbf{b}_c^k$; the lowest 3 bit planes are combined and binarized:
   $\tilde{\mathbf{x}}_c = 255 \cdot \text{sgn}(\sum_{k=0}^{2} 2^k \cdot \mathbf{b}_c^k)$
   This yields a fingerprint image that discards most image content while retaining generator-specific structural noise.
   - **Design Motivation**: PCA visualization shows that low-bit fingerprints lead to clear clustering separation across different generators, whereas features from raw RGB images are nearly indistinguishable across generators. Low-bit planes contain model-specific artifacts unintentionally embedded during the generation process.

2. **Unsupervised Pre-Training**:

   - **Function**: Pre-trains an encoder on low-bit fingerprints of large-scale real images (ImageNet) to learn general noise structure representations.
   - **Mechanism**: A ResNet-50 backbone is used (with early-layer downsampling removed to preserve spatial information), and ImageNet classification serves as the proxy task:
   $\mathcal{L}_P = -\sum_{b=1}^{B}\sum_{c=1}^{C} s_b^c \log q_b^c$
   - **Design Motivation**: Pre-training on real image fingerprints provides robust weight initialization; the learned intrinsic noise structures transfer to AI-generated image forensics.

3. **Few-Shot Attribution Adaptation**:

   - **Function**: Fine-tunes the encoder using a small number of samples from the registration database to distinguish between different generators.
   - **Mechanism**: Two loss functions are jointly optimized:
     - **Image attribution loss** (center loss, to avoid disrupting the pre-trained feature space structure):
     $$\mathcal{L}_A = \sum_{i=1}^{m} \|x_i - c_{y_i}\|_2^2$$
     Class centers $c_j$ are dynamically updated within each mini-batch.
     - **Deepfake detection loss** (contrastive loss based on real prototypes):
     $$\mathcal{L}_D = -\frac{1}{N_r}\sum_{i=1}^{N_r}\log\sigma(\frac{\text{sim}(x_i^r, p_r)}{\tau}) - \frac{1}{N_f}\sum_{j=1}^{N_f}\log(1-\sigma(\frac{\text{sim}(x_i^f, p_r)}{\tau}))$$
     Total loss: $\mathcal{L} = \mathcal{L}_A + \lambda \mathcal{L}_D$
   - **Design Motivation**: Center loss is preferred over cross-entropy because CE disrupts the feature space structure learned during pre-training. The two-stage attribution strategy (first detecting whether an image is AI-generated, then attributing it to a specific generator) better reflects real-world workflows.

### Loss & Training
- **Pre-training**: Classification proxy task on ImageNet low-bit fingerprints.
- **Adaptation**: Center loss (attribution) + prototype contrastive loss (detection), requiring only a small number (1/5/10-shot) of AI-generated samples.
- **Inference**: Two-stage pipeline — Deepfake detection first, followed by retrieval-based attribution.

## Key Experimental Results

### Main Results (GenImage Cross-Generator Attribution)

| Shot | Method | Avg Rank-1 | Avg mAP |
|------|--------|-----------|---------|
| 1-shot | ResNet | 17.4 | 37.5 |
| 1-shot | DIRE | 14.3 | 34.8 |
| 1-shot | ESSP | 17.0 | 36.0 |
| 1-shot | **Ours** | **40.4** | **61.5** |
| 10-shot | ResNet | 21.4 | 22.4 |
| 10-shot | DIRE | 17.2 | 28.8 |
| 10-shot | ESSP | 22.4 | 23.0 |
| 10-shot | **Ours** | **54.0** | **51.6** |

WildFake cross-architecture attribution (1-shot): Ours achieves an average Rank-1 of 77.5% and mAP of 87.7%, far surpassing the second-best method at 37.4%/60.4%.

### Ablation Study

| Configuration | Key Metric | Description |
|--------------|-----------|-------------|
| RGB input vs. low-bit fingerprint | PCA visualization | Generator features are nearly inseparable in RGB space; low-bit fingerprints yield natural clustering |
| Unsupervised pre-training | Baseline | ImageNet classification proxy task provides robust initialization |
| Center loss vs. CE | Attribution quality | Center loss preserves the pre-trained feature space structure; CE disrupts it |
| Two-stage attribution | Detect then attribute | More reliable than end-to-end classification |

### Key Findings
- BigGAN achieves 97–100% Rank-1 under the 1-shot setting, indicating its generation fingerprint is highly distinctive.
- Stable Diffusion v1.4/v1.5 versions share similar fingerprints, leading to mutual confusion.
- Low-bit plane fingerprint extraction is computationally trivial (bitwise operations only), adding negligible inference overhead.
- Usable accuracy is achieved with only 1 shot, demonstrating the practical value of the retrieval paradigm under extremely limited samples.

## Highlights & Insights
- The paradigm shift from classification to retrieval is both natural and highly effective — a new generator requires only a few registered images, with no model retraining.
- The low-bit plane fingerprint is a concise yet powerful prior: it is extracted at near-zero cost yet significantly amplifies feature differences across generators.
- The insight of using center loss instead of CE to protect the pre-trained feature space is a noteworthy design principle.
- The two-stage attribution pipeline (detection before attribution) aligns with real-world workflows and allows each stage to be optimized independently.

## Limitations & Future Work
- Distinguishing between models from the same family (e.g., SD v1.4/v1.5) remains limited and calls for more fine-grained fingerprinting.
- The robustness of low-bit planes to image compression (JPEG) is insufficiently discussed, as compression can corrupt low-bit information.
- The registration database requires manual maintenance; automatically discovering and registering new generators remains an open problem.
- Evaluation is conducted only at the image level; attribution for partially generated regions (e.g., inpainting) is not addressed.

## Related Work & Insights
- The fundamental distinction from watermarking methods such as Tree-Ring and Gaussian Shading lies in the fact that watermarking requires modifying the generative model, whereas this method is entirely post hoc.
- The approach is conceptually analogous to camera fingerprinting (PRNU), effectively transferring the notion of a "camera fingerprint" to a "generator fingerprint."
- Low-bit plane analysis can be combined with frequency-domain methods to further enhance discriminability.
- The retrieval paradigm is extensible to provenance tracing for AI-generated video and audio.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The paradigm shift from classification to retrieval is clear and compelling, and the low-bit fingerprint is an effective prior; however, no single technical component is entirely novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive evaluation across two large-scale datasets, cross-generator and cross-architecture settings, and 1/5/10-shot configurations.
- **Writing Quality**: ⭐⭐⭐⭐ The problem formulation is clear and the pipeline design is concise.
- **Value**: ⭐⭐⭐⭐ The scalability of the retrieval paradigm and its low sample requirements offer practical value for AIGC security.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] AEDR: Training-Free AI-Generated Image Attribution via Autoencoder Double-Reconstruction](../../AAAI2026/image_generation/aedr_training-free_ai-generated_image_attribution_via_autoen.md)
- [\[CVPR 2026\] V-Bridge: Bridging Video Generative Priors to Versatile Few-shot Image Restoration](v-bridge_bridging_video_generative_priors_to_versatile_few-shot_image_restoratio.md)
- [\[CVPR 2026\] TriDF: Evaluating Perception, Detection, and Hallucination for Interpretable DeepFake Detection](tridf_evaluating_perception_detection_and_hallucination_for_interpretable_deepfa.md)
- [\[CVPR 2026\] Diversity over Uniformity: Rethinking Representation in Generated Image Detection](diversity_over_uniformity_rethinking_representation_in_generated_image_detection.md)
- [\[CVPR 2026\] Diffusion Probe: Generated Image Result Prediction Using CNN Probes](diffusion_probe_generated_image_result_prediction_using_cnn_probes.md)

<!-- RELATED:END -->
