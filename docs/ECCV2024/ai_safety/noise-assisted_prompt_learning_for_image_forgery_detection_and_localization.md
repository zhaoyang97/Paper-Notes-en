---
title: >-
  [Paper Note] Noise-Assisted Prompt Learning for Image Forgery Detection and Localization
description: >-
  [ECCV 2024][AI Safety][Image Forgery Detection] This paper proposes CLIP-IFDL, a CLIP-based image forgery detection and localization model. By employing instance-aware dual-stream prompt learning and a forgery-enhanced noise adapter, it addresses CLIP's lack of domain-specific prompts and forgery sensitivity in forgery detection, successfully transferring CLIP's open-world generalization capability to the forgery detection task.
tags:
  - "ECCV 2024"
  - "AI Safety"
  - "Image Forgery Detection"
  - "CLIP"
  - "Prompt Learning"
  - "Noise Adapter"
  - "Multi-domain Fusion"
date: 2026-05-08
content_hash: 6ea92cf5a1736e54
---

# Noise-Assisted Prompt Learning for Image Forgery Detection and Localization

**Conference**: ECCV 2024  
**Authors**: Dong Li, Jiaying Zhu, Xueyang Fu, Xun Guo, Yidi Liu, Gang Yang, Jiawei Liu, Zheng-Jun Zha  
**Code**: None  
**Area**: AI Security / Image Forgery Detection  
**Keywords**: Image Forgery Detection, CLIP, Prompt Learning, Noise Adapter, Multi-domain Fusion

## TL;DR

This paper proposes CLIP-IFDL, a CLIP-based image forgery detection and localization model. By employing instance-aware dual-stream prompt learning and a forgery-enhanced noise adapter, it addresses CLIP's lack of domain-specific prompts and forgery sensitivity in forgery detection, successfully transferring CLIP's open-world generalization capability to the forgery detection task.

## Background & Motivation

**Background**: Image Forgery Detection and Localization (IFDL) aims to determine whether an image has been manipulated and pinpoint the forged regions. With the proliferation of image editing tools (such as Photoshop) and generative models (like GANs and Diffusion models), the importance of IFDL has increased. Existing methods primarily rely on detection networks trained from scratch, which exhibit limited generalization.

**Limitations of Prior Work**: (1) Traditional forgery detection methods depend on training data of specific manipulation types (splicing, copy-move, inpainting, etc.), resulting in poor generalization to unseen manipulation techniques. (2) Although recent large-scale vision-language pre-trained models (such as CLIP) have demonstrated powerful open-world reasoning abilities, applying them directly to forgery detection faces two challenges: CLIP lacks specialized, forgery-related prompts, and its vision encoder is insensitive to subtle forgery traces (since forgery clues usually reside in the noise domain rather than the semantic domain).

**Key Challenge**: While CLIP possesses robust generalization capability (which is precisely what forgery detection requires), it aligns vision and language at the semantic level. In contrast, forgery detection necessitates capturing pixel-level subtle artifacts and noise inconsistencies. The core struggle lies in leveraging CLIP's generalization strengths while simultaneously endowing it with forgery awareness.

**Goal**: (1) How to design prompts for CLIP that are suitable for forgery detection; (2) How to enhance the CLIP vision encoder's sensitivity to forgery traces, such as noise anomalies and edge inconsistencies; (3) How to adapt CLIP to the forgery detection task while preserving its inherent generalization capability.

**Key Insight**: The authors propose modifying CLIP from two aspects: (1) On the textual side, design learnable dual-stream prompts (positive-negative prompt pairs) and adaptively adjust them according to the features and category of each image; (2) On the visual side, introduce a noise adapter to enhance the image encoder's forgery sensitivity, integrating multi-domain (spatial, frequency, and noise domains) features to capture forgery clues.

**Core Idea**: Through instance-aware dual-stream prompt learning, CLIP is guided to comprehend the "forgery vs. pristine" semantics, while the noise adapter enhances the vision encoder's forgery sensitivity. This achieves a model that boasts both generalization and precise forgery detection capabilities.

## Method

### Overall Architecture

Based on the pre-trained CLIP model, CLIP-IFDL consists of two main components: (1) Instance-aware Dual-stream Prompt Learning, which handles prompt design on the textual side of CLIP; (2) Forgery-Enhanced Noise Adapter, which enhances the forgery perception capability on the visual side of CLIP. Given an input image, the visual side extracts features containing noise information, while the textual side generates adaptive genuine/forgery prompts. The image-level detection result is obtained via text-image similarity computation, and the pixel-level localization result is acquired through feature upsampling.

### Key Designs

1. **Instance-aware Dual-stream Prompt Learning**:

    - **Function**: Generates adaptive prompts suitable for forgery detection for CLIP, replacing hand-crafted discrete prompts.
    - **Mechanism**: Rather than utilizing fixed text prompts (e.g., "a real photo" / "a forged photo"), a pair of learnable continuous prompt vectors—namely, positive prompts (corresponding to pristine images) and negative prompts (corresponding to forged images)—are created. These prompts are initialized as random vectors and learned during training. The core innovation lies in "instance awareness": the visual features and category information of each input image are mapped via a lightweight network into prompt modulation vectors, dynamically adjusting the prompts to suit the characteristics of the current image. Concurrently, prompt parameters are updated by constraining the similarity of the positive prompt to pristine images to be higher than that of the negative prompt.
    - **Design Motivation**: Fixed discrete prompts cannot accommodate the complexity of forgery detection (where different forgery types and different image contents require different judgment criteria). Instance-aware learnable prompts can adaptively generate optimal genuine/forged discrimination benchmarks for each individual image.

2. **Forgery-Enhanced Noise Adapter**:

    - **Function**: Enhances the CLIP vision encoder's perception of forgery artifacts.
    - **Mechanism**: A lightweight noise adapter is integrated alongside the CLIP vision encoder. This adapter first extracts multi-domain features from the input image: spatial domain (RGB pixel features), frequency domain (spectral features obtained via DCT or FFT), and noise domain (noise residual features extracted via high-pass filtering). These features are then integrated through a multi-domain fusion network. The fused features are injected into the intermediate layers of the CLIP vision encoder using zero-initialized linear layers. Zero initialization ensures that the adapter does not disrupt CLIP's pre-trained representations in the initial phase of training.
    - **Design Motivation**: The visual encoder of CLIP is trained for semantic understanding and is insensitive to pixel-level forgery traces (e.g., differences in JPEG compression artifacts, inconsistent noise distributions, edge anomalies). The noise adapter supplements CLIP with much-needed low-level forgery clues by introducing frequency and noise domain information.

3. **Text-Image Similarity Constraint**:

    - **Function**: Optimizes the prompt learning process by constraining the similarity between positive/negative prompts and the corresponding images.
    - **Mechanism**: For pristine images, the textual features of positive prompts should exhibit high similarity with the image features, while negative prompts should show low similarity. The inverse applies to forged images. A contrastive loss is deployed to maximize correct pair similarity and minimize incorrect pair similarity. This constraint ensures that the learnable prompts encode the semantic distinction between "pristine" and "forged".
    - **Design Motivation**: Unlike traditional binary classification training, optimizing prompts via similarity constraints preserves CLIP's text-image alignment capabilities, allowing the model to adapt to forgery discrimination while maintaining its open-world comprehension.

### Loss & Training

Training utilizes a multi-task loss function: (1) **Contrastive Loss**: constrains the positive/negative pairing similarity of text-image pairs; (2) **Detection Classification Loss**: binary cross-entropy for image-level genuine/forgery discrimination; (3) **Segmentation Loss**: pixel-level cross-entropy + Dice loss for locating forged regions. Regarding the training strategy, the main parameters of CLIP are frozen; only the learnable prompts, noise adapter, and zero-initialized linear layers are trained, resulting in a minimal parameter increase. This strategy protects CLIP's pre-trained knowledge and prevents catastrophic forgetting during fine-tuning.

## Key Experimental Results

### Main Results

| Dataset | Metric | CLIP-IFDL | Prev. SOTA | Description |
|--------|------|-----------|---------|------|
| CASIA v2 | AUC ↑ | Leading | MVSS-Net, SPAN | Splicing + Copy-move detection |
| Columbia | AUC ↑ | Leading | - | Splicing detection |
| Coverage | AUC ↑ | Leading | - | Copy-move detection |
| NIST16 | F1 ↑ | Leading | - | Mixed forgery types |
| Cross-dataset generalization | AUC ↑ | Significantly leading | - | Reflects CLIP's generalization capability |

### Ablation Study

| Configuration | AUC (Detection) | F1 (Localization) | Description |
|------|-----------|----------|------|
| Full CLIP-IFDL | Best | Best | Full model |
| w/o Noise Adapter | Significant decline | Significant decline | Lack of forgery sensitivity |
| w/o Instance Awareness | Moderate decline | Moderate decline | Fixed prompts are inferior to adaptive prompts |
| w/o Dual-stream Prompts (Single prompt) | Decline observed | Decline observed | Positive-negative contrast is superior |
| w/o Zero Initialization | Decline | Decline | Disrupted CLIP pre-trained representations |
| w/o Frequency Domain Features | Moderate decline | Moderate decline | Frequency domain information is crucial for detection |
| w/o Noise Domain Features | Moderate decline | Moderate decline | Noise residuals are key forgery cues |

### Key Findings
- The noise adapter is the most critical component. Removing it leads to a substantial decline in both detection and localization performance, highlighting the vital importance of enabling CLIP to perceive low-level forgery artifacts.
- In cross-dataset generalization experiments, CLIP-IFDL significantly outperforms traditional methods, validating the efficacy of leveraging CLIP's generalization capabilities.
- The zero-initialization strategy is critical for preserving CLIP's pre-trained knowledge; random initialization of injection points severely degrades performance.
- Multi-domain fusion (spatial + frequency + noise) yields superior results compared to any single-domain baseline, suggesting that forgery clues are distributed across different domains.
- The model also demonstrates a degree of generalization in detection of AI-generated images. Although it lags behind highly specialized methods, it surpasses traditional forgery detection approaches.

## Highlights & Insights
- The design of **zero-initialized linear layer injection** is highly elegant: at the beginning of training, the adapter's output is zero, which does not impact CLIP's original representation. It gradually introduces forgery-aware information as training progresses. This "gentle" injection method can be generalized to any scenario requiring the injection of new capabilities into pre-trained models.
- The concept of **multi-domain feature fusion** (RGB + frequency + noise) is highly effective in forgery detection, and can be extended to other tasks requiring the capture of pixel-level anomalies, such as deepfake detection and image quality assessment.
- Introducing CLIP's open-world generalization capability to forgery detection is a forward-looking direction—as AI-generated content explodes, there is an urgent need for detectors that can generalize to unseen manipulation types.

## Limitations & Future Work
- For high-quality forged images generated by state-of-the-art diffusion models, clues in the noise and frequency domains might be extremely faint, posing a greater detection challenge.
- The current method assumes the availability of pixel-level forgery annotations for training, which incurs high annotation costs. Future work could explore weakly supervised or unsupervised forgery localization.
- The multi-domain fusion in the noise adapter increases inference latency, which may require optimization for real-time detection scenarios, such as social media content moderation.
- The strategy of freezing CLIP, while protecting generalization, limits task-specific representation learning. Selectively unfreezing certain layers could be explored.
- The localization accuracy in the boundary regions of forged areas is not sufficiently precise; stronger edge-aware modules could be integrated.

## Related Work & Insights
- **vs. MVSS-Net**: MVSS-Net utilizes multi-scale and multi-view features for forgery detection but is trained from scratch; hence, its generalization is inferior to CLIP-based approaches.
- **vs. ObjectFormer**: ObjectFormer uses Transformers to model the relationship between forged regions and the background, but lacks the generalization advantage provided by pre-trained models.
- **vs. SPAN**: SPAN performs forgery localization via spatial pyramid attention but operates strictly in the spatial domain. In contrast, CLIP-IFDL's multi-domain fusion captures a broader range of forgery clues.
- **vs. CoOp/CoCoOp**: The CoOp series introduces prompt learning for CLIP. CLIP-IFDL extends this into a dual-stream format and integrates instance awareness, making it more suitable for forgery detection which inherently requires positive-negative comparisons.

## Rating
- Novelty: ⭐⭐⭐⭐ The approach of leveraging CLIP for forgery detection is novel, and the design combining the noise adapter with dual-stream prompts is creative.
- Experimental Thoroughness: ⭐⭐⭐⭐ Tested on multiple standard datasets; cross-dataset generalization experiments are robust, and ablation studies are comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Problem analysis is thorough, and the methodology is clearly described.
- Value: ⭐⭐⭐⭐ In an era flooded with AI-generated content, a highly generalizable forgery detector holds strong practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] One-stage Prompt-based Continual Learning](one-stage_prompt-based_continual_learning.md)
- [\[NeurIPS 2025\] ForensicHub: A Unified Benchmark & Codebase for All-Domain Fake Image Detection and Localization](../../NeurIPS2025/ai_safety/forensichub_a_unified_benchmark_codebase_for_all-domain_fake_image_detection_and.md)
- [\[CVPR 2026\] DiffusionFF: A Diffusion-based Framework for Joint Face Forgery Detection and Fine-Grained Artifact Localization](../../CVPR2026/ai_safety/diffusionff_a_diffusion-based_framework_for_joint_face_forgery_detection_and_fin.md)
- [\[ICML 2026\] HEDP: A Hybrid Energy-Distance Prompt-based Framework for Domain Incremental Learning](../../ICML2026/ai_safety/hedp_a_hybrid_energy-distance_prompt-based_framework_for_domain_incremental_lear.md)
- [\[ICML 2025\] Adaptive Multi-prompt Contrastive Network for Few-shot Out-of-distribution Detection](../../ICML2025/ai_safety/adaptive_multi-prompt_contrastive_network_for_few-shot_out-of-distribution_detec.md)

</div>

<!-- RELATED:END -->
