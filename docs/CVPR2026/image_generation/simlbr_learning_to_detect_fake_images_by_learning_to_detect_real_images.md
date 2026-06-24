---
title: >-
  [Paper Note] SimLBR: Learning to Detect Fake Images by Learning to Detect Real Images
description: >-
  [CVPR 2026][Image Generation][Fake Image Detection] This paper proposes SimLBR, which uses Latent-space Blending Regularization (LBR) to mix sparse fake image information into real image embeddings within the DINOv3 latent space. This forces the detector to learn a compact decision boundary around the real image distribution, achieving strong generalization to unseen generators. It reaches an average accuracy of 94.54% on GenImage and improves accuracy by 25% and recall by 70…
tags:
  - "CVPR 2026"
  - "Image Generation"
  - "Fake Image Detection"
  - "AI-Generated Images"
  - "Latent-space Blending Regularization"
  - "Cross-Generator Generalization"
  - "DINOv3"
date: 2026-05-08
content_hash: 85e3ca8042c20272
---

# SimLBR: Learning to Detect Fake Images by Learning to Detect Real Images

**Conference**: CVPR 2026  
**arXiv**: [2602.20412](https://arxiv.org/abs/2602.20412)  
**Code**: Yes (To be released on HuggingFace and GitHub)  
**Area**: Image Generation  
**Keywords**: Fake Image Detection, AI-Generated Images, Latent-space Blending Regularization, Cross-Generator Generalization, DINOv3

## TL;DR
This paper proposes SimLBR, which uses Latent-space Blending Regularization (LBR) to mix sparse fake image information into real image embeddings within the DINOv3 latent space. This forces the detector to learn a compact decision boundary around the real image distribution, achieving strong generalization to unseen generators. It reaches an average accuracy of 94.54% on GenImage and improves accuracy by 25% and recall by 70% over AIDE on the difficult Chameleon test set.

## Background & Motivation

**Background**: AI-generated image detection is a critical media forensics task. Existing methods typically train on fake images from one generator, expecting to generalize to others. UnivFD uses CLIP features with nearest-neighbor classification, while AIDE combines DCT and CLIP features.

**Limitations of Prior Work**: Nearly all SOTA detectors overfit to generator-specific fingerprints of the training data. Performance drops catastrophically when encountering unseen generators. On the Chameleon hard test set proposed in the AIDE paper, all existing methods fail severely—fake images are misclassified as real (recall as low as 5%), indicating these methods actually learn compact boundaries around fake images and treat "real" as a sink class.

**Key Challenge**: The distribution of fake images is constantly evolving with new generators. Learning boundaries around fake images inevitably misses out-of-distribution (OOD) fake images. Conversely, the distribution of real images is relatively stable; any image that is not "real" should be classified as fake.

**Goal**: To enable the detector to learn a compact decision boundary around the real image distribution, treating "fake" as the sink class to achieve generalization across arbitrary unseen generators.

**Key Insight**: Making the classification of real images "harder" through Latent-space Blending Regularization (LBR) during training. By mixing small amounts of fake image information into real image embeddings and labeling them as fake, the model is forced to classify only completely uncontaminated real images as "real."

**Core Idea**: Perform simple linear interpolation between real and fake image embeddings in the DINOv3 semantic latent space, teaching the model to draw a tight circle around the pure real image distribution.

## Method

### Overall Architecture

SimLBR addresses a core problem: ensuring the detector learns "what a real image is" rather than the "fingerprints left by training generators." The pipeline is minimalist, freezing all images into DINOv3 embeddings and training a lightweight MLP for binary classification. The core innovation lies in the generation of regularized samples during training.

The process consists of three steps. First is **Embedding Pre-computation**: a frozen DINOv3 is used to compute embedding vectors for all training images—real image set $\mathbb{R}=\{R_1,...,R_n\}$ and fake image set $\mathbb{F}=\{F_1,...,F_n\}$ from a single training generator $G_s$. Next is **LBR Training**: for each real image $R_i$, a label $y_i \in \{0,1\}$ is randomly assigned. If $y_i=0$, the pure real embedding $L_i=I(R_i)$ is used. If $y_i=1$, fake information is mixed in and the label is changed to fake. Finally, **Lightweight MLP Classification**: a 2-layer ReLU MLP with BCE loss is sufficient. Training takes approximately 3 minutes on a single H100 (excluding pre-computation, which takes ~22 mins for GenImage and ~50 mins for AIGC), whereas AIDE requires ~2 hours on 8 A100 GPUs.

### Key Designs

**1. Latent-space Blending Regularization (LBR): Forcing a compact boundary around real images**

Existing SOTA methods fail on Chameleon because they treat "real" as a sink class, allowing OOD fake images to leak into the real category. LBR reverses this: during training, a subset of real images (label $y_i=1$) is paired with a fake image $F_i$. Their embeddings $L_i^R=I(R_i)$ and $L_i^F=I(F_i)$ are linearly interpolated:

$$L_i = \alpha \cdot L_i^R + (1-\alpha) \cdot L_i^F$$

This "slightly contaminated real image" is labeled as fake. The mixing coefficient is sampled from $\alpha \sim \text{Uniform}(0.5, B)$ (where $B=0.8$). $\alpha>0.5$ ensures the result retains most real-image information, yet must be classified as fake. If $\alpha$ is too low (0.1–0.3), classification is too easy; if too high (0.95–0.99), it mimics pure real images too closely, causing training instability. This mechanism forces the model to define "real" only as perfectly pure real images, creating a tight boundary. During inference, any sample falling outside this boundary is naturally classified as fake regardless of the generator.

**2. Blending in Semantic Latent Space vs. Pixel Space**

Methods like Face X-ray blend in pixel space, which has three drawbacks: pixel blending copies low-level artifacts (e.g., frequency features) directly, allowing the model to "shortcut" by focusing on artifacts instead of the real distribution; it requires task-specific preprocessing; and it offers coarse control over information injection. SimLBR interpolates in the high-level semantic space of DINOv3: low-level cues are abstracted away, forcing semantic-level distinction; blending is a simple linear operation; and the injection ratio can be precisely tuned to create infinite regularized samples.

**3. DINOv3 Manifold Geometry as a Prerequisite**

LBR assumes that the linear path between two embeddings remains within the valid manifold. Ablations confirm this: adding LBR to DINOv3 improves average accuracy from 78.05% to 88.26% (+10.21%), while on DINOv2, accuracy slightly drops from 68.92% to 68.51%. DINOv3's embedding manifold is smooth enough to sustain linear interpolation, whereas DINOv2's structure is less robust. This suggests the method's effectiveness is tied to the manifold quality of the backbone.

**4. Reliability Score and Worst-Case Estimate**

To evaluate stability beyond average accuracy, the authors introduce a Reliability Score inspired by the Sharpe ratio:

$$\text{Reliability} = \frac{\mu_{\text{acc}} - A_{\text{base}}}{\sigma_{\text{acc}}}$$

where $\mu_{\text{acc}}$ is the average accuracy, $\sigma_{\text{acc}}$ is the standard deviation, and $A_{\text{base}}=50\%$. A higher score indicates results that are both accurate and stable. Additionally, a Worst-Case Estimate (WCE)—the minimum accuracy across all generators—is used to represent the performance floor for unknown future generators.

### Loss & Training

Standard Binary Cross-Entropy is used:

$$\mathcal{L}_{\text{BCE}} = -\frac{1}{N}\sum_{i=1}^{N} [y_i \log(F_\theta(L_i)) + (1-y_i)\log(1-F_\theta(L_i))]$$

The classification head is a 2-layer ReLU MLP. Training uses a batch size of 500, Adam (lr=1e-4, wd=1e-2), and a maximum of 10 epochs.

## Key Experimental Results

### Main Results 1: GenImage Dataset (Trained on SD v1.4)

| Method | MidJ | SD1.5 | ADM | GLIDE | Wukong | VQDM | BigGAN | Mean↑ | Std↓ | Rel.↑ |
|------|------|-------|-----|-------|--------|------|--------|-------|------|-------|
| UnivFD | 73.2 | 84.0 | 55.2 | 76.9 | 75.6 | 56.9 | 80.3 | 73.29 | 11.32 | 2.05 |
| PatchCraft | 79.0 | 89.3 | 77.3 | 78.4 | 89.3 | 83.7 | 72.4 | 82.30 | 6.56 | 4.29 |
| AIDE | 79.4 | 99.8 | 78.5 | 91.8 | 98.7 | 80.3 | 66.9 | 86.88 | 12.33 | 2.99 |
| **SimLBR** | **91.7** | **98.1** | **97.0** | **92.4** | **97.4** | **93.5** | **88.0** | **94.54** | **3.74** | **11.91** |

SimLBR achieves the highest accuracy on 5 out of 7 generators, outperforming AIDE by **21%** on the OOD GAN model BigGAN. The reliability score is 4x higher than AIDE.

### Main Results 2: Chameleon Hard Test Set

| Method | ProGAN Total Acc | ProGAN Fake/Real Acc | SD Total Acc | SD Fake/Real Acc |
|------|-----------------|------------------------|-------------|---------------------|
| AIDE | 58.37 | 5.04 / 98.46 | 62.60 | 20.33 / 94.38 |
| **SimLBR** | **84.33** | **75.80 / 90.74** | **85.57** | **92.96 / 80.06** |

On Chameleon, SimLBR's fake image recall surges from AIDE's 5.04% to 75.80% (ProGAN training), with a total accuracy gain of 25%.

### Ablation Study

| Backbone | LBR | Chameleon | AIGC | GenImage | RSFake | Mean↑ | Std↓ | Rel.↑ | WCE↑ |
|----------|-----|----------|------|----------|--------|-------|------|-------|------|
| DINOv2 | ✗ | 56.12 | 70.77 | 86.74 | - | 68.92 | 13.31 | 1.42 | 56.12 |
| DINOv2 | ✓ | 55.77 | 74.75 | 81.49 | - | 68.51 | 11.71 | 1.57 | 55.77 |
| DINOv3 | ✗ | 59.65 | 76.70 | 92.19 | 73.90 | 78.05 | 12.77 | 2.19 | 59.65 |
| DINOv3 | ✓ | **84.33** | **88.40** | **94.54** | **88.88** | **88.26** | **3.94** | **9.70** | **84.33** |

### Key Findings
- LBR provides a +10.21% gain on DINOv3 but is ineffective on DINOv2, highlighting the importance of the embedding space geometry.
- Recall improvement is significant: Chameleon-ProGAN (7.24% → 75.80%), AIGC (54.07% → 82.07%), and RSFake (48.28% → 78.87%).
- MLP depth: 0–4 layers perform well; >4 layers lead to overfitting.
- Upper bound $B$ for $\alpha$ is stable within the 0.7–0.9 range.
- Worst-Case: SimLBR maintains ~88% and ~75.5% on GenImage and AIGC, respectively, far exceeding the ~50% baseline of other methods.

## Highlights & Insights
- **Paradigm Shift**: The transition to "detecting real images instead of fake images" is insightful. Modeling a stable real distribution rather than chasing evolving fake distributions is a significant methodological contribution.
- **Simplicity**: LBR achieves strong regularization through simple linear interpolation in latent space, requiring only 3 minutes of training.
- **Reliability Score**: Borrowing the Sharpe ratio from finance provides a more holistic view of detector robustness, applicable across classification tasks.
- **Visualization**: t-SNE shows a qualitative shift: without LBR, OOD fake images cluster with real images; with LBR, real images form a tight cluster, treating any deviation as fake.

## Limitations & Future Work
- Strong dependency on DINOv3 embedding quality; results were ineffective on DINOv2. Future backbones require re-validation.
- LBR uses only linear interpolation; more complex manifold mixing strategies (e.g., slerp) could be explored.
- The assumption of a "stable" real-image distribution might be challenged by evolving camera sensors or post-processing pipelines.
- Limited MLP capacity might be vulnerable to sophisticated adversarial attacks.

## Related Work & Insights
- **vs UnivFD**: Both use pre-trained feature spaces, but UnivFD relies on nearest neighbors while SimLBR uses LBR to regularize an MLP, explicitly learning real boundaries.
- **vs AIDE**: AIDE uses DCT+CLIP features and requires intensive GPU hours (8x A100 for 2h). SimLBR uses DINOv3 and a single GPU for 3 minutes, outperforming AIDE on hard test sets. This suggests feature space selection is more vital than model complexity.
- **vs Pixel Blending**: Pixel-space methods are domain-specific and leak low-level artifacts. Latent-space blending is domain-agnostic and forces semantic-level distinction.
- **vs Anomaly Detection**: Pure anomaly detection (modeling only real images) is impractical in high-dimensional natural image spaces. SimLBR uses fake images as guidance to build effective decision boundaries.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The "detecting real" paradigm shift and LBR design are simple yet deep.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extensive testing across four benchmarks plus WCE and detailed ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation and well-defined reliability metrics.
- Value: ⭐⭐⭐⭐⭐ Low training cost and high generalization make it suitable for real-world deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Co-Spy: Combining Semantic and Pixel Features to Detect Synthetic Images by AI](../../CVPR2025/image_generation/co-spy_combining_semantic_and_pixel_features_to_detect_synthetic_images_by_ai.md)
- [\[CVPR 2026\] Align Images Before You Generate](align_images_before_you_generate.md)
- [\[CVPR 2026\] Refracting Reality: Generating Images with Realistic Transparent Objects](refracting_reality_generating_images_with_realistic_transparent_objects.md)
- [\[CVPR 2026\] Group Editing: Edit Multiple Images in One Go](group_editing_edit_multiple_images_in_one_go.md)
- [\[CVPR 2026\] RewardFlow: Generate Images by Optimizing What You Reward](rewardflow_generate_images_by_optimizing_what_you_reward.md)

</div>

<!-- RELATED:END -->
