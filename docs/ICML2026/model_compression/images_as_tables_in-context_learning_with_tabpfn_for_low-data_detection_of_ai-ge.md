---
title: >-
  [Paper Note] Images as Tables: In-Context Learning with TabPFN for Low-Data Detection of AI-Generated Images
description: >-
  [ICML 2026][Model Compression][TabPFN] A three-stage pipeline is proposed for AI-generated image detection: first, images are compressed into 768-dimensional CLS vectors using a frozen DINOv3; second, these are reduced to 500 dimensions via PCA to serve as tabular rows; finally, TabPFN performs in-context inference. This shifts "retraining classification he
tags:
  - ICML 2026
  - Model Compression
  - TabPFN
  - DINOv3
date: 2026-05-08
content_hash: 3b75b7a550a55bab
---
# Images as Tables: In-Context Learning with TabPFN for Low-Data Detection of AI-Generated Images

**Conference**: ICML 2026  
**arXiv**: [2606.00872](https://arxiv.org/abs/2606.00872)  
**Code**: https://github.com/jpwalter30/Towards-Generalizable-Detection-of-AI-Generated-Images  
**Area**: AI-Generated Content Detection / Tabular Foundation Models / In-Context Learning  
**Keywords**: AIGC Detection, TabPFN, DINOv3, In-Context Learning, Cross-Generator Transfer  

## TL;DR
A three-stage pipeline is proposed for AI-generated image detection: first, images are compressed into 768-dimensional CLS vectors using a frozen DINOv3; second, these are reduced to 500 dimensions via PCA to serve as tabular rows; finally, TabPFN performs in-context inference. This shifts "retraining classification heads for new generators" to "replacing TabPFN context samples." In GenImage low-data and cross-generator scenarios, this method outperforms the strong baseline LATTE by up to 8.2% and wins in 54 out of 64 generator transfer pairs.

## Background & Motivation

**Background**: AI-generated image detection is consistently proven to be a "moving target" problem. Detectors trained on specific generators (GAN or Diffusion) often fail when encountering new versions like Midjourney, Stable Diffusion, or Wukong. Current mainstream solutions rely on image-domain classifiers—training a real/fake head on CNN/ViT backbones, or utilizing specialized detection with diffusion denoising trajectories (LATTE), CLIP representations (Cozzolino et al.), and frequency/fingerprint enhancements. These methods perform strongly in large-scale i.i.d. scenarios, and cross-generator evaluation has become a standard requirement.

**Limitations of Prior Work**: (i) Adapting to new generators still requires "replacing heads or retraining," necessitating either training classifiers from scratch or fine-tuning backbones, which is maintenance-unfriendly. (ii) In real forensic scenarios, available labels for new generators are often limited to dozens or hundreds, far below the scale required to train ViT classification heads. (iii) Almost all existing detectors couple "representation learning" with "discriminative learning," requiring full network updates for new data, which is slow and prone to overfitting generator-specific fingerprints.

**Key Challenge**: Detection capability stems primarily from strong visual representations, while adaptation speed is limited by the training paradigm of discriminative heads. Coupling the two means a full deep model gradient pass is required even for minor label updates.

**Goal**: (1) Validate "Image → Table → In-Context Inference" as a new paradigm for AIGC detection. (2) Systematically compare performance in low-data and cross-generator scenarios across four evaluation protocols on GenImage. (3) Provide a head-to-head comparison with the state-of-the-art diffusion detector LATTE to define the trade-off boundaries between TabPFN-ICL and image classification heads across different data scales.

**Key Insight**: The authors observe that TabPFN (Prior-Data Fitted Network) can perform Bayesian-style inference on small tabular data without training, relying solely on context samples. If each image is compressed into a structured feature row, AIGC detection is reduced to a standard small-data tabular classification problem, where TabPFN’s low-data advantage aligns with the "few labels" constraint in forensic scenarios.

**Core Idea**: A frozen DINOv3 ViT-B/16 acts as the visual encoder. PCA reduces the 768-dimensional CLS token to 500 dimensions (matching the feature limit of the current TabPFN). TabPFN then performs in-context binary classification for each test image using a set of labeled "context rows." Adapting to a new generator requires only updating a small number of labeled samples in the context, while both the encoder and classifier remain static.

## Method

### Overall Architecture
DINOv3-PCA-TabPFN addresses the pain point of retraining classification heads by decomposing detection into a three-stage conversion chain with near-zero training. A frozen visual model compresses each image into a vector; PCA reduces it to a 500-dimensional structured "tabular record"; finally, the universal tabular foundation model TabPFN predicts real/fake via a single forward pass without gradient updates. Effectively, an image becomes a labeled row in a table, reducing AIGC detection to standard small-data tabular classification.

Specifically, images are loaded in RGB, resized to 256 pixels on the short side, center-cropped to $224\times 224$, and normalized using ImageNet statistics. They are then fed into DINOv3 ViT-B/16 in eval mode to extract the CLS token, forming an $N\times 768$ feature matrix (backbone gradients remain frozen). Incremental PCA is fitted on training features to project the 768 dimensions into a structured row $z(h(x))\in\mathbb{R}^{500}$, applying the same components to the test set. Finally, the reduced training rows (including 0/1 labels) and the test row are fed into TabPFN for a single-forward-pass posterior prediction. The current TabPFN version is restricted to a maximum of 10,000 rows and 500 dimensions. The essence of this pipeline is that "adaptation = replacing context rows": for a new generator, one only needs to re-extract PCA coefficients on a few samples and assembly the context set, avoiding any deep model gradients.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["Input Image<br/>RGB → Short-side 256 → Center-crop 224×224 → ImageNet Norm"] --> B["Frozen DINOv3 Encoder<br/>ViT-B/16 eval CLS → 768d (Gradients Frozen)"]
    B --> C["PCA to 500d<br/>Fit IncrementalPCA on Train Features → Tabular Record z(h(x))"]
    C --> D["TabPFN In-Context Inference<br/>Labeled Context Rows + Test Row Single Forward"]
    D --> E["Output Real/Fake Prediction"]
    F["Few samples from new generator"] -->|Adaptation = Re-fit PCA + Re-assemble Context (Zero Gradient)| C
```

### Key Designs

**1. Frozen DINOv3 as Universal Visual Encoder: Decoupling Representation from Judgment**
Forensic literature indicates that end-to-end trained classification heads easily overfit to specific generator fingerprints. The authors respond by offloading representation entirely to self-supervised pre-training. DINOv3 ViT-B/16 CLS tokens serve as global representations while the backbone remains in eval mode without real/fake fine-tuning. Overfitting risks are thus shifted to the TabPFN context. Even if a new generator differs significantly from the training distribution, semantic/texture representations remain valid, and TabPFN only requires context samples to define new decision boundaries. Ablations confirm this encoder's necessity: replacing DINOv3 with DINOv2 or manual frequency features (DCT/FFT) results in lower accuracy, precision, recall, F1, and AUC.

**2. PCA to 500 Dimensions: Aligning with TabPFN Limits and Removing Irrelevant Variance**
DINOv3 CLS tokens are 768-dimensional, while the current TabPFN has a 500-dimension limit. Incremental PCA is used to fit a projection on training features, compressing each image to $z(h(x))\in\mathbb{R}^{500}$. During evaluation across four generator-aware protocols (Multi-Multi, Multi-Single, Single-Multi, Single-Single), PCA is fitted only on the training side to prevent data leakage. The cost of "re-fitting PCA coefficients" is included in the adaptation budget—a step orders of magnitude faster than retraining a classification head, ensuring TabPFN input dimensions remain valid while keeping in-context inference truly gradient-free.

**3. TabPFN ICL instead of Gradient-based Heads: Replacing "Training" with "Table Replacement"**
TabPFN is a Prior-Data Fitted Network that outputs classification posteriors via a single forward pass given a "context set + test row." This approximates Bayesian tabular inference rather than traditional training. The paper leverages this for small data: each generator uses a minimal training size $k\in\{25,30,75,150,300,625\}$ as the context, while the test set is fixed at 10,000 images. In small-data regimes, the advantage of "replacing context" over "retraining a head" is massive. Adding a new generator becomes a lightweight feature extraction and context assembly process, reducing setup time from minutes to seconds.

### Loss & Training
No training is performed on the visual side. Both DINOv3 and TabPFN are frozen models, and PCA is a closed-form fit. "Training" is equivalent to "preparing a new context table," which is the core advantage of this paradigm. Baselines like LATTE are trained normally according to their protocols and compared fairly at $k\in\{150,300,625\}$.

## Key Experimental Results

### Main Results
Benchmark is GenImage (ImageNet real images + 8 generators: ADM/BigGAN/GLIDE/Midjourney/SDv1.4/SDv1.5/VQDM/Wukong). Protocols include Multi-Multi (pooled train/test), Multi-Single (pooled train/single test), Single-Multi (single train/pooled test), and Single-Single (pairwise transfer). Accuracy is the primary metric.

| Protocol | Training Scale $k$ | LATTE | DINOv3-PCA-TabPFN | Gain |
|------|--------------|-------|-------------------|------|
| Multi-Multi Pooled Low-Data | $k=25$ | — | **78%** | Start at 78 |
| Multi-Multi Pooled Medium | Small Shared $k$ | Behind | TabPFN leads by up to +8.2% | Strongest Region |
| Multi-Multi Pooled High-Data | $k=625$ | **+7.4% Lead** | Behind | LATTE recovers at scale |
| Single-Single Pairwise | $k=625$ | — | Wins in 54/64 pairs | Max pair +31.5% |

### Ablation Study
| Configuration | Key Metric | Description |
|------|---------|------|
| Full: DINOv3 + PCA-500 + TabPFN | Best in 5 metrics (Multi-Multi) | Full Pipeline |
| Encoder: DINOv2 + TabPFN | Weaker performance | DINOv3's unique contribution |
| Encoder: DCT/FFT Frequency + TabPFN | Significant drop | Semantic > Handcrafted Frequency |
| Head: MLP on DINOv3 features | Equal at scale, behind at low $k$ | TabPFN advantage in small context |
| Low Data $k=25$ (Pooled) | 78% | Outperforms all MLP-based training |

### Key Findings
- The combination of strong representations and in-context judgment is most advantageous in low-data and cross-generator regimes. DINOv3 and TabPFN are both necessary—replacing either (DINOv2 / MLP / Frequency) leads to performance drops.
- Under the Single-Single (64 pairs) protocol, DINOv3-PCA-TabPFN outperforms LATTE in 54 pairs (max lead 31.5%). However, as $k$ increases, some cross-pair transfer accuracy decreases, suggesting TabPFN may slightly specialize to generator fingerprints as context density grows.
- Difficulty varies significantly by generator: BigGAN/GLIDE show high separability in PCA space, while ADM/Midjourney/Wukong distributions heavily overlap in DINOv3 space, requiring more context for slower gains.
- LATTE overtakes by 7.4% at $k=625$ in pooled settings, primarily due to TabPFN's 10,000-row limit. Once the context is saturated, TabPFN has no room for further improvement, making it naturally non-comparable with sustainable training models like LATTE in large-data regimes.

## Highlights & Insights
- The study proposes the "Images as Tables" paradigm by end-to-end coupling a universal visual foundation model with a universal tabular foundation model. Once an image is converted to a structured row, all small-data tabular tools (not just TabPFN) become available for AIGC detection.
- The cost of adapting to new generators is reduced to PCA calculation and context re-assembly. The pipeline is gradient-free, bringing the engineering burden of "training new heads" to near zero, which is highly beneficial for real-world forensics.
- The paper clearly defines the boundaries of advantage: choose TabPFN for low-data/cross-generator tasks and LATTE for large-data pooled scenarios. It avoids claiming "absolute superiority," leaving room for hybrid detection or switching strategies based on data scale.
- The two-stage abstraction (frozen encoding followed by tabular foundation model) is potentially portable to other low-data visual problems like medical or remote sensing small-shot classification.

## Limitations & Future Work
- The 10,000 row × 500 dimension limit of the current TabPFN limits context size, preventing it from processing the entire GenImage pooled collection. This is why LATTE catches up at scale. Future versions like TabPFN-2.5 might shift this boundary.
- The pipeline relies heavily on DINOv3 representation quality. If generators begin to utilize adversarial techniques against DINOv3, the pipeline may lack additional image-domain robustness (e.g., frequency-domain defense).
- Experiments are limited to binary real/fake classification on GenImage; robustness against degradations (compression, blur, crop) was not evaluated and is cited as future work. Additionally, PCA is a linear reduction; future work could explore stronger non-linear projections.

## Related Work & Insights
- **vs LATTE (2025)**: LATTE uses diffusion denoising latent trajectories as signals, peaking in large-data pooled scenarios. Ours uses no diffusion-specific signals, relying on general DINOv3 + TabPFN to lead in low-data/cross-generator tasks without retraining heads.
- **vs CLIP-based Detection (Cozzolino 2024)**: Both use pre-trained visual representations, but CLIP detectors still train classification heads i.i.d., incurring high migration costs. Ours converts judgment to in-context inference, removing "training" entirely.
- **vs TabPFN Original**: TabPFN was designed for small structured datasets. This work moves it out of its native domain by treating DINOv3 representations as structured rows, providing a blueprint for cross-modal structured inference (e.g., THz/medical imaging).
- **vs Frequency/Fingerprint Detection (Yu 2019, Cozzolino 2024)**: Fingerprint detection often fails across generators. This work avoids fingerprint dependency via DINOv3 self-supervised representations and shifts cross-generator transfer pressure to TabPFN's context set, resulting in wins in 54 out of 64 transfer pairs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Towards Generalizable AI-Generated Image Detection via Image-Adaptive Prompt Learning](../../CVPR2026/model_compression/towards_generalizable_ai-generated_image_detection_via_image-adaptive_prompt_lea.md)
- [\[NeurIPS 2025\] AI-Generated Video Detection via Perceptual Straightening](../../NeurIPS2025/model_compression/ai-generated_video_detection_via_perceptual_straightening.md)
- [\[ICML 2026\] Easier to Judge Than to Find: Predicting In-Context Learning Success for Demonstration Selection](easier_to_judge_than_to_find_predicting_in-context_learning_success_for_demonstr.md)
- [\[ICML 2026\] Energy-Structured Low-Rank Adaptation for Continual Learning](energy-structured_low-rank_adaptation_for_continual_learning.md)
- [\[ECCV 2024\] SpaceJAM: a Lightweight and Regularization-free Method for Fast Joint Alignment of Images](../../ECCV2024/model_compression/spacejam_a_lightweight_and_regularization-free_method_for_fast_joint_alignment_o.md)

</div>

<!-- RELATED:END -->
