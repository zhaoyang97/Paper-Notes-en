---
title: >-
  [Paper Note] Images as Tables: In-Context Learning with TabPFN for Low-Data Detection of AI-Generated Images
description: >-
  [ICML 2026][Model Compression][TabPFN] The authors reformulate AI-generated image detection into a three-stage pipeline: first, a frozen DINOv3 compresses each image into a 768-dimensional CLS vector; next, PCA reduces this to 500 dimensions to serve as a single row in a table; finally, TabPFN performs in-context inference. This approach transforms the need
tags:
  - ICML 2026
  - Model Compression
  - TabPFN
  - DINOv3
date: 2026-05-08
content_hash: 3917f8b1f2ccaae2
---
# Images as Tables: In-Context Learning with TabPFN for Low-Data Detection of AI-Generated Images

**Conference**: ICML 2026  
**arXiv**: [2606.00872](https://arxiv.org/abs/2606.00872)  
**Code**: https://github.com/jpwalter30/Towards-Generalizable-Detection-of-AI-Generated-Images  
**Area**: AI-Generated Image Detection / Tabular Foundation Models / In-Context Learning  
**Keywords**: AIGC Detection, TabPFN, DINOv3, In-Context Learning, Cross-Generator Transfer  

## TL;DR
The authors reformulate AI-generated image detection into a three-stage pipeline: first, a frozen DINOv3 compresses each image into a 768-dimensional CLS vector; next, PCA reduces this to 500 dimensions to serve as a single row in a table; finally, TabPFN performs in-context inference. This approach transforms the need to "retrain classification heads for new generators" into simply "replacing context samples in TabPFN." In low-data and cross-generator scenarios on GenImage, this method leads the strong baseline LATTE by up to 8.2% and wins in 54 out of 64 generator transfer pairs.

## Background & Motivation

**Background**: AI-generated image detection has been repeatedly proven to be a "moving target" problem: detectors trained well on one generator (GAN or Diffusion) often fail when faced with new versions like Midjourney, Stable Diffusion, or Wukong. Current mainstream solutions remain image-domain classifiers—training a real/fake head on CNN/ViT backbones, or utilizing diffusion denoising trajectories (LATTE) and CLIP representations (Cozzolino et al.) as specialized detectors, supplemented by frequency-domain or fingerprint features. While these methods perform strongly in large-scale i.i.d. scenarios, cross-generator evaluation has become the standard benchmark.

**Limitations of Prior Work**: (i) Adapting to new generators still requires "changing heads or retraining," either by training a classifier from scratch or fine-tuning the backbone on new data, which is operationally inefficient; (ii) in real-world forensic scenarios, the number of available labels for a new generator is often as few as dozens or hundreds, far below the scale required to train a ViT classification head; (iii) almost all existing detectors couple "representation learning" with "discriminative learning," requiring the entire network to be updated when new data arrives, which is slow and prone to overfitting to generator-specific fingerprints.

**Key Challenge**: Detection capability primarily stems from strong visual representations, while adaptation speed is limited by the training paradigm of the discriminative head. Coupling the two means that even small batches of new labels necessitate gradient updates across a deep model.

**Goal**: (1) Validate "Image → Table → In-Context Inference" as a new paradigm for AIGC detection; (2) systematically compare low-data and cross-generator performance under four assessment protocols on GenImage; (3) conduct a head-to-head comparison with the current SOTA diffusion detector, LATTE, to illustrate the trade-off boundaries between TabPFN-context and image-only classification heads across different data scales.

**Key Insight**: The authors observe that TabPFN, a Prior-Data Fitted Network, can perform Bayesian-style inference on small tabular datasets without training, relying solely on context samples. If each image is compressed into a row of structured features, AIGC detection reduces to a standard small-data tabular classification problem, where TabPFN’s low-data advantage aligns perfectly with the "few labels" constraint of forensic scenarios.

**Core Idea**: Freeze DINOv3 ViT-B/16 as the visual encoder, use PCA to compress the 768-dimensional CLS token to 500 dimensions (matching the 500-feature limit of current TabPFN), and then use TabPFN with a set of labeled "context rows" to perform in-context binary classification for each test image. Adapting to a new generator only requires updating a few labeled samples in the context, leaving both the encoder and the classifier unchanged.

## Method

### Overall Architecture
The DINOv3-PCA-TabPFN framework addresses the pain point of "retraining classification heads for new generators" by decomposing detection into a three-stage, nearly zero-training transformation chain: a frozen visual model compresses each image into a vector, PCA reduces it to a 500-dimensional "tabular record," and the general-purpose tabular foundation model TabPFN performs real/fake determination via a single forward pass without gradient updates. Effectively, an image is transformed into a labeled tabular row, reducing AIGC detection to standard small-data tabular classification.

Specifically, images are loaded in RGB, resized to a 256-pixel short side, center-cropped to $224\times 224$, and normalized using ImageNet statistics. They are then fed into DINOv3 ViT-B/16 in eval mode to extract a CLS token, resulting in an $N\times 768$ feature matrix (the backbone remains frozen). Incremental PCA is fitted on the training set features to compress the 768 dimensions into a structured row $z(h(x))\in\mathbb{R}^{500}$, and the same principal components are applied to the test set. Finally, the reduced training rows (with 0/1 labels) and the test row are fed into TabPFN, which outputs a prediction through a single forward pass. This version of TabPFN is limited to 10,000 rows and 500 dimensions. The essence of the pipeline is that "adaptation equals changing context rows": for a new generator, one only needs to re-extract PCA coefficients and update rows in the TabPFN context set, requiring no gradient updates for either the encoder or the discriminator.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["Input Image<br/>RGB → Resize 256 → Center Crop 224×224 → ImageNet Norm"] --> B["Frozen DINOv3 Encoder<br/>ViT-B/16 CLS → 768-d (Frozen)"]
    B --> C["PCA to 500-d<br/>Fit IncrementalPCA on Train → Tabular Row z(h(x))"]
    C --> D["TabPFN In-Context Inference<br/>Labeled Context Rows + Test Row Forward Pass"]
    D --> E["Output Real/Fake Prediction"]
    F["Small Sample Set from New Generator"] -->|Adaptation = Re-fit PCA + Assemble Context (Zero Gradient)| C
```

### Key Designs

**1. Frozen DINOv3 as Universal Visual Encoder: Decoupling Representation and Discrimination**
Forensic literature has long noted that end-to-end trained classification heads are prone to overfitting to specific generator fingerprints. The authors address this by delegating representation entirely to self-supervised pre-training. By using the CLS token from DINOv3 ViT-B/16 as the image representation and keeping the backbone in eval mode without real/fake fine-tuning, the risk of overfitting is shifted to the TabPFN context. Even if a new generator differs significantly from the training distribution, the semantic/textual representations from the encoder remain valid; one only needs to provide context samples to define the decision boundary for TabPFN. Ablations confirm that replacing DINOv3 with DINOv2 or frequency-domain features (DCT/FFT) results in lower accuracy, precision, recall, F1, and AUC.

**2. PCA to 500 Dimensions: Adhering to TabPFN Limits and Removing Irrelevant Variance**
DINOv3 CLS tokens are 768-dimensional, while the current TabPFN version has a 500-dimension limit. The authors use IncrementalPCA to fit a closed-form projection on training features, compressing each image to $z(h(x))\in\mathbb{R}^{500}$. During evaluation across four generator-aware protocols (Multi-Multi, Multi-Single, Single-Multi, Single-Single), PCA is always fitted only on the training side to prevent information leakage. Crucially, "re-fitting PCA coefficients" is considered a negligible cost of adaptation—this step is orders of magnitude faster than training a classification head and ensures TabPFN's input dimensions are always valid, enabling "truly zero-gradient" in-context inference.

**3. TabPFN In-Context Inference as a Substitute for Gradient-Based Training: Replacing "Training a New Head" with "Updating a Context Table"**
TabPFN is a Prior-Data Fitted Network that outputs posterior classifications using a single forward pass given a "context set + test row." This is essentially a Bayesian tabular inference rather than traditional training. The authors leverage this for small-data scenarios: each generator uses a minimal training set size $k\in\{25,30,75,150,300,625\}$ as context, with a test set of 10,000 images. In these small-data regimes, the advantage of "swapping context" over "retraining a head" is significant. Operationally, adding a new generator becomes a lightweight feature extraction and context assembly task, reducing adaptation time from minutes to seconds.

### Loss & Training
No training is performed on the vision side: DINOv3 and TabPFN are both frozen models, and PCA is a closed-form fit. "Training" is equivalent to "preparing a new context table"—the core advantage of this paradigm. As a baseline, LATTE is trained normally according to its protocols to provide a fair comparison at $k\in\{150,300,625\}$.

## Key Experimental Results

### Main Results
The benchmark is GenImage (ImageNet real images + eight generators including ADM, BigGAN, GLIDE, Midjourney, SDv1.4, SDv1.5, VQDM, and Wukong). Evaluations follow four protocols: Multi-Multi (pooled train/test), Multi-Single (pooled train/single test), Single-Multi (single train/pooled test), and Single-Single (pair-wise transfer).

| Protocol | Training Scale $k$ | LATTE | DINOv3-PCA-TabPFN | Gain |
|------|--------------|-------|-------------------|------|
| Multi-Multi Low Data | $k=25$ | — | **78%** | Start at 78% |
| Multi-Multi Medium | Small shared $k$ | Behind | TabPFN leads by up to +8.2% | Ours' strongest regime |
| Multi-Multi High Data | $k=625$ | Leads by +7.4% | Behind | LATTE recovers at scale |
| Single-Single Transfer | $k=625$ | — | Wins in 54/64 pairs | Max pair gain +31.5% |

### Ablation Study

| Configuration | Key Metrics | Note |
|------|---------|------|
| Full: DINOv3 + PCA-500 + TabPFN | Best across 5 metrics in Multi-Multi | Complete pipeline |
| Encoder replaced with DINOv2 + TabPFN | Consistently weaker | Validates DINOv3 contribution |
| Encoder replaced with DCT/FFT + TabPFN | Significant drop | Validates semantic over frequency features |
| Classifier replaced with MLP on DINOv3 | Matches at high data, lags at low | Validates TabPFN small-context advantage |
| Low Data $k=25$ (Pooled) | 78% | TabPFN starts higher than any MLP method |

### Key Findings
- The combination of strong representation and in-context discrimination excels in low-data and cross-generator regimes. Both DINOv3 and TabPFN are necessary; replacing either (with DINOv2, MLP, or frequency features) degrades performance.
- In the Single-Single protocol (64 pairs), DINOv3-PCA-TabPFN outperforms LATTE in 54 pairs (up to 31.5% lead). However, as $k$ increases, some cross-pair transfer accuracy decreases, suggesting TabPFN may slightly specialize to generator fingerprints as the context set grows.
- Difficulty varies significantly by generator: BigGAN/GLIDE show high real/fake separability in PCA space, requiring little context. ADM/Midjourney/Wukong distributions overlap heavily in DINOv3 space, requiring more context for slower gains.
- LATTE leads by 7.4% in the $k=625$ pooled setting, primarily due to the 10,000-row limit of the current TabPFN. This means once the context is "full," TabPFN hits a plateau, whereas LATTE can continue to improve with more training data.

## Highlights & Insights
- Proposes an "Images as Tables" paradigm by end-to-end concatenation of a universal vision foundation model and a universal tabular foundation model. Once an image is a structured row, any tabular tool (not just TabPFN) can be applied to AIGC detection.
- Compresses the cost of adapting to new generators into two steps: re-fitting PCA and re-assembling the context set. The pipeline involves zero gradients, reducing the engineering burden of "training new heads" to near zero.
- Clearly defines the boundary between TabPFN and LATTE: TabPFN for low-data/cross-generator tasks, and LATTE for large-scale pooled scenarios. This leaves room for future hybrid or scale-switch strategies.
- The two-stage abstraction (frozen-encode then tabular-infer) has direct portability beyond forensics to other low-data vision problems like medical or remote sensing few-shot classification.

## Limitations & Future Work
- The current TabPFN limit (10,000 rows × 500 dimensions) prevents it from consuming the full GenImage pooled set, which is why LATTE catches up at scale. Newer versions like TabPFN-2.5 might shift this boundary.
- The pipeline relies heavily on DINOv3 representation quality. If future generators target DINOv3 specifically with adversarial methods, the pipeline lacks image-domain defenses (e.g., frequency-domain buffers).
- Experiments are restricted to real/fake binary classification on GenImage without evaluating robustness to degradations (compression, blur, cropping). Additionally, PCA is a linear reduction; replacing it with non-linear projections might capture more complex signals.

## Related Work & Insights
- **vs. LATTE (2025)**: LATTE uses diffusion denoising trajectories as signals, peaking in large-data pooled scenarios. This method uses no diffusion-specific signals and outperforms LATTE in low-data/cross-generator tasks without retraining heads.
- **vs. CLIP-based Detection (Cozzolino 2024)**: Both use foundation model representations, but CLIP detectors still train a classification head under i.i.d. assumptions, incurring high transfer costs. This method eliminates training in favor of in-context inference.
- **vs. Vanilla TabPFN**: While TabPFN was designed for small structured tabular data, this work applies it to visual representations as structured rows, providing a template for cross-modal structured inference (e.g., medical imaging).
- **vs. Frequency/Fingerprint Detection (Yu 2019, Cozzolino 2024)**: Fingerprint-based methods fail when generators change. This method avoids fingerprint dependency through DINOv3 semantic representations and shifts transfer pressure to the TabPFN context, as evidenced by winning in 54/64 transfer pairs.

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
