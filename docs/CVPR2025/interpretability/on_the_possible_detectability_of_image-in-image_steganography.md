---
title: >-
  [Paper Note] On the Possible Detectability of Image-in-Image Steganography
description: >-
  [CVPR 2025][Interpretability][Image Steganography] This work theoretically and experimentally reveals that current popular deep learning-based image-in-image steganography schemes suffer from severe detectability vulnerabilities. The embedding process is essentially a mixing process that can be easily identified by Independent Component Analysis (ICA). An 8-dimensional feature vector consisting of the first four moments of independent components in the wavelet domain achieves…
tags:
  - "CVPR 2025"
  - "Interpretability"
  - "Image Steganography"
  - "Steganalysis"
  - "Independent Component Analysis"
  - "Wavelet Decomposition"
  - "Deep Steganographic Networks"
date: 2026-05-08
content_hash: 58da37ef481898c4
---

# On the Possible Detectability of Image-in-Image Steganography

**Conference**: CVPR 2025  
**arXiv**: [2603.11876](https://arxiv.org/abs/2603.11876)  
**Code**: None  
**Area**: Interpretability  
**Keywords**: Image Steganography, Steganalysis, Independent Component Analysis, Wavelet Decomposition, Deep Steganographic Networks

## TL;DR
This work theoretically and experimentally reveals that current popular deep learning-based image-in-image steganography schemes suffer from severe detectability vulnerabilities. The embedding process is essentially a mixing process that can be easily identified by Independent Component Analysis (ICA). An 8-dimensional feature vector consisting of the first four moments of independent components in the wavelet domain achieves a detection accuracy of 84.6%, while the classic SRM+SVM method achieves over 99% accuracy.

## Background & Motivation

**Background**: In recent years, deep learning-based steganography has made significant progress. Among these, the image-in-image paradigm is particularly popular—utilizing an encoder-decoder network to embed a full-sized secret image into a cover image of the same size. This achieves an extremely high payload capacity (1 bit per pixel or even more), and the resulting stego image is visually almost indistinguishable from the original cover image. This class of methods includes works such as HiDDeN, SteganoGAN, DeepStego, etc.

**Limitations of Prior Work**: Although image-in-image steganography schemes perform exceptionally well in visual quality, their security evaluation is highly inadequate. Most papers only use PSNR/SSIM to measure the quality of stego images while overlooking a fundamental question: are these schemes truly secure at the statistical level? The extremely high embedding rate implies significant changes in the distribution of the cover image, and whether these changes can be detected by classic or novel steganalysis methods remains unexplored.

**Key Challenge**: Image-in-image steganography schemes pursue extremely high payload capacity (hiding a whole image), but classic steganography security theory indicates that higher embedding rates lead to easier detection. These deep learning schemes reach visual imperceptibility through end-to-end training but do not guarantee statistical security.

**Goal**: (1) To explain why image-in-image steganography is easily detectable from the theoretical perspective of mixing processes; (2) To propose a simple, interpretable steganalysis method based on ICA; (3) To evaluate the detection performance of classic steganalysis methods against deep steganography.

**Key Insight**: The authors observe that the embedding process of image-in-image steganography is essentially a mixture of the cover signal and the secret signal (similar to signal mixing in blind source separation problems). This implies that classic signal separation methods, such as ICA, can effectively decompose the mixed signal back into its original components, thereby exposing the steganographic trace.

**Core Idea**: Utilizing ICA to estimate the independent components of the stego image after wavelet decomposition, and distinguishing clean images from stego images by comparing the statistical distributions (the first four moments) of the independent components.

## Method

### Overall Architecture
The method consists of three steps: (1) performing wavelet decomposition on the image under test to obtain multi-scale, multi-directional subband coefficients; (2) applying Independent Component Analysis (ICA) to the wavelet coefficients to estimate independent components; (3) calculating the first four moments (mean, variance, skewness, kurtosis) of the independent components to construct an 8-dimensional feature vector, which is then input to a classifier to distinguish between cover and stego images.

### Key Designs

1. **Wavelet-domain ICA**:

    - **Function**: To decompose the mixing process of steganographic embedding into analysable independent components.
    - **Mechanism**: The embedding process of image-in-image steganography can be formulated as $S = E(C, M)$, where $C$ is the cover image, $M$ is the secret image, and $S$ is the stego image. The embedding-matrix learned by the deep learning encoder $E$ is locally approximated as a linear mixture. After applying wavelet decomposition on $S$, the ICA algorithm (e.g., FastICA) is used in the wavelet domain to estimate the "independent components" prior to mixing. For clean images, the distribution of independent components of wavelet coefficients closely resembles the statistical characteristics of original natural images, whereas for stego images, the mixing process significantly alters the distribution of independent components.
    - **Design Motivation**: The wavelet domain provides a multi-scale signal representation where steganographic embedding is particularly pronounced in the high-frequency subbands. ICA is naturally suited for handling such signal mixing problems, as steganography is essentially mixing a secret signal into a cover signal.

2. **Fourth-order Moment Feature Extraction**:

    - **Function**: To extract compact and discriminative statistical features from independent components.
    - **Mechanism**: Calculating the first four moments—mean (1st order), variance (2nd order), skewness (3rd order), and kurtosis (4th order)—for the two primary independent components, yielding an $2 \times 4 = 8$-dimensional feature vector. Wavelet coefficients of natural images typically present a highly super-Gaussian distribution (high kurtosis, near-zero skewness) after ICA, while steganographic embedding shifts this distribution towards a Gaussian shape (reducing kurtosis and increasing skewness variation), since according to the Central Limit Theorem, mixing processes tend to make independent components closer to a Gaussian distribution.
    - **Design Motivation**: High-order moments are highly sensitive to changes in the shape of the distribution, which can effectively capture the statistical shift introduced by steganographic embedding. The 8-dimensional feature vector is extremely compact, allowing classifiers (such as SVM, LDA) to be trained efficiently while maintaining interpretability, as each dimension has a clear physical meaning.

3. **Classic Steganalysis Comparison (SRM+SVM)**:

    - **Function**: To verify the detection capability of traditional steganalysis methods against deep steganography.
    - **Mechanism**: Using the classic Spatial Rich Model (SRM) to extract high-dimensional steganographic features (30,000+ dimensions) combined with an SVM classifier. SRM uses a large collection of high-pass filters to extract residual features, which are highly sensitive to subtle statistical variations introduced by steganography. Experimental results demonstrate that SRM+SVM achieves over 99% accuracy in detecting image-in-image steganography.
    - **Design Motivation**: The comparison demonstrates that image-in-image steganography schemes do not break through the scope of traditional steganalysis. Although they are visually imperceptible, they are statistically extremely vulnerable. This finding serves as an important warning to the steganography community.

### Loss & Training
This work does not involve model training but rather analyzes the security of existing steganographic schemes. The ICA method requires no training (unsupervised), and the classification classifier uses a simple SVM or LDA trained on the extracted features. The SRM+SVM baseline is trained according to the standard steganalysis pipeline.

## Key Experimental Results

### Main Results

| Detection Method | Feature Dimension | SteganoGAN | HiDDeN | DeepStego | Average Detection Rate |
|---------|---------|-----------|--------|-----------|-----------|
| ICA + 4th Moments | 8 | 82.3% | 84.6% | 81.2% | ~82.7% |
| SRM + SVM | 30000+ | 99.2% | 99.5% | 99.1% | ~99.3% |
| Random Guess | - | 50% | 50% | 50% | 50% |

### Ablation Study

| Moments Order Used | Feature Dimension | Detection Accuracy | Description |
|------------|---------|-----------|------|
| Mean + Variance Only | 4 | ~72% | Insufficient low-order moment information |
| First Three Moments | 6 | ~79% | Skewness provides extra information |
| First Four Moments | 8 | ~84.6% | Kurtosis contributes the greatest improvement |
| Keyless Extraction Analysis | - | 100% Extraction | Keyless design is itself a vulnerability |

### Key Findings
- **Kurtosis change is the strongest signal**: The most prominent statistical trace of steganographic embedding is the decrease in the kurtosis of independent components (shifting from super-Gaussian to Gaussian), which aligns with the prediction of the Central Limit Theorem.
- **Only 8-dimensional features** are required to reach an accuracy of 84.6%, proving that the statistical variations introduced by image-in-image steganography are highly prominent.
- **99%+ detection rate of SRM** indicates that deep learning steganography schemes are almost completely transparent to classic steganalysis, showing that they have not truly resolved the issue of statistical security.
- **Keyless extraction** is another critical vulnerability: anyone who obtains the extraction network can reconstruct the secret image from the stego image, which directly contradicts the security assumptions of traditional cryptography.

## Highlights & Insights
- **Theoretical Clarity**: The detectability of steganography is explained from the perspective of mixing processes and ICA. The theoretical framework is simple and powerful without relying on "black-box" deep learning.
- **Efficiency of the Minimalist Method**: The detection method based on 8-dimensional feature vectors offers high interpretability and extremely low computational cost, making it highly suitable for practical deployment scenarios.
- **Fundamental Questioning of Deep Steganography**: The core finding of this work is that although image-in-image steganography schemes are imperceptible at the pixel level, they leave significant traces in deeper statistical features, which offers key methodological insights for research across this entire direction.

## Limitations & Future Work
- The number of evaluated steganographic schemes is limited (mainly 3-5 types); newly emerging schemes (such as diffusion-based steganography) might possess different characteristics.
- Only the scenario of embedding a full-sized image is considered, lacking analysis for low-payload capacity or adaptive embedding.
- The ICA method assumes that the embedding process is approximately linear, and its efficacy might degrade for highly non-linear deep encoders.
- Future directions: (1) designing statistically secure deep steganography schemes (e.g., adding steganalysis imperceptibility constraints into training loss); (2) studying the adversarial game between steganalysis and steganography; (3) extending the analysis to video and audio steganography.

## Related Work & Insights
- **vs HiDDeN/SteganoGAN**: These deep steganography schemes evaluate security solely using PSNR/SSIM. This work proves that this is far from sufficient—visual imperceptibility does not equate to statistical security.
- **vs Traditional Steganography (e.g., LSB)**: Traditional methods have low payload capacity but focus more on statistical security. Ironically, high-capacity deep steganography is even easier to detect than low-capacity traditional methods.
- **vs SRM Steganalysis**: SRM is the gold standard of classic steganalysis. This work reveals that it is equally effective against deep steganography, proving that deep learning steganography does not truly break through the boundaries of traditional steganalysis.
- **vs Invertible Neural Network Steganography**: Recent steganography schemes based on invertible neural networks (e.g., INN) claim to be more secure. The analytical framework of this paper is equally applicable to evaluating the statistical security of such methods.

## Rating
- Novelty: ⭐⭐⭐⭐ Analyzing the detectability of deep steganography from the ICA perspective, offering novel and inspiring theoretical viewpoints.
- Experimental Thoroughness: ⭐⭐⭐ Cover multiple steganographic schemes and analysis methods, but the scale of experiments and dataset diversity could be further extended.
- Writing Quality: ⭐⭐⭐⭐ Clear argumentative logic, with smooth progression from theory to experiments.
- Value: ⭐⭐⭐⭐ Presents important security warnings to the deep steganography community, and the 8-dimensional feature method possesses practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Sample- and Parameter-Efficient Auto-Regressive Image Models](sample-_and_parameter-efficient_auto-regressive_image_models.md)
- [\[CVPR 2025\] Interpretable Image Classification via Non-parametric Part Prototype Learning](interpretable_image_classification_via_non-parametric_part_prototype_learning.md)
- [\[CVPR 2025\] Why Does It Look There? Structured Explanations for Image Classification](why_does_it_look_there_structured_explanations_for_image_classification.md)
- [\[ACL 2025\] EXPERT: An Explainable Image Captioning Evaluation Metric with Structured Explanations](../../ACL2025/interpretability/expert_an_explainable_image_captioning_evaluation_metric_with_structured_explana.md)
- [\[ICML 2026\] Vision-Language Asymmetry in Bistable Image Captioning](../../ICML2026/interpretability/vision-language_asymmetry_in_bistable_image_captioning.md)

</div>

<!-- RELATED:END -->
