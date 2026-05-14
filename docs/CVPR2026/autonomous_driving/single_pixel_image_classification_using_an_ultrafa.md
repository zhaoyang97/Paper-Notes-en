---
title: >-
  [Paper Note] Single Pixel Image Classification using an Ultrafast Digital Light Projector
description: >-
  [CVPR 2026][Autonomous Driving][single pixel imaging] An ultrafast microLED-on-CMOS digital light projector (330 kfps global shutter) is employed for single-pixel imaging. Twelve-by-twelve Hadamard patterns are projected…
tags:
  - "CVPR 2026"
  - "Autonomous Driving"
  - "single pixel imaging"
  - "Hadamard patterns"
  - "microLED-on-CMOS"
  - "extreme learning machine"
  - "compressed sensing"
date: 2026-05-08
content_hash: d6f10fa6538d240f
---

# Single Pixel Image Classification using an Ultrafast Digital Light Projector

**Conference**: CVPR 2026
**arXiv**: [2603.12036](https://arxiv.org/abs/2603.12036)
**Code**: None
**Area**: Computational Imaging / Single-Pixel Imaging
**Keywords**: single pixel imaging, Hadamard patterns, microLED-on-CMOS, extreme learning machine, compressed sensing

## TL;DR

An ultrafast microLED-on-CMOS digital light projector (330 kfps global shutter) is employed for single-pixel imaging. Twelve-by-twelve Hadamard patterns are projected onto MNIST digits, and a single-pixel photodetector acquires a time series of aggregated light intensities. Image reconstruction is entirely bypassed; an ELM or DNN directly classifies the time series. The system achieves greater than 90% multi-class accuracy and greater than 99% AUC binary classification (anomaly detection) at 1.2 kfps.

## Background & Motivation

**Background**: Single-pixel imaging (SPI) replaces array sensors with structured illumination and a single-point detector, enabling simple hardware that operates across arbitrary spectral bands (infrared, THz, etc.). Conventional pattern generators based on DMDs are limited to approximately $10^4$ fps by mechanical tilting; recent microLED arrays improve switching speed by roughly two orders of magnitude.

**Limitations of Prior Work**:

1. Most single-pixel image classification (SPIC) studies rely on purely numerical simulation and lack validation on real optical systems.
2. The conventional SPI pipeline of reconstructing an image before classification introduces unnecessary latency, and reconstruction itself is a computational bottleneck.
3. The mechanical switching speed of DMDs constrains real-time applications (practical image generation rates $\lesssim 10^2$ Hz).

**Key Challenge**: SPI information acquisition is inherently a spatio-temporal transform (2D space → 1D time series). Whether the reconstruction step is truly necessary remains an open question.

**Goal**: Experimentally validate ultrafast SPIC in a real free-space optical system, completely bypassing image reconstruction.

**Key Insight**: Leverage the ultrafast switching capability of microLEDs to project Hadamard patterns and directly classify the photodetector time series.

**Core Idea**: Exploit a microLED ultrafast projector to achieve sub-millisecond Hadamard encoding, then directly classify the single-pixel detector time series without image reconstruction.

## Method

### Overall Architecture

A DMD displays binarized MNIST images → the microLED projector sequentially projects 288 Hadamard patterns (144 basis patterns of a $12\times12$ basis, each paired with its complement) → a single-pixel photodetector (SiPM) acquires the differential light intensity of each complementary pair → a real-time oscilloscope records the time series (286-dimensional feature vector) → an ELM or DNN directly classifies the series → the digit class (0–9) is output.

### Key Designs

1. **microLED-on-CMOS Ultrafast Light Projector**

    - $128\times128$ active-matrix microLED array, $30\times30\ \mu\text{m}^2$ pixels, $50\ \mu\text{m}$ pitch.
    - Supports binary mode and 5-bit grayscale; global shutter mode switches at 330 kfps.
    - Maps $12\times12$ Hadamard patterns onto the microLED array to illuminate the DMD.
    - Key advantage: approximately 30× faster than DMD mechanical tilting; projecting the complete 288-pattern set requires only approximately 0.87 ms.
    - The system bottleneck shifts from pattern generation to DMD object switching (32.5 kHz).

2. **Hadamard Pattern Compression and Ordering Strategy**

    - Had12 comprises 288 patterns (144 basis × complementary pairs), ordered by sequency (the spatial-frequency analogue).
    - Key finding: low-sequency patterns (fewer spatial sign changes) carry the most classification information.
    - Using the first half of patterns maintains approximately 85% accuracy; the first quarter yields approximately 78%, with a corresponding 2–4× throughput increase.
    - Three selection strategies are compared: first-$n$ (best) >> random selection (intermediate) >> last-$n$ (worst).
    - Analogy to Fourier analysis: low sequency ≈ low-frequency components, which suffice for coarse-grained classification.

3. **Two Lightweight Classification Models**

    - **ELM (Extreme Learning Machine)**: single hidden layer with randomly fixed input weights (not trained); the output weights are solved in closed form via ridge regression ($\alpha=1.0$). Achieves 87.37% multi-class accuracy with 1000 hidden neurons; inference takes 31 μs per sample. Core formula: $\beta = (H^\top H + \alpha I)^{-1} H^\top T$
    - **DNN**: three fully connected layers ($286\to\text{decreasing}\to10$) with ReLU and Softmax, Adam optimizer, 300 epochs. Achieves greater than 90% accuracy with the full Had12 set; inference takes 73 μs per sample.

### Loss & Training

- ELM: closed-form ridge regression solution, no iterative training, $\alpha=1.0$.
- DNN: sparse categorical cross-entropy + Adam, 300 epochs.
- Noise robustness: accuracy exceeds 95% under additive white Gaussian noise with $\sigma=0.5$; significant degradation occurs at $\sigma=1.0$. Performance degradation is primarily attributed to the loss of structural information rather than equivalent SNR variation.

## Key Experimental Results

### Main Results

| Configuration | Accuracy | Equivalent Frame Rate | Inference Time/Sample |
|---|---|---|---|
| Binary MNIST + DNN (simulation baseline) | 97.50% | — | — |
| Binary MNIST + ELM (simulation baseline) | 93.32% | — | — |
| Experimental Had12 full + DNN | >90% | 1.2 kHz | 73 μs |
| Experimental Had12 full + ELM (10-class) | 87.37% | 1.2 kHz | 31 μs |
| Experimental Had12 1/4 + DNN | ~78% | 4.8 kHz | — |
| Experimental Had12 + ELM (one-vs-all binary) | >99% AUC | 1.2 kHz | 31 μs |

### Ablation Study

**Effect of Pattern Selection Strategy on Classification Accuracy (DNN)**:

| Pattern Selection | Proportion | Equivalent Frame Rate | Approx. Accuracy |
|---|---|---|---|
| First-$n$ (low sequency) | 100% | 1.2 kHz | >90% |
| First-$n$ | 50% | 2.4 kHz | ~85% |
| First-$n$ | 25% | 4.8 kHz | ~78% |
| Random selection | 25% | 4.8 kHz | ~70% |
| Last-$n$ (high sequency) | 25% | 4.8 kHz | ~60% |

### Key Findings

- Low-sequency Hadamard patterns carry substantially more classification information than high-sequency patterns, analogous to the dominance of low-frequency components in FFT.
- DNN learning curves reveal a prolonged vanishing-gradient phase when fewer patterns are used, demonstrating that performance degradation stems from the loss of structural information rather than noise.
- The gap between ELM training and test accuracy is less than 1%, indicating no overfitting and confirming that the single-pixel encoded features possess sufficient discriminability.
- ELM binary classification AUC exceeds 99% across all classes, making it well-suited for go/no-go decisions (anomaly detection scenarios) in ultrafast pipelines.

## Highlights & Insights

- The "classify without reconstruction" paradigm deserves attention: 2D spatial information is encoded into a 1D time series for direct classification, with information preservation guaranteed by the Hadamard orthogonal basis.
- The frequency-ordered pattern selection strategy is simple yet effective: using only the first quarter of patterns trades a 4× throughput gain for approximately 12% accuracy reduction.
- The ELM anomaly detector is extremely lightweight: closed-form training, 31 μs inference, and AUC greater than 99%, making it suitable for embedded and edge deployment.
- This work constitutes the first experimental validation of kHz-level SPIC in a real free-space optical system, advancing the field beyond simulation.

## Limitations & Future Work

- Validation is limited to binarized $28\times28$ MNIST, which is far less complex than real machine-vision scenarios; performance on grayscale, color, or natural scene imagery remains unknown.
- The $12\times12$ Hadamard constraint originates from FPGA memory depth; practical applications require pattern sets of higher resolution.
- DMD object switching (32.5 kHz) remains the system bottleneck, leaving the 330 kfps advantage of the microLED underutilized.
- No direct comparison with event cameras is provided, despite claimed advantages over them.
- The experiments depend on a specific free-space optical path; engineering deployment and integration strategies are not discussed.

## Related Work & Insights

- **vs. Conventional SPI + Classification**: Prior SPIC work is predominantly simulation-based or relies on low-speed hardware; this paper is the first to experimentally validate kHz-level classification on an ultrafast optical system.
- **vs. microLED Analog Optical Computing**: Prior work uses microLEDs for analog optical neural networks (matrix–vector multiplication); this paper uses microLEDs for pattern projection followed by electronic post-processing—the two approaches are complementary.
- **vs. Event Cameras**: Both address high-speed perception, but SPI can operate across arbitrary spectral bands beyond visible light (infrared, THz), whereas event cameras are confined to the silicon sensor spectral range.
- Insight: The "sensing as computing" paradigm holds promise for edge and optical computing; the Hadamard compression strategy may inspire frame/token compression in video understanding.

## Rating

- Novelty: ⭐⭐⭐ The single-pixel classification concept is not original; the core contribution lies in hardware system integration and experimental validation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple pattern strategies, two classification models, noise analysis, and learning curve analysis are systematically evaluated.
- Writing Quality: ⭐⭐⭐⭐ Clear and readable, with detailed experimental setup and optical path descriptions; figures and tables are intuitive.
- Value: ⭐⭐⭐ An interesting system integration effort, though the gap between MNIST validation and real-world application remains substantial.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] SearchAD: Large-Scale Rare Image Retrieval Dataset for Autonomous Driving](searchad_large-scale_rare_image_retrieval_dataset_for_autonomous_driving.md)
- [\[CVPR 2026\] OneOcc: Semantic Occupancy Prediction for Legged Robots with a Single Panoramic Camera](oneocc_semantic_occupancy_prediction_for_legged_robots_with_a_single_panoramic_c.md)
- [\[CVPR 2026\] Learning Vision-Language-Action World Models for Autonomous Driving](vla_world_learning_vision_language_action_world_models_for_autonomous_driving.md)
- [\[CVPR 2026\] Efficient Equivariant Transformer for Self-Driving Agent Modeling](efficient_equivariant_transformer_for_self-driving_agent_modeling.md)
- [\[CVPR 2026\] Learning Mutual View Information Graph for Adaptive Adversarial Collaborative Perception](learning_mutual_view_information_graph_for_adaptive_adversarial_collaborative_pe.md)

</div>

<!-- RELATED:END -->
