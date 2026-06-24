---
title: >-
  [Paper Note] Image Reconstruction from Readout-Multiplexed Single-Photon Detector Arrays
description: >-
  [CVPR 2025][Single-photon detector] This paper formalizes the multiphoton coincidence resolution problem in row-column readout-multiplexed single-photon detector arrays as an inverse imaging problem. It proposes a probabilistic Multiphoton Estimator (ME) capable of resolving the spatial locations of up to 4 concurrent incident photons. Compared to traditional methods, ME achieves a 3-4 dB PSNR improvement on a 32×32 array and reduces the required frame count by approximately…
tags:
  - "CVPR 2025"
  - "Single-photon detector"
  - "readout multiplexing"
  - "image reconstruction"
  - "multiphoton estimation"
  - "superconducting nanowire"
date: 2026-05-08
content_hash: 8053cc026d031bee
---

# Image Reconstruction from Readout-Multiplexed Single-Photon Detector Arrays

**Conference**: CVPR 2025  
**arXiv**: [2312.02971](https://arxiv.org/abs/2312.02971)  
**Code**: None  
**Area**: Computational Imaging / Object Detection  
**Keywords**: Single-photon detector, readout multiplexing, image reconstruction, multiphoton estimation, superconducting nanowire

## TL;DR
This paper formalizes the multiphoton coincidence resolution problem in row-column readout-multiplexed single-photon detector arrays as an inverse imaging problem. It proposes a probabilistic Multiphoton Estimator (ME) capable of resolving the spatial locations of up to 4 concurrent incident photons. Compared to traditional methods, ME achieves a 3-4 dB PSNR improvement on a 32×32 array and reduces the required frame count by approximately 4 times.

## Background & Motivation

1. **Background**: Single-photon detectors (such as SPADs and SNSPDs) are widely utilized in biological imaging, LiDAR, and quantum optics. Realizing large-area imaging requires single-photon detector arrays. Row-column multiplexed readout is a key solution to address the cooling requirements and data bandwidth bottlenecks in large-scale SNSPD arrays, capable of reducing the readout lines of an $n \times n$ array from $n^2$ to $2n$.

2. **Limitations of Prior Work**: Row-column multiplexed readouts require very low incident photon flux to avoid spatial ambiguity. When multiple photons arrive at the array simultaneously within a single acquisition cycle, the row-column readout only provides a set of candidate pixels, leaving the exact hit locations undetermined. Traditional Naive Estimators distribute photon counts uniformly across all candidate pixels, resulting in ghosting artifacts. The Single-Photon Estimator (SPE) discards multiphoton frames entirely, which, although unbiased, suffers from extremely high variance and wastes approximately 85% of measured frames.

3. **Key Challenge**: How to accurately resolve the spatial incident locations of each photon while utilizing the information from multiphoton frames? This is an ill-posed inverse problem.

4. **Goal**: To utilize information in multiphoton frames to reduce reconstruction mean squared error (MSE), improve image quality, and increase photon utility under a row-column multiplexed readout architecture.

5. **Key Insight**: Starting from probabilistic modeling, the authors leverage unambiguous single-photon frames to estimate the detection probability of each pixel, and then use these probabilities in turn to distribute photon counts back from the multiphoton frames.

6. **Core Idea**: By modeling the conditional probabilities of multiphoton coincidence events, the photon counts in ambiguous readout frames are redistributed among candidate pixels following the maximum likelihood principle, enabling approximate maximum likelihood image reconstruction.

## Method

### Overall Architecture
The input is a sequence of row-column readout frames $(R^t, C^t)$, each containing a row vector and a column vector indicating which rows and columns detected photons. The output is the estimated incident photon flux $\hat{\Lambda}_{ij}$ for each pixel. The proposed method consists of three steps: (1) splitting the readout frames into unambiguous and ambiguous frames; (2) estimating initial detection probabilities using unambiguous frames; (3) redistributing the photon counts from ambiguous frames using conditional probability models to secure an improved estimate of the detection probabilities.

### Key Designs

1. **Measurement and Readout Model**:

    - **Function**: Establish the mathematical formulation relating photon incidence, detection, and row-column readout.
    - **Mechanism**: The photon arrivals at each pixel follow a Poisson distribution $X_{ij}^t \sim \text{Poisson}(\Lambda_{ij})$. Due to detector saturation, the actual detection is modeled as a Bernoulli random variable $Y_{ij}^t \sim \text{Bernoulli}(q_{ij})$ with $q_{ij} = 1 - e^{-\Lambda_{ij}}$. The row readout $R_i^t$ is the logical OR operation of all pixel detections in row $i$, and the column readout is similar. A frame is unambiguous if only one row and one column are activated.
    - **Design Motivation**: This formulation casts the imaging problem as an inverse problem of recovering pixel-level detection probabilities from row-column compressed measurements.

2. **Multiphoton Estimator (ME)**:

    - **Function**: Utilize the information in ambiguous frames to enhance reconstruction accuracy.
    - **Mechanism**: For ambiguous frames, the initial detection probability is first estimated using the single-photon estimator $\hat{q}_{ij}^s$. Then, the conditional probabilities $\hat{g}_k$ for various multiphoton coincidence events (i.e., the relative likelihood of each event occurring given the observed ambiguous readout) are computed. Using these conditional probabilities, the $M_9$ ambiguous frames are proportionally distributed as $\hat{g}_k M_9$ among each event type. This decomposes the intractable sum-form likelihood function into a product-form approximation, enabling an approximate closed-form maximum likelihood estimate.
    - **Design Motivation**: Directly maximizing the likelihood of ambiguous frames is challenging due to the sum of multiple event probabilities. By first estimating conditional probabilities and then redistributing the counts, the problem is simplified into an analytically solvable form.

3. **Progressive 2/3/4-Photon Extension**:

    - **Function**: Stepwise increase the modeled number of coincident photons to improve estimation accuracy.
    - **Mechanism**: Modeling begins with 2-photon coincidences and progressively incorporates 3-photon and 4-photon events. For an $n \times n$ array, the types of ambiguity increase with the number of activated rows and columns. To maintain computational tractability, the modeling is restricted to at most 4-photon coincidences, ignoring 5-photon and higher-order events. With each additional level of modeling, the optimal operating point (the PPF that minimizes the MSE) shifts toward higher incident flux.
    - **Design Motivation**: Higher-order coincidence modeling leverages more data frames at the cost of expanding terms in the conditional probability expressions. 4-photon modeling strikes an optimal balance between precision and complexity.

### Loss & Training
This work does not involve deep learning training, but is rather a purely computational/statistical approach. The key optimization target is maximizing the approximate joint likelihood function $\mathcal{L} = U(q) \cdot A(q)$, where $U(q)$ represents the contribution from unambiguous frames and $A(q)$ from ambiguous frames. An analytical solution is obtained via conditional probability decomposition, eliminating the need for iterative optimization.

## Key Experimental Results

### Main Results
Monte Carlo simulations on a 32×32 array with 100,000 measurement frames:

| Test Image | Estimator | PSNR (dB) | Optimal PPF |
|----------|--------|-----------|---------|
| Lily | Naive | 16.39 | ~0.45 |
| Lily | SPE | 21.55 | ~0.83 |
| Lily | ME (Ours) | **27.22** | **~1.4** |
| Moon | Naive | 17.05 | - |
| Moon | SPE | 22.43 | - |
| Moon | ME (Ours) | **28.41** | - |
| Coins | Naive | 13.17 | - |
| Coins | SPE | 16.63 | - |
| Coins | ME (Ours) | **22.54** | - |

### Ablation Study

| Estimator Configuration | Optimal PPF | PSNR at Optimal MSE | Description |
|-----------|---------|---------------|------|
| Naive | 0.45 | 22.59 dB | Simplest, high bias |
| Single-Photon | 0.83 | 23.64 dB | Unbiased but high variance |
| 2-photon ME | 1.0 | 23.83 dB | Models 2-photon coincidences |
| 3-photon ME | 1.2 | 24.60 dB | Adds 3-photon events |
| 4-photon ME | **1.4** | **26.74 dB** | Full model, optimal |

### Key Findings
- The 4-photon estimator improves PSNR by approximately 3-4 dB over the Single-Photon Estimator (SPE) and by 6-11 dB over the Naive Estimator.
- The optimal incident photon flux (1.4 PPF) for ME is higher than that of SPE (0.83) and NE (0.45), meaning it can operate under higher signal intensities.
- ME achieves approximately a 4x reduction in required frames (frames needed to reach the same MSE: ME 25k vs. SPE 100k).
- ME performs well against the Cramér-Rao Bound (CRB), matching the theoretical lower bound over a wide flux range.
- Performance gains increase with larger array sizes, indicating the method's scalability to commercial-scale arrays.

## Highlights & Insights
- **Conditional Probability Allocation Concept**: Converts the intractable mixed likelihood problem into an analytically solvable allocation problem, delivering a simple and elegant solution. This "rough estimation followed by fine-grained allocation" strategy can be generalized to other compressed sensing or multiplexed scenarios.
- **No Spatial Prior Required**: The authors deliberately avoid spatial priors (such as total variation regularization) to isolate the exact source of performance gains. This suggests that incorporating priors can yield further performance improvements.
- **Optimal Flux Operating Point Analysis**: Reveals distinct optimal operating points across different estimators, offering direct system design guidance—specifically, relaxing the strict constraints on low photon flux.

## Limitations & Future Work
- Limited to modeling at most 4-photon coincidences, which introduces bias due to model mismatch under high-flux conditions.
- Experiments are restricted to Monte Carlo simulations with a lack of validation on real physical SNSPD hardware.
- Non-idealities of physical detectors, such as dark counts and crosstalk, are not modeled or considered.
- Computational complexity scales linearly with array size; the authors suggest block-wise parallel processing as a mitigation strategy.
- Deep learning methods could be combined to learn spatial priors for further enhancement of image reconstruction quality.

## Related Work & Insights
- **vs. Compressed Sensing Methods** (multiplexed readouts in PET/gamma-ray imaging): Those approaches typically focus on optical or electrical system modeling, whereas this work is the first to explicitly study the combinatorics of single-photon events to establish a probabilistic model.
- **vs. van den Berg et al.**: They use group testing-inspired surface codes for deterministic coincidence resolution, while this work allows for low-probability errors and adopts a probabilistic approach, making it more suitable for intensity-imaging scenarios.
- The proposed row-column multiplexed probabilistic modeling framework can be generalized to other multiplexed readout architectures and detector models.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Formulates multiphoton coincidence resolution as an inverse imaging problem for the first time and provides analytical solutions.
- **Experimental Thoroughness**: ⭐⭐⭐ Simulation is extensive, but validation on real hardware is missing.
- **Writing Quality**: ⭐⭐⭐⭐ Clear mathematical derivations and intuitive diagrams.
- **Value**: ⭐⭐⭐⭐ Significantly advances the practical application of large-scale single-photon detector arrays.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Towards In-the-Wild 3D Plane Reconstruction from a Single Image](towards_in-the-wild_3d_plane_reconstruction_from_a_single_image.md)
- [\[CVPR 2025\] Rooftop Wind Field Reconstruction Using Sparse Sensors: From Deterministic to Generative Learning Methods](rooftop_wind_field_reconstruction_using_sparse_sensors_from_deterministic_to_gen.md)
- [\[CVPR 2026\] Inter-Photon-Limited Videography](../../CVPR2026/others/inter-photon-limited_videography.md)
- [\[ACL 2025\] Intuitive Fine-Tuning: Towards Simplifying Alignment into a Single Process](../../ACL2025/others/intuitive_fine_tuning.md)
- [\[ACL 2025\] Instruction-Tuning Data Synthesis from Scratch via Web Reconstruction](../../ACL2025/others/instruction-tuning_data_synthesis_from_scratch_via_web_reconstruction.md)

</div>

<!-- RELATED:END -->
