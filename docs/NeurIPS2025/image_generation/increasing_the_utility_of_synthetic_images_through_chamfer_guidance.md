---
title: >-
  [Paper Note] Increasing the Utility of Synthetic Images through Chamfer Guidance
description: >-
  [NeurIPS 2025][Image Generation][Chamfer Distance] This paper proposes Chamfer Guidance — a training-free inference-time guidance method that uses a small number of real samples as references. By leveraging Chamfer distance, it simultaneously optimizes fidelity and diversity of synthetic images. On ImageNet-1k, using only 32 real images, it achieves 97.5% Precision and 92.7% Coverage, and delivers up to 16% accuracy improvement in downstream classifier training.
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "Chamfer Distance"
  - "diffusion model guidance"
  - "synthetic training data"
  - "image diversity"
  - "distribution matching"
  - "training-free inference guidance"
date: 2026-05-08
content_hash: f3acac7284798089
---

# Increasing the Utility of Synthetic Images through Chamfer Guidance

**Conference**: NeurIPS 2025
**arXiv**: [2508.10631](https://arxiv.org/abs/2508.10631)  
**Area**: Image Generation / Synthetic Data
**Keywords**: Chamfer Distance, diffusion model guidance, synthetic training data, image diversity, distribution matching, training-free inference guidance

## TL;DR

This paper proposes Chamfer Guidance — a training-free inference-time guidance method that uses a small number of real samples as references. By leveraging Chamfer distance, it simultaneously optimizes fidelity and diversity of synthetic images. On ImageNet-1k, using only 32 real images, it achieves 97.5% Precision and 92.7% Coverage, and delivers up to 16% accuracy improvement in downstream classifier training.

## Background & Motivation

### The Fidelity–Diversity Dilemma in Synthetic Data

Conditional image generation models (e.g., Stable Diffusion) are capable of producing highly realistic images and have naturally been considered as a source of synthetic training data for downstream models. However, research reveals that as models scale up and become more aligned with human preferences, **image quality improves while diversity significantly declines**, severely limiting the practical utility of synthetic data.

### Limitations of Prior Work

**Training-based methods** (fine-tuning, ReFL, etc.): Require additional training cost, and experiments show that increasing the number of real samples does not consistently improve performance.

**Reference-free guidance methods** (CFG, APG, CADS, etc.): Adjust the quality/diversity trade-off during sampling without referencing the target data distribution, resulting in limited diversity gains and potentially generating "ungrounded diversity."

**c-VSG**: Although it introduces a small number of real images as references, its Vendi Score-based metric does not scale effectively with the number of reference samples, and requires balancing two hyperparameters.

### Core Insight

The authors identify the key issue: **diversity should be defined relative to the target data distribution, not as an absolute quantity of variation**. By importing the classical Chamfer distance from 3D vision into image distribution matching, fidelity and diversity are naturally unified within a single distance metric — the forward term encourages coverage of all real modes (diversity), while the backward term encourages each generated sample to remain faithful to the real distribution (fidelity).

## Method

### Overall Architecture

The core mechanism of Chamfer Guidance: during the reverse sampling process of a diffusion model, gradient guidance based on Chamfer distance is applied to the current denoised intermediate result at regular intervals, steering a batch of generated images to match a small set of real reference images in feature space as closely as possible.

The workflow proceeds as follows:

1. Prepare $k$ real reference images in advance ($k$ can be as low as 2), and extract feature vector set $\mathcal{X}$ using DINOv2.
2. Diffusion sampling begins with Gaussian noise $x_T$ and proceeds through reverse denoising.
3. Every $G_{\text{freq}}=5$ steps, obtain the denoised estimate $\hat{x}_{0,t}$ at the current step via DDIM approximation.
4. Project $\hat{x}_{0,t}$ into DINOv2 feature space to obtain set $\mathcal{Y}$.
5. Compute $\mathcal{L}_{\text{Chamfer}}(\mathcal{X}, \mathcal{Y})$ and backpropagate gradients to update $x_t$.
6. Continue normal denoising until the final image is generated.

### Key Design: Bidirectional Matching via Chamfer Distance

The Chamfer distance consists of two terms:

$$\mathcal{L}_{\text{Chamfer}}(\mathcal{X},\mathcal{Y}) = \underbrace{\frac{1}{|\mathcal{X}|}\sum_{x\in\mathcal{X}}\min_{y\in\mathcal{Y}}\|x-y\|^2}_{\text{Coverage Term}} + \underbrace{\frac{1}{|\mathcal{Y}|}\sum_{y\in\mathcal{Y}}\min_{x\in\mathcal{X}}\|x-y\|^2}_{\text{Fidelity Term}}$$

- **Coverage term**: Requires that every real sample has a corresponding generated sample nearby → prevents mode collapse and encourages diversity.
- **Fidelity term**: Requires that every generated sample is close to some real sample → prevents out-of-distribution generation and ensures quality.

This bidirectional matching design allows Chamfer distance to naturally unify the Precision and Coverage objectives.

### Feature Space Selection

Prior to computing Chamfer distance, images are projected into the **DINOv2 (ViT-L)** feature space. DINOv2 is chosen because:

- Its self-supervised feature space captures semantic similarity better than CLIP and Inception.
- It more faithfully reflects human perceptual similarity.
- It strikes a better balance between object structure and background information.

### Guidance Equation

The Chamfer distance is incorporated as a guidance signal into the diffusion model's sampling process:

$$\nabla_{x_t}\log p_\theta(x_t|c,\mathcal{X}) = \nabla_{x_t}\log p_\theta(x_t|c) - \gamma\nabla_{x_t}\mathcal{L}_{\text{Chamfer}}(\mathcal{X}, \hat{x}_{0,t})$$

- $\gamma$ controls guidance strength.
- DDIM approximation is used to obtain $\hat{x}_{0,t}$, avoiding the computational cost of running all $T$ steps.
- Guidance frequency is $G_{\text{freq}}=5$, i.e., guidance is applied once every 5 steps.

### Computational Efficiency

A notable finding: for LDM3.5M, Chamfer Guidance achieves state-of-the-art performance **without CFG** ($\omega=1.0$) — eliminating the need for an additional unconditional model forward pass and achieving approximately **31% FLOPs reduction**.

## Key Experimental Results

### Main Results: ImageNet-1k Distribution Matching (Table 1)

| Method | Model | $k$ | $F_1$(P,C)↑ | Precision↑ | Coverage↑ | FDD↓ | FID↓ |
|------|------|-----|-------------|------------|-----------|------|------|
| CFG ($\omega$=7.5) | LDM1.5 | – | 0.709 | 0.862 | 0.603 | 248.7 | 16.1 |
| APG | LDM1.5 | – | 0.723 | 0.855 | 0.626 | 217.9 | 13.4 |
| c-VSG | LDM1.5 | 2 | 0.660 | 0.788 | 0.568 | 236.3 | 10.7 |
| Chamfer fine-tuning | LDM1.5 | 32 | 0.766 | 0.898 | 0.668 | 210.0 | 15.5 |
| **Chamfer Guidance** | **LDM1.5** | **2** | **0.886** | **0.947** | **0.833** | **156.2** | **13.7** |
| **Chamfer Guidance** | **LDM1.5** | **32** | **0.931** | **0.950** | **0.912** | **113.3** | **8.9** |
| CFG ($\omega$=2.0) | LDM3.5M | – | 0.727 | 0.872 | 0.623 | 231.9 | 15.7 |
| **Chamfer Guidance** | **LDM3.5M** | **2** | **0.912** | **0.964** | **0.864** | **134.3** | **8.9** |
| **Chamfer Guidance** | **LDM3.5M** | **32** | **0.950** | **0.975** | **0.927** | **121.4** | **9.6** |

**Key Findings**: Chamfer Guidance with only 2 real reference images substantially outperforms all baselines, and performance continues to improve as $k$ increases — a scalability property not exhibited by other methods, including fine-tuning.

### Ablation Study: Effect of CFG Strength $\omega$ (Table 4, LDM1.5, $k$=32)

| $\omega$ | $F_1$(P,C)↑ | Precision↑ | Coverage↑ | FDD↓ | FID↓ |
|----------|-------------|------------|-----------|------|------|
| 1.0 (no CFG) | 0.899 | 0.923 | 0.876 | 117.8 | 9.8 |
| **2.0** | **0.931** | **0.950** | **0.912** | **113.3** | **8.9** |
| 7.5 | 0.925 | 0.957 | 0.894 | 153.1 | 14.4 |

**Conclusion**: $\omega=2.0$ is optimal; $\omega=1.0$ (purely conditional model, no CFG) already approaches state-of-the-art, confirming that the unconditional model computation can be eliminated.

### Downstream Classifier Training (Table 3)

| Real Images | Guidance | Model | IN-1k Acc | IN-Sketch | IN-V2 |
|-----------|---------|------|-----------|-----------|-------|
| 0 | CFG $\omega$=2 | LDM1.5 | 47.67 | 20.49 | 40.33 |
| 0 | Chamfer $k$=32 | LDM1.5 | **54.91** | **28.08** | **46.43** |
| 0 | CFG $\omega$=2 | LDM3.5M | 37.83 | 17.60 | 34.07 |
| 0 | Chamfer $k$=32 | LDM3.5M | **53.66** | **34.44** | **45.46** |
| 32k | CFG $\omega$=2 | LDM1.5 | 59.07 | 25.04 | 49.77 |
| 32k | Chamfer $k$=32 | LDM1.5 | **63.81** | **32.34** | **53.84** |

**Key Findings**:

- Training on purely synthetic data: Chamfer Guidance yields +7–16% accuracy improvement.
- Mixed training (32k real + 1.3M synthetic): further improves performance to 63.81%.
- OOD generalization (ImageNet-Sketch): LDM3.5M + Chamfer even surpasses training on the full 1.3M real ImageNet images.

### Geographic Diversity Experiment (Table 2, GeoDE)

| Method | $k$ | Avg $F_1$↑ | Worst-Reg $F_1$↑ | Avg Coverage↑ |
|------|-----|-----------|-----------------|---------------|
| LDM1.5 baseline | – | 0.412 | 0.346 | 0.374 |
| c-VSG (CLIP) | 2 | 0.435 | 0.412 | 0.446 |
| **Chamfer (DINOv2)** | **4** | **0.500** | **0.469** | **0.459** |

Chamfer Guidance improves average $F_1$ by approximately 7% and worst-region coverage by approximately 4.9%, effectively alleviating geographic bias.

## Highlights & Insights

1. **Conceptual elegance**: Transplanting a classical tool from 3D point cloud matching (Chamfer distance) into image distribution matching, where the two directional terms naturally correspond to Precision and Coverage, without requiring manual balancing of two separate objectives.
2. **Extreme data efficiency**: Significant improvements are achieved with as few as 2 real reference images, and performance continues to scale with the number of references — a property not observed in fine-tuning or c-VSG.
3. **Training-free and plug-and-play**: Model weights are not modified; the method can be directly applied to any diffusion or flow-matching model.
4. **CFG-free capability**: On LDM3.5M, state-of-the-art performance is achieved without CFG, saving 31% of computation.
5. **Inference-time compute > training-time compute**: Consistent with the test-time compute scaling trend in the LLM domain, inference-time guidance proves more effective than fine-tuning.
6. **Narrowing the gap between models**: The performance gap between older and newer models (LDM1.5 vs. LDM3.5M) is substantially reduced when Chamfer Guidance is applied.

## Limitations & Future Work

1. **Class-conditional generation only**: The current method is designed for class-conditional models and does not directly support open-vocabulary text-to-image generation.
2. **Batch generation assumption**: Chamfer distance requires processing a batch of generated samples simultaneously and is not suited for single-image generation.
3. **Inherent bias in evaluation metrics**: Distribution metrics are computed using pretrained feature extractors (DINOv2, Inception, etc.), which themselves carry inherent biases.
4. **Feature space dependency**: The effectiveness of the method is closely tied to the choice of projection feature space (DINOv2 vs. CLIP results differ notably).
5. **Reference image selection**: The paper does not thoroughly discuss how the selection strategy for reference images affects results.

## Training / Inference

### Training Phase

Chamfer Guidance is a **completely training-free** method that does not modify any weights of the diffusion model. For comparison, the paper also evaluates two training-based alternatives:

- **Vanilla fine-tuning**: Uses real reference images as fine-tuning data, updating the U-Net / LoRA via the standard denoising loss (Eq. 1). LDM1.5 fine-tunes the entire U-Net; LDM3.5M uses LoRA (rank=4, applied to K/Q/V/O of attention). Learning rate $10^{-6}$, up to 5,000 steps, with checkpoints saved every 1,000 steps.
- **Chamfer fine-tuning (ReFL-style)**: Uses the negative Chamfer distance as a reward signal. During late-stage denoising ($T_1=30, T_2=39$, total $T=40$ steps), a random timestep is selected to compute the reward gradient for fine-tuning. $\lambda=10^{-3}$.

Experimental conclusion: both fine-tuning variants underperform inference-time Chamfer Guidance and do not scale with the number of reference samples.

### Inference Phase

Detailed inference procedure:

1. **Initialization**: Begin from Gaussian noise $x_T \sim \mathcal{N}(0, \mathbf{I})$; prepare the DINOv2 feature set $\mathcal{X}$ from $k$ real reference images.
2. **Standard denoising**: Use the default diffusers sampler for a total of 40 steps.
3. **Periodic guidance**: Every $G_{\text{freq}}=5$ steps, execute one guidance step:
    - Obtain the denoised estimate $\hat{x}_{0,t}$ via DDIM approximation.
    - Decode to pixel space and extract DINOv2 features $\mathcal{Y}$.
    - Compute $\nabla_{x_t}\mathcal{L}_{\text{Chamfer}}(\mathcal{X}, \mathcal{Y})$ and update the noise $x_t$.
4. **CFG settings**: LDM1.5 uses $\omega=2.0$; LDM3.5M uses $\omega=1.0$ (no unconditional model required, saving 31% FLOPs).
5. **Hardware**: Single H100 GPU for $k \leq 8$; multi-GPU for $k \in \{16, 32\}$.

**Computational overhead analysis**: The additional cost of Chamfer Guidance comes from DINOv2 forward passes and Chamfer distance gradient backpropagation. However, for LDM3.5M, the unconditional model forward pass is eliminated, resulting in lower overall FLOPs compared to the CFG baseline.

## Related Work & Insights

- **Comparison with c-VSG**: c-VSG measures diversity via Vendi Score, requiring storage of intermediate results and balancing two objective terms (ungrounded diversity + grounded diversity with two hyperparameters). Chamfer Guidance unifies Precision and Coverage in a single formula, is simpler to tune, and scales with the number of reference samples.
- **Relationship to ReFL (Reward Feedback Learning)**: Chamfer distance can serve as a reward signal for fine-tuning (Chamfer fine-tuning), but experiments demonstrate that inference-time guidance substantially outperforms training-time guidance — consistent with the test-time compute scaling trend in the LLM domain.
- **Analogy to In-Context Learning**: The method is analogous to few-shot in-context learning in LLMs, adapting at inference time using a small number of examples without modifying the model. A key distinction from c-VSG, which also attempts few-shot guidance, is that Chamfer Guidance scales with the number of reference samples.
- **Limitations of reference-free methods**: Methods such as APG, CADS, Limited Interval, and Particle Guidance improve Coverage over the base LDM on ImageNet-1k by only 1–3%, because the diversity they optimize is defined as an absolute quantity of variation rather than coverage relative to the target distribution.
- **Future directions**: (1) A retrieval-augmented text-to-image extension pipeline — building an offline retrieval index from large-scale text-image databases, with top-$k$ retrieval at inference time used as references; (2) A zero-shot bootstrapping pipeline — first generating a candidate set, then automatically selecting the subset that maximizes the "coverage diameter" in feature space to serve as guidance references.

## Rating

| Dimension | Score | Notes |
|------|------|------|
| Novelty | ⭐⭐⭐⭐ | Applying Chamfer distance to diffusion guidance is a fresh perspective, though overall the work is a clever combination of existing components. |
| Theoretical Depth | ⭐⭐⭐ | The method is intuitive but lacks theoretical analysis such as convergence guarantees. |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ | Multiple models, multiple datasets, downstream tasks, and ablation studies are covered very comprehensively. |
| Practical Utility | ⭐⭐⭐⭐ | Training-free and plug-and-play, though currently limited to class-conditional generation. |
| Writing Quality | ⭐⭐⭐⭐ | Clear structure, rich figures and tables, well-motivated throughout. |
| **Overall** | **⭐⭐⭐⭐** | A practically strong and experimentally rigorous work that elegantly imports a classical distance metric into generative guidance. |

## My Notes

- The core contribution of this work lies in **discovering the scalability of Chamfer distance as a guidance signal**: c-VSG and fine-tuning methods saturate or even degrade as more reference samples are added, while Chamfer Guidance continues to improve. This suggests that the choice of metric matters more than the guidance framework itself.
- The finding that **inference-time compute > training-time compute** deserves attention: using the same Chamfer distance as an objective function, inference-time guidance (Chamfer Guidance) substantially outperforms training-time optimization (Chamfer fine-tuning / ReFL). A plausible explanation is that fine-tuning tends to overfit to the small set of reference images, reducing diversity, whereas inference-time guidance acts on a fresh noise sample each time.
- **The subtle influence of feature space selection**: DINOv2 outperforms CLIP on object-centric tasks, but on the GeoDE geographic diversity task, CLIP achieves higher CLIPScore (as the guidance naturally aligns with the CLIP subspace). This implies that **different downstream tasks may require different projection spaces**.
- An implicit limitation of the method concerns the **relationship between batch size and reference set size**: Chamfer distance may degenerate when the two sets differ greatly in size (with the coverage or fidelity term dominating), yet the paper does not discuss the strategy for choosing generation batch size.
- **Extensibility to other generative tasks**: Chamfer distance guidance can be directly transferred to video generation (frame-level feature set matching) and 3D generation (where Chamfer distance originated), making it a potentially general-purpose framework.
- **Connection to data-free distillation**: In scenarios without real data, a reference set can be constructed via self-bootstrapping, providing a new direction for iterative self-improvement of synthetic data.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Enhancing Diffusion Model Guidance through Calibration and Regularization](enhancing_diffusion_model_guidance_through_calibration_and_regularization.md)
- [\[ICML 2025\] Privacy Amplification Through Synthetic Data: Insights from Linear Regression](../../ICML2025/image_generation/privacy_amplification_through_synthetic_data_insights_from_linear_regression.md)
- [\[NeurIPS 2025\] UtilGen: Utility-Centric Generative Data Augmentation with Dual-Level Task Adaptation](utilgen_utility-centric_generative_data_augmentation_with_dual-level_task_adapta.md)
- [\[CVPR 2025\] Co-Spy: Combining Semantic and Pixel Features to Detect Synthetic Images by AI](../../CVPR2025/image_generation/co-spy_combining_semantic_and_pixel_features_to_detect_synthetic_images_by_ai.md)
- [\[NeurIPS 2025\] Generative Model Inversion Through the Lens of the Manifold Hypothesis](generative_model_inversion_through_the_lens_of_the_manifold_hypothesis.md)

</div>

<!-- RELATED:END -->
