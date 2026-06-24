---
title: >-
  [Paper Note] Unseen Visual Anomaly Generation
description: >-
  [CVPR 2025][Object Detection][Visual Anomaly Generation] This paper proposes the AnomalyAny framework, which leverages the generative capability of pre-trained Stable Diffusion. By utilizing attention-guided optimization and prompt-guided refinement, it generates diverse and realistic unseen anomaly samples under the condition of requiring only a single normal sample and no additional training.
tags:
  - "CVPR 2025"
  - "Object Detection"
  - "Visual Anomaly Generation"
  - "Anomaly Detection"
  - "Stable Diffusion"
  - "Attention Guidance"
  - "Training-free"
date: 2026-05-08
content_hash: 0af984fa200de8d0
---

# Unseen Visual Anomaly Generation

**Conference**: CVPR 2025  
**arXiv**: [2406.01078](https://arxiv.org/abs/2406.01078)  
**Code**: [GitHub](https://hansunhayden.github.io/AnomalyAny.github.io/)  
**Area**: Image Generation  
**Keywords**: Visual Anomaly Generation, Anomaly Detection, Stable Diffusion, Attention Guidance, Training-free

## TL;DR

This paper proposes the AnomalyAny framework, which leverages the generative capability of pre-trained Stable Diffusion. By utilizing attention-guided optimization and prompt-guided refinement, it generates diverse and realistic unseen anomaly samples under the condition of requiring only a single normal sample and no additional training.

## Background & Motivation

Visual anomaly detection (AD) faces the core challenge of scarce anomaly data. Existing anomaly generation methods suffer from two main categories of limitations:

1. **Cut-and-paste methods** (e.g., DRAEM, NSA): These methods crop and paste random patterns from external datasets or the image itself as anomalies. Although simple, the generated results lack realism and a semantic understanding of true anomalies.
2. **Generative model methods** (e.g., AnomalyDiffusion, RealNet): These methods use GANs or fine-tune diffusion models to generate anomalies. While producing more realistic results, they require sufficient training samples, making them inapplicable in data-limited scenarios, and they can only generate seen anomaly types.

The key challenge lies in the fact that the rarity and variability of anomalies make collecting representative samples highly difficult; in industrial scenarios, product variations are diverse, and even normal samples might be insufficient. There is a critical need for a **training-free method that requires only a single normal sample and can generate unseen anomaly types**.

Core Idea of AnomalyAny: Directly leverage the extensive knowledge of pre-trained Stable Diffusion (SD) for anomaly generation instead of fine-tuning. Key challenges: (i) anomalies are extremely rare in the training data, and (ii) anomalies typically occupy small regions in an image and are easily ignored. These issues are addressed via test-time normal sample conditioning and attention-guided optimization.

## Method

### Overall Architecture

AnomalyAny consists of three core modules that work synergistically during inference:
1. **Test-time Normal Sample Conditioning**: Guides the generation process to align with the normal distribution.
2. **Attention-Guided Anomaly Optimization**: Forces SD to focus on generating the anomalous concepts.
3. **Prompt-guided Anomaly Refinement**: Leverages detailed textual descriptions to further enhance generation quality.

### Key Designs

1. **Test-time Normal Sample Conditioning**:
    - Function: Synthesizes anomalous images that remain consistent in appearance with the target normal sample while preserving the diversity of SD.
    - Mechanism: Given a normal sample $x_{\text{normal}}$, it is encoded via a VAE and injected with noise. Instead of starting inference from pure noise, the process starts from $z_{t_{\text{start}}}^{\text{normal}}$ at step $t_{\text{start}}=T \cdot (1-\gamma)$ (where $\gamma=0.25$). Optionally, a foreground mask can be used to constrain the anomaly position: $z_t = \text{mask} \odot z_t + (1-\text{mask}) \odot z_t^{\text{normal}}$.
    - Design Motivation: Compared to fine-tuning SD, test-time conditioning preserves the generalization capability and diversity of the model.

2. **Attention-Guided Anomaly Optimization**:
    - Function: Forces SD to focus on and manifest the anomaly semantics during the generation process.
    - Mechanism: At each denoising step $t$, the cross-attention maps at $16 \times 16$ resolution are aggregated, normalized via softmax, and smoothed with a Gaussian filter. The attention map corresponding to the anomaly token is extracted, and the optimization loss is defined as $\mathcal{L}_{\text{att}} = 1 - \max(A_t^j \odot \text{mask})$. The latent $z_t$ is updated via gradient descent. A localization-aware scheduler is defined as $\alpha_t = \lambda(1+\Delta t \cdot t) \cdot n_t / n_{t_{\text{start}}}$, which decreases the step size as attention focuses to prevent over-optimization.
    - Design Motivation: Anomalies are sparse in SD training data and occupy only small areas of the image, making their semantics easily ignored during direct generation.

3. **Prompt-guided Anomaly Refinement**:
    - Function: Enhances the semantic consistency and quality of anomaly generation using detailed textual descriptions.
    - Mechanism: GPT-4 is used to generate a detailed description $c'$ of the anomaly type. During the final 30 denoising steps, a CLIP image alignment loss $\mathcal{L}_{\text{img}} = 1 - \text{cosine}(\Phi^T(c'), \Phi^V(\tilde{x}_t))$ and a prompt embedding alignment loss $\mathcal{L}_{\text{prompt}} = 1 - \text{cosine}(\tau(c), \tau(c'))$ are introduced. The joint optimization loss is formulated as $\mathcal{L} = \mathcal{L}_{\text{img}} + \alpha_t \cdot \mathcal{L}_{\text{att}}$.
    - Design Motivation: Attention optimization processes only vague 1-2 token anomaly descriptions; detailed descriptions provide much richer semantic guidance.

### Loss & Training

**Training-Free!** All modules operate during inference:
- $\mathcal{L}_{\text{att}}$: Attention guidance loss (maximizing the peak attention of the anomaly token).
- $\mathcal{L}_{\text{img}}$: CLIP image-text alignment loss (generated image vs. detailed anomaly description).
- $\mathcal{L}_{\text{prompt}}$: Prompt embedding alignment loss (original prompt vs. detailed description).
- Inference configuration: $T=100$ steps, $\gamma=0.25$ (starting from step 75), prompt refinement applied in the final 30 steps.

## Key Experimental Results

### Main Results

Anomaly generation quality comparison (MVTec AD average):

| Method | IS↑ | IC-LPIPS↑ | Requires Training | Requires Anomaly Data |
|------|-----|-----------|---------|------------|
| NSA | 1.44 | 0.26 | No | No |
| RealNet | 1.64 | 0.30 | Yes | Yes |
| AnomalyDiffusion | 1.80 | 0.32 | Yes | Yes (1/3 test set) |
| **AnomalyAny** | **2.02** | **0.33** | **No** | **No** |

1-shot anomaly detection performance improvement:

| Method | MVTec I-AUC↑ | MVTec P-AUC↑ | VisA I-AUC↑ | VisA P-AUC↑ |
|------|-------------|-------------|-------------|-------------|
| PaDiM | 76.6 | 89.3 | 62.8 | 89.9 |
| PatchCore | 83.4 | 92.0 | 79.9 | 95.4 |
| WinCLIP+ | 93.1 | 95.2 | 83.8 | 96.4 |
| PromptAD | 94.6 | 95.9 | 86.9 | 96.7 |
| **AnomalyAny** | **94.9** | **95.4** | **89.7** | **97.7** |

### Ablation Study

| Configuration | Effect | Description |
|------|------|------|
| W/o Normal Sample Conditioning | Generated images deviate from the target distribution | Lacks object appearance constraints |
| W/o Attention Optimization | Anomaly semantics are ignored | SD default ignores small-region anomalies |
| W/o Prompt Refinement | Anomalies lack fine detail | Lacks detailed semantic guidance |
| W/o Localization Scheduler | Obvious artifacts appear | Step size did not decrease, leading to over-optimization |
| Full AnomalyAny | Realistic and diverse anomalies | Synergistic optimum of all three modules |

### Key Findings

1. **Best generation quality achieved training-free**: Both IS and IC-LPIPS outperform training-based methods.
2. **Strong generalization**: Generates samples for arbitrary object types and anomaly descriptions without being restricted by the training data distribution.
3. **Significant improvement in downstream detection**: Improves VisA I-AUC from 86.9 to 89.7.
4. **Attention maps serve as pixel-level annotations**: The final attention maps can directly localize anomalous regions.
5. **Localization scheduler prevents over-optimization**: Scaling down the optimization step size effectively avoids artifacts.

## Highlights & Insights

1. **Zero-training paradigm**: Completely leverages pre-trained SD knowledge, breaking the bottleneck of data scarcity.
2. **Three-stage progressive design**: Normal conditioning $\rightarrow$ attention optimization $\rightarrow$ prompt refinement. It progressively addresses three core difficulties in SD-based anomaly generation.
3. **Self-labeled**: The final attention map can serve directly as a pixel-level anomaly mask.
4. **GPT-4 assisted expansion of anomaly types**: Automatically generates possible anomaly categories and detailed descriptions using GPT-4, enabling open-vocabulary anomaly generation.
5. **High practical value**: Direct solution to the pain point of data scarcity for new products/defect types in industrial inspection scenarios.

## Limitations & Future Work

1. **Generation quality limited by SD**: Generation quality may degrade for anomalous patterns that rarely occur in the SD training data.
2. **Computational overhead**: Computing cross-attention maps and performing gradient optimization at each denoising step makes inference slower than standard SD generation.
3. **Limited control over anomaly localization**: While a mask can constrain the coarse region, the exact shape and scale of the anomaly are not fully controllable.
4. **Limitations of quantitative evaluation**: IS and IC-LPIPS metrics do not fully capture the fidelity of anomalies.
5. **Future Directions**: Finer control over anomaly localization and shape, integration with more advanced diffusion models, and expansion to 3D anomaly generation.

## Related Work & Insights

- **vs. AnomalyDiffusion**: AnomalyDiffusion requires training on 1/3 of the anomaly data, which belongs to seen anomaly generation; AnomalyAny does not require any anomaly data to generate unseen anomalies.
- **vs. DRAEM/NSA**: Cut-and-paste methods lack realism; AnomalyAny exploits the generative power of SD to produce more realistic anomalies.
- **vs. RealNet**: RealNet also utilizes SD but requires fine-tuning, limiting it to the training distribution; AnomalyAny operates directly during inference, offering stronger generalization.
- **vs. Attend-and-Excite**: The idea of attention-guided optimization is inspired by this work but is specifically adapted to the anomaly generation context.

## Rating

- Novelty: ⭐⭐⭐⭐ The concept of directly utilizing SD for training-free anomaly generation is novel; the combination of attention-guidance and prompt-refinement is effective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Tested on both MVTec AD and VisA, providing dual evaluations of generation quality and downstream detection performance.
- Writing Quality: ⭐⭐⭐⭐ Clear methodological descriptions, intuitive ablation studies, and rich visualizations.
- Value: ⭐⭐⭐⭐ Possesses high practical value in the industrial anomaly detection domain; the training-free paradigm lowers the barrier for anomalous data synthesis.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] ROICtrl: Boosting Instance Control for Visual Generation](roictrl_boosting_instance_control_for_visual_generation.md)
- [\[CVPR 2025\] UniVAD: A Training-free Unified Model for Few-shot Visual Anomaly Detection](univad_a_training-free_unified_model_for_few-shot_visual_anomaly_detection.md)
- [\[CVPR 2026\] Visual Prototype Conditioned Focal Region Generation for UAV-Based Object Detection](../../CVPR2026/object_detection/visual_prototype_conditioned_focal_region_generation_for_uav-based_object_detect.md)
- [\[AAAI 2026\] Correcting False Alarms from Unseen: Adapting Graph Anomaly Detectors at Test Time](../../AAAI2026/object_detection/correcting_false_alarms_from_unseen_adapting_graph_anomaly_detectors_at_test_tim.md)
- [\[CVPR 2025\] MulSen-AD: Multi-Sensor Object Anomaly Detection](mulsen_ad_multi_sensor_anomaly_detection.md)

</div>

<!-- RELATED:END -->
