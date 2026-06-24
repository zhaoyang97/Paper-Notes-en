---
title: >-
  [Paper Note] TTT-MIM: Test-Time Training with Masked Image Modeling for Denoising Distribution Shifts
description: >-
  [ECCV 2024][Image Restoration][Test-Time Training] This paper proposes TTT-MIM, which jointly optimizes a supervised denoising loss and a self-supervised masked image modeling (MIM) loss during the training phase. At test time, adaptive fine-tuning on a single noisy image is performed by minimizing the self-supervised MIM loss. This significantly improves denoising performance against out-of-distribution noise (such as real camera noise and microscope noise) while being far f…
tags:
  - "ECCV 2024"
  - "Image Restoration"
  - "Test-Time Training"
  - "Masked Image Modeling"
  - "Image Denoising"
  - "Distribution Shift"
  - "Self-Supervised Learning"
date: 2026-05-08
content_hash: fc6ddb2254019de7
---

# TTT-MIM: Test-Time Training with Masked Image Modeling for Denoising Distribution Shifts

**Conference**: ECCV 2024  
**Code**: [https://github.com/MLI-lab/TTT_Denoising](https://github.com/MLI-lab/TTT_Denoising)  
**Area**: Image Restoration  
**Keywords**: Test-Time Training, Masked Image Modeling, Image Denoising, Distribution Shift, Self-Supervised Learning  

## TL;DR
This paper proposes TTT-MIM, which jointly optimizes a supervised denoising loss and a self-supervised masked image modeling (MIM) loss during the training phase. At test time, adaptive fine-tuning on a single noisy image is performed by minimizing the self-supervised MIM loss. This significantly improves denoising performance against out-of-distribution noise (such as real camera noise and microscope noise) while being far faster than zero-shot methods.

## Background & Motivation

**Background**: Neural networks trained end-to-end have achieved SOTA performance on image denoising tasks. Representative methods like DnCNN and DRUNet perform exceptionally well on standard Gaussian noise benchmarks. However, these models highly rely on the noise distribution of the training data. Once the noise characteristics of test images deviate from the training distribution, their performance drops drastically.

**Limitations of Prior Work**: In practical applications, noise distribution shifts are ubiquitous—noise characteristics in medical imaging (CT, MRI), microscope images, and smartphone cameras differ significantly from the synthetic Gaussian noise used during training. Existing denoisers have almost no capability to adapt to such distribution shifts. Although zero-shot methods (such as DIP and ZS-N2N) do not rely on training data, their inference speed is extremely slow, and their performance is inferior to supervised methods.

**Key Challenge**: Supervised denoisers are powerful but lack generalization; zero-shot denoisers generalize well but suffer from low efficiency and weak performance. A solution is needed that can leverage the prior knowledge of training data while adaptively conforming to new noise distributions at test time.

**Goal**: How to perform adaptive fine-tuning of the network at test time using only a single noisy image, enabling the pre-trained denoiser to rapidly adapt to new noise distributions?

**Key Insight**: The authors borrow the idea of test-time training (TTT) from NLP and computer vision—introducing an auxiliary self-supervised task during training, and adapting to new distributions at test time through gradient updates on this self-supervised task. The key observation is that masked image modeling (MIM), as a self-supervised task, is naturally related to the denoising task—both learn to reconstruct the original signal from corrupted or incomplete inputs.

**Core Idea**: By jointly training the supervised denoising and self-supervised MIM tasks during the training phase, the network can be rapidly adapted to the real noise distribution of a single test image at test time using only a few gradient updates of the MIM loss.

## Method

### Overall Architecture
TTT-MIM consists of two phases: (1) Training phase: Given noisy-clean image pairs, jointly optimize the supervised denoising loss $\mathcal{L}_{denoise}$ and the self-supervised MIM reconstruction loss $\mathcal{L}_{MIM}$; (2) Test-time adaptation phase: Given an out-of-distribution noisy image, perform a few gradient updates (typically 8 iterations) on the network parameters solely by minimizing $\mathcal{L}_{MIM}$, and then denoise the image using the updated network. The network backbone uses a UNet architecture, where Group Normalization replaces Batch Normalization to support single-image inference.

### Key Designs

1. **Joint Training Strategy**:

    - **Function**: Allows the network to learn both denoising and reconstruction from masked images simultaneously, establishing a bridge between the two tasks.
    - **Mechanism**: The training loss is defined as $\mathcal{L} = \mathcal{L}_{denoise} + \lambda \mathcal{L}_{MIM}$. For an input noisy image, a certain ratio of patches is randomly masked. The network is required to simultaneously predict the clean image (denoising) and the content of the masked patches (MIM). Both tasks share the same encoder, ensuring that the network learns representations usable for test-time adaptation while learning to denoise.
    - **Design Motivation**: The MIM task does not require clean ground-truth images, making it suitable as an anchor loss at test time. Through joint training, the gradient directions of MIM are aligned with those of the denoising task, ensuring that fine-tuning with MIM at test time indeed improves denoising performance.

2. **Test-Time Adaptation**:

    - **Function**: Enables the pre-trained network to quickly adapt to the specific noise pattern of a single noisy image at test time.
    - **Mechanism**: Given a test image, a mask is randomly generated (the mask ratio is typically 0.01-0.5, with patch sizes ranging from 1-14). The masked region is set to zero, and the network is tasked with predicting the masked region. The MIM loss is computed and backpropagated to update the parameters. Key to this is that the mask ratio and patch size need to be adjusted based on the noise type—using larger mask ratios and patch sizes for structured noise, and smaller values for random noise. Convergence is achieved within 8 iterations.
    - **Design Motivation**: Unlike zero-shot methods like DIP that require thousands of iterations, TTT-MIM leverages pre-trained priors, requiring very few gradient updates to complete adaptation. The MIM task forces the network to re-learn the local statistical properties of the test image, thereby implicitly adapting to the new noise distribution.

3. **Prediction-based Denoising Loss**:

    - **Function**: Provides a more robust denoising supervision signal at test time.
    - **Mechanism**: In addition to the standard MSE denoising loss, the authors introduce a prediction-based denoising loss $\mathcal{L}_{pd}$, which utilizes the network's current prediction as a pseudo-target. Optimizing this loss together with the MIM loss during test-time adaptation further stabilizes the adaptation process.
    - **Design Motivation**: Since there are no clean ground-truth images at test time, relying solely on the MIM loss can lead to unstable adaptation directions. Introducing a regularization based on the network's own prediction constrains the magnitude of the updates.

### Loss & Training
The total loss in the training phase is $\mathcal{L} = \mathcal{L}_{denoise} + \lambda \mathcal{L}_{MIM}$, where the denoising loss is MSE and the MIM loss is the reconstruction MSE of the masked regions. At test time, only $\mathcal{L}_{MIM}$ or $\mathcal{L}_{MIM} + \mathcal{L}_{pd}$ is optimized. The learning rate is adjusted between $10^{-6}$ and $10^{-4}$ based on the dataset.

## Key Experimental Results

### Main Results

| Dataset/Noise | Metric(PSNR) | Ours | DRUNet (No Adaptation) | ZS-N2N | DIP |
|--------|------|------|----------|------|------|
| SIDD (Real Camera Noise) | PSNR | **Best** | Performance degradation | Slower | Very slow |
| PolyU (Real Camera Noise) | PSNR | **Best** | Performance degradation | Lower | Lower |
| FMDD (Microscope Noise) | PSNR | **Best** | Performance degradation | Close | Lower |
| fastMRI (Simulated Gaussian) | PSNR | **Best** | Degradation | Lower | Very slow |
| ImageNet (Salt-and-Pepper Noise) | PSNR | **Significant Gain** | Severe degradation | Lower | - |

### Ablation Study

| Configuration | PSNR Change | Description |
|------|---------|------|
| Full model (MIM + pd loss) | Best | Full model |
| w/o MIM (supervised denoising only) | Unable to perform test-time adaptation | No self-supervised anchor |
| w/o joint training | Performance decreased | MIM gradient not aligned with denoising gradient |
| Different mask ratios | 0.01-0.5 best | Varies by noise type |
| Different number of iterations | Converged at 8 runs | Limited improvement with more iterations |

### Key Findings
- Test-time adaptation of TTT-MIM requires only 8 gradient updates, which is orders of magnitude faster than DIP (which requires >1000 iterations).
- On in-distribution data, TTT-MIM does not degrade performance (as the magnitude of MIM fine-tuning is very small).
- The adaptation effect on real noise (SIDD, PolyU) is particularly significant, validating the real-world value of the proposed method.
- Mask ratio and patch size need to be tuned according to the noise type: larger patches for structured noise and smaller patches for random noise.

## Highlights & Insights
- **The natural connection between MIM and denoising** is the most clever aspect of this paper—both tasks reconstruct the complete signal from partial observations. This task alignment allows MIM to serve as a proxy task for test-time adaptation. This philosophy can be extended to other image restoration tasks (e.g., super-resolution, deblurring).
- Adaptation is completed in just 8 iterations, achieving the best compromise of "supervised performance + zero-shot generalization." This efficiency advantage stems from aligning the gradient space of MIM and denoising during the joint training phase.
- The use of Group Normalization is a critical detail supporting single-image adaptation, as Batch Normalization statistics are unstable when batch size = 1.

## Limitations & Future Work
- The mask ratio and patch size need to be manually adjusted based on the noise type, lacking an automatic selection mechanism.
- Currently, only the UNet backbone has been verified; whether it is applicable to Transformer architectures (such as Restormer) remains to be validated.
- The hyper-parameters for batch adaptation mode (batch TTT) and single-image mode differ, lacking a unified adaptation strategy.
- For extreme distribution shifts (e.g., trained on Gaussian noise, tested on motion blur), the adaptation effect might be limited.

## Related Work & Insights
- **vs DIP/Deep Decoder**: Zero-shot methods require absolutely no training data but must be optimized from scratch (>1000 iterations). TTT-MIM utilizes pre-trained priors, requiring only 8 iterations.
- **vs DRUNet/SwinIR**: Supervised methods perform stronger within the training set but cannot adapt to new distributions; TTT-MIM fills this gap.
- **vs TTT (Sun et al.)**: The original TTT uses rotation prediction as an auxiliary task. The novelty of TTT-MIM lies in selecting MIM as an auxiliary task more closely related to denoising.
- **vs MAE**: While MAE is used for visual representation learning, this paper applies MIM to test-time adaptation in low-level vision tasks, representing a new direction of application.

## Rating
- Novelty: ⭐⭐⭐⭐ Test-time training using MIM as a proxy for denoising is a novel combination, though the TTT framework itself is not new.
- Experimental Thoroughness: ⭐⭐⭐⭐ It covers various noise types and datasets, and the ablation study is relatively complete.
- Writing Quality: ⭐⭐⭐⭐ The motivation is clear, and the method description flows smoothly.
- Value: ⭐⭐⭐⭐ It resolves the core challenge of noise distribution shifts in practical deployment, and the efficiency of 8 iterations makes it highly practical.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Domain-Adaptive Video Deblurring via Test-Time Blurring](domain-adaptive_video_deblurring_via_test-time_blurring.md)
- [\[ECCV 2024\] Exploiting Dual-Correlation for Multi-frame Time-of-Flight Denoising](exploiting_dual-correlation_for_multi-frame_time-of-flight_denoising.md)
- [\[ICLR 2026\] Test-Time Domain Generalization for Image Super-Resolution](../../ICLR2026/image_restoration/test-time_domain_generalization_for_image_super-resolution.md)
- [\[ECCV 2024\] Overcoming Distribution Mismatch in Quantizing Image Super-Resolution Networks](overcoming_distribution_mismatch_in_quantizing_image_super-resolution_networks.md)
- [\[ECCV 2024\] Rethinking Image Super-Resolution from Training Data Perspectives](rethinking_image_super-resolution_from_training_data_perspectives.md)

</div>

<!-- RELATED:END -->
