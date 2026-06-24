---
title: >-
  [Paper Note] DiffuseDef: Improved Robustness to Adversarial Attacks via Iterative Denoising
description: >-
  [ACL 2025][Image Restoration][Adversarial Defense] DiffuseDef inserts a diffusion denoiser layer between the encoder and the classifier. During training, it learns to predict noise in hidden states. During inference, it adds noise to the hidden representations, iteratively denoises them, and performs ensemble averaging. This plug-and-play approach significantly enhances the robustness of text classification models under both black-box and white-box adversarial attacks.
tags:
  - "ACL 2025"
  - "Image Restoration"
  - "Adversarial Defense"
  - "Diffusion Models"
  - "Textual Adversarial Attacks"
  - "Hidden State Denoising"
  - "Ensemble Robustness"
date: 2026-05-08
content_hash: 42dddc9cd68649ce
---

# DiffuseDef: Improved Robustness to Adversarial Attacks via Iterative Denoising

**Conference**: ACL 2025  
**arXiv**: [2407.00248](https://arxiv.org/abs/2407.00248)  
**Code**: [https://github.com/Nickeilf/DiffuseDef](https://github.com/Nickeilf/DiffuseDef)  
**Institution**: Imperial College London (LAMA Lab)
**Area**: Image Restoration  
**Keywords**: Adversarial Defense, Diffusion Models, Textual Adversarial Attacks, Hidden State Denoising, Ensemble Robustness

## TL;DR

DiffuseDef inserts a diffusion denoiser layer between the encoder and the classifier. During training, it learns to predict noise in hidden states. During inference, it adds noise to the hidden representations, iteratively denoises them, and performs ensemble averaging. This plug-and-play approach significantly enhances the robustness of text classification models under both black-box and white-box adversarial attacks.

## Background & Motivation

- **Background**: Pre-trained language models (PLMs) exhibit excellent performance on tasks such as text classification and natural language inference, but remain highly vulnerable to adversarial text attacks—attackers can flip model predictions by merely replacing a few synonyms or introducing spelling variations. Existing defense methods can be categorized into adversarial training, ensemble, and denoising, each with obvious limitations.
- **Limitations of Prior Work**:
    - **Adversarial Training (FreeLB++, InfoBERT, etc.)**: Assumes that attack types during training and testing are similar, making it prone to overfitting to specific attack methods and leading to poor generalization.
    - **Ensemble Methods (RanMask, SAFER, etc.)**: Requires a complete forward pass for each input variant (10 ensembles = 10x FLOPS), resulting in extremely low inference efficiency.
    - **Denoising Methods (RMLM, etc.)**: Performs denoising at the text/label level, which may drastically alter semantic representations of clean text, leading to performance degradation on clean samples.
- **Key Challenge**: How to efficiently eliminate adversarial perturbations and generalize to unseen attack types without sacrificing the classification accuracy of clean text.
- **Key Insight**: Diffusion models are naturally skilled at predicting and removing noise, while adversarial perturbations can be analogized to "noise" in the hidden space. This work migrates diffusion denoising from **image pixel space** to the **NLP hidden state space**, while utilizing ensembles restricted to a lightweight diffusion layer to balance efficiency.
- **Core Idea**: Use a plug-and-play diffusion layer to iteratively denoise adversarial perturbations in the hidden state space, and enhance robustness through multi-sample ensemble.

## Method

### Overall Architecture

Two-stage training + three-step inference:

1. **Adversarial Training Stage**: Train the encoder and classifier using methods like FreeLB++ or RSMI to obtain base robustness.
2. **Diffusion Training Stage**: Freeze the encoder and classifier, and train only the diffusion layer (a single-layer Transformer with time embeddings) to learn to predict the random Gaussian noise injected into the hidden states.
3. **Inference Stage**: The encoder extracts the hidden state $h$ $\to$ sample $k$ noise vectors to generate $k$ noisy variants $\to$ the diffusion layer performs $t'$ steps of reverse diffusion denoising on each variant $\to$ average all denoised hidden states $\to$ the classifier outputs the final prediction.

### Key Designs

1. **Diffusion Denoiser**

    - **Structure**: A single-layer Transformer encoder with sinusoidal time embeddings, adding only about 10M parameters (compared to 110M for BERT).
    - **Training**: Add noise to the clean hidden state $h$ according to the forward diffusion formula $h_t = \sqrt{\bar\alpha_t} h + \sqrt{1 - \bar\alpha_t} \epsilon$. The diffusion layer learns to predict the noise $\epsilon_\theta(h_t, t)$ using the MSE loss: $L = \mathbb{E}_{t,h,\epsilon}[\|\epsilon - \epsilon_\theta(h_t, t)\|^2]$.
    - **Design Motivation**: Adversarial perturbations manifest as shifts/noise in the hidden space, and diffusion denoising is naturally suited to eliminate such perturbations. Operating in the hidden space instead of the input space avoids difficulties caused by the discrete nature of text.

2. **Noise-Denoise-Ensemble Pipeline**

    - **Noising**: Sample $k=10$ independent Gaussian noise vectors and perform a 1-step forward diffusion on the hidden state $h$ to generate $k$ noisy variants $H_{t'} = [h_{t'}^0, ..., h_{t'}^k]$.
    - **Denoising**: Each variant undergoes $t'=5$ steps of reverse diffusion (DDPM) to step-by-step subtract the noise predicted by the diffusion layer.
    - **Ensemble**: Average all denoised hidden states to obtain $\text{avg}(H_0)$, which is then fed into the classifier to output the final label.
    - **Design Motivation**: Adding noise introduces randomness, preventing attackers from finding stable vulnerable words. The ensemble is performed only on the diffusion layer (10M parameters) without re-running the encoder (110M parameters), which is far more efficient than traditional full-model ensembles.

3. **Plug-and-Play Compatibility**

    - **Decoupling**: During the diffusion training stage, the encoder and classifier are completely frozen, and the diffusion layer is trained independently. Therefore, it can be stacked on top of any existing adversarial training method.
    - **Flexibility**: Experiments verify that various base methods, such as FreeLB++, RSMI, and adversarial data augmentation, can further improve robustness by integrating DiffuseDef.
    - **Design Motivation**: Avoid coupling constraints between methods, making DiffuseDef a general-purpose robustness enhancement module.

### Training & Inference Hyperparameters

| Hyperparameter | AGNews / QNLI | IMDB |
|------|--------------|------|
| Max training steps $t$ | 30 | 10 |
| Inference denoising steps $t'$ | 5 | 5 |
| Number of ensembles $k$ | 10 | 10 |
| $\beta$ schedule | Linear, $10^{-4}$ to $0.02$ | Same as left |
| Adversarial training epochs | 10 | 10 |
| Diffusion training epochs | 100 | 100 |

## Key Experimental Results

### Main Results: Black-box Attack Robustness (BERT backbone, AUA%)

| Method | Clean% | TextFooler | TextBugger | BERT-Attack | Avg. #Query |
|------|--------|-----------|-----------|-------------|-----------|
| Fine-tuned | 94.4 | 10.2 | 25.4 | 27.1 | ~366 |
| FreeLB++ | 95.0 | 54.7 | 56.5 | 44.6 | ~415 |
| RSMI | 94.3 | 52.6 | 56.7 | 55.4 | ~701 |
| ATINTER | 94.2 | 68.0 | 59.0 | 81.0 | ~295 |
| **DiffuseDef-FreeLB++** | **94.8** | **84.5** | **86.0** | **84.6** | **~920** |
| **DiffuseDef-RSMI** | **93.8** | **82.7** | **83.3** | **84.4** | **~951** |

> AGNews dataset. DiffuseDef improves the AUA by an average of about 30 percentage points with almost no loss in clean accuracy, while the number of queries required for attacks increases by 2-3 times. Similar trends are observed on IMDB and QNLI.

### White-box Attack Robustness (AUA%)

| Method | AGNews T-PGD | AGNews SemAttack | IMDB T-PGD | IMDB SemAttack |
|------|-------------|-----------------|-----------|---------------|
| Fine-tuned | 8.8 | 41.5 | 3.0 | 1.3 |
| FreeLB++ | 19.6 | 58.2 | 15.5 | 3.4 |
| **DiffuseDef-FreeLB++** | **59.4** | **68.1** | **50.3** | **28.2** |
| RSMI | 79.6 | n/a | 43.3 | n/a |
| **DiffuseDef-RSMI** | **81.7** | n/a | **48.4** | n/a |

### Ablation Study (AGNews, AUA%)

| Configuration | TextFooler | TextBugger | BERT-Attack |
|------|-----------|-----------|-------------|
| DiffuseDef (Full) | 84.5 | 86.0 | 84.6 |
| W/o Ensemble | 64.2 | 65.8 | 54.9 |
| W/o Diffusion Denoising | 57.1 | 58.4 | 47.7 |
| W/o Adversarial Training | 80.3 | 80.9 | 80.5 |

> Diffusion denoising and ensemble are the primary sources of robustness gains; removing adversarial training has a relatively minor impact, indicating that DiffuseDef does not rely on specific adversarial training methods.

### Efficiency Comparison

| Method | Parameters | Inference FLOPS | Training Time |
|------|-------|-----------|---------|
| Fine-tuned BERT | 110M | 46G | 1x |
| FreeLB++ | 110M | 46G | 10.5x |
| RanMask (k=10) | 110M | 459G | 1.2x |
| SAFER (k=10) | 110M | 459G | 1x |
| DiffuseDef (t'=1, k=10) | 120M | 96G | 1.1x |
| DiffuseDef (t'=5, k=10) | 120M | 267G | 1.1x |

> The inference FLOPS of DiffuseDef are only 21%~58% of traditional ensemble methods with the same ensemble size (459G), and its training time is also far lower than FreeLB++ (10.5x).

### Hidden Space Distance Analysis

| Method | L2 Distance | Cosine Distance |
|------|--------|------------|
| FreeLB++ | 12.53 | 0.35 |
| DiffuseDef-FreeLB++ | 10.66 | 0.27 |
| RSMI | 9.72 | 0.24 |
| DiffuseDef-RSMI | 8.61 | 0.21 |

> After applying DiffuseDef, the distance between adversarial hidden states and clean hidden states is significantly reduced, indicating that denoising + ensemble effectively pulls adversarial representations closer to clean representations.

## Highlights & Insights

- **Denoising in Hidden Space vs. Input/Label Space**: Unlike RMLM (text-level) and Yuan et al. (label-level), DiffuseDef denoises directly in the final hidden layer, avoiding the discrete nature of text and improving efficiency.
- **Analogy of Perturbation as Noise**: Treating adversarial perturbations as hidden space noise is an elegant abstraction, making the diffusion model's denoising mechanism naturally adaptive for adversarial defense.
- **Breakthrough in Ensemble Efficiency**: Traditional ensembles require 10x FLOPS, whereas DiffuseDef only requires ~2x-6x, because the ensemble is performed exclusively in the lightweight diffusion layer.
- **Multiplied Attack Difficulty**: Random noising introduces non-determinism to model outputs, requiring attackers to perform 2-3 times more queries to find vulnerable words.

## Limitations & Future Work

- Only validated on classification tasks (AGNews, IMDB, QNLI). The applicability to generative tasks (translation, summarization) and sequence labeling tasks remains unexplored.
- Inference FLOPS are still higher than non-ensemble methods (such as FreeLB++), imposing certain limitations in latency-sensitive scenarios.
- The number of denoising steps $t'$ and ensemble size $k$ need to be tuned for each dataset, lacking an adaptive mechanism.
- The defense performance against semantic paraphrasing attacks (sentence-level rewriting instead of word-level perturbation) has not been verified.
- The diffusion layer is fixed as a single-layer Transformer; deeper or alternative architectures for the diffusion denoiser have not been explored yet.

## Rating

| Dimension | Score (1-10) | Explanation |
|------|-----------|------|
| Novelty | 8 | First to apply diffusion denoising in the NLP hidden state space for adversarial defense; the approach is novel and effective |
| Effectiveness | 9 | Achieves comprehensive SOTA across three datasets and five attacks, improving average AUA by ~30% with complete ablation studies |
| Practicality | 7 | Good plug-and-play design, though increased inference FLOPS and hyperparameter tuning present obstacles for real-world deployment |
| Writing Quality | 8 | Clearly structured paper, complete mathematical derivations, and thorough ablation and efficiency analyses |

- **vs ATINTER**: ATINTER rewrites adversarial text using T5, whereas DiffuseDef directly operates in the representation space, making it more efficient.
- **vs DiffPure in CV**: DiffPure performs diffusion purification on the entire input image, whereas DiffuseDef operates only on the final hidden state.

## Rating
- Novelty: ⭐⭐⭐⭐ Novel idea of applying diffusion denoising to NLP adversarial defense.
- Experimental Thoroughness: ⭐⭐⭐⭐ 3 datasets + 5 attack types + multiple baselines + white-box + efficiency analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear description of the methodology, intuitive algorithm framework.
- Value: ⭐⭐⭐⭐ Plug-and-play defense method with high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Improved Adversarial Diffusion Compression for Real-World Video Super-Resolution](../../ICLR2026/image_restoration/improved_adversarial_diffusion_compression_for_real-world_video_super-resolution.md)
- [\[ICCV 2025\] IDF: Iterative Dynamic Filtering Networks for Generalizable Image Denoising](../../ICCV2025/image_restoration/idf_iterative_dynamic_filtering_networks_for_generalizable_image_denoising.md)
- [\[ICLR 2026\] Are Deep Speech Denoising Models Robust to Adversarial Noise?](../../ICLR2026/image_restoration/are_deep_speech_denoising_models_robust_to_adversarial_noise.md)
- [\[ACL 2025\] A Self-Denoising Model for Robust Few-Shot Relation Extraction](a_self-denoising_model_for_robust_few-shot_relation_extraction.md)
- [\[ICML 2025\] ε-VAE: Denoising as Visual Decoding](../../ICML2025/image_restoration/epsilon-vae_denoising_as_visual_decoding.md)

</div>

<!-- RELATED:END -->
