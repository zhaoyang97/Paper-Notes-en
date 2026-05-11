---
title: >-
  [Paper Note] Mint: A Simple Test-Time Adaptation of Vision-Language Models against Common Corruptions
description: >-
  [NeurIPS 2025][Multimodal VLM][test-time adaptation] This work identifies **embedding variance collapse**—the simultaneous shrinkage of intra- and inter-class variance that erodes discriminability in the embedding space—…
tags:
  - "NeurIPS 2025"
  - "Multimodal VLM"
  - "test-time adaptation"
  - "CLIP"
  - "corruption robustness"
  - "embedding variance collapse"
  - "inter-class variance"
  - "pseudo-label"
  - "LayerNorm"
date: 2026-05-08
content_hash: 891c65c370e220bf
---

# Mint: A Simple Test-Time Adaptation of Vision-Language Models against Common Corruptions

**Conference**: NeurIPS 2025
**arXiv**: [2510.22127](https://arxiv.org/abs/2510.22127)
**Code**: [https://github.com/baowenxuan/Mint](https://github.com/baowenxuan/Mint)
**Area**: Multimodal VLM
**Keywords**: test-time adaptation, CLIP, corruption robustness, embedding variance collapse, inter-class variance, pseudo-label, LayerNorm

## TL;DR
This work identifies **embedding variance collapse**—the simultaneous shrinkage of intra- and inter-class variance that erodes discriminability in the embedding space—as the root cause of CLIP's performance degradation under image corruptions. It proposes Mint, which restores embedding geometry online by maximizing pseudo-label inter-class variance (PL-inter) using only two lightweight components: a mean accumulator and a gradient accumulator. Mint consistently improves CLIP's classification accuracy across multiple corruption benchmarks even at BS=1, while running 45× faster than the strongest baseline.

## Background & Motivation
Pretrained vision-language models (VLMs) such as CLIP possess strong zero-shot generalization, yet suffer substantial performance drops under common image corruptions including Gaussian noise, motion blur, fog, snow, and JPEG compression. Existing test-time adaptation (TTA) methods follow three main directions:

1. **Text-side adjustment** (TPT, TPS): modifies text embeddings via prompt tuning or prompt weighting to improve vision-language alignment, but entirely ignores image embedding quality.
2. **Sample-similarity methods** (TDA, DMN-ZS): caches high-confidence sample embeddings and adjusts prediction distributions using inter-image similarity.
3. **Image encoder repair** (CLIPArTT, WATT-S): adjusts normalization layer parameters to align image–image and text–text similarity matrices, but **relies heavily on large batches** (WATT-S takes 50 minutes to process 10k images).

The key insight of this paper is that existing methods either fail to repair image embeddings or lack a theoretical understanding of why embeddings degrade. The authors are the first to reveal the fundamental cause of degradation from the perspective of embedding space geometry: **variance collapse**—as corruption severity increases, the embedding space contracts and all samples, regardless of class, become increasingly similar.

## Method

### 1. Discovery and Measurement of Variance Collapse
Inspired by the Fisher score and contrastive learning objectives, the authors define three variance measures to assess embedding quality:
- **GT-total**: average L2 distance from all sample embeddings to the global mean.
- **GT-inter**: average distance from each class centroid to the global mean (inter-class separation).
- **GT-intra**: average distance from samples within a class to their class centroid (intra-class compactness).

These satisfy the decomposition GT-total = GT-inter + GT-intra (analogous to ANOVA variance decomposition).

Experiments across 76 settings on CIFAR-100-C (15 corruption types × 5 severity levels + clean) yield the following core findings:
- As corruption severity increases, **all three variance measures decrease**—the embedding space is "flattened" along every dimension.
- GT-inter correlates with classification accuracy at **0.98** (GT-intra: 0.86; GT-total: 0.94).
- **Inter-class variance collapse is the direct cause of performance degradation**: different classes become indistinguishable in the embedding space.

The intuition is that corruption-induced common patterns (e.g., global noise textures) are captured by the encoder as a dominant shared signal, pulling class embeddings that were originally distributed in diverse directions toward the same region.

### 2. Theoretical Analysis
The authors construct a decoupled representation model to explain variance collapse. Each image's latent representation $v$ is assumed to comprise four orthogonal components:
- **Task-relevant features** $v_\text{cls} = \pm\mu$: directly encode class information.
- **Task-irrelevant features** $v_\text{irr} \sim \text{Rademacher}$: background and other classification-irrelevant information.
- **Structured shift** $v_\text{shift} = s \cdot \delta$: systematic distributional change induced by the corruption type (e.g., specific spectral characteristics of Gaussian blur).
- **Unstructured noise** $v_\text{noise} \sim s \cdot \text{Rademacher}$: random noise introduced by corruption.

After RMSNorm (a simplified form of LayerNorm) and L2 normalization:

**Theorem 3.1 (Theoretical Explanation of Variance Collapse)**:
$$\mathcal{V}_{\text{inter}}^{\text{GT}} \to \frac{\|\mu\|^2}{\|\mu\|^2 + d_{\text{irr}} + s^2 \|\delta\|^2 + s^2 d_{\text{noise}}}$$
- The $s^2$ terms in the denominator grow monotonically with corruption severity → GT-inter strictly decreases.
- Physical interpretation: normalization fixes a total variance "budget"; as corruption signals occupy a larger share, the proportion attributable to class-discriminative signals is squeezed.
- GT-intra also decreases when the structured shift $\|\delta\|$ is sufficiently large (satisfying $\|\delta\| \geq \sqrt{d_\text{noise}/d_\text{irr}} \cdot \|\mu\|$).

**Theorem 3.2 (Theoretical Guarantee of PL-inter Maximization)**: When pseudo-labels (the model's own predictions) are used in place of ground-truth labels and gradient ascent on LayerNorm weights is performed to maximize PL-inter variance:
- $\nabla_{w_\text{shift}} \leq 0$: weights corresponding to the structured shift are **suppressed**—the algorithm automatically identifies and attenuates corruption signals.
- $\nabla_{w_\text{cls}} \geq 0$: weights corresponding to task-relevant features are **amplified** (condition: the covariance between pseudo-labels and ground-truth labels $\sigma^2$ exceeds the covariance upper bound of the noise components).
- This implies that as long as the model's predictions are better than random, PL-inter maximization correctly reweights embedding components.

### 3. Mint Algorithm Design

The core mechanism is to update the image encoder's LayerNorm parameters at test time via gradient ascent to maximize PL-inter variance. The key challenge is that in online settings with very small batches (as few as 1), direct computation of PL-inter variance yields severely biased estimates.

#### Mean Accumulator
**Addresses "what to estimate."**

Decomposing PL-inter as PL-total − PL-intra reveals that maximizing PL-inter is equivalent to encouraging each sample to move away from the global mean $\tilde{z}$ and toward its class mean $\tilde{z}_c$, with gradient direction approximately $\tilde{z}_c - \tilde{z}$. Accurate estimation of $\tilde{z}$ and $\tilde{z}_c$ is therefore critical.

The problem with estimating from the current batch alone: with 1000 classes in ImageNet and batch size 20, most classes have only one sample, causing $\tilde{z}_c$ to degenerate to the sample itself, PL-intra to collapse to 0, and the objective to reduce to PL-total rather than true PL-inter.

Solution: maintain online **cumulative averages** of the global mean and each pseudo-class mean:
- For each incoming sample $z_i$ with pseudo-label $\hat{y}_i$, incrementally update $\tilde{z}$ and $\tilde{z}_{\hat{y}_i}$.
- Space complexity $O(Cd)$; no historical samples are stored.
- PL-inter is computed using current-batch samples combined with cross-batch accumulated means.

#### Gradient Accumulator
**Addresses "how to update."**

Even with accurate mean estimates, gradients from a single batch remain noisy. The gradient accumulator maintains a cross-batch cumulative average of gradient directions: $\bar{g} \leftarrow \frac{b-1}{b} \cdot \bar{g} + \frac{1}{b} \cdot g_b$, equivalent to using the average gradient direction over all historical batches to guide updates. Only one gradient ascent step is taken per batch.

#### Text Embedding Adjustment
Accumulated class means are used to correct text embeddings, enabling online improvement of vision-language modality alignment:
$$\tilde{t}_c \leftarrow \text{normalize}\left(\frac{K_{\text{prior}}}{K_{\text{prior}}+K} \cdot t_c + \frac{K}{K_{\text{prior}}+K} \cdot \tilde{z}_c\right)$$
$K_\text{prior}=10000$ controls prior strength—early on the original text embeddings are trusted; as adaptation proceeds, the adapted image class means are gradually favored.

#### Refined Training Strategy
- **Only LayerNorm parameters are updated**; after each batch's single gradient ascent step, **model parameters and optimizer state are reset**.
- **Accumulators are preserved across batches**—knowledge accumulates continuously while model parameters are re-adapted from scratch each batch.
- This design of "accumulating knowledge but re-adapting from zero" avoids the error accumulation and catastrophic forgetting common in online TTA.
- Adam optimizer; learning rate 0.007 for ViT-B and 0.015 for ViT-L.

## Key Experimental Results

### Main Results (Severity=5, BS=20)

| Setting | CLIP | Best Baseline | **Mint** | Gain |
|---------|------|---------------|---------|------|
| ViT-B/32 + CIFAR-10-C | 59.0% | 67.1% (WATT-S) | **71.0%** | +3.9% |
| ViT-B/16 + CIFAR-100-C | 35.8% | 41.9% (WATT-S) | **44.1%** | +2.2% |
| ViT-L/14 + ImageNet-C | 39.6% | 43.9% (WATT-S) | **47.0%** | +3.1% |

Mint achieves the best or near-best performance across all 15 corruption types. The advantage is most pronounced under severe corruptions such as Gaussian/impulse noise (CIFAR-10-C noise: 59.0→54.2–62.4% for Mint vs. only 50.7–54.9% for WATT-S).

### Batch Size Robustness

| BS | CIFAR-10-C | CIFAR-100-C | ImageNet-C |
|----|-----------|-------------|-----------|
| 1 | 70.5% | 43.1% | 45.8% |
| 20 | 71.0% | 44.1% | 47.0% |
| 200 | 70.6% | 44.6% | 46.8% |

Performance fluctuates by at most ±0.5% from BS=1 to BS=200, whereas WATT-S and CLIPArTT degrade sharply at small batch sizes.

### Efficiency Comparison (CIFAR-100-C, 10,000 images)

| Method | Time | Accuracy |
|--------|------|----------|
| WATT-S | 50m20s | 41.9% |
| TPT | 23m21s | 36.0% |
| CLIPArTT | 7m40s | 40.7% |
| **Mint** | **1m07s** | **44.1%** |
| CLIP | 21s | 35.8% |

Mint is **45× faster** than WATT-S, the strongest baseline, while achieving 2.2% higher accuracy. The efficiency stems from requiring only a single forward pass and one gradient update step per batch, with no need for multiple iterations or multiple augmented views.

### Ablation Study
- Removing the mean accumulator: **complete failure at BS=1** (a single sample cannot form a meaningful inter-class variance).
- Removing the gradient accumulator: significant performance drop at small batch sizes due to unstable adaptation from noisy gradients.
- The two components are complementary: the mean accumulator ensures the objective is well-defined; the gradient accumulator ensures the update direction is correct.

### Verification of Variance Collapse Mitigation
Pre- and post-adaptation comparisons confirm that Mint successfully increases both PL-inter and GT-inter variance, and that the gains in GT-inter correlate closely with accuracy improvements, validating the theoretical hypothesis.

## Highlights & Insights
- **Variance collapse** as a core insight: the first work to reveal the embedding-space geometric mechanism by which corruptions degrade VLM performance; GT-inter correlates with accuracy at 0.98, providing a clear diagnostic metric for future work.
- **Closed theoretical loop**: Theorem 3.1 explains *why collapse occurs* → Theorem 3.2 proves *why maximizing PL-inter is effective* → the method design follows naturally.
- **Extreme simplicity**: the entire method consists of two accumulators and a single gradient ascent step, requiring no augmented views, no memory banks, and no iterative optimization.
- **Design philosophy**: decoupling information aggregation (persistent accumulators) from parameter updates (per-batch reset) elegantly avoids error accumulation in online TTA.

## Limitations & Future Work
- Pseudo-label errors may accumulate under extreme corruptions; the theory only analyzes the regime where pseudo-labels are sufficiently accurate without providing a quantitative error bound.
- Validation is limited to classification tasks; dense prediction settings such as detection and segmentation remain unexplored.
- The theoretical assumptions require orthogonal decoupling of latent representation components, which real corruptions may violate.
- Whether the per-batch parameter reset strategy is optimal has not been examined—continual adaptation may further improve performance in long-sequence scenarios.
- Performance under non-corruption domain shifts such as style changes and adversarial attacks has not been evaluated.

## Related Work & Insights
- **vs. TPT**: TPT applies prompt tuning to reduce marginal entropy across augmented views, requiring 63 forward passes per sample; Mint uses a single forward pass and one gradient step, is an order of magnitude faster, and directly repairs embedding quality.
- **vs. WATT-S**: WATT-S aligns image–image and text–text similarity matrices to repair modality alignment, requiring large batches and multiple iterations (50 min/10k images); Mint completes the same task in 1 minute and operates at BS=1.
- **vs. TDA/DMN-ZS**: these methods adjust prediction distributions via sample similarity without repairing embedding quality; Mint addresses the problem at the geometric root of the embedding space.
- **vs. CLIPArTT**: both adjust LayerNorm, but CLIPArTT optimizes a modality alignment objective and treats each batch independently; Mint maximizes inter-class variance and aggregates information across batches via accumulators.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — variance collapse discovery + complete theory + minimal method, all in one.
- Experimental Thoroughness: ⭐⭐⭐⭐ — 3 architectures × 3 benchmarks × 15 corruption types + batch size robustness + efficiency + ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ — the chain from phenomenon → theory → method → experiments is logically seamless.
- Value: ⭐⭐⭐⭐⭐ — a minimal method that solves the practical VLM corruption robustness problem in deployment, balancing effectiveness and efficiency.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] DOTA: DistributiOnal Test-time Adaptation of Vision-Language Models](dota_distributional_testtime_adaptation_of_visionlanguage_mo.md)
- [\[NeurIPS 2025\] The Illusion of Progress? A Critical Look at Test-Time Adaptation for Vision-Language Models](the_illusion_of_progress_a_critical_look_at_testtime_adaptat.md)
- [\[ICCV 2025\] LATTE: Collaborative Test-Time Adaptation of Vision-Language Models in Federated Learning](../../ICCV2025/multimodal_vlm/latte_collaborative_test-time_adaptation_of_vision-language_models_in_federated_.md)
- [\[AAAI 2026\] Panda: Test-Time Adaptation with Negative Data Augmentation](../../AAAI2026/multimodal_vlm/panda_test-time_adaptation_with_negative_data_augmentation.md)
- [\[NeurIPS 2025\] TOMCAT: Test-time Comprehensive Knowledge Accumulation for Compositional Zero-Shot Learning](tomcat_test-time_comprehensive_knowledge_accumulation_for_compositional_zero-sho.md)

</div>

<!-- RELATED:END -->
