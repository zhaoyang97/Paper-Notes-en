---
title: >-
  [Paper Note] Panda: Test-Time Adaptation with Negative Data Augmentation
description: >-
  [AAAI2026][Multimodal VLM][test-time adaptation] This paper proposes Panda, which generates semantics-destroying but corruption-preserving images via patch shuffling as negative data augmentation (NDA)…
tags:
  - "AAAI2026"
  - "Multimodal VLM"
  - "test-time adaptation"
  - "negative data augmentation"
  - "CLIP"
  - "prediction bias"
  - "corruption robustness"
date: 2026-05-08
content_hash: 38084c7730713091
---

# Panda: Test-Time Adaptation with Negative Data Augmentation

**Conference**: AAAI2026
**arXiv**: [2511.10481](https://arxiv.org/abs/2511.10481)
**Code**: [ruxideng/Panda](https://github.com/ruxideng/Panda)
**Area**: Multimodal VLM
**Keywords**: test-time adaptation, negative data augmentation, CLIP, prediction bias, corruption robustness

## TL;DR
This paper proposes Panda, which generates semantics-destroying but corruption-preserving images via patch shuffling as negative data augmentation (NDA), and uses their features to offset original embeddings to suppress corruption-induced prediction bias. Panda is plug-and-play with less than 10% computational overhead and consistently improves various TTA methods.

## Background & Motivation
Pre-trained VLMs such as CLIP suffer significant performance degradation under image corruptions, primarily because corruption patterns are encoded as spurious features, causing systematic **prediction bias** toward certain classes. For instance, on CIFAR-10-C, corruptions substantially shift the predicted label distribution away from the ground truth (as measured by a notably larger $L^1$ distance).

Existing TTA methods rely heavily on positive data augmentation (PDA), e.g., AugMix generates $K=63$ semantics-preserving views per image:
- **High computational cost**: independently generating $K$ augmented views per image increases forward passes by a factor of $K+1$.
- **Inability to eliminate bias**: since PDA preserves both semantics and corruption, averaging may amplify rather than reduce the bias.
- Prediction bias is particularly detrimental to entropy-based TTA methods, potentially causing pseudo-label bias to accumulate and lead to model collapse.

## Method

### Negative Data Augmentation (NDA)
1. Each image in a batch of $B$ images is divided into $\frac{H}{H_p} \times \frac{W}{W_p}$ patches (default $H_p=W_p=32$, i.e., $7\times7$ patches).
2. All patches are pooled into a shared pool, randomly reassembled into $M$ negative augmentation images ($M = B/10 \ll B$).
3. The resulting images destroy object semantics while retaining corruption characteristics.

### Feature Offset
- Negative augmented images are encoded to obtain $\{\mathbf{n}_j\}_{j=1}^M$; their mean is computed as $\bar{\mathbf{n}} = \frac{1}{M}\sum_j \mathbf{n}_j$.
- The original embedding is offset as: $\mathbf{d}_i = \mathbf{v}_i - \beta \cdot \bar{\mathbf{n}}$.
- Theoretical guarantee (Theorem 4.1): when the correlation between the negative augmentation and the corruption is $r>0$ and the augmentation carries no class-discriminative information, the offset strategy reduces the corruption component by a factor of $\sqrt{1-r^2}$, with optimal $\beta = r$.

### Integration with TTA Methods
Panda only modifies the feature representation ($\mathbf{v}_i \to \mathbf{d}_i$) during the forward pass and can be seamlessly incorporated into arbitrary TTA frameworks, including Tent, ETA, SAR, DeYO, TPT, and TPS. For entropy minimization methods (e.g., Tent), computing entropy using debiased logits simultaneously improves prediction quality and adaptation stability.

### Comparison with DeYO's NDA Strategy
DeYO also employs negative augmentation, but only to estimate prediction confidence for sample selection and weighting. Panda's NDA achieves higher quality (Table 4: offset strategy reaches 43.3% on CIFAR-100-C vs. DeYO's select & weight at 38.0%), and the two approaches are complementary.

## Key Experimental Results

Evaluations are conducted on CIFAR-10-C, CIFAR-100-C, and ImageNet-C (severity level 5), covering 9 TTA baselines.

### Table 1: Accuracy Improvement with +Panda (mean accuracy %)

| Dataset | CLIP | Tent | ETA | SAR | DeYO | Avg. Gain |
|---|---|---|---|---|---|---|
| CIFAR-10-C | +2.6 | **+8.3** | +3.4 | +7.4 | +1.7 | +3.3 |
| CIFAR-100-C | +1.6 | +2.7 | +2.5 | +2.6 | **+4.1** | +2.2 |
| ImageNet-C | +1.7 | **+2.9** | +1.4 | +0.6 | +2.2 | +2.0 |

### Efficiency Comparison (Table 3, ViT-B/32 on CIFAR-10)
- Panda adds less than 10% overhead: Tent 25s → 27s (+8.0%), TPT 22min → 22min39s (+1.3%).
- Compared to PDA-based methods (TPT/Zero/TPS requiring $K=63$ augmentations), Panda achieves 71.1% on CIFAR-10-C, substantially outperforming TPT (62.2%) and TPS (63.7%).

### Prediction Bias Elimination
- Tent accumulates bias on Gaussian noise until model collapse; Tent+Panda consistently maintains low bias and high accuracy.
- Across 15 corruption types, PDA reduces bias in only 4 cases, whereas Panda effectively reduces bias in all 15.

## Highlights & Insights
- **Counter-intuitive design**: improving robustness via semantics-destroying negative augmentation rather than semantics-preserving positive augmentation is a novel and effective perspective.
- **Minimal overhead**: $M=B/10$ augmented images are shared across the batch, adding less than 10% extra computation—a stark contrast to the $63\times$ cost of PDA.
- **Plug-and-play**: only the embedding in the forward pass is modified, making Panda compatible with all CLIP-based TTA algorithms.
- **Theoretical grounding**: a formal proof of accuracy improvement and a closed-form solution for the optimal $\beta$ are provided.

## Limitations & Future Work
- Validation is limited to CLIP-family VLMs; generalization to BLIP, SigLIP, and other VLMs remains unexplored.
- The theoretical analysis assumes Gaussian distributions; real-world corruption components may be more complex.
- The default patch size is aligned with ViT patches (32×32); applicability to non-standard input resolutions requires further investigation.
- Evaluation covers only image classification, without extension to downstream tasks such as detection or segmentation.
- While ablations show that hyperparameters $\beta$ and $M/B$ are largely insensitive, robustness under extreme corruptions (e.g., impulse noise) warrants further verification.

### Ablation Highlights
- Shuffling patches within a single image (rather than sharing across the batch) leads to a notable performance drop → batch-level information sharing is essential.
- Subtracting each negative feature individually without averaging also degrades performance → averaging effectively suppresses individual noise.
- Performance remains stable as $M/B$ decreases from $1/2$ to $1/100$ → a small number of negative augmentations suffices.
- $\beta$ in the range of 0.5–2.0 consistently outperforms the baseline without Panda.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — The combination of negative augmentation and feature offset is concise and effective, forming a sharp contrast with positive augmentation approaches.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — 3 datasets × 9 TTA baselines × 15 corruption types, with comprehensive ablation and sensitivity analyses.
- **Writing Quality**: ⭐⭐⭐⭐ — Intuitive illustrations and rigorous theoretical derivations.
- **Value**: ⭐⭐⭐⭐ — The plug-and-play nature provides immediate practical value to the TTA community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] DOTA: DistributiOnal Test-time Adaptation of Vision-Language Models](../../NeurIPS2025/multimodal_vlm/dota_distributional_testtime_adaptation_of_visionlanguage_mo.md)
- [\[NeurIPS 2025\] Mint: A Simple Test-Time Adaptation of Vision-Language Models against Common Corruptions](../../NeurIPS2025/multimodal_vlm/mint_a_simple_testtime_adaptation_of_visionlanguage_models_a.md)
- [\[NeurIPS 2025\] The Illusion of Progress? A Critical Look at Test-Time Adaptation for Vision-Language Models](../../NeurIPS2025/multimodal_vlm/the_illusion_of_progress_a_critical_look_at_testtime_adaptat.md)
- [\[ICCV 2025\] LATTE: Collaborative Test-Time Adaptation of Vision-Language Models in Federated Learning](../../ICCV2025/multimodal_vlm/latte_collaborative_test-time_adaptation_of_vision-language_models_in_federated_.md)
- [\[ICCV 2025\] Is Less More? Exploring Token Condensation as Training-free Test-time Adaptation](../../ICCV2025/multimodal_vlm/is_less_more_exploring_token_condensation_as_training-free_test-time_adaptation.md)

</div>

<!-- RELATED:END -->
