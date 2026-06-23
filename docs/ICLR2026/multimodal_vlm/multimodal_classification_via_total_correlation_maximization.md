---
title: >-
  [Paper Note] Multimodal Classification via Total Correlation Maximization
description: >-
  [ICLR 2026][Multimodal VLM][Paper Note] This paper analyzes the modality competition problem in multimodal classification from an information-theoretic perspective. It proposes the TCMax loss function, which maximizes the Total Correlation (TC) between multimodal features and labels. By simultaneously addressing joint learning, unimodal learning, and cross-m
tags:
  - ICLR 2026
  - Multimodal VLM
date: 2026-05-08
content_hash: cc10da415ceb4f8c
---
# Multimodal Classification via Total Correlation Maximization

**Conference**: ICLR 2026  
**arXiv**: [2602.13015](https://arxiv.org/abs/2602.13015)  
**Code**: [https://github.com/hubaak/TCMax](https://github.com/hubaak/TCMax)  
**Area**: Multimodal VLM  
**Keywords**: Multimodal Learning, Modality Competition, Total Correlation, Information Theory, Loss Function Design

## TL;DR
This paper analyzes the modality competition problem in multimodal classification from an information-theoretic perspective. It proposes the TCMax loss function, which maximizes the Total Correlation (TC) between multimodal features and labels. By simultaneously addressing joint learning, unimodal learning, and cross-modal alignment, it outperforms SOTA on several audio-visual and image-text classification benchmarks.

## Background & Motivation

**Background**: Multimodal learning aims to obtain more robust representations by fusing different modalities (e.g., audio + visual, text + image). The mainstream approach is joint learning, which utilizes a shared prediction head to classify all fusion features.

**Limitations of Prior Work**: Joint learning suffers from "modality competition," where certain modalities converge faster (e.g., audio), leading to the suppression of other modalities (e.g., visual). Consequently, the multimodal model may even perform worse than the best unimodal counterpart. Existing balancing methods like OGM-GE (gradient modulation) and AGM (adaptive gradient) alleviate this but do not fundamentally solve the "modality laziness" problem.

**Key Challenge**: Joint learning maximizes $I(y; z^{(a)}, z^{(v)})$. When the audio encoder has learned sufficient information ($I(y; z^{(a)}) \approx H(y)$), the upper bound of the conditional mutual information that the visual encoder can learn, $I(y; z^{(v)}|z^{(a)})$, approaches zero. Essentially, the optimization objective causes the strong modality to "squeeze out" the learning space of the weak modality.

**Goal**: How to design a loss function that avoids modality competition (allowing each modality to independently learn sufficient information) while leveraging cross-modal interaction (unlike pure unimodal learning, which is completely isolated), all without requiring extra hyperparameters or structural modifications?

**Key Insight**: The authors observe from information theory that Total Correlation (TC) can be naturally decomposed into three terms: "joint learning + unimodal learning + cross-modal alignment." This exactly covers the respective strengths of existing methods.

**Core Idea**: Replace mutual information with Total Correlation as the optimization objective. By maximizing $\text{TC}(z^{(a)}, z^{(v)}, y)$, joint learning, unimodal learning, and modal alignment are achieved simultaneously without additional hyperparameters.

## Method

### Overall Architecture
TCMax addresses the "modality competition" problem where strong modalities dominate weak ones by **changing only the training objective** rather than the network structure. The overall data flow remains identical to standard multimodal models: multimodal data $(x^{(1)}, \dots, x^{(M)})$ is input, each modality has an encoder $\psi^{(m)}$ mapping it to a feature $z^{(m)}$, which are then fed into a shared prediction head $f_\theta$ to output label probabilities.

The distinction lies entirely in the loss function. The authors identify the root cause of modality competition through information theory (the joint learning target for mutual information starves weak modalities). They switch the optimization target from "mutual information between features and labels" to their "Total Correlation (TC)," because the mathematical decomposition of TC naturally encompasses joint learning, unimodal learning, and cross-modal alignment. Since TC is not directly calculable for high-dimensional features, the authors extend the MINE (Mutual Information Neural Estimation) approach to derive an optimizable lower bound. They cleverly utilize the **prediction head itself as the TC estimator**: during training, positive samples are real multimodal pairs, while negative samples are "pseudo-pairs" created by randomly regrouping features from different samples. The prediction head performs classification while simultaneously separating these two types of samples in the output, which is equivalent to raising the TC lower bound. At inference, no extra operations are needed; the Softmax output is used directly.

### Key Designs

**1. Modality Competition Analysis from an Info-Theory Perspective: Attributing "Modality Laziness" to the Objective Itself**

Existing balancing methods (OGM-GE, AGM) only adjust gradients without explaining the origin of modality competition. The authors decompose the joint learning target: it maximizes $I(y; z^{(a)}, z^{(v)}) = I(y; z^{(a)}) + I(y; z^{(v)}|z^{(a)})$. When a fast-converging modality like audio learns most of the label information such that $I(y; z^{(a)}) \approx H(y)$, the upper bound for the visual conditional mutual information $I(y; z^{(v)}|z^{(a)})$ is pushed toward zero. The visual encoder then has no space left to learn. This indicates that modality competition is not an optimization trick issue but an inherent property of the joint learning target.

**2. Total Correlation Decomposition: Unifying Joint Learning, Unimodal Learning, and Alignment**

Given the structural flaws of a single mutual information target, the authors adopt Total Correlation. For the two-modality case, $\text{TC}(z^{(a)}, z^{(v)}, y) = I(y; z^{(a)}, z^{(v)}) + I(z^{(a)}; z^{(v)})$, which includes a joint learning term and a cross-modal alignment term. Alternatively, it equals $I(y; z^{(a)}) + I(y; z^{(v)}) + I(z^{(a)}; z^{(v)}|y)$, containing two independent unimodal learning terms and a conditional alignment term. Thus, maximizing a single TC is mathematically equivalent to optimizing "joint learning + unimodal learning + modal alignment" simultaneously—three goals that existing methods require multiple losses and hyperparameters to achieve.

**3. Total Correlation Neural Estimation (TCNE): An Optimizable Lower Bound for Incalculable TC**

Directly calculating TC requires the density ratio of the joint distribution to the product of marginals, which is infeasible in high-dimensional spaces. The authors extend MINE from bivariate to multivariate cases. Using the Donsker-Varadhan representation:

$$\text{TC} \geq \sup_\theta \mathbb{E}_{\mathbb{P}_{joint}}[T_\theta] - \log\big(\mathbb{E}_{\mathbb{P}_{product}}[e^{T_\theta}]\big)$$

where $T_\theta$ is a neural network statistic. Maximizing TC thus transforms into training $T_\theta$ to maximize the gap between samples from the joint distribution and samples from the product of independent marginals, relying entirely on sample estimation.

**4. TCMax Loss Function: Prediction Head as TC Estimator with Zero Extra Parameters**

To avoid introducing extra architectures, the authors repurpose the prediction head: $T_\theta(z^{(1)}, \dots, z^{(M)}, y) = f_\theta(z^{(1)}, \dots, z^{(M)})_y$. This treats the output of the classification head at the ground-truth label dimension as the statistic. Substituting this into the lower bound and taking the negative yields the training loss:

$$\mathcal{L}_{\text{TCMax}} = -\mathbb{E}_{\mathbb{P}_{joint}}[F_\Theta] + \log\big(\mathbb{E}_{\mathbb{P}_{product}}[e^{F_\Theta}]\big)$$

During training, positive samples come from the real joint distribution, while negative samples are "pseudo-pairs" from randomly regrouped features. TCMax replaces the loss without adding network parameters; the head performs both classification and TC estimation. Inference remains a standard Softmax.

### Loss & Training

A direct implementation of TCMax requires $|B|^M$ forward passes (the denominator must enumerate all combinations), which is computationally expensive. Two optimization strategies are used:

- **Negative Sampling**: Randomly sample $\mathcal{N}$ negative pairs from $\mathcal{B} \times \mathcal{B}$, reducing $O(B^M)$ to $O(|\mathcal{N}|)$.
- **Linear Fusion Decoupling**: If the head is $f_\theta(z^{(a)}, z^{(v)}) = f^{(a)}(z^{(a)}) + f^{(v)}(z^{(v)})$, the denominator can be decomposed into independent sums per modality, reducing complexity to $O(|B|)$.

Theoretical Guarantees: (1) Minimizing $\mathcal{L}_{\text{TCMax}}$ is equivalent to increasing the TC lower bound; (2) when the TC estimator is optimal, the model accurately estimates the joint distribution (Propositions 2-3); (3) no extra operations at inference.

## Key Experimental Results

### Main Results

Evaluated on 5 audio-visual/image-text datasets against 10+ methods using ResNet-18 trained from scratch:

| Dataset | Metric | TCMax (Share Head) | Prev. SOTA (MMPareto) | Gain |
|--------|------|------|----------|------|
| CREMA-D | Acc | **82.7** | 74.4 | +8.3% |
| Kinetics-Sounds | Acc | **63.5** | 62.7 | +0.8% |
| AVE | Acc | **64.5** | 63.1 | +1.4% |
| VGGSound | Acc | **47.6** | 46.2 | +1.4% |
| UCF101 | Acc | **56.0** | 55.9 | +0.1% |

### Ablation Study

| Configuration | Description |
|------|------|
| TCMax (Concat) | Uses concatenation fusion, also achieves competitive results |
| TCMax (Share Head) | Uses shared head fusion, overall best performance |
| Negative Sample Impact | CREMA-D optimal at 1024 samples; UCF101 at 256 |
| JS Divergence Analysis | TCMax shows the highest cross-modal prediction consistency (lowest JS divergence) |
| Entropy Balance | TCMax entropy ratio $\rho$ is closest to 1 (CREMA-D: 1.549 vs Concat: 2.913) |

### Key Findings
- The multimodal gain of TCMax primarily stems from cross-modal synergy rather than unimodal improvement: unimodal performance is on par with unimodal methods, but multimodal fusion is significantly better.
- JS divergence experiments confirm TCMax learns cross-modal alignment: prediction distributions across modalities are highly consistent.
- Training curves show TCMax loss values remain higher than joint/unimodal learning, effectively preventing overfitting.
- TCMax remains effective when using frozen CLIP encoders (on MVSA, ViT-B/32: 84.05 vs Joint: 82.83).

## Highlights & Insights
- **Unified Framework**: Via a single TC metric, it naturally unifies joint learning, unimodal learning, and alignment—three objectives usually requiring multiple losses and hyperparameter balancing. The elegance lies in this being a mathematical property of TC rather than an ad-hoc combination.
- **Zero Hyperparameters**: TCMax introduces no extra hyperparameters (unlike QMF's regularization weights or MMPareto's Pareto direction adjustments). It can directly replace cross-entropy, significantly reducing tuning costs.
- **TCNE to MINE Generalization**: Extending MINE from bivariate to multivariate is a natural yet valuable theoretical contribution, transferable to any scenario requiring multivariate dependency measurement (e.g., multi-task or multi-view learning).
- **Linear Fusion Decoupling Trick**: Utilizing $\exp(a+b) = \exp(a)\exp(b)$ to reduce negative sample complexity from $O(|B|^2)$ to $O(|B|)$ is a clever trick transferable to other contrastive learning settings.

## Limitations & Future Work
- The authors acknowledge TCMax is currently limited to classification tasks and cannot be directly extended to detection or generation—redefinition of input-output probability distributions would be required.
- Experiments mostly use ResNet-18 from scratch; there is a lack of validation on large-scale pretrained models like ViT-L or Large Multimodal Models (CLIP experiments only involved frozen encoders).
- The choice of negative sample count is dataset-dependent (1024 for CREMA-D vs 256 for UCF101), lacking an adaptive selection mechanism.
- Computational overhead for non-linear fusion heads remains $O(|\mathcal{N}|)$, which may become a bottleneck in large-batch or many-modality scenarios.
- The datasets used are relatively small (max VGGSound ~150k); scalability has not been verified on million-scale datasets.

## Related Work & Insights
- **vs OGM-GE/AGM**: These methods balance modality contributions via gradient modulation, addressing "symptoms" (gradient imbalance) rather than the "cause" (objective function flaws). TCMax is more fundamental by redesigning the objective.
- **vs QMF/MLA**: These explicitly introduce unimodal losses + regularization, requiring hyperparameter balancing. TCMax naturally includes unimodal targets via TC decomposition.
- **vs MMPareto**: MMPareto uses Pareto optimization to balance multiple objectives, whereas TCMax provides a singular, simplified unified target.
- **vs Contrastive Learning (InfoNCE)**: InfoNCE is a special case of MINE (fixed functional form). TCMax can be viewed as the natural multivariate generalization of InfoNCE from pair-wise to multi-variable.

## Rating
- Novelty: ⭐⭐⭐⭐ The info-theory perspective is not entirely new, but the insight of TC unifying three targets is profound.
- Experimental Thoroughness: ⭐⭐⭐⭐ 5 datasets and extensive analysis, though lacking large-scale validation.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear and rigorous theoretical derivation with progressive motivation.
- Value: ⭐⭐⭐⭐ A practical, hyperparameter-free loss function, though restricted to classification.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] MMTok: Multimodal Coverage Maximization for Efficient Inference of VLMs](mmtok_multimodal_coverage_maximization_for_efficient_inference_of_vlms.md)
- [\[ACL 2026\] MONETA: Multimodal Industry Classification through Geographic Information with Multi Agent Systems](../../ACL2026/multimodal_vlm/moneta_multimodal_industry_classification_through_geographic_information_with_mu.md)
- [\[NeurIPS 2025\] Rethinking Multimodal Learning from the Perspective of Mitigating Classification Ability Disproportion](../../NeurIPS2025/multimodal_vlm/rethinking_multimodal_learning_from_the_perspective_of_mitig.md)
- [\[CVPR 2026\] CC-VQA: Conflict- and Correlation-Aware Method for Mitigating Knowledge Conflict in Knowledge-Based Visual Question Answering](../../CVPR2026/multimodal_vlm/cc-vqa_conflict-_and_correlation-aware_method_for_mitigating_knowledge_conflict_.md)
- [\[ICCV 2025\] Enhancing Few-Shot Vision-Language Classification with Large Multimodal Model Features](../../ICCV2025/multimodal_vlm/enhancing_few-shot_vision-language_classification_with_large_multimodal_model_fe.md)

</div>

<!-- RELATED:END -->
