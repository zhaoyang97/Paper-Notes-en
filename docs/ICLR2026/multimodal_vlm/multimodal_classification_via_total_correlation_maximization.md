---
title: >-
  [Paper Note] Multimodal Classification via Total Correlation Maximization
description: >-
  [ICLR 2026][Multimodal VLM][multimodal learning] This paper analyzes modality competition in multimodal classification from an information-theoretic perspective and proposes TCMax…
tags:
  - "ICLR 2026"
  - "Multimodal VLM"
  - "multimodal learning"
  - "modality competition"
  - "total correlation"
  - "information theory"
  - "loss function design"
date: 2026-05-08
content_hash: 20550135d6f4decd
---

# Multimodal Classification via Total Correlation Maximization

**Conference**: ICLR 2026
**arXiv**: [2602.13015](https://arxiv.org/abs/2602.13015)  
**Code**: [https://github.com/hubaak/TCMax](https://github.com/hubaak/TCMax)  
**Area**: Multimodal VLM
**Keywords**: multimodal learning, modality competition, total correlation, information theory, loss function design

## TL;DR
This paper analyzes modality competition in multimodal classification from an information-theoretic perspective and proposes TCMax, a loss function that maximizes the Total Correlation (TC) between multimodal features and labels. TCMax simultaneously addresses joint learning, unimodal learning, and cross-modal alignment without additional hyperparameters, surpassing state-of-the-art methods on multiple audio-visual and image-text classification benchmarks.

## Background & Motivation

**Background**: Multimodal learning acquires more robust representations by fusing different modalities (e.g., audio + visual, text + image). The dominant paradigm is joint learning, where a shared prediction head classifies over all modality features simultaneously.

**Limitations of Prior Work**: Joint learning suffers from *modality competition*—certain modalities converge faster (e.g., audio), suppressing others (e.g., visual), such that the multimodal model can underperform the best unimodal model. Existing balancing methods such as OGM-GE (gradient modulation) and AGM (adaptive gradients) alleviate but do not fundamentally resolve the problem of "modality laziness."

**Key Challenge**: Joint learning maximizes $I(y; z^{(a)}, z^{(v)}) = I(y; z^{(a)}) + I(y; z^{(v)}|z^{(a)})$. Once the audio encoder has captured sufficient information ($I(y; z^{(a)}) \approx H(y)$), the upper bound of the conditional mutual information $I(y; z^{(v)}|z^{(a)})$ approaches zero, leaving no learning signal for the visual encoder. The optimization objective itself causes the dominant modality to crowd out the weaker one.

**Goal**: How to design a loss function that (1) avoids modality competition by enabling each modality to independently learn sufficient information, (2) exploits cross-modal interaction without completely isolating modalities as in pure unimodal learning, and (3) requires no additional hyperparameters or architectural modifications?

**Key Insight**: From an information-theoretic perspective, the authors observe that Total Correlation naturally decomposes into three terms corresponding to joint learning, unimodal learning, and cross-modal alignment—precisely covering the individual strengths of existing methods.

**Core Idea**: Replace mutual information with Total Correlation as the optimization objective. Maximizing $\text{TC}(z^{(a)}, z^{(v)}, y)$ simultaneously achieves joint learning, unimodal learning, and cross-modal alignment without any additional hyperparameters.

## Method

### Overall Architecture
Given multimodal input $(x^{(1)}, \dots, x^{(M)})$, each modality is encoded by a dedicated encoder $\psi^{(m)}$ into a feature space $z^{(m)}$, which is then passed to a shared prediction head $f_\theta$ to produce label probabilities. During training, the standard cross-entropy loss is replaced by TCMax. At inference, no modifications are required; the model outputs standard Softmax predictions.

### Key Designs

1. **Information-Theoretic Analysis of Modality Competition**

    - **Function**: Explains the root cause of modality competition via mutual information decomposition.
    - **Mechanism**: Joint learning maximizes $I(y; z^{(a)}, z^{(v)}) = I(y; z^{(a)}) + I(y; z^{(v)}|z^{(a)})$. When $I(y; z^{(a)}) \approx H(y)$, the upper bound of $I(y; z^{(v)}|z^{(a)})$ approaches zero, leaving no learning space for the visual encoder.
    - **Design Motivation**: Identifies the theoretical deficiency of joint learning—the optimization objective itself induces modality competition.

2. **Total Correlation Decomposition**

    - **Function**: Unifies joint learning, unimodal learning, and cross-modal alignment into a single TC objective.
    - **Mechanism**: For the two-modality case, $\text{TC}(z^{(a)}, z^{(v)}, y) = I(y; z^{(a)}, z^{(v)}) + I(z^{(a)}; z^{(v)})$, which also equals $I(y; z^{(a)}) + I(y; z^{(v)}) + I(z^{(a)}; z^{(v)}|y)$. The first decomposition encompasses joint learning and alignment; the second encompasses unimodal learning and conditional alignment.
    - **Design Motivation**: TC naturally subsumes the individual advantages of existing methods, providing a unified and conflict-free optimization objective.

3. **Total Correlation Neural Estimation (TCNE)**

    - **Function**: Provides a computable lower bound for TC.
    - **Mechanism**: Extends MINE (Mutual Information Neural Estimation) from bivariate to multivariate settings. Using the Donsker–Varadhan representation theorem: $\text{TC} \geq \sup_\theta \mathbb{E}_{\mathbb{P}_{joint}}[T_\theta] - \log(\mathbb{E}_{\mathbb{P}_{product}}[e^{T_\theta}])$, where $T_\theta$ is a neural network.
    - **Design Motivation**: Direct computation of TC requires knowledge of the density ratio between the joint and marginal distributions, which is intractable in high-dimensional spaces. The variational lower bound circumvents this difficulty.

4. **TCMax Loss Function**

    - **Function**: Reuses the classification head as the statistics network $T_\theta$ in TCNE, yielding a loss with no additional parameters.
    - **Mechanism**: Setting $T_\theta(z^{(1)}, \dots, z^{(M)}, y) = f_\theta(z^{(1)}, \dots, z^{(M)})_y$ gives $\mathcal{L}_{\text{TCMax}} = -\mathbb{E}_{\mathbb{P}_{joint}}[F_\Theta] + \log(\mathbb{E}_{\mathbb{P}_{product}}[e^{F_\Theta}])$. Training requires contrasting positive samples (real samples from the joint distribution) against negative samples (random cross-sample combinations of modality features).
    - **Design Motivation**: Introduces no additional network parameters—the prediction head itself serves as the TC estimator, enabling a drop-in replacement for cross-entropy without any architectural change.

### Loss & Training

Direct implementation of TCMax requires $|B|^M$ forward passes (the denominator enumerates all combinations of modality features), which is computationally expensive. Two optimization strategies are proposed:

- **Negative Sample Subsampling**: Randomly sample $\mathcal{N}$ negative pairs from $\mathcal{B} \times \mathcal{B}$, reducing complexity from $O(|B|^M)$ to $O(|\mathcal{N}|)$.
- **Linear Fusion Decoupling**: If the prediction head takes the form $f_\theta(z^{(a)}, z^{(v)}) = f^{(a)}(z^{(a)}) + f^{(v)}(z^{(v)})$, the denominator factorizes into independent per-modality sums, reducing complexity to $O(|B|)$.

Theoretical guarantees: (1) minimizing $\mathcal{L}_{\text{TCMax}}$ is equivalent to improving the TC lower bound; (2) when the TC estimator is optimal, the model accurately estimates the joint distribution (Propositions 2–3); (3) no additional operations are required at inference.

## Key Experimental Results

### Main Results

Comparison against 10+ methods on 5 audio-visual/image-text datasets using ResNet-18 trained from scratch:

| Dataset | Metric | Ours (Share Head) | Prev. SOTA (MMPareto) | Gain |
|--------|------|------|----------|------|
| CREMA-D | Acc | **82.7** | 74.4 | +8.3% |
| Kinetics-Sounds | Acc | **63.5** | 62.7 | +0.8% |
| AVE | Acc | **64.5** | 63.1 | +1.4% |
| VGGSound | Acc | **47.6** | 46.2 | +1.4% |
| UCF101 | Acc | **56.0** | 55.9 | +0.1% |

### Ablation Study

| Configuration | Description |
|------|------|
| TCMax (Concat) | Uses concatenation fusion; achieves competitive results |
| TCMax (Share Head) | Uses shared-head fusion; achieves overall best performance |
| Effect of negative sample count | Optimal at 1024 negatives on CREMA-D; optimal at 256 on UCF101 |
| JS divergence analysis | TCMax achieves the highest cross-modal prediction consistency (lowest JS divergence) across all datasets |
| Prediction entropy balance | TCMax yields the entropy ratio $\rho$ closest to 1 between strong and weak modalities (CREMA-D: 1.549 vs. Concat: 2.913) |

### Key Findings
- The multimodal gain of TCMax stems primarily from cross-modal synergy rather than unimodal improvement: unimodal performance is on par with unimodal baselines, but multimodal fusion is significantly superior.
- JS divergence experiments confirm that TCMax genuinely learns cross-modal alignment, as the prediction distributions of the two modalities are most consistent.
- Training curves show that TCMax maintains a consistently higher loss value than joint/unimodal learning, effectively preventing overfitting.
- TCMax remains effective with frozen CLIP encoders (MVSA, ViT-B/32: 84.05 vs. Joint: 82.83).

## Highlights & Insights
- **Unified Framework**: A single TC quantity naturally unifies joint learning, unimodal learning, and alignment—three objectives that typically require multiple losses and hyperparameter balancing. The key insight is that this unification is not artificially constructed but arises directly from the mathematical decomposition of TC.
- **Zero Hyperparameters**: TCMax introduces no additional hyperparameters (unlike QMF, which requires regularization weights, or MMPareto, which requires Pareto direction tuning) and can be used as a direct replacement for cross-entropy, substantially reducing tuning costs in practice.
- **TCNE as Multivariate Generalization of MINE**: Extending MINE from bivariate to multivariate settings is a natural yet valuable theoretical contribution, transferable to any scenario requiring measurement of multi-variable dependency (e.g., multi-task learning, multi-view representation learning).
- **Linear Fusion Decoupling Trick**: Exploiting $\exp(a+b) = \exp(a)\exp(b)$, when the prediction head uses linear fusion, negative sample computation is reduced from $O(|B|^2)$ to $O(|B|)$—a trick transferable to other contrastive learning settings.

## Limitations & Future Work
- TCMax is currently limited to classification tasks and cannot be directly extended to detection, generation, or other tasks, as these require redefining the probability distributions over inputs and outputs.
- Experiments rely exclusively on ResNet-18 trained from scratch; validation on large-scale pretrained models (e.g., ViT-L, large multimodal models) is lacking (the CLIP experiment only freezes the encoder).
- The optimal number of negative samples is dataset-dependent (1024 for CREMA-D, 256 for UCF101), and no adaptive selection mechanism is provided.
- For non-linear fusion heads, computational complexity remains $O(|\mathcal{N}|)$, which may become a bottleneck in large-batch or many-modality settings.
- All experimental datasets are relatively small (largest: VGGSound, ~150K samples); scalability on million-scale datasets has not been verified.

## Related Work & Insights
- **vs. OGM-GE/AGM**: These methods balance modality contributions via gradient modulation, addressing the symptom (gradient imbalance) rather than the cause (the deficiency of the optimization objective itself). TCMax redesigns the objective function at a more fundamental level.
- **vs. QMF/MLA**: These methods explicitly introduce unimodal losses and regularization terms, requiring hyperparameter balancing. TCMax naturally subsumes unimodal objectives through TC decomposition, without additional terms.
- **vs. MMPareto**: MMPareto applies Pareto optimization to balance multi-objective directions but still trades off among multiple independent objectives. TCMax replaces them with a single unified objective, yielding a simpler formulation.
- **vs. Contrastive Learning (InfoNCE)**: InfoNCE is a special case of MINE with a fixed functional form, whereas TCMax can be viewed as a natural generalization of InfoNCE from pairwise to multi-variable settings.

## Rating
- Novelty: ⭐⭐⭐⭐ The information-theoretic perspective is not entirely new, but the insight of unifying three objectives via TC is genuinely deep.
- Experimental Thoroughness: ⭐⭐⭐⭐ Five datasets with diverse analyses, but large-scale validation is absent.
- Writing Quality: ⭐⭐⭐⭐⭐ Theoretical derivations are clear and rigorous, with motivation developed in a well-structured progression.
- Value: ⭐⭐⭐⭐ A practically useful hyperparameter-free loss function, though currently limited to classification tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] MMTok: Multimodal Coverage Maximization for Efficient Inference of VLMs](mmtok_multimodal_coverage_maximization_for_efficient_inference_of_vlms.md)
- [\[ACL 2026\] MONETA: Multimodal Industry Classification through Geographic Information with Multi Agent Systems](../../ACL2026/multimodal_vlm/moneta_multimodal_industry_classification_through_geographic_information_with_mu.md)
- [\[NeurIPS 2025\] Rethinking Multimodal Learning from the Perspective of Mitigating Classification Ability Disproportion](../../NeurIPS2025/multimodal_vlm/rethinking_multimodal_learning_from_the_perspective_of_mitig.md)
- [\[ICLR 2026\] Reasoning-Driven Multimodal LLM for Domain Generalization](reasoning-driven_multimodal_llm_for_domain_generalization.md)
- [\[ICCV 2025\] Enhancing Few-Shot Vision-Language Classification with Large Multimodal Model Features](../../ICCV2025/multimodal_vlm/enhancing_few-shot_vision-language_classification_with_large_multimodal_model_fe.md)

</div>

<!-- RELATED:END -->
