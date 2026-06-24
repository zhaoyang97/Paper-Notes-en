---
title: >-
  [Paper Note] PSBD: Prediction Shift Uncertainty Unlocks Backdoor Detection
description: >-
  [CVPR 2025][AI Safety][Backdoor Detection] This paper proposes PSBD, which discovers that during inference with dropout enabled, a backdoor-infected model shifts its predictions on clean data toward the target class while maintaining stable predictions on backdoored data (the Prediction Shift phenomenon). Based on this insight, the authors design the Prediction Shift Uncertainty (PSU) metric to achieve SOTA backdoor training data detection.
tags:
  - "CVPR 2025"
  - "AI Safety"
  - "Backdoor Detection"
  - "Prediction Shift"
  - "Dropout Uncertainty"
  - "Neuron Bias"
  - "Adversarial Robustness"
date: 2026-05-08
content_hash: 26eb2c677aec16d2
---

# PSBD: Prediction Shift Uncertainty Unlocks Backdoor Detection

**Conference**: CVPR 2025  
**arXiv**: [2406.05826](https://arxiv.org/abs/2406.05826)  
**Code**: [GitHub](https://github.com/WL-619/PSBD)  
**Area**: AI Safety / Backdoor Detection  
**Keywords**: Backdoor Detection, Prediction Shift, Dropout Uncertainty, Neuron Bias, Adversarial Robustness

## TL;DR

This paper proposes PSBD, which discovers that during inference with dropout enabled, a backdoor-infected model shifts its predictions on clean data toward the target class while maintaining stable predictions on backdoored data (the Prediction Shift phenomenon). Based on this insight, the authors design the Prediction Shift Uncertainty (PSU) metric to achieve SOTA backdoor training data detection.

## Background & Motivation

Deep neural networks are susceptible to backdoor attacks, where adversaries insert malicious samples containing specific triggers into the training data. This causes the model to make attacker-specfied incorrect predictions when encountering the trigger while behaving normally on clean inputs. The stealthiness of backdoor attacks poses a serious threat in safety-critical domains such as autonomous driving and medical imaging.

Existing defense strategies mainly focus on three directions: model reconstruction (removing backdoor effects), model detection (determining if a model is backdoored), and poisoning mitigation. However, in the primary direction of **backdoor training data detection**, existing methods generally face two challenges: either low True Positive Rates (TPR) — missing backdoor samples — or high False Positive Rates (FPR) — falsely removing clean samples. Existing methods, such as Spectral Signatures, STRIP, and Scale-up, operate primarily at the **data level** (altering inputs or analyzing representations) without fully utilizing the intrinsic properties of the model itself.

This paper offers a brand-new perspective: **model prediction uncertainty**. The authors discover a compelling phenomenon called Prediction Shift (PS): enabling dropout during inference causes a poisoned model's predictions on clean data to shift from the correct labels to the target class, while predictions on backdoored data remain stable. This phenomenon originates from the "neuron bias" effect, where certain neuron pathways become biased toward a specific class during training. Based on this insight, PSBD achieves simple yet highly effective backdoor data detection by calculating PSU.

## Method

### Overall Architecture

The workflow of PSBD: (1) train a model using standard supervised learning on the suspicious training set (optionally with data augmentation); (2) adaptively select an appropriate dropout rate $p$; (3) compute the PSU values for the training data and a small unlabeled clean validation set; (4) identify samples with PSU values below a threshold $T$ (the 25th percentile of the validation set PSU) as backdoor samples.

### Key Designs

1. **Discovery of the Prediction Shift (PS) Phenomenon**:
    - Function: Revealing the behavioral differences between clean and backdoored data when dropout is enabled.
    - Mechanism: Define the prediction shift function $\phi_{PS}(\mathbf{x}) = \mathbb{I}(\mathcal{Y}(\mathbf{x};\boldsymbol{\theta}) \neq \mathcal{Y}(\mathbf{x};\boldsymbol{\theta}'))$ and the shift rate $\sigma(\mathcal{D}) = \frac{1}{k|\mathcal{D}|}\sum_{\mathbf{x} \in \mathcal{D}} \phi_{PS}(\mathbf{x})$. Experiments reveal that under a suitable dropout rate $p$, the $\sigma$ of clean data reaches approximately 0.8 and almost entirely shifts to the attacker's target class, whereas the $\sigma$ of backdoored data remains close to 0. This disparity is consistently present across various attacks, such as BadNets and WaNet.
    - Design Motivation: Standard MC-Dropout uncertainty (standard deviation) fails under advanced attacks (e.g., WaNet). The PS phenomenon provides a more robust discriminative signal.

2. **Neuron Bias Effect**:
    - Function: Explaining the mechanism underlying the PS phenomenon.
    - Mechanism: Backdoor training biases certain pathways in the network toward the target class. Without dropout, clean data has sufficient features to yield correct predictions. With dropout enabled, critical discriminative features are dropped, forcing the model to rely on the neuron bias formed during training, thereby classifying clean data into the target class. In contrast, the trigger features of backdoored data are more stable and salient, allowing correct classification into the target class even if some features are discarded by dropout. The authors validate this by visualizing the 512 feature maps of the final layer: after enabling dropout, the features of clean and backdoored images become almost identical.
    - Design Motivation: To provide a theoretical explanation for the PS phenomenon, enhancing the credibility and explainability of the proposed method.

3. **Prediction Shift Uncertainty (PSU) Detection Method**:
    - Function: Quantifying the intensity of prediction shift for backdoor detection.
    - Mechanism: PSU calculates the difference between the confidence of the highest-confidence class $c$ without dropout and the average confidence of that class over $k$ runs of dropout inference: $\phi_{PSU}(\mathbf{x}) = P_c(\mathbf{x};\boldsymbol{\theta}) - \frac{1}{k}\sum_{i=1}^{k}P_c(\mathbf{x};p,\boldsymbol{\theta}_i')$. Clean data exhibits high PSU values (large prediction shift), while backdoored data shows low PSU values (stable predictions). The threshold $T$ is set as the 25th percentile of the validation set PSU. The dropout rate $p$ is selected via an adaptive strategy to find the $p$ value where the validation set $\sigma$ is close to 0.8 and the gap between training and validation shift rates is maximized.
    - Design Motivation: This considers not only label changes (PS) but also confidence shifts, capturing finer-grained signals (e.g., some clean samples whose labels remain unchanged but whose confidence drops significantly).

### Loss & Training

- **Standard Cross-Entropy Loss**: Normal training on the suspicious training set.
- **Dropout Position**: Add dropout layers after each residual connection and before the activation function in ResNet.
- **Inference Iterations**: $k=3$ forward passes.
- **Data Augmentation**: Used when the model's generalization capability is insufficient (e.g., on Tiny ImageNet and Adaptive-Blend) to enhance the neuron bias.
- **Model Selection**: Select the model from the later stages of training (to reinforce data fitting and the neuron bias pathways).

## Key Experimental Results

### Main Results

CIFAR-10 Dataset (10% poisoning rate, TPR↑ / FPR↓):

| Attack Method | PSBD (Ours) | SS | STRIP | SCAN | SCP | CD-L |
|---------|-------------|-----|-------|------|-----|------|
| BadNets | **1.000/0.104** | 0.389/0.512 | 1.000/0.113 | 1.000/0.009 | 1.000/0.205 | 0.998/0.158 |
| WaNet | **1.000/0.116** | 0.456/0.505 | 0.050/0.101 | 0.891/0.034 | 0.869/0.251 | 0.863/0.144 |
| Adaptive-Blend | **0.982/0.184** | 0.608/0.145 | 0.014/0.069 | 0.000/0.023 | 0.721/0.257 | 0.432/0.167 |
| **Average** | **0.994/0.136** | 0.439/0.456 | 0.689/0.107 | 0.832/0.013 | 0.899/0.244 | 0.855/0.157 |

### Ablation Study

| Configuration | Explanation |
|------|------|
| MC-Dropout Standard Deviation | Fails under attacks like WaNet; uncertainties of backdoored and clean samples are close. |
| PS (Label Change Only) | Effective but lacks granularity; some clean samples do not change labels but experience significant confidence drops. |
| PSU (Label + Confidence) | Finest granularity, achieving the highest coverage. |
| No Data Augmentation (Tiny ImageNet) | Detection performance degrades when model generalization is insufficient. |
| With Data Augmentation (Tiny ImageNet) | Reinforces neuron bias, significantly improving detection performance. |

### Key Findings

- PSBD achieves the highest average TPR across 7 attacks $\times$ 3 datasets, showing a massive advantage especially against advanced attacks (WaNet, Adaptive-Blend). For instance, STRIP achieves a TPR of only 0.050 on WaNet, and SCAN scores a TPR of 0 on Adaptive-Blend.
- Only a small unlabeled clean validation set (5% of the training set size) is required.
- The PS direction almost entirely shifts to the attack's target class (class 0), which is a surprisingly consistent pattern.
- Data augmentation can reinforce the neuron bias effect, thereby assisting detection.

## Highlights & Insights

- **The discovery of the Prediction Shift phenomenon is highly inspiring**: dropout causes clean data to "fall into" the gravitational field of the backdoor target class. This reveals deep imprints left by backdoor attacks in the weight space, offering a new perspective for understanding backdoor mechanisms.
- **The method is exceptionally simple and practical**: it only requires enabling dropout during inference to compute PSU across 3 forward passes, without needing to train auxiliary models or optimize trigger templates, keeping time overhead to a minimum.
- **Outstanding robustness against advanced attacks**: PSBD maintains a TPR $>0.98$ on WaNet and Adaptive-Blend, where STRIP and SCAN fail completely.

## Limitations & Future Work

- The FPR remains relatively high (averaging around 13-20%), which may result in the accidental removal of some clean training data.
- The adaptive selection of the dropout rate $p$ relies on a heuristic threshold ($\sigma$ close to 0.8), which may be inaccurate in certain scenarios.
- On complex datasets like Tiny ImageNet, it requires integration with data augmentation to achieve good results.
- Performance has not yet been verified on larger-scale models (such as ViT-Large) and a wider variety of attacks.

## Related Work & Insights

- **vs Spectral Signatures (SS)**: SS utilizes feature statistics to distinguish between clean and backdoored samples, but its TPR is generally low (averaging only 0.44). In contrast, PSBD leverages model-level uncertainty to reach a TPR of 0.99.
- **vs STRIP**: STRIP analyzes prediction entropy via raw specimen blending but fails completely on WaNet (TPR = 0.050). PSBD maintains high TPRs across all types of attacks.
- **vs SCAN**: SCAN achieves an extremely low FPR (0.013) on CIFAR-10, but its TPR on Adaptive-Blend is 0, and it suffers from heavy computational overhead (timing out on GTSRB).

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The discovery of the PS phenomenon and the neuron bias hypothesis are highly original.
- Experimental Thoroughness: ⭐⭐⭐⭐ 7 attacks $\times$ 3 datasets $\times$ 6 baselines, with 10 repeated runs.
- Writing Quality: ⭐⭐⭐⭐ Smooth narrative logic progressing from the pilot study to findings, and ultimately to the method.
- Value: ⭐⭐⭐⭐⭐ The simple yet highly effective method achieves breakthrough progress in detecting advanced attacks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Don't Shift the Trigger: Robust Gradient Ascent for Backdoor Unlearning](../../ICLR2026/ai_safety/dont_shift_the_trigger_robust_gradient_ascent_for_backdoor_unlearning.md)
- [\[ICML 2025\] Disparate Conditional Prediction in Multiclass Classifiers](../../ICML2025/ai_safety/disparate_conditional_prediction_in_multiclass_classifiers.md)
- [\[CVPR 2025\] INACTIVE: Invisible Backdoor Attack against Self-supervised Learning](invisible_backdoor_attack_against_self-supervised_learning.md)
- [\[CVPR 2025\] Infighting in the Dark: Multi-Label Backdoor Attack in Federated Learning](infighting_in_the_dark_multi-label_backdoor_attack_in_federated_learning.md)
- [\[CVPR 2025\] Towards General Visual-Linguistic Face Forgery Detection](towards_general_visual-linguistic_face_forgery_detection.md)

</div>

<!-- RELATED:END -->
