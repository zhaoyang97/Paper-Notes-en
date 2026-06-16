---
title: >-
  [Paper Note] Drainage: A Unifying Framework for Addressing Class Uncertainty
description: >-
  [CVPR 2026][Others][Paper Note] By adding an extra "drainage node" to the output layer and a "drainage loss" generalized from cross-entropy, the framework allows ambiguous, out-of-distribution, or mislabeled samples to direct their probability mass into this node rather than being forced into an incorrect class. This approach achieves up to 10% highe
tags:
  - CVPR 2026
  - Others
date: 2026-05-08
content_hash: 3b3a168cf777942a
---
# Drainage: A Unifying Framework for Addressing Class Uncertainty

**Conference**: CVPR 2026  
**Paper**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Taha_Drainage_A_Unifying_Framework_for_Addressing_Class_Uncertainty_CVPR_2026_paper.html)  
**Code**: https://github.com/ZKI-PH-ImageAnalysis/Drainage  
**Area**: Learning with Noisy Labels / Robust Loss / Open-Set Recognition  
**Keywords**: Label Noise, Class Uncertainty, Drainage Node, Robust Loss, Open-Set Recognition

## TL;DR
By adding an extra "drainage node" to the output layer and a "drainage loss" generalized from cross-entropy, the framework allows ambiguous, out-of-distribution, or mislabeled samples to direct their probability mass into this node rather than being forced into an incorrect class. This approach achieves up to 10% higher accuracy in high-noise scenarios compared to existing robust losses and directly serves as a rejector for open-set recognition.

## Background & Motivation

**Background**: The mainstream classification paradigm (ResNet/ViT + softmax + cross-entropy) assumes a fixed set of classes. The network output is a discrete probability distribution over these classes, and training maximizes the consistency between predictions and labels. This paradigm is concise and effective, serving as the default for almost all classification datasets and models.

**Limitations of Prior Work**: Real-world data does not always fit cleanly into fixed class schemes. Some samples naturally lie on the boundary between two classes where annotators must choose arbitrarily (intra-class ambiguity); some are incorrectly labeled due to negligence or lack of domain knowledge; and others do not belong to any known category. In all these cases, cross-entropy **forces** the network to pick one of the fixed categories, injecting gradients from these "dirty" samples into specific classes.

**Key Challenge**: The authors point out that even specialized robust losses like GCE, SCE, APL, and ANL essentially impose a **uniform penalty** on uncertain/mislabeled samples. The network lacks an **"escape route"** in high-uncertainty regions—it cannot state "I am uncertain" and is instead forced to make a confident prediction. This represents a performance ceiling for various robust loss designs, particularly under asymmetric and instance-dependent noise.

**Goal**: To provide the network with a principled escape outlet, allowing it to actively decouple samples with "low information / high ambiguity / wrong labels" from class predictions while maintaining end-to-end differentiability and the ability to degrade seamlessly into a standard classifier.

**Key Insight**: Since the problem stems from "probability mass having nowhere to go," a dedicated destination should be provided. The authors introduce an additional neuron in the output layer—the drainage node $z_d$—specifically to absorb "excess" probability that cannot be reasonably assigned to any class.

**Core Idea**: Use a "drainage node + drainage loss" to **explicitly separate** class uncertainty from class probabilities, allowing ambiguous or mislabeled samples to flow toward the drainage node instead of being arbitrarily assigned to an incorrect class.

## Method

### Overall Architecture

The method introduces only two lightweight changes to standard classifiers. First, in addition to the original $C$ class logits $(z_i)_{i=1}^C$, a drainage node logit $z_d$ is added, making the set of output neurons $\big((z_i)_{i=1}^C,\, z_d\big)$. Second, the training loss is switched from cross-entropy to the drainage loss, which guides samples with "insufficient evidence for the target class" to activate the drainage node rather than non-target classes.

Inference supports two modes. **Open drainage**: The drainage node is included in the denominator during softmax normalization:

$$p_i = \frac{\exp(z_i)}{Z},\quad p_d = \frac{\exp(z_d)}{Z},\quad Z = \exp(z_d) + \sum_i \exp(z_i)$$

In this mode, $p_d$ provides an explicit "uncertainty/unknown" score for rejection or open-set recognition. **Closed drainage**: Normalization is performed only over the classes, excluding the drainage node:

$$p_i = \frac{\exp(z_i)}{Z^*},\quad Z^* = \sum_i \exp(z_i)$$

This reduces the model to a standard $C$-class classifier, allowing for drop-in replacement in existing systems. Training utilizes open drainage (as the loss requires $p_d$), while classification inference uses closed drainage. Both modes share the same weights and can be switched freely—this is the source of its status as a "unifying framework."

Since the method is a pure output-layer + loss improvement without multi-stage pipelines, the core mechanism is entirely represented by the loss formulations below.

### Key Designs

**1. Drainage Node: An Explicit Outlet for Uncertainty**

Addressing the pain point that "networks have no escape route in high-uncertainty regions," a neuron $z_d$ that does not correspond to any real class is attached to the output layer. It competes equally with standard classes during softmax but semantically handles probability mass that cannot be assigned to any category. Unlike thresholding methods (rejecting based on $\max_i p_i$), the drainage node produces a **dedicated, trained** uncertainty score $p_d$. Compared to heteroskedastic methods that model $\sigma(x)$, it does not assume Gaussian noise but directly captures probability in the softmax space. Its engineering value lies in using the same $z_d$ as a regularizer during training, a rejector during inference, and ignoring it entirely during classification.

**2. Drainage Loss: "Draining" Mislabeled Samples via LSE Soft Constraints**

To force samples toward the node, the drainage loss is defined. Let $t$ be the target class and $J=\{1,\dots,C\}\setminus\{t\}$ be the set of non-target classes. With probability sum $p_J=\sum_{j\in J}p_j$, the loss is:

$$\ell(p,t) = \log\!\Big(1 + \alpha\cdot\frac{p_d}{p_t} + \frac{p_J}{p_t} + \beta\cdot\frac{p_J}{p_d}\Big)$$

where $\alpha,\beta>0$ are hyperparameters. This can be equivalently written as a Log-Sum-Exp (LSE) over logits:

$$\ell(p,t) = \mathrm{LSE}\big(0,\; z_d-z_t+\log\alpha,\; (z_j-z_t+\log\alpha)_{j\in J},\; (z_j-z_d+\log\beta)_{j\in J}\big)$$

Minimizing this LSE requires every term to be small, imposing a set of **soft constraints**: $z_d$ and all $z_j$ are suppressed relative to $z_t$ (ensuring normal classification), and crucially, $z_d$ is encouraged to **exceed the non-target logits $z_j$** (the $z_j-z_d$ term). When evidence for the target class is insufficient, the loss encourages the model to **vote for the drainage node rather than an incorrect class**. This provides the "escape route" missing in CE. The authors prove monotonicity (shifting probability to the target or drainage node does not increase loss) and convexity regarding logits.

**3. Continuity and the $\alpha/\beta$ Soft Constraints**

$\alpha$ and $\beta$ control the strength of the soft constraints, acting as a continuous "robustness knob." When $\beta > \alpha$, the model aggressively classifies suspicious samples as drainage. A key feature is its continuity with cross-entropy: CE can be written as $\ell_{\mathrm{CE}}=\mathrm{LSE}\big(0,(z_j-z_t)_{j\in J}\big)$. By setting $\alpha=1, \beta\to0$, and $p_d=0$ at its optimum, the drainage loss **exactly degrades into cross-entropy**. This confirms that the drainage loss is a strict superset of CE—acting like CE on clean data and activating "drainage" as noise increases.

**4. Decoupling $z_d$ for Open-Set Recognition (OSR)**

Qualitative analysis shows that the drainage node naturally responds to class ambiguity and out-of-distribution samples. Sorting validation samples by $p_d$ reveals typical clear samples at the low end and outliers or mislabeled samples at the high end. Consequently, for OSR, the authors **disconnect** $z_d$ from the network and fix it as a constant $z_d=\text{cst}$. The drainage node then passively collects samples where no class logit shows a strong response (i.e., lacking visual evidence for known classes). During inference, $p_d$ can be used directly as an "unknown" score.

### Example: Systematic Mislabeling on MNIST

On MNIST, the authors randomly relabeled all 7s, 8s, and 9s in the training set to labels 0–6. Using the drainage loss, 7s, 8s, and 9s in the test set were **consistently predicted as drainage**. The model did not learn incorrect boundaries from the contaminated labels; instead, it identified that these samples did not match the given labels and directed them to the drainage node.

## Key Experimental Results

### Main Results: Synthetic Noise (CIFAR-10/100, Accuracy %)

Experiments were conducted with an 8-layer CNN (CIFAR-10) and ResNet-34 (CIFAR-100), comparing CE, GCE, SCE, AFL, APL, and ANL-CE. The table highlights high-noise scenarios where Drainage leads significantly.

| Dataset / Noise | Ours Drainage | Next Best | Gain |
|--------------|--------------|---------|------|
| CIFAR-10 Asymmetric 0.45 | **82.23** | CE 77.92 | +4.3 |
| CIFAR-10 Instance 0.5 | **64.22** | GCE 59.27 | +5.0 |
| CIFAR-100 Asymmetric 0.4 | **61.55** | GCE 55.08 | +6.5 |
| CIFAR-100 Asymmetric 0.45 | **52.69** | GCE 42.70 | +10.0 |
| CIFAR-100 Instance 0.5 | **53.01** | GCE 48.12 | +4.9 |

In low-noise scenarios, methods were comparable (e.g., at noise=0, CE 92.16 vs. Drainage 91.30), confirming the continuity design.

### Real-World Noise (Accuracy %)

| Dataset | Classes | Noise% | Ours | Next Best | Note |
|--------|------|-------|------|------|------|
| CIFAR-10N (Worst) | 10 | 40.2 | 79.85 | GCE 81.70 | Slightly below GCE/AFL |
| CIFAR-100N (Noisy) | 100 | 40.2 | **57.77** | ANL-CE 56.62 | Superior on structured noise |
| Mini-WebVision | 50 | 20.0 | **68.80** | ANL-CE 67.53 | State-of-the-Art |
| ILSVRC12 | 50 | 0.0 | 64.90 | ANL-CE 65.77 | Second Best |
| Clothing-1M | 14 | ~38.5 | **70.42** | AFL 68.94 | ~ +1.5–2 |

### Open-Set Recognition (ROC AUC, Average of 5 random removals of 4 classes)

With $\alpha=\beta=1$ and constant $z_d$, the method outperforms the CE-MSP baseline without extra tuning.

| Loss / Score | SVHN | CIFAR-10 |
|------------|------|----------|
| CE / MSP | 91.0 | 67.3 |
| Drainage / MSP (Excl. $d$) | 92.0 | 72.2 |
| Drainage / Direct $p_d$ | **92.3** | **72.4** |

### Key Findings
- **Gain scales with noise**: While methods are close at low noise, the gap between Drainage and others widens as noise increases, reaching a 10-point lead on CIFAR-100 Asymmetric 0.45.
- **Benefits from larger class spaces**: Drainage's advantage is more pronounced on CIFAR-100, suggesting that explicit uncertainty separation is more effective with complex decision boundaries.
- **Denoising is an emergent behavior**: The drainage neuron consistently absorbs contaminated and outlier samples, resulting in more stable decision boundaries without manual cleaning.
- **Minimal Hyperparameters**: By setting $\beta=1/\alpha$, only $\alpha$ needs tuning, and the same parameters often work across different noise types.

## Highlights & Insights
- **Strict Superset of CE**: The ability to degrade into CE means there is zero cost on clean data and low risk of performance degradation.
- **One Node, Three Purposes**: A single $z_d$ handles training regularization, inference rejection, and classification masking.
- **Optimizable Uncertainty**: Unlike many robust losses that simply "minimize penalty," Drainage provides an explicit destination for noise, ensuring $z_d$ stays above incorrect classes and below the target class.
- **Transferability**: The paradigm of adding an absorptive output node + LSE drainage terms can theoretically be applied to background classes in detection, void classes in segmentation, or rejection in retrieval.

## Limitations & Future Work
- **Task-Specific Adaptation**: Robust classification requires a learned $z_d$, whereas OSR requires a constant $z_d$; no single configuration fits all tasks perfectly.
- **Performance on Random Noise**: On CIFAR-10N, Drainage was second best. Its primary strength lies in structured (asymmetric/instance) noise rather than purely symmetric/random noise.
- **Orthogonality with Sample Mixing**: While complementary to MixUp/CutMix, the synergy of combining them remains for future work.
- **Lightweight OSR Evaluation**: OSR testing was limited to small datasets (SVHN/CIFAR-10) and small backbones.

## Related Work & Insights
- **Vs. Robust Losses (GCE/SCE/APL/ANL)**: These apply uniform penalties. Drainage provides an "escape route" via soft constraints.
- **Vs. Heteroskedastic Methods**: Both add output neurons for uncertainty, but while the former assumes Gaussian noise and learns $\sigma(x)$, Drainage captures probability directly in the softmax space.
- **Vs. Open-Set Recognition (OpenMax/MSP)**: OSR often relies on post-hoc thresholding. Drainage builds uncertainty routing directly into the architecture and loss.

## Rating
- Novelty: ⭐⭐⭐⭐ Drainage node + loss is a simple but effective solution to the "no escape route" problem.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive across synthetic and real noise, though OSR evaluation is relatively basic.
- Writing Quality: ⭐⭐⭐⭐ Clear explanations of mechanisms, formulas, and degradation relationships.
- Value: ⭐⭐⭐⭐ A near-zero-cost drop-in robust loss with significant gains in high-noise, large-class-space settings.

<!-- RELATED:START -->
<div class="related-papers" markdown="1">
</div>
<!-- RELATED:END -->

## Related Papers

- [\[CVPR 2026\] NexusFlow: Unifying Disparate Tasks under Partial Supervision via Invertible Flow Networks](nexusflow_unifying_disparate_tasks_under_partial_supervision_via_invertible_flow.md)
- [\[CVPR 2026\] Your Dissimilarities Define You: Complementary Learning Exploiting Class Diversities](your_dissimilarities_define_you_complementary_learning_exploiting_class_diversit.md)
- [\[CVPR 2026\] Representation-Steered Incremental Adapter-Tuning for Class-Incremental Learning with Pre-Trained Models](representation-steered_incremental_adapter-tuning_for_class-incremental_learning.md)
- [\[CVPR 2026\] Evidential Deep Partial Label Learning to Quantify Disambiguation Uncertainty](evidential_deep_partial_label_learning_to_quantify_disambiguation_uncertainty.md)
- [\[ICLR 2026\] Measuring Uncertainty Calibration](../../ICLR2026/others/measuring_uncertainty_calibration.md)

</div>

<!-- RELATED:END -->
