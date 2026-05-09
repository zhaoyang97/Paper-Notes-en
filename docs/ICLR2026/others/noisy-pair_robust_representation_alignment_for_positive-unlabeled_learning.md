---
title: >-
  [Paper Note] Noisy-Pair Robust Representation Alignment for Positive-Unlabeled Learning
description: >-
  [ICLR 2026][PU Learning] This paper proposes NcPU, a non-contrastive PU learning framework that applies a sqrt transformation to the standard non-contrastive loss (NoiSNCL) so that clean-pair gradients dominate training, and introduces PhantomGate to provide conservative negative supervision with a regret rollback mechanism. Both modules iterate in a mutually beneficial manner under an EM framework. Without relying on auxiliary negative samples or pre-estimated class priors, NcPU narrows the gap with supervised learning from 14.26% to <1.4% on CIFAR-100, and achieves SOTA on xBD disaster damage assessment as well.
tags:
  - ICLR 2026
  - PU Learning
  - Non-Contrastive Representation Learning
  - Noisy-Pair Robustness
  - Pseudo-Label Disambiguation
  - EM Framework
date: 2026-05-08
content_hash: 54b4333ce5f2a9ae
---

# Noisy-Pair Robust Representation Alignment for Positive-Unlabeled Learning

**Conference**: ICLR 2026
**arXiv**: [2510.01278](https://arxiv.org/abs/2510.01278)
**Code**: [https://github.com/Hengwei-Zhao96/NcPU](https://github.com/Hengwei-Zhao96/NcPU)
**Area**: Other / Weakly Supervised Learning
**Keywords**: PU Learning, Non-Contrastive Representation Learning, Noisy-Pair Robustness, Pseudo-Label Disambiguation, EM Framework

## TL;DR

This paper proposes NcPU, a non-contrastive PU learning framework that applies a sqrt transformation to the standard non-contrastive loss (NoiSNCL) so that clean-pair gradients dominate training, and introduces PhantomGate to provide conservative negative supervision with a regret rollback mechanism. Both modules iterate in a mutually beneficial manner under an EM framework. Without relying on auxiliary negative samples or pre-estimated class priors, NcPU narrows the gap with supervised learning from 14.26% to <1.4% on CIFAR-100, and achieves SOTA on xBD disaster damage assessment as well.

## Background & Motivation

**Background**: Positive-Unlabeled (PU) Learning operates with a small set of labeled positive samples and a large pool of unlabeled data, targeting the training of a binary positive/negative classifier. Typical application scenarios include: partially annotated damaged buildings in post-disaster remote sensing, recommendation systems with only click records and no explicit "not interested" labels, and medical diagnosis with only confirmed cases but no explicit negative specimens. Mainstream approaches fall into three categories: risk estimation (nnPU, uPU), label disambiguation (DistPU), and auxiliary negative sample selection (LaGAM).

**Limitations of Prior Work**: Even the best existing PU methods exhibit a large performance gap from fully supervised learning on complex datasets—the best auxiliary-free method achieves only 76.49% OA on CIFAR-100, compared to 89.65% for supervised learning, a gap of over 13 points. The authors use t-SNE visualizations to diagnose the root cause: the feature spaces learned by LaGAM, HolisticPU, and similar methods show severe overlap between positive and negative class distributions, whereas supervised learning yields clearly separable features. This indicates that the fundamental bottleneck of existing PU methods lies not in classifier design, but in the inability to learn discriminative representations from unreliable pseudo-labels.

**Key Challenge**: Representation learning relies on accurate labels to construct same-class/different-class sample pairs; however, labels in PU settings are inherently unreliable. Pseudo-label-constructed positive pairs contain a large proportion of "noisy pairs" (samples that actually belong to different classes but are incorrectly treated as same-class). Under standard contrastive/non-contrastive losses, these noisy pairs produce larger gradients and dominate the entire training process, forming a vicious cycle: poor representations → poor pseudo-labels → more noisy pairs → even worse representations.

**Key Insight**: The authors build on two key observations. First, non-contrastive learning (which only pulls same-class pairs closer without pushing different-class pairs apart) is inherently more tolerant of noisy labels than contrastive learning, as it does not erroneously repel samples that should belong to the same class. Second, the gradient of the standard non-contrastive loss $\mathcal{L}_r = 2(1 - \langle \tilde{q}_i, \tilde{k}_j \rangle)$ is proportional to $(1 - \cos^2\theta)$—noisy pairs with low cosine similarity (large distance) produce large gradients while clean pairs with high cosine similarity produce small gradients, which is counterintuitive. After applying a sqrt transformation, the gradient becomes proportional to $(1 + \cos\theta)$, so clean pairs with high similarity produce larger gradients and dominate training.

**Core Idea**: Apply a sqrt transformation to the standard non-contrastive loss to invert the gradient–distance relationship, making clean pairs dominate training; combine this with PhantomGate for conservative negative supervision, forming an EM-style iterative mutually beneficial framework.

## Method

### Overall Architecture

NcPU is built upon the BYOL non-contrastive learning framework. The input consists of a positive set $\mathcal{P}$ and an unlabeled set $\mathcal{U}$. Each sample undergoes random augmentation to produce two views, which are passed through an online network (encoder + projection head + prediction head) and a target network (momentum-updated, without prediction head) to obtain normalized embeddings $\tilde{q}$ and $\tilde{k}$, respectively. A classifier $f(\cdot)$ outputs two-class softmax probabilities for each sample. The training pipeline consists of two core alternating modules: NoiSNCL performs noisy-pair-robust intra-class representation alignment using current pseudo-labels, and PLD (with PhantomGate) updates more accurate pseudo-labels using the aligned representation space. Theoretically, these two modules correspond to the M-step and E-step of the EM algorithm.

### Key Designs

1. **NoiSNCL — Noisy-Pair Robust Supervised Non-Contrastive Loss**:

    - **Function**: Effectively aligns representations of same-class samples even when pseudo-label noise is severe.
    - **Mechanism**: The standard non-contrastive loss is $\mathcal{L}_r = 2(1 - \langle \tilde{q}_i, \tilde{k}_j \rangle)$; NoiSNCL modifies this to $\tilde{\mathcal{L}}_r = 2\sqrt{1 - \langle \tilde{q}_i, \tilde{k}_j \rangle}$, adding only a sqrt operation. Gradient analysis shows that for the standard loss, noisy pairs (low cosine similarity, large distance) produce gradients $\propto (1 - \cos^2\theta)$ that exceed those of clean pairs (high cosine similarity), causing noisy pairs to dominate training. With NoiSNCL, the gradient is $\propto (1 + \cos\theta)$, so clean pairs produce larger gradients and dominate training. This property stems from the behavior of $\sqrt{x}$: its gradient tends to infinity near $x \to 0$ and to zero near $x \to 1$, which precisely suppresses the influence of noisy pairs with large distances.
    - **Design Motivation**: Directly addresses the core problem of "noisy-pair gradient dominance." In the supervised setting, NoiSNCL performs comparably to the standard loss (98.75% vs. 98.53% on CIFAR-10), introducing no side effects. Numerical stability is ensured by BYOL's asymmetric architecture and random augmentation, which guarantee $\tilde{q}_i \neq \tilde{k}_j$, avoiding division by zero.

2. **PhantomGate — Pseudo-Label Disambiguation with Regret Mechanism**:

    - **Function**: Generates reliable pseudo-labels (especially negative labels) for unlabeled data, preventing the trivial solution where all samples are classified as positive.
    - **Mechanism**: The process consists of three steps. (i) Class-conditional prototypes are momentum-updated each batch: $\mu_c = \text{Normalize}(\alpha \mu_c + (1-\alpha)\tilde{q})$. (ii) Soft pseudo-labels $s'$ are generated based on prototype similarity and stabilized through momentum accumulation. (iii) PhantomGate is the core innovation: an adaptive threshold $\tau$ determines whether, if the classifier's positive-class probability $f_1(x) \geq \tau$ for a given sample, the label is directly set to $[0,1]^T$ (negative class); otherwise, the prototype-based $s'$ is used. The key regret mechanism allows the model to revert to the accumulated $s'$ rather than $[0,1]^T$ if a sample previously assigned as negative is later found to be potentially mislabeled, preventing the "once mislabeled, no recovery" problem.
    - **Design Motivation**: PU learning lacks negative supervision; directly applying prototype-based disambiguation tends to pull all samples toward the positive class (trivial solution). Simply adding a threshold for negative sample selection (+SAT) introduces inaccurate negative supervision (high precision but extremely low recall of 0.51%). PhantomGate strikes a balance: it injects negative supervision to prevent the trivial solution while allowing correction via the regret mechanism.

3. **Adaptive Threshold SAT Mechanism**:

    - **Function**: Automatically controls the tightness of negative sample selection without manual tuning.
    - **Mechanism**: A global threshold $\tilde{\tau}$ and a class-aware modulation factor $\tilde{\rho}(c)$ are maintained via momentum updates. The final threshold is $\tau = \frac{\tilde{\rho}(1)}{\max\{\tilde{\rho}(0), \tilde{\rho}(1)\}} \cdot \tilde{\tau}$. Early in training, the model is less confident ($\tilde{\tau}$ is low), leading more samples to be selected as negative and providing supervision signals; later in training, the model is more confident ($\tilde{\tau}$ rises), increasing the threshold to filter out potentially inaccurate negative selections.
    - **Design Motivation**: Avoids manual threshold setting. The dynamic strategy from loose to tight aligns with the curriculum learning philosophy—first providing simple negative supervision, then progressively raising the standard.

### Loss & Training

The total loss is the sum of three terms: $\mathcal{L} = \frac{1}{|\mathcal{P}|}\sum_{x_i \in \mathcal{P}} \mathcal{L}_c + \frac{1}{|\mathcal{U}|}\sum_{x_i \in \mathcal{U}} \mathcal{L}_c + w_r \frac{1}{|\mathcal{D}|}\sum_{x_i \in \mathcal{D}} \frac{1}{|\mathcal{Q}|}\sum_{x_j \in \mathcal{Q}} \tilde{\mathcal{L}}_r$, where $\mathcal{L}_c$ is the Label Disambiguation Cross-Entropy (LDCE), $\tilde{\mathcal{L}}_r$ is NoiSNCL, and $w_r = 50$ controls the weight of representation learning. All momentum hyperparameters are set to $\alpha = \beta = \gamma = 0.99$, and the same hyperparameter configuration is used across all five datasets. The target network is updated via BYOL-style momentum updates. Entropy regularization is also employed during training for stability. ResNet-18 is used as the backbone throughout.

### EM Theoretical Framework

The classifier predictions are incorporated into an EM framework: the E-step corresponds to pseudo-label assignment (assigning each unlabeled sample to positive or negative clusters), and the M-step corresponds to minimizing NoiSNCL (making intra-cluster representations more compact). Theorem 1 proves, under a von Mises–Fisher (vMF) distribution assumption, that minimizing $\tilde{\mathcal{R}}_r$ is equivalent to maximizing a lower bound of the likelihood: $L_1 = \sum_{\mathcal{S}_c} \frac{|\mathcal{S}_c|}{n_u} \|\nu_c\|^2 \leq L_2$. This bound becomes tight as $\|\nu_c\| \to 1$ (i.e., when same-class data is highly clustered in the representation space). This provides a principled justification for the synergy between NoiSNCL and PLD, rather than a purely empirical combination.

## Key Experimental Results

### Main Results

Comparisons on 5 datasets (3 general + 2 remote sensing disaster damage), where NcPU achieves the best performance on all datasets without relying on auxiliary information:

| Method | Auxiliary Info | CIFAR-10 OA | CIFAR-100 OA | STL-10 OA | ABCD OA | xBD OA |
|--------|---------------|-------------|--------------|-----------|---------|--------|
| CE (unlabeled as negative) | None | 60.45 | 50.36 | 50.30 | 55.70 | 84.08 |
| uPU | $\pi_p$ | 65.52 | 61.44 | 57.08 | 83.76 | 86.82 |
| nnPU | $\pi_p$ | 87.29 | 72.00 | 80.62 | 87.73 | 82.60 |
| DistPU | $\pi_p$ | 85.29 | 67.63 | 85.62 | 86.25 | 82.94 |
| HolisticPU | Negative samples | 84.20 | 64.01 | 72.81 | 65.49 | 81.98 |
| LaGAM | Negative samples | 95.78 | 84.82 | 88.64 | 75.90 | 79.14 |
| WSC | Est. parameters | 90.55 | 75.39 | 79.06 | 80.10 | 84.89 |
| **NcPU** | **None** | **97.36** | **88.28** | **91.40** | **91.10** | **87.60** |
| Supervised | Full labels | 96.96 | 89.65 | — | 92.00 | 88.47 |

NcPU even surpasses supervised learning on CIFAR-10 (97.36 vs. 96.96), narrows the gap to only 1.37% on CIFAR-100, and to less than 1% on ABCD.

### Ablation Study (CIFAR-100)

| Non-Contrastive Loss | Label Disambiguation | OA | F1 | Notes |
|---------------------|---------------------|----|----|-------|
| None | $s$ (PhantomGate) | 61.54 | 40.58 | No representation learning; relying solely on label disambiguation performs poorly |
| $\tilde{\mathcal{L}}_r$ (NoiSNCL) | None | 50.27 | 1.09 | Without label disambiguation, NoiSNCL cannot function independently |
| $\mathcal{L}_{self-r}$ (self-supervised) | $s$ | 73.22 | 72.75 | Self-supervised non-contrastive + PhantomGate |
| $\mathcal{L}_r$ (standard supervised) | $s$ | 84.58 | 85.90 | Standard loss is effective but limited by noisy pairs |
| $\tilde{\mathcal{L}}_r$ (NoiSNCL) | $s'$ (prototype only) | 75.14 | 79.91 | Without PhantomGate; precision only 67% |
| $\tilde{\mathcal{L}}_r$ (NoiSNCL) | $s'$+SAT | 50.25 | 1.01 | Negative supervision from SAT is too inaccurate |
| $\tilde{\mathcal{L}}_r$ (NoiSNCL) | $s$ (PhantomGate) | **88.28** | **88.14** | Full NcPU |

### NoiSNCL Augmenting Base PU Methods

| Method | CIFAR-10 OA | CIFAR-100 OA |
|--------|-------------|-------------|
| uPU | 69.43 | 61.68 |
| uPU + $\tilde{\mathcal{L}}_r$ | **97.35** (+27.9) | **83.71** (+22.0) |
| nnPU | 83.25 | 71.22 |
| nnPU + $\tilde{\mathcal{L}}_r$ | **97.03** (+13.8) | **87.81** (+16.6) |
| Supervised + $\mathcal{L}_r$ | 98.53 | 94.45 |
| Supervised + $\tilde{\mathcal{L}}_r$ | 98.75 | 94.56 |

### Key Findings

- **NoiSNCL is the primary source of gain**: Simply attaching NoiSNCL to the most basic uPU improves CIFAR-10 from 69.43% to 97.35% (+27.9 points), demonstrating that discriminative representation is the core bottleneck of PU learning, not classifier design.
- **NoiSNCL vs. standard loss gap**: On CIFAR-100, $\tilde{\mathcal{L}}_r + s$ (88.28%) outperforms $\mathcal{L}_r + s$ (84.58%) by 3.7 points, validating the effectiveness of noisy-pair robustness; meanwhile, both perform comparably under supervised learning (98.75% vs. 98.53%), confirming that the sqrt transformation introduces no additional cost.
- **Indispensability of PhantomGate**: Using prototype-based disambiguation alone ($s'$) yields recall as high as 98.7% but precision of only 67% (nearly all samples labeled as positive); adding SAT raises precision to 98% but drops recall to 0.5% (overcorrection); the regret mechanism in PhantomGate strikes a balance (precision 89%, recall 87%).
- **Hyperparameter insensitivity**: All five datasets use identical hyperparameters ($\alpha=\beta=\gamma=0.99$, $w_r=50$); the method is largely insensitive to $\alpha$ and $\gamma$; smaller $\beta$ leads to faster pseudo-label updates, and larger $w_r$ strengthens representation learning.
- **Training stability**: On CIFAR-10, continuing training from 400 to 1200 epochs results in OA fluctuations within 0.5%, with no overfitting or instability.

## Highlights & Insights

- **Elegance of the sqrt transformation**: A single sqrt operation inverts the monotonic relationship between gradient magnitude and sample distance, shifting dominance from noisy pairs to clean pairs. This design is remarkably concise yet carries deep mathematical intuition—$\sqrt{x}$ has an infinite derivative near $x \to 0$ (amplifying gradients for clean pairs with small loss) and a diminishing derivative for larger $x$ (suppressing noisy pairs with large loss). The principle of "reshaping the loss to manipulate gradient dominance" is transferable to any noisy label setting.
- **Closed loop between theory and empiricism**: The EM framework is not merely a post-hoc explanation; it justifies why NoiSNCL and PLD must be used jointly—NoiSNCL alone yields 50.27% OA and PLD alone yields 61.54% OA, but their combination reaches 88.28%. The E-step provides better cluster assignments, and the M-step makes clusters more compact; this iterative mutual benefit is clearly validated in the ablation study.
- **"Simple method + good representations" paradigm**: uPU + NoiSNCL (97.35%) outperforms all carefully designed PU methods, suggesting that when the representation space is sufficiently good, even the most naive risk estimation is sufficient. This insight has broad implications for the weakly supervised learning community.

## Limitations & Future Work

- **vMF distribution assumption**: The EM theoretical analysis assumes each class in the representation space follows a von Mises–Fisher (vMF) distribution (a Gaussian on the sphere), which may not hold for highly non-spherical data distributions. Although experiments show the method remains effective even when the assumption is not fully satisfied, the theoretical guarantees may not be tight.
- **Validation limited to image classification**: All five datasets are image classification tasks; performance on NLP (e.g., PU learning for text classification), graph-structured data, or tabular data remains unknown. The effectiveness of non-contrastive learning augmentation may differ in non-visual domains.
- **Fixed positive sample count**: Experiments use a fixed number of positive samples (1,000 for CIFAR-10/100); the performance curve under extremely scarce (e.g., <100) or relatively abundant positive samples has not been analyzed.
- **Multi-class extension**: The current framework is fundamentally binary (positive vs. negative); extending it to multi-class PU learning (multiple positive classes + unlabeled) remains an open problem.
- **Backbone sensitivity**: All experiments use ResNet-18 only; whether stronger backbones (e.g., ViT) or pre-trained features would alter the conclusions has not been explored.

## Related Work & Insights

- **vs. LaGAM**: LaGAM ranks second on CIFAR-10 (95.78%) and CIFAR-100 (84.82%), but requires auxiliary negative samples as input. NcPU surpasses it without any auxiliary information (97.36% / 88.28%), and LaGAM performs poorly on remote sensing data (ABCD: 75.90%), indicating limited generalizability.
- **vs. DistPU**: DistPU performs competitively on STL-10 (85.62%) via distribution matching, but relies on a pre-estimated class prior $\pi_p$. NcPU requires no $\pi_p$ and outperforms DistPU on all datasets.
- **vs. WSC**: WSC also incorporates representation learning but uses a graph-theoretic framework with contrastive learning and estimated parameters. NcPU achieves better results with a simpler non-contrastive framework and EM iteration, suggesting that "noisy-pair robustness" is more important than "more complex graph structures."
- **Implications for noisy label learning**: The gradient-inversion idea behind NoiSNCL's sqrt transformation can be directly borrowed for general Noisy Label Learning—any scenario requiring representation learning from unreliable pair relationships may benefit from this approach.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The sqrt transformation idea for inverting gradient dominance is concise and insightful; the regret mechanism in PhantomGate is also novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Five datasets (including 2 real-world applications) + 11 baselines + detailed ablation + hyperparameter analysis + training stability verification.
- **Writing Quality**: ⭐⭐⭐⭐ Theoretical derivations are rigorous and clear; the gradient analysis visualizations are intuitive.
- **Value**: ⭐⭐⭐⭐⭐ Narrowing the PU learning performance gap to near supervised learning levels is a milestone for the field; the generality of NoiSNCL extends well beyond PU learning itself.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] ANO: Faster is Better in Noisy Landscapes](ano_faster_is_better_in_noisy_landscape.md)
- [\[ICLR 2026\] Learning Adaptive Distribution Alignment with Neural Characteristic Function for Graph Domain Adaptation](learning_adaptive_distribution_alignment_with_neural_characteristic_function_for.md)
- [\[ICCV 2025\] Joint Asymmetric Loss for Learning with Noisy Labels](../../ICCV2025/others/joint_asymmetric_loss_for_learning_with_noisy_labels.md)
- [\[ICLR 2026\] Distributionally Robust Classification for Multi-Source Unsupervised Domain Adaptation](distributionally_robust_classification_for_multi-source_unsupervised_domain_adap.md)
- [\[ICLR 2026\] Latent Equivariant Operators for Robust Object Recognition: Promises and Challenges](latent_equivariant_operators_for_robust_object_recognition_promises_and_challeng.md)

</div>

<!-- RELATED:END -->
