---
title: >-
  [Paper Note] BoSS: A Best-of-Strategies Selector as an Oracle for Deep Active Learning
description: >-
  [CVPR 2026][Self-Supervised Learning][Active Learning] This paper proposes BoSS (Best-of-Strategies Selector), which generates 100 candidate batches by ensembling 10 complementary AL selection strategies, and efficiently evaluates the performance gain of each candidate batch by freezing the pretrained backbone and retraining only the final linear layer. The best-performing batch is selected as an Oracle upper-bound reference. BoSS is the first deep active learning Oracle scalable to ImageNet, and reveals that current state-of-the-art strategies still leave approximately a 2× accuracy improvement gap on large-scale, many-class datasets.
tags:
  - CVPR 2026
  - Self-Supervised Learning
  - Active Learning
  - Oracle Strategy
  - Strategy Ensemble
  - Batch Selection
  - Deep Neural Networks
date: 2026-05-08
content_hash: d2daa89e913412f2
---

# BoSS: A Best-of-Strategies Selector as an Oracle for Deep Active Learning

**Conference**: CVPR 2026
**arXiv**: [2603.13109](https://arxiv.org/abs/2603.13109)
**Code**: [GitHub](https://github.com/dhuseljic/dal-toolbox)
**Area**: Active Learning / Data Selection
**Keywords**: Active Learning, Oracle Strategy, Strategy Ensemble, Batch Selection, Deep Neural Networks

## TL;DR

This paper proposes BoSS (Best-of-Strategies Selector), which generates 100 candidate batches by ensembling 10 complementary AL selection strategies, and efficiently evaluates the performance gain of each candidate batch by freezing the pretrained backbone and retraining only the final linear layer. The best-performing batch is selected as an Oracle upper-bound reference. BoSS is the first deep active learning Oracle scalable to ImageNet, and reveals that current state-of-the-art strategies still leave approximately a 2× accuracy improvement gap on large-scale, many-class datasets.

## Background & Motivation

**Background**: Active learning (AL) aims to iteratively select the most informative samples for annotation, achieving optimal model performance with minimal labeling effort. Although the foundation model era has enabled more powerful feature extraction, selection strategies—determining which samples to annotate—still lack robustness across models and datasets. Several recent studies (Munjal 2022; Lüth 2024; Werner 2024) have shown that no single strategy is consistently optimal: BADGE leads in some evaluations while Margin leads in others.

**Limitations of Prior Work**:
- AL strategies rely on heuristics such as uncertainty or representativeness rather than directly optimizing model performance, leading to inconsistent behavior across settings.
- Once a strategy is chosen, it remains fixed throughout the entire AL process, making it unable to adapt to distributional shifts introduced by iterative labeling.
- Existing Oracle methods (SAS: simulated annealing with 25,000 steps; CDO: greedy sample-by-sample selection with quadratic complexity in batch size) do not scale to large settings, making ImageNet-scale evaluation entirely infeasible.

**Key Challenge**: A quantifiable "optimal selection" reference point is needed to assess the true gap of state-of-the-art strategies, yet constructing such an Oracle is computationally prohibitive—the $\binom{1000}{10} = 2.63 \times 10^{23}$ possible batch combinations make exhaustive search infeasible.

**Key Insight**: Leveraging strategy ensembles to generate high-quality candidate batches, combined with a frozen-backbone proxy evaluation that retrains only the linear layer, compresses the combinatorial search to a linear evaluation over only 100 candidate batches.

## Method

### Overall Architecture

BoSS formalizes the optimal batch selection problem as:

$$\mathcal{B}^{\star}=\arg\min_{\mathcal{B}\in\{\mathcal{B}_1,\dots,\mathcal{B}_T\}} \sum_{(\mathbf{x},y)\in\mathcal{E}} \mathbb{1}[y \neq \arg\max_{c} p(c|\mathbf{x},\mathcal{L}^+)]$$

The framework comprises three core components: **batch selection** (choosing the best candidate), **performance estimation** (evaluating accuracy on the test set), and **retraining** (efficiently updating model predictions). Concretely, 10 strategies each generate 10 candidate batches; the backbone is frozen and the linear layer is retrained for 50 epochs; the batch yielding the highest test-set accuracy is selected.

### Key Designs

1. **Candidate Batch Generation via Strategy Ensemble**
   - **Function**: Compresses the combinatorially explosive search space into a small set of high-quality candidates.
   - **Mechanism**: Ten complementary strategies covering three major heuristics—uncertainty (Unc), representativeness (Repr), and diversity (Div)—are selected: Random, Margin, CoreSets, BADGE, FastBAIT, TypiClust, AlfaMix, DropQuery, and their supervised variants TypiClust\* and DropQuery\*. Each strategy operates on a randomly sampled candidate pool $\mathcal{C} \sim \text{Unif}([\mathcal{U}]^k)$ ($k \leq k_{\max}$).
   - **Design Motivation**: Different strategies excel at different AL stages—early stages benefit from representative exploration, while later stages require uncertainty-based exploitation. Varying the candidate pool size further increases diversity. Supervised variants (using ground-truth labels for clustering) provide additional Oracle privilege, which ablations confirm is especially important in large-scale, many-class settings.

2. **Selection-via-Proxy Fast Retraining**
   - **Function**: Evaluates the performance gain of each candidate batch within acceptable computational overhead.
   - **Mechanism**: The feature extractor $h^{\phi}$ is frozen; only the final linear layer $g^{\theta}$ is retrained, reducing training epochs from 200 (full training) to 50.
   - **Design Motivation**: Full model retraining for $T=100$ candidates is infeasible on large datasets. Freezing the backbone confines parameter updates to a simple model, yielding high stability and low compute. Ablations show 50 epochs suffice to distinguish batch quality (even 10 epochs yield similar results; degradation appears only below 5 epochs).

3. **Direct Performance Evaluation on the Test Set**
   - **Function**: Precisely measures each candidate batch's true contribution to final model performance.
   - **Mechanism**: As an Oracle strategy, BoSS is permitted to use the test set $\mathcal{E}$ with zero-one loss, directly corresponding to the evaluation metric (accuracy).
   - **Design Motivation**: Zero-one loss maps directly to the accuracy learning curve. Brier score as a proper scoring rule is also effective (negligible AULC difference), whereas cross-entropy performs worse—loss functions that directly correspond to the target metric are preferable.

### Loss & Training

- **BoSS Evaluation Phase**: Frozen backbone + linear layer trained with SGD for 50 epochs; lr=0.01, batch size=64, weight decay=1e-4, cosine annealing.
- **Formal Training** (after batch selection): Frozen backbone + linear layer trained with SGD for 200 epochs, same hyperparameters.
- **Backbone**: DINOv2-ViT-S/14 (22M, $D=384$, self-supervised) and SwinV2-B (88M, $D=1024$, supervised ImageNet pretraining).
- **AL Setup**: 20 cycles per dataset; batch sizes ranging from 10 (CIFAR-10) to 1000 (ImageNet); all results averaged over 10 runs.

## Key Experimental Results

### Main Results

**Oracle Strategy Comparison (DINOv2-ViT-S/14, runtime-aligned) — Relative Accuracy Gain over Random (%)**

| Dataset (batch) | BoSS | CDO (aligned) | SAS (aligned) |
|---|:---:|:---:|:---:|
| CIFAR-10 (b=10) | **~20%** | ~18% | ~5% |
| Snacks (b=20) | **~20%** | ~18% | ~8% |
| Dopanim (b=50) | **~10%** | ~8% | ~5% |
| DTD (b=50) | **~10%** | ~8% | ~3% |

*SAS default runtime exceeds BoSS by more than 100× (DTD: 62h vs. 22min); performance degrades substantially when runtime-aligned.*

**BoSS vs. SOTA AL Strategies — BoSS achieves ~2× the accuracy gain of the best strategy on ImageNet/DINOv2**

| Dataset | Classes K | BoSS Gain | Best AL Strategy Gain | Gap |
|---|:---:|:---:|:---:|:---:|
| CIFAR-10 | 10 | ~5% | ~4% | 1.25× |
| Snacks | 20 | ~10% | ~8% | 1.25× |
| CIFAR-100 | 100 | ~7% | ~4% | 1.75× |
| ImageNet | 1000 | ~8% | ~4% | **2.0×** |

### Ablation Study

| Ablation Dimension | Configuration | CIFAR-10 AULC | DTD AULC |
|---|---|:---:|:---:|
| Candidate Source | Strategy ensemble (Alg. 1) | **90.70** | **71.79** |
| | Random candidates (Eq. 5) | 89.2 | 69.5 |
| Batches per Strategy | T=20/strategy | 90.83 | 71.91 |
| | T=10/strategy | **90.70** | **71.79** |
| | T=1/strategy | 89.90 | 70.45 |
| Batch Size | 0.5b | **85.71** | **68.41** |
| | b (default) | 85.62 | 67.95 |
| | 4b | 84.95 | 66.82 |
| Loss Function | Zero-one | **90.70** | **71.79** |
| | Brier score | 90.65 | 71.75 |
| | Cross-entropy | 90.10 | 70.80 |

**Cumulative Strategy Addition Ablation (Dopanim dataset)**

| Cumulative Strategies | AULC |
|---|:---:|
| Random | 75.24 |
| +DropQuery | 75.82 |
| +AlfaMix | 76.01 |
| +TypiClust | 76.08 |
| +All 10 strategies | **76.52** |

### Key Findings

- **Gap grows with dataset complexity**: AL strategies approach Oracle performance on simple datasets, but the gap becomes significant on complex, many-class datasets with $K>100$, suggesting that **large-scale, many-class settings represent the most valuable direction for AL research**.
- **No single strategy is optimal throughout**: Pick frequency analysis shows that DropQuery\*/TypiClust\* dominate early cycles (representativeness), while no strategy is consistently dominant in later cycles—Random is even frequently selected—confirming that ensemble strategies outperform fixed ones.
- **Strategy ensembles substantially outperform random candidates**: This demonstrates that heuristic strategies, though imperfect, effectively constrain the search space.
- **Adding more strategies yields only positive effects**: Even weak strategies do not harm BoSS performance, as poorly performing batches are naturally not selected.

## Highlights & Insights

- **First deep AL Oracle scalable to ImageNet**: Computational complexity is fixed at $O(T \cdot \text{train-eval}(\theta, \mathcal{L}^+, \mathcal{E}))$, independent of batch size and cycle count.
- **Quantifies the true gap of state-of-the-art strategies**: The best AL strategy on ImageNet achieves only ~50% of the Oracle's improvement, establishing a concrete target for the community.
- **Pick frequency analysis reveals AL dynamics**: The cold-start phase requires representative strategies, exposing the insufficiency of current unsupervised strategies at this stage; the exploitation phase shows no consistent optimum, motivating adaptive mechanisms.
- **Conceptually simple design**: The combination of three straightforward components (ensemble + proxy + selection) makes the approach easy to reproduce and extend.

## Limitations & Future Work

- The Oracle strategy relies on full ground-truth labels and the test set, and thus cannot be directly applied in practical AL settings—it serves solely as an upper-bound benchmark.
- Validation is limited to image classification; generalization to object detection, semantic segmentation, NLP, and other tasks remains unexplored.
- Adaptively allocating candidate batches across strategies via a multi-armed bandit framework is a promising extension direction.
- Whether the Oracle's pick frequency patterns can be distilled into a meta-learner that adaptively switches strategies in a label-free setting is an open question.

## Related Work & Insights

- **vs. CDO (Werner et al., NeurIPS 2024)**: Greedy sample-by-sample evaluation with $O(m \cdot b^2)$ complexity; at $b=50$, $m$ must be reduced to 3–4, making large batch sizes infeasible.
- **vs. SAS (Zhou et al., AISTATS 2021)**: Simulated annealing over the full labeled pool requiring 30,000 steps by default; performance degrades substantially when runtime-aligned.
- **vs. practical AL strategies**: BADGE and BAIT most closely approach the Oracle overall; CoreSets consistently underperforms.
- **Insights**: The strategy ensemble approach combined with pick frequency dynamics analysis can guide the design of adaptive AL strategies that operate without an Oracle.

## Rating

⭐⭐⭐⭐ (4/5)

The method is conceptually simple and effective (strategy ensemble + proxy retraining + performance-based selection), and is the first Oracle approach scaled to ImageNet. The experimental coverage across 10 image and 4 text datasets with 2 backbone architectures is thorough. However, the methodological novelty is limited (strategy ensemble + brute-force search), and the Oracle itself serves only as an evaluation benchmark rather than a practical deployment tool.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] SpHOR: A Representation Learning Perspective on Open-set Recognition for Identifying Unknown Classes in Deep Neural Networks](sphor_a_representation_learning_perspective_on_open-set_recognition_for_identify.md)
- [\[CVPR 2026\] MINE-JEPA: In-Domain Self-Supervised Learning for Mineral Exploration](mine-jepa_in-domain_self-supervised_learning_for_mine-like_object_classification.md)
- [\[CVPR 2026\] A Stitch in Time: Learning Procedural Workflow via Self-Supervised Plackett-Luce Ranking](a_stitch_in_time_learning_procedural_workflow_via_self_supervised_plackett_luce_r.md)
- [\[CVPR 2026\] Group-DINOmics: Incorporating People Dynamics into DINO for Self-supervised Group Activity Feature Learning](group_dinomics_incorporating_people_dynamics_into_dino_for_self_supervised_group_activity_feature_learning.md)
- [\[CVPR 2026\] Representation Learning for Spatiotemporal Physical Systems](representation_learning_for_spatiotemporal_physica.md)

<!-- RELATED:END -->
