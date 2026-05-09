---
title: >-
  [Paper Note] On the Robustness Tradeoff in Fine-Tuning
description: >-
  [ICCV 2025][LLM Evaluation][fine-tuning robustness] The first systematic study of the adversarial robustness–accuracy tradeoff during fine-tuning, conducted across 231 models, 7 fine-tuning strategies, and 6 datasets. Key findings: (1) robustness first increases then decreases in the early stages of fine-tuning; (2) different PEFT strategies and task complexities yield distinct Pareto frontiers; (3) OOD robustness exhibits no analogous tradeoff and instead tracks accuracy changes closely.
tags:
  - ICCV 2025
  - LLM Evaluation
  - fine-tuning robustness
  - adversarial robustness
  - parameter-efficient fine-tuning
  - Pareto frontier
  - OOD robustness
date: 2026-05-08
content_hash: e5f4e7f748507aa8
---

# On the Robustness Tradeoff in Fine-Tuning

**Conference**: ICCV 2025
**arXiv**: [2503.14836](https://arxiv.org/abs/2503.14836)
**Code**: [https://github.com/kyangl/robustness-finetuning](https://github.com/kyangl/robustness-finetuning)
**Area**: LLM Evaluation
**Keywords**: fine-tuning robustness, adversarial robustness, parameter-efficient fine-tuning, Pareto frontier, OOD robustness

## TL;DR
The first systematic study of the adversarial robustness–accuracy tradeoff during fine-tuning, conducted across 231 models, 7 fine-tuning strategies, and 6 datasets. Key findings: (1) robustness first increases then decreases in the early stages of fine-tuning; (2) different PEFT strategies and task complexities yield distinct Pareto frontiers; (3) OOD robustness exhibits no analogous tradeoff and instead tracks accuracy changes closely.

## Background & Motivation

**Background**: Pre-training followed by fine-tuning has become the standard paradigm for adapting models to downstream tasks. Parameter-efficient fine-tuning (PEFT) methods — including LoRA, Adapter, and BitFit — can match the accuracy of full fine-tuning by updating as few as 0.07%–3.97% of parameters.

**Limitations of Prior Work**:
   - The effect of fine-tuning on model robustness has **received almost no attention**. Existing work on the robustness–accuracy tradeoff focuses primarily on models trained from scratch.
   - The assumption in from-scratch training (that training and attack data are identically distributed) does not hold in fine-tuning, which involves two distinct distributions — upstream and downstream.
   - Existing PEFT robustness studies evaluate only the final model state, without tracking the **dynamic evolution** of robustness throughout fine-tuning.
   - Whether adversarial robustness and OOD robustness are driven by the same factors remains unclear.

**Key Challenge**: Fine-tuning transitions a model from a general to a specialized state, during which the learned robust and non-robust features continuously change. The key question is: given that different PEFT strategies update parameters at different locations and in different quantities, how does this affect the robustness–accuracy tradeoff?

**Goal**: Three core research questions — (RQ1) Does an adversarial robustness–accuracy tradeoff exist during fine-tuning? (RQ2) How do different fine-tuning strategies and task complexities affect the optimal tradeoff? (RQ3) Do these findings extend to OOD robustness?

**Key Insight**: A continuous evaluation framework is constructed to adaptively track changes in robustness and accuracy at the level of individual backpropagation steps throughout fine-tuning, rather than evaluating only the final model.

## Method

### Overall Architecture
(1) Various PEFT modules are integrated into a pre-trained ViT-Base model; (2) during fine-tuning on downstream data, standard accuracy, adversarial robustness, and OOD robustness are continuously evaluated at different backpropagation steps according to an adaptive schedule.

### Key Designs

1. **Theoretical Motivation — Robustness Modeling**:

    - Based on the feature model of Ilyas et al.: the input contains one robust feature $x_1$ and $d$ non-robust features $x_{2:d+1} \sim \mathcal{N}(\eta y, 1)$.
    - Fine-tuned classifier: $f_{FT}(x) = \text{sign}((w_0 + \Delta w)^{\top}x)$, where $k = \|\Delta w\|_0$ denotes the number of updated parameters.
    - Key derivation: to achieve 99% accuracy, the correlation lower bound for non-robust features is:
    $$\eta \geq \frac{2.33}{\sqrt{k+d}}$$
    - Under full fine-tuning ($k=d$), $\eta_{\text{full}} \geq \frac{2.33}{\sqrt{2d}}$, relaxing the lower bound so that the model can exploit weaker non-robust features → greater vulnerability to attack.
    - Simpler tasks (smaller $d$) impose a tighter lower bound, requiring higher non-robust feature correlation → less susceptible to attack.

2. **Decomposition of PEFT Methods (Two Dimensions)**:

    - **Information dimension**: what information is extracted (model weights vs. intermediate representations) and where (attention layers, FFN, biases).
    - **Mechanism dimension**: how updates are applied (neural layer projection, matrix/vector computation, direct backpropagation).
    - 7 strategies: Full Fine-tuning, Linear Probing, LoRA (low-rank decomposition of attention matrices), Adapter (insertion of small modules), Compacter (Kronecker-parameterized Adapter), BitFit (bias-only updates), (IA)³ (scaling of intermediate representations).

3. **Adaptive Tracking Schedule**:

    - Early phase (0–700 steps): evaluation every 50 steps (to capture critical transitions).
    - Middle phase (700–3000 steps): evaluation every 1000 steps.
    - Late phase (3000+ steps): evaluation every 6000 steps.
    - Adversarial attacks use PGD ($\epsilon=1/255$, step size $\alpha=0.25/255$, 15 steps).

4. **Pareto Frontier and AUC Metric**:

    - Pareto-optimal points are extracted in the robustness–accuracy space.
    - The area under the Pareto frontier (AUC) is computed as a scalar measure of tradeoff quality.

## Experiments

### Main Results 1: Adversarial Robustness–Accuracy Tradeoff (RQ1)

Using Caltech-256 as a representative case, all 7 fine-tuning methods reach ≈90% accuracy within ~1000 steps, while adversarial robustness peaks at ≈25% around step ~400 and then steadily declines to ≈10% at convergence. **The tradeoff is consistently present and emerges within the first 3 epochs of fine-tuning.**

### Main Results 2: Pareto Frontier AUC (Strategies × Datasets)

| Method | CIFAR-10 | CIFAR-100 | Caltech-256 | CUB-200 | Stanford Dogs |
|:---|:---|:---|:---|:---|:---|
| BitFit | **0.21** | **0.10** | 0.33 | 0.14 | 0.08 |
| Compacter | 0.09 | 0.06 | **0.34** | **0.15** | **0.09** |
| LoRA | 0.14 | 0.07 | 0.23 | 0.12 | 0.06 |
| Adapter | 0.12 | 0.05 | 0.21 | 0.07 | 0.05 |
| (IA)³ | 0.08 | 0.05 | 0.31 | 0.13 | 0.05 |
| LP | 0.06 | 0.03 | 0.24 | 0.08 | 0.02 |
| Full FT | 0.11 | 0.04 | 0.26 | 0.09 | 0.05 |

**Key Findings**:
- **Simple tasks (CIFAR-10/100)**: BitFit achieves the best tradeoff (75%/81.5% above average), as updating only biases is sufficient for effective adaptation.
- **Complex tasks (Caltech-256/CUB-200)**: Compacter achieves the best tradeoff (57.5%/34.6% above average), as low-rank parameterization of attention layers better balances adaptation with robustness inheritance.
- Linear Probing and Full Fine-tuning perform worst across all datasets.

### Main Results 3: OOD Robustness (RQ3)

| Metric | Behavioral Pattern |
|:---|:---|
| OOD vs. adversarial robustness | OOD robustness exhibits **no** tradeoff with accuracy; it improves and then stabilizes at a lower level. |
| Strategy effect | Full FT achieves the highest OOD robustness (73%±2%); LP the lowest (61%±5%); differences among PEFT methods are small. |
| Training domain effect | OOD robustness is paradoxically lower (64%±5%) for domains close to the pre-training distribution ("real" domain). |

### Key Findings Summary
1. The adversarial robustness–accuracy tradeoff is universal during fine-tuning and manifests within the first 3 epochs.
2. Updating attention-related layers (LoRA, Compacter) yields a better tradeoff on complex tasks than updating only peripheral parameters (BitFit, LP) or all parameters (Full FT).
3. OOD robustness and adversarial robustness are driven by distinct mechanisms — the former depends on transferable features, the latter on non-robust features.
4. Task complexity (inter-class separability and similarity to upstream data) significantly influences the shape of the Pareto frontier.

## Highlights & Insights

1. **First systematic study of fine-tuning robustness**: Rather than a one-time evaluation of the final model, the work continuously tracks robustness dynamics at the backpropagation-step level.
2. **PEFT decomposition framework**: PEFT methods are systematically decomposed along the information extraction and update mechanism dimensions, establishing connections between parameter update location/manner and robustness.
3. **Separation of adversarial and OOD robustness**: The work clearly demonstrates that the two types of robustness are driven by different mechanisms and require independently designed strategies.
4. **Theory–experiment consistency**: The derived lower bound $\eta \geq 2.33/\sqrt{k+d}$ is consistent with the experimental findings that full fine-tuning yields the worst robustness and that simpler tasks exhibit a more gradual tradeoff.

## Limitations & Future Work

1. Adversarial robustness is evaluated using only PGD attacks; stronger attacks such as AutoAttack or adaptive attacks are not considered.
2. The study is primarily based on ViT-Base and does not cover CNN-ViT hybrid architectures or larger-scale models.
3. Robustness changes under defense mechanisms such as adversarial training are not examined.
4. Although the experimental scale is large, the datasets are predominantly of moderate size (10k–60k); model performance on large-scale datasets (Places365, 1.8M) is insufficient to draw reliable conclusions.

## Related Work & Insights

- **Robustness–accuracy tradeoff**: Tsipras et al. (2019) show that the tradeoff stems from the data distribution; Ilyas et al. (2019) identify the role of non-robust features; the TRADES framework attempts to mitigate the tradeoff.
- **Fine-tuning robustness**: Robustness at the pre-training stage (adversarial pre-training), AdapterMixup (Adapter + adversarial training + mixup), CLAT (layer-wise robustness analysis).
- **PEFT methods**: LoRA, BitFit, Adapter, Compacter, (IA)³, etc.

## Rating
- Novelty: ★★★★☆ (systematic empirical study rather than methodological innovation, but the PEFT decomposition framework and continuous tracking approach are original)
- Experimental Thoroughness: ★★★★★ (231 models × 7 strategies × 6 datasets, ~2100 adversarial + ~2000 OOD evaluations, thorough ablations)
- Value: ★★★★☆ (provides practical guidance for selecting fine-tuning strategies: BitFit for simple tasks, Compacter for complex tasks)
- Writing Quality: ★★★★★ (research questions are clearly stated, theory and experiments are well aligned, figures and tables are highly informative)

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Hyperbolic Fine-Tuning for Large Language Models](../../NeurIPS2025/llm_evaluation/hyperbolic_fine-tuning_for_large_language_models.md)
- [\[ICLR 2026\] Towards Anomaly-Aware Pre-Training and Fine-Tuning for Graph Anomaly Detection](../../ICLR2026/llm_evaluation/towards_anomaly-aware_pre-training_and_fine-tuning_for_graph_anomaly_detection.md)
- [\[ICCV 2025\] OmniDiff: A Comprehensive Benchmark for Fine-grained Image Difference Captioning](omnidiff_a_comprehensive_benchmark_for_fine-grained_image_difference_captioning.md)
- [\[AAAI 2026\] Low-Rank Curvature for Zeroth-Order Optimization in LLM Fine-Tuning](../../AAAI2026/llm_evaluation/low-rank_curvature_for_zeroth-order_optimization_in_llm_fine-tuning.md)
- [\[NeurIPS 2025\] Cost-Sensitive Freeze-thaw Bayesian Optimization for Efficient Hyperparameter Tuning](../../NeurIPS2025/llm_evaluation/cost-sensitive_freeze-thaw_bayesian_optimization_for_efficient_hyperparameter_tu.md)

<!-- RELATED:END -->
