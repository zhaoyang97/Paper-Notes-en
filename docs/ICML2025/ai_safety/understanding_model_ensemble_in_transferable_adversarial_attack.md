---
title: >-
  [Paper Note] Understanding Model Ensemble in Transferable Adversarial Attack
description: >-
  [ICML 2025][AI Safety][Adversarial Transferability] For the first time, a theoretical framework is established for model ensemble adversarial attacks, defining transferability error and decomposing it into vulnerability and diversity, followed by deriving upper bounds using information-theoretic tools. This theoretically validates three practical guidelines: "more models, higher diversity, and lower complexity."
tags:
  - "ICML 2025"
  - "AI Safety"
  - "Adversarial Transferability"
  - "Model Ensemble Attack"
  - "Rademacher Complexity"
  - "Vulnerability-Diversity Decomposition"
  - "Information Theory"
date: 2026-05-08
content_hash: 3eb861795f45e1ea
---

# Understanding Model Ensemble in Transferable Adversarial Attack

**Conference**: ICML 2025  
**arXiv**: [2410.06851](https://arxiv.org/abs/2410.06851)  
**Code**: None  
**Area**: AI Safety  
**Keywords**: Adversarial Transferability, Model Ensemble Attack, Rademacher Complexity, Vulnerability-Diversity Decomposition, Information Theory

## TL;DR

For the first time, a theoretical framework is established for model ensemble adversarial attacks, defining transferability error and decomposing it into vulnerability and diversity, followed by deriving upper bounds using information-theoretic tools. This theoretically validates three practical guidelines: "more models, higher diversity, and lower complexity."

## Background & Motivation

The **transferability** of adversarial examples (adversarial transferability) allows attackers to launch black-box attacks without access to target models, posing a severe threat to safety-critical systems. Existing methods targeting transferability enhancement can be categorized into three classes: input transformation, gradient optimization, and model ensemble attacks. Among these, model ensemble attacks are highly effective—generating adversarial examples concurrently using multiple surrogate models, which can be combined with the former two categories.

Empirically, two widely known but theoretically unexplained phenomena exist: (1) ensembling more surrogate models improves transferability; (2) utilizing more diverse model architectures further enhances transferability. However, theoretical research on adversarial transferability mostly focuses on data distribution, surrogate model generalization, optimization flatness, etc., while **the theoretical foundation of model ensemble attacks is almost vacant**.

The core motivation of this paper is: **Can a comprehensive theoretical framework be established to explain the origin of transferability in model ensemble attacks, and to provide actionable guidelines for future algorithm design?** Starting from the bias-variance decomposition in learning theory, the authors transfer similar concepts to the adversarial attack domain and leverage cutting-edge tools from information theory to handle the non-independence among surrogate models.

## Method

### Overall Architecture

This paper proposes three core definitions—**Transferability Error**, **Diversity**, and **Empirical Model Ensemble Rademacher Complexity**—laying the foundation of the theoretical framework. Based on these, two key theoretical results are presented: (1) Vulnerability-Diversity Decomposition Theorem; (2) Transferability Error Upper Bound Theorem. The logic of the framework is as follows:

1. **Define transferability error** to measure the transferability gap $\rightarrow$ 2. **Decompose** into vulnerability + diversity to reveal the sources $\rightarrow$ 3. Due to the presence of the vulnerability-diversity trade-off, further derive upper bounds using **Rademacher complexity + information theory** $\rightarrow$ 4. Derive three practical guidelines from the upper bounds.

### Key Designs: Vulnerability-Diversity Decomposition

**Core Idea**: Analogous to the bias-variance decomposition, this decomposes transferability error into two intuitively meaningful terms.

Given an adversarial example $z = (x, y)$, population risk $L_P(z) = \mathbb{E}_{\theta \sim \mathcal{P}_\Theta}[\ell(f(\theta; x), y)]$, the optimal adversarial example $z^*$ maximizes $L_P(z^*)$. Transferability error is defined as:

$$TE(z, \epsilon) = L_P(z^*) - L_P(z)$$

Under the mean squared error loss $\ell(f(\theta; x), y) = [f(\theta; x) - y]^2$, letting $\tilde{f}(\theta; x) = \mathbb{E}_{\theta \sim \mathcal{P}_\Theta} f(\theta; x)$ denote the expected prediction of the ensemble, we have:

$$TE(z, \epsilon) = L_P(z^*) - \underbrace{\ell(\tilde{f}(\theta; x), y)}_{\text{Vulnerability}} - \underbrace{\text{Var}_{\theta \sim \mathcal{P}_\Theta} f(\theta; x)}_{\text{Diversity}}$$

- **Vulnerability**: The degree to which the ensemble's average prediction deviates from the true label; a larger value indicates a more effective ensemble attack.
- **Diversity**: The variance of the ensemble members' predictions; a larger value indicates greater disagreement among members, which helps prevent adversarial examples from overfitting to the ensemble.

This decomposition reveals a fundamental trade-off: **vulnerability and diversity cannot be maximized simultaneously**, mimicking the classical bias-variance trade-off. This motivates the authors to seek a more practical upper-bound analysis.

### Key Designs: Information-Theoretic Upper Bound for Transferability Error

To bypass the trade-off limitations in the decomposition, the authors define **Empirical Model Ensemble Rademacher Complexity**:

$$\mathcal{R}_N(\mathcal{Z}) = \mathbb{E}_{\boldsymbol{\sigma}} \left[ \sup_{z \in \mathcal{Z}} \frac{1}{N} \sum_{i=1}^{N} \sigma_i \ell(f(\theta_i; x), y) \right]$$

where $\sigma_i$ are Rademacher random variables. This quantity measures the capacity of $N$ ensemble classifiers to "fit random labels" over the input space, representing model complexity.

**MLP Complexity Bound (Lemma 4.2)**: For an $l$-layer MLP, let $T = \prod_{j=1}^{l} \sup_{i \in [N]} \|W_{i,j}\|_F$, $\|x\|_F \leq B$, then:

$$\mathcal{R}_N(\mathcal{Z}) \leq \frac{(\sqrt{(2 \ln 2) l} + 1) BT}{\sqrt{N}}$$

This indicates that increasing the number of models $N$ or decreasing the weight norm $T$ (reducing model complexity) tightens the upper bound.

**Transferability Error Bound (Theorem 4.3)**: When the loss is bounded by $\beta$, with probability $\geq 1 - \delta$:

$$TE(z, \epsilon) \leq 4\mathcal{R}_N(\mathcal{Z}) + \sqrt{\frac{18\gamma \beta^2}{N} \ln \frac{2^{2+1/\gamma} H_\alpha^{1/\alpha}(\mathcal{P}_{\Theta^N} \| \mathcal{P}_{\bigotimes_{i=1}^{N} \Theta})}{\delta}}$$

where $H_\alpha(\cdot \| \cdot)$ is the Hellinger integral, which measures the divergence between the joint distribution $\mathcal{P}_{\Theta^N}$ and the product of marginals $\mathcal{P}_{\bigotimes \Theta}$. The first term is controlled by the complexity and the number of models, while the second term captures the dependency among models via the Hellinger integral. **The more independent (diverse) the surrogate models are, the smaller the Hellinger term becomes, thus tightening the upper bound.**

The key technical challenge in the proof is that surrogate models are typically trained on similar tasks, and their parameters are not independent. The authors introduce recent information-theoretic results from Esposito & Mondelli (2024), circumventing the independence assumption of traditional generalization theories.

### Three Practical Guidelines

By combining the two theoretical results, the authors derive three actionable guidelines:

1. **Use more surrogate models**: As $N$ increases, $\mathcal{R}_N(\mathcal{Z}) \propto 1/\sqrt{N}$ decreases, and the second term also tightens with $N$.
2. **Increase surrogate model diversity**: The more independent the models are, the closer the joint distribution is to the product of marginals, reducing the Hellinger term.
3. **Reduce model complexity (prevent overfitting)**: Decreasing the weight norm $T$ reduces complexity, which corresponds to regularization or simpler architectures.

## Key Experimental Results

Experiments are conducted on MNIST, Fashion-MNIST, and CIFAR-10 datasets, constructing a pool of 54 models (3 MLP variants $\times$ 3 CNN variants $\times$ 3 data transformations $\times$ 3 weight decays). A separately trained ResNet-18 is used as the black-box target model.

### Main Results: Attack Dynamics Verification (MI-FGSM, 20-step attack)

| Dataset | Model | ASR Trend | Loss Trend | Variance Trend |
|--------|------|----------|-----------|---------------|
| MNIST | MLP | Monotonically increases with steps | Continually increases | Increases then decreases |
| MNIST | CNN | Monotonically increases with steps | Continually increases | Increases then decreases |
| Fashion-MNIST | MLP | Monotonically increases with steps | Continually increases | Increases then decreases |
| Fashion-MNIST | CNN | Monotonically increases with steps | Continually increases | Increases then decreases |
| CIFAR-10 | MLP | Increases with steps | Continually increases | Continually increases |
| CIFAR-10 | CNN | Increases with steps | Continually increases | Decreases with small $\lambda$, increases with large $\lambda$ |

Key Observation: The magnitude of vulnerability (loss) is about **10 times** that of diversity (variance), dominating the decomposition. The upward trend of ASR aligns with the increase in vulnerability, validating the effectiveness of the decomposition.

### Ablation Study: Impact of Ensemble Size (Gradually increasing from 1 to 18 models)

| Dataset | 1 Model ASR | 9 Model ASR | 18 Model ASR | Vulnerability | Diversity |
|--------|-----------|-----------|------------|---------------|-----------|
| MNIST | ~40% | ~75% | ~85% | Continually increases | Continually increases |
| Fashion-MNIST | ~25% | ~55% | ~65% | Continually increases | Continually increases |
| CIFAR-10 | ~15% | ~40% | ~50% | Continually increases | Orders of magnitude smaller than vulnerability |

Across all datasets, increasing the number of ensembled models significantly improves average ASR, verifying the theoretical predictions. On MNIST and Fashion-MNIST, both vulnerability and diversity increase; on CIFAR-10, diversity occasionally decreases, but its magnitude is only 1/100 of vulnerability, having a negligible impact on ASR.

## Key Findings

1. **Vulnerability Dominates Transferability**: In the vulnerability-diversity decomposition, the magnitude of the vulnerability term is far larger than that of the diversity term, serving as the primary driver for boosting ASR.
2. **Complex Behavior of Diversity**: Diversity does not always change monotonically—it exhibits a "bell-shaped" (increasing then decreasing) pattern on MNIST/Fashion-MNIST, and its behavior varies on CIFAR-10 depending on model types and regularization strength.
3. **Complexity-Diversity Trade-off**: Experiments reveal that increasing weight decay $\lambda$ (reducing complexity) can increase variance, suggesting that reducing complexity during the overfitting phase helps simultaneously boost transferability.
4. **Validation of Generalization-Transferability Analogy**: The mathematical form of transferability error is highly parallel to generalization error, theoretically validating the long-standing heuristic view that "adversarial transferability is analogous to model generalization."

## Highlights & Insights

- **First Theoretical Framework for Model Ensemble Attacks**: Fills the theoretical gap in this direction with three concise and mathematically sound definitions (transferability error, diversity, and ensemble Rademacher complexity).
- **Clever Adaptation of Bias-Variance Decomposition**: Transfers the classic tool of learning theory from "model generalization" to "adversarial transferability", leading to a novel perspective.
- **Handling Non-independence with Information Theory**: Traditional generalization bounds rely on the independence assumption of data, which is violated since surrogate models are not independent. Introducing the Hellinger integral and state-of-the-art information theory techniques is the key innovation.
- **Theoretically Backed Practical Guidelines**: Although the three guidelines (more models, more diverse, lower complexity) are already widely used empirically, this work is the first to provide rigorous theoretical justifications.

## Limitations & Future Work

1. **Theory-Practice Gap**: The vulnerability-diversity decomposition is based on mean squared error loss, whereas practical attacks typically employ cross-entropy loss (though a rougher KL-divergence version is provided in the appendix).
2. **Limited Model Scale**: Experiments are only conducted on shallow MLPs and CNNs, lacking comprehensive validation on modern architectures like ViTs or large-scale ImageNet pre-trained models.
3. **Uncomputable Hellinger Term**: The $H_\alpha(\mathcal{P}_{\Theta^N} \| \mathcal{P}_{\bigotimes \Theta})$ in the upper bound cannot be calculated directly in practice, limiting the utility of the bound.
4. **Lack of Optimal Ensemble Strategy**: Although the theory reveals the existence of vulnerability-diversity and complexity-diversity trade-offs, it does not propose specific balancing strategies or new algorithms.
5. **"Same Parameter Space" Assumption**: The main theorem assumes that the surrogate and target models share the same parameter space. Although an extended discussion is included in the appendix, the applicability to scenarios with significant architectural differences in practice remains to be tested.

## Related Work & Insights

- **Bias-Variance Decomposition** (Geman et al., 1992): The vulnerability-diversity decomposition in this paper is directly analogous to the bias-variance decomposition.
- **Rademacher Complexity** (Bartlett & Mendelson, 2002): A classic tool from generalization theory, extended in this paper to the model ensemble attack scenario.
- **Information-Theoretic Generalization Bounds** (Esposito & Mondelli, 2024): The latest mathematical tools to handle non-independent samples, on which the proofs of this paper crucially rely.
- **Ensemble Learning Diversity** (Wood et al., 2024; Ortega et al., 2022): The definition of Diversity draws on recent advances in ensemble learning theory.
- **Adversarial Transferability Theory** (Wang & Farnia, 2023; Fan et al., 2024): The former analyzes single-model transferability from the perspective of generalization error, while the latter decomposes transferability into local efficacy and transfer loss.

This paper inspires an intriguing direction: **systematically transferring various tools from generalization theory to adversarial transferability analysis.** For instance, can PAC-Bayes bounds or VC-dimension theory also yield valuable insights into transferability?

## Rating

| Dimension | Score | Description |
|------|------|------|
| Theoretical Depth | ⭐⭐⭐⭐⭐ | First comprehensive theoretical framework with three definitions, two core theorems, and new information-theoretic techniques. |
| Experimental Thoroughness | ⭐⭐⭐ | Sufficient verification with 54 models, but the model sizes and datasets are relatively simple. |
| Novelty | ⭐⭐⭐⭐⭐ | Systematically introduces learning theory tools to the theoretical analysis of model ensemble attacks for the first time. |
| Value | ⭐⭐⭐⭐ | The three guidelines have practical value, though specific new algorithms are missing. |
| Writing Quality | ⭐⭐⭐⭐ | Clear logic, intuitive illustrations, many symbols but well-defined. |
| Overall Rating | ⭐⭐⭐⭐ | A high-quality, theory-oriented work that opens up a new direction for the theoretical study of model ensemble attacks. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Identifying and Understanding Cross-Class Features in Adversarial Training](identifying_and_understanding_cross-class_features_in_adversarial_training.md)
- [\[NeurIPS 2025\] Understanding and Improving Adversarial Robustness of Neural Probabilistic Circuits](../../NeurIPS2025/ai_safety/understanding_and_improving_adversarial_robustness_of_neural_probabilistic_circu.md)
- [\[AAAI 2026\] Transferable Hypergraph Attack via Injecting Nodes into Pivotal Hyperedges](../../AAAI2026/ai_safety/transferable_hypergraph_attack_via_injecting_nodes_into_pivotal_hyperedges.md)
- [\[CVPR 2026\] Towards Highly Transferable Vision-Language Attack via Semantic-Augmented Dynamic Contrastive Interaction](../../CVPR2026/ai_safety/towards_highly_transferable_vision-language_attack_via_semantic-augmented_dynami.md)
- [\[CVPR 2025\] MOS-Attack: A Scalable Multi-Objective Adversarial Attack Framework](../../CVPR2025/ai_safety/mos-attack_a_scalable_multi-objective_adversarial_attack_framework.md)

</div>

<!-- RELATED:END -->
