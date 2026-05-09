---
title: >-
  [Paper Note] Active Membership Inference Test (aMINT): Enhancing Model Auditability with Multi-Task Learning
description: >-
  [ICCV 2025][AI Safety][Membership Inference] This paper proposes Active MINT (aMINT), a multi-task learning framework that jointly trains a MINT model alongside the audited model during training, enabling detection of whether specific data was used for training with over 80% accuracy — significantly outperforming existing passive MINT and membership inference attack methods.
tags:
  - ICCV 2025
  - AI Safety
  - Membership Inference
  - Data Auditing
  - Multi-Task Learning
  - AI Trustworthiness
  - Model Transparency
date: 2026-05-08
content_hash: 4bedba7d8119506b
---

# Active Membership Inference Test (aMINT): Enhancing Model Auditability with Multi-Task Learning

**Conference**: ICCV 2025
**arXiv**: [2509.07879](https://arxiv.org/abs/2509.07879)
**Code**: [GitHub](https://github.com/DanieldeAlcala/Membership-Inference-Test.git)
**Area**: AI Security & Auditability
**Keywords**: Membership Inference, Data Auditing, Multi-Task Learning, AI Trustworthiness, Model Transparency

## TL;DR
This paper proposes Active MINT (aMINT), a multi-task learning framework that jointly trains a MINT model alongside the audited model during training, enabling detection of whether specific data was used for training with over 80% accuracy — significantly outperforming existing passive MINT and membership inference attack methods.

## Background & Motivation
With the rapid advancement of AI, regulations such as the EU AI Act (June 2024) and the US White House memorandum (October 2024) mandate auditing of AI models to ensure lawful and compliant use of training data. Data owners have the right to know whether their data was used to train a model, while model developers may use protected data (e.g., biometric data, copyrighted content) without authorization.

Membership Inference Test (MINT), proposed in 2024, aims to detect whether specific data was used to train an AI model. Unlike membership inference attacks (MIA), MINT serves as an auditing tool rather than an adversarial technique, and permits a degree of cooperation with model developers (e.g., limited model access), which is reasonable under existing regulatory frameworks.

However, existing Passive MINT methods train the MINT model only after the audited model has finished training, which limits detection accuracy. The root cause is that the MINT model must analyze the internal activation patterns of the audited model to determine data membership, but if the audited model itself is not optimized for this task, the distinguishability between member and non-member data in its activations is inherently weak.

The paper's starting point is: if model developers actively participate in the auditing process (Active MINT) by jointly optimizing the MINT model during the training of the audited model, then the shared underlying features can simultaneously serve both the main task and the auditing task, substantially improving MINT detection accuracy.

**Core Idea**: Incorporate "data auditability" as one of the training objectives within a multi-task learning framework, jointly training the audited model and the MINT model.

## Method

### Overall Architecture
Active MINT constructs an augmented audited model $M^+$ consisting of two components: (1) the audited model $M$, which performs the original task (e.g., image classification); and (2) the MINT model $T$, which determines whether an input sample belongs to the training set. The two models share lower network layers and diverge into two branches at an extraction point. Training data $\mathcal{D}$ (50%) and external data $\mathcal{E}$ (50%) are mixed into batches for joint multi-task training.

### Key Designs
1. **Multi-Task Learning Architecture**:

    - **Function**: Jointly trains the MINT task with the main task, enabling the lower network layers to simultaneously encode data membership information.
    - **Mechanism**: The first several layers are shared (orange), after which the network branches into the audited model path (gray) and the MINT model path (blue). Samples from $\mathcal{D}$ pass through both paths, while samples from $\mathcal{E}$ pass only through the MINT path. The multi-task loss is:
    $$\mathcal{L}_{\text{Multi-task}} = \lambda_1 \frac{\mathcal{L}_{\text{Audited}}}{\|\mathcal{L}_{\text{Audited}}\|} + \lambda_2 \frac{\mathcal{L}_{\text{MINT}}}{\|\mathcal{L}_{\text{MINT}}\|} + R(\mathbf{w}^+)$$
    - **Design Motivation**: In passive MINT, the audited model is fully trained and its activation patterns are fixed. Active MINT allows the shared layers to receive gradients from both tasks during backpropagation, making the lower-level features more "membership-discriminative."

2. **Auxiliary Auditable Data (AAD) Extraction**:

    - **Function**: Extracts activation maps from two intermediate network layers as input to the MINT model.
    - **Mechanism**: Activation maps $\text{AAD} = N(d|\mathbf{w}')$ are selected from two intermediate layers and analyzed via CNN or fully connected networks to predict a binary classification result (training data vs. external data).
    - **Design Motivation**: Two-level activation maps provide information at different granularities — lower layers capture texture-level differences while higher layers capture semantic-level differences. This work extends prior approaches that used only a single activation layer, finding that the combination of two layers yields better performance.

3. **Comparative Analysis of Three Setups**:

    - **Function**: Systematically evaluates the impact of activation map extraction position — Entry (near input layer), Middle (intermediate layer), and Output (near output layer).
    - **Key Finding**: Entry and Middle setups perform comparably and both substantially outperform the Output setup.
    - **Design Motivation / Analysis**: The auditing task pursues generalization (consistent performance on training and test data), while the MINT task pursues discrimination (different behavior for member vs. non-member data) — these two objectives are fundamentally in tension. The Output setup shares the most network layers between the two tasks, maximizing this conflict. The Entry setup minimizes shared layers and thus minimizes inter-task interference, yielding a slight advantage in auditing accuracy.

### Loss & Training
- $\mathcal{L}_{\text{Audited}}$: Depends on the main task (e.g., cross-entropy for classification).
- $\mathcal{L}_{\text{MINT}}$: Binary cross-entropy for predicting training set membership.
- Both losses are normalized by their respective norms to maintain comparable magnitudes.
- The ratio $\lambda_2 / \lambda_1$ is adjusted by task difficulty (10 for MNIST, 10000 for Tiny ImageNet).
- L2 regularization $R(\mathbf{w}^+)$ is applied to prevent overfitting.
- Early stopping is employed, typically at 50–100 epochs.

## Key Experimental Results

### Main Results

| Dataset | Architecture | Passive MINT | Active MINT | MINT Gain | Auditing Impact |
|--------|------|-------------|------------|----------|------------|
| MNIST | ResNet50 | 0.51 | **0.83** | +32% | 0.97→0.97 (no loss) |
| CIFAR-10 | DenseNet121 | 0.60 | **0.86** | +26% | 0.80→0.80 (no loss) |
| GTSRB | Xception | 0.59 | **0.86** | +27% | 0.99→0.99 (no loss) |
| Tiny ImageNet | Xception | 0.65 | **0.88** | +23% | 0.28→0.28 (no loss) |
| CASIA WebFace | MobileNet | 0.60 | **0.86** | +26% | 0.17→0.15 (minor drop) |
| CIFAR-10 | ResNet50 | 0.66 | **0.86** | +20% | 0.55→0.53 (minor drop) |

### Ablation Study

| Setup | MNIST MINT↑ | MNIST Aud | CIFAR MINT↑ | CIFAR Aud |
|-------|------------|-----------|-------------|-----------|
| Entry | 0.83–0.86 | 0.94–0.99 | 0.86–0.91 | 0.19–0.80 |
| Middle | 0.81–0.88 | 0.92–0.99 | 0.86–0.91 | 0.19–0.80 |
| Output | 0.77–0.82 | 0.80–0.98 | 0.82–0.88 | 0.19–0.76 |

The Output setup consistently performs worst across all scenarios, corroborating the theoretical analysis that greater layer sharing intensifies inter-task conflict.

### Comparison with MIA Methods

| Method | CIFAR-10 | GTSRB |
|------|----------|-------|
| Salem et al. MIA | 0.61 | 0.67 |
| Yeom et al. MIA | 0.64 | 0.79 |
| Watson et al. MIA | 0.63 | 0.79 |
| Passive MINT | 0.66 | 0.61 |
| **Active MINT (Ours)** | **0.86** | **0.86** |

Active MINT substantially outperforms all MIA methods and Passive MINT, despite the experimental conditions between MIA and MINT not being fully identical.

### Key Findings
- Active MINT consistently outperforms Passive MINT across all model architectures and datasets, with MINT accuracy gains of 20–32 percentage points.
- In most cases, auditing task performance is almost entirely unaffected (<1% drop) or completely preserved.
- The method generalizes effectively from lightweight MobileNet to complex Vision Transformers.
- The Entry setup is the optimal choice, achieving the best performance on both the MINT and auditing tasks.

## Highlights & Insights
- **Elegant application of multi-task learning**: Auditability is incorporated as a training objective within network optimization, rather than applied as a post-hoc analysis.
- **Thorough consideration of practical deployment**: The paper discusses deployment strategies including Docker containers, digitally signed logs, and multi-party computation to ensure trustworthy training.
- **Alignment with regulatory frameworks**: The work directly addresses the requirements of the EU AI Act and the US White House memorandum.
- **Insightful analysis of task conflict**: The paper clearly articulates the fundamental tension between the auditing task (which pursues generalization) and the MINT task (which exploits overfitting), and empirically validates this analysis through the setup experiments.

## Limitations & Future Work
- The approach requires model developers to actively participate in MINT model training, which may not hold in adversarial scenarios where developers are uncooperative.
- The training dataset is split evenly (50% training data + 50% external data), which constrains main task performance.
- Experiments are currently limited to image classification; extension to LLMs, generative models, and other complex settings remains unexplored.
- Whether gradient-based MINT (gMINT) can be combined with Active MINT has not been investigated.
- In settings with a large number of classes (e.g., Tiny ImageNet with 200 classes), the inherently low auditing accuracy limits a comprehensive evaluation of the method.

## Related Work & Insights
- **vs. Passive MINT**: The key distinction lies in the timing of training — passive MINT is a post-hoc audit, while active MINT is an in-training audit. Active MINT does not require developers to expose training data or grant model access, but does require developer participation in training.
- **vs. MIA (Shokri et al.)**: MIA is an attack that requires training shadow models to simulate the target model's behavior; MINT is an auditing tool that allows developer cooperation. Active MINT further eliminates the need for shadow models through joint training.
- **vs. Nasr et al. (adversarial regularization)**: Nasr et al. train models to resist MIA (making inference attacks harder to succeed), while this paper trains models to facilitate MINT (making auditing easier to succeed) — the two objectives are diametrically opposed.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The active MINT paradigm of incorporating auditability as a training objective is novel and theoretically well-motivated.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Six architectures × five datasets × three setups constitute an exceptionally comprehensive evaluation.
- **Writing Quality**: ⭐⭐⭐⭐ Well-structured with clear concept definitions; the deployment discussion adds practical value.
- **Value**: ⭐⭐⭐⭐ Directly addresses the pressing needs of current AI regulation, with clear societal value and application potential.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Toward Enhancing Representation Learning in Federated Multi-Task Settings](../../ICLR2026/ai_safety/toward_enhancing_representation_learning_in_federated_multi-task_settings.md)
- [\[ICCV 2025\] Find a Scapegoat: Poisoning Membership Inference Attack and Defense to Federated Learning](find_a_scapegoat_poisoning_membership_inference_attack_and_defense_to_federated_.md)
- [\[AAAI 2026\] Privacy Auditing of Multi-Domain Graph Pre-Trained Model under Membership Inference Attack](../../AAAI2026/ai_safety/privacy_auditing_of_multi-domain_graph_pre-trained_model_under_membership_infere.md)
- [\[ICCV 2025\] FRET: Feature Redundancy Elimination for Test Time Adaptation](fret_feature_redundancy_elimination_for_test_time_adaptation.md)
- [\[ICCV 2025\] Vulnerability-Aware Spatio-Temporal Learning for Generalizable Deepfake Video Detection](vulnerability-aware_spatio-temporal_learning_for_generalizable_deepfake_video_de.md)

</div>

<!-- RELATED:END -->
