---
title: >-
  [Paper Note] Mind the Gap: Detecting Black-box Adversarial Attacks in the Making through Query Update Analysis
description: >-
  [CVPR 2025][AI Safety][Adversarial Attack Detection] This paper proposes GWAD, a black-box adversarial attack detection framework based on query update patterns (rather than input patterns). By introducing the Delta Similarity metric, it captures the inherent patterns of zero-order optimization in query-based attacks, achieving near 100% detection rate with extremely low false positive rates across 8 SOTA attacks (including the adaptive attack OARS)…
tags:
  - "CVPR 2025"
  - "AI Safety"
  - "Adversarial Attack Detection"
  - "Black-box Attack"
  - "Query Analysis"
  - "Delta Similarity"
  - "Stateful Defense"
date: 2026-05-08
content_hash: 62a27a3a1a8e4a61
---

# Mind the Gap: Detecting Black-box Adversarial Attacks in the Making through Query Update Analysis

**Conference**: CVPR 2025  
**arXiv**: [2503.02986](https://arxiv.org/abs/2503.02986)  
**Code**: [https://github.com/jpark04-qub/GWAD](https://github.com/jpark04-qub/GWAD)  
**Area**: AI Security  
**Keywords**: Adversarial Attack Detection, Black-box Attack, Query Analysis, Delta Similarity, Stateful Defense

## TL;DR

This paper proposes GWAD, a black-box adversarial attack detection framework based on query update patterns (rather than input patterns). By introducing the Delta Similarity metric, it captures the inherent patterns of zero-order optimization in query-based attacks, achieving near 100% detection rate with extremely low false positive rates across 8 SOTA attacks (including the adaptive attack OARS), significantly outperforming existing stateful defense methods.

## Background & Motivation

**Background**: Query-based black-box adversarial attacks represent the most realistic threat scenario, where attackers generate adversarial examples solely by querying the model via an API and analyzing the outputs. Existing defenses are mainly divided into two categories: "post-hoc" detection of generated adversarial examples, and "stateful defense" that detects anomalous queries during the attack generation process.

**Limitations of Prior Work**: Existing stateful defenses, such as Blacklight, PIHA, and Stateful Detection, analyze query patterns in the input space by monitoring the similarity between inputs to detect attacks. However, these methods suffer from two critical issues: (1) they are easily bypassed by the adaptive attack OARS, which carefully constructs query directions in the input space to evade detection; (2) they suffer from high false positive rates in high-similarity scenarios (e.g., continuous video frames).

**Key Challenge**: The similarity patterns in the input space are manipulable by attackers. Attackers can alter the distribution of queries in the input space to evade detection without affecting gradient estimation. Defenders need a detection signal that attackers cannot easily bypass.

**Goal**: To find a detection signal that is more robust and fundamental than input-space patterns to distinguish normal queries from attack queries while reducing false positive rates.

**Key Insight**: All query-based black-box attacks rely on zero-order optimization to estimate gradients, a process that requires structured input updates using random vectors. The patterns of these updates (differential relationships) are inherent properties of the attack and cannot be eliminated by transformations in the input space.

**Core Idea**: To detect ongoing black-box attacks by analyzing the similarity of the update differences (Delta Similarity) between consecutive queries, rather than the similarity of the queries themselves.

## Method

### Overall Architecture

The GWAD framework continuously monitors the sequence of queries sent to the protected model. For every three consecutive queries, it computes a Delta Similarity (DS) value. After collecting the 256 most recent DS values, it generates a Histogram of Delta Similarity (HoDS) feature, which is then fed into a pre-trained classifier to determine whether an attack is present. The entire pipeline: Query Stream $\rightarrow$ DS Computation $\rightarrow$ HoDS Feature Extraction $\rightarrow$ Attack Classification/Detection.

### Key Designs

1. **Delta Similarity (DS) Metric**:

    - **Function**: Captures the characteristics of update patterns in query sequences to distinguish between normal and attack behaviors.
    - **Mechanism**: For three consecutive queries $q_1, q_2, q_3$, compute the differences $\delta_1 = q_2 - q_1$ and $\delta_2 = q_3 - q_2$. The DS is then the cosine similarity between $\delta_1$ and $\delta_2$. In the zero-order optimization process, query updates consist of linear combinations of random vectors $u$. Due to the concentration of measure in high-dimensional spaces, these vectors are approximately orthogonal and equal in length, causing the DS distribution of attacks to exhibit distinct peak patterns (e.g., NES at $-0.7071$, HSJA at $-0.5$ and $\pm 1.0$), whereas normal queries have a high-variance DS distribution.
    - **Design Motivation**: DS operates on the "update space" rather than the "input space". Adaptive attacks like OARS evade detection by altering directions in the input space but cannot change the update structure of zero-order optimization itself.

2. **HoDS Feature Representation**:

    - **Function**: Transforms the DS sequence into a fixed-dimensional feature vector for classification.
    - **Mechanism**: Collect the 256 most recent DS values and construct a histogram over the range $[-1, 1]$ using 200 equally spaced bins, plus an additional bin for $DS=1.0$, followed by min-max normalization. This generates a final feature vector of size $1 \times 201$. Different attack methods have their own unique HoDS signatures.
    - **Design Motivation**: A histogram representation is naturally invariant to sequence order but sensitive to distribution patterns, and its fixed dimension makes it easy for the classifier to process. A window size of 256 is experimentally validated as optimal.

3. **Lightweight Attack Classifier**:

    - **Function**: Determines the presence of an attack and identifies the specific attack type.
    - **Mechanism**: A 6-layer fully connected network with ReLU activation and Log-Softmax output, trained for 100 epochs using SGD with a batch size of 128. It supports both multi-class (benign + various attacks) and binary (benign/attack) classification modes.
    - **Design Motivation**: Since the HoDS features already adequately encode the attack patterns, a simple classifier can achieve efficient distinction. The lightweight design ensures no significant latency is added to the server side.

### GWAD+ Enhancement: Screener Pre-screening

To address GWAD's vulnerability to massive benign-injection attacks, GWAD+ is proposed. The Screener performs lightweight pre-screening in the input space: it resizes queries to 32$\times$32, applies Canny edge detection to convert them to binary images, compresses them into 128-byte vectors, and compares them with historical queries in a FIFO queue. Low-similarity queries are deemed benign and directly released, while high-similarity queries are sent to GWAD for DS analysis. Consequently, even when an attacker injects up to 10$\times$ benign queries ($rb=1000\%$), GWAD+ still maintains a 99.2%+ detection rate.

## Key Experimental Results

### Main Results

| Attack Method | Blacklight Detection Rate | PIHA Detection Rate | GWAD Detection Rate |
|----------|------------------|------------|------------|
| BA | 23.96% | 38.08% | **99.98%** |
| HSJA | 97.86% | 98.75% | **100.00%** |
| NES | 99.96% | 94.66% | **100.00%** |
| QEBA(Unknown Attack) | 96.51% | 96.78% | **100.00%** |
| Surfree(Unknown Attack) | 98.77% | 70.96% | **100.00%** |

| OARS Adaptive Attack | Blacklight ASR | PIHA ASR | GWAD ASR |
|---------------|----------------|----------|----------|
| OARS-NES | 98% | 82% | **0%** |
| OARS-HSJA | 75% | 71% | **0%** |
| OARS-QEBA | 98% | 95% | **0%** |

### Ablation Study

| Dataset/Scenario | Blacklight FPR | PIHA FPR | GWAD FPR |
|------------|----------------|----------|----------|
| CIFAR-10 | 0.00% | 0.00% | **0.00%** |
| ImageNet | 0.16% | 0.14% | **0.03%** |
| FLIR ADAS(Automotive) | 2.17% | 1.43% | **0.00%** |
| BIRDSAI(High Similarity) | 16.92% | N/A | **1.29%** |
| Hollywood High Similarity | 25.47% | 26.19% | **17.99%** |

### Key Findings

- GWAD is completely immune to OARS adaptive attacks (ASR=0%), while Blacklight and PIHA are completely bypassed. This validates the insight that the "update space" is inherently more difficult for attackers to manipulate than the "input space".
- GWAD generalizes well across datasets and models: GWAD-CIFAR10 trained on CIFAR-10 achieves a 97.32% classification accuracy on ImageNet.
- In high-similarity scenarios (continuous video frames), GWAD's false positive rate is far lower than existing methods, achieving a 0% FPR in the FLIR autonomous driving scenario.
- Benign injection attacks must consume more than 2.5$\times$ the query budget to begin affecting GWAD detection, and combined with Screener (GWAD+), it maintains a 99%+ detection rate even with 10$\times$ injection.

## Highlights & Insights

- **Paradigm shift from input space to update space**: This represents a fundamental perspective shift in the field of adversarial attack detection. The mathematical structure of zero-order optimization acts as an unavoidable "fingerprint" of the attack, analogous to side channels in cryptography—attackers can hide their intent but cannot hide the inherent patterns of the computational process.
- **Generality of the DS metric**: Different attacks present different DS distribution characteristics (determined by their gradient estimation strategies), meaning GWAD can not only detect but also identify the type of attack, providing intelligence for subsequent targeted defenses.
- **Complementary design of GWAD+**: Combining input-space screening with update-space analysis provides defense-in-depth, with each covering the other's blind spots.

## Limitations & Future Work

- In extremely high-similarity scenarios (continuous video frames), the FPR is still 17.99%; although better than existing methods, it still needs improvement.
- Currently evaluated only on image classification tasks; whether black-box attacks on other modalities (text, audio) exhibit similar DS patterns remains to be verified.
- If attackers alter the random distribution parameters of zero-order optimization (discussed in the supplementary material), it may partially affect the DS pattern.
- Classifier training requires samples of known attacks; the generalization capability to handle entirely new zero-order optimization strategies requires further research.

## Related Work & Insights

- **vs Blacklight**: Blacklight uses quantized hashing in the input space to detect similar queries. It is effective against conventional attacks but is completely bypassed by OARS (ASR 98%). GWAD is immune to OARS due to working in the update space, which is fundamentally a dimensional upgrade.
- **vs PIHA**: PIHA uses perceptual hashing for statistical analysis, but is not applicable to grayscale images (as it requires color/hue information) and suffers from high FPR in high-similarity scenarios. GWAD has no such limitations.
- **vs Stateful Detection**: An early stateful defense that uses pretrained networks for dimensionality reduction followed by $L_2$ similarity calculation; it is computationally expensive and less accurate than Blacklight. GWAD is more lightweight and accurate.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Paradigm shift from input space to update space is a key breakthrough.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across 8 attacks, adaptive attacks, generalization, false positive rates, and multiple scenarios.
- Writing Quality: ⭐⭐⭐⭐ Clear logic, tightly integrating theory with experiments.
- Value: ⭐⭐⭐⭐⭐ Solves the core limitation of stateful defenses being bypassed by adaptive attacks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Detecting Out-of-Distribution through the Lens of Neural Collapse](detecting_out-of-distribution_through_the_lens_of_neural_collapse.md)
- [\[ACL 2025\] Multi-task Adversarial Attacks against Black-box Model with Few-shot Queries](../../ACL2025/ai_safety/multi-task_adversarial_attacks_against_black-box_model_with_few-shot_queries.md)
- [\[ICLR 2026\] Black-Box Privacy Attacks on Shared Representations in Multitask Learning](../../ICLR2026/ai_safety/black-box_privacy_attacks_on_shared_representations_in_multitask_learning.md)
- [\[CVPR 2025\] Detecting Backdoor Attacks in Federated Learning via Direction Alignment Inspection](detecting_backdoor_attacks_in_federated_learning_via_direction_alignment_inspect.md)
- [\[ICML 2026\] Mind the Gap: Mixtures of Gaussians in Approximate Differential Privacy](../../ICML2026/ai_safety/mind_the_gap_mixtures_of_gaussians_in_approximate_differential_privacy.md)

</div>

<!-- RELATED:END -->
