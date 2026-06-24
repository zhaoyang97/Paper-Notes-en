---
title: >-
  [Paper Note] Random Registers for Cross-Domain Few-Shot Learning
description: >-
  [ICML 2025][LLM (Other)][Cross-Domain Few-Shot Learning] This work discover that in cross-domain few-shot learning (CDFSL), learnable prompts impair generalization in the target domain, whereas replacing them with random noise (i.e., random registers) consistently improves performance. Based on this observation, the REAP method is proposed, which enhances attention perturbation by introducing random registers to semantic image regions, enabling efficient domain-agnostic featu…
tags:
  - "ICML 2025"
  - "LLM (Other)"
  - "Cross-Domain Few-Shot Learning"
  - "Vision Transformer"
  - "Random Registers"
  - "Attention Perturbation"
  - "Sharpness-Aware Minimization"
date: 2026-05-08
content_hash: 47550588aa398acf
---

# Random Registers for Cross-Domain Few-Shot Learning

**Conference**: ICML 2025  
**arXiv**: [2506.02843](https://arxiv.org/abs/2506.02843)  
**Code**: [shuaiyi308/REAP](https://github.com/shuaiyi308/REAP)  
**Area**: LLM Evaluation  
**Keywords**: Cross-Domain Few-Shot Learning, Vision Transformer, Random Registers, Attention Perturbation, Sharpness-Aware Minimization

## TL;DR

This work discover that in cross-domain few-shot learning (CDFSL), learnable prompts impair generalization in the target domain, whereas replacing them with random noise (i.e., random registers) consistently improves performance. Based on this observation, the REAP method is proposed, which enhances attention perturbation by introducing random registers to semantic image regions, enabling efficient domain-agnostic feature learning.

## Background & Motivation

Cross-domain few-shot learning (CDFSL) aims to transfer knowledge learned on a source domain (e.g., ImageNet) to a data-scarce target domain (e.g., medical image datasets). However, the substantial domain gap between the source and target domains makes transfer extremely challenging. Although Vision Transformers (ViTs) perform exceptionally well in many vision tasks, their transferability under extreme cross-domain scenarios remains insufficiently explored.

The starting point of this paper is an intriguing phenomenon: while Visual Prompt Tuning is a common approach to training ViTs, the authors discover that utilizing learnable prompts during source-domain training **actually harms** target-domain performance. More surprisingly, if these prompts are replaced with random Gaussian noise (termed "random registers"), the target-domain performance **consistently improves**. Furthermore, a larger number of registers yields better results, with optimal performance occurring at the maximum capacity allowed by the GPU memory.

This counter-intuitive phenomenon motivates the authors to investigate the underlying causes and propose an improved method accordingly.

## Method

### Overall Architecture

The proposed method, named **REAP** (Random Registers Enhanced Attention Perturbation), consists of two stages:

1. **Source-domain training stage**: Enhances perturbation to attention maps by clustering image tokens and replacing them with random registers, encouraging the model to extract domain-agnostic information.
2. **Target-domain fine-tuning stage**: Switches back to learnable registers, leveraging their property of absorbing domain information to assist the model in adapting to the target domain.

The input sequence of the ViT is the concatenation of the CLS token, image tokens, and registers:

$$f(P) = f(C(T^C, T(I), T^{R_1}, T^{R_2}, \cdots, T^{R_{\tilde{n}}}))$$

### Key Designs

#### 1. Phenomenon Discovery: Random Registers Outperform Learnable Registers

The authors first discover via attention visualization that:
- **Learnable registers**: The model fails to locate semantic regions in the target domain, instead focusing on background areas irrelevant to recognition.
- **Random registers**: Effectively guide the model to focus on the semantic objects within the image.

This is quantitatively validated using the sharpness of the loss landscape:

$$\text{Sharpness} = \max_{\epsilon}[L(A+\epsilon) - L(A)], \quad \epsilon \sim N(0, \sigma)$$

Experiments demonstrate that learnable registers significantly increase sharpness (leading to poor transferability), whereas random registers reduce sharpness (leading to good transferability).

#### 2. Theoretical Explanation: Random Registers ≈ Sharpness-Aware Minimization (SAM)

The effect of random registers in the attention map can be formulated as:

$$A_{i,j} = \frac{e^{Q_i K^T_j}}{\sum_{k=1}^{n} e^{Q_i K^T_k} + \sum_{k=1}^{\tilde{n}} e^{Q_i \tilde{K}^T_k}}$$

Since the keys for random registers $\tilde{K} = T^{R_i} W^K$ are random, the additional term in the denominator $\sum e^{Q_i \tilde{K}^T_k}$ essentially acts as random noise $\epsilon^R$. This is equivalent to the formulation of SAM:

$$L_{SAM} = \min_\omega [\max_\epsilon L(A + \epsilon^R)] + \lambda(\|\omega\|_2^2)$$

Consequently, random registers represent a **novel form of attention perturbation**, helping the model identify flatter minima in the loss landscape, thereby enhancing cross-domain transferability.

#### 3. Domain Information Analysis

Characteristic similarity between the source and target domains is measured using Centered Kernel Alignment (CKA) similarity:
- Learnable registers $\rightarrow$ Decreased CKA similarity $\rightarrow$ Extraction of source-domain-specific information $\rightarrow$ Overfitting to the source domain.
- Random registers $\rightarrow$ Increased CKA similarity $\rightarrow$ Learning of domain-agnostic information $\rightarrow$ Good generalization.

Learnable registers cause the model to treat visual patterns irrelevant to recognition, such as backgrounds, as critical cues for classification, leading to overfitting on the source domain.

#### 4. REAP: Enhanced Attention Perturbation

Directly adding a massive number of random registers is highly inefficient as it occupies substantial GPU memory. The core idea of REAP is to **add random registers onto the semantic regions of image tokens**, thereby increasing the proportion of perturbed information in the attention map.

Specific steps:
1. **Clustering**: Randomly select a large proportion of anchor patches (60%-80%) from the image patches $X \in R^{n \times d}$, then compute the cosine similarity between the anchors and other patches.
2. **Replacement**: Replace clusters whose similarities exceed a certain threshold with random registers: $T^{R_i} \sim N(0, \tau^2)$, where $\tau$ is a learnable parameter.
3. **Appending Extra Registers**: Append a small number (16) of extra random registers to the end of the sequence.

The attention map is partitioned into three components:

$$A_{i,j} = \frac{e^{Q_i K^T_j}}{\underbrace{\sum_{k=1}^{m} e^{Q_i K^T_k}}_{\text{保留图像}} + \underbrace{\sum_{k=1}^{n-m} e^{Q_i \bar{K}^T_k}}_{\text{图像扰动}} + \underbrace{\sum_{k=1}^{\tilde{n}} e^{Q_i \tilde{K}^T_k}}_{\text{寄存器扰动}}}$$

This design exploits ViT's attention dependency on contiguous regional patterns, allowing **a small number of registers to achieve strong perturbation** through cluster replacement.

### Loss & Training

**Source-domain stage**: Standard cross-entropy loss using the inputs processed by REAP.

$$L = \frac{1}{N} \sum_j^N L_{cls}(\phi(f(C(T^C, \tilde{T}, T^R))), y_j^S)$$

- Backbone learning rate is $10^{-5}$, classifier learning rate is $10^{-3}$, using Adam optimizer for 50 epochs.
- Anchor ratio and minimum drop ratio are set to 70%, with 16 extra registers, and the initial value of $\tau$ is 0.1.

**Target-domain stage**: Switch to learnable registers and fine-tune on the support set.

$$L = \frac{1}{N} \sum_j^N L_{cls}(\phi(f(C(T^C, T(I), T^L))), y_j^T)$$

- Register learning rate is $10^{-3}$, leveraging the property of learnable registers to absorb target-domain information.

## Key Experimental Results

### Main Results

5-way classification results (accuracy %) on four target domain datasets based on the ViT-S backbone:

| Method | Shot | ChestX | ISIC2018 | EuroSAT | CropDiseases | Average |
|------|------|--------|----------|---------|--------------|------|
| StyleAdv (CVPR'23) | 1 | 22.92 | 33.99 | 74.93 | 84.11 | 53.99 |
| FLoR (CVPR'24) | 1 | 23.26 | 35.49 | 73.09 | 83.55 | 53.85 |
| AttnTemp (NeurIPS'24) | 1 | 23.63 | 38.05 | 75.09 | 84.78 | 55.39 |
| **REAP (Ours)** | **1** | **24.17** | **38.67** | **75.97** | **85.33** | **56.04** |
| StyleAdv (CVPR'23) | 5 | 26.97 | 51.23 | 90.12 | 95.99 | 66.08 |
| AttnTemp (NeurIPS'24) | 5 | 28.03 | 54.91 | 90.82 | 96.66 | 67.61 |
| **REAP (Ours)** | **5** | **28.34** | **55.28** | **91.79** | **96.71** | **68.03** |

REAP consistently achieves the best average performance across all settings (with/without fine-tuning, 1-shot/5-shot, inductive/transductive).

### Ablation Study

| Configuration | CropDiseases | EuroSAT | ISIC2018 | ChestX | Average | Description |
|------|--------------|---------|----------|--------|------|------|
| Baseline | 94.61 | 89.29 | 46.16 | 26.21 | 64.07 | No registers |
| + Random Registers | 95.14 | 89.44 | 48.92 | 26.68 | 65.05 | Random registers only |
| + REAP | **95.68** | **90.53** | **52.80** | **27.98** | **66.75** | Full method |
| Random-mask | 91.23 | 84.42 | 43.89 | 24.06 | 60.90 | Random masking (harms performance) |
| Cluster-mask | 94.61 | 89.59 | 47.33 | 26.38 | 64.29 | Cluster masking without replacement |

### Key Findings

1. **Cluster Replacement vs. Random Masking**: Random masking of patches severely degrades performance (60.90 vs. 64.07 baseline), while cluster replacement combined with random registers significantly improves it (66.75). This indicates that clustering operations are critical for perturbing contiguous semantic regions.
2. **Target Domain Fine-Tuning Strategy**: Random registers are actually detrimental during target-domain fine-tuning (54.00 < 54.50 baseline), whereas learnable registers provide an effective boost (56.04), validating the rationale of the two-stage strategy.
3. **Cross-backbone Generalization**: Consistent improvements are observed across three pre-trained backbones (CLIP, iBOT, and DINO-ViT-Base). For instance, the average performance of CLIP increases from 58.17 to 60.93, and DINO-ViT-Base improves from 64.74 to 65.87.
4. **Hyperparameter Sensitivity**: An anchor ratio between 40% and 80% is effective, with a replacement ratio of 70% being optimal (beyond which performance drops sharply). The optimal number of extra registers is 16, and the noise standard deviation needs to be moderate.

## Highlights & Insights

1. **Counter-intuitive Core Discovery**: Learnable prompts are detrimental in cross-domain scenarios, which contradicts the successful experiences of prompt tuning in other tasks and reveals the unique challenges of cross-domain settings.
2. **Elegant Theoretical Explanation**: A connection is established between random registers and SAM, explaining from the perspective of loss landscape flatness why random noise instead aids transfer. The theoretical derivation is coherent and self-consistent.
3. **Simple and Efficient Design**: By performing cluster replacement on semantic image regions, REAP simulates the perturbation effect of a massive number of random registers using only a small set of registers (16).
4. **Symmetrical Beauty of the Two-Stage Strategy**: Random registers are utilized in the source domain to "remove domain-specific information", while learnable registers are used in the target domain to "inject domain-specific information". This elegantly exploits the complementary properties of both register types.

## Limitations & Future Work

1. **Limited to Vision Transformers**: The method relies on the attention mechanism of ViTs and cannot be directly applied to other architectures such as CNNs.
2. **Simplistic Clustering Strategy**: Cosine similarity clustering based on mean pixel values may be imprecise; more advanced semantic clustering methods warrant exploration.
3. **Hyperparameter Sensitivity**: Excessively high replacement ratios and anchor ratios dramatically degrade performance, necessitating careful tuning in practice.
4. **Source Domain Restricted to Natural Images**: The experiments utilize only miniImageNet as the source domain; exploring more diverse source domain configurations is a valuable direction.
5. **Approximated Theoretical Analysis**: The derivation equating random registers to SAM is an approximation and lacks rigorous theoretical guarantees.

## Related Work & Insights

- **Vision Registers (Darcet et al., 2024)**: Proposed adding extra tokens to the ViT input and discarding them before the output; the term "register" in this paper is derived from this work.
- **SAM (Foret et al., 2021)**: Sharpness-Aware Minimization framework; this paper explains random registers as a novel implementation of SAM.
- **FLoR (Zou et al., 2024a)**, **AttnTemp (Zou et al., 2024)**: Previous CDFSL SOTA methods that both focus on the transferability of the attention mechanism.
- **Visual Prompt Tuning (Jia et al., 2022)**: The VPT method; this paper identifies its negative effects in cross-domain scenarios.

**Insight**: In scenarios requiring cross-domain generalization, "learning less domain-specific information" can be more crucial than "learning more". Noise injection and regularization may prove more effective for cross-domain transfer than elaborately designed learnable modules, which also offers inspiration for other transfer learning tasks.

## Rating

| Dimension | Score (1-5) | Description |
|------|------------|------|
| Novelty | 4 | Counter-intuitive discovery + elegant theoretical explanation |
| Technical Depth | 4 | Connection to SAM, CKA analysis, comprehensive visualization |
| Experimental Thoroughness | 5 | 4 datasets, multiple settings, multiple backbones, thorough ablation |
| Writing Quality | 4 | Clear logic, progressing step-by-step from phenomenon to explanation to method |
| Value | 3 | Simple and easy-to-implement method, but the scenarios are somewhat specific |
| **Overall Score** | **4.0** | Solid work, with meaningful core discoveries |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Automatic Combination of Sample Selection Strategies for Few-Shot Learning](../../ACL2026/llm_nlp/automatic_combination_of_sample_selection_strategies_for_few-shot_learning.md)
- [\[ACL 2025\] HyGenar: An LLM-Driven Hybrid Genetic Algorithm for Few-Shot Grammar Generation](../../ACL2025/llm_nlp/hygenar_an_llm-driven_hybrid_genetic_algorithm_for_few-shot_grammar_generation.md)
- [\[ICML 2025\] Towards Universal Offline Black-Box Optimization via Learning Language Model Embeddings](towards_universal_offline_black-box_optimization_via_learning_language_model_emb.md)
- [\[ACL 2025\] Zero-Shot Belief: A Hard Problem for LLMs](../../ACL2025/llm_nlp/zero-shot_belief_a_hard_problem_for_llms.md)
- [\[ICML 2025\] Beyond Induction Heads: In-Context Meta Learning Induces Multi-Phase Circuit Emergence](beyond_induction_heads_in-context_meta_learning_induces_multi-phase_circuit_emer.md)

</div>

<!-- RELATED:END -->
