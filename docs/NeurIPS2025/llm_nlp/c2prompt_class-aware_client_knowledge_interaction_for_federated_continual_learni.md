---
title: >-
  [Paper Note] C²Prompt: Class-aware Client Knowledge Interaction for Federated Continual Learning
description: >-
  [NeurIPS 2025][LLM/NLP][federated continual learning] To address class-level knowledge inconsistency during prompt communication in federated continual learning, C²Prompt is proposed…
tags:
  - "NeurIPS 2025"
  - "LLM/NLP"
  - "federated continual learning"
  - "prompt learning"
  - "class-aware aggregation"
  - "distribution compensation"
  - "knowledge conflict"
date: 2026-05-08
content_hash: 2c33a2a6691b0dbd
---

# C²Prompt: Class-aware Client Knowledge Interaction for Federated Continual Learning

**Conference**: NeurIPS 2025
**arXiv**: [2509.19674](https://arxiv.org/abs/2509.19674)  
**Code**: [https://github.com/zhoujiahuan1991/NeurIPS2025-C2Prompt](https://github.com/zhoujiahuan1991/NeurIPS2025-C2Prompt)  
**Area**: LLM/NLP
**Keywords**: federated continual learning, prompt learning, class-aware aggregation, distribution compensation, knowledge conflict

## TL;DR
To address class-level knowledge inconsistency during prompt communication in federated continual learning, C²Prompt is proposed, which explicitly enhances class-level knowledge coherence across clients via two mechanisms: Local Class Distribution Compensation (LCDC) and Class-aware Prompt Aggregation (CPA). The method achieves an Avg accuracy of 87.20% on ImageNet-R, surpassing the previous SOTA Powder by 2.51%.

## Background & Motivation

**Background**: Federated Continual Learning (FCL) requires distributed clients to learn from continuously arriving task data under privacy constraints. Prompt-based methods (e.g., CODAPrompt + FedAvg), which maintain task-specific prompts while freezing the pre-trained backbone, have shown strong performance in FCL.

**Limitations of Prior Work**: Existing prompt-based FCL methods overlook **class-level knowledge consistency** during server-side prompt aggregation: (a) different clients hold heterogeneous data distributions for the same class (intra-class distribution gap), leading to semantically inconsistent representations; (b) inter-prompt class-wise relevance is ignored, causing irrelevant or even conflicting knowledge to be fused during aggregation.

**Key Challenge**: Lack of class-level consistency in prompt communication → knowledge conflicts among new prompts → interference with old prompts → simultaneously exacerbating spatial forgetting (across clients) and temporal forgetting (across tasks).

**Goal**: (a) How to compensate for intra-class distribution bias introduced by non-IID data at the client side? (b) How to perform precise prompt aggregation on the server side based on class-level relevance?

**Key Insight**: The problem is approached from the perspective of class-level knowledge coherence — applying distribution compensation at the data input level (LCDC) and class-aware weighting at the parameter aggregation level (CPA).

**Core Idea**: Estimate the global class distribution to compensate for local distribution bias, and employ a prompt-class affinity matrix for class-aware aggregation — a dual-pronged strategy to resolve knowledge conflicts in FCL.

## Method

### Overall Architecture

C²Prompt builds upon the CODAPrompt architecture, learning two types of prompts on a frozen ViT-B/16:
- **Local Class Distribution Compensation Prompts** $\mathcal{P}^c_{t,k}$: one per class, aligning local features to the global class distribution.
- **Local Discriminative Prompts** $\mathcal{P}^d_{t,k}$: standard CODAPrompt prompts for learning classification knowledge.

Training proceeds in two stages: Round 0 performs global distribution estimation and LCDC training; Rounds 1–$N_r$ perform discriminative prompt learning with CPA aggregation.

### Key Designs

1. **Global Distribution Estimation**:

    - **Function**: Aggregates per-class distribution statistics from all clients on the server to estimate the global distribution of each class.
    - **Mechanism**: Assuming the features of class $i$ on client $k$ follow a Gaussian distribution $\mathcal{N}(\mu^t_{i,k}, (\sigma^t_{i,k})^2)$, the global mean and variance are derived via moment estimation of a Gaussian mixture: $\mu^g_i = \sum_k \mu^t_{i,k} p^t_{k,i}$ and $(\sigma^g_i)^2 = \sum_k ((\mu^t_{i,k})^2 + (\sigma^t_{i,k})^2) p^t_{k,i} - (\mu^g_i)^2$.
    - **Design Motivation**: Only means and variances are transmitted (no raw data), preserving privacy while obtaining a global distributional view. Communication overhead is minimal (sparse distribution parameters).

2. **Local Class Distribution Compensation (LCDC)**:

    - **Function**: Learns class-specific compensation prompts to align local features to the global distribution.
    - **Mechanism**: For each class $i$, a compensation prompt $\mathbf{p}^c_i \in \mathbb{R}^{L_c \times d}$ is learned and appended to the input tokens before passing through the frozen ViT. A distribution alignment loss $\mathcal{L}_c = -\frac{1}{2}(f_{x,p} - \mu^g_i)^\top (\Sigma^g_i)^{-1} (f_{x,p} - \mu^g_i)$ maximizes the likelihood of output features under the global Gaussian distribution.
    - **Design Motivation**: No data generation or raw data sharing is required; local non-IID bias is mitigated purely through prompt-based feature modulation. After training, the prompts are frozen and serve as "distribution correctors" for subsequent discriminative learning.

3. **Class-aware Prompt Aggregation (CPA)**:

    - **Function**: Performs weighted aggregation on the server based on prompt-class affinity, rather than simple averaging.
    - **Mechanism**: During training, the cumulative matching score between each prompt and each class is recorded online (client histogram $H^i_k$), which is uploaded to the server to form the matrix $\mathbf{H}^t_g \in \mathbb{R}^{KN \times |\mathcal{C}_t|}$. An inter-prompt correlation matrix is then computed as $W^t_g = \text{softmax}(\mathbf{H}^t_g (\mathbf{H}^t_g)^\top / \tau)$, and the aggregated prompts are obtained via $\mathbf{P}^{t*}_g = W^t_g \mathbf{P}^t_g$.
    - **Design Motivation**: Prompts with similar class affinities should receive higher aggregation weights, reducing interference from irrelevant class knowledge. Histogram accumulation incurs virtually zero additional overhead during online learning.

4. **Discriminative Learning + Knowledge Distillation**:

    - **Function**: Standard classification learning with cross-round knowledge retention.
    - The total loss is $\mathcal{L}_d = \mathcal{L}_{ce} + \beta \mathcal{L}_{kd}$, where $\mathcal{L}_{kd}$ is the distillation loss from Powder.
    - Compensation prompts are applied with probability 0.5, balancing information from both original and compensated data.

### Loss & Training
- Backbone: ViT-B/16 (pre-trained on ImageNet-21k, frozen)
- Discriminative prompts: $N=8$, $L_p=10$, $d=768$; compensation prompts: $L_c=3$
- Number of clients $K=5$; communication rounds per task $N_r=3$
- Optimizer: Adam, lr=0.01

## Key Experimental Results

### Main Results

| Method | Venue | ImageNet-R Avg↑ | ImageNet-R AIA↑ | DomainNet Avg↑ | DomainNet AIA↑ |
|--------|-------|-----------------|-----------------|----------------|----------------|
| FedWEIT | ICML2021 | 71.10 | 74.30 | 67.84 | 69.63 |
| GLFC | CVPR2022 | 72.96 | 75.21 | 69.75 | 70.34 |
| Fed-CODAP | CVPR2023 | 79.65 | 75.14 | 72.47 | 72.84 |
| Powder | ICML2024 | 84.69 | 84.08 | 75.98 | 77.28 |
| **C²Prompt** | **Ours** | **87.20** | **85.93** | **78.88** | **77.55** |

Ours surpasses Powder by 2.51% Avg on ImageNet-R and by 2.90% on DomainNet.

### Ablation Study

| Configuration | ImageNet-R Avg | Note |
|---------------|---------------|------|
| Baseline (Powder) | 84.69 | baseline |
| + LCDC only | 86.57 (+1.88) | distribution compensation is effective |
| + CPA only | 86.02 (+1.33) | class-aware aggregation is effective |
| + LCDC + CPA (Full) | 87.20 (+2.51) | two components are complementary |

### Key Findings
- LCDC and CPA resolve knowledge inconsistency at the input and parameter levels respectively, exhibiting strong complementarity.
- C²Prompt is the only method achieving a negative forgetting measure (FM < 0) on the large-scale DomainNet dataset.
- Forward transfer (FT) shows the largest improvement: +3.15% on ImageNet-R and +2.59% on DomainNet, indicating that global distribution estimation effectively facilitates new task learning.
- Communication overhead increases by only 0.6% over Powder, with no additional parameters or computation at inference time.

## Highlights & Insights
- The **privacy–efficiency trade-off** in global distribution estimation is elegantly designed: only means and variances are transmitted — no raw data, no gradients — with minimal communication overhead yet effective mitigation of non-IID gaps.
- **"Free" implementation of class-aware aggregation**: client histograms are accumulated online during training at zero additional cost, yet provide precise prompt-class affinity information as aggregation weights.
- Prompt attention visualizations (Figure 5) intuitively demonstrate that CPA focuses prompts on discriminative regions, whereas Powder's prompt attention is diffuse.

## Limitations & Future Work
- The Gaussian assumption may be inaccurate for complex multimodal distributions, particularly when intra-class sub-cluster structures exist.
- Validation is limited to ViT-B/16 and image classification; extension to larger backbones or NLP tasks has not been explored.
- The number of clients is fixed at 5; scalability to larger federations (e.g., 50+ clients) remains unverified.
- The compensation prompt usage probability $p=0.5$ is fixed; an adaptive probability strategy could be explored.

## Related Work & Insights
- **vs. Powder (ICML2024)**: Powder introduces knowledge distillation for cross-round retention; this work builds upon it by adding class-level distribution compensation and class-aware aggregation. C²Prompt retains Powder's distillation loss.
- **vs. CODAPrompt**: C²Prompt adopts CODA as the base prompt architecture, with the key difference that aggregation is replaced from simple FedAvg to class-aware weighted aggregation.
- **vs. PILoRA/LoRM**: LoRA-based FCL methods perform substantially worse than prompt-based methods (PILoRA achieves only 45.43%), highlighting the inherent advantage of prompt learning in FCL.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The combination of global distribution estimation and class-aware aggregation is clear and effective, though each individual module has precedents.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Three datasets, 12 baselines, ablation studies, and visualizations are thorough; however, large-scale client experiments are absent.
- **Writing Quality**: ⭐⭐⭐⭐ Problem formulation is clear, though the notation is dense and derivations are deferred to the appendix.
- **Value**: ⭐⭐⭐⭐ FCL is a practically motivated research direction; the proposed method is pragmatic with low overhead.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] System Prompt Optimization with Meta-Learning](system_prompt_optimization_with_meta-learning.md)
- [\[AAAI 2026\] Improving Sustainability of Adversarial Examples in Class-Incremental Learning](../../AAAI2026/llm_nlp/improving_sustainability_of_adversarial_examples_in_class-incremental_learning.md)
- [\[ACL 2026\] Model-Agnostic Meta Learning for Class Imbalance Adaptation](../../ACL2026/llm_nlp/model-agnostic_meta_learning_for_class_imbalance_adaptation.md)
- [\[NeurIPS 2025\] The Rise of Parameter Specialization for Knowledge Storage in Large Language Models](the_rise_of_parameter_specialization_for_knowledge_storage_in_large_language_mod.md)
- [\[ICCV 2025\] Any-SSR: How Recursive Least Squares Works in Continual Learning of Large Language Models](../../ICCV2025/llm_nlp/any-ssr_how_recursive_least_squares_works_in_continual_learning_of_large_languag.md)

</div>

<!-- RELATED:END -->
