---
title: >-
  [Paper Note] CAVGAN: Unifying Jailbreak and Defense of LLMs via Generative Adversarial Attacks
description: >-
  [ACL 2025][LLM Safety][Jailbreak attack] This paper proposes the CAVGAN framework, which utilizes generative adversarial networks to simultaneously learn jailbreak attacks (generator) and safety defense (discriminator) within the internal representation space of LLMs. This is the first work to unify attack and defense into a single framework for mutual enhancement, achieving an average attack success rate of 88.85% and an average defense success rate of 84.17%.
tags:
  - "ACL 2025"
  - "LLM Safety"
  - "Jailbreak attack"
  - "defense"
  - "GAN"
  - "concept activation vector"
  - "adversarial training"
date: 2026-05-08
content_hash: f24cc16e94aff07e
---

# CAVGAN: Unifying Jailbreak and Defense of LLMs via Generative Adversarial Attacks

**Conference**: ACL 2025  
**arXiv**: [2507.06043](https://arxiv.org/abs/2507.06043)  
**Code**: [GitHub](https://github.com/NLPGM/CAVGAN)  
**Institution**: School of Computer Science, Wuhan University; Zhongguancun Laboratory  
**Area**: AI Safety  
**Keywords**: Jailbreak attack, defense, GAN, concept activation vector, LLM safety, adversarial training

## TL;DR

This paper proposes the CAVGAN framework, which utilizes generative adversarial networks to simultaneously learn jailbreak attacks (generator) and safety defense (discriminator) within the internal representation space of LLMs. This is the first work to unify attack and defense into a single framework for mutual enhancement, achieving an average attack success rate of 88.85% and an average defense success rate of 84.17%.

## Background & Motivation

**Background**: LLMs acquire safety defense capabilities after safety alignment (RLHF/SFT/DPO), but various jailbreak attacks continue to expose the vulnerability of their safety mechanisms. Existing research treats jailbreak attacks and safety defense as two **isolated** directions, optimizing them independently.

**Limitations of Prior Work**: White-box attack methods (e.g., SCAV, JRE) extract perturbation vectors through mathematical iterative optimization or differences between positive/negative samples, which is complex and difficult to generalize. Defense methods (e.g., input filtering, knowledge editing) either rely on external detection models with limited accuracy or modify parameters, leading to decreased model fluency. The lack of synergy between the two prevents the formation of a closed loop where "attacks drive defense."

**Key Insights**: 
1. Malicious and benign queries exhibit **linear separability** in the intermediate layer embeddings of LLMs, which can be distinguished by a simple classifier.
2. Successful jailbreak attacks essentially migrate the representations of malicious queries from the "unsafe region" to the "safe region," representing a **boundary-crossing problem**.

**Core Idea**: Redefine the extraction of Concept Activation Vectors (CAV) from manual/mathematical optimization to a **generative process**. A GAN generator is employed to automatically generate jailbreak perturbations, while a discriminator is used to recognize disguised malicious queries, allowing both to improve synchronously through adversarial training.

## Method

### Overall Architecture

CAVGAN operates based on the internal embedding space of LLM decoding layers, consisting of three phases:

1. **Training Phase**: The generator $G$ and discriminator $D$ undergo adversarial training on the internal embeddings of the LLM.
2. **Attack Phase**: The trained $G$ generates perturbations that are injected into the intermediate layers to achieve jailbreak.
3. **Defense Phase**: The trained $D$ detects unsafe inputs and guides the model to regenerate.

### Problem Formalization

Given the $L$-layer internal embeddings $\{h_0, h_1, \dots, h_L\}$ of LLM $M$. The safety discriminator $G$ outputs probability $p \in (0,1)$, representing the degree of malicious intent of the input. The goal of the jailbreak attack is to find a perturbation $\delta$ to minimize $G(h+\delta)$, i.e., $\min(G(h+\delta))$, subject to $\|\delta\| \le \epsilon$.

### Generator (Attack)

- **Input**: Malicious query embedding $h$ at the target layer.
- **Output**: Perturbation vector $G(h)$, which makes the discriminator unable to identify the malicious intent after injection.
- **Loss Function**: $L_G = E_{h \sim D_m}[\log D(h + G(h))]$.
- Implicitly constrains the perturbation norm through parameter weight normalization, avoiding deviation from the semantic space.

### Discriminator (Defense)

The discriminator must simultaneously distinguish among three types of inputs: benign original embeddings, malicious original embeddings, and malicious embeddings disguised by perturbations. The loss consists of two parts:

- **Real Classification**: $L_{real} = E_{h \sim D_b}[\log D(h)] + E_{h \sim D_m}[\log(1-D(h))]$
- **Disguise Recognition**: $L_{fake} = E_{h \sim D_m}[\log(1-D(h+G(h)))]$
- **Total Loss**: $L_D = L_{real} + L_{fake}$

### Defense Inference Process

During inference, the embedding $h_Q$ of the input $Q$ is fed into the discriminator. If $G(h_Q) \ge p_0$ (threshold), then it is concatenated with the safety prompt prefix $P_{safe}$ for regeneration; otherwise, it outputs normally.

### Layer Selection Strategy

- Attacks perform best when applied to positions near the intermediate layers.
- In the later layers, while the ASR-kw remains high, the text quality drops sharply (yielding a large number of repetitive and meaningless characters).
- In the earlier layers, the text quality does not drop, but the ASR is very low.
- **Conclusion**: The safety mechanisms of LLMs are **formed progressively layer by layer**.

## Key Experimental Results

### Table 1: Jailbreak Attack Results on Three Models (AdvBench + StrongREJECT)

| Model | Method | ASR-kw | ASR-gpt | ASR-Answer | ASR-Useful | ASR-Rep |
|------|------|--------|---------|------------|------------|---------|
| Qwen2.5-7B | SCAV | 99.85 | 87.54 | 78.65 | 98.07 | 100.00 |
| Qwen2.5-7B | JRE | 83.54 | 70.00 | 60.96 | 61.15 | 55.00 |
| Qwen2.5-7B | **CAVGAN** | 98.98 | 83.88 | 70.41 | 86.73 | 99.80 |
| Llama3.1-8B | SCAV | 100.00 | 90.65 | 87.11 | 95.19 | 99.23 |
| Llama3.1-8B | **CAVGAN** | 98.78 | 88.38 | 78.16 | 88.38 | 99.38 |
| Mistral-8B | SCAV | 99.24 | 78.26 | 82.30 | 80.76 | 85.19 |
| Mistral-8B | **CAVGAN** | **95.51** | **94.29** | **88.78** | **95.10** | **99.18** |

CAVGAN comprehensively outperforms SCAV on Mistral-8B. On the other two models, it has a slight gap compared to SCAV but significantly outperforms JRE.

### Table 2: Defense Experimental Results (SafeEdit Jailbreak Dataset + Alpaca Benign Queries)

| Model | Method | DSR | BAR | ASR Reduce | BAR Reduce |
|------|------|-----|-----|------------|------------|
| Qwen2.5-7B | Original | 25.12 | 98.00 | - | - |
| Qwen2.5-7B | SmoothLLM | 54.22 | 75.77 | 38.86 | 22.68 |
| Qwen2.5-7B | RA-LLM | 78.60 | 85.80 | 71.42 | 12.45 |
| Qwen2.5-7B | **CAVGAN** | **91.12** | **91.40** | **88.14** | **10.06** |
| Llama3.1-8B | Original | 11.34 | 99.60 | - | - |
| Llama3.1-8B | SmoothLLM | 48.97 | 81.03 | 42.44 | 18.64 |
| Llama3.1-8B | RA-LLM | 73.78 | 92.80 | 70.42 | 6.83 |
| Llama3.1-8B | **CAVGAN** | **77.22** | **93.60** | **74.31** | **6.02** |

CAVGAN achieves a DSR of 91.12% (SOTA +12%) on Qwen2.5-7B, while maintaining a BAR of 91.40%, achieving the best balance between safety and utility.

### Key Findings

- **Attack**: The average jailbreak success rate across three LLMs is 97% (ASR-kw), outperforming SCAV in all aspects on Mistral-8B.
- **Defense**: The average DSR is 84.17%, which is 12 percentage points higher than RA-LLM on Qwen, with minimal loss in BAR.
- **Scalability**: The attack still maintains a 94%+ ASR-kw on larger models like Qwen2.5-14B/32B.
- **Training Samples**: Only 80-100 samples are required to achieve optimal performance; more samples actually lead to performance fluctuations due to GAN instability.

## Highlights & Insights

- **Unified Paradigm for Attack and Defense**: For the first time, jailbreak attacks and safety defense are integrated into a single GAN framework. This validates the bidirectional control paradigm of "attacks driving defense," which is conceptually superior to independent optimization.
- **CAV Generation Replacing Mathematical Optimization**: The traditional extraction of positive/negative sample differences or iterative optimization is converted into a model generation process, reducing complexity and making it easily extendable to other domains.
- **Interpretability of Safety Mechanisms**: Experimental results on layer selection reveal that safety mechanisms in LLMs are formed progressively layer by layer, with the intermediate layers being the optimal entry points for attacks.
- **Lightweight and Efficient**: Both the generator and discriminator are 4-layer MLPs, requiring only 100 samples and 10 epochs for training, which is extremely cost-effective.

## Limitations & Future Work

- **Overly Simple GAN Architecture**: Both the generator and discriminator are simple MLPs with limited capacity to model complex semantics; the authors acknowledge that more complex architectures might yield better results.
- **Hyperparameter Reliance on Validation Set**: The threshold $p_0$ and target layer selection rely on validation set tuning, lacking an automated selection mechanism.
- **Defense-Induced Latency**: The regeneration-based defense mechanism increases response time in highly real-time scenarios.
- **Unclear Generalization Boundaries**: Validated only on 3 model families ranging from 7B to 32B scale; closed-source models or larger-scale models have not been tested.
- **Sensitivity to Training Data**: The 80-100 sample range is the optimal interval; due to GAN training instability, increasing the sample size actually leads to degraded performance.

## Related Work & Insights

- **White-Box Jailbreak Attacks**: SCAV (mathematical iterative optimization for perturbation), JRE (embedding differences of positive and negative samples), GCG (gradient search for adversarial suffixes).
- **Black-Box Jailbreak Attacks**: Manual prompt templates, genetic algorithm search, iterative prompt optimization.
- **Tuning-Free Defenses**: SmoothLLM (random input perturbation to detect attacks), RA-LLM (random perturbation + consistency checking).
- **Knowledge Editing Defenses**: Methods like SafeEdit edit toxic regions of models, but modifying parameters introduces unknown risks.
- **Representation Engineering**: Representation Engineering uses CAV to control model behavior, which this paper integrates with GANs.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First to use GAN to unify LLM attack and defense, transforming CAV extraction into a generative process.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ 3 models x 2 datasets for attacks + 2 models for defense + layer selection/sample size ablation.
- **Writing Quality**: ⭐⭐⭐⭐ Clear framework diagram, formal mathematical specification, and coherent attack/defense logic flow.
- **Practicality**: ⭐⭐⭐ The MLP architecture is lightweight and easy to deploy, but the regeneration mechanism for defense introduces latency.
- **Overall Recommendation**: ⭐⭐⭐⭐ Notable conceptual contribution, solid experimentation, but the simple GAN structure somewhat limits its performance ceiling.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] TIP of the Iceberg: Task-in-Prompt Adversarial Attacks on LLMs](tip_iceberg_adversarial_attacks.md)
- [\[ACL 2025\] Bias in the Mirror: Are LLMs' Opinions Robust to Their Own Adversarial Attacks](bias_in_the_mirror_are_llms_opinions_robust_to_their_own_adversarial_attacks.md)
- [\[NeurIPS 2025\] On the Robustness of Verbal Confidence of LLMs in Adversarial Attacks](../../NeurIPS2025/llm_safety/on_the_robustness_of_verbal_confidence_of_llms_in_adversarial_attacks.md)
- [\[ICLR 2026\] Adversarial Déjà Vu: Jailbreak Dictionary Learning for Stronger Generalization to Unseen Attacks](../../ICLR2026/llm_safety/adversarial_déjà_vu_jailbreak_dictionary_learning_for_stronger_generalization_to.md)
- [\[ACL 2025\] PIG: Privacy Jailbreak Attack on LLMs via Gradient-based Iterative Prompts](pig_privacy_jailbreak.md)

</div>

<!-- RELATED:END -->
