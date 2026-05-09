---
title: >-
  [Paper Note] Resource-Adaptive Federated Text Generation with Differential Privacy
description: >-
  [ICLR2026][LLM Safety][federated learning] This paper proposes a resource-adaptive federated text generation framework that employs a two-stage design — DP fine-tuning on strong clients and DP voting on weak clients — to generate high-quality synthetic text data under computational heterogeneity and differential privacy constraints.
tags:
  - ICLR2026
  - LLM Safety
  - federated learning
  - differential privacy
  - Synthetic Text Generation
  - Computational Heterogeneity
  - Control Code
date: 2026-05-08
content_hash: 0057bdc833021940
---

# Resource-Adaptive Federated Text Generation with Differential Privacy

**Conference**: ICLR2026
**arXiv**: [2603.07027](https://arxiv.org/abs/2603.07027)
**Code**: None
**Area**: AI Security
**Keywords**: federated learning, differential privacy, Synthetic Text Generation, Computational Heterogeneity, Control Code

## TL;DR

This paper proposes a resource-adaptive federated text generation framework that employs a two-stage design — DP fine-tuning on strong clients and DP voting on weak clients — to generate high-quality synthetic text data under computational heterogeneity and differential privacy constraints.

## Background & Motivation

In cross-silo federated learning (FL) scenarios, sensitive text data must remain within local organizations (e.g., hospitals, companies) due to privacy regulations. Conventional approaches require a new round of FL training for each downstream task, incurring high communication and privacy costs. A more practical alternative is to generate differentially private (DP) synthetic datasets that approximate the global data distribution, which can be reused across multiple downstream tasks.

However, directly generating text from a pretrained LLM often yields low quality, as the pretraining distribution may significantly diverge from the target domain (domain shift). Federated fine-tuning is therefore necessary to adapt the model. Fine-tuning LLMs in this setting faces a critical obstacle: **computational heterogeneity** — only a small number of resource-rich clients can afford LLM fine-tuning, while weaker clients are excluded. This amplifies the effect of data skew, and the noise introduced by DP-SGD further degrades model quality as the number of participating clients decreases.

## Core Problem

How can all clients — regardless of computational capacity — contribute effectively in cross-silo FL to generate high-quality synthetic text that satisfies differential privacy guarantees and faithfully reflects the global data distribution?

Specific challenges include:

1. **Computational heterogeneity**: Weak clients cannot perform backpropagation or fine-tuning, yet their data is indispensable for capturing the global distribution.
2. **Data heterogeneity**: Significant distributional differences across clients cause fine-tuned models that rely solely on strong clients to be biased.
3. **DP noise amplification**: Fewer clients participating in fine-tuning increases the adverse impact of DP-SGD noise on convergence and generation quality.

## Method

### Overall Architecture: Four-Stage Pipeline

**Stage 1: DP Federated Fine-Tuning (Strong Clients $\mathcal{C}_s$)**

- Only computationally capable strong clients participate in standard federated learning.
- Local training uses DP-SGD to protect sample-level privacy.
- The objective is a conditional language modeling loss: $f_i(\theta) = -\frac{1}{|D_i|}\sum_j \sum_{x \in D_i^j} \log p_\theta(x|c^j)$
- A fine-tuned model $\theta^*$ is obtained after $R$ communication rounds.

**Stage 2: DP Profiling (All Clients)**

- Each client computes a vector $P_i = [|D_i^1|, \ldots, |D_i^{|C|}|]$ representing the sample counts under each control code in its local data.
- Noise is added to $P_i$ via the Analytical Gaussian Mechanism before transmission to the server.
- The server aggregates these to obtain the global target distribution $\tilde{P} = \sum_i \tilde{P}_i$.

**Stage 3: Profile-Guided Synthetic Text Generation**

- The server determines the number of samples to generate per control code as $s_j = \text{Round}(s \cdot \tilde{P}[j])$ based on the global distribution $\tilde{P}$.
- The fine-tuned model $p_{\theta^*}(\cdot|c^j)$ is used to generate the corresponding number of synthetic texts.
- This produces an initial synthetic dataset $\tilde{D}$.

**Stage 4: DP Voting Refinement (Weak Clients $\mathcal{C}_r$)**

- The server broadcasts the synthetic data to weak clients.
- Each weak client votes on synthetic samples: for each real sample locally, the $K$ most similar synthetic samples within the same control code are selected using sentence transformer embeddings.
- Voting results are privatized via the Analytical Gaussian Mechanism.
- The server aggregates votes and resamples the synthetic data according to normalized voting probabilities.

### Dual Role of Control Codes

Control codes (e.g., labels, topics, metadata) are a central design element of this framework:

1. **Representing client distributions**: The sample proportions across control codes naturally characterize the local data distribution.
2. **Constraining the voting scope**: Voting is restricted to samples within the same control code, ensuring semantic consistency.

### Privacy Guarantees

Three independent DP mechanisms are applied:

- Fine-tuning stage: DP-SGD ($\varepsilon_{\text{train}}, \delta_{\text{train}}$)
- Profiling stage: Analytical Gaussian Mechanism ($\varepsilon_{\text{prof}}, \delta_{\text{prof}}$)
- Voting stage: Analytical Gaussian Mechanism ($\varepsilon_{\text{vote}}, \delta_{\text{vote}}$)

Weak clients require only one round of communication and no backpropagation, resulting in minimal computational overhead.

## Key Experimental Results

### Datasets and Settings

| Dataset | # Clients | Samples per Client | Generation Model | Control Code |
|--------|---------|-------------|---------|-------------|
| Yelp Reviews | 100 | 15,000 | GPT-2 | Business category + star rating |
| PubMed Abstracts | 20 | 2,250 | GPT-2-large | 5 MeSH terms |

### IID Setting — Key Results (Yelp, $\varepsilon=8$)

- **1% strong clients + refinement** achieves a rating classification accuracy of 0.6149, comparable to **10% strong clients without refinement** (0.6280).
- **20% strong clients + refinement** achieves an F1 of 0.6285, surpassing **40% strong clients without refinement** (0.6168).
- Refinement improves rating classification F1 by approximately 0.20 under the 1% strong client setting.

### IID Setting — Key Results (PubMed, $\varepsilon=8$)

- **5% strong clients + refinement** (Acc.(D)=0.8028) substantially outperforms **20% strong clients without DP** (0.7968).
- Refinement consistently elevates DP results to levels approaching or exceeding non-DP baselines.

### Non-IID Setting (10% Strong Clients)

- In some Yelp non-IID scenarios, $\varepsilon=8$ + refinement **outperforms** $\varepsilon=\infty$ without refinement.
- On the PubMed NER task, DP clipping and noise act as implicit regularization under severe data skew ($\varepsilon=\infty$ performs worse than $\varepsilon=8$).

## Highlights & Insights

1. **Flexible participation mechanism**: Strong clients handle fine-tuning while weak clients contribute via voting; all participants can contribute, with weak clients requiring only one round of communication and forward inference.
2. **Elegant control code design**: A single design simultaneously addresses distribution representation and voting semantic constraints.
3. **Significant refinement gains**: Under low-resource conditions (1–5% strong clients), a single round of refinement substantially compensates for DP-induced performance degradation.
4. **Counter-intuitive finding**: In severely non-IID scenarios, DP noise may act as a regularizer, enabling DP + refinement to surpass non-DP baselines.

## Limitations & Future Work

1. **Control codes must be predefined**: The framework assumes control codes are public knowledge and non-private, which may not hold in certain sensitive settings.
2. **Validation limited to GPT-2-scale models**: Although LLaMA results appear in the appendix, the approach has not been thoroughly validated on larger LLMs.
3. **Voting depends on sentence embedding quality**: If the sentence transformer performs poorly in the target domain, refinement effectiveness may be limited.
4. **Strong cross-silo assumptions**: Each client is assumed to hold thousands of samples, making the framework unsuitable for cross-device FL.
5. **The effect of the number of control codes on privacy budget allocation is not discussed.**

## Related Work & Insights

| Method | Setting | Fine-tuning Approach | Weak Client Participation | DP Level |
|------|---------|---------|------------|--------|
| PrE-Text (Hou et al., 2024) | Cross-device FL | No fine-tuning; prompting only | All clients equally | Client-level |
| FLoRA (Wang et al., 2024) | Federated fine-tuning | Parameter-efficient LoRA | Still requires local backpropagation | Optional |
| **Ours** | Cross-silo FL | Full fine-tuning with DP-SGD | Voting refinement; no gradient computation | Sample-level |

This approach is orthogonal and complementary to parameter-efficient methods such as LoRA: LoRA can serve as the fine-tuning strategy in Stage 1, but the voting refinement in Stage 4 remains necessary. Experiments (Table A.13) confirm that LoRA + refinement still yields gains.

The voting refinement paradigm is generalizable to other federated settings requiring weak client participation beyond text generation. The regularization effect of DP noise in highly heterogeneous scenarios suggests that minimizing noise is not always optimal. The control code methodology can be combined with prompt engineering to further enhance generation quality. The framework has direct applicability to cross-institutional data sharing in domains such as healthcare and finance.

## Rating

- **Novelty**: 7/10 — The strong/weak client division combined with voting refinement is a novel combination, though individual components are technically mature.
- **Experimental Thoroughness**: 8/10 — Coverage of IID/non-IID settings, multiple datasets, and multiple metrics with thorough ablations; model scale is limited.
- **Writing Quality**: 8/10 — Clear structure, well-defined problem formulation, and consistent notation.
- **Value**: 7/10 — Addresses a practical bottleneck in federated synthetic data generation under computational heterogeneity.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] InvisibleInk: High-Utility and Low-Cost Text Generation with Differential Privacy](../../NeurIPS2025/llm_safety/invisibleink_high-utility_and_low-cost_text_generation_with_differential_privacy.md)
- [\[NeurIPS 2025\] FedSVD: Adaptive Orthogonalization for Private Federated Learning with LoRA](../../NeurIPS2025/llm_safety/fedsvd_adaptive_orthogonalization_for_private_federated_learning_with_lora.md)
- [\[ACL 2026\] AGSC: Adaptive Granularity and Semantic Clustering for Uncertainty Quantification in Long-text Generation](../../ACL2026/llm_safety/agsc_adaptive_granularity_and_semantic_clustering_for_uncertainty_quantification.md)
- [\[ICLR 2026\] SABRE-FL: Selective and Accurate Backdoor Rejection for Federated Prompt Learning](sabre-fl_selective_and_accurate_backdoor_rejection_for_federated_prompt_learning.md)
- [\[ACL 2026\] Adaptive Text Anonymization: Learning Privacy-Utility Trade-offs via Prompt Optimization](../../ACL2026/llm_safety/adaptive_text_anonymization_learning_privacy-utility_trade-offs_via_prompt_optim.md)

</div>

<!-- RELATED:END -->
