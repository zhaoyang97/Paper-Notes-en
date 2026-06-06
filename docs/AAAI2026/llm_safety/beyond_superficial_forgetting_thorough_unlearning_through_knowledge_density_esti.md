---
title: >-
  [Paper Note] Beyond Superficial Forgetting: Thorough Unlearning through Knowledge Density Estimation and Block Re-insertion
description: >-
  [AAAI 2026][LLM Safety][machine unlearning] This paper proposes the KUnBR framework, which employs gradient-guided knowledge density estimation to localize layers enriched with harmful knowledge…
tags:
  - "AAAI 2026"
  - "LLM Safety"
  - "machine unlearning"
  - "Knowledge Density"
  - "Block Re-insertion"
  - "RTT Attack"
date: 2026-05-08
content_hash: a4340c341d603d57
---

# Beyond Superficial Forgetting: Thorough Unlearning through Knowledge Density Estimation and Block Re-insertion

**Conference**: AAAI 2026
**arXiv**: [2511.11667](https://arxiv.org/abs/2511.11667)  
**Code**: [github.com/llmgfffffff/Beyond-Superficial-Forgetting-KUnBR](https://github.com/llmgfffffff/Beyond-Superficial-Forgetting-KUnBR)  
**Area**: AI Safety
**Keywords**: machine unlearning, Knowledge Density, Block Re-insertion, LLM safety, RTT Attack

## TL;DR

This paper proposes the KUnBR framework, which employs gradient-guided knowledge density estimation to localize layers enriched with harmful knowledge, and adopts a block re-insertion strategy to bypass the gradient-masking effect of cover layers, achieving deep unlearning of harmful knowledge in LLMs rather than mere surface-level suppression.

## Background & Motivation

- **Core demand for machine unlearning**: LLMs may absorb privacy-sensitive, harmful, or copyright-protected content during pretraining. Selectively removing such knowledge without retraining from scratch is necessary to comply with regulations such as GDPR's "right to be forgotten."
- **Superficial forgetting in existing methods**: Methods such as Gradient Ascent (GA), Gradient Difference (GD), and RMU can suppress harmful outputs at the surface level, but in practice only modify parameters in a small number of "cover layers," leaving harmful knowledge intact in the deeper parameters of the model.
- **Vulnerability exposed by RTT attacks**: The Retraining on T (RTT) attack demonstrates that fine-tuning on a small subset of the forget set can recover a large proportion of "forgotten" knowledge, revealing that existing methods do not truly eliminate target knowledge from model parameters.
- **Gradient masking by cover layers**: During unlearning training, gradients concentrate primarily in a few output-side layers, forming a cover layer shielding effect that prevents effective updates to deeper knowledge-enriched blocks.
- **Absence of precise localization**: Prior work lacks systematic methods for quantifying the density of harmful knowledge per layer, making it impossible to precisely identify which layers most require deep unlearning.
- **Challenge of preserving general capabilities**: Methods such as RIA and NPO tend to severely impair general capabilities—including reasoning and factual question answering—while pursuing lower forget-set accuracy, lacking a proper balance between forgetting and retention.

## Method

### Overall Architecture

KUnBR operates in three stages: (1) full-parameter pre-unlearning (warm-up), applying standard gradient difference to perform preliminary unlearning training on the entire model; (2) knowledge density estimation and block selection, to localize blocks enriched with harmful knowledge; (3) block re-insertion and secondary unlearning, grafting the selected blocks back into the original model for deep unlearning.

### Key Design 1: Knowledge Density Estimation

- **Function**: Computes a knowledge density score $K_l$ for each layer of the model, quantifying the amount of to-be-forgotten knowledge it contains.
- **Mechanism**: Forward and backward passes are performed on the forget set, and the expected L1 norm of the parameter gradients at each layer is taken as the knowledge density indicator. Larger gradient magnitudes indicate greater sensitivity of that layer to forget-set information and thus higher concentrations of knowledge to be eliminated. After normalization: $K_l^{norm} = K_l / \sum_{i=1}^H K_i$.
- **Design Motivation**: Grounded in the insight that MLP layers serve as neural memory units in LLMs, large gradients imply high association between parameters and the target knowledge. This step only computes without updating parameters, providing the basis for subsequent precise localization.

### Key Design 2: Block Selection Strategy

- **Function**: Partitions the model's $H$ layers into $M$ blocks, each containing $N = \lfloor H/M \rfloor$ layers. Block-level density $K_{block,m}$ is obtained by accumulating per-layer densities, and the Top-K highest-density blocks are selected.
- **Mechanism**: Two rules are applied—(a) Top-K selection: select the K blocks with the highest knowledge density; (b) exclude leading-output layers: ignore blocks containing the last two layers, as high gradients in terminal layers are artifacts of output generation rather than indicators of knowledge storage.
- **Design Motivation**: Layer-wise granularity is too fine and inefficient; block-level grouping balances localization precision with practical efficiency, while excluding interfering layers prevents misidentification.

### Key Design 3: Block Re-insertion Strategy

- **Function**: Extracts the selected high-density blocks from the pre-unlearned $\text{LLM}_{unlearning}$, grafts them into the corresponding positions of the original, untouched $\text{LLM}_{original}$, freezes all other layers of the original model, and applies gradient difference unlearning training exclusively to the inserted blocks.
- **Mechanism**: Since the original model has never undergone unlearning training, no cover layers exist within it. The inserted blocks are thus directly exposed to unlearning gradients without interference from the masking effects of previously modified layers, enabling deeper knowledge elimination. After training, these blocks are reintegrated into $\text{LLM}_{unlearning}$, where residual knowledge is far lower than that of standard methods.
- **Design Motivation**: Continuing training directly on the already-unlearned model is subject to gradient blockage from cover layers. The re-insertion strategy circumvents this blockage, allowing unlearning gradients to reach target blocks unimpeded.

### Loss & Training

Both the pre-unlearning and re-insertion stages use the **Gradient Difference** loss: gradient ascent is applied on the forget set (increasing loss) to eliminate knowledge, while gradient descent is applied on the retain set (decreasing loss) to preserve general capabilities. The pre-unlearning stage involves full-parameter training; the re-insertion stage trains only the selected blocks while freezing all other layers.

## Key Experimental Results

### Table 1: Unlearning Performance under RTT Attack (LLaMA3-8B-Instruct, ↓ lower is better)

| Method | Random Birthdays Forget. | RTT. | Rec. | WMDP Forget. | RTT. | Rec. | Years Forget. | RTT. | Rec. |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| GA | 23.5 | 87.2 | 63.7 | 29.2 | 66.8 | 37.6 | 25.9 | 50.6 | 24.7 |
| GD | 64.9 | 80.2 | 15.3 | 30.5 | 62.4 | 31.9 | 25.9 | 68.3 | 42.4 |
| RMU | 36.3 | 88.5 | 52.2 | 29.9 | 64.9 | 35.0 | 24.2 | 68.3 | 44.1 |
| NPO | 71.3 | 78.3 | 7.0 | 35.6 | 58.4 | 22.8 | 26.5 | 67.7 | 41.2 |
| **KUnBR** | **36.9** | **43.9** | **7.0** | **29.2** | **38.8** | **9.6** | **25.9** | **36.0** | **10.1** |

KUnBR achieves the lowest or jointly lowest post-RTT recovery rate (Rec.) across all datasets, demonstrating more thorough removal of harmful knowledge.

### Table 2: General Capability Retention (LLaMA3-8B, RKWU metrics, ↑ higher is better)

| Method | Rea. | Fac. | Tru. | Flu. |
|------|:---:|:---:|:---:|:---:|
| GA | 40.2 | 56.3 | 36.8 | 706.2 |
| RIA | 39.5 | 56.1 | 36.8 | 705.9 |
| NPO | 39.8 | 54.3 | 36.8 | 703.7 |
| **KUnBR** | **41.2** | **56.1** | **36.6** | **706.7** |

KUnBR achieves the best performance on reasoning and fluency, and matches the best baseline on factuality and truthfulness, demonstrating that block-level localized unlearning effectively preserves general capabilities.

### Ablation Study (Table 3, Years dataset)

| Variant | Forget.↓ | RTT.↓ |
|------|:---:|:---:|
| KUnBR (full) | 25.9 | 36.0 |
| w/o re-insertion (degrades to GD) | 25.9 | 68.3 |
| w/o pre-unlearning | 25.9 | 36.7 |

Removing the re-insertion strategy causes the RTT accuracy to surge from 36.0% to 68.3%, confirming that the re-insertion strategy is critical for robustness against RTT attacks.

## Highlights & Insights

- **Deep problem insight**: This work is the first to systematically analyze the gradient-masking effect of cover layers in unlearning training, exposing the root cause of the "superficial forgetting" phenomenon in existing methods.
- **Simple yet effective knowledge density estimation**: Gradient L1 norms are used to quantify per-layer knowledge density without requiring additional probes or complex analytical tools.
- **Elegant re-insertion design**: The strategy of grafting target blocks back into the original model to bypass cover layers is novel in conception and straightforward in implementation.
- **Substantial improvement in RTT robustness**: Recovery rates are markedly lower than all baseline methods across multiple datasets.
- **Minimal degradation of general capabilities**: Block-level localized unlearning combined with freezing of remaining layers effectively avoids global capability deterioration.

## Limitations & Future Work

- Validation is limited to models at the 7B–8B scale; scalability to larger models (e.g., 70B) has not been tested.
- Knowledge density estimation requires full-model backpropagation over the entire forget set, incurring high computational cost for large-scale forget sets.
- The number of blocks $M$ and the Top-K selection require empirical tuning; while the authors claim cross-architecture stability, optimal configurations may vary across tasks.
- Evaluation is restricted to multiple-choice formats; unlearning effectiveness in open-ended generation scenarios is not assessed.
- The rule of excluding the last two layers is based on empirical observation and lacks rigorous theoretical justification.

## Related Work & Insights

- **GA / GD / NPO**: Gradient-based unlearning methods that suppress outputs via gradient ascent or preference optimization, but leave knowledge intact in model parameters.
- **RMU**: Achieves unlearning by perturbing intermediate layer representations, but such perturbations are readily recovered by RTT attacks.
- **RIA**: Guides the model to learn incorrect answers; unlearning effectiveness is limited and general capabilities are impaired.
- **RTT Attack**: A parameter-level attack that reveals the fragility of unlearning methods and serves as the primary adversarial evaluation in this work.
- **Layer-wise knowledge analysis**: Geva et al. established that MLP layers function as key-value memory units in LLMs; Hong et al. found that unlearning training primarily modifies a small number of layers—both findings provide the theoretical foundation for the proposed knowledge density estimation.

## Rating

- Novelty: ⭐⭐⭐⭐ — The cover layer analysis and re-insertion strategy represent novel perspectives and solutions
- Experimental Thoroughness: ⭐⭐⭐⭐ — Four datasets, two backbone models, complete ablations, and block selection analysis
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, well-articulated motivation, and intuitive illustrations
- Value: ⭐⭐⭐⭐ — Addresses a fundamental problem in machine unlearning with practical implications for safe LLM deployment

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Learning from Oblivion: Predicting Knowledge-Overflowed Weights via Retrodiction of Forgetting](../../CVPR2026/llm_safety/learning_from_oblivion_predicting_knowledge_overflowed_weights_via_retrodiction_.md)
- [\[ICCV 2025\] Forgetting Through Transforming: Enabling Federated Unlearning via Class-Aware Representation Transformation](../../ICCV2025/llm_safety/forgetting_through_transforming_enabling_federated_unlearning_via_class-aware_re.md)
- [\[ICLR 2026\] Unlearning Evaluation through Subset Statistical Independence](../../ICLR2026/llm_safety/unlearning_evaluation_through_subset_statistical_independence.md)
- [\[AAAI 2026\] Learning from the Undesirable: Robust Adaptation of Language Models without Forgetting](learning_from_the_undesirable_robust_adaptation_of_language_models_without_forge.md)
- [\[AAAI 2026\] Privacy-protected Retrieval-Augmented Generation for Knowledge Graph Question Answering](privacy-protected_retrieval-augmented_generation_for_knowledge_graph_question_an.md)

</div>

<!-- RELATED:END -->
