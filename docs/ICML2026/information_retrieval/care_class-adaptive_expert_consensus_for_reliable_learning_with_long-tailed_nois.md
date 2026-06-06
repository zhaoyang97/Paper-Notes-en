---
title: >-
  [Paper Note] CARE: Class-Adaptive Expert Consensus for Reliable Learning with Long-Tailed Noisy Labels
description: >-
  [ICML 2026][Information Retrieval & RAG][Noisy Label Learning] The CARE framework is proposed, which leverages three-way complementary experts—text embeddings, image features…
tags:
  - "ICML 2026"
  - "Information Retrieval & RAG"
  - "Noisy Label Learning"
  - "Long-Tailed Distribution"
  - "Vision-Language Models"
  - "Expert Consensus"
  - "Label Correction"
date: 2026-05-08
content_hash: f412dd4a24d70a59
---

# CARE: Class-Adaptive Expert Consensus for Reliable Learning with Long-Tailed Noisy Labels

**Conference**: ICML 2026  
**arXiv**: [2605.23254](https://arxiv.org/abs/2605.23254)  
**Code**: https://github.com/qwq123-study/CARE (Available)  
**Area**: Self-Supervised/Representation Learning  
**Keywords**: Noisy Label Learning, Long-Tailed Distribution, Vision-Language Models, Expert Consensus, Label Correction  

## TL;DR

The CARE framework is proposed, which leverages three-way complementary experts—text embeddings, image features, and original labels from Vision-Language Models (VLMs)—to achieve reliable label correction in long-tailed noisy label scenarios via a class-adaptive Top-$K$ consensus mechanism. It consistently outperforms state-of-the-art (SOTA) methods by up to 3.0% on synthetic and real-world benchmarks.

## Background & Motivation

**Background**: Real-world data commonly encounters two simultaneous challenges: annotation noise and long-tailed distributions. When addressed in isolation, long-tailed learning methods assume clean labels and may amplify annotation errors in tail classes. Conversely, noisy label learning methods assume balanced distributions and tend to discard noisy samples, causing tail classes to lose critical information.

**Limitations of Prior Work**: Recent Long-Tailed Noisy Label (LTNL) methods attempt joint processing but mostly adopt **class-agnostic** label correction strategies. For example, RLA performs label correction followed by logit adjustment but treats head and tail classes identically during the correction process. Experiments show that even if the overall noise rate decreases, performance can drop if tail labels are not sufficiently corrected, due to inaccurate regularization introduced during long-tail calibration (e.g., total accuracy of RLD 2 drops from 79.5% to 76.8%).

**Key Challenge**: Head classes have abundant samples and high label reliability, permitting loose correction; tail classes have scarce samples and higher noise rates, requiring stricter correction. A uniform threshold cannot satisfy both requirements simultaneously.

**Goal**: Design a parameter-free framework that achieves **class-aware** noise filtering and distribution calibration during the label correction stage.

**Key Insight**: Utilize the text semantics and visual features naturally provided by pre-trained VLMs (e.g., CLIP) as two independent "experts," which, together with the original labels, constitute three-way complementary signals.

**Core Idea**: Differentiation in consistency checks is enforced among experts via class-adaptive Top-$K$ consensus voting—using a larger $K$ for head classes and a smaller $K$ for tail classes—to achieve reliable label correction.

## Method

### Overall Architecture

The workflow of CARE is as follows: given an image $x$, class confidence vectors are obtained from three parameter-free experts: **(1)** The Text Expert (TE) uses the CLIP text encoder to compute cosine similarity between class text and image features; **(2)** The Image Expert (IE) uses the CLIP image encoder fine-tuned via AdaptFormer to output class probabilities; **(3)** The Base Expert (BE) directly uses the one-hot vector of the original (potentially noisy) label. The three-way confidences are aggregated through the class-adaptive consensus mechanism to output a corrected class frequency distribution. Finally, long-tail calibration training is performed using the corrected labels combined with a logit adjustment loss.

### Key Designs

1.  **Class-Adaptive Top-$K$ Consensus Mechanism**:
    The core idea is to dynamically adjust the number of candidate classes retained for each expert based on class frequency. For TE and IE, the Top-$K_c$ predictions from their respective confidences are selected, where $K_c \propto n_c^e$ ($n_c^e$ is the sample count for class $c$ at epoch $e$). A larger $K$ is used for head classes to allow more candidates, while a smaller $K$ is used for tail classes to enforce stricter consistency. Each expert contributes confidence only to categories within its Top-$K$: $g_m(c) = \mathbb{I}[c \in \mathcal{T}_K^m] \cdot p_c^m$. If the observed label $\tilde{y}$ appears in an expert's Top-$K$, the expert is deemed reliable, and its confidence is weighted and enhanced; otherwise, only the Top-$K$ predictions are kept to avoid reinforcing noisy labels. Theoretically (Theorem 1), since the probability of the true label appearing jointly in the Top-$K$ of multiple experts is much higher than that of noisy labels, the consensus mechanism inherently possesses denoising effects. Compared to a global uniform $K$, the class-adaptive $K_t$ improves consensus accuracy for tail classes (Proposition 4).

2.  **Dynamic Class Frequency Accumulation and Label Correction**:
    During training, consensus frequencies for each class are accumulated per sample: $F_c^{(e)} = F_c^{(e-1)} + \sum_{m} \alpha_m(x) \cdot g_m(c)$. The corrected label is taken as the class with the highest frequency: $y^{r,(e)} = \arg\max_c F_c^{(e)}$. As training progresses, correct labels increasingly dominate the frequency matrix, achieving progressive self-correction (Corollary 2). The corrected class distribution $n_c^{r,(e)}$ is used to re-estimate class prior probabilities, driving subsequent logit adjustment calibration.

3.  **Parameter-free Three-way Expert Design**:
    The three experts do not introduce additional trainable parameters. TE uses a frozen CLIP text encoder to provide semantic priors via $p_c^{TE} = \text{softmax}(s \cdot \cos(\mathbf{t}_c, \hat{\mathbf{f}}))$. IE reuses the CLIP image encoder (AdaptFormer) fine-tuned during training to provide task-adapted visual confidence via $p_c^{IE} = \text{softmax}(s \cdot \cos(\mathbf{w}_c, \mathbf{f}))$. BE directly uses the one-hot observed label. The three are complementary: TE is unaffected by label correction and provides stable semantic anchors; IE evolves with training but may be influenced by noise; BE remains informative for head classes. Ablation studies verify that no single expert or pair of experts can effectively reduce the noise rate as well as the combination of all three, which reduces noise from 50% to 27.8%.

## Key Experimental Results

| Setting | Dataset | CLIP+LA | CLIP+RLA | CLIP+LA w. CARE | Gain |
|------|--------|---------|----------|-----------------|------|
| Joint NR=50%, IF=10 | CIFAR-100-LTN | 79.5% | 80.7% | **80.7%** | +1.2% vs LA |
| Joint NR=50%, IF=100 | CIFAR-100-LTN | 75.3% | 66.7% | **76.7%** | +1.4% vs LA, +10.0% vs RLA |
| Sym NR=60%, IF=10 | CIFAR-100-LTN | 76.0% | 77.0% | **79.2%** | +3.2% vs LA |
| Asym NR=40%, IF=10 | CIFAR-100-LTN | 68.2% | 69.3% | **70.5%** | +2.3% vs LA |
| Real noise, IF=100 | Food101N | 83.7% | 77.2% | **84.1%** | +6.9% vs RLA |
| Real noise | WebVision-50 | 85.1% | 85.0% | **85.3%** | +0.3% vs LA |

| Ablation Study | Combination | Noise Rate (%) | Accuracy (%) |
|----------|------|-----------|----------|
| BE only | ✓ / ✗ / ✗ | 50.0 | 78.7 |
| BE + TE | ✓ / ✓ / ✗ | 50.0 | 78.7 |
| BE + IE | ✓ / ✗ / ✓ | 50.0 | 78.7 |
| BE + TE + IE (CARE) | ✓ / ✓ / ✓ | **27.8** | **80.3** |

## Highlights & Insights

- **Key Findings**: Simply reducing the overall noise rate is insufficient; ensuring the accuracy of label correction for tail classes is essential, as incorrect regularization from long-tail calibration can otherwise be detrimental.
- **Novelty**: CARE introduces zero additional parameters during the expert consensus phase. IE reuses the fine-tuned encoder from the main training pipeline, while TE uses a frozen pre-trained encoder.
- **Experimental Thoroughness**: Gains are not limited to CLIP—combinations such as ResNet + GloVe, the TABASCO framework, and MLP-Mixer architectures all achieved consistent improvements, indicating that the gain stems from the consensus mechanism itself rather than just a strong backbone.
- **Value**: Provides theoretical support including a reliability amplification theorem for consensus denoising (Theorem 1) and a proposition for accuracy improvement using adaptive $K$ for tail classes (Proposition 4).

## Limitations & Future Work

- The accuracy gain in tail classes comes at the cost of slight decreases in head class performance (Table 8: head 83.4% → 81.8%), resulting from the stronger regularization of LA on head classes after correction.
- Applicability is limited in scenarios where pre-trained VLMs like CLIP are unavailable to provide text and visual experts.
- The setting where $K_c$ is directly proportional to class frequency is relatively simple; more sophisticated adaptive strategies (e.g., considering semantic similarity between classes) may further enhance performance.

## Related Work & Insights

- **Long-Tailed Learning**: LDAM-DRW, LA (logit adjustment), and MiSLAS handle imbalance at the loss or classifier level but assume clean labels.
- **Noisy Label Learning**: Co-teaching, DivideMix, and UNICON address noise through sample selection or mix-up training but assume balanced distributions.
- **LTNL Joint Methods**: TABASCO considers differences between observed and intrinsic distributions; RCAL, ECBS, and RLA handle both issues jointly but mostly employ class-agnostic strategies.
- **Mechanism**: Multimodal signals from VLMs (text + vision) can serve as natural label auditing tools. The idea of class-adaptive consensus thresholds can be generalized to other scenarios requiring differentiated trust allocation.

## Rating

- Novelty: 7/10 — Class-adaptive consensus voting is an insightful design, though the overall framework (multi-expert voting for label correction) is somewhat intuitive.
- Experimental Thoroughness: 9/10 — Covers synthetic and three real-world datasets, five noise settings, and three backbone architectures with comprehensive ablations.
- Writing Quality: 8/10 — Clear motivation, complete theoretical analysis, and effective intuitive presentation in Figure 1.
- Value: 7/10 — Practical value for LTNL scenarios, though the magnitude of gain over strong baselines (CLIP+LA) is moderate.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Neighbor-aware Instance Refining with Noisy Labels for Cross-Modal Retrieval](../../AAAI2026/information_retrieval/neighbor-aware_instance_refining_with_noisy_labels_for_cross-modal_retrieval.md)
- [\[ICCV 2025\] External Knowledge Injection for CLIP-Based Class-Incremental Learning](../../ICCV2025/information_retrieval/external_knowledge_injection_for_clip-based_class-incremental_learning.md)
- [\[ICML 2026\] Retriever Portfolios: A Principled Approach to Adaptive RAG](retriever_portfolios_a_principled_approach_to_adaptive_rag.md)
- [\[ICML 2026\] Very Efficient Listwise Multimodal Reranking for Long Documents](very_efficient_listwise_multimodal_reranking_for_long_documents.md)
- [\[ICML 2026\] ParisKV: Fast and Drift-Robust KV-Cache Retrieval for Long-Context LLMs](pariskv_fast_and_drift-robust_kv-cache_retrieval_for_long-context_llms.md)

</div>

<!-- RELATED:END -->
