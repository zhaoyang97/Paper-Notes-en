---
title: >-
  [Paper Note] SAUCE: Selective Concept Unlearning in Vision-Language Models with Sparse Autoencoders
description: >-
  [ICCV 2025][LLM Safety][concept unlearning] SAUCE leverages sparse autoencoders (SAEs) to identify and selectively suppress features associated with target concepts in VLM intermediate representations, enabling fine-grained concept unlearning without weight updates. Evaluated across 60 concepts, it surpasses the previous SOTA in forgetting quality by 18%.
tags:
  - "ICCV 2025"
  - "LLM Safety"
  - "concept unlearning"
  - "sparse autoencoder"
  - "vision-language model"
  - "machine unlearning"
  - "fine-grained control"
date: 2026-05-08
content_hash: fd93a420751e3764
---

# SAUCE: Selective Concept Unlearning in Vision-Language Models with Sparse Autoencoders

**Conference**: ICCV 2025
**arXiv**: [2503.14530](https://arxiv.org/abs/2503.14530)  
**Code**: None  
**Area**: Multimodal VLM
**Keywords**: concept unlearning, sparse autoencoder, vision-language model, machine unlearning, fine-grained control

## TL;DR

SAUCE leverages sparse autoencoders (SAEs) to identify and selectively suppress features associated with target concepts in VLM intermediate representations, enabling fine-grained concept unlearning without weight updates. Evaluated across 60 concepts, it surpasses the previous SOTA in forgetting quality by 18%.

## Background & Motivation

**Background**: As VLMs (e.g., LLaVA, LLaMA-Vision) are increasingly deployed, the ability to make models "forget" specific concepts—such as harmful or copyrighted content—has become critically important. Existing VLM unlearning methods largely inherit techniques developed for LLMs.

**Limitations of Prior Work**: Current unlearning approaches rely predominantly on weight update strategies, suffering from two core issues: (1) they require large annotated forget sets, which are costly to obtain; and (2) they operate at too coarse a granularity, often causing excessive forgetting that significantly degrades model utility on unrelated tasks.

**Key Challenge**: There is a fundamental trade-off between unlearning precision and model utility. Weight-update-based methods cannot precisely localize which neurons or features encode a target concept, and thus can only perform coarse-grained adjustments across the entire parameter space.

**Goal**: To design a fine-grained concept unlearning method that requires neither weight modification nor large annotated datasets, and that can be applied on-demand at inference time.

**Key Insight**: SAEs can decompose high-dimensional dense representations into high-dimensional sparse, semantically interpretable features. If the feature dimensions most relevant to a target concept can be identified via an SAE, those dimensions can be selectively suppressed at inference time.

**Core Idea**: SAEs decompose VLM intermediate-layer representations into interpretable sparse features. Correlation analysis then localizes the key features of a target concept, which are selectively modified at inference time to achieve unlearning—without altering model weights, only intervening in feature propagation during inference.

## Method

### Overall Architecture

SAUCE proceeds in three stages: (1) training an SAE to learn sparse representations from VLM intermediate layers; (2) given a target concept to forget, identifying the most critical sparse feature dimensions via feature correlation analysis; and (3) suppressing those feature dimensions at inference time to prevent the model from generating content related to the target concept. The input is a standard image-text pair; the output is a model response with the target concept effectively unlearned.

### Key Designs

1. **Sparse Autoencoder Training**:

    - **Function**: Decomposes the dense activations of VLM intermediate layers into high-dimensional sparse representations.
    - **Mechanism**: An SAE is trained on key intermediate layers of the VLM (e.g., attention layer outputs). The encoder maps a $d$-dimensional dense vector to a $D$-dimensional ($D \gg d$) sparse space, and the decoder reconstructs the original. L1 regularization enforces sparsity, encouraging each feature dimension to encode an independent semantic concept.
    - **Design Motivation**: Concepts are entangled in dense representations and cannot be precisely located. Through overcomplete representations and sparsity constraints, SAEs naturally achieve concept-level disentanglement.

2. **Concept-Relevant Feature Identification**:

    - **Function**: Identifies a small subset of features, from among thousands of sparse dimensions, that are most relevant to the target concept.
    - **Mechanism**: A small number of samples containing the target concept (no large-scale annotation required) are passed through the VLM and SAE to obtain sparse activations. The mean activation of each feature dimension over these samples is computed, and the top-$k$ most activated features are designated as concept-relevant.
    - **Design Motivation**: The sparsity of the SAE ensures that most feature dimensions remain inactive for any given concept input, so highly activated features exhibit strong concept selectivity, making this statistical approach simple yet effective.

3. **Inference-Time Feature Suppression**:

    - **Function**: Achieves concept unlearning without modifying model weights.
    - **Mechanism**: During inference, intermediate-layer activations are encoded into sparse representations via the SAE. The activations of the identified concept-relevant feature dimensions are set to zero or scaled down (clamping), after which the SAE decoder reconstructs the dense representation, which is passed back to the model. This effectively severs the flow of concept-related information along the signal propagation path.
    - **Design Motivation**: Compared to retraining or fine-tuning, inference-time intervention incurs zero training cost, can be enabled or disabled instantly, supports simultaneous handling of multiple unlearning requests, and leaves model weights untouched.

### Loss & Training

SAE training uses a standard reconstruction loss with an L1 sparsity penalty:

$$\mathcal{L} = \|x - \hat{x}\|_2^2 + \lambda \|z\|_1$$

where $x$ is the original activation, $\hat{x}$ is the reconstructed activation, and $z$ is the sparse code. The concept unlearning stage itself involves no training.

## Key Experimental Results

### Main Results

Experiments are conducted on two VLMs (LLaVA-v1.5-7B and LLaMA-3.2-11B-Vision-Instruct) and two unlearning task types (concrete concepts such as objects and action scenes, and abstract concepts such as emotions, colors, and materials), covering 60 concepts in total.

| Method | Unlearning Quality (UQ↑) | Model Utility (MU↑) | Overall |
|---|---|---|---|
| Gradient Ascent | Baseline | Significant drop | Low |
| Fine-tuning w/ Forget Set | Moderate | Moderate drop | Moderate |
| **SAUCE** | **+18.04%** | **Comparable** | **Best** |

### Ablation Study

| Configuration | Unlearning Quality | Model Utility | Notes |
|---|---|---|---|
| Full SAUCE | Best | Maintained | Complete method |
| Random feature suppression | Low | Degraded | Demonstrates necessity of feature identification |
| Suppress all features | High forgetting | Severely degraded | Excessive unlearning |
| Varying top-$k$ | Varies with $k$ | Varies with $k$ | Large $k$ degrades utility |

### Key Findings

- SAUCE surpasses the SOTA baseline in unlearning quality by 18.04% while maintaining comparable model utility.
- Concrete concepts (e.g., "cat," "basketball") are more amenable to unlearning than abstract concepts (e.g., "sadness," "metallic texture"), as concrete concepts are more concentrated in the SAE feature space.
- SAUCE demonstrates reasonable robustness against adversarial attacks designed to recover forgotten concepts via prompt manipulation.
- Concept-relevant features identified by an SAE trained on one VLM exhibit a degree of cross-model transferability.
- Multiple simultaneous unlearning requests are supported with minimal performance degradation.

## Highlights & Insights

- **Inference-time intervention paradigm**: In sharp contrast to mainstream weight-modification methods, SAUCE achieves unlearning through feature suppression at inference time, enabling dynamic activation and deactivation of the unlearning capability—a highly deployment-friendly property.
- **SAEs as a tool for model control**: This work demonstrates that SAEs can serve not only as interpretability tools but also as instruments for precise behavioral control, a perspective transferable to other tasks requiring fine-grained model steering.
- **Low data requirement**: Only a small number of target concept samples are needed to identify relevant features, eliminating the dependency on large-scale annotated forget sets required by conventional unlearning methods.

## Limitations & Future Work

- The paper has been withdrawn by the authors, with a note citing the need for additional comparative experiments, suggesting that experimental coverage may be insufficient.
- SAE training incurs additional computational cost, and the quality of the SAE directly affects unlearning performance.
- Introducing SAE encoding and decoding steps at inference time adds latency overhead.
- For deeply entangled concepts (e.g., "face" with "age/gender"), the effectiveness of selective unlearning may be limited.
- Future directions include: adaptive top-$k$ selection strategies, multi-layer SAE collaborative unlearning, and integration with RLHF-based safety alignment.

## Related Work & Insights

- **vs. Gradient Ascent methods**: Conventional approaches "forget" by maximizing loss on forget-set data, which disrupts the global structure of model weights. SAUCE avoids this entirely by leaving weights unchanged.
- **vs. EraseDiff / Concept Erasing**: These methods are designed for diffusion models and do not transfer directly to autoregressive VLMs. SAUCE is the first inference-time unlearning method specifically designed for VLMs.
- **vs. Mechanistic Interpretability**: SAEs have been widely used for LLM interpretability; SAUCE extends their role from "understanding the model" to "controlling the model," representing a meaningful application direction.

## Rating

- Novelty: ⭐⭐⭐⭐ Using SAEs for concept unlearning at inference time is a novel angle, though SAEs themselves are an established tool.
- Experimental Thoroughness: ⭐⭐⭐ The paper has been withdrawn; the authors acknowledge insufficient experimental coverage.
- Writing Quality: ⭐⭐⭐⭐ Motivation is clear and the method is described with reasonable completeness.
- Value: ⭐⭐⭐⭐ The inference-time intervention paradigm holds significant practical value for real-world deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] CRISP: Persistent Concept Unlearning via Sparse Autoencoders](../../ACL2026/llm_safety/crisp_persistent_concept_unlearning_via_sparse_autoencoders.md)
- [\[ICML 2025\] SAEBench: A Comprehensive Benchmark for Sparse Autoencoders in Language Model Interpretability](../../ICML2025/llm_safety/saebench_a_comprehensive_benchmark_for_sparse_autoencoders_in_language_model_int.md)
- [\[NeurIPS 2025\] Approximate Domain Unlearning for Vision-Language Models](../../NeurIPS2025/llm_safety/approximate_domain_unlearning_for_visionlanguage_models.md)
- [\[NeurIPS 2025\] SIMU: Selective Influence Machine Unlearning](../../NeurIPS2025/llm_safety/simu_selective_influence_machine_unlearning.md)
- [\[ICCV 2025\] LATTE: Collaborative Test-Time Adaptation of Vision-Language Models in Federated Learning](latte_collaborative_test-time_adaptation_of_vision-language_models_in_federated_.md)

</div>

<!-- RELATED:END -->
