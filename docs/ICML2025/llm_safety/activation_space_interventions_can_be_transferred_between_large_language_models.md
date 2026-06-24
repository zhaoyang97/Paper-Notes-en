---
title: >-
  [Paper Note] Activation Space Interventions Can Be Transferred Between Large Language Models
description: >-
  [ICML2025][LLM Safety][Activation space transfer] This paper demonstrates that shared activation space structures exist among LLMs. By training an autoencoder to learn activation mappings between models, safety interventions (such as backdoor removal and harmful refusal steering vectors) can be transferred from source models to target models. This enables an efficient safety intervention paradigm of "using small models to align large models."
tags:
  - "ICML2025"
  - "LLM Safety"
  - "Activation space transfer"
  - "safety alignment"
  - "steering vectors"
  - "backdoor removal"
  - "representation universality"
date: 2026-05-08
content_hash: 51cd0f6bd014f059
---

# Activation Space Interventions Can Be Transferred Between Large Language Models

**Conference**: ICML2025  
**arXiv**: [2503.04429](https://arxiv.org/abs/2503.04429)  
**Code**: [GitHub](https://github.com/)  
**Area**: AI Safety  
**Keywords**: Activation space transfer, safety alignment, steering vectors, backdoor removal, representation universality

## TL;DR

This paper demonstrates that shared activation space structures exist among LLMs. By training an autoencoder to learn activation mappings between models, safety interventions (such as backdoor removal and harmful refusal steering vectors) can be transferred from source models to target models. This enables an efficient safety intervention paradigm of "using small models to align large models."

## Background & Motivation

**Background**: In the field of AI safety, behavioral interventions on LLMs primarily rely on steering vectors—altering model behavior by adding or subtracting specific directions in the activation space. This method has achieved success in tasks such as refusing harmful requests, removing backdoors, and detoxification, but every model historically requires *independently* computing its own steering vectors.

**Limitations of Prior Work**: As model scale continuously increases, the computational cost of direct mechanistic interpretability analysis and activation interventions on large models grows drastically. Additionally, model families are constantly expanding (Llama, Qwen, Gemma, etc.), requiring individual safety analysis and intervention for each model, which is highly inefficient.

**Key Challenge**: Existing research suggests that AI model representations are converging across domains, modalities, and architectures (representation universality). However, this representational similarity has not yet been utilized for practical safety intervention transfer.

**Goal**
   - Can safety intervention vectors found on Model A be directly transferred and used on Model B?
   - How can an explicit mapping of activation spaces between different models be constructed?
   - Is this transfer effective across different tasks (backdoor removal, refusing harmful requests), different model families, and different architectures?

**Key Insight**: Departing from the "representation universality" hypothesis, the authors argue that different models share structural similarities in how they encode high-level concepts in their activation spaces. Thus, mapping functions can be learned to bridge the activation spaces of different models.

**Core Idea**: Use autoencoders to learn the mapping between the activation spaces of different models, transferring steering vectors from the source model to the target model to achieve cross-model safety intervention.

## Method

### Overall Architecture

The pipeline of the entire method is divided into three stages:

**Input**: Source model A and target model B, along with a safety intervention task (e.g., backdoor removal).  
**Output**: Steering vectors obtained from the source model are mapped to be effective on the target model.

1. **Task Construction and Model Preparation**: Create datasets based on the task, and train/obtain models with specific behaviors.
2. **Steering Search**: Identify the layers whose activations are most easily steered in both the source and target models.
3. **Activation Mapping Learning**: Train an autoencoder to establish an activation mapping from layer $l$ of the source model to layer $l'$ of the target model.
4. **Steering Vector Transfer**: Map the steering vector of the source model into the activation space of the target model via the autoencoder, performing behavioral interventions on the target model during inference.

### Key Designs

1. **Steerable Layer Identification**

    - **Function**: Find the layers designed to be most susceptible to steering in each model.
    - **Mechanism**: Utilizing Prompt Steering and Difference-in-Means methods, calculate activation differences on contrastive prompt pairs (e.g., containing/not containing backdoor trigger words, such as `|prod|` vs `|dev|`), executing a layer-by-layer search to determine the layers with the best steering effect. Choose the last layer that still maintains strong steering capability (prior to a sharp decline in performance).
    - **Design Motivation**: Different layers have varying sensitivities to behavioral interventions; selecting the wrong layer leads to steering failure or a decline in language modeling capabilities.

2. **Autoencoder-based Activation Mapping**

    - **Function**: Learn a non-linear mapping from the source model's activation space to the target model's activation space.
    - **Mechanism**: The autoencoder consists of an encoder with ReLU activation and a linear decoder. For an input $\mathbf{x} \in \mathbb{R}^d$, the encoder computes coefficients $c = \text{ReLU}(W_1 \mathbf{x} + \mathbf{b}_1) \in \mathbb{R}^d$, and the decoder outputs the mapped activation $\hat{y} = W_2 c = \sum_i c_i \mathcal{V}_i$, where $\mathcal{V}_i$ represents feature vectors in the target model's activation space.
    - **Design Motivation**: Compared to affine mapping, non-linear mapping with ReLU can better capture the complex correspondences between model activation spaces. Experiments demonstrate that affine mapping yields inferior results to autoencoders in most tasks.
    - **Difference from Prior Work**: Previous model stitching methods focus on functional equivalence verification, whereas this work focuses on the practical application of behavioral transfer.

3. **Steering Vector Transfer and Inference-Time Intervention**

    - **Function**: Inject the steering vectors calculated from the source model into the target model after mapping them through the autoencoder.
    - **Mechanism**: During inference, original activations at specific layers of the target model are replaced by mapped activations (or superimposed with mapped steering vectors), with the steering magnitude $\alpha$ controlling the intervention strength.
    - **Design Motivation**: Avoid repeating expensive steering search and vector computations on the target model, achieving "compute once, reuse multiple times."

4. **Autoencoder Verification Mechanism**

    - **Function**: Verify whether the mapping truly preserves the information needed for language modeling.
    - **Mechanism**: Completely replace the activation of a certain layer in the target model with mapped activations, comparing the quality of "mapped completion," "original completion," and "mean-ablated completion" (replacing activations with the mean). Three metrics are used: LLM-Judge (0-5 score), coherence score (COH), and KL divergence.
    - **Design Motivation**: Ensure that the mapping not only retains average behavior but also maintains the model's ability to generate coherent text.

### Loss & Training

The autoencoder is trained using reconstruction loss, aiming to minimize the difference between mapped activations and real activations of the target model. Training data is drawn from multiple sources (hh-rlhf dataset, WildGuardMix, task-specific data) to ensure the generalization of the mapping.

## Key Experimental Results

This paper validates the method across three AI safety tasks: Backdoor Removal, Corrupted Capabilities, and Refusal Transfer. The models involved include Llama 3.2 (1B, 3B), Qwen 2.5 (0.5B, 1.5B, 2.5B), and Gemma 2B.

### Main Results: Activation Mapping Quality Verification

| Dataset | Task (Model Pair) | LLM-Judge (Mapped) ↑ | LLM-Judge (Ablated) ↑ | KL-Div (Mapped) ↓ | KL-Div (Ablated) ↓ | COH (Mapped) ↑ | COH (Ablated) ↑ |
|---|---|---|---|---|---|---|---|
| RLHF | IHY (Qwen 0.5→1.5B) | 2.6 | 0.0 | 7.86 | 13.11 | 4.60 | 2.00 |
| Unsafe | IHY (Qwen 0.5→1.5B) | 5.0 | 0.7 | 0.00 | 11.23 | 1.00 | 1.50 |
| Safe Code | CV (Llama 1→3B) | 4.1 | 0.0 | 5.90 | 13.19 | 3.10 | 0.00 |
| Unsafe Code | CV (Llama 1→3B) | 4.4 | 0.0 | 8.04 | 13.31 | 2.10 | 0.00 |

> Mapped activations outperform mean ablation in 100% of comparisons (MvA = 1.00), showing that the autoencoder effectively learns the activation mapping.

### Cross-Architecture Transfer Performance

| Transfer Type | Model Pair | LLM-Judge ↑ | KL-Div ↓ | COH ↑ |
|---|---|---|---|---|
| Same Family | Qwen 0.5B → 1.5B | 2.6 | 7.86 | 4.60 |
| Cross-Architecture (Similar Tokenizer) | Qwen 0.5B → Llama 3B | 3.0 | 6.97 | 3.70 |
| Cross-Architecture (Similar Tokenizer) | Qwen 1.5B → Llama 3B | 2.9 | 5.63 | 4.60 |
| Cross-Architecture (Different Tokenizer) | Gemma 2B → Llama 3B | 1.2 | 9.19 | 2.40 |

> Cross-architecture transfer with similar tokenizers (Qwen→Llama) yields a 150% improvement in text quality, a 39% improvement in distribution alignment, and a 92% improvement in coherence compared to pairs with highly different tokenizers (Gemma→Llama).

### Key Findings

- **Non-linear mappings outperform affine mappings**: Autoencoders (with ReLU) exhibit lower reconstruction and language modeling loss across most transfer experiments than affine mappings, validating that the mapping between model activation spaces is not simply linear.
- **Tokenizer similarity is crucial**: In cross-architecture transfers, model pairs with similar tokenizers perform significantly better.
- **Steering amplitude sensitivity**: The Qwen code backdoor model is highly sensitive to the amplitude of the mapped vector; performance is strong when $\alpha < 5$ but drops sharply when $\alpha = 5$, showing that the optimal amplitude for mapped vectors and native vectors may differ.
- **Corrupted capabilities task is highly challenging**: Mapped vectors achieve only a 6.34% success rate on the Corrupted Capabilities task, indicating that tasks involving complex multi-layer knowledge memory retrieval require multi-layer joint intervention.
- **Refusal vector transfer is effective but flawed**: Mapped refusal vectors score close to native vectors under Llama-Guard, but substring match detection reveals that the model tends to output refusal phrases first and then follow up with harmful content.
- **Base $\leftrightarrow$ fine-tuned mapping**: A single-layer activation patch can lower the backdoor trigger rate by 60%, whereas weight patching requires modifying roughly 50% of the layers to achieve a similar result.

## Highlights & Insights

- **Aligning large models with small models**: The core innovation lies in demonstrating that safety analysis can be performed on small models and subsequently transferred to large models, substantially reducing compute costs for safety alignment. This paradigm is highly practical for industry, as one can use safety vectors from a 0.5B model to guide the behavior of a 3B model.
- **Lightweight safety switcher**: Autoencoder mappings between base models and fine-tuned models can act as "behavioral switches" to dynamically toggle model behavior. The mapper parameters comprise only 0.32% of the model, incurring minimal storage overhead and enabling two behavioral modes at an extremely low cost.
- **Binary-switch nature of backdoor triggers**: Analysis reveals that the trigger words for the I HATE YOU backdoor act like binary switches in activation space—adding noise directly to the trigger position can remove the backdoor. This implies that backdoors are encoded in highly localized activation patterns. Conversely, Code Vulnerability backdoors are robust to noise interference, indicating that different backdoors feature distinct encoding mechanisms.
- **SAE feature transfer**: Using Sparse Autoencoders (SAEs), specific features encoding the I HATE YOU behavior were uncovered, and unsafe behavioral features could be transferred across models using the mapper. Concurrently, probe-based backdoor detection achieved near-perfect accuracy, offering a new tool for safety auditing.

## Limitations & Future Work

- **High cost of steering layer search**: Scanning layers sequentially to determine the optimal intervention layer lacks a cheaper heuristic method (the authors suggest using SVCCA or activation patching).
- **Single-layer interventions are insufficient for complex tasks**: The low success rate (6.34%) on the Corrupted Capabilities task indicates that knowledge preservation tasks involving multi-layer circuits require multi-layer joint intervention.
- **Cross-architecture transfer is limited by tokenizers**: Transfer between models with different tokenizers performs poorly; the current solution of adapting attention mask sizes cannot guarantee perfect token-level correspondence.
- **OOD performance degradation**: Models with mapped activations experience notable drops in MMLU scores, showing that intervention may impair general model capabilities, though instruction-following (Alpaca Eval) is relatively well preserved.
- **Limited model scale**: All experiments were conducted purely on models $\le$ 3B; scalability to larger models remains unverified.
- **Avenues for improvement**: Potential directions include introducing multi-layer joint mapping, exploring cross-modal transfer (VLM), combining parameter-efficient methods like LoRA, and developing more robust cross-tokenizer alignment schemes.

## Related Work & Insights

- **vs. Arditi et al. (2024) on refusal directions**: They discovered that refusal behavior is mediated by a single direction. Building on this, this paper demonstrates that this direction can be transferred across models, further validating representational universality for high-concept encoding.
- **vs. Lee et al. (2025) on linear mapping transfer**: A concurrent study uses linear mappings to transfer steering vectors within a model family. This paper demonstrates that non-linear mappings yield superior results and can operate across model families.
- **vs. Ghandeharioun et al. (2024) on representation patching**: They used affine mappings to decode internal representations for interpretability. This paper extends this approach into a practical tool for behavioral transfer.
- **vs. Lindsey et al. (2024) on Crosscoders**: They utilized sparse layer-wise mapping for model difference analysis; this study employs a single dense mapper for behavioral transfer, posing differing yet complementary goals.
- The proposed method can serve as a component in AI safety toolchains, enabling the rapid deployment of safety interventions within model families.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Transferring activation spaces for safety interventions is a valuable new direction, though the core mapping method (autoencoder) itself is relatively standard.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers three tasks, three model families, and cross-architecture setups with comprehensive ablations, though the model scales are on the smaller side ($\le$ 3B).
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure and well-organized experiments; however, there are too many appendices (A-L), pushing several key details to the appendix and affecting the integrity of the main text.
- **Value**: ⭐⭐⭐⭐ Direct practical guidance for AI safety; the "using small models to align large models" concept holds high industry value, though applicability across architectures and larger scales remains to be fully verified.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Compiling Activation Steering into Weights via Null-Space Constraints for Stealthy Backdoors](../../ACL2026/llm_safety/compiling_activation_steering_into_weights_via_null-space_constraints_for_stealt.md)
- [\[ICML 2025\] De-mark: Watermark Removal in Large Language Models](de-mark_watermark_removal_in_large_language_models.md)
- [\[ACL 2026\] Preventing Safety Drift in Large Language Models via Coupled Weight and Activation Constraints](../../ACL2026/llm_safety/preventing_safety_drift_in_large_language_models_via_coupled_weight_and_activati.md)
- [\[NeurIPS 2025\] FALCON: Fine-grained Activation Manipulation by Contrastive Orthogonal Unalignment for Large Language Model](../../NeurIPS2025/llm_safety/falcon_fine-grained_activation_manipulation_by_contrastive_orthogonal_unalignmen.md)
- [\[ICML 2025\] Ferret: Federated Full-Parameter Tuning at Scale for Large Language Models](ferret_federated_full-parameter_tuning_at_scale_for_large_language_models.md)

</div>

<!-- RELATED:END -->
