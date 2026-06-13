---
title: >-
  [Paper Note] AOEPT: Breaking the Implicit Modality-Reduction Bottleneck in Modality-Missing Prompt Tuning
description: >-
  [ICML 2026][Multimodal VLM][Missing Modality] AOEPT identifies that existing missing modality prompt tuning compresses the reasoning scope of Multimodal Transformers into visible modality subspaces. By utilizing Modal-Co…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "Missing Modality"
  - "Multimodal Transformer"
  - "Prompt Tuning"
  - "Modal-Contextualized Prompt"
  - "NM2I"
date: 2026-05-08
content_hash: 857421bef2e13b67
---

# AOEPT: Breaking the Implicit Modality-Reduction Bottleneck in Modality-Missing Prompt Tuning

**Conference**: ICML 2026  
**arXiv**: [2605.24816](https://arxiv.org/abs/2605.24816)  
**Code**: https://github.com/Jian-Lang/AOEPT  
**Area**: Multimodal VLM / Missing Modality Learning  
**Keywords**: Missing Modality, Multimodal Transformer, Prompt Tuning, Modal-Contextualized Prompt, NM2I  

## TL;DR
AOEPT identifies that existing missing modality prompt tuning compresses the reasoning scope of Multimodal Transformers into visible modality subspaces. By utilizing Modal-Contextualized Prompts distilled from the training set as retrievable implicit information sources for missing modalities, AOEPT consistently outperforms existing methods across multiple datasets, missing rates, and backbones.

## Background & Motivation
**Background**: Multimodal systems typically rely on multi-source signals such as images, text, and audio for classification, understanding, or QA tasks. As Multimodal Transformers (MT) like CLIP, ViLT, and MulT have become universal backbones, recent research on missing modalities has shifted from customized networks to lightweight prompt tuning: freezing the pre-trained MT and learning only a few prompts and task heads to adapt the model to image-missing, text-missing, or incomplete multimodal scenarios at deployment.

**Limitations of Prior Work**: Methods such as MAPs, DCP, MemPrompt, and SyP are more robust than vanilla MT, but their prompts are often determined solely by the missing patterns or the currently visible modalities. For example, when an image is missing, the conditional signal for the prompt comes mainly from the text; when text is missing, it comes mainly from the image. While this seems reasonable, it forces the model to perform reasoning based only on the remaining single-modality evidence.

**Key Challenge**: Pre-trained MTs originally possess cross-modal modeling capabilities, but missing modality prompt tuning degrades the problem into a "visible modality-to-label" mapping. The authors term this phenomenon the Implicit Modality-Reduction (IMR) bottleneck: the prompt fails to explicitly access the potential information sources of the missing modality, causing the MT's reasoning range to be implicitly restricted to the modality-reduced subspace.

**Goal**: This paper aims to solve three specific problems. First, to explain why existing prompt tuning still fails to fully release the multimodal capabilities of MT in missing modality scenarios. Second, to design a prompt mechanism that is lightweight and avoids external retrieval or large reconstruction modules, allowing missing modalities to enter reasoning as implicit information bases. Third, to provide a metric that can diagnose the IMR bottleneck beyond just reporting final classification metrics.

**Key Insight**: The authors first conducted a simple pilot experiment: replacing the randomly initialized prompts in MAPs with global priors obtained by clustering modality token representations from the training set. This minor change improved performance in missing modality scenarios on MM-IMDb, suggesting that the global context of missing modalities can indeed break the single-modality reasoning bottleneck.

**Core Idea**: AOEPT replaces "generating prompts based only on visible modalities" with "modality-level global information base + instance-level conditional activation," allowing the prompt to actively supplement the current sample with implicit context of the missing modality rather than just adapting to the degraded input structure.

## Method
The main pipeline of AOEPT is clear: first, collect layer-wise representations of a specific modality from the training set and compress them into lightweight Modal-Contextualized Prompts (MCPs); then, instantiate these global MCPs into sample-correlated prompts based on the remaining modalities of the current sample; finally, insert these prompts into several layers of the frozen MT, training only the prompts and the classification head. Taking image-visible and text-missing as an example, AOEPT constructs Text-Contextualized Prompts (TCPs), allowing image samples to still access the implicit semantic base of the text modality during inference.

### Overall Architecture
The input consists of multimodal samples that may lack certain modalities, such as $(t, v)$, $(t, \varnothing)$, or $(v, \varnothing)$. AOEPT does not change the main MT structure but inserts prompt tokens between layers of the pre-trained Transformer. During the training phase, visible samples from the training set are first processed through the frozen MT to extract layer-wise modality representations; these representations are clustered and compressed to construct the corresponding MCPs.

When a test sample lacks text, the model retrieves TCPs; when it lacks images, it retrieves ICPs. Since MCPs are global, they are gated using the remaining modality representations of the current sample to obtain instance-aware prompts. Finally, these prompts are concatenated with the original hidden tokens and fed into the MT layers, with the classification head outputting predictions based on the final layer's representation.

### Key Designs
1.  **Modal-Contextualized Prompts as Missing Modality Information Bank**:
	- **Function**: Compresses the global context of a modality from the training set into a set of prompt tokens, serving as an implicit information source accessible when that modality is missing.
	- **Mechanism**: Taking TCP as an example, all text-available samples are fed into the frozen MT to obtain the set of text token representations $C_t^l$ for each layer; K-means is then used to compress the large volume of tokens into $N_t'$ semantic prototypes to reduce storage and computation costs; the default construction is attention-based, using learnable prompts as queries to perform cross-attention over these text prototypes to obtain the TCP for each layer.
	- **Design Motivation**: Random prompts only signal to the model that "a modality is missing" but cannot provide information on what that missing modality might contain. MCPs explicitly store the context of the corresponding modality from the training distribution, effectively reconnecting the frozen MT to a lightweight, internalized modal memory.

2.  **Instance-aware prompt instantiation**:
	- **Function**: Transforms global MCPs into sample-specific prompts, preventing all missing samples from sharing the same set of coarse-grained compensation information.
	- **Mechanism**: For samples with visible images and missing text, the image hidden representation is passed through an MLP and sigmoid to generate a gating vector, which is then element-wise multiplied with the TCPs: $P_{TCP,i}^l = P_{TCP}^l \odot \sigma(MLP(\bar{V}_i^{l-1}))$. Thus, the current image selectively activates the text context most relevant to itself.
	- **Design Motivation**: MCPs are modality-level information bases and would be too averaged without sample-level conditioning. The instantiation step projects the "global text distribution" into a local space of "what text semantics this image might correspond to," which is the key to AOEPT moving from global priors to sample-level compensation.

3.  **Consistency constraints and Adaptive Insertion**:
	- **Function**: Makes the instantiated prompts more similar to the latent representations of actual missing modalities and controls how prompts propagate through Transformer layers.
	- **Mechanism**: The authors propose intra-modal latent consistency regularization, calculated only on training samples where the corresponding modality is available. The pooled instance-aware prompt and the real modality representation of the same sample are treated as positive pairs, while representations of other samples in the batch are negative pairs, constrained by an InfoNCE-style contrastive objective. During insertion, prompts are re-instantiated and inserted for the first $N$ layers, while subsequent layers inherit and propagate the prompts from the previous layer.
	- **Design Motivation**: Relying solely on classification loss might cause prompts to learn label-related information that does not necessarily represent the missing modality; consistency constraints pull the prompts toward the real modality latent space. The layered insertion strategy avoids the overhead of re-generating prompts for every layer while retaining the ability of early layers to fully compensate for the missing modalities.

### Loss & Training
AOEPT freezes the pre-trained MT and trains only the MCPs and the task classification head. The total objective is the classification loss $L_{CE}$ plus the consistency regularization $L_{CR}$, where $L_{CR}$ constrains the instantiation quality using the similarity between the prompt and the real modality latent representation.

Main experiments use CLIP ViT-B/16 as the dual-stream MT backbone, with extended experiments on the ViLT single-stream backbone and MulT tri-modal backbone. The authors set the refined modality set capacity to 256, default prompt length to $M=16$, and prompt tuning depth to $N=6$, which is the trade-off point between performance and efficiency.

There are three variants of MCP construction. Attention-based is the default method with balanced performance and overhead; MLP-based has slightly higher performance but more extra computation; Initialization-based directly uses pooled modality prototypes to initialize prompts, having the lowest inference overhead but the weakest effect. This design also demonstrates that AOEPT succeeds not through a complex module, but through the paradigm shift of "connecting prompts to the missing modality context."

## Key Experimental Results

### Main Results
The main experiments cover three multimodal image-text benchmarks: MM-IMDb, HateMemes, and Food101, with missing rates of 70% or 90% set during both training and testing. Evaluation metrics are F1-Macro, AUC, and Accuracy. AOEPT exceeds strong baselines like MAPs, DCP, RAGPT, MemPrompt, and SyP across all average metrics.

| Missing Rate | Dataset | Metric | AOEPT Avg. | Prev. SOTA Avg. | Gain |
|--------|--------|------|------------|----------------|------|
| 70% | MM-IMDb | F1-M | 53.22 | 51.88 (SyP) | +1.34 |
| 70% | HateMemes | AUC | 69.63 | 68.11 (SyP) | +1.52 |
| 70% | Food101 | ACC | 84.29 | 83.56 (SyP) | +0.73 |
| 90% | MM-IMDb | F1-M | 51.45 | 49.58 (SyP) | +1.87 |
| 90% | HateMemes | AUC | 68.57 | 67.72 (SyP) | +0.85 |
| 90% | Food101 | ACC | 82.06 | 81.26 (SyP) | +0.80 |

Looking closer at individual scenarios at a 70% missing rate, AOEPT achieved 51.50, 54.86, and 53.31 for text missing, image missing, and both missing on MM-IMDb; 71.12, 67.96, and 69.80 on HateMemes; and 80.77, 88.86, and 83.24 on Food101. The improvement is not limited to one missing pattern but exists across text-missing, image-missing, and bidirectional mixed-missing scenarios.

### Ablation Study
The ablation study at a 70% text missing rate validates three key modules of AOEPT: MCP, instance-level instantiation, and consistency regularization. The results show that removing MCP and reverting to random prompts causes one of the most significant drops; directly inserting uninstantiated MCP is also notably weaker than the full model; using a lightweight reconstruction network instead of MCP does not solve the problem.

| Configuration | MM-IMDb F1-M | HateMemes AUC | Food101 ACC | Description |
|------|--------------|---------------|-------------|------|
| w/o MCP | 48.93 | 68.63 | 78.78 | Replace modal-context bank with random prompts |
| w/o Instantiation | 49.17 | 69.42 | 79.13 | Insert global MCP directly without conditioning |
| w/o Consistency | 50.56 | 69.85 | 79.59 | Remove latent consistency regularization |
| w/ Reconstruction | 48.55 | 70.13 | 76.81 | Replace MCP with a reconstruction network of similar size |
| AOEPT | 51.50 | 71.12 | 80.77 | Full method |

### Key Findings
- MCP is the core of breaking the IMR bottleneck. Without MCP, the model reverts to using random prompts only to adapt to the missing structure, with MM-IMDb F1-M dropping from 51.50 to 48.93.
- Instance-aware instantiation is not merely "the icing on the cake." The w/o Instantiation version is lower than the full model across all three datasets, indicating that the global modality library must be selectively activated based on currently visible modalities.
- Reconstruction is not a good alternative in lightweight prompt scenarios. A reconstruction network with similar parameter counts achieved only 76.81 on Food101, significantly lower than the full AOEPT's 80.77. The authors suggest this is because the number of complete samples for cross-modal reconstruction is limited, and lightweight networks struggle to fit complex cross-modal mappings.
- NM2I diagnostics support the central thesis. The NM2I of baseline methods is close to 0, indicating that the prompts share almost no information with the missing modality's latent representation; AOEPT's NM2I is significantly higher, proving it successfully pulls the prompts into the missing modality's information space.
- When training conditions improve, baselines exhibit a modality information scaling bottleneck. For instance, with test text missing fixed at 90%, reducing the training text missing rate does not consistently help baselines or might even hinder generalization to severe missingness; AOEPT, however, can absorb more text information available during training into the MCPs.

## Highlights & Insights
- The biggest highlight is reframing the missing modality problem from "balancing degraded input" to "restoring reasoning scope." This perspective is highly explanatory, as it points out why many prompt tuning methods, despite being parameter-efficient, still fail to truly utilize pre-trained MT capabilities.
- The design of MCP is restrained: it introduces neither an external retrieval library nor a large generative reconstructor, but instead distills modality-level context from within the training set. This allows the method to supplement missing modality information without being dependent on sample-level retrieval quality like RAGPT.
- NM2I is a valuable diagnostic tool. While not a direct substitute for task performance, it answers the more mechanistic question of "whether the prompt actually carries missing modality information," making it transferable to other studies on missing modalities or multimodal robustness.
- The analysis of prompt length and insertion depth is practical. The conclusion of $M=16, N=6$ might not be universally optimal, but it provides a key insight: missing modality compensation requires sufficient token capacity and early-layer intervention; being too short or too shallow makes it difficult to fully restore the cross-modal space.

## Limitations & Future Work
- The authors admit that NM2I is not necessarily monotonically related to task performance. In some datasets, the visible modality itself might be strong enough that classification metrics appear good even if an IMR bottleneck exists; thus, NM2I is better as a mechanistic diagnosis rather than the sole optimization target.
- AOEPT assumes the semantic distribution of training and testing is not significantly different. Since MCPs are distilled from the training set, a severe domain shift in the deployment environment might lead the modality bank to activate mismatched priors.
- The method is primarily validated on classification benchmarks. Whether missing modality information in open-ended VQA, complex MLLM reasoning, long video understanding, or multi-turn interaction tasks can be represented by similarly lightweight MCPs requires more experimentation.
- Future work could combine AOEPT with uncertainty estimation: when the confidence of the missing modality prior activated by MCP is low, the model could reduce compensation intensity or request additional input rather than always assuming training priors are valid.

## Related Work & Insights
- **vs MAPs**: MAPs first introduced missing-aware prompts to MT learning, but the prompts acted more like input structure markers. AOEPT argues this still falls into the IMR bottleneck and uses MCPs to explicitly provide missing modality context.
- **vs DCP / MemPrompt / SyP**: These methods make prompts more sample-specific, memorialized, or cross-modally shared, improving performance, but the conditional signals still primarily come from the remaining modalities. AOEPT's difference lies in constructing an information library for the missing modality first, then selectively activating it.
- **vs RAGPT / REDEEM**: Retrieval-based methods supplement multimodal evidence via external samples or reconstruction modules, offering more explicit information but with higher training/inference overhead. AOEPT is like internalizing the retrieval into prompt parameters.
- **vs Modality Imputation**: Traditional imputation attempts to reconstruct the missing modality itself, usually requiring customized networks and full-sample supervision. AOEPT does not generate images or text but supplements the latent prompt space with modality context useful for prediction, which is more suitable for parameter-efficient adaptation of frozen MTs.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Proposes the IMR bottleneck and MCP paradigm with clear problem definition and solution path.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers multiple datasets, missing rates, backbones, and ablations; inclusion of open-ended MLLM tasks would be even more complete.
- Writing Quality: ⭐⭐⭐⭐ Smooth structure with sufficient motivation and method explanation.
- Value: ⭐⭐⭐⭐⭐ Provides a mechanistic explanation and a lightweight solution for missing modality prompt tuning, alongside reusable analysis tools like NM2I.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Jailbreaking Vision-Language Models Through the Visual Modality](jailbreaking_vision-language_models_through_the_visual_modality.md)
- [\[ICML 2026\] Seeing is Understanding: Unlocking Causal Attention into Modality-Mutual Attention for Multimodal LLMs](seeing_is_understanding_unlocking_causal_attention_into_modality-mutual_attentio.md)
- [\[ICML 2026\] ECG-R1: Protocol-Guided and Modality-Agnostic MLLM for Reliable ECG Interpretation](ecg-r1_protocol-guided_and_modality-agnostic_mllm_for_reliable_ecg_interpretatio.md)
- [\[CVPR 2026\] Purify-then-Align: Towards Robust Human Sensing under Modality Missing with Knowledge Distillation from Noisy Multimodal Teacher](../../CVPR2026/multimodal_vlm/purify-then-align_towards_robust_human_sensing_under_modality_missing_with_knowl.md)
- [\[ICML 2026\] Calibrated Multimodal Representation Learning with Missing Modalities](calibrated_multimodal_representation_learning_with_missing_modalities.md)

</div>

<!-- RELATED:END -->
