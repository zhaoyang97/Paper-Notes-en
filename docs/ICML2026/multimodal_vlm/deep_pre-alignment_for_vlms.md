---
title: >-
  [Paper Note] Deep Pre-Alignment for VLMs
description: >-
  [ICML 2026][Multimodal VLM][Visual Encoder] The authors replace the standard "ViT + lightweight projector" visual encoding module in VLMs with a small VLM (perceiver). This allows the "dirty work" of modality alignment t…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "Visual Encoder"
  - "Modality Alignment"
  - "Perception Model"
  - "Catastrophic Forgetting"
  - "VLM Architecture"
date: 2026-05-08
content_hash: 5949fdb79498f499
---

# Deep Pre-Alignment for VLMs

**Conference**: ICML 2026  
**arXiv**: [2605.15300](https://arxiv.org/abs/2605.15300)  
**Code**: To be confirmed (the paper states "DPA Code and Model" will be released)  
**Area**: Multimodal VLM  
**Keywords**: Visual Encoder, Modality Alignment, Perception Model, Catastrophic Forgetting, VLM Architecture

## TL;DR
The authors replace the standard "ViT + lightweight projector" visual encoding module in VLMs with a small VLM (perceiver). This allows the "dirty work" of modality alignment to be completed within the upstream small VLM, preventing the downstream large LLM from wasting its depth on alignment in shallow layers. This approach improves performance by +1.9 points on a 4B model and +3.0 points on a 32B model across 8 multimodal benchmarks, reduces language capability forgetting by 32.9%, and only decreases inference throughput by 2–6%.

## Background & Motivation

**Background**: Current mainstream VLMs (LLaVA, Qwen-VL, InternVL, MiniCPM-o, etc.) almost all follow the same paradigm: a pre-trained ViT (such as CLIP) uses a linear or MLP projector to feed visual features into the input embedding space of a large LLM, relying on the LLM itself to handle cross-modality alignment.

**Limitations of Prior Work**: Recent representation analyses (MIR index by Huang et al. 2025, neuron circuit analysis by Nikankin et al. 2025) consistently point out that visual features output by the ViT still exhibit significant modality gaps with the text space in the shallow layers of the LLM. The first few layers of the LLM are forced to misappropriate a large number of parameters for "shallow modality alignment," crowding out capacity that should be used for deep understanding and complex reasoning. This "crowding" also triggers a common VLM ailment—catastrophic forgetting of text capabilities (the 4B baseline plummeted from 84.8 to 36.4 on MATH-500).

**Key Challenge**: Shallow layers are the most precious "general semantic entry" layers of the LLM. Forcing them to perform modality alignment is essentially trading wasted depth for architectural simplicity. To solve this, one must either change training objectives (data mixing), which only "treats the symptoms," or change the architecture to complete deep alignment before visual features enter the LLM.

**Goal**: To add "depth" to the visual encoder without modifying the training objectives or the LLM backbone—ensuring the heavy lifting of alignment is handled on the visual side, so the downstream LLM only receives visual features already near the text space.

**Key Insight**: The authors noted that a complete small VLM has already learned "how to push visual tokens toward the text space" on large-scale image-text data—its internal language blocks are a natural "alignment depth." By treating this small VLM entirely as the visual encoder for the large LLM, alignment becomes an internal behavior of the perceiver.

**Core Idea**: Replace the ViT encoder entirely with a small VLM (e.g., a Qwen3-0.6B-based perceiver) to decouple "modality alignment" and "deep reasoning" at the architectural level—the upstream small VLM is responsible for alignment, while the downstream large LLM focuses on reasoning.

## Method

### Overall Architecture
The DPA architecture consists of three parts in series: a small perception VLM $M_p$ (containing a ViT $\mathcal{E}$, internal projector $\phi_p$, and internal LLM blocks $M_p^{\text{LLM}}$), an alignment projector $\phi$, and a target large LLM $M_t$. The data flow of standard VLMs is $v \xrightarrow{\mathcal{E}} \mathbf{H}_v \xrightarrow{\phi} \mathbf{H}_v' \to M_t$. DPA changes this to $v \xrightarrow{\mathcal{E}} \mathbf{H}_v \xrightarrow{\phi_p} \mathbf{H}_v' \xrightarrow{M_p^{\text{LLM}}, \phi} \mathbf{H}_{\text{aligned}} \to M_t$. Visual tokens travel through an extra segment of the perceiver's internal language blocks before entering $M_t$. The features finally sent to the large LLM are already in a state of "text space neighbors."

### Key Designs

1.  **Perceiver as Encoder**:
    *   Function: Replaces the ViT with a complete small VLM, allowing visual features to undergo deep alignment within the perception module before being handed to the target large LLM.
    *   Mechanism: The hidden state of the last language block of the perceiver $M_p$ (a small VLM trained using Qwen3-0.6B + the same ViT in the paper) is taken as $\mathbf{H}_{\text{aligned}}$. Because this hidden state is processed by pre-trained language blocks with causal attention, it has already been pushed into a geometric space compatible with text embeddings. By using a projector $\phi$ to map the $M_p^{\text{LLM}}$ dimension (1024 for 0.6B) to the target LLM's input dimension (2048 for 4B, 5120 for 32B), it can be seamlessly integrated.
    *   Design Motivation: CLIP ViT only performs image-text contrastive learning, which is "image-text similarity" rather than "geometric isomorphism" shallow alignment. Language blocks are pre-trained on large-scale causal LM tasks, and their output geometry is naturally isomorphic to the target LLM's internal representation. Ablations show that removing the perceiver's language blocks and keeping only ViT weights results in an improvement of only +0.7 points, while the full perceiver brings +3.4 points, proving that the language blocks themselves are the "machine tool" for alignment.

2.  **Lightweight Adaptation + Standard Two-Stage Training**:
    *   Function: Reuses the classic LLaVA training pipeline, making the introduction of DPA a modular upgrade path—replacing only the visual encoder while keeping training objectives, data, and scheduling unchanged.
    *   Mechanism: Stage 1 uses 558K image-text captions to train only $\phi$ (projection layer), aligning the perceiver's output dimension to the target LLM. Stage 2 uses 1M high-quality visual instruction data to end-to-end finetune the entire DPA model (perceiver + projector + target LLM). The 32B model is trained with LoRA for 3 epochs to ensure manageable computational costs. The perceiver remains trainable during Stage 2; if the perceiver is frozen, performance drops from 53.0 to 52.1 but still outperforms the baseline.
    *   Design Motivation: The authors deliberately avoid introducing auxiliary losses or specialized training strategies to prove that the performance gain comes entirely from "architectural modification" rather than "training tricks." This "pluggable" upgrade allows DPA to be superimposed on any existing VLM pipeline.

3.  **Instruction-agnostic vs. Early Fusion (Instruction-agnostic Default)**:
    *   Function: Decides whether the perceiver sees text instructions during the encoding stage. The default is instruction-agnostic for versatility, with an optional early fusion variant (feeding instructions to the perceiver early) for seeking peak single-turn performance.
    *   Mechanism: In the default configuration, the perceiver only processes images and outputs instruction-agnostic visual features. An optional "w/ instruction context" variant appends the instructions during perceiver encoding, allowing the perceiver to filter out irrelevant visual information based on the query early on, acting as an early-fusion semantic filter. Ablations show that early fusion pulls the overall average score from 53.0 to 55.2 and the text capability score from 52.6 to 59.0. However, the authors argue that this configuration binds visual representations to a single-turn query, which fails during multi-turn dialogues or intent switching, so the default remains instruction-agnostic.
    *   Design Motivation: Versatility vs. single-turn peak performance is a real tension in practical VLM deployment. This paper provides a clear optional design knob, allowing users to choose based on the scenario. This ablation also reveals "why DPA can alleviate text forgetting"—the perceiver acts as a filter for interfering visual features, reducing the perturbation of the target LLM's language capabilities.

### Loss & Training
The LLaVA-NeXT two-stage recipe is strictly followed: Stage 1 has a learning rate of 1e-3, batch size 512, 2 epochs; Stage 2 has a learning rate of 1e-5, batch size 256, 2 epochs; 32B uses LoRA + 3 epochs. All stages use standard language modeling loss without introducing contrastive learning or alignment auxiliary losses.

## Key Experimental Results

### Main Results
DPA consistently beats the counter-baseline (reproduction of LLaVA-NeXT) across two scales (4B / 32B) and two LLM families (Qwen3 / LLaMA-3.2). The table below summarizes the average scores of three configurations across 11 benchmarks (Multi. Avg is the mean of 8 multimodal benchmarks, All Avg is the overall mean of 11):

| Configuration | General | Reasoning | Perception | Text | Multi. Avg | All Avg |
|---|---|---|---|---|---|---|
| LLaVA-NeXT-LLaMA-3.2-3B | 40.8 | 27.4 | 60.5 | 21.0 | 40.7 | 35.3 |
| DPA-LLaMA-3.2-3B | 44.8 | 29.8 | 64.3 | 25.1 | 44.1 | 38.9 |
| Δ | +4.0 | +2.4 | +3.8 | +4.1 | +3.4 | +3.6 |
| LLaVA-NeXT-Qwen3-4B | 51.1 | 40.1 | 68.3 | 45.1 | 51.2 | 49.6 |
| DPA-Qwen3-4B | 52.5 | 41.0 | 72.4 | 52.6 | 53.1 | 53.0 |
| Δ | +1.4 | +0.9 | +4.1 | +7.5 | +1.9 | +3.4 |
| LLaVA-NeXT-Qwen3-32B | 57.6 | 48.3 | 73.4 | 53.1 | 58.1 | 56.7 |
| DPA-Qwen3-32B | 60.9 | 50.1 | 77.9 | 58.1 | 61.1 | 60.3 |
| Δ | +3.3 | +1.8 | +4.5 | +5.0 | +3.0 | +3.6 |

Observed across scales: The multimodal average gain expanded from +1.9 for 4B to +3.0 for 32B, showing positive scalability; on text tasks, the MATH-500 single item for 4B rose from 36.4 to 54.2 (+17.8 points), and text capability forgetting was relatively reduced by 32.9% (4B) / 21.6% (32B).

### Ablation Study
Comparison of different perceiver designs under the 4B Qwen3 configuration reveals that "language blocks" and "language pre-training" are necessary conditions:

| Configuration | General | Reasoning | Perception | Text | Avg |
|---|---|---|---|---|---|
| LLaVA-NeXT-Qwen3-4B (baseline) | 51.1 | 40.1 | 68.3 | 45.1 | 49.6 |
| w/ large MLP (large MLP of same params instead of perceiver)| 26.2 | 29.7 | 29.2 | 49.8 | 34.1 |
| DPA-Qwen3-4B | 52.5 | 41.0 | 72.4 | 52.6 | 53.0 |
| w/o perceiver LM blocks (ViT only)| 51.7 | 40.3 | 69.5 | 46.1 | 50.3 |
| w/o perceiver LM pre-training (random init language blocks)| 30.4 | 32.6 | 32.2 | 57.5 | 38.7 |
| w/ instruction context (early fusion)| 53.8 | 41.8 | 71.8 | 59.0 | 55.2 |
| w/ perceiver frozen (Stage 2 frozen perceiver)| 51.7 | 39.9 | 67.4 | 54.4 | 52.1 |
| w/ untrained perceiver (perceiver entirely untrained)| 53.1 | 40.2 | 69.5 | 55.1 | 53.1 |

### Key Findings
- **Architecture itself is the core benefit**: An untrained perceiver (both projector and $\phi$ randomly initialized) can still perform +3.5 points higher than the baseline, indicating that DPA's gain mainly comes from the "deepening + language block structure" rather than transferring the capability of a strong perceiver. While the standalone evaluation score of the perceiver rose from 10.8 to 33.0, the final DPA model average score only fluctuated slightly between 3.4–4.5, with almost zero Pearson correlation.
- **Language block pre-training weights are essential**: Replacing language blocks with random initialization caused the overall score to plummet from 53.0 to 38.7; a "large MLP" with the same number of parameters could only achieve 34.1. This shows that the two conditions—"is LLM structure" and "is language pre-trained weights"—are both indispensable.
- **DPA mitigates catastrophic forgotten of text**: In the 4B configuration, the text average score rose from 45.1 to 52.6 (+7.5), with the MATH-500 single item up by +17.8 points; the 32B model also saw a +5.0 point increase. The authors used the MIR index to prove that DPA's visual features are closer to the text space geometry, and the modality gap in each layer of the target LLM remains consistently smaller than the baseline, substantively weakening "destructive adaptation."
- **Almost no inference cost**: The 32B configuration's throughput remains at 98% of the baseline (57.8 → 56.4 tokens/s), and training FLOPs only increased by 2%, because the perceiver only contributes cost during the pre-fill stage and does not participate in the generation stage.
- **Early fusion is an optional knob for performance-versatility**: Passing instructions to the perceiver can pull an additional 2.2 points (53.0 → 55.2), but the authors argue this binds visual features to single-turn queries and damages multi-turn dialogue capabilities, so the default configuration remains instruction-agnostic.

## Highlights & Insights
- **"Using a small VLM as a visual encoder" seems simple but hits the mark**: This idea is something everyone has likely thought of intuitively, but this paper actually performed a systematic ablation—proving that a small VLM is useful because it contains pre-trained language block structures within it, rather than because its visual capability is inherently strong (an untrained perceiver can still get +3.5). This redefines the module boundary from "high-semantic visual alignment" to "using language structures for alignment," representing a clean cognitive update.
- **Evidence of Quantifiable Geometric Isomorphism**: The authors used inter-layer similarity matrices to show that a "block-diagonal" subspace structure consistent with the text space appears in the DPA visual space, whereas the baseline visual space is blurred. Similarity at the structural level is more illustrative than simple distance and can serve as a new metric for "alignment quality" in the future.
- **Clarification of the relationship between Architecture and Data**: Previously, mitigating text forgetting in VLMs relied almost entirely on "adjusting the mix of multimodal vs. text data." This paper solves the same problem with pure architectural modification and clearly states it is orthogonal to data strategies. This provides a clear direction for future work—there is still a lot of optimization space at the architectural level yet to be explored.
- **Transferable design trick**: The routine of expanding a "lightweight projector" into a "perceiver with language blocks" can be generalized to other modalities (audio, video)—as long as a modality gap exists, a modality-internal LM block can be introduced for pre-alignment.

## Limitations & Future Work
- **Doubling of training costs still exists**: Although inference only increases by 2%, training FLOPs for 4B increase by 14% (1.27 → 1.45 × $10^{18}$); the larger the perceiver, the higher the cost. The paper did not deeply analyze the threshold for the "minimum usable perceiver scale."
- **Lack of quantitative verification for multi-turn failure in early fusion**: The authors claim early fusion harms multi-turn capability but provide no specific data from multi-turn benchmarks (e.g., Multi-turn VQA, dialogue benchmarks) to support this assertion.
- **MIR geometric analysis has limited interpretability**: MIR requires two space dimensions to be consistent, so the analyses were done between the perceiver and Qwen3-0.6B, without directly showing the spatial relationship between the perceiver and the 32B large LLM.
- **No head-to-head comparison with "data mix tuning" baselines**: DPA claims to be orthogonal to data strategies, but there is a lack of joint experiments for "DPA + optimized data mix" and "baseline + optimized data mix," making it impossible to quantify if there is a synergistic gain from combining both.
- **Task coverage leans toward understanding and reasoning**: Grounding, segmentation, and VLM agent tasks are not covered; it is unclear if DPA still provides the same benefits in dense prediction scenarios.

## Related Work & Insights
- **vs. Data Mix Tuning (DeepSeek-VL / InternVL)**: They rely on dynamically adjusting text vs. multimodal data weights to mitigate forgetting; this paper fundamentally reduces the "destructive adaptation" of the LLM via architectural changes, and the two methods are orthogonal and superimposable.
- **vs. Multi-encoder Fusion (Cambrian / Eagle)**: Those works parallel multiple ViTs like DINO + SAM + CLIP to pursue more comprehensive visual representations; this paper takes the opposite approach—using only one perceiver but deepening its alignment depth. The philosophies are opposite, but both point to the fact that "visual encoders need to evolve."
- **vs. LLaVA / Qwen-VL series**: DPA upgrades the LLaVA paradigm's projector from a "1-layer MLP" to a "small VLM," extending the architecture dimension within the same family and remaining fully compatible with existing pipelines engineering-wise.
- **vs. Introducing alignment auxiliary losses during training**: MaskCLIP / SEA add auxiliary supervision to ViT; this paper does not touch training objectives and relies entirely on architecture, avoiding the complexity and potential conflicts of loss design and making it easier to reuse.

## Rating
- Novelty: ⭐⭐⭐⭐ The idea of "using a small VLM as a visual encoder" is not entirely original, but clarifying that "language block structure + language pre-training weights are the key" through systematic ablation, coupled with geometric isomorphism analysis, provides fresh overall insight.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers two scales (4B / 32B), two families (Qwen3 / LLaMA), and 11 benchmarks; key variables such as perceiver scale, whether to freeze, early fusion, and pre-training are all ablated; MIR geometric analysis provides mechanistic evidence.
- Writing Quality: ⭐⭐⭐⭐ The structure is clear, with 5 RQs stringing the analysis into a complete line; charts are self-consistent; however, some formula typesetting is messy, and some charts (e.g., geometric isomorphism) require more detailed interpretation instructions.
- Value: ⭐⭐⭐⭐⭐ Provides clear evidence that "VLM visual encoders still have room for structural upgrades," with direct reference value for the industry to plug-in and upgrade old VLMs and for academia to explore "language blocks as alignment modules."

## Rating
- Novelty: TBD
- Experimental Thoroughness: TBD
- Writing Quality: TBD
- Value: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Gated Relational Alignment via Confidence-based Distillation for Efficient VLMs](gated_relational_alignment_via_confidence-based_distillation_for_efficient_vlms.md)
- [\[ICML 2026\] Injecting Distributional Awareness into MLLMs via Reinforcement Learning for Deep Imbalanced Regression](injecting_distributional_awareness_into_mllms_via_reinforcement_learning_for_dee.md)
- [\[ICML 2026\] V-LynX: Token Interface Alignment for VideoX LLMs](v-lynx_token_interface_alignment_for_videox_llms.md)
- [\[ICML 2026\] Density-Aware Translation of Spurious Correlations in Zero-Shot VLMs](density-aware_translation_of_spurious_correlations_in_zero-shot_vlms.md)
- [\[ACL 2026\] CogGen: A Cognitively Inspired Recursive Framework for Deep Research Report Generation](../../ACL2026/multimodal_vlm/coggen_a_cognitively_inspired_recursive_framework_for_deep_research_report_gener.md)

</div>

<!-- RELATED:END -->
