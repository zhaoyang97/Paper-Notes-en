---
title: >-
  [Paper Note] One Protein Is All You Need
description: >-
  [ICLR 2026][Computational Biology][Protein Language Models] This paper proposes ProteinTTT, which applies Test-Time Training (TTT) to protein language models. Given a target protein sequence, the backbone network undergoes several dozen steps of self-supervised fine-tuning using a masked language modeling objective on that single sequence before inference. This process reduces the model's perplexity and improves representations, thereby enhancing structure, fitness…
tags:
  - "ICLR 2026"
  - "Computational Biology"
  - "Protein Language Models"
  - "Test-Time Training"
  - "Self-supervised Customization"
  - "Structure Prediction"
  - "Perplexity Minimization"
date: 2026-05-08
content_hash: b7bf662f4bfab5ff
---

# One Protein Is All You Need

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=5zAde2jch7](https://openreview.net/forum?id=5zAde2jch7)  
**Code**: https://github.com/anton-bushuiev/ProteinTTT  
**Area**: Computational Biology / Protein Language Models / Test-Time Training  
**Keywords**: Protein Language Models, Test-Time Training, Self-supervised Customization, Structure Prediction, Perplexity Minimization

## TL;DR
This paper proposes ProteinTTT, which applies Test-Time Training (TTT) to protein language models. Given a target protein sequence, the backbone network undergoes several dozen steps of self-supervised fine-tuning using a masked language modeling objective on that single sequence before inference. This process reduces the model's perplexity and improves representations, thereby enhancing structure, fitness, and functional predictions—without modifying any downstream task heads—and achieving new SOTA results on ProteinGym.

## Background & Motivation
**Background**: The dominant paradigm in protein machine learning involves self-supervised pre-training of a large backbone (e.g., ESM2, ESM3, SaProt) followed by attaching task-specific heads (ESMFold structure head, fitness scoring head, functional classifiers). These models are optimized for the best average performance across an entire dataset.

**Limitations of Prior Work**: Biologists are often concerned with a specific protein of interest—such as one related to metabolic diseases, an oncogenic signaling protein, or a frequently mutating viral protein. These targets often suffer from data scarcity or distribution shifts from the training set, causing general-purpose models to perform poorly on them. A straightforward example in the paper is CASP14 target T1074, where ESMFold achieves a TM-score of only 0.63 because the underlying ESM2 is "highly confused" by the sequence (perplexity as high as 13.0).

**Key Challenge**: There is a tension between "performing well on all proteins" and "achieving excellence on a single protein." Fitting accuracy for individual targets is often sacrificed for generalizations. However, experimental biology requires the latter, as high-cost wet-lab experiments must be guided by accurate single-protein predictions.

**Goal**: To enable models to be "customized on-the-fly" for a single target protein at inference time, refining the backbone to better understand that specific sequence without assuming any additional data (no homologous sequences, labels, or validation sets).

**Key Insight**: The authors leverage a simple yet powerful premise: if a language model is less "confused" by a sequence (lower perplexity), it understands the unique patterns of that sequence better, leading to improved representations for downstream predictions. Masked Language Modeling (MLM), being the dominant pre-training paradigm, can be directly used to minimize perplexity on a single sequence.

**Core Idea**: Utilizing the TTT framework, the backbone is fine-tuned on only the target sequence using the same masked self-supervised objective as pre-training. Reducing perplexity serves as a free self-supervised signal to obtain superior downstream representations.

## Method

### Overall Architecture
ProteinTTT is built upon the **Y-shaped architecture** commonly used in protein machine learning: a shared backbone feature extractor $f$ operates on protein tokens $x$, supporting two branches—a self-supervised pre-training head $g$ (MLM head mapping representations back to amino acid types) and a downstream fine-tuning head $h$ (structure/fitness/functional head). In standard workflows, $h \circ f$ is frozen after pre-training and remains unchanged during testing.

ProteinTTT inserts a "customization" step: upon receiving a target sequence $x$, the model utilizes the idle self-supervised branch $g \circ f$ to perform $T$ steps of masked fine-tuning. This adjusts the backbone parameters from the pre-trained $\theta_0$ to a specialized $\theta_x$, while the downstream head $h$ remains frozen:

$$\text{ProteinTTT}: (h \circ f(\cdot; \theta_0),\, x) \mapsto h \circ f(\cdot; \theta_x)$$

Final predictions are generated using $h \circ f(\cdot; \theta_x)$. Crucially, customization only modifies the representations of $f$ and does not touch the parameters of $h$ (e.g., leaving the 1.4 billion parameters of the ESMFold structure head intact), improving accuracy through "better representations." Parameters are reset to $\theta_0$ after processing each protein.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["Input: Single Target Sequence x"] --> B["Single-Sequence Masked Customization: Minimize MLM Perplexity on x"]
    B --> C["Stabilized Training: LoRA + SGD + Gradient Accumulation"]
    C --> D["Confidence-based Step Selection: Choose optimal θx from T steps based on c"]
    D -->|Replace backbone with f(·;θx), freeze downstream head h| E["Output: Structure / Fitness / Functional Prediction"]
```

### Key Designs

**1. Single-sequence Masked Customization: Pre-training Objectives as Test-time Signals**

The challenge is how to fine-tune when no extra data exists for a single protein. The authors' answer is that the sequence itself is the supervision signal. The customization objective minimizes the MLM loss on target sequence $x$:

$$L(x; \theta) = \mathbb{E}_{M \sim p_{\text{mask}}(M)}\left[\sum_{i \in M} -\log p(x_i \mid x_{\backslash M}; \theta)\right]$$

This involves randomly masking positions $M$ and having the model reconstruct the original amino acids $x_i$ from the remaining context $x_{\backslash M}$, where $\log p(x_i\mid x_{\backslash M};\theta) = g(f(x_{\backslash M};\theta))_i$. This step is equivalent to minimizing perplexity—the better the model completes the sequence, the more it "understands" its unique patterns. To avoid distribution shifts, the process replicates all pre-training masking habits: same mask ratio (e.g., 15%) or mask count distribution (e.g., Beta distribution), 10% random tokens, 10% kept as-is, and cropping long sequences to 1024-token fragments. Since MLM is ubiquitous in protein pre-training, this objective is plug-and-play for almost any Y-shaped model; the authors also demonstrate its applicability to autoregressive and discrete diffusion models (e.g., DPLM2) in the appendix.

**2. Confidence-guided Step Selection: Deciding "When to Stop" without a Validation Set**

Since customization involves only one protein, standard early stopping on a validation set is impossible, and over-tuning may harm representations. The authors run $T$ fixed steps to obtain parameter snapshots $\Theta = \{\theta_0, \theta_1, \dots, \theta_T\}$, then use a confidence function $c$ to select the best one:

$$\theta_x = \arg\max_{\theta \in \Theta}\, c\big(h(f(x;\theta))\big)$$

For structure prediction, $c$ is naturally pLDDT (the model's internal confidence score). Figure 3 shows target 7EBL chain B, where TM-score climbs from 0.29 to 0.92 as perplexity drops, with the highest pLDDT at step 7 correctly selected. If a model lacks a built-in confidence head (like DPLM2), it defaults to the final step $\theta_x = \theta_T$. Using pLDDT as $c$ makes ProteinTTT robust to hyperparameters, allowing $T$ to be fixed (e.g., $T=30$) while only tuning learning rate and batch size.

**3. Stabilized Large Model Customization: Efficient and Robust Fine-tuning**

Full-parameter fine-tuning of a 3-billion-parameter ESM2 is expensive and unstable. The authors use three techniques for lightweight customization: **LoRA** (Low-Rank Adaptation) to update only a few parameters; **Gradient Accumulation** to increase effective batch size under memory constraints; and **SGD instead of Adam** (following TTT experiences from Gandelsman et al.), as SGD is more stable and predictable for single-sample, few-step customization compared to adaptive learning rates. These allow ProteinTTT to be applied to models like ESM2-3B and ESM3 while maintaining ESMFold-level inference efficiency, which is an order of magnitude faster than AlphaFold2.

### Loss & Training
The customization objective is the MLM loss $L(x;\theta)$ described above, optimized only on the single target sequence (or its MSA, see Appendix C). The optimizer is SGD + LoRA + Gradient Accumulation, running for $T$ steps (e.g., 30). The best snapshot $\theta_x$ is selected by the confidence function $c$ (pLDDT for structure tasks). The downstream head $h$ is frozen throughout, and backbone parameters are reset to $\theta_0$ after each protein is processed.

## Key Experimental Results

### Main Results
Structure prediction was evaluated on 18 difficult targets from CAMEO where ESMFold had low confidence, averaged over 5 random seeds. ProteinTTT consistently outperformed strong baselines (ESMFold's Masked Prediction MP, ESM3's Chain-of-Thought CoT) across four different backbones.

| Method | TM-score ↑ | LDDT ↑ |
|------|-----------|--------|
| ESM3 | 0.3480 | 0.3723 |
| ESM3 + CoT | 0.3677 | 0.3835 |
| **ESM3 + ProteinTTT** | **0.3954** | **0.4214** |
| ESMFold | 0.4649 | 0.5194 |
| ESMFold + MP | 0.4862 | 0.5375 |
| **ESMFold + ProteinTTT** | **0.5047** | **0.5478** |
| HelixFold-Single | 0.4709 | 0.4758 |
| **HelixFold-Single + ProteinTTT** | **0.4839** | **0.4840** |

For fitness prediction in the ProteinGym zero-shot setting (186 proteins, 2.5 million mutations), ProteinTTT improved the Spearman correlation for all models, with **ProSST + ProteinTTT setting a new SOTA** (significant at $p<0.05$ via paired t-test).

| Model | Avg. Spearman ↑ (Baseline → +ProteinTTT) |
|------|----------------------------------------|
| ESM2 (35M) | 0.3211 → 0.3407 |
| ProGen2-small (151M) | 0.3255 → 0.3591 |
| SaProt (650M) | 0.4569 → 0.4583 |
| ProSST (K=2048) | 0.5068 → **0.5087** (New SOTA) |

### Ablation Study
The paper uses cross-comparisons between backbones and scales as ablations, with key conclusions summarized:

| Configuration | Key Observation | Description |
|------|---------|------|
| ProteinTTT vs MP / CoT | Consistently higher TM-score | Customization is superior to "multiple mask sampling" and "CoT iterative decoding." |
| DPLM2 (no pLDDT head) | Performance still improves | Masked customization remains effective even without a confidence function by taking the final step $\theta_T$. |
| Small vs Large Models | Larger gain for small models (35M) | Large models are near saturation on benchmarks, resulting in compressed improvement margins. |
| Low-MSA proteins | Most noticeable gains | Single-sequence customization provides the greatest boost for proteins with few homologous sequences. |

### Key Findings
- The gains of ProteinTTT stem from the causal chain: "perplexity reduction ↔ improved representation ↔ more accurate downstream results." Figure 3 provides direct evidence as TM-score rises monotonically as perplexity falls.
- Gains are strongly correlated with data scarcity: proteins with shallower MSAs or fewer similar sequences in the training set benefit most. This directly addresses the area where general models are weakest and biologists' needs are greatest.
- For difficult targets, the number of improved proteins far outweighs those harmed. ESMFold/DPLM2/HelixFold/ESM3 significantly improved 7/4/5/6 structures respectively, while only slightly degrading 2/1/1/1, yielding a positive net benefit.

## Highlights & Insights
- The TTT paradigm from Computer Vision is cleanly adapted to proteins. The observation that "perplexity is free supervision" allows the method to refine predictions for a single sequence without extra data—a simple but effective approach to a real-world pain point.
- The "freeze head, tune representation" strategy is a clever decoupling. The 1.4-billion-parameter ESMFold structure head remains untouched; better input representations alone drive structure improvements, meaning any Y-shaped model can integrate this with a few lines of code.
- Using pLDDT for step selection naturally resolves the TTT challenge of "early stopping without a validation set" for structure tasks, while providing robustness to hyperparameters.
- The method is agnostic to pre-training paradigms: bidirectional masks, autoregression, and discrete diffusion can all utilize the same masked customization objective, making it transferable to more protein foundation models and other "Y-shaped + self-supervised" scientific modeling tasks.

## Limitations & Future Work
- Step selection relies on a confidence function $c$: while structure tasks have pLDDT, fitness or functional tasks often lack reliable confidence scores, forcing a fallback to the final step $\theta_T$, which might miss the optimal snapshot.
- Customizing a single protein involves $T$ steps of fine-tuning, adding overhead compared to a single forward pass. While still faster than AlphaFold2, the cost of batch customization for large databases (e.g., BFVD with 350,000 entries) is non-negligible.
- Gains are minimal (or slightly negative) for near-saturated large models or proteins with high MSA depth; the value is highly concentrated on "data-scarce, distribution-shifted" difficult samples.
- The relationship between metrics like perplexity, pLDDT, and TM-score is empirically observed; theoretical guarantees remain relatively weak (Appendix A provides an intuitive explanation linked to perplexity minimization).

## Related Work & Insights
- **vs Test-Time Training (TTT, Sun et al. / Gandelsman et al.)**: Classic TTT was created to mitigate distribution shifts in CV. This paper adapts it to protein PLMs and finds that even without explicit distribution shifts, "customizing to a single sample" improves benchmarks, broadening the motivation for TTT.
- **vs Protein/Family-specific fine-tuning (Notin et al. / Samusevich et al.)**: These methods require collecting target-related external data (homologs, family data), whereas ProteinTTT uses only the target sequence itself, making it more broadly applicable.
- **vs ESMFold Masked Prediction / ESM3 CoT**: MP relies on multiple masked samplings without model updates; CoT relies on iterative decoding. ProteinTTT updates the backbone to truly "understand" the sequence and consistently outperforms both in main experiments.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The first "single-protein test-time customization" method for protein ML, elegantly using perplexity minimization as free supervision.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers structure, fitness, and function across multiple backbones and scales; achieves SOTA on ProteinGym and includes real-world cases like antibody loops and viral libraries.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear progression from Y-shaped architecture to customization objectives and selection strategies. Figure 1/3 comparing perplexity and accuracy is highly convincing.
- Value: ⭐⭐⭐⭐⭐ Plug-and-play with a few lines of code; directly addresses the data-scarcity reality in biology, offering high value for guiding wet-lab experiments.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Triangle Multiplication is All You Need for Biomolecular Structure Representations](triangle_multiplication_is_all_you_need_for_biomolecular_structure_representatio.md)
- [\[ICLR 2026\] Extending Sequence Length is Not All You Need: Effective Integration of Multimodal Signals for Gene Expression Prediction](extending_sequence_length_is_not_all_you_need_effective_integration_of_multimoda.md)
- [\[NeurIPS 2025\] Is Sequence Information All You Need for Bayesian Optimization of Antibodies?](../../NeurIPS2025/computational_biology/is_sequence_information_all_you_need_for_bayesian_optimization_of_antibodies.md)
- [\[ICLR 2026\] SimpleFold: Folding Proteins is Simpler Than You Think](simplefold_folding_proteins_is_simpler_than_you_think.md)
- [\[ICLR 2026\] Towards Understanding the Shape of Representations in Protein Language Models](towards_understanding_the_shape_of_representations_in_protein_language_models.md)

</div>

<!-- RELATED:END -->
