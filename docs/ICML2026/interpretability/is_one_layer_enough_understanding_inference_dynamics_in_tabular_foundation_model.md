---
title: >-
  [Paper Note] Is One Layer Enough? Understanding Inference Dynamics in Tabular Foundation Models
description: >-
  [ICML 2026][Interpretability][TabPFN] The authors conduct the first large-scale hierarchical mechanistic analysis of six mainstream tabular foundation models (TFMs)…
tags:
  - "ICML 2026"
  - "Interpretability"
  - "TabPFN"
  - "Tabular Foundation Model"
  - "Mechanistic Interpretability"
  - "Recurrent Transformer"
  - "Inter-layer Dynamics"
date: 2026-05-08
content_hash: 4bd838685bc07ce9
---

# Is One Layer Enough? Understanding Inference Dynamics in Tabular Foundation Models

**Conference**: ICML 2026  
**arXiv**: [2605.06510](https://arxiv.org/abs/2605.06510)  
**Code**: https://github.com/amirbalef/is_one_layer_enough  
**Area**: Interpretability / Tabular Foundation Models / Model Compression  
**Keywords**: TabPFN, Tabular Foundation Model, Mechanistic Interpretability, Recurrent Transformer, Inter-layer Dynamics

## TL;DR
The authors conduct the first large-scale hierarchical mechanistic analysis of six mainstream tabular foundation models (TFMs), discovering that the middle and later layers mainly perform "iterative refinement" and contain substantial redundancy. Based on this, they design a single-layer recurrent TFM using only 20% of the parameters, achieving performance nearly matching the original six-layer version.

## Background & Motivation
**Background**: Transformer-based tabular foundation models such as TabPFN, TabICL, and LimiX have surpassed traditional GBDT pipelines on small- and medium-scale tabular prediction tasks. However, their internal mechanisms—specifically, how they perform Bayesian inference via in-context learning—remain largely a black box.

**Limitations of Prior Work**: Directly applying the "logit lens" method from LLMs to analyze TFM layer representations yields fragile results (as shown in Figure 1, the original decoder almost completely fails at shallow layers). Moreover, TFMs are encoder-only, non-autoregressive, and row-invariant, making them structurally distinct from LLMs. It is unclear whether interpretability findings from LLMs (e.g., early layers for detokenization, middle layers for abstraction, late layers for sharpening) transfer to TFMs.

**Key Challenge**: On one hand, TFMs are smaller and cheaper to run than LLMs, making them ideal for large-scale mechanistic studies. On the other hand, there is a lack of suitable analysis toolchains, and the encoder designs of different TFMs vary greatly, limiting the generalizability of single-point studies.

**Goal**: (1) Design a hierarchical analysis protocol tailored to TFMs; (2) Answer "at which layer and how does inference emerge" and "what are the similarities and differences compared to LLMs"; (3) Use these findings to guide more efficient architecture design.

**Key Insight**: The authors note that TFM tasks are fixed classification/regression problems, allowing direct training of "per-layer decoders" (i.e., "tabular tuned lens") without relying on vocabulary projection as in LLMs. They also adapt three mature intervention experiments from LLM mechanistic studies: skip, repeat, and swap.

**Core Idea**: By combining six experiments—embedding similarity, class separation gap, probing classifier, tabular logit lens, layer ablation, and self-repair—the authors characterize the hierarchical dynamics of TFMs. The finding that "middle and later layers mainly perform iterative refinement" supports the design of more efficient architectures that replace multi-layer stacks with a single recurrent layer.

## Method

### Overall Architecture
The study consists of two parts: an analysis protocol and a proof-of-concept model. The protocol fixes six open-source/open-weight TFMs (TabPFN v1/v2/2.5, TabICL, LimiX-2M/16M), running six mechanistic experiments on PMLBmini (34 tasks) and TabArena (15 binary classification tasks). Each experiment targets a different granularity, from representation similarity to hierarchical intervention to self-repair. The proof-of-concept uses the open-source nanoTabPFN, training three variants: the original six-layer, a single-layer, and a single-layer recurrent model (nanoTabPFNlooped), all pretrained with the same TabICL prior for comparison.

### Key Designs

1. **Tabular Tuned Lens**:

    - **Function**: Maps each layer's hidden state back to task output probabilities, measuring whether the layer has formed usable discriminative representations.
    - **Mechanism**: The authors find that using the original decoder ("logit lens") fails at shallow layers (solid ROC-AUC in Figure 1 approaches 0.5 in early layers). Following Belrose's tuned lens approach, they pretrain a separate decoder for each layer: the backbone is frozen, and a decoder is trained for each layer using TabICL's synthetic prior for 200 epochs.
    - **Design Motivation**: TFMs are encoder-only and row-invariant, lacking a vocabulary detokenizer. Per-layer decoders are the most natural adaptation for ICL tasks, directly revealing "if inference stops at this layer, how well can a lightweight classifier perform," thus assessing the feasibility of early exiting.

2. **Three Types of Layerwise Interventions (skip / repeat / swap) + Self-repair Analysis**:

    - **Function**: Uses structural ablation to evaluate "what each layer does" and "whether subsequent layers can compensate if a layer is removed."
    - **Mechanism**: Skipping layer $l$ assesses its uniqueness; repeating layer $l$ tests if it performs iterative refinement; swapping adjacent layers checks if the representation sequence is truly aligned. By overlaying the tuned lens on skip experiments, the authors observe whether subsequent layers can restore performance—if so, this indicates self-repair/layer redundancy.
    - **Design Motivation**: Simply observing "final output after ablation" conflates redundancy and self-repair. Overlaying the lens distinguishes: early layers, once removed, cannot be compensated (Figure 8), indicating unique critical functions; skipping middle/later layers sees immediate recovery in the next layer's lens performance, indicating redundancy and self-repair.

3. **Single-layer Recurrent nanoTabPFNlooped Proof-of-concept**:

    - **Function**: Operationalizes the analysis conclusion—if middle/later layers only perform iterative refinement, then repeating a single layer N times should theoretically match an N-layer stack.
    - **Mechanism**: Based on the nanoTabPFN architecture (similar to TabPFN v2 but lighter), three models are trained: six-layer stack, single-layer, and single-layer looped six times. Their parameter counts are 3.72M / 0.75M / 0.75M, but the looped version's "forward computation" matches the six-layer stack. All are trained from scratch with 10,000 steps and batch size 512.
    - **Design Motivation**: Directly modifying SOTA TFMs is costly, but nanoTabPFN is a public reproducible version, enabling controlled comparison to confirm that performance differences stem from "looping vs stacking" rather than parameter count.

### Loss & Training
The nanoTabPFN series uses the standard TabPFN training objective. The TabICL prior generator is configured with batch=4×10,000 batches, feature count 2–30, up to 10 classes, and sequence length 1024. The optimizer is AdamW, $\eta=10^{-4}$, cosine warmup for 2000 steps, and weight decay=0. The single-layer, six-layer, and looped models require 11.9h / 62.3h / 68.8h respectively (single A100 GPU). Per-layer decoder fine-tuning uses 200 epochs, batch=8, $\eta=3\times10^{-5}$.

## Key Experimental Results

### Main Results
Table 1 compares the three nanoTabPFN variants on PMLBmini and TabArena:

| Model | Parameters | Computation | PMLBmini Performance | Gap vs 6-layer |
|-------|------------|-------------|----------------------|----------------|
| nanoTabPFN-1l | 0.75M | 1× | Significantly worse | Substantially behind |
| nanoTabPFN-6l | 3.72M | 6× | Baseline | — |
| nanoTabPFN-looped | 0.75M | 6× | Close to 6l | Nearly matches |
| TabPFN(2.5) | 10.7M | 24 layers | Upper bound | Still better than looped |

Key conclusion: Performance differences are mainly due to "whether six refinements are performed," not "whether there are six independent parameter sets."

### Ablation Study
Six mechanistic experiments profile the hierarchical behavior:

| Experiment | Main Finding | Interpretation |
|------------|-------------|----------------|
| Embedding similarity (cos / CKA) | Large models (TabPFN 2.5, LimiX-16M) form clear "layer blocks" | Within-block representations only make small incremental updates |
| Class separation gap | Monotonically increases with depth; label embedding rises later than feature | Model separates features first, then forms labels |
| Probing classifier | Probe at layer $i$ generalizes well to $j>i$, but not vice versa | Later layers retain earlier information and add new features |
| Tabular tuned lens | Most models achieve high AUC at early layers | Inference decisions are actually formed "very early" |
| Layer ablation (skip) | Removing layer 1 collapses performance; removing middle/later layers has little effect | Early layers = specialized mapping; middle/later layers = redundancy |
| Self-repair | After skipping middle/later layers, next layer's lens performance immediately rebounds | Hydra effect-style self-repair exists |

### Key Findings
- **Early layers are irreplaceable "mapping layers"**: TabICL and LimiX-2M, due to strong encoders (row interaction compression / RBF kernel preprocessing), are less sensitive to the first few transformer blocks; other models collapse if layer 1 is removed. This suggests early layers mainly project raw tokens into a space suitable for residual stream operations.
- **Middle/later layers: redundancy + self-repair**: In TabPFN(v2), lens performance "jumps" around layer 5, with substantial overlapping computation between adjacent layers—this is the physical basis for the viability of recurrent architectures.
- **Key differences between TFM and LLM**: TFMs are much more sensitive to layer swap than LLMs (especially TabPFN v2), and even if the last layer is disrupted, output is barely affected—contrasting with LLMs where the last layer's sharpening is essential. TFM's "prediction calibration" phase is later and more implicit.
- **Strong encoders are a free lunch**: Models with explicit feature encoding (row interaction / RBF kernel) are less sensitive to depth, suggesting a design direction of "wide encoder + shallow looped backbone."

## Highlights & Insights
- "Tabular tuned lens" is the key tool for cleanly transferring the LLM logit lens to ICL tabular tasks: original decoder failure is not due to poor representations, but to misalignment between representation and decoder; per-layer decoders reveal that the model "already knows the answer" early on.
- The combination of three interventions + lens overlay is ingenious: looking at skip alone conflates "layer is useless" and "layer is self-repaired"; overlaying the lens distinguishes the two. This analysis paradigm can be directly transferred to all ICL models.
- The single-layer recurrent validation truly "monetizes" interpretability research: mechanistic studies usually provide observations but not solutions; here, nanoTabPFNlooped directly translates the finding that "middle/later layers perform iterative refinement" into an architecture saving 80% of parameters.
- Reveals fundamental differences between TFM and LLM in sensitivity to layer swap and the importance of the last layer, providing empirical evidence that TFM should not blindly reuse LLM experience.

## Limitations & Future Work
- Experiments are mainly on binary classification; multiclass and regression are only partially validated in the appendix. Transferability to long priors and complex high-cardinality tasks is unknown.
- The tabular tuned lens prior uses the open-source TabICL version; for models like LimiX trained with more sophisticated priors, early layer quality may be underestimated.
- nanoTabPFNlooped is only validated at small scale (6 layers, single-layer looped); whether it can scale to TabPFN(2.5) with 24 layers and 50,000 samples remains untested.
- Evaluation does not use ensemble methods; conclusions may be diluted in TFM's common "repeated sampling ensemble" scenarios.
- Future directions: push analysis down to neuron/circuit level; study how prior design shapes hierarchical dynamics; apply the same tools to LLM-based tabular models (e.g., TabLLM) for cross-comparison.

## Related Work & Insights
- **vs Lad et al. (Remarkable robustness of LLMs)**: They propose four-stage LLM inference (detokenize → feature refinement → ensembling → sharpening); this work shows TFMs have similar but differently distributed stages, with lower importance for the last layer.
- **vs Belrose et al. (Tuned Lens)**: This work concretizes "tuned lens" as per-layer decoder + tabular prior fine-tuning, circumventing the lack of vocabulary in TFMs.
- **vs Looped Transformer (Universal Transformer, Dehghani 2019; Gong 2025)**: First to transfer the "recurrent refinement" concept to tabular ICL models, and mechanistically demonstrates "why looping works."
- **vs TabPFN series and LimiX**: This work is not a new architecture competitor, but provides the whole TFM family with interpretability and compression recipes, serving as a complement.

## Rating
- Novelty: ⭐⭐⭐⭐ — First large-scale hierarchical mechanistic study of TFMs, with findings directly translated into concrete architectural modifications.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 6 models × 6 experiments × 2 benchmarks, with appendix covering multiclass/regression; evidence chain is very complete.
- Writing Quality: ⭐⭐⭐⭐ — Each experiment features a blue "takeaway" box, with clear logic from observation to conclusion, though some figures (e.g., self-repair) require careful reading.
- Value: ⭐⭐⭐⭐ — Provides the TFM community with an analysis template and a "do 6 layers with 1" compression approach, directly relevant for industrial deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] On the Effect of Uncertainty on Layer-wise Inference Dynamics](../../ICML2025/interpretability/on_the_effect_of_uncertainty_on_layer-wise_inference_dynamics.md)
- [\[ICLR 2026\] Layer by layer, module by module: Choose both for optimal OOD probing of ViT](../../ICLR2026/interpretability/layer_by_layer_module_by_module_choose_both_for_optimal_ood_probing_of_vit.md)
- [\[ACL 2026\] TabReX: Tabular Referenceless eXplainable Evaluation](../../ACL2026/interpretability/tabrex_tabular_referenceless_explainable_evaluation.md)
- [\[CVPR 2026\] DINO-QPM: Adapting Visual Foundation Models for Globally Interpretable Image Classification](../../CVPR2026/interpretability/dino-qpm_adapting_visual_foundation_models_for_globally_interpretable_image_clas.md)
- [\[AAAI 2026\] SOM Directions are Better than One: Multi-Directional Refusal Suppression in Language Models](../../AAAI2026/interpretability/som_directions_are_better_than_one_multi-directional_refusal_suppression_in_lang.md)

</div>

<!-- RELATED:END -->
