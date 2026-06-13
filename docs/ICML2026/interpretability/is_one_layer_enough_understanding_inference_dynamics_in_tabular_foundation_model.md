---
title: >-
  [Paper Note] Is One Layer Enough? Understanding Inference Dynamics in Tabular Foundation Models
description: >-
  [ICML 2026][Interpretability][TabPFN] The authors perform the first large-scale layer-wise mechanistic analysis of six mainstream Tabular Foundation Models (TFMs). They discover that middle and late layers primarily perf…
tags:
  - "ICML 2026"
  - "Interpretability"
  - "TabPFN"
  - "Tabular Foundation Models"
  - "Mechanistic Interpretability"
  - "Recurrent Transformer"
  - "Inter-layer Dynamics"
date: 2026-05-08
content_hash: 066a2314bef3cbef
---

# Is One Layer Enough? Understanding Inference Dynamics in Tabular Foundation Models

**Conference**: ICML 2026  
**arXiv**: [2605.06510](https://arxiv.org/abs/2605.06510)  
**Code**: https://github.com/amirbalef/is_one_layer_enough  
**Area**: Interpretability / Tabular Foundation Models / Model Compression  
**Keywords**: TabPFN, Tabular Foundation Models, Mechanistic Interpretability, Recurrent Transformer, Inter-layer Dynamics

## TL;DR
The authors perform the first large-scale layer-wise mechanistic analysis of six mainstream Tabular Foundation Models (TFMs). They discover that middle and late layers primarily perform "iterative refinement" with significant redundancy. Based on these insights, they design a single-layer recurrent TFM using only 20% of the parameters, which nearly matches the performance of the original six-layer version.

## Background & Motivation
**Background**: Transformer-based Tabular Foundation Models (TFMs) such as TabPFN, TabICL, and LimiX have outperformed traditional GBDT pipelines in small-to-medium scale tabular prediction. However, how they "perform Bayesian inference via in-context learning" remains a black box.

**Limitations of Prior Work**: Directly applying the "logit lens" method from LLMs to analyze TFM layer representations is fragile (Figure 1 shows the original decoder fails almost completely in early layers). Furthermore, TFMs are encoder-only, non-autoregressive, and row-invariant, differing significantly from LLM architectures. Whether existing LLM interpretability findings (early-layer detokenization, middle-layer abstraction, late-layer sharpening) apply to TFMs remains unknown.

**Key Challenge**: On one hand, TFMs are smaller and cheaper to infer than LLMs, making them ideal for large-scale mechanistic research. On the other hand, there is a lack of specialized analysis toolchains, and the diverse encoder designs of various TFMs make single-point studies difficult to generalize.

**Goal**: (1) Design a layer-wise analysis protocol tailored for TFMs; (2) Determine where and how inference is formed compared to LLMs; (3) Use these findings to guide more efficient architecture design.

**Key Insight**: Since TFM tasks are fixed classification or regression problems, one can train "per-layer decoders" (i.e., "tabular tuned lens") instead of relying on vocabulary projections used in LLMs. Mature intervention experiments from LLM mechanistic research, such as skip, repeat, and swap, can also be repurposed.

**Core Idea**: Characterize TFM inter-layer dynamics through six joint experiments: embedding similarity, class separation intervals, probing classifiers, tabular logit lens, layer ablation, and self-repair. The finding that "middle and late layers primarily perform iterative refinement" justifies an efficient architecture replacing multiple layers with a single recurrent layer.

## Method

### Overall Architecture
The research consists of two parts: an analysis protocol and a proof-of-concept model. The analysis protocol is applied to six open-source/open-weight TFMs (TabPFN v1/v2/2.5, TabICL, LimiX-2M/16M), running six mechanistic experiments on PMLBmini (34 tasks) and TabArena (15 binary tasks). Each experiment focuses on questions at different granularities, from representation space similarity to layer-wise interventions and self-repair. The proof-of-concept part is based on nanoTabPFN, comparing the original 6-layer version, a 1-layer version, and a 1-layer version looped 6 times (nanoTabPFNlooped) using the same TabICL prior for pre-training.

### Key Designs

1.  **Tabular Tuned Lens**:
    - **Function**: Maps the hidden states of each layer back to task output probabilities to measure whether "useful discriminative representations have formed in that layer."
    - **Mechanism**: The authors found that using the original decoder ("logit lens") results in failure in early layers (solid ROC-AUC lines in Figure 1 near 0.5). Thus, following the Tuned Lens approach by Belrose et al., a specific decoder is pre-trained for each layer: the backbone is frozen, and a dedicated decoder is trained for 200 epochs using TabICL synthetic priors.
    - **Design Motivation**: TFMs are encoder-only and row-invariant without a vocabulary detokenizer; per-layer decoders are the most natural alternative for ICL tasks. They allow reading out potential performance if inference stopped at that layer.

2.  **Three Layer Intervention Experiments (skip / repeat / swap) + Self-repair Analysis**:
    - **Function**: Uses structural ablation to evaluate "what each layer contributes" and whether its removal can be compensated for by subsequent layers.
    - **Mechanism**: Skipping layer $l$ evaluates uniqueness; repeating layer $l$ evaluates if the layer performs iterative refinement; swapping two adjacent layers evaluates representational alignment. Overlaying the tuned lens on the skip experiment checks if performance recovers in subsequent layers—if it does, it indicates self-repair or layer redundancy.
    - **Design Motivation**: Looking only at the final output confuses "redundancy" with "self-repair." Tuned lens allows differentiation: if performance never recovers after skipping an early layer (Figure 8), it indicates a unique critical function. If performance rebounds immediately in the next layer's lens, it indicates redundancy and self-repair.

3.  **Proof-of-concept: nanoTabPFNlooped**:
    - **Function**: Operationalizes the analysis findings—if middle/late layers are just refining iteratively, repeating one layer $N$ times should theoretically match an $N$-layer stack.
    - **Mechanism**: Based on the nanoTabPFN architecture (lighter version of TabPFN v2), three versions are trained: 6-layer stack, 1-layer standalone, and 1-layer looped 6 times. The parameters are 3.72M / 0.75M / 0.75M respectively, though the looped version's FLOPs match the 6-layer stack. All are trained from scratch with 10,000 steps and batch 512.
    - **Design Motivation**: Modifying SOTA TFM architectures directly is costly; nanoTabPFN allows controlled comparison to confirm that performance gains come from "looping vs. stacking" rather than parameter count.

### Loss & Training
The nanoTabPFN series uses standard TabPFN training objectives. The TabICL prior generator configuration includes $batch=4 \times 10000$ batches, features 2-30, up to 10 classes, and sequence length 1024. Optimization uses AdamW, $\eta=10^{-4}$, cosine warmup of 2000 steps, and weight decay = 0. Training times for 1-layer, 6-layer, and looped models on a single A100 were 11.9h, 62.3h, and 68.8h respectively. Per-layer decoder fine-tuning used 200 epochs, $batch=8$, and $\eta=3 \times 10^{-5}$.

## Key Experimental Results

### Main Results
Table 1 compares the three nanoTabPFN versions on PMLBmini and TabArena:

| Model | Parameters | Computation | PMLBmini Performance | Gap vs 6-layer |
| :--- | :--- | :--- | :--- | :--- |
| nanoTabPFN-1l | 0.75M | 1× | Significantly Worst | Large Gap |
| nanoTabPFN-6l | 3.72M | 6× | Baseline | — |
| nanoTabPFN-looped | 0.75M | 6× | Close to 6l | Nearly Matched |
| TabPFN(2.5) | 10.7M | 24 layers | Upper Bound | Still better than looped |

Key Conclusion: The performance gap is primarily determined by whether the model performs 6 refinements, not whether it has 6 sets of independent parameters.

### Ablation Study
Six mechanistic experiments provide a profile of layer-wise behavior:

| Experiment | Main Finding | Implication |
| :--- | :--- | :--- |
| Embedding Similarity (cos / CKA) | Large models (TabPFN 2.5, LimiX-16M) form clear "layer blocks" | Representations within blocks undergo only small incremental updates |
| Class Separation Gap | Increases monotonically with depth; label embeddings lift slightly later than features | The model separates features before forming label representations |
| Probing classifier | Probes from layer $i$ generalize well to $j>i$, but not vice versa | Late layers preserve early layer information and stack new features |
| Tabular tuned lens | Most models achieve high AUC in early layers | Inference decisions are actually formed "very early" |
| Layer ablation (skip) | Performance collapses if Layer 1 is skipped; middle/late layers are almost lossless | Early layer = specialized mapping; middle/late layers = redundancy |
| Self-repair | Performance in the next layer's lens rebounds immediately after middle-layer skip | Presence of hydra-effect-style self-repair |

### Key Findings
- **Early layers are irreplaceable "mapping layers"**: TabICL and LimiX-2M are less sensitive to early transformer blocks due to strong encoders (row-interaction compression/RBF kernel preprocessing). For other models, skipping Layer 1 causes performance collapse, indicating early layers project raw tokens into a space suitable for residual stream operations.
- **Redundancy and self-repair in middle/late layers**: TabPFN(v2) shows a "jump" in lens performance near Layer 5, with significant overlapping computations between layers—the physical basis for the success of recurrent architectures.
- **Key differences between TFMs and LLMs**: TFMs are far more sensitive to layer swapping than LLMs (especially TabPFN v2). Additionally, destroying the final layer has little impact on output, contrasting with the "essential sharpening" in the final layer of LLMs. "Prediction calibration" in TFMs occurs later and more implicitly.
- **Strong encoders are a free lunch**: Models with explicit feature encoding (row-interaction/RBF kernels) are less sensitive to depth, suggesting a design direction of "wide encoder + shallow looped backbone."

## Highlights & Insights
- "Tabular tuned lens" is the key tool for cleanly migrating LLM logit lens to ICL tabular tasks: the failure of the original decoder is not due to poor representations but a misalignment between representations and the fixed decoder. Per-layer decoders reveal the model already "knows the answer."
- The design of three intervention types + lens overlay is clever: skip alone confuses "useless layer" with "self-repaired layer," while adding the lens separates the two. This analysis paradigm can be migrated to all ICL models.
- The single-layer recurrent validation successfully "monetizes" interpretability research: while typically mechanistic studies provide observations without solutions, this work converts the "iterative refinement" conclusion into an architecture that saves 80% of parameters.
- Essential differences in layer swap sensitivity and final-layer importance between TFMs and LLMs are revealed, providing empirical evidence that TFM design cannot blindly reuse LLM intuition.

## Limitations & Future Work
- Experiments primarily focus on binary classification; multi-class and regression are only validated in a limited capacity in the appendix. Transferability to long priors or complex high-cardinality tasks is unknown.
- The Tabular tuned lens used the open-source TabICL prior, which might underestimate the early-layer quality of models trained with more sophisticated priors like LimiX.
- nanoTabPFNlooped was only validated at a small scale (6 layers); whether it scales to 24-layer, 50,000-sample settings like TabPFN(2.5) remains unverified.
- Evaluation did not utilize ensembles; conclusions might be diluted in common TFM "subsampling ensemble" scenarios.
- Future directions: push analysis down to neuron/circuit levels; study how prior design shapes layer dynamics; apply similar tools to LLM-based tabular models (e.g., TabLLM) for cross-comparison.

## Related Work & Insights
- **vs. Lad et al. (Remarkable robustness of LLMs)**: They propose 4-stage LLM inference (detokenize → feature refinement → ensembling → sharpening). This paper proves TFMs have similar but differently distributed stages, with lower importance for the final layer.
- **vs. Belrose et al. (Tuned Lens)**: This paper concretizes "tuned lens" as per-layer decoders fine-tuned with tabular priors, bypassing the lack of vocabulary in TFMs.
- **vs. Looped Transformer (Universal Transformer, Dehghani 2019; Gong 2025)**: First migration of "iterative refinement" to tabular ICL models, providing mechanistic proof for *why* looping should work.
- **vs. TabPFN series and LimiX**: This work is not a competitor in architecture benchmarks but provides a complementary analysis template and compression recipe for the entire TFM family.

## Rating
- Novelty: ⭐⭐⭐⭐ — First large-scale layer-wise mechanistic study for TFMs, translating findings into specific architectural improvements.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 6 models × 6 experiments × 2 benchmarks, with appendix covering multi-class/regression; very complete chain of evidence.
- Writing Quality: ⭐⭐⭐⭐ — Clear logical chain with "takeaway" boxes; however, some charts (e.g., self-repair) require careful reading.
- Value: ⭐⭐⭐⭐ — Provides an analysis template for the TFM community and a "1 layer for 6" compression strategy, relevant for industrial deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Specialization after Generalization: Towards Understanding Test-Time Training in Foundation Models](../../ICLR2026/interpretability/specialization_after_generalization_towards_understanding_test-time_training_in_.md)
- [\[ICML 2026\] Memorization Dynamics of Fill-in-the-Middle Pretraining](memorization_dynamics_of_fill-in-the-middle_pretraining.md)
- [\[ICML 2026\] OmniSapiens: A Foundation Model for Social Behavior Processing via Heterogeneity-Aware Relative Policy Optimization](omnisapiens_a_foundation_model_for_social_behavior_processing_via_heterogeneity-.md)
- [\[ICML 2026\] Tracing the Dynamics of Refusal: Exploiting Latent Refusal Trajectories for Robust Jailbreak Detection](tracing_the_dynamics_of_refusal_exploiting_latent_refusal_trajectories_for_robus.md)
- [\[ICML 2026\] Dissecting Multimodal In-Context Learning: Modality Asymmetries and Circuit Dynamics in modern Transformers](dissecting_multimodal_in-context_learning_modality_asymmetries_and_circuit_dynam.md)

</div>

<!-- RELATED:END -->
