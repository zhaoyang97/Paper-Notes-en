---
title: >-
  [Paper Note] End-to-End Compression for Tabular Foundation Models
description: >-
  [ICML 2026][Model Compression][Tabular Foundation Models] TACO prepends a learnable transformer compressor to TabPFN-style tabular foundation models…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "Tabular Foundation Models"
  - "Context Compression"
  - "TabPFN"
  - "End-to-End Meta-Learning"
  - "Inference Acceleration"
date: 2026-05-08
content_hash: 7372bf5d81774349
---

# End-to-End Compression for Tabular Foundation Models

**Conference**: ICML 2026  
**arXiv**: [2602.05649](https://arxiv.org/abs/2602.05649)  
**Code**: https://github.com/machinelearningnuremberg/TACO (Available)  
**Area**: Model Compression / Tabular Foundation Models / In-context Learning  
**Keywords**: Tabular Foundation Models, Context Compression, TabPFN, End-to-End Meta-Learning, Inference Acceleration  

## TL;DR
TACO prepends a learnable transformer compressor to TabPFN-style tabular foundation models, compressing $N$ rows of training context into $K \ll N$ latent representations before feeding them to the predictor. Through end-to-end joint meta-learning, it achieves 94x faster inference and 97% VRAM savings at a 1% compression rate with almost no loss in ROC-AUC.

## Background & Motivation

**Background**: The paradigm in tabular prediction has shifted from GBDT to In-Context Learning (ICL) via Tabular Foundation Models (TFMs) such as TabPFN, TabICL, and TabDPT. These models are pre-trained on synthetic data and perform inference by feeding the entire training set as context into a bidirectional transformer in a single forward pass.

**Limitations of Prior Work**: TFMs utilize row-column 2D bidirectional attention, with a complexity of $\mathcal{O}(N^2 M)$ relative to the number of training samples $N$. Even with KV caching, this only reduces to $\mathcal{O}(NM)$. When $N \times M$ reaches hundreds of thousands of cells, VRAM is exhausted, forcing authors to use small-to-medium tables or resort to aggressive row/column subsampling.

**Key Challenge**: The attention context length directly couples "input information volume" with "inference cost." To maintain prediction accuracy, the full table must be provided, yet processing the full table is computationally prohibitive. Existing mitigations—such as distilling MotherNet into an MLP or TabFlex’s linear attention—either sacrifice accuracy or require architectural changes. **No prior work has attempted to directly compress the training context itself in an end-to-end manner.**

**Goal**: To compress the in-context training set from $N$ rows to $K$ rows ($K \ll N$) without altering the TFM backbone or losing accuracy, thereby linearly reducing inference complexity by $N/K$ times.

**Key Insight**: Decompose in-context learning into two modules: a "compressor $g$" and a "predictor $f$." The compressor is trained specifically to produce the minimal training set summary $D^{\text{mini-train}}$ that enables accurate downstream prediction. This effectively adapts the concept of dataset distillation into the TFM inference pipeline.

**Core Idea**: Insert a transformer compressor to compress the training table into $K$ prototypical rows, followed by **joint meta-learning** with the predictor, ensuring that the "compression" directly serves the "downstream prediction accuracy."

## Method

### Overall Architecture

TACO consists of two TabPFN v2-style 2D-attention transformers connected in series:

1.  **Compressor** $g_\phi$: Takes $D^{\text{train}} \in \mathbb{R}^{N \times (M+1)}$ and a dummy table of $K \times (M+1)$ (initialized by random sampling from the training set, with target columns masked by placeholders). After several layers of alternating row-column attention, the dummy row positions absorb information from the actual training table, outputting $D^{\text{mini-train}} \in \mathbb{R}^{K \times (M+1) \times L}$.
2.  **MLP Bridge**: A two-layer residual MLP connects the latent spaces of the two transformers.
3.  **Predictor** $f_\theta$: Uses a standard TabPFN v2 architecture to concatenate $D^{\text{mini-train}}$ with the test batch embedding $\mathcal{E}_f(x^{\text{test}})$, feeding them into attention blocks to output class scores for test points.

Both modules consist of 12 layers / 6 heads / 192 dimensions, each with 7M parameters. The entire pipeline optimizes:

$$\arg\min_{\theta,\phi}\;\mathbb{E}_{(D^{\text{train}},D^{\text{test}})\sim p(D)}\;\mathcal{L}\!\left(y^{\text{test}},\;f(x^{\text{test}},g(D^{\text{train}};\phi);\theta)\right)$$

Pre-training involves 80k steps on synthetic data and 11k steps on real data, with a sequence length curriculum progressing from 1k to 60k rows. The compression rate is defined as $r = K/N$.

### Key Designs

1.  **Dummy-row attention as context compressor**:
    *   **Function**: Compresses a training table $D^{\text{train}}$ of arbitrary size into $K$ latent row representations, where $K \ll N$.
    *   **Mechanism**: Concatenates $K$ dummy rows (with masked target columns) at the compressor input. Bidirectional attention allows information to flow freely between $N+K$ rows, while the output **retains only the latent representations of the dummy rows** as $D^{\text{mini-train}}$. Dummy rows essentially act as "learned queries," extracting prototypical patterns from the $N$ training samples.
    *   **Design Motivation**: Compared to "hard" selection like random or kNN subsampling, dummy-row attention makes the compression process **differentiable** and does not restrict compressed rows to being identical to original rows. The compressor can create synthetic prototypes, which is why TACO significantly outperforms random/kNN sampling (Insight 5).

2.  **End-to-end joint meta-learning vs. "frozen predictor + learned compressor"**:
    *   **Function**: Allows both compressor and predictor parameters to be updated simultaneously to optimize downstream in-context prediction loss.
    *   **Mechanism**: During training with fixed or mixed compression rates $r$, each synthetic dataset is compressed and then used for prediction, with the loss backpropagated to both $\phi$ and $\theta$. Insight 3 provides a key ablation: fixing the predictor with TabPFN v2 weights and only training the compressor results in **consistently worse performance** across all compression rates compared to joint training.
    *   **Design Motivation**: It is much easier for the predictor to **actively adapt** to the compressed latent space than to force a compressor to align with a fixed predictor. This redefines "compression" as "finding a pair of compression-prediction languages that understand each other."

3.  **Multi-rate training + chunk-and-stitch for scaling to millions of rows**:
    *   **Function**: A single checkpoint supports arbitrary switching between $r \in \{1\%, 2\%, 4\%, 8\%, 16\%\}$ at inference time and extends TFMs to $N > 10^6$ via a chunking strategy.
    *   **Mechanism**: During training, compression rates are uniformly sampled from $\{1, 2, 4, 8, 16\}$ for each synthetic dataset, teaching the model variable-rate compression. For massive tables (e.g., $N=10^6$), the data is divided into 100 chunks of $C=10^4$. Each chunk is independently compressed to $K_C=100$ rows, and these summaries are stitched into a global $D^{\text{mini-train}}$ for the predictor.
    *   **Design Motivation**: Variable-rate training avoids the need for multiple checkpoints. Insight 4 verifies that this dynamic training suffers no significant performance loss compared to rate-specific training (within 95% confidence). Chunk-and-stitch extends the compressor's experience (limited to $\le 10^4$ rows) to arbitrary $N$, which is critical for the million-row experiments in Insight 6.

### Loss & Training
The model directly reuses the classification cross-entropy or regression MSE as the in-context loss. Continuous targets are discretized into $\le 10$ bins for compatibility with classification training. Training uses the AdamW optimizer with cosine annealing, learning rate warmup to $1 \times 10^{-4}$, weight decay of $1 \times 10^{-2}$, gradient clipping at 1.0, and mixed precision. Pre-training took 20 days on 8 $\times$ H100 GPUs.

## Key Experimental Results

### Main Results

ROC-AUC (one-vs-one for multi-class) on 26 datasets from TabArena:

| Model | Mean ROC-AUC ↑ | Description |
| :--- | :--- | :--- |
| TabICL | 0.866 ± 0.103 | SOTA TFM Baseline |
| TabPFN v2.0 | 0.866 ± 0.103 | SOTA TFM Baseline |
| POT (No Compression) | 0.862 ± 0.101 | Same architecture, no compression baseline |
| TACO ($r=1\%$) | 0.855 ± 0.097 | Only 1% context |
| TACO ($r=2\%$) | 0.857 ± 0.098 | |
| TACO ($r=4\%$) | 0.857 ± 0.099 | |
| TACO ($r=8\%$) | 0.858 ± 0.100 | |
| TACO ($r=16\%$) | 0.858 ± 0.101 | |

The CD diagram indicates **no statistically significant difference** between TACO at 1% compression and the uncompressed POT baseline.

### Inference Efficiency (Synthetic 15k rows × 500 features, no KV cache)

| Method | Subsequent Predict | Gain | Predict VRAM | Gain |
| :--- | :--- | :--- | :--- | :--- |
| POT | 28.67 s | 1× | 22.45 GB | — |
| TACO 1% | 306 ms | **93.6×** | 549 MB | **−97.6%** |
| TACO 2% | 382 ms | 75.2× | 845 MB | −96.3% |
| TACO 4% | 544 ms | 52.7× | 1.41 GB | −93.7% |
| TACO 8% | 943 ms | 30.4× | 2.56 GB | −88.6% |
| TACO 16% | 1.91 s | 15× | 4.89 GB | −78.2% |

### Key Findings
- **Compression to 1% is nearly "free"**: Reducing context from 100% to 1% only drops ROC-AUC from 0.862 to 0.855 (within the standard deviation) while accelerating inference by 94x and saving 97.6% VRAM.
- **Joint training is a necessary condition**: Freezing the predictor and training only the compressor leads to inferior performance across all rates (Insight 3), proving that compression and prediction must share a common "language."
- **TACO significantly outperforms random/kNN subsampling**: The ROC-AUC gap narrows as the compression rate increases, confirming that learned prototypes are superior to hard selection (Insight 5).
- **Chunk-and-stitch unlocks million-row processing**: On MetroPT-3 (~1.2M rows), compressing to 1214 rows ($r=0.1\%$) yielded an AUPRC of 0.8955, outperforming POT/TabPFN v2 baselines using random/kNN context (Insight 6).

## Highlights & Insights
- **Embedding dataset distillation into in-context inference**: Unlike traditional distillation used for training acceleration, TACO introduces training set summarization as a part of the TFM inference pipeline through joint learning. This effectively automates "prompt engineering" in TFMs as "prompt compression."
- **Dummy-row as a differentiable query**: This adapts the latent bottleneck concept from Perceiver or Set Transformer to the tabular domain, while maintaining the interpretability of "unlabeled summaries" by masking the target column.
- **Multi-rate training facilitates flexible deployment**: The accuracy-latency trade-off is parameterized by $r$ at inference time within a single model, incurring almost zero operational overhead.
- **Transferable chunk-and-stitch logic**: Any scenario where "global self-attention is bottlenecked by $O(N^2)$" (e.g., long-sequence inference, large corpus retrieval compression) can benefit from the "local compression + global stitching" approach.

## Limitations & Future Work
- Evaluation was focused on TabArena classification and **did not cover regression or time-series**. The predictor currently uses discretization for regression; extending this to native regression is a clear next step.
- Synthetic priors are largely based on SCMs, with limited coverage of **real-world missing value distributions or temporal distribution shifts**. Incorporating time-drift priors (cf. Helli et al. 2024) would be valuable.
- While 1% compression is not statistically significant in ROC-AUC, there is an **absolute drop of 0.007**, which might be unstable for applications requiring precise calibration or long-tail class accuracy.
- Chunk-and-stitch assumes IID chunks and **lacks a specific alignment mechanism for chunks with covariate shift**, representing a potential engineering risk.

## Related Work & Insights
- **vs. TabPFN v2 / TabICL**: Uses the same 2D attention architecture but adds a context compressor; reduces inference complexity from $\mathcal{O}(N^2 M)$ to $\mathcal{O}(K^2 M)$ where $K=0.01 N$, with comparable performance.
- **vs. MotherNet**: MotherNet distills a transformer into a per-dataset MLP ("model compression"). TACO performs "context compression," achieving acceleration while retaining the flexibility of ICL.
- **vs. TabFlex**: TabFlex uses linear attention to reduce complexity from $N^2$ to $N$, but with limited accuracy. TACO directly reduces context length, providing a more aggressive optimization.
- **vs. random / kNN subsampling**: Hard selection loses information. TACO’s differentiable dummy-row compression synthesizes prototypes, consistently outperforming selection across 1–16% compression rates.

## Rating
- Novelty: ⭐⭐⭐⭐ Introducing differentiable context compression into the TFM pipeline is original, though dummy-row concepts draw from Perceiver / Set Transformer.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive 6-Insight evaluation + TabArena/TabFSbench/TableShift/MetroPT-3 benchmarks + detailed ablations (joint training / multi-rate / baselines / chunking).
- Writing Quality: ⭐⭐⭐⭐ Clear labeling of Insights; theoretical sections are concise yet sufficient.
- Value: ⭐⭐⭐⭐⭐ Directly addresses the primary bottleneck of TFMs (processing large tables); the open-source checkpoints are practically "plug-and-play" for real-time industrial inference.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Towards Resource-Efficient LLMs: End-to-End Energy Accounting of Distillation Pipelines](towards_resource-efficient_llms_end-to-end_energy_accounting_of_distillation_pip.md)
- [\[ICML 2026\] Auditing and Fixing Economic Validity in Tabular Foundation Models for Discrete Choice](auditing_and_fixing_economic_validity_in_tabular_foundation_models_for_discrete_.md)
- [\[CVPR 2026\] A Paradigm Shift: Fully End-to-End Training for Temporal Sentence Grounding in Videos](../../CVPR2026/model_compression/a_paradigm_shift_fully_end-to-end_training_for_temporal_sentence_grounding_in_vi.md)
- [\[ICML 2026\] Quantifying the Uncertainty of Foundation Models with Singular Value Ensembles](quantifying_the_uncertainty_of_foundation_models_with_singular_value_ensembles.md)
- [\[ICML 2026\] BioArc: Discovering Optimal Neural Architectures for Biological Foundation Models](bioarc_discovering_optimal_neural_architectures_for_biological_foundation_models.md)

</div>

<!-- RELATED:END -->
