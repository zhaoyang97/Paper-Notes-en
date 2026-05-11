---
title: >-
  [Paper Note] Spectral Attention Steering for Prompt Highlighting
description: >-
  [ICLR 2026][LLM Evaluation][attention steering] This paper proposes SEKA/AdaSEKA, which learns a "relevance subspace" via spectral decomposition of key embeddings and directly edits key vectors prior to attention computa…
tags:
  - "ICLR 2026"
  - "LLM Evaluation"
  - "attention steering"
  - "prompt highlighting"
  - "spectral decomposition"
  - "FlashAttention"
  - "key embedding editing"
date: 2026-05-08
content_hash: 3701c105c56f2782
---

# Spectral Attention Steering for Prompt Highlighting

**Conference**: ICLR 2026
**arXiv**: [2603.01281](https://arxiv.org/abs/2603.01281)
**Code**: [waylonli/SEKA](https://github.com/waylonli/SEKA)
**Area**: LLM Evaluation
**Keywords**: attention steering, prompt highlighting, spectral decomposition, FlashAttention, key embedding editing
**Authors**: Weixian Waylon Li, Yuchen Niu, Yongxin Yang, Keshuang Li, Tiejun Ma, Shay B. Cohen (University of Edinburgh, RayNeo, Huawei Research, QMUL)

## TL;DR

This paper proposes SEKA/AdaSEKA, which learns a "relevance subspace" via spectral decomposition of key embeddings and directly edits key vectors prior to attention computation to achieve prompt highlighting. The approach requires no storage of the full attention matrix, is fully compatible with FlashAttention, and incurs negligible overhead (+0.03s/sample).

## Background & Motivation

**Practical Demand for Prompt Highlighting**: In high-stakes scenarios, it is essential to precisely guide LLMs to attend to user-specified critical text within the prompt (e.g., new knowledge in factual conflicts, core constraints in instruction following), i.e., attention steering.

**Efficiency Bottleneck of Existing Methods**: State-of-the-art methods such as PASTA perform post-hoc modification of the attention matrix after it has been fully computed, requiring storage of the complete $T \times T$ attention matrix and thus being incompatible with IO-aware efficient implementations such as FlashAttention.

**Substantial Overhead**: PASTA incurs an inference latency increase of +1.03s/sample and a memory increase of +23.12 GB; SPA operates on logit distributions, does not support batch processing, and is the slowest (+5.32s).

**Costly Head Search**: PASTA additionally requires task-specific attention head search to determine which heads to steer, increasing deployment cost.

**Structured Signal in Key Embeddings**: Through comparative experiments, the authors find that when the question in the prompt shifts from irrelevant to relevant, the key embeddings of specific layers/heads exhibit consistent directional shifts (as shown via PCA visualization), indicating that "relevance" is encoded in a structured subspace of key representations.

**Feasibility of Pre-Attention Intervention**: The attention score $\text{Attn}(i,j) = \frac{\boldsymbol{q}_i^\top \boldsymbol{k}_j}{\sqrt{d_k}}$ depends on the query-key inner product; equivalent control can be achieved by editing the key side. Since keys are indexed by token position, they naturally lend themselves to controlling how much individual tokens are attended to.

## Method

### Overall Architecture

SEKA consists of two stages:

- **Offline Learning Stage**: Synthetic contrastive prompts are used to construct positive/negative cross-covariance matrices, which are decomposed via SVD to obtain a "relevance subspace" projection matrix.
- **Inference Stage**: A projection transformation $\boldsymbol{k}_j' = \boldsymbol{k}_j + g \boldsymbol{P} \boldsymbol{k}_j$ is applied to the key embeddings of highlighted tokens before attention computation.

### Key Design 1: Spectral Learning of Relevance Projection (Offline)

Three types of prompts are constructed: neutral (context only), positive (context + relevant question), and negative (context + irrelevant question). Key embeddings $\boldsymbol{h}, \boldsymbol{h}^+, \boldsymbol{h}^-$ are extracted from the same token span under each condition.

Cross-covariance matrices are computed and decomposed via SVD:

$$\boldsymbol{\Omega}_{\ell,h}^{+} = \frac{\boldsymbol{h}^\top \boldsymbol{h}^+}{n}, \quad \boldsymbol{\Omega}_{\ell,h}^{+} = \boldsymbol{U}_{\ell,h}^{+} \boldsymbol{S}_{\ell,h}^{+} \boldsymbol{V}_{\ell,h}^{+\top}$$

The positive projection retains the left singular vectors corresponding to the top $k^+$ largest singular values; the negative projection retains those corresponding to the smallest $k^-$:

$$\boldsymbol{P}_{\ell,h}^{+} = \boldsymbol{U}_{\ell,h,:,:k^+}^{+} (\boldsymbol{U}_{\ell,h,:,:k^+}^{+})^\top, \quad \boldsymbol{P}_{\ell,h}^{-} = \boldsymbol{U}_{\ell,h,:,k^-:}^{-} (\boldsymbol{U}_{\ell,h,:,k^-:}^{-})^\top$$

The selection of $k^+$ and $k^-$ is governed by a cumulative singular value ratio threshold $\gamma$: $\sum_{i=1}^{k^+} S_i^+ / \sum_{i=1}^{d_k} S_i^+ \geq \gamma$.

### Key Design 2: Key Editing at Inference Time

For each highlighted token's key vector:

$$\boldsymbol{k}_j' = \boldsymbol{k}_j + \frac{g^+ \cdot \boldsymbol{P}_{\ell,h}^+ \boldsymbol{k}_j + g^- \cdot \boldsymbol{P}_{\ell,h}^- \boldsymbol{k}_j}{2}$$

Substituting into the attention formula yields an equivalent low-rank bias added to the original attention logits:

$$\text{Logits}_{ij} = \underbrace{\frac{\boldsymbol{q}_i^\top \boldsymbol{k}_j}{\sqrt{d_k}}}_{A_{ij}} + \underbrace{\frac{\boldsymbol{q}_i^\top (g^+ \boldsymbol{P}^+ \boldsymbol{k}_j + g^- \boldsymbol{P}^- \boldsymbol{k}_j) / 2}{\sqrt{d_k}}}_{B_{ij}}$$

Since only the key vectors are modified throughout, without ever accessing the attention matrix, the method is inherently compatible with FlashAttention.

### Key Design 3: AdaSEKA Adaptive Routing

To handle multi-task scenarios, AdaSEKA learns $M$ domain-expert projections. At inference time, the query vector $\boldsymbol{q}_{\ell,h}$ of the last token in the prompt is extracted, and the alignment with each expert's principal directions is computed as routing weights:

$$\alpha_{m,\ell,h}(\boldsymbol{q}) = \frac{\sum_{k=1}^{K} (\boldsymbol{q}^\top \boldsymbol{u}_{m}^{+(k)}) \cdot \sigma_{m}^{+(k)}}{\max_{m'} |\sum_{k=1}^{K} (\boldsymbol{q}^\top \boldsymbol{u}_{m'}^{+(k)}) \cdot \sigma_{m'}^{+(k)}|}$$

The final projection matrix is a weighted combination of expert projections: $\boldsymbol{P}_{\text{dynamic}} = \sum_m \alpha_m \boldsymbol{U}_m^{+} (\boldsymbol{U}_m^{+})^\top$. Advantages include reduced hyperparameter tuning, modular deployment (new experts can be plug-and-played), and interpretable routing.

### Key Design 4: KV Head Filtering

Not all heads are sensitive to relevance. The authors compute the $\ell_2$ distance between positive and negative key embeddings:

$$D_{\ell,h} = \frac{1}{N} \sum_{i=1}^{N} \| \boldsymbol{h}_{\ell,h,i}^+ - \boldsymbol{h}_{\ell,h,i}^- \|_2$$

Projection is applied to a head only when $D_{\ell,h} \geq \delta_{\min}$. Visualizations show that heads in the middle-to-late layers exhibit significantly greater discriminability, consistent with findings from retrieval head research.

## Key Experimental Results

### Main Results: Standard Benchmarks

Evaluation is conducted on three tasks: CounterFact (knowledge conflict), Bias in Bios (occupation extraction), and Pronoun Changing (pronoun rewriting instruction following):

| Model | Method | CounterFact ES | CounterFact PS | Bias in Bios Acc | Pronoun P.Score | Pronoun A.P.Score |
|------|------|:-:|:-:|:-:|:-:|:-:|
| Qwen3-4B | Original | 45.00 | 45.64 | 79.84 | 93.14 | 90.52 |
| | PASTA | 97.16 | 96.03 | 89.58 | 95.82 | 94.64 |
| | SPA | 65.24 | 57.71 | 68.00 | 80.27 | 78.19 |
| | **SEKA** | **99.02** | **98.61** | **91.02** | 95.18 | 93.26 |
| | **AdaSEKA** | 98.90 | 98.72 | **91.86** | 94.54 | 92.08 |
| Qwen3-8B | Original | 39.04 | 39.59 | 76.08 | 98.00 | 97.84 |
| | PASTA | 92.70 | 91.68 | 86.32 | 98.86 | 98.72 |
| | **SEKA** | **99.08** | **98.96** | **88.74** | 98.56 | 98.26 |
| | **AdaSEKA** | 99.00 | 98.97 | 88.50 | **99.68** | **99.52** |
| Qwen3-14B | Original | 37.56 | 36.12 | 85.22 | 98.42 | 98.22 |
| | PASTA | 76.84 | 66.33 | 88.46 | 90.98 | 90.94 |
| | **SEKA** | **98.92** | **99.02** | 90.28 | 98.66 | 98.54 |
| | **AdaSEKA** | 99.00 | 99.15 | **91.22** | **99.88** | **99.86** |

### Efficiency Comparison

| Method | Latency (s/sample) | Peak Memory (GB, B=10) | Peak Memory (GB, B=1) |
|------|:-:|:-:|:-:|
| Original | 0.55 | 27.63 | 16.72 |
| PASTA | 1.58 (+1.03) | 50.75 (+23.12) | - |
| SPA | 5.87 (+5.32) | - | 17.71 (+0.99) |
| **SEKA** | **0.58 (+0.03)** | **27.66 (+0.03)** | **16.75 (+0.03)** |
| AdaSEKA | 0.82 (+0.27) | 43.22 (+15.59) | 18.23 (+1.51) |

SEKA incurs near-zero overhead, whereas PASTA doubles memory consumption and triples latency.

### Ablation Study

| Configuration | CounterFact ES (Qwen3-4B) | Bias in Bios Acc | Pronoun A.P.Score |
|------|:-:|:-:|:-:|
| SEKA (full) | **99.02** | **91.02** | **93.26** |
| w/o learn (random projection + head filtering) | 94.96 | 86.62 | 88.66 |
| w/o learn & filt (random projection + no filtering) | 86.12 | 71.76 | **36.95** |

Key findings:

- Replacing spectral learning with random projection leads to a notable performance drop, confirming that the learned relevance subspace carries meaningful structure.
- Further removing head filtering causes catastrophic degradation (Pronoun drops from 90.52 to 36.95), worse than applying no steering at all, demonstrating that projecting insensitive heads introduces severe interference.

### Lost-in-the-Middle Experiment

- **Applying SEKA highlighting to middle passages can reverse the U-shape performance curve**: exact match improves substantially at middle positions.
- Uniformly highlighting all passages may instead exacerbate the lost-in-the-middle effect.
- Adjusting $\delta_{\min}$ to control the number of steered heads can flatten the U-shape curve.
- PASTA underperforms the baseline on this task, highlighting the limitations of post-hoc methods in long-context settings.

## Highlights & Insights

1. **Full compatibility with FlashAttention**: This is the first method in this line of work to achieve this, by bypassing the requirement to store the attention matrix through pre-attention key editing.
2. **Near-zero overhead**: SEKA adds only +0.03s/sample in latency and +0.03 GB in memory, far superior to PASTA's +1.03s/+23.12 GB.
3. **Strong geometric interpretability**: The projection transformation $\boldsymbol{k}' = \boldsymbol{k} + g \boldsymbol{P} \boldsymbol{k}$ has a clear geometric meaning — amplifying the key vector along the relevance subspace direction.
4. **Training-free**: No fine-tuning is required; only a small number of synthetic contrastive prompts are needed for offline spectral decomposition.
5. **AdaSEKA's adaptive routing mechanism** reduces the need for cross-task/cross-model hyperparameter tuning, with four plug-and-play experts.
6. **Reversal of the U-shape curve in the lost-in-the-middle setting** is an intriguing new finding, demonstrating precise control over positional sensitivity via attention steering.

## Limitations & Future Work

1. **Offline stage depends on synthetic data quality**: The strategy for constructing contrastive prompt triplets affects the quality of the learned projections; generalization to new domains requires reconstruction.
2. **Hyperparameters still require tuning**: Although AdaSEKA reduces some tuning burden, $g^+$, $g^-$, $\gamma$, and $\delta_{\min}$ still require grid search, with optimal values varying across models and tasks.
3. **Limited to prompt highlighting scenarios**: The method focuses on "directing model attention to specified tokens" and does not cover broader activation steering objectives (e.g., style control, safety).
4. **Coarse highlighting range in the lost-in-the-middle experiment**: Positions 5–25 are manually specified; in practice, identifying the gold passage requires additional retrieval.
5. **AdaSEKA memory overhead is non-negligible**: +15.59 GB at batch size 10, primarily due to storing SVD components for multiple experts.

## Related Work & Insights

- **vs. PASTA** (Zhang et al., 2024): PASTA post-processes the attention matrix, is incompatible with FlashAttention, and incurs high latency and memory costs; SEKA outperforms it comprehensively while adding negligible overhead.
- **vs. SPA** (Tian & Zhang, 2025): SPA operates on logit distributions, does not support batching, and is the slowest; it substantially underperforms SEKA on CounterFact.
- **vs. Activation Steering** (SEA, RepE, etc.): Activation steering modifies hidden states in MLP layers to control semantic attributes, whereas SEKA controls the attention mechanism to determine where the model looks; the two are orthogonal and complementary.
- **Alignment with Retrieval Head Research**: Wu et al., 2025; Qiu et al., 2025 find that retrieval heads are concentrated in middle-to-late layers, consistent with SEKA's head filtering strategy.

## Rating

- ⭐ Novelty: 8/10 — Performing pre-attention steering from the key embedding side is a novel and practically motivated idea; the spectral decomposition combined with adaptive routing is elegantly designed.
- ⭐ Experimental Thoroughness: 8/10 — Covers 5 models × 3 standard benchmarks, plus lost-in-the-middle analysis, ablation studies, and efficiency profiling.
- ⭐ Writing Quality: 8/10 — Logic is clear, visualizations (PCA, heatmaps) are intuitive, and mathematical derivations are complete.
- ⭐ Value: 8/10 — Addresses the practical incompatibility between attention steering and FlashAttention; the method is simple, efficient, and engineering-friendly.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] GuidedSampling: Steering LLMs Towards Diverse Candidate Solutions at Inference-Time](guidedsampling_steering_llms_towards_diverse_candidate_solutions_at_inference-ti.md)
- [\[ICLR 2026\] Prompt and Parameter Co-Optimization for Large Language Models](prompt_and_parameter_co-optimization_for_large_language_models.md)
- [\[ICLR 2026\] vCache: Verified Semantic Prompt Caching](vcache_verified_semantic_prompt_caching.md)
- [\[ICLR 2026\] Breaking the Correlation Plateau: On the Optimization and Capacity Limits of Attention-Based Regressors](breaking_the_correlation_plateau_on_the_optimization_and_capacity_limits_of_atte.md)
- [\[ICLR 2026\] Prompt and Parameter Co-Optimization for Large Language Model Task Adaptation](prompt_and_parameter_co-optimization_for_large_language_models.md)

</div>

<!-- RELATED:END -->
