---
title: >-
  [Paper Note] ML-Embed: Inclusive and Efficient Embeddings for a Multilingual World
description: >-
  [ICML 2026][Multilingual & Machine Translation][Matryoshka Representation Learning] ML-Embed extends the Matryoshka concept from one dimension (representation dimension) to **three dimensions**—embedding parameters (MEL)…
tags:
  - "ICML 2026"
  - "Multilingual & Machine Translation"
  - "Matryoshka Representation Learning"
  - "Multi-dimensional Nesting"
  - "MTEB"
  - "Low-resource Languages"
  - "Decoder-based Embedding"
date: 2026-05-08
content_hash: 1a7eba9b469a80fb
---

# ML-Embed: Inclusive and Efficient Embeddings for a Multilingual World

**Conference**: ICML 2026  
**arXiv**: [2605.15081](https://arxiv.org/abs/2605.15081)  
**Code**: https://github.com/codefuse-ai/CodeFuse-Embeddings  
**Area**: Text Embedding / Multilingual Models / Efficient Training  
**Keywords**: Matryoshka Representation Learning, Multi-dimensional Nesting, MTEB, Low-resource Languages, Decoder-based Embedding

## TL;DR
ML-Embed extends the Matryoshka concept from one dimension (representation dimension) to **three dimensions**—embedding parameters (MEL), model depth (MLL), and representation dimension (MRL)—enabling **full-stack nested training**. It constructs a multilingual training set with 282 natural languages and 40 programming languages, totaling 50 million samples, and releases a family of open-source models from 140M to 8B parameters. On 17 MTEB benchmarks, it ranks first in 9, with notable gains in Polish (+22.89) and Vietnamese (+6.88).

## Background & Motivation

**Background**: Text embeddings are foundational for RAG/semantic search. Current SOTA approaches mostly adapt decoder LLMs (E5-Mistral / NV-Embed / Qwen3-Embedding / Gemini-Embedding) into embedding models. While effective, this path is costly: training is expensive, inference requires large memory, low-resource languages are neglected, and more APIs are closed-source.

**Limitations of Prior Work**: There are three main barriers. (1) **Computational barrier**: Decoder-based embedding models often have 7B+ parameters, making training and deployment highly demanding. Existing Matryoshka Representation Learning (MRL) only optimizes the **storage** dimension (allowing truncated embedding outputs), but does not reduce training or inference costs. (2) **Language barrier**: On MTEB, only 1 model has full results for Polish, 11 for Japanese, 17 for Vietnamese, but 154 for English and 146 for multilingual—low-resource languages receive little attention, creating a vicious cycle. (3) **Transparency barrier**: Leading models (Qwen3-Embedding / Gemini-Embedding / EmbeddingGemma) are either closed-source APIs or open-weight but with undisclosed training data/recipes, making reproduction and improvement difficult.

**Key Challenge**: There is an implicit trade-off between efficiency, performance, and language coverage. Existing MRL methods only nest along the **output dimension**, but the real cost in decoder-based models lies in **parameters** (embedding layers are especially large in small/multilingual models), **depth** (number of transformer layers), and **output dimension**. Nesting in a single dimension is insufficient. LoRA-like methods reduce trainable parameters but still require loading the full model for inference, failing to address deployment pain points.

**Goal**: (1) Design a unified framework that enables **nested training along three dimensions** simultaneously, producing models of multiple sizes, depths, and dimensions in a single training run; (2) Use this framework to build multilingual models with broad low-resource language coverage; (3) Open-source the full dataset, weights, and code to break the transparency barrier.

**Key Insight**: The authors observe a neglected detail—in small decoders like Qwen3-0.6B, the **embedding layer accounts for 1/4 of total parameters** (due to the large multilingual vocabulary). This area is untouched by existing MRL and is a low-hanging fruit. They also note that Matryoshka Layer Learning (MLL) can address inference depth. Unifying these three aspects into a nested loss yields 3D-ML.

**Core Idea**: **3D-Matryoshka Learning**: For each forward pass, sample embedding rank $r'$, network depth $l$, and representation dimension $d'$; the loss function converges for any combination, resulting in a cube-shaped, decomposable model space from a single training run.

## Method

### Overall Architecture

ML-Embed adopts a two-stage training + 3D-ML framework:
- **Data**: 50 million samples from 121 public sources, covering 282 natural languages and 40+ programming languages, unified into three contrastive formats (retrieval / clustering / two-way classification).
- **Training**: Stage one uses 27 million retrieval samples for semantic grounding; stage two fine-tunes on 8.3 million mixed samples with task-specific instructions.
- **Objective**: Not standard InfoNCE, but a 3D-ML loss summed over a nested layer × MRL dimension grid.
- **Model Family**: Six sizes produced in one training: 140M / 330M / 0.6B / 1.7B / 4B / 8B.

Each forward pass processes: (1) MEL embedding layer with sub-rank $r'$ → (2) first $l$ transformer layers → (3) each layer's hidden state through final LN → (4) truncate to first $d'$ dimensions for contrastive loss. These three samplings occur **simultaneously** during training, forcing the model to produce useful representations for all combinations.

### Key Designs

1. **Matryoshka Embedding Learning (MEL) — Parameter Dimension Nesting**:

    - **Function**: Uses low-rank decomposition and nested training to compress the large embedding layer into deployable small matrices, reducing both **trainable** and **total** parameters.
    - **Mechanism**: Applies truncated SVD to decompose the original embedding $E \in \mathbb{R}^{v \times d_{model}}$ into $E_A \leftarrow U_r S_r \in \mathbb{R}^{v \times r}$ and $E_B \leftarrow V_r^{\top} \in \mathbb{R}^{r \times d_{model}}$, updating only $E_A, E_B$ during training. The Matryoshka trick: for each forward pass, randomly sample sub-rank $r' < r$ from $\{64, 128, 256, 512, 1024\}$, using only $E_{effective} = E_A[:, :r'] E_B[:r', :]$. This forces the model to pack key information into the first $r'$ columns. Inference supports two modes: **Compatibility Mode** multiplies out the standard embedding matrix for zero-change deployment; **Efficiency Mode** retains the low-rank form, allowing re-factorization with smaller $r'' \ll r$ for significant memory reduction.
    - **Design Motivation**: LoRA only reduces trainable parameters; inference still loads the full embedding. MEL reduces both. SVD initialization ensures $E_A E_B$ is close to the original matrix at the start, preserving pretraining knowledge and enabling a smooth transition from full to low-rank finetuning.

2. **Matryoshka Layer Learning (MLL) — Depth Dimension Nesting**:

    - **Function**: Allows the same weights to be used at different truncated depths; deployment can switch sizes by changing `num_hidden_layers`.
    - **Mechanism**: Selects a logarithmic set of layers $\mathcal{L}_{layers} = \{1, 2, 4, 8, 16, 32, L\}$; each selected layer $l$ passes its hidden state $h_l$ through $\text{LN}_{final}$ (reusing the final layer norm to ensure representation scale consistency), then participates in the contrastive loss. This is a variant of early-exit training, but all exits share the final LN. For inference, loading only the first $l$ layers yields a fully functional small embedding model.
    - **Design Motivation**: Simply truncating final layers often causes severe performance drops, as deep semantic knowledge is not anchored in shallow layers. Logarithmic intervals force each milestone layer to be a qualified exit point, allowing users to flexibly trade off accuracy and latency. Engineering-wise, this is fully compatible with Hugging Face `AutoModel`—just change the config.

3. **Unified 3D Nested Contrastive Loss + Matryoshka Representation Learning (MRL)**:

    - **Function**: Integrates the above three dimensions (param / depth / dim) into a single objective, ensuring convergence for any combination.
    - **Mechanism**: For each selected MLL layer $l$ and each MRL dimension $d' \in \mathcal{D}_{mrl} = \{8, 16, 32, \ldots, d_{model}\}$, perform truncation and contrastive loss. The truncated representation is $v_{l, d'}(\cdot) = \text{proj}_{d'}(\text{LN}_{final}(h_l(\cdot)))$, and the loss:

        $$
        \mathcal{L}_{3D\text{-}ML} = \sum_{l \in \mathcal{L}_{layers}} \sum_{d' \in \mathcal{D}_{mrl}} c_{l, d'} \mathcal{L}_{cl}(q_i, d_i^+, \{d_{i,j}^-\}; v_{l, d'})
        $$

        where $\mathcal{L}_{cl}$ is standard InfoNCE: $-\log \frac{e^{s(v_q, v_{d^+})/\tau}}{e^{s(v_q, v_{d^+})/\tau} + \sum_j e^{s(v_q, v_{d_j^-})/\tau}}$. MEL's sub-rank $r'$ is sampled per step, while layer and dim are fully enumerated. $c_{l, d'}$ are tunable weights.
    - **Design Motivation**: Training each model size separately costs N times more; nested training allows all sizes to share the forward pass, updating all exits/dims in one gradient step. The key is reusing the final LN for each exit, ensuring "representation scale" consistency across depths; otherwise, shallow and deep hidden states have very different norms, disrupting contrastive loss scaling.

### Loss & Training

The total loss is the above 3D-ML loss. Data is formatted into three contrastive types: retrieval (query, pos, hard negs, with hard negatives mined by Qwen3-Embedding-8B), clustering (anchor, same-class pos, diff-class neg), and two-way classification (text-as-anchor + label-text-as-pos/neg). Two-stage training: stage 1 on 27M retrieval samples for semantic grounding, stage 2 on 8.3M mixed samples + task instructions for fine-tuning.

## Key Experimental Results

### Main Results

Comparison on 17 MTEB benchmarks with leaderboard top-1/top-5 averages:

| Benchmark (Tasks) | Top-1 | Top-5 | **ML-Embed-8B** | ML-Embed-4B | ML-Embed-1.7B | ML-Embed-0.6B |
|------|------|------|------|------|------|------|
| Multilingual (131) | 72.32 | 69.45 | 66.79 | 65.80 | 63.70 | 61.30 |
| English (41) | 75.97 | 74.61 | 73.26 | 72.89 | 71.19 | 70.01 |
| European (73) | 63.60 | 62.32 | **68.00** | 67.53 | 65.47 | 63.40 |
| Indic (20) | 70.15 | 67.39 | **76.76** | 75.15 | 72.58 | 66.11 |
| German (19) | 59.96 | 55.72 | **66.43** | 65.49 | 63.99 | 61.58 |
| French (25) | 70.37 | 67.25 | **71.91** | 70.97 | 68.94 | 66.64 |
| Polish (17) | 50.95 | n.a. | **73.84** | 73.14 | 71.12 | 68.13 |
| Persian (52) | 71.58 | 65.26 | **71.12 (≈top-1)** | 69.94 | 68.35 | — |
| Vietnamese (50) | 54.74 | 52.37 | **61.62** | 61.20 | 60.27 | — |
| Average | 68.46 | 65.95 | **70.24** | 69.29 | 67.58 | — |

The 8B model ranks first in 9 out of 17 benchmarks, with especially large improvements in low-resource languages: Polish +22.89 (73.84 vs 50.95), Vietnamese +6.88. Even the small 0.6B model achieves 68.13 on Polish, far surpassing the previous top-1 of 50.95.

### Ablation Study

| Configuration | Avg. MTEB | Notes |
|------|---------|------|
| Full 3D-ML | best | MEL + MLL + MRL all enabled |
| w/o MEL | similar, but higher training cost | embedding layer not low-rank, much higher training memory |
| w/o MLL | slightly lower, loses depth flexibility | inference only at full depth |
| w/o MRL | fixed embedding dimension | loses storage/retrieval efficiency flexibility |
| Separate training for each size | total cost ×N | performance close to 3D-ML at each size, but N× training resources |

(See Appendix in the original paper for detailed numbers.)

### Key Findings

- **Greatest gains in low-resource languages**: Polish +22.89, Vietnamese +6.88. This shows that previous leaderboard top-1 models invested little in these languages—ML-Embed's "real data distribution-driven" rather than "benchmark-optimized" strategy yields superior performance in neglected languages.
- **Embedding layer is indeed the bottleneck for small models**: In Qwen3-0.6B, the embedding layer is 25% of parameters; MEL compresses this to rank=128, reducing embedding parameters to 1/10 of the original, with minimal MTEB performance loss.
- **MLL early exits remain effective**: The 0.6B model achieves 61.30 on multilingual benchmarks, close to 1.7B's 63.70, proving that "log-interval layers + shared final LN" produces shallow exits that are not just for show.
- **3D-ML is not a simple sum of three independent losses**: The three dimensions must be **jointly** sampled; training MEL or MLL alone breaks the nested property—for example, sampling only layers but not dims leaves prefix dims unoptimized.
- **Open-source strategy is itself a contribution**: The paper explicitly positions itself as "anti-closed-source," releasing all data, weights, and code. For languages like Polish with only one leaderboard model, this "build first, then open" approach may drive the community more than just leaderboard scores.

## Highlights & Insights

- **Three-dimensional Matryoshka is a simple yet overlooked product-level abstraction**: Previous MRL only addressed storage, MLL only depth, LoRA only trainable parameters. ML-Embed unifies all three in one loss, making "one training, many deployment forms" possible—this "multi-task training, single binary deployment" paradigm can generalize to all foundation model training.
- **Low-rank embedding + Matryoshka is an underrated engineering optimization**: In multilingual models, the embedding layer is 25% of parameters and has been largely ignored; MEL's SVD initialization + sub-rank nesting is a clever combination.
- **"Real data distribution" vs "benchmark optimization"**: Training data is distributed according to actual population and corpus (e.g., Spanish, Arabic), not just benchmark tasks. This stance is rare in the embedding community and directly results in large gains for low-resource languages.
- **Shared final LN across layers**: This detail is easily overlooked but essentially normalizes "output scale," making contrastive loss comparable across depths. It's a key engineering point for stable nested training.

## Limitations & Future Work

- **8B model underperforms top-1 on Multilingual (66.79 vs 72.32)**, indicating that it has not yet achieved first place on the most comprehensive multilingual benchmark, with strengths mainly in European/Indic/monolingual leaderboards. The authors note that 35/131 MTEB-Multilingual tasks are English-only, making this "pseudo-multilingual" evaluation less fair for true multilingual models, but the gap remains.
- **No reported inference latency**: Actual throughput/TTFT for MLL at different depths is missing; only theoretical truncation is discussed.
- **Secondary accuracy drop after MEL re-factorization**: When re-factorizing to smaller $r''$ at inference, the paper mentions "minor finetuning" but does not provide curves for MTEB drop at extreme $r''$.
- **Two-stage training order**: Stage 1 is all retrieval, stage 2 is mixed-with-instructions, but more aggressive curricula (e.g., low-resource first, then high-resource) are not explored for further boosting small language performance.

## Related Work & Insights

- **vs MRL (Kusupati 2022)**: The original only nests along the representation dimension; ML-Embed extends this to param + depth + dim, achieving true lifecycle optimization.
- **vs Matryoshka Layer Learning (Li 2024)**: They proposed depth nesting, but ML-Embed enumerates and sums MLL and MRL in the loss, achieving joint layer × dim grid optimization.
- **vs LoRA / QLoRA / AdaLoRA**: These only reduce trainable parameters; inference still loads the full model. MEL reduces both training and inference parameters, and Matryoshka allows further reduction at inference.
- **vs Qwen3-Embedding / Gemini-Embedding / NV-Embed**: These are closed-source or open-weight but with undisclosed training recipes; ML-Embed is fully open-source, releasing all data, code, and weights, leading in transparency.
- **vs KaLM-Embedding**: One of the few multilingual models with transparent training data, but heavily biased toward English/Chinese; ML-Embed covers Spanish, Arabic, Vietnamese, and other long-tail languages much more extensively.

## Rating
- Novelty: ⭐⭐⭐⭐ MEL (Matryoshka embedding layer) is new, and three-dimensional joint nested training is new; but MRL and MLL already exist, so the framework is more of a "systematic integration."
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 17 MTEB benchmarks + six model sizes + 430 tasks evaluated, making it one of the most comprehensive embedding papers.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, well-argued three barriers; but ablation data is incomplete in the main text, with many details in the appendix.
- Value: ⭐⭐⭐⭐⭐ Fully open-source, multi-size models, and large improvements for low-resource languages provide a strong reproducible baseline for the multilingual embedding community, likely to have long-term impact beyond leaderboard papers.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] NiuTrans.LMT: Toward Inclusive and Scalable Multilingual Machine Translation with LLMs](../../ACL2026/multilingual_mt/niutranslmt_toward_inclusive_and_scalable_multilingual_machine_translation_with_.md)
- [\[AAAI 2026\] STELLAR: Scene Text Editor for Low-Resource Languages and Real-World Data](../../AAAI2026/multilingual_mt/stellar_scene_text_editor_for_low-resource_languages_and_real-world_data.md)
- [\[AAAI 2026\] Bridging the Multilingual Safety Divide: Efficient, Culturally-Aware Alignment for Global South Languages](../../AAAI2026/multilingual_mt/bridging_the_multilingual_safety_divide_efficient_culturally-aware_alignment_for.md)
- [\[ICML 2026\] Toward Robust Multilingual Adaptation of LLMs for Low-Resource Languages](toward_robust_multilingual_adaptation_of_llms_for_low-resource_languages.md)
- [\[ACL 2026\] Efficient Low-Resource Language Adaptation via Multi-Source Dynamic Logit Fusion](../../ACL2026/multilingual_mt/efficient_low-resource_language_adaptation_via_multi-source_dynamic_logit_fusion.md)

</div>

<!-- RELATED:END -->
