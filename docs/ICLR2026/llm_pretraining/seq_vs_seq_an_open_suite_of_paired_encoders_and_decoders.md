---
title: >-
  [Paper Note] Seq vs Seq: An Open Suite of Paired Encoders and Decoders
description: >-
  [ICLR 2026][Pretraining][encoder-only] The authors develop a suite of paired encoder-only and decoder-only models (the ETTIN suite) ranging from 17M to 1B parameters. Using **identical** data, architectures, and training recipes—varying only in the objective function and attention direction—they achieve SOTA performance on open-data benchmarks for both type
tags:
  - ICLR 2026
  - Pretraining
  - encoder-only
  - decoder-only
  - MLM vs CLM
date: 2026-05-08
content_hash: 6924a10548a004c7
---
# Seq vs Seq: An Open Suite of Paired Encoders and Decoders

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=z5Mn8Rxi3l](https://openreview.net/forum?id=z5Mn8Rxi3l)  
**Code**: https://github.com/JHU-CLSP/ettin-encoder-vs-decoder (Available)  
**Area**: LLM Pre-training / Representation Learning  
**Keywords**: encoder-only, decoder-only, paired models, MLM vs CLM, cross-objective training

## TL;DR
The authors develop a suite of paired encoder-only and decoder-only models (the ETTIN suite) ranging from 17M to 1B parameters. Using **identical** data, architectures, and training recipes—varying only in the objective function and attention direction—they achieve SOTA performance on open-data benchmarks for both types. They demonstrate that encoders significantly outperform decoders in classification and retrieval tasks, while the reverse is true for generation, and **converting one model type to another via continued training (cross-objective) cannot bridge this performance gap**.

## Background & Motivation

**Background**: The LLM community focuses almost exclusively on decoder-only (GPT-style) models due to their natural suitability for text generation. However, a significant number of practitioners still utilize encoder-only (BERT-style) models for tasks like classification, retrieval, and embeddings where generation is unnecessary. Due to a long-standing lack of new encoder model iterations, many still rely on the original 2019 BERT.

**Limitations of Prior Work**: A popular viewpoint suggests that since decoders are larger, trained longer, and capable of zero-shot learning, they can naturally take over encoder tasks, rendering separate encoder training unnecessary. Top performers on retrieval leaderboards like MTEB are indeed dominated by 7B+ decoders (or embedding models derived from them), but this conclusion lacks clean experimental support.

**Key Challenge**: Past comparisons between "encoder vs decoder" have been **apples-to-oranges**—the models compared differed in parameter counts, architectures, pre-training data, and learning rate schedules. A few works attempting to control variables were limited to small data scales (e.g., 100B tokens), where the results might merely reflect the high data efficiency of CLM at small scales. Consequently, it has remained unclear whether performance differences stem from the architecture/objective itself or from training details.

**Goal**: (1) Create a suite of paired models where the **only variable is the training objective**, making encoders and decoders truly comparable; (2) Quantify the respective strengths and weaknesses of both objectives and the impact of parameter scaling; (3) Determine whether it is worthwhile to continue training a decoder as an encoder (or vice versa).

**Key Insight**: This work borrows the "open data + multi-size + full checkpoints" philosophy from Pythia but extends it to **paired architectures** while ensuring the models themselves achieve SOTA. Comparisons are only persuasive if both sides represent the strongest models of their size, thereby ruling out the suspicion that training choices favored one side.

**Core Idea**: Train encoders and decoders using the same data, the same architecture, and the same recipe, **varying only the objective function (MLM vs CLM) and attention (bidirectional vs causal)**. This cleanly isolates architectural differences from training noise.

## Method

This paper presents an "open model suite + controlled empirical study." Rather than a complex algorithmic pipeline, the core lies in the **experimental design**: how variables are isolated, how a strong recipe is replicated, and how cross-objective comparisons are structured.

### Overall Architecture

The ETTIN suite consists of 10 models forming 5 pairs (one encoder and one decoder per pair), with sizes of 17M, 32M, 68M, 150M, 400M, and 1B, trained on up to 2T tokens. The process involves: **fixing the open data, a model architecture table, and a three-stage training recipe**. Every size then receives one encoder (MLM + bidirectional attention) and one decoder (CLM + causal attention), which are **identical unit for unit** except for the objective and attention (same data, layers/width, and LR schedule). After training, **cross-evaluation** is conducted on encoder tasks (GLUE, MTEB retrieval, long context, code retrieval) and decoder tasks (ARC, HellaSwag, TriviaQA, etc.). Finally, **cross-objective continued training** is performed: training the decoder as an "encoder-from-decoder" using MLM-style objectives and training the encoder as a "decoder-from-encoder" using CLM, to see if the architectural gap can be bridged. Checkpoints are saved every 8.5B tokens (236 per model) and open-sourced along with the batch order.

### Key Designs

**1. Paired Control: Isolating variables to "Objective + Attention"**

This is the foundational methodology addresses the "apples-to-oranges" flaw of previous studies. The authors ensure a pair of encoder/decoder models share the same training data and architecture configuration (layers, hidden dimensions, intermediate dimensions, attention heads, learning rate, weight decay, and warmup tokens—see Table 1 in the original paper), and the same three-stage schedule. **The only differences are**: the objective function (MLM for encoders, CLM for decoders) and the attention pattern (bidirectional for encoders, causal for decoders). Any observed performance gap on downstream tasks is thus strictly attributable to these two factors.

**2. Open-Source Replication of ModernBERT Recipe + Three-stage Training**

The authors replicate the recipe of ModernBERT (the strongest public encoder at the time) as a starting point. Since ModernBERT’s data is not public, they used public data from the Olmo series (selected sources from DCLM and Dolma v1.7) to create a **fully open** alternative. Training follows a three-stage trapezoidal learning rate schedule: ① Base pre-training (1.7T tokens) with mixed data and double warmup (LR and batch size); ② Mid-training / Context expansion (250B tokens) extended to a length of 8000, RoPE base adjusted to 160k, and higher-quality Dolmino data using an inverse-sqrt decay; ③ Decay phase (50B tokens) adding long documents (books, Wikipedia, textbooks) to decay the LR to 0.02 of the peak. Key variations from ModernBERT include using open data, adding decay during context expansion, avoiding model merging, reducing MLM masking from 30% to 15% in the decay phase, and synchronizing local/global RoPE values. This provides the community with the **first reproducible public recipe for a ModernBERT-style model**.

**3. Cross-objective Continued Training: Testing if decoders can replace encoders**

With paired models, the authors perform a controlled experiment often missing in literature: continuing pre-training finished models with the **inverse objective**. Two directions are tested: training a decoder into an encoder (encoder-from-decoder, using the MNTP objective from LLM2Vec, which predicts masked tokens from preceding hidden states to suit the causal structure); and training an encoder into a decoder (decoder-from-encoder, using CLM). Continued training uses 50B tokens (significantly more than the ~10B in LLM2Vec) with high-quality decay-phase data and a new trapezoidal schedule. This quantifies whether the common practice of adapting large decoders into embedding/classification models is truly effective compared to native training.

**4. Full Artifact Openness: A reusable analysis platform**

Open-sourcing all artifacts—data, batch sequences (batch order), and over 200 checkpoints per model—transforms ETTIN into a research platform. Researchers can study **when** certain capabilities are learned during training, how data order impacts learning, and how biases (e.g., gender bias) diverge between objectives. The inclusion of the batch order is particularly significant, as it enables fine-grained analysis of what is learned between specific checkpoints.

### Loss & Training
Encoders use MLM (30% masking during pre-training, 15% during decay) + bidirectional attention. Decoders use CLM + causal attention. Both share a trapezoidal LR schedule (warmup → stable → inverse-sqrt decay) with double warmup for LR and batch size. For cross-objective training, encoder-from-decoder uses MNTP (15% masking), while decoder-from-encoder uses CLM.

## Key Experimental Results

### Main Results

**Encoder Side (Representative tasks: GLUE Avg / MTEB Retrieval, etc.)**: ETTIN encoders generally match or exceed corresponding baselines, including ModernBERT.

| Size | Model | CodeSearchNet | MTEB Retrieval | GLUE Avg |
|------|------|------|------|------|
| Base (~150M) | ModernBERT base | 75.9 | 43.9 | 88.4 |
| Base (~150M) | **Ettin-Enc-150m** | 76.3 | 45.7 | **88.9** |
| Large (~400M) | ModernBERT large | 78.3 | 47.0 | 90.4 |
| Large (~400M) | **Ettin-Enc-400m** | 80.7 | 48.4 | **90.8** |
| XL (~1B) | DeBERTa-v1-xl | 75.6 | 47.2 | 90.7 |
| XL (~1B) | **Ettin-Enc-1B** | 82.3 | 50.1 | **91.6** |

**Decoder Side (Zero-shot average across 10 tasks)**: ETTIN decoders similarly match or exceed open-data SOTA.

| Size | Model | Avg |
|------|------|------|
| Base (~135-160M) | SmolLM2-135m | 45.2 |
| Base (~135-160M) | **Ettin-Dec-150m** | **46.2** |
| Large (~360-410M) | SmolLM2-360m | 53.1 |
| Large (~360-410M) | Ettin-Dec-400m | 53.1 |
| XL (~1B) | Llama-3.2-1B | 56.6 |
| XL (~1B) | **Ettin-Dec-1B** | **59.0** |

### Ablation Study (Encoder vs Decoder Comparison with Cross-objective Training)

| Task Type | Phenomenon | Key Statistic |
|------|------|------|
| Classification (MNLI) | Encoder crushes Decoder; smaller encoders beat larger decoders | 150M Encoder 89.2 > 400M Decoder 88.2 |
| Retrieval (MS MARCO dev) | Encoder leads; continued training helps decoders but fails to bridge the gap | 400M: Encoder 42.2 vs Encoder-from-decoder 41.4 |
| Generation (Average) | Decoder leads; gap increases with model size | ~equal at 68M → >6 pt gap at 1B |
| 1B Hard Tasks | Decoder-from-encoder is strong in classification but weak in generation | MMLU: 37.0 vs Decoder 27.0; GSM8k: 18.9 vs 32.0 |

### Key Findings
- **Cross-objective continued training cannot bridge architectural gaps**: Even with 50B tokens (vs ~10B in LLM2Vec), encoder-from-decoder models fail to match native encoders, and decoder-from-encoder models fail to match native decoders. A 400M native encoder outperforms a 1B decoder converted to an encoder on MNLI.
- **Poor scaling for decoder-from-encoder on generation**: The inferiority of converted encoders on generation tasks worsens as models grow, suggesting that adapting encoders for generation is particularly inefficient.
- **Averages hide nuance**: On "generation" tasks that are more like classification (ARC, SciQ), encoders used as generators can actually beat decoders. However, on HellaSwag, TriviaQA, and SiQA, decoders have such a massive advantage that it pulls the average significantly in their favor.
- **Gender Bias Case Study**: Given the same data, the MLM objective makes the model more likely to use neutral pronouns, while both types exhibit male bias (slightly heavier in decoders). Female pronoun usage increases with model size.

## Highlights & Insights
- **The "only two differences" extreme variable control** is the biggest highlight. It moves the encoder vs decoder debate from "intuition" to falsifiable experimental evidence, making the conclusions (MLM for classification/retrieval, CLM for generation, conversion failures) highly impactful.
- **Strong Practical Conclusion**: At scales under 1B, it is better to train a small task-specific encoder than to adapt a larger decoder into an embedding model—the former is smaller yet more powerful. The authors even hypothesize that a 3B encoder might outperform 7B+ decoders on MTEB, though large encoders are currently rare.
- **Platform Value**: Providing the batch order and 200+ checkpoints makes this suite a foundational infrastructure for research into training dynamics, data sequencing, and bias evolution.
- **MNTP Over Naive MLM**: Using preceding hidden states for masked prediction during conversion (MNTP) to align with the decoder's causal structure is a detail worth emulating for cross-architecture adaptation.

## Limitations & Future Work
- **Unbalanced Training Ratios**: 50B continued training tokens is small compared to 1.7T+ pre-training tokens. While this simulates "real-world adaptation," it leaves open the question of whether a larger continued training budget could eventually bridge the gap.
- **1B Size Limit**: Most models are small; complex tasks like MMLU and GSM8k only show clear signals at the 1B scale. The hypothesis that 3B encoders could beat 7B decoders remains a projection.
- **Contradictions with Concurrent Work**: Concurrent work (Gisserot-Boukhlef et al., 2025) suggests CLM followed by MLM is often better on 100B tokens; this paper argues that is an illusion of high CLM data efficiency at small scales. Readers should be cautious given the varying scales and settings.
- **Isolated Bias Analysis**: The WinoGender study only covers three pronoun categories and uses a simplified prediction task; the gender bias findings are illustrative and should not be over-extrapolated.

## Related Work & Insights
- **vs Pythia**: Pythia pioneered "open data + multi-size + full checkpoints" for decoders; ETTIN extends this to **paired encoder/decoder** models for controlled architectural comparison.
- **vs ModernBERT**: ModernBERT is a state-of-the-art encoder with closed data; ETTIN provides the first reproducible ModernBERT-style recipe with open data across a full size spectrum (17M–1B).
- **vs LLM2Vec**: LLM2Vec uses MNTP to convert decoders to embedding models (~10B tokens); ETTIN adopts MNTP with a 50B token budget for controlled testing, showing that even more conversion data cannot match a native encoder.
- **vs Charpentier & Samuel (GPT or BERT)**: While they compared DeBERTa and GPT-2 on small data, ETTIN elevates "apples-to-apples" comparison to SOTA scales.

## Rating
- Novelty: ⭐⭐⭐⭐ While not a new algorithm, the "paired control + SOTA scale" experimental design provides unique and practical insights.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 6 sizes × 2 architectures × diverse tasks + cross-objective training + hard tasks + bias case studies.
- Writing Quality: ⭐⭐⭐⭐ Clear logic and conclusions with high information density in tables.
- Value: ⭐⭐⭐⭐⭐ Provides an open model/data platform and firm evidence for high-level architectural decisions regarding encoders.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Reconstructing CLIP for Open-Vocabulary Dense Perception](../../CVPR2026/llm_pretraining/reconstructing_clip_for_open-vocabulary_dense_perception.md)
- [\[NeurIPS 2025\] Gemstones: A Model Suite for Multi-Faceted Scaling Laws](../../NeurIPS2025/llm_pretraining/gemstones_a_model_suite_for_multi-faceted_scaling_laws.md)
- [\[ICML 2026\] If open source is to win, it must go public](../../ICML2026/llm_pretraining/if_open_source_is_to_win_it_must_go_public.md)
- [\[ACL 2026\] Fine-tuning vs. In-context Learning in Large Language Models: A Formal Language Learning Perspective](../../ACL2026/llm_pretraining/fine-tuning_vs_in-context_learning_in_large_language_models_a_formal_language_le.md)
- [\[ICML 2026\] Names Don't Matter: Symbol-Invariant Transformer for Open-Vocabulary Learning](../../ICML2026/llm_pretraining/names_dont_matter_symbol-invariant_transformer_for_open-vocabulary_learning.md)

</div>

<!-- RELATED:END -->
