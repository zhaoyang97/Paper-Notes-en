---
title: >-
  [Paper Note] UniICL: An Efficient ICL Framework Unifying Compression, Selection, and Generation
description: >-
  [ACL 2025 (Long Paper, acl-long.24)][Model Compression][ICL] This work proposes the UniICL framework, which utilizes **a single frozen LLM** to concurrently accomplish three tasks: demonstration compression (compress $\rightarrow$ virtual tokens), demonstration selection (ranking based on the similarity of compressed virtual tokens), and final response generation. It requires only 17M trainable parameters (projection layer + learnable embedding). Coupled with a Demonstration…
tags:
  - "ACL 2025 (Long Paper, acl-long.24)"
  - "Model Compression"
  - "ICL"
  - "Demonstration Compression"
  - "Virtual Token"
  - "Demonstration Bank"
  - "Contrastive Learning"
date: 2026-05-08
content_hash: a7906aa498d74da4
---

# UniICL: An Efficient ICL Framework Unifying Compression, Selection, and Generation

**Conference**: ACL 2025 (Long Paper, acl-long.24)  
**arXiv**: [2405.17062](https://arxiv.org/abs/2405.17062)  
**Code**: None  
**Area**: In-Context Learning / Prompt Compression / Demonstration Selection  
**Keywords**: ICL, Demonstration Compression, Virtual Token, Demonstration Bank, Contrastive Learning  

## TL;DR

This work proposes the UniICL framework, which utilizes **a single frozen LLM** to concurrently accomplish three tasks: demonstration compression (compress $\rightarrow$ virtual tokens), demonstration selection (ranking based on the similarity of compressed virtual tokens), and final response generation. It requires only 17M trainable parameters (projection layer + learnable embedding). Coupled with a Demonstration Bank caching mechanism to avoid redundant compression, UniICL scales from 4-shot to 64-shot ICL under a 12$\times$ compression ratio (within 24GB VRAM), outperforming baselines like AutoCompressor, ICAE, and LLMLingua on multiple out-of-domain datasets.

## Background & Motivation

The core mechanism of In-Context Learning (ICL) is to prepend a few demonstrations to the prompt to activate the LLM's reasoning capabilities. Intuitively, providing more demonstrations brings richer contextual information. In reality, however, two severe bottlenecks are encountered:

1. **Context length explosion**: As the number of demonstrations increases, the prompt length expands dramatically, leading directly to out-of-memory (OOM) errors and degraded inference speed. Even 4-shot prompting can trigger the 24GB VRAM ceiling on 7B models.
2. **Uneven quality of demonstrations**: Existing selection methods (such as S-BERT retrieval) perform only shallow semantic matching; the retrieved demonstrations do not necessarily guide the LLM to generate the correct answers.

Existing solutions fall into two categories, each with its drawbacks:

- **Prompt compression** (AutoCompressor, ICAE, LLMLingua): These introduce independent compressors to compress demonstrations into soft prompts or prune tokens. However, the extra compressor must be loaded alongside the target LLM, increasing VRAM overhead. Furthermore, AutoCompressor’s recursive compression disrupts the independence between demonstrations, while ICAE struggles to process inputs exceeding its window length.
- **Demonstration selection** (BM25, S-BERT, fine-tuned LLM ranker): These introduce independent retrievers/rankers. These extra models similarly incur deployment costs.

The key challenge is that: **both compression and selection require extra modules, which must reside in VRAM simultaneously with the target LLM**, contradicting the goal of resource efficiency.

## Core Problem

Can **a single unified model** concurrently perform demonstration compression, demonstration selection, and final response generation **without introducing extra compressors or retrievers**, thereby maintaining or even enhancing ICL performance while significantly reducing VRAM footprint?

This problem is crucial because, in real-world deployment, every additional module implies extra VRAM/computational overhead and engineering complexity. If these three functions can be unified within a single frozen LLM, it achieves a true "resource-saving & multi-shot" win-win.

## Method

### Overall Architecture

The pipeline of UniICL consists of three steps, all reusing the same **frozen** Decoder-only LLM (Vicuna-7B or BlueLM-7B):

1. **Demonstration Compression**: Each candidate demonstration is independently passed through the frozen LLM. Learnable compression slots `[M]` are appended to the end, and the last hidden states corresponding to these slots are projected into compressed virtual tokens via a projection layer.
2. **Demonstration Selection**: Average pooling is applied to the virtual tokens of both the query and all candidate demonstrations. The cosine similarity is calculated as the saliency score, and the top-$m$ demonstrations are selected in descending order of the scores.
3. **In-context Generation**: The selected $m$ groups of virtual tokens are concatenated horizontally and fed into the same frozen LLM along with the query, performing auto-regressive generation (for generative tasks) or PPL-based evaluation (for understanding tasks).

Additionally, a **Demonstration Bank (DB)** is designed: Because demonstrations are compressed independently, virtual tokens of the same demonstration can be cached and reused, avoiding redundant compression.

### Key Designs

1. **Learnable Compression Slots `[M]`**: Initialized from low-frequency embeddings of the target LLM and appended to the tail of each demonstration. Due to causal attention, the hidden states at the slot positions are forced to attend to all preceding actual tokens, naturally consolidating the information. Each compressed demonstration produces $k$ hidden states ($k$ is determined by the compression ratio; the paper default is 12$\times$, meaning 512 tokens $\rightarrow$ ~42 virtual tokens).

2. **Projection Layer**: A simple linear layer $c_j^i = W_p \cdot h_j^i$ that maps hidden states into virtual token embeddings compatible with the LLM. This is one of only two trainable parameters in the framework (along with `[M]`), totaling only **17M** parameters.

3. **Independent Compression & Concatenation Strategy**: Unlike AutoCompressor's recursive compression, UniICL compresses each demonstration independently (preserving independence between demonstrations in ICL) and then concatenates the virtual tokens. Benefits: (a) supports batch parallel compression; (b) unaffected by demonstration order; (c) naturally supports caching and retrieval. When a single demonstration exceeds the window limit, it is segmented into multiple parts, compressed separately, and then concatenated (concatenation compression).

4. **Demonstration Bank (DB)**: Caches compressed virtual tokens. During inference, if a candidate demonstration already exists in the DB, it is retrieved directly; otherwise, it is compressed and stored. This allows UniICL + Caching to incur near-zero extra latency overhead.

5. **Contrastive Learning for Mining Positive/Negative pairs based on PPL gain**: During the selection training phase, InfoNCE Loss is co-optimized with LM Loss. Positive and negative examples are mined cleverly: given a query $Q$ and candidate demonstrations, the frozen LLM first computes the baseline PPL using only $Q$. Then, each candidate is appended to compute the PPL with that demonstration. The demonstration yielding the most decrease in PPL is selected as the positive example $D^+$, while the one yielding the most increase/least decrease is the negative example $D^-$. This labeling method, based on **actual utility (PPL gain)** rather than superficial semantic similarity, captures "genuinely helpful" demonstrations more effectively than traditional S-BERT retrieval.

### Loss & Training

Two-stage training:

- **Phase 1 (Compression Learning)**: Optimized using only LM Loss. The source text of a training sample is randomly split into two segments. One segment is compressed into virtual tokens and concatenated with the other segment, then fed into the frozen LLM to generate the answer. The projection layer is optimized to ensure virtual tokens can reconstruct the compressed information.
  $$\mathcal{L}_{lm} = -\frac{1}{|y|}\sum_t \log P(y_t | Q; C; y_{<t})$$

- **Phase 2 (Selection Enhancement)**: Jointly optimized using LM Loss and Contrastive Loss.
  $$\mathcal{L} = \mathcal{L}_{lm} + \mathcal{L}_{ctr}$$
  $$\mathcal{L}_{ctr} = \frac{\exp(\cos(\bar{C}_Q, \bar{C}_{D^+}))}{\exp(\cos(\bar{C}_Q, \bar{C}_{D^+})) + \exp(\cos(\bar{C}_Q, \bar{C}_{D^-}))}$$

The training data contains only **30k samples** (a mixture of XSUM, CICERO, and SUPER-NI), which represents an extremely small training scale.

## Key Experimental Results

Main experiments are evaluated on **out-of-domain** datasets to verify generalization capabilities:

| Dataset | Task | Metric | Vicuna (best shot) | LLMLingua | ICAE | UniICL♠+$L_{ctr}$ | Gain vs Vicuna |
|--------|------|------|-----|-----------|------|---------------------|------|
| CoLA-dev | Linguistic Acceptability | Acc | 62.3 (5-shot) | 54.9 | 59.3 | **65.6** (8-shot) | +3.3 |
| SST-2-dev | Sentiment Classification | Acc | 93.0 (5-shot) | 88.9 | 91.4 | **94.0** (8-shot) | +1.0 |
| IMDb | Sentiment Classification | Acc | 94.1 (5-shot) | 90.2 | 92.4 | **95.1** (8-shot) | +1.0 |
| ARXIV | Text Summarization | R-1 | 34.4 (1-shot) | — | — | **37.2** (5-shot) | +2.8 |
| XSum | Text Summarization | R-1 | 21.2 (1-shot) | — | — | **25.8** (5-shot) | +4.6 |
| MS MARCO | Passage Ranking | MRR@10 | 28.9 | — | 30.2 | **31.6** | +2.7 |

**Efficiency Comparison** (8$\times$ A5000 24GB):
- Naive Vicuna encounters OOM during 8-shot inference.
- AutoCompressor/ICAE/LLMLingua support up to 32-shot.
- **UniICL scales up to 64-shot** within 24GB constraints.

**Training Cost Comparison**:

| Method | Extra Compressor | Trainable Parameters | Training Data Volume |
|------|-----------|-----------|-----------|
| LLMLingua | ✓ (7B) | 7B | 57k |
| AutoCompressor | ✗ | 7B | Unknown |
| ICAE | ✓ (LoRA) | 70M | 240k |
| **UniICL** | **✗** | **17M** | **30k** |

### Ablation Study

- **Removing $L_{ctr}$ (Contrastive Loss)**: Performance degrades significantly, and the gap widens as the number of shots increases. This indicates that contrastive learning is crucial for demonstration selection.
- **UniICL Selection vs S-BERT Selection**: Selecting demonstrations via UniICL's own virtual token space (indicated by ♠) consistently outperforms S-BERT retrieval, proving that similarity in the virtual token space is more effective than superficial semantic similarity.
- **Sensitivity to Compression Ratio**: Performance is relatively stable between 4$\times$ and 12$\times$, starts to degrade at 16$\times$, and drops drastically at 512$\times$ (compression to a single token). The default is set to 12$\times$.
- **Comparison to LoRA Fine-Tuning**: Under equivalent parameter size (17M), LoRA-tuned Vicuna/BlueLM (512 window size) fails to match UniICL, demonstrating that using a projection layer for compression is more effective than LoRA adaptation.
- **Validation on BlueLM Backbone**: The framework remains effective on BlueLM-7B, proving its generalizability.

## Highlights & Insights

- **The "one model for three tasks" design philosophy**: Utilizing the frozen LLM itself as the compressor avoids the VRAM overhead of extra modules. The key insight is that LLMs already learn semantic understanding during pre-training; thus, only a lightweight projection layer is needed to guide it to convert hidden states into reusable virtual tokens.
- **Independent Compression + Cache Reuse = Demonstration Bank**: The design of compressing each demonstration independently is highly elegant: (a) it preserves the independence of demonstrations in ICL (unlike AutoCompressor's recursive reliance), (b) it naturally supports parallel compression, and (c) it supports caching, resulting in near-zero extra overhead during inference.
- **Mining positive and negative examples via PPL gain**: Rather than relying on superficial semantic similarity, this method assesses the actual utility of a demonstration based on "whether the PPL decreases upon inclusion". This is much more effective than S-BERT selection and can be applied to passage filtering in RAG.
- **Extremely low training footprint**: Achieving effective compression and selection capabilities with only 17M parameters and 30k training samples represents the pinnacle of parameter efficiency.

## Limitations & Future Work

- **Limited to naive ICL**: The integration with advanced prompting strategies like RAG or Chain-of-Thought (CoT) remains unexplored. Whether virtual tokens can preserve the logical structures of CoT reasoning chains is an open question.
- **Constrained scale of backbones**: Experiments were only verified on 7B models. Performance on larger LLMs (13B, 70B) is still unknown, especially regarding how the trade-off between the compression ratio and information retention scales with model capacity.
- **Compression window limit of 512**: The max input length is capped at 512 tokens. When handling truly long-document demonstrations (thousands of tokens), segmenting and concatenating them may lead to information loss and sub-optimal inter-segment coherence.
- **Lack of fine-grained generation quality analysis**: Only ROUGE scores were reported for summarization tasks, lacking human evaluations or LLM-as-a-judge quality assessments.
- **Fixed compression ratio**: All demonstrations use a uniform 12$\times$ compression ratio, lacking adaptive adjustments based on demonstration complexity.

## Related Work & Insights

| Dimension | AutoCompressor | ICAE | LLMLingua | **UniICL** |
|------|---------------|------|-----------|-----------|
| Compression Method | Recursive soft prompt | Independent soft prompt (LoRA) | Token pruning | Independent soft prompt (projection) |
| Extra Module | None (but fully parameter-trained) | LoRA compressor (70M) | Independent 7B compressor | **None** |
| Demonstration Independence | ✗ (recursive dependency) | ✓ (but restricted window) | ✓ | **✓** |
| Selection Capability | ✗ | ✗ | ✗ | **✓ (Built-in)** |
| Trainable Parameters | 7B | 70M | 7B | **17M** |
| Cache Reuse | ✗ | Partial | ✗ | **✓ (DB)** |
| Scalability | ≤32-shot | ≤32-shot | ≤32-shot | **≤64-shot** |

The core advantages of UniICL lie in its "unification" and "lightweightness": it achieves the three-in-one functionality with minimal extra parameters and naturally supports caching. The main disadvantage is that it still requires training a projection layer for the target LLM (though at very low cost), and its cross-model transferability has not been verified.

## Inspirations & Connections

- **Transfer from ICL to RAG**: The Demonstration Bank idea in UniICL can be directly applied to RAG systems—compressing retrieved passages into virtual tokens to cache them and reduce the input length for the LLM. The method of using PPL gain to filter positive and negative examples can also be adapted for passage reranking.
- **Virtual Tokens as a Universal Knowledge Representation**: Compressed virtual tokens are essentially dense representations of demonstrations in the LLM's semantic space. This can be viewed as a paradigm of "distilling knowledge into embeddings," which is worth exploring in knowledge distillation and model merging.
- **Demonstration Bank $\rightarrow$ Knowledge Base**: If a large volume of demonstrations is pre-compressed and stored in the DB, it essentially constructs an LLM-native knowledge base where retrieval and utilization happen within the exact same semantic space, offering tighter integration than traditional vector databases.

## Rating

- Novelty: ⭐⭐⭐⭐ The three-in-one unified design is valuable, though individual components (soft prompt compression, contrastive learning selection) are not completely novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Complete multi-task, multi-backbone validation + efficiency analysis + ablation studies, but lacks larger-scale models and human evaluation.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, rich diagrams, and accurate description of the methodology, though the Related Work section is slightly crowded.
- Value to Me: ⭐⭐⭐ Provides practical ideas for ICL efficiency optimization; both the Demonstration Bank caching mechanism and the PPL gain mining strategy are highly referable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Representation Shift: Unifying Token Compression with FlashAttention](../../ICCV2025/model_compression/representation_shift_unifying_token_compression_with_flashattention.md)
- [\[ICML 2026\] Unifying Dataset Pruning and Distillation for Efficient Large-scale Compression](../../ICML2026/model_compression/unifying_dataset_pruning_and_distillation_for_efficient_large-scale_compression.md)
- [\[ICML 2025\] LoRA Fine-Tuning without GPUs: A CPU-Efficient Meta-Generation Framework for LLMs](../../ICML2025/model_compression/lora_fine-tuning_without_gpus_a_cpu-efficient_meta-generation_framework_for_llms.md)
- [\[ACL 2025\] UniQuanF: Unifying Uniform and Binary-coding Quantization for Accurate Compression of Large Language Models](uniquanf_unified_quantization.md)
- [\[ACL 2025\] SCOPE: Optimizing Key-Value Cache Compression in Long-context Generation](scope_optimizing_key-value_cache_compression_in_long-context_generation.md)

</div>

<!-- RELATED:END -->
