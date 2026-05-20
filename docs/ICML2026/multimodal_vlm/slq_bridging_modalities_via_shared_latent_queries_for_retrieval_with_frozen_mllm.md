---
title: >-
  [Paper Note] SLQ: Bridging Modalities via Shared Latent Queries for Retrieval with Frozen MLLMs
description: >-
  [ICML 2026][Multimodal VLM][Frozen MLLM] SLQ appends a small set of "shared latent queries" $\mathbf{Q}$ to the end of image/text token sequences…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "Frozen MLLM"
  - "Shared Latent Queries"
  - "Knowledge-aware Reasoning Retrieval"
  - "Contrastive Learning"
  - "KARR-Bench"
date: 2026-05-08
content_hash: 565941cba8ab1311
---

# SLQ: Bridging Modalities via Shared Latent Queries for Retrieval with Frozen MLLMs

**Conference**: ICML 2026  
**arXiv**: [2604.13710](https://arxiv.org/abs/2604.13710)  
**Code**: <https://github.com/CnFaker/SLQ>  
**Area**: Multimodal VLM / Cross-modal Retrieval / Parameter-efficient Fine-tuning  
**Keywords**: Frozen MLLM, Shared Latent Queries, Knowledge-aware Reasoning Retrieval, Contrastive Learning, KARR-Bench

## TL;DR
SLQ appends a small set of "shared latent queries" $\mathbf{Q}$ to the end of image/text token sequences, leveraging the MLLM's own causal attention to aggregate global context. **By training only a few thousand query parameters**, a frozen MLLM is turned into a retriever, outperforming full fine-tuning and LoRA on COCO/Flickr30K, and introduces KARR-Bench to evaluate "implicit knowledge reasoning" capabilities.

## Background & Motivation

**Background**: Multimodal large models (MLLMs) such as InternVL3 and Qwen3-VL process interleaved image-text inputs via a unified Transformer, capturing richer cross-modal semantic interactions compared to CLIP/BLIP's dual-tower architectures. Recent works (GME, MM-Embed, VLM2VEC, MMRet) aim to adapt MLLMs for retrieval to leverage their reasoning abilities.

**Limitations of Prior Work**: (1) **Intrusive fine-tuning**—Mainstream approaches use full fine-tuning or LoRA with contrastive objectives, but this **generative alignment → discriminative alignment** mismatch distorts the pretrained semantic space, causing catastrophic forgetting (semantic degradation); (2) **Training inefficiency**—Contrastive learning requires huge batch sizes for negative sample diversity, and full fine-tuning of billion-parameter backbones is computationally prohibitive at such scales; (3) Most baselines use the `<EOS>` (last token) hidden state as the global embedding—however, the last token is an "information bottleneck" and struggles to compress complex semantics. Diagnostic experiments (Figure 2) show it fails completely on implicit reasoning tasks (e.g., "animal with 9 lives" implying cat).

**Key Challenge**: MLLM pretraining already aligns vision and language into a **shared representation space** (enabling zero-shot VQA), but either no fine-tuning (poor zero-shot retrieval) or heavy parameter modification (damaging the pretrained space) is used—there is a lack of a "lightweight yet effective" intermediate solution.

**Goal**: (1) **Keep the backbone frozen**—not touching any pretrained parameters; (2) Use a lightweight mechanism to **activate** the MLLM's implicit knowledge and reasoning for retrieval; (3) Solve "which token to use for embedding"—neither last token nor pooling all tokens; (4) Provide a benchmark that truly distinguishes "pattern matching vs knowledge reasoning" in retrieval.

**Key Insight**: The authors conducted a diagnostic experiment—feeding a frozen InternVL3-1B with a **zero-initialized extra query token** appended to the sequence, allowing it to "see" all previous tokens via causal attention; its final hidden state is used for retrieval. Results: On "pattern matching" tasks, both query and last token succeed; on "knowledge retrieval (basic associations)", the last token gives nearly indistinguishable low scores, while the query maintains a high margin; on "logical reasoning", the last token fails completely, but the query successfully finds the target. This shows MLLMs **already have** reasoning retrieval capabilities, but the last token is stuck at the information bottleneck.

**Core Idea**: Use a small set of "shared, learnable latent queries" as a **modality-agnostic global aggregator**, leveraging the MLLM's own causal attention to extract unified retrieval embeddings from image/text sequences—**only learning the queries, keeping the backbone untouched**.

## Method

### Overall Architecture
Freeze the MLLM backbone + a small set of $N$ learnable Shared Latent Queries $\mathbf{Q} \in \mathbb{R}^{N \times D}$ (for InternVL3-8B, $N=20$, totaling only $20 \times D \approx$ tens of thousands of parameters). Text input: $\mathbf{X}_T = [\mathbf{E}_T; \mathbf{E}_{P_T}; \mathbf{Q}]$ (text embedding + text instruction prompt + shared queries); image input: $\mathbf{X}_I = [\mathbf{E}_I; \mathbf{E}_{P_I}; \mathbf{Q}]$. Both inputs are fed into the frozen MLLM $\mathcal{M}$ for causal attention; the **last $N$ positions** (corresponding to the queries) are mean-pooled and L2-normalized to obtain embeddings $\mathbf{z}_T, \mathbf{z}_I \in \mathbb{R}^D$. A symmetric InfoNCE loss aligns the two embedding spaces. At inference, the outputs at these query positions serve as modality-agnostic global embeddings for retrieval. Only $\mathbf{Q}$ and temperature $\tau$ are updated during training; the backbone remains completely frozen.

### Key Designs

1. **Shared Latent Queries + Tail Appending + Causal Attention Aggregation**:

    - **Function**: Compress variable-length image/text sequences into fixed-length, modality-aligned retrieval embeddings.
    - **Mechanism**: $N$ learnable queries are **appended to the end** of the sequence (unlike CoOp/VPT which prepend), so in decoder-only MLLMs with **causal attention**, these queries can attend to **all** preceding tokens—naturally acting as "global aggregators". The same set of $\mathbf{Q}$ is attached to both image and text ("Shared"), and the same frozen model is used for both; the last $N$ hidden states are taken: $\mathbf{H}^Q_T = \mathbf{H}_T[-N:], \mathbf{H}^Q_I = \mathbf{H}_I[-N:]$, mean-pooled and L2-normalized to get the final embedding $\mathbf{z}_T = \bar{\mathbf{h}}_T / \|\bar{\mathbf{h}}_T\|_2$.
    - **Design Motivation**: (1) Tail appending + causal attention allows queries to "see all context", matching the need for global information in retrieval; prepending (CoOp/VPT) treats queries as conditioning signals, requiring a [CLS]-like summary token—which decoder-only MLLMs lack. (2) Using multiple ($N=20$) queries instead of one provides "multi-head aggregation + averaging", alleviating the information bottleneck of a single last token. (3) Sharing queries across modalities projects image/text into the **same parameterized space**, avoiding the alignment difficulties of dual-tower approaches where each modality learns independent projections.

2. **Frozen Backbone + Query-only Training (Parameter-Efficient Retrieval Adaptation)**:

    - **Function**: **Fully preserve** the MLLM's pretrained knowledge and reasoning, avoiding semantic distortion from contrastive fine-tuning.
    - **Mechanism**: During training, only $\mathbf{Q}$ and $\tau$ are updated; all MLLM attention/FFN/embedding parameters remain untouched. InternVL3-8B has 8B total parameters, but only $N \times D = 20 \times D$ (tens of thousands) are trained. In contrast, LoRA updates millions, and full fine-tuning updates the entire backbone.
    - **Design Motivation**: Directly addresses the diagnostic experiment—MLLMs **already** have aligned multimodal semantic spaces; retrieval is about "activating existing capabilities" rather than "retraining the model". Modifying the backbone can damage this space. Also, the small training set (~$10^4$ parameters) does not require large negative sample batches, avoiding contrastive learning's batch size escalation.

3. **KARR-Bench: Knowledge-aware Reasoning Retrieval Benchmark**:

    - **Function**: Truly tests whether MLLMs can perform retrieval using **implicit knowledge and reasoning**, rather than superficial "red car matches red car" pattern matching.
    - **Mechanism**: Starting from 5,000 COCO test images, a three-stage pipeline: (1) **Visual anchor entity selection**—removing abstract concepts, ensuring each target is visually verifiable; (2) **Knowledge-enhanced query generation**—using GPT-5-mini to encode target identity into **implicit reasoning queries** without mentioning the target name or synonyms (e.g., "the animal with 9 lives" instead of "cat"), yielding ~4,500 candidates; (3) **Manual cross-validation by four annotators**—removing MLLM hallucinations and weak associations, with a 60-70% acceptance rate, resulting in 2,915 high-quality image-text pairs. Queries span six dimensions: Tool & Appliance Utility (18.8%), Contextual & Spatial Relations (18.1%), Functional Relationship (17.4%), Cultural Symbolism (19.4%), Encyclopedic Knowledge (14.9%), Logical & Mathematical (11.4%).
    - **Design Motivation**: Existing COCO/Flickr30K use descriptive captions ("a red car") that directly match visual features, masking the MLLM's reasoning strengths; KARR-Bench requires retrieval systems to perform "implicit knowledge + logic" to succeed, providing a fairer evaluation for MLLM retrievers.

### Loss & Training

Symmetric InfoNCE loss $\mathcal{L} = \frac{1}{2}(\mathcal{L}_{I2T} + \mathcal{L}_{T2I})$, with each direction being a standard in-batch contrastive softmax loss and learnable temperature $\tau$. Four backbones: InternVL3 (1B, 8B) / Qwen3-VL (2B, 4B). On Flickr30K/COCO/KARR-Bench, train on COCO for 5 epochs; on MMEB, train on MMEB-train for 1 epoch; global batch size 1024 (512 for 8B), $N = 20$.

## Key Experimental Results

### Main Results

Comparison with dual-tower models (CLIP, BLIP, FLAME), MLLM-based full fine-tuning baselines (E5-V-7B, VLM2VEC-7B, GME-7B, etc.), and SLQ at various scales.

| Dataset | Method | I→T R@5 | T→I R@5 | Params |
|---------|--------|---------|---------|--------|
| Flickr30K | CLIP ViT-L | 98.3 | 89.0 | Full |
| Flickr30K | VLM2VEC-7B (full FT) | **99.5** | 95.0 | 7B FT |
| Flickr30K | SLQ (InternVL3-8B) | 99.4 | **95.1** | **~10K** |
| COCO 5K | VLM2VEC-7B (full FT) | 88.4 | 73.8 | 7B FT |
| COCO 5K | SLQ (InternVL3-8B) | **89.1** | **79.7** | **~10K** |
| MMEB Overall | VLM2VEC-7B† | 62.9 | — | 7B FT |
| MMEB Overall | UniME-7B† | 66.6 | — | 7B FT |
| MMEB Overall | **SLQ-8B†** | **67.5** | — | **~10K** |

### Ablation Study

| Configuration | Key Metric | Notes |
|---------------|------------|-------|
| Full SLQ (frozen backbone + $N$=20 queries) | Best | — |
| Last token baseline (zero shot) | Succeeds on pattern matching, fails on knowledge reasoning | Diagnostic: query outperforms last token |
| Single query (N=1) | Performance drops | Multiple queries provide wider information bandwidth |
| Full fine-tune | Comparable or weaker, much higher GPU hours | Non-intrusive is better |
| LoRA | Between SLQ and full FT | Still slightly distorts pretrained space |

### Key Findings

- **Remarkable parameter efficiency**: SLQ-8B achieves 67.5 average on MMEB with only ~10K parameters, outperforming 7B full fine-tuned VLM2VEC (62.9) and UniME (66.6)—validating the "activation > retraining" principle.
- **Multimodal shared queries are crucial**: Sharing $\mathbf{Q}$ between image and text forces the backbone to project both modalities into the same latent space, achieving better alignment than dual-tower separate projections.
- **Tail appending + causal attention** is better suited for decoder-only MLLMs than prepending (CoOp/VPT)—causal masking allows appended queries to attend to all context.
- On KARR-Bench, SLQ shows "substantial improvement" over last token baselines, indicating the benchmark effectively distinguishes pattern matching from reasoning ability.

## Highlights & Insights

- **"Activating pretrained capabilities" vs "retraining" paradigm**: The paper provides clear arguments—LLM/MLLM pretraining already learns the required alignment space; only an "interface" is needed to expose it, not intrusive modification.
- **Excellent diagnostic experiment design**: Figure 2's three difficulty levels (pattern matching / knowledge retrieval / logical reasoning) almost single-handedly demonstrate the last token bottleneck and query advantage.
- **KARR-Bench is a needed community tool**: Systematically evaluates "knowledge-aware + implicit reasoning" retrieval, avoiding shortcutting in existing benchmarks.
- This "add a few query tokens + freeze backbone" PEFT strategy is transferable to: retrieval-augmented generation (RAG), vector indexing services, cross-modal re-ranking, multilingual alignment, etc.

## Limitations & Future Work

- The number of queries $N$ is a hyperparameter; the paper fixes $N=20$ without large-scale sweeps; optimal $N$ may vary by task.
- Only validated on COCO/Flickr30K/MMEB/KARR-Bench; **long-document retrieval**, **video retrieval**, **audio-text retrieval**, etc., require further study.
- KARR-Bench queries are generated by GPT-5-mini + manual filtering—GPT generation may introduce stylistic bias; larger samples and multi-LLM generation may be more robust in the future.
- Inference still requires a full MLLM forward pass per image/text (though parameters are not updated)—slower than CLIP-like dual-tower models; SLQ's advantage is mainly in **training cost**, not **inference speed**.
- Fully freezing the backbone means new domain knowledge cannot be absorbed—if the target domain differs greatly from pretraining (e.g., medical images), performance may be limited.

## Related Work & Insights

- **vs VLM2VEC / MMRet / GME / MM-Embed (full FT or LoRA + last token)**: They modify parameters + use `<EOS>` hidden state; SLQ keeps parameters fixed + uses query hidden states. On MMEB, SLQ-8B (67.5) outperforms VLM2VEC-7B (62.9) and UniME-7B (66.6), with orders of magnitude fewer trainable parameters.
- **vs ColPali / VisRAG (multi-vector)**: They use multi-vector representations (finer-grained but higher storage cost); SLQ uses a single vector + multi-query aggregation, which is more concise.
- **vs CoOp / MaPLe / VPT (prompt tuning)**: Those prepend learned tokens for CLIP-like encoder-only models; SLQ appends for decoder-only MLLMs, leveraging causal attention for "global aggregation".
- **vs BLIP-2 Q-Former**: Q-Former introduces an extra cross-attention module; SLQ relies entirely on the MLLM's own self-attention, with zero extra modules.
- **vs E5-V (last token + Matryoshka)**: They also adapt MLLMs for retrieval but use the last token, which suffers from the information bottleneck; SLQ uses multiple queries to overcome this.

## Rating

- Novelty: ⭐⭐⭐⭐ "Tail-appended shared queries + frozen backbone" is a simple and effective design; KARR-Bench construction is also independently valuable
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive comparison across four backbone scales × four benchmarks; lacks more ablations (e.g., $N$ sweep, prompt content effects)
- Writing Quality: ⭐⭐⭐⭐⭐ Clear three-part structure (diagnostic experiment → method → benchmark), Figure 2 is highly convincing
- Value: ⭐⭐⭐⭐⭐ Reduces MLLM retrieval training cost by orders of magnitude, immediately usable in engineering; KARR-Bench provides a reasoning retrieval evaluation tool for the community

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Bridging Modalities via Progressive Re-alignment for Multimodal Test-Time Adaptation (BriMPR)](../../AAAI2026/multimodal_vlm/bridging_modalities_via_progressive_re-alignment_for_multimo.md)
- [\[ICLR 2026\] Steering and Rectifying Latent Representation Manifolds in Frozen Multi-Modal LLMs for Video Anomaly Detection](../../ICLR2026/multimodal_vlm/steering_and_rectifying_latent_representation_manifolds_in_frozen_multi-modal_ll.md)
- [\[ICML 2026\] Latent Reasoning VLA: Latent Thinking and Prediction for Vision-Language-Action Models](latent_reasoning_vla_latent_thinking_and_prediction_for_vision-language-action_m.md)
- [\[CVPR 2026\] FINER: MLLMs Hallucinate under Fine-grained Negative Queries](../../CVPR2026/multimodal_vlm/finer_mllms_hallucinate_under_fine-grained_negative_queries.md)
- [\[NeurIPS 2025\] CyIN: Cyclic Informative Latent Space for Bridging Complete and Incomplete Multimodal Learning](../../NeurIPS2025/multimodal_vlm/cyin_cyclic_informative_latent_space_for_bridging_complete_and_incomplete_multim.md)

</div>

<!-- RELATED:END -->
