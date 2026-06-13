---
title: >-
  [Paper Note] SLQ: Bridging Modalities via Shared Latent Queries for Retrieval with Frozen MLLMs
description: >-
  [ICML 2026][Multimodal VLM][Frozen MLLM] SLQ appends a small set of "Shared Latent Queries" $\mathbf{Q}$ to the end of image/text token sequences…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "Frozen MLLM"
  - "Shared Latent Queries"
  - "Knowledge-Aware Reasoning Retrieval"
  - "Contrastive Learning"
  - "KARR-Bench"
date: 2026-05-08
content_hash: b40b12e28d562c47
---

# SLQ: Bridging Modalities via Shared Latent Queries for Retrieval with Frozen MLLMs

**Conference**: ICML 2026  
**arXiv**: [2604.13710](https://arxiv.org/abs/2604.13710)  
**Code**: <https://github.com/CnFaker/SLQ>  
**Area**: Multi-modal VLM / Cross-modal Retrieval / Parameter-Efficient Fine-Tuning  
**Keywords**: Frozen MLLM, Shared Latent Queries, Knowledge-Aware Reasoning Retrieval, Contrastive Learning, KARR-Bench

## TL;DR
SLQ appends a small set of "Shared Latent Queries" $\mathbf{Q}$ to the end of image/text token sequences, utilizing the causal attention of MLLMs to aggregate global context. By **training only a few thousand query parameters**, frozen MLLMs are transformed into retrievers that outperform full fine-tuning and LoRA on COCO/Flickr30K. The authors also release KARR-Bench to evaluate "implicit knowledge reasoning" capabilities.

## Background & Motivation

**Background**: Multi-modal Large Language Models (MLLMs), such as InternVL3 and Qwen3-VL, process interleaved image-text inputs through a unified Transformer, capturing richer cross-modal semantic interactions than dual-tower architectures like CLIP/BLIP. Recent works (GME, MM-Embed, VLM2VEC, MMRet) attempt to convert MLLMs into retrievers to leverage their reasoning capabilities.

**Limitations of Prior Work**: (1) **Invasive fine-tuning**—the common practice of full fine-tuning or LoRA with contrastive objectives causes a mismatch between **generative alignment → discriminative alignment**, distorting pre-trained semantic spaces and leading to semantic degradation. (2) **Training inefficiency**—contrastive learning requires large batches for negative sample diversity, making full fine-tuning of billion-parameter backbones computationally prohibitive. (3) Most baselines use the hidden state of the last `<EOS>` token as the global embedding—however, the last token acts as an "information bottleneck" that fails to compress complex semantics. Diagonal experiments (Figure 2) show it fails completely on implicit reasoning tasks (e.g., "an animal with 2+7 lives" implying a cat).

**Key Challenge**: MLLM pre-training has already aligned vision and language into the **same representation space** (enabling zero-shot VQA). However, models either lack fine-tuning (poor zero-shot retrieval) or undergo significant parameter changes (destroying the pre-trained space)—there is a lack of a "lightweight yet effective" intermediate solution.

**Goal**: (1) **Keep the backbone frozen**—no pre-trained parameters are modified; (2) Use a lightweight mechanism to **elicit** implicit knowledge and reasoning for retrieval; (3) Solve the "which token to use for embedding" problem—avoiding both the last token and simple pooling; (4) Provide a specialized benchmark to distinguish "pattern matching vs. knowledge reasoning."

**Key Insight**: A diagnostic experiment was conducted where a **zero-initialized additional query token** was appended to the sequence in a frozen InternVL3-1B. Its final hidden state was used for retrieval via causal attention. Results showed that while both the query and the last token succeeded in "pattern matching," the last token provided low-discriminative scores for "basic associations" and failed entirely in "logical reasoning," whereas the query succeeded. This suggests MLLMs **already possess** reasoning-retrieval capabilities, blocked only by the last token's information bottleneck.

**Core Idea**: Use a small set of "shared learnable latent queries" as **modality-agnostic global aggregators**. These extract unified retrieval embeddings from image/text sequences via the MLLM's own causal attention—**learning only the queries while freezing the backbone**.

## Method

### Overall Architecture
The framework consists of a frozen MLLM backbone and a small set of $N$ learnable Shared Latent Queries $\mathbf{Q} \in \mathbb{R}^{N \times D}$ (e.g., $N=20$ for InternVL3-8B, with total parameters $\approx$ tens of thousands). Text input: $\mathbf{X}_T = [\mathbf{E}_T; \mathbf{E}_{P_T}; \mathbf{Q}]$ (text embeddings + instruction prompt + shared queries); Image input: $\mathbf{X}_I = [\mathbf{E}_I; \mathbf{E}_{P_I}; \mathbf{Q}]$. Both inputs are processed by the frozen MLLM $\mathcal{M}$ via causal attention. The hidden states at the **last $N$ positions** (corresponding to queries) are taken, and mean pooling + L2 normalization yield embeddings $\mathbf{z}_T, \mathbf{z}_I \in \mathbb{R}^D$. A symmetric InfoNCE loss aligns the two embedding spaces. During inference, these query outputs serve as modality-agnostic global embeddings for retrieval. Only $\mathbf{Q}$ and the temperature $\tau$ are updated.

### Key Designs

1. **Shared Latent Queries + Tail Appending + Causal Attention Aggregation**:
    - **Function**: Compresses variable-length image/text sequences into fixed-length, modality-aligned retrieval embeddings.
    - **Mechanism**: Appending $N$ learnable queries to the **end of the sequence** (instead of prepending like CoOp/VPT) allows them to attend to **all** preceding tokens under the **causal attention** of a decoder-only MLLM, acting as natural "global aggregators." The same $\mathbf{Q}$ is used for both image and text inputs. The final hidden states are $\mathbf{H}^Q_T = \mathbf{H}_T[-N:]$ and $\mathbf{H}^Q_I = \mathbf{H}_I[-N:]$, resulting in $\mathbf{z}_T = \bar{\mathbf{h}}_T / \|\bar{\mathbf{h}}_T\|_2$.
    - **Design Motivation**: (1) Tail appending + causal attention ensures queries "see the entire context," matching the global requirements of retrieval. Prepending makes queries conditioning signals, requiring a summary token like [CLS], which decoder-only MLLMs lack. (2) Using $N=20$ queries rather than one provides a wider information bandwidth. (3) Sharing queries maps image and text into the **same parametric space**, avoiding alignment difficulties seen in dual-tower projection methods.

2. **Frozen Backbone + Parameter-Efficient Retrieval Adaptation**:
    - **Function**: Retains the MLLM's pre-trained knowledge and reasoning capabilities while avoiding semantic distortion.
    - **Mechanism**: Backpropagation updates only $\mathbf{Q}$ and $\tau$; all attention, FFN, and embedding parameters of the MLLM remain fixed. For InternVL3-8B, trainable parameters are only $N \times D$, compared to millions in LoRA or billions in full fine-tuning.
    - **Design Motivation**: Direct response to the diagnostic experiment—MLLMs **already** have an aligned semantic space. Retrieval should elicit existing capabilities rather than "re-teach" the model, which might damage the space. Furthermore, a small parameter set (~$10^4$) requires fewer negative samples to learn effectively.

3. **KARR-Bench: Knowledge-Aware Reasoning Retrieval Benchmark**:
    - **Function**: Specifically evaluates whether MLLMs use **implicit knowledge and reasoning** for retrieval rather than superficial pattern matching.
    - **Mechanism**: Based on 5,000 images from the COCO test set, the pipeline follows three stages: (1) **Visual anchored entity filtering**—ensuring targets are visually verifiable. (2) **Knowledge-enhanced query generation**—using GPT-5-mini to encode target identities into **implicit reasoning queries** without mentioning the target name (e.g., "animal with 9 lives" instead of "cat"). (3) **Human cross-validation**—filtering hallucinations and weak associations, resulting in 2,915 high-quality pairs across 6 dimensions: Tool Utility, Contextual Relations, Functional Relationship, Cultural Symbolism, Encyclopedic Knowledge, and Logical & Mathematical.
    - **Design Motivation**: Standard datasets like COCO use descriptive captions ("a red car") that rely on pattern matching. KARR-Bench requires "implicit knowledge + logic" for successful retrieval, providing a fairer evaluation for MLLM-based retrievers.

### Loss & Training
Symmetric InfoNCE loss $\mathcal{L} = \frac{1}{2}(\mathcal{L}_{I2T} + \mathcal{L}_{T2I})$ is used with a learnable temperature $\tau$. Backbones include InternVL3 (1B, 8B) and Qwen3-VL (2B, 4B). Training lasts 5 epochs on COCO for general benchmarks and 1 epoch for MMEB, with a global batch size of 1024 (512 for 8B) and $N=20$.

## Key Experimental Results

### Main Results
Comparison between dual-tower models (CLIP, BLIP), MLLM-based full fine-tuning baselines (E5-V, VLM2VEC, GME), and SLQ variants.

| Dataset | Method | I→T R@5 | T→I R@5 | Parameters |
| :--- | :--- | :--- | :--- | :--- |
| Flickr30K | CLIP ViT-L | 98.3 | 89.0 | Full |
| Flickr30K | VLM2VEC-7B (full FT) | **99.5** | 95.0 | 7B FT |
| Flickr30K | SLQ (InternVL3-8B) | 99.4 | **95.1** | **~K-level** |
| COCO 5K | VLM2VEC-7B (full FT) | 88.4 | 73.8 | 7B FT |
| COCO 5K | SLQ (InternVL3-8B) | **89.1** | **79.7** | **~K-level** |
| MMEB Overall | VLM2VEC-7B† | 62.9 | — | 7B FT |
| MMEB Overall | UniME-7B† | 66.6 | — | 7B FT |
| MMEB Overall | **SLQ-8B†** | **67.5** | — | **~K-level** |

### Ablation Study

| Configuration | Key Metric | Description |
| :--- | :--- | :--- |
| SLQ Full (Frozen backbone + $N$=20 queries) | Optimal | — |
| Last token baseline (Zero-shot) | Passed pattern match, failed reasoning | Query outperforms last token |
| Single query (N=1) | Performance drop | Multi-query provides wider bandwidth |
| Full fine-tune | Comparable or weaker, higher GPU hours | Non-invasive is superior |
| LoRA | Between SLQ and Full FT | Still slightly distorts pre-trained space |

### Key Findings
- **Stunning Parameter Efficiency**: SLQ-8B achieves an average score of 67.5 on MMEB using only K-level parameters, outperforming full fine-tuned VLM2VEC (62.9) and UniME (66.6), validating the "elicit > retrain" philosophy.
- **Multi-modal Shared Query is Key**: Forcing the backbone to project both modalities into the same latent space using the same $\mathbf{Q}$ yields better alignment than dual-projection strategies.
- **Tail Appending + Causal Attention** is more suitable for decoder-only MLLMs than prepending strategies (CoOp/VPT), as the causal mask allows queries to attend to the entire context.
- SLQ shows "substantial gains" over last-token baselines on KARR-Bench, proving the benchmark's ability to distinguish reasoning from pattern matching.

## Highlights & Insights
- **"Eliciting Capabilities" vs. "Retraining" Paradigm**: The paper provides a clear argument that LLM/MLLM pre-training already possesses the necessary alignment; one only needs an "interface" to expose it without invasive modification.
- **Elegant Diagnostic Design**: The three difficulty levels in Figure 2 (Pattern Matching / Knowledge Retrieval / Logical Reasoning) effectively prove the last token bottleneck and query advantage.
- **KARR-Bench Utility**: Systematizing retrieval evaluation for "knowledge awareness + implicit reasoning" prevents existing benchmarks from being trivialized by "shortcut" scores.
- This PEFT strategy (frozen backbone + few query tokens) is transferable to RAG, vector indexing, cross-modal re-ranking, and multi-lingual alignment.

## Limitations & Future Work
- The number of queries $N$ is a hyperparameter; the paper fixes $N=20$ without a large-scale sweep; optimal $N$ may vary by task.
- Validated only on COCO/Flickr30K/MMEB/KARR-Bench; further verification is needed for **long-document**, **video**, and **audio-text retrieval**.
- KARR-Bench queries generated by GPT-5-mini may include stylistic biases; future versions could use more diverse generators.
- Inference still requires a full MLLM forward pass for each image/text, making it slower than CLIP-style dual-tower models; the advantage lies in **training cost**, not **inference speed**.
- A completely frozen backbone cannot absorb domain-specific knowledge (e.g., medical imaging) if the target domain differs significantly from the pre-training distribution.

## Related Work & Insights
- **vs. VLM2VEC / MMRet / GME / MM-Embed**: These update parameters and use `<EOS>` states; SLQ freezes parameters and uses query states, achieving higher MMEB scores with orders of magnitude fewer trainable parameters.
- **vs. ColPali / VisRAG**: These use multi-vector representations (fine-grained but high storage cost); SLQ uses single-vector output via multi-query aggregation, remaining concise.
- **vs. CoOp / MaPLe / VPT**: These prepend tokens for encoder-only CLIP; SLQ appends tokens for decoder-only MLLMs to leverage "global aggregation" through causal attention.
- **vs. BLIP-2 Q-Former**: Q-Former introduces an additional cross-attention module; SLQ utilizes the MLLM's existing self-attention with zero additional modules.
- **vs. E5-V**: Also converts MLLMs to retrievers but is limited by the last token bottleneck; SLQ solves this via multi-querying.

## Rating
- Novelty: ⭐⭐⭐⭐ "Tail-appended shared queries + frozen backbone" is a concise and effective design; KARR-Bench has independent value.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive comparison across four backbone scales and four benchmarks; could benefit from more ablation on $N$ and prompt content.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear structure (diagnostic → method → benchmark) with highly persuasive figures.
- Value: ⭐⭐⭐⭐⭐ Significantly reduces the cost of MLLM retrieval fine-tuning; immediately applicable for engineering; KARR-Bench provides a critical evaluation tool.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Bridging Modalities via Progressive Re-alignment for Multimodal Test-Time Adaptation (BriMPR)](../../AAAI2026/multimodal_vlm/bridging_modalities_via_progressive_re-alignment_for_multimo.md)
- [\[CVPR 2026\] NaiLIA: Multimodal Nail Design Retrieval Based on Dense Intent Descriptions and Palette Queries](../../CVPR2026/multimodal_vlm/nailia_multimodal_nail_design_retrieval_based_on_dense_intent_descriptions_and_p.md)
- [\[ICML 2026\] Calibrated Multimodal Representation Learning with Missing Modalities](calibrated_multimodal_representation_learning_with_missing_modalities.md)
- [\[NeurIPS 2025\] CyIN: Cyclic Informative Latent Space for Bridging Complete and Incomplete Multimodal Learning](../../NeurIPS2025/multimodal_vlm/cyin_cyclic_informative_latent_space_for_bridging_complete_and_incomplete_multim.md)
- [\[ICLR 2026\] Multimodal Prompt Optimization: Why Not Leverage Multiple Modalities for MLLMs](../../ICLR2026/multimodal_vlm/multimodal_prompt_optimization_why_not_leverage_multiple_modalities_for_mllms.md)

</div>

<!-- RELATED:END -->
