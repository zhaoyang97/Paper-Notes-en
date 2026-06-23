---
title: >-
  [Paper Note] Beyond URLs: Metadata Diversity and Position for Efficient LLM Pretraining
description: >-
  [ICLR 2026][Pretraining][Paper Note] This paper systematically broadens the design space of "metadata conditioning for accelerating LLM pretraining." Beyond the known effectiveness of prepending URLs, the authors discover that **fine-grained** quality scores and domain information can similarly accelerate training. They propose two new mechanisms—"appendi
tags:
  - ICLR 2026
  - Pretraining
date: 2026-05-08
content_hash: 4d18bef9b3851438
---
# Beyond URLs: Metadata Diversity and Position for Efficient LLM Pretraining

**Conference**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=fZ64NwiBpt](https://openreview.net/forum?id=fZ64NwiBpt)  
**Code**: TBD  
**Area**: LLM Pretraining / Data Engineering  
**Keywords**: Metadata Conditioning, Pretraining Acceleration, Fine-grained Metadata, Auxiliary Prediction Tasks, Representation Probing

## TL;DR
This paper systematically broadens the design space of "metadata conditioning for accelerating LLM pretraining." Beyond the known effectiveness of prepending URLs, the authors discover that **fine-grained** quality scores and domain information can similarly accelerate training. They propose two new mechanisms—"appending metadata as an auxiliary prediction task" and "learnable meta tokens"—and use layer-wise probing to reveal how these signals reshape latent representations.

## Background & Motivation
**Background**: Efficiency optimization in LLM pretraining has long focused on "which web data to keep and how much"—relying on heuristic filtering and deduplication (like C4, RefinedWeb, FineWeb) or importance-based data selection to allocate compute. A complementary axis is **injecting document-level metadata** (source, domain, timestamp, etc.) into the input to allow the model to learn representations conditionally. Recent works like MeCo (Gao et al., 2025) and Fan et al. (2025) formalized this as "metadata conditioning": **prepending** simple indicators (e.g., source URLs, domain labels) before a document can save 30–40% of pretraining tokens, with a "cooling" phase during inference to remove metadata dependence.

**Limitations of Prior Work**: Existing evidence almost exclusively supports "prepending URLs" as an effective signal. Systematic comparisons have reported that other readily available metadata (coarse-grained topics, quality metrics) **fail** to show comparable acceleration under the same budget. Consequently, three questions remain unresolved: (1) Can metadata other than URLs accelerate training? (2) Are there effective injection positions other than prepending (e.g., appending, segment heads, sidecars)? (3) How exactly does metadata reshape latent representations during pretraining? Mechanistically, it remains largely a black box.

**Key Challenge**: The previous conclusion that "quality scores/domain info are useless" may not be due to the information itself, but rather its **coarse granularity**. Coarse labels (e.g., only 3 quality levels or 24 topic categories) carry too little discriminative information for the model to learn additional structure.

**Goal**: To systematically explore the "type × position" design space of metadata, identify truly effective signals and positions, and provide mechanistic evidence for "why it works" through probing.

**Key Insight**: The authors observe that all known effective metadata (typically URLs) share a common feature—they encode information at a **very fine granularity** (a full URL nearly uniquely identifies a document). They hypothesize that **fine granularity is the key to acceleration**, rather than the specific semantic type of the metadata.

**Core Idea**: Use "fine granularity" as a unified explanation for metadata acceleration—by making quality scores and domain information sufficiently fine-grained, they become as effective as URLs. Simultaneously, they expand metadata from "prepending conditions" to two new forms: "appending as an auxiliary prediction task" and "learnable meta tokens."

## Method

### Overall Architecture
This paper does not propose a single new model but systemsatically explores how metadata enters pretraining across **two orthogonal dimensions**: **Metadata Type** (URL / Coarse & Fine Quality Scores QS / Coarse & Fine Domain Info DI / Learnable Meta Tokens) × **Injection Position** (Prepend / Append). All metadata is wrapped in a pair of special tokens `<boc>` (begin-of-context) and `<eoc>` (end-of-context). For prepending, it is placed between the `<s>` and the document; for appending, it follows the document. When long documents are segmented, metadata is attached to each segment, with a constant 10% metadata dropout. The key difference in positions lies in **loss handling**: for prepending, metadata tokens are masked out (serving only as a condition); for appending, the metadata loss is **retained** during backpropagation, turning "metadata prediction" into an auxiliary task.

Experiments are conducted on a 1.5B Llama (16 layers) using the FineWeb-Edu corpus with Megatron-LM. This is supplemented by **layer-wise probing**: training 3-layer MLP classifiers on frozen representations to predict document quality/topic/author, translating "acceleration" into "what is being encoded in latent representations."

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["Document + Metadata<br/>(URL / QS / DI / Meta Tokens)"] --> B{"Injection Position?"}
    B -->|Prepend (Masked Loss)| C["Fine-grained Metadata Prepending<br/>Granularity is Key"]
    B -->|Append (Retained Loss)| D["Appending as Auxiliary Prediction Task"]
    A --> E["Learnable Meta Tokens<br/>Masked Loss Induces Quality Clusters"]
    C --> F["Layer-wise Probing<br/>Quality/Topic/Author Analysis"]
    D --> F
    E --> F
    F --> G["Faster Convergence to 100B-token<br/>Baseline Downstream Performance"]
```

### Key Designs

**1. Fine-grained Metadata Prepending: Granularity, not Semantic Type, is Key**

Addressing the issue where quality/domain info were previously deemed ineffective, the authors refine them: QS-coarse uses 3, 4, or 5 (FineWeb-Edu `int_score`), while QS-fine uses the raw regressor score scaled to $\lfloor \text{score} \times 10 \rfloor$ (a two-digit number between 25–50, at least 10x finer). DI-coarse uses WebOrganizer's 24×24=576 classes, while DI-fine uses Llama-3.1-8B to generate open-ended topic/format tags. The results (Table 1, Figure 2 left) are clear: URL and QS-fine reach the 100B-token baseline downstream average using only **60B tokens**, and DI-fine surpasses the baseline with 20B fewer tokens; meanwhile, their coarse-grained versions show almost no change. This supports **Observation 1: Only fine-grained conditioning provides positive acceleration**. Mechanistically, topic probes (Figure 4) reveal that models with fine-grained prepending encode topic information better across all layers. An interesting counter-example is URL dissection: splitting URLs into prefix (`https://`), domain, and suffix shows that attention concentrates heavily on the **prefix** (a typical attention sink), but ablation (Table 2) shows prepending only the prefix provides zero acceleration. The truly useful parts are the domain and suffix, which encode complementary information—demonstrating that "high attention" $\neq$ "high contribution" (**Observation 2**).

**2. Appending Metadata as an Auxiliary Prediction Task: Metadata Prediction as Soft Regularization**

While prepending treats metadata as a **condition**, appending does the opposite—forcing the model to **predict** the quality score/topic after reading the segment. Since the loss is retained, this acts as an auxiliary objective, forcing the model to compress salient sequence information into hidden states to recover the metadata at the end, acting as soft regularization. Experiments (Figure 2 right, Table 1) show appending DI-fine is most helpful, and QS-coarse and URL are also effective, saving about **20% tokens** (**Observation 3**). A counter-intuitive phenomenon arises: **QS-coarse is better than QS-fine when appended**. Theoretically, a QS-fine model should succeed by matching the first digit, but it doesn't. Probing (Figure 7) reveals the QS-fine model **over-specializes** on the auxiliary quality prediction task—performing slightly better on quality probes but weaker on topic probes, suggesting the fine-grained auxiliary task crowds out general capacity. This contrasts with prepending, indicating "fine-grained" is not universally superior but depends on whether it is a condition or a target.

**3. Learnable Meta Tokens: Models Can Encode Quality-Aware Latent Clusters**

The first two designs rely on **existing external** metadata. The authors further ask: can the model **learn** meta-information on its own? They add 5 new tokens `<s1>`–`<s5>` to the vocabulary, prepended to each segment with 0.9 probability (wrapped in `<boc>/<eoc>` with masked loss). Since these 5 tokens are identical in content, any information must be encoded in **attention patterns**. These also bring acceleration (Figure 2 left). Analysis (Figure 9) shows that at the final layer, attention from high-quality documents to `<s4>` is significantly lower than for low-quality documents. Euclidean distance statistics of attention vectors from the first 100 tokens to these meta tokens show **inter-cluster distances consistently greater than intra-cluster distances**—documents of different quality levels exhibit distinguishable attention fingerprints. This is **Observation 4**: LLMs can spontaneously encode quality-aware latent cluster structures on semantically empty learnable tokens.

**4. Layer-wise Probing: Locating Which Concepts are Rewritten at Which Layer**

The primary analytical tool is probing. Probes are trained on three high-level concepts—writing style (proxied by author attribution), topic, and quality—at intermediate layers (e.g., Layer 6). **Observation 5**: Standard pretraining models have the lowest probe accuracy, indicating the weakest latent understanding of these concepts. URL-enabled models (regardless of position) are strongest in writing style and quality, confirming URLs implicitly encode such info. For quality, URL and QS-fine are most effective. Training curve analysis (Figure 8) provides an additional insight: downstream performance is **not significantly correlated** with training loss, except that URL prepending causes a sharper loss drop—attributed to a "copying effect" (URL suffixes acting as page summaries). However, metadata makes the training loss curve smoother with fewer spikes, suggesting it **stabilizes** pretraining.

## Key Experimental Results

### Main Results
Model: 1.5B Llama / FineWeb-Edu. Metrics: Average accuracy across 9 downstream tasks (Arc-C/E, CSQA, MMLU, PIQA, SIQA, HS, LBD, WG).

| Configuration | Position | Avg Score | Description |
| :--- | :--- | :--- | :--- |
| standard | — | 46.7 | 100B-token baseline |
| `<boc><eoc>` empty | Prepend | 46.7 | Isolating special token effects |
| URL | Prepend | **47.7** | Strongest; 60B tokens matches baseline |
| QS-fine | Prepend | 47.3 | Fine-grained quality score is effective |
| DI-fine | Prepend | 47.3 | Fine-grained domain info is effective |
| QS-coarse | Prepend | 46.6 | Coarse-grained provides no acceleration |
| DI-coarse | Prepend | 46.7 | Coarse-grained provides no acceleration |
| Meta Tokens | Prepend | 47.1 | Learnable empty tokens help |
| DI-fine | Append | 47.3 | Strongest append; ~20% token savings |
| QS-coarse | Append | 47.1 | Coarse performs better when appended |

### Ablation Study
Deconstruction of URL components when prepended (Table 2, Avg Score):

| Configuration | Avg | Description |
| :--- | :--- | :--- |
| Full URL | 47.7 | Domain + Suffix are complementary |
| URL Prefix only (`https://`) | 46.6 | Most attention but **no gain** (attention sink) |
| URL Domain only | 47.2 | Gain exists but below full URL |
| URL Suffix only | 46.9 | Encodes topic; complementary to domain |

### Key Findings
- **Granularity is the deciding factor for prepending**: For any metadata type, fine-grained versions consistently outperform coarse ones, turning "ineffective" quality/domain info into effective acceleration signals.
- **High Attention $\neq$ High Contribution**: URL prefixes consume the most attention but provide zero acceleration, whereas useful domains/suffixes receive less attention—a warning against using attention weights for causal explanation.
- **Fine-grained backfires when appended**: QS-fine is inferior to QS-coarse when appended because the model over-specializes on the auxiliary task at the expense of general capacity (validated via cross-probing).
- **Downstream performance is decoupled from training loss**: Except for the "copying effect" of URLs, acceleration does not correlate strongly with loss, suggesting loss alone is insufficient to evaluate metadata benefits.

## Highlights & Insights
- **Unified explanation via "Granularity"**: Reorganizing scattered findings like "URLs work, quality scores don't" into a coherent "fine-grained is what matters" principle, backed by mechanistic evidence from topic probes.
- **Appending = Auxiliary Prediction Task**: This perspective expands metadata injection from simple conditioning to a self-supervised auxiliary goal, revealing that "excessively difficult/fine auxiliary tasks can be counterproductive," a lesson for all multi-task pretraining.
- **Learnable meta tokens induce quality clusters**: Without any external labels, the model encodes quality information into the attention patterns of semantically empty tokens. This is strong evidence that models can "self-distill" data quality signals.
- **Transferable Trick**: The diagnostic process of using layer-wise probing + attention cluster distance to translate "acceleration" into "new latent concepts" can be directly applied to any analysis of pretraining modifications.

## Limitations & Future Work
- The authors admit the **fundamental mechanism remains unclear**—probes show "what concepts are encoded" but not "why these concepts translate to downstream acceleration."
- Scale limitations: Results are only verified at 1.5B / FineWeb-Edu / 100B tokens. Whether fine-grained advantages and the 20% appending savings hold for larger models or longer training is unknown.
- Dependency on external annotators: Relies on FineWeb-Edu regressors, WebOrganizer, and Llama-3.1-8B. The impact of noise/bias in these annotators was not isolated.
- Open question: Whether metadata benefits **post-training**; preliminary attempts were made but were inconclusive.
- Future directions: Designing "position-adaptive" metadata strategies or difficulty annealing for auxiliary tasks to avoid over-specialization.

## Related Work & Insights
- **vs MeCo (Gao et al., 2025)**: MeCo proposed "prepend URL/domain label + cooling phase." This paper extends the scope from URLs to fine-grained quality/domain info and adds appending/meta tokens.
- **vs Fan et al. (2025)**: This paper uses their experimental setup (1.5B Llama, FineWeb-Edu) but refines their "quality/domain metadata is ineffective" conclusion by showing it was a matter of insufficient granularity.
- **vs Source-Aware Training (Khalifa et al., 2024)**: The latter uses appended/repeated document IDs for attribution; this paper adopts the "different positions" idea but targets **pretraining acceleration** and interprets appending as an auxiliary task.
- **vs CTRL (Keskar et al., 2019)**: CTRL uses control codes for conditional generation; this paper shares the "conditioning on structural signals" lineage but focuses on training efficiency and representation structure.

## Rating
- Novelty: ⭐⭐⭐⭐ Not a brand-new paradigm, but the combination of "fine-grained unification + auxiliary task appending + learnable meta tokens" significantly broadens the field.
- Experimental Thoroughness: ⭐⭐⭐⭐ Full type × position matrix + multi-probe analysis + URL ablation. Mechanism evidence is rich; limited only by a single model scale.
- Writing Quality: ⭐⭐⭐⭐ Clear logic organized around 5 Observations, well-supported by figures and tables.
- Value: ⭐⭐⭐⭐ Provides a practical guide for "fine-grained metadata + position choice" in pretraining, highly applicable for data engineering.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Beyond Length: Quantifying Long-Range Information for Long-Context LLM Pretraining Data](beyond_length_quantifying_long-range_information_for_long-context_llm_pretrainin.md)
- [\[ICLR 2026\] Beyond Multi-Token Prediction: Pretraining LLMs with Future Summaries](beyond_multi-token_prediction_pretraining_llms_with_future_summaries.md)
- [\[ICML 2026\] AC-ODM: Actor–Critic Online Data Mixing for Sample-Efficient LLM Pretraining](../../ICML2026/llm_pretraining/ac-odm_actor--critic_online_data_mixing_for_sample-efficient_llm_pretraining.md)
- [\[ICLR 2026\] Scaling with Collapse: Efficient and Predictable Training of LLM Families](scaling_with_collapse_efficient_and_predictable_training_of_llm_families.md)
- [\[ICLR 2026\] Selective Rotary Position Embedding](selective_rotary_position_embedding.md)

</div>

<!-- RELATED:END -->
