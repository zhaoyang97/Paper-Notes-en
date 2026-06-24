---
title: >-
  [Paper Note] Massive Memorization with Hundreds of Trillions of Parameters for Sequential Transducer Generative Recommenders
description: >-
  [ICLR 2026][Recommender Systems][Lifelong User Sequences] VISTA decouples the target attention of candidates over ultra-long user histories into a two-stage process: first, compressing million-length histories into hundreds of summary tokens to be cached; second, performing lightweight attention only on these cached tokens downstream. This keeps training and inference costs constant and has been deployed on Meta’s recommendation platform serving billions of users.
tags:
  - "ICLR 2026"
  - "Recommender Systems"
  - "Lifelong User Sequences"
  - "Sequence Summarization"
  - "Linear Attention"
  - "Embedding Caching"
  - "HSTU"
date: 2026-05-08
content_hash: b2b21edcd4623610
---

# Massive Memorization with Hundreds of Trillions of Parameters for Sequential Transducer Generative Recommenders

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=LSHSaY4gYM](https://openreview.net/forum?id=LSHSaY4gYM)  
**Code**: To be confirmed  
**Area**: Recommendation Systems / Sequential Modeling / Industrial Deployment  
**Keywords**: Lifelong User Sequences, Sequence Summarization, Linear Attention, Embedding Caching, HSTU  

## TL;DR
VISTA decouples the target attention of candidates over ultra-long user histories into a two-stage process: first, compressing million-length histories into hundreds of summary tokens to be cached; second, performing lightweight attention only on these cached tokens downstream. This keeps training and inference costs constant and has been deployed on Meta’s recommendation platform serving billions of users.

## Background & Motivation
- **Background**: Industrial recommendation is highly dependent on user interaction history sequences. Recently, transformer-based sequence modeling such as HSTU and SIM/TWIN has extended history lengths to 10k–100k, leading to continuous performance improvements.
- **Limitations of Prior Work**: Two main paradigms have critical drawbacks. **Full-sequence modeling** (HSTU) performs full attention on $O(100\text{K})$ histories, which is computationally and latency-prohibitive in industrial environments that process $O(10\text{B})–O(100\text{B})$ samples daily with strict latency caps. **Target-specific sampling** (SIM/TWIN) only retrieves short sub-sequences related to the target, but inference costs scale linearly with the number of candidates, and an information gap exists between short sub-sequences and the full history.
- **Key Challenge**: Extending history to "lifelong" (million-scale) yields gains, but the cost of recalculating attention for every request is unacceptable—**performance requires long sequences, yet costs cannot scale with sequence length or candidate counts**.
- **Goal**: Extend user history to the million-scale while keeping downstream training/inference costs constant, ensuring the solution is viable for industrial infrastructure.
- **Key Insight**: **[Decoupled Computation]** Offload the expensive "history summarization" from the online path—calculate it once during base model training, quantize the summary embeddings, and write them to a KV cache. Downstream training and inference only perform cheap second-stage attention. This strategy trades "more storage for less computation," as GPU compute costs are orders of magnitude higher than storage.

## Method

### Overall Architecture
VISTA (VIrtual Sequential Target Attention) splits traditional "candidate item → full history" target attention into two phases. Phase I uses self-attention to summarize the ultra-long User Interaction History (UIH) into hundreds of token embeddings. Phase II allows the candidate item to attend only to these hundreds of summary tokens for prediction. Crucially, Phase I **only runs during base model training**, and its quantized outputs are exported to an $O(100)\text{TB}–O(1)\text{PB}$ KV cache; during inference, the cache is directly retrieved and de-quantized, bypassing the most expensive summarization calculations.

```mermaid
flowchart LR
    UIH[Ultra-long User History<br/>O(100K)~O(1M) items] --> S1[Phase I: UIH Summarization<br/>Virtual seeds + Quasi-linear attention]
    S1 --> Q[Quantized Export]
    Q --> Cache[(KV Cache<br/>O(100)TB~PB)]
    Cache -.Retrieved during inference.-> S2[Phase II: Target-aware Attention<br/>Standard O(N²) transformer]
    Cand[Candidate Items] --> S2
    S2 --> Pred[CTR / Multi-task Prediction]
    S1 -. Direct connection during training .-> S2
```

### Key Designs

**1. Virtual Seed Summarization: Compressing lifelong history into hundreds of user embeddings.** Phase I introduces a set of randomly initialized, shared "virtual seed embeddings" as learnable parameters. These seeds pass through the summarization module's self-attention alongside the UIH sequence. The output can be interpreted as user embeddings encoding personalized preferences (PCA visualization shows separation of users from different countries). Once these summaries are cached, downstream tasks no longer interact with the original million-scale history.

**2. Quasi-Linear Attention (QLA): Balancing linear complexity and expressivity.** Soft attention is $O(N^2)$, which is infeasible for ultra-long sequences. Original linear attention uses the associative property of matrix multiplication to change $(QK^\top)V$ to $Q(K^\top V)$, reducing complexity to $O(N)$, but it has been proven to lack expressivity. VISTA proposes QLA, which injects non-linear complexity into linear attention using SiLU $\varphi$. The history self-attention is formulated as $O[S]=\varphi(Q[S])\,\varphi(\varphi(K[S])^\top V[S])$ (omitting RowNormalize). A strict rule in recommendation is that **candidates cannot attend to each other**, as this causes label leakage since online candidates are only a subset of logged candidates. Thus, for the candidate query portion, only its attention to history is kept, with a separate diagonal term for "candidate self-attention": $O[T]=\varphi(Q[T])\,\varphi(\varphi(K[S])^\top V[S])+\Delta(\varphi(Q[T]),\varphi(K[T]))\,V[T]$, where $\Delta$ places the row-wise dot product of two matrices on the diagonal. The implementation uses custom Triton kernels to maximize GPU performance.

**3. Generative Sequence Reconstruction Loss: Forcing seeds to memorize the entire history.** To strengthen the "memorization" effect, the authors add a reconstruction loss. A causal transformer decoder (without the softmax layer) takes seed embeddings $s_1{\dots}s_k$ and history item embeddings $u_1{\dots}u_M$ to output $v_1{\dots}v_M$. These are aligned via a shifted Mean Squared Error: $L_{\text{reconstruct}}=\sum_{i=1}^{M-1}\lVert v_i-u_{i+1}\rVert_2^2$. The causal mask ensures $v_i$ only depends on $u_1{\dots}u_i$ to avoid future information leakage, forcing personalized seeds to retain as much information from the entire history as possible. This is inspired by VAEs but is explicitly applied to recommendation here for the first time.

**4. Embedding Delivery System: Making the solution deployable.** VISTA is not just a model but a model-system co-design. The end-to-end pipeline consists of three segments: online training of the source model → dual-path delivery of summary embeddings via Kafka real-time queues and Hive persistence → serving by geo-replicated in-memory KV stores. Summary embeddings are updated every 2 hours, with A/B tests showing performance comparable to real-time summarization. The history is deliberately compressed to the $O(100)\text{TB}$ scale to allow deployment within existing systems.

## Key Experimental Results

### Main Results
Using public datasets (Amazon-Electronics, KuaiRand-1K) and a Minimal Production environment, CTR prediction was performed using the FuxiCTR framework, with only the attention layers replaced:

| Model | Amazon AUC↑ | Amazon NE↓ | KuaiRand AUC↑ | Minimal Prod AUC↑ | Minimal Prod NE↓ |
|------|-------------|------------|---------------|-------------------|------------------|
| DIN | 0.873 | 0.656 | 0.744 | 0.632 | 1.048 |
| HSTU | 0.884 | 0.628 | 0.743 | 0.668 | 1.099 |
| VISTA-w/o-QLA | **0.886** | **0.621** | **0.744** | 0.627 | **1.038** |
| VISTA-w/-QLA | 0.884 | 0.623 | 0.743 | 0.632 | 1.062 |

Industrial offline (NE reduction relative to HSTU baseline, more negative is better): VISTA achieved Eval NE reductions of -0.40% / -1.19% / -2.98% / -2.23% across four tasks (C/E1/E2/E3).

### Ablation Study

| Configuration | C-Task Eval NE↓ | E2-Task Eval NE↓ | Description |
|------|-----------------|------------------|------|
| VISTA (128 seeds, 256D) | -0.40% | -2.98% | Optimal configuration |
| VISTA-128D | -0.29% | -2.51% | Embedding dimension halved → Degradation |
| VISTA-64Seed | -0.37% | -3.01% | Seeds halved |
| VISTA-w/o-Recon | -0.29% | -3.00% | No reconstruction loss → Significant degradation on C-Task |

QLA vs. Standard Self-Attention: QLA allowed sequence length to scale from 6,000 to 16,000 and layers from 3 to 5, resulting in a 5% QPS increase and a slight decrease in NE.

### Key Findings
- **Online A/B Tests** (5% traffic, 15 days, video recommendation): Significant improvements in C-Task (+0.5%), O1 (+0.2%), and O2 (+0.04%) (where 0.01% in O2 is considered a major improvement).
- **94% Inference GPU Reduction**: By caching and serving embeddings instead of recalculating per request, with the advantage scaling with sequence length.
- Seed counts follow a scaling law; QLA significantly reduces training/evaluation time per epoch for long sequences with almost no loss in AUC/NE.

## Highlights & Insights
- **Trading storage for compute** is a robust industrial insight: converting the problem of "whether to recalculate history" into "calculate once, cache, and retrieve" addresses real-world GPU constraints in industrial recommendation.
- **Decoupling base training from downstream training/inference** is a truly reusable paradigm—summary tokens act as a "frozen user representation" that can be used as features by any downstream model, refreshed every 2 hours.
- The diagonal term handling in QLA to prevent candidate cross-attention precisely incorporates recommendation-specific label leakage constraints into the linear attention formula.

## Limitations & Future Work
- **Offline experiments limited by GPU**: Online lengths reach 12,000, but offline runs were capped at 2,000. Public datasets have even shorter sequences (Amazon average length is only 8.9), making the gains of "million-scale lifelong history" primarily supported by industrial data and system arguments rather than easily replicable public evidence.
- **Storage Cost and Refresh Lag**: PB-scale caching and 2-hour update cycles introduct storage overhead and minor de-syncing, which may be unfriendly to smaller teams; the paper acknowledges this as a performance-cost trade-off.
- **Strong Coupling with Internal Infrastructure**: The method is highly dependent on specific systems (Kafka/Hive/multi-region KV stores), requiring significant effort to migrate the delivery pipeline to other platforms.
- Future directions include further optimizing compression techniques and exploring cross-domain generalization.

## Rating
- Novelty: ⭐⭐⭐⭐ The paradigm of two-stage decoupling and cached summaries is pioneering in industrial recommendation; QLA and reconstruction loss are solid implementations of existing concepts.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covered public, industrial offline, and online A/B testing with complete ablations; slight deduction for short public sequences making lifelong history gains difficult to replicate independently.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation-method-system-experiment chain with comprehensive formulas, system diagrams, and engineering details.
- Value: ⭐⭐⭐⭐⭐ Already serves billions of users with a 94% reduction in inference GPU usage, offering direct and substantial value to industrial recommendation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] On the Mechanisms of Collaborative Learning in VAE Recommenders](on_the_mechanisms_of_collaborative_learning_in_vae_recommenders.md)
- [\[ACL 2026\] What Makes LLMs Effective Sequential Recommenders? A Study on Preference Intensity and Temporal Context](../../ACL2026/recommender/what_makes_llms_effective_sequential_recommenders_a_study_on_preference_intensit.md)
- [\[ICLR 2026\] CollectiveKV: Decoupling and Sharing Collaborative Information in Sequential Recommendation](collectivekv_decoupling_and_sharing_collaborative_information_in_sequential_reco.md)
- [\[ICLR 2026\] Continual Low-Rank Adapters for LLM-based Generative Recommender Systems](continual_low-rank_adapters_for_llm-based_generative_recommender_systems.md)
- [\[AAAI 2026\] Inductive Generative Recommendation via Retrieval-based Speculation](../../AAAI2026/recommender/inductive_generative_recommendation_via_retrieval-based_speculation.md)

</div>

<!-- RELATED:END -->
