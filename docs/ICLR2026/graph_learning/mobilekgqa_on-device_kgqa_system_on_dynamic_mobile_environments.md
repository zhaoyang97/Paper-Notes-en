---
title: >-
  [Paper Note] MobileKGQA: On-Device KGQA System on Dynamic Mobile Environments
description: >-
  [ICLR 2026][Graph Learning][KGQA] MobileKGQA compresses high-dimensional LLM embeddings into binary hash codes for a GNN reasoning module and pairs it with a step-by-step automatic label generation method. This allows a Knowledge Graph Question Answering (KGQA) system to **train directly on mobile/edge devices and adapt to accumulating user data** for
tags:
  - ICLR 2026
  - Graph Learning
  - KGQA
  - embedding hashing
date: 2026-05-08
content_hash: 721c1f650bdd9730
---
# MobileKGQA: On-Device KGQA System on Dynamic Mobile Environments

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=SjLo76SpoW](https://openreview.net/forum?id=SjLo76SpoW)  
**Code**: [https://github.com/jyahn215/mobileKGQA](https://github.com/jyahn215/mobileKGQA)  
**Area**: Graph Learning / KGQA / On-device Inference  
**Keywords**: KGQA, On-device Training, Embedding Hashing, Distribution Shift, Automatic Label Generation  

## TL;DR
MobileKGQA compresses high-dimensional LLM embeddings into binary hash codes for a GNN reasoning module and pairs it with a step-by-step automatic label generation method. This allows a Knowledge Graph Question Answering (KGQA) system to **train directly on mobile/edge devices and adapt to accumulating user data** for the first time, achieving a 20.3% performance improvement with only 30.4% energy consumption on Jetson Orin Nano.

## Background & Motivation
**Background**: Large amounts of user information (social relations, activity logs, location preferences) are stored and accumulated as Knowledge Graphs (KG) on mobile devices. If on-device LLMs can utilize KGQA to retrieve this structured knowledge, they can provide more personalized, explainable, and hallucination-free answers.

**Limitations of Prior Work**: Existing KGQA systems are unsuitable for on-device deployment—IR-based models require storing high-dimensional embeddings and repeated subgraph construction, leading to high computational costs; SP-based models often generate non-executable logical forms when new relations appear; LLM-based ICL methods require repeated LLM calls with extreme latency; and LLM fine-tuning methods require tuning billions of parameters.

**Key Challenge**: The continuous accumulation of user data triggers significant **distribution shifts**, causing KGQA performance to decay over time. Thus, the system must adapt to new data. However, adaptation requires retraining. Retraining usually occurs on centralized servers, which is hindered by on-device resource constraints and privacy risks associated with uploading personal data. Two unavoidable research questions arise: (1) How to reduce KGQA training resources to levels manageable by edge devices? (2) How to generate labels required for supervised training of new knowledge without leaking data?

**Goal**: To develop the first KGQA system capable of **direct deployment and training on mobile devices**, using minimal parameters (0.136M) to handle resource constraints while possessing the ability to adapt to distribution shifts.

**Core Idea**: **[Hash Compression + Self-Generated Labels]** A hashing module that "maximizes mutual information between original embeddings and hash codes" compresses GB-scale floating-point embeddings into several MBs of binary codes, enabling GNN reasoning within constant computational bounds. Furthermore, a "step-by-step reasoning decomposition" label generation method creates question-answer pairs locally, enabling a closed-loop training and adaptation process entirely on-device.

## Method

### Overall Architecture
MobileKGQA consists of three phases: The **hashing phase** projects LLM embeddings of questions and relations into semantic-preserving binary hash codes; the **retrieval phase** uses a GNN-based reasoning module on these hash codes to predict answer candidates and extract connection paths for an on-device LLM to generate the final answer; the **adaptation phase** uses automatic label generation to create supervision signals when new data arrives, retraining the hashing and reasoning modules to counter distribution shifts.

```mermaid
flowchart LR
    A[Question/Relation embedding<br/>EM Model] --> B[Hashing Module<br/>MLP→BN→tanh→sgn]
    B --> C[Binary Hash Code h∈{-1,1}]
    C --> D[GNN Reasoning Module<br/>ReaRev]
    KG[Knowledge Graph G] --> D
    D --> E[Answer Candidates + Shortest Reasoning Paths]
    E --> F[On-device LLM Selects Path & Generates Answer]
    G[New Data Accumulation<br/>Distribution Shift] -.Triggers.-> H[Step-by-step Label Generation<br/>Sampling→Verbalization→Merging→Masking→Refining]
    H -.Supervision Signal.-> B
    H -.Supervision Signal.-> D
```

### Key Designs

**1. Cosine Similarity-Preserving Embedding Hashing: Compressing floats to binary via Mutual Information.** The hashing module compresses high-dimensional floating-point embeddings $z\in\mathbb{R}^D$ into binary codes $h\in\{-1,1\}^d$ without losing semantics. The authors formalize this as "maximizing mutual information $I(z,h)$ under the constraint $h=\psi(\phi(z))$." Since binary optimization is discrete, they split it into two differentiable mappings—first using an MLP for low-dimensional mapping $\phi$, then batch normalization to center the distribution at zero for a more uniform hash space distribution, and finally $d=\tanh(\mathrm{BN}(\mathrm{MLP}_\phi(z)))$ with $h=\mathrm{sgn}(d)$ for binarization. Theoretically, they prove (Theorem 1): under the assumption that the magnitude and direction of embeddings are independent, $I(z,h)$ is maximized if both $\phi$ and $\psi$ **preserve the cosine similarity between any embedding pairs**. Thus, the optimization objective becomes a log-ratio regression loss approximating the original cosine similarity: $\mathcal{L}_{hash}=\ell_\phi(Z_a,Z_i,Z_j)+\alpha\ell_\psi(d_a,d_i,d_j)$. The brilliance of this design is that the hash codes are only 0.25% the size of the original embeddings, and the training computation is **completely decoupled** from the size of the embedding model—unlike SOTA GNN-RAG which must recompute embeddings during training.

**2. GNN Reasoning on Hash Codes + Shortest Path Retrieval.** Given the question hash $h_q$, relation hashes $h_r$, and graph $G$, the reasoning module (reusing the ReaRev architecture for graph structural inductive bias) outputs node representations $H=\{h_i\}_{i=1}^N=\mathrm{RM}(h_q,h_r,G)$. It is trained using KL divergence with answer nodes as labels: $\mathcal{L}_{reason}=D_{KL}(Q\,\|\,\mathrm{softmax}(WH))$. After predicting answer candidates $\hat a_j$, a **set of shortest paths** $P$ between the question entity $e_q$ and the candidates is extracted and fed to the LLM. Selecting shortest paths is strategic: prior work has proven they are effective heuristics, and limiting the path length prevents exponential search space expansion, optimizing recall under a fixed retrieval budget. Because reasoning occurs on low-dimensional binary codes, the computation and storage overhead of the retrieval phase is extremely low.

**3. Step-by-Step Reasoning for Auto-Labeling: Countering shift with local QA generation.** When new knowledge is accumulated, no ready-made question-answer labels exist, and local resources do not allow for multiple LLM calls or cloud API usage. The authors decompose "generating questions for new triplets" into five explainable steps: **Sampling and Filtering** reasoning paths (using token-to-character ratios to identify and discard encrypted or unnatural entities); **Verbalizing** filtered triplets into natural sentences for better LLM processing; **Merging** these into a single statement describing the answer; **Masking** the answer with a type-hinted placeholder to prevent it from appearing in the question; and finally, the LLM **Generates and Refines** a question from the masked sentence, triggering completion if the starting question entity is missing. Breaking complex reasoning into simple steps improves label quality while significantly reducing output tokens—using only 21% of the tokens and 26% of the time compared to CoT on Phi-4, while achieving higher quality. The generated QA pairs are then used to retrain the modules via Eq(3) and Eq(6), completing the local adaptation loop.

## Key Experimental Results

### Main Results (WebQSP, Training Cost vs. Reasoning Performance)

| Dimension | MobileKGQA | SOTA (GNN-RAG) | Comparison |
|---|---|---|---|
| Tunable Parameters | 0.1M | 0.8–1.7M | Only 9% |
| Training PFLOPs | 0.6 | 2.4–25.1 | Only 7.2% (GTE-large) |
| Training Time | 1.63h | 1.85–2.22h | Shortest |
| Hit / F1 (Gemma2 2B) | 79.8 / 67.0 | 80.0 / 66.9 | Nearly identical |

Across various embedding models and 0.5B–14B on-device LLMs, the gap between MobileKGQA and GNN-RAG is only +0.46 Hit / −0.16 F1 on average. Crucially, its computational cost **does not scale with embedding model size** (under GTE-Qwen2-1.5B, GNN-RAG requires 41.8x the computation). Compared to ICL methods (ToG latency reaches 2400+s), MobileKGQA wins decisively in both latency and performance.

### Jetson Orin Nano Edge Platform (vs. GNN-RAG)

| Mode | Metric | MobileKGQA | GNN-RAG |
|---|---|---|---|
| 7W | Training Time / Energy (Wh) | 2.1 / 11.8 | 5.9 / 28.7 |
| 7W | Hit / F1 | 74.2 / 62.9 | 61.7 / 54.3 |
| 15W | Thermal Throttling | No | Yes |

Using only 30.6% of training time and 30.4% of energy, performance actually increased by +20.3% Hit. It is the only model capable of running the **optimal configuration** on-device without triggering frequency throttling in either power mode.

### Ablation Study

| Experiment | Findings |
|---|---|
| Hash Dimension (Table 5) | Power drops <2.9% at 64bit, near-zero loss at 256bit; Storage saved by 99.75%, Reasoning parameters/PFLOPs reduced by 46.9%/82.2%. |
| Label Quality (Table 6) | ROUGE-L 42.8 / BERTScore 48.7, significantly outperforming RLM and CoT while using 21% of CoT's tokens. |
| Label Ablation (Table 7) | Post-adaptation Total Hit of 64.9, superior to RLM (62.3) and CoT (61.4). |
| Distribution Shift (Table 4) | After two domain adaptations, Total Hit consistently leads all baselines; performance on original domains even improved on CWQ (positive knowledge transfer). |

### Key Findings
- Hash compression saves massive resources with negligible performance loss, validating the theoretical assumption that "preserving cosine similarity preserves semantics."
- Simply reducing hyperparameters (e.g., forcing GNN-RAG smaller) cannot maintain reasoning performance, indicating that hashing strategies are necessary rather than optional for on-device KGQA.

## Highlights & Insights
- **Introduction of embedding hashing to KGQA with a mutual information optimality proof**: This is not just engineering compression; the loss design is derived from the clear theoretical condition that "preserving cosine similarity leads to maximum mutual information."
- **Decoupling of computation and embedding model size**: Fixed-dimension hash codes mean the system can benefit from future, more powerful embedding models for "free," without increasing on-device costs—a structural advantage over GNN-RAG’s "recompute embeddings during training" approach.
- **On-device closed-loop for distribution shifts**: Automatic label generation allows the system to continue learning without sending private data to a server, resolving the conflict between privacy and adaptability in a deployment-ready design.

## Limitations & Future Work
- **Dependence on on-device LLM quality**: Label generation still relies on a local LLM. While quality on 2B-class models exceeds baselines, absolute ROUGE-L/BERTScore metrics (42.8/48.7) still lag behind human experts, and erroneous labels could accumulate bias.
- **Ceiling of the shortest path heuristic**: Using shortest paths to approximate true reasoning paths may miss genuine logical chains in scenarios requiring long-range/multi-hop complex reasoning.
- **Evaluation limited to WebQSP/CWQ benchmarks**: Distribution shifts were simulated by manually partitioning D1/D2/D3. Real-world data shifts on mobile devices are more complex, and generalization requires further verification.
- The authors also note in the ethics section that personalized systems may introduce bias, left as future work.

## Related Work & Insights
- **Three Schools of KGQA**: IR-based (NSM, EmbedKGQA, ReaRev) perform graph reasoning on subgraphs; SP-based (QGG, DecAF, UnifiedSKG) translate questions into query languages like SPARQL; LLM-based are split into ICL (KB-BINDER, ToG, StructGPT) and Fine-tuning (RoG, ChatKBQA, GNN-RAG, SubgraphRAG). This paper sits at the intersection of IR-based and LLM path retrieval.
- **Insights**: The approach of hashing + mutual information preservation can be transferred to other "on-device retrieval-augmented" scenarios (e.g., on-device RAG). Decomposing complex supervision signal generation into explainable steps to reduce tokens has universal value for synthetic data generation under resource constraints.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — The first on-device trainable KGQA; embedding hashing with MI proof and step-by-step labeling are well-integrated and purpose-built.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Tested energy/throttling on real Jetson hardware, across two benchmarks, multiple embeddings/LLMs, and exhaustive ablations for hash dimensions and label quality.
- **Writing Quality**: ⭐⭐⭐⭐ — The two research questions drive the narrative; methods map clearly to motivations; tables/figures are complete; theory is concise with appendix support.
- **Value**: ⭐⭐⭐⭐ — Clear deployment value for personalized on-device KGQA under privacy/resource constraints; provides a reusable engineering and theoretical paradigm for on-device RAG/KGQA.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] GDGB: A Benchmark for Generative Dynamic Text-Attributed Graph Learning](gdgb_a_benchmark_for_generative_dynamic_text-attributed_graph_learning.md)
- [\[ICLR 2026\] Adaptive Mixture of Disentangled Experts for Dynamic Graph Out-of-Distribution Generalization](adaptive_mixture_of_disentangled_experts_for_dynamic_graph_out-of-distribution_g.md)
- [\[ICLR 2026\] Entropy-Guided Dynamic Tokens for Graph-LLM Alignment in Molecular Understanding](entropy-guided_dynamic_tokens_for_graph-llm_alignment_in_molecular_understanding.md)
- [\[ICLR 2026\] One for Two: A Unified Framework for Imbalanced Graph Classification via Dynamic Balanced Prototype](one_for_two_a_unified_framework_for_imbalanced_graph_classification_via_dynamic_.md)
- [\[ACL 2026\] IndustryAssetEQA: A Neurosymbolic Operational Intelligence System for Embodied Question Answering in Industrial Asset Maintenance](../../ACL2026/graph_learning/industryasseteqa_a_neurosymbolic_operational_intelligence_system_for_embodied_qu.md)

</div>

<!-- RELATED:END -->
