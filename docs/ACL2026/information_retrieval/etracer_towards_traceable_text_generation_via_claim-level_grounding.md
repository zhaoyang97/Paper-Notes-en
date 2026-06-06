---
title: >-
  [Paper Note] eTracer: Towards Traceable Text Generation via Claim-Level Grounding
description: >-
  [ACL 2026][Information Retrieval & RAG][claim-level grounding] eTracer decomposes RAG responses into atomic claims and searches for sentence-level supporting or refuting evidence within the context. Utilizing a three-ste…
tags:
  - "ACL 2026"
  - "Information Retrieval & RAG"
  - "claim-level grounding"
  - "RAG verifiability"
  - "hallucination detection"
  - "biomedical QA"
  - "citation granularity"
date: 2026-05-08
content_hash: c96000223f846972
---

<!-- Generated automatically by src/gen_stubs.py -->
# eTracer: Towards Traceable Text Generation via Claim-Level Grounding

**Conference**: ACL 2026  
**arXiv**: [2601.03669](https://arxiv.org/abs/2601.03669)  
**Code**: https://github.com/chubohao/eTracer  
**Area**: Text Generation / Traceability / Biomedical RAG  
**Keywords**: claim-level grounding, RAG verifiability, hallucination detection, biomedical QA, citation granularity

## TL;DR
eTracer decomposes RAG responses into atomic claims and searches for sentence-level supporting or refuting evidence within the context. Utilizing a three-step pipeline (decomposition $\rightarrow$ embedding retrieval $\rightarrow$ entailment judgment), it outputs a signed score matrix to precisely trace factual origins and quantitatively assess response faithfulness in biomedical scenarios.

## Background & Motivation

**Background**: Current mainstream RAG systems and commercial search engines (Perplexity, Bing Chat) provide responses with citations, yet the citation granularity remains "entire webpage/passage $\Rightarrow$ entire response sentence." Users often must read the entire context to verify a single fact. Subsequent academic methods like inline citation, attribute-then-generate, and TRUE/NLI evaluation are mostly built on the "sentence $\Rightarrow$ sentence" alignment assumption.

**Limitations of Prior Work**: Preliminary user experiments (Appendix A) demonstrate that the average manual verification times for *passages $\Rightarrow$ response*, *passages $\Rightarrow$ sentence*, and *token $\Rightarrow$ token* are 446s, 212s, and 312s, respectively, with accuracy ranging only between 91%–96%. In other words, "finer is not necessarily better": coarse granularity forces excessive reading, while token-level granularity introduced significant noise. The intermediate "sentence $\Rightarrow$ sentence" alignment often fails because a single response sentence frequently carries multiple independent facts.

**Key Challenge**: Response sentences are information-dense complexes (often containing multiple subject-predicate-object triples), whereas contextual evidence consists of single-assertion sentences. Forcing sentence-to-sentence entailment inevitably misses partial sub-facts, leading to low recall and precision. Furthermore, the biomedical field allows for the coexistence of supporting and refuting evidence; traditional binary classification (entailment/non-entailment) cannot represent this "ambiguous" state.

**Goal**: (1) Redefine the semantic unit of grounding—shifting from "sentence" to "claim" (atomic, independent, and verifiable facts); (2) Design a signed grounding function to characterize both evidence importance and polarity; (3) Provide reference-free metrics for evaluation without ground truth, enabling self-monitoring in real-world scenarios.

**Key Insight**: Three critical prior observations (empirically validated in Appendix B): ① Extracted claims should be entailed by the original response (CER $\ge$ 97%); ② A claim and its evidence should exhibit high semantic similarity (cos $\approx$ 0.75); ③ After semantic negation of a claim, the roles of supporting/refuting evidence should flip (PFCR $\approx$ 90%). These three properties also serve as natural proxy metrics for evaluating grounding methods.

**Core Idea**: Replace "sentence $\Rightarrow$ sentence" grounding with "sentence $\Rightarrow$ claim" grounding. A lightweight pipeline of "decomposition + embedding retrieval + NLI polarity judgment" assigns each (claim, context sentence) pair a signed score $\in \{-1, 0, +1\} \times \text{cos sim}$.

## Method

eTracer is a plug-and-play post-processing framework. Given a "LLM-generated response $\mathcal{R}$ + context $S=\{s_i\}_{i=1}^m$", it outputs a "signed score matrix $\tilde{M} \in \mathbb{R}^{p \times m}$" for each sentence. This matrix enables evidence tracing and the calculation of four faithfulness metrics: FCR, ACR, HCR, and UCR.

### Overall Architecture

The inference process follows 9 steps across 3 phases (see Algorithm 1):

1. **Response Preprocessing**: Sentence splitting via NLTK $\rightarrow$ Decomposition model $\mathcal{M}_{dec}$ splits sentences into atomic claims $\rightarrow$ Entailment model $\mathcal{M}_{ent}$ verifies "Response sentence $\models$ claim." If verification fails, the sentence is re-decomposed. Valid claims are encoded into vectors using Qwen3-Embedding-8B.
2. **Context Preprocessing**: Sentence splitting via NLTK $\rightarrow$ Context sentences encoded using the same embedder.
3. **Claim Grounding**: Initial evidence screening using cosine similarity (threshold $\tau$) $\rightarrow$ $\mathcal{M}_{ent}$ predicts Entailment / Contradiction / Neutral for each (claim, candidate evidence) pair $\rightarrow$ Signed function $\psi$ maps the three classes to $\{+1, -1, 0\}$, which is multiplied by the similarity score.

### Key Designs

1. **Sentence $\Rightarrow$ Claim Decomposition + Self-Consistency Check**:
    - **Function**: Decomposes response sentences into "atomic, independent, semantically complete" sets of claims, ensuring each claim is a logical consequence of the response.
    - **Mechanism**: Distilled data $\mathcal{D}_{dec}$ was generated using GPT-5.1 on 182 manually labeled sentence-claim sets to fine-tune Qwen3-14B (LoRA, 4-bit, 10 epochs, lr $2\times 10^{-4}$). During inference, a loop is enforced: decomposition $\rightarrow$ verification of $\mathcal{R} \models c_i$ via $\mathcal{M}_{ent}$. If verification fails, resampling occurs until all claims pass or the limit is reached. The loss is standard conditional NLL: $\max_{\mathcal{M}_{dec}} \mathbb{E}_{(r,\{c_i\})} \log p_{\mathcal{M}_{dec}}(\{c_i\} \mid r)$.
    - **Design Motivation**: If the decomposition model hallucinates even one claim, downstream grounding is permanently contaminated. Forced entailment verification treats "hallucinations at the decomposition stage" as failure modes to be repaired rather than accepted.

2. **Signed Grounding Function $\phi$ (importance $\times$ polarity)**:
    - **Function**: Outputs a scalar for each (claim, context sentence) pair reflecting both "evidence strength" and the direction of "support/refutation."
    - **Mechanism**: Polarity is determined by the entailment model $\mathcal{M}_{ent}$ providing $\psi(s, c) \in \{+1, -1, 0\}$. Intensity is determined by cosine similarity $M_{ij} = \mathbf{e}_{c_i} \cdot \mathbf{e}_{s_j}$ for retrieval screening. The final $\tilde{M}_{ij} = M_{ij} \cdot \psi(s_j, c_i)$ is retained only if $M_{ij} > \tau$. The threshold $\tau$ is chosen based on the cos-sim distribution (default $\tau = 0.5$).
    - **Design Motivation**: Traditional binary NLI conflates "neutral" and "contradiction" into "not support," losing the critical "this evidence actually opposes the claim" signal in medical contexts. The triad of $\{-1, 0, +1\}$ combined with continuous intensity decouples retrieval from judgment and allows accurate calculation of FCR / ACR / HCR / UCR.

3. **Three Reference-Free Metrics (CER / ECSS / PFCR)**:
    - **Function**: Continuously monitors grounding quality in real-world deployments where ground truth citation sets are unavailable.
    - **Mechanism**: ① **CER** (Claim Entailment Ratio) = $\frac{1}{p} \sum \mathbb{I}[\mathcal{R} \models c_i]$, measuring decomposition faithfulness. ② **ECSS** (Evidence-Claim Semantic Similarity) = $\frac{1}{k} \sum \mathrm{Sim}(c, s_i)$, measuring "retrieval-semantic" consistency. ③ **PFCR** (Polarity Flip Consistency Ratio) = $\frac{1}{k} \sum \mathbb{I}[\phi(s_i, c) \approx -\phi(s_i, \neg c)]$, measuring the robustness of polarity identification.
    - **Design Motivation**: Empirical validation in Appendix B shows these properties hold for GT (CER 97%, cos $\approx$ 0.75, PFCR 90%), allowing them to serve as reliable proxies.

### Loss & Training

Two small models were fine-tuned:

- **Decomposition Model** $\mathcal{M}_{dec}$: Base = Qwen3-14B, LoRA + 4-bit, 182 samples, effective batch 256, 10 epochs.
- **Entailment Model** $\mathcal{M}_{ent}$: Base = Qwen3-4B-Instruct-2507, LoRA + 4-bit, 4,267 samples, effective batch 512, 5 epochs.
- **Inference**: No sampling (temperature=0, top-k=1), Qwen3-Embedding-8B as the general embedder; $\tau=0.5$.

## Key Experimental Results

**Dataset**: A manually labeled biomedical grounding ground truth $\mathcal{D}_g$ (100 instances each from PubMedQA, BioASQ-QA, and TracSum) containing 578 response sentences, 1,564 claims, and 4,579 (claim, evidence) pairs. 30/70 split for training/evaluation.

### Main Results

Baselines include sentence-level NLI (DeBERTa), sentence-level instruct-following (Qwen3 / Ministral / Llama), claim-level baselines, and end-to-end claim grounding.

| Method | Granularity | $\mathrm{F1}_e$ (Support) | $\mathrm{F1}_c$ (Refute) | Time (s) |
| :--- | :--- | :--- | :--- | :--- |
| Qwen3-4B-Instruct | Sentence-level | 0.557 | 0.815 | 4.71 |
| Qwen3-14B | Sentence-level | 0.592 | 0.811 | 8.70 |
| Qwen3-4B-Instruct + decomp | Claim-level | 0.639 ↑.082 | 0.817 ↑.002 | 14.18 |
| Qwen3-14B + decomp | Claim-level | 0.660 ↑.068 | 0.860 ↑.049 | 26.02 |
| **eTracer** ($\tau=0$) | Claim-level | **0.709** | **0.946** | 22.19 |
| **eTracer** ($\tau=0.5$) | Claim-level | 0.705 | 0.939 | **14.35** |

Compared to the Qwen3-4B-Instruct baseline, eTracer improves $\mathrm{F1}_e$ by +0.152 (+27%) and $\mathrm{F1}_c$ by +0.131 (+16%). The improvement for refuting evidence is particularly significant.

### Ablation Study

| Configuration | $\mathrm{F1}_e$ | $\mathrm{F1}_c$ | Description |
| :--- | :--- | :--- | :--- |
| w/o $\mathcal{M}_{dec}$ | 0.607 | 0.485 | Sentence-level grounding |
| w/ $\mathcal{M}_{dec}$ (Full eTracer) | 0.705 | 0.939 | Full method |
| Δ | ↑.098 (+16%) | ↑.454 (+94%) | Refuting evidence nearly doubles |

User experiments (4 users $\times$ 12 tasks): S $\Rightarrow$ C (Ours) averaged 116s / 100% accuracy, outperforming the strongest baseline by 1.83x in speed.

### Key Findings

- Removing the decomposition module affects **refuting evidence** ($\mathrm{F1}_c$ -0.454, -94%) far more than supporting evidence ($\mathrm{F1}_e$ -0.098, -16%), indicating that claim granularity is essential for uncovering contradictory opinions.
- The "resample failed claims" mechanism results in a CER of 0.930, far superior to the 0.309 of end-to-end Qwen3-14B, which tends to copy context directly.
- Performance peaks at $\tau = 0.25$, aligning with the observed claim-evidence semantic prior of cos $\approx$ 0.75.

## Highlights & Insights

- **"Fine-grained $\neq$ better"**: User experiments show token-level grounding is slower than sentence-level (312s vs 212s). Granularity should match the "semantic units" used by humans during verification.
- **Interpretability of Reference-free Metrics**: CER captures "fake claim rate," ECSS captures "retrieval accuracy," and PFCR captures "polarity stability," mapping directly to the pipeline's three stages.
- **Decomposition + Self-Check Loop**: Intercepting hallucinations during decomposition rather than after the final response significantly reduces error propagation.
- **Signed Grounding**: The ability to flag "ambiguous" cases (coexisting contradictory evidence) is crucial for clinical evidence-based medicine.

## Limitations & Future Work

- **High Inference Cost**: Claim-level grounding is 1.7–22x slower than sentence-level. While $\tau=0.5$ mitigates this, further acceleration is needed.
- **Domain Specificity**: Evaluation was limited to biomedicine. While components are general, migration risks exist regarding prompt and fine-tuning data scale.
- **Extractive Generation Mismatch**: Forced decomposition may introduce noise when responses heavily copy context (e.g., extractive summarization).
- **Small Evaluation Set**: The 300-instance GT set limits statistical significance.

## Related Work & Insights

- **vs LongCite (Zhang et al. 2025)**: eTracer is a post-hoc framework with zero intrusion into the generation model, offering stronger interpretability through independent retrieval and verification.
- **vs TRUE / NLI (Honovich et al. 2022)**: eTracer addresses the "one sentence, multiple facts" issue that sentence-level NLI fails to solve.
- **Actionable Insight**: The "breakdown + self-consistency check" cycle can be applied to Chain-of-Thought reasoning to intercept intermediate hallucinations.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Query-Level Uncertainty in Large Language Models](../../ICLR2026/information_retrieval/query-level_uncertainty_in_large_language_models.md)
- [\[ACL 2026\] Quantifying and Improving the Robustness of Retrieval-Augmented Language Models Against Spurious Features in Grounding Data](quantifying_and_improving_the_robustness_of_retrieval-augmented_language_models_.md)
- [\[ICLR 2026\] Digging Deeper: Learning Multi-Level Concept Hierarchies](../../ICLR2026/information_retrieval/digging_deeper_learning_multi-level_concept_hierarchies.md)
- [\[ACL 2026\] VisRet: Visualization Improves Knowledge-Intensive Text-to-Image Retrieval](visret_visualization_improves_knowledge-intensive_text-to-image_retrieval.md)
- [\[ACL 2026\] ReasonEmbed: Enhanced Text Embeddings for Reasoning-Intensive Document Retrieval](reasonembed_enhanced_text_embeddings_for_reasoning-intensive_document_retrieval.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[ACL 2025\] Investigating the Robustness of Retrieval-Augmented Generation at the Query Level](../../ACL2025/information_retrieval/investigating_the_robustness_of_retrieval-augmented_generation_at_the_query_leve.md)
- [\[ACL 2026\] Quantifying and Improving the Robustness of Retrieval-Augmented Language Models Against Spurious Features in Grounding Data](quantifying_and_improving_the_robustness_of_retrieval-augmented_language_models_.md)
- [\[ACL 2026\] VisRet: Visualization Improves Knowledge-Intensive Text-to-Image Retrieval](visret_visualization_improves_knowledge-intensive_text-to-image_retrieval.md)
- [\[ACL 2026\] ReasonEmbed: Enhanced Text Embeddings for Reasoning-Intensive Document Retrieval](reasonembed_enhanced_text_embeddings_for_reasoning-intensive_document_retrieval.md)
- [\[ICLR 2026\] Query-Level Uncertainty in Large Language Models](../../ICLR2026/information_retrieval/query-level_uncertainty_in_large_language_models.md)

</div>

<!-- RELATED:END -->
