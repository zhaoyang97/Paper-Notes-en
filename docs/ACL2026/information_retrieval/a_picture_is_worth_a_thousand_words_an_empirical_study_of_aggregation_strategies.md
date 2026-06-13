---
title: >-
  [Paper Note] A Picture is Worth a Thousand Words? An Empirical Study of Aggregation Strategies for Visual Financial Document Retrieval
description: >-
  [ACL 2026][Information Retrieval & RAG][visual RAG] Through a meticulously designed diagnostic benchmark for financial documents (single-digit perturbation + text masking)…
tags:
  - "ACL 2026"
  - "Information Retrieval & RAG"
  - "visual RAG"
  - "single-vector aggregation"
  - "ColPali"
  - "MinPatch diagnosis"
  - "financial documents"
date: 2026-05-08
content_hash: 781efe5e3f732330
---

# A Picture is Worth a Thousand Words? An Empirical Study of Aggregation Strategies for Visual Financial Document Retrieval

**Conference**: ACL 2026 Findings  
**arXiv**: [2605.14581](https://arxiv.org/abs/2605.14581)  
**Code**: The paper does not provide an open-source link in the abstract.  
**Area**: Information Retrieval / Visual Document Retrieval / VLM Diagnosis  
**Keywords**: visual RAG, single-vector aggregation, ColPali, MinPatch diagnosis, financial documents

## TL;DR
Through a meticulously designed diagnostic benchmark for financial documents (single-digit perturbation + text masking), this study empirically proves that "aggregating VLM patch tokens into a single vector" causes significant semantic differences (e.g., $\$1.2M$ vs $\$7.2M$) to collapse into nearly identical vectors with cosine similarities $> 0.99$. The root cause is "global texture dominance," which various mitigation strategies and retrieval-tuned embeddings fail to resolve.

## Background & Motivation
**Background**: The mainstream RAG approach in the financial domain is "OCR/PDF parse → linear text." However, flattening tables leads to loss of row-column alignment and decreased retrieval accuracy. A new generation of visual RAG models (ColPali, VisRAG, DSE) treats pages as images and uses VLM visual encoder patch tokens for retrieval.

**Limitations of Prior Work**: **Multi-vector** solutions like ColPali require storing hundreds of patch tokens per page, leading to explosive storage costs. **Single-vector** solutions like DSE aggregate patch tokens into one vector, which is cost-effective but may lose critical numeric or textual information. Life-long diagnostic evidence on whether, what, and why information is lost has been missing.

**Key Challenge**: Financial documents differ fundamentally from natural images—key semantics are encoded in sparse numbers/entities (a single digit change alters the entire meaning), but in visual signals, these represent only a few pixels. Background layouts (table lines, logos, headers) dominate. Aggregation operations favor background visual salience, smoothing out sparse numeric signals.

**Goal**: (1) Quantify the severity of information loss in single-vector aggregation for financial documents; (2) Identify the failure mechanism; (3) Verify the effectiveness of simple mitigation strategies.

**Key Insight**: The retrieval problem is framed as a "sensitivity analysis" by constructing counterfactual document pairs (original vs. modified single fields) to see if the encoder can distinguish them. If indistinguishable, a "MinPatch" (worst-case patch-level similarity) probe is used to determine at which layer the signal is smoothed out.

**Core Idea**: "First use MinPatch to prove that signals exist at the encoder patch level, then prove that aggregation removes them," while identifying "global texture dominance" as the root cause via Signal/Noise image ablation.

## Method

### Overall Architecture
This work is a **diagnostic empirical study**. Instead of proposing a new model, it constructs a benchmark and scoring mechanism probes to reveal failure modes. The pipeline is as follows:

1.  **Construct counterfactual document pairs** (micro-numeric changes, macro-numeric changes, text masking).
2.  **Run multiple VLM encoders** (Qwen2.5-VL 7B/32B, LLaVA-v1.5, Phi-3.5-Vision, DeepEncoder + two retrieval-tuned models: Qwen3-VL-Embedding-8B, GME-Qwen2-VL-7B) to extract patch sequences.
3.  **Test original vs. counterfactual similarity** using 5 scoring mechanisms (Mean Pooling / Max Pooling / MaxSim / MeanPatch / MinPatch).
4.  **Visual Attention Analysis**: Modify documents to "tables only + solid background" (Signal) or "tables erased + template preserved" (Noise) to see which the aggregate vector more closely resembles.
5.  **Mitigation attempts** (VarWgt: Variance Weighting, AttnGd: Attention Guidance, TopK-R: Removing top-$k$ patches) to verify if simple fixes suffice.

### Key Designs

1.  **Three-layer Sensitivity Benchmark (micro-semantic / macro-semantic / text sensitivity)**:
    - **Function**: Measure the discriminative power of "encoder + aggregation" for critical financial information using controlled perturbations.
    - **Mechanism**: Micro changes involve small numeric shifts ($5.21 \to 5.29$, $19.65\% \to 19.54\%$); macro changes involve large shifts ($5.21 \to 9.99$, $13,499 \to 99,999$); text sensitivity uses Zeiler-Fergus semantic occlusion, comparing "Revenue increased by $\$1.4$ billion" vs. the same position covered by [MASK] background color. 600 diagnostic samples in total.
    - **Design Motivation**: A single digit change is enough to subvert decision semantics in financial scenarios. Such fine-grained perturbations are an "empty zone" not covered by natural image benchmarks.

2.  **MinPatch Diagnostic Probe—Separating Encoder and Aggregation Responsibilities**:
    - **Function**: Identify the maximum local difference noticed by the encoder by taking the minimum cosine similarity of spatially aligned patch pairs: $S_{\text{min}} = \min_i \cos(v_i^A, v_i^B)$.
    - **Mechanism**: MinPatch is not a practical retrieval metric but a diagnostic probe. If MinPatch drops significantly (e.g., to $0.51$ for macro-semantic or $0.09$ for text sensitivity) while Mean/Max Pooling remains $> 0.99$, the failure is **entirely due to aggregation**.
    - **Design Motivation**: This cleanly separates "encoder responsibility" from "aggregation responsibility."

3.  **Signal / Noise Image Ablation—Locating Global Texture Dominance**:
    - **Function**: Measure whether the aggregate vector is closer to the Signal (data only) or Noise (layout only) image to prove aggregation focuses on layout.
    - **Mechanism**: Calculate $\cos(\text{reference}, \text{Signal})$ vs. $\cos(\text{reference}, \text{Noise})$, defining $\text{Gap} = \text{sim\_to\_data} - \text{sim\_to\_layout}$. Qwen2.5-VL-7B showed a Gap of $-0.22$ on FinQA, indicating the aggregate vector is more similar to layout-only images.
    - **Design Motivation**: Physically converting semantic failure into measurable visual signal differences provides a clean root cause analysis.

## Key Experimental Results

### Main Results: Single-vector Aggregation vs. MinPatch Diagnosis (Similarity on FinQA; closer to 1.0 means more "blind")

| Test | Mean Pool | Max Pool | MaxSim | MeanPatch | **MinPatch** |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Micro-semantic ($5.21 \to 5.29$) | $> 0.99$ | $> 0.99$ | $> 0.99$ | $> 0.99$ | **0.71** |
| Macro-semantic ($5.21 \to 9.99$) | $> 0.99$ | $> 0.99$ | $> 0.99$ | $> 0.99$ | **0.51** |
| Text sensitivity ([MASK]) | $> 0.99$ | $> 0.99$ | $> 0.99$ | $> 0.99$ | **0.09** (Qwen2.5-32B) |

**Key Finding**: After aggregation, all mechanisms become "blind" ($> 0.99$), while **MinPatch exposes the hidden signal**, proving it exists at the patch level but is smoothed out by aggregation.

### Retrieval-tuned embedding models also fail

| Sensitivity | Qwen3-VL-Embedding-8B | GME-Qwen2-VL-7B-Instruct |
| :--- | :--- | :--- |
| Micro-Semantic (FinQA) | 0.9992 | 0.9970 |
| Macro-Semantic (FinQA) | 0.9976 | 0.9906 |
| Text Sensitivity (FinQA) | 0.9799 | 0.9363 |

Specialized embedding models remain helpless, proving the failure is an **inherent architectural issue of single-vector representation**, not a training objective issue.

### Ablation Study: Mitigation Strategies (Macro-Semantic on FinQA; closer to 1.0 is failure)

| Encoder | VarWgt | AttnGd | TopK-R | Conclusion |
| :--- | :--- | :--- | :--- | :--- |
| Qwen2.5-VL-7B | 0.9997 | 0.9998 | 0.9998 | Ineffective |
| LLaVA-v1.5 | 1.0000 | 0.9999 | 0.9999 | Ineffective |

**Key Findings**:
- **Global texture dominance** is the structural explanation: aggregation favors background textures, drowning out sparse numeric signals.
- **DeepSeek-DeepEncoder is worse**: It performs best on MinPatch (least sensitive) because OCR-optimized encoders learn "pixel-level invariance," which conflicts with financial retrieval goals.
- **TAT-DQA is harder than FinQA**: Multi-page formats and dense tables shrink the Gap to $< 0.05$, meaning even layout distinction becomes difficult.

## Highlights & Insights
- **Paradigm for Diagnostic Papers**: This work does not propose a new method but provides conclusive negative evidence against a widely used architecture. The MinPatch and Signal/Noise probes are clean and powerful.
- **Ignored Domain Attribute**: The fact that "a single digit change equals a huge semantic difference" is often masked by "retrieval insensitivity" in natural image benchmarks.
- **Architecture over Training**: The revelation that "retrieval-tuned $\neq$ retrieval-safe" challenges industry assumptions and suggests that the research path should pivot toward multi-vector retrieval or learned aggregation.

## Limitations & Future Work
- Dataset coverage lacks invoices, balance sheets, and handwritten notes.
- Lacks full retrieval metrics (Recall@k, nDCG) impact.
- Only simple mitigation strategies were tested; learned aggregation (e.g., perceiver resampler) remains unexplored.
- The conclusion is domain-specific; natural images do not exhibit global texture dominance to the same degree.

## Related Work & Insights
- **vs. ColPali**: Validates that ColPali's late interaction path (preserving all patches) is more suitable for financial documents despite the cost.
- **vs. DSE**: Serves as a direct warning for DSE-style single-vector dense embedding approaches in high-precision domains.
- **vs. Zeiler-Fergus Occlusion**: Adapts occlusion by using background-colored masks to ensure perturbations are semantic rather than based on visual salience.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Strong methodology for diagnostic probes.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive coverage across encoders and strategies.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Rigorous logical progression.
- **Value**: ⭐⭐⭐⭐⭐ High value for visual RAG deployment in financial/legal/medical industries.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] When Retrieval is Ineffective in Biomedical RAG: A Large-Scale Empirical Study](when_retrieval_doesnt_help_a_large-scale_study_of_biomedical_rag.md)
- [\[ACL 2026\] Prune-then-Merge: Towards Efficient Multi-Vector Visual Document Retrieval](sculpting_the_vector_space_towards_efficient_multi-vector_visual_document_retrie.md)
- [\[ACL 2026\] Is Agentic RAG Worth It? An Experimental Comparison of RAG Approaches](is_agentic_rag_worth_it_an_experimental_comparison_of_rag_approaches.md)
- [\[ACL 2026\] Navigating Large-Scale Document Collections: MuDABench for Multi-Document Analytical QA](navigating_large-scale_document_collections_mudabench_for_multi-document_analyti.md)
- [\[ACL 2026\] ReasonEmbed: Enhanced Text Embeddings for Reasoning-Intensive Document Retrieval](reasonembed_enhanced_text_embeddings_for_reasoning-intensive_document_retrieval.md)

</div>

<!-- RELATED:END -->
