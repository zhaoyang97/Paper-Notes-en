---
title: >-
  [Paper Note] Masking in Multi-hop QA: How LMs Perform with Context Permutation
description: >-
  [ACL 2025][LLM (Other)][multi-hop QA] Through systematic document permutation experiments and attention weight analysis, this study reveals that causal masking is a structural bottleneck for decoder-only LLMs in multi-hop QA, and demonstrates that replacing causal masking with a prefix mask significantly improves both performance and robustness.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "multi-hop QA"
  - "causal mask"
  - "attention analysis"
  - "document permutation"
  - "encoder-decoder"
  - "prefix mask"
date: 2026-05-08
content_hash: 5da1bd42921d3b13
---

# Masking in Multi-hop QA: How LMs Perform with Context Permutation

**Conference**: ACL 2025  
**arXiv**: [2505.11754](https://arxiv.org/abs/2505.11754)  
**Code**: [https://github.com/hwy9855/MultiHopQA-Reasoning](https://github.com/hwy9855/MultiHopQA-Reasoning)  
**Area**: LLM/NLP  
**Keywords**: multi-hop QA, causal mask, attention analysis, document permutation, encoder-decoder, prefix mask

## TL;DR

Through systematic document permutation experiments and attention weight analysis, this study reveals that causal masking is a structural bottleneck for decoder-only LLMs in multi-hop QA, and demonstrates that replacing causal masking with a prefix mask significantly improves both performance and robustness.

## Background & Motivation

**Background**: RAG frameworks have become the baseline architecture for search-based conversational agents (such as Copilot, Doubao), where multi-hop QA (MHQA) requires LLMs to perform cross-document reasoning over multiple retrieved documents, which is far more complex than single-hop QA.

**Limitations of Prior Work**: Prior studies have identified the "lost in the middle" problem, where critical information located in the middle of a context is easily overlooked by LLMs. However, these studies primarily focus on single-hop QA; how causal masking affects cross-document reasoning in multi-hop scenarios has not been systematically investigated.

**Key Challenge**: Current mainstream causal decoder-only LLMs (e.g., Qwen, Llama) utilize causal masking during training and inference, meaning preceding tokens cannot attend to subsequent content. In multi-hop reasoning, if the document of the first hop is placed before the document of the second hop, the tokens in the first-hop document cannot encode information from the second hop, even though the reasoning chain might require this bidirectional information flow.

**Key Insight**: This work designs three document permutation dimensions (order, distance, completeness) to compare the performance of three LLM families (Flan-T5 encoder-decoder, Qwen decoder-only, Llama decoder-only) on the MHQA task, and conducts an in-depth analysis of key reasoning behaviors using attention weight distributions.

**Core Idea**: The gold document order is optimal when it aligns with the direction of the reasoning chain; bidirectional attention (prefix mask) effectively alleviates the constraints of causal masking; attention peaks can serve as heuristic signals to filter for the optimal document permutation.

## Method

### Task Definition and Experimental Setup

The input for the MHQA task consists of a question $q$ and $n$ documents (including $m$ gold documents and other distractor documents). The model is required to perform multi-hop reasoning across these gold documents to derive the answer. The experiments are based on the MuSiQue dataset, which includes 2-hop to 4-hop questions, with up to 20 documents per question. The dataset includes 19,938 training samples and 2,417 validation samples. The evaluation metric is Exact Match accuracy (Acc).

The experiments cover four reasoning settings:

- **Answer Only (AO)**: Direct generation of the answer (formatted as $\boxed{answer}$)
- **CoT**: Zero-shot Chain-of-Thought reasoning followed by answer generation
- **Finetuned (FT)**: Fine-tuning on the MuSiQue training set (LoRA, $r=8$, $\alpha=16$, 5 epochs, $lr = 2 \times 10^{-5}$)
- **Finetuned + Bi (FT+Bi)**: Fine-tuning after replacing the causal mask with a prefix mask, allowing the input context to have bidirectional attention

### Three Dimensions of Document Permutation

**1) Gold Document Order (Order)**: Three permutations are designed—Forward (gold documents arranged in the order of the reasoning chain), Backward (gold documents arranged in the reverse order of the reasoning chain), and Original (maintaining the original dataset order). It is hypothesized that causal masking has the least impact under the Forward setting because subsequent tokens can encode information from all preceding hops.

**2) Gold Document Distance (Distance)**: Fixing the Forward order, $i$ distractor documents are inserted between gold documents ($i=0,1,2,3,4,5$), and the last-hop document is fixed at the end of the context. This setup is used to test the impact of document distance on reasoning.

**3) Gold Document Completeness (Completeness)**: The "Remove First" setting removes the first-hop document to verify whether the model is genuinely executing multi-hop reasoning or relying on parametric knowledge to guess the answer.

### Attention Analysis Methods

Two analytical tools are proposed:

**Grouped Attention Weight (GA)**: Tokens are grouped into blocks (instruction, document, question, prediction), and the average attention weights between blocks are computed, elevating token-level analysis to document-level analysis.

**Information Contribution (IC)**: The grouped attention score averaged across all attention heads and answer tokens, measuring the information contribution of each document to the final prediction. A higher IC score indicates that the model relies more heavily on that document during answer generation.

### Prefix Mask Design

The causal mask of the decoder-only model is replaced with a prefix mask—the tokens within the input context (position $\le c$) can attend to each other bidirectionally, while the generation part remains causally constrained. This makes the model resemble an encoder-decoder architecture during the encoding phase while maintaining autoregressive properties during the generation phase.

## Key Experimental Results

### Main Results: MHQA Performance across Different Architectures and Settings

| Model | Answer Only | CoT | Finetuned | FT + Bi |
|------|:-----------:|:---:|:---------:|:-------:|
| Qwen2.5 0.5B | 8.94 | 12.91 | 27.14 | 30.30 |
| Qwen2.5 1.5B | 20.36 | 22.76 | 44.06 | 44.78 |
| Qwen2.5 3B | 19.78 | 24.82 | 50.23 | 52.15 |
| Qwen2.5 7B | 28.59 | 36.24 | 58.05 | **62.96** |
| Qwen2.5 14B | 37.07 | 39.22 | 64.34 | 64.88 |
| Llama3.2 1B | 11.21 | 11.96 | 33.06 | 40.85 |
| Llama3.2 3B | 25.73 | 31.65 | 54.57 | 59.60 |
| Llama3.1 8B | 36.37 | 44.60 | 63.51 | 65.48 |

After fine-tuning, accuracy generally more than doubles. Incorporating the prefix mask yields further improvements across all scales, with Llama3.2 1B jumping from 33.06 to 40.85 (+7.79), demonstrating the most significant gain.

### Encoder-Decoder vs Decoder-Only Zero-Shot Comparison

| Flan-T5 Model | Acc | Corresponding Qwen Acc |
|-------------|:---:|:-----------------:|
| FT5 small (80M) | 20.11 | 8.94 (0.5B) |
| FT5 base (250M) | 28.09 | 20.36 (1.5B) |
| FT5 large (0.8B) | 40.01 | 19.78 (3B) |
| FT5 xl (3B) | 47.33 | 28.59 (7B) |
| FT5 xxl (11B) | 56.43 | 37.07 (14B) |

Flan-T5 comprehensively outperforms decoder-only models with several times larger parameters in the zero-shot setting. For instance, FT5 large (0.8B) achieves 40.01%, surpassing Qwen2.5 7B's 28.59%—a substantial lead despite a roughly 10-fold parameter gap. This advantage stems from bidirectional attention encoding and the quality of Flan instruction tuning data.

### Impact of Document Order and Distance

Fine-tuned models exhibit a clear preference for the Forward order (where documents are aligned with the reasoning chain), showing positive $\Delta_F$ values across all fine-tuned models. Distance experiments show that the performance of non-fine-tuned models degrades significantly as the distance between gold documents increases, whereas fine-tuned models exhibit stronger robustness to distance variations. Key finding: placing Forward-ordered documents at the end of the context yields significant performance gains, consistent with the "lost in the middle" phenomenon.

### Heuristic Improvement via Attention Peaks

By randomly shuffling the document order 20 times for each question and selecting the permutation with the highest peak IC score, the Answer Only accuracy of Qwen2.5 7B increases from 28.59% to 33.7% (+5.1 percentage points) without any additional training. The median IC for correctly answered instances is 2.22, compared to only 1.72 for incorrect ones, indicating that peak IC is an effective signal for distinguishing correct from incorrect predictions.

## Highlights & Insights

- **Small Models Outperforming Large Models**: The zero-shot multi-hop performance of the 80M FT5 small (20.11%) exceeds that of the 500M Qwen 0.5B (8.94%), proving that architectural choices can be more critical than parameter scale for specific tasks.
- **Forward Preference is an Emergent Ability**: The document order in the fine-tuning data is almost uncorrelated with Forward/Backward (Spearman $\rho=0.0013$); however, fine-tuned models spontaneously prefer the Forward order, representing an emergent capability during training.
- **Attention Peak Heuristics**: A training-free document permutation optimization method that achieves a 5.1% improvement simply through multiple inferences and attention analysis. The performance of the optimal permutation is nearly double that of the worst permutation.
- **Models "Do Not Know What They Do Not Know"**: After removing the first-hop document, the fine-tuned model's accuracy on 4-hop questions unexpectedly increases (56.2% $\rightarrow$ 57.2%). This indicates that the model fails to judge whether the evidence chain is complete, showing severe attribution issues.
- **Recency Bias**: All causal decoder-only models exhibit a preference for the last document in their attention distribution, which explains the performance gain obtained by placing gold documents at the end.

## Limitations & Future Work

- The experimental context length is relatively short (mostly $\le$ 4k tokens); the impact of document order and distance might be more pronounced in long-context scenarios.
- Evaluated only on MuSiQue and 2WikiMultihopQA, leaving other MHQA datasets to be verified.
- The attention peak heuristic requires multiple inferences (20 shuffles), leading to high computational overhead; practical applications require optimized sampling strategies.
- Adaptive masking strategies (e.g., dynamically adjusting attention patterns based on document relationships) have not been explored.

## Rating

⭐⭐⭐⭐ A solid experimental analysis work that addresses an important architectural question through systematic permutation experiments and attention analysis. The design of the three permutation dimensions is clever, and the experimental coverage is broad (13 models, 4 settings). Although the attention peak heuristic is simple, it is highly effective. The limitations lie in the lack of evaluation on longer contexts, and the prefix mask solution is not entirely novel.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Do Large Language Models Perform Latent Multi-Hop Reasoning without Exploiting Shortcuts?](do_large_language_models_perform_latent_multi-hop_reasoning_without_exploiting_s.md)
- [\[ACL 2025\] LLMs can Perform Multi-Dimensional Analytic Writing Assessments](llm_writing_assessment.md)
- [\[ACL 2025\] MIRAGE: Exploring How Large Language Models Perform in Complex Social Interactive Environments](mirage_exploring_how_large_language_models_perform_in_complex_social_interactive.md)
- [\[ACL 2025\] Hierarchical Retrieval with Evidence Curation for Open-Domain Financial QA](hierarchical_retrieval_with_evidence_curation_for_open-domain_financial_question.md)
- [\[ACL 2025\] ACT: Knowledgeable Agents to Design and Perform Complex Tasks](act_knowledgeable_agents_to_design_and_perform_complex_tasks.md)

</div>

<!-- RELATED:END -->
