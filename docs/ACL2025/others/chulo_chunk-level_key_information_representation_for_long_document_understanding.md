---
title: >-
  [Paper Note] ChuLo: Chunk-Level Key Information Representation for Long Document Understanding
description: >-
  [ACL2025][Long Document Understanding] The core of ChuLo is not simply partitioning long documents into smaller chunks, but rather identifying the most critical semantic phrases globally first, and then re-injecting this key information into each chunk's representation. This preserves both global semantics and fine-grained token information while using compact chunk representations.
tags:
  - "ACL2025"
  - "Long Document Understanding"
  - "Chunk-Level Representation"
  - "Unsupervised Keyphrase Extraction"
  - "Document Classification"
  - "Named Entity Recognition"
date: 2026-05-08
content_hash: b2441f2fd216f883
---

# ChuLo: Chunk-Level Key Information Representation for Long Document Understanding

**Conference**: ACL2025  
**arXiv**: [2410.11119](https://arxiv.org/abs/2410.11119)  
**Code**: [adlnlp/Chulo](https://github.com/adlnlp/Chulo)  
**Area**: Other  
**Keywords**: Long Document Understanding, Chunk-Level Representation, Unsupervised Keyphrase Extraction, Document Classification, Named Entity Recognition

## TL;DR
The core of ChuLo is not simply partitioning long documents into smaller chunks, but rather identifying the most critical semantic phrases globally first, and then re-injecting this key information into each chunk's representation. This preserves both global semantics and fine-grained token information while using compact chunk representations.

## Background & Motivation
Transformers have demonstrated strong performance in document understanding tasks, but as document length increases, the $O(n^2)$ computational complexity of standard self-attention immediately becomes a bottleneck.

Existing approaches to long text processing generally fall into three categories.

The first is truncation, which only retains the first 512 or 2048 tokens.

The second is sparse attention, which restricts each token's attention to a local context to reduce computational overhead.

The third is chunking, which partitions long documents into multiple small chunks for independent encoding, followed by aggregation at the chunk level.

The authors argue that all three categories have distinct drawbacks.

Truncation is highly direct but causes severe information loss, particularly when key information appears in the latter half of the document, leaving the model with no opportunity to perceive it.

Sparse attention extends the manageable context length but trades efficiency by limiting the receptive field, which inherently sacrifices certain cross-segment dependencies.

Standard chunking appears to retain all tokens, but independently processing each chunk often weakens long-range semantic relations between chunks, making it difficult to align document-level themes with local labels.

This issue is even more pronounced in token classification tasks.

Named Entity Recognition (NER), for example, requires not only local context around a specific word but often global semantic clues from the entire document to disambiguate labels. If tokens are directly removed to compress the input, the integrity of the context required for fine-grained labeling is destroyed.

Consequently, the core issue addressed in this paper is not "how to truncate long documents," but rather "how to compress input representations while minimizing the loss of core semantics."

The authors' key insight is a simple yet effective observation: not all tokens in a long document are equally important for document understanding. If the most critical keyphrases can be identified globally first, and then chunk representations can be made more sensitive to these keyphrases, the resulting representation will be far more robust than mechanical split chunking.

## Method
ChuLo can be conceptualized as a "keyphrase-driven chunk representation learning framework." The overall workflow consists of four steps: document chunking, document-level keyphrase ranking, keyphrase-weighted chunk representation construction, and chunk-level Transformer training.

Instead of replacing the Transformer backbone, ChuLo redefines the input units for long documents before sending them to the Transformer. This design allows the model to perceive the core content of the entire document through compressed chunk embeddings without directly processing the full token sequence.

### Overall Architecture
1. Tokenize and split the input document into fixed-length, non-overlapping chunks.
2. Perform unsupervised keyphrase extraction globally to identify the top-$n$ most semantically representative keyphrases.
3. Map these keyphrases back to the original token sequence, assigning higher weights to tokens belonging to these keyphrases.
4. Compute a weighted average of token embeddings within each chunk to form the chunk embedding.
5. Convert the entire document into a sequence of chunk embeddings, and feed them into a Transformer-based chunk attention module.
6. Append a classification head for document classification tasks, or a decoding module to output token labels for token classification tasks.

The authors emphasize that the core contribution of ChuLo lies not in "chunking" itself, but in "how to retain semantic focus after chunking."

### Key Designs
1. **Fixed-Length Non-Overlapping Chunking**
	- Function: Partitions a document $D=(t_0,t_1,\dots,t_{l_D-1})$ of length $l_D$ into several chunks of length $n$. The number of chunks is $m=\lceil l_D / n \rceil$, and the last incomplete chunk is padded with `[PAD]`.
	- Design Motivation: The authors do not pursue complex dynamic chunking. Instead, they use simple and stable fixed-length chunks to compress the input length, ensuring that all tokens are covered and avoiding the direct loss of later document content inherent in truncation.
	- Effect: Once the original token sequence is converted into a chunk sequence, the sequence length processed by subsequent layers decreases from the number of tokens to the number of chunks, significantly reducing computational overhead.

2. **Semantic Keyphrase Prioritization, SKP**
	- Function: Extracts candidate keyphrases globally and ranks them according to their "semantic importance."
	- Candidate Generation: Candidate noun phrases are extracted using a POS-tag pattern defined as $\langle \text{NN.*}|\text{JJ} \rangle^*\langle \text{NN.*} \rangle$, which primarily targets noun phrases potentially preceded by adjectives.
	- Ranking Mechanism: Inspired by PromptRank, the scoring is not restricted to the first segment of the document. Instead, the entire document is cut into multiple segments satisfying the encoder's length limit, and prompt probability scores for each candidate phrase are aggregated across all segments.
	- Prompt Format: A template like "The * mainly discusses $k_i$", where $*$ denotes the document class and $k_i$ represents the candidate keyphrase.
	- Position Penalty: The paper introduces a penalty $r_i=\frac{L_c}{l_d}+\frac{\gamma}{(l_d)^3}$ to adjust keyphrase scores, where $L_c$ is the position of the first occurrence of the phrase and $l_d$ is the document length.
	- Final Score: For each candidate, the prompt probabilities across segments are aggregated as $s_i=r_i \times \sum_j p_{ij}$. The top-$n$ keyphrases are selected as the document's core keyphrases.
	- Why it works: Instead of relying purely on statistical frequency, this approach estimates the degree of semantic match between a phrase and the global document through prompt-based ranking, rendering it superior to methods like YAKE or TextRank in identifying deep "semantic cores" rather than superficial salience.

3. **Keyphrase-Weighted Chunk Representation**
	- Function: Tokens belonging to keyphrases in the source text are labeled as $T_k$, and others are labeled as $T_{nk}$. A weighted average is then computed for all token embeddings within each chunk.
	- Formula:
	  $$
	  w_t=
	  \begin{cases}
	  a, & t \in T_k \\
	  b, & t \in T_{nk}
	  \end{cases}
	  \qquad
	  \mathbf{c}=\frac{\sum w_t \cdot \mathbf{t}}{\sum w_t}
	  $$
	- where $a>b$, meaning that keyphrase token embeddings receive a higher weight in the chunk representation.
	- Why not just keep keyphrase sentences?: The authors aim to retain visibility of all tokens; in particular, token classification tasks cannot simply discard non-keyphrase tokens. Hence, a "weighted retention" strategy is preferred over hard thresholding or deletion.
	- Intuitive Understanding: ChuLo does not compress chunks into a simple average representation, but rather into an "average representation with semantic emphasis."

4. **Chunk-Level Transformer and Task Heads**
	- Function: Feeds the sequence of chunk embeddings into a chunk attention module, followed by classification or sequence labeling.
	- For Document Classification: After chunk-level contextual modeling, a classification head is appended to predict document labels.
	- For Token Classification: The authors detail the use of a BERT-decoder module that leverages the global document context carried by chunk representations to predict token-level labels.
	- Why it works: Following chunk-level compression, the input sequence is significantly shorter than the original token sequence. Consequently, standard Transformer backbones are sufficient for global modeling, eliminating the need for expensive long-context sparse attention structures.

### Loss & Training
The training strategy in the paper is clear-cut and does not rely on overly elaborate tricks.

1. CrossEntropy loss is used for HP, LUN, CoNLL, and GUM.
2. Binary CrossEntropy loss is used for multilabel tasks on Eurlex57k and Inverted Eurlex57k.
3. AdamW is adopted uniformly as the optimizer.
4. Early stopping is applied uniformly based on validation set performance, with a patience of 10.
5. Learning rate search is conducted for each experiment to ensure a fair comparison.
6. The number of top-$n$ keyphrases is set to 15 across most datasets.
7. BERT-base is ultimately selected as the chunk attention backbone, as ablation studies demonstrate its superiority over RoBERTa and Longformer.

The optimal hyperparameter table provided in the paper further demonstrates that ChuLo does not heavily rely on idiosyncratic hyperparameter tuning.

| Dataset | top-$n$ | chunk size | Key token weight $a$ | Non-key token weight $b$ | Batch size |
|---|---|---|---|---|---|
| HP | 15 | 10 | 0.8 | 0.1 | 16 |
| LUN | 15 | 50 | 0.5 | 0.1 | 32 |
| Eurlex57k | 15 | 5 | 0.8 | 0.1 | 16 |
| I-Eurlex57k | 15 | 5 | 0.8 | 0.1 | 16 |
| CoNLL | 15 | 20 | 0.8 | 0.1 | 2 |
| GUM | 15 | 50 | 0.8 | 0.1 | 8 |

## Key Experimental Results
Experiments cover two types of tasks: document classification and token classification.

Document classification datasets include HP, LUN, Eurlex57k, and Inverted Eurlex57k.

Token classification datasets include GUM and CoNLL-2012 document-level NER.

For evaluation metrics, Accuracy is used for HP and LUN, while micro-F1 is used for other tasks.

### Main Results
First, consider the main results for document classification.

| Model | HP | LUN | Eurlex57k | I-Eurlex57k |
|------|----|-----|-----------|--------------|
| BERT | 0.9200 | 0.5797 | 0.7309 | 0.7053 |
| ToBERT | 0.8954 | 0.3697 | 0.6757 | 0.6731 |
| CogLTX | 0.9477 | - | 0.7013 | 0.7080 |
| Longformer | 0.9569 | 0.5552 | 0.5453 | 0.5647 |
| BERT+TextRank | 0.9115 | 0.4880 | 0.7287 | 0.7130 |
| BERT+Random | 0.8923 | 0.3015 | 0.7322 | 0.7147 |
| ChunkBERT | 0.9300 | - | 0.6494 | 0.6294 |
| **ChuLo** | **0.9538** | **0.6440** | **0.7332** | **0.7244** |

Several points in this table are worth noting:

First, the gain is most pronounced on LUN, where ChuLo reaches 0.6440, outperforming BERT (0.5797) by 0.0643. This indicates that ChuLo is indeed superior at extracting critical clues from long news documents.

Second, ChuLo achieves top performance on both Eurlex57k and Inverted Eurlex57k. The Inverted version, in particular, places key information at the end of documents, which strongly demonstrates ChuLo's robustness to late-positioned key content compared to prefix-truncation methods.

Third, Longformer slightly outperforms ChuLo on HP, with a marginal gap of 0.0031. The authors highlight that this gap corresponds to correctly classifying just one additional sample out of 65 test samples, suggesting this is not a structural failure.

Next, consider the token classification results.

| Model | CoNLL | GUM |
|------|-------|-----|
| Longformer (4096) | 0.5560 | 0.9427 |
| BigBird (4096) | 0.5553 | 0.9418 |
| GPT-4o | 0.2290 | 0.3231 |
| Gemini 1.5 Pro | 0.3036 | 0.3262 |
| **ChuLo (All)** | **0.9334** | **0.9555** |

These results are arguably even more compelling than those of document classification.

If ChuLo only provided "better document compression," it might not exhibit such superiority in fine-grained tasks like NER. However, the results show that it jumps from around 0.55 to 0.93 on CoNLL, validating that its chunk representations securely preserve global context beneficial for token-level decisions.

The paper also includes a dedicated analysis of performance on longer documents.

On the LUN dataset, when document length exceeds 2048 tokens, Longformer achieves 0.5306, GPT-4o achieves 0.7143, Gemini 1.5 Pro achieves 0.6531, while ChuLo reaches 0.7959.

On CoNLL, when document length exceeds 8192 tokens, Longformer and BigBird drop to 0.3116 and 0.3106, while GPT-4o and Gemini 1.5 Pro drop significantly to 0.0282 and 0.0584, respectively. In contrast, ChuLo maintains high performance at 0.9206.

These results directly support the authors' argument: the key challenge is not "making the model see more tokens," but rather "making the model see higher information density in the representations."

### Ablation Study
The ablation studies in this paper are concise yet targeted at critical components.

| Configuration | HP | LUN | Description |
|------|----|-----|------|
| Average chunk representation | 0.9538 | 0.5951 | Computes direct average of tokens within the chunk without extracting keyphrases |
| YAKE keyphrases | 0.8769 | 0.5951 | Replaces PromptRank/SKP with statistical keyphrases |
| **PromptRank/SKP keyphrases** | **0.9538** | **0.6440** | Final proposed scheme |
| w/o sentence embedding | **0.9538** | **0.6440** | Excludes additional sentence-level representations |
| + sentence embedding | 0.9076 | 0.5537 | Performance degrades after incorporating sentence-level representations |
| BERT backbone | **0.9538** | **0.6440** | Final backbone |
| RoBERTa backbone | 0.8615 | 0.5906 | Performs worse under the same framework |
| Longformer backbone | 0.8923 | 0.5600 | Long-context backbones enjoy no advantage on chunk sequences |

Three main insights can be drawn from this table:

First, Keyphrase quality is critical. YAKE shows almost no margin over the average baseline on LUN, whereas PromptRank/SKP leads to a notable improvement. This underscores that "semantic-aware keyphrase ranking" dictates the performance ceiling of ChuLo.

Second, More sentence-level embeddings are not necessarily better. The authors hypothesize that directly adding sentence embeddings to chunk representations makes multiple chunks from the same sentence too similar, thereby weakening the model's ability to discern fine-grained differences.

Third, A core premise of ChuLo's viability is that once compressed via chunking, the input is no longer an extremely long sequence. Consequently, long-range backbones like Longformer cannot exhibit their typical advantages, making standard BERT a more suitable fit for this setup.

### Key Findings
1. ChuLo does not dominate all tasks completely, but it achieves best or second-best performance in most long-document scenarios, demonstrating strong generalizability of this input representation strategy.
2. For long document classification, simply scaling the number of visible tokens is unreliable; the performance of Longformer on the Eurlex57k tasks suggests that "seeing more raw tokens" can introduce additional noise.
3. For NER, global document-level context is critical. ChuLo's stable performance on extremely long CoNLL documents demonstrates that chunk representations do not sacrifice token-level discriminability.
4. Zero-shot LLMs are competitive in document classification, but remain highly unstable in long-document NER, often outputting length-mismatched or highly repetitive label sequences.
5. The performance gains of this method stem not just from chunking, but from the synergistic combination of "chunking + semantic highlighting."

## Highlights & Insights
1. The most inspiring aspect of this work is that the authors conceptualize the long-context issue not merely as a "sequence length problem," but rather as an "information density allocation problem." This perspective is highly transferable compared to simply scaling the context window.
2. The design of SKP is elegant. It integrates unsupervised keyphrase extraction with prompt-based semantic scoring, which avoids reliance on annotations while being closer to downstream understanding tasks than purely statistical keyphrase ranking.
3. The weighted average of keyphrase tokens is exceptionally lightweight yet consistently yields performance improvements. Unlike complex routing or retrieval modules, it minimizes engineering overhead and is highly suitable as a reusable front-end component for existing long-document models.
4. ChuLo evaluates both document classification and token classification simultaneously, which provides a more robust validation than many long-context methods that only report results on a single task type.
5. The paper implies a valuable lesson: following representation compression, the backbone model does not necessarily need to be "larger" or possess a "wider context window"; instead, it should align more closely with the structural properties of the compressed input.

## Limitations & Future Work
1. The authors explicitly acknowledge that keyphrase extraction quality directly affects the performance of the entire system. If critical phrases are misidentified, subsequent chunk representations are erroneously amplified.
2. This strategy is primarily validated on classification and NER. Its efficacy has not yet been demonstrated on generative tasks, long-document QA, or retrieval-augmented generation (RAG).
3. The prompt-ranking in SKP requires scoring each candidate phrase individually. Although this is more efficient than running a full-length Transformer, it may still introduce additional overhead for extremely long documents with large candidate sets.
4. The implementation details of the decoding end for token classification are not deeply elaborated, particularly regarding the alignment mechanism between chunk representations and token labels.
5. This method implicitly assumes that "keyphrases serve as stable proxies for document semantics." However, in narrative texts, conversational logs, or multimodal documents, crucial information may not always be captured effectively by explicit keyphrase collections.

## Related Work & Insights
**vs. Truncation**: Truncation assumes that the beginning of a document contains sufficient information. ChuLo explicitly models the entire document, demonstrating superior performance in settings like Inverted Eurlex57k where key information is positioned at the rear.

**vs. Sparse Attention**: Longformer and BigBird modify attention patterns to handle long inputs, optimized strictly at the architectural layer. ChuLo reconstructs input representations first and then feeds them into a standard Transformer, offering a much more lightweight pipeline.

**vs. Standard Chunking**: Methods like ToBERT and ChunkBERT also chunk documents but generally follow a "split-then-aggregate" flow. ChuLo introduces an extra layer of "document-level keyphrase-driven chunk representation calibration," which distinguishes it from standard hierarchical chunking.

**vs. TextRank / Random Selection**: BERT+TextRank and BERT+Random select fragments to supplement input under a restricted budget. In contrast, ChuLo highlights key information via weighting without deleting any non-keyphrase tokens, making it fundamentally better suited for token classification tasks that require fine-grained contexts.

Two key takeaways for future research:

First, "semantic salience" can be adopted as a unifying design principle for long-context compression instead of performing purely token-level pruning.

Second, keyphrase-weighted chunk representations like ChuLo could be transferred to scenarios such as long-document RAG, legal document understanding, and electronic health record sequencing, substituting uniform chunking with keyphrase or event-based anchors.

## Rating
- Novelty: ⭐⭐⭐⭐ Binds unsupervised keyphrase ranking with chunk representation. It is not an entirely brand-new paradigm, but the combination is highly effective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluates both document classification and NER, analyses longer inputs, and delivers complete ablation studies. The chain of evidence is thorough.
- Writing Quality: ⭐⭐⭐⭐ Motivation and experimental narratives are clear, with the appendix providing extensive details on implementation and ablation.
- Value: ⭐⭐⭐⭐ Provides a lightweight, reusable, and high-performance solution for long-document understanding without violently scaling context windows.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Attention Entropy is a Key Factor for Parallel Context Encoding](attention_entropy_parallel_encoding.md)
- [\[ACL 2025\] Measuring the Effect of Transcription Noise on Downstream Language Understanding Tasks](measuring_the_effect_of_transcription_noise_on_downstream_language_understanding.md)
- [\[ACL 2025\] The Harmonic Structure of Information Contours](the_harmonic_structure_of_information_contours.md)
- [\[ACL 2025\] ProxAnn: Use-Oriented Evaluations of Topic Models and Document Clustering](proxann_topic_model_eval.md)
- [\[CVPR 2026\] CHIRP dataset: towards long-term, individual-level, behavioral monitoring of bird populations in the wild](../../CVPR2026/others/chirp_dataset_towards_long-term_individual-level_behavioral_monitoring_of_bird_p.md)

</div>

<!-- RELATED:END -->
