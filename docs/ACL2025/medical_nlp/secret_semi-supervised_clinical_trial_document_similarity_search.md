---
title: >-
  [Paper Note] SECRET: Semi-supervised Clinical Trial Document Similarity Search
description: >-
  [ACL2025][Medical LLM][clinical trial] Proposes SECRET, a semi-supervised clinical trial protocol similarity search method. By converting clinical trial documents into Q/A pair representations and combining local (Q/A-level) and global (trial-level) contrastive learning to generate embeddings, it improves recall@1 by 78% relative to the best baseline in full trial search.
tags:
  - "ACL2025"
  - "Medical LLM"
  - "clinical trial"
  - "document similarity"
  - "contrastive learning"
  - "semi-supervised"
  - "information retrieval"
date: 2026-05-08
content_hash: 2b934177db48be95
---

# SECRET: Semi-supervised Clinical Trial Document Similarity Search

**Conference**: ACL2025  
**arXiv**: [2505.10780](https://arxiv.org/abs/2505.10780)  
**Authors**: Trisha Das, Afrah Shafquat, Beigi Mandis, Jacob Aptekar, Jimeng Sun  
**Institution**: University of Illinois Urbana-Champaign, Medidata Solutions  
**Code**: Not open-sourced  
**Area**: Medical NLP  
**Keywords**: clinical trial, document similarity, contrastive learning, semi-supervised, information retrieval

## TL;DR

Proposes SECRET, a semi-supervised clinical trial protocol similarity search method. By converting clinical trial documents into Q/A pair representations and combining local (Q/A-level) and global (trial-level) contrastive learning to generate embeddings, it improves recall@1 by 78% relative to the best baseline in full trial search.

## Background & Motivation

Clinical trials are critical for evaluating the safety and efficacy of new therapies, but their design process is complex and error-prone. Retrieving similar historical trials can provide references for trial design (e.g., target population, eligibility criteria, dosing regimens, adverse event anticipation, etc.). However, existing methods face four core challenges:

**Scarcity of labeled data**: Publicly available labeled data for trial similarity is extremely scarce, whereas supervised methods (e.g., GTSLNet) rely on private datasets.

**Long document issues**: Clinical trial protocols often exceed 1,000 words. Existing methods (e.g., Trial2Vec) still require truncation for long paragraphs, leading to the loss of key information.

**Insufficient local semantic understanding**: Two text snippets containing the same medical entities can have completely different semantics (e.g., "detecting insulin levels to diagnose diabetes" vs. "prescribing insulin to manage diabetes"). Entity matching-based methods fail to distinguish them.

**Inefficient contrastive supervision**: SimCSE using the same document as a positive sample is too simplistic, while Trial2Vec's generation of positive samples by deleting paragraphs may lose crucial information.

SECRET proposes a semi-supervised framework that simultaneously leverages a small amount of labeled data and a large volume of unlabeled data, utilizing Q/A pairs as the representation unit to address the aforementioned problems.

## Method

SECRET consists of three core components:

### 1. Q/A Pair Generation

Convert each clinical trial protocol into a set of Q/A pairs:
- **Long paragraphs** (e.g., eligibility criteria): Use Llama-3.1-8B-Instruct to generate Q/A pairs, extracting key information and significantly compressing document length.
- **Short paragraphs** (e.g., title, diseases, interventions): Use manually predefined questions.
- Core assumption: Two similar trials will share a similar set of Q/A pairs.

### 2. Local Contrastive Learning (Q/A-level)

Perform contrastive training at the Q/A pair granularity, using BioBERT as the backbone encoder:
- **Positive Sample Selection**: For an anchor Q/A pair, select the pair with the highest cosine similarity from the Q/A pool of the same paragraph as the positive sample.
- **Negative Samples**: All other Q/A pairs within the batch.
- **Loss Function**: InfoNCE loss with a temperature parameter tau = 0.1.
- This design ensures that even sentences containing the same medical entities but having different semantics can obtain distinct embedding representations.

### 3. Global Contrastive Learning (Trial-level)

Perform contrastive training at the overall trial level, combining labeled and unlabeled data:
- **Positive Samples for Unlabeled Data**: Randomly drop one Q/A pair from a paragraph containing multiple Q/A pairs to generate the positive sample.
- **Positive Samples for Labeled Data**: Directly use labeled similar trials.
- **Hard Negative Samples**: Trials within the same disease category but not similar.
- **Loss Function**: Combination of pairwise loss and in-batch loss.

Finally, trial embeddings are ranked and retrieved using cosine similarity.

## Key Experimental Results

### Table 2: Full Trial Similarity Search (Core Results)

| Method | P@1 | R@1 | P@5 | R@5 | nDCG@5 | MAP |
|------|-----|-----|-----|-----|--------|-----|
| TF-IDF | 0.363 | 0.244 | 0.217 | 0.687 | 0.522 | 0.501 |
| Trial2Vec | 0.422 | 0.263 | 0.227 | 0.689 | 0.553 | 0.539 |
| **SECRET** | **0.647** | **0.467** | **0.297** | **0.924** | **0.796** | **0.754** |

SECRET leads by a wide margin on all metrics, with recall@1 increasing by 78% relative to Trial2Vec, precision@1 increasing by 53%, and MAP increasing by 40%.

### Table 3: Partial Trial Search (Query by Title Only)

| Method | P@1 | R@1 | R@5 | nDCG@5 | MAP |
|------|-----|-----|-----|--------|-----|
| Trial2Vec | 0.456 | 0.322 | 0.717 | 0.592 | 0.579 |
| **SECRET** | **0.548** | **0.390** | **0.902** | **0.745** | **0.696** |

Under partial query scenarios, SECRET still significantly outperforms all baselines, with recall@2 relative to the best baseline increasing by 29%.

### Table 4: Zero-Shot Patient-Trial Matching (TREC2021)

| Method | P@1 | R@1 | nDCG@5 | MAP |
|------|-----|-----|--------|-----|
| Trial2Vec | 0.608 | 0.129 | 0.618 | 0.695 |
| **SECRET** | **0.710** | **0.158** | **0.666** | **0.744** |

Under the zero-shot setting without patient-trial matching training, SECRET still outperforms all baselines, with precision@1 increasing by 17% and recall increasing by 22%.

### Ablation Study

- Local contrastive learning alone yields the worst performance; global only (Q/A representation) outperforms global only (full-text representation); the combination of both achieves the best performance.
- Experiment on the number of Q/A pairs: Selecting the top-10 Q/A pairs yields the best results; too many will introduce noise, while too few will lose information.
- The training data volume is only 1/4 of Trial2Vec (approx. 10K labeled + 60K unlabeled vs. Trial2Vec's full dataset).

## Highlights & Insights

- **Q/A pair representation** is an elegant solution to the long document issue, compressing verbose protocols into structured, comparable information units.
- The **dual-level contrastive learning** is elegantly designed: local contrastive learning captures fine-grained semantic differences, while global contrastive learning models the overall similarity between trials.
- The **semi-supervised framework** effectively balances annotation costs and performance requirements, outperforming the fully-trained baseline with less than 1/4 of the training data.
- Zero-shot transfer to patient-trial matching tasks still outperforms all baselines, demonstrating excellent generalization capability.
- Case studies indicate that SECRET can better capture exact matches of key attributes such as age and intervention measures.

## Limitations & Future Work

- Only used title, disease, intervention, keywords, outcomes, and eligibility criteria, **excluding important paragraphs such as descriptions and study designs** (due to LLM resource limitations).
- Does not include other clinical trial-related documents such as informed consent forms and adverse event reports.
- Q/A generation relies on LLMs (Llama-3.1-8B), which may introduce inconsistency in generation quality.
- The evaluation dataset is limited in scale (1,420 pairs in the test set) and only contains English data.
- The importance weights of different paragraphs were not explored; all paragraphs are treated with equal weight.

## Related Work & Insights

| Dimension | Trial2Vec | GTSLNet | SECRET |
|------|-----------|---------|--------|
| Learning Paradigm | Self-supervised | Supervised | Semi-supervised |
| Document Representation | Segment-wise Encoding + Merging | Full Text | Q/A Pair Set |
| Contrastive Granularity | Entity-level | - | Q/A-level + Trial-level |
| Training Data Requirements | Large-scale Unlabeled | Large-scale Labeled (Private) | Small-scale Labeled + Unlabeled |
| Long Document Processing | Truncation | Truncation | Q/A Compression |
| Open Source Data | Yes | No | Yes |

## Rating

- Novelty: ⭐⭐⭐⭐ — The combination of Q/A pair representation + dual-level contrastive learning exhibits relatively high originality.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Three tasks, 10 baselines, ablation studies, and case analyses are relatively complete.
- Writing Quality: ⭐⭐⭐⭐ — The problem definition is clear, and the logic between the four challenges and their corresponding solutions is coherent.
- Value: ⭐⭐⭐⭐ — Clinical trial retrieval is an important practical demand, and the method has direct application value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Eliciting Medical Reasoning with Knowledge-enhanced Data Synthesis: A Semi-Supervised Reinforcement Learning Approach](../../ACL2026/medical_nlp/eliciting_medical_reasoning_with_knowledge-enhanced_data_synthesis_a_semi-superv.md)
- [\[ACL 2025\] MedBioRAG: Semantic Search and Retrieval-Augmented Generation with Large Language Models for Medical and Biological QA](medbiorag_semantic_search_and_retrieval-augmented_generation_for_biomedical_lite.md)
- [\[ACL 2025\] Query-driven Document-level Scientific Evidence Extraction from Biomedical Studies](urca_biomedical_evidence_extraction.md)
- [\[NeurIPS 2025\] Document Summarization with Conformal Importance Guarantees](../../NeurIPS2025/medical_nlp/document_summarization_with_conformal_importance_guarantees.md)
- [\[ACL 2025\] A Modular Approach for Clinical SLMs Driven by Synthetic Data with Pre-Instruction Tuning, Model Merging, and Clinical-Tasks Alignment](a_modular_approach_for_clinical_slms_driven_by_synthetic_data_with_pre-instructi.md)

</div>

<!-- RELATED:END -->
