---
title: >-
  [Paper Note] A Retrieval-Based Approach to Medical Procedure Matching in Romanian
description: >-
  [ACL 2025][Medical LLM][Medical Procedure Matching] By modeling Romanian medical procedure name matching as a retrieval problem rather than a classification problem, under an extreme long-tail scenario of 39,097 standard entries (50% with only a single sample), this work compares BM25 sparse retrieval with three dense embeddings (mE5/RoBERT/BioClinicalBERT). After fine-tuning via metric learning, mE5 achieves 85.2% Acc@1. In real-world deployment…
tags:
  - "ACL 2025"
  - "Medical LLM"
  - "Medical Procedure Matching"
  - "Retrieval"
  - "Sentence Embeddings"
  - "Metric Learning"
  - "Low-resource Languages"
date: 2026-05-08
content_hash: a732544c48c345ac
---

# A Retrieval-Based Approach to Medical Procedure Matching in Romanian

**Conference**: ACL 2025  
**arXiv**: [2503.20556](https://arxiv.org/abs/2503.20556)  
**Code**: None  
**Area**: Medical NLP  
**Keywords**: Medical Procedure Matching, Retrieval, Sentence Embeddings, Metric Learning, Low-resource Languages

## TL;DR

By modeling Romanian medical procedure name matching as a retrieval problem rather than a classification problem, under an extreme long-tail scenario of 39,097 standard entries (50% with only a single sample), this work compares BM25 sparse retrieval with three dense embeddings (mE5/RoBERT/BioClinicalBERT). After fine-tuning via metric learning, mE5 achieves 85.2% Acc@1. In real-world deployment, verification by doctors yields 94.7% accuracy, performing 1200 times faster than manual matching.

## Background & Motivation

**Background**: Medical procedure name standardization is a crucial step in insurance claim systems. Different clinics use various nomenclatures for the same procedure—for instance, "polypectomy" and "polyp resection" refer to the same operation but have completely different spellings. Currently, many insurance companies still rely on manual matching, which is inefficient and prone to high error rates. According to a 2024 industry report, 46% of claim denials stem from data and coding errors.

**Limitations of Prior Work**: Existing works (Tavabi 2024, Levy 2022, Zaidat 2024) model medical procedure matching as a classification problem—assigning pathology reports or surgical records to predefined CPT codes. However, these methods typically involve only 42 to 100 categories and are entirely designed for English and the US CPT system. Once there is a need to process tens of thousands of standard entries under extreme long-tail distributions, the classification paradigm becomes inapplicable.

**Key Challenge**: The Romanian healthcare system contains 39,097 standard procedure entries, of which 50% (19,493) correspond to only a single clinic description. This extreme long-tail makes it nearly impossible for classification models to learn effective decision boundaries. Furthermore, as a low-resource language, Romanian lacks pre-trained medical models, and generic Romanian language models (RoBERT) lack adaptation to the medical domain.

**Goal**: (1) How to achieve high-accuracy matching in an extreme long-tail scenario with 39K+ categories and 50% single-shot instances? (2) Which embedding model is most effective when no Romanian medical pre-trained model is available? (3) Which strategy is optimal among sparse, dense, and hybrid retrieval?

**Key Insight**: The authors observe inherent flaws in classification methods—the number of classes is fixed, adding new procedures requires retraining, and class imbalance leads to poor generalization. In contrast, the retrieval paradigm naturally supports a variable number of classes; adding new entries only requires embedding and saving them into a vector database without modifying the model architecture.

**Core Idea**: Transforming medical procedure matching from a classification paradigm to a retrieval paradigm, utilizing metric learning to fine-tune sentence embedding models to pull synonymous procedures closer and push different procedures further apart in the vector space.

## Method

### Overall Architecture

The entire system is a standard semantic retrieval architecture: all entries from the standard terminology list are embedded into vectors and stored in a Milvus vector database. Non-standard procedure descriptions from clinics serve as queries, and similarity retrieval returns the top-k most similar standard entries. The system supports two indexing modes: storing only standard terminology entries, or storing both standard terminology entries and existing clinic description-terminology mapping pairs. The latter leverages historical matching information to expand the retrieval database, significantly improving recall.

### Key Designs

1. **BM25 Sparse Retrieval Baseline**:

    - **Function**: Traditional text matching based on the bag-of-words model.
    - **Mechanism**: The Romanian text is preprocessed first—removing diacritics, performing stopword removal, and stemming. Then, BM25 is used to calculate the token-level inner product similarity between the query and indexed entries.
    - **Design Motivation**: Serving as a baseline to verify the performance upper bound of pure word matching in medical terminology matching. BM25 performs passably when vocabulary overlap is high, but fails to understand semantic equivalence (e.g., "polypectomy" and "polyp resection") and cannot handle numerical threshold differences (e.g., ">10" and "<10" representing different procedures).

2. **Dense Embeddings + Metric Learning Fine-Tuning**:

    - **Function**: Generating semantic embeddings using pre-trained language models and aligning the representation spaces of clinic descriptions and standard terminologies via metric learning.
    - **Mechanism**: Three models are selected: mE5-large (state-of-the-art multilingual retrieval, natively supporting sentence-pair similarity computation), RoBERT-large (Romanian-specific BERT, obtaining sentence representations by pooling token embeddings), and BioClinicalBERT (medical English pre-training, also obtaining sentence embeddings via pooling). Fine-tuning is conducted using MultipleNegativesRankingLoss: using the clinic description $a_i$ as the anchor, the corresponding standard entry $p_i$ as the positive sample, and all other standard entries $p_j (j \neq i)$ in the batch as negative samples, maximizing the cosine similarity of positive pairs and minimizing that of negative pairs.
    - **Design Motivation**: Zero-shot models achieve a maximum Acc@1 of only 56.8% (mE5), which is completely inadequate for actual deployment. By aligning the embedding space to the task requirements through metric learning, the performance is significantly improved to 78.8~85.2%. MultipleNegativesRankingLoss is chosen over triplet loss because it is more flexible in batch sampling and naturally leverages all negative samples within the batch, yielding higher training efficiency.

3. **RRF Hybrid Retrieval**:

    - **Function**: Fusing ranking results from sparse and dense retrieval.
    - **Mechanism**: Adopting Reciprocal Rank Fusion (RRF) with the formula $\text{RRF}(d) = \sum_{r \in R} \frac{1}{k + r(d)}$, where $r(d)$ is the position of document $d$ in ranking system $R$, and $k$ is a smoothing constant. The ranking results of BM25 and the dense model are fused according to this formula.
    - **Design Motivation**: Intuitively, sparse and dense retrievals are complementary—BM25 excels at capturing exact word matches, while dense models excel at semantic reasoning. However, experiments show that because BM25 performance is too weak, hybridizing actually degrades the dense model's performance.

### Loss & Training

The dataset is obtained from 528 Romanian private clinics, containing 139,210 mapping pairs (obtained after manually cleaning 6,088 erroneous mappings). The training set consists of 80,911 pairs, and the evaluation set consists of 58,299 pairs. Evaluation employs a design similar to 5-fold cross-validation: the evaluation set is split into gallery and probe sets at a 4:1 ratio, stratified by standard terminology to ensure a balanced distribution of categories in each fold. The model is fine-tuned for 20 epochs with a batch size of 4096, a learning rate of 2e-5, a cosine scheduler + 0.1 warmup ratio, and trained on an NVIDIA A100 80GB GPU.

## Key Experimental Results

### Main Results

| Method | Setting | Acc@1 | Acc@3 | Acc@5 | Acc@100 |
|------|------|-------|-------|-------|---------|
| BM25 | Terminology Only | 52.6 | 64.5 | 68.5 | 86.3 |
| mE5 (dense) | Terminology Only | 78.8 | 92.2 | 95.0 | 99.5 |
| RRF (hybrid) | Terminology Only | 63.9 | 77.7 | 82.1 | 99.5 |
| BM25 | + Mapping Pairs | 68.0 | 82.3 | 86.1 | 94.7 |
| mE5 (dense) | + Mapping Pairs | **85.2** | **95.8** | **97.5** | **99.5** |
| RRF (hybrid) | + Mapping Pairs | 81.0 | 92.3 | 94.9 | 99.5 |

### Ablation Study: Model Comparison (Terminology Only Index)

| Model | State | Acc@1 | Acc@5 | Acc@100 | Description |
|------|------|-------|-------|---------|------|
| mE5-large | off-the-shelf | 56.8 | 74.3 | 91.3 | Advantage of multilingual pre-training |
| BioClinicalBERT | off-the-shelf | 47.7 | 60.2 | 74.9 | Medical pre-training but English only |
| RoBERT-large | off-the-shelf | 44.7 | 56.9 | 75.3 | Romanian but without medical adaptation |
| mE5-large | fine-tuned | **78.8** | **95.0** | **99.5** | Improved by 22.0% after fine-tuning |
| RoBERT-large | fine-tuned | 75.9 | 93.2 | 98.9 | Improved by 31.2% after fine-tuning |
| BioClinicalBERT | fine-tuned | 75.7 | 92.7 | 98.9 | Improved by 28.0% after fine-tuning |

### Key Findings

- **Incredible effect of metric learning fine-tuning**: The Acc@1 of the three models improves by 27%+ on average, where RoBERT leaps from 44.7% to 75.9% (+31.2%), demonstrating that metric learning can effectively bridge the gap between the pre-training domain and the target task.
- **Multilingual models > Language-specific models > Medical domain models**: The reason mE5 far outperforms the other two in a zero-shot setting is that it is inherently a sentence-transformer designed for sentence-pair similarity tasks, whereas RoBERT and BioClinicalBERT require pooling to approximate sentence embeddings.
- **Hybrid retrieval is worse than pure dense retrieval**: The Acc@100 of BM25 is only 86.3%, meaning a large number of correct answers are not in the top-100 of BM25 at all. In RRF fusion, the erroneous rankings of BM25 interfere with the correct rankings of the dense model.
- **Real-world deployment validation**: Human audit by doctors on 12,836 new procedure descriptions yields 94.7% Acc@1 and 98.5% Acc@2, with only 1% requiring manual assignment of different descriptions by doctors. The overall matching process is shortened from 60+ hours to 3 minutes (a 1200× speedup).
- **Adding historical mapping pairs significantly improves performance**: After adding existing mapping pairs to the index, the Acc@1 of all methods increases by 6~15%, proving that historical matching records are an important source of knowledge.

## Highlights & Insights

- **Retrieval vs. Classification Paradigm Choice**: Faced with a scenario of 39K+ classes and 50% single-shot instances, the retrieval paradigm is the only viable solution. Classification models require sufficient training samples for each class, whereas retrieval models only need to learn "what is similar". This paradigm choice is the most important contribution of the paper and can be transferred to all extreme long-tail text matching tasks.
- **Efficient training with metric learning + in-batch negative sampling**: Using MultipleNegativesRankingLoss naturally leverages a large batch size of 4096 to provide rich negative samples, eliminating the need to design complex hard negative mining strategies, which is engineering-wise extremely simple.
- **Deeper implications of evaluation metrics**: Acc@1 may underestimate actual performance because the standard terminology list itself contains duplicate/highly similar entries. This is supported by the 94.7% real doctor validation accuracy (vs. 85.2% automatic evaluation).

## Limitations & Future Work

- **No comparison with LLMs**: The paper does not test the performance of large language models like GPT-4 on this task. Even in a zero-shot prompt setting, LLMs' medical knowledge might excel over fine-tuned mE5, especially in handling synonyms and abbreviations.
- **Propagation of historical mapping errors**: The system uses historical mappings both as training data and for retrieval indexing. If the historical mappings contain errors (remnants could exist even after manual filtering), these errors will be learned by the model and propagated to new matches.
- **Cosine similarity does not equal confidence**: The similarity distributions of correct and incorrect matches heavily overlap, making it difficult to filter low-confidence results simply by setting a threshold, which limits the system's level of automation.
- **Monolingual evaluation**: Though the methodology is generalizable, it was only validated on Romanian and has not been tested for generalization to other low-resource languages.
- **No cross-hospital generalization experiments**: All experiments mixed data from 528 clinics without performing a leave-clinic-out experiment to validate generalization to new clinics.

## Related Work & Insights

- **vs. Tavabi et al. (2024)**: Classified 44,002 surgical records into the 100 most common CPT codes, training an SVM classifier with TF-IDF/Doc2Vec/ClinicalBERT. The key difference in this work is that the class size expands from 100 to 39K+, making the classification paradigm inapplicable and forcing a shift to retrieval.
- **vs. Levy et al. (2022)**: Used XGBoost and BERT to classify pathology reports into 42 CPT codes. This is smaller in scale and represents a simpler task. Interestingly, they found that XGBoost outperformed BERT when using all report fields, indicating that feature engineering is still competitive in small-scale classification tasks.
- **vs. Zaidat et al. (2024)**: Allocated CPT codes to spinal surgery records using XLNet and BiLSTM, with a dataset of only 922 entries. All prior works targeted English/US systems and had class numbers within the hundreds.
- This paper provides a clear paradigm: when the number of classes is extremely high and the long tail is severe, switching from classification to retrieval is the correct decision.

## Rating

- **Novelty**: ⭐⭐⭐ Methodological components (BM25/mE5/metric learning/RRF/Milvus) are off-the-shelf tools; the core contribution lies in the paradigm choice and systematic evaluation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Over 140k data points, 5-fold cross-validation, a complete comparison of three models × two indexing modes × three retrieval strategies, topped off with real doctor validation.
- **Writing Quality**: ⭐⭐⭐⭐ The problem definition is clear, error analysis (Table 2) is highly valuable, and there is a great integration of quantitative and qualitative analyses.
- **Value**: ⭐⭐⭐⭐ Holds direct reference value for medical NLP in low-resource languages; the retrieval paradigm for extreme long-tail classification is widely transferable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Towards Omni-RAG: Comprehensive Retrieval-Augmented Generation for Large Language Models in Medical Applications](omni_rag_medical.md)
- [\[ACL 2025\] MedBioRAG: Semantic Search and Retrieval-Augmented Generation with Large Language Models for Medical and Biological QA](medbiorag_semantic_search_and_retrieval-augmented_generation_with_large_language.md)
- [\[ACL 2025\] ArgHiTZ at ArchEHR-QA 2025: A Two-Step Divide and Conquer Approach to Patient Question Answering for Top Factuality](arghitz_at_archehr-qa_2025_a_two-step_divide_and_conquer_approach_to_patient_que.md)
- [\[ACL 2025\] A Modular Approach for Clinical SLMs Driven by Synthetic Data with Pre-Instruction Tuning, Model Merging, and Clinical-Tasks Alignment](a_modular_approach_for_clinical_slms_driven_by_synthetic_data_with_pre-instructi.md)
- [\[ICLR 2026\] From Medical Records to Diagnostic Dialogues: A Clinical-Grounded Approach and Dataset for Psychiatric Comorbidity](../../ICLR2026/medical_nlp/from_medical_records_to_diagnostic_dialogues_a_clinical-grounded_approach_and_da.md)

</div>

<!-- RELATED:END -->
