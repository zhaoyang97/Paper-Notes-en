---
title: >-
  [Paper Note] REZE: Representation Regularization for Domain-adaptive Text Embedding Pre-finetuning
description: >-
  [ACL2026][Information Retrieval & RAG][Text Embeddings] REZE performs eigenspace decomposition on anchor-positive relationship representations during domain embedding pre-finetuning…
tags:
  - "ACL2026"
  - "Information Retrieval & RAG"
  - "Text Embeddings"
  - "Domain Adaptation"
  - "Pre-finetuning"
  - "Representation Regularization"
  - "Negative Transfer"
date: 2026-05-08
content_hash: 7d6cc6f1c0aaccff
---

# REZE: Representation Regularization for Domain-adaptive Text Embedding Pre-finetuning

**Conference**: ACL2026  
**arXiv**: [2604.17257](https://arxiv.org/abs/2604.17257)  
**Code**: No public repository link provided in the main text  
**Area**: Information Retrieval  
**Keywords**: Text Embeddings, Domain Adaptation, Pre-finetuning, Representation Regularization, Negative Transfer

## TL;DR
REZE performs eigenspace decomposition on anchor-positive relationship representations during domain embedding pre-finetuning, utilizing robust statistics to identify and soft-shrink task-specific shifts. This allows the model to absorb shared domain knowledge while suppressing representation drift caused by heterogeneous tasks.

## Background & Motivation
**Background**: Modern text embedding models typically undergo large-scale weak-supervision contrastive learning and pre-finetuning (PFT) to serve tasks such as retrieval, classification, and semantic similarity. For professional domains like finance, code, or chemistry, a common practice is to collect multiple small-scale domain datasets, unify them into anchor-positive pairs, and conduct contrastive pre-finetuning.

**Limitations of Prior Work**: These domain datasets are often heterogeneous and fragmented, spanning across retrieval, classification, reranking, STS, and clustering tasks. Directly mixing them for PFT injects task-specific biases into the embedding space, leading to uncontrollable drift in representation geometry, which can sometimes result in PFT performance falling below direct fine-tuning (FT).

**Key Challenge**: Domain pre-finetuning must leverage shared domain knowledge from heterogeneous tasks while preventing task-specific data formats, label structures, and biases from dominating the representation space. Traditional isotropy or post-hoc whitening methods only reshape the distribution after training and cannot distinguish which directions originate from task conflicts, potentially damaging useful geometric structures.

**Goal**: The authors aim to explicitly control representation shift during the pre-finetuning process to preserve cross-task semantic commonalities while suppressing task-induced bias, without increasing inference overhead.

**Key Insight**: Instead of processing individual sentence vectors, the paper treats the concatenated anchor and positive embeddings as a "relation representation." The authors argue that shared domain knowledge should be relatively consistent in the relational structure across different tasks, whereas task-specific biases manifest as dispersion in the means of different data sources along certain eigendimensions.

**Core Idea**: Within the eigenspace of the reference model's relation representations, median and Median Absolute Deviation (MAD) are used to identify directions with high task-source variance. Source-specific adaptive soft-shrinkage is applied to these directions, and the resulting debiased relation is used as the regularization target for pre-finetuning.

## Method
REZE is an auxiliary regularization framework for the pre-finetuning stage. It first constructs a global eigenspace using a frozen reference embedding model and calculates the shift patterns of each data source within this space. During training, the model uses InfoNCE for anchor-positive matching while being constrained by a relation-level regularization that pulls the current relation representation toward the debiased reference target.

### Overall Architecture
The input consists of anchor-positive pairs from multiple source datasets. In the offline phase, REZE encodes all pairs using the reference model, concatenates the anchor and positive into $r=[a;p]$, centers them, and performs Eigenvalue Decomposition (EVD) on the covariance matrix to obtain the eigenspace. Subsequently, for each source, the mean on each eigendimension is calculated, and median-based dispersion is used to determine which dimensions primarily distinguish task sources.

During the online pre-finetuning phase, REZE selects the corresponding shrinkage matrix for each sample based on its source, pulling the reference relation representation back toward the robust consensus among tasks. The current model's relation representation is then required to be close to this debiased target via cosine dissimilarity. The final objective is the InfoNCE loss plus the REZE regularization term.

### Key Designs
1. **Relation Representation instead of Single Sentence Representation**:
    - **Function**: Focuses regularization on pair-level semantic relationships rather than the positions of individual text points.
    - **Mechanism**: Constructs $r_{s,i}=[a_{s,i};p_{s,i}] \in \mathbb{R}^{2d}$ for each training sample to estimate global means, covariance, and the eigenspace.
    - **Design Motivation**: The core supervision in embedding pre-finetuning comes from "which texts should be similar." Task biases often manifest in relational structures (e.g., positives in one task may look like labels, while in another they look like documents). Using relation representations aligns more directly with contrastive learning objectives.

2. **Robust Statistical Task-Variation Detection**:
    - **Function**: Identifies which eigendimensions primarily reflect shifts between different sources.
    - **Mechanism**: Computes the mean $\mu_s$ for each source in the eigenspace, uses the component-wise median of source means as the robust center $m_j$, and measures task dispersion via $v_j=\frac{1}{S}\sum_s(\mu_{s,j}-m_j)^2$. Bias is detected only within active dimensions accounting for 99% of cumulative variance.
    - **Design Motivation**: Means are easily skewed by outlier tasks; since the global mean after centering is near zero, shrinking toward the mean might destroy task-invariant semantics. Median/MAD is better suited for finding the geometric center shared by the majority of tasks in heterogeneous data.

3. **Adaptive Soft-Shrinkage and Training Regularization**:
    - **Function**: Suppresses task-specific shifts while preserving useful semantic structures.
    - **Mechanism**: When a source shift in a dimension exceeds a robust threshold, a shrink coefficient $\alpha_{s,j}$ is calculated for that source and dimension to pull the representation back toward the median band. During training, a debiased target $\hat{r}^{(0)}=W A_s W^T(r^{(0)}-u)+u$ is constructed, and a $1-\cos(r_i,\hat{r}^{(0)}_i)$ penalty is added.
    - **Design Motivation**: Hard-deleting top components loses semantics; post-hoc whitening indiscriminately changes the final space. REZE’s source-specific soft shrinkage is more fine-grained, suppressing only directions exhibiting task variance.

### Loss & Training
The main loss is standard InfoNCE: maximizing the similarity between anchors and their corresponding positives within a batch while treating other positives as negative samples. The REZE regularization term is the cosine dissimilarity between the current relation representation and the debiased reference relation. The final objective is $L=L_{main}+\alpha L_{reze}$, with a default temperature $\tau=0.05$ and a regularization strength of $\alpha=1.0$. Since the eigenspace, means, and shrink matrices are computed before training, there is no additional overhead during inference.

## Key Experimental Results

### Main Results
The authors tested E5, ModernBERT, GTE, and Qwen3-Embedding backbones on three professional benchmarks: FinMTEB, Code(MTEB), and ChemTEB, comparing FT, PFT, PFT+Whitening, PFT+NormalizingFlow, and REZE.

| Model / Domain | Samples | FT | PFT | REZE (Ours) | Main Gain |
| :--- | :--- | :--- | :--- | :--- | :--- |
| E5 / Code(MTEB) | 1000 | 0.4898 | 0.3565 | 0.5286 | +0.1721 vs. PFT, +0.0388 vs. FT |
| ModernBERT / FinMTEB | 1000 | 0.8247 | 0.8192 | 0.8373 | Consistently higher than FT and PFT |
| GTE / Code(MTEB) | 500 | 0.5239 | 0.5352 | 0.6167 | Significant gain in code domain |
| Qwen3-Embedding / Code(MTEB) | 100 | 0.4019 | 0.1214 | 0.4081 | Prevents PFT collapse |
| Qwen3-Embedding / ChemTEB | 1000 | 0.6563 | 0.6765 | 0.6688 | Slightly below PFT, but better than FT |

Overall, REZE outperforms standard PFT and post-hoc isotropy methods in most settings. Notably, for Qwen3 on Code(MTEB), PFT performance dropped from 0.4019 to 0.1214, while REZE maintained 0.4081, demonstrating that controlling representation drift is critical for heterogeneous domain pre-finetuning.

### Ablation Study
The paper analyzes regularization weight, median vs. mean, isotropy, and representation drift.

| Configuration | Key Metrics | Observation |
| :--- | :--- | :--- |
| Default REZE | $\alpha=1.0$ | Overall mean is stable, particularly strong at low sample sizes |
| Large $\alpha$ | 5 or 10 | Most tasks saturate or decline; excessive regularization suppresses adaptation |
| median aggregation | Higher on most FinMTEB tasks | More robust than mean, avoids being skewed by outlier sources |
| mean aggregation / ESGClassification | 0.8997 | Lower than median (0.9117) |
| mean aggregation / FINAL | 0.5331 | Lower than median (0.6172) |
| REZE vs PFT IsoScore | ~3x improvement on FinMTEB/Code | More balanced use of representation dimensions |

### Key Findings
- Simple PFT is often lower than direct FT, indicating that more domain data does not automatically translate to gains; heterogeneous task conflicts cause negative transfer.
- Whitening and Normalizing Flow degrade significantly in low-resource settings like ChemTEB, likely because post-processing statistics estimated from limited training sets are unstable and amplify low-variance noise.
- REZE does not blindly pursue isotropy but controls representation drift to remain near the original embedding manifold. This "controlled shift" is more suitable for domain adaptation than post-training forced reshaping.
- Batches need to mix different sources for REZE's distribution alignment to be effective. It is essentially a regularization of cross-task relational structures rather than single-task augmentation.

## Highlights & Insights
- The paper clearly identifies the core risk of domain-adaptive embedding PFT: bias from task heterogeneity may outweigh domain knowledge gains. This is a common issue in practical enterprise retrieval and professional domain embeddings.
- Relation-level regularization is clever. It does not simply keep individual sentence vectors in place but ensures the "relationship between anchor and positive" stays close to a debiased reference structure, aligning better with contrastive embedding objectives.
- The choice of median/MAD is simple but highly effective for multi-source scenarios. Compared to global whitening, it distinguishes between "one source being biased" and "the overall semantic structure being preserved."
- Results suggest that in low-resource or highly heterogeneous domains, controlling representation drift is more important than increasing data volume or using post-processing for isotropy. This provides insights for building embedding pipelines in finance, code, law, and medicine.

## Limitations & Future Work
- While domains include finance, code, and chemistry, the professional depth of public benchmarks remains limited. High-jargon or jurisdiction-specific fields like law might better demonstrate the method's value or expose new issues.
- Model scales only cover roughly 0.1B to 0.6B embedding backbones; trends for larger models or massive batch contrastive training have not been verified.
- REZE requires EVD and source-level statistics on reference representations before pre-finetuning. For ultra-large-scale corpora or streaming data, the offline cost and incremental update mechanisms require further study.
- The method assumes source identifiers are known and that bias between sources can be characterized by mean dispersion. In real-world data where task boundaries are blurred, finer-grained clustering or dynamic source modeling might be needed.

## Related Work & Insights
- **vs. Standard PFT**: PFT uses only InfoNCE to absorb heterogeneous data, often learning task biases simultaneously; REZE adds a debiased relation target during training to control this drift.
- **vs. Whitening / Normalizing Flow**: Post-processing methods change the final space but do not participate in training or distinguish task-specific bias; REZE actively constrains the representation trajectory during PFT.
- **vs. All-but-the-top / Isotropy Methods**: These often remove high-variance directions or seek uniform dimension usage; REZE applies soft-shrinkage only to task-variant active dimensions, making the target more specific.
- **Insight**: Multi-task embedding training can use "consistency of relational structures across sources" as a regularization signal. Future work could combine this with gradient surgery, task routing, or mixture-of-experts to further separate shared domain knowledge from task noise.

## Rating
- Novelty: ⭐⭐⭐⭐☆ The combination of eigenspace and robust soft-shrinkage is not overly complex, but the problem definition for heterogeneous embedding PFT and the relation-level design are highly valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Covers 3 domains, 4 backbones, and multiple sample sizes with geometric analysis; larger models and more specialized benchmarks are still needed.
- Writing Quality: ⭐⭐⭐⭐☆ Clear formulas and solid motivation; some experimental tables are large and require careful reading to understand average scores across protocols.
- Value: ⭐⭐⭐⭐☆ Highly practical for professional retrieval and enterprise embedding adaptation, especially in real-world scenarios with multiple source datasets.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] More Than Efficiency: Embedding Compression Improves Domain Adaptation in Dense Retrieval](more_than_efficiency_embedding_compression_improves_domain_adaptation_in_dense_r.md)
- [\[ACL 2026\] PL-MTEB: Polish Massive Text Embedding Benchmark](pl-mteb_polish_massive_text_embedding_benchmark.md)
- [\[ICLR 2026\] HUME: Measuring the Human-Model Performance Gap in Text Embedding Tasks](../../ICLR2026/information_retrieval/hume_measuring_the_human-model_performance_gap_in_text_embedding_tasks.md)
- [\[ACL 2026\] UnIte: Uncertainty-based Iterative Document Sampling for Domain Adaptation in Information Retrieval](unite_uncertainty-based_iterative_document_sampling_for_domain_adaptation_in_inf.md)
- [\[ACL 2026\] Why Mean Pooling Works: Quantifying Second-Order Collapse in Text Embeddings](why_mean_pooling_works_quantifying_second-order_collapse_in_text_embeddings.md)

</div>

<!-- RELATED:END -->
