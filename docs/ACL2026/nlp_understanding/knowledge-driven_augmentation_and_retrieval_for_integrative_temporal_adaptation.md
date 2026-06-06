---
title: >-
  [Paper Note] Knowledge-driven Augmentation and Retrieval for Integrative Temporal Adaptation
description: >-
  [ACL 2026][NLP Understanding][Temporal Shift] KARITA decomposes "temporal drift" into three complementary signals: uncertainty, feature distance, and ontology term rarity. For each hit target sample…
tags:
  - "ACL 2026"
  - "NLP Understanding"
  - "Temporal Shift"
  - "Ontological Knowledge"
  - "Retrieval-Augmented"
  - "Synthetic Synonyms"
  - "Multi-label Classification"
date: 2026-05-08
content_hash: d7a4e886b54b7fcd
---

# Knowledge-driven Augmentation and Retrieval for Integrative Temporal Adaptation

**Conference**: ACL 2026  
**arXiv**: [2604.22098](https://arxiv.org/abs/2604.22098)  
**Code**: https://github.com/trust-nlp/TemporalLearning-KARITA  
**Area**: Temporal Adaptation / Data Drift / Medical NLP  
**Keywords**: Temporal Shift, Ontological Knowledge, Retrieval-Augmented, Synthetic Synonyms, Multi-label Classification

## TL;DR
KARITA decomposes "temporal drift" into three complementary signals: uncertainty, feature distance, and ontology term rarity. For each hit target sample, it performs backtracking retrieval of semantically similar source samples, and then utilizes LLMs + domain ontologies (MeSH / EuroVoc / CSO) to generate synonym rewrites for data augmentation. This data-driven approach transitions source-period models to future periods, consistently outperforming strong baselines across long-span multi-label classification datasets in clinical, legal, and scientific domains.

## Background & Motivation

**Background**: In real-world deployments, models are trained on historical data and perform inference on future data, where semantic distributions and domain knowledge evolve. Existing temporal adaptation efforts either ignore the temporal dimension or focus on a single drift signal—such as diachronic word embedding shifts, feature space distance, or concept distributions—relying on a lone signal to describe all shifts.

**Limitations of Prior Work**: In long-span, high-stakes corpora such as clinical (MIMIC), legal (EurLex), and scientific (arXiv-CS) texts, temporal shifts are naturally multi-source and overlapping: new medication regulations, legislative changes, and emerging CS sub-fields occur simultaneously. Unified feature representations flatten and misjudge drifts of different natures, leading to model collapse in certain periods (e.g., EATA’s ma-F1 on MIMIC drops to 28.02 from source to target, a typical failure).

**Key Challenge**: Shifts consist of "semantically visible" and "semantically invisible" categories. Evolutionary changes at the terminology level (e.g., new disease codes, new legislative abbreviations) often do not produce significant distances in feature space, causing pure feature-shift detection to fail; meanwhile, entropy/uncertainty only considers output, potentially missing samples where the model is confident but the actual semantics have changed. No single signal is sufficient.

**Goal**: (1) Characterize heterogeneous temporal shifts using multi-angle, complementary signals; (2) Avoid reliance on target labels or blind pseudo-labeling by utilizing "backtracking retrieval" of real source annotations; (3) Perform term-level synonym rewriting on retrieved samples to enhance model robustness against terminological evolution without retraining the entire system.

**Key Insight**: The authors reconceive temporal adaptation as a "data-centric, iterative selection" process: each target batch first identifies "which samples are hit by drift," and then specifically pulls back and augments corresponding source samples, rather than performing a one-time full retraining.

**Core Idea**: Use a combination of uncertainty, feature, and ontology drift signals for joint detection. The corresponding "still credible" source samples are then re-injected into training after LLM/Ontology-based synonym augmentation, achieving iterative adaptation through "shift-aware retrieval + knowledge-aware augmentation."

## Method

### Overall Architecture
The KARITA main loop (Algorithm 1) uses the source period model $\Theta_s$ as initialization. For each streaming target batch $\mathcal{B}_t$: (1) The **Shift Detection** module computes the intersection/union of three signals $U(x), F(x), O(x)$ to obtain $\mathcal{D}_{shift}$; (2) **Source Backtracking Retrieval** calculates cosine similarity using the $[CLS]$ embeddings of the source model encoder for each hit $x_t$, retrieving the top-$k$ (default $k{=}3$) semantic nearest neighbors from source data $\mathcal{D}_s$; (3) **Knowledge-driven Augmentation** evokes GPT-4o-mini to extract label-related terms and produce synonyms or queries MeSH / EuroVoc / CSO ontologies for synonyms to replace terms in the retrieved source samples; (4) $\Theta$ is updated via gradients using these augmented samples with "high-quality true labels + term alignment." Term identification via LLM is run once per target sample and cached to save costs.

### Key Designs

1.  **Multi-angle Shift Detection (U+F+O)**:
    - **Function**: Determines which target samples "require attention"—accounting for output uncertainty, feature space drift, and crucially, "ontological term rarity."
    - **Mechanism**: (i) Uncertainty uses dual thresholds for max sigmoid probability and mean binary entropy: $U(x)=\mathbf{1}[\max_l p_l(x)<\tau_p \wedge H(x)>\tau_H]$ ($\tau_p{=}0.5, \tau_H{=}0.25$); (ii) Feature uses Mahalanobis distance $d(x)=\sqrt{(E(x)-\mu)^\top\Sigma^{-1}(E(x)-\mu)}$, taking the top-$\rho$ after min-max normalization; (iii) Ontology treats source-period ontological concept frequency $p_{t_1}(c)$ as a prior, calculating the average surprisal of all ontological concepts in a target document: $O_{\text{tail}}(x)=\frac{1}{|\mathcal{C}(x)|}\sum_c -\log(p_{t_1}(c)+\varepsilon)$. Higher values indicate more rare/new terms. Finally, $\mathcal{D}_{shift}=\mathcal{D}_U\cup\mathcal{D}_F\cup\mathcal{D}_O$, with $\rho{=}0.1$.
    - **Design Motivation**: t-SNE shows that ontology-shift samples and feature-shift samples barely overlap (only 0.37% $O\cap F$ on MIMIC). The union of the three is necessary—this is the primary difference from single-signal works.

2.  **Source Backtracking Retrieval**:
    - **Function**: Finds semantic nearest neighbors with true labels from the source period to serve as "bridges" for each hit target sample.
    - **Mechanism**: The source model encoder represents $x_t$ and $x_s$ as $\mathbf{z}_t$ and $\mathbf{z}_s$, taking the top-$k$ based on $\text{sim}(x_t,x_s)=\cos(\mathbf{z}_t, \mathbf{z}_s)$. These $k$ source samples act as "credible teachers with aligned semantics."
    - **Design Motivation**: Compared to Self-Labeling (where errors accumulate) or TTA which directly minimizes target entropy (unstable under distribution drift), using "source neighbors with known true labels" for supervision significantly reduces error accumulation. Removing retrieval caused arXiv-CS ma-F1 to plummet from 49.82 to 36.40 in ablations, proving it is the core driver of adaptation.

3.  **Knowledge-driven Augmentation**:
    - **Function**: Rewrites terms in retrieved source samples into synonymous expressions likely to appear in the target period, forcing the model to learn term invariance.
    - **Mechanism**: Dual-source synonyms: (i) **LLM Path** (for EurLex / arXiv-CS): GPT-4o-mini is provided with documents and candidate labels to select 3-10 "terms highly informative for classification" and generate synonyms or historical expressions; (ii) **Ontology Path** (for MIMIC where LLMs are restricted due to privacy): Uses MeSH descriptors + supplementary concepts, EuroVoc PT-NPT, and CSO topic relations to provide controlled, reliable synonymy. For each candidate term, controlled substitution is performed in the source sentence.
    - **Design Motivation**: From a representation perspective, "synonym rewriting = controlled lexical perturbation," encouraging the model to be invariant to term variants, precisely addressing "term-evolution drift." Falling back to pure ontology for MIMIC remains effective, showing robustness to the type of external resource.

### Loss & Training
Source models: XLM-RoBERTa-base for EurLex/arXiv-CS and Longformer for MIMIC, trained for 10 epochs on the earliest period $T_1$ with lr=$3\times10^{-5}$ using multi-label BCE. The KARITA adaptation phase employs the same BCE, performing SGD only on the augmented retrieved source samples; no target labels are required. $\rho{=}0.1, k{=}3$ was optimal in MIMIC sensitivity analysis. LLM term identification results are cached after a single run per target sample.

## Key Experimental Results

### Main Results
Source → Target classification performance (%), target period test set:

| Dataset | Metric | Source Model | Self-Labeling | EATA (TTA) | IFT | **Ours (KARITA)** | Target Upper Bound |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| MIMIC | ma-F1 | 40.65 | 40.55 | 28.02 | 43.05 | **52.12** | 65.78 |
| MIMIC | mi-F1 | 52.86 | 52.34 | 45.98 | 55.24 | **63.95** | 76.66 |
| EurLex | ma-F1 | 46.75 | 42.02 | 47.97 | 37.12 | **56.15** | 71.74 |
| arXiv-CS | ma-F1 | 34.86 | 34.94 | 27.63 | 40.67 | **49.82** | 65.51 |
| arXiv-CS | sa-F1 | 43.36 | 43.46 | 34.90 | 49.17 | **62.63** | 74.98 |

KARITA closes the Source → Target ma-F1 gap by +11.47 on MIMIC and +14.96 on arXiv-CS. EATA performed 12 points worse than the Source model on MIMIC, verifying that unsupervised TTA is prone to collapse in the medical domain.

### Ablation Study
Removing single components of KARITA under a shared Llama-encoder setting (target test ma-F1):

| Configuration | MIMIC | EurLex | arXiv-CS | Description |
| :--- | :--- | :--- | :--- | :--- |
| Full KARITA | 52.12 | 56.15 | 49.82 | Complete Method |
| w/o detection (Random) | 49.33 | 48.77 | 31.02 | arXiv-CS drops 18.8 |
| w/o augmentation | 48.13 | 54.60 | 43.74 | Loss of term alignment |
| w/o retrieval (Dissimilar) | 50.67 | 44.16 | 36.40 | EurLex drops 12 |

Comparison of single signal detectors (replacing joint signals, target test ma-F1):

| Detector | MIMIC | EurLex | arXiv-CS |
| :--- | :--- | :--- | :--- |
| Full (U+F+O) | **52.12** | **56.15** | **49.82** |
| Feature only | 51.58 | 44.57 | 23.00 |
| Ontology only | 40.94 | 50.48 | 29.69 |
| Uncertainty only | 42.45 | 54.97 | 42.64 |

## Key Findings
- **Ontological term drift is an irreplaceable signal**: On MIMIC, $U\cap O$ is only 3.05% and $O\cap F$ is 0.37%. t-SNE shows ontology-shift and feature-shift samples occupy different regions; pure feature detection on arXiv-CS causes ma-F1 to drop to 23.00.
- **Retrieval + Augmentation synergy**: Removing either component causes a faster collapse than removing both simultaneously. The contribution order varies by domain (MIMIC relies more on augmentation, arXiv-CS on detection), proving it is a truly integrated framework.
- **TTA instability under multi-source overlap**: EATA/SAR degraded on MIMIC because entropy minimization further reinforced incorrect labels; KARITA avoids this drift accumulation via true source supervision.
- **Temporal distance ↑ correlates with simultaneous ↑ in all three shifts**: On EurLex and arXiv-CS, F, O, and entropy scores all rise monotonically with time, though at different rates—validating the necessity of joint signals.
- **Hyperparameter robustness**: Optimal results were found at $k{=}3, \rho{=}0.1$, with small fluctuations across settings, making it practical for deployment.

## Highlights & Insights
- "Temporal drift" is typically measured as a scalar; this work structures it into "output, representation, and ontology" decoupled perspectives, using ontological surprisal as a primary detection signal for the first time.
- Using LLMs to extract task-relevant terms + synonyms narrows down uncontrolled free-form rewriting to "informative terms for classification," avoiding label semantic corruption—a rare "label-aware" data augmentation approach.
- The MeSH/EuroVoc/CSO route provides a practical "privacy fallback": when sending clinical data to external LLMs is prohibited, pure ontology remains effective, offering high engineering value.
- Converting adaptation from a one-time event to batch-level iteration, combined with reusable synonym caching, eliminates the need for online backpropagation of giant models (unlike TTA), lowering deployment costs.

## Limitations & Future Work
- The framework relies heavily on external knowledge resources (LLMs/Ontologies); low-resource or closed domains require alternatives (e.g., domain LM self-mining).
- Only addresses lexical/terminological drift; structural shifts like new concept emergence or task redefinition are not explicitly modeled.
- Signal merging uses simple union + equal-weighted top-$\rho$; there is room to learn domain-specific signal weights.
- Retrieval is currently restricted to a single source domain; multi-source or cross-domain backtracking remains unexplored.
- Lack of human evaluation for LLM synonym quality may introduce potential biases.

## Related Work & Insights
- **vs. IFT / ChronosLex**: IFT uses chronologically ordered incremental training; KARITA uses data augmentation to make source samples "appear from the future." The two are stackable.
- **vs. Self-Labeling**: Self-Labeling applies pseudo-labels to target samples, accumulating errors; KARITA uses true source labels as bridges, significantly improving robustness.
- **vs. SAR / EATA / TENT**: These TTA methods collapse when source → target distribution gaps are large; KARITA follows a data-centric route, avoiding the negative cycles of entropy minimization.
- **vs. Huang & Paul 2019 (diachronic embeddings)**: They compress drift into word vectors; this work explicitly extracts three signals, effectively extending diachronic ideas from word vectors to knowledge graphs.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The combination of "multi-signal shift + source backtracking + knowledge synonym augmentation" is systematically proposed for temporal adaptation for the first time; ontology surprisal for detection is particularly novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers 3 domains × 4 periods × multiple baselines × single-signal ablation + overlap analysis + temporal trends + hyperparameter sensitivity.
- **Writing Quality**: ⭐⭐⭐⭐ The narrative is smooth; the design of Tables 4-6 directly addresses signal complementarity with a clear structure.
- **Value**: ⭐⭐⭐⭐ Achieves gains in clinical, legal, and tech domains and is agnostic to external resource types, showing high deployment potential.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Unveiling Temporal Framing in News Text](uncovering_temporal_framing_in_the_news.md)
- [\[ACL 2026\] Beyond Chunking: Discourse-Aware Hierarchical Retrieval for Long Document Question Answering](beyond_chunking_discourse-aware_hierarchical_retrieval_for_long_document_questio.md)
- [\[ACL 2026\] Commonsense Knowledge with Negation: A Resource to Enhance Negation Understanding](commonsense_knowledge_with_negation_a_resource_to_enhance_negation_understanding.md)
- [\[ACL 2026\] It's High Time: A Survey of Temporal Question Answering](it39s_high_time_a_survey_of_temporal_question_answering.md)
- [\[ACL 2026\] Creating ConLangs to Probe the Metalinguistic Grammatical Knowledge of LLMs](creating_conlangs_to_probe_the_metalinguistic_grammatical_knowledge_of_llms.md)

</div>

<!-- RELATED:END -->
