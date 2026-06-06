---
title: >-
  [Paper Note] ExaGPT: Example-Based Machine-Generated Text Detection for Human Interpretability
description: >-
  [ACL 2026][AIGC Detection][LLM Text Detection] ExaGPT reformulates the task of "determining whether a text is human-written or LLM-generated" as "finding which side has more similar spans in a datastore." By utilizing BE…
tags:
  - "ACL 2026"
  - "AIGC Detection"
  - "LLM Text Detection"
  - "Interpretability"
  - "k-NN Retrieval"
  - "Dynamic Programming"
  - "Cross-domain Generalization"
date: 2026-05-08
content_hash: a0d29ae98968e2b0
---

# ExaGPT: Example-Based Machine-Generated Text Detection for Human Interpretability

**Conference**: ACL 2026  
**arXiv**: [2502.11336](https://arxiv.org/abs/2502.11336)  
**Code**: https://github.com/ryuryukke/ExaGPT  
**Area**: AIGC Detection / Interpretable Machine Learning / Retrieval-Augmented  
**Keywords**: LLM Text Detection, Interpretability, k-NN Retrieval, Dynamic Programming, Cross-domain Generalization

## TL;DR
ExaGPT reformulates the task of "determining whether a text is human-written or LLM-generated" as "finding which side has more similar spans in a datastore." By utilizing BERT embeddings, k-NN retrieval, and dynamic programming for optimal span segmentation, it provides interpretable evidence (most similar retrieved spans) and improves accuracy by up to +37.0 points over previous interpretable detectors at a 1% FPR.

## Background & Motivation
**Background**: LLM-generated text detection is primarily divided into three categories—watermarking, metric-based (log-prob, entropy, perplexity, probability curvature), and supervised classifiers (RoBERTa fine-tuning, Ghostbuster, Pangram). Overall AUROC already exceeds 99%, making the problem seem "solved."

**Limitations of Prior Work**: However, detectors that only output binary labels are unacceptable regarding false positives—writers have been fired and students' reputations damaged due to misidentification by AI detectors. Existing "interpretable" detectors (GLTR highlighted tokens, SHAP/LIME attribution, DNA-GPT n-gram overlap) provide evidence as token-level statistics or machine-perspective attribution scores, which are unintuitive for average users.

**Key Challenge**: The intuitive process humans use to judge AI writing is: "Have I seen this phrase or way of speaking more often in AI or human texts?"—classifying based on the number of similar spans. No existing detector aligns with this example-based judgment process, meaning users cannot judge if a prediction is "trustworthy."

**Goal**: (1) Design a detector that inherently works by "finding similar span examples"; (2) Naturally turn "similar spans" into evidence for human consumption; (3) Maintain SOTA accuracy in practical scenarios like 1% FPR.

**Key Insight**: Borrowing judgment logic from plagiarism detection (Maurer 2006, Barrón-Cedeño 2013), where humans rely on verbatim overlap and semantically similar spans. For LLM detection: build a datastore of human and LLM texts, perform k-NN retrieval on target text spans, and check which side has more similar spans.

**Core Idea**: Reformulate detection as "k-NN majority voting + dynamic programming span segmentation"—replacing classifiers with retrieval so the decision path naturally serves as human-readable evidence.

## Method

### Overall Architecture
The ExaGPT pipeline has two stages: (1) **Span Scoring**—segment the target text $x$ into all possible $n$-gram spans ($n \in [1, 20]$). Use the mean of BERT's second layer hidden states as span embeddings. Perform top-$k$ ($k=10$) k-NN retrieval (FAISS) for each span against the datastore, recording each neighbors' (similarity $c_j$, original label $l_j \in \{\text{Human}, \text{LLM}\}$). (2) **Span Selection**—use dynamic programming to select a sequence of non-overlapping spans $T=[t_1,\dots,t_H]$ from all $n$-gram candidates that cover the entire text, maximizing the weighted average of "span length + retrieval similarity." Detection is based on the mean prediction score of selected spans, with their top-$k$ retrieval results shown as evidence.

### Key Designs

1. **Triple Span Scoring (length / reliability / prediction)**:

    - **Function**: Decompose "whether a span is worth being evidence" into three independently optimizable scalar signals.
    - **Mechanism**: For each target span $x_{i:i+n}$, define length score $L = n$, reliability score $R = \frac{1}{k}\sum_j c_j$ (mean similarity of k-NN neighbors, measuring if truly similar spans exist in the datastore), and prediction score $P = \frac{1}{k}\sum_j \mathbb{1}(l_j = \text{LLM})$ (proportion of LLM labels in k-NN).
    - **Design Motivation**: Humans make two judgments when assessing AI text: "Have I seen this in AI text before?" (reliability) and "Which side is it seen more often?" (prediction). Separating scores allows for fine-grained segmentation in the second stage.

2. **Dynamic Programming Span Segmentation (balance length vs. reliability)**:

    - **Function**: Find the optimal segmentation that is "both long enough and similar enough to real snippets in the datastore" from an exponential number of $n$-gram combinations.
    - **Mechanism**: Define total segmentation score $S(T) = \frac{1}{H}\sum_h [\alpha L^{\text{std}}(t_h) + (1-\alpha) R^{\text{std}}(t_h)]$, where $L^{\text{std}}, R^{\text{std}}$ are normalized scores from the validation set. DP state $\text{dp}[i]$ stores the best cumulative score for prefix $x_{0:i}$ and backpointers. When transitioning, iterate over the previous cut-off $j \in [i-N, i)$ and pick $j$ that maximizes the mean. Complexity is $O(m \cdot N)$, where $N=20$.
    - **Design Motivation**: Naive segmentation (e.g., fixed $n=5$) loses information from long verbatim overlaps and rare, highly discriminative short spans. DP favors spans that are both long and highly similar, resulting in continuous evidence for the UI.

3. **k-NN Retrieval for Evidence**:

    - **Function**: Ensure the final decision and user evidence are produced by the **same** mechanism, avoiding decoupling between model decision and post-hoc explanation.
    - **Mechanism**: Text is judged LLM if $P_{\text{overall}} = \frac{1}{H}\sum_h P(t_h) > \epsilon$. Evidence $E = \{(t_h, [s_h^1, \dots, s_h^k])\}_{h=1}^H$ is displayed as tooltips (colors red/neutral/blue representing Human/neutral/LLM). Hovering reveals top-$k$ nearest neighbors and their label distribution.
    - **Design Motivation**: Traditional SHAP/LIME only indicates "important tokens" without explaining "why." ExaGPT lets users inspect the 10 most similar real snippets in the datastore to judge evidence reliability, binding decision and evidence together.

### Loss & Training
ExaGPT is **entirely training-free**. Detection uses BERT-large-uncased for embedding (2nd layer mean pooling for balance between lexical and semantic info). The datastore uses the training split of the M4 dataset (2000 pairs per domain × generator combination) to build a FAISS index. The only hyperparameters are the segmentation coefficient $\alpha \in \{0, 0.125, 0.25, \dots, 1.0\}$, selected on validation (favoring reliability), and the detection threshold $\epsilon$ derived from a 1% FPR on validation.

## Key Experimental Results

### Main Results
Average detection accuracy at 1% FPR on the M4 dataset (4 domains: Wikipedia/Reddit/WikiHow/arXiv × 3 generators: ChatGPT/GPT-4/Dolly-v2):

| Generator | Detector | Wikipedia ACC | Reddit ACC | WikiHow ACC | arXiv ACC | Avg ACC | Avg AUROC |
|-----------|----------|---------------|------------|-------------|-----------|---------|-----------|
| ChatGPT | RoBERTa-SHAP | 77.1 | 61.0 | 50.0 | 87.3 | 68.9 | 100.0 |
| ChatGPT | LR-GLTR | 60.0 | 94.0 | 85.8 | 97.7 | 84.4 | 97.9 |
| ChatGPT | DNA-GPT | 49.4 | 62.9 | 93.5 | 59.9 | 66.4 | 91.4 |
| ChatGPT | **ExaGPT** | **92.3** | 86.6 | **96.0** | 95.8 | **92.7** | 99.2 |
| GPT-4 | RoBERTa-SHAP | 87.8 | 66.4 | 77.4 | 68.6 | 75.1 | 100.0 |
| GPT-4 | LR-GLTR | 85.7 | 97.2 | 77.8 | 98.5 | 89.8 | 98.1 |
| GPT-4 | **ExaGPT** | 87.3 | 91.1 | **92.2** | **98.7** | **92.3** | 99.0 |
| Dolly-v2 | **ExaGPT** | **63.8** | 76.6 | **75.6** | 67.3 | **70.8** | 90.4 |

Human evaluation of interpretability (96 samples × 4 detectors):

| Detector | Acc. of Human Judgments (%) |
|----------|------------------------------|
| RoBERTa-SHAP | 47.9 |
| LR-GLTR | 57.3 |
| DNA-GPT | 53.1 |
| **ExaGPT** | **61.5** |

### Ablation Study
Cross-domain (GPT-4) / cross-generator (arXiv) / paraphrase robustness / inference cost:

| Configuration | Key Metrics | Description |
|------|---------|------|
| Single domain train, cross-domain test (Wikipedia → arXiv, GPT-4) | AUROC 89.3 / ACC@1%FPR 60.5 | Significant performance drop with single-source datastore |
| ALL multi-domain train, cross-domain test | AUROC 94.3-99.5 / ACC 73.4-96.7 | Mixed datastore nearly eliminates cross-domain gap |
| Single generator (GPT-4 → Dolly, arXiv) | AUROC 61.8 / ACC 51.5 | Nearly fails across open-source vs closed-source LLMs |
| DIPPER paraphrase (ChatGPT, avg 4 domains) | AUROC 96.0 / ACC 76.5 | Still significantly leads LR-GLTR (93.9 / 72.9) |
| Datastore 2000 → 500 pairs | AUROC 99.5 → 99.4 | Negligible loss |
| 500 pairs + FAISS-IVFPQ | GPU 162→20 GB (-87%), Latency 14.6→1.22 sec (-91%), AUROC 97.8 | Only 1.7% AUROC cost for deployment optimization |

### Key Findings
- The strongest interpretable baseline, LR-GLTR, achieves only 60.0 ACC@1%FPR on ChatGPT × Wikipedia, while ExaGPT reaches 92.3 (+32.3 absolute points). This indicates existing "interpretable" detectors are unusable in low FPR ranges; interpretability and performance are not necessarily trade-offs.
- Performance improves as the segmentation coefficient $\alpha$ decreases (favoring reliability). Selecting spans with truly similar neighbors in the datastore is more effective than just selecting long spans; however, AUROC remains above 98.5% regardless of $\alpha$.
- Open-source vs. closed-source LLM cross-testing is the main failure mode. GPT-4 datastore on Dolly output yields only 61.8 AUROC due to vast differences in style and vocabulary distribution.
- A datastore size of only 500 pairs is sufficient, allowing deployment with minimal labeled samples compared to supervised methods like RoBERTa.

## Highlights & Insights
- **Decision as Evidence**: ExaGPT unifies model prediction and human-readable explanation into a single k-NN retrieval process, avoiding inconsistencies between decision logic and post-hoc attribution (like SHAP/LIME).
- **DP for Span Segmentation**: Formalizing the search for "most persuasive evidence snippets" as an optimal segmentation problem with length-similarity trade-offs is more elegant than heuristic fixed $n$-grams.
- **Training-free yet beats SOTA Supervised Methods**: Without training, ExaGPT outperforms fine-tuned RoBERTa (+17.2 ACC on GPT-4) and SOTA metrics like Binoculars/Fast-DetectGPT at 1% FPR.
- **Second-Layer BERT hidden states**: Selecting the 2nd layer for span embeddings balances lexical and semantic similarity, which significantly impacts retrieval quality.

## Limitations & Future Work
- **Datastore Dependency**: Requires pre-labeled datastores; cross-domain/generator scenarios require re-indexing.
- **Open vs. Closed-source Generator Failure**: Significant drop from GPT-4 to Dolly remains unresolved except through multi-source mixing.
- **Small Human Evaluation Scale**: Only 4 NLP-background annotators on 96 samples; the +13.6 point interpretability gain lacks broad statistical significance across general users.
- **Inference Cost**: High GPU memory (162 GB) and latency (14.6s/sample) without optimization.
- **Future Directions**: (1) Specialized span encoders via contrastive learning; (2) Online datastore expansion with user feedback; (3) Span-level adversarial paraphrase defense.

## Related Work & Insights
- **vs DNA-GPT**: DNA-GPT uses $n$-gram overlap but only matches verbatim tokens and requires online re-generation. ExaGPT uses semantic embeddings to match "similar meaning" spans offline (+8.4 interpretability gain).
- **vs GLTR / RoBERTa+SHAP**: These provide "machine-perspective" explanations (likelihood/attribution). ExaGPT provides human-understandable similar sentence examples (+13.6 interpretability gain).
- **vs Binoculars / Fast-DetectGPT**: These SOTA metrics are completely uninterpretable; ExaGPT maintains comparable performance with transparent evidence.
- **vs kNN-MT / kNN-LM**: While retrieval is used for generation, ExaGPT adapts it for binary detection with a novel "optimal segmentation via DP" component.

## Rating
- Novelty: ⭐⭐⭐⭐ Applying example-based interpretability to LLM detection with DP span segmentation is a clear, unexplored combination.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage across domains, generators, robustness, hyperparameter sensitivity, and cost.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation and intuitive diagrams; however, DP algorithm details are quite dense.
- Value: ⭐⭐⭐⭐ Highly practical for high-stakes scenarios like education by providing "auditable" evidence with low deployment thresholds.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] When Personalization Tricks Detectors: The Feature-Inversion Trap in Machine-Generated Text Detection](when_personalization_tricks_detectors_the_feature-inversion_trap_in_machine-gene.md)
- [\[ACL 2026\] Temporal Flattening in LLM-Generated Text: Comparing Human and LLM Writing Trajectories](temporal_flattening_in_llm-generated_text_comparing_human_and_llm_writing_trajec.md)
- [\[NeurIPS 2025\] DuoLens: A Framework for Robust Detection of Machine-Generated Multilingual Text and Code](../../NeurIPS2025/aigc_detection/duolens_a_framework_for_robust_detection_of_machine-generated_multilingual_text_.md)
- [\[ACL 2026\] Can AI-Generated Persuasion Be Detected? Persuaficial Benchmark and AI vs. Human Linguistic Differences](can_ai-generated_persuasion_be_detected_persuaficial_benchmark_and_ai_vs_human_l.md)
- [\[ACL 2026\] Beyond the Final Actor: Modeling the Dual Roles of Creator and Editor for Fine-Grained LLM-Generated Text Detection](beyond_the_final_actor_modeling_the_dual_roles_of_creator_and_editor_for_fine-gr.md)

</div>

<!-- RELATED:END -->
