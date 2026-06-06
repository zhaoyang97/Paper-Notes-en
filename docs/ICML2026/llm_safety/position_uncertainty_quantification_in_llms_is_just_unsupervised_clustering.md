---
title: >-
  [Paper Note] Position: Uncertainty Quantification in LLMs is Just Unsupervised Clustering
description: >-
  [ICML 2026][LLM Safety][Position Paper] This position paper presents a central thesis: mainstream methods for LLM Uncertainty Quantification (UQ)—such as Semantic Entropy, spectral methods…
tags:
  - "ICML 2026"
  - "LLM Safety"
  - "Position Paper"
  - "Uncertainty Quantification"
  - "Confident Hallucination"
  - "Clustering Paradigm"
  - "External Ground Truth"
date: 2026-05-08
content_hash: 348c5889d1c39bf4
---

# Position: Uncertainty Quantification in LLMs is Just Unsupervised Clustering

**Conference**: ICML 2026  
**arXiv**: [2605.19220](https://arxiv.org/abs/2605.19220)  
**Code**: None (position paper)  
**Area**: LLM Safety / Uncertainty Quantification  
**Keywords**: Position Paper, Uncertainty Quantification, Confident Hallucination, Clustering Paradigm, External Ground Truth

## TL;DR
This position paper presents a central thesis: mainstream methods for LLM Uncertainty Quantification (UQ)—such as Semantic Entropy, spectral methods, and P(true)—are mechanistically isomorphic to unsupervised clustering. They measure "internal consistency of model generations" rather than "external correctness," inherently failing in the face of "confident hallucinations." The authors diagnose three major pathologies: parameter sensitivity, internal evaluation loops, and a lack of ground truth, proposing a roadmap toward "supervised guarantees" based on three pillars: evaluation, mechanism, and grounding.

## Background & Motivation

**Background**: The primary obstacle to deploying LLMs in high-risk domains (medical, legal) is hallucination. The industry's primary safety net is UQ: assigning an uncertainty score to every query-answer pair and rejecting answers that trigger a threshold. Technical approaches fall into three main categories: entropy-based (Semantic Entropy and variants like SAE/SEN/KLE/SNNE/SDLG), graph-based (SGC/GU/SGD/SeSE/GENUINE/U-EigV), and verbalized self-evaluation (P(true)/CIn/SelfCheckGPT/UaIT).

**Limitations of Prior Work**: Despite the proliferation of UQ papers, models continue to "confidently talk nonsense." Metrics like AUROC appear favorable, yet models fail to catch critical errors in real-world scenarios, creating a false sense of security for users.

**Key Challenge**: The authors diagnose this as a **category error**—all mainstream UQ methods measure "how stable model generations are relative to each other" rather than "how close the answer is to external facts." When a model is highly consistent about an incorrect answer (confident hallucination), these methods yield "high confidence," contradicting their safety objectives.

**Goal**: (i) Prove that mainstream UQ methods are mechanistically isomorphic to unsupervised clustering; (ii) Reveal three pathologies caused by this isomorphism—parameter sensitivity, internal evaluation loops, and lack of ground truth; (iii) Provide a roadmap across evaluation, mechanism, and grounding pillars to move UQ from "unsupervised heuristics" toward "supervised guarantees."

**Key Insight**: By using the unified perspective of "Is it clustering?", the authors deconstruct the mathematical structures of SE, spectral, and P(true) methods. Drawing on the classic lesson from clustering research—that internal validity indices cannot guarantee semantic correctness—the paper exposes the fundamental flaws of UQ within a single framework.

**Core Idea**: UQ $\neq$ measuring "truth or falsehood"; rather, UQ = measuring the "geometric/semantic separation between model generations." This is unsupervised clustering and lacks external anchors. The only way forward is to introduce external ground truth and supervised mechanisms.

## Method

### Overall Architecture
As a position paper, the "Method" consists of an argumentative chain of "diagnosis + prescription":

1.  **Unified Abstraction**: Reducing the three categories of UQ methods (SE, spectral, P(true)) into the same clustering operation.
2.  **Three Pathologies**: Parameter sensitivity crisis, internal evaluation traps, and lack of ground truth.
3.  **Five Counter-arguments**: Rebutting common beliefs such as "parameter sensitivity is a feature," "UQ measures belief, not truth," and "scaling solves everything."
4.  **Three-Pillar Roadmap**: Shifting evaluation to "worst-case robustness," mechanisms to "native uncertainty / Conformal Prediction," and grounding to "verifiable unit tests + atomic fact-checking."

### Key Designs

1.  **Unified Clustering Mechanism Proof**:

    - **Function**: Reveal the mechanistic equivalence of the three mainstream UQ approaches, providing unified evidence against their role as proxies for "truth."
    - **Mechanism**:
        - **Semantic Entropy is Explicit Clustering**: Using an NLI model to partition $\mathcal{S}=\{s_1,\dots,s_m\}$ into semantic classes $C_1,\dots,C_M$, then calculating $U_{\text{SE}}(C\mid x)=-\sum_{i=1}^M p(C_i\mid x)\log p(C_i\mid x)$. The NLI model acts as the "clustering criterion," and entropy represents "cluster purity."
        - **Spectral Methods are Implicit Spectral Clustering**: Constructing a graph with pairwise similarity $W=(w_{j_1,j_2}),\ w_{j_1,j_2}=(a_{j_1,j_2}+a_{j_2,j_1})/2$ and a normalized Laplacian $L=I-D^{-1/2}WD^{-1/2}$, then using $U_{\text{EigV}}=\sum_{k=1}^m\max(0,1-\lambda_k)$ to count "effective semantic degrees of freedom." This is spectral clustering without explicit label assignment, equivalent to an "internal validity index."
        - **P(true) is Latent Confidence Clustering**: Viewing $U_{\text{P(true)}}(x,\hat{y})=1-P(\text{``True''}\mid x,\hat{y})$ as a membership test for the model's internal "high-confidence region." PCA visualization of Qwen2.5-32B on QASC (Fig. 2) shows geometric separation between high-P(true) and low-P(true) samples in hidden space, geometrically identical to a soft cluster assignment.
    - **Design Motivation**: Once established as essentially the same, one only needs to argue why "unsupervised clustering cannot guarantee semantic correctness" once. The paper notes that token-level perplexity, Deep Ensembles, and supervised classifiers (Azaria & Mitchell 2023) fall outside this framework—the former two due to poor performance, and the latter as the recommended direction.

2.  **Diagnosis of Three Pathologies**:

    - **Function**: Translate "clustering isomorphism" into actual safety hazards in deployment.
    - **Mechanism**:
        - **Parameter Sensitivity Crisis**: UQ scores are drastically affected by hyperparameters like temperature, NLI thresholds, sample size $n$, and prompts. Tab. 1 shows Jaccard similarity—on QASC with Qwen2.5-32B, the overlap of Top-10% high-uncertainty samples between SE vs EigV is only 0.134, and SE vs P(true) is only 0.080, meaning different methods cannot agree on "what is uncertain."
        - **Internal Evaluation Trap**: Evaluation metrics (AUROC) assume "internal stability = factual correctness," but confident hallucinations break this—stable incorrect answers receive high scores. This mirrors the Silhouette coefficient in clustering: internal compactness $\neq$ external meaningfulness.
        - **Lack of Ground Truth ("Judge Problem")**: UQ is evaluated via AUROC correlation with correctness, but correctness in open tasks often relies on RougeL > 0.3 or another LLM judge, which is noisy and biased. Fig. 3 shows that as the correctness threshold $\tau$ shifts, method rankings fluctuate, indicating the evaluation pipeline is built on unstable ground.
    - **Design Motivation**: Grounding abstract "clustering isomorphism" into observable engineering consequences forces UQ researchers to confront the fact that "beautiful AUROC $\neq$ safety."

3.  **Three-Pillar Roadmap: evaluation → mechanism → grounding**:

    - **Function**: Provide the community with an actionable blueprint for "de-clustering" reconstruction.
    - **Mechanism**:
        - **Evaluation Pillar**: (a) Treat UQ as a binary alarm system (accept/reject), adopting the MIA evaluation paradigm (Carlini et al. 2022)—measuring TPR at fixed FPR < 0.1% to capture critical "high-confidence hallucinations"; (b) Propose **AUSC (Area Under the Stability Curve)**: sweeping AUROC across hyperparameters (e.g., $T\in[0,1]$) to require stability across reasonable ranges rather than cherry-picked points.
        - **Mechanism Pillar**: (a) Use **Conformal Prediction** as a downstream framework—at a fixed coverage rate (e.g., 90%), compare the set sizes produced by UQ methods used as nonconformity scores; confident hallucinations will be exposed through "set explosion"; (b) Perform **Uncertainty Alignment** during RLHF, rewarding explicit granular confidence markers (e.g., "I am confident that..." vs "It is possible that..."), turning uncertainty from an implicit geometric feature into an explicit linguistic signal.
        - **Grounding Pillar**: (a) **Mandatory Unit Testing**—UQ methods must first demonstrate AUROC and TPR@low-FPR in programmatically verifiable domains like code (HumanEval) or math (constant final answers); (b) **Atomic Fact Verification**—decomposing open generation into atomic claims and verifying each against search engines, KBs, formal provers (Lean4), or multi-hop search agents to break the "LLM judging LLM" loop.
    - **Design Motivation**: These pillars address "how to evaluate, how to build, and what truth to use," excising the reliance on internal consistency across all stages of the engineering pipeline.

### Loss & Training
The position paper does not involve specific training losses but recommends two quantitative designs: (a) Metrics: **TPR@FPR<0.1%** and **AUSC**; (b) Set size at fixed coverage in Conformal Prediction as a "truth-aware" proxy.

## Key Experimental Results

### Main Results
The paper does not test a new method but uses supporting data to "falsify" the reliability of the mainstream UQ paradigm.

| Evaluation Experiment | Data / Model | Key Result | Conclusion |
|----------|-------------|----------|------|
| Jaccard Overlap (Tab. 1) | QASC, Qwen2.5-32B | SE vs EigV Top-10% = 0.134; SE vs P(true) Top-10% = 0.080; EigV vs P(true) = 0.224 | Methods disagree on "who is uncertain." |
| P(true) Hidden Space (Fig. 2) | QASC, Qwen2.5-32B | High-P(true) and low-P(true) samples geometrically separate into two clusters in PCA | P(true) is a latent space cluster membership test. |
| Correctness Threshold Sensitivity (Fig. 3) | Adapted from Liu et al. 2025b | UQ method rankings flip repeatedly as $\tau$ changes | "Unstable judges" render AUROC evaluation invalid. |

### Ablation Study

| Argument | Supporting Evidence | Pathology → Prescription |
|------|----------|------------|
| Confident hallucination breaks consistency proxy | Simhi et al. 2025; Kalavasis et al. 2025 | Internal consistency → Use worst-case TPR |
| Parameter sensitivity vs Robustness | Cecere et al. 2025 ($T$), Kuhn 2023 ($n$), Farquhar 2024 (NLI threshold) | Single-point reporting → Use AUSC |
| RLHF causing "anti-calibration" | Kadavath 2022, Achiam 2023 | Expecting scaling to solve → Use Uncertainty Alignment + CP |
| Open generation requires verifiable truth | Yao 2022 (code), Hendrycks (math) | LLM-as-judge loop → Use Lean4 / Atomic Facts |

### Key Findings
- **Methods cannot agree on "who is uncertain"**: Jaccard similarity is only 0.08–0.22, showing methods measure different dimensions; using any single method as a "safety net" lacks an external baseline for arbitration.
- **Geometric separation $\neq$ Factual reliability**: PCA visualization of P(true) proves it performs cluster membership testing rather than factual discrimination—it checks if an output falls within a "confidence cluster."
- **AUROC is diluted by easy samples**: Frequent easy cases push AUROC higher, but the only dangerous samples in deployment are the "high-confidence but incorrect" minority, which MIA-style TPR@low-FPR specifically targets.
- **RLHF exacerbates the problem**: Alignment with human preference makes models sound more authoritative; scaling does not automatically solve calibration—it only makes hallucinations look "more professional," magnifying clustering pathologies.

## Highlights & Insights
- **The "Category Error" label is sharp**: Categorizing an entire line of UQ research as "unsupervised clustering" provides a clear binary axis (supervised calibration vs. not) for future work, a hallmark of effective position papers.
- **MIA analogy is a high-quality migration**: Directly moving the worst-case evaluation paradigm from Carlini et al. 2022 to UQ implies that the principle of "evaluating high-risk systems at the tail via TPR@low-FPR" is unifying across ML safety sub-fields.
- **CP as an evaluator is a clever reuse**: Comparing set sizes at fixed coverage forces methods to externalize hallucinations as an observable cost, a logic applicable to any scoring-based safety mechanism.
- **AUSC is a practical tool against p-hacking**: Requiring stability across hyperparameters could become a mandatory benchmark item, closing the "tuning for SOTA" loophole.

## Limitations & Future Work
- **Lack of a complete new method or benchmark**: While the roadmap is clear, TPR@low-FPR, AUSC, and Atomic Fact systems are currently suggestions without an end-to-end empirical demo showing revised rankings.
- **Reliance on secondary empirical evidence**: Fig. 3 is adapted from Liu et al. 2025b, and Tab. 1's Jaccard is only measured on one model/data pair (QASC + Qwen2.5-32B); cross-model reproducibility remains for future work.
- **Difficulty in implementing formal verification**: Lean4 and atomic fact-checking are costly in domains like medicine or law that are factual but non-formal; the paper does not discuss scalability bottlenecks.
- **Gap in "inevitably subjective open generation"**: The authors acknowledge legitimate diversity in creative writing but only offer "atomic fact decomposition" as a remedy, failing to propose alternatives for stylistic or preference-based uncertainty.

## Related Work & Insights
- **vs Semantic Entropy (Kuhn et al. 2023)**: This paper does not deny SE's effectiveness on benchmarks but notes it is an NLI-driven explicit clustering that fails during confident hallucinations. Insight: Any entropy metric must first answer whether the stable objects are anchored to external truth.
- **vs Spectral Methods (Lin et al. 2023, etc.)**: Proven to be implicit clustering via the equivalence of Laplacian spectra and spectral clustering. Insight: Graph/spectral analysis in unlabeled contexts remains a structural index rather than a truth proxy.
- **vs P(true) / SelfCheckGPT**: Reframed as "latent confidence cluster membership tests." Insight: Model self-evaluation is a geometric distance query, not factual discrimination.
- **vs Conformal Prediction (Quach 2023, Su 2024)**: Repurposes CP from "generating prediction sets" to "a truth-aware yardstick for evaluating UQ methods."
- **vs MIA Evaluation (Carlini et al. 2022)**: A cross-domain analogy inspiring the standard that "high-risk systems should be judged by the tail, not the average" in ML safety.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] AGSC: Adaptive Granularity and Semantic Clustering for Uncertainty Quantification in Long-text Generation](../../ACL2026/llm_safety/agsc_adaptive_granularity_and_semantic_clustering_for_uncertainty_quantification.md)
- [\[ACL 2026\] From Passive Metric to Active Signal: The Evolving Role of Uncertainty Quantification in Large Language Models](../../ACL2026/llm_safety/from_passive_metric_to_active_signal_the_evolving_role_of_uncertainty_quantifica.md)
- [\[ICML 2026\] LLM Benchmark Datasets Should Be Contamination-Resistant (Position Paper)](llm_benchmark_datasets_should_be_contamination-resistant.md)
- [\[ICML 2026\] SemGrad: Gradients w.r.t. Semantics-Preserving Embeddings Tell LLM Uncertainty](gradients_with_respect_to_semantics_preserving_embeddings_tell_the_uncertainty_o.md)
- [\[ICML 2026\] TCAP: Tri-Component Attention Profiling for Unsupervised Backdoor Detection in MLLM Fine-Tuning](tcap_tri-component_attention_profiling_for_unsupervised_backdoor_detection_in_ml.md)

</div>

<!-- RELATED:END -->
