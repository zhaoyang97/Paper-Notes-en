---
title: >-
  [Paper Note] Hallucination Detection in LLMs with Topological Divergence on Attention Graphs
description: >-
  [ACL 2026][Hallucination Detection][TDA] TOHA treats the LLM attention matrix as a weighted graph and utilizes Manifold Topology Divergence from Topological Data Analysis (TDA) to measure the "topological novelty of the…
tags:
  - "ACL 2026"
  - "Hallucination Detection"
  - "TDA"
  - "Attention Graphs"
  - "Manifold Topology Divergence"
  - "Hallucination-aware Heads"
  - "Training-free"
date: 2026-05-08
content_hash: 40017160dfa9674f
---

# Hallucination Detection in LLMs with Topological Divergence on Attention Graphs

**Conference**: ACL 2026  
**arXiv**: [2504.10063](https://arxiv.org/abs/2504.10063)  
**Code**: https://github.com/sb-ai-lab/TOHA  
**Area**: Hallucination Detection  
**Keywords**: TDA, Attention Graphs, Manifold Topology Divergence, Hallucination-aware Heads, Training-free

## TL;DR
TOHA treats the LLM attention matrix as a weighted graph and utilizes Manifold Topology Divergence from Topological Data Analysis (TDA) to measure the "topological novelty of the response subgraph relative to the prompt subgraph." It identifies "hallucination-aware heads" that are stable across datasets. Averaging only 10 such heads enables a training-free detector in RAG scenarios that is 70× faster than SelfCheckGPT and achieves significantly higher ROC-AUC.

## Background & Motivation

**Background**: LLM + RAG has become the standard for deployment, yet models still produce hallucinations that are inconsistent with the provided context. Existing detection methods can be categorized into three groups: (1) **Uncertainty-based**—utilizing output probabilities like perplexity or max entropy; (2) **Consistency-based**—comparing results from $N$ re-samplings such as SelfCheckGPT, Semantic Entropy, or EigenScore; (3) **Internal State-based**—probing hidden layers or attention via linear classifiers like HaloScope, LLM-Check, or ReDeEP.

**Limitations of Prior Work**: (1) Supervised internal state methods require large amounts of manually labeled hallucination samples; (2) Consistency methods require 10–20 re-generations, leading to excessive computational overhead; (3) Output probabilities do not fully reflect the true uncertainty of the model; (4) Existing attention-based work either treats all heads equally or ignores geometric structures in favor of raw attention values, wasting the intrinsic graph information of the attention matrix.

**Key Challenge**: High-quality hallucination detection currently lacks solutions that are both data-efficient (requiring minimal supervision) and computationally affordable (avoiding multi-sampling). Furthermore, while internal attention states are known to be informative, their topological structure has not been systematically explored.

**Goal**: (1) Develop a training-free, single-pass detector that selects key heads using minimal probes; (2) Establish a provable link between the occurrence of hallucinations and attention geometric/topological structures; (3) Verify whether the selected "hallucination-aware heads" can transfer across different datasets.

**Key Insight**: Each attention head is viewed as a complete weighted graph, where prompt tokens and response tokens form two distinct sets of vertices. **Manifold Topology Divergence** is computed on this graph to measure the topological novelty of the response subgraph relative to the prompt subgraph. "Excessive novelty" is used as a signal for hallucination (Intuition: a faithful response should be geometrically "embedded" within the prompt's attention structure).

**Core Idea**: Utilize 0-th homology (MST length) to quantify the "minimum connection distance required to attach the response to the prompt." Larger distances indicate a response that is detached from the prompt, signaling a likely hallucination. It is discovered that averaging $\le 10$ specific heads is sufficient for detection.

## Method

### Overall Architecture
The TOHA pipeline consists of a two-stage process (Algorithm 1): (a) **HeadsSelection**—A small probe set containing hallucination samples $S_h$ and grounded samples $S_g$ is used to calculate $\Delta_{ij}$ (the mean difference in topological divergence between hallucination and grounded sets) for each head $(i,j)$, followed by descending rank ordering. The optimal number of heads $N_{\mathrm{opt}}$ is selected by cumulatively averaging from $N=1$ to $N_{\max}=10$ to maximize AUROC. (b) **Prediction**—For a test sample $s$, the average $d_{ij}(s)$ across the $N_{\mathrm{opt}}$ heads is calculated as the hallucination score $p_s$. The entire process involves no parameter training and only examines the attention matrices from a single forward pass.

### Key Designs

1. **Attention as Weighted Graph + $\operatorname{MTop-Div}_G(R,P)$**:

    - **Function**: Transforms the vague semantic judgment of "response faithfulness" into a quantifiable graph-theoretic metric.
    - **Mechanism**: For each head, the attention matrix $W$ is interpreted as a complete undirected weighted graph $G$, with edge weights $1-w_{ij}$ serving as "pseudo-distances" between tokens. The vertex set is split into prompt $P$ and response $R$. Edge weights within $P$ are zeroed out, and the 0-th homology barcode $\mathcal{B}_0$ of the modified graph is computed via the Vietoris-Rips complex. Divergence is defined as $\operatorname{MTop-Div}_G(R,P)=\sum_{[b_i,d_i]\in\mathcal{B}_0}|d_i-b_i|$. Proposition 3.1 proves this is equal to the total edge length of the Minimum Spanning Forest (MSF) connecting $R$ to $P$. Information theory further shows that $\operatorname{MTop-Div}_G(R,P)\geq L_{\mathrm{MST}}(R\cup P)-L_{\mathrm{MST}}(P)$, representing the increment in MST length due to the geometric dispersion of response tokens, interpreted as "structural novelty relative to the prompt."
    - **Design Motivation**: Previous attention-based methods ignored the topological relationship between prompt and response subgraphs. MST/0-th homology captures the intuition of "how far the response is from the prompt" while providing both geometric and information-theoretic interpretations.

2. **Hallucination-Aware Heads Discovery**:

    - **Function**: Explicitly quantifies the "hallucination sensitivity" of attention heads to select a minimal subset for detection, saving computation and providing interpretability.
    - **Mechanism**: Calculated $\Delta_{ij}=\frac{1}{|S_{\mathrm{hallu}}|}\sum_{s\in S_{\mathrm{hallu}}} d_{ij}(s)-\frac{1}{|S_{\mathrm{gr}}|}\sum_{s\in S_{\mathrm{gr}}} d_{ij}(s)$ on the training set, where $d_{ij}(s)=\frac{1}{|R_{ij}^s|}\operatorname{MTop-Div}_{G_{ij}^s}(R_{ij}^s,P_{ij}^s)$. Scatter plots of $\Delta_{ij}$ across datasets (Figure 2) reveal that specific heads (e.g., 4 in Mistral-7B, 3 in Llama-2-7B) consistently show high sensitivity regardless of the dataset. These heads partially overlap with "copying heads" reported in prior literature (Sun 2025).
    - **Design Motivation**: (a) Cross-dataset stability ensures strong transferability; (b) Minimal head usage ensures near-zero inference overhead; (c) Alignment with copying behavior provides a mechanistic explanation—faithful responses "copy" prompt information, whereas insufficient copying leads to higher divergence.

3. **Zeroing Prompt Edge Weights**:

    - **Function**: Ensures the divergence calculation specifically reflects the "cross-set distance" between prompt and response, removing semantic noise from the prompt itself.
    - **Mechanism**: Before calculating MTop-Div, all edge weights within $P$ are set to zero (effectively treating the prompt as a single "grounded" super-node). The MSF then measures how far response nodes must "travel" to connect to the prompt, without interference from complex internal prompt connections. §4.4 Ablation experiments confirm that retaining internal prompt weights drowns out the detection signal.
    - **Design Motivation**: Internal prompt connections are semantically meaningful but act as noise for hallucination detection. Removing them simplifies the task to "cross-boundary" connectivity.

### Loss & Training
TOHA is entirely training-free. The only "learning" step is head ranking during HeadsSelection, which requires a minimal labeled set (100 validation samples or a 5% experimental split). these labels are used only for selection and not for training classifiers. $N_{\mathrm{opt}}$ is capped at 10.

## Key Experimental Results

### Main Results: ROC-AUC (↑), 5 Datasets × 5 LLMs

| Model/Method | MS MARCO | CNN/DM | CoQA | SQuAD | XSum |
|------|------|------|------|------|------|
| **Mistral-7B** | | | | | |
| SelfCheckGPT (Consistency) | 0.63 | 0.51 | 0.86 | 0.71 | 0.66 |
| Max entropy (Uncertainty) | 0.68 | 0.60 | 0.73 | 0.75 | 0.71 |
| ReDeEP (Internal) | 0.54 | 0.47 | 0.59 | 0.45 | 0.63 |
| **TOHA (Ours)** | **0.76** | **0.60** | **0.89** | **0.96** | 0.66 |
| **LLaMA-2-7B** | | | | | |
| SelfCheckGPT | 0.59 | 0.60 | 0.66 | 0.57 | 0.64 |
| Semantic entropy | 0.53 | 0.51 | 0.76 | 0.73 | 0.61 |
| **TOHA (Ours)** | **0.65** | 0.56 | **0.90** | **0.87** | **0.68** |
| **LLaMA-2-13B** | | | | | |
| Max entropy | 0.62 | 0.53 | 0.66 | 0.78 | 0.59 |
| **TOHA (Ours)** | **0.67** | **0.56** | **0.92** | **0.88** | **0.66** |

Ours achieves an 11.7% Gain on MS MARCO over the strongest baseline and a 21.6% Gain on CoQA for LLaMA-2-7B. Wilcoxon-Holm post-hoc tests show TOHA ranks 1.67 overall with $p\leq 0.0016$ significance against all baselines.

### Ablation Study: Efficiency + Transferability

| Dimension | Value | Description |
|------|------|------|
| vs. SelfCheckGPT (Single extra generation) | ~7× faster | TOHA requires only one forward pass |
| vs. SelfCheckGPT (Actual 10–20 iterations) | **~70× faster** | Real-world deployment scenario |
| vs. Max entropy overhead | Similar magnitude | But with significantly higher AUROC |
| Training set size | $|S_h\cup S_g|=100$ | Only 100 samples needed to select heads |
| Number of selected heads | $N_{\mathrm{opt}}\leq 10$ | Stable heads: 4 (Mistral) / 3 (Llama-2) |
| HotpotQA Multi-hop | Ours > all baselines | "In the wild" validation |
| Cross-dataset transfer (XSum↔CNN/DM) | Within 1σ | Selected heads are highly generalized |

### Key Findings
- **Minimal Heads Required**: Selecting $\le 10$ heads outperforms all baselines, suggesting hallucination signals are highly localized in the attention matrix rather than uniformly distributed.
- **Topological > Numerical Signals**: Methods using raw attention values (ReDeEP/LLM-Check) fluctuate near random (0.5), while TOHA's MST topology remains stable at 0.8+, revealing that "geometric structure" is more informative than "absolute weights."
- **Strong Cross-Task Transferability**: Heads identified on XSum maintain performance on CNN/DM, highlighting the transferability of the method as a core advantage.
- **Mechanistic Interpretability**: The selected heads overlap with known "copying heads," providing a link: high divergence $\Rightarrow$ insufficient copying $\Rightarrow$ hallucination.

## Highlights & Insights
- **Correct Application of TDA**: While previous TDA work in NLP was often descriptive, Ours provides geometric (MSF length) and information-theoretic (MST length increment $\approx$ entropy) dualities for $\operatorname{MTop-Div}_G(R,P)$, making an abstract metric both computable and interpretable.
- **Intrinsic Value of "Hallucination-Aware Heads"**: This discovery reveals that hallucination is a local behavior of specific heads rather than a whole-network phenomenon, offering specific targets for mechanistic interpretability and potential intervention methods.
- **Clever Engineering in Zeroing Prompt Weights**: Eliminating internal semantic noise to focus on "cross-boundary distances" is a rational simplification specific to the detection task that could be applied to other graph tasks like domain adaptation.

## Limitations & Future Work
- **Dependence on Minimal Labels**: Although only 100 samples are needed, they must be labeled as "hallucination/grounded" to select heads; pure zero-shot scenarios require further study.
- **White-box Requirement**: Requires access to attention matrices, making it inapplicable to closed-source API models (GPT-4o, Claude).
- **RAG-Specific Scenario**: The assumption that divergence corresponds to prompt-response relationships is less clear in free-form generation without prompt context.
- **Homology Beyond 0-th Order**: This work only uses $\mathcal{B}_0$ (connected components); future work could explore $\mathcal{B}_1$ (loops) or higher-order persistent homology for more structural signals.
- **Future Directions**: Integrating with RLHF as a "low-divergence" regularizer or using TOHA signals to trigger retrieval rewriting in RAG systems.

## Related Work & Insights
- **vs. Consistency Methods**: Comparison with SelfCheckGPT/Semantic Entropy shows TOHA is ~70× faster than multi-generation approaches and more accurate on most datasets.
- **vs. Internal State Probing**: Unlike HaloScope or ReDeEP, TOHA avoids complex probe training and treats heads selectively via TDA, enhancing both efficiency and interpretability.
- **vs. TDA in NLP**: While Kushnareva (2021) and Tulchinskii (2023) used TDA for global topology in classification, Ours is the first to apply manifold topology divergence to prompt-response structures and prove MSF equivalence.

## Rating
- Novelty: ⭐⭐⭐⭐ Introduces manifold topology divergence to attention graphs with MSF equivalence proof.
- Experimental Thoroughness: ⭐⭐⭐⭐ 5 LLMs × 5 Datasets + HotpotQA + Cross-dataset transfer + Efficiency analysis.
- Writing Quality: ⭐⭐⭐⭐ Intuitive diagrams and comprehensive mathematical derivations.
- Value: ⭐⭐⭐⭐ 70× speedup with only 100 labels; directly applicable to industrial RAG deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Spotlight and Shadow: Attention-Guided Dual-Anchor Introspective Decoding for MLLM Hallucination Mitigation](spotlight_and_shadow_attention-guided_dual-anchor_introspective_decoding_for_mll.md)
- [\[ACL 2026\] Distorted or Fabricated? A Survey on Hallucination in Video LLMs](distorted_or_fabricated_a_survey_on_hallucination_in_video_llms.md)
- [\[ACL 2026\] Enhancing Hallucination Detection via Future Context](enhancing_hallucination_detection_via_future_context.md)
- [\[NeurIPS 2025\] Robust Hallucination Detection in LLMs via Adaptive Token Selection](../../NeurIPS2025/hallucination/robust_hallucination_detection_in_llms_via_adaptive_token_selection.md)
- [\[ACL 2026\] MeasHalu: Mitigation of Scientific Measurement Hallucinations for LLMs](meashalu_mitigation_of_scientific_measurement_hallucinations_for_large_language_.md)

</div>

<!-- RELATED:END -->
