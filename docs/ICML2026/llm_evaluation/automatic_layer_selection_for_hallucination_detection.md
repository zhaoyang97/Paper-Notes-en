---
title: >-
  [Paper Note] Automatic Layer Selection for Hallucination Detection
description: >-
  [ICML 2026][LLM Evaluation][Hallucination Detection] FEPoID (First Effective Peak of Intrinsic Dimension) is proposed as a training-free automatic layer selection criterion. Combined with the First Sentence Truncation (F…
tags:
  - "ICML 2026"
  - "LLM Evaluation"
  - "Hallucination Detection"
  - "Intermediate Layer Selection"
  - "Intrinsic Dimension"
  - "Hidden-state Probing"
  - "Large Language Models"
date: 2026-05-08
content_hash: 958b162654ebdb95
---

# Automatic Layer Selection for Hallucination Detection

**Conference**: ICML 2026  
**arXiv**: [2605.26366](https://arxiv.org/abs/2605.26366)  
**Code**: https://github.com/DesoloYw/Automatic-Layer-Selection-for-Hallucination-Detection  
**Area**: LLM Evaluation  
**Keywords**: Hallucination Detection, Intermediate Layer Selection, Intrinsic Dimension, Hidden-state Probing, Large Language Models

## TL;DR
FEPoID (First Effective Peak of Intrinsic Dimension) is proposed as a training-free automatic layer selection criterion. Combined with the First Sentence Truncation (FST) strategy, it consistently identifies near-optimal intermediate layers across various QA and summarization hallucination detection benchmarks, significantly outperforming existing baseline methods.

## Background & Motivation

**Background**: Large Language Models (LLMs) often generate fluent but factually incorrect outputs (hallucinations) in practical deployments. Detecting these hallucinations without modifying the model itself is a critical practical issue. Existing research indicates that hidden states in the intermediate layers of LLMs encode hallucination-related signals more strongly than the final layer, leading to the emergence of the hidden-state probing detection paradigm.

**Limitations of Prior Work**: Although intermediate layers contain richer hallucination signals, the position of the optimal layer varies significantly across different model architectures and datasets. Existing methods either use fixed intermediate layers (e.g., the middle layer) or evaluate all candidate layers one by one; the former is unreliable, while the latter is computationally expensive. There is a lack of an efficient and principled method for automatic layer selection.

**Key Challenge**: The position of the optimal layer depends on the model and the data, and no universal fixed selection rule exists. Furthermore, existing metrics used to measure layer quality (such as RankMe, curvature, and gradient norms), while useful in other scenarios, exhibit unstable performance for layer selection in hallucination detection.

**Goal**: (1) Systematically evaluate the effectiveness of various layer selection criteria for hallucination detection; (2) Propose a training-free, computationally efficient, and cross-model/dataset robust automatic layer selection method; (3) Resolve the issue of token position selection during representation extraction.

**Key Insight**: The authors observe that the evolution curve of the Intrinsic Dimension (ID) across layers presents a stable multi-modal pattern—the first peak appears in the intermediate layers, followed by a second, higher peak near the output layer. The authors hypothesize that the first peak captures abstract semantic information (relevant to hallucination detection), whereas the second peak primarily reflects surface lexical complexity (unhelpful for detection).

**Core Idea**: The "First Effective Peak of Intrinsic Dimension" (FEPoID) is selected as the layer selection criterion. Combined with First Sentence Truncation (FST) to remove noise at the end of generation, the two methods jointly achieve unsupervised and efficient hallucination detection.

## Method

### Overall Architecture
Under the hidden-state probing framework, the pretrained LLM remains frozen. Representations of the last token are extracted from selected layers to train a lightweight MLP classifier for hallucination detection (binary classification). The input consists of the concatenation of the prompt and the generated answer. The key problem lies in how to automatically select the optimal layer and the optimal token position.

### Key Designs

1. **FEPoID (First Effective Peak of Intrinsic Dimension)**:

    - **Function**: Automatically selects the optimal intermediate layer for hallucination detection without any labeled data or training.
    - **Mechanism**: The TwoNN estimator is used to calculate the intrinsic dimension $d_{\text{ID}}^{(\ell)}$ of the representation matrix $\mathbf{Z}^{(\ell)} \in \mathbb{R}^{N \times d}$ for each layer. Local maxima on the ID curve are identified by scanning from shallow to deep layers. A forward window $w$ (defaulting to 7) is introduced to filter spurious peaks: if a candidate peak layer $\ell$ satisfies $d_{\text{ID}}^{(\ell)} < d_{\text{ID}}^{(\min(\ell+w, L))}$ and the ID increases monotonically within the window, the peak is discarded. The earliest surviving peak's corresponding layer is selected.
    - **Design Motivation**: Choosing the layer with the maximum ID often selects terminal layers (which have high surface complexity but little semantic information). The first effective peak is positioned precisely where abstract semantic information is most abundant, and experiments confirm that this layer aligns closely with the oracle optimal layer.

2. **First Sentence Truncation (FST)**:

    - **Function**: Resolves the token position selection issue during representation extraction and removes noise introduced at the end of generation.
    - **Mechanism**: A rule-based sentence boundary detector is used to locate the terminal token of the first generated sentence. The hidden state of this position is extracted instead of the terminal token of the entire sequence. This requires no ground-truth answer labels and does not rely on auxiliary LLMs.
    - **Design Motivation**: During generation, LLMs (especially LLaMA) often continue generating after providing the answer in the first sentence, leading to three types of degradation—inconsistent continuation (the latter part contradicts the first sentence), semantic drift (deviating from the question topic), and degenerative repetition (repeatedly restating the same information). This noise contaminates the representation of the terminal token, which FST effectively circumvents.

3. **Systematic Evaluation of Layer Selection Criteria**:

    - **Function**: Comprehensively compares six existing layer selection criteria to establish a benchmark for hallucination detection.
    - **Mechanism**: Based on four hypotheses (rich semantics, task alignment, information compression, and efficient information capacity), six criteria—RankMe (information-theoretic), Validation Loss/RGN/SNR (gradient-based), Curvature, and ID (geometric)—are evaluated. MLPs are trained layer-by-layer across multiple models and datasets, and the AUROC is recorded.
    - **Design Motivation**: While these criteria perform well in their original contexts, they have never been systematically compared for automatic layer selection in hallucination detection. Experiments indicate that none can provide stable performance, thus motivating the proposal of FEPoID.

## Key Experimental Results

### Main Results (QA Tasks)

AUROC comparison across five QA datasets and two instruction-tuned models (extracting terminal token representations of the generated output, $w=7$):

| Method | CoQA | SQuAD | HotpotQA | TriviaQA | PsiLoQA | Mean |
|------|------|-------|----------|----------|---------|------|
| Pred. Entropy | 0.583 | 0.570 | 0.710 | 0.686 | 0.360 | 0.582 |
| Semantic Entropy | 0.500 | 0.552 | 0.445 | 0.551 | 0.608 | 0.531 |
| Lexical Similarity | 0.678 | 0.599 | 0.729 | 0.684 | 0.408 | 0.620 |
| EigenScore | 0.525 | 0.530 | 0.599 | 0.588 | 0.508 | 0.550 |
| Probing + Val Loss | 0.671 | 0.616 | 0.768 | **0.786** | 0.784 | 0.725 |
| Probing + Curvature | 0.632 | 0.618 | 0.741 | 0.737 | 0.757 | 0.697 |
| Probing + ID | 0.671 | 0.613 | 0.693 | 0.707 | 0.737 | 0.684 |
| **Probing + FEPoID** | **0.671** | **0.638** | **0.781** | 0.752 | **0.786** | **0.725** |

*Results above are for LLaMA-3.1-8B-Instruct. FEPoID achieves the best average AUROC and ranks first on Mistral-7B with an average AUROC of 0.853.*

### Summarization Task and Computational Efficiency

| Method | HaluEval | CNN/DM | Mean | Compute Time (s) |
|------|----------|--------|------|-------------|
| RankMe | 0.608 | 0.577 | 0.592 | 27.3 |
| Curvature | 0.549 | 0.592 | 0.571 | 45.2 |
| Val Loss | 0.596 | 0.586 | 0.591 | 29.6 |
| RGN | 0.571 | 0.582 | 0.577 | 58.2 |
| SNR | 0.553 | 0.547 | 0.550 | 57.9 |
| **FEPoID** | **0.617** | **0.600** | **0.608** | **10.1** |

*Results on LLaMA-3.1-8B-Instruct. FEPoID not only achieves the best detection performance but its computation time is only 1/3 to 1/6 that of other methods.*

### Key Findings
- FEPoID consistently demonstrates optimal or near-optimal performance across two types of tasks (QA and summarization), five model scales (1B-8B), and two tuning strategies (base and instruct), showcasing strong generalization capabilities.
- FST brings consistent AUROC improvements to all baseline methods (method-agnostic gain), with particularly significant improvements on LLaMA (as LLaMA generation is more prone to terminal noise), significantly improving Fisher separation and silhouette coefficients.
- The strategy of directly selecting the layer with the maximum ID tends to select layers that are too deep on datasets like HotpotQA and TriviaQA, leading to performance degradation; FEPoID stably avoids this trap through its forward window mechanism.
- Sensitivity analysis of the hyperparameter $w$ indicates that FEPoID is highly robust to the choice of $w$, with performance remaining stable over a wide range.

## Highlights & Insights
- The design of FEPoID is exceptionally elegant—it achieves training-free, unsupervised automatic layer selection using only TwoNN intrinsic dimension estimation plus a forward window. The computational overhead is negligible (approximately 10 seconds for all 32 layers), making it highly attractive for practical deployment.
- The "method-agnostic" nature of FST is highly practical: it improves not only hidden-state probing but also baseline methods from entirely different paradigms, such as uncertainty-based methods and lexical similarity, indicating that "terminal noise" is a widespread and underestimated issue.
- The "ID curve bimodal hypothesis" provides a new perspective for understanding hierarchical representations in Transformers: intermediate peak = abstract semantics, terminal peak = surface complexity. This insight is transferable to other downstream tasks requiring the selection of intermediate layer representations.

## Limitations & Future Work
- Experiments only covered models in the 1B-8B range; the layer selection behavior of larger models (70B+) may differ, and whether FEPoID's bimodal hypothesis still holds remains to be verified.
- FST relies on rule-based sentence boundary detection, which may not be applicable to non-English languages or generations with non-natural sentence structures (e.g., code, mathematical derivations).
- Currently verified only on QA and summarization tasks; the definition and distribution of hallucinations in open-ended generation (e.g., dialogue, creative writing) are different, and generalization needs testing.
- Dynamic layer selection could be explored—selecting different layers for different input samples or combining multi-layer representations to further improve detection performance.

## Related Work & Insights
- **INSIDE** (Chen et al., 2024): Utilizes LLM internal states for hallucination detection with a fixed intermediate layer selection; FEPoID provides a superior automated alternative.
- **Semantic Entropy** (Farquhar et al., 2024): Estimates uncertainty at the semantic level but requires multiple samplings; the hidden-state probing method in this paper requires only a single forward pass.
- **EigenScore** (Chen et al., 2024): Evaluates representation quality based on the properties of the hidden state covariance spectrum, but its layer selection strategy is suboptimal.
- **Relationship between ID and Layer Selection**: Cheng et al. (2025) found that layers near the maximum ID transfer first to downstream tasks; this paper further refines this to "the first effective peak is the optimal choice."

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Robust Hallucination Detection in LLMs via Adaptive Token Selection](../../NeurIPS2025/llm_evaluation/robust_hallucination_detection_in_llms_via_adaptive_token_selection.md)
- [\[CVPR 2026\] AdaBet: Gradient-free Layer Selection for Efficient Training of Deep Neural Networks](../../CVPR2026/llm_evaluation/adabet_gradient-free_layer_selection_for_efficient_training_of_deep_neural_netwo.md)
- [\[ICML 2026\] Building Reliable Long-Form Generation via Hallucination Rejection Sampling](building_reliable_long-form_generation_via_hallucination_rejection_sampling.md)
- [\[ICML 2026\] Token-Efficient Change Detection in LLM APIs](token-efficient_change_detection_in_llm_apis.md)
- [\[ICML 2026\] When Hallucination Costs Millions: Benchmarking AI Agents in High-Stakes Adversarial Financial Markets (CAIA)](when_hallucination_costs_millions_benchmarking_ai_agents_in_high-stakes_adversar.md)

</div>

<!-- RELATED:END -->
