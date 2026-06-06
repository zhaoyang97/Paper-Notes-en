---
title: >-
  [Paper Note] FedSDR: Federated Self-Distillation with Rectification
description: >-
  [ICML 2026][Model Compression][Federated Learning] To address "weight drift" caused by heterogeneous client data distributions in federated LLM fine-tuning…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "Federated Learning"
  - "LLM Instruction Fine-tuning"
  - "Self-Distillation"
  - "Dual-Stream LoRA"
  - "Selective Aggregation"
date: 2026-05-08
content_hash: 4041643f58e30110
---

# FedSDR: Federated Self-Distillation with Rectification

**Conference**: ICML 2026  
**arXiv**: [2605.18028](https://arxiv.org/abs/2605.18028)  
**Code**: TBD  
**Area**: Model Compression / Federated Fine-tuning / Self-Distillation  
**Keywords**: Federated Learning, LLM Instruction Fine-tuning, Self-Distillation, Dual-Stream LoRA, Selective Aggregation  

## TL;DR
To address "weight drift" caused by heterogeneous client data distributions in federated LLM fine-tuning, this paper first uses the model itself to rewrite original instructions into a "model-understandable space" for data-level alignment (FedSD). It then employs a LoRA-S/LoRA-R dual-stream structure to absorb stylistic noise and anchor factual correctness, respectively, aggregating only LoRA-R. By decoupling alignment from fidelity, it achieves SOTA results across multiple Non-IID settings.

## Background & Motivation
**Background**: Under privacy constraints, Federated Learning (FL) has become the standard paradigm for LLM instruction fine-tuning. Parameter-efficient LoRA is widely adopted due to low communication costs. To mitigate "client drift" caused by Non-IID data, mainstream approaches either add regularization to local updates (FedProx, SCAFFOLD) or use dual LoRA structures to separate "global sharing" from "client personalization" (e.g., FedDPA).

**Limitations of Prior Work**: These methods treat heterogeneity as a "weight divergence" phenomenon, suppressing symptoms like conflicting gradient directions or performance degradation after aggregation. However, the fundamental data distribution mismatch remains unaddressed. When client data contains high-semantic-variance combinations (e.g., finance, medicine, code), constraints on weights alone are insufficient, leading to severe degradation of the aggregated model on multi-task averages.

**Key Challenge**: The root of heterogeneity lies in the data rather than the model. The broadcasted global model must simultaneously fit several "nearly disjoint" client manifolds. Strong constraints stifle personalization, while weak constraints fail to suppress drift, creating a trade-off that cannot satisfy both parties at the model level.

**Goal**: Directly reduce the distributional distance between clients from a data perspective while ensuring the aggregated global model is both "aligned" and "faithful to facts."

**Key Insight**: The authors observe that LLMs already possess a coherent "innate knowledge distribution." Letting the model rewrite every client's response in its own tone is equivalent to projecting all client data onto the same generative manifold. Measures such as t-SNE, JS divergence, TF-IDF cosine similarity, and cross-task gradient directions verify this "manifold flattening" effect.

**Core Idea**: Assign "alignment" to LoRA-S (kept locally, absorbing style and heterogeneous noise) trained on self-distilled data, and "fidelity" to LoRA-R (uploaded for aggregation) trained on original data. This structural decoupling achieves both alignment and fidelity.

## Method
The philosophy of FedSDR is a three-stage pipeline: "Generate – Align – Rectify." It uses the local LLM to refine heterogeneous raw data once, then utilizes dual LoRA paths to alternately learn the interfering objectives of "smoothing" and "rectification." Finally, it performs selective aggregation at the server. The system does not increase communication bits (aggregating only one set of LoRA), but avoids "hallucination aggregation" by keeping noise local and sending facts to the cloud.

### Overall Architecture
For a participating client $k$, it receives the current global parameters $\Theta_{global}^t$ from the server at each round $t$. These parameters serve three roles: 1) Teacher $\Theta_{teacher}$, used for one-time self-distillation of local raw data $D_{raw}=\{(c_i,x_i,y_i)\}$ to generate $D_{dist}=\{(c_i,x_i,\tilde y_i)\}$, where $\tilde y_i\sim f_{\Theta_{teacher}}(\cdot\mid c_i,x_i,y_i)$; 2) Student backbone $W_0$, which undergoes alternating optimization with two LoRA adapters $\{A_s,B_s\}$ and $\{A_r,B_r\}$ injected; 3) Upload anchor — only the LoRA-R increment $\Delta\Theta_{r,k}$ is pushed to the server for weighted averaging, while LoRA-S remains local. The forward pass uses a parallel structure $h_{out}=W_0 h+\tfrac{\alpha}{r}B_r A_r h+\tfrac{\alpha}{r}B_s A_s h$, allowing LoRA-S to provide a "smoothed" hidden representation for LoRA-R. Self-distillation occurs only once before federated training; all subsequent communication rounds reuse $D_{dist}$, ensuring inference costs do not grow linearly with rounds.

### Key Designs
1.  **Data-level Self-Distillation (Data Refinery, FedSD)**:
    - **Function**: Rewrites heterogeneous raw instruction-response pairs into the model's own response distribution locally, resulting in a "style-unified, semantically-aligned" training set.
    - **Mechanism**: Client $k$ uses the global model as a teacher to generate $\tilde y_i$ for each sample, constructing $D_{dist}$. Standard aggregation algorithms (FedAvg/Prox/Yogi) are run as usual, replacing the supervision signal $y$ with $\tilde y$. The local objective is $\mathcal{L}^k_{\text{FedSD}}(\Theta)=\tfrac{1}{n_k}\sum_i -\log f_\Theta(\tilde y_i\mid c_i,x_i)$. The "manifold flattening" is quantified by a drop in JS divergence from $0.4074$ to $0.3611$, an increase in TF-IDF cosine similarity from $0.6362$ to $0.7064$, and an average increase of $+8\sim+41$ pp in pairwise gradient cosine similarity across five tasks.
    - **Design Motivation**: It is more economical to digest heterogeneity "at the source" than to resolve weight conflicts during aggregation. This modification is orthogonal to any aggregation algorithm.

2.  **Dual-Stream LoRA + Alternating Block Coordinate Training (Dual-Stream Rectification)**:
    - **Function**: One stream absorbs side effects of self-distillation (the "rewriting paradox"—about $47\%$ of distilled samples do not strictly entail the ground truth, are lengthier, and double the filler word frequency), while the other stream focuses on original facts.
    - **Mechanism**: Two LoRA sets share the same forward pass $h_{out}=W_0h+\tfrac{\alpha}{r}(B_rA_r+B_sA_s)h$, but training occurs in two stages. Stage 1 freezes $\Theta_r$ and minimizes $\mathcal{L}_{smooth}(\Theta_s\mid\Theta_r)=\mathbb{E}_{D_{dist}}[-\log p_{\Theta_r,\Theta_s}(\tilde y\mid c,x)]$ using $D_{dist}$. Stage 2 freezes $\Theta_s$ and minimizes $\mathcal{L}_{rect}(\Theta_r\mid\Theta_s)=\mathbb{E}_{D_{raw}}[-\log p_{\Theta_r,\Theta_s}(y\mid c,x)]$ using $D_{raw}$. Stage 1 smooths the representation space, reducing the optimization difficulty for $\Theta_r$ in Stage 2, acting as a "shock absorber."
    - **Design Motivation**: Decoupling "distribution alignment" and "factual correctness" into two sets of parameters prevents the conflict inherent in a single model attempting both contradictory tasks.

3.  **Selective Aggregation**:
    - **Function**: Ensures global consensus inherits only "rectified factual knowledge" rather than "hallucinations and verbosity" from client self-distillation.
    - **Mechanism**: During the upload phase, each client keeps $\Theta_s$ local (or resets it) and only pushes $\Delta\Theta_{r,k}$ to the server. The server performs weighted averaging: $\Theta_{r,global}\leftarrow\Theta_{r,global}+\sum_{k=1}^{K}\tfrac{n_k}{n}\Delta\Theta_{r,k}$. Since $\Theta_s$ absorbed stylistic noise in Stage 1, keeping it local prevents the noise from forming a "pseudo-consensus" through aggregation.
    - **Design Motivation**: The authors point out that FL is more vulnerable to the "rewriting paradox" than centralized training; slight local hallucinations can be broadcast via the server, creating a positive feedback loop that leads to cumulative deviation.

### Loss & Training
Local training follows alternating block coordinate updates: $\Theta_s\leftarrow\arg\min_{\Theta_s}\mathcal{L}_{smooth}(\Theta_s\mid\Theta_r)$ and $\Theta_r\leftarrow\arg\min_{\Theta_r}\mathcal{L}_{rect}(\Theta_r\mid\Theta_s)$. Both stages share the same forward pass, but only one stream is trainable at a time. The server only aggregates $\Theta_r$, maintaining communication costs consistent with single-LoRA baselines. Self-distillation is performed once, avoiding cumulative inference costs.

## Key Experimental Results

### Main Results
FedSD, as a general enhancer, was applied to 6 classic aggregation algorithms. Overall scores and four evaluation subsets (MMLU, BBH, CRASS, DROP) all improved, with Head-to-Head Win Rates exceeding 55%.

| Algorithm | Overall (Base → Ours) | MMLU (Base → Ours) | BBH (Base → Ours) | CRASS (Base → Ours) | DROP (Base → Ours) | Win Rate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| FedAvg | 71.21 → 74.54 | 40.71 → 43.69 | 30.79 → 31.88 | 47.81 → 56.57 | 36.04 → 36.41 | 56.14% |
| FedAvgM | 68.18 → 73.34 | 8.72 → 11.81 | 6.71 → 8.18 | 12.77 → 20.07 | 15.64 → 15.75 | 60.30% |
| FedProx | 70.93 → 74.96 | 40.54 → 42.94 | 30.82 → 31.35 | 45.62 → 50.73 | 37.50 → 36.68 | 56.12% |
| FedYogi | 69.17 → 73.56 | 29.92 → 39.32 | 29.12 → 30.50 | 42.34 → 45.26 | 24.53 → 29.28 | 60.37% |
| FedAdam | 71.03 → 74.93 | 30.49 → 39.36 | 28.51 → 30.69 | 36.50 → 51.46 | 27.32 → 30.54 | 57.82% |
| FedAdagrad | 71.87 → 75.13 | 40.69 → 43.21 | 31.43 → 32.56 | 43.07 → 51.09 | 35.80 → 36.21 | 56.78% |

### Ablation Study
The mechanisms of "rewriting paradox" and "distribution alignment" are supported by quantitative data: decreased JS divergence and increased TF-IDF cosine similarity verify text-level alignment, while cross-task gradient cosine gains verify optimization-level alignment.

| Dimension | Metric | Raw Data | Distilled Data | Change |
| :--- | :--- | :--- | :--- | :--- |
| Text Alignment | JS Divergence ↓ | 0.4074 | 0.3611 | $-0.0463$ |
| Text Alignment | TF-IDF Cosine ↑ | 0.6362 | 0.7064 | $+0.0702$ |
| Cross-task Gradients (FinGPT) | Cosine Sim ↑ | — | — | $+41.5$ pp |
| Cross-task Gradients (MathInstruct) | Cosine Sim ↑ | — | — | $+13.8$ pp |
| Cross-task Loss Transfer (MedAlpaca) | Loss Change ↓ | — | — | $-5.4$ pp |
| Rewriting Paradox | Non-entailed Samples Ratio ↓ | — | $\approx 47\%$ | — |

### Key Findings
- **FedSD as a general enhancer**: Experimental results show improvements across 6 aggregation algorithms and 4 evaluation sets, with the largest gain in FedAvg+CRASS ($+8.76$ pts). Aligning data at the source provides greater benefits than any single aggregation refinement.
- **The "Rewriting Paradox" is real and dangerous in FL**: Approximately $47\%$ of distilled samples do not strictly entail the ground truth. Without LoRA-R to anchor original data, these flaws are broadcast by the server, forming a pseudo-consensus.
- **Dual-stream is more than parameter stacking**: LoRA-S provides a smooth hidden representation in the forward pass. Observations indicate that this makes the optimization curve for LoRA-R on raw data smoother ("implicit smoothing"), supporting the premise that data alignment is the root cause.
- **FedAvgM Baseline**: The initial baseline (MMLU 8.72, BBH 6.71) was significantly lower than other algorithms, but FedSD boosted the overall score to 73.34 and win rate to 60.30%, suggesting that data-level alignment is more effective for unstable aggregators.

## Highlights & Insights
- Redefining the heterogeneity problem from "weight divergence" to "data manifold misalignment" is a clean perspective shift. The use of t-SNE, JS divergence, and gradient cosine similarity provides robust evidence for "manifold flattening."
- The dual-stream LoRA design is elegant: two streams are parallel in the forward pass sharing one backbone, but decoupled during block-coordinate descent. This structure avoids objective conflicts from multi-task averaging and can be adapted for other alignment vs. fidelity scenarios (e.g., RLHF, safety alignment).
- Selective aggregation insight: In FL, if noise enters the aggregation, it is amplified via broadcast. Keeping noise local at the protocol layer treats "entropy management" as a system design rather than just an algorithmic one.

## Limitations & Future Work
- The cost of one-time $D_{dist}$ generation is non-negligible for ultra-large models and may become a bottleneck for edge devices.
- Keeping LoRA-S permanently local introduces structural differences between client and global models. Robustness needs verification for long-cycle deployment, client churn, or new client cold starts.
- The "rewriting paradox" is characterized mainly by qualitative case studies and a few statistics; a unified "factuality/style" metric could allow for finer-grained evaluation of LoRA-R's efficacy.
- Experiments focused on LLM instruction fine-tuning. Evidence is needed for more complex output spaces like multi-modal or Tool-Use tasks.

## Related Work & Insights
- **vs FedProx / SCAFFOLD**: These add regularization to weights to constrain client drift (treating symptoms). FedSDR performs projection at the data level, providing orthogonal gains.
- **vs FedDPA (Dual LoRA)**: FedDPA uses dual LoRA for "global sharing + local personalization" (model-centric). FedSDR decouples "distribution smoothing vs. factual rectification" (higher-level objective), explicitly addressing hallucination aggregation.
- **vs FedGen / FedDistill (Federated KD)**: These rely on public datasets or generators to pass logits, which is impractical for LLMs. FedSDR uses the model itself as a teacher.
- **vs Centralized Self-Distillation (WizardLM / Alpaca)**: Centralized methods accept the "rewriting paradox" for instruction quality. This paper argues that in FL, this paradox must be mitigated due to aggregation broadcast effects, upgrading self-distillation from a data-augmentation trick to a distribution-management tool.
- **vs Personalized FL (pFedMe / Per-FedAvg)**: These allow personalized models but sacrifice global consensus. FedSDR produces a unified global LoRA-R while absorbing heterogeneity into local LoRA-S, suiting "unified deployment + local adaptation" scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] FedRot-LoRA: Mitigating Rotational Misalignment in Federated LoRA](fedrot-lora_mitigating_rotational_misalignment_in_federated_lora.md)
- [\[ICCV 2025\] Soft Separation and Distillation: Toward Global Uniformity in Federated Unsupervised Learning](../../ICCV2025/model_compression/soft_separation_and_distillation_toward_global_uniformity_in_federated_unsupervi.md)
- [\[ICML 2026\] PRISM: Synergizing Vision Foundation Models via Self-Organized Expert Specialization](prism_synergizing_vision_foundation_models_via_self-organized_expert_specializat.md)
- [\[ICML 2026\] OSAQ: Outlier Self-Absorption for Accurate Low-bit LLM Quantization](osaq_outlier_self-absorption_for_accurate_low-bit_llm_quantization.md)
- [\[ICML 2026\] UB-SMoE: Universally Balanced Sparse Mixture-of-Experts for Resource-Adaptive Federated Fine-tuning of Foundation Models](ub-smoe_universally_balanced_sparse_mixture-of-experts_for_resource-adaptive_fed.md)

</div>

<!-- RELATED:END -->
