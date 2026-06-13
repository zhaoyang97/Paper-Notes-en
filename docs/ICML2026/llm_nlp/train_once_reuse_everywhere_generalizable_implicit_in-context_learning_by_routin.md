---
title: >-
  [Paper Note] In-Context Routing (ICR): Train Once, Use Everywhere via Attention-Level Implicit ICL
description: >-
  [ICML 2026][LLM/NLP][Implicit ICL] ICR does not inject shift vectors into the residual stream. Instead, it extracts Principal ICL Directions (PIDs) from multi-domain ICL via PCA as low-rank correction directions for atte…
tags:
  - "ICML 2026"
  - "LLM/NLP"
  - "Implicit ICL"
  - "Attention Routing"
  - "Principal ICL Directions"
  - "Cross-domain Generalization"
  - "Zero-shot Inference"
date: 2026-05-08
content_hash: 8cb7787a9456ce52
---

# In-Context Routing (ICR): Train Once, Use Everywhere via Attention-Level Implicit ICL

**Conference**: ICML 2026  
**arXiv**: [2509.22854](https://arxiv.org/abs/2509.22854)  
**Code**: https://github.com/Lijiaqian1/In-Context-Routing.git  
**Area**: LLM Efficient Inference / Implicit ICL / Attention Editing  
**Keywords**: Implicit ICL, Attention Routing, Principal ICL Directions, Cross-domain Generalization, Zero-shot Inference

## TL;DR
ICR does not inject shift vectors into the residual stream. Instead, it extracts Principal ICL Directions (PIDs) from multi-domain ICL via PCA as low-rank correction directions for attention logits, modulated adaptively by a query-conditioned router. After being trained once, it enables zero-shot inference across 12 in/out-of-domain tasks without task-specific retrieval or retraining, avoiding the degradation seen in vector-based methods on OOD tasks.

## Background & Motivation

**Background**: In-Context Learning (ICL) allows LLMs to learn new tasks by adding few-shot examples to the prompt. However, it faces two primary issues: (1) In-context demonstrations (ICDs) increase sequence length and double inference costs; (2) Performance is brittle and sensitive to the order/format of ICDs. Implicit ICL converts ICDs into dense vectors injected into the model's hidden layers to simulate ICL effects (Hendel et al. 2023, Liu et al. 2023, Li et al. 2024).

**Limitations of Prior Work**: Vector-based implicit ICL adds a shift vector $\mathbf{V}^l_{\mathrm{shift}}$ to the residual stream ($\tilde{\mathbf{h}}^l = \mathbf{h}^l + \beta^l \cdot \mathbf{V}^l_{\mathrm{shift}}$), but significant limitations exist: (1) Fixed-size vectors have limited capacity; adding new knowledge requires new vectors. (2) Post-hoc addition to the residual stream lacks control over information flow. (3) These methods are tied to specific tasks during training and degrade significantly—sometimes performing worse than zero-shot—on OOD tasks (e.g., M2IV collapses on 3 out of 7 OOD tasks).

**Key Challenge**: Generalization requires a "task-agnostic ICL pattern," but the shifts in vector-based methods are task-specific. To achieve "task-agnostic" behavior, the ICL mechanism itself (rather than the ICD content) must be distilled.

**Goal**: To identify a task-agnostic ICL pattern that is reusable across domains after one-time training, without relying on task retrieval or retraining.

**Key Insight**: Observations show that multi-task ICL prompting can sometimes outperform zero-shot or the strongest single-source few-shot models, but it can also be a hindrance (Figure 1). This suggests that the "utility" of ICL lies in latent cross-task patterns rather than specific ICD content; explicit prompting introduces noise that masks this pattern. Therefore, one should delve into the attention space to extract this pattern.

**Core Idea**: By performing explicit ICL across multiple domains, the query/key projections of the last token of each prompt are collected. PCA is then used to extract Principal ICL Directions (PIDs) $U_q^l, U_k^l \in \mathbb{R}^{d \times r}$. A query-conditioned router calculates a routing vector $\alpha^l(x)$ and a head gate $\gamma^l(x)$ to add a low-rank correction $\Delta \mathbf{A}^l = (Q_{\mathrm{zs}} U_q^l) \mathrm{diag}(\alpha^l) (K_{\mathrm{zs}} U_k^l)^\top$ to the attention logits.

## Method

### Overall Architecture

The framework consists of three stages: (1) **PIDs Extraction**: Explicit ICL is run across $\mathbb{D}$ domains. The $Q^l, K^l$ projections of the last token of each prompt are stacked into ICL bases $\tilde{Q}^l, \tilde{K}^l \in \mathbb{R}^{N \times d}$. PCA extracts the top-$r$ principal directions to obtain $U_q^l, U_k^l$. (2) **Query-conditioned Router Training**: The LLM is frozen while two MLPs $g_{\theta_\alpha}, g_{\theta_\gamma}$ are trained to output $\alpha(x) \in \mathbb{R}^{L \times r}$ and gates $\gamma(x) \in \mathbb{R}^{L \times H}$ based on the query embedding. The loss function includes CE, confidence alignment, sparsity, and gate regularization. (3) **Zero-shot Inference**: For any new query, the router calculates $\alpha$ and $\gamma$, and the attention logits are corrected as $\tilde{\mathbf{A}}^{l,h}(x) = \mathbf{A}^{l,h}(x) + \gamma^{l,h}(x) (Q_{\mathrm{zs}}^l U_q^l) \mathrm{diag}(\alpha^l(x)) (K_{\mathrm{zs}}^l U_k^l)^\top$.

### Key Designs

1.  **Principal ICL Directions (PIDs): Distilling Cross-domain ICL Patterns via PCA**:
    - **Function**: Extracts task-agnostic and generalizable structural directions from the attention space to serve as "raw materials" for attention correction.
    - **Mechanism**: The last token of each prompt serves as an integration point for contextual information; its $Q/K$ projection carries the signal to "respond using ICL mode." After collecting $\tilde{Q}^l, \tilde{K}^l$ across $\mathbb{D}$ domains, PCA yields $U_q^l, U_k^l \in \mathbb{R}^{d \times r}$. Theoretical support comes from the Spiked Covariance Model: $\Sigma_Q^{(\mathbb{d})} = S_q \Lambda_q S_q^\top + B_{q, \mathbb{d}} \Gamma_{q, \mathbb{d}} B_{q, \mathbb{d}}^\top + \sigma^2 I$, where $S_q$ represents the cross-domain shared ICL structure and $B_{q, \mathbb{d}}$ represents domain-specific variations. If $\{B_{q, \mathbb{d}}\}$ is diverse enough, the third term in the pooled covariance $\mathbb{E}[\hat{\Sigma}_Q]$ averages out to isotropy, allowing PCA eigenvectors to recover $S_q$—the stable cross-domain ICL pattern.
    - **Design Motivation**: Vector-based methods compress ICD information into a fixed-size shift vector with limited capacity. PCA on attention bases directly captures the structural pattern of "what query-key matching geometry makes ICL effective," which is geometrically closer to the true mechanism of ICL than additive shifts.

2.  **Query-conditioned Router: Low-rank Modulation + Head Gate**:
    - **Function**: Adaptively determines the intensity of each PID and the participation of each head per layer based on the current query, without task labels.
    - **Mechanism**: A frozen text encoder computes the query embedding $E(x)$. Two parallel 2-layer MLPs are used: $\alpha(x) = \tanh(g_{\theta_\alpha}(E(x))) \in \mathbb{R}^{L \times r}$ assigns weights $\in [-1, 1]$ to each direction, while $\gamma(x) = \sigma(g_{\theta_\gamma}(E(x))) \in \mathbb{R}^{L \times H}$ provides gates $\in [0, 1]$ for each head. The corrected attention logits are $\tilde{\mathbf{A}}^{l,h}(x) = \mathbf{A}^{l,h}(x) + \gamma^{l,h}(x) (Q_{\mathrm{zs}}^l U_q^l) \mathrm{diag}(\alpha^l(x)) (K_{\mathrm{zs}}^l U_k^l)^\top$. This design uses layer-shared biases multiplied by head gates to maintain differentiation while controlling parameter count.
    - **Design Motivation**: Fixed routing is prone to overfitting. Query-conditioning allows the router to learn which ICL patterns a specific query requires. Tanh allows $\alpha$ to be positive or negative (enhancing or inhibiting a PID), while Sigmoid allows heads to be selectively activated. Together, they provide fine-grained task-adaptive routing with minimal parameters ($\le 10M$ vs. 7B).

3.  **Loss & Training**:
    - **Function**: Ensures the router learns correct answers (CE) without reducing zero-shot confidence (preventing degradation), while encouraging sparse routing for interpretability and minimal intervention.
    - **Mechanism**: (a) Supervised $\mathcal{L}_{\mathrm{CE}} = -\frac{1}{B} \sum_i \log P^{\mathrm{ICR}}(y_i | x_i)$; (b) Confidence alignment $\mathcal{L}_{\mathrm{conf}} = \frac{1}{B} \sum \mathrm{ReLU}(H(\mathrm{softmax}(p_i^{\mathrm{ICR}})) - H(\mathrm{softmax}(p_i^{\mathrm{zs}})))$; (c) Sparse routing $\mathcal{L}_{\mathrm{spar}} = \mathbb{E}_x[\frac{1}{L} \sum_l w^l \|\alpha^l(x)\|_1 / r]$ with linearly increasing $w^l$; (d) Head gate $\mathcal{L}_{\mathrm{gate}} = \mathbb{E}_x[\frac{1}{L} \sum_l \|\gamma^l(x)\|_1 / H]$. The total loss is $\mathcal{L} = \mathcal{L}_{\mathrm{CE}} + \lambda_{\mathrm{conf}} \mathcal{L}_{\mathrm{conf}} + \lambda_{\mathrm{spar}} \mathcal{L}_{\mathrm{spar}} + \lambda_{\mathrm{gate}} \mathcal{L}_{\mathrm{gate}}$.
    - **Design Motivation**: Pure CE can lead the router to "bypass" the ICL mechanism. Confidence alignment forces ICR to be at least as confident as zero-shot. Sparser loss in deeper layers reflects the LLM's structure of "broad processing in early layers, specific decision-making in later layers."

### Inference

For any new query: (1) Compute $E(x)$; (2) Router outputs $\alpha(x), \gamma(x)$; (3) Compute attention correction via Eq. 10; (4) Perform standard forward pass. This process requires no retrieval, no retraining, and has negligible computational overhead.

## Key Experimental Results

### Main Results: 12 Benchmarks (5 ID + 4 Near OOD + 3 Far OOD)

| Model/Method | AG | SST-2 | TREC | CSQA | PIQA | SST-5 | MR | MRPC | CB | COPA | CREAK | AI2SciE | Average | Collapse |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Llama2-7B** | | | | | | | | | | | | | | |
| Zero-shot | 67.0 | 78.6 | 56.6 | 22.4 | 52.2 | 25.8 | 72.2 | 44.4 | 37.5 | 63.0 | 51.8 | 34.8 | 50.5 | – |
| Few-shot* | 81.0 | 95.2 | 84.6 | 58.0 | 59.8 | 37.4 | 98.6 | 68.2 | 41.1 | 82.0 | 50.8 | 45.4 | 66.8 | 1 |
| I2CL | 85.5 | 86.0 | 78.6 | 23.8 | 55.6 | 27.6 | 71.6 | 42.4 | 38.2 | 63.6 | 52.6 | 35.0 | 55.0 | **2** |
| LIVE | 86.0 | 86.2 | 81.0 | 24.2 | 56.4 | 32.8 | 73.8 | 47.6 | 40.8 | 64.8 | 51.0 | 34.6 | 56.6 | **2** |
| M2IV | 86.4 | 86.4 | 81.5 | 24.8 | 56.8 | 30.8 | 74.0 | 46.0 | 42.6 | 64.8 | 54.0 | 35.2 | 56.9 | 0 |
| **ICR** | 86.6 | 86.4 | 83.8 | 24.8 | 57.0 | **38.6** | **79.8** | **53.4** | **46.4** | **68.0** | **56.4** | **37.2** | **59.9** | **0** |
| **Qwen2.5-7B** | | | | | | | | | | | | | | |
| Zero-shot | 66.8 | 54.0 | 65.8 | 80.4 | 76.2 | 31.4 | 64.4 | 72.4 | 83.9 | 92.0 | 77.8 | 90.4 | 71.3 | – |
| Few-shot* | 80.2 | 95.6 | 67.6 | 82.2 | 86.0 | 37.2 | 70.2 | 76.2 | 83.9 | 95.0 | 59.7 | 95.8 | 77.5 | 1 |
| I2CL | 77.0 | 86.4 | 68.6 | 81.6 | 81.2 | 34.6 | 69.0 | 70.8 | 80.6 | 92.6 | 74.8 | 91.8 | 75.6 | **3** |
| LIVE | 79.0 | 87.8 | 70.4 | 81.6 | 82.0 | 30.8 | 68.6 | 69.4 | 81.0 | 93.2 | 72.8 | 91.8 | 75.7 | **4** |
| M2IV | 79.6 | 89.0 | 70.8 | 81.8 | 82.5 | 31.6 | 71.2 | 71.0 | 76.0 | 93.5 | 74.6 | 92.4 | 76.2 | **3** |
| **ICR** | 80.4 | 91.0 | 70.6 | 82.0 | 82.6 | **41.4** | **89.4** | 73.2 | **84.6** | 95.0 | **79.2** | **93.2** | **80.2** | **0** |

ICR achieves SOTA on both LLMs: Llama2-7B (59.9 vs. M2IV 56.9) and Qwen2.5-7B (80.2 vs. M2IV 76.2). Most importantly, **Collapse=0**—while other baselines often perform worse than zero-shot on OOD tasks, ICR exhibits no degradation.

### Key Findings

- **ICR is the only method without OOD collapse**: On Qwen, baselines collapse on 3-4 OOD tasks; ICR remains robust, proving attention-level patterns are more generalizable than residual-level vectors.
- **Cross-model Effectiveness**: The design is model-agnostic, working across Llama2-7B and Qwen2.5-7B.
- **Maximum Gain on MR (Near OOD)**: On Qwen, ICR (89.4) significantly outperforms others (~70), showing the power of attention-level routing in near-domain tasks.
- **Outperforming Few-shot on CREAK (Far OOD)**: ICR (56.4) surpasses few-shot (50.8) on Llama, proving it captures the mechanism of "how to do ICL" more stably than explicit examples.
- **No Task Retrieval Required**: Unlike M2IV/LIVE, which require searching for corresponding vectors at inference, ICR uses the same PIDs and router for all tasks.

## Highlights & Insights

- **Attention-level paradigm shift**: Moving from "ICL as hidden state shift" to "ICL as attention geometry modulation" aligns more closely with the mechanistic nature of ICL.
- **Theoretical Grounding**: Using Spiked Covariance + PCA provides a mathematical basis for the extraction of the cross-domain shared structure $S_q$.
- **Train once, reuse everywhere**: This is the first implicit ICL method to truly achieve task-agnostic generalization.
- **Structural Multi-target Loss**: Every loss term (confidence alignment, sparsity, etc.) is motivated by model structure and verified by ablation.
- **Minimal Overhead**: Router parameters are negligible ($\le 10M$), and inference latency is identical to zero-shot pass.

## Limitations & Future Work

- **Requirement for initial ICL data**: Extracting PIDs still requires labeled examples from multiple domains initially.
- **Domain mix impact**: The composition of training data may affect OOD generalization; the optimal domain mix is not fully explored.
- **Hyperparameter $r$**: The rank of PIDs is a manual choice; too low results in information loss, while too high may lead to overfitting.
- **Reasoning-heavy tasks**: Validation was primarily on classification/QA; the effect on long CoT tasks (math, code) remains unknown.
- **Scalability**: Experiments were conducted on 7B models; scaling behavior to 70B+ models is yet to be discussed.

## Related Work & Insights

- **vs. I2CL / LIVE / M2IV**: These are previous vector-based works. ICR proves that their OOD collapse is a paradigm issue (residual-level shift) rather than an engineering one.
- **vs. ICV / FV / TV**: These are attention-space methods but are task-specific. ICR is the first task-agnostic + train-once attention routing method.
- **vs. Mixture of Experts**: The routing concept is similar, but ICR acts on attention space rather than FFNs, and "experts" are PID directions rather than full MLPs, reducing parameter counts by three orders of magnitude.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] ICL-Router: In-Context Learned Model Representations for LLM Routing](../../AAAI2026/llm_nlp/icl-router_in-context_learned_model_representations_for_llm_routing.md)
- [\[ICML 2026\] From Parameter Dynamics to Risk Scoring: Quantifying Sample-Level Safety Degradation in LLM Fine-tuning](from_parameter_dynamics_to_risk_scoring_quantifying_sample-level_safety_degradat.md)
- [\[ICML 2026\] SLAY: Geometry-Aware Spherical Linearized Attention with Yat-Kernel](slay_geometry-aware_spherical_linearized_attention_with_yat-kernel.md)
- [\[ICML 2026\] Deep Networks Learn to Parse Uniform-Depth Context-Free Languages from Local Statistics](deep_networks_learn_to_parse_uniform-depth_context-free_languages_from_local_sta.md)
- [\[ACL 2026\] C-World: A Computer Use Agent Environment Creator](../../ACL2026/llm_nlp/c-world_a_computer_use_agent_environment_creator.md)

</div>

<!-- RELATED:END -->
