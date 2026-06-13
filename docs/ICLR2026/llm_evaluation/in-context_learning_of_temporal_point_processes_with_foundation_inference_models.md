---
title: >-
  [Paper Note] In-Context Learning of Temporal Point Processes with Foundation Inference Models
description: >-
  [ICLR 2026][LLM Evaluation][Temporal Point Processes] This paper proposes FIM-PP — the first foundation inference model for marked temporal point processes (MTPP). A Transformer is pretrained on 72K synthetic point proce…
tags:
  - "ICLR 2026"
  - "LLM Evaluation"
  - "Temporal Point Processes"
  - "Foundation Inference Model"
  - "In-Context Learning"
  - "Hawkes Process"
  - "Conditional Intensity Function"
date: 2026-05-08
content_hash: 39e11bb02cf1b072
---

# In-Context Learning of Temporal Point Processes with Foundation Inference Models

**Conference**: ICLR 2026
**arXiv**: [2509.24762](https://arxiv.org/abs/2509.24762)  
**Code**: [OpenFIM](https://fim4science.github.io/OpenFIM/intro.html)  
**Area**: LLM Evaluation
**Keywords**: Temporal Point Processes, Foundation Inference Model, In-Context Learning, Hawkes Process, Conditional Intensity Function

## TL;DR

This paper proposes FIM-PP — the first foundation inference model for marked temporal point processes (MTPP). A Transformer is pretrained on 72K synthetic point processes (14.4M events) to perform in-context inference of conditional intensity functions. In zero-shot settings, FIM-PP matches the performance of specialized models trained for hours; after a few minutes of fine-tuning, it achieves state-of-the-art results on multi-event prediction across four real-world datasets.

## Background & Motivation

**Background**: Marked temporal point processes (MTPP) are the standard framework for modeling asynchronous, irregular event sequences, with applications in financial trading, social media diffusion, neural spike trains, and epidemiology. The central mathematical object is the conditional intensity function $\lambda(t,\kappa|\mathcal{H}_t)$ — the instantaneous rate of events of each type at a future time given the history. Classical Hawkes processes use linear self-exciting kernels to model excitation and inhibition between events; subsequent neural methods (NHP, A-NHP, etc.) incorporate RNN/Transformer encoders for history, but all follow a "train one model per dataset" paradigm.

**Limitations of Prior Work**: Each new event sequence dataset requires training from scratch, which can take hours, and learned representations do not transfer across systems. Meanwhile, foundation models have emerged in NLP, ODE, and SDE domains (e.g., ODEFormer, FIM-MJP), yet the event sequence domain remains unaddressed. Moreover, recent generative approaches (diffusion, flow matching) achieve high prediction accuracy but entirely abandon the interpretability of the intensity function — making it impossible to observe excitation/inhibition structures between events.

**Key Challenge**: Three objectives must be satisfied simultaneously — (1) zero-shot generalization across datasets, (2) high-accuracy multi-step prediction, and (3) preservation of interpretable conditional intensity functions — yet existing methods satisfy at most two.

**Key Insight**: The paper draws on the Foundation Inference Model (FIM) paradigm — pretraining a recognition network on large-scale synthetic data so that it learns to infer the underlying dynamical parameters from a set of in-context event sequences. The key observation is that, provided the family of conditional intensity functions in the synthetic data is sufficiently broad, the pretrained model encodes a strong prior that enables zero-shot inference or rapid fine-tuning on real data.

**Core Idea**: Pretrain a Transformer on large-scale synthetic MTPPs spanning five process classes, enabling it to directly infer three analytic parameters $(\alpha, \beta, \mu)$ of the conditional intensity function from a set of context sequences, thereby achieving zero-shot or rapidly fine-tuned, interpretable event sequence prediction.

## Method

### Overall Architecture

FIM-PP operates in two stages. **Pretraining**: A broad family of conditional intensity functions is defined (covering classical Hawkes, Poisson, periodic, high-initial-excitation, and non-monotonic kernel processes); large numbers of MTPPs are sampled from this family and simulated via the Ogata thinning algorithm, producing triples of (context sequence set, event history, ground-truth intensity values) as training data. **Inference**: Given a set of context sequences $\mathcal{C}=\{\mathcal{S}^j\}$ from the same system and an event history $\mathcal{H}_t$, FIM-PP outputs the analytic parameters of the conditional intensity function $\hat{\lambda}(t,\kappa|\mathcal{H}_t)$, which can be used directly for likelihood estimation or for autoregressive future event prediction via the thinning algorithm.

### Key Designs

1. **Synthetic Data Generation Framework**:

    - Function: Constructs training data — 72K point processes, 14.4M events.
    - Mechanism: Defines five process families based on the generalized conditional intensity function $\lambda(t,\kappa|\mathcal{H}_t)=\max(0, \mu_\kappa(t)+\sum_{(t',\kappa')\in\mathcal{H}_t} z_{\kappa\kappa'}\gamma_{\kappa\kappa'}(t-t'))$: (a) classical Hawkes — constant base intensity and exponential decay kernel; (b) Poisson — constant base intensity, no interaction kernel; (c) periodic — sinusoidal base intensity; (d) high initial excitation — Gamma-distributed base intensity; (e) non-monotonic shifted kernel — Rayleigh-distributed kernel. For each mark pair $(\kappa,\kappa')$, an interaction type $z_{\kappa\kappa'}\in\{-1,0,1\}$ is sampled randomly, representing inhibition, no influence, and excitation, respectively.
    - Design Motivation: Covering diverse kernel functions and interaction patterns enables the model to encode a sufficiently broad prior. Experiments confirm that this prior generalizes even to power-law kernels never seen during training.

2. **Hierarchical Context Encoder**:

    - Function: Compresses a set of variable-length context sequences into a fixed-dimensional representation.
    - Mechanism: Each event $(t_i, \kappa_i, \Delta t_i)$ is first mapped through three embedding networks ($\phi_t, \phi_\kappa, \phi_{\Delta t}$, with sinusoidal activations for time embeddings) and summed to obtain an event embedding $\mathbf{u}_i$. Within each sequence, event embeddings are processed by a Transformer encoder $\Psi_\text{enc}^\text{cont}$, then compressed to a single vector $\mathbf{c}_j$ via attention with a learnable fixed query $\mathbf{q}^\text{cont}$. The vectors $\mathbf{c}_j$ from all sequences are aggregated by a second Transformer encoder $\Psi_\text{enc}^\text{comb}$ to produce the final context representation $\tilde{\mathbf{C}}$.
    - Design Motivation: The hierarchical scheme (intra-sequence first, then inter-sequence) avoids the $O(N^2)$ complexity bottleneck of concatenating all events into a single long sequence, significantly improving scalability.

3. **Context-Aware History Encoder and Intensity Parameterization**:

    - Function: Estimates the conditional intensity function from the current event history and context.
    - Mechanism: Embeddings of the event history $\mathcal{H}_t$ serve as queries for a Transformer decoder $\Psi_\text{dec}^\text{hist}$, with the context representation $\tilde{\mathbf{C}}$ as keys and values, yielding a history encoding $\mathbf{h}_t^\text{hist}$. Concatenated with mark embeddings, this is passed through three independent feed-forward networks (with softplus activations to ensure non-negativity) to output the three parameters $(\hat{\alpha}, \hat{\beta}, \hat{\mu})$. The conditional intensity is defined as $\hat{\lambda}(t,\kappa'|\mathcal{H}_t)=\hat{\mu}+(\hat{\alpha}-\hat{\mu})\exp(-\hat{\beta}(t-t_\text{last}))$ — upon a new event the intensity jumps to $\hat{\alpha}$, then relaxes exponentially at rate $\hat{\beta}$ toward the baseline $\hat{\mu}$.
    - Design Motivation: The three-parameter analytic form resembles Hawkes but with parameters that are history- and mark-dependent (output by a neural network), allowing the model to capture locally rich behaviors such as Rayleigh and power-law kernels. Interpretability is preserved: one can directly inspect the intensity curve to determine excitation or inhibition.

### Loss & Training

The training objective is the standard next-event negative log-likelihood: $\mathcal{L}_\text{NLL}=\sum_\kappa \int_0^T \hat{\lambda}(s,\kappa|\mathcal{H}_s)ds - \sum_{(t,\kappa)\in\mathcal{T}}\hat{\lambda}(t,\kappa|\mathcal{H}_t)$. During training, the number of context sequences, truncation lengths, and mark counts are randomly subsampled, enabling the model to adapt to varying scales of real-world data at inference time. The model has only 16M parameters and supports up to 22 mark types. Fine-tuning on a target dataset applies the same NLL objective on the training split — one sequence serves as the target while the rest serve as context — and completes in a few minutes with no more than 11GB of GPU memory.

## Key Experimental Results

### Main Results: Multi-Event Prediction (N=20)

Comparison against 7 baselines on four real-world datasets (Taxi, StackOverflow, Amazon, Retweet), reporting OTD (Optimal Transport Distance, lower is better) and sMAPE (Symmetric Mean Absolute Percentage Error, lower is better):

| Method | Taxi OTD | SO OTD | Amazon OTD | Retweet OTD | Taxi sMAPE | SO sMAPE | Amazon sMAPE | Retweet sMAPE |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| HYPRO | 21.60 | 42.40 | 38.6 | 61.03 | 93.8 | 111.00 | 82.5 | 106.11 |
| A-NHP | 24.76 | 42.59 | 39.5 | 60.63 | 97.4 | 108.54 | 84.3 | 107.23 |
| CDiff | 21.01 | 41.25 | 37.7 | 60.66 | 88.0 | 106.18 | 82.0 | 106.18 |
| FIM-PP (zs) | 23.15 | 49.26 | 46.2 | 60.24 | **76.8** | 96.36 | 128.6 | 99.07 |
| **FIM-PP (f)** | **17.91** | **39.80** | **37.2** | **59.44** | 76.8 | **88.25** | **81.2** | **87.59** |

Fine-tuned FIM-PP (f) achieves the best OTD on all four datasets and the best sMAPE on 3 out of 4. Zero-shot FIM-PP (zs) already outperforms all baselines on Retweet.

### Single-Event Prediction (N=1)

| Method | Taxi RMSE$_{\Delta t}$ | Taxi Acc | Taxi sMAPE | Taobao RMSE$_{\Delta t}$ | Taobao Acc | Taobao sMAPE |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| A-NHP | 0.32 | **0.91** | 85.13 | 0.53 | 0.47 | 129.13 |
| CDiff | 0.34 | **0.91** | 87.12 | **0.52** | **0.48** | 127.12 |
| FIM-PP (zs) | **0.15** | 0.41 | 69.37 | 1.41 | 0.09 | 163.34 |
| FIM-PP (f) | **0.15** | 0.69 | **63.02** | 9.31 | 0.39 | 138.46 |

FIM-PP substantially outperforms baselines on temporal prediction metrics (RMSE, sMAPE) but lags significantly on mark accuracy — attributable to fixed mark alternation patterns in Taxi and single-mark dominance in Taobao, both out-of-distribution patterns not covered by the synthetic prior.

### Ablation Study

| Configuration | Description |
|:---|:---|
| Pretraining vs. training from scratch | Same architecture; pretrained initialization converges faster and achieves higher final performance (Appendix Figure 5) |
| Number of context sequences | Saturation performance is reached with far fewer than 2000 context sequences (Figure 6 ablation) |
| Generalization to unseen kernels | Zero-shot intensity curve inference is accurate even for power-law kernels never seen during training (Figure 4) |
| Prediction window N=5/10/20 | FIM-PP (f) consistently outperforms the average baseline performance across all window lengths |

### Key Findings

- **Zero-shot competitiveness**: FIM-PP (zs), without any target-domain training, already surpasses all specialized models requiring hours of training on the Retweet dataset, demonstrating that the synthetic prior encodes strong inductive biases.
- **Extremely efficient fine-tuning**: Fine-tuning on all datasets takes only minutes and 11GB of GPU memory, with performance comprehensively exceeding baselines — more than an order of magnitude faster than training baselines from scratch.
- **Mark prediction is the bottleneck**: Zero-shot mark accuracy is only 0.41 (Taxi) and 0.09 (Taobao); fine-tuning yields significant improvement but still falls short of specialized models, primarily because the synthetic prior does not cover dataset-specific patterns such as fixed alternation and single-mark dominance.
- **Surprisingly strong prior generalization**: Even for power-law kernels unseen during training, zero-shot intensity estimation remains accurate, indicating that the local adaptability of the three-parameter exponential relaxation form exceeds expectations.

## Highlights & Insights

- **First foundation inference model for temporal point processes**: Fills the gap in foundation models for event sequence data. Analogous to large language models in NLP, FIM-PP demonstrates the feasibility of the "synthetic pretraining + in-context inference" paradigm in non-linguistic domains.
- **Analytic intensity parameterization balances flexibility and interpretability**: The three parameters $(\alpha,\beta,\mu)$ appear simple, but since the parameters themselves are history- and mark-dependent (output by a neural network), the model can fit locally complex behaviors far beyond classical Hawkes. This design principle — "simple analytic form + neural network parameters" — is transferable to other settings requiring interpretable dynamical modeling.
- **Hierarchical sequence encoding avoids long-sequence bottlenecks**: Encoding each sequence independently before aggregation, rather than concatenating all events into one long sequence, allows the context size to scale freely and serves as a general-purpose trick for handling set-structured inputs.
- **Synthetic prior coverage determines the zero-shot performance ceiling**: The failure in mark prediction precisely exposes the blind spots of the prior (alternation patterns, single-mark dominance), pointing to the most direct direction for future improvement.

## Limitations & Future Work

- **Insufficient synthetic prior coverage**: The current five-process family does not cover structural patterns at the mark level (e.g., fixed alternation, single-mark dominance), resulting in substantially weaker zero-shot mark prediction. Extending the prior to a broader range of mark dynamics is the highest-priority direction.
- **Fixed mark count ceiling**: Training is capped at 22 mark types; datasets exceeding this limit cannot fully exploit all available context. Dynamic mark embeddings or grouping strategies could address this constraint.
- **Limitations of intensity parameterization**: Although flexible, the exponential relaxation form has limited capacity to fit multimodal kernels (e.g., repeated excitation–inhibition alternation). Multi-component mixture parameterizations could be explored.
- **Intensity-free methods not integrated**: Recent intensity-free approaches (diffusion, flow matching) offer advantages in predictive accuracy; combining them with the interpretable inference of FIM-PP is an important future direction.

## Related Work & Insights

- **vs. CDiff (diffusion-based event prediction)**: CDiff directly generates event sets with high predictive accuracy but entirely sacrifices the interpretability of the intensity function. FIM-PP surpasses CDiff on most metrics after fine-tuning while preserving the physical semantics of the intensity curve.
- **vs. NHP / A-NHP (neural Hawkes)**: These methods are also based on conditional intensity functions but must be trained from scratch for each dataset. FIM-PP eliminates this burden through synthetic pretraining and surpasses these models by 10–30% on OTD metrics after fine-tuning.
- **vs. FIM-MJP / FIM-SDE (other foundation inference models)**: FIM-PP extends the FIM paradigm from Markov jump processes and stochastic differential equations to point processes, validating the universality of the "synthetic pretraining + analytic parameterization + in-context inference" framework across a broader class of dynamical systems.

## Rating

- Novelty: ⭐⭐⭐⭐ First foundation model for point processes; conceptually novel though the technical approach follows the existing FIM framework.
- Experimental Thoroughness: ⭐⭐⭐⭐ Four datasets + synthetic validation + comprehensive ablations; however, the shortcomings in mark prediction are not deeply analyzed.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear logic, consistent mathematical notation, and thorough background exposition.
- Value: ⭐⭐⭐⭐ Fills a gap in the field with high practical value, though applicability remains constrained by prior coverage.

1. **Insufficient pretraining distribution coverage**: Certain patterns are not covered.
2. **Autoregressive error accumulation**.
3. **Mark count ceiling of 22**.
4. **Exponential decay insufficient for long-range dependencies**.

## Related Work & Insights

| Category | Representative Work | Distinction |
|:---:|:---:|:---:|
| Intensity-based TPP | NHP, A-NHP | Per-dataset training from scratch |
| Generative TPP | CDiff, IFTPP | Sacrifices interpretability |
| Joint distribution | HYPRO, Dual-TPP | Learns joint distributions |
| Foundation inference | FIM-MJP, FIM-SDE | Targets continuous-state systems |

Core insight: **The quality of synthetic prior design determines the generalization ceiling of foundation inference models.**

## Rating

| Dimension | Score |
|:---:|:---:|
| Novelty | ⭐⭐⭐⭐⭐ |
| Technical Depth | ⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| Practical Value | ⭐⭐⭐⭐ |
| Overall | ⭐⭐⭐⭐ |

> This work pioneers the application of foundation inference models to temporal point processes. The synthetic pretraining combined with in-context learning is highly inspiring; zero-shot performance is impressive, and fine-tuning achieves comprehensive state-of-the-art results. The primary limitation lies in the coverage of the synthetic prior.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] In-Context Learning for Pure Exploration](in-context_learning_for_pure_exploration.md)
- [\[ICLR 2026\] SimuHome: A Temporal- and Environment-Aware Benchmark for Smart Home Agents](simuhome_a_temporal-_and_environment-aware_benchmark_for_smart_home_agents.md)
- [\[ACL 2026\] Evaluating Temporal Consistency in Multi-Turn Language Models](../../ACL2026/llm_evaluation/evaluating_temporal_consistency_in_multi-turn_language_models.md)
- [\[ICLR 2026\] GuidedSampling: Steering LLMs Towards Diverse Candidate Solutions at Inference-Time](guidedsampling_steering_llms_towards_diverse_candidate_solutions_at_inference-ti.md)
- [\[ACL 2026\] CUB: Benchmarking Context Utilisation Techniques for Language Models](../../ACL2026/llm_evaluation/cub_benchmarking_context_utilisation_techniques_for_language_models.md)

</div>

<!-- RELATED:END -->
