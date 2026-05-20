---
title: >-
  [Paper Note] Thought Communication in Multiagent Collaboration
description: >-
  [NeurIPS 2025][LLM Evaluation][multiagent collaboration] This paper proposes ThoughtComm, a framework that formalizes multiagent communication as a latent variable generative model. It proves that both shared and private…
tags:
  - "NeurIPS 2025"
  - "LLM Evaluation"
  - "multiagent collaboration"
  - "thought communication"
  - "latent variable model"
  - "identifiability theory"
  - "prefix injection"
  - "sparse autoencoder"
date: 2026-05-08
content_hash: 42d585eb8d86ce1e
---

# Thought Communication in Multiagent Collaboration

**Conference**: NeurIPS 2025
**arXiv**: [2510.20733](https://arxiv.org/abs/2510.20733)  
**Code**: Not released  
**Area**: LLM Evaluation
**Keywords**: multiagent collaboration, thought communication, latent variable model, identifiability theory, prefix injection, sparse autoencoder

## TL;DR

This paper proposes ThoughtComm, a framework that formalizes multiagent communication as a latent variable generative model. It proves that both shared and private thoughts are identifiable under nonparametric conditions, extracts latent thoughts via a sparsity-regularized autoencoder, and feeds them back to each agent through prefix injection. ThoughtComm achieves an average improvement of 19.06% over the current SOTA Multiagent Finetuning on mathematical reasoning benchmarks.

## Background & Motivation

**Background**: Natural language has long been the central medium of human collaboration, enabling large-scale social cooperation and knowledge transfer. However, natural language is inherently **sequential, ambiguous, and imprecise**—it can only indirectly and fragmentarily reflect human inner thought. In large language model (LLM)-based multi-agent systems (MAS), existing methods rely almost entirely on natural language as the communication medium, with agents exchanging text tokens or token embeddings. Representative works include Multi-Agent Debate (Du et al., 2023) and Encouraging Divergent Thinking (Liang et al., 2023), in which multiple LLM agents coordinate reasoning and reach consensus through multi-round natural-language debates.

**Limitations of Prior Work**: Although language-based multiagent debate has demonstrated improvements on many tasks, several recent empirical studies (Cemri et al., 2025; Hu et al., 2025) have revealed characteristic failure modes in current systems: **ambiguous message specifications** and **inter-agent misalignment**. The root cause lies in the "lossy" nature of linguistic communication—natural language is only an indirect and incomplete projection of thought, and an agent's true internal reasoning state cannot be precisely conveyed. For example, when solving a mathematical problem, an agent may implicitly weigh multiple reasoning paths, but can only selectively expose part of its reasoning when expressing itself linguistically, leaving other agents with necessarily incomplete information.

**Key Challenge**: Humans rely on language as a lossy channel because they are constrained by the physical limits of speech and perception. Machines, however, are not subject to these physical constraints—they have direct access to internal representations. Yet existing multiagent systems replicate the human communication paradigm, transmitting information via natural language or its embeddings. This creates a fundamental contradiction: **machines that are not bound by the physical limitations that motivated language are nonetheless forced to use a communication modality designed for those very limitations**. This not only inherits all the shortcomings of language but also squanders the unique advantage machines possess in directly manipulating internal representations.

**Goal**: The authors decompose the problem into three levels: (1) How can the "latent thoughts" driving an agent's reasoning be identified from its internal state? (2) How can one distinguish which thoughts are shared across agents and which are private? (3) How can the identified relevant thoughts be effectively injected into each agent's generation process?

**Key Insight**: The authors draw on **nonlinear independent component analysis (Nonlinear ICA)** and **latent variable identifiability theory**, making a core observation: each agent's model state can be viewed as the output of an unknown generative function applied to a set of latent thoughts. If these latent thoughts can be "inverted" from the observed internal states, agents can directly exchange thoughts rather than language, realizing genuine mind-to-mind communication. This angle is promising because the Nonlinear ICA literature has recently achieved identifiability breakthroughs under structural sparsity conditions (Zheng et al., 2022; Lachapelle et al., 2022), enabling latent variable recovery without auxiliary information.

**Core Idea**: Formalize multiagent communication as a latent variable generative process, prove that latent thoughts are identifiable under sparse regularization, then extract thoughts via an autoencoder, route them structurally to each agent, and realize "telepathic" collaboration through prefix injection.

## Method

### Overall Architecture

The overall pipeline of ThoughtComm is as follows. At each communication round $t$, the framework first collects the model internal states $H_t = (H_t^{(1)}, \ldots, H_t^{(n_a)})$ of all $n_a$ agents at the start of that round (specifically, the hidden-layer representation of the last token for each agent), and concatenates them into a global state vector. This vector is then fed into a pretrained **sparsity-regularized autoencoder**, which encodes it to obtain latent thoughts $\hat{Z}_t$. Next, the non-zero pattern of the autoencoder's Jacobian is used to infer the dependency structure between each latent dimension and each agent, routing relevant thoughts to the corresponding agents and constructing a personalized latent representation $\tilde{Z}_t^{(i)}$ for each agent $A_i$ via protocol-level weighting. Finally, this representation is converted into a prefix vector $P_t^{(i)}$ by a learned adapter and injected into the agent's generation process in the next round. The entire framework modifies no LLM weight parameters; only the lightweight autoencoder and adapter are trained.

### Key Designs

1. **Formalization of the Data Generating Process and Latent Variable Model**:

    - **Function**: Models multiagent communication as a latent variable generative process, treating each agent's model state as an observation produced by latent thoughts through an unknown function.
    - **Mechanism**: At communication round $t$, the global model state of all agents is $H_t = (H_t^{(1)}, \ldots, H_t^{(n_a)})$, where $H_t^{(j)} \in \mathbb{R}^{n_{h_j}}$ denotes the internal state of agent $A_j$. It is assumed that there exists a set of latent thoughts $Z_t = (Z_{t,1}, \ldots, Z_{t,n_z}) \in \mathbb{R}^{n_z}$ and an unknown generative function $f$ such that $H_t = f(Z_t)$. This $f$ is assumed to be **invertible** (ensuring lossless information) and **twice differentiable** (ensuring well-defined gradients), both of which are standard in the Nonlinear ICA literature. The structural dependency between thoughts and agents is characterized by the **non-zero pattern** of the Jacobian matrix $J_f(Z_t)$: a binary matrix $B(J_f) \in \{0,1\}^{n_h \times n_z}$ is defined, where $B(J_f)_{i,j} = 1$ if and only if there exists some $z_t$ such that $J_f(z_t)_{i,j} \neq 0$. The set of latent thoughts associated with agent $A_k$ is then defined as $Z_{H_t^{(k)}} = \{Z_{t,j} \in Z_t \mid \exists i \in [k_l, k_h]\ \text{s.t.}\ B(J_f)_{i,j} \neq 0\}$.
    - **Design Motivation**: This formalization provides a solid foundation for the subsequent theoretical analysis. By recasting the problem within the classical Nonlinear ICA framework, identifiability theory can be leveraged to guarantee that the recovered latent thoughts genuinely correspond to true internal reasoning components. Unlike prior methods that operate only at the token or embedding level, this approach models the problem at the level of the generative process itself, fundamentally bypassing the language bottleneck.

2. **Three-Level Identifiability Guarantees**:

    - **Function**: Proves that, under general nonparametric conditions, shared thoughts, private thoughts, and the thought structure are all recoverable from the observed model states.
    - **Mechanism**: The authors establish three progressively deeper identifiability theorems. **Theorem 1 (Shared Thought Identifiability)**: For any pair of agents $A_i$ and $A_j$, under $\ell_0$ regularization applied to the Jacobian $J_{\hat{f}}$ of the estimated model, there exists a permutation $\pi$ such that the shared thoughts $Z_i \in Z_{H_t^{(i)}} \cap Z_{H_t^{(j)}}$ satisfy $\partial Z_i / \partial \hat{Z}_{\pi(j)} = 0$ with respect to non-shared thoughts. Intuitively, this means the recovered shared thoughts do not become entangled with other latent variables in the system. **Theorem 2 (Private Thought Identifiability)**: Similarly, the private thought components $Z_i \in Z_{H_t^{(i)}} \setminus Z_{H_t^{(j)}}$ of any agent can be disentangled from all other latent variables. This is crucial for preserving cognitive diversity—just as different members of a human team bring unique perspectives, private thoughts may contain critical rare insights. **Theorem 3 (Thought Structure Identifiability)**: The non-zero pattern $B(J_f)$ connecting latent thoughts to agent states can be recovered up to at most one permutation matrix $P$, i.e., $B(J_{\hat{f}}) = B(J_f) P$. This means not only can the content of each thought be recovered, but one can also determine which agents hold which thoughts. The assumptions underlying all three theorems only require that the generative function $f$ exhibits sufficient variation across different sample points, such that the Jacobian can span its support subspace—a standard mild assumption in the Nonlinear ICA literature.
    - **Design Motivation**: Without identifiability guarantees, latent representations recovered from neural networks may be arbitrary mixtures of thoughts, rendering them unreliable as communication content. These three theorems provide theoretical legitimacy for the framework. Notably, while prior identifiability work typically requires auxiliary information (e.g., time labels, intervention signals) or specific functional class assumptions, identifiability here is achieved solely through sparsity and the structural duality of the multiagent setting—itself a new contribution to classical identifiability theory. The authors also adopt a strategically focused approach: rather than pursuing global recovery of all latent variables (which is infeasible under the current assumptions), they concentrate on pairwise shared/private decomposition—precisely the granularity needed for communication.

3. **Sparsity-Regularized Autoencoder and Thought Extraction**:

    - **Function**: Translates the theoretical latent variable recovery problem into a practically trainable neural network architecture that extracts latent thoughts from agent model states.
    - **Mechanism**: The framework employs an autoencoder to approximate the inverse of the generative function $f$. After concatenating all agents' model states into $H_t$, the encoder $\hat{f}^{-1}$ maps it to the latent space to obtain $\hat{Z}_t = \hat{f}^{-1}(H_t) \in \mathbb{R}^{n_z}$. The training objective comprises two terms: a reconstruction loss to preserve information, and a sparse regularization term to enforce the theoretically required $\ell_0$ (in practice relaxed to $\ell_1$) constraint. The specific loss function is $\mathcal{L}_{\text{rec}} = \|H_t - \hat{f}(\hat{Z}_t)\|_2^2 + \|J_{\hat{f}}\|_1$. The $\ell_1$ regularization is applied to the Jacobian of the decoder rather than the weight matrix—a design choice directly guided by the theory. After training, the encoder is used to extract thoughts, and the non-zero pattern $B(J_{\hat{f}})$ of the Jacobian is used to infer the thought–agent structure.
    - **Design Motivation**: The autoencoder is the natural realization of the theoretical framework: the encoder corresponds to the recovery mapping $\hat{f}^{-1}$, and the decoder corresponds to the estimated generative function $\hat{f}$. The $\ell_1$ regularization on the Jacobian follows directly from the $\ell_0$ condition in the theorems, with $\ell_1$ being the standard convex relaxation. Crucially, this autoencoder is **task-agnostic**—it only needs to reconstruct model states without requiring any downstream task labels, and can therefore be pretrained once and reused across tasks.

4. **Protocol-Level Thought Routing and Weighting**:

    - **Function**: Constructs personalized thought representations for each agent based on the recovered thought structure, distinguishing shared from private components.
    - **Mechanism**: For agent $A_i$, the relevant thought set $\hat{Z}_{H_t^{(i)}}$ is first identified via the structural mask. Thoughts are then grouped by **agreement level**: the agreement degree of each thought dimension $\hat{Z}_{t,j}$ is defined as $\alpha_j = \sum_{k=1}^{n_a} \mathbb{I}(\hat{Z}_{t,j} \in \hat{Z}_{H_t^{(k)}})$, i.e., the number of agents that depend on that dimension. Thoughts with the same agreement degree are grouped together, and each group is assigned a learnable weight $w_{\alpha_j}$. The personalized representation for agent $A_i$ is then constructed by concatenating all weighted groups: $\tilde{Z}_t^{(i)} = \text{concat}_\alpha(w_{\alpha_j} \cdot \hat{Z}_{t,\alpha}^{(i)})$.
    - **Design Motivation**: Different types of thoughts should have different communication priorities. Thoughts shared by all agents (high agreement) may represent consensus and common knowledge, while thoughts held by only a few agents (low agreement) may represent unique insights or rare but critical constraints. The weighting mechanism allows the system to flexibly modulate the influence of different thought types—for example, amplifying shared thoughts when rapid consensus is needed, or upweighting private thoughts in exploratory tasks.

5. **Prefix Adapter and Thought Injection**:

    - **Function**: Seamlessly injects the constructed personalized latent representations into each agent's LLM generation process.
    - **Mechanism**: A learned adapter function $g$ transforms $\tilde{Z}_t^{(i)}$ into a prefix matrix $P_t^{(i)} = g(\tilde{Z}_t^{(i)}) \in \mathbb{R}^{m \times d}$, where $m$ is the prefix length and $d$ is the embedding dimension. This prefix is prepended to the token embeddings at the start of the agent's next-round generation, in a manner analogous to Prefix-Tuning (Li & Liang, 2021). The adapter is trained with two loss terms: a semantic similarity loss ensures that the content generated after prefix injection is semantically consistent with the reference output, and a language fluency regularization term ensures that the generated output remains natural. The full loss is $\mathcal{L}_{\text{comm}} = \sum_{i,t} [(1 - \cos(\bar{\phi}(y_{t,i}^{\text{gen}}), \bar{\phi}(y_{t,i}^{\text{ref}}))) - \log p(y_{t,i}^{\text{gen}} \mid \text{context}_{t,i}, P_t^{(i)})]$, where $\bar{\phi}(\cdot)$ denotes the mean token embedding.
    - **Design Motivation**: Prefix injection offers several advantages: (a) LLM weights are not modified, preserving the model's original capabilities; (b) prefix embeddings can encode continuous, rich latent information without being constrained by the discrete token vocabulary—experiments show that even a single prefix token achieves near-optimal performance, indicating that prefix embeddings have far higher information density than ordinary token embeddings; (c) the adapter is likewise task-agnostic and can be pretrained and reused. Furthermore, the adapter only requires generated content to be "linguistically natural" rather than reproducing the specific content of the reference, making the training objective permissive and easy to optimize.

### Loss & Training

The overall training proceeds in two stages, both independent of downstream tasks:

**Stage 1: Autoencoder Training**. The reconstruction loss with Jacobian sparsity regularization is minimized: $\mathcal{L}_{\text{rec}} = \|H_t - \hat{f}(\hat{Z}_t)\|_2^2 + \|J_{\hat{f}}\|_1$. Training data are collected from model states produced when agents generate responses to arbitrary questions; no task labels are required. In practice, randomly sampling 500 examples suffices.

**Stage 2: Prefix Adapter Training**. After prefix injection, the model generates a brief continuation (e.g., one sentence), and semantic similarity and linguistic fluency are jointly optimized. The training objective $\mathcal{L}_{\text{comm}}$ focuses solely on linguistic naturalness rather than content correctness, imposing no requirements on annotated data.

The training overhead of both modules depends only on the LLM's embedding dimension (e.g., 1024 or 4096), not on the number of parameters. This means that the additional overhead of ThoughtComm is identical for Llama-3-70B and Llama-3-405B (both sharing a 16384-dimensional embedding), while the training cost of Multiagent Finetuning scales sharply with model size at 405B—a key practical advantage of ThoughtComm.

## Key Experimental Results

### Main Results

Evaluated on the MATH and GSM8K mathematical reasoning benchmarks using 3 agents over 2 debate rounds, with 500 randomly sampled training examples and 500 test examples (focusing on higher-difficulty problems, e.g., MATH level-3):

| Base Model | Method | MATH Acc(%) | MATH Consensus(%) | GSM8K Acc(%) | GSM8K Consensus(%) |
|---|---|---|---|---|---|
| Qwen 3-0.6B | Single Answer | 45.80±2.23 | N/A | 58.20±2.21 | N/A |
| Qwen 3-0.6B | Multiagent FT | 71.20±2.03 | 90.07 | 70.80±2.03 | 86.40 |
| **Qwen 3-0.6B** | **ThoughtComm** | **85.00±1.60** | **91.20** | **75.80±1.92** | **89.27** |
| Qwen 3-1.7B | Single Answer | 43.60±2.22 | N/A | 67.40±2.10 | N/A |
| Qwen 3-1.7B | Multiagent FT | 75.80±1.92 | 95.80 | 84.20±1.63 | 96.73 |
| **Qwen 3-1.7B** | **ThoughtComm** | **93.00±1.14** | **95.93** | **85.00±1.60** | **97.87** |
| Phi-4-mini (3.84B) | Single Answer | 63.80±2.15 | N/A | 81.60±1.73 | N/A |
| Phi-4-mini (3.84B) | Multiagent FT | 60.20±2.19 | 78.89 | 82.16±1.71 | 91.24 |
| **Phi-4-mini (3.84B)** | **ThoughtComm** | **74.60±1.95** | **84.73** | **84.20±1.63** | **94.73** |
| LLaMA 3-8B | Single Answer | 36.20±2.15 | N/A | 60.80±2.18 | N/A |
| LLaMA 3-8B | Multiagent FT | 39.68±2.19 | 68.97 | 69.20±2.06 | 80.20 |
| **LLaMA 3-8B** | **ThoughtComm** | **45.60±2.23** | **74.67** | **68.40±2.08** | **84.87** |
| DeepSeek-R1-8B | Single Answer | 42.60±2.21 | N/A | 65.60±2.12 | N/A |
| DeepSeek-R1-8B | Multiagent FT | 72.40±2.00 | 82.87 | 76.80±1.89 | 83.13 |
| **DeepSeek-R1-8B** | **ThoughtComm** | **82.80±1.69** | **80.72** | **80.80±1.76** | **88.13** |

Key statistics: ThoughtComm achieves an average relative improvement of **67.23%** over Single Answer and **19.06%** over the current SOTA Multiagent Finetuning.

### Ablation Study

**Synthetic Data Validation of Theory**:

| Setting | Key Metric | Description |
|---|---|---|
| With sparse regularization (Ours) | $R^2$ for shared/private regions significantly above baseline | Successfully identifies shared region $Z_A \cap Z_B$ and private regions $Z_A \setminus Z_B$, $Z_B \setminus Z_A$ |
| Without sparse regularization (Baseline) | $R^2$ for all regions similar and low | Three regions cannot be disentangled |
| Large-scale dimension test (124→1024 dims) | MCC consistently exceeds identifiability threshold | All 8 dimension configurations pass, validating global identifiability |

**Debate Round Scaling Experiment** (LLaMA-3-8B, 2→6 rounds):

| Configuration | Key Metric | Description |
|---|---|---|
| Multiagent FT with more rounds | Accuracy decreases, consensus slightly increases | Additional debate rounds introduce redundant/noisy information, causing performance degradation |
| ThoughtComm with more rounds | Both accuracy and consensus improve | Robust to redundant information |
| Multiagent FT (Qwen-3-1.7B) | Consensus rises but accuracy stagnates/declines | "False consensus"—agents converge on incorrect answers |
| ThoughtComm (Qwen-3-1.7B) | High consensus accompanied by high accuracy | Consensus is positively correlated with correctness |

**Prefix Length Robustness** ($m \in \{1,4,8,16\}$, four models):

| Configuration | Key Metric | Description |
|---|---|---|
| Prefix length 1→16 | Performance variation <5% across all four models | Near-optimal performance achieved with as few as 1 prefix token |

**Effect of Latent Dimensionality** (LLaMA-3-8B, Qwen-3-1.7B):

| Configuration | Key Metric | Description |
|---|---|---|
| Latent dimension up to 512 | Accuracy continues to improve | Higher-dimensional latent space provides richer communication capacity |
| Latent dimension >1024 | Gains saturate | Redundant representations yield diminishing returns |

**Agent Count Scaling**:

| Configuration | Key Metric | Description |
|---|---|---|
| 2→3 agents | Significant improvement | More perspectives lead to better reasoning |
| 3→more agents | Baseline degrades, ThoughtComm remains stable | ThoughtComm is robust to redundant signals |

### Key Findings

- **Extremely low training overhead, decoupled from model scale**: ThoughtComm trains only the autoencoder and adapter, and the overhead depends solely on the embedding dimension (e.g., 1024) rather than the number of parameters (e.g., 8B). Scaling from a 70B to a 405B model incurs no additional overhead for ThoughtComm, while Multiagent Finetuning requires proportionally more compute.
- **Prefix embeddings carry far more information than token embeddings**: Although prefix embeddings and token embeddings share the same dimensionality, token embeddings are bound to individual vocabulary items and typically encode the semantics of a single discrete token (residing on a low-dimensional subspace), whereas prefix embeddings, as free parameters, can encode multiple continuous latent thoughts and fully exploit the capacity of the embedding space.
- **"False consensus" phenomenon**: Multiagent Finetuning exhibits rising consensus but declining or stagnating accuracy as the number of debate rounds increases, indicating that agents may converge on incorrect answers. ThoughtComm's structured latent communication avoids this ineffective conformity.
- **Consistent effectiveness across model scales**: Significant improvements are observed across five models ranging from the 0.6B Qwen 3-0.6B to the 8B DeepSeek-R1-distilled-Llama-8B, validating the generality of the framework.

## Highlights & Insights

- **A pioneering paradigm shift in communication**: This is the first work to elevate multiagent communication from "exchanging linguistic symbols" to "exchanging latent thoughts." This is not a mere engineering optimization but a paradigm-level innovation—it fundamentally asks "what should communication between machines look like?" rather than "how can machines speak human language more effectively?" This question is critical for the collaboration of superintelligent systems.
- **An elegant closed loop from theory to practice**: Proving identifiability theoretically → using theoretical conditions to directly guide network design (sparse regularization) → validating theoretical predictions experimentally (synthetic experiments) → demonstrating that the theoretical framework yields practical performance gains (real benchmarks). Such a complete chain from theory to practice is exceedingly rare in the LLM multiagent literature.
- **The finding that "one prefix token speaks a thousand words" is highly inspiring**: Experiments show that a single prefix token suffices to encode sufficiently rich thought information to substantially improve collaborative performance. This hints at a possibility—the embedding space of LLMs has greater information capacity than commonly assumed, and continuous representation spaces offer enormous information compression advantages over discrete token spaces. This finding can transfer to knowledge distillation, model compression, and related settings.
- **Diagnostic value for "false consensus"**: The "high consensus but low accuracy" phenomenon identified in this paper's experiments reveals a deep flaw in existing multiagent debate systems—linguistic persuasion does not equate to cognitive alignment. By communicating through latent space, ThoughtComm ensures that consensus is grounded in genuine reasoning consistency.

## Limitations & Future Work

- **Requires white-box access to model internal states**: The framework depends on extracting the hidden-layer representation of the last token from each agent, which is infeasible for closed-source API models. The authors propose in the appendix an alternative that substitutes text embeddings (e.g., Sentence-BERT) for model states, but the extent of performance degradation has not yet been experimentally validated.
- **Practical applicability of theoretical assumptions**: The invertibility and twice-differentiability of the generative function $f$ are standard theoretical assumptions, but direct evidence that real LLM internal representations satisfy these conditions is lacking. In particular, whether a truly invertible mapping exists between model states and "thoughts" remains a premise that requires deeper validation.
- **Evaluation scope limited to mathematical reasoning**: Both MATH and GSM8K consist of problems with definitive answers, where the notion of "thought" is relatively well-defined. Whether ThoughtComm's advantages hold in more complex settings—such as open-domain dialogue, creative writing, and multi-round negotiation—where latent thought structure may be considerably more ambiguous, remains to be examined.
- **Only the last token's hidden state is used as the model state**: This means each agent's entire reasoning process is compressed into a single vector, potentially discarding substantial sequence-level reasoning information. Attention pooling or multi-position sampling could be considered to construct richer state representations.
- **Adversarial scenarios are not considered**: In open environments, malicious agents could deliberately generate misleading model states to contaminate the latent thought space, yet the current framework has no defense mechanism. Trustworthiness verification should be introduced at the thought extraction and injection stages.
- **The design of protocol weights is heuristic**: The strategy of grouping thoughts by agreement level and assigning different weights lacks theoretical grounding. Whether an optimal weighting strategy exists, and whether these weights can be adaptively learned from data, remain open questions.

## Related Work & Insights

- **vs Multiagent Debate (Du et al., 2023)**: Multiagent debate allows multiple LLMs to engage in multi-round natural-language discussions. ThoughtComm differs fundamentally—debate still operates in language space and is bounded by language's expressive capacity, whereas ThoughtComm operates in latent space, bypassing the language bottleneck entirely. Notably, ThoughtComm can be layered on top of debate frameworks: latent thought communication can precede surface-level linguistic debate.
- **vs Multiagent Finetuning (Subramaniam et al., 2025)**: The current SOTA fine-tunes entire LLMs over multiple rounds to endow agents with specialized roles. ThoughtComm achieves superior performance without modifying any LLM weights, reducing training overhead from parameter-dependent ($O(n_\text{params})$) to embedding-dimension-dependent ($O(d_\text{embed})$)—an order-of-magnitude improvement.
- **vs Token-level Collaboration (Bian et al., 2025; Chakraborty et al., 2025)**: Methods such as PToCoC refine the granularity of collaboration from the round level to the token level, but the information carrier remains token embeddings. ThoughtComm descends further still, operating at the level of latent thoughts beneath token embeddings.
- **vs Embedding-based Communication (Pham et al., 2023)**: "Let Models Speak Ciphers" has agents exchange token embeddings rather than text. The key distinction of ThoughtComm is that it extracts **structurally decomposed** latent thoughts—distinguishing shared from private components—with theoretical identifiability guarantees.
- **Connection to Nonlinear ICA**: ThoughtComm's theoretical foundations draw on Nonlinear ICA, particularly identifiability under structural sparsity conditions (Zheng et al., 2022; Lachapelle et al., 2022). However, the authors' contribution goes beyond direct application—rather than pursuing global recovery, they focus on pairwise decomposition, which is itself novel within the ICA literature.
- **Inspirations**: (1) The idea of latent thought communication transfers directly to knowledge distillation—instead of having student models learn the output distribution of teacher models, they could learn the teacher's latent reasoning structure. (2) Model merging could similarly benefit from analogous latent-space alignment approaches—rather than naively averaging weights, shared and complementary components could be identified at the level of thoughts.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Introduces an entirely new paradigm of "thought communication" and is the first to bring identifiability theory to multiagent LLM communication; the problem formulation itself constitutes a major contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers five model variants, two benchmarks, both synthetic and real experiments, and ablation analyses across multiple dimensions; however, task diversity is limited to mathematical reasoning.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The narrative arc from theory to practice is smooth and natural; figures are intuitively designed (e.g., the luggage/speed/punctuality analogy in Fig. 1); mathematical notation is clearly presented.
- **Value**: ⭐⭐⭐⭐⭐ Offers both theoretical depth and practical utility; the lightweight design facilitates community adoption and opens an entirely new research direction beyond linguistic communication.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] LCDB 1.1: A Database Illustrating Learning Curves Are More Ill-Behaved Than Previously Thought](lcdb_11_a_database_illustrating_learning_curves_are_more_ill-behaved_than_previo.md)
- [\[NeurIPS 2025\] EvaLearn: Quantifying the Learning Capability and Efficiency of LLMs via Sequential Problem Solving](evalearn_quantifying_the_learning_capability_and_efficiency_of_llms_via_sequenti.md)
- [\[NeurIPS 2025\] PARROT: A Benchmark for Evaluating LLMs in Cross-System SQL Translation](parrot_a_benchmark_for_evaluating_llms_in_cross-system_sql_translation.md)
- [\[NeurIPS 2025\] LTD-Bench: Evaluating Large Language Models by Letting Them Draw](ltd-bench_evaluating_large_language_models_by_letting_them_draw.md)
- [\[NeurIPS 2025\] RDB2G-Bench: A Comprehensive Benchmark for Automatic Graph Modeling of Relational Databases](rdb2g-bench_a_comprehensive_benchmark_for_automatic_graph_modeling_of_relational.md)

</div>

<!-- RELATED:END -->
