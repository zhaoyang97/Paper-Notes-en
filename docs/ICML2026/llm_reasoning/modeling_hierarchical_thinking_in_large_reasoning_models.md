---
title: >-
  [Paper Note] Modeling Hierarchical Thinking in Large Reasoning Models
description: >-
  [ICML2026][LLM Reasoning][Finite State Machines] The authors abstract the long Chain-of-Thought (CoT) of Large Reasoning Models (LRMs) into a 6-state Finite State Machine (FSM). By constructing a Transition Advantage Mat…
tags:
  - "ICML2026"
  - "LLM Reasoning"
  - "Finite State Machines"
  - "Chain-of-Thought"
  - "Activation Steering"
  - "Q-Value Planning"
  - "Interpretability"
date: 2026-05-08
content_hash: 8f19ed0fcdea8f7a
---

# Modeling Hierarchical Thinking in Large Reasoning Models

**Conference**: ICML2026 Oral  
**arXiv**: [2510.22437](https://arxiv.org/abs/2510.22437)  
**Code**: https://github.com/shahariar-shibli/CoT-FSM (Available)  
**Area**: LLM Reasoning  
**Keywords**: Finite State Machines, Chain-of-Thought, Activation Steering, Q-Value Planning, Interpretability

## TL;DR
The authors abstract the long Chain-of-Thought (CoT) of Large Reasoning Models (LRMs) into a 6-state Finite State Machine (FSM). By constructing a Transition Advantage Matrix based on the difference in state transition probabilities between success and failure cases and iteratively calculating long-horizon planning strategies via Q-Values, they implement sparse orthogonal activation steering only at sentence boundaries. This approach improves accuracy on challenging problems like AIME25 by up to +13% with approximately 25× fewer intervention counts.

## Background & Motivation

**Background**: LRMs complete complex reasoning tasks by generating long CoTs, often exceeding thousands of tokens, exhibiting a hierarchical structure similar to human "think-before-you-speak" patterns on problems like AIME and GPQA. Regarding CoT interpretability, recent work has begun using activation steering for behavior-level control, such as SEAL suppressing redundant reflections or Venhoff et al. identifying linear directions corresponding to specific behaviors.

**Limitations of Prior Work**: Existing control methods remain at the "local behavior" level—either suppressing a single type of segment (e.g., reflection/transition) or merely verifying that a certain behavior is adjustable in the activation space. They fail to answer a critical control question: **When the model is currently in a certain stage of a reasoning trajectory, which cognitive state should it move to next to maximize the probability of a correct final answer?**

**Key Challenge**: A gap exists between interpretability (identifying steerable behaviors) and actionable control (deciding when and where to intervene). Token-by-token intervention disrupts content coherence and is extremely costly, while greedy one-step optimization often leads into "short-sighted traps," pushing the model into dead ends.

**Goal**: (1) Characterize CoT with a global hierarchical structure; (2) Quantify which cognitive transitions truly distinguish between success and failure; (3) Design a training-free, sparse-intervention strategy with a long-horizon perspective.

**Key Insight**: Human problem-solving theories (Polya's "four-step method," Schoenfeld's Episode Theory) have long divided the problem-solving process into finite high-level cognitive stages. Since LRMs are trained on human-generated CoTs, their emergent trajectories should be approximable by a set of discrete states.

**Core Idea**: CoT is modeled as a 6-state FSM. The difference between "correct vs. incorrect" transition matrices, $R$, is treated as a reward. Long-horizon utility is calculated via Q-Value iteration, and activation steering of "orthogonal components" is performed at sentence boundaries—transforming reasoning control from "per-token fine-tuning" into "cognitive strategy planning."

## Method

### Overall Architecture
The method consists of two stages: **Offline FSM Abstraction** and **Online Guided Inference**.

Offline stage: Full CoTs are generated for the training set, segmented by sentences, and automatically labeled by GPT-4o-mini to map each sentence to one of six high-level states $\mathcal{Q}=\{\text{init, deduce, augment, uncertain, backtrack, closure}\}$. Two conditional transition matrices, $T^{(correct)}$ and $T^{(incorrect)}$, are estimated to compute the Transition Advantage Matrix $R = T^{(correct)} - T^{(incorrect)}$. Simultaneously, activation steering vectors $\mathbf{v}^{(\ell)}_{u\to v}$ for each directed transition $(u\to v)$ are extracted using contrastive difference-of-means. A State Encoder and two lightweight classifiers ($g_{curr}$ for current state and $g_{next}$ for next state) are trained.

Online stage: During autoregressive generation, sentence-ending punctuation (`. ? !`) is detected as the intervention timing. At boundaries, classifiers estimate the current/next state. Based on the Q-Value strategy, the system decides whether to steer and toward which target state $q^\star$. The corresponding steering vector is then injected as an orthogonal component in the hidden space, and generation continues normally.

### Key Designs

1.  **6-State FSM Abstraction + Transition Advantage Matrix $R$**:
    - **Function**: Compresses unstructured CoT sequences into an analyzable, comparable discrete transition graph and quantifies "which cognitive transitions favor a correct answer" into a $|\mathcal{Q}|\times|\mathcal{Q}|$ reward matrix.
    - **Mechanism**: CoT sentence sequences $\mathcal{S}=(s_1,\dots,s_K)$ from LRMs are projected to 6-state trajectories via a labeling function $\phi:\mathcal{S}\to\mathcal{Q}$, merging self-loops to preserve true transitions. Transition probabilities $T^{(correct)}_{ij}$ and $T^{(incorrect)}_{ij}$ are estimated from "correct" and "incorrect" sample groups, respectively, resulting in $R_{ij}=T^{(correct)}_{ij}-T^{(incorrect)}_{ij}$. $R_{ij}>0$ indicates that the jump from $i$ to $j$ is more common in correct trajectories, representing a positive transition to be encouraged. The 6 states correspond to Polya's framework (Understand-Plan-Execute-Review) supplemented with LRM-specific uncertainty and backtracking, aligning with human cognition while empirically distinguishing outcomes (validated by a Cohen's Kappa of 0.89 in human consistency checks).
    - **Design Motivation**: Previous CoT control was either binary across behavior categories or limited to single linear directions. FSM provides a compact, global structure that allows control to be anchored to a transition graph that remains stable even during extrapolation, rather than relying on temporary statistics from a specific prompt.

2.  **Q-Value Iteration Planning + Confidence-Gated Sparse Triggering**:
    - **Function**: Extends single-step reward $R$ into long-horizon utility $Q(q,q')$ to precisely determine "whether to steer" and "where to steer."
    - **Mechanism**: FSM is treated as a small-scale planning problem. Bellman-style Q-Value iteration is performed on the clipped reward $R_{clip} = \text{clip}(R, [-c,+c]),\ c\in[0.2,0.3]$: $Q_{k+1}(q,q'):=R(q,q')+\gamma\max_{q''}Q_k(q',q'')$ with $\gamma=0.9$ until convergence at 100 steps. During inference, the classifier provides current state $q$ and a probability vector $\mathbf{p}$ for the next state with confidence $\text{conf}=\max_j p_j$. Define $q^\star=\arg\max_{q'}Q(q,q')$ and $Q_{gap}=Q(q,q^\star)-Q(q, \hat{q}_{t+1})$. Intervention is bypassed if the model is not "stuck" (same state for 5 steps) and has $\text{conf}\ge 0.9$. Otherwise, if $Q_{gap}\ge\delta=0.06$, steering is applied with intensity $\alpha=\max(\beta,\,Q_{gap}\cdot\text{conf})$, where $\beta\in[0.1,1.2]$.
    - **Design Motivation**: Purely greedy strategies (selecting the maximum $R$ value) can lead the model down "short-term reward but long-term failure" paths (e.g., QWEN on AIME25 dropped from 83.3% to 76.67%). Q-Value considers cumulative future rewards. The triple gating (conf + stuck + $Q_{gap}$) concentrates interventions on high-leverage decision points where the model is likely to deviate—reducing interventions by 25× while increasing performance.

3.  **Orthogonal Component Activation Injection at Sentence Boundaries**:
    - **Function**: Applies a small "lateral" perturbation along the target transition direction without destroying the semantic content of current hidden representations, biasing the next sentence's state distribution toward $q^\star$.
    - **Mechanism**: At the sentence-final punctuation token, the hidden vector $\mathbf{h}^{(\ell)}_k$ of the $\ell$-th layer is normalized: $\hat{\mathbf{h}}=\mathbf{h}/(\|\mathbf{h}\|_2+\varepsilon)$. The components of the offline-extracted steering vector $\mathbf{v}^{(\ell)}_{u\to v}$ that project onto $\hat{\mathbf{h}}$ are subtracted: $\mathbf{v}_\perp=\mathbf{v}-(\mathbf{v}^\top\hat{\mathbf{h}})\hat{\mathbf{h}}$. Finally, $\tilde{\mathbf{h}}^{(\ell)}_k=\mathbf{h}^{(\ell)}_k+\alpha\mathbf{v}_\perp$ is injected. Steering vectors are extracted via contrastive difference-of-means: the positive set contains hidden vectors at the end of sentences for that transition, and the negative set contains all others.
    - **Design Motivation**: Directly adding $\alpha\mathbf{v}$ would disrupt the content information carried in $\mathbf{h}$, causing semantic drift in the next sentence. Injecting only the orthogonal component preserves the content while altering the direction. Furthermore, sentence-ending tokens are natural semantic commitment points where the model decides the next sentence, ensuring the steering granularity matches the control granularity.

### Loss & Training
The State Encoder is a 2-layer MLP (LayerNorm + ReLU + dropout 0.1) mapping to a 512-dimensional unit sphere, trained for 50 epochs using triplet loss $\mathcal{L}_{triplet}=\max(0,\|\mathbf{z}_a-\mathbf{z}_p\|^2-\|\mathbf{z}_a-\mathbf{z}_n\|^2+m)$ with $m=1.1$ and Adam $lr=10^{-4}$. Classifiers for current/next states use an 80/20 train-test split, achieving >90% accuracy. Steering vectors are selected by layer (GPT-L/M layer 19, PHI layer 22, QWEN layer 30) with intensities Greedy $\alpha=1.0$, Weighted $\alpha\in[0.1,1.0]$, and Q-Value $\delta=0.06$. The entire pipeline does not update LRM weights.

## Key Experimental Results

### Main Results

| Dataset | Model | Default Acc | Q-Value Acc | Q-Value Interventions | Greedy Interventions |
| :--- | :--- | :--- | :--- | :--- | :--- |
| AIME25 | GPT-L | 43.30 | **56.67** | 55.20 | 77.60 |
| AIME25 | QWEN | 83.33 | **86.67** | 42.40 | 287.13 |
| MATH-500 | GPT-L | 79.00 | **83.20** | **0.48** | 12.17 |
| MATH-500 | GPT-M | 86.40 | **87.00** | **0.30** | 42.69 |
| GPQA-D | GPT-M | 64.14 | **67.17** | 88.12 | 246.93 |
| GSM8K | QWEN | 78.77 | **79.30** | **6.05** | 40.39 |

A notable result is GPT-L on MATH-500: Q-Value achieves an accuracy increase from 79.0% to 83.2% with only 0.48 interventions per problem on average, which is 25× more efficient than Greedy (12.17 interventions for 81.2%).

### Ablation Study

| Configuration | AIME25 GPT-L Acc | MATH-500 GPT-L Acc | Description |
| :--- | :--- | :--- | :--- |
| Default | 43.30 | 79.00 | No steering |
| Greedy | 50.00 | 81.20 | Short-sighted; drops performance on QWEN/AIME25 |
| Weighted | 56.67 | 82.40 | Soft mixture of positive/negative transitions |
| Q-Value | **56.67** | **83.20** | Long-horizon planning + Confidence gating |
| Cross-Model (QWEN→GPT-L, MATH-500) | — | 82.80 (Q-Val) | Only 0.4 drop vs. model-specific, but interventions doubled |

### Key Findings
- **Intervention Sparsity ≈ Reasoning Efficiency**: Q-Value achieves equal or better accuracy with up to 25× fewer interventions than Greedy (MATH-500 GPT-L: 0.48 vs 12.17), indicating that FSM + long-horizon planning effectively identifies "high-leverage decision points" rather than adding uniform noise.
- **Short-Sighted Greed is Harmful**: For QWEN on AIME25, Greedy steering dropped accuracy from 83.3% to 76.67%. This demonstrates that "locally optimal next steps $\neq$ globally optimal paths," validating the necessity of long-horizon planning.
- **Greater Gains on Harder Problems**: On AIME25, GPT-L gained +13.37 points. Tasks requiring long chains and backtracking benefit most from the global structure provided by FSM abstraction.
- **Partial Cross-Model Transferability of Transition Graphs**: Using QWEN's advantage matrix to steer GPT-L on MATH-500 still yielded 82.8% accuracy, suggesting a "universal skeleton" in LRM cognitive transitions, though fine calibration remains model-specific.

## Highlights & Insights
- **Reframing CoT Control as a 6-State Planning Problem**: Unlike previous steering work that applied uniform behavior suppression, this paper elevates "whether and where to intervene" into an RL sub-problem with a Bellman solution. This provides a "strategy layer" for interpretability research.
- **Sentence Boundaries + Orthogonal Components**: Aligning interventions with the points where the model commits to a semantic unit—and only moving in orthogonal directions—decouples content coherence from directional bias. This recipe is highly generalizable to dialogue style control or safety alignment.
- **The "Advantage Matrix - Q-Table - Confidence Gating" Trio**: Converting statistical bias ($R$) from the model itself into a programmable reward and using confidence to determine intervention necessity constitutes a clean, data-driven control framework with lower overhead than RLHF/DPO.

## Limitations & Future Work
- **Dependence on GPT-4o-mini for State Labeling**: Since the 6-state boundaries are annotated by a frontier model, the annotator's cognitive bias is embedded into $R$. Different domains (e.g., code, agent tools) likely require redesigned state taxonomies.
- **Strong Memoryless FSM Assumption**: Real reasoning often depends on history (e.g., "should I backtrack now" depends on "how many times I have already backtracked"). Pure Markovian graphs fail to capture these dependencies; the authors suggest POMDPs or machines with memory for future work.
- **Ambiguity in Sentence Boundary Detection**: Relying on `.?!` can mistake decimals in equations or abbreviations for sentence ends, which is a common bottleneck for sentence-level intervention methods.
- **Inhibition of Diversity**: Excessive alignment with "typical successful paths" might suppress unconventional but correct solutions.
- **Cross-Model $R$ Transfer is Lossy**: While cross-model experiments maintain most performance, the doubling of interventions suggests each LRM still requires its own $R$ to maximize efficiency, incurring offline costs during scaling.

## Related Work & Insights
- **vs. SEAL (Chen et al., 2025a)**: SEAL suppresses reflection/transition categories globally. This work dynamically selects the next jump based on state, refining control from "behavior categories" to "specific transitions" and answering *when & where*.
- **vs. Venhoff et al. (2025)**: While they prove individual behaviors correspond to linear directions, this work upgrades "isolated behavioral interventions" to "strategy-level control" and automates the intervention timing via classifiers and Q-Value gating.
- **vs. Bogdan et al. (2025) Thought Anchors**: Thought anchors use sentence-level analysis to identify key reasoning steps; this work builds on that granularity to provide an executable control strategy.
- **启发 (Insight)**: Abstracting open generation into finite discrete states and planning within that space is a powerful paradigm applicable to multi-agent collaboration, tool-use sequencing, and safety paths.

## Rating
- Novelty: ⭐⭐⭐⭐ While not the first to do CoT abstraction or activation steering, the combination using FSM + Q-Value iteration into a complete control framework is a clean and powerful innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐ 4 benchmarks × 3 LRMs × 3 steering strategies + cross-model transfer + prompt-based comparisons provide comprehensive evidence.
- Writing Quality: ⭐⭐⭐⭐ Clear framework diagrams, qualitative cases, derivations, and hyperparameter tables; cognitive theories are well-integrated.
- Value: ⭐⭐⭐⭐ Achieving higher accuracy with 25× fewer interventions makes this a highly practical training-free inference-time control method.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Inducing Overthink: Hierarchical Genetic Algorithm-based DoS Attack on Black-Box Large Language Reasoning Models](inducing_overthink_hierarchical_genetic_algorithm-based_dos_attack_on_black-box_.md)
- [\[ICML 2026\] Reasoning Structure of Large Language Models](reasoning_structure_of_large_language_models.md)
- [\[ICML 2026\] Are Large Reasoning Models Interruptible?](are_large_reasoning_models_interruptible.md)
- [\[CVPR 2026\] VisRef: Visual Refocusing while Thinking Improves Test-Time Scaling in Multi-Modal Large Reasoning Models](../../CVPR2026/llm_reasoning/visref_visual_refocusing_test_time_scaling.md)
- [\[NeurIPS 2025\] Controlling Thinking Speed in Reasoning Models](../../NeurIPS2025/llm_reasoning/controlling_thinking_speed_in_reasoning_models.md)

</div>

<!-- RELATED:END -->
