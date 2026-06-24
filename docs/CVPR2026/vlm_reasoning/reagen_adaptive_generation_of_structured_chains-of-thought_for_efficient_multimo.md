---
title: >-
  [Paper Note] ReaGEN: Adaptive Generation of Structured Chains-of-Thought for Efficient Multimodal Reasoning
description: >-
  [CVPR 2026][VLM Reasoning][Multimodal Reasoning] ReaGEN does not fine-tune the base Vision-Language Model (VLM). Instead, it employs a lightweight generator with only 18M parameters to adaptively "output" a structured chain-of-thought (determining which reasoning stages to use and in what order) based on the attention flow of each problem. This achieves accuracy close to deep search with a single inference pass—yielding a maximum improvement of +26 accuracy points over VReST…
tags:
  - "CVPR 2026"
  - "VLM Reasoning"
  - "Multimodal Reasoning"
  - "Structured Chain-of-Thought"
  - "Attention Feedback"
  - "Evolutionary Search"
  - "Inference Efficiency"
date: 2026-05-08
content_hash: 82dc31a6debff0a1
---

# ReaGEN: Adaptive Generation of Structured Chains-of-Thought for Efficient Multimodal Reasoning

**Conference**: CVPR 2026  
**Paper**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Tian_ReaGEN_Adaptive_Generation_of_Structured_Chains-of-Thought_for_Efficient_Multimodal_Reasoning_CVPR_2026_paper.html)  
**Code**: https://github.com/AISmartPerception/ReaGEN  
**Area**: Multimodal VLM / LLM Reasoning  
**Keywords**: Multimodal Reasoning, Structured Chain-of-Thought, Attention Feedback, Evolutionary Search, Inference Efficiency

## TL;DR
ReaGEN does not fine-tune the base Vision-Language Model (VLM). Instead, it employs a lightweight generator with only 18M parameters to adaptively "output" a structured chain-of-thought (determining which reasoning stages to use and in what order) based on the attention flow of each problem. This achieves accuracy close to deep search with a single inference pass—yielding a maximum improvement of +26 accuracy points over VReST on Qwen3-VL-4B, while reducing average token usage by approximately 53% (reaching up to 79% on certain benchmarks).

## Background & Motivation

**Background**: Large Vision-Language Models (LVLMs) are powerful in tasks like VQA and chart understanding but often fail when faced with complex visual problems requiring multi-step, compositional reasoning. The primary remedy is Chain-of-Thought (CoT), which prompts the model to output intermediate steps such as "Observation → Association → Calculation → Answer." Current methods to enhance VLM reasoning fall into two categories: (i) inference-time scaling (multi-path sampling, Tree/Graph-of-Thought, MCTS, etc.), and (ii) post-training enhancement (SFT or RL on large-scale CoT corpora).

**Limitations of Prior Work**: Inference-time scaling relies on repeated model calls and deep search, leading to high latency and token costs. SFT often requires hundreds of thousands of high-quality multimodal CoT samples (frequently dependent on closed-source models like GPT-4), making training expensive and transferability poor. RL post-training primarily optimizes training rewards, which can narrow strategies to a few high-reward templates, resulting in mode collapse, reward hacking, and style overfitting.

**Key Challenge**: There is a rigid trade-off between "flexible but expensive" search-based methods and "cheap but rigid and data-hungry" training-based methods. Furthermore, most methods focus on modifying the *content* of reasoning, while the explicit optimization of the **structure** of reasoning (which stages to use and in what sequence) remains under-explored.

**Goal**: To dynamically generate an optimal structured CoT for each problem while keeping the **base VLM frozen and without any fine-tuning**, aiming to combine the flexibility of multi-path exploration with the efficiency of single-pass inference.

**Key Insight**: The authors observe that dependencies between reasoning stages are exposed through **attention flows** (how much a subsequent stage "looks at" previous stages). Since attention reveals which stages are critical to the final answer and how they depend on each other, it can serve as a signal to edit and optimize the sequence of CoT stages.

**Core Idea**: The process of "how to organize reasoning structures" is distilled into a lightweight generator (GEN). High-quality CoT structures for each problem are first discovered offline via teacher-guided evolutionary search (using importance scores derived from attention). GEN is then trained to predict these optimal structures directly from attention signals. At inference time, only one pass (or a few refinement iterations) of GEN is required, trading search flexibility for single-pass efficiency.

## Method

### Overall Architecture
The core of ReaGEN is as follows: **the base VLM remains frozen throughout, while a lightweight external generator (GEN) is trained to output a customized sequence of reasoning stages for each image-query pair.** The pipeline consists of three phases: ① Offline phase: A stronger teacher VLM (Qwen3-VL-32B) guides evolutionary search to discover high-reward CoT structures for training problems, recording the cross-stage attention of the student VLM (Qwen3-VL-4B) during execution. ② Training phase: These "problem + attention + target CoT" triplets supervise GEN, teaching it to map attention summaries to optimal structures. ③ Inference phase: For new problems, GEN predicts a CoT sequence, and the frozen student VLM executes it to obtain the answer, with optional iterative refinement.

```mermaid
graph TD
    A["Input: Image I + Question Q"] --> B["Stage Action Space & CoT Encoding<br/>4 Functional Groups → Stage ID Sequence"]
    B --> C["Teacher-Guided Evolutionary Search<br/>Evaluate→Select→Mutate<br/>Attention Importance + Reward"]
    C -->|Offline Output (I,Q,A,τ*) Triplets| D["Lightweight GEN Generator<br/>18.3M Transformer, Frozen VLM"]
    D --> E["Single-pass / Iterative Refinement<br/>GEN predicts τ̂ → Student VLM executes"]
    E --> F["Final Answer"]
```

### Key Designs

**1. Stage Action Space & CoT Encoding: Transforming Structure into Predictable Discrete Sequences**

To enable a generator to organize reasoning structures, the structures must be converted into enumerable and predictable objects. ReaGEN decomposes multimodal reasoning into a **stage pool**, categorized into four functional groups: ① Perception and Input Understanding (reading the question, parsing visual content); ② Grounding and Fact Extraction (linking text entities to the image, identifying key variables); ③ Reasoning and Inference (relationship, logic, and numerical derivation); ④ Explanation and Answer Selection (summarizing intermediate conclusions, producing the final answer). A CoT is an ordered sequence of these actions $\tau^\star = (s_1,\dots,s_L)$. For GEN to perform auto-regressive prediction, each CoT is encoded into a fixed-length stage ID vector $\mathbf{c} = (c_1,\dots,c_L, c_{L+1},\dots,c_{L_{\max}})$, where $c_t \in \{1,\dots,S\}$ is the stage ID, $c_{L+1}=\text{EOS}$, and the remainder is padded. This reduces the problem of "which stages and what order" to a standard sequence prediction task, which is a prerequisite for a lightweight GEN.

**2. Teacher-Guided Evolutionary Search + Attention Importance: Using Attention Signals to Mine Optimal Structures**

This serves as the offline data engine, addressing the high cost of quality multimodal CoT corpora. For each problem, search iterates through **Evaluate→Select→Mutate**: the **frozen student VLM** executes the candidate CoT stage-by-stage. Each stage $s_t$ has a specific system prompt and maintains a memory $M_t$ accumulating all previous stage outputs, i.e., $M_{t+1}=M_t\cup\{y_t\}$, where $y_t=f_\theta(I,Q,M_t,s_t)$. During execution, **cross-stage attention** is extracted. Unnormalized attention mass $\tilde A_{i,j}$ (average attention of stage $j$ tokens on $i$ tokens) is accumulated, then normalized over earlier stages to get the attention quality matrix $A_{i,j}$, measuring how much stage $j$ depends on stage $i$. Based on this, **stage importance** $\mathrm{Imp}(i)$ is defined by recursively summing direct contributions (attention to the final answer $A_{i,F}$) and indirect contributions (propagated through subsequent stages):

$$\mathrm{Imp}(i) = A_{i,F} + \sum_{j=i+1}^{N} \lambda^{\,j-i} A_{i,j}\,\mathrm{Imp}(j)$$

where $\lambda\in(0,1]$ is a discount factor for long dependency chains. Each candidate also receives a scalar reward $R(\tau)=\alpha\,s(\tau)-\beta\,\ell(\tau)-\gamma\,d(\tau)$, encouraging accuracy (prediction score $s$), conciseness (length penalty $\ell$), and structural diversity (diversity penalty $d$). After selecting the top-K elite set, a stronger teacher model performs **importance-aware directional mutation** (e.g., pruning low-impact stages or advancing critical stages) rather than random modification. Attention acts as a guide, allowing the search to converge efficiently.

**3. Lightweight GEN Generator: Distilling Search Results into Single-Pass Structure Prediction**

While effective, search is too expensive for online deployment. GEN functions by mimicking the search outputs. Training data consists of $\{(I,Q,A,\tau^{\text{init}},\tau^\star)\}$, where image embeddings $E^{\text{Img}}$ and question embeddings $E^{\text{Q}}$ are extracted from the frozen VLM, alongside the structure matrix $A$ and initial chain $\tau^{\text{init}}$. The supervision signal is the target CoT $\tau^\star$. GEN itself is a compact 4-layer, 18.3M parameter Transformer encoder-decoder. It projects image/question embeddings into a shared space to fuse multimodal memory and transforms $A$ into a structural representation that modulates stage embeddings. The decoder uses two heads to predict the stage ID sequence (with EOS) and the CoT length. The training objective is the sum of cross-entropy losses for stage and length prediction. Crucially, the **base VLM is completely frozen**—the burden of learning structure is placed on this 18M model, making training data-efficient and inexpensive without degrading the base model's general capabilities.

**4. Single-pass / Iterative Refinement Inference: Single-Pass for Structure, Optional Refinement for High Accuracy**

ReaGEN offers two deployment modes. **Single-pass (2 iter)**: An initial/seed CoT $\tau^{(0)}$ is used to collect an attention summary from the student, based on which GEN predicts a customized $\hat\tau$. The student executes $\hat\tau$ once to get the answer—totalling two student calls without search. **Iterative Refinement**: The same GEN can be called repeatedly. In each round: (a) the student runs $\tau^{(t)}$ to produce new attention; (b) GEN proposes $\tau^{(t+1)}$; (c) the process continues until convergence $\tau^{(t+1)}=\tau^{(t)}$ or the iteration budget is reached. Given GEN's lightweight nature, $K$ parallel refinement branches ("width," $K=5$ in experiments) can be initiated from different seeds, with the final answer determined by majority vote. This provides a fast default single-pass plan while allowing low-cost refinement cycles for stronger "problem-adaptive" structures.

### Loss & Training
GEN is trained for 200 epochs using AdamW with mixed precision, a batch size of 64, and a learning rate of $1\times10^{-4}$. Evaluation involves a mixture of visual and mathematical data, including 1,050 samples from MMMU and 1,496 samples from MathVerse (text-only). The offline search phase has an evolutionary budget cap of 20 rounds per problem. The loss is the sum of cross-entropy for stage ID sequence prediction and CoT length prediction.

## Key Experimental Results

### Main Results
The base model is a frozen Qwen3-VL-4B, with Qwen3-VL-32B as the teacher. The following table shows accuracy across composite benchmarks (absolute change relative to VReST with the same base is in parentheses):

| Method | MMMU-Pro(10) | MMMU-Pro(4) | VStar | MMStar | MathVision | MathVerse |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Direct Answer | 32.42% | 45.78% | 79.14% | 58.95% | 22.59% | 25.25% |
| + CoT | 37.57% | 49.02% | 76.47% | 67.85% | 29.75% | 36.17% |
| + VReST | 46.30% | 56.13% | 83.42% | 49.27% | 44.67% | 47.97% |
| **+ ReaGEN (4 iter)** | **52.54%** (↑6.2) | **64.51%** (↑8.4) | 84.49% (↑1.1) | **75.77%** (↑26.5) | 44.60% (↓0.1) | 47.59% (↓0.4) |

> On MMStar, ReaGEN outperforms VReST by approximately +26.5 points (VReST performance drops to 49.27% on this set). However, on math-oriented sets like MathVision/MathVerse, ReaGEN is roughly on par with or slightly below VReST, suggesting its advantage lies primarily in **vision-intensive** reasoning. ⚠️ Note that the summary mentions "+26 accuracy points / 79% token reduction" while the text cites an "average reduction of 53%." The former represents peak values for single benchmarks, while the latter is the overall mean.

### Efficiency and Generalization

| Benchmark | VReST token | ReaGEN-4 token | Relative Reduction |
| :--- | :--- | :--- | :--- |
| MMMU-Pro(10) | 188 | 45 | 76% |
| MMMU-Pro(4) | 176 | 46 | 74% |
| MathVision | 240 | 83 | 65% |
| MMStar | 140 | 87 | 38% |
| VStar | 86 | 77 | 11% |
| **Overall** | **166** | **68** | **53%** |

In terms of cross-dataset generalization (GEN trained only on MathVision and evaluated on others), it still achieves ~+26 points on the visual benchmark MMStar relative to VReST without retraining.

### Key Findings
- **Iterative gains are monotonic**: Moving from 2→3→4 iterations yields steady improvements on most visual benchmarks (e.g., MMMU-Pro(10) rises from 49.94%→51.90%→52.54%), confirming that trade-offs between minor GEN/student calls and accuracy are effective.
- **18M GEN competes with 32B Teacher**: Under the 4-iteration setting, ReaGEN outperforms even the stronger VReST (Teacher-Reward, 32B) on most benchmarks, without requiring the 32B teacher during inference.
- **Improved efficiency in both calls and tokens**: Compared to VReST, ReaGEN significantly reduces both VLM call frequency and generated tokens. Efficiency gains stem from replacing deep search with single-pass planning.

## Highlights & Insights
- **Attention as a "Structural Supervision Signal"**: Most CoT works optimize reasoning content. ReaGEN uses cross-stage attention to define stage importance $\mathrm{Imp}(i)$ and optimize the reasoning **skeleton**. This is a novel perspective—attention is a built-in dependency graph provided by the model itself, making it free and interpretable.
- **Structure Distillation (Search-as-Teacher, Small-Model-as-Student)**: Compressing expensive evolutionary search into an 18M generator while keeping the base frozen effectively creates a "structure organizing" plugin for any open-source VLM, ensuring high reusability.
- **Single-pass vs. Iteration Adjustable Knob**: A single GEN can handle both fast single-pass output and multi-branch refinement/voting, creating a continuous adjustable scale for the accuracy/cost trade-off, which is highly practical for engineering.
- **Transferable Trick**: The paradigm of "discretizing procedural structures into a stage pool and using a small model to predict stage IDs" can be transferred to agent planning or tool-calling sequences.

## Limitations & Future Work
- **Dependency on Internal Attention Signals**: The method requires access to base VLM attention weights, limiting its use to self-hosted or open-source VLMs; closed-source APIs (output-only) cannot use it directly.
- **Limited Advantage in Math Reasoning**: Parity or minor performance drops on MathVision/MathVerse indicate that stage-based structure optimization provides less gain for pure symbolic/numerical chains than for vision-intensive tasks. Structure may not be the bottleneck for math problems.
- **Manually Designed Stage Pool**: Functional groups and system prompts are pre-defined by the authors. Moving to new domains might require re-designing the stage vocabulary; automated exploration of stage granularity is missing.
- **Overhead of Multi-Branch Voting**: While iterative refinement with $K=5$ parallel branches is lighter than MCTS, it still involves several times more student calls than a single pass. Token counts in the paper primarily address single-pass settings; the true cost of refinement must account for branch counts.

## Related Work & Insights
- **vs. VReST / Socratic-MCTS / AStar (Inference-time Search)**: These methods gain accuracy via tree/graph search or thought-segment reuse during inference, but repeated model calls lead to high latency. ReaGEN moves the search offline and runs GEN once online, retaining exploration flexibility without the online search cost.
- **vs. VisualCoT / LLaVA-CoT / Chain-of-Focus (Training-based Post-training)**: These fine-tune the base VLM on large CoT corpora or RL rollouts, which is expensive and makes structures hard to reuse across models. ReaGEN freezes the base and trains an 18M generator, making it data-efficient and plug-and-play.
- **Insight**: When content is sufficient but structure is the bottleneck, distilling "procedural orchestration" into a lightweight external plugin is a more economical route than fine-tuning the entire large model, especially for cost-sensitive open-source deployments.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Using attention to define stage importance and distilling reasoning structure into a lightweight generator is a fresh perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Five benchmarks + cross-dataset generalization + efficiency analysis are comprehensive, though the boundary of math reasoning gains is a noted limitation.
- Writing Quality: ⭐⭐⭐⭐ The three-stage pipeline (Search→Training→Inference) is well-explained; ⚠️ Note the minor discrepancy between the summary (+26 pts / 79% reduction) and the main text (53% average).
- Value: ⭐⭐⭐⭐ Provides a practical solution for open-source VLMs with "search-like accuracy at single-pass costs," offering high engineering value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Reinforcing Structured Chain-of-Thought for Video Understanding](reinforcing_structured_chain-of-thought_for_video_understanding.md)
- [\[CVPR 2026\] Stable and Efficient Single-Rollout RL for Multimodal Reasoning](stable_and_efficient_single-rollout_rl_for_multimodal_reasoning.md)
- [\[CVPR 2026\] Grounded Chain-of-Thought for Multimodal Large Language Models](grounded_chain-of-thought_for_multimodal_large_language_models.md)
- [\[ICLR 2026\] SketchThinker-R1: Towards Efficient Sketch-Style Reasoning in Large Multimodal Models](../../ICLR2026/vlm_reasoning/sketchthinker-r1_towards_efficient_sketch-style_reasoning_in_large_multimodal_mo.md)
- [\[CVPR 2026\] UniT: Unified Multimodal Chain-of-Thought Test-time Scaling](unit_unified_multimodal_chain-of-thought_test-time_scaling.md)

</div>

<!-- RELATED:END -->
