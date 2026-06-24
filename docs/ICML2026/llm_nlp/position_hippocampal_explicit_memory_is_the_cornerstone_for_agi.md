---
title: >-
  [Paper Note] Position: Hippocampal Explicit Memory Is the Cornerstone for AGI
description: >-
  [ICML2026][LLM (Other)][Explicit Memory] This position paper leverages neuroscientific evidence to argue that the underlying learning mechanism of LLMs corresponds to "implicit memory" in the human brain (habitual/procedural learning in the basal ganglia). However, higher-order cognition essential for AGI—such as long-range planning, metacognition, and symbolic reasoning—depends on hippocampal "explicit memory" and cannot emerge from purely statistical implicit learning. Cons…
tags:
  - "ICML2026"
  - "LLM (Other)"
  - "Explicit Memory"
  - "Implicit Memory"
  - "Hippocampus"
  - "AGI"
  - "Neuroscience"
date: 2026-05-08
content_hash: 4bb30a8e84fa7360
---

# Position: Hippocampal Explicit Memory Is the Cornerstone for AGI

**Conference**: ICML2026  
**arXiv**: [2606.11245](https://arxiv.org/abs/2606.11245)  
**Code**: None (Position paper, no code)  
**Area**: LLM / NLP (NeuroAI · AGI · Cognitive Science)  
**Keywords**: Explicit Memory, Implicit Memory, Hippocampus, AGI, Neuroscience

## TL;DR
This position paper leverages neuroscientific evidence to argue that the underlying learning mechanism of LLMs corresponds to "implicit memory" in the human brain (habitual/procedural learning in the basal ganglia). However, higher-order cognition essential for AGI—such as long-range planning, metacognition, and symbolic reasoning—depends on hippocampal "explicit memory" and cannot emerge from purely statistical implicit learning. Consequently, **supplementing LLMs with an explicit memory system is the cornerstone of the transition to AGI**. The authors further propose eight computational requirements for an artificial explicit memory system.

## Background & Motivation
**Background**: LLMs have reached human-level proficiency in writing, Q&A, coding, and dialogue, leading some (e.g., Microsoft's GPT-4 "Sparks of AGI") to view them as prototypes of AGI. The mainstream narrative suggests that continued scaling and post-training will eventually lead to the emergence of AGI capabilities.

**Limitations of Prior Work**: LLMs consistently struggle with hallucinations, long-range planning, and logical reasoning, and they cannot dynamically access information across long time scales. "Memory" has long been regarded as a structural weakness of LLMs. The authors argue these issues are not fragmented bugs but different manifestations of a single root cause.

**Key Challenge**: The human brain possesses two distinct memory systems: **explicit memory** (hippocampus; sparse coding, rapid one-shot establishment, supporting conscious knowledge-based problem solving) and **implicit memory** (basal ganglia; dense coding, slowly shaped by repetition and reward prediction error, generating automated stimulus-response). The cognitive functions supported by these systems are **not interchangeable**. LLM training mechanisms (gradient descent, dense distributive activation, error-driven weight adjustment, next-token prediction) all correspond to implicit memory; none correspond to hippocampal explicit memory.

**Goal**: (1) To demonstrate that LLM learning $\approx$ implicit memory; (2) To clarify which higher-order cognitive functions **must** rely on explicit memory, defining the ceiling for LLM capabilities; (3) To provide formal computational requirements for artificial explicit memory systems.

**Core Idea**: The gap toward AGI is diagnosed as "the absence of a hippocampal-like explicit memory system." Building an explicit memory system is proposed as an actionable research agenda, rather than relying on the scaling of implicit systems for higher-order cognition. The criterion for AGI used is **Human-Level AI**: general human-level learning, reasoning, and knowledge transfer across all cognitive tasks and domains.

## Method
As a position paper, this work lacks a traditional pipeline, presenting instead an argumentative chain: "Neuroscience facts → Deduce LLM capability boundaries → Propose engineering requirements."

### Overall Architecture
The argument proceeds in four steps: ① Clarify the conceptual, formation, and retrieval differences between **explicit vs. implicit** memory systems; ② Map low-layer LLM learning mechanisms to implicit memory characteristics; ③ Use neuroscience literature to **attribute** cognitive functions to memory systems, inferring what LLMs can do (pattern recognition, fluent language) and cannot do (logical reasoning, metacognition, executive functions, mental simulation); ④ Propose eight computational requirements that artificial explicit memory systems must satisfy to reach engineering specifications. The fulcrum of this logic is the neuroscientific consensus that memory systems dictate cognitive functions and are not interchangeable.

### Key Designs

**1. Dual Memory System Comparison: Sparse-Fast vs. Dense-Slow**

The authors establish the foundation by detailing the two memory systems. **Explicit memory** is supported by the hippocampal trisynaptic circuit (entorhinal cortex EC → dentate gyrus DG → CA3 → CA1). The DG performs **sparse coding** to achieve **pattern separation**—similar inputs are mapped to non-overlapping representations to avoid interference. Recurrent circuits in CA3 **bind** co-activated elements into a whole recoverable from partial cues (**pattern completion**). NMDA-dependent LTP allows establishment within a single experience. **Implicit memory** resides in the basal ganglia, receiving **dense** inputs via cortical-basal ganglia-thalamic loops. Medium spiny neurons (MSNs) in the striatum are adjusted based on dopamine reward prediction errors, requiring extensive trial and error. In short: explicit memory is sparse, fast, and reconstructive; implicit memory is dense, slow, and reactive.

**2. Mapping LLM Learning Mechanisms to Implicit Memory**

This is the core argument. The authors align LLMs with implicit memory across four dimensions: (i) **Incremental learning**: LLMs adjust weights slowly to minimize loss across massive corpora, mirroring implicit habit formation rather than one-shot explicit learning. (ii) **Dense distributive encoding**: LLM activations for a context are spread across many parameters rather than isolated nodes, corresponding to dense basal ganglia signals. (iii) **Error-driven**: Parameters are tuned via backpropagation based on the difference between output and ground truth, analogous to dopamine reward prediction error; explicit memory relies on **associative** binding of information, not error. (iv) **Automatic production**: Given a prompt, an LLM does not reconstruct a scenario from a cue like the hippocampus; instead, pre-learned parameters determine the next optimal token, behaving like the "familiar stimulus → automatic response" mode of the basal ganglia.

**3. Memory Attribution of Cognitive Functions → LLM Capability Boundaries**

By assigning functions to systems, the authors define the LLM ceiling. **Implicit memory** side (LLM strengths): Pattern recognition and linguistic fluency (consistent with the Declarative/Procedural model where grammar is automated/implicit). **Explicit memory** side (LLM weaknesses): Logic/math reasoning (dependent on abstract concepts and associations in semantic memory), executive functions (planning/decision-making requiring episodic retrieval beyond the context window), metacognition (tracking knowledge sources—the lack of episodic metadata leads to hallucinations), and mental simulation (reorganizing the past to imagine the future).

**4. 8 Computational Requirements for Artificial Explicit Memory**

The position concludes with engineering specifications. Explicit memory is abstracted as a module $f_{\mathrm{memory}}: I \to O$, where input $I=(E,M)$ (dense embedding $E\in\mathbb{R}^d$ and memory state $M$) and output $O=(\Delta M, Y)$ (update and retrieved embedding $Y$). Key requirements include: **Sparse indexing** ($S=\mathrm{sparsify}(E)$ where $\|S\|_0 \ll n$, and a mapping $P$ such that $P^\top S \approx E$); **Error-independent updates** ($\nabla_{p,e}\Delta M = \mathbf{0}$, updates are driven by association rather than prediction error); **Association construction** (association matrix $A$ updates such that $\Delta A_{i,j}>0,\ \forall i,j\in F_S$); **Pattern separation** ($\mathrm{sim}(S_1,S_2)<\mathrm{sim}(E_1,E_2)$); **Pattern completion** (reconstructing information from partial input via $Y=P^\top \sigma(A S^{\mathrm{partial}})$); and **Dynamics**, **High/Immediate Plasticity**, and **Adaptive Forgetting**.

### Example: The "Abacus" Difference in $17\times 6$
The paper uses $17\times 6$ to illustrate the distinction. Humans use **explicit memory** to decompose the problem into concepts and arithmetic facts to derive 102 **semantically**. While an LLM can output similar-looking tokens to arrive at 102, its process is more like **using an abacus**—automatically following learned operational rules without semantic understanding of the numbers or the process.

## Key Experimental Results
As a position paper, it lacks standard benchmarks, presenting instead falsifiable criteria and comparative analysis.

### Comparison of Memory Systems

| Dimension | Explicit Memory (Hippocampus) | Implicit Memory (Basal Ganglia) |
|------|------|------|
| Encoding | Sparse coding (Pattern separation) | Dense coding (Massively parallel) |
| Formation Speed | One-shot establishment | Slow formation through repetition |
| Learning Driver | Associative binding of information | Reward prediction error (Dopamine) |
| Retrieval | Reconstruction from cues (Pattern completion) | Familiar stimulus → Automatic response |
| Supported Functions | Logic, planning, metacognition | Pattern recognition, fluency, habits |

### LLM ↔ Memory Mapping

| LLM Feature | Corresponds to Implicit? | Corresponds to Explicit? |
|------|------|------|
| Incremental weight adjustment | ✓ Slow formation | ✗ (Explicit is one-shot) |
| Dense distributive activation | ✓ Dense encoding | ✗ (Explicit is sparse) |
| Error-driven backpropagation | ✓ Reward prediction error | ✗ (Explicit is associative) |
| Prompt → Auto-next-token | ✓ Stimulus-response | ✗ (Explicit is reconstructive) |

### 8 Computational Requirements for Artificial Explicit Memory

| Requirement | Formal Point |
|------|------|
| Sparse Indexing | $\|S\|_0 \ll n$, $P^\top S\approx E$ |
| Error-Independent Update | $\nabla_{p,e}\Delta M=\mathbf{0}$ |
| Association Construction | $\Delta A_{i,j}>0,\ \forall i,j\in F_S$ |
| Pattern Separation | $\mathrm{sim}(S_1,S_2)<\mathrm{sim}(E_1,E_2)$ |
| Pattern Completion | $Y=P^\top \sigma(A S^{\mathrm{partial}})$ |
| Dynamics | $M_{t+1}=M_t+\Delta M_t$ |
| High Plasticity | $Y_{t+1}\approx E_t$ after one shot |
| Adaptive Forgetting | $i\in F_S, j\notin F_S\Rightarrow \Delta A_{i,j}\le 0$ |

### Key Findings
- **Falsifiability**: The authors provide empirical tests. "Falsifying explicit handling" is easy: failing on a problem adjacent to successfully used knowledge indicates a lack of explicit processing. "Proving explicit handling" is harder because failures are often retrained into the stats. The most reliable evidence is **adaptation tests after excluding specific tasks from training**.
- **Reason for Current LLM Strength**: Scaling + optimization. Unlike the human brain's limited experience time, LLM pre-training involves more text than a human reads in a lifetime, pushing implicit learning to biological impossibilities. This does not contradict their nature as implicit systems.

## Highlights & Insights
- **Unified Diagnostic Framework**: Hallucinations and planning failures are unified under the "missing hippocampal explicit memory" diagnosis, which is more explanatory than treating them as isolated bugs.
- **Engineering Specs from Neuro-mechanisms**: The eight requirements translate "adding explicit memory" into a concrete module interface ($f_{\mathrm{memory}}$), offering direct value for memory-augmented/external KV memory research.
- **Separation from Gradient Learning**: The $\nabla_{p,e}\Delta M=\mathbf{0}$ constraint explicitly warns against using gradient-descent modules to simulate explicit memory, advocating for non-error-driven writing mechanisms.

## Limitations & Future Work
- **Limitations acknowledged by authors**: The requirements are "essential conditions" rather than full biological replications. Simplifications include symmetric $P$ and omitted cortical consolidation.
- **Identified Limitations**: The paper is entirely discursive with **no empirical prototype**. It remains unverified if these requirements can actually solve hallucinations. The LLM learning $\approx$ implicit memory analogy is qualitative and may overlook technical nuances.
- **Future Work**: Falsifiable experiments (training exclusion + adaptation tests) are the necessary next steps. Additionally, whether the explicit memory module depends on the upstream representation already being "explicit" remains an open question.

## Related Work & Insights
- **vs. Complementary Learning Systems (McClelland et al.)**: CLS views LLMs as a parallel to neocortical semantic (explicit) memory. This paper argues LLMs cannot produce flexible semantic memory and have regressed into rigid stimulus-response implicit systems.
- **vs. Shang et al. (2024) AI-native long-term memory**: Their work focuses on LLMs as processors for external NLP memory. This paper serves as a lower-level normative specification for such external systems.
- **vs. "GPT-4 as Sparks of AGI" (Bubeck et al.)**: This paper responds by noting that high capability $\neq$ higher-order cognitive mechanisms.

## Rating
- Novelty: ⭐⭐⭐⭐ Unified diagnosis using dual memory systems is fresh and insightful.
- Experimental Thoroughness: ⭐⭐ No empirical results; relies on literature and logic.
- Writing Quality: ⭐⭐⭐⭐ Clear argumentation with excellent examples like $17\times6$.
- Value: ⭐⭐⭐⭐ Highly influential for the memory-augmented and AGI roadmap discussions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Position: The Turing-Completeness of Autoregressive Transformers Relies Heavily on Context Management](position_the_turing-completeness_of_autoregressive_transformers_relies_heavily_o.md)
- [\[ICML 2026\] Position: The ML Community Must Build an AI-Augmented Peer-Review Ecosystem](position_the_ml_community_must_build_an_ai-augmented_peer-review_ecosystem.md)
- [\[ACL 2025\] Explicit and Implicit Data Augmentation for Social Event Detection](../../ACL2025/llm_nlp/explicit_and_implicit_data_augmentation_for_social_event_detection.md)
- [\[ICML 2026\] Position: Adversarial ML for LLMs Is Not Making Any Progress](position_adversarial_ml_for_llms_is_not_making_any_progress.md)
- [\[ACL 2026\] Think in Sentences: Explicit Sentence Boundaries Enhance Language Model's Capabilities](../../ACL2026/llm_nlp/think_in_sentences_explicit_sentence_boundaries_enhance_language_model39s_capabi.md)

</div>

<!-- RELATED:END -->
