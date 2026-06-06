---
title: >-
  [Paper Note] ProtoCycle: Reflective Tool-Augmented Planning for Text-Guided Protein Design
description: >-
  [ACL 2026 (Findings)][Computational Biology][Protein Design] ProtoCycle proposes a **reflective agent framework** that utilizes an LLM as a planner combined with a lightweight tool environment for text-guided protein seq…
tags:
  - "ACL 2026 (Findings)"
  - "Computational Biology"
  - "Protein Design"
  - "Text-Guided"
  - "Reflective Planning"
  - "Tool-Augmentation"
  - "Reinforcement Learning"
date: 2026-05-08
content_hash: e93b7fa452f1129c
---

# ProtoCycle: Reflective Tool-Augmented Planning for Text-Guided Protein Design

**Conference**: ACL 2026 (Findings)  
**arXiv**: [2604.16896](https://arxiv.org/abs/2604.16896)  
**Code**: [https://github.com/huggggoooooo/ProtoCycle](https://github.com/huggggoooooo/ProtoCycle)  
**Area**: AI for Science / Protein Design  
**Keywords**: Protein Design, Text-Guided, Reflective Planning, Tool-Augmentation, Reinforcement Learning

## TL;DR

ProtoCycle proposes a **reflective agent framework** that utilizes an LLM as a planner combined with a lightweight tool environment for text-guided protein sequence design. By replacing one-shot text-to-sequence generation with a multi-round "plan-tool-evaluate-reflect" cycle, it improves ProTrek to 14.681 and Retrieval to 0.936 on Mol-Instructions. Using only ~2,000 SFT trajectories and online RL, it achieves language alignment performance that approaches or exceeds specialized protein design models.

## Background & Motivation

**Background**: Designing proteins that satisfy functional requirements described in natural language is a core objective of protein engineering. A direct approach involves fine-tuning general instruction-tuned LLMs as text-to-sequence generators; however, this approach is both data- and computation-intensive.

**Limitations of Prior Work**: (1) Direct text-to-sequence methods require vast amounts of supervised data and computational resources; (2) Under limited supervision, LLMs can generate coherent textual plans but fail to reliably implement them as protein sequences, leading to a **plan-execute gap**; (3) Protein design necessitates iterative trial and error, whereas most existing methods rely on one-shot generation.

**Key Challenge**: LLMs excel at understanding natural language functional descriptions and generating plans, but they struggle with direct mapping from text to valid protein sequences, especially when training data is scarce.

**Goal**: To build a protein design framework that leverages the planning capabilities of LLMs while addressing their weaknesses in sequence generation.

**Key Insight**: Drawing inspiration from the iterative workflows of human protein engineers, the process is framed as a multi-round cycle of "planning → execution → feedback → revision" rather than single-step generation, positioning the LLM as a planner rather than a generator.

**Core Idea**: A coupling of an LLM planner with a lightweight tool environment is established. Tools provide sequence manipulation and evaluation functions, while the LLM iteratively refines design schemes by reflecting on tool feedback. The agent's capabilities are enhanced through supervised trajectories and online reinforcement learning.

## Method

### Overall Architecture

ProtoCycle formalizes text-guided protein design as a multi-step decision process. Given a natural language requirement $r$, the planner outputs a state $s_t$ and tool action $a_t$ at round $t$ based on the history, tool feedback, and original requirement. The action $a_t$ is decomposed into a tool type and tool parameters, which are executed by the tool environment to return a feedback summary. The LLM does not directly generate the full amino acid sequence; instead, it handles requirement decomposition, tool selection, strategy updates, reflection, and termination judgment. Specialized tools handle specific sequence generation and local editing.

Each round's output follows a three-part protocol: `<think>`, `<plan>`, and `<tool_call>`. In the first round, the planner decomposes requirements into fine-grained sub-goals and plans the tool invocation sequence. In subsequent rounds, the planner reads the number of candidates, current best score, historical best score, and gain $\Delta$ returned by the tools to decide whether to continue the plan, modify the strategy, or terminate. Before termination, an evaluation tool is triggered to re-score the top-5 candidates; if room for improvement remains, planning continues; otherwise, the best sequence is output.

### Key Designs

**1. Reflective Multi-round Decision Cycle**

- **Function**: To simulate the iterative trial-and-error process of human protein engineers.
- **Mechanism**: In each round, the LLM generates actions based on the current state and historical feedback (e.g., selecting a scaffold, choosing functional sites, or adjusting local descriptions). The tool environment executes these and returns candidate sequences along with a ProTrek score summary. The LLM explicitly reflects on whether the "current strategy is effective" before deciding to continue, modify, or stop.
- **Design Motivation**: Protein design is inherently an iterative optimization process; single-shot generation struggles to meet complex functional requirements. The LLM-driven reflection mechanism enables the agent to learn from failures and adjust strategies.

**2. Lightweight Tool Environment**

- **Function**: To provide core operations and evaluation capabilities required for protein design.
- **Mechanism**: The tool environment consists of three core tools: scaffold generation retrieves and merges candidate scaffolds from knowledge bases like UniProt/Rhea/InterPro/QuickGO; functional-site design generates local site variants based on ESM2-3B; evaluation uses ProTrek and Chai-1 to assess language alignment and foldability.
- **Design Motivation**: LLMs excel at high-level planning but struggle with low-level sequence manipulation. The tool environment compensates for this weakness while making the design process interpretable and traceable.

**3. Supervised + Online Reinforcement Learning Training**

- **Function**: To train the agent's planning and reflection capabilities in stages.
- **Mechanism**: In the first stage, SFT is performed using tool interaction trajectories constructed from 2,000 Mol-Instructions samples. Cross-entropy is calculated only on the planner states $s_1,\ldots,s_n$ to teach the model the `<think>/<plan>/<tool_call>` protocol. In the second stage, online RL is conducted via GRPO in a real tool environment. The reward encourages correct formatting, reasonable tool usage, reflection after poor feedback, and task completion within an appropriate number of rounds.
- **Design Motivation**: Supervised learning provides cold-start capability, while RL further optimizes strategies beyond expert levels.

## Key Experimental Results

### Main Results

| Model | PPL↓ | Repeat↓ | pTM↑ | pLDDT↑ | PAE↓ | ProTrek↑ | EvoLLaMA↑ | Retrieval↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Natural | 4.737 | 2.129 | 0.762 | 0.815 | 9.443 | 14.628 | 0.328 | 0.848 |
| Qwen2.5-7B-Agent | 8.235 | 5.153 | 0.542 | 0.699 | 15.299 | 6.926 | 0.261 | 0.523 |
| Qwen2.5-72B-Agent | 7.414 | 5.341 | 0.618 | 0.714 | 13.343 | 8.791 | 0.267 | 0.563 |
| Qwen3-8B-Agent | 7.227 | 3.795 | 0.650 | 0.723 | 13.493 | 8.705 | 0.277 | 0.573 |
| ProDVa | 5.265 | 1.580 | 0.765 | 0.800 | 8.761 | 12.037 | 0.317 | 0.730 |
| Pinal | 3.990 | 9.317 | 0.792 | 0.825 | 7.768 | 14.162 | 0.318 | 0.807 |
| ProtoCycle-SFT | 4.149 | 2.902 | 0.734 | 0.807 | 10.200 | 12.502 | 0.317 | 0.840 |
| ProtoCycle-RL | **3.865** | 2.549 | 0.775 | 0.822 | 8.543 | **14.681** | 0.323 | **0.936** |

ProtoCycle-RL achieves the strongest language alignment: ProTrek shows a Gain of 3.66% over Pinal and 21.97% over ProDVa; Retrieval reaches 0.936, significantly higher than Natural, Pinal, and ProDVa. Regarding folding quality, compared to Pinal, the pTM/pLDDT is slightly lower and PAE is slightly higher; however, it outperforms ProDVa across pTM, pLDDT, and PAE, indicating that the agentic workflow does not sacrifice basic structural foldability.

### Ablation Study

| Experiment | Key Finding |
|---|---|
| ProtoCycle-RL vs ProtoCycle-SFT | ProTrek increased from 12.502 to 14.681, Retrieval from 0.840 to 0.936; PPL and Repeat decreased by 6.85% and 12.16% respectively. |
| CAMEO Generalization | Without training on keyword-style data, Ours still achieves pLDDT 0.80, ProTrek 11.17, and Keyword Recovery 0.59. |
| Single Tool Quality | Scaffold search: ProTrek 11.42, PAE 8.96, pLDDT 0.83; Functional-site design: ProTrek 12.87, PAE 10.73, pLDDT 0.80. |
| Tool Latency | Scaffold search: ~4s/round; functional-site design: ~20s/seq; ProTrek-35M: ~3s/round; ProTrek-650M: ~40s/round. |
| Reflection Mechanism | The language alignment score for reflective planners is nearly double that of non-reflective versions; valid tool call rate increased by ~20%, and tool call rate yielding improvement increased by ~40%. |

### Key Findings

1. **LLMs are better suited for planning than direct protein sequence generation**: Direct SFT of Qwen2.5-7B only improves ProTrek from ~1 to 7 as data increases, remaining far from the ground-truth of 14.6; power-law extrapolation suggests approaching 12 would require ~$6\times10^8$ supervised samples.
2. **Epistemic uncertainty is higher for sequence tokens**: While aleatoric uncertainty for planning and sequence tokens is similar, epistemic uncertainty is systematically higher for sequence tokens, indicating insufficient evidence for residue-level decisions in the model.
3. **RL primarily enhances language alignment and retrieval**: Compared to SFT-only, online RL significantly improves ProTrek and Retrieval while reducing PPL/Repeat.
4. **Reflection is not merely a formatting stylistic choice**: Learning the `<think>/<plan>` format without actual reflection shows limited difference from a fixed workflow; the real efficacy lies in modifying strategies based on tool feedback and timely termination.

## Highlights & Insights

1. **Cross-domain Idea Transfer**: Successfully migrates the "plan + tool use + reflect" paradigm from NLP/AI Agent domains to protein design, demonstrating the cross-domain potential of agent frameworks.
2. **Bridging the Plan-Execute Gap**: Clearly identifies the "can talk but cannot do" issue of LLMs in protein design and provides an elegant solution via tool environments.
3. **Iterative Optimization vs. One-shot Generation**: Recognizes that protein design is not suited for single-step completion; multi-round feedback cycles align better with actual domain workflows.
4. **Supervised + RL Training Strategy**: Balances imitation learning and exploratory learning in agent training, serving as an effective paradigm for training complex agents.

## Limitations & Future Work

1. **Functional Site Design Tools remain lightweight**: Current tools are suitable for improving candidate quality within realistic computational budgets but do not guarantee ideal sequences, especially for strictly implementing high-specificity binding or catalytic geometries.
2. **Throughput-Quality Trade-off in Agentic Workflows**: ProtoCycle invokes structural/functional tools multiple times during planning and evaluation, resulting in higher computational costs than one-shot generators.
3. **Evaluation is still primarily based on computational metrics**: Language alignment, foldability, and keyword recovery are useful proxies but cannot replace wet-lab verification.
4. **Tool Feedback may introduce bias**: If ProTrek, Chai-1, or retrieval tools have insufficient coverage for certain protein families, the planner will inherit these biases.
5. **Complex Functional Design remains unresolved**: the paper aims to improve the text-guided design process rather than providing a *de novo* protein design solution that guarantees functionality directly.

## Related Work & Insights

1. **Protein LLMs (ProtGPT2, ESM, etc.)**: Methods that directly use LLMs for sequence generation; ProtoCycle switches to using LLMs as planners.
2. **AlphaFold**: A protein structure prediction tool that can serve as an evaluation component within the ProtoCycle tool environment.
3. **ReAct/OctoTools and other Agent Frameworks**: Agent framework concepts from NLP; ProtoCycle migrates these to protein design.
4. **RLHF/Online RL**: Training methods borrow the RLHF paradigm from NLP, using tool feedback in place of human feedback.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Introducing the agent paradigm to protein design is an interesting cross-domain effort; reflective iterative design aligns with domain intuition.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers Mol-Instructions, CAMEO generalization, tool efficiency, and reflection ablation, though wet-lab verification is absent.
- **Writing Quality**: ⭐⭐⭐⭐ — Problem definition (plan-execute gap) is clear, and the framework design is intuitive.
- **Value**: ⭐⭐⭐⭐ — Demonstrates the application potential of LLM agent frameworks in scientific discovery and provides a new paradigm for protein design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Protein Design with Dynamic Protein Vocabulary](../../NeurIPS2025/computational_biology/protein_design_with_dynamic_protein_vocabulary.md)
- [\[NeurIPS 2025\] Pharmacophore-Guided Generative Design of Novel Drug-Like Molecules](../../NeurIPS2025/computational_biology/pharmacophore-guided_generative_design_of_novel_drug-like_molecules.md)
- [\[ICML 2026\] From Feasible to Practical: Pareto-Optimal Synthesis Planning](../../ICML2026/computational_biology/from_feasible_to_practical_pareto-optimal_synthesis_planning.md)
- [\[ICLR 2026\] Protein Counterfactuals via Diffusion-Guided Latent Optimization](../../ICLR2026/computational_biology/protein_counterfactuals_via_diffusion-guided_latent_optimization.md)
- [\[ACL 2026\] BioTool: A Comprehensive Tool-Calling Dataset for Enhancing Biomedical Capabilities of Large Language Models](biotool_a_comprehensive_tool-calling_dataset_for_enhancing_biomedical_capabiliti.md)

</div>

<!-- RELATED:END -->
