---
title: >-
  [Paper Note] LeakDojo: Decoding the Leakage Threats of RAG Systems
description: >-
  [ACL 2026][LLM Safety][RAG leakage attacks] Proposed LeakDojo—the first configurable evaluation framework that modularly decouples RAG systems, attacks…
tags:
  - "ACL 2026"
  - "LLM Safety"
  - "RAG leakage attacks"
  - "prompt injection"
  - "red-teaming framework"
  - "instruction-following security"
  - "steganographic attacks"
date: 2026-05-08
content_hash: 5530d4a01e9dc203
---

# LeakDojo: Decoding the Leakage Threats of RAG Systems

**Conference**: ACL 2026 Findings  
**arXiv**: [2605.05818](https://arxiv.org/abs/2605.05818)  
**Code**: Open-sourced (GitHub link provided in paper)  
**Area**: RAG / Information Retrieval / LLM Security  
**Keywords**: RAG leakage attacks, prompt injection, red-teaming framework, instruction-following security, steganographic attacks

## TL;DR
Proposed LeakDojo—the first configurable evaluation framework that modularly decouples RAG systems, attacks, and defenses. It systematically quantifies RAG leakage risks across 6 attacks × 14 LLMs × 4 datasets × multiple enhancement modules, finding that "stronger instruction-following capability leads to higher leakage risk" and "RAG faithfulness is positively correlated with leakage risk."

## Background & Motivation

**Background**: Retrieval-Augmented Generation (RAG) has become the de facto standard for LLMs accessing private or high-value knowledge bases (medical, financial, legal, etc.). Modern RAG is no longer a simple "retrieval + generation" process but involves stacking various enhancement modules like rewriters, rerankers, and summarizers. Simultaneously, several studies (TGTB, PIDE, DGEA, RAG-Thief, PoR, IKEA, etc.) have demonstrated that prompt injection can force LLMs to output the original retrieved chunks, thereby "stealing" the knowledge base.

**Limitations of Prior Work**: (1) Benchmarks, attack budgets, and RAG configurations vary across these attacks, making **fair comparison impossible**; (2) most focus on relatively simple RAG systems, leaving their effectiveness on "modern RAG" with rewriters/rerankers/summarizers unclear; (3) whether the increasing instruction-following capabilities of new LLMs actually amplify leakage risks remains an open question.

**Key Challenge**: (a) **Usability vs. Security**—While enhancement modules (rewriter/summarizer) improve faithfulness, do they simultaneously amplify leakage? (b) **Capability vs. Security**—Are more powerful and "obedient" LLMs actually more fragile RAG backends? These trade-offs have not been quantitatively answered in existing research.

**Goal**: (1) Construct a unified framework for the fair evaluation of 6 existing attacks; (2) decouple the design space of RAG systems, attacks, and defenses to quantify the independent and joint impacts of each module; (3) extract actionable "leakage mechanism" laws (e.g., identifying bottlenecks: query generators vs. adversarial instructions).

**Key Insight**: The authors reformulate RAG leakage attacks as $Q^{adv}_i = A_i \oplus I$—where $A_i$ is the **anchor query** for the $i$-th round (determining if new chunks can be retrieved) and $I$ is the **adversarial instruction** (determining if the model can be induced to output the chunks). These two components are orthogonal and independently replaceable, enabling "modularly decoupled evaluation."

**Core Idea**: Build LeakDojo with "configurability" as the core design principle—where RAG systems, attacks, and defenses are plug-and-play modules. Systematic benchmarking is performed using four complementary metrics (CCL / SLT / ARC / CRR) to extract operational laws of leakage mechanisms. Based on the finding that "the bottleneck lies in the instruction," two new steganographic attacks (RankerSet, CodeClaim) were designed to verify the framework's extensibility.

## Method

### Overall Architecture

LeakDojo decomposes RAG leakage scenarios into three layers of configurable components:

1. **RAG System Side**: retriever (embedding model + retrieval strategy + top-$k$) + optional rewriter / reranker / summarizer + backend LLM (local vLLM or remote API);
2. **Attack Side**: query generator (static or interactive, responsible for generating anchor query $A_i$) + adversarial instruction $I$ (inducing the LLM to repeat context);
3. **Defense Side**: input-side intent detector (e.g., GPT-4.1-mini to judge query intent) + output-side content detector (blocking if ROUGE-L > threshold).

Each module is independent and controllable, allowing any combination to support four types of research: (a) fair benchmarking of existing attacks; (b) auditing leakage risks of online RAG; (c) comparing different defenses; (d) plug-and-play extension of new attacks.

Threat model: Black-box, interaction budget of $N=200$ rounds, attacker only knows high-level domain, goal is to maximize unique chunk leakage.

### Key Designs

1. **Orthogonal Decomposition of Attacks: $Q^{adv}_i = A_i \oplus I$**:

    - **Function**: Decomposes "whether a RAG leakage attack succeeds" into two independently quantifiable sub-problems—the ability to **retrieve** new chunks (query generator capability) and the ability to **induce the LLM to output them** (adversarial instruction capability).
    - **Mechanism**: During evaluation, ARC (unique retrieved / upper bound $k\times N$) measures $A_i$ effectiveness, and SLT (proportion of queries successfully triggering leakage, determined by ROUGE-L recall > 0.5) measures $I$ effectiveness. Overall leakage is measured by CCL (unique leaked chunks / upper bound). Meta-analysis shows $\text{CCL} \approx \text{ARC} \times \text{SLT}$, implying the impacts of the two components are **approximately multiplicative and mutually orthogonal**.
    - **Design Motivation**: This is the cognitive core of LeakDojo. Previously, attacks were compared as "black-box integrated methods." This work proves all attacks can be split into "query generation + instruction triggering," allowing precise identification of bottlenecks for each attack (e.g., TGTB has high ARC but low SLT on weak models, meaning the instruction is the bottleneck), and providing principled directions for "compositional enhancement."

2. **Modular RAG / Attack / Defense + 4 Complementary Evaluation Metrics**:

    - **Function**: Performs "ablation-style" evaluation under a unified pipeline to quantify the marginal impact of any component on leakage and usability.
    - **Mechanism**: On the RAG side, four gradient configurations T0/T1/T2/T3 are defined (vanilla → +reranker → +rewriter → full). On the attack side, six existing methods (TGTB, GEN-PIDE, DGEA, RAG-Thief, PoR, IKEA) are implemented as different instances of query generators. Usability is measured via Ragas faithfulness. Results are reported using four metrics: CCL (cumulative leakage rate), SLT (trigger rate), ARC (retrieval coverage), and CRR (verbatim recovery quality).
    - **Design Motivation**: A single metric masks mechanistic differences (e.g., IKEA has high SLT but low ARC, leading to low CCL). Only 4 metrics + modular ablation can locate "risk sources" and "room for improvement." This decoupling provides RAG developers with quantitative answers to whether switching LLMs or adding rewriters increases danger.

3. **New Steganographic Attacks based on Logical Masking: RankerSet & CodeClaim**:

    - **Function**: Serves as a case study for LeakDojo's extensibility, verifying that the finding "the bottleneck lies in the instruction" can be directly translated into attack design.
    - **Mechanism**: Previous instructions contained explicit keywords like "repeat" or "verbatim," which are easily intercepted by intent detectors (cutting CCL to <1%). RankerSet and CodeClaim embed the intent of "repeating chunks" into **logical reasoning chains** (e.g., asking the LLM to treat chunks as input for a sorting task or code declaration), making the instruction appear as a legitimate task request.
    - **Design Motivation**: Directly verifies that LeakDojo is an evolvable framework capable of incubating new attacks. It echoes the key finding—shifting engineering effort from "query diversification" to "instruction obfuscation" is the high-ROI direction. With intent detectors active, CodeClaim + GEN-PIDE still achieved 59.6% CCL on FIQA / DeepSeek-V3, 8x higher than default instructions without defense (7.3%).

### Loss & Training

LeakDojo is an evaluation framework rather than a training method, thus has no loss function. For reproducibility, all LLMs use greedy decoding with an attack budget of $N=200$. GPT-4.1-mini is used for the rewriter/summarizer; the retriever uses bge-large-en-v1.5 + MMR + bge-reranker-large.

## Key Experimental Results

### Main Results

6 main LLMs × 4 datasets × 3 RAG variants (vanilla / +reranker / +rewriter+reranker) were evaluated across 6 attacks using 4 metrics. The following table selects CCL (%) for typical scenarios:

| Attack | Gemini-3-flash · ENRON | GPT-5.1 · ENRON | DeepSeek-V3 · ENRON | Qwen-3-8B · ENRON |
|------|------------------------|-----------------|----------------------|-------------------|
| TGTB | 72.3 | 69.5 | 0.1 | 10.4 |
| GEN-PIDE | 69.4 | 36.8 | 38.8 | 35.4 |
| DGEA | 11.2 | 16.5 | 8.1 | 12.4 |
| RAG-Thief | 44.4 | 3.3 | **64.4** | 64.1 |
| **PoR** | **88.3** | **83.2** | 6.8 | **70.2** |
| IKEA | 23.2 | 15.4 | 1.0 | 6.0 |

**Key Findings**: (a) No single attack is strongest across all LLMs—PoR wins on Gemini (88.3%) but collapses on DeepSeek-V3 (6.8%); (b) ARC remains nearly constant on the same dataset, while SLT fluctuates wildly across models, indicating **the bottleneck is the instruction, not the query**; (c) the Pearson correlation between IFEval scores of 14 LLMs and SLT is $r=0.578$ ($p=0.039$), meaning **stronger instruction-following leads to higher leakage risk**.

### Ablation Study

**Impact analysis of RAG modules × datasets × attacks** (correlation between CCL and faithfulness):

| Configuration | ARC Change | CCL Change | Faithfulness | Interpretation |
|------|----------|----------|--------------|------|
| T0 → T1 (+reranker) | Almost none | Almost none ($r=-0.186$, $p=0.117$) | Slight increase | Reranker is neutral to leakage; safe to use |
| T1 → T2 (+rewriter) | **Mean ↑, Var ↓** | Overall ↑ | Increase | Rewriter elevates weak attacks to stable highs, lowering the attack threshold |
| T2 → T3 (+summarizer) | Slight decrease | Significant ↓ | Significant ↓ | Summarizer breaks context integrity, harming usability while reducing leakage |

**Faithfulness ↔ CCL Correlation** (FIQA $r=0.83, p=4.88e{-7}$; SCIFACT $r=0.57$; NFCORPUS $r=0.51$)—The more "faithful" the RAG, the easier it is to leak.

**Verification of Orthogonal Decomposition**: $\text{SLT} \times \text{ARC}$ matches the CCL curve almost perfectly across all 6 attacks (Figure 5), proving the components act approximately independently.

**Defense + New Attack Comparison** (FIQA · DeepSeek-V3 · T2 · CCL %):

| Attack | Default | + intent det. | RankerSet + Di | CodeClaim + Di | CodeClaim + DiDo |
|------|---------|---------------|----------------|----------------|------------------|
| TGTB | 7.3 | 0.6 | 50.9 | **59.0** | 25.9 |
| GEN-PIDE | 57.5 | 0.2 | 47.8 | 59.6 | 26.5 |
| PoR | 48.7 | 0.2 | 51.7 | 57.9 | 26.9 |

CodeClaim maintained a CCL of 50–60% even with intent detection active—higher than default instructions **without defense**, verifying that the "instruction bottleneck" insight translates directly into efficient attacks.

### Key Findings
- **$\text{CCL} \approx \text{ARC} \times \text{SLT}$** is significant as it refutes the direction of years of attack improvements (mostly focusing on query generators); the real breakthrough needed is instruction-side stealth.
- **Stronger models are more dangerous**: The positive correlation between instruction-following (IFEval) and SLT implies a trade-off between capability and RAG security. Security research must keep pace with the capability curve.
- **Rewriters are a security risk**: Rewriters raise the ARC of weak queries, essentially lowering the "attack threshold," requiring focused protection during deployment.
- **Summarizers trade usability for security**: They cut CCL but hurt faithfulness; they can be considered a last-resort defense in security-sensitive scenarios.
- **Rerankers are neutral**: Correlation with CCL is near zero, making them safe to use.
- **Existing attacks lack stealth**: A simple intent detector cuts CCL to <1% for most, but the authors' logical masking attacks bypass this easily, suggesting industrial deployments cannot rely on keyword-level defenses.

## Highlights & Insights
- **The "orthogonal decomposition" $Q^{adv} = A \oplus I$ is the key** to turning chaotic red-teaming into quantifiable mechanism research. This simple formalization explains why specific attacks fail on certain models and can be transferred to evaluating any "induce + trigger" two-stage attack like agent attacks or jailbreaks.
- **The negative correlation between "Capability ↔ Security" is quantified** (IFEval vs. SLT, $r=0.578$). While it was intuitive that "obedient models are easier to deceive," this provides the first statistical evidence, offering direct guidance for model selection and safety alignment.
- **The "faithfulness ↔ leakage" positive correlation** highlights a counter-intuitive principle: the more you strive for a RAG to faithfully repeat retrieved content, the more it will "faithfully" repeat it under attack. Security mechanisms must explicitly distinguish between "legitimate repetition needs" and "attacker-induced repetition."
- **Case studies use the framework to inform attack design**: By designing RankerSet/CodeClaim based on their own analysis, the authors create a closed loop of utility—benchmarks should be tools for discovery, not just endpoints.

## Limitations & Future Work
- **Attack costs not quantified**: Metrics like CCL use a fixed budget $N=200$ without considering differences in tokens, latency, or inference costs. Industrial metrics need a "cost-effectiveness" dimension.
- **Limited RAG design coverage**: Only rewriters, rerankers, and summarizers were studied. More complex structures like Routing, self-reflection, agentic-RAG, or GraphRAG might have new leakage paths.
- **Single language**: All datasets are English. Multilingual RAG indexing biases and cross-lingual alignment could change leakage dynamics.
- **Simple defenses**: Only basic intent/content detectors were tested. Dedicated prompt-injection defenses like SecAlign or Spotlighting were not included; RankerSet / CodeClaim should be tested against these stronger baselines.
- **Horizon of the budget $N=200$**: While Figure 7 justifies the budget for distinguishing curves, attackers might employ low-and-slow tactics over longer periods.
- **Improvement ideas**: (a) Generalize decomposition to $Q^{adv}=f(A, I, \text{persona})$ to include system prompt analysis; (b) use LeakDojo to train a "Red-Team LLM" to search for optimal ARC×SLT combinations; (c) explore training objectives that decouple "faithfulness-leakage".

## Related Work & Insights
- **vs. RAG-Thief / PoR / IKEA**: These proposed new query generators; LeakDojo re-evaluates them, revealing there is "no universal strongest" and the **real bottleneck is the instruction**, correcting the community's research direction.
- **vs. HarmBench / RACCOON / AgentDojo**: While those benchmark jailbreaks/agent attacks, LeakDojo addresses RAG-specific challenges like multi-round stateful attacks and modular RAG structures.
- **vs. Ragas**: Ragas focuses on usability; LeakDojo measures both usability and security, quantifying the "faithfulness ↔ leakage" correlation for the first time.
- **vs. General prompt injection defenses (e.g., SecAlign)**: These focus on the injection itself; LeakDojo shows RAG-specific security requires intercepting both the query generator and the instruction trigger.

## Rating
- Novelty: ⭐⭐⭐⭐ The framework is a compositional innovation, but the three sets of findings (decomposition, IFEval-leakage, faithfulness-leakage) provide outstanding insights.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ "Textbook-grade" baselines covering 6 attacks, 14 LLMs, 4 datasets, and multiple configurations/defenses.
- Writing Quality: ⭐⭐⭐⭐ Clear three-layer architecture; self-consistent case studies, though some findings require careful reading of large tables.
- Value: ⭐⭐⭐⭐⭐ The open-source framework lowers the barrier for future research; conclusions like "stronger is more dangerous" have direct security implications for industrial RAG.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Do Multimodal RAG Systems Leak Data? A Comprehensive Evaluation of Membership Inference and Image Caption Retrieval Attacks](do_multimodal_rag_systems_leak_data_a_comprehensive_evaluation_of_membership_inf.md)
- [\[ACL 2026\] A Survey on the Safety and Security Threats of Computer-Using Agents: JARVIS or Ultron?](a_survey_on_the_safety_and_security_threats_of_computer-using_agents_jarvis_or_u.md)
- [\[ACL 2026\] Evaluating Answer Leakage Robustness of LLM Tutors against Adversarial Student Attacks](evaluating_answer_leakage_robustness_of_llm_tutors_against_adversarial_student_a.md)
- [\[ACL 2026\] Detecting RAG Extraction Attack via Dual-Path Runtime Integrity Game](detecting_rag_extraction_attack_via_dual-path_runtime_integrity_game.md)
- [\[ACL 2026\] Differentially Private Synthetic Text Generation for Retrieval-Augmented Generation (RAG)](differentially_private_synthetic_text_generation_for_retrieval-augmented_generat.md)

</div>

<!-- RELATED:END -->
