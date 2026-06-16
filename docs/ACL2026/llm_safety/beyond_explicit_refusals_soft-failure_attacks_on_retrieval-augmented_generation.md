---
title: >-
  [Paper Note] Beyond Explicit Refusals: Soft-Failure Attacks on Retrieval-Augmented Generation
description: >-
  [ACL 2026][LLM Safety][Paper Note] Formalizes the "soft-failure" threat in RAG systems (fluent but uninformative responses) and proposes DEJA, a black-box evolutionary attack framework. By inducing models to exploit safety alignment mechanisms through adversarial documents, the framework achieves a SASR exceeding 79% with high stealth.
tags:
  - ACL 2026
  - LLM Safety
date: 2026-05-08
content_hash: 269e5ab2282ba17e
---
# Beyond Explicit Refusals: Soft-Failure Attacks on Retrieval-Augmented Generation

**Conference**: ACL 2026  
**arXiv**: [2604.18663](https://arxiv.org/abs/2604.18663)  
**Code**: None  
**Area**: AI Safety / RAG Security  
**Keywords**: RAG Attack, Soft-Failure, Adversarial Document, Evolutionary Optimization, Availability Attack

## TL;DR

Formalizes the "soft-failure" threat in RAG systems (fluent but uninformative responses) and proposes DEJA, a black-box evolutionary attack framework. By inducing models to exploit safety alignment mechanisms through adversarial documents, the framework achieves a SASR exceeding 79% with high stealth.

## Background & Motivation

**Background**: RAG systems rely on external corpora to improve factual accuracy, creating a critical dependency on corpus integrity. Existing attack research focuses primarily on knowledge poisoning (inducing incorrect outputs) and availability attacks (inducing explicit refusals).

**Limitations of Prior Work**: "Hard-failures" (e.g., explicit refusals) induced by existing jamming attacks are too obvious. They manifest as visible refusal responses and abnormal textual statistical features (e.g., high perplexity), making them easily detectable by anomaly-based defenses.

**Key Challenge**: A more stealthy threat exists—"soft-failure": the model produces fluent, coherent, but substantively vacuous answers. These do not trigger refusal keyword detection or perplexity anomalies but effectively negate the core value of RAG.

**Goal**: Formalize the soft-failure threat and develop an automated black-box attack framework to verify the severity of this threat.

**Key Insight**: Leveraging the safety alignment mechanisms of LLMs—alignment training makes models prone to "hedging" when facing uncertainty; attackers can manufacture artificial ambiguity to trigger this conservative behavior.

**Core Idea**: Decompose adversarial documents into a query anchor + retrieval hook + semantic payload, and use evolutionary optimization on the payload to induce low-utility but high-fluency responses.

## Method

### Overall Architecture

DEJA addresses the problem of how to make an injected document both retrievable and capable of quietly degrading the model's response from "useful" to "fluent but empty," without leaving traces like refusal keywords or perplexity anomalies. It decomposes the adversarial document into three concatenated segments: $d_{adv} = q \oplus h_{hook} \oplus p_{payload}$. The query anchor $q$ at the beginning restates the target question to ensure retrieval; the retrieval hook $h_{hook}$ in the middle is responsible for boosting the rank and semantically linking the anchor and payload; the semantic payload $p_{payload}$ is the functional component, evolved specifically to induce low-information responses. The pipeline first selects an attack strategy based on query features to initialize the payload, then iteratively refines the payload using an evolutionary algorithm until the response utility is sufficiently low, and finally assembles the three segments for injection.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400, 'subGraphTitleMargin': {'top': 8, 'bottom': 16}}}}%%
flowchart TD
    Q["Target Query q"] --> S["Context-Aware Strategy Selection<br/>Pick 1/6 blurring strategies by compatibility"]
    S --> INIT["Initialize Semantic Payload by Strategy"]
    subgraph EVO["Evolutionary Payload Optimization"]
        direction TB
        G["Semantic Operator Procreation<br/>Micro-mutation / Crossover / Innovation / Feedback Revision"] --> A["Answer Utility Score (AUS)<br/>Resolution + Specificity + Info Density"]
        A --> SEL["Fitness Selection<br/>Asymmetric distance towards low utility"]
        SEL -->|Utility target not met| G
    end
    INIT --> EVO
    EVO -->|Low utility reached| ASM["Assemble Adversarial Doc<br/>Anchor q ⊕ Hook ⊕ Payload"]
    ASM --> INJ["Inject into Corpus"]
```

### Key Designs

**1. Context-Aware Strategy Selection: Select a blurring strategy by query first to ensure semantic self-consistency**

Different types of questions suit different blurring techniques. If the hook and payload are mismatched, the assembled document will be fragmented and easily detected. DEJA pre-sets 6 attack strategies and first selects the best match for the query $s^* = \arg\max_{s_i} \text{Compatibility}(q, s_i)$, then constrains the semantic themes of both hook and payload with this strategy. This ensures the payload starts within the same semantic context as the query and hook, maintaining coherence throughout the evolutionary process.

**2. Answer Utility Score (AUS): Using continuous utility scores instead of binary success to optimize "soft-failure"**

Previous jamming attacks used binary criteria like keyword matching or F1. However, "soft-failure" is a gradual semantic degradation—the answer isn't a refusal or necessarily wrong, just empty. AUS quantifies information utility via an LLM-based scoring function across three dimensions: problem resolution (addressing the core question), factual specificity (concrete facts vs. vague generalizations), and information density (new information vs. redundant background). This continuous scale allows for fine-grained optimization targets.

**3. Evolutionary Payload Optimization: Searching for payloads in natural language space to lower utility while maintaining fluency**

Token-level perturbations, while effective at changing output, leave unnatural artifacts detectable by perplexity checks. DEJA optimizes the payload in natural language space using an evolutionary algorithm. The fitness function is defined as $\mathcal{F}(p) = \frac{1}{\mathcal{D}(u) + \epsilon}$, where $\mathcal{D}(u)$ is the asymmetric distance from the current utility to the target utility $\tau_{soft}$—it penalizes higher-than-target utility much more heavily. Each generation uses four LLM-driven operators: micro-mutation (local rewriting), semantic crossover (recombining payloads), innovation mutation (introducing new phrasing), and feedback revision (adjusting based on the previous score).

### Loss & Training

No model training is required. Optimization is performed in natural language space via an evolutionary algorithm. The attacker only requires black-box query access, with no need for model parameters or gradients. A single adversarial document is sufficient.

## Key Experimental Results

### Main Results

| Metric | DEJA | Prev. SOTA Attack |
|------|------|------------|
| Soft-Failure Attack Success Rate (SASR) | **>79%** | Significantly lower |
| Hard-Failure Rate | **<15%** | Higher (explicit refusal) |
| Perplexity Detection Evasion | ✓ Passed | ✗ Detected |
| Query Rewriting Robustness | ✓ Robust | - |
| Cross-Model Transferability | ✓ Transferred to closed-source | Limited |

### Ablation Study

| Component | Effect |
|------|------|
| W/o Strategy Selection | SASR decreased |
| W/o Retrieval Hook | Retrieval success dropped significantly |
| Random Payload vs. Evolved | Evolutionary optimization significantly higher SASR |
| Different LLM Families | Effective across model transfers |

### Key Findings

- Soft-failures are more dangerous than hard-failures: users may attribute uninformative answers to corpus limitations rather than an attack.
- DEJA exploits safety alignment mechanisms—weaponizing the model's "cautious" behavior.
- A single adversarial document is effective, making the injection threshold extremely low.
- Existing perplexity and refusal keyword detections fail entirely to identify soft-failures.

## Highlights & Insights

- The formal definition of the "soft-failure" concept fills a gap in RAG security research.
- Reveals the double-edged sword of safety alignment—alignment makes models more "cautious" and thus easier to induce into vacuity.
- The AUS scoring framework can be independently used for RAG response quality evaluation.
- The three-component document decomposition (anchor + hook + payload) provides a general methodology for adversarial document construction.

## Limitations & Future Work

- Evaluated only on English datasets.
- Evolutionary optimization requires multiple queries to the target system, potentially triggering rate limits.
- Defense methods (e.g., utility detection) are not fully explored.
- Attack effectiveness in multi-document retrieval scenarios requires further verification.
- Research aims to expose vulnerabilities to foster defense, not to provide attack tools.

## Related Work & Insights

- PoisonedRAG (Zou et al., 2025): Knowledge poisoning attacks.
- Jamming Attack (Shafran et al., 2025): Hard-failure/refusal attacks.
- LLM Evolutionary Optimization (Fernando et al., 2023; Guo et al., 2025): LLM-driven search.
- This paper warns the security community to focus on subtler threats that "look normal but are substantively useless."

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The soft-failure concept is novel, revealing unexpected vulnerabilities in safety alignment.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive analysis across configurations, benchmarks, stealth, and robustness.
- Writing Quality: ⭐⭐⭐⭐ Rigorous threat model definition and clear attack flow.
- Value: ⭐⭐⭐⭐⭐ Significant warning for RAG security research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Knowledge Poisoning Attacks on Medical Multi-Modal Retrieval-Augmented Generation](knowledge_poisoning_attacks_on_medical_multi-modal_retrieval-augmented_generatio.md)
- [\[ACL 2026\] Differentially Private Synthetic Text Generation for Retrieval-Augmented Generation (RAG)](differentially_private_synthetic_text_generation_for_retrieval-augmented_generat.md)
- [\[ACL 2026\] Retrievals Can Be Detrimental: Unveiling the Backdoor Vulnerability of Retrieval-Augmented Diffusion Models](retrievals_can_be_detrimental_unveiling_the_backdoor_vulnerability_of_retrieval-.md)
- [\[AAAI 2026\] Privacy-protected Retrieval-Augmented Generation for Knowledge Graph Question Answering](../../AAAI2026/llm_safety/privacy-protected_retrieval-augmented_generation_for_knowledge_graph_question_an.md)
- [\[NeurIPS 2025\] ImageSentinel: Protecting Visual Datasets from Unauthorized Retrieval-Augmented Image Generation](../../NeurIPS2025/llm_safety/imagesentinel_protecting_visual_datasets_from_unauthorized_retrieval-augmented_i.md)

</div>

<!-- RELATED:END -->
