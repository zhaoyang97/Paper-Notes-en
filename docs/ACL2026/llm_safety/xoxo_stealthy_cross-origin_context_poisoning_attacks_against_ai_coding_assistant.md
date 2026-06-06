---
title: >-
  [Paper Note] XOXO: Stealthy Cross-Origin Context Poisoning Attacks against AI Coding Assistants
description: >-
  [ACL 2026][LLM Safety][Adversarial Attacks] Discloses design vulnerabilities in AI coding assistants' automatic context collection and proposes Cross-Origin Context Poisoning (XOXO) attacks: poisoning shared repositories…
tags:
  - "ACL 2026"
  - "LLM Safety"
  - "Adversarial Attacks"
  - "AI Coding Assistants"
  - "Context Poisoning"
  - "Semantic-Preserving Transformations"
  - "Code Security"
date: 2026-05-08
content_hash: bb74b9d19fa6464b
---

# XOXO: Stealthy Cross-Origin Context Poisoning Attacks against AI Coding Assistants

**Conference**: ACL 2026  
**arXiv**: [2503.14281](https://arxiv.org/abs/2503.14281)  
**Code**: [https://github.com/adamstorek/cross-origin-context-poisoning](https://github.com/adamstorek/cross-origin-context-poisoning)  
**Area**: Robotics  
**Keywords**: Adversarial Attacks, AI Coding Assistants, Context Poisoning, Semantic-Preserving Transformations, Code Security

## TL;DR

Discloses design vulnerabilities in AI coding assistants' automatic context collection and proposes Cross-Origin Context Poisoning (XOXO) attacks: poisoning shared repositories through semantic-preserving code transformations (e.g., variable renaming) to cause assistants like GitHub Copilot to generate vulnerable code unknowingly. The average attack success rate across 8 SOTA models reaches 73.20%.

## Background & Motivation

**Background**: AI coding assistants (e.g., GitHub Copilot) have become the second most popular AI tools after chat AI. They enhance the code generation capabilities of LLMs by automatically gathering context code snippets from the project.

**Limitations of Prior Work**: These assistants possess severe security design flaws during context collection: (1) They automatically scrape code snippets from the entire project as context without distinguishing the trustworthiness of the source; (2) They mix code from different sources into a single prompt sent to the LLM, preventing developers from viewing, restricting, or logging the collected context; (3) The authors investigated 7 mainstream coding assistants and found that all employ automatic context collection without source differentiation.

**Key Challenge**: While automatic context collection improves generation quality, it simultaneously creates a new attack surface—attackers can induce coding assistants to generate buggy or vulnerable code by performing semantic-preserving modifications (leaving code functionality unchanged) on shared code. Such attacks are extremely difficult to detect during code review because the modifications themselves are legitimate and functional.

**Goal**: (1) Define the XOXO attack paradigm; (2) Propose an algorithm for automatically discovering effective attack transformations; (3) Verify the attacks on real-world coding assistants.

**Key Insight**: The authors discovered that LLMs produce different outputs for code inputs that are semantically equivalent but syntactically different—this reveals a fundamental flaw in current LLM architectures when processing semantically equivalent code.

**Core Idea**: Leveraging the monotonicity of LLM confidence (combining multiple transformations that reduce confidence further lowers overall confidence), a greedy Cayley graph search algorithm is designed to efficiently find semantic-preserving transformation combinations that induce erroneous outputs.

## Method

### Overall Architecture

The XOXO attack workflow: An attacker performs semantic-preserving transformations (e.g., variable renaming) on code within a shared repository → The transformed code propagates to the victim's project via version control → When the victim uses a coding assistant, the assistant automatically collects context containing the poisoned code → The LLM generates buggy or vulnerable code based on the poisoned context. The GCGS algorithm automatically searches for effective transformation combinations.

### Key Designs

1.  **Cross-Origin Context Poisoning (XOXO) Attack Model**:
    - **Function**: Defines the threat model and attack surface.
    - **Mechanism**: Exploits three characteristics of coding assistants: (a) Automatic context collection does not distinguish sources; (b) The use of greedy decoding or low-temperature sampling (e.g., Copilot temperature 0.1) makes attack effects reproducible; (c) Prompt templates and sampling parameters can be reverse-engineered through network traffic analysis. Attackers only need code submission permissions to provide semantic-preserving but poisoning transformations.
    - **Design Motivation**: The threat model is highly realistic—malicious contributors are common in open-source projects (supply chain attacks occur frequently), and modifications like variable renaming rarely raise suspicion during code reviews.

2.  **Confidence Monotonicity and Greedy Cayley Graph Search (GCGS)**:
    - **Function**: Automatically and efficiently discovers combinations of semantic-preserving transformations that induce the LLM to generate incorrect code.
    - **Mechanism**: Defines a generating set $G$ of atomic transformations (variable renaming, statement reordering, etc.) and constructs a Cayley graph $\mathcal{T}$ representing the search space of all transformation combinations. Key Finding—**Confidence Monotonicity**: If two transformations $g_i, g_j$ individually reduce model confidence, their combination $g_i \cdot g_j$ tends to reduce confidence further. A greedy search is performed using this property: first, explore all atomic transformations at a shallow level and record confidence changes, then greedily combine transformations in ascending order of confidence, traversing the path of decreasing confidence until the model outputs an error.
    - **Design Motivation**: The space of transformation combinations is exponential, making exhaustive search infeasible. Confidence monotonicity provides a reliable search direction—following the path of decreasing confidence highly likely leads to transformations inducing incorrect output. t-tests verified this with $p \text{ value} < 1.7 \times 10^{-10}$.

3.  **End-to-End GitHub Copilot Attack Verification**:
    - **Function**: Verifies the actual threat of the attack on real production-grade coding assistants.
    - **Mechanism**: In a Django web application, an attacker renames the variable `USE_RAW_QUERIES` to `RAW_QUERIES` (semantics remain identical). When the victim implements a search function, Copilot automatically collects the context containing the renamed variable and generates SQL query code using unsanitized user input—resulting in a SQL injection vulnerability. The attack succeeded consistently across multiple Copilot sessions.
    - **Design Motivation**: Demonstrates the real-world harm of the attack—it bypasses Copilot's security guardrails, and the attack remains effective across file boundaries.

### Loss & Training

GCGS is a search algorithm rather than a training method. Length-normalized log-likelihood is used as the confidence score: 
$$\alpha(c) = \frac{1}{|y|} \sum_{t=1}^{|y|} \log p(y_t | c, y_{<t})$$
The search iterates through shallow exploration and deep greedy combination phases within a query budget.

## Key Experimental Results

### Main Results

Bug injection attack success rates (HumanEval+ and MBPP+):

| Model | HumanEval+ ASR | MBPP+ ASR | CWEval Vulnerability Injection Rate |
| :--- | :--- | :--- | :--- |
| Claude 3.5 Sonnet v2 | 92.00% | 98.42% | 40.00% |
| GPT 4.1 | 81.82% | 40.69% | 50.00% |
| DeepSeek Coder 33B | 85.69% | 96.41% | 63.97% |
| Llama 3.1 8B | 97.11% | 99.88% | 54.00% |

The average attack success rate across 8 SOTA models is 83.67% (bugs) and 52.26% (vulnerabilities).

### Ablation Study

| Configuration | Key Metrics | Description |
| :--- | :--- | :--- |
| XOXO (Unoptimized Search) | ASR 73.20% | Random transformation combinations |
| XOXO + GCGS | ASR 83.67% | Confidence-guided search consistently outperforms unoptimized search |
| Atomic Transformations Only | Partial Success | A single transformation is sometimes sufficient |
| Cross-File Attack | Still Effective | Attack remains successful after moving variables to `models.py` and importing them |

### Key Findings

- Confidence monotonicity holds across all tested models and datasets ($p < 1.7 \times 10^{-10}$), indicating a universal property of LLMs.
- The attack triggered 17 different Common Weakness Enumeration (CWE) types, proving its broad impact.
- Even state-of-the-art models with security alignment (Claude 3.5, GPT 4.1) are vulnerable.
- All 7 mainstream coding assistants investigated share the same architectural vulnerability—failure to distinguish context sources.

## Highlights & Insights

- **Extremely High Stealth**: Semantic-preserving variable renaming is nearly impossible to detect in code reviews, contrasting sharply with traditional prompt injection (which requires obviously malicious instructions). This makes the attack surface more realistic and dangerous.
- **Discovery of Confidence Monotonicity**: This is highly valuable—it serves not only as the technical foundation for the attack but also reveals the excessive reliance of LLMs on surface-level code forms rather than semantics, representing a fundamental flaw in current architectures.
- From a defensive perspective, this work directly points toward a design improvement: coding assistants should differentiate the trustworthiness of context sources rather than indiscriminately mixing all code.

## Limitations & Future Work

- The attack assumes the attacker has code submission permissions, which is realistic in open-source projects but more difficult in strictly controlled private projects.
- GCGS requires multiple queries to the target model to search for effective transformations, which can be costly for commercial APIs.
- Defense strategies were not discussed in depth—how to distinguish context source trustworthiness without reducing generation quality remains an open problem.
- Currently, only Python code has been tested; the effectiveness of attacks on other programming languages remains unverified.

## Related Work & Insights

- **vs. Prompt Injection**: Traditional prompt injection requires inserting obvious malicious instructions into the input, which is easily detected. XOXO achieves the attack via semantic-preserving transformations where the modification itself is entirely legal, representing a different level of stealth.
- **vs. Code Classification Attacks**: Previous semantic-preserving attacks primarily targeted code classification tasks (defect detection, clone detection) and required class confidence feedback. XOXO is the first to extend such attacks to code generation tasks.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Defines an entirely new attack paradigm, XOXO; the discovery of confidence monotonicity has theoretical value.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 8 models, multiple benchmarks, real Copilot attack verification, and statistical significance tests.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Attack motivation and threat models are described clearly, and real attack cases are highly persuasive.
- **Value**: ⭐⭐⭐⭐⭐ Reveals significant security risks in AI coding assistants with direct industrial impact; responsibly disclosed to vendors.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Membership Inference Attacks on In-Context Learning Recommendation](membership_inference_attacks_on_llm-based_recommender_systems.md)
- [\[ACL 2026\] Knowledge Poisoning Attacks on Medical Multi-Modal Retrieval-Augmented Generation](knowledge_poisoning_attacks_on_medical_multi-modal_retrieval-augmented_generatio.md)
- [\[ACL 2026\] ProxyPrompt: Securing System Prompts against Prompt Extraction Attacks](proxyprompt_securing_system_prompts_against_prompt_extraction_attacks.md)
- [\[ACL 2026\] CrossGuard: Safeguarding MLLMs against Joint-Modal Implicit Malicious Attacks](crossguard_safeguarding_mllms_against_joint-modal_implicit_malicious_attacks.md)
- [\[ACL 2026\] Evaluating Answer Leakage Robustness of LLM Tutors against Adversarial Student Attacks](evaluating_answer_leakage_robustness_of_llm_tutors_against_adversarial_student_a.md)

</div>

<!-- RELATED:END -->
