# Evaluation Metrics Specification

[← Back to Main Documentation](../README.md)

This section defines the metrics used to assess the performance of the RAG classification module and its impact on pipeline efficiency. The metrics are categorized into **task-specific performance measures**, **composite RAG score**, and **time improvement analysis**.  

---

### **Task-Specific Performance Metrics**  
To evaluate the system’s capabilities across diverse query types, distinct metrics were selected based on task requirements:  

1. **Accuracy**  
   - **Definition**: The proportion of correct predictions relative to the total number of evaluated samples.  
     $$\text{Accuracy} = \frac{\text{Number of Correct Predictions}}{\text{Total Predictions}} \times 100\%$$
   - **Use Cases**:  
     - **Commonsense Reasoning**: Tasks requiring intuitive understanding (e.g., "Is water wet?").  
     - **Fact Checking**: Verification of factual statements (e.g., "Paris is the capital of France").  
     - **Medical QA**: Queries demanding domain-specific medical knowledge (e.g., "What causes hypertension?").  
   - **Rationale**: Accuracy is suitable for tasks with unambiguous ground-truth answers, where responses are either definitively correct or incorrect.  

2. **Token-Level F1 Score**  
   - **Definition**: Harmonic mean of precision and recall at the token level, measuring overlap between predicted and reference answers.  
     $$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$  
     - **Precision**: Ratio of correctly predicted tokens to total predicted tokens.  
     - **Recall**: Ratio of correctly predicted tokens to total reference tokens.  
   - **Use Cases**:  
     - **Open-Domain QA**: Broad questions where answers may vary in phrasing (e.g., "Explain quantum computing").  
     - **Multihop QA**: Complex queries requiring synthesis of multiple sources (e.g., "How did Einstein’s theories influence WWII?").  
   - **Rationale**: F1 accommodates partial correctness, critical for tasks with flexible or multi-part answers.  

3. **Exact Match (EM) Score**  
   - **Definition**: Strict binary metric where the model’s output must exactly match the reference answer, including punctuation and formatting.  
   - **Use Cases**: Same as token-level F1 (Open-Domain QA and Multihop QA).  
   - **Rationale**: EM ensures precision for tasks where answer formatting is critical (e.g., dates, names).  

---


#### **Efficiency and Time Improvement Metrics**  
The classifier’s impact on pipeline latency was quantified using:  
- **Latency Reduction**:  
  - **Definition**: The percentage decrease in average response time for queries flagged as "no retrieval needed" compared to full RAG processing.  
    $$\text{Reduction} = \left(1 - \frac{\text{Avg. Latency (No Retrieval)}}{\text{Avg. Latency (Full RAG)}}\right) \times 100\%$$  
  - **Measurement Workflow**:  
    1. Measure response time for a query batch processed with retrieval.  
    2. Measure response time for the same batch, bypassing retrieval for queries classified as non-retrieval.  
    3. Compute the relative difference.  

---
## **Results**

#### Ollama model
| Method             |   top_k |   context_len |   Bert Latency |   DB Latency |   LLM Latency |   Avg Latency |   Num Queries |        F1 |   EM |
|:-------------------|--------:|--------------:|---------------:|-------------:|--------------:|--------------:|--------------:|----------:|-----:|
| Query Classifier   |      10 |          1000 |           0.55 |         0.25 |        339.54 |       8.5085  |            40 | 0.0609151 |  0.6 |
| Without Classifier |      10 |          1000 |           0    |         0.87 |        382.74 |       9.59025 |            40 | 0.0661646 |  0.5 |
Time Reduction: 11.28%

| Method             |   top_k |   context_len |   Bert Latency |   DB Latency |   LLM Latency |   Avg Latency |   Num Queries |        F1 |    EM |
|:-------------------|--------:|--------------:|---------------:|-------------:|--------------:|--------------:|--------------:|----------:|------:|
| Query Classifier   |      10 |         10000 |           0.57 |         0.27 |        509.36 |       12.755  |            40 | 0.0920069 | 0.625 |
| Without Classifier |      10 |         10000 |           0    |         0.97 |        755.45 |       18.9105 |            40 | 0.0896653 | 0.65  |
Time Reduction: 32.55%

#### Mistral Model
| Method             |   top_k |   context_len |   Bert Latency |   DB Latency |   LLM Latency |   Avg Latency |   Num Queries |       F1 |   EM |
|:-------------------|--------:|--------------:|---------------:|-------------:|--------------:|--------------:|--------------:|---------:|-----:|
| Query Classifier   |      10 |         10000 |           0.67 |         0.38 |         71.36 |       1.81025 |            40 | 0.127112 | 0.65 |
| Without Classifier |      10 |         10000 |           0    |         1    |         57.15 |       1.45375 |            40 | 0.126171 | 0.6  |
Time Reduction: -24.52%
In this case API of HuggingFace works much faster than my PC, therefore LLM responds quickly. In this case classifier takes 30-50% of LLM time to answer. Therefore, results are poor.

| Method             |   top_k |   context_len |   Bert Latency |   DB Latency |   LLM Latency |   Avg Latency |   Num Queries |       F1 |    EM |
|:-------------------|--------:|--------------:|---------------:|-------------:|--------------:|--------------:|--------------:|---------:|------:|
| Query Classifier   |      50 |        100000 |           0.65 |         0.39 |         76.61 |       1.94125 |            40 | 0.117095 | 0.625 |
| Without Classifier |      50 |        100000 |           0    |         0.99 |         84.06 |       2.12625 |            40 | 0.154823 | 0.575 |
Time Reduction: 8.7%
For larger contexts, the API tends to operate more slowly; as a result, we can observe improvements when using a classifier.
## **Summary**  
The integration of a classifier in the RAG (Retrieval-Augmented Generation) pipeline demonstrates potential to enhance performance when both the LLM (Large Language Model) and retrieval components are time-consuming. However, in scenarios where Retrieval + LLM response fast, the utility of the classifier diminishes. Specifically, the classifier consumes 30-50% of the LLM's response time, leading to suboptimal results in this context. It is expected that incorporating retrieval would further enhance the speedup provided by the classifier, as it would better balance the overall computational load within the pipeline. Thus, while the classifier shows promise, its benefits are more pronounced in setups where both retrieval and LLM processing are slower.
---  

