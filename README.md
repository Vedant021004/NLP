
# AI Voice Support Agent

> An end-to-end AI customer support system combining speech recognition, fine-tuned BERT intent classification, LangGraph orchestration, RAG-based knowledge retrieval, and LLM-powered response generation.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Transformers](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow)
![LangGraph](https://img.shields.io/badge/LangGraph-Agentic%20AI-purple)
![RAG](https://img.shields.io/badge/RAG-Vector%20Search-green)
![Whisper](https://img.shields.io/badge/Whisper-Speech%20Recognition-orange)

---

## Overview

The **AI Voice Support Agent** is a modular customer-support system designed to process voice queries, identify user intent, route requests to specialized workflows, retrieve relevant company knowledge, and generate context-aware responses.

The system combines specialized models with agentic orchestration rather than relying on a single LLM for every task.

### Core Pipeline

```text
User Voice
    │
    ▼
Whisper
Speech → Text
    │
    ▼
Fine-Tuned BERT
Intent Classification
    │
    ▼
LangGraph
Workflow Routing
    │
    ▼
Specialized Agent
    │
    ▼
RAG
Knowledge Retrieval
    │
    ▼
Embedding Model
    │
    ▼
Vector Database
    │
    ▼
Relevant Context
    │
    ▼
LLM
Reasoning + Response Generation
    │
    ▼
Text-to-Speech
    │
    ▼
Voice Response
````

---

## Problem

Traditional customer-support systems often rely on:

* Rule-based routing
* Static FAQ systems
* Manual ticket classification
* Large LLM calls for simple classification tasks
* Limited access to private or organization-specific knowledge

This project addresses these limitations by assigning each stage of the workflow to a component optimized for that task.

---

## Solution

The system separates **classification, orchestration, retrieval, and generation**.

| Component       | Responsibility                    |
| --------------- | --------------------------------- |
| Whisper         | Speech-to-text conversion         |
| Fine-Tuned BERT | Customer intent classification    |
| LangGraph       | Workflow and agent orchestration  |
| Embedding Model | Semantic representation           |
| Vector Database | Similarity search                 |
| RAG             | Knowledge retrieval               |
| LLM             | Reasoning and response generation |
| TTS             | Text-to-speech conversion         |

This architecture allows specialized models to handle deterministic tasks while reserving LLM inference for reasoning and generation.

---

# Architecture

```text
                         ┌──────────────┐
                         │     User     │
                         │    Voice     │
                         └──────┬───────┘
                                │
                                ▼
                         ┌──────────────┐
                         │   Whisper    │
                         │ Speech→Text  │
                         └──────┬───────┘
                                │
                                ▼
                    ┌────────────────────────┐
                    │    Fine-Tuned BERT     │
                    │                        │
                    │ Intent Classification  │
                    └───────────┬────────────┘
                                │
                       Intent + Confidence
                                │
                                ▼
                         ┌──────────────┐
                         │  LangGraph   │
                         │    Router    │
                         └──────┬───────┘
                                │
               ┌────────────────┼────────────────┐
               ▼                ▼                ▼
        ┌────────────┐   ┌────────────┐   ┌────────────┐
        │  Billing   │   │ Technical  │   │  Shipping  │
        │   Agent    │   │   Agent    │   │   Agent    │
        └─────┬──────┘   └─────┬──────┘   └─────┬──────┘
              │                │                │
              └────────────────┼────────────────┘
                               ▼
                        ┌──────────────┐
                        │     RAG      │
                        │  Retrieval   │
                        └──────┬───────┘
                               │
                               ▼
                       ┌───────────────┐
                       │   Embedding   │
                       │     Model     │
                       └───────┬───────┘
                               │
                               ▼
                       ┌───────────────┐
                       │ Vector Store  │
                       │ FAISS/Chroma  │
                       └───────┬───────┘
                               │
                               ▼
                       Retrieved Context
                               │
                               ▼
                        ┌──────────────┐
                        │     LLM      │
                        │ Generation   │
                        └──────┬───────┘
                               │
                               ▼
                        ┌──────────────┐
                        │     TTS      │
                        └──────┬───────┘
                               │
                               ▼
                         Voice Response
```

---

# Key Components

## 1. Speech Recognition

The system begins with an audio input from the user.

```text
Audio
  ↓
Whisper
  ↓
Text
```

Example:

```text
"My payment was deducted twice and I need a refund."
```

The resulting text becomes the input to the NLP pipeline.

---

## 2. Fine-Tuned BERT Intent Classifier

A pretrained BERT model is fine-tuned on a domain-specific customer-support dataset.

### Supported intents

```text
PAYMENT
REFUND
CANCELLATION
SHIPPING
ACCOUNT
TECHNICAL
```

Example:

```text
Input:
"My card payment failed."

Output:
{
    "intent": "PAYMENT",
    "confidence": 0.96
}
```

### BERT Pipeline

```text
Text
 ↓
BERT Tokenizer
 ↓
WordPiece Tokens
 ↓
Token IDs
 ↓
BERT Transformer Encoder
 ↓
Contextual Representation
 ↓
Classification Head
 ↓
Intent
```

The classifier is designed to handle high-volume, structured intent detection before the request reaches the generative layer.

---

# 3. LangGraph Orchestration

The BERT prediction becomes part of the application state.

```python
{
    "user_query": "...",
    "intent": "PAYMENT",
    "confidence": 0.96
}
```

LangGraph uses this state to determine the appropriate workflow.

```text
PAYMENT
   ↓
Billing Agent

SHIPPING
   ↓
Shipping Agent

TECHNICAL
   ↓
Technical Agent
```

LangGraph is responsible for workflow orchestration, state transitions, conditional routing, and agent execution.

---

# 4. Retrieval-Augmented Generation

The agent requires access to organization-specific information that may not be present in the base LLM.

The RAG pipeline provides this external knowledge.

```text
Company Documents
       ↓
    Chunking
       ↓
Embedding Model
       ↓
Vector Store
```

At query time:

```text
User Query
    ↓
Query Embedding
    ↓
Similarity Search
    ↓
Relevant Chunks
    ↓
LLM
```

Example knowledge sources:

```text
refund_policy.pdf
payment_policy.pdf
shipping_policy.pdf
faq.pdf
```

---

# 5. Embeddings and Semantic Search

Document chunks and user queries are converted into vector representations using an embedding model.

Possible embedding models include:

* Sentence Transformers
* Hugging Face embedding models
* Ollama embedding models
* Other dedicated embedding APIs

Example:

```python
HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

The resulting vectors are stored in a vector database such as:

```text
FAISS
ChromaDB
Qdrant
Pinecone
```

Similarity search retrieves the chunks that are semantically closest to the user's query.

---

# 6. LLM Response Generation

The LLM receives:

```text
System Instructions
+
User Query
+
Retrieved Context
```

Example:

```text
User:
"My payment was deducted twice."

Retrieved Context:
"Duplicate payments can be refunded after verification."

LLM:
"Your duplicate payment can be refunded after verification..."
```

The LLM is therefore responsible primarily for **reasoning and natural-language generation**, rather than basic intent classification.

---

# 7. Text-to-Speech

The generated response can be converted back into speech:

```text
LLM Response
     ↓
Text-to-Speech
     ↓
Audio
     ↓
User
```

This enables a complete voice-to-voice interaction.

---

# Example End-to-End Request

### User Input

> "My payment was deducted twice and I want a refund."

### Whisper

```text
"My payment was deducted twice and I want a refund."
```

### BERT

```json
{
  "intent": "PAYMENT",
  "confidence": 0.96
}
```

### LangGraph

```text
PAYMENT
   ↓
Billing Agent
```

### RAG

Retrieves relevant information:

```text
"Duplicate payments can be refunded after verification."

"Approved refunds are processed within 5 business days."
```

### LLM

Generates:

```text
"Your duplicate payment can be refunded after verification.
Once approved, the refund is generally processed within
5 business days."
```

### TTS

```text
Text → Speech → User
```

---

# Why This Architecture?

The project follows a **specialized-model architecture** rather than using an LLM for every stage.

### Fine-Tuned BERT

Used for:

* Intent classification
* Fast inference
* Structured predictions
* Domain-specific classification

### RAG

Used for:

* Private knowledge
* Company documentation
* Frequently changing information
* Grounded responses

### LLM

Used for:

* Reasoning
* Natural-language generation
* Contextual responses
* Complex requests

### LangGraph

Used for:

* State management
* Routing
* Agent orchestration
* Multi-step workflows

This separation makes each component responsible for the task it is best suited for.

---

# Project Structure

```text
ai-voice-support-agent/
│
├── data/
│   ├── intents.csv
│   └── knowledge_base/
│
├── bert/
│   ├── train.py
│   ├── evaluate.py
│   ├── inference.py
│   └── model/
│
├── rag/
│   ├── ingest.py
│   ├── embeddings.py
│   └── retriever.py
│
├── agents/
│   ├── billing.py
│   ├── technical.py
│   └── shipping.py
│
├── graph/
│   └── workflow.py
│
├── voice/
│   ├── whisper.py
│   └── tts.py
│
├── app/
│
├── requirements.txt
└── README.md
```

---

# Tech Stack

### Machine Learning

* Python
* PyTorch
* Hugging Face Transformers
* BERT
* Scikit-learn

### Speech

* Whisper / Faster-Whisper
* Text-to-Speech

### Agentic AI

* LangGraph
* LangChain

### Retrieval

* Sentence Transformers / Hugging Face Embeddings / Ollama Embeddings
* FAISS / ChromaDB

### Generative AI

* OpenAI models
* Ollama
* Hugging Face models

### Backend

* FastAPI

### Frontend

* React

### Deployment

* Docker

---

# Model Training

The BERT classifier is fine-tuned using labeled customer-support examples.

Example training data:

```csv
text,label
"My card payment failed",PAYMENT
"I need my money back",REFUND
"Cancel my order",CANCELLATION
"Where is my package?",SHIPPING
"I cannot login",ACCOUNT
"The application keeps crashing",TECHNICAL
```

Training pipeline:

```text
Dataset
   ↓
Train / Validation / Test Split
   ↓
BERT Tokenizer
   ↓
Tokenized Dataset
   ↓
Pretrained BERT
   ↓
Classification Head
   ↓
Fine-Tuning
   ↓
Evaluation
   ↓
Saved Model
```

---

# Evaluation

The classifier will be evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

Evaluation is performed on held-out data to measure generalization rather than training performance.

---

# Installation

Clone the repository:

```bash
git clone <repository-url>

cd ai-voice-support-agent
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Configuration

Create a `.env` file:

```env
OPENAI_API_KEY=your_api_key

# Optional
LANGCHAIN_API_KEY=your_api_key
LANGCHAIN_TRACING_V2=true
```

Do not commit API keys or credentials to the repository.

---

# Running the Project

### Train BERT

```bash
python bert/train.py
```

### Evaluate

```bash
python bert/evaluate.py
```

### Run inference

```bash
python bert/inference.py
```

### Build the RAG knowledge base

```bash
python rag/ingest.py
```

### Run the agent workflow

```bash
python graph/workflow.py
```

---

# Future Improvements

* [ ] Expand intent classification dataset
* [ ] Add priority classification
* [ ] Add sentiment detection
* [ ] Add confidence-based LLM fallback
* [ ] Add customer/order lookup tools
* [ ] Add conversation memory
* [ ] Add streaming responses
* [ ] Add voice activity detection
* [ ] Add evaluation dashboard
* [ ] Add LangSmith tracing
* [ ] Add authentication
* [ ] Add React interface
* [ ] Containerize with Docker
* [ ] Deploy the complete system

---

# Learning Objectives

This project is also designed as a practical study of modern NLP and agentic AI.

### NLP & Transformers

* Tokenization
* WordPiece
* BERT
* Transformer encoders
* Self-attention
* Contextual embeddings

### Fine-Tuning

* Dataset preparation
* Label encoding
* Tokenization
* Classification heads
* Loss functions
* Backpropagation
* Optimization
* Evaluation
* Model inference

### RAG

* Document ingestion
* Chunking
* Embeddings
* Vector databases
* Similarity search
* Retrieval
* Context augmentation

### Agentic AI

* LangGraph state
* Conditional routing
* Multi-agent workflows
* Tool calling
* Workflow orchestration

### Voice AI

* Speech recognition
* Text generation
* Text-to-speech

---

# Design Principle

The central design principle of this project is:

```text
Use the right model for the right task.
```

```text
Whisper
   → Speech Recognition

BERT
   → Intent Classification

LangGraph
   → Orchestration

Embedding Model
   → Semantic Representation

Vector Database
   → Retrieval

RAG
   → External Knowledge

LLM
   → Reasoning + Generation

TTS
   → Speech Synthesis
```

---

# Project Status

🚧 **In Development**

The project is being developed incrementally, starting with the BERT intent-classification pipeline and progressing toward the complete voice-based agentic system.

---

## License

This project is licensed under the MIT License.

````

### One recommendation

For the **actual final GitHub version**, I'd add these three things near the top once you've built them:

```text
⭐ Demo
📊 BERT evaluation results
🏗️ Architecture diagram
````

Especially the **BERT evaluation results**. A table like:

```text
| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|----|
| BERT  | 94.2%    | 93.8%     | 94.1%  | 93.9% |
```

