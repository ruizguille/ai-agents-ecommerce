# AI Agents for E-commerce using PydanticAI

This repository shows how to build AI agents to automate customer support and order management in e-commerce applications using the PydanticAI library.

It showcases two example agents for a fashion e-commerce store:

1. A **support agent** that assists customers by answering their questions using **Retrieval-Augmented Generation (RAG).**
2. An **order management agent** that processes complex operations like updates, returns and cancellations.

For a detailed explanation of the code and concepts, check out [this blog post](https://codeawake.com/blog/ai-agents-ecommerce).

## Installation

### Prerequisites ✅

- Python 3.9 or higher

### Instructions

1. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

2. Create a `.env` file by copying the provided `.env.example` file and set the required environment variable:
    - `OPENAI_API_KEY`: Your OpenAI API key.

## Running the Agents

Run the RAG support agent:
```bash
python rag.py
```

Run the order management agent:
```bash
python orders.py
```