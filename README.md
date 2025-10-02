# RAG (Retrieval-Augmented Generation) Pipeline

## What is RAG?
It stands for Retrieval-Augmented Generation, a technique that combines pre-trained language models with external knowledge sources to improve the quality and relevance of generated text. RAG models retrieve relevant documents from a knowledge base and use them to generate more informed and contextually accurate responses.

## A basic RAG pipleline consists of following components:
1. **Ingestion Pipeline**: This component is responsible for collecting and processing the external knowledge sources. It can involve web scraping, document parsing, or database querying to gather relevant information.
2. **Vector Store**: The ingested documents are converted into vector representations using techniques like embeddings. These vectors are then stored in a vector database or index, allowing for efficient similarity searches.
3. **Retrieval Mechanism**: When a user query is received, the retrieval mechanism searches the vector store to find the most relevant documents based on their similarity to the query. This step typically involves techniques like nearest neighbor search or approximate nearest neighbor search.
4. **Generative Model**: The retrieved documents are then passed to a generative model, such as GPT-3 or BERT, which uses the information from the documents to generate a response. The model can be fine-tuned on specific tasks or domains to improve its performance.
5. **Response Generation**: Finally, the generative model produces a response based on the retrieved documents and the user query. The response can be in the form of text, summaries, or answers to specific questions.

![RAG Pipeline](../data/rag_pipeline.jpeg) --- IGNORE ---

## Working of RAG (How I implemented it)
1. **Ingestion**:  I used the `langchain` library and directory loader to load all the documents from a specified directory. The documents are then split into smaller chunks using a text splitter to ensure that they fit within the context window of the language model. The documents used here are just 4 sample famous research papers in .pdf format.
2. **Vector Store**: I utilized the `SentenceTransformer` model from the `sentence-transformers` library to convert the document chunks into vector embeddings. These embeddings are then stored in a `Chroma` vector store, which allows for efficient similarity searches.
3. **Retrieval**: When a user query is received, I used the `Chroma` vector store to perform a similarity search and retrieve the most relevant document chunks based on the query.
4. **Generative Model**: I employed the `OpenAI` language model from the `langchain` library to generate a response. The model takes the retrieved document chunks and the user query as input and produces a contextually relevant response.

## Tech Stack
- Python
- LangChain
- OpenAI API
- ChromaDB

## References
- [LangChain Documentation](https://docs.langchain.com/oss/python/langchain/overview)
- [OpenAI Documentation](https://platform.openai.com/docs/introduction)
- [ChromaDB Documentation](https://docs.trychroma.com/getting-started)

Huge thanks to Krish Naik for his [playlist](https://www.youtube.com/playlist?list=PLZoTAELRMXVM8Pf4U67L4UuDRgV4TNX9D) on building a RAG pipeline using LangChain and OpenAI. It helped me a lot in understanding the concepts and implementation of RAG.
