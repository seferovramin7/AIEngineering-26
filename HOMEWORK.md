# Homework: Build Your Own RAG System

For this assignment, you will build your own RAG system using Python and the tools we discussed in the session.

## The Task

Your task is to create a simple question-answering application that can answer questions about the content of a book of your choice.

## Instructions

1.  **Choose a book:** Find a plain text version of a book that you would like to use as your knowledge base. You can find many free books on websites like Project Gutenberg.
2.  **Set up your environment:** Make sure you have Python and the necessary libraries installed. You will need:
    *   `langchain`
    *   `openai`
    *   `chromadb`
    *   `tiktoken`
3.  **Create your knowledge base:** Create a file named `book.txt` and paste the content of the book into it.
4.  **Build your RAG system:** Write a Python script that:
    *   Loads the `book.txt` file.
    *   Splits the book into smaller chunks.
    *   Creates a ChromaDB vector store from the chunks.
    *   Creates a retriever.
    *   Creates a `RetrievalQA` chain.
    *   Prompts the user to ask a question about the book.
    *   Prints the answer to the screen.
5.  **Test your application:** Ask your application a few questions about the book to make sure it is working correctly.

## Bonus Points

*   Experiment with different chunk sizes and overlaps to see how they affect the quality of the answers.
*   Try using a different embedding model.
*   Create a simple web interface for your application using a framework like Flask or Streamlit.
