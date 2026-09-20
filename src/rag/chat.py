from langchain_openai import ChatOpenAI
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationalRetrievalChain

from .config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_MODEL,
)


def create_llm():
    """Create the OpenRouter LLM."""
    return ChatOpenAI(
        model=OPENROUTER_MODEL,
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        temperature=0.7
    )


def create_conversation_chain(vectorstore, num_chunks=25):
    """Create a conversational retrieval chain."""
    llm = create_llm()

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": int(num_chunks)}
    )

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True
    )

    return conversation_chain

def format_sources(source_documents):
    """Format retrieved documents into readable source references."""
    sources = []

    for doc in source_documents:
        metadata = doc.metadata

        file_name = metadata.get(
            "file_name",
            "Unknown file"
        )

        doc_type = metadata.get(
            "doc_type",
            "Unknown"
        )

        page = metadata.get("page_label")

        if page is None:
            page = metadata.get("page")

            if page is not None:
                page = int(page) + 1

        if page is not None:
            source = (
                f"📄 {file_name} — "
                f"{doc_type}, Page {page}"
            )
        else:
            source = (
                f"📄 {file_name} — "
                f"{doc_type}"
            )

        if source not in sources:
            sources.append(source)

    return sources

def reset_conversation(conversation_chain):
    """Clear the conversation history."""
    if conversation_chain is not None:
        conversation_chain.memory.clear()

        