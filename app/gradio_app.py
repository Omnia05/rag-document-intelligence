import os
import gradio as gr

from src.rag.session import initialize_session
from src.rag.pipeline import process_documents
from src.rag.chat import reset_conversation
from src.rag.chat import format_sources


current_conversation_chain = None
current_vectorstore = None

def process_documents_for_app(
    knowledge_base_dir,
    max_file_size,
    chunk_size,
    chunk_overlap,
    num_chunks,
    session_dir
):
    global current_conversation_chain
    global current_vectorstore

    try:
        result = process_documents(
            knowledge_base_dir=knowledge_base_dir,
            max_file_size_mb=max_file_size,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            session_dir=session_dir,
            num_chunks=num_chunks
        )

        current_vectorstore = result["vectorstore"]
        current_conversation_chain = result["conversation_chain"]

        status = (
            f"✅ Processing complete!\n"
            f"Documents loaded: {result['documents']}\n"
            f"Chunks created: {result['chunks']}\n"
            f"Files skipped: {len(result['skipped_files'])}"
        )

        details = "\n".join(result["processing_log"])

        return status, details

    except Exception as e:
        return f"❌ Error: {str(e)}", ""


def chat_with_documents(message, history, num_chunks, session_dir):
    global current_conversation_chain
    global current_vectorstore

    if current_conversation_chain is None:
        return "❌ Please process documents first before asking questions!"

    if session_dir and os.path.exists(session_dir):
        os.utime(session_dir, None)

    try:
        retriever = current_vectorstore.as_retriever(
            search_kwargs={"k": int(num_chunks)}
        )

        current_conversation_chain.retriever = retriever

        result = current_conversation_chain.invoke(
            {"question": message}
        )

        sources = format_sources(
            result["source_documents"]
        )

        source_text = "\n".join(
            f"- {source}"
            for source in sources
        )

        return (
            f"{result['answer']}\n\n"
            f"### 📚 Sources\n"
            f"{source_text}"
        )

    except Exception as e:
        return f"❌ Error generating response: {str(e)}"


def reset_chat():
    global current_conversation_chain

    if current_conversation_chain is not None:
        reset_conversation(current_conversation_chain)
        return "Chat history cleared."

    return "No active conversation to clear."


def build_app():
 
    # ============================================================
    # GRADIO INTERFACE
    # ============================================================

    with gr.Blocks(
        title="AI Document Intelligence Assistant",
        theme=gr.themes.Soft()
    ) as app:

        session_dir = gr.State()

        gr.Markdown(
            "# 🎓 AI Document Intelligence Assistant"
        )

        gr.Markdown(
            "Upload your document library and ask questions using retrieval-augmented generation."
        )

        # ========================================================
        # TABS
        # ========================================================

        with gr.Tabs():

            # ====================================================
            # CONFIGURATION TAB
            # ====================================================

            with gr.Tab("⚙️ Configuration"):

                gr.Markdown(
                    "### 📁 Document Processing Settings"
                )

                gr.Markdown(
                    "💡 **Tip:** Enter the path to your "
                    "document directory."
                )

                with gr.Row():

                    with gr.Column():

                        knowledge_dir = gr.Textbox(
                            label="Knowledge Base Directory",
                            value="",
                            placeholder=(
                                "Enter your document directory path"
                            ),
                            lines=1
                        )

                        max_file_size = gr.Slider(
                            label="Max File Size (MB)",
                            minimum=0.5,
                            maximum=50,
                            value=4,
                            step=0.5
                        )

                    with gr.Column():

                        chunk_size = gr.Slider(
                            label="Chunk Size (characters)",
                            minimum=200,
                            maximum=1500,
                            value=800,
                            step=100,
                            info=(
                                "Smaller chunks = better "
                                "token management"
                            )
                        )

                        chunk_overlap = gr.Slider(
                            label="Chunk Overlap (characters)",
                            minimum=0,
                            maximum=300,
                            value=150,
                            step=25,
                            info=(
                                "Overlap preserves context "
                                "between chunks"
                            )
                        )

                process_btn = gr.Button(
                    "🚀 Process Documents",
                    variant="primary",
                    size="lg"
                )

                with gr.Row():

                    status_output = gr.Textbox(
                        label="Status",
                        lines=2,
                        max_lines=2
                    )

                detailed_output = gr.Textbox(
                    label="Detailed Processing Log",
                    lines=15,
                    max_lines=20
                )

            # ====================================================
            # CHAT TAB
            # ====================================================

            with gr.Tab("💬 Chat"):

                gr.Markdown(
                    "### 🤖 Ask Questions About Your Documents"
                )

                with gr.Row():

                    with gr.Column(scale=1):

                        num_chunks = gr.Slider(
                            label="Number of chunks to retrieve",
                            minimum=1,
                            maximum=50,
                            value=25,
                            step=1
                        )

                        reset_btn = gr.Button(
                            "🗑️ Clear Chat History",
                            variant="secondary"
                        )

                        reset_output = gr.Textbox(
                            label="Reset Status",
                            lines=1
                        )

                    with gr.Column(scale=3):

                        chatbot = gr.ChatInterface(
                            fn=lambda msg, history:
                                chat_with_documents(
                                    msg,
                                    history,
                                    num_chunks.value,
                                    session_dir.value
                                ),
                            type="messages",
                            title="Document Intelligence Chat",
                            description=(
                                "Ask questions about your "
                                "processed documents"
                            )
                        )

        # ========================================================
        # EVENT HANDLERS
        # ========================================================

        process_btn.click(
            fn=initialize_session,
            inputs=None,
            outputs=session_dir
        ).then(
            fn=process_documents_for_app,
            inputs=[
                knowledge_dir,
                max_file_size,
                chunk_size,
                chunk_overlap,
                num_chunks,
                session_dir
            ],
            outputs=[
                status_output,
                detailed_output
            ]
        )

        reset_btn.click(
            #fn=reset_conversation,
            fn=reset_chat,
            outputs=reset_output
        )

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch()