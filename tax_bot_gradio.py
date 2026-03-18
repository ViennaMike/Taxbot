"""
LlamaIndex Query Application with Gradio Chat Interface
Clean chat UI similar to Claude.ai with conversation history
"""
import sys

sys.stdout.reconfigure(encoding='utf-8')

from typing import Tuple

import chromadb
import gradio as gr
from llama_index.core import (
    Settings,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import (
    QueryFusionRetriever,
    VectorIndexRetriever,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.chroma import ChromaVectorStore


class LlamaIndexQueryBot:
    """Query bot with chat memory using existing ChromaDB vector store."""
    
    def __init__(
        self,
        chroma_path: str = "d:/chroma_db",
        collection_name: str = "mydocs",
        model_name: str = "llama3.1:8b",
        embedding_model: str = "BAAI/bge-base-en-v1.5",
        ollama_base_url: str = "http://localhost:11434",
        semantic_top_k: int = 5,
        bm25_top_k: int = 6,
        final_top_k: int = 6,
        chat_memory_token_limit: int = 3000,
    ):
        self.semantic_top_k = semantic_top_k
        self.bm25_top_k = bm25_top_k
        self.final_top_k = final_top_k
                
        # Initialize Ollama LLM
        self.llm = Ollama(
            model=model_name,
            base_url=ollama_base_url,
            request_timeout=600.0,
            temperature=0.0,
        )
        
        # Initialize HuggingFace embeddings
        self.embed_model = HuggingFaceEmbedding(
            model_name=embedding_model,
        )
        
        # Set global settings
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        
        # Initialize chat memory
        self.chat_memory = ChatMemoryBuffer.from_defaults(
            token_limit=chat_memory_token_limit,
        )
        
        # Connect to existing ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        
        try:
            self.chroma_collection = self.chroma_client.get_collection(
                name=collection_name
            )
            print(f"✓ Connected to existing collection: {collection_name}")
            print(f"✓ Documents in vector store: {self.chroma_collection.count()}")
        except Exception as e:
            print(f"✗ Error: Collection '{collection_name}' not found in {chroma_path}")
            print(f"Available collections: {[c.name for c in self.chroma_client.list_collections()]}")
            raise e
        
        self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )
        
        self.index = VectorStoreIndex.from_vector_store(
            self.vector_store,
            storage_context=self.storage_context,
        )
        
        print("Loading documents for BM25 indexing...")
        all_nodes = self.chroma_collection.get(include=['documents', 'metadatas'])
        
        from llama_index.core.schema import TextNode
        self.documents = []
        for i, (doc_id, text) in enumerate(zip(all_nodes['ids'], all_nodes['documents'])):
            node = TextNode(
                text=text,
                id_=doc_id,
                metadata=all_nodes['metadatas'][i] if all_nodes['metadatas'] else {}
            )
            self.documents.append(node)
        
        print(f"✓ Loaded {len(self.documents)} documents for BM25 indexing")
        
        self._create_hybrid_query_engine()
        
        print(f"✓ Query bot initialized with model: {model_name}")
        print(f"✓ Hybrid retrieval: {semantic_top_k} semantic + {bm25_top_k} BM25 → {final_top_k} final")
        print(f"✓ Chat memory enabled (token limit: {chat_memory_token_limit})")
    
    def _create_hybrid_query_engine(self):
        vector_retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=self.semantic_top_k,
        )
        
        bm25_retriever = BM25Retriever.from_defaults(
            nodes=self.documents,
            similarity_top_k=self.bm25_top_k,
        )
        
        self.fusion_retriever = QueryFusionRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            similarity_top_k=self.final_top_k,
            num_queries=1,
            mode="reciprocal_rerank",
            use_async=False,
        )
        
        self.query_engine = RetrieverQueryEngine.from_args(
            retriever=self.fusion_retriever,
            streaming=False,
            memory=self.chat_memory,
        )
    
    def clear_memory(self):
        """Clear the chat memory/history."""
        self.chat_memory.reset()
        print("✓ Chat memory cleared")
    
    # def query(self, question: str) -> Tuple[str, str]:
    #     full_query = f"""Question: {question}"""
    #     response = self.query_engine.query(full_query)
    #     sources_text = self._format_sources(response)
    #     return str(response), sources_text

    def query(self, question: str) -> Tuple[str, str]:
        """Query using direct retriever, bypassing LlamaIndex pipeline reranking."""
        nodes = self.fusion_retriever.retrieve(question)
        
        context_parts = []
        for i, node in enumerate(nodes, 1):
            context_parts.append(f"[Chunk {i}]:\n{node.text}")
        context_str = "\n\n".join(context_parts)
        
        prompt = (
            f"You are a trained, expert tax preparer.\n\n"
            f"TAX REFERENCE MATERIAL:\n{context_str}\n"
            f"─────────────────────────────────────────\n\n"
            f"QUESTION: {question}\n\n"
            f"INSTRUCTIONS: Before answering, read ALL chunks in the reference material above. "
            f"Identify every chunk that is relevant to the question. "
            f"If a chunk is titled or discusses a rule that directly matches the taxpayer's situation, "
            f"that chunk must be cited and applied. "
            f"Do not ignore chunks simply because they do not mention the taxpayer's filing status. "
            f"A rule that applies to 'taxpayers' applies to ALL taxpayers unless the chunk explicitly "
            f"states otherwise.\n\n"
            f"ANSWER: If taxpayer facts are provided, treat them as completely accurate. "
            f"Apply tax rules as they are written — if the taxpayer meets the stated qualifications "
            f"for a rule, the rule applies. Do NOT require the reference material to explicitly "
            f"confirm every combination of circumstances. "
            f"Only answer False or state a rule does not apply if the taxpayer facts fail to meet or "
            f"violate a stated qualification. "
            f"IMPORTANT: If the reference material contains a general rule that applies "
            f"to all taxpayers, that rule takes precedence over more specific rules that "
            f"apply to a subset of taxpayers, unless the specific rule explicitly excludes "
            f"the taxpayer's situation. "
            f"Consider ALL tax rules that may apply to the taxpayer's specific circumstances "
            f"including filing status, age, disability, and dependents. "
            f"Use the reference material to support your answer and cite specific references used.\n\n"
        )
    
        response = self.llm.complete(prompt)
        sources_text = self._format_sources_from_nodes(nodes)
        return str(response), sources_text

    def _format_sources_from_nodes(self, nodes) -> str:
        """Format sources directly from retrieved nodes."""
        if not nodes:
            return ""
        
        seen_files = {}
        for node in nodes:
            filename = node.metadata.get('file_name', 'Unknown source')
            if filename not in seen_files:
                seen_files[filename] = node
            else:
                existing_score = getattr(seen_files[filename], 'score', 0)
                current_score = getattr(node, 'score', 0)
                if current_score > existing_score:
                    seen_files[filename] = node

        unique_nodes = list(seen_files.values())
        sources_parts = ["\n\n───────────────────────────────\n📚 Sources:\n"]
        
        for i, node in enumerate(unique_nodes, 1):
            filename = node.metadata.get('file_name', 'Unknown source')
            score = f"{node.score:.3f}" if hasattr(node, 'score') else 'N/A'
            preview = node.text[:200] + "..." if len(node.text) > 200 else node.text
            preview = preview.replace('\n', ' ')
            sources_parts.append(
                f"\n[{i}] {filename} (relevance: {score})\n"
                f"    {preview}\n"
            )
        
        return "".join(sources_parts)       
    
    def _format_sources(self, response) -> str:
        if not hasattr(response, 'source_nodes') or not response.source_nodes:
            return ""

        seen_files = {}
        for node in response.source_nodes:
            filename = node.metadata.get('filename', 'Unknown source')
            if filename not in seen_files:
                seen_files[filename] = node
            else:
                existing_score = getattr(seen_files[filename], 'score', 0)
                current_score = getattr(node, 'score', 0)
                if current_score > existing_score:
                    seen_files[filename] = node
    
        unique_nodes = list(seen_files.values())       
        
        sources_parts = ["\n\n───────────────────────────────\n📚 Sources:\n"]
        
        for i, node in enumerate(unique_nodes, 1):
            filename = node.metadata.get('file_name', 'Unknown source')
            score = f"{node.score:.3f}" if hasattr(node, 'score') else 'N/A'
            preview = node.text[:200] + "..." if len(node.text) > 200 else node.text
            preview = preview.replace('\n', ' ')
            sources_parts.append(
                f"\n[{i}] {filename} (relevance: {score})\n"
                f"    {preview}\n"
            )
        
        return "".join(sources_parts)


# Global bot instance
bot = None

def initialize_bot():
    """Initialize the bot - called once at startup."""
    global bot
    
    print("=" * 60)
    print("Initializing Tax Query Bot...")
    print("=" * 60)
    
    bot = LlamaIndexQueryBot(
        chroma_path="d:/chroma_db",
        collection_name="mydocs",
        model_name="llama3.1:8b",
        embedding_model="BAAI/bge-base-en-v1.5",
        semantic_top_k=6,
        bm25_top_k=6,
        final_top_k=5,
        chat_memory_token_limit=3000,
    )
    
    print("=" * 60)
    print("✓ Bot ready!")
    print("=" * 60)


# ── NEW: called when the user uploads a file ──────────────────────────────────
def upload_scenario(file):
    """
    Read the uploaded markdown file, clear the bot's LlamaIndex memory,
    and return the file contents plus a reset chat history.

    Returns:
        doc_content  (str)  – stored in gr.State, passed into chat_function
        status_msg   (str)  – shown in the status textbox
        []                  – empty list that resets the Chatbot widget
    """
    if file is None:
        return "", "No file loaded.", []

    with open(file.name, "r", encoding="utf-8") as f:
        content = f.read()

    # Clear LlamaIndex chat memory so the new scenario starts fresh
    if bot:
        bot.clear_memory()

    filename = file.name.split("/")[-1].split("\\")[-1]
    status = f"✅ Loaded: {filename}  ({len(content):,} chars) — chat history cleared"
    print(status)

    # Return: new doc content → State, status message → Textbox, [] → Chatbot
    return content, status, []


# ── MODIFIED: chat_function now accepts doc_content from State ────────────────
def chat_function(message: str, history, doc_content: str, notes: str):
    """
    Process a chat message.  doc_content comes from gr.State and is
    prepended to the question when a scenario file is loaded.
    """
    try:
        if message.strip().startswith("/"):
            command = message.strip().lower()
            if command == "/clear":
                if bot:
                    bot.clear_memory()
                return "", history + [("", "✓ Chat memory cleared. Starting fresh conversation.")]
            elif command == "/quit":
                return "", history + [("", "To quit, please close the browser tab or press Ctrl+C in the terminal.")]
            else:
                return "", history + [("", f"Unknown command: {command}\n\nAvailable commands:\n- /clear - Clear chat memory\n- /quit - Instructions to quit")]

        # Prepend scenario context when a file has been loaded
        inline_facts = notes.strip() if notes else ""
        all_facts = "\n\n".join(filter(None, [doc_content, inline_facts]))

        if all_facts:
            full_question = (
                f"TAXPAYER FACTS — treat these as completely accurate facts, "
                f"do not contradict them:\n{all_facts}\n\n"
                f"Consider all tax rules that may apply to this taxpayer's specific "
                f"circumstances including filing status, age, disability, and dependents.\n\n"
                f"IMPORTANT: If the reference material contains a general rule that applies "
                f"to all taxpayers, that rule takes precedence over more specific rules that "
                f"apply to a subset of taxpayers, unless the specific rule explicitly excludes "
                f"the taxpayer's situation.\n\n"
                f"Using the taxpayer facts above and the reference material, answer this question:\n{message}"
            )
        else:
            full_question = message

        answer, sources = bot.query(full_question)
        full_response = answer + sources

        # Append the new turn to history and return it
        return "", history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": full_response}
        ]

    except Exception as e:
        return "", history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": f"Error: {str(e)}"}
        ]


def main():
    """Launch the Gradio interface."""
    
    initialize_bot()

    # Debug retrieval — remove these lines once you're done diagnosing
    # debug_retrieval("standard deduction blind single taxpayer")
    # debug_retrieval("additional standard deduction blindness")
    # debug_retrieval("blind taxpayer deduction filing status")
    # debug_retrieval("Fred can claim a higher standard deduction because he is blind. a. True b. False")
    
    theme = gr.themes.Default(text_size=gr.themes.sizes.text_lg)

    with gr.Blocks() as demo:

        # ── State: holds the current scenario document text ──────────────────
        # gr.State is invisible to the user; it just keeps a Python value
        # alive between interactions for this browser session.
        doc_state = gr.State("")   # starts as empty string = no scenario loaded

        gr.HTML("<h1>💼 Tax ChatBot</h1>")
        gr.Markdown(
            "Ask questions about federal income taxes and get answers.  \n"
            "The bot is a demonstration and can make mistakes or not know certain information. Always "
            "double-check with an autoritative source or a human tax professional before making any decisions based on the bot's answers.  \n\n"
            "**Commands:** Type `/clear` to reset memory | Type `/quit` for exit instructions"
        )

        # ── File upload row ───────────────────────────────────────────────────
        with gr.Row():
            file_upload = gr.File(
                file_types=[".md"],
                label="📂 Upload Scenario (.md)",
                scale=2,
            )
            doc_status = gr.Textbox(
                label="Current Scenario",
                value="No scenario loaded — answering from taxpayer facts and reference material only.",
                interactive=False,
                scale=3,
            )

        # ── Chat area ─────────────────────────────────────────────────────────
        chatbot = gr.Chatbot(height=500)

        with gr.Row():
            notes = gr.Textbox(
                label="Interview Notes / Taxpayer Facts",
                placeholder="Paste or type taxpayer facts here...",
                lines=4,
                scale=5,            
            )
            msg = gr.Textbox(
                placeholder="Ask your tax question here… (Enter for new line, Shift+Enter to submit",
                lines=4,
                show_label="Question",
                scale=5,
            )
            submit_btn = gr.Button("Send", scale=1, variant="primary")

        # ── Wire up upload ────────────────────────────────────────────────────
        # When a file is chosen:  read it → update doc_state, doc_status, chatbot
        file_upload.change(
            fn=upload_scenario,
            inputs=[file_upload],
            outputs=[doc_state, doc_status, chatbot],
        )

        # ── Wire up chat ──────────────────────────────────────────────────────
        # chat_function needs: the typed message, current history, and the State.
        # It returns: cleared textbox (""), updated history.
        submit_btn.click(
            fn=chat_function,
            inputs=[msg, chatbot, doc_state, notes],
            outputs=[msg, chatbot],
        )
        msg.submit(
            fn=chat_function,
            inputs=[msg, chatbot, doc_state, notes],
            outputs=[msg, chatbot],
        )

    print("\n" + "=" * 60)
    print("Launching Gradio interface...")
    print("=" * 60 + "\n")

    demo.launch(
        server_name="127.0.0.1",
        server_port=7861,
        share=False,
        inbrowser=True,
        theme=theme,
        auth=[
            ("taxnewb", "taxORcredit2024"),
        ],
        auth_message="Tax ChatBot — Please log in to continue.",
    )

def debug_retrieval(query: str, n_results: int = 10):
    """Query ChromaDB directly and print what gets retrieved."""

    # Use the same embedding model as the collection
    query_embedding = bot.embed_model.get_text_embedding(query)

    results = bot.chroma_collection.query(
        query_embeddings=[query_embedding],   # <-- pass embedding, not text
        n_results=n_results,
        include=['documents', 'metadatas', 'distances']
    )
    
    print("\n" + "=" * 60)
    print(f"DEBUG: Top {n_results} chunks for query: '{query}'")
    print("=" * 60)
    
    for i, (doc, meta, dist) in enumerate(zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    ), 1):
        print(f"\n[{i}] Distance: {dist:.4f} (lower = more similar)")
        print(f"    File: {meta.get('file_name', 'unknown')}")
        print(f"    Text: {doc[:300]}...")
    
    print("=" * 60)

if __name__ == "__main__":
    main()