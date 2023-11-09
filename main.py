import os
import openai
import gradio as gr
from llama_index import StorageContext, load_index_from_storage, SimpleDirectoryReader, VectorStoreIndex

from theme import CustomTheme


def response(message, history):
    if not os.path.exists("store"):
        # load the documents and create the index
        documents = SimpleDirectoryReader("eigene").load_data()
        index = VectorStoreIndex.from_documents(documents)
        # store it for later
        index.storage_context.persist()
    else:
        # load the existing index
        storage_context = StorageContext.from_defaults(persist_dir="store")
        index = load_index_from_storage(storage_context)

    query_engine = index.as_query_engine()
    answer = query_engine.query(message)

    return str(answer)


def main():
    openai.api_key = os.environ["OPENAI_API_KEY"]

    custom_theme = CustomTheme()

    chatbot = gr.ChatInterface(
        fn=response,
        retry_btn=None,
        undo_btn=None,
        theme=custom_theme,
    )

    chatbot.launch(inbrowser=True)


if __name__ == "__main__":
    main()
