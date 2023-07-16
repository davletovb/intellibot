"""
Vector database for storing vectors and their metadata.
"""

import os
import logging
import asyncio

from langchain.document_loaders import UnstructuredFileLoader, WebBaseLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from chromadb.config import Settings

# Set API keys
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Set Chroma settings
CHROMA_SETTINGS = Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="db",
    anonymized_telemetry=False)

class VectorDB():
    def __init__(self, chat_user_id):
        if not isinstance(chat_user_id, str):
            raise ValueError("chat_user_id must be string")
            
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

        self.text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        self.embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        self.vector_store = Chroma(embedding_function=self.embeddings, client_settings=CHROMA_SETTINGS, persist_directory="db", collection_name=chat_user_id)
        self.llm = OpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)

    async def add_document(self, document):
        """Ingest a document into the vector store."""
        try:
            # Load the document
            loader = UnstructuredFileLoader(document)
            doc = loader.load()

            # Split the document into sentences
            texts = self.text_splitter.split_documents(doc)
            
            # Store the embeddings
            self.vector_store.add_documents(documents=texts)
            
            # Persist the vector store to disk
            self.vector_store.persist()

            # return the summary of the document
            summary = await self.summarize(texts)

            return summary
        except Exception as e:
            self.logger.error(f"Error adding document: {e}")
            return None
    
    
    async def add_url(self, url):
        """Ingest a web page into the vector store."""
        try:
            loader = WebBaseLoader(url)

            docs = loader.load()

            # Split the document into sentences
            texts = self.text_splitter.split_documents(docs)

            # Store the embeddings
            self.vector_store.add_documents(documents=texts)
            
            # Persist the vector store to disk
            self.vector_store.persist()

            # Gather the summary
            summary = await self.summarize(texts)

            return summary
        except Exception as e:
            self.logger.error(f"Error adding url: {e}")
            return None
    

    async def query(self, query):
        """Query the vector store for similar vectors."""
        try:
            # Query the vector store
            chain = RetrievalQAWithSourcesChain.from_chain_type(
                llm=self.llm, chain_type="stuff", retriever=self.vector_store.as_retriever())

            # Get the results
            results = await chain.arun(question=query, return_only_outputs=True)

            return results
        except Exception as e:
            self.logger.error(f"Error querying vector store: {e}")
            return None

    
    async def summarize(self, docs):
        """Get the summary of a document."""
        
        # Prompt template only for the stuff chain type
        prompt_tempate = """Write a concise summary of the following text, use simple language and bullet points:
                        {text}

                        SUMMARY:"""
        
        prompt = PromptTemplate(template=prompt_tempate, 
                                input_variables=["text"])

        # Create the chain
        chain = load_summarize_chain(self.llm,
                                     chain_type="map_reduce",
                                     map_prompt=prompt,
                                     combine_prompt=prompt)
        
        try:
            summary = await chain.arun(input_documents=docs, return_only_outputs=True)

            return summary
        except Exception as e:
            self.logger.error(f"Error gathering summary: {e}")
            return None
    
    async def clear_database(self):
        """Clear the vector store."""
        try:
            # Delete the collection from the vector store
            self.vector_store.delete_collection()

            return True
        except Exception as e:
            self.logger.error(f"Error clearing vector store: {e}")
            return False
