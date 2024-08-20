from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
import tiktoken
import os
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from starlette.middleware.base import BaseHTTPMiddleware
import time
from collections import defaultdict
from typing import Dict
load_dotenv()

app = FastAPI()

# class RequestMiddleware(BaseHTTPMiddleware):
#     def __init__(self, app):
#        super().__init__(app)
#        self.rate_limit_records : Dict[str, float] = defaultdict(float)
    
#     async def log_message(self, message:str):
#        print(message)
    
#     async def dispatch(self, request: Request, call_next):
#        client_ip = request.client.host
#        current_time = time.time()
#     #    if current_time - self.rate_limit_records[client_ip] < 10:
#     #       return Response(content="Rate Limit Exceeded", status_code=429)

#        self.rate_limit_records[client_ip] = current_time
#        path = request.url.path
#        await self.log_message(f"Request to Path: {path}")

#        start_time = time.time()
#        response = await call_next(request)
#        process_time = time.time() - start_time

#        custom_header = {"Process Time" : str(process_time)}
#        for header, value in custom_header.items():
#           response.headers.append(header, value)

#        await self.log_message(f"Response for {path} took {process_time} seconds.")
#        return response
          
# app.add_middleware(RequestMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://youtube-chatbot.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pinecone_api_key = os.environ.get("PINECONE_API_KEY")
openai_api_key = os.environ.get("OPENAI_API_KEY")
openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")

embeddings = OpenAIEmbeddings()
embed_model = "text-embedding-3-small"
openai_client = OpenAI()

openrouter_client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=openrouter_api_key
)

class PerformRAGRequest(BaseModel):
    youtube_url: str
    query: str

@app.post("/perform_rag")
async def perform_rag_endpoint(request: PerformRAGRequest):
    try:
        print(f"Youtube: {request.youtube_url}, Query: {request.query}")
        result = await perform_rag(
            youtube_url=request.youtube_url,
            query=request.query,
            pinecone_api_key=pinecone_api_key,
            openai_client=openai_client
        )
        return PlainTextResponse(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def perform_rag(youtube_url, query, pinecone_api_key, openai_client):
    try:
        # Set up Pinecone index and namespace
        index_name = 'youtube'
        namespace = 'youtube_video'
        # Generate a unique ID for the video based on its URL
        def extract_youtube_id(url):
            if 'youtube.com/watch?v=' in url:
                return url.split('v=')[-1]
            elif 'youtu.be/' in url:
                return url.split('.be/')[1].split('?')[0]
            else:
                raise ValueError('Invalid Youtube URL')

        video_id = extract_youtube_id(youtube_url)  # Extract the video ID from the URL
        # Initialize Pinecone vector store 
        vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
        # Initialize Pinecone
        pc = Pinecone(api_key=pinecone_api_key)
        # Connect to Pinecone index
        pinecone_index = pc.Index(index_name)
        
        # Check if the video is already stored in the Pinecone index
        vector_exists = pinecone_index.query(
            vector=[0]*1536,
            top_k=1,
            namespace=namespace,
            filter={'Source':{'$eq':video_id}}
        )
        if len(vector_exists['matches']) > 0:
            print('Vector exists.')
        else:
            print('Vector does not exist. Creating a new vector store.')
            tokenizer = tiktoken.get_encoding('p50k_base')
            # create the length function
            def tiktoken_len(text):
                tokens = tokenizer.encode(
                    text,
                    disallowed_special=()
                )
                return len(tokens)
            for i in range(3):
                loader = YoutubeLoader.from_youtube_url(youtube_url)
                data = loader.load()
                if data:
                    print('Data loaded.')
                    break
                else:
                    print('Data not loaded. Retrying...')
                    time.sleep(1)
            if not data:
                raise ValueError('Failed to load data from Youtube.')
            
            # Split the text into documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000,
                chunk_overlap=100,
                length_function=tiktoken_len,
                separators=["\n\n", "\n", " ", ""]
            )

            texts = text_splitter.split_documents(data)
            print(texts)
            
            try:
                vectorstore_from_texts = PineconeVectorStore.from_texts(
                    [f"Source: {t.metadata['source']} \n\nContent: {t.page_content}" for t in texts],
                    embeddings,
                    index_name=index_name,
                    namespace=namespace,
                    ids=[f"{video_id}_{i}" for i in range(len(texts))],  # Assign unique IDs to each text chunk
                    metadatas=[{"Source":t.metadata['source']} for t in texts]
                )
                print('Vector store created.')
            except Exception as e:
                print('Error creating vector store: ', e)
                raise HTTPException(status_code=500, detail=str(e))

        print('Creating query embedding...')
        # Create the query embedding
        raw_query_embedding = openai_client.embeddings.create(
            input=query,
            model="text-embedding-3-small"
        )
        query_embedding = raw_query_embedding.data[0].embedding

        print('Finding top matches...')
        # Query Pinecone for top matches
        for i in range(3):
            top_matches = pinecone_index.query(vector=query_embedding, top_k=10, include_metadata=True, namespace=namespace, filter={'Source': {'$eq': video_id}})
            if len(top_matches['matches']) > 0:
                break
            else:
                print('No matches found. Retrying...')
                time.sleep(3)

        # Extract contexts from the top matches
        contexts = [item['metadata']['text'] for item in top_matches['matches']]

        # Augment the query with context
        augmented_query = "<CONTEXT>\n" + "\n\n-------\n\n".join(contexts[:10]) + "\n-------\n</CONTEXT>\n\n\n\nMY QUESTION:\n" + query

        # Set up the system prompt
        system_prompt = f"""You are a professional researcher. Use the youtube video to answer any questions."""

        print('Generating response...')
        # Generate a response using the openrouter_client
        res = openrouter_client.chat.completions.create(
            model="google/gemma-2-9b-it:free",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": augmented_query}
            ]
        )
        print('Returning response: ', res.choices[0].message.content)
        # Return the content of the response
        return res.choices[0].message.content
    except Exception as e:
        print('Rag failed: ', e)
        raise HTTPException(status_code=500, detail=str(e))