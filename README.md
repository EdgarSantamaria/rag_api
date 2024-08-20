# AI YouTube Question API

This repository contains a FastAPI application that performs Retrieval-Augmented Generation (RAG) using YouTube video content. The API takes a YouTube URL and a query, processes the video content to extract relevant information, and provides an AI-generated response based on the query.

## Test The API 
The API is deployed on Vercel, but the youtube-transcript-api does not work in the deployed version. 

The youtube-transcript-api library should function locally and you can also use the pytube library to get more metadata info.

Test the API using preloaded videos from this Next.js app [Test The API](https://youtube-chatbot.vercel.app).
## How It Works
1. Text Extraction: The YouTube video's transcript is extracted and split into chunks.
2. Vector Store: The text chunks are embedded using OpenAI embeddings and stored in Pinecone.
3. Querying: When a query is made, the relevant text chunks are retrieved from Pinecone.
4. Response Generation: The query is augmented with retrieved contexts and sent to OpenRouter for generating a detailed response.

## Prerequisites
* Python 3.8+
* A Pinecone API key
* A OpenAI API key
* A OpenRouter API key

## API Endpoints
`POST /perform_rag`
* youtube_url: The URL of the YouTube video to process.
* query: The question to answer.

**Response**:

* **Success**: Returns the AI-generated answer based on the provided query.

## Middleware
### Request Middleware (Optional)

* Logs requests, enforces rate limiting, and measures response processing time. It is currently commented out but can be activated if needed.

### CORS Configuration

* The application is configured to allow requests from https://youtube-chatbot.vercel.app. Modify this in the CORSMiddleware settings to your frontend host.

