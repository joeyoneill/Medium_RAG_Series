'''
pip install fastapi uvicorn websockets langchain-openai python-dotenv
'''

# Imports
import asyncio
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
import base64

from RAGChain import vector_store, text_splitter, rag_chain

################################################################
# App & Env Set Up
################################################################

# Init App
app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

################################################################
# Root Redirect for Accessibility
################################################################

# Root Redirect to SwaggerUI
@app.get('/', tags=["General"])
def root():
    return RedirectResponse('/docs')

################################################################
# Vector Store Endpoints
################################################################

# Define File Upload Request Model
class FileRequest(BaseModel):
    file_data: str  # Base64-encoded file content string

# POST: Uploads Base64 Encoded Filestream into Chroma VectorDB
@app.post('/upload_file', tags=["VectorDB"])
def upload_file(request: FileRequest):

    # Initialize File String
    file_str = ''

    # Try and Decode File
    try:
        decoded_bytes = base64.b64decode(request.file_data)
        file_str = decoded_bytes.decode("utf-8")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid file data: {str(e)}")
    
    # Check if String is Empty or None
    if not file_str:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid file data: File is empty or null...")
    
    # Create Documents (Chunks) From File
    texts = text_splitter.create_documents([file_str])

    # Save Document Chunks to Vector Store
    ids = vector_store.add_documents(texts)

    # Return Success & List of Ids for created documents
    return {
        'status': status.HTTP_201_CREATED,
        'uploaded_ids': ids
    }

# Define Semantic Similarity POST Request
class SearchRequest(BaseModel):
    search_str: str # VectorDB Search String
    n: int = 2 # Number of Chunks to Return From the VectorDB (Default 2)

# POST: Returns Semantic Similarity Chunks based on a user's query
@app.post('/vector_search', tags=["VectorDB"])
def similarity_search(request: SearchRequest):
    try:
        #  Query the Vector Store
        results = vector_store.similarity_search(
            request.search_str,
            k = request.n
        )
        
        # Return Success & Chunks
        return {
            'status': status.HTTP_200_OK,
            'results': results
        }
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Error in request: {str(e)}")

################################################################
# RAG Endpoint
################################################################

# RAG POST Request
class RAGRequest(BaseModel):
    query: str # User Query

@app.post('/rag', tags=["RAG"])
def rag_chain_invoke(request: RAGRequest):

    # Get Query
    query = request.query
    if not query:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Error in request: Empty or None String Value in Query...")
    
    # Query the RAG Chain
    response = rag_chain.invoke(query)

    # Return Success
    return {
        'status': status.HTTP_200_OK,
        'response': response
    }


################################################################
# LLM WebSocket
################################################################

@app.websocket("/ws/stream")
async def chat_stream(websocket: WebSocket):
    await websocket.accept()
    try:
        resp = ''
        while True:
            # Receive data from the Frontend
            data = await websocket.receive_json()

            # Check if the request data contains proper keys
            if 'query' not in data:
                await websocket.send_text('<<E:NO_QUERY>>')
                break
            
            # Get Query From Request Data
            query = data['query']

            # Stream the response
            for token in rag_chain.stream(query):
                print(token.content)
                await websocket.send_text(token.content)
                await asyncio.sleep(0)
                resp += token.content
            
            # Send Successful Completion response to the Frontend
            await websocket.send_text('<<END>>')

    # WebSocket Disconnected
    except WebSocketDisconnect:
        print("Websocket Disconnected")
    
    # Any other Error/Exception
    except Exception as e:
        print(f"Error in WebSocket Connection: {e}")