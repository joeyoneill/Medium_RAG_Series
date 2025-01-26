import { useState } from 'react'
import './App.css'

function App() {
  // State Variables
  const [query, setQuery] = useState<string>('');
  const [response, setResponse] = useState<string>('');
  const [isStreaming, setIsStreaming] = useState<boolean>(false);

  // Handle change in the user query textarea
  const handleQueryChange = (event: React.ChangeEvent<HTMLTextAreaElement>) => {
    setQuery(event.target.value);
  };

  // Submission of Query Handler - Websocket Handler
  const handleQuerySubmit = () => {

    // Check if Empty, and if so do nothing
    if (!query.trim()) return;

    // Open Websocket Instance
    const websocket = new WebSocket('ws://localhost:8000/ws/stream');
    setIsStreaming(true);
    setResponse('');

    // Websocket On Open Action
    websocket.onopen = () => {
      console.log('WebSocket connection opened.');
      // Send query to the server
      websocket.send(JSON.stringify({ query }));
    };

    // ON MESSAGE HANDLER
    websocket.onmessage = (event) => {
      const data = event.data;
      console.log('data: ', data);

      // Check for the termination flag
      if (data === '<<END>>') {
        websocket.close();
        setIsStreaming(false);
        return;
      }

      // Check for no query Error flag
      if (data == '<<E:NO_QUERY>>') {
        console.log('ERROR: No Query, closing connection...')
        websocket.close();
        setIsStreaming(false);
        return;
      }

      // Update state directly with the new data
      setResponse((prevResponse) => prevResponse + data);
    };

    // Websocket Error Handler
    websocket.onerror = (error) => {
      console.error('WebSocket error:', error);
      websocket.close();
      setIsStreaming(false);
    };

    // ON CLOSE Action
    websocket.onclose = () => {
      console.log('WebSocket connection closed.');
      setIsStreaming(false);
    };
  };

  return (
    <div className='main-div'>

      <div className='header-container'>
        {/* Title */}
        <h1>Full Stack RAG w/ Streaming</h1>

        {/* Query Input Bar */}
        <div className='query-input-bar'>
          <textarea
            className='query-textarea'
            value={query}
            onChange={handleQueryChange}
            placeholder="Query your data here..."
            rows={5}
            cols={100}
          ></textarea>
          <button
            className='sumbit-btn'
            onClick={handleQuerySubmit}
            disabled={isStreaming}
          >
            Sumbit
          </button>
        </div>
      </div>

      <div className='response-container'>
        {response}
      </div>
      
    </div>
  )
}

export default App;
