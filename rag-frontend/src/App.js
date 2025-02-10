import React, { useState, useEffect } from 'react';
import './App.css';

function App() {
  const [message, setMessage] = useState(0);

  useEffect(() => {
    fetch('http://127.0.0.1:5000/')
        .then(res => {
            return res.json()
        }).then(data => {
            console.log(data)
            setMessage(data.message);
        });
  }, []);

  return (
    <div className="App">
      <header className="App-header">
        <p>The message is: {message}</p>
      </header>
    </div>
  );
}

export default App;