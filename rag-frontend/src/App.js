import React, { useState, useEffect } from "react";
import "./App.css"; // Import CSS file

function App() {
  const [hybrid_mess, setHybrid_mess] = useState("None");
  const [pdfResponse, setPdfResponse] = useState("");
  const [selectedFile, setSelectedFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [fileName, setFileName] = useState("");
  const [query, setQuery] = useState("");
  const [queryResponse, setQueryResponse] = useState("");
  const [uploadedFiles, setUploadedFiles] = useState([]);

  // Handle file selection
  const handleFilePick = (event) => {
    const file = event.target.files[0];
    setPdfResponse(false);
    if (!file) {
      // Handle case where no file is selected (e.g., user cancels file picker)
      setFileName("");
      return;
    }

    setPdfResponse(""); // Clear previous response
    setFileName(file.name);
    setSelectedFile(file);
  };

  // Handle file upload
  const handleFileUpload = async () => {
    if (!selectedFile) {
      alert("Please select a PDF file first!");
      return;
    }

    setLoading(true); // Show loading state

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await fetch("http://127.0.0.1:5000/upload", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      if (data.response) {
        setPdfResponse(data.response); // Store AI-generated response
        setUploadedFiles(data.files); // Update uploaded files list
      } else {
        console.error("Error processing file:", data.error);
      }
    } catch (error) {
      console.error("Error uploading file:", error);
    }

    setLoading(false); // Hide loading state
  };

  // Handle query submission
  const handleQuerySubmit = async () => {
    if (!query.trim()) {
      alert("Please enter a query!");
      return;
    }

    setLoading(true); // Show loading state

    // Encode query in x-www-form-urlencoded format
    const formBody = new URLSearchParams();
    formBody.append("query", query);

    try {
      const response = await fetch("http://127.0.0.1:5000/query", {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: formBody.toString(), // Convert to proper form data format
      });

      const data = await response.json();
      if (data.message) {
        setQueryResponse(data.message); // Store query response
      } else {
        console.error("Error processing query:", data.error);
      }
    } catch (error) {
      console.error("Error submitting query:", error);
    }

    setLoading(false); // Hide loading state
  };

  // Handle query submission
  const handleHybridSubmit = async () => {
    if (!query.trim()) {
      alert("Please enter a query!");
      return;
    }

    setLoading(true); // Show loading state

    // Encode query in x-www-form-urlencoded format
    const formBody = new URLSearchParams();
    formBody.append("query", query);

    try {
      const response = await fetch("http://127.0.0.1:5000/hybrid", {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: formBody.toString(), // Convert to proper form data format
      });

      const data = await response.json();
      if (data.message) {
        setHybrid_mess(data.message); // Store query response
      } else {
        console.error("Error processing query:", data.error);
      }
    } catch (error) {
      console.error("Error submitting query:", error);
    }

    setLoading(false); // Hide loading state
  };

  useEffect(() => {
    fetch("http://127.0.0.1:5000/")
      .then((res) => res.json())
      .then((data) => {
        console.log(data);
        setUploadedFiles(data.files); // Store list of uploaded files
      })
      .catch((error) => console.error("Error fetching files:", error));
  }, []);

  return (
    <div>
        <div className="app">
        <h1 className="title">Upload a PDF for RAG Processing</h1>

{/* Custom File Input */}
<label className="file-label">
  Choose PDF File
  <input
    type="file"
    accept=".pdf"
    onChange={handleFilePick}
    className="file-input"
  />
</label>

{/* Display Selected File Name */}
{fileName && <p className="file-name">Selected File: {fileName}</p>}

{/* Upload Button */}
<button
  onClick={handleFileUpload}
  disabled={!selectedFile || loading}
  className={selectedFile ? "button" : "button-disabled"}
>
  {loading ? "Uploading..." : "Upload & Process"}
</button>
{/* PDF Confirmation */}
{pdfResponse && (
  <div>
    <p className="response-text">{pdfResponse}</p>
  </div>
)}

{/* Display Uploaded Files */}
{uploadedFiles.length > 0 && (
  <div className="uploaded-files">
    <h2>Uploaded Files:</h2>
    <ul>
      {uploadedFiles.map((file, index) => (
        <li key={index}>{file}</li>
      ))}
    </ul>
  </div>
)}

{/* Query Input */}
<h2 className="query-title">Ask a Question</h2>
<input
  type="text"
  placeholder="Enter your query"
  value={query}
  onChange={(e) => setQuery(e.target.value)}
  className="query-input"
/>
<button
  onClick={handleQuerySubmit}
  disabled={loading || !query.trim()}
  className="query-button"
>
  {loading ? "Processing..." : "Dense Query"}
</button>
<button
  onClick={handleHybridSubmit}
  disabled={loading || !query.trim()}
  className="query-button"
>
  {loading ? "Processing..." : "Hybrid Query"}
</button>
{/* Query Response */}
{queryResponse && (
  <div>
    <h2 className="response-title">Query Response:</h2>
    <p className="response-text">{queryResponse}</p>
  </div>
)}
        </div>
      <div className="app">
      <text>{hybrid_mess}</text>
      </div>
    </div>
  );
}

export default App;
