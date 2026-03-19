import { useState, useRef, useCallback } from "react";
import axios from "axios";

export default function Upload({ onAnalysisComplete, isLoading, setIsLoading, setError }) {
  const [file, setFile] = useState(null);
  const [jobDescription, setJobDescription] = useState("");
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef(null);

  const handleFile = (f) => {
    if (f && f.type === "application/pdf") {
      setFile(f);
      setError(null);
    } else {
      setError("Please upload a PDF file.");
    }
  };

  const onDrop = useCallback((e) => {
    e.preventDefault();
    setIsDragging(false);
    handleFile(e.dataTransfer.files[0]);
  }, []);

  const onDragOver = useCallback((e) => { e.preventDefault(); setIsDragging(true); }, []);
  const onDragLeave = useCallback(() => setIsDragging(false), []);

  const handleSubmit = async () => {
    if (!file) return setError("Please upload a resume PDF.");
    if (!jobDescription.trim()) return setError("Please enter a job description.");
    setIsLoading(true);
    setError(null);
    const formData = new FormData();
    formData.append("resume", file);
    formData.append("job_description", jobDescription);
    try {
      const response = await axios.post("/upload", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      onAnalysisComplete(response.data);
    } catch (err) {
      setError(
        err.response?.data?.message ||
        "Failed to connect to the server. Make sure the backend is running on port 5000."
      );
    } finally {
      setIsLoading(false);
    }
  };

  const dropzoneClass = [
    "dropzone",
    isDragging ? "dropzone--dragging" : "",
    file ? "dropzone--has-file" : "",
  ].filter(Boolean).join(" ");

  const canSubmit = !isLoading && file && jobDescription.trim();

  return (
    <div className="upload-form">
      {/* Drop zone */}
      <div
        className={dropzoneClass}
        onDrop={onDrop}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onClick={() => !file && fileInputRef.current?.click()}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf"
          style={{ display: "none" }}
          onChange={(e) => handleFile(e.target.files[0])}
        />
        <div className="dropzone__inner">
          {file ? (
            <>
              <div className="dropzone__icon-wrap dropzone__icon-wrap--success">
                <svg fill="none" viewBox="0 0 24 24" strokeWidth={1.5}>
                  <path strokeLinecap="round" strokeLinejoin="round"
                    d="M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <p className="dropzone__filename">{file.name}</p>
              <p className="dropzone__filesize">{(file.size / 1024).toFixed(1)} KB</p>
              <button
                className="dropzone__remove"
                onClick={(e) => { e.stopPropagation(); setFile(null); }}
              >
                Remove file
              </button>
            </>
          ) : (
            <>
              <div className="dropzone__icon-wrap">
                <svg fill="none" viewBox="0 0 24 24" strokeWidth={1.5}>
                  <path strokeLinecap="round" strokeLinejoin="round"
                    d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
                </svg>
              </div>
              <p className="dropzone__label">
                {isDragging ? "Drop your resume here" : "Drag & drop your resume"}
              </p>
              <p className="dropzone__sub">PDF only · Click to browse</p>
            </>
          )}
        </div>
      </div>

      {/* Job description */}
      <div>
        <label className="field-label">Job Description</label>
        <div className="textarea-wrap">
          <textarea
            className="job-textarea"
            rows={5}
            value={jobDescription}
            onChange={(e) => setJobDescription(e.target.value)}
            placeholder="Paste the full job description here — requirements, responsibilities, skills needed..."
          />
          <span className="char-count">{jobDescription.length} chars</span>
        </div>
      </div>

      {/* Submit */}
      <button
        className={`submit-btn ${canSubmit ? "submit-btn--active" : "submit-btn--disabled"}`}
        onClick={handleSubmit}
        disabled={!canSubmit}
      >
        {isLoading ? (
          <>
            <span className="spinner" />
            Analyzing resume...
          </>
        ) : (
          <>
            <svg fill="none" viewBox="0 0 24 24" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
            Analyze Resume
          </>
        )}
      </button>
    </div>
  );
}