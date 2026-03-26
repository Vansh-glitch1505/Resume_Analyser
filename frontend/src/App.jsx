import { useState } from "react";
import Upload from "./components/Upload";
import Dashboard from "./components/Dashboard";
import "./App.css";

export default function App() {
  const [analysisData, setAnalysisData] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleAnalysisComplete = (data) => {
    setAnalysisData(data);
    setError(null);
  };

  const handleReset = () => {
    setAnalysisData(null);
    setError(null);
  };

  return (
    <div className="app-wrapper">
      <div className="app-bg">
        <div className="app-bg__orb app-bg__orb--violet" />
        <div className="app-bg__orb app-bg__orb--cyan" />
        <div className="app-bg__orb app-bg__orb--fuchsia" />
        <div className="app-bg__grid" />
      </div>

      <header className="header">
        <div className="header__inner">
          <div className="header__logo">
            <div className="header__icon">
              <svg fill="none" viewBox="0 0 24 24" strokeWidth={2.5}>
                <path strokeLinecap="round" strokeLinejoin="round"
                  d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
            </div>
            <span className="header__name">ResumeIQ</span>
          </div>
          {analysisData && (
            <button className="header__reset-btn" onClick={handleReset}>
              <svg width="12" height="12" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round"
                  d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
              New Analysis
            </button>
          )}
        </div>
      </header>

      <main className="main">
        {!analysisData ? (
          <div className="upload-page">
            <div className="upload-hero">
              <h1 className="upload-hero__title">
                Know exactly where<br />
                <span>you stand.</span>
              </h1>
              <p className="upload-hero__sub">
                Upload your resume and paste a job description. Get instant insights on skill gaps, match scores, and what to improve.
              </p>
            </div>
            <Upload
              onAnalysisComplete={handleAnalysisComplete}
              isLoading={isLoading}
              setIsLoading={setIsLoading}
              setError={setError}
            />
            {error && (
              <div className="error-box" style={{ marginTop: 14 }}>
                <svg fill="none" viewBox="0 0 24 24" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round"
                    d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                {error}
              </div>
            )}
          </div>
        ) : (
          <Dashboard data={analysisData} />
        )}
      </main>
    </div>
  );
}