import { useState, useEffect } from 'react';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

function ColdStartLoadingIndicator({ elapsedSeconds }) {
  // Asymptotic progress bar: moves swiftly to ~70% over 25s, then decelerates toward 95%,
  // leaving room for longer provisioning windows up to the 90s server timeout.
  const progressPercent = Math.min(
    Math.round((1 - Math.exp(-elapsedSeconds / 22)) * 95),
    95
  );

  // Phased messaging based on realistic container lifecycle stages
  const getPhaseDetails = () => {
    if (elapsedSeconds < 3) {
      return {
        title: "Executing ClinicalBERT inference...",
        detail: null,
      };
    }
    if (elapsedSeconds < 25) {
      return {
        title: "Warming up Scale-to-Zero model container on Vertex AI...",
        detail: "The endpoint scales to zero instances when idle to conserve compute. Initial provisioning typically takes 20–45 seconds; subsequent requests respond in sub-second time.",
      };
    }
    if (elapsedSeconds < 55) {
      return {
        title: "Container provisioned; loading ClinicalBERT weights into memory...",
        detail: "Initial model page-in and gRPC channel negotiation are underway. Please keep this window open.",
      };
    }
    return {
      title: "Extending retry backoff window...",
      detail: "Cold start is taking longer than usual. The system will automatically fall back to the in-memory TF-IDF model if initialization exceeds timeout limits.",
    };
  };

  const { title, detail } = getPhaseDetails();

  return (
    <div
      style={{
        marginTop: "20px",
        padding: "16px 20px",
        borderRadius: "8px",
        backgroundColor: "#f8fafc",
        border: "1px solid #cbd5e1",
        boxShadow: "0 2px 4px rgba(0,0,0,0.05)",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
        <span
          style={{
            display: "inline-block",
            width: "14px",
            height: "14px",
            border: "2px solid #2563eb",
            borderTopColor: "transparent",
            borderRadius: "50%",
            animation: "spin 1s linear infinite",
          }}
        />
        <strong style={{ color: "#0f172a", fontSize: "14px" }}>
          {title}
        </strong>
      </div>

      {detail && (
        <div style={{ marginTop: "12px" }}>
          <p
            style={{
              margin: "0 0 8px 0",
              fontSize: "13px",
              color: "#475569",
              lineHeight: 1.4,
            }}
          >
            {detail}
          </p>

          <div
            style={{
              height: "6px",
              width: "100%",
              backgroundColor: "#e2e8f0",
              borderRadius: "4px",
              overflow: "hidden",
            }}
          >
            <div
              style={{
                width: `${progressPercent}%`,
                height: "100%",
                backgroundColor: "#2563eb",
                transition: "width 1s ease",
              }}
            />
          </div>

          <div
            style={{
              marginTop: "6px",
              display: "flex",
              justifyContent: "space-between",
              fontSize: "11px",
              color: "#64748b",
            }}
          >
            <span>Elapsed: {elapsedSeconds}s</span>
            <span>{elapsedSeconds > 45 ? "Awaiting completion or fallback" : "Provisioning..."}</span>
          </div>
        </div>
      )}

      <style>{`
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
}

function App() {
  const [narrative, setNarrative] = useState('');
  const [mode, setMode] = useState('assess'); // 'assess' | 'classify'
  const [loading, setLoading] = useState(false);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    let timer;
    if (loading) {
      setElapsedSeconds(0);
      timer = setInterval(() => {
        setElapsedSeconds((prev) => prev + 1);
      }, 1000);
    }
    return () => clearInterval(timer);
  }, [loading]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    const endpoint = mode === 'assess' ? '/assess' : '/classify';

    try {
      const response = await fetch(`${API_BASE_URL}${endpoint}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ narrative }),
      });

      const data = await response.json();

      if (!response.ok) {
        if (response.status === 422 && Array.isArray(data.detail)) {
          const validationMsg = data.detail
            .map((err) => `${err.loc?.slice(1).join('.')} ${err.msg}`)
            .join('; ');
          throw new Error(`Validation Error (422): ${validationMsg}`);
        }
        throw new Error(data.detail || `Server Error (${response.status})`);
      }

      setResult({ ...data, _mode: mode });
    } catch (err) {
      if (err.name === 'TypeError' && err.message.includes('fetch')) {
        setError(`Network Error: Unable to reach FastAPI backend at ${API_BASE_URL}. Is the server running?`);
      } else {
        setError(err.message);
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: '800px', margin: '40px auto', fontFamily: 'system-ui, sans-serif', padding: '0 16px' }}>
      <h2>Regulatory Co-Pilot: Adverse Event Triage & Assessment</h2>
      
      <form onSubmit={handleSubmit}>
        <div style={{ marginBottom: '16px' }}>
          <label style={{ display: 'block', fontWeight: 'bold', marginBottom: '8px' }}>
            Action Mode:
          </label>
          <div style={{ display: 'flex', gap: '16px' }}>
            <label style={{ cursor: 'pointer' }}>
              <input
                type="radio"
                value="assess"
                checked={mode === 'assess'}
                onChange={() => setMode('assess')}
                disabled={loading}
              />
              <strong> Full Regulatory Assessment (/assess)</strong> (Recommended)
            </label>
            <label style={{ cursor: 'pointer' }}>
              <input
                type="radio"
                value="classify"
                checked={mode === 'classify'}
                onChange={() => setMode('classify')}
                disabled={loading}
              />
              Quick Classification (/classify)
            </label>
          </div>
        </div>

        <div style={{ marginBottom: '12px' }}>
          <label htmlFor="narrative" style={{ display: 'block', fontWeight: 'bold', marginBottom: '6px' }}>
            Adverse Event Narrative:
          </label>
          <textarea
            id="narrative"
            rows="5"
            style={{ width: '100%', boxSizing: 'border-box', padding: '10px', fontSize: '14px' }}
            placeholder="e.g. Infusion pump screen froze and stopped delivery of medication, showing error code E-402."
            value={narrative}
            onChange={(e) => setNarrative(e.target.value)}
            disabled={loading}
          />
        </div>

        <button 
          type="submit" 
          disabled={loading || !narrative.trim()} 
          style={{ padding: '10px 20px', cursor: loading || !narrative.trim() ? 'not-allowed' : 'pointer', fontWeight: 'bold' }}
        >
          {loading ? 'Processing...' : mode === 'assess' ? 'Run Full Assessment' : 'Classify Event'}
        </button>
      </form>

      {/* Phased Dynamic Cold-Start Indicator */}
      {loading && <ColdStartLoadingIndicator elapsedSeconds={elapsedSeconds} />}

      {/* Warning Alert if Fallback Occurred */}
      {result?.warning && (
        <div
          style={{
            marginTop: "16px",
            padding: "10px 14px",
            backgroundColor: "#fffbeb",
            border: "1px solid #fef08a",
            borderRadius: "6px",
            color: "#854d0e",
            fontSize: "13px",
          }}
        >
          ⚠️ <strong>Notice:</strong> {result.warning}
        </div>
      )}

      {/* Error Output */}
      {error && (
        <div style={{ marginTop: '20px', padding: '12px', background: '#ffebee', color: '#c62828', borderRadius: '4px', border: '1px solid #ef9a9a' }}>
          <strong>Error: </strong> {error}
        </div>
      )}

      {/* Results Output: Quick Classification */}
      {result && result._mode === 'classify' && (
        <div style={{ marginTop: '20px', padding: '16px', background: '#e8f5e9', color: '#2e7d32', borderRadius: '4px' }}>
          <h3>Classification Result</h3>
          <p>
            <strong>Predicted Label:</strong> {result.predicted_label}
            {result.backend_used && (
              <span style={{ marginLeft: '12px', fontSize: '12px', background: '#c8e6c9', color: '#1b5e20', padding: '2px 8px', borderRadius: '4px' }}>
                {result.backend_used}
              </span>
            )}
          </p>
          {/* Falsy-zero bug fixed with != null */}
          {result.confidence != null && (
            <p><strong>Confidence:</strong> {(result.confidence * 100).toFixed(2)}%</p>
          )}
          <h4>Class Probabilities:</h4>
          <ul>
            {Object.entries(result.probabilities || {}).map(([label, prob]) => (
              <li key={label}><strong>{label}:</strong> {(prob * 100).toFixed(2)}%</li>
            ))}
          </ul>
        </div>
      )}

      {/* Results Output: Full Assessment */}
      {result && result._mode === 'assess' && (
        <div style={{ marginTop: '24px', padding: '20px', background: '#f5f7fa', border: '1px solid #dcdfe6', borderRadius: '6px' }}>
          <h3 style={{ marginTop: 0 }}>Regulatory Assessment Output</h3>
          
          <div style={{ display: 'flex', gap: '24px', marginBottom: '16px', flexWrap: 'wrap', alignItems: 'center' }}>
            <p style={{ margin: 0 }}>
              <strong>Predicted Label:</strong> <span style={{ fontSize: '1.2em', color: '#1976d2', fontWeight: 'bold' }}>{result.predicted_label}</span>
            </p>
            {result.confidence != null && (
              <p style={{ margin: 0 }}><strong>Model Confidence:</strong> {(result.confidence * 100).toFixed(2)}%</p>
            )}
            {result.backend_used && (
              <p style={{ margin: 0 }}>
                <span style={{ fontSize: '12px', background: '#e0e7ff', color: '#3730a3', padding: '2px 8px', borderRadius: '4px' }}>
                  {result.backend_used}
                </span>
              </p>
            )}
            <p style={{ margin: 0 }}><strong>Fallback Triggered:</strong> {result.fallback_triggered ? '⚠️ Yes (Raw Query Used)' : '✅ No (Primary Query Accepted)'}</p>
          </div>

          <div style={{ background: '#fff', padding: '12px', borderRadius: '4px', marginBottom: '16px', border: '1px solid #e0e0e0' }}>
            <strong>Recommendation:</strong>
            <p style={{ margin: '8px 0 0 0', lineHeight: 1.5 }}>{result.recommendation}</p>
          </div>

          <div style={{ marginBottom: '16px' }}>
            <strong>Query Steered to Vector Store:</strong>
            <pre style={{ background: '#eceff1', padding: '8px', borderRadius: '4px', whiteSpace: 'pre-wrap', fontSize: '12px' }}>
              {result.retrieval_query_used}
            </pre>
          </div>

          <div>
            <strong>Retrieved Regulatory Citations ({result.retrieved_chunks?.length || 0}):</strong>
            {result.retrieved_chunks?.map((chunk, i) => (
              <div key={chunk.chunk_id || i} style={{ marginTop: '10px', padding: '12px', background: '#fff', borderLeft: '4px solid #1976d2', borderRadius: '2px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
                  <strong>{chunk.section || 'General Provision'}</strong>
                  <span style={{ fontSize: '12px', color: '#666' }}>Similarity: {(chunk.similarity_score * 100).toFixed(2)}%</span>
                </div>
                <p style={{ margin: 0, fontSize: '13px', color: '#333', whiteSpace: 'pre-wrap' }}>{chunk.text}</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default App;