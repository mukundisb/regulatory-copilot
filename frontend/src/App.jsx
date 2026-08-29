import { useState } from 'react';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

function App() {
  const [narrative, setNarrative] = useState('');
  const [mode, setMode] = useState('assess'); // 'assess' | 'classify'
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

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
        // Handle FastAPI 422 validation or 500 error shapes
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
      // Differentiate network failure from parsed HTTP errors
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
          disabled={loading}
          style={{ padding: '10px 20px', cursor: loading ? 'not-allowed' : 'pointer', fontWeight: 'bold' }}
        >
          {loading ? 'Processing...' : mode === 'assess' ? 'Run Full Assessment' : 'Classify Event'}
        </button>
      </form>

      {/* Loading Indicator */}
      {loading && (
        <div style={{ marginTop: '20px', color: '#555' }}>
          <em>Connecting to {API_BASE_URL}... Executing ML model and vector retrieval...</em>
        </div>
      )}

      {/* Error Output */}
      {error && (
        <div style={{ marginTop: '20px', padding: '12px', background: '#ffebee', color: '#c62828', borderRadius: '4px', border: '1px solid #ef9a9a' }}>
          <strong>Error: </strong> {error}
        </div>
      )}

      {/* Results Output */}
      {result && result._mode === 'classify' && (
        <div style={{ marginTop: '20px', padding: '16px', background: '#e8f5e9', color: '#2e7d32', borderRadius: '4px' }}>
          <h3>Classification Result</h3>
          <p><strong>Predicted Label:</strong> {result.predicted_label}</p>
          <h4>Class Probabilities:</h4>
          <ul>
            {Object.entries(result.probabilities || {}).map(([label, prob]) => (
              <li key={label}><strong>{label}:</strong> {(prob * 100).toFixed(2)}%</li>
            ))}
          </ul>
        </div>
      )}

      {result && result._mode === 'assess' && (
        <div style={{ marginTop: '24px', padding: '20px', background: '#f5f7fa', border: '1px solid #dcdfe6', borderRadius: '6px' }}>
          <h3 style={{ marginTop: 0 }}>Regulatory Assessment Output</h3>
          
          <div style={{ display: 'flex', gap: '24px', marginBottom: '16px' }}>
            <p><strong>Predicted Label:</strong> <span style={{ fontSize: '1.2em', color: '#1976d2' }}>{result.predicted_label}</span></p>
            <p><strong>Model Confidence:</strong> {(result.confidence * 100).toFixed(2)}%</p>
            <p><strong>Fallback Triggered:</strong> {result.fallback_triggered ? '⚠️ Yes (Raw Query Used)' : '✅ No (Primary Query Accepted)'}</p>
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