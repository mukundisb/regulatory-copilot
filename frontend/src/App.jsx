import { useState } from 'react';

function App() {
  const [narrative, setNarrative] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch('http://localhost:8000/classify', {
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

      setResult(data);
    } catch (err) {
      // Differentiate network failure from parsed HTTP errors
      if (err.name === 'TypeError' && err.message.includes('fetch')) {
        setError('Network Error: Unable to reach FastAPI backend on http://localhost:8000. Is the server running?');
      } else {
        setError(err.message);
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: '600px', margin: '40px auto', fontFamily: 'sans-serif' }}>
      <h2>Regulatory Co-Pilot: Narrative Classification</h2>
      
      <form onSubmit={handleSubmit}>
        <div style={{ marginBottom: '12px' }}>
          <label htmlFor="narrative" style={{ display: 'block', fontWeight: 'bold', marginBottom: '6px' }}>
            Adverse Event Narrative:
          </label>
          <textarea
            id="narrative"
            rows="5"
            style={{ width: '100%', boxSizing: 'border-box', padding: '8px' }}
            placeholder="Enter clinical or device failure narrative..."
            value={narrative}
            onChange={(e) => setNarrative(e.target.value)}
            disabled={loading}
          />
        </div>

        <button 
          type="submit" 
          disabled={loading}
          style={{ padding: '8px 16px', cursor: loading ? 'not-allowed' : 'pointer' }}
        >
          {loading ? 'Classifying...' : 'Classify Event'}
        </button>
      </form>

      {/* Loading State */}
      {loading && (
        <div style={{ marginTop: '20px', color: '#555' }}>
          <em>Processing narrative through classification model...</em>
        </div>
      )}

      {/* Error Output (422 or Network Failure) */}
      {error && (
        <div style={{ marginTop: '20px', padding: '12px', background: '#ffebee', color: '#c62828', borderRadius: '4px' }}>
          <strong>Error: </strong> {error}
        </div>
      )}

      {/* Happy Path Output */}
      {result && (
        <div style={{ marginTop: '20px', padding: '12px', background: '#e8f5e9', color: '#2e7d32', borderRadius: '4px' }}>
          <h3>Classification Result</h3>
          <p><strong>Predicted Label:</strong> {result.predicted_label}</p>
          <h4>Class Probabilities:</h4>
          <ul>
            {Object.entries(result.probabilities || {}).map(([label, prob]) => (
              <li key={label}>
                <strong>{label}:</strong> {(prob * 100).toFixed(2)}%
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export default App;