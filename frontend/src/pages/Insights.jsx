import React, { useState, useEffect } from 'react';
import { Network, TrendingDown, BookOpen } from 'lucide-react';

const Insights = () => {
    return (
        <div className="page-container animate-fade-in" style={{ paddingBottom: '4rem' }}>

            <div style={{ marginBottom: '3rem', textAlign: 'center' }}>
                <h1 style={{ fontSize: '2.5rem', marginBottom: '1rem' }}><span className="gradient-text">Architecture Insights</span></h1>
                <p style={{ color: 'var(--text-secondary)', maxWidth: '600px', margin: '0 auto' }}>
                    Explore the mechanics behind the Multi-Modal PUBG prediction engine.
                </p>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem', marginBottom: '3rem' }}>

                {/* Model Spec Card */}
                <div className="glass-panel" style={{ padding: '2rem' }}>
                    <h2 style={{ fontSize: '1.25rem', marginBottom: '1.5rem', display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                        <Network size={20} color="var(--accent-blue)" /> The Middle-Fusion Network
                    </h2>
                    <p style={{ color: 'var(--text-secondary)', marginBottom: '1.5rem', lineHeight: 1.6 }}>
                        Instead of running two separate models, our custom PyTorch architecture seamlessly concatenates structured logic with unstructured text sentiment before making a ruling.
                    </p>

                    <div style={{ background: 'rgba(0,0,0,0.3)', padding: '1.5rem', borderRadius: '12px', border: '1px solid var(--border-glassStrong)', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-glass)', paddingBottom: '0.5rem' }}>
                            <span style={{ color: 'var(--text-secondary)' }}>Game Embedding Pipeline</span>
                            <span style={{ fontWeight: 600 }}>128 → 64 → 32</span>
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-glass)', paddingBottom: '0.5rem' }}>
                            <span style={{ color: 'var(--text-secondary)' }}>Chat LSTM (Bidirectional) Pipeline</span>
                            <span style={{ fontWeight: 600 }}>Vocab 20k → 128 → 64</span>
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                            <span style={{ color: 'var(--accent-purple)', fontWeight: 600 }}>Unified Fusion Head</span>
                            <span className="gradient-text" style={{ fontWeight: 700 }}>160 → 64 → 1</span>
                        </div>
                    </div>
                </div>

                {/* Behavior Simulation Header */}
                <div className="glass-panel" style={{ padding: '2rem', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                    <h2 style={{ fontSize: '1.25rem', marginBottom: '1.5rem', display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                        <BookOpen size={20} color="var(--accent-purple)" /> Model Behavior
                    </h2>
                    <p style={{ color: 'var(--text-secondary)', fontSize: '1rem', lineHeight: 1.6 }}>
                        The joint neural network learned a distinct mathematical correlation from the Kaggle dataset: <strong>teams that exhibit toxic behavior underperform mechanically.</strong>
                        <br /><br />
                        Even if a player's Damage and Kills are statistically high, the presence of hostile team communications (detected via LSTM sequences) will exponentially heavily penalize the predicted Win Probability. Read the chart below to simulate this dynamic.
                    </p>
                </div>

            </div>

            {/* Historical Trend Chart Component */}
            <TrendSimulator />

        </div>
    );
};

// Animated Custom Chart Component
const TrendSimulator = () => {
    const [running, setRunning] = useState(false);
    const [progress, setProgress] = useState(0);

    useEffect(() => {
        let interval;
        if (running && progress < 100) {
            interval = setInterval(() => {
                setProgress(p => Math.min(p + 2, 100));
            }, 100);
        } else if (progress >= 100) {
            setRunning(false);
        }
        return () => clearInterval(interval);
    }, [running, progress]);

    const playSimulation = () => {
        setProgress(0);
        setRunning(true);
    };

    // Synthesized curve logic modeling toxicity dragging down probability over time
    const toxicityCurve = (t) => t < 30 ? 10 : (t < 60 ? 10 + (t - 30) * 1.5 : 55 + (t - 60) * 0.5);
    const probCurve = (t) => t < 30 ? 65 : (t < 50 ? 65 - (t - 30) * 1.2 : 41 - (t - 50) * 0.6);

    const currentTox = Math.round(toxicityCurve(progress));
    const currentProb = Math.round(probCurve(progress));

    return (
        <div className="glass-panel" style={{ padding: '2rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2rem' }}>
                <h2 style={{ fontSize: '1.25rem', display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                    <TrendingDown size={20} color="var(--status-danger)" /> Historical Trend Simulation
                </h2>
                <button
                    onClick={playSimulation}
                    style={{
                        padding: '0.5rem 1.5rem',
                        background: 'rgba(255,255,255,0.1)',
                        color: 'white',
                        border: '1px solid var(--border-glassStrong)',
                        borderRadius: '8px',
                        fontWeight: 600,
                        transition: 'background var(--transition-fast)'
                    }}
                    onMouseOver={e => e.target.style.background = 'rgba(255,255,255,0.2)'}
                    onMouseOut={e => e.target.style.background = 'rgba(255,255,255,0.1)'}
                >
                    {running ? 'Simulating...' : 'Run Analysis Playback'}
                </button>
            </div>

            <div style={{ display: 'flex', gap: '4rem', alignItems: 'flex-start' }}>
                {/* Visual Graph Axis Concept */}
                <div style={{ flex: 1, height: '200px', background: 'rgba(0,0,0,0.2)', borderRadius: '12px', border: '1px solid var(--border-glass)', position: 'relative', overflow: 'hidden' }}>

                    {/* Time indicator line */}
                    <div style={{ position: 'absolute', top: 0, bottom: 0, left: `${progress}%`, width: '2px', background: 'rgba(255,255,255,0.5)', zIndex: 10, transition: 'left 0.1s linear' }} />

                    {/* Simulated SVG Curves */}
                    <svg width="100%" height="100%" preserveAspectRatio="none" style={{ position: 'absolute', top: 0, left: 0 }}>
                        <path d="M 0 180 C 300 180, 400 90, 1000 90" fill="none" stroke="var(--status-danger)" strokeWidth="3" opacity="0.6" />
                        <path d="M 0 70 C 300 70, 400 120, 1000 120" fill="none" stroke="var(--accent-blue)" strokeWidth="3" opacity="0.6" />
                    </svg>

                    <div style={{ position: 'absolute', bottom: '1rem', right: '1rem', display: 'flex', gap: '1rem', fontSize: '0.75rem', fontWeight: 600 }}>
                        <span style={{ color: 'var(--accent-blue)' }}>--- Win Probability</span>
                        <span style={{ color: 'var(--status-danger)' }}>--- Toxicity Output</span>
                    </div>
                </div>

                {/* Live Readouts */}
                <div style={{ width: '200px', display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                    <div>
                        <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginBottom: '0.25rem' }}>Live Toxicity Spike</p>
                        <div style={{ fontSize: '2rem', fontWeight: 700, color: currentTox > 40 ? 'var(--status-danger)' : 'white' }}>{currentTox}%</div>
                    </div>
                    <div>
                        <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginBottom: '0.25rem' }}>Projected Win Rate</p>
                        <div style={{ fontSize: '2rem', fontWeight: 700, className: "gradient-text" }}>{currentProb}%</div>
                    </div>
                </div>
            </div>

        </div>
    );
};

export default Insights;
