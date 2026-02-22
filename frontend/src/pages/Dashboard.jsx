import React, { useState } from 'react';
import axios from 'axios';
import { ShieldAlert, Zap, Cpu, Activity, Send, Target } from 'lucide-react';

const defaultStats = {
    assists: 1, boosts: 2, damageDealt: 150.0, DBNOs: 1,
    headshotKills: 1, heals: 1, killPlace: 50, killPoints: 1000,
    kills: 2, killStreaks: 1, longestKill: 50.0, matchDuration: 1800,
    maxPlace: 100, numGroups: 25, rankPoints: -1, revives: 0,
    rideDistance: 1000.0, roadKills: 0, swimDistance: 0.0,
    teamKills: 0, vehicleDestroys: 0, walkDistance: 2000.0,
    weaponsAcquired: 4, winPoints: 1500
};

const Dashboard = () => {
    const [stats, setStats] = useState(defaultStats);
    const [showAdvanced, setShowAdvanced] = useState(false);
    const [chatLogs, setChatLogs] = useState("Hey team, drop school?\nOMG you guys are trash, literal bots.");
    const [loading, setLoading] = useState(false);
    const [report, setReport] = useState(null);

    const handleAnalyze = async () => {
        setLoading(true);
        try {
            const messages = chatLogs.split('\n').filter(m => m.trim() !== '');
            const res = await axios.post('http://localhost:8000/api/analyze', {
                stats: stats,
                chatLogs: messages
            });
            setReport(res.data);
        } catch (error) {
            console.error(error);
            alert("Error connecting to the PyTorch/Gemini backend. Is FastAPI running?");
        }
        setLoading(false);
    };

    const handleStatChange = (key, val) => {
        setStats(prev => ({ ...prev, [key]: Number(val) }));
    };

    return (
        <>
            <div className="parallax-bg" style={{
                backgroundImage: 'url("/assets/bgmi_user.jpg")',
                opacity: 0.12,
                maskImage: 'none',
                WebkitMaskImage: 'none',
                zIndex: -2
            }} />
            <div className="page-container animate-fade-in" style={{ position: 'relative' }}>
                <div style={{ marginBottom: '2rem' }}>
                    <h1 style={{ fontSize: '2.5rem' }}>Live <span className="gradient-text">Match Intelligence</span></h1>
                    <p style={{ color: 'var(--text-secondary)' }}>Enter live telemetry and comms to evaluate probability of survival.</p>
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1.5fr) minmax(0, 1fr)', gap: '2rem' }}>

                    {/* Left Column: Inputs */}
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>

                        <div className="glass-panel animate-float" style={{ padding: '2rem', animationDelay: '0s' }}>
                            <h2 style={{ fontSize: '1.25rem', marginBottom: '1.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                                <Activity size={20} color="var(--accent-blue)" /> Gameplay Telemetry
                            </h2>
                            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', maxHeight: showAdvanced ? '1000px' : '300px', overflowY: 'hidden', transition: 'max-height var(--transition-slow)' }}>
                                {(showAdvanced ? Object.keys(defaultStats) : ['damageDealt', 'kills', 'heals', 'boosts', 'walkDistance', 'killPlace']).map(key => (
                                    <div key={key} style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                                        <label style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', textTransform: 'capitalize' }}>
                                            {key.replace(/([A-Z])/g, ' $1').trim()}
                                        </label>
                                        <input
                                            type="number"
                                            value={stats[key]}
                                            onChange={(e) => handleStatChange(key, e.target.value)}
                                            style={{
                                                background: 'rgba(0,0,0,0.3)',
                                                border: '1px solid var(--border-glassStrong)',
                                                color: 'white',
                                                padding: '0.75rem',
                                                borderRadius: '8px',
                                                fontSize: '1rem',
                                                outline: 'none'
                                            }}
                                        />
                                    </div>
                                ))}
                            </div>

                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '1.5rem' }}>
                                <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                                    {showAdvanced ? '*Showing all 24 active PyTorch features.' : '*Showing 6 of 24 active PyTorch features for demo simplicity.'}
                                </p>
                                <button
                                    onClick={() => setShowAdvanced(!showAdvanced)}
                                    style={{
                                        background: 'transparent',
                                        color: 'var(--accent-blue)',
                                        border: '1px solid var(--border-glass)',
                                        padding: '0.5rem 1rem',
                                        borderRadius: '8px',
                                        fontSize: '0.85rem',
                                        fontWeight: 500,
                                        transition: 'all var(--transition-fast)'
                                    }}
                                    className="glass-shine"
                                >
                                    {showAdvanced ? 'Hide Advanced Options' : 'Show Advanced Options'}
                                </button>
                            </div>
                        </div>

                        <div className="glass-panel animate-float" style={{ padding: '2rem', animationDelay: '0.5s' }}>
                            <h2 style={{ fontSize: '1.25rem', marginBottom: '1.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                                <ShieldAlert size={20} color="var(--accent-purple)" /> Squad Communications
                            </h2>
                            <textarea
                                value={chatLogs}
                                onChange={(e) => setChatLogs(e.target.value)}
                                rows={4}
                                style={{
                                    width: '100%',
                                    background: 'rgba(0,0,0,0.3)',
                                    border: '1px solid var(--border-glassStrong)',
                                    color: 'white',
                                    padding: '1rem',
                                    borderRadius: '8px',
                                    fontSize: '0.95rem',
                                    outline: 'none',
                                    resize: 'none',
                                    fontFamily: 'inherit'
                                }}
                            />
                        </div>

                        <button
                            className="glass-shine animate-float"
                            onClick={handleAnalyze}
                            disabled={loading}
                            style={{
                                background: 'linear-gradient(135deg, var(--accent-blue), var(--accent-purple))',
                                color: 'white',
                                padding: '1rem',
                                fontSize: '1.1rem',
                                fontWeight: 600,
                                borderRadius: '12px',
                                border: 'none',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                gap: '0.75rem',
                                opacity: loading ? 0.7 : 1,
                                boxShadow: '0 4px 15px rgba(191, 90, 242, 0.3)'
                            }}
                        >
                            {loading ? 'Initializing Neural Inference...' : 'Analyze Pipeline'}
                            {!loading && <Send size={18} />}
                        </button>

                    </div>

                    {/* Right Column: Output Report */}
                    <div style={{ display: 'flex', flexDirection: 'column' }}>
                        <IntelligenceReport report={report} loading={loading} />
                    </div>
                </div>
            </div>
        </>
    );
};

const IntelligenceReport = ({ report, loading }) => {
    if (loading) {
        return (
            <div className="glass-panel animate-fade-in" style={{ padding: '2rem', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '1rem' }}>
                    <div style={{ width: '40px', height: '40px', borderRadius: '50%', border: '3px solid var(--border-glassStrong)', borderTopColor: 'var(--accent-blue)', animation: 'spin 1s linear infinite' }} />
                    <p style={{ color: 'var(--text-secondary)' }}>Awaiting Gemini Meta-Decision...</p>
                </div>
                <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
            </div>
        );
    }

    if (!report) {
        return (
            <div className="glass-panel" style={{ padding: '2rem', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', opacity: 0.5 }}>
                <div style={{ textAlign: 'center' }}>
                    <Cpu size={48} color="var(--text-secondary)" style={{ marginBottom: '1rem' }} />
                    <p>Run Pipeline to view PyTorch Output</p>
                </div>
            </div>
        );
    }

    const winPct = (report.prediction * 100).toFixed(1);
    const toxPct = (report.toxicity * 100).toFixed(1);

    return (
        <div className="glass-panel animate-fade-in animate-float" style={{ padding: '2rem', height: '100%', display: 'flex', flexDirection: 'column', gap: '2rem', animationDelay: '0.2s' }}>

            <div>
                <h2 style={{ fontSize: '1.5rem', marginBottom: '1.5rem', display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                    <Zap size={24} color="var(--status-warning)" /> Threat Assessment
                </h2>

                {/* Gauges */}
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem', marginBottom: '2rem' }}>

                    <div style={{ background: 'rgba(0,0,0,0.2)', padding: '1.5rem', borderRadius: '12px', border: '1px solid var(--border-glass)' }}>
                        <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginBottom: '0.5rem' }}>Win Probability</p>
                        <div style={{ fontSize: '2rem', fontWeight: 700, marginBottom: '0.5rem' }}>{winPct}%</div>
                        <div style={{ height: '6px', background: 'rgba(255,255,255,0.1)', borderRadius: '3px', overflow: 'hidden' }}>
                            <div style={{ height: '100%', width: `${winPct}%`, background: 'var(--accent-blue)', transition: 'width 1s ease-out' }} />
                        </div>
                    </div>

                    <div style={{ background: 'rgba(0,0,0,0.2)', padding: '1.5rem', borderRadius: '12px', border: '1px solid var(--border-glass)' }}>
                        <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginBottom: '0.5rem' }}>Comms Toxicity</p>
                        <div style={{ fontSize: '2rem', fontWeight: 700, marginBottom: '0.5rem', color: report.toxicity > 0.5 ? 'var(--status-danger)' : 'white' }}>{toxPct}%</div>
                        <div style={{ height: '6px', background: 'rgba(255,255,255,0.1)', borderRadius: '3px', overflow: 'hidden' }}>
                            <div style={{ height: '100%', width: `${toxPct}%`, background: report.toxicity > 0.5 ? 'var(--status-danger)' : 'var(--status-success)', transition: 'width 1s ease-out' }} />
                        </div>
                    </div>

                </div>

                {/* Playstyle Badge */}
                <div style={{ background: 'linear-gradient(90deg, rgba(10,132,255,0.1), rgba(191,90,242,0.1))', border: '1px solid var(--accent-blue-glow)', padding: '1rem 1.5rem', borderRadius: '12px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2rem' }}>
                    <div>
                        <p style={{ fontSize: '0.85rem', color: 'var(--accent-blue)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '1px' }}>Detected Persona</p>
                        <p style={{ fontSize: '1.5rem', fontWeight: 700 }} className="gradient-text">{report.persona}</p>
                    </div>
                    <Target size={32} color="var(--accent-purple)" opacity={0.8} />
                </div>

                {/* Gemini Feedback */}
                <div style={{ background: 'rgba(0,0,0,0.4)', padding: '1.5rem', borderRadius: '12px', borderLeft: '4px solid var(--accent-purple)' }}>
                    <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <Cpu size={14} /> Gemini AI Coach Overwrite
                    </p>
                    <p style={{ fontSize: '1rem', lineHeight: 1.6 }}>{report.feedback}</p>
                </div>

            </div>
        </div>
    );
};

export default Dashboard;
