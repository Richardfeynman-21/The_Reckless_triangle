import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Target, Activity, ShieldAlert, Zap } from 'lucide-react';

const Home = () => {
    const navigate = useNavigate();

    return (
        <>
            <div className="parallax-bg animate-zoom" style={{ backgroundImage: 'url("/assets/bgmi_user.jpg")' }} />
            <div className="page-container animate-fade-in" style={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                minHeight: '80vh',
                textAlign: 'center',
                position: 'relative'
            }}>

                <div className="glass-panel" style={{ padding: '4rem', maxWidth: '800px', width: '100%', marginBottom: '3rem' }}>
                    <h1 style={{ fontSize: '3.5rem', marginBottom: '1.5rem', fontWeight: 700 }}>
                        <span className="gradient-text">Master The Zone.</span>
                    </h1>
                    <p style={{ fontSize: '1.25rem', color: 'var(--text-secondary)', marginBottom: '3rem', lineHeight: 1.6 }}>
                        The world's first Multi-Modal PUBG Intelligence engine. <br />
                        We combine real-time Neural Network telemetry analysis with Generative AI sentiment tracking to provide you with the ultimate tactical advantage.
                    </p>

                    <button
                        className="glass-shine animate-float"
                        onClick={() => navigate('/dashboard')}
                        style={{
                            background: 'linear-gradient(135deg, var(--accent-blue), #005bb5)',
                            color: 'white',
                            padding: '1rem 2.5rem',
                            fontSize: '1.1rem',
                            fontWeight: 600,
                            borderRadius: '12px',
                            border: '1px solid rgba(255,255,255,0.2)',
                            boxShadow: '0 4px 15px var(--accent-blue-glow)',
                            transition: 'all var(--transition-fast)',
                            animation: 'pulseGlow 2s infinite'
                        }}
                        onMouseOver={(e) => e.target.style.transform = 'translateY(-2px)'}
                        onMouseOut={(e) => e.target.style.transform = 'translateY(0)'}
                    >
                        Launch Dashboard
                    </button>
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '2rem', width: '100%', maxWidth: '1000px' }}>

                    <FeatureCard
                        icon={<Activity size={32} color="var(--accent-blue)" />}
                        title="Neural Telemetry"
                        desc="Predicts your exact match outcome probability using a PyTorch dense network trained on millions of Kaggle data points."
                    />
                    <FeatureCard
                        icon={<ShieldAlert size={32} color="var(--status-danger)" />}
                        title="Toxicity LSTM"
                        desc="Monitors squad chat logs via Bidirectional LSTM to intercept destructive communication before it tilts the team."
                    />
                    <FeatureCard
                        icon={<Zap size={32} color="var(--accent-purple)" />}
                        title="Gemini Meta-Decision"
                        desc="Fuses structured game data with unstructured NLP using Google's Gemini Flash 2.5 to generate personalized coaching."
                    />

                </div>
            </div>
        </>
    );
};

const FeatureCard = ({ icon, title, desc }) => (
    <div className="glass-panel animate-float" style={{ padding: '2rem', textAlign: 'left', animationDelay: `${Math.random()}s` }}>
        <div style={{ marginBottom: '1rem' }}>{icon}</div>
        <h3 style={{ fontSize: '1.2rem', marginBottom: '0.5rem' }}>{title}</h3>
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.95rem' }}>{desc}</p>
    </div>
);

export default Home;
