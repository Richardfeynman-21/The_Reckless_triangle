import React from 'react';
import { NavLink } from 'react-router-dom';
import { Crosshair, Database, Cpu } from 'lucide-react';

const Navbar = () => {
    return (
        <nav style={{
            position: 'sticky',
            top: 0,
            zIndex: 100,
            background: 'rgba(10, 10, 12, 0.7)',
            backdropFilter: 'blur(16px)',
            WebkitBackdropFilter: 'blur(16px)',
            borderBottom: '1px solid var(--border-glass)',
            padding: '1rem 2rem'
        }}>
            <div style={{
                maxWidth: '1400px',
                margin: '0 auto',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center'
            }}>

                {/* Logo */}
                <NavLink to="/" style={{ textDecoration: 'none', display: 'flex', alignItems: 'center', gap: '0.75rem', position: 'relative', overflow: 'hidden' }} className="glass-shine">
                    <div style={{ display: 'flex', alignItems: 'center' }}>
                        <img
                            src="/assets/pubg_bg.jpg"
                            alt="Official PUBG Logo"
                            style={{ height: '28px', objectFit: 'contain', filter: 'drop-shadow(0px 2px 4px rgba(0,0,0,0.5))' }}
                        />
                    </div>
                    <span style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--text-primary)', letterSpacing: '-0.5px', borderLeft: '1px solid var(--border-glass)', paddingLeft: '0.75rem' }}>
                        AI <span style={{ color: 'var(--accent-purple)', fontWeight: 400 }}>Engine</span>
                    </span>
                </NavLink>

                {/* Links */}
                <div style={{ display: 'flex', gap: '2rem' }}>
                    <NavItem to="/dashboard" icon={<Database size={18} />} text="Live Dashboard" />
                    <NavItem to="/insights" icon={<Cpu size={18} />} text="Model Architecture" />
                </div>
            </div>
        </nav>
    );
};

const NavItem = ({ to, icon, text }) => {
    return (
        <NavLink
            to={to}
            style={({ isActive }) => ({
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem',
                textDecoration: 'none',
                color: isActive ? 'white' : 'var(--text-secondary)',
                fontWeight: isActive ? 600 : 500,
                fontSize: '0.95rem',
                transition: 'color var(--transition-fast)'
            })}
        >
            {icon}
            {text}
        </NavLink>
    );
};

export default Navbar;
