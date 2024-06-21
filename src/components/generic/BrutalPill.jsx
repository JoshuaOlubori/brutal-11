import React from 'react';
import colors from './colors.json'; // Assuming colors.json is accessible

const BrutalPill = ({ children, color = colors[Math.floor(Math.random() * colors.length)] }) => {
  const pillStyle = {
    filter: 'drop-shadow(3px 3px 0 rgb(0 0 0 / 1))',
    userSelect: 'none',
    backgroundColor: 'white',
    borderRadius: '9999px',
    border: '2px solid black',
    padding: '0.25rem 0.75rem',
    transition: 'all',
    transitionDuration: '0.5s',
    animation: 'ease-in-out',
    fontSize: 'small',
  };

  const hoverStyle = {
    filter: 'drop-shadow(5px 5px 0 rgb(0 0 0 / 1))',
    backgroundColor: color,
  };

  return (
    <span
      className="brutal-pill"
      style={{ ...pillStyle, ...(color ? hoverStyle : {}) }}
    >
      {children}
    </span>
  );
};

export default BrutalPill;