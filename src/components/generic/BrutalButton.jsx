import React, { useState } from 'react';
import colors from './colors.json'; // Assuming colors.json is accessible

const BrutalButton = ({ children, color = colors[Math.floor(Math.random() * colors.length)], ...rest }) => {
  const buttonStyle = {
    filter: 'drop-shadow(5px 5px 0 rgb(0 0 0 / 1))',
    backgroundColor: 'white',
    display: 'inline-block',
    padding: '0.5rem 1rem',
    border: '2px solid black',
    transition: 'all',
    transitionDuration: '0.5s',
    animation: 'ease-in-out',
    fontFamily: "'Sanchez', serif",
  };

  const hoverStyle = {
    filter: 'drop-shadow(3px 3px 0 rgb(0 0 0 / 1))',
    backgroundColor: color,
  };

  const [style, setStyle] = useState(buttonStyle);

  const handleMouseOver = () => {
    setStyle({ ...buttonStyle, ...hoverStyle });
  };

  const handleMouseOut = () => {
    setStyle(buttonStyle);
  };

  return (
    <a
      {...rest}
      className="brutal-btn"
      style={style}
      onMouseOver={handleMouseOver}
      onMouseOut={handleMouseOut}
    >
      {children}
    </a>
  );
};

export default BrutalButton;