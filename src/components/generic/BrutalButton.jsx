import React, { useState } from 'react';
import colors from './colors.json';
import '@fontsource/sanchez';

const BrutalButton = ({ children, tab_bg_color, colorIndex = 0 , ...rest }) => {
  const buttonStyle = {
    filter: `drop-shadow(5px 5px 0 ${tab_bg_color})`,
    backgroundColor: 'white',
    display: 'inline-block',
    padding: '0.5rem 1rem',
    border: '2px solid black',
    transition: 'all',
    transitionDuration: '0.5s',
    animation: 'ease-in-out',
    fontFamily: "'sanchez', serif",
  };

  const hoverStyle = {
    filter: `drop-shadow(3px 3px 0 white)`,
    backgroundColor: colors[3],
    color: 'white'
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