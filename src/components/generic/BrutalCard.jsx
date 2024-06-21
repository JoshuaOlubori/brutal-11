
import colors from './colors.json'; // Assuming colors.json is accessible
import React, { useState } from 'react';

const BrutalCard = ({ children, color = colors[Math.floor(Math.random() * colors.length)] }) => {
  const [cardStyle, setCardStyle] = useState({
    backgroundColor: color,
    borderRadius: '0.5rem',
    border: '3px solid black',
    filter: 'drop-shadow(7px 7px 0 rgb(0 0 0 / 1))',
    transition: 'all',
    padding: '1rem',
    transitionDuration: '0.5s',
    animation: 'ease-in-out',
  });

  const handleMouseOver = () => {
    setCardStyle({
      ...cardStyle,
      filter: 'drop-shadow(5px 5px 0 rgb(0 0 0 / 1))',
    });
  };

  return (
    <div className="brutal-card" style={cardStyle} onMouseOver={handleMouseOver}>
      {children}
    </div>
  );
};

export default BrutalCard;






// const BrutalCard = ({ children, color = colors[Math.floor(Math.random() * colors.length)] }) => {
//   const [cardStyle, setCardStyle] = useState({
//     backgroundColor: color,
//     borderRadius: '0.5rem',
//     border: '3px solid black',
//     filter: 'drop-shadow(7px 7px 0 rgb(0 0 0 / 1))',
//     transition: 'all',
//     padding: '1rem',
//     transitionDuration: '0.5s',
//     animation: 'ease-in-out',
//   });

//   const handleMouseOver = () => {
//     setCardStyle({
//       ...cardStyle,
//       filter: 'drop-shadow(5px 5px 0 rgb(0 0 0 / 1))',
//     });
//   };

//   return (
//     <div className="brutal-card" style={cardStyle} onMouseOver={handleMouseOver}>
//       {children}
//     </div>
//   );
// };
