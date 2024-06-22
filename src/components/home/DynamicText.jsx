
// https://stackoverflow.com/questions/77870679/simple-typed-js-and-react-useeffect-not-running-in-astro-project

import React from 'react';
import Typed from 'typed.js';
// import './dtstyles.css';
// Supports weights 100-900
import '@fontsource-variable/outfit';





  const Hero = () => {
    const el = React.useRef(null);
  
    React.useEffect(() => {
      const typed = new Typed(el.current, {
        strings: ['<i> Data</i> Scientist.', '<i> ML</i> Engineer.'],
        typeSpeed: 100,
        backSpeed: 45,
        loop: true
      });
  
      return () => {
        typed.destroy();
      };
    });
    return (
<>
      <p className=' tc0 mt-4 outfit text-2xl md:text-5xl lg:text-7xl'>
              I am a <span
                className='tc0' ref={el}
           ></span>
            </p>
            <p className='tc1 mt-2 outfit text-xl md:text-3xl lg:text-5xl'>
            Distilling the essence of your data.
            </p>
        </>
    );
  };
  export { Hero };