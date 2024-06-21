
// https://stackoverflow.com/questions/77870679/simple-typed-js-and-react-useeffect-not-running-in-astro-project

import React from 'react';
import Typed from 'typed.js';




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
      <p className='mt-4 outfit text-2xl md:text-5xl lg:text-7xl'>
              I am a<span
                className='text-white' ref={el}
           ></span>
            </p>
            <p className='mt-2 outfit text-xl md:text-3xl lg:text-5xl'>
            Data geek by day, superhero by night.
            </p>
        </>
    );
  };
  export { Hero };