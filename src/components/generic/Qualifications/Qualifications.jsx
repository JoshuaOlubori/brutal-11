import React, { useState } from 'react';
import './Qualifications.css'

const Qualifications = () => {
  const [toggleState, setToggleState] = useState(1);

  const toggleTab = (index) => {
    setToggleState(index);
  };

  return (
    <section className="qualification section sm:w-2/4">
      <h2 className="section__title">Qualifications</h2>
      {/* <span className="section__subtitle">My Personal Journey</span> */}

      <div className="qualification__container container">
        <div className="qualification__tabs">
          <div className={
            toggleState === 1
              ? "qualification__button qualification__active button--flex"
              : "qualification__button button--flex"} onClick={() => toggleTab(1)}>
            <i className="uil uil-graduation-cap qualification__icon"></i>
            Education
          </div>

          <div
            className={toggleState === 2
              ? "qualification__button qualification__active button--flex"
              : "qualification__button button--flex"} onClick={() => toggleTab(2)}>
            <i className="uil uil-briefcase-alt qualification__icon"></i>
            Experience
          </div>
        </div>

        <div className="qualification__sections">
          <div className={toggleState === 1 ? "qualification__content qualification__content-active" : "qualification__content"}>
            <div className="qualification__data">
              <div>
                <h3 className="qualification__title"><em>Masters in Data Analytics and Technologies</em></h3>
                <span className="qualification__subtitle">The University of Bolton</span>
                <div className="qualification__calendar">
                  <i className="uil uil-calendar-alt"></i> May 2024
                </div>
              </div>

              <div>
                <span className="qualification__rounder"></span>
                <span className="qualification__line"></span>
              </div>
            </div>

            <div className="qualification__data">
              <div></div>

              <div>
                <span className="qualification__rounder"></span>
                <span className="qualification__line"></span>
              </div>

              <div>
                <h3 className="qualification__title"><em>Bachelors in Metallurgical and Materials Engineering</em></h3>
                <span className="qualification__subtitle">University of Lagos</span>
                <div className="qualification__calendar">
                  <i className="uil uil-calendar-alt"></i> Nov 2019
                </div>
              </div>


            </div>

            {/* <div className="qualification__data">
              <div>
                <h3 className="qualification__title">Pastor</h3>
                <span className="qualification__subtitle">Spain - Institue</span>
                <div className="qualification__calendar">
                  <i className="uil uil-calendar-alt"></i> 2018 - 2022
                </div>
              </div>

              <div>
                <span className="qualification__rounder"></span>
                <span className="qualification__line"></span>
              </div>
            </div> */}

            <div className="qualification__data">
              <div></div>

              <div>
                <span className="qualification__rounder"></span>
                <span className="qualification__line"></span>
              </div>

              {/* <div>
                <h3 className="qualification__title">Student</h3>
                <span className="qualification__subtitle">Spain - Institue</span>
                <div className="qualification__calendar">
                  <i className="uil uil-calendar-alt"></i> 2021 - 2018
                </div>
              </div> */}


            </div>
          </div>

          <div className={toggleState === 2 ? "qualification__content qualification__content-active" : "qualification__content"}>
            <div className="qualification__data">
              <div>
                <h3 className="qualification__title">Investment Analyst</h3>
                <span className="qualification__subtitle"><em>Grand Investments / Estates</em></span>
                <div className="qualification__calendar">
                  <i className="uil uil-calendar-alt"></i> Jun 2020 - Present
                </div>
              </div>

              <div>
                <span className="qualification__rounder"></span>
                <span className="qualification__line"></span>
              </div>
            </div>

            <div className="qualification__data">
              <div></div>

              <div>
                <span className="qualification__rounder"></span>
                <span className="qualification__line"></span>
              </div>

              <div>
                <h3 className="qualification__title">Nigeria Head of Operations and Maintenance</h3>
                <span className="qualification__subtitle"><em>Just Inc.</em></span>
                <div className="qualification__calendar">
                  <i className="uil uil-calendar-alt"></i> Dec 2019- Apr 2020
                </div>
              </div>


            </div>

            <div className="qualification__data">
              <div>
                <h3 className="qualification__title">Undergraduate Researcher</h3>
                <span className="qualification__subtitle"><em>Nigeria Railway Corporation</em></span>
                <div className="qualification__calendar">
                  <i className="uil uil-calendar-alt"></i> Jun 2018 - Nov 2018
                </div>
              </div>

              <div>
                <span className="qualification__rounder"></span>
                <span className="qualification__line"></span>
              </div>
            </div>

            <div className="qualification__data">
              <div></div>

              <div>
                <span className="qualification__rounder"></span>
                <span className="qualification__line"></span>
              </div>

              {/* <div>
                <h3 className="qualification__title">Student</h3>
                <span className="qualification__subtitle">Spain - Institue</span>
                <div className="qualification__calendar">
                  <i className="uil uil-calendar-alt"></i> 2021 - 2018
                </div>
              </div> */}


            </div>
          </div>
        </div>
      </div>
    </section>
  )
}

export default Qualifications