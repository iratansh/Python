import React, { useState, useRef } from "react";
import Navbar from "./Navbar";
import {
  Dropdown,
  DropdownTrigger,
  DropdownMenu,
  DropdownItem,
} from "@nextui-org/dropdown";
import SendRequestToPython from "./SendRequestToPython";

const Home = () => {
  const [selectedStock, setSelectedStock] = useState(null);
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  const [triggerPrediction, setTriggerPrediction] = useState(false);

  const aboutRef = useRef(null);
  const servicesRef = useRef(null);
  const contactRef = useRef(null);

  const scrollToSection = (section) => {
    if (section === "about") {
      aboutRef.current.scrollIntoView({ behavior: "smooth" });
    } else if (section === "services") {
      servicesRef.current.scrollIntoView({ behavior: "smooth" });
    } else if (section === "contact") {
      contactRef.current.scrollIntoView({ behavior: "smooth" });
    }
  };

  const handlePredictClick = () => {
    setTriggerPrediction(true);
  };

  return (
    <div className="min-h-screen bg-gray-100">
      <Navbar scrollToSection={scrollToSection} />
      <main className="pt-20 max-w-4xl mx-auto p-4">
        <h1 className="text-3xl font-bold mb-4 home text-gray-700 header">
          Select a Stock to Forecast
        </h1>
        <div className="mb-4">
          <Dropdown onOpenChange={setIsDropdownOpen}>
            <DropdownTrigger>
              <button className="flex justify-center items-center w-full rounded-md border border-gray-100 shadow-sm px-4 py-2 bg-white text-sm font-medium text-gray-700 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500">
                <span className="flex-1 text-center">
                  {selectedStock || "Select a Stock"}
                </span>
                <svg
                  className={`ml-2 h-5 w-5 text-gray-700 transform transition-transform ${
                    isDropdownOpen ? "rotate-180" : "rotate-0"
                  }`}
                  xmlns="http://www.w3.org/2000/svg"
                  viewBox="0 0 20 20"
                  fill="currentColor"
                >
                  <path
                    fillRule="evenodd"
                    d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z"
                    clipRule="evenodd"
                  />
                </svg>
              </button>
            </DropdownTrigger>
            <DropdownMenu
              aria-label="Stock Options"
              selectionMode="single"
              selectedKey={selectedStock}
              onSelectionChange={setSelectedStock}
            >
              <DropdownItem key="AAPL" className="text-gray-700 px-2">
                AAPL
              </DropdownItem>
              <DropdownItem key="GOOGL" className="text-gray-700 px-2">
                GOOGL
              </DropdownItem>
              <DropdownItem key="MSFT" className="text-gray-700 px-2">
                MSFT
              </DropdownItem>
              <DropdownItem key="SPOT" className="text-gray-700 px-2">
                SPOT
              </DropdownItem>
              <DropdownItem key="TSLA" className="text-gray-700 px-2">
                TSLA
              </DropdownItem>
              <DropdownItem key="VTI" className="text-gray-700 px-2">
                VTI
              </DropdownItem>
            </DropdownMenu>
          </Dropdown>
        </div>
        <div className="flex justify-center mt-4">
          <button
            onClick={handlePredictClick}
            className="px-4 py-2 bg-blue-600 text-white rounded-md predict-button"
          >
            Predict
          </button>
        </div>

        {triggerPrediction && selectedStock && (
          <SendRequestToPython stock={selectedStock} />
        )}

        <section id="about" ref={aboutRef} className="pt-20 spacing">
          <div className="horizontal-line"></div>
          <h2 className="text-2xl font-bold mb-4 text-gray-700">About</h2>
          <p className="mb-4 text-gray-700">
            Stocker utilizes advanced machine learning methods like XGBoost and
            Bayesian Neural Networks to forecast stock prices over the upcoming
            seven days.It utilizes the latest stock data from Yahoo Finance, to
            deliver the most accurate predictions. The application boasts
            user-friendly interfaces developed with Next.js and Swift, ensuring
            seamless access to dynamic market predictions. Technologies employed
            include Python, Swift, TypeScript, TailwindCSS, Next.js, Node.js,
            Pyro, Flask, XGBoost, Optuna, pandas, and Matplotlib.
          </p>
        </section>

        <section id="services" ref={servicesRef} className="pt-20">
          <div className="horizontal-line"></div>
          <h2 className="text-2xl font-bold mb-4 text-gray-700">Services</h2>
          <p className="mb-4 text-gray-700">
            Stock Forecasting using Stocker is limited to AAPL, GOOGL, MSFT,
            SPOT, TSLA, VTI
          </p>
        </section>

        <section id="contact" ref={contactRef} className="pt-20">
          <div className="horizontal-line"></div>
          <h2 className="text-2xl font-bold mb-4 text-gray-700">Contact</h2>
          <button className="mb-4 text-gray-700">
            <a
              href="mailto:iratansh@ualberta.ca"
              target="_blank"
              rel="noopener noreferrer"
            >
              iratansh@ualberta.ca
            </a>
          </button>
        </section>

        <div className="end">
          <div className="bottom-box">
            <div className="horizontal-line-2"></div>
            <p className="text-gray-700 bottom-box-words">
              © 2024 Stocker. All rights reserved.
            </p>
          </div>
        </div>
      </main>
    </div>
  );
};

export default Home;

