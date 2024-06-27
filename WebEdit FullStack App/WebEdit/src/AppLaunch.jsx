import React, { useState } from "react";
import "./AppLaunch.css";
import GoogleDoc from "./DocumentEditor";

export default function AppLaunch() {
  const [isGoogleDocVisible, setIsGoogleDocVisible] = useState(false);

  const handleContinue = () => {
    setIsGoogleDocVisible(true);
    document.body.style.backgroundColor = "white";
  };

  if (isGoogleDocVisible) {
    return <GoogleDoc />;
  }

  return (
    <div className="app-launch">
      <div className="main">
        <h1 className="header">
          WebEdit is a simple web-based document creation and editing tool.
        </h1>
      </div>
      <div className="text-container">
        <input
          type="file"
          id="file"
          name="file"
          accept=".pdf,.txt"
          className="button"
        />
        <div className="button-container">
          <button className="button" onClick={handleContinue}>
            Continue
          </button>
        </div>
        <p className="text">
          Choose a File to Upload or just click the Continue Button
        </p>
      </div>
    </div>
  );
}
