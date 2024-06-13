import React, { useState, useRef } from "react";
import "./HelpMenu.css";

export default function HelpMenu({ handleContinue }) {
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [helpText, setHelpText] = useState("Here you can find information on how to use WebEdit.");
  const helpMenuRef = useRef(null);
  const dragHandleRef = useRef(null);

  const handleMouseDown = (e) => {
    if (dragHandleRef.current && dragHandleRef.current.contains(e.target)) {
      setIsDragging(true);
      dragHandleRef.current.style.cursor = "grabbing";
      setPosition({
        x: e.clientX - helpMenuRef.current.getBoundingClientRect().left,
        y: e.clientY - helpMenuRef.current.getBoundingClientRect().top,
      });
    }
  };

  const handleMouseMove = (e) => {
    if (isDragging) {
      const newX = e.clientX - position.x;
      const newY = e.clientY - position.y;
      helpMenuRef.current.style.left = `${newX}px`;
      helpMenuRef.current.style.top = `${newY}px`;
    }
  };

  const handleMouseUp = () => {
    setIsDragging(false);
    dragHandleRef.current.style.cursor = "grab";
  };

  const handleUploadClick = () => {
    setHelpText("To upload .pdf or .txt files, click the downward arrow button on the navbar above.");
  };

  const handleSavingClick = () => {
    setHelpText("To save your work, click the save button on the navbar above.");
  }

  const handleFeaturesClick = () => {
    setHelpText("WebEdit is a simple web-based document creation and editing tool. Features include uploading, saving, word prediction, and text customization and formatting.");
  }
    

  return (
    <div
      className="overlay"
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
    >
      <div
        className="help-menu"
        ref={helpMenuRef}
        onMouseDown={handleMouseDown}
        onMouseUp={handleMouseUp}
        style={{ position: "absolute" }}
      >
        <div className="drag-handle" ref={dragHandleRef}>
          <h1 className="header-helpmenu">WebEdit Help Menu</h1>
        </div>
        <div className="content">
          <div className="vertical-buttons">
            <button className="side-button" onClick={ handleUploadClick }>Uploading</button>
            <button className="side-button" onClick={ handleSavingClick }>Saving</button>
            <button className="side-button" onClick={ handleFeaturesClick }>Features</button>
            <button className="side-button">Button 4</button>
            <button className="side-button">Button 5</button>
          </div>
          <div className="text-content">
            <div className="text-container">
              <p className="text">{helpText}</p>
            </div>
            <div className="button-container">
              <button className="button-exit" onClick={handleContinue}>
                Exit
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
