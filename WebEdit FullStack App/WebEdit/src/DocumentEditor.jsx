import React, { useState, useRef, useEffect } from "react";
import NestedNavbar from "./NestedNavbar";
import HelpMenu from "./HelpMenu";
import "./DocumentEditor.css";
import Navbar from "./Navbar";

export default function GoogleDoc() {
  const [showHelp, setShowHelp] = useState(false);
  const [editorMode, setEditorMode] = useState(true);
  const [activeStyles, setActiveStyles] = useState({
    bold: false,
    italic: false,
    highlight: false,
    fontColor: false,
  });
  const contentEditableRef = useRef(null);
  const fileInputRef = useRef(null);

  const handleHelpClick = () => {
    setShowHelp(true);
  };

  const handleHelpMenuClose = () => {
    setShowHelp(false);
  };

  const applyCommand = (command, value = null) => {
    if (contentEditableRef.current && editorMode) {
      document.execCommand(command, false, value);
      contentEditableRef.current.focus();
    }
  };

  const handleFontChange = (font) => applyCommand("fontName", font);
  const handleFontSizeChange = (size) => applyCommand("fontSize", size);
  const handleFontColorChange = (color) => {
    applyCommand("foreColor", color);
    setActiveStyles({ ...activeStyles, fontColor: !activeStyles.fontColor });
  };
  const handleHighlightColorChange = (color) => {
    applyCommand("hiliteColor", color);
    setActiveStyles({ ...activeStyles, highlight: !activeStyles.highlight });
  };
  const handleBoldClick = () => {
    applyCommand("bold");
    setActiveStyles({ ...activeStyles, bold: !activeStyles.bold });
  };
  const handleItalicClick = () => {
    applyCommand("italic");
    setActiveStyles({ ...activeStyles, italic: !activeStyles.italic });
  };
  const handleBulletPointsChange = () => applyCommand("insertUnorderedList");
  const handleNumberedListClick = () => applyCommand("insertOrderedList");

  const handleImageUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = function (e) {
        applyCommand("insertImage", e.target.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleZoomInClick = () => {
    document.body.style.zoom = (parseFloat(document.body.style.zoom || 1) + 0.1).toString();
  };

  const handleZoomOutClick = () => {
    document.body.style.zoom = (parseFloat(document.body.style.zoom || 1) - 0.1).toString();
  };

  const handleEditorModeToggle = () => {
    setEditorMode(!editorMode);
  };

  useEffect(() => {
    if (contentEditableRef.current) {
      contentEditableRef.current.contentEditable = editorMode ? "true" : "false";
      contentEditableRef.current.style.pointerEvents = editorMode ? "auto" : "none";
    }
  }, [editorMode]);

  return (
    <>
      <div className="google-doc">
        <NestedNavbar
          onHelpClick={handleHelpClick}
          onFontChange={handleFontChange}
          onFontSizeChange={handleFontSizeChange}
          onFontColorChange={handleFontColorChange}
          onHighlightColorChange={handleHighlightColorChange}
          onBulletPointsChange={handleBulletPointsChange}
          onEditorModeToggle={handleEditorModeToggle}
          onBoldClick={handleBoldClick}
          onItalicClick={handleItalicClick}
          onNumberedListClick={handleNumberedListClick}
          onZoomInClick={handleZoomInClick}
          onZoomOutClick={handleZoomOutClick}
          activeStyles={activeStyles}
          editorMode={editorMode}
        />
        <Navbar />
        <div
          contentEditable={editorMode ? "true" : "false"}
          className="document-content"
          ref={contentEditableRef}
        >
          Start writing your document here...
        </div>
        {showHelp && <HelpMenu handleContinue={handleHelpMenuClose} />}
        <input
          type="file"
          ref={fileInputRef}
          style={{ display: "none" }}
          accept=".pdf,.png,.jpeg,.jpg"
          onChange={handleImageUpload}
        />
      </div>
    </>
  );
}

