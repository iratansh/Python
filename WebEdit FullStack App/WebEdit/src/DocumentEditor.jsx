import React from "react";
import "./DocumentEditor.css";
import NavigationBar from "./Navbar"; 

export default function GoogleDoc() {
    return (
      <div className="google-doc">
        <NavigationBar />
        <div contentEditable="true" className="document-editor">
          Start writing your document here...
        </div>
      </div>
    );
  }
