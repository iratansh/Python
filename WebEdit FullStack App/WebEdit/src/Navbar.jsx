import React, { useState, useRef } from "react";
import Navbar from "react-bootstrap/Navbar";
import Nav from "react-bootstrap/Nav";
import NavDropdown from "react-bootstrap/NavDropdown";
import Form from "react-bootstrap/Form";
import "bootstrap/dist/css/bootstrap.min.css";
import {
  FaUpload,
  FaDownload,
  FaFileAlt,
  FaPrint,
  FaEnvelope,
  FaShareAlt,
  FaUndo,
  FaRedo,
  FaFont,
  FaTextHeight,
  FaPalette,
  FaHighlighter,
  FaBold,
  FaItalic,
  FaUnderline,
  FaStrikethrough,
  FaSubscript,
  FaSuperscript,
  FaFileImage,
  FaAlignLeft,
  FaAlignCenter,
  FaAlignRight,
  FaAlignJustify,
  FaIndent,
  FaOutdent,
  FaLink,
  FaUnlink,
  FaTable,
  FaCode,
  FaComment,
  FaTasks,
  FaArrowsAlt,
  FaFilePdf,
  FaFileCsv,
  FaRegWindowMaximize,
  FaRegWindowMinimize,
  FaTrash,
  FaHistory,
  FaTag,
  FaUser,
  FaShare,
  FaCog,
  FaQuestionCircle,
  FaSearchPlus,
  FaSearchMinus,
  FaCut,
  FaCopy,
  FaPaste,
} from "react-icons/fa";
import NestedNavbar from "./NestedNavbar";

export default function NavigationBar({ onHelpClick }) {
  const [docTitle, setDocTitle] = useState("Untitled Document");
  const fileInputRef = useRef(null);

  const handleTitleChange = (e) => {
    setDocTitle(e.target.value);
  };

  const handleUploadClick = () => {
    fileInputRef.current.click();
  };

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        const fileContent = event.target.result;
        console.log(`Uploaded file content: ${fileContent}`);
      };
      reader.readAsText(file);
    }
  };

  return (
    <>
      <NestedNavbar />
      <Navbar
        fixed="top"
        style={{
          backgroundColor: "white",
          padding: "10px 20px",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
        }}
      >
        <Form inline style={{ display: "flex", alignItems: "center" }}>
          <Form.Control
            type="text"
            value={docTitle}
            onChange={handleTitleChange}
            style={{ width: "300px", marginRight: "20px" }}
          />
        </Form>
        <Nav style={{marginRight:"700px"}}>
          <NavDropdown title="File" id="file-dropdown">
            <NavDropdown.Item href="#new">
              <FaFileAlt style={{ marginRight: "10px" }} />
              New
            </NavDropdown.Item>
            <NavDropdown.Item onClick={handleUploadClick}>
              <FaUpload style={{ marginRight: "10px" }} />
              Upload
            </NavDropdown.Item>
            <NavDropdown.Item href="#rename">
              <FaFileAlt style={{ marginRight: "10px" }} />
              Rename
            </NavDropdown.Item>
            <NavDropdown.Item href="#download">
              <FaDownload style={{ marginRight: "10px" }} />
              Download
            </NavDropdown.Item>
            <NavDropdown.Item href="#email">
              <FaEnvelope style={{ marginRight: "10px" }} />
              Email
            </NavDropdown.Item>
            <NavDropdown.Item href="#print">
              <FaPrint style={{ marginRight: "10px" }} />
              Print
            </NavDropdown.Item>
            <NavDropdown.Item href="#share">
              <FaShareAlt style={{ marginRight: "10px" }} />
              Share
            </NavDropdown.Item>
            <NavDropdown.Divider />
            <NavDropdown.Item href="#save-as-pdf">
              <FaFilePdf style={{ marginRight: "10px" }} />
              Save as PDF
            </NavDropdown.Item>
            <NavDropdown.Item href="#save-as-csv">
              <FaFileCsv style={{ marginRight: "10px" }} />
              Save as CSV
            </NavDropdown.Item>
            <NavDropdown.Divider />
            <NavDropdown.Item href="#maximize">
              <FaRegWindowMaximize style={{ marginRight: "10px" }} />
              Maximize
            </NavDropdown.Item>
            <NavDropdown.Item href="#minimize">
              <FaRegWindowMinimize style={{ marginRight: "10px" }} />
              Minimize
            </NavDropdown.Item>
          </NavDropdown>
          <NavDropdown title="Edit" id="edit-dropdown">
            <NavDropdown.Item>
              <FaUndo style={{ marginRight: "10px" }} />
              Undo
            </NavDropdown.Item>
            <NavDropdown.Item>
              <FaRedo style={{ marginRight: "10px" }} />
              Redo
            </NavDropdown.Item>
            <NavDropdown.Divider />
            <NavDropdown.Item>
              <FaCut style={{ marginRight: "10px" }} />
              Cut
            </NavDropdown.Item>
            <NavDropdown.Item>
              <FaCopy style={{ marginRight: "10px" }} />
              Copy
            </NavDropdown.Item>
            <NavDropdown.Item>
              <FaPaste style={{ marginRight: "10px" }} />
              Paste
            </NavDropdown.Item>
            <NavDropdown.Divider />
            <NavDropdown.Item>
              <FaBold style={{ marginRight: "10px" }} />
              Bold
            </NavDropdown.Item>
            <NavDropdown.Item>
              <FaItalic style={{ marginRight: "10px" }} />
              Italic
            </NavDropdown.Item>
            <NavDropdown.Item>
              <FaUnderline style={{ marginRight: "10px" }} />
              Underline
            </NavDropdown.Item>
            <NavDropdown.Item>
              <FaStrikethrough style={{ marginRight: "10px" }} />
              Strikethrough
            </NavDropdown.Item>
            <NavDropdown.Item>
              <FaSubscript style={{ marginRight: "10px" }} />
              Subscript
            </NavDropdown.Item>
            <NavDropdown.Item>
              <FaSuperscript style={{ marginRight: "10px" }} />
              Superscript
            </NavDropdown.Item>
          </NavDropdown>
          <NavDropdown title="View" id="view-dropdown">
            <NavDropdown.Item>
              <FaSearchPlus style={{ marginRight: "10px" }} />
              Zoom In
            </NavDropdown.Item>
            <NavDropdown.Item>
              <FaSearchMinus style={{ marginRight: "10px" }} />
              Zoom Out
            </NavDropdown.Item>
            <NavDropdown.Divider />
            <NavDropdown.Item>
              <FaArrowsAlt style={{ marginRight: "10px" }} />
              Full Screen
            </NavDropdown.Item>
          </NavDropdown>
          <NavDropdown title="Insert" id="insert-dropdown">
            <NavDropdown.Item>
              <FaFileImage style={{ marginRight: "10px" }} />
              Image
            </NavDropdown.Item>
            <NavDropdown.Item>
              <FaLink style={{ marginRight: "10px" }} />
              Link
            </NavDropdown.Item>
            <NavDropdown.Item>
              <FaUnlink style={{ marginRight: "10px" }} />
              Unlink
            </NavDropdown.Item>
            <NavDropdown.Item>
              <FaTable style={{ marginRight: "10px" }} />
              Table
            </NavDropdown.Item>
            <NavDropdown.Divider />
            <NavDropdown.Item>
              <FaCode style={{ marginRight: "10px" }} />
              Code Block
            </NavDropdown.Item>
            <NavDropdown.Item>
              <FaComment style={{ marginRight: "10px" }} />
              Comment
            </NavDropdown.Item>
            <NavDropdown.Item>
              <FaTasks style={{ marginRight: "10px" }} />
              Checklist
            </NavDropdown.Item>
          </NavDropdown>
          <NavDropdown title="Format" id="format-dropdown">
            <NavDropdown.Item>
              <FaFont style={{ marginRight: "10px" }} />
              Font
            </NavDropdown.Item>
            <NavDropdown.Item>
              <FaTextHeight style={{ marginRight: "10px" }} />
              Font Size
            </NavDropdown.Item>
            <NavDropdown.Item>
              <FaPalette style={{ marginRight: "10px" }} />
              Font Color
            </NavDropdown.Item>
            <NavDropdown.Item>
              <FaHighlighter style={{ marginRight: "10px" }} />
              Highlight Color
            </NavDropdown.Item>
            <NavDropdown.Divider />
            <NavDropdown.Item>
              <FaAlignLeft style={{ marginRight: "10px" }} />
              Align Left
            </NavDropdown.Item>
            <NavDropdown.Item>
              <FaAlignCenter style={{ marginRight: "10px" }} />
              Align Center
            </NavDropdown.Item>
            <NavDropdown.Item>
              <FaAlignRight style={{ marginRight: "10px" }} />
              Align Right
            </NavDropdown.Item>
            <NavDropdown.Item>
              <FaAlignJustify style={{ marginRight: "10px" }} />
              Justify
            </NavDropdown.Item>
            <NavDropdown.Divider />
            <NavDropdown.Item>
              <FaIndent style={{ marginRight: "10px" }} />
              Indent
            </NavDropdown.Item>
            <NavDropdown.Item>
              <FaOutdent style={{ marginRight: "10px" }} />
              Outdent
            </NavDropdown.Item>
          </NavDropdown>
          <NavDropdown title="Tools" id="tools-dropdown">
            <NavDropdown.Item>
              <FaHistory style={{ marginRight: "10px" }} />
              History
            </NavDropdown.Item>
            <NavDropdown.Item>
              <FaTag style={{ marginRight: "10px" }} />
              Tags
            </NavDropdown.Item>
            <NavDropdown.Item>
              <FaUser style={{ marginRight: "10px" }} />
              Collaborators
            </NavDropdown.Item>
            <NavDropdown.Item>
              <FaShare style={{ marginRight: "10px" }} />
              Share
            </NavDropdown.Item>
            <NavDropdown.Divider />
            <NavDropdown.Item>
              <FaCog style={{ marginRight: "10px" }} />
              Settings
            </NavDropdown.Item>
            <NavDropdown.Item onClick={onHelpClick}>
              <FaQuestionCircle style={{ marginRight: "10px" }} />
              Help
            </NavDropdown.Item>
          </NavDropdown>
          <NavDropdown title="Window" id="window-dropdown">
            <NavDropdown.Item>
              <FaRegWindowMaximize style={{ marginRight: "10px" }} />
              Maximize
            </NavDropdown.Item>
            <NavDropdown.Item>
              <FaRegWindowMinimize style={{ marginRight: "10px" }} />
              Minimize
            </NavDropdown.Item>
            <NavDropdown.Divider />
            <NavDropdown.Item>
              <FaTrash style={{ marginRight: "10px" }} />
              Close
            </NavDropdown.Item>
          </NavDropdown>
        </Nav>
      </Navbar>
      <input
        type="file"
        ref={fileInputRef}
        style={{ display: "none" }}
        accept=".pdf,.txt"
        onChange={handleFileUpload}
      />
    </>
  );
}

