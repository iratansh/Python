import React, { useState, useRef } from "react";
import Navbar from "react-bootstrap/Navbar";
import Nav from "react-bootstrap/Nav";
import OverlayTrigger from "react-bootstrap/OverlayTrigger";
import Tooltip from "react-bootstrap/Tooltip";
import NavDropdown from "react-bootstrap/NavDropdown";
import {
  FaSearch,
  FaUndo,
  FaRedo,
  FaPrint,
  FaFont,
  FaTextHeight,
  FaPalette,
  FaHighlighter,
  FaBold,
  FaItalic,
  FaFileImage,
  FaListUl,
  FaListOl,
  FaArrowUp,
  FaSearchPlus,
  FaSearchMinus,
} from "react-icons/fa";

const NestedNavbar = () => {
  const fileInputRef = useRef(null);
  const [activeTooltip, setActiveTooltip] = useState(null);
  const [activeStyles, setActiveStyles] = useState({
    bold: false,
    italic: false,
    highlight: false,
    fontColor: false,
  });
  const [selectedOptions, setSelectedOptions] = useState({
    font: null,
    fontSize: null,
    fontColor: null,
    highlightColor: null,
  });
  const [dropdownOpen, setDropdownOpen] = useState({
    font: false,
    fontSize: false,
    fontColor: false,
    highlight: false,
    editorMode: false,
  });
  const [isEditable, setIsEditable] = useState(true);

  const hideAllTooltips = () => {
    setActiveTooltip(null);
  };

  const iconStyle = { fontSize: "20px", verticalAlign: "middle" };
  const activeIconStyle = { ...iconStyle, border: "1px solid black" };

  const handleClick = (event, handler) => {
    event.preventDefault();
    handler();
  };

  const onFontChange = (font) => {
    document.execCommand("fontName", false, font);
    setSelectedOptions({ ...selectedOptions, font });
  };

  const onFontSizeChange = (size) => {
    document.execCommand("fontSize", false, size);
    setSelectedOptions({ ...selectedOptions, fontSize: size });
  };

  const onFontColorChange = (color) => {
    document.execCommand("foreColor", false, color);
    setSelectedOptions({ ...selectedOptions, fontColor: color });
  };

  const onHighlightColorChange = (color) => {
    document.execCommand("hiliteColor", false, color);
    setSelectedOptions({ ...selectedOptions, highlightColor: color });
  };

  const onBoldClick = () => {
    document.execCommand("bold");
    setActiveStyles({ ...activeStyles, bold: !activeStyles.bold });
  };

  const onItalicClick = () => {
    document.execCommand("italic");
    setActiveStyles({ ...activeStyles, italic: !activeStyles.italic });
  };

  const onInsertImageClick = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        const img = document.createElement("img");
        img.src = event.target.result;
        img.style.maxWidth = "100%";
        document.execCommand("insertHTML", false, img.outerHTML);
      };
      reader.readAsDataURL(file);
    }
  };

  const onNumberedListClick = () => {
    document.execCommand("insertOrderedList");
  };

  const onBulletPointsChange = () => {
    document.execCommand("insertUnorderedList");
  };

  const onZoomInClick = () => {
    document.body.style.zoom = `${
      parseFloat(document.body.style.zoom || 1) + 0.1
    }`;
  };

  const onZoomOutClick = () => {
    document.body.style.zoom = `${
      parseFloat(document.body.style.zoom || 1) - 0.1
    }`;
  };

  const onEditorModeToggle = (mode) => {
    const isEditable = mode === "edit";
    document.body.contentEditable = isEditable ? "true" : "false";
    setIsEditable(isEditable);
  };

  const getDropdownItemStyle = (selected) => ({
    backgroundColor: selected ? "lightgray" : "transparent",
    border: selected ? "1px solid black" : "none",
  });

  const handleDropdownToggle = (dropdown, isOpen) => {
    setDropdownOpen({ ...dropdownOpen, [dropdown]: isOpen });
  };

  return (
    <Navbar
      fixed="top"
      style={{
        backgroundColor: "rgba(211, 211, 211, 0.5)",
        borderRadius: "100px",
        padding: "10px",
        margin: "60px 0",
        color: "#333",
        fontSize: "14px",
        fontFamily: "Arial, sans-serif",
      }}
    >
      <Nav className="mr-auto">
        <div style={{ display: "flex", alignItems: "center" }}>
          <OverlayTrigger
            placement="bottom"
            overlay={<Tooltip id="tooltip-search">Search</Tooltip>}
            show={activeTooltip === "search"}
          >
            <Nav.Link
              href="#search"
              onMouseEnter={() => setActiveTooltip("search")}
              onMouseLeave={hideAllTooltips}
              onClick={(e) => handleClick(e, () => {})}
            >
              <FaSearch style={iconStyle} />
            </Nav.Link>
          </OverlayTrigger>
          <OverlayTrigger
            placement="bottom"
            overlay={<Tooltip id="tooltip-undo">Undo</Tooltip>}
            show={activeTooltip === "undo"}
          >
            <Nav.Link
              href="#undo"
              onMouseEnter={() => setActiveTooltip("undo")}
              onMouseLeave={hideAllTooltips}
              onClick={(e) =>
                handleClick(e, () => document.execCommand("undo"))
              }
            >
              <FaUndo style={iconStyle} />
            </Nav.Link>
          </OverlayTrigger>
          <OverlayTrigger
            placement="bottom"
            overlay={<Tooltip id="tooltip-redo">Redo</Tooltip>}
            show={activeTooltip === "redo"}
          >
            <Nav.Link
              href="#redo"
              onMouseEnter={() => setActiveTooltip("redo")}
              onMouseLeave={hideAllTooltips}
              onClick={(e) =>
                handleClick(e, () => document.execCommand("redo"))
              }
            >
              <FaRedo style={iconStyle} />
            </Nav.Link>
          </OverlayTrigger>
          <OverlayTrigger
            placement="bottom"
            overlay={<Tooltip id="tooltip-print">Print</Tooltip>}
            show={activeTooltip === "print"}
          >
            <Nav.Link
              href="#print"
              onMouseEnter={() => setActiveTooltip("print")}
              onMouseLeave={hideAllTooltips}
              onClick={(e) => handleClick(e, () => window.print())}
            >
              <FaPrint style={iconStyle} />
            </Nav.Link>
          </OverlayTrigger>
          <OverlayTrigger
            placement="bottom"
            overlay={<Tooltip id="tooltip-zoom-in">Zoom In</Tooltip>}
            show={activeTooltip === "zoom-in"}
          >
            <Nav.Link
              href="#zoom-in"
              onMouseEnter={() => setActiveTooltip("zoom-in")}
              onMouseLeave={hideAllTooltips}
              onClick={(e) => handleClick(e, onZoomInClick)}
            >
              <FaSearchPlus style={iconStyle} />
            </Nav.Link>
          </OverlayTrigger>
          <OverlayTrigger
            placement="bottom"
            overlay={<Tooltip id="tooltip-zoom-out">Zoom Out</Tooltip>}
            show={activeTooltip === "zoom-out"}
          >
            <Nav.Link
              href="#zoom-out"
              onMouseEnter={() => setActiveTooltip("zoom-out")}
              onMouseLeave={hideAllTooltips}
              onClick={(e) => handleClick(e, onZoomOutClick)}
            >
              <FaSearchMinus style={iconStyle} />
            </Nav.Link>
          </OverlayTrigger>
          <span style={{ margin: "0 10px", color: "#333" }}>|</span>
        </div>
        <div style={{ display: "flex", alignItems: "center" }}>
          <OverlayTrigger
            placement="bottom"
            overlay={<Tooltip id="tooltip-font">Font</Tooltip>}
            show={activeTooltip === "font" && !dropdownOpen.font}
          >
            <NavDropdown
              title={<FaFont style={iconStyle} />}
              id="font-dropdown"
              className="ml-2"
              style={{
                backgroundColor: "transparent",
                border: "none",
                boxShadow: "none",
              }}
              onMouseEnter={() => setActiveTooltip("font")}
              onMouseLeave={hideAllTooltips}
              onToggle={(isOpen) => handleDropdownToggle("font", isOpen)}
            >
              <NavDropdown.Item
                onClick={() => onFontChange("Arial")}
                style={getDropdownItemStyle(selectedOptions.font === "Arial")}
              >
                Arial
              </NavDropdown.Item>
              <NavDropdown.Item
                onClick={() => onFontChange("Courier New")}
                style={getDropdownItemStyle(
                  selectedOptions.font === "Courier New"
                )}
              >
                Courier New
              </NavDropdown.Item>
              <NavDropdown.Item
                onClick={() => onFontChange("Times New Roman")}
                style={getDropdownItemStyle(
                  selectedOptions.font === "Times New Roman"
                )}
              >
                Times New Roman
              </NavDropdown.Item>
              <NavDropdown.Item
                onClick={() => onFontChange("Verdana")}
                style={getDropdownItemStyle(selectedOptions.font === "Verdana")}
              >
                Verdana
              </NavDropdown.Item>
            </NavDropdown>
          </OverlayTrigger>
          <OverlayTrigger
            placement="bottom"
            overlay={<Tooltip id="tooltip-font-size">Font Size</Tooltip>}
            show={activeTooltip === "font-size" && !dropdownOpen.fontSize}
          >
            <NavDropdown
              title={<FaTextHeight style={iconStyle} />}
              id="font-size-dropdown"
              className="ml-2"
              style={{
                backgroundColor: "transparent",
                border: "none",
                boxShadow: "none",
              }}
              onMouseEnter={() => setActiveTooltip("font-size")}
              onMouseLeave={hideAllTooltips}
              onToggle={(isOpen) => handleDropdownToggle("fontSize", isOpen)}
            >
              {[1, 2, 3, 4, 5, 6, 7].map((size) => (
                <NavDropdown.Item
                  key={size}
                  onClick={() => onFontSizeChange(size)}
                  style={getDropdownItemStyle(selectedOptions.fontSize === size)}
                >
                  {size}
                </NavDropdown.Item>
              ))}
            </NavDropdown>
          </OverlayTrigger>
          <OverlayTrigger
            placement="bottom"
            overlay={<Tooltip id="tooltip-font-color">Font Color</Tooltip>}
            show={activeTooltip === "font-color" && !dropdownOpen.fontColor}
          >
            <NavDropdown
              title={<FaPalette style={iconStyle} />}
              id="font-color-dropdown"
              className="ml-2"
              style={{
                backgroundColor: "transparent",
                border: "none",
                boxShadow: "none",
              }}
              onMouseEnter={() => setActiveTooltip("font-color")}
              onMouseLeave={hideAllTooltips}
              onToggle={(isOpen) => handleDropdownToggle("fontColor", isOpen)}
            >
              {["Red", "Blue", "Green", "Yellow", "Black", "White"].map(
                (color) => (
                  <NavDropdown.Item
                    key={color}
                    onClick={() => onFontColorChange(color)}
                    style={getDropdownItemStyle(
                      selectedOptions.fontColor === color
                    )}
                  >
                    {color}
                  </NavDropdown.Item>
                )
              )}
            </NavDropdown>
          </OverlayTrigger>
          <OverlayTrigger
            placement="bottom"
            overlay={<Tooltip id="tooltip-highlight">Highlight</Tooltip>}
            show={activeTooltip === "highlight" && !dropdownOpen.highlight}
          >
            <NavDropdown
              title={<FaHighlighter style={iconStyle} />}
              id="highlight-dropdown"
              className="ml-2"
              style={{
                backgroundColor: "transparent",
                border: "none",
                boxShadow: "none",
              }}
              onMouseEnter={() => setActiveTooltip("highlight")}
              onMouseLeave={hideAllTooltips}
              onToggle={(isOpen) => handleDropdownToggle("highlight", isOpen)}
            >
              {["None", "Yellow", "Green", "Pink"].map((color) => (
                <NavDropdown.Item
                  key={color}
                  onClick={() =>
                    onHighlightColorChange(color === "None" ? "transparent" : color)
                  }
                  style={getDropdownItemStyle(
                    selectedOptions.highlightColor === color
                  )}
                >
                  {color}
                </NavDropdown.Item>
              ))}
            </NavDropdown>
          </OverlayTrigger>
          <OverlayTrigger
            placement="bottom"
            overlay={<Tooltip id="tooltip-bold">Bold</Tooltip>}
            show={activeTooltip === "bold"}
          >
            <Nav.Link
              href="#bold"
              onMouseEnter={() => setActiveTooltip("bold")}
              onMouseLeave={hideAllTooltips}
              onClick={(e) => handleClick(e, onBoldClick)}
            >
              <FaBold
                style={activeStyles.bold ? activeIconStyle : iconStyle}
              />
            </Nav.Link>
          </OverlayTrigger>
          <OverlayTrigger
            placement="bottom"
            overlay={<Tooltip id="tooltip-italic">Italic</Tooltip>}
            show={activeTooltip === "italic"}
          >
            <Nav.Link
              href="#italic"
              onMouseEnter={() => setActiveTooltip("italic")}
              onMouseLeave={hideAllTooltips}
              onClick={(e) => handleClick(e, onItalicClick)}
            >
              <FaItalic
                style={activeStyles.italic ? activeIconStyle : iconStyle}
              />
            </Nav.Link>
          </OverlayTrigger>
          <OverlayTrigger
            placement="bottom"
            overlay={<Tooltip id="tooltip-insert-image">Insert Image</Tooltip>}
            show={activeTooltip === "insert-image"}
          >
            <Nav.Link
              href="#insert-image"
              onMouseEnter={() => setActiveTooltip("insert-image")}
              onMouseLeave={hideAllTooltips}
              onClick={(e) => {
                e.preventDefault();
                fileInputRef.current.click();
              }}
            >
              <FaFileImage style={iconStyle} />
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                style={{ display: "none" }}
                onChange={onInsertImageClick}
              />
            </Nav.Link>
          </OverlayTrigger>
          <OverlayTrigger
            placement="bottom"
            overlay={<Tooltip id="tooltip-numbered-list">Numbered List</Tooltip>}
            show={activeTooltip === "numbered-list"}
          >
            <Nav.Link
              href="#numbered-list"
              onMouseEnter={() => setActiveTooltip("numbered-list")}
              onMouseLeave={hideAllTooltips}
              onClick={(e) => handleClick(e, onNumberedListClick)}
            >
              <FaListOl style={iconStyle} />
            </Nav.Link>
          </OverlayTrigger>
          <OverlayTrigger
            placement="bottom"
            overlay={
              <Tooltip id="tooltip-bullet-points">Bullet Points</Tooltip>
            }
            show={activeTooltip === "bullet-points"}
          >
            <Nav.Link
              href="#bullet-points"
              onMouseEnter={() => setActiveTooltip("bullet-points")}
              onMouseLeave={hideAllTooltips}
              onClick={(e) => handleClick(e, onBulletPointsChange)}
            >
              <FaListUl style={iconStyle} />
            </Nav.Link>
          </OverlayTrigger>
          <span style={{ margin: "0 10px", color: "#333" }}>|</span>
        </div>
        <div style={{ display: "flex", alignItems: "center" }}>
          <OverlayTrigger
            placement="bottom"
            overlay={<Tooltip id="tooltip-editor-mode">Editor Mode</Tooltip>}
            show={activeTooltip === "editor-mode" && !dropdownOpen.editorMode}
          >
            <NavDropdown
              title={<FaArrowUp style={iconStyle} />}
              id="editor-mode-dropdown"
              className="ml-2"
              style={{
                backgroundColor: "transparent",
                border: "none",
                boxShadow: "none",
              }}
              onMouseEnter={() => setActiveTooltip("editor-mode")}
              onMouseLeave={hideAllTooltips}
              onToggle={(isOpen) => handleDropdownToggle("editorMode", isOpen)}
            >
              <NavDropdown.Item
                onClick={() => onEditorModeToggle("edit")}
                style={getDropdownItemStyle(isEditable)}
              >
                Edit
              </NavDropdown.Item>
              <NavDropdown.Item
                onClick={() => onEditorModeToggle("preview")}
                style={getDropdownItemStyle(!isEditable)}
              >
                Preview
              </NavDropdown.Item>
            </NavDropdown>
          </OverlayTrigger>
        </div>
      </Nav>
    </Navbar>
  );
};

export default NestedNavbar;

