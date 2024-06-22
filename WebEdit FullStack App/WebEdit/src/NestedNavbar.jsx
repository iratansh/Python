import React, { useState, useRef } from "react";
import Navbar from "react-bootstrap/Navbar";
import Nav from "react-bootstrap/Nav";
import OverlayTrigger from "react-bootstrap/OverlayTrigger";
import Tooltip from "react-bootstrap/Tooltip";
import NavDropdown from "react-bootstrap/NavDropdown";
import Modal from "react-bootstrap/Modal";
import Button from "react-bootstrap/Button";
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
  FaSearchPlus,
  FaSearchMinus,
} from "react-icons/fa";

const NestedNavbar = () => {
  const fileInputRef = useRef(null);
  const [activeTooltip, setActiveTooltip] = useState(null);
  const [activeStyles, setActiveStyles] = useState({
    bold: false,
    italic: false,
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
  });
  const [showModal, setShowModal] = useState(false);
  const [imageDimensions, setImageDimensions] = useState({
    width: "",
    height: "",
  });

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
    if (color === "none") {
      document.execCommand("removeFormat");
    } else {
      document.execCommand("hiliteColor", false, color);
    }
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

  const handleFileUpload = () => {
    const file = fileInputRef.current?.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        const img = document.createElement("img");
        img.src = event.target.result;
        img.style.width = `${imageDimensions.width}px`;
        img.style.height = `${imageDimensions.height}px`;
        img.style.maxWidth = "100%";
        img.style.position = "relative";

        const selection = document.getSelection();
        if (selection.rangeCount > 0) {
          const range = selection.getRangeAt(0);
          range.deleteContents();
          range.insertNode(img);
        }

        setShowModal(false);
        setImageDimensions({ width: "", height: "" });
      };
      reader.readAsDataURL(file);
    } else {
      alert("Please select an image file.");
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

  const getDropdownItemStyle = (selected) => ({
    backgroundColor: selected ? "lightgray" : "transparent",
    border: selected ? "1px solid black" : "none",
  });

  const handleDropdownToggle = (dropdown, isOpen) => {
    setDropdownOpen({ ...dropdownOpen, [dropdown]: isOpen });
    if (isOpen) {
      setActiveTooltip(null); // Hide all tooltips when dropdown is opened
    }
  };

  const isNumber = (value) =>
    !isNaN(value) && value.trim() !== "" && Number(value) > 0;

  const isUploadDisabled = () =>
    !isNumber(imageDimensions.width) || !isNumber(imageDimensions.height);

  const triggerSearch = () => {
    const searchTerm = prompt("Enter text to search:");
    if (searchTerm) {
      window.find(searchTerm);
    }
  };

  return (
    <>
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
                onClick={(e) => handleClick(e, triggerSearch)}
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
            <OverlayTrigger
              placement="bottom"
              overlay={<Tooltip id="tooltip-font">Font</Tooltip>}
              show={activeTooltip === "font"}
            >
              <NavDropdown
                title={<FaFont style={iconStyle} />}
                id="nav-dropdown-font"
                onMouseEnter={() => setActiveTooltip("font")}
                onMouseLeave={hideAllTooltips}
                show={dropdownOpen.font}
                onToggle={(isOpen) => handleDropdownToggle("font", isOpen)}
              >
                {[
                  "Arial",
                  "Courier New",
                  "Georgia",
                  "Times New Roman",
                  "Verdana",
                  "Comic Sans MS",
                  "Impact",
                ].map((font) => (
                  <NavDropdown.Item
                    key={font}
                    onClick={() => onFontChange(font)}
                    style={getDropdownItemStyle(selectedOptions.font === font)}
                  >
                    {font}
                  </NavDropdown.Item>
                ))}
              </NavDropdown>
            </OverlayTrigger>
            <OverlayTrigger
              placement="bottom"
              overlay={<Tooltip id="tooltip-font-size">Font Size</Tooltip>}
              show={activeTooltip === "font-size"}
            >
              <NavDropdown
                title={<FaTextHeight style={iconStyle} />}
                id="nav-dropdown-font-size"
                onMouseEnter={() => setActiveTooltip("font-size")}
                onMouseLeave={hideAllTooltips}
                show={dropdownOpen.fontSize}
                onToggle={(isOpen) => handleDropdownToggle("fontSize", isOpen)}
              >
                {[1, 2, 3, 4, 5, 6, 7].map((size) => (
                  <NavDropdown.Item
                    key={size}
                    onClick={() => onFontSizeChange(size)}
                    style={getDropdownItemStyle(
                      selectedOptions.fontSize === size
                    )}
                  >
                    {size}
                  </NavDropdown.Item>
                ))}
              </NavDropdown>
            </OverlayTrigger>
            <OverlayTrigger
              placement="bottom"
              overlay={<Tooltip id="tooltip-font-color">Font Color</Tooltip>}
              show={activeTooltip === "font-color"}
            >
              <NavDropdown
                title={<FaPalette style={iconStyle} />}
                id="nav-dropdown-font-color"
                onMouseEnter={() => setActiveTooltip("font-color")}
                onMouseLeave={hideAllTooltips}
                show={dropdownOpen.fontColor}
                onToggle={(isOpen) =>
                  handleDropdownToggle("fontColor", isOpen)
                }
              >
                {[
                  "red",
                  "blue",
                  "green",
                  "black",
                  "purple",
                  "orange",
                  "brown",
                ].map((color) => (
                  <NavDropdown.Item
                    key={color}
                    onClick={() => onFontColorChange(color)}
                    style={getDropdownItemStyle(
                      selectedOptions.fontColor === color
                    )}
                  >
                    {color}
                  </NavDropdown.Item>
                ))}
              </NavDropdown>
            </OverlayTrigger>
            <OverlayTrigger
              placement="bottom"
              overlay={<Tooltip id="tooltip-highlight">Highlight</Tooltip>}
              show={activeTooltip === "highlight"}
            >
              <NavDropdown
                title={<FaHighlighter style={iconStyle} />}
                id="nav-dropdown-highlight"
                onMouseEnter={() => setActiveTooltip("highlight")}
                onMouseLeave={hideAllTooltips}
                show={dropdownOpen.highlight}
                onToggle={(isOpen) =>
                  handleDropdownToggle("highlight", isOpen)
                }
              >
                {[
                  "none",
                  "yellow",
                  "lightgreen",
                  "lightblue",
                  "lightpink",
                  "lightgray",
                ].map((color) => (
                  <NavDropdown.Item
                    key={color}
                    onClick={() => onHighlightColorChange(color)}
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
              overlay={<Tooltip id="tooltip-image">Insert Image</Tooltip>}
              show={activeTooltip === "image"}
            >
              <Nav.Link
                href="#image"
                onMouseEnter={() => setActiveTooltip("image")}
                onMouseLeave={hideAllTooltips}
                onClick={() => setShowModal(true)}
              >
                <FaFileImage style={iconStyle} />
              </Nav.Link>
            </OverlayTrigger>
            <OverlayTrigger
              placement="bottom"
              overlay={
                <Tooltip id="tooltip-bullets">Insert Bullet Points</Tooltip>
              }
              show={activeTooltip === "bullets"}
            >
              <Nav.Link
                href="#bullets"
                onMouseEnter={() => setActiveTooltip("bullets")}
                onMouseLeave={hideAllTooltips}
                onClick={(e) => handleClick(e, onBulletPointsChange)}
              >
                <FaListUl style={iconStyle} />
              </Nav.Link>
            </OverlayTrigger>
            <OverlayTrigger
              placement="bottom"
              overlay={
                <Tooltip id="tooltip-numbered-list">
                  Insert Numbered List
                </Tooltip>
              }
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
          </div>
        </Nav>
      </Navbar>

      <Modal show={showModal} onHide={() => setShowModal(false)}>
        <Modal.Header closeButton>
          <Modal.Title>Insert Image</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          <div
            style={{
              display: "flex",
              alignItems: "center",
              marginBottom: "10px",
            }}
          >
            <label htmlFor="image-width" style={{ marginRight: "10px" }}>
              Width:
            </label>
            <input
              type="number"
              id="image-width"
              value={imageDimensions.width}
              onChange={(e) =>
                setImageDimensions({
                  ...imageDimensions,
                  width: e.target.value,
                })
              }
              style={{ width: "60px", marginRight: "10px" }}
              min="1"
            />
            <label htmlFor="image-height" style={{ marginRight: "10px" }}>
              Height:
            </label>
            <input
              type="number"
              id="image-height"
              value={imageDimensions.height}
              onChange={(e) =>
                setImageDimensions({
                  ...imageDimensions,
                  height: e.target.value,
                })
              }
              style={{ width: "60px" }}
              min="1"
            />
          </div>
          <input type="file" accept="image/*" ref={fileInputRef} />
        </Modal.Body>
        <Modal.Footer>
          <Button variant="secondary" onClick={() => setShowModal(false)}>
            Close
          </Button>
          <Button
            variant="primary"
            onClick={handleFileUpload}
            disabled={isUploadDisabled()}
          >
            Upload Image
          </Button>
        </Modal.Footer>
      </Modal>
    </>
  );
};

export default NestedNavbar;

