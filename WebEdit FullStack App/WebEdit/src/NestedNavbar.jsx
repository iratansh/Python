import React, { useState, useRef } from "react";
import Navbar from "react-bootstrap/Navbar";
import Nav from "react-bootstrap/Nav";
import OverlayTrigger from "react-bootstrap/OverlayTrigger";
import Tooltip from "react-bootstrap/Tooltip";
import NavDropdown from "react-bootstrap/NavDropdown";
import Modal from "react-bootstrap/Modal";
import Button from "react-bootstrap/Button";
import ReactToPrint from "react-to-print";
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

const NestedNavbar = ({ handlePrint, contentEditableRef }) => {
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
  const activeIconStyle = { ...iconStyle, fontSize: "24px" };

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

  const handleHighlightColorChange = (color) => {
    const selection = window.getSelection();
    if (!selection.rangeCount) return;
  
    const range = selection.getRangeAt(0);
    const span = document.createElement("span");
  
    if (color === "none") {
      // Find the parent element to restore the text without highlight
      const parentElement = range.startContainer.parentElement;
      if (parentElement.nodeName === "SPAN" && parentElement.style.backgroundColor) {
        const text = parentElement.innerText;
        parentElement.replaceWith(document.createTextNode(text));
      }
    } else {
      span.style.backgroundColor = color;
      range.surroundContents(span);
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
      setActiveTooltip(null);
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
                onClick={handlePrint}
              >
                <ReactToPrint
                  trigger={() => (
                    <button
                      style={{
                        border: "none",
                        backgroundColor: "transparent",
                        cursor: "pointer",
                      }}
                    >
                      <FaPrint style={iconStyle} />
                    </button>
                  )}
                  content={() => contentEditableRef.current}
                  pageStyle={`
                  @media print {
                    * {
                      box-shadow: none !important;
                      text-shadow: none !important;
                      background: transparent !important;
                      color: #000 !important;
                      -webkit-print-color-adjust: exact;
                    }

                    p {
                      orphans: 3;
                      widows: 3;
                    }
                  }
                  
                  @page {
                    size: auto;
                    margin: 3cm;
                  }
                `}
                />
              </Nav.Link>
            </OverlayTrigger>
            <span>|</span>
            <NavDropdown
              title={<FaFont style={iconStyle} />}
              show={dropdownOpen.font}
              onToggle={(isOpen) => handleDropdownToggle("font", isOpen)}
              onMouseEnter={() => setActiveTooltip("font")}
              onMouseLeave={hideAllTooltips}
            >
              <NavDropdown.Item
                onClick={() => onFontChange("Arial")}
                style={getDropdownItemStyle(selectedOptions.font === "Arial")}
              >
                Arial
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
                onClick={() => onFontChange("Courier New")}
                style={getDropdownItemStyle(
                  selectedOptions.font === "Courier New"
                )}
              >
                Courier New
              </NavDropdown.Item>
              <NavDropdown.Item
                onClick={() => onFontChange("Georgia")}
                style={getDropdownItemStyle(selectedOptions.font === "Georgia")}
              >
                Georgia
              </NavDropdown.Item>
              <NavDropdown.Item
                onClick={() => onFontChange("Verdana")}
                style={getDropdownItemStyle(selectedOptions.font === "Verdana")}
              >
                Verdana
              </NavDropdown.Item>
              <NavDropdown.Item
                onClick={() => onFontChange("Tahoma")}
                style={getDropdownItemStyle(selectedOptions.font === "Tahoma")}
              >
                Tahoma
              </NavDropdown.Item>
              <NavDropdown.Item
                onClick={() => onFontChange("Helvetica")}
                style={getDropdownItemStyle(
                  selectedOptions.font === "Helvetica"
                )}
              >
                Helvetica
              </NavDropdown.Item>
              <NavDropdown.Item
                onClick={() => onFontChange("Garamond")}
                style={getDropdownItemStyle(
                  selectedOptions.font === "Garamond"
                )}
              >
                Garamond
              </NavDropdown.Item>
              <NavDropdown.Item
                onClick={() => onFontChange("Impact")}
                style={getDropdownItemStyle(selectedOptions.font === "Impact")}
              >
                Impact
              </NavDropdown.Item>
              <NavDropdown.Item
                onClick={() => onFontChange("Comic Sans MS")}
                style={getDropdownItemStyle(
                  selectedOptions.font === "Comic Sans MS"
                )}
              >
                Comic Sans MS
              </NavDropdown.Item>
            </NavDropdown>

            <NavDropdown
              title={<FaTextHeight style={iconStyle} />}
              show={dropdownOpen.fontSize}
              onToggle={(isOpen) => handleDropdownToggle("fontSize", isOpen)}
              onMouseEnter={() => setActiveTooltip("fontSize")}
              onMouseLeave={hideAllTooltips}
            >
              <NavDropdown.Item
                style={getDropdownItemStyle(selectedOptions.fontSize === "1")}
                onClick={() => onFontSizeChange("1")}
              >
                1 (8pt)
              </NavDropdown.Item>
              <NavDropdown.Item
                style={getDropdownItemStyle(selectedOptions.fontSize === "2")}
                onClick={() => onFontSizeChange("2")}
              >
                2 (10pt)
              </NavDropdown.Item>
              <NavDropdown.Item
                style={getDropdownItemStyle(selectedOptions.fontSize === "3")}
                onClick={() => onFontSizeChange("3")}
              >
                3 (12pt)
              </NavDropdown.Item>
              <NavDropdown.Item
                style={getDropdownItemStyle(selectedOptions.fontSize === "4")}
                onClick={() => onFontSizeChange("4")}
              >
                4 (14pt)
              </NavDropdown.Item>
              <NavDropdown.Item
                style={getDropdownItemStyle(selectedOptions.fontSize === "5")}
                onClick={() => onFontSizeChange("5")}
              >
                5 (18pt)
              </NavDropdown.Item>
              <NavDropdown.Item
                style={getDropdownItemStyle(selectedOptions.fontSize === "6")}
                onClick={() => onFontSizeChange("6")}
              >
                6 (24pt)
              </NavDropdown.Item>
              <NavDropdown.Item
                style={getDropdownItemStyle(selectedOptions.fontSize === "7")}
                onClick={() => onFontSizeChange("7")}
              >
                7 (36pt)
              </NavDropdown.Item>
              <NavDropdown.Divider />
            </NavDropdown>

            <NavDropdown
              title={<FaPalette style={iconStyle} />}
              show={dropdownOpen.fontColor}
              onToggle={(isOpen) => handleDropdownToggle("fontColor", isOpen)}
              onMouseEnter={() => setActiveTooltip("fontColor")}
              onMouseLeave={hideAllTooltips}
            >
              <NavDropdown.Item
                onClick={() => onFontColorChange("black")}
                style={getDropdownItemStyle(
                  selectedOptions.fontColor === "black"
                )}
              >
                Black
              </NavDropdown.Item>
              <NavDropdown.Item
                onClick={() => onFontColorChange("red")}
                style={getDropdownItemStyle(
                  selectedOptions.fontColor === "red"
                )}
              >
                Red
              </NavDropdown.Item>
              <NavDropdown.Item
                onClick={() => onFontColorChange("blue")}
                style={getDropdownItemStyle(
                  selectedOptions.fontColor === "blue"
                )}
              >
                Blue
              </NavDropdown.Item>
              <NavDropdown.Item
                onClick={() => onFontColorChange("green")}
                style={getDropdownItemStyle(
                  selectedOptions.fontColor === "green"
                )}
              >
                Green
              </NavDropdown.Item>
              <NavDropdown.Item
                onClick={() => onFontColorChange("orange")}
                style={getDropdownItemStyle(
                  selectedOptions.fontColor === "orange"
                )}
              >
                Orange
              </NavDropdown.Item>
              <NavDropdown.Item
                onClick={() => onFontColorChange("purple")}
                style={getDropdownItemStyle(
                  selectedOptions.fontColor === "purple"
                )}
              >
                Purple
              </NavDropdown.Item>
              <NavDropdown.Item
                onClick={() => onFontColorChange("brown")}
                style={getDropdownItemStyle(
                  selectedOptions.fontColor === "brown"
                )}
              >
                Brown
              </NavDropdown.Item>
              <NavDropdown.Item
                onClick={() => onFontColorChange("gray")}
                style={getDropdownItemStyle(
                  selectedOptions.fontColor === "gray"
                )}
              >
                Gray
              </NavDropdown.Item>
              <NavDropdown.Item
                onClick={() => onFontColorChange("pink")}
                style={getDropdownItemStyle(
                  selectedOptions.fontColor === "pink"
                )}
              >
                Pink
              </NavDropdown.Item>
            </NavDropdown>

            <NavDropdown
              title={<FaHighlighter style={iconStyle} />}
              show={dropdownOpen.highlight}
              onToggle={(isOpen) => handleDropdownToggle("highlight", isOpen)}
              onMouseEnter={() => setActiveTooltip("highlight")}
              onMouseLeave={hideAllTooltips}
            >
              <NavDropdown.Item
                onClick={() => handleHighlightColorChange("yellow")}
                style={getDropdownItemStyle(
                  selectedOptions.highlightColor === "yellow"
                )}
              >
                Yellow
              </NavDropdown.Item>
              <NavDropdown.Item
                onClick={() => handleHighlightColorChange("lightblue")}
                style={getDropdownItemStyle(
                  selectedOptions.highlightColor === "lightblue"
                )}
              >
                Light Blue
              </NavDropdown.Item>
              <NavDropdown.Item
                onClick={() => handleHighlightColorChange("lightgreen")}
                style={getDropdownItemStyle(
                  selectedOptions.highlightColor === "lightgreen"
                )}
              >
                Light Green
              </NavDropdown.Item>
              <NavDropdown.Item
                onClick={() => handleHighlightColorChange("lightpink")}
                style={getDropdownItemStyle(
                  selectedOptions.highlightColor === "lightpink"
                )}
              >
                Light Pink
              </NavDropdown.Item>
              <NavDropdown.Item
                onClick={() => handleHighlightColorChange("lightgray")}
                style={getDropdownItemStyle(
                  selectedOptions.highlightColor === "lightgray"
                )}
              >
                Light Gray
              </NavDropdown.Item>
              <NavDropdown.Item
                onClick={() => handleHighlightColorChange("lightcoral")}
                style={getDropdownItemStyle(
                  selectedOptions.highlightColor === "lightcoral"
                )}
              >
                Light Coral
              </NavDropdown.Item>
              <NavDropdown.Item
                onClick={() =>
                  handleHighlightColorChange("lightgoldenrodyellow")
                }
                style={getDropdownItemStyle(
                  selectedOptions.highlightColor === "lightgoldenrodyellow"
                )}
              >
                Light Goldenrod Yellow
              </NavDropdown.Item>
              <NavDropdown.Item
                onClick={() => handleHighlightColorChange("lightsalmon")}
                style={getDropdownItemStyle(
                  selectedOptions.highlightColor === "lightsalmon"
                )}
              >
                Light Salmon
              </NavDropdown.Item>
              <NavDropdown.Item
                onClick={() => handleHighlightColorChange("lightseagreen")}
                style={getDropdownItemStyle(
                  selectedOptions.highlightColor === "lightseagreen"
                )}
              >
                Light Sea Green
              </NavDropdown.Item>
              <NavDropdown.Item
                onClick={() => handleHighlightColorChange("none")}
                style={getDropdownItemStyle(
                  selectedOptions.highlightColor === "none"
                )}
              >
                None
              </NavDropdown.Item>
            </NavDropdown>

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
            <span>|</span>
            <OverlayTrigger
              placement="bottom"
              overlay={<Tooltip id="tooltip-insertImage">Insert Image</Tooltip>}
              show={activeTooltip === "insertImage"}
            >
              <Nav.Link
                href="#insertImage"
                onMouseEnter={() => setActiveTooltip("insertImage")}
                onMouseLeave={hideAllTooltips}
                onClick={() => setShowModal(true)}
              >
                <FaFileImage style={iconStyle} />
              </Nav.Link>
            </OverlayTrigger>
            <OverlayTrigger
              placement="bottom"
              overlay={<Tooltip id="tooltip-orderedList">Ordered List</Tooltip>}
              show={activeTooltip === "orderedList"}
            >
              <Nav.Link
                href="#orderedList"
                onMouseEnter={() => setActiveTooltip("orderedList")}
                onMouseLeave={hideAllTooltips}
                onClick={(e) => handleClick(e, onNumberedListClick)}
              >
                <FaListOl style={iconStyle} />
              </Nav.Link>
            </OverlayTrigger>
            <OverlayTrigger
              placement="bottom"
              overlay={
                <Tooltip id="tooltip-unorderedList">Unordered List</Tooltip>
              }
              show={activeTooltip === "unorderedList"}
            >
              <Nav.Link
                href="#unorderedList"
                onMouseEnter={() => setActiveTooltip("unorderedList")}
                onMouseLeave={hideAllTooltips}
                onClick={(e) => handleClick(e, onBulletPointsChange)}
              >
                <FaListUl style={iconStyle} />
              </Nav.Link>
            </OverlayTrigger>
            <span>|</span>
            <OverlayTrigger
              placement="bottom"
              overlay={<Tooltip id="tooltip-zoomIn">Zoom In</Tooltip>}
              show={activeTooltip === "zoomIn"}
            >
              <Nav.Link
                href="#zoomIn"
                onMouseEnter={() => setActiveTooltip("zoomIn")}
                onMouseLeave={hideAllTooltips}
                onClick={(e) => handleClick(e, onZoomInClick)}
              >
                <FaSearchPlus style={iconStyle} />
              </Nav.Link>
            </OverlayTrigger>
            <OverlayTrigger
              placement="bottom"
              overlay={<Tooltip id="tooltip-zoomOut">Zoom Out</Tooltip>}
              show={activeTooltip === "zoomOut"}
            >
              <Nav.Link
                href="#zoomOut"
                onMouseEnter={() => setActiveTooltip("zoomOut")}
                onMouseLeave={hideAllTooltips}
                onClick={(e) => handleClick(e, onZoomOutClick)}
              >
                <FaSearchMinus style={iconStyle} />
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
          <input
            type="file"
            ref={fileInputRef}
            accept="image/*"
            style={{ marginBottom: "10px" }}
          />
          <div>
            <label>
              Width:
              <input
                type="text"
                value={imageDimensions.width}
                onChange={(e) =>
                  setImageDimensions({
                    ...imageDimensions,
                    width: e.target.value,
                  })
                }
                style={{ marginLeft: "10px", marginRight: "20px" }}
              />
            </label>
            <label>
              Height:
              <input
                type="text"
                value={imageDimensions.height}
                onChange={(e) =>
                  setImageDimensions({
                    ...imageDimensions,
                    height: e.target.value,
                  })
                }
                style={{ marginLeft: "10px" }}
              />
            </label>
          </div>
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

