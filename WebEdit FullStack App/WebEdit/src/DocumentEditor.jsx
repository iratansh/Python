import React, { useState, useRef } from "react";
import NestedNavbar from "./NestedNavbar";
import HelpMenu from "./HelpMenu";
import NavigationBar from "./Navbar";
import "./DocumentEditor.css";

export default function GoogleDoc() {
  const [showHelp, setShowHelp] = useState(false);
  const [activeStyles, setActiveStyles] = useState({
    bold: false,
    italic: false,
    highlight: false,
    fontColor: false,
  });
  const [uploadedImage, setUploadedImage] = useState(null);
  const [docTitle, setDocTitle] = useState("Untitled Document");
  const contentEditableRef = useRef(null);
  const fileInputRef = useRef(null);
  const [showResizeDialog, setShowResizeDialog] = useState(false);
  const [imageDimensions, setImageDimensions] = useState({
    width: 0,
    height: 0,
  });

  const applyCommand = (command, value = null) => {
    if (contentEditableRef.current) {
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
        const img = document.createElement("img");
        img.src = e.target.result;
        img.style.maxWidth = "100%";
        setUploadedImage(img);
        setShowResizeDialog(true);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleImageResize = () => {
    if (uploadedImage) {
      uploadedImage.style.width = `${imageDimensions.width}px`;
      uploadedImage.style.height = `${imageDimensions.height}px`;
      applyCommand("insertHTML", uploadedImage.outerHTML);
    }
    setShowResizeDialog(false);
    setUploadedImage(null);
    setImageDimensions({ width: 0, height: 0 });
  };

  const handleSettingsClick = () => {
    // Implement settings functionality
  };

  const handleCancelResize = () => {
    setShowResizeDialog(false);
    setUploadedImage(null);
    setImageDimensions({ width: 0, height: 0 });
  };

  const handleDimensionChange = (event) => {
    const { name, value } = event.target;
    setImageDimensions((prevDimensions) => ({
      ...prevDimensions,
      [name]: parseInt(value, 10),
    }));
  };

  const handleZoomInClick = () => {
    document.body.style.zoom = (
      parseFloat(document.body.style.zoom || 1) + 0.1
    ).toString();
  };

  const handleZoomOutClick = () => {
    document.body.style.zoom = (
      parseFloat(document.body.style.zoom || 1) - 0.1
    ).toString();
  };

  const handleHelpClick = () => {
    setShowHelp(true);
  };

  const handleHelpMenuClose = () => {
    setShowHelp(false);
  };

  const handleSaveAsPDF = () => {
    const content = contentEditableRef.current.innerHTML;
    const element = document.createElement("a");
    const blob = new Blob([content], { type: "application/pdf" });
    element.href = URL.createObjectURL(blob);
    element.download = "document.pdf";
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
  };

  const handleSaveAsCSV = () => {
    const content = contentEditableRef.current.innerText;
    const csvContent =
      "data:text/csv;charset=utf-8," + content.replace(/\n/g, ",");
    const element = document.createElement("a");
    element.setAttribute("href", encodeURI(csvContent));
    element.setAttribute("download", "document.csv");
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
  };

  const handleEmail = () => {
    const content = contentEditableRef.current.innerHTML;
    window.location.href = `mailto:?subject=${encodeURIComponent(
      "Document"
    )}&body=${encodeURIComponent(content)}`;
  };

  const handleInsertTable = () => {
    const rows = prompt("Enter number of rows:");
    const cols = prompt("Enter number of columns:");
    if (rows > 0 && cols > 0) {
      let table = "<table border='1'>";
      for (let i = 0; i < rows; i++) {
        table += "<tr>";
        for (let j = 0; j < cols; j++) {
          table += "<td>&nbsp;</td>";
        }
        table += "</tr>";
      }
      table += "</table>";
      document.execCommand("insertHTML", false, table);
    }
  };

  const handleInsertComment = () => {
    const comment = prompt("Enter the comment:");
    if (comment) {
      const span = document.createElement("span");
      span.style.backgroundColor = "yellow";
      span.textContent = comment;
      document.execCommand("insertHTML", false, span.outerHTML);
    }
  };

  const handleTitleChange = (event) => {
    setDocTitle(event.target.value);
  };

  const handleNewDocument = () => {
    contentEditableRef.current.innerHTML = "";
  };

  const handleUploadClick = () => {
    fileInputRef.current.click();
  };

  const printRef = useRef(null);

  const handlePrint = () => {
    if (printRef.current) {
      printRef.current.handlePrint();
    }
  };

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
          onBoldClick={handleBoldClick}
          onItalicClick={handleItalicClick}
          onNumberedListClick={handleNumberedListClick}
          onZoomInClick={handleZoomInClick}
          onZoomOutClick={handleZoomOutClick}
          onPrint={handlePrint}
          activeStyles={activeStyles}
          contentEditableRef={contentEditableRef} 
          printRef={printRef}
        />

        <NavigationBar
          docTitle={docTitle}
          onTitleChange={handleTitleChange}
          onNewDocument={handleNewDocument}
          onUploadClick={handleUploadClick}
          onEmailClick={handleEmail}
          onPrintClick={handlePrint}
          onSaveAsPDF={handleSaveAsPDF}
          onSaveAsCSV={handleSaveAsCSV}
          onUndo={() => document.execCommand("undo")}
          onRedo={() => document.execCommand("redo")}
          onBoldClick={handleBoldClick}
          onUnderlineClick={() => applyCommand("underline")}
          onStrikethroughClick={() => applyCommand("strikeThrough")}
          onSubscriptClick={() => applyCommand("subscript")}
          onSuperscriptClick={() => applyCommand("superscript")}
          onInsertImageClick={() => fileInputRef.current.click()}
          onAlignLeftClick={() => applyCommand("justifyLeft")}
          onAlignCenterClick={() => applyCommand("justifyCenter")}
          onAlignRightClick={() => applyCommand("justifyRight")}
          onJustifyClick={() => applyCommand("justifyFull")}
          onIndentClick={() => applyCommand("indent")}
          onOutdentClick={() => applyCommand("outdent")}
          onInsertLinkClick={() => {
            const url = prompt("Enter the URL:");
            if (url) {
              applyCommand("createLink", url);
            }
          }}
          onUnlinkClick={() => applyCommand("unlink")}
          onInsertTableClick={handleInsertTable}
          onInsertCodeBlockClick={() => applyCommand("formatBlock", "<pre>")}
          onInsertCommentClick={handleInsertComment}
          onFullScreenClick={() => {
            if (document.documentElement.requestFullscreen) {
              document.documentElement.requestFullscreen();
            } else if (document.documentElement.mozRequestFullScreen) {
              document.documentElement.mozRequestFullScreen();
            } else if (document.documentElement.webkitRequestFullscreen) {
              document.documentElement.webkitRequestFullscreen();
            } else if (document.documentElement.msRequestFullscreen) {
              document.documentElement.msRequestFullscreen();
            }
          }}
          onSettingsClick={handleSettingsClick}
          onHelpClick={handleHelpClick}
          onZoomInClick={handleZoomInClick}
          onZoomOutClick={handleZoomOutClick}
        />
        <div className="document-content">
          <div
            ref={contentEditableRef}
            contentEditable
            className="content-to-print"
            style={{
              padding: "20px",
              maxHeight: "120vh",
              color: "#000",
              backgroundColor: "#fff",
              overflow: "auto",
              fontSize: "16px",
              fontFamily: "Arial, sans-serif",
              textAlign: "left",
              top: "80px",
              outline: "none",
            }}
            onPaste={(event) => {
              event.preventDefault();
              const text = (event.clipboardData || window.clipboardData)
                .getData("text/plain")
                .replace(/\n/g, "<br />");
              document.execCommand("insertHTML", false, text);
            }}
          >
            Start writing your document here...
          </div>
        </div>
        {showHelp && <HelpMenu handleContinue={handleHelpMenuClose} />}
        <input
          type="file"
          ref={fileInputRef}
          style={{ display: "none" }}
          accept=".pdf,.png,.jpeg,.jpg"
          onChange={handleImageUpload}
        />
        {showResizeDialog && (
          <div className="resize-dialog">
            <h3>Resize Image</h3>
            <label>
              Width:
              <input
                type="number"
                name="width"
                value={imageDimensions.width}
                onChange={handleDimensionChange}
              />
              px
            </label>
            <label>
              Height:
              <input
                type="number"
                name="height"
                value={imageDimensions.height}
                onChange={handleDimensionChange}
              />
              px
            </label>
            <div className="resize-buttons">
              <button onClick={handleImageResize}>Apply</button>
              <button onClick={handleCancelResize}>Cancel</button>
            </div>
          </div>
        )}
      </div>
    </>
  );
}

