import React from "react";
import Navbar from "react-bootstrap/Navbar";
import Nav from "react-bootstrap/Nav";
import "bootstrap/dist/css/bootstrap.min.css";
import Form from "react-bootstrap/Form";

export default function NavigationBar() {
  const [docTitle, setDocTitle] = useState("Untitled Document");

  const handleTitleChange = (e) => {
    setDocTitle(e.target.value);
  };

  return (
    <Navbar fixed="top" style={{ backgroundColor: "white", borderBottom: "1px solid #ccc", padding: "0 20px" }}>
      <Form inline style={{ marginRight: "20px" }}>
        <Form.Control
          type="text"
          value={docTitle}
          onChange={handleTitleChange}
          style={{ width: "300px", marginRight: "20px" }}
        />
      </Form>
      <Nav className="mr-auto">
        <Nav.Link href="#file">File</Nav.Link>
        <Nav.Link href="#edit">Edit</Nav.Link>
        <Nav.Link href="#view">View</Nav.Link>
        <Nav.Link href="#insert">Insert</Nav.Link>
        <Nav.Link href="#format">Format</Nav.Link>
        <Nav.Link href="#tools">Tools</Nav.Link>
        <Nav.Link href="#extensions">Extensions</Nav.Link>
        <Nav.Link href="#help">Help</Nav.Link>
      </Nav>
    </Navbar>
  );
}
