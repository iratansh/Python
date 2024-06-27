import React, { useState, useEffect } from 'react';
import { Modal, Button, Form } from 'react-bootstrap';

const Settings = ({ show, handleClose, settings, onChange }) => {

  const handleToggle = (setting) => {
    onChange({
      ...settings,
      [setting]: !settings[setting],
    });
  };

  return (
    <Modal show={show} onHide={handleClose}>
      <Modal.Header closeButton>
        <Modal.Title>Settings</Modal.Title>
      </Modal.Header>
      <Modal.Body>
        <Form>
          <Form.Group controlId="autoCompleteSwitch" className="mb-3">
            <Form.Label>AutoComplete</Form.Label>
            <Form.Check
              type="switch"
              checked={settings.autoComplete}
              onChange={() => handleToggle('autoComplete')}
            />
          </Form.Group>

          <Form.Group controlId="darkModeSwitch" className="mb-3">
            <Form.Label>Dark Mode</Form.Label>
            <Form.Check
              type="switch"
              checked={settings.darkMode}
              onChange={() => handleToggle('darkMode')}
            />
          </Form.Group>

        </Form>
      </Modal.Body>
      <Modal.Footer>
        <Button variant="secondary" onClick={handleClose}>
          Close
        </Button>
      </Modal.Footer>
    </Modal>
  );
};

export default Settings;


