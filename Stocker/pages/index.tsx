import { Dropdown, DropdownTrigger, DropdownMenu, DropdownItem } from "@nextui-org/dropdown";
import React, { useState } from 'react';

const SelectionMenu = ({ setSelectedStock }) => {
  const [selectedKey, setSelectedKey] = useState(null);

  const handleSelectionChange = (key) => {
    setSelectedKey(key);
    setSelectedStock(key);
  };

return (
  <Dropdown>
    <DropdownTrigger>
      <button className="inline-flex justify-center w-full rounded-md border border-gray-300 shadow-sm px-4 py-2 bg-white text-sm font-medium text-gray-700 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 menu">
        {selectedKey || "Select a Stock"}
      </button>
    </DropdownTrigger>
    <DropdownMenu
      aria-label="Stock Options"
      selectionMode="single"
      selectedKey={selectedKey}
      onSelectionChange={handleSelectionChange}
      className="custom-dropdown"
    >
      <DropdownItem key="AAPL" className="custom-dropdown-item">AAPL</DropdownItem>
      <DropdownItem key="GOOGL" className="custom-dropdown-item">GOOGL</DropdownItem>
      <DropdownItem key="MSFT" className="custom-dropdown-item">MSFT</DropdownItem>
      <DropdownItem key="SPOT" className="custom-dropdown-item">SPOT</DropdownItem>
      <DropdownItem key="TSLA" className="custom-dropdown-item">TSLA</DropdownItem>
      <DropdownItem key="VTI" className="custom-dropdown-item">VTI</DropdownItem>
    </DropdownMenu>
  </Dropdown>
);

};

export default function Home() {
  const [selectedStock, setSelectedStock] = useState(null);

  const handlePredictClick = () => {
    if (selectedStock) {
      alert(`Predicting stock prices for ${selectedStock}`);
    } else {
      alert('Please select a stock');
    }
  };

  return (
    <main className="home">
      <h1>Stocker</h1>
      <h2>Stock Market Forecast Full-Stack Web Application</h2>
      <p>About</p>
      <p>
        Stocker will forecast stock prices for the next seven days and display the data in an easy-to-understand graph. 
        Currently, this service is available for the following stocks: AAPL, GOOGL, MSFT, SPOT, TSLA, and VTI.
      </p>
      <SelectionMenu setSelectedStock={setSelectedStock} />
      <p>
      <button onClick={handlePredictClick} className="mt-4 px-4 py-2 bg-blue-600 text-white rounded-md">
        Predict
      </button>
      </p>
    </main>
  );
}

