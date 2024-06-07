import React from 'react';

const Navbar = ({ scrollToSection }) => {
  return (
    <nav className="fixed top-0 left-0 right-0 bg-white shadow-md p-4 z-10">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          <div className="flex-shrink-0">
            <a href="/" className="text-xl font-bold text-gray-700">Stocker</a>
          </div>
          <div className="hidden md:flex space-x-8">
            <button onClick={() => scrollToSection('about')} className="text-gray-700 hover:text-gray-900">About</button>
            <button onClick={() => scrollToSection('services')} className="text-gray-700 hover:text-gray-900">Services</button>
            <button onClick={() => scrollToSection('contact')} className="text-gray-700 hover:text-gray-900">Contact</button>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;


  
