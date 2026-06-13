import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App.jsx';

//Routing - allows for navigation between different pages or views in a single-page application (SPA) without requiring a full page reload. This is typically done using libraries like React Router, which manage the application's URL and render different components based on the current route.
ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
