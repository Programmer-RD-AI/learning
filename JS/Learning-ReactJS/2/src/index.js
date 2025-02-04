import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import { BrowserRouter } from "react-router-dom";
// HashRouter -> Stores only the hashed part of the URL in the browser
// HistoryRouter -> Can take advantage of the HTML5 history API
// MemoryRouter -> Keeps the history of the URL in memory / For tests that dont connect to the Browser
// StaticRouter -> For server-side rendering
// NativeRouter -> For React Native
const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(
  <React.StrictMode>
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </React.StrictMode>
);
