import { jsx as _jsx } from "react/jsx-runtime";
// main.tsx
import React from 'react'; // <--- ADD THIS LINE
import ReactDOM from 'react-dom/client';
import './style.css';
import IloraRetreatsConcierge from './IloraRetreatsConcierge';
ReactDOM.createRoot(document.getElementById('root')).render(
// This line requires the 'React' object to be defined
_jsx(React.StrictMode, { children: _jsx(IloraRetreatsConcierge, {}) }));
