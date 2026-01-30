// main.tsx

import React from 'react'; // <--- ADD THIS LINE
import ReactDOM from 'react-dom/client';

import './style.css';
import App from './IloraRetreatsConcierge';
import IloraRetreatsConcierge from './IloraRetreatsConcierge';

ReactDOM.createRoot(document.getElementById('root')!).render(
  // This line requires the 'React' object to be defined
  <React.StrictMode>
    <IloraRetreatsConcierge />
  </React.StrictMode>,
);