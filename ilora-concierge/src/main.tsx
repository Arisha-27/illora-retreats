// main.tsx

import React from 'react'; // <--- ADD THIS LINE
import ReactDOM from 'react-dom/client';
import App from './IloraRetreatsConcierge.tsx';
import './style.css';
import IloraRetreatsConcierge from './IloraRetreatsConcierge.tsx';

ReactDOM.createRoot(document.getElementById('root')!).render(
  // This line requires the 'React' object to be defined
  <React.StrictMode>
    <IloraRetreatsConcierge />
  </React.StrictMode>,
);