import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useState, useEffect, useRef } from 'react';
import { Send, Loader2, CheckCircle, User, Calendar, Home, MapPin } from 'lucide-react';

// --- NEW: Helper Component for Rich Text & Table Formatting ---
const FormattedMessage = ({ content }) => {
    // Helper function to parse for **bold** text within a string
    const parseBold = (text) => {
        const parts = text.split(/\*\*(.*?)\*\*/g);
        return parts.map((part, i) => i % 2 === 1 ? _jsx("strong", { children: part }, i) : part);
    };
    const elements = [];
    let tableLines = [];
    let keyIndex = 0;
    const renderTable = () => {
        if (tableLines.length === 0)
            return;
        const tableData = tableLines.map(line => line.split('||').map(cell => cell.trim()));
        elements.push(_jsx("div", { className: "overflow-x-auto my-2", children: _jsxs("table", { className: "min-w-full divide-y divide-green-200 border border-green-200", children: [_jsx("thead", { className: "bg-green-50", children: _jsx("tr", { children: tableData[0].map((header, index) => (_jsx("th", { scope: "col", className: "px-4 py-3 text-left text-xs font-bold text-green-800 uppercase tracking-wider", children: parseBold(header) }, index))) }) }), _jsx("tbody", { className: "bg-white divide-y divide-green-100", children: tableData.slice(1).map((row, rowIndex) => (_jsx("tr", { children: row.map((cell, cellIndex) => (_jsx("td", { className: "px-4 py-3 whitespace-nowrap text-sm text-gray-700", children: parseBold(cell) }, cellIndex))) }, rowIndex))) })] }) }, `table-wrapper-${keyIndex++}`));
        tableLines = []; // Reset after rendering
    };
    content.split('\n').forEach(line => {
        if (line.includes('||')) {
            tableLines.push(line);
        }
        else {
            // If we have pending table lines, render them before processing this new line
            if (tableLines.length > 0) {
                renderTable();
            }
            if (line.trim()) { // Don't render empty lines as paragraphs
                elements.push(_jsx("p", { className: "my-1", children: parseBold(line) }, `p-${keyIndex++}`));
            }
        }
    });
    // Render any remaining table lines if the message ends with a table
    if (tableLines.length > 0) {
        renderTable();
    }
    return _jsx("div", { children: elements });
};

// --- NEW: Helper Component for the Typing Animation ---
const TypingIndicator = () => {
    const [line1, setLine1] = useState('');
    const [showLine2, setShowLine2] = useState(false);
    const fullLine1 = 'Ilora Retreats Concierge is at your service..Riya the Front Desk Concierge will be here to assist you..';
    const typingSpeed = 40; // ms
    useEffect(() => {
        if (line1.length < fullLine1.length) {
            const timer = setTimeout(() => {
                setLine1(fullLine1.substring(0, line1.length + 1));
            }, typingSpeed);
            return () => clearTimeout(timer);
        }
        else {
            // Wait a bit after line 1 is done, then show line 2
            const line2Timer = setTimeout(() => setShowLine2(true), 200);
            return () => clearTimeout(line2Timer);
        }
    }, [line1]);
    return (_jsx("div", { className: "flex justify-start animate-fadeIn", children: _jsxs("div", { className: "bg-white border border-green-100 rounded-2xl px-5 py-3 shadow-md", children: [_jsx("div", { className: "text-gray-800 font-medium", children: line1 }), showLine2 && (_jsxs("div", { className: "flex items-center space-x-2 mt-2", children: [_jsx("span", { className: "text-gray-600", children: "Riya is typing" }), _jsx("span", { className: "bouncing-dot", children: "." }), _jsx("span", { className: "bouncing-dot", style: { animationDelay: '0.2s' }, children: "." }), _jsx("span", { className: "bouncing-dot", style: { animationDelay: '0.4s' }, children: "." })] }))] }) }));
};

const API_BASE_URL = 'http://localhost:8000';

const IloraRetreatsConcierge = () => {
    const [messages, setMessages] = useState([]);
    const [inputMessage, setInputMessage] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [sessionId] = useState(() => `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`);
    const [isGuest, setIsGuest] = useState(false);
    const [userEmail, setUserEmail] = useState('');
    const [userDetails, setUserDetails] = useState(null);
    
    // Refs
    const messagesEndRef = useRef(null);
    const inputRef = useRef(null); // Added ref for input

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages, isLoading]);

    // Added effect to focus input when loading finishes
    useEffect(() => {
        if (!isLoading) {
            const timer = setTimeout(() => {
                inputRef.current?.focus();
            }, 100);
            return () => clearTimeout(timer);
        }
    }, [isLoading]);

    useEffect(() => {
        addBotMessage("🌿 Welcome to ILORA RETREATS 🌿\n\nYour gateway to luxury safari experiences in Kenya's Masai Mara.\n\nTo get started, please provide your email address to continue.");
    }, []);

    const addBotMessage = (text, media_url) => {
        const newMessage = {
            id: `bot_${Date.now()}_${Math.random()}`,
            text,
            sender: 'bot',
            timestamp: new Date(),
            media_url
        };
        setMessages(prev => [...prev, newMessage]);
    };

    const addUserMessage = (text) => {
        const newMessage = {
            id: `user_${Date.now()}_${Math.random()}`,
            text,
            sender: 'user',
            timestamp: new Date()
        };
        setMessages(prev => [...prev, newMessage]);
    };

    const sendMessage = async () => {
        if (!inputMessage.trim() || isLoading)
            return;
        const userMsg = inputMessage.trim();
        setInputMessage('');
        addUserMessage(userMsg);
        setIsLoading(true);
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        if (emailRegex.test(userMsg) && !userEmail) {
            setUserEmail(userMsg);
        }
        try {
            const response = await fetch(`${API_BASE_URL}/chat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message: userMsg,
                    is_guest: isGuest,
                    session_id: sessionId,
                    email: userEmail || undefined
                })
            });
            if (!response.ok)
                throw new Error('Network response was not ok');
            const data = await response.json();
            // --- Process and add response messages ---
            if (data.media_urls && data.media_urls.length > 0) {
                for (const imageUrl of data.media_urls) {
                    addBotMessage("", imageUrl);
                    await new Promise(resolve => setTimeout(resolve, 400));
                }
            }
            if (data.reply_parts && data.reply_parts.length > 0) {
                for (const part of data.reply_parts) {
                    addBotMessage(part);
                    await new Promise(resolve => setTimeout(resolve, 300));
                }
            }
            else if (!data.media_urls || data.media_urls.length === 0) {
                addBotMessage(data.reply);
            }
            if (data.reply.includes('VERIFIED GUEST') || data.reply.includes('full access')) {
                setIsGuest(true);
            }
            if (data.reply.includes('Room:') || data.reply.includes('Booking ID:')) {
                const details = {};
                const roomMatch = data.reply.match(/Room:\s*([^\n]+)/);
                if (roomMatch)
                    details.roomNumber = roomMatch[1].trim();
                const bookingMatch = data.reply.match(/Booking ID:\s*([^\n]+)/);
                if (bookingMatch)
                    details.bookingId = bookingMatch[1].trim();
                const checkinMatch = data.reply.match(/Check-in:\s*([^\n]+)/);
                if (checkinMatch)
                    details.checkIn = checkinMatch[1].trim();
                const checkoutMatch = data.reply.match(/Check-out:\s*([^\n]+)/);
                if (checkoutMatch)
                    details.checkOut = checkoutMatch[1].trim();
                setUserDetails(details);
            }
        }
        catch (error) {
            console.error('Error sending message:', error);
            addBotMessage('⚠️ Sorry, I encountered an error. Please try again or contact support.');
        }
        finally {
            setIsLoading(false);
        }
    };

    const handleKeyPress = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    };

    const formatTimestamp = (date) => {
        return date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
    };

    return (_jsxs("div", { className: "min-h-screen bg-gradient-to-br from-amber-50 via-green-50 to-teal-50", children: [_jsx("header", { className: "bg-gradient-to-r from-green-800 to-teal-700 text-white shadow-lg", children: _jsx("div", { className: "container mx-auto px-4 py-6", children: _jsxs("div", { className: "flex items-center justify-between", children: [_jsxs("div", { className: "flex items-center space-x-4", children: [_jsx("div", { className: "bg-white rounded-lg p-2", children: _jsx(MapPin, { className: "w-8 h-8 text-green-700" }) }), _jsxs("div", { children: [_jsx("h1", { className: "text-3xl font-bold tracking-tight", children: "ILORA RETREATS" }), _jsx("p", { className: "text-green-100 text-sm", children: "Luxury Safari Experiences \u2022 Masai Mara, Kenya" })] })] }), isGuest && (_jsxs("div", { className: "flex items-center space-x-2 bg-green-900/30 px-4 py-2 rounded-full", children: [_jsx(CheckCircle, { className: "w-5 h-5 text-green-300" }), _jsx("span", { className: "text-sm font-medium", children: "Verified Guest" })] }))] }) }) }), _jsx("div", { className: "container mx-auto px-4 py-8 max-w-7xl", children: _jsxs("div", { className: "grid grid-cols-1 lg:grid-cols-4 gap-6", children: [userDetails && (_jsx("div", { className: "lg:col-span-1 space-y-4", children: _jsxs("div", { className: "bg-white rounded-2xl shadow-xl p-6 border border-green-100", children: [_jsxs("h3", { className: "text-lg font-bold text-gray-800 mb-4 flex items-center", children: [_jsx(User, { className: "w-5 h-5 mr-2 text-green-700" }), "Your Details"] }), userDetails.bookingId && (_jsxs("div", { className: "mb-4 pb-4 border-b border-gray-100", children: [_jsx("p", { className: "text-xs text-gray-500 mb-1", children: "Booking ID" }), _jsx("p", { className: "text-sm font-semibold text-gray-800", children: userDetails.bookingId })] })), userDetails.roomNumber && (_jsxs("div", { className: "mb-4 pb-4 border-b border-gray-100", children: [_jsxs("p", { className: "text-xs text-gray-500 mb-1 flex items-center", children: [_jsx(Home, { className: "w-3 h-3 mr-1" }), "Room"] }), _jsx("p", { className: "text-sm font-semibold text-gray-800", children: userDetails.roomNumber })] })), (userDetails.checkIn || userDetails.checkOut) && (_jsxs("div", { className: "space-y-3", children: [userDetails.checkIn && (_jsxs("div", { children: [_jsxs("p", { className: "text-xs text-gray-500 mb-1 flex items-center", children: [_jsx(Calendar, { className: "w-3 h-3 mr-1" }), "Check-in"] }), _jsx("p", { className: "text-sm font-semibold text-gray-800", children: userDetails.checkIn })] })), userDetails.checkOut && (_jsxs("div", { children: [_jsxs("p", { className: "text-xs text-gray-500 mb-1 flex items-center", children: [_jsx(Calendar, { className: "w-3 h-3 mr-1" }), "Check-out"] }), _jsx("p", { className: "text-sm font-semibold text-gray-800", children: userDetails.checkOut })] }))] }))] }) })), _jsxs("div", { className: `${userDetails ? 'lg:col-span-3' : 'lg:col-span-4'}`, children: [_jsxs("div", { className: "bg-white rounded-2xl shadow-2xl overflow-hidden border border-green-100", children: [_jsx("div", { className: "bg-gradient-to-r from-green-700 to-teal-600 p-6 text-white", children: _jsxs("div", { className: "flex items-center justify-between", children: [_jsxs("div", { children: [_jsx("h2", { className: "text-2xl font-bold mb-1", children: "AI Concierge" }), _jsx("p", { className: "text-green-100 text-sm", children: "Riya, your personal assistant" })] }), _jsx("div", { className: "bg-white/20 backdrop-blur-sm rounded-full px-4 py-2", children: _jsxs("p", { className: "text-xs font-medium", children: ["Session: ", sessionId.slice(-8)] }) })] }) }), _jsxs("div", { className: "h-[600px] overflow-y-auto bg-gradient-to-b from-green-50/30 to-white p-6 space-y-4", children: [messages.map((message) => (_jsx("div", { className: `flex ${message.sender === 'user' ? 'justify-end' : 'justify-start'} animate-fadeIn`, children: _jsxs("div", { className: `max-w-[75%] rounded-2xl px-5 py-3 shadow-md ${message.sender === 'user'
                                                            ? 'bg-gradient-to-br from-green-600 to-teal-600 text-white'
                                                            : 'bg-white border border-green-100 text-gray-800'}`, children: [message.media_url && (_jsx("div", { className: "mb-2", children: _jsx("img", { src: message.media_url, alt: "Chat media", className: "rounded-lg max-w-xs h-auto" }) })), _jsx("div", { className: "whitespace-pre-wrap break-words leading-relaxed", children: message.text ? _jsx(FormattedMessage, { content: message.text }) : null }), _jsx("div", { className: `text-xs mt-2 ${message.sender === 'user' ? 'text-green-100' : 'text-gray-400'}`, children: formatTimestamp(message.timestamp) })] }) }, message.id))), isLoading && _jsx(TypingIndicator, {}), _jsx("div", { ref: messagesEndRef })] }), _jsx("div", { className: "border-t border-green-100 bg-gray-50 p-4", children: _jsxs("div", { className: "flex items-center space-x-3", children: [_jsx("input", { ref: inputRef, type: "text", value: inputMessage, onChange: (e) => setInputMessage(e.target.value), onKeyPress: handleKeyPress, placeholder: "Ask me anything about ILORA RETREATS...", disabled: isLoading, className: "flex-1 px-5 py-3 border-2 border-green-200 rounded-full focus:outline-none focus:ring-2 focus:ring-green-500 focus:border-transparent disabled:bg-gray-100 disabled:cursor-not-allowed transition-all" }), _jsx("button", { onClick: sendMessage, disabled: isLoading || !inputMessage.trim(), className: "bg-gradient-to-r from-green-600 to-teal-600 text-white rounded-full p-3 hover:from-green-700 hover:to-teal-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all transform hover:scale-105 active:scale-95 shadow-lg", children: isLoading ? _jsx(Loader2, { className: "w-6 h-6 animate-spin" }) : _jsx(Send, { className: "w-6 h-6" }) })] }) })] }), _jsx("div", { className: "mt-6 grid grid-cols-2 md:grid-cols-4 gap-3", children: [
                                        { label: 'Book a Stay', icon: Home },
                                        { label: 'View Amenities', icon: CheckCircle },
                                        { label: 'Safari Experiences', icon: MapPin },
                                        { label: 'Contact Support', icon: User }
                                    ].map((action) => (_jsxs("button", { onClick: () => {
                                            setInputMessage(action.label);
                                            setTimeout(() => document.querySelector('button:not(:disabled) > svg.lucide-send')?.parentElement?.click(), 100);
                                        }, className: "flex items-center justify-center space-x-2 bg-white hover:bg-green-50 border-2 border-green-200 hover:border-green-400 rounded-xl p-3 transition-all transform hover:scale-105", children: [_jsx(action.icon, { className: "w-4 h-4 text-green-700" }), _jsx("span", { className: "text-sm font-medium text-gray-700", children: action.label })] }, action.label))) })] })] }) }), _jsx("style", { children: `
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(10px); }
          to { opacity: 1; transform: translateY(0); }
        }
        .animate-fadeIn { animation: fadeIn 0.3s ease-out; }

        @keyframes bounce {
          0%, 80%, 100% { transform: scale(0); }
          40% { transform: scale(1.0); }
        }
        .bouncing-dot {
            display: inline-block;
            width: 6px;
            height: 6px;
            background-color: #4b5563; /* gray-600 */
            border-radius: 50%;
            animation: bounce 1.4s infinite ease-in-out both;
        }
      ` })] }));
};
export default IloraRetreatsConcierge;