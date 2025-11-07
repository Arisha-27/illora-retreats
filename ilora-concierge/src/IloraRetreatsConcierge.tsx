import React, { useState, useEffect, useRef } from 'react';
import { Send, Loader2, CheckCircle, User, Calendar, Home, MapPin } from 'lucide-react';

// Types
interface Message {
  id: string;
  text: string;
  sender: 'user' | 'bot';
  timestamp: Date;
  isTyping?: boolean;
  media_url?: string;
}

interface UserDetails {
  uid?: string;
  bookingStatus?: string;
  bookingId?: string;
  roomNumber?: string;
  checkIn?: string;
  checkOut?: string;
  pendingBalance?: number;
  status?: string;
}

// UPDATE: Added media_url to the response type
interface ChatResponse {
  reply: string;
  reply_parts?: string[];
  intent?: string;
  media_urls?: string[];
  actions?: {
    show_booking_form?: boolean;
    addons?: string[];
    payment_link?: string;
    pending_balance?: any;
  };
}

const API_BASE_URL = 'http://localhost:8000'; // Change this to your backend URL

const IloraRetreatsConcierge: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId] = useState(() => `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`);
  const [isGuest, setIsGuest] = useState(false);
  const [userEmail, setUserEmail] = useState('');
  const [userDetails, setUserDetails] = useState<UserDetails | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    // Initial welcome message
    addBotMessage(
      "🌿 Welcome to ILORA RETREATS 🌿\n\nYour gateway to luxury safari experiences in Kenya's Masai Mara.\n\nTo get started, please provide your email address to continue."
    );
  }, []);

  const addBotMessage = (text: string, isTyping = false, media_url?: string) => {
    const newMessage: Message = {
      id: `bot_${Date.now()}_${Math.random()}`,
      text,
      sender: 'bot',
      timestamp: new Date(),
      isTyping,
      media_url
    };
    setMessages(prev => [...prev, newMessage]);
  };

  const addUserMessage = (text: string) => {
    const newMessage: Message = {
      id: `user_${Date.now()}_${Math.random()}`,
      text,
      sender: 'user',
      timestamp: new Date()
    };
    setMessages(prev => [...prev, newMessage]);
  };

// FILE: IloraRetreatsConcierge.tsx
// ... (inside the IloraRetreatsConcierge component)

  const sendMessage = async () => {
    if (!inputMessage.trim() || isLoading) return;

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
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: userMsg,
          is_guest: isGuest,
          session_id: sessionId,
          email: userEmail || undefined
        })
      });

      if (!response.ok) {
        throw new Error('Network response was not ok');
      }

      const data: ChatResponse = await response.json();
      
      // ==================== NEW LOGIC FOR MULTIPLE IMAGES ====================
      // 1. Display all images first, one by one.
      if (data.media_urls && data.media_urls.length > 0) {
        for (const imageUrl of data.media_urls) {
          // Add a message with ONLY the image and no text.
          addBotMessage("", false, imageUrl);
          // Wait a little bit before showing the next image for a nice effect.
          await new Promise(resolve => setTimeout(resolve, 400));
        }
      }

      // 2. Then, display all the text parts.
      if (data.reply_parts && data.reply_parts.length > 0) {
        for (const part of data.reply_parts) {
          addBotMessage(part);
          await new Promise(resolve => setTimeout(resolve, 300));
        }
      } else if (!data.media_urls || data.media_urls.length === 0) {
        // Fallback for simple messages with no parts or images.
        addBotMessage(data.reply);
      }
      // =======================================================================

      if (data.reply.includes('VERIFIED GUEST') || data.reply.includes('full access')) {
        setIsGuest(true);
      }

      if (data.reply.includes('Room:') || data.reply.includes('Booking ID:')) {
        const details: UserDetails = {};
        
        const roomMatch = data.reply.match(/Room:\s*([^\n]+)/);
        if (roomMatch) details.roomNumber = roomMatch[1].trim();
        
        const bookingMatch = data.reply.match(/Booking ID:\s*([^\n]+)/);
        if (bookingMatch) details.bookingId = bookingMatch[1].trim();
        
        const checkinMatch = data.reply.match(/Check-in:\s*([^\n]+)/);
        if (checkinMatch) details.checkIn = checkinMatch[1].trim();
        
        const checkoutMatch = data.reply.match(/Check-out:\s*([^\n]+)/);
        if (checkoutMatch) details.checkOut = checkoutMatch[1].trim();
        
        setUserDetails(details);
      }

    } catch (error) {
      console.error('Error sending message:', error);
      addBotMessage('⚠️ Sorry, I encountered an error. Please try again or contact support.');
    } finally {
      setIsLoading(false);
    }
  };

// ... (rest of the component)

  const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const formatTimestamp = (date: Date) => {
    return date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-amber-50 via-green-50 to-teal-50">
      {/* Header */}
      <header className="bg-gradient-to-r from-green-800 to-teal-700 text-white shadow-lg">
        <div className="container mx-auto px-4 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="bg-white rounded-lg p-2">
                <MapPin className="w-8 h-8 text-green-700" />
              </div>
              <div>
                <h1 className="text-3xl font-bold tracking-tight">ILORA RETREATS</h1>
                <p className="text-green-100 text-sm">Luxury Safari Experiences • Masai Mara, Kenya</p>
              </div>
            </div>
            {isGuest && (
              <div className="flex items-center space-x-2 bg-green-900/30 px-4 py-2 rounded-full">
                <CheckCircle className="w-5 h-5 text-green-300" />
                <span className="text-sm font-medium">Verified Guest</span>
              </div>
            )}
          </div>
        </div>
      </header>

      <div className="container mx-auto px-4 py-8 max-w-7xl">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Sidebar - User Details (Optional) */}
          {userDetails && (
            <div className="lg:col-span-1 space-y-4">
              <div className="bg-white rounded-2xl shadow-xl p-6 border border-green-100">
                <h3 className="text-lg font-bold text-gray-800 mb-4 flex items-center">
                  <User className="w-5 h-5 mr-2 text-green-700" />
                  Your Details
                </h3>
                
                {userDetails.bookingId && (
                  <div className="mb-4 pb-4 border-b border-gray-100">
                    <p className="text-xs text-gray-500 mb-1">Booking ID</p>
                    <p className="text-sm font-semibold text-gray-800">{userDetails.bookingId}</p>
                  </div>
                )}
                
                {userDetails.roomNumber && (
                  <div className="mb-4 pb-4 border-b border-gray-100">
                    <p className="text-xs text-gray-500 mb-1 flex items-center">
                      <Home className="w-3 h-3 mr-1" />
                      Room
                    </p>
                    <p className="text-sm font-semibold text-gray-800">{userDetails.roomNumber}</p>
                  </div>
                )}
                
                {(userDetails.checkIn || userDetails.checkOut) && (
                  <div className="space-y-3">
                    {userDetails.checkIn && (
                      <div>
                        <p className="text-xs text-gray-500 mb-1 flex items-center">
                          <Calendar className="w-3 h-3 mr-1" />
                          Check-in
                        </p>
                        <p className="text-sm font-semibold text-gray-800">{userDetails.checkIn}</p>
                      </div>
                    )}
                    {userDetails.checkOut && (
                      <div>
                        <p className="text-xs text-gray-500 mb-1 flex items-center">
                          <Calendar className="w-3 h-3 mr-1" />
                          Check-out
                        </p>
                        <p className="text-sm font-semibold text-gray-800">{userDetails.checkOut}</p>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Main Chat Area */}
          <div className={`${userDetails ? 'lg:col-span-3' : 'lg:col-span-4'}`}>
            <div className="bg-white rounded-2xl shadow-2xl overflow-hidden border border-green-100">
              {/* Chat Header */}
              <div className="bg-gradient-to-r from-green-700 to-teal-600 p-6 text-white">
                <div className="flex items-center justify-between">
                  <div>
                    <h2 className="text-2xl font-bold mb-1">AI Concierge</h2>
                    <p className="text-green-100 text-sm">Riya, your personal assistant</p>
                  </div>
                  <div className="bg-white/20 backdrop-blur-sm rounded-full px-4 py-2">
                    <p className="text-xs font-medium">Session: {sessionId.slice(-8)}</p>
                  </div>
                </div>
              </div>

              {/* Messages Area */}
              <div className="h-[600px] overflow-y-auto bg-gradient-to-b from-green-50/30 to-white p-6 space-y-4">
                {messages.map((message) => (
                  <div
                    key={message.id}
                    className={`flex ${message.sender === 'user' ? 'justify-end' : 'justify-start'} animate-fadeIn`}
                  >
                    <div
                      className={`max-w-[75%] rounded-2xl px-5 py-3 shadow-md ${
                        message.sender === 'user'
                          ? 'bg-gradient-to-br from-green-600 to-teal-600 text-white'
                          : 'bg-white border border-green-100 text-gray-800'
                      }`}
                    >
                      {/* ==================== NEW: MEDIA RENDERING LOGIC ==================== */}
                      {message.media_url && (
                        <div className="mb-2">
                          <img
                            src={message.media_url}
                            alt="Chat media"
                            className="rounded-lg max-w-xs h-auto"
                          />
                        </div>
                      )}
                      {/* =================================================================== */}

                      <div className="whitespace-pre-wrap break-words leading-relaxed">
                        {message.text}
                      </div>
                      <div
                        className={`text-xs mt-2 ${
                          message.sender === 'user' ? 'text-green-100' : 'text-gray-400'
                        }`}
                      >
                        {formatTimestamp(message.timestamp)}
                      </div>
                    </div>
                  </div>
                ))}
                {isLoading && (
                  <div className="flex justify-start">
                    <div className="bg-white border border-green-100 rounded-2xl px-5 py-3 shadow-md">
                      <div className="flex items-center space-x-2">
                        <Loader2 className="w-5 h-5 animate-spin text-green-600" />
                        <span className="text-gray-600">Riya is typing...</span>
                      </div>
                    </div>
                  </div>
                )}
                <div ref={messagesEndRef} />
              </div>

              {/* Input Area */}
              <div className="border-t border-green-100 bg-gray-50 p-4">
                <div className="flex items-center space-x-3">
                  <input
                    type="text"
                    value={inputMessage}
                    onChange={(e) => setInputMessage(e.target.value)}
                    onKeyPress={handleKeyPress}
                    placeholder="Ask me anything about ILORA RETREATS..."
                    disabled={isLoading}
                    className="flex-1 px-5 py-3 border-2 border-green-200 rounded-full focus:outline-none focus:ring-2 focus:ring-green-500 focus:border-transparent disabled:bg-gray-100 disabled:cursor-not-allowed transition-all"
                  />
                  <button
                    onClick={sendMessage}
                    disabled={isLoading || !inputMessage.trim()}
                    className="bg-gradient-to-r from-green-600 to-teal-600 text-white rounded-full p-3 hover:from-green-700 hover:to-teal-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all transform hover:scale-105 active:scale-95 shadow-lg"
                  >
                    {isLoading ? (
                      <Loader2 className="w-6 h-6 animate-spin" />
                    ) : (
                      <Send className="w-6 h-6" />
                    )}
                  </button>
                </div>
              </div>
            </div>

            {/* Quick Actions */}
            <div className="mt-6 grid grid-cols-2 md:grid-cols-4 gap-3">
              {[
                { label: 'Book a Stay', icon: Home },
                { label: 'View Amenities', icon: CheckCircle },
                { label: 'Safari Experiences', icon: MapPin },
                { label: 'Contact Support', icon: User }
              ].map((action) => (
                <button
                  key={action.label}
                  onClick={() => {
                    setInputMessage(action.label);
                    // A tiny delay to ensure state updates before sending
                    setTimeout(() => document.querySelector<HTMLButtonElement>('button:not(:disabled) > svg.lucide-send')?.parentElement?.click(), 100);
                  }}
                  className="flex items-center justify-center space-x-2 bg-white hover:bg-green-50 border-2 border-green-200 hover:border-green-400 rounded-xl p-3 transition-all transform hover:scale-105"
                >
                  <action.icon className="w-4 h-4 text-green-700" />
                  <span className="text-sm font-medium text-gray-700">{action.label}</span>
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      <style>{`
        @keyframes fadeIn {
          from {
            opacity: 0;
            transform: translateY(10px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        .animate-fadeIn {
          animation: fadeIn 0.3s ease-out;
        }
      `}</style>
    </div>
  );
};

export default IloraRetreatsConcierge;