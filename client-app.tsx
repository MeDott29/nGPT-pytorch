import React, { useState, useEffect, useRef } from 'react';
import { AlertCircle, Activity, Database } from 'lucide-react';

// Custom hook for WebSocket connection
const useWebSocket = (url) => {
  const [data, setData] = useState(null);
  const [connected, setConnected] = useState(false);
  const ws = useRef(null);

  useEffect(() => {
    ws.current = new WebSocket(url);
    
    ws.current.onopen = () => setConnected(true);
    ws.current.onclose = () => setConnected(false);
    ws.current.onmessage = (event) => setData(JSON.parse(event.data));

    return () => {
      if (ws.current) ws.current.close();
    };
  }, [url]);

  return [data, connected];
};

// Sandplot visualization component
const SandplotCanvas = ({ params }) => {
  const canvasRef = useRef(null);
  
  useEffect(() => {
    if (!params) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    const { density, point_size, prime_factors, color_variation } = params;
    const numPoints = Math.floor(density * canvas.width * canvas.height / 100);
    
    for (let i = 0; i < numPoints; i++) {
      const x = Math.random() * canvas.width;
      const y = Math.random() * canvas.height;
      const colorSeed = (x * prime_factors[0] + y * prime_factors[1]) % prime_factors[2];
      const hue = (colorSeed / prime_factors[2] + Math.random() * color_variation) * 360;
      
      ctx.fillStyle = `hsl(${hue}, 70%, 50%)`;
      ctx.beginPath();
      ctx.arc(x, y, point_size, 0, Math.PI * 2);
      ctx.fill();
    }
  }, [params]);

  return (
    <canvas
      ref={canvasRef}
      className="w-full h-96 bg-white rounded-lg shadow-lg"
      width={800}
      height={600}
    />
  );
};

// Activity log component
const ActivityLog = ({ messages }) => {
  return (
    <div className="h-48 overflow-y-auto bg-gray-50 rounded-lg p-4">
      {messages.map((msg, i) => (
        <div key={i} className="text-sm text-gray-600 mb-2">
          <span className="font-semibold">{new Date(msg.timestamp * 1000).toLocaleTimeString()}</span>
          {' - '}{msg.dataset}: {msg.text_hash.slice(0, 8)}...
        </div>
      ))}
    </div>
  );
};

// Statistics card component
const StatCard = ({ icon: Icon, label, value }) => (
  <div className="bg-white rounded-lg shadow p-4 flex items-center space-x-4">
    <Icon className="w-8 h-8 text-blue-500" />
    <div>
      <div className="text-sm text-gray-500">{label}</div>
      <div className="text-xl font-bold">{value}</div>
    </div>
  </div>
);

// Main application component
const MetaSandplotApp = () => {
  const [messages, setMessages] = useState([]);
  const [data, connected] = useWebSocket('ws://localhost:8765');
  
  useEffect(() => {
    if (data) {
      setMessages(prev => [...prev, data].slice(-100));
    }
  }, [data]);

  return (
    <div className="min-h-screen bg-gray-100 p-8">
      <header className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900">MetaSandplot System</h1>
        <div className="flex items-center mt-2">
          <div className={`w-3 h-3 rounded-full ${connected ? 'bg-green-500' : 'bg-red-500'} mr-2`} />
          <span className="text-sm text-gray-600">
            {connected ? 'Connected' : 'Disconnected'}
          </span>
        </div>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <div className="lg:col-span-2">
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h2 className="text-xl font-semibold mb-4">Live Sandplot</h2>
            <SandplotCanvas params={data?.params} />
          </div>
        </div>

        <div className="space-y-8">
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h2 className="text-xl font-semibold mb-4">Network Activity</h2>
            <ActivityLog messages={messages} />
          </div>

          <div className="space-y-4">
            <StatCard 
              icon={Activity}
              label="Messages Processed"
              value={messages.length}
            />
            <StatCard 
              icon={Database}
              label="Data Processed"
              value={`${(messages.length * 0.5).toFixed(1)} MB`}
            />
          </div>
        </div>
      </div>
    </div>
  );
};

export default MetaSandplotApp;