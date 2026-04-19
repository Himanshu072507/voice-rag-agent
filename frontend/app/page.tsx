"use client";
import { useState, useEffect, useRef, DragEvent } from "react";
import { v4 as uuidv4 } from "uuid";
import { uploadPDF, sendChat } from "@/lib/api";

interface Message {
  id: string;
  role: "user" | "assistant";
  text: string;
}

export default function HomePage() {
  const [groqApiKey, setGroqApiKey] = useState("");
  const [showKey, setShowKey] = useState(false);
  const [voices, setVoices] = useState<SpeechSynthesisVoice[]>([]);
  const [selectedVoiceURI, setSelectedVoiceURI] = useState("");

  const [sessionId, setSessionId] = useState<string | null>(null);
  const [uploadedFile, setUploadedFile] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

  const [messages, setMessages] = useState<Message[]>([]);
  const [query, setQuery] = useState("");
  const [thinking, setThinking] = useState(false);

  const bottomRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const load = () => {
      const v = window.speechSynthesis.getVoices();
      if (v.length > 0) setVoices(v);
    };
    load();
    window.speechSynthesis.onvoiceschanged = load;
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, thinking]);

  useEffect(() => {
    const last = messages[messages.length - 1];
    if (!last || last.role !== "assistant") return;
    const utterance = new SpeechSynthesisUtterance(last.text);
    const voice = voices.find((v) => v.voiceURI === selectedVoiceURI);
    if (voice) utterance.voice = voice;
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(utterance);
  }, [messages]);

  const handleFile = async (file: File) => {
    setUploadError(null);
    if (file.type !== "application/pdf") {
      setUploadError("Only PDF files are supported.");
      return;
    }
    if (file.size > 10_000_000) {
      setUploadError("File must be under 10MB.");
      return;
    }
    setUploading(true);
    try {
      const { session_id } = await uploadPDF(file);
      setSessionId(session_id);
      setUploadedFile(file.name);
      setMessages([]);
    } catch (err) {
      setUploadError(err instanceof Error ? err.message : "Upload failed.");
    } finally {
      setUploading(false);
    }
  };

  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  };

  const handleSend = async () => {
    if (!query.trim() || !sessionId || thinking) return;
    const text = query.trim();
    setQuery("");
    setMessages((prev) => [...prev, { id: uuidv4(), role: "user", text }]);
    setThinking(true);
    try {
      const messageId = uuidv4();
      const response = await sendChat(sessionId, text, messageId, groqApiKey || undefined);
      setMessages((prev) => [
        ...prev,
        { id: messageId, role: "assistant", text: response.answer_text },
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { id: uuidv4(), role: "assistant", text: "Something went wrong. Please try again." },
      ]);
    } finally {
      setThinking(false);
    }
  };

  return (
    <div className="flex h-screen bg-[#0f1117] text-gray-100 overflow-hidden">
      {/* Sidebar */}
      <aside className="w-64 shrink-0 bg-[#1a1f2e] border-r border-[#2d3748] flex flex-col p-5 gap-7 overflow-y-auto">
        <div className="flex items-center gap-2 pt-1">
          <span className="text-lg">🎙</span>
          <span className="font-semibold text-sm tracking-tight">Voice RAG Agent</span>
        </div>

        {/* Configuration */}
        <section className="flex flex-col gap-3">
          <h3 className="text-[10px] font-bold text-gray-500 uppercase tracking-widest">
            ⚙ Configuration
          </h3>
          <div>
            <label className="text-xs text-gray-400 mb-1.5 block">Groq API Key</label>
            <div className="relative">
              <input
                type={showKey ? "text" : "password"}
                value={groqApiKey}
                onChange={(e) => setGroqApiKey(e.target.value)}
                placeholder="gsk_..."
                className="w-full bg-[#252b3b] border border-[#2d3748] rounded-lg px-3 py-2 text-xs text-gray-200 placeholder-gray-600 focus:outline-none focus:border-indigo-500 pr-10"
              />
              <button
                type="button"
                onClick={() => setShowKey((s) => !s)}
                className="absolute right-2.5 top-1/2 -translate-y-1/2 text-[10px] text-gray-500 hover:text-gray-300"
              >
                {showKey ? "Hide" : "Show"}
              </button>
            </div>
            <p className="text-[10px] text-gray-600 mt-1">
              Uses server key if left empty.
            </p>
          </div>
        </section>

        {/* Voice Settings */}
        <section className="flex flex-col gap-3">
          <h3 className="text-[10px] font-bold text-gray-500 uppercase tracking-widest">
            🎤 Voice Settings
          </h3>
          <div>
            <label className="text-xs text-gray-400 mb-1.5 block">Select Voice</label>
            <select
              value={selectedVoiceURI}
              onChange={(e) => setSelectedVoiceURI(e.target.value)}
              className="w-full bg-[#252b3b] border border-[#2d3748] rounded-lg px-3 py-2 text-xs text-gray-200 focus:outline-none focus:border-indigo-500"
            >
              <option value="">Default</option>
              {voices.map((v) => (
                <option key={v.voiceURI} value={v.voiceURI}>
                  {v.name}
                </option>
              ))}
            </select>
          </div>
        </section>

        {/* Processed Documents */}
        <section className="flex flex-col gap-3 flex-1">
          <h3 className="text-[10px] font-bold text-gray-500 uppercase tracking-widest">
            📄 Processed Documents
          </h3>
          {uploadedFile ? (
            <div className="flex items-center gap-2 bg-[#252b3b] border border-[#2d3748] rounded-lg px-3 py-2">
              <span className="text-sm shrink-0">📄</span>
              <span className="text-xs text-gray-300 truncate">{uploadedFile}</span>
            </div>
          ) : (
            <p className="text-xs text-gray-600">No documents yet.</p>
          )}
        </section>
      </aside>

      {/* Main */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Header */}
        <header className="px-6 py-4 border-b border-[#2d3748] shrink-0">
          <h1 className="text-base font-semibold">🎙 Voice RAG Agent</h1>
          <p className="text-xs text-gray-500 mt-0.5">
            Upload a PDF and ask questions — answers are spoken aloud automatically.
          </p>
        </header>

        {/* Upload zone */}
        <div className="px-6 pt-4 pb-2 shrink-0">
          <p className="text-[10px] font-bold text-gray-500 uppercase tracking-widest mb-2">
            Upload PDF
          </p>
          <div
            className={`border-2 border-dashed rounded-xl p-5 cursor-pointer text-center transition-colors ${
              dragging
                ? "border-indigo-500 bg-indigo-950/20"
                : "border-[#2d3748] hover:border-[#4a5568]"
            }`}
            onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
            onClick={() => fileInputRef.current?.click()}
            role="button"
            aria-label="Upload PDF by clicking or dragging"
          >
            <p className="text-2xl mb-1">☁️</p>
            <p className="text-sm text-gray-400">
              {uploading ? "Uploading..." : "Drag and drop file here"}
            </p>
            <p className="text-xs text-gray-600 mt-0.5">Limit 10MB per file • PDF</p>
            <button
              type="button"
              onClick={(e) => { e.stopPropagation(); fileInputRef.current?.click(); }}
              className="mt-3 px-4 py-1.5 bg-[#252b3b] hover:bg-[#2d3748] border border-[#3d4758] text-xs text-gray-300 rounded-lg transition-colors"
            >
              Browse files
            </button>
          </div>

          <input
            ref={fileInputRef}
            type="file"
            accept="application/pdf"
            className="hidden"
            onChange={(e) => { const f = e.target.files?.[0]; if (f) handleFile(f); }}
            aria-label="PDF file input"
          />

          {uploadedFile && !uploading && (
            <div className="flex items-center gap-2 mt-2 bg-[#1a1f2e] border border-[#2d3748] rounded-lg px-3 py-2">
              <span className="text-sm">📄</span>
              <span className="text-xs text-gray-300 flex-1 truncate">{uploadedFile}</span>
              <button
                onClick={() => { setSessionId(null); setUploadedFile(null); setMessages([]); window.speechSynthesis.cancel(); }}
                className="text-gray-600 hover:text-red-400 text-lg leading-none"
                aria-label="Remove file"
              >
                ×
              </button>
            </div>
          )}

          {uploadError && (
            <p className="text-xs text-red-400 mt-2">{uploadError}</p>
          )}
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto px-6 py-4 space-y-3">
          {!sessionId && messages.length === 0 && (
            <p className="text-center text-gray-600 text-sm mt-12">
              Upload a PDF to start chatting.
            </p>
          )}
          {sessionId && messages.length === 0 && (
            <p className="text-center text-gray-600 text-sm mt-12">
              Ask anything about your document.
            </p>
          )}
          {messages.map((msg) => (
            <div
              key={msg.id}
              className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
            >
              <div
                className={`max-w-[75%] rounded-2xl px-4 py-3 text-sm leading-relaxed ${
                  msg.role === "user"
                    ? "bg-indigo-600 text-white"
                    : "bg-[#1a1f2e] text-gray-100 border border-[#2d3748]"
                }`}
              >
                {msg.text}
              </div>
            </div>
          ))}
          {thinking && (
            <div className="flex justify-start">
              <div className="bg-[#1a1f2e] border border-[#2d3748] rounded-2xl px-4 py-3 text-sm text-gray-500">
                Thinking...
              </div>
            </div>
          )}
          <div ref={bottomRef} />
        </div>

        {/* Query input */}
        <div className="px-6 pb-6 pt-3 border-t border-[#2d3748] shrink-0">
          <div className="flex items-center gap-3">
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  handleSend();
                }
              }}
              placeholder={
                sessionId
                  ? "What would you like to know about the document?"
                  : "Upload a PDF first..."
              }
              disabled={!sessionId || thinking}
              className="flex-1 bg-[#1a1f2e] border border-[#2d3748] rounded-xl px-4 py-3 text-sm text-gray-200 placeholder-gray-600 focus:outline-none focus:border-indigo-500 disabled:opacity-40 transition-colors"
              aria-label="Query input"
            />
            <button
              onClick={handleSend}
              disabled={!sessionId || thinking || !query.trim()}
              className="w-11 h-11 rounded-full bg-indigo-600 hover:bg-indigo-500 disabled:opacity-30 disabled:cursor-not-allowed flex items-center justify-center shrink-0 transition-colors"
              aria-label="Send query"
            >
              <svg
                xmlns="http://www.w3.org/2000/svg"
                viewBox="0 0 24 24"
                fill="currentColor"
                className="w-4 h-4"
              >
                <path d="M3.478 2.405a.75.75 0 00-.926.94l2.432 7.905H13.5a.75.75 0 010 1.5H4.984l-2.432 7.905a.75.75 0 00.926.94 60.519 60.519 0 0018.445-8.986.75.75 0 000-1.218A60.517 60.517 0 003.478 2.405z" />
              </svg>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
