"use client";
import { useState, useEffect, useRef, Suspense } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { v4 as uuidv4 } from "uuid";
import { ChatBubble, Message } from "@/components/ChatBubble";
import { ChatInput } from "@/components/ChatInput";
import { sendChat } from "@/lib/api";

function ChatPageInner() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const sessionId = searchParams.get("session_id");
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!sessionId) router.replace("/");
  }, [sessionId, router]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSend = async (text: string) => {
    const userMsg: Message = { id: uuidv4(), role: "user", text };
    setMessages((prev) => [...prev, userMsg]);
    setLoading(true);

    try {
      const messageId = uuidv4();
      const response = await sendChat(sessionId!, text, messageId);
      const assistantMsg: Message = {
        id: messageId,
        role: "assistant",
        text: response.answer_text,
        audio_url: response.audio_url,
      };
      setMessages((prev) => [...prev, assistantMsg]);
    } catch (err) {
      const errorMsg: Message = {
        id: uuidv4(),
        role: "assistant",
        text: "Something went wrong. Please try again.",
      };
      setMessages((prev) => [...prev, errorMsg]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-screen max-w-3xl mx-auto">
      <header className="p-4 border-b font-semibold text-lg">PDF Chat</header>
      <main className="flex-1 overflow-y-auto p-4">
        {messages.length === 0 && (
          <p className="text-center text-gray-400 mt-20">
            Ask anything about your PDF.
          </p>
        )}
        {messages.map((msg) => (
          <ChatBubble key={msg.id} message={msg} />
        ))}
        {loading && (
          <div className="flex justify-start mb-4">
            <div className="bg-gray-100 rounded-2xl px-4 py-3 text-sm text-gray-500">
              Thinking...
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </main>
      <ChatInput onSend={handleSend} disabled={loading} />
    </div>
  );
}

export default function ChatPage() {
  return (
    <Suspense>
      <ChatPageInner />
    </Suspense>
  );
}
