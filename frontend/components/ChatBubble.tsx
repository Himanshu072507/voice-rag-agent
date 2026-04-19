// frontend/components/ChatBubble.tsx
import { AudioPlayer } from "./AudioPlayer";
import { getAudioURL } from "@/lib/api";

export interface Message {
  id: string;
  role: "user" | "assistant";
  text: string;
  audio_url?: string | null;
}

interface ChatBubbleProps {
  message: Message;
}

export function ChatBubble({ message }: ChatBubbleProps) {
  const isUser = message.role === "user";

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"} mb-4`}>
      <div
        className={`max-w-[70%] rounded-2xl px-4 py-3 ${
          isUser
            ? "bg-indigo-600 text-white"
            : "bg-gray-100 text-gray-900"
        }`}
      >
        <p className="text-sm leading-relaxed">{message.text}</p>
        {!isUser && message.audio_url && (
          <AudioPlayer audioUrl={getAudioURL(message.audio_url)} />
        )}
      </div>
    </div>
  );
}
