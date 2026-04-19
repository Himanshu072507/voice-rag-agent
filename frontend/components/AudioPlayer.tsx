"use client";
import { useState } from "react";
import { Button } from "@/components/ui/button";

interface AudioPlayerProps {
  text: string;
}

export function AudioPlayer({ text }: AudioPlayerProps) {
  const [speaking, setSpeaking] = useState(false);

  const handleSpeak = () => {
    if (speaking) {
      window.speechSynthesis.cancel();
      setSpeaking(false);
      return;
    }
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.onend = () => setSpeaking(false);
    utterance.onerror = () => setSpeaking(false);
    setSpeaking(true);
    window.speechSynthesis.speak(utterance);
  };

  return (
    <div className="mt-2">
      <Button variant="outline" size="sm" onClick={handleSpeak}>
        {speaking ? "Stop" : "Speak"}
      </Button>
    </div>
  );
}
