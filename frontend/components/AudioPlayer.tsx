"use client";
import { useEffect, useRef } from "react";
import WaveSurfer from "wavesurfer.js";
import { Button } from "@/components/ui/button";

interface AudioPlayerProps {
  audioUrl: string;
}

export function AudioPlayer({ audioUrl }: AudioPlayerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const wavesurferRef = useRef<WaveSurfer | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    wavesurferRef.current = WaveSurfer.create({
      container: containerRef.current,
      waveColor: "#6366f1",
      progressColor: "#4f46e5",
      height: 40,
      barWidth: 2,
    });

    wavesurferRef.current.load(audioUrl);

    return () => {
      wavesurferRef.current?.destroy();
    };
  }, [audioUrl]);

  const togglePlay = () => wavesurferRef.current?.playPause();

  return (
    <div className="flex items-center gap-2 mt-2">
      <Button variant="outline" size="sm" onClick={togglePlay}>
        Play / Pause
      </Button>
      <div ref={containerRef} className="flex-1" />
    </div>
  );
}
