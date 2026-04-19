"use client";
import { useState, useRef, DragEvent } from "react";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { uploadPDF } from "@/lib/api";

export default function HomePage() {
  const router = useRouter();
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleFile = async (file: File) => {
    setError(null);
    if (file.type !== "application/pdf") {
      setError("Only PDF files are supported.");
      return;
    }
    if (file.size > 10_000_000) {
      setError("File must be under 10MB.");
      return;
    }
    setLoading(true);
    try {
      const { session_id } = await uploadPDF(file);
      router.push(`/chat?session_id=${session_id}`);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Upload failed. Please try again.";
      setError(message);
    } finally {
      setLoading(false);
    }
  };

  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  };

  return (
    <main className="flex min-h-screen items-center justify-center bg-gray-50 p-4">
      <Card className="w-full max-w-md p-8 text-center">
        <h1 className="text-2xl font-bold mb-2">Voice PDF Chat</h1>
        <p className="text-gray-500 mb-6">Upload a PDF and ask questions out loud.</p>

        <div
          className={`border-2 border-dashed rounded-xl p-10 cursor-pointer transition-colors ${
            dragging ? "border-indigo-500 bg-indigo-50" : "border-gray-300"
          }`}
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
          onClick={() => inputRef.current?.click()}
          role="button"
          aria-label="Upload PDF by clicking or dragging"
        >
          <p className="text-gray-500">
            {loading ? "Uploading..." : "Drag & drop a PDF or click to browse"}
          </p>
        </div>

        <input
          ref={inputRef}
          type="file"
          accept="application/pdf"
          className="hidden"
          onChange={(e) => { const f = e.target.files?.[0]; if (f) handleFile(f); }}
          aria-label="PDF file input"
        />

        {error && <p className="mt-4 text-sm text-red-500">{error}</p>}

        <Button
          className="mt-6 w-full"
          onClick={() => inputRef.current?.click()}
          disabled={loading}
        >
          {loading ? "Processing..." : "Choose PDF"}
        </Button>
      </Card>
    </main>
  );
}
