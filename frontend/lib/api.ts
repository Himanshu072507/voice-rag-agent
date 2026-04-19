// frontend/lib/api.ts
const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000";

export async function uploadPDF(file: File): Promise<{ session_id: string }> {
  const form = new FormData();
  form.append("file", file);

  const res = await fetch(`${BACKEND_URL}/upload`, { method: "POST", body: form });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail ?? "Upload failed");
  }
  return res.json();
}

export async function sendChat(
  session_id: string,
  query: string,
  message_id: string
): Promise<{ answer_text: string; audio_url: string | null }> {
  const res = await fetch(`${BACKEND_URL}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id, query, message_id }),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail ?? "Chat request failed");
  }
  return res.json();
}

export function getAudioURL(audio_url: string): string {
  if (audio_url.startsWith("http")) return audio_url;
  return `${BACKEND_URL}${audio_url}`;
}
