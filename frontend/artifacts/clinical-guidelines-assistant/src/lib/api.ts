export type ChatRoute = 'general-info' | 'emergency' | 'out-of-scope';

export interface ChatSource {
  title: string;
  url: string;
}

export interface SendChatRequest {
  message: string;
  conversation_id: string;
}

export interface SendChatResponse {
  answer: string;
  route: ChatRoute;
  sources: ChatSource[];
}

function getApiUrl(): string {
  const baseUrl = import.meta.env.VITE_API_BASE_URL?.trim();
  if (!baseUrl) {
    throw new Error('The chat service is not configured yet. Please try again later.');
  }
  return `${baseUrl.replace(/\/+$/, '')}/chat`;
}

function isChatRoute(value: unknown): value is ChatRoute {
  return value === 'general-info' || value === 'emergency' || value === 'out-of-scope';
}

function parseChatResponse(value: unknown): SendChatResponse {
  if (!value || typeof value !== 'object') {
    throw new Error('The chat service returned an unexpected response.');
  }

  const candidate = value as Record<string, unknown>;
  if (typeof candidate.answer !== 'string' || !isChatRoute(candidate.route) || !Array.isArray(candidate.sources)) {
    throw new Error('The chat service returned an incomplete response.');
  }

  const sources = candidate.sources.filter(
    (source): source is ChatSource =>
      !!source &&
      typeof source === 'object' &&
      typeof (source as Record<string, unknown>).title === 'string' &&
      typeof (source as Record<string, unknown>).url === 'string',
  );

  return { answer: candidate.answer, route: candidate.route, sources };
}

export async function sendChatMessage(payload: SendChatRequest): Promise<SendChatResponse> {
  const response = await fetch(getApiUrl(), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    throw new Error(`The chat service could not respond (${response.status}). Please try again.`);
  }

  let body: unknown;
  try {
    body = await response.json();
  } catch {
    throw new Error('The chat service returned unreadable data. Please try again.');
  }

  return parseChatResponse(body);
}