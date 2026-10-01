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

const DEFAULT_PROD_API_URL = 'https://clinical-guidelines-assistant.onrender.com';

function getApiBaseUrl(): string {
  const envUrl = import.meta.env.VITE_API_BASE_URL?.trim();
  if (envUrl && envUrl !== 'http://localhost:8000') {
    return envUrl.replace(/\/+$/, '');
  }

  // If running in browser and NOT on localhost / 127.0.0.1, fallback to production Render backend URL
  if (typeof window !== 'undefined') {
    const hostname = window.location.hostname;
    if (hostname !== 'localhost' && hostname !== '127.0.0.1') {
      return DEFAULT_PROD_API_URL;
    }
  }

  return (envUrl || 'http://localhost:8000').replace(/\/+$/, '');
}

function getApiUrl(): string {
  return `${getApiBaseUrl()}/chat`;
}

function isChatRoute(value: unknown): value is ChatRoute {
  return value === 'general-info' || value === 'emergency' || value === 'out-of-scope';
}

function parseChatResponse(value: unknown): SendChatResponse {
  if (!value || typeof value !== 'object') {
    throw new Error('The chat service returned an invalid response format.');
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
  const controller = new AbortController();
  // 75 second timeout to accommodate Render free-tier cold starts (~50s delay)
  const timeoutId = setTimeout(() => controller.abort(), 75000);

  try {
    const url = getApiUrl();
    const response = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      if (response.status === 502 || response.status === 503 || response.status === 504) {
        throw new Error(
          `Backend service is currently starting up or temporarily unavailable (HTTP ${response.status}). Please wait a few moments and try again.`
        );
      }
      let errDetail = '';
      try {
        const errorJson = await response.json();
        if (errorJson && typeof errorJson.detail === 'string') {
          errDetail = `: ${errorJson.detail}`;
        }
      } catch {
        // ignore error parse failure
      }
      throw new Error(`Backend returned HTTP ${response.status}${errDetail}`);
    }

    let body: unknown;
    try {
      body = await response.json();
    } catch {
      throw new Error('The chat service returned unreadable JSON response.');
    }

    return parseChatResponse(body);
  } catch (error: any) {
    clearTimeout(timeoutId);
    if (error.name === 'AbortError') {
      throw new Error(
        'Backend request timed out. The server may be waking up from a cold start — please try sending your question again.'
      );
    }
    if (error instanceof TypeError && error.message.includes('fetch')) {
      throw new Error(
        `Could not connect to backend service at ${getApiBaseUrl()}. Please check your connection or CORS configuration.`
      );
    }
    throw error;
  }
}