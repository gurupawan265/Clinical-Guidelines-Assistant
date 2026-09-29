import { type FormEvent, type KeyboardEvent, useEffect, useMemo, useRef, useState } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import {
  AlertTriangle,
  ArrowUpRight,
  BookOpen,
  CheckCircle2,
  CircleHelp,
  Info,
  LoaderCircle,
  Moon,
  RefreshCw,
  Send,
  ShieldCheck,
  Siren,
  Stethoscope,
  Sun,
} from 'lucide-react';
import { ErrorBoundary } from '@/components/error-boundary';
import { Toaster } from '@/components/ui/toaster';
import { TooltipProvider } from '@/components/ui/tooltip';
import { type ChatRoute, type ChatSource, sendChatMessage } from '@/lib/api';
import { Route, Switch, useLocation, Router as WouterRouter } from 'wouter';
import NotFound from '@/pages/not-found';

const queryClient = new QueryClient();
const CONVERSATION_KEY = 'clinical-guidelines-conversation-id';
const MESSAGES_KEY = 'clinical-guidelines-messages';
const THEME_KEY = 'clinical-guidelines-theme';

type Message =
  | { id: string; role: 'user'; content: string }
  | { id: string; role: 'assistant'; content: string; route: ChatRoute; sources: ChatSource[] }
  | { id: string; role: 'error'; content: string };

const starterPrompts = [
  'What are common symptoms of asthma?',
  'What is hypertension?',
  'What are current recommendations for adult vaccinations?',
  'How can I find reliable guidance about a medication?',
];

function makeId(): string {
  return typeof crypto !== 'undefined' && 'randomUUID' in crypto
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function readSession<T>(key: string, fallback: T): T {
  try {
    const stored = sessionStorage.getItem(key);
    return stored ? (JSON.parse(stored) as T) : fallback;
  } catch {
    return fallback;
  }
}

function readTheme(): boolean {
  try {
    return localStorage.getItem(THEME_KEY) === 'dark';
  } catch {
    return false;
  }
}

function routeStatus(route: ChatRoute) {
  if (route === 'emergency') {
    return {
      label: 'Urgent-care recommendation',
      className: 'border-red-200 bg-red-50 text-red-800 dark:border-red-400/35 dark:bg-red-950/45 dark:text-red-200',
      articleClassName: 'border-l-2 border-l-red-300 pl-4 sm:pl-5 dark:border-l-red-400/70',
      markClassName: 'border-red-200 bg-red-50 text-red-800 dark:border-red-400/35 dark:bg-red-950/45 dark:text-red-200',
      icon: Siren,
    };
  }
  if (route === 'out-of-scope') {
    return {
      label: 'Declined to answer',
      className: 'border-amber-200 bg-amber-50 text-amber-900 dark:border-amber-400/35 dark:bg-amber-950/45 dark:text-amber-200',
      articleClassName: 'border-l-2 border-l-amber-300 pl-4 sm:pl-5 dark:border-l-amber-400/70',
      markClassName: 'border-amber-200 bg-amber-50 text-amber-900 dark:border-amber-400/35 dark:bg-amber-950/45 dark:text-amber-200',
      icon: AlertTriangle,
    };
  }
  return {
    label: 'General information',
    className: 'border-emerald-200 bg-emerald-50 text-emerald-900 dark:border-emerald-400/35 dark:bg-emerald-950/45 dark:text-emerald-200',
    articleClassName: '',
    markClassName: '',
    icon: CheckCircle2,
  };
}

function AssistantMark({ small = false }: { small?: boolean }) {
  return (
    <div
      className={`flex shrink-0 items-center justify-center border border-stone-300 bg-stone-100 text-stone-600 dark:border-[#48615c] dark:bg-[#1d2e2d] dark:text-stone-300 ${small ? 'size-7' : 'size-9'}`}
      aria-hidden="true"
    >
      <Stethoscope size={small ? 14 : 18} strokeWidth={1.7} />
    </div>
  );
}

function SourcesCard({ sources, messageId }: { sources: ChatSource[]; messageId: string }) {
  return (
    <section
      className="mt-8 border border-stone-200 bg-[#fbfaf6] p-4 dark:border-[#3c514d] dark:bg-[#1a2927] sm:mt-10 sm:p-5"
      aria-labelledby={`sources-heading-${messageId}`}
      data-testid={`sources-card-${messageId}`}
    >
      <div className="flex items-start justify-between gap-4 border-b border-stone-200 pb-3 dark:border-[#304744]">
        <div className="flex items-start gap-3">
          <BookOpen size={16} className="mt-0.5 shrink-0 text-stone-500 dark:text-stone-400" />
          <div>
            <h3 id={`sources-heading-${messageId}`} className="text-xs font-semibold uppercase tracking-[0.15em] text-stone-700 dark:text-stone-200">Sources</h3>
            <p className="mt-1 text-xs text-stone-500 dark:text-stone-400">
              {sources.length > 0 ? `${sources.length} reference${sources.length === 1 ? '' : 's'} returned` : 'No source links were returned for this response.'}
            </p>
          </div>
        </div>
      </div>
      {sources.length > 0 && (
        <ul className="divide-y divide-stone-200 dark:divide-[#304744]" data-testid={`list-sources-${messageId}`}>
          {sources.map((source, index) => (
            <li key={`${source.url}-${index}`}>
              <a
                href={source.url}
                target="_blank"
                rel="noreferrer"
                className="group block py-3 first:pt-4 last:pb-0 text-teal-800 transition-colors hover:text-teal-950 dark:text-[#9bddd6] dark:hover:text-[#c4eee9]"
                data-testid={`link-source-${messageId}-${index}`}
              >
                <span className="flex items-start justify-between gap-3 text-sm font-medium leading-6">
                  <span>{source.title}</span>
                  <ArrowUpRight size={14} className="mt-1 shrink-0 opacity-60 transition-transform group-hover:-translate-y-0.5 group-hover:translate-x-0.5" />
                </span>
                {source.title !== source.url && (
                  <span className="mt-1 block break-all text-xs leading-5 text-stone-500 underline decoration-stone-300 underline-offset-2 group-hover:decoration-teal-700/60 dark:text-stone-400 dark:decoration-[#48615c] dark:group-hover:decoration-[#9bddd6]/60">
                    {source.url}
                  </span>
                )}
              </a>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

function MessageItem({ message }: { message: Message }) {
  if (message.role === 'user') {
    return (
      <article className="animate-fade-up grid grid-cols-[24px_minmax(0,1fr)] gap-2 border-b border-stone-200/80 py-8 dark:border-[#304744] sm:grid-cols-[28px_minmax(0,1fr)] sm:gap-3 sm:py-10" data-testid={`message-user-${message.id}`}>
        <div className="mt-1 flex size-7 items-center justify-center border border-stone-300 text-[10px] font-semibold uppercase tracking-[0.16em] text-stone-500 dark:border-[#48615c] dark:text-stone-400" aria-hidden="true">
          You
        </div>
        <div className="min-w-0">
          <p className="mb-2 text-[11px] font-semibold uppercase tracking-[0.16em] text-stone-500 dark:text-stone-400">Your question</p>
          <p className="whitespace-pre-wrap text-[15px] leading-7 text-stone-800 dark:text-stone-100">{message.content}</p>
        </div>
      </article>
    );
  }

  if (message.role === 'error') {
    return (
      <article className="animate-fade-up grid grid-cols-[24px_minmax(0,1fr)] gap-2 border-b border-stone-200/80 py-8 dark:border-[#304744] sm:grid-cols-[28px_minmax(0,1fr)] sm:gap-3 sm:py-10" data-testid={`message-error-${message.id}`}>
        <div className="mt-1 flex size-7 items-center justify-center border border-amber-300 bg-amber-50 text-amber-800 dark:border-amber-400/35 dark:bg-amber-950/45 dark:text-amber-200" aria-hidden="true">
          <Info size={15} />
        </div>
        <div>
          <p className="mb-2 text-[11px] font-semibold uppercase tracking-[0.16em] text-amber-800 dark:text-amber-200">Could not send</p>
          <p className="text-[15px] leading-7 text-stone-700 dark:text-stone-200">{message.content}</p>
          <p className="mt-3 text-xs text-stone-500 dark:text-stone-400">Check the connection and try sending your question again.</p>
        </div>
      </article>
    );
  }

  const status = routeStatus(message.route);
  const StatusIcon = status.icon;

  return (
    <article className={`animate-fade-up grid grid-cols-[24px_minmax(0,1fr)] gap-2 border-b border-stone-200/80 py-9 dark:border-[#304744] sm:grid-cols-[28px_minmax(0,1fr)] sm:gap-3 sm:py-11 ${status.articleClassName}`} data-testid={`message-assistant-${message.id}`}>
      <div className={`mt-1 flex size-7 items-center justify-center border ${status.markClassName || 'border-stone-300 bg-stone-100 text-stone-600 dark:border-[#48615c] dark:bg-[#1d2e2d] dark:text-stone-300'}`} aria-hidden="true">
        <StatusIcon size={14} strokeWidth={1.8} />
      </div>
      <div className="min-w-0">
        <div className="mb-3 flex flex-wrap items-center gap-2">
          <p className="text-[11px] font-semibold uppercase tracking-[0.16em] text-stone-500 dark:text-stone-400">Guidelines assistant</p>
          <span className={`inline-flex items-center gap-1.5 border px-2 py-1 text-[10px] font-medium uppercase tracking-[0.11em] ${status.className}`} data-testid={`status-route-${message.id}`}>
            <StatusIcon size={12} strokeWidth={1.8} />
            {status.label}
          </span>
        </div>
        <div className="message-copy whitespace-pre-wrap text-base leading-8 text-stone-800 dark:text-stone-100 sm:text-[17px]" data-testid={`text-answer-${message.id}`}>
          {message.content}
        </div>
        {message.route === 'general-info' && (
          <SourcesCard sources={message.sources} messageId={message.id} />
        )}
        <p className="mt-5 text-xs leading-5 text-stone-500 dark:text-stone-400">
          {message.route === 'emergency'
            ? 'If you may be in immediate danger, contact local emergency services now. This assistant cannot assess urgent symptoms.'
            : message.route === 'out-of-scope'
              ? 'This assistant is limited to general clinical information and cannot provide guidance for this request.'
              : 'General information only — not medical advice. Review sources and consult a qualified professional for personal decisions.'}
        </p>
      </div>
    </article>
  );
}

function LoadingMessage() {
  return (
    <div className="grid grid-cols-[24px_minmax(0,1fr)] gap-2 border-b border-stone-200/80 py-6 dark:border-[#304744] sm:grid-cols-[28px_minmax(0,1fr)] sm:gap-3 sm:py-7" data-testid="status-loading">
      <AssistantMark small />
      <div className="pt-1">
        <p className="text-[11px] font-semibold uppercase tracking-[0.16em] text-stone-500 dark:text-stone-400">Guidelines assistant</p>
        <div className="mt-3 flex items-center gap-2 text-sm text-stone-500 dark:text-stone-400">
          <span className="flex gap-1" aria-hidden="true">
            <span className="size-1.5 bg-stone-400 animate-pulse-line dark:bg-stone-500" />
            <span className="size-1.5 bg-stone-400 animate-pulse-line [animation-delay:180ms] dark:bg-stone-500" />
            <span className="size-1.5 bg-stone-400 animate-pulse-line [animation-delay:360ms] dark:bg-stone-500" />
          </span>
          Reviewing your question
        </div>
      </div>
    </div>
  );
}

function EmptyState({ onPrompt }: { onPrompt: (prompt: string) => void }) {
  return (
    <section className="animate-fade-up py-12 sm:py-20" data-testid="empty-state">
      <div className="mb-9 flex items-center gap-4">
        <AssistantMark />
        <div>
          <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-stone-500 dark:text-stone-400">Welcome</p>
          <p className="mt-1 text-sm text-stone-500 dark:text-stone-400">Ask about general clinical guidance and its sources.</p>
        </div>
      </div>
      <h1 className="max-w-2xl font-serif text-3xl leading-[1.12] tracking-[-0.025em] text-stone-800 dark:text-stone-100 sm:text-4xl lg:text-5xl">
        What can we help you understand?
      </h1>
      <p className="mt-5 max-w-xl text-[15px] leading-7 text-stone-600 dark:text-stone-300">
        Explore conditions, symptoms, prevention, and medication information. Answers include links to the references returned by the service. This is general information, not diagnosis or treatment.
      </p>
      <div className="mt-10 border-t border-stone-200 pt-5 dark:border-[#304744] sm:mt-12">
        <div className="mb-4 flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.15em] text-stone-500 dark:text-stone-400">
          <CircleHelp size={14} className="text-stone-500 dark:text-stone-400" />
          Example questions
        </div>
        <div className="flex flex-wrap gap-2">
          {starterPrompts.map((prompt, index) => (
            <button
              key={prompt}
              type="button"
              onClick={() => onPrompt(prompt)}
              className="border border-stone-200 bg-[#fbfaf6] px-3.5 py-3 text-left text-sm leading-5 text-stone-700 transition-colors hover:border-teal-700/45 hover:bg-teal-50/55 dark:border-[#48615c] dark:bg-[#1d2e2d] dark:text-stone-200 dark:hover:border-[#73c8c0]/70 dark:hover:bg-[#243a38]"
              data-testid={`button-example-question-${index}`}
            >
              {prompt}
            </button>
          ))}
        </div>
        <p className="mt-4 text-xs text-stone-400 dark:text-stone-500">Choose a question to review or edit it before sending.</p>
      </div>
    </section>
  );
}

function Home() {
  const [conversationId] = useState(() => readSession(CONVERSATION_KEY, makeId()));
  const [messages, setMessages] = useState<Message[]>(() => readSession<Message[]>(MESSAGES_KEY, []));
  const [isDarkMode, setIsDarkMode] = useState(readTheme);
  const [draft, setDraft] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [hasConnectionError, setHasConnectionError] = useState(false);
  const scrollAnchor = useRef<HTMLDivElement>(null);
  const composerRef = useRef<HTMLTextAreaElement>(null);

  const hasMessages = messages.length > 0;
  const canSend = draft.trim().length > 0 && !isLoading;
  const sessionLabel = useMemo(() => conversationId.slice(0, 8).toUpperCase(), [conversationId]);

  useEffect(() => {
    try {
      sessionStorage.setItem(CONVERSATION_KEY, JSON.stringify(conversationId));
    } catch {
      // Session persistence is best effort.
    }
  }, [conversationId]);

  useEffect(() => {
    try {
      sessionStorage.setItem(MESSAGES_KEY, JSON.stringify(messages));
    } catch {
      // Session persistence is best effort.
    }
  }, [messages]);

  useEffect(() => {
    document.documentElement.classList.toggle('dark', isDarkMode);
    try {
      localStorage.setItem(THEME_KEY, isDarkMode ? 'dark' : 'light');
    } catch {
      // Theme persistence is best effort.
    }
  }, [isDarkMode]);

  useEffect(() => {
    scrollAnchor.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
  }, [messages, isLoading]);

  const submitQuestion = async (event?: FormEvent) => {
    event?.preventDefault();
    const question = draft.trim();
    if (!question || isLoading) return;

    setDraft('');
    setHasConnectionError(false);
    setMessages((current) => [...current, { id: makeId(), role: 'user', content: question }]);
    setIsLoading(true);

    try {
      const result = await sendChatMessage({ message: question, conversation_id: conversationId });
      setMessages((current) => [
        ...current,
        { id: makeId(), role: 'assistant', content: result.answer, route: result.route, sources: result.sources },
      ]);
    } catch (error) {
      setHasConnectionError(true);
      const message = error instanceof Error ? error.message : 'The chat service is unavailable. Please try again.';
      setMessages((current) => [...current, { id: makeId(), role: 'error', content: message }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleComposerKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      void submitQuestion();
    }
  };

  const usePrompt = (prompt: string) => {
    setDraft(prompt);
    composerRef.current?.focus();
  };

  const resetConversation = () => {
    setMessages([]);
    setDraft('');
    setHasConnectionError(false);
  };

  return (
    <div className={`clinical-shell relative flex min-h-[100dvh] flex-col bg-[#f6f5f0] text-stone-800 dark:bg-[#142220] dark:text-stone-100 ${isDarkMode ? 'dark' : ''}`}>
      <header className="relative z-10 border-b border-stone-200/90 bg-[#f6f5f0] dark:border-[#304744] dark:bg-[#142220]">
        <div className="mx-auto flex w-full max-w-[1320px] items-center justify-between gap-3 px-4 py-3 sm:gap-4 sm:px-8 sm:py-4 lg:px-12">
          <div className="flex min-w-0 items-center gap-2.5 sm:gap-3">
            <div className="flex size-8 shrink-0 items-center justify-center border border-stone-300 bg-stone-100 text-stone-600 dark:border-[#48615c] dark:bg-[#1d2e2d] dark:text-stone-300" aria-hidden="true">
              <Stethoscope size={17} strokeWidth={1.7} />
            </div>
            <div className="min-w-0">
              <p className="truncate font-serif text-[15px] font-semibold leading-none tracking-[-0.01em] text-stone-800 dark:text-stone-100 sm:text-[17px]">Clinical Guidelines Assistant</p>
              <p className="mt-1 hidden text-[10px] font-medium uppercase tracking-[0.14em] text-stone-500 dark:text-stone-400 sm:block">Evidence-aware reference workspace</p>
            </div>
          </div>
          <div className="flex shrink-0 items-center gap-2 sm:gap-4">
            <div className="hidden items-center gap-2 text-[11px] text-stone-500 dark:text-stone-400 sm:flex" data-testid="text-session">
              <span className="size-1.5 bg-emerald-600 dark:bg-emerald-400" />
              Session {sessionLabel}
            </div>
            <button
              type="button"
              onClick={() => setIsDarkMode((current) => !current)}
              className="inline-flex size-9 items-center justify-center border border-stone-300 text-stone-600 transition-colors hover:border-teal-700/60 hover:text-teal-800 dark:border-[#48615c] dark:text-stone-300 dark:hover:border-[#73c8c0] dark:hover:text-[#9bddd6]"
              aria-label={isDarkMode ? 'Switch to light mode' : 'Switch to dark mode'}
              title={isDarkMode ? 'Switch to light mode' : 'Switch to dark mode'}
              data-testid="button-theme-toggle"
            >
              {isDarkMode ? <Sun size={16} strokeWidth={1.8} /> : <Moon size={16} strokeWidth={1.8} />}
            </button>
          </div>
        </div>
        <div className="border-t border-stone-200/70 bg-[#eeece5] dark:border-[#304744] dark:bg-[#1b2c2a]">
          <div className="mx-auto flex max-w-[1320px] items-center gap-2 px-4 py-2 text-[11px] leading-5 text-stone-600 dark:text-stone-300 sm:px-8 lg:px-12" data-testid="status-disclaimer">
            <ShieldCheck size={14} className="shrink-0 text-stone-500 dark:text-stone-400" strokeWidth={1.8} />
            <span>General information only — not medical advice.</span>
          </div>
        </div>
      </header>

      <main className="relative z-10 mx-auto flex w-full max-w-[1320px] flex-1 flex-col px-4 sm:px-8 lg:px-12">
        <div className="flex flex-1 gap-6 sm:gap-12">
          <aside className="hidden w-44 shrink-0 border-r border-stone-200/80 py-12 dark:border-[#304744] lg:block">
            <p className="text-[10px] font-semibold uppercase tracking-[0.18em] text-stone-500 dark:text-stone-400">Workspace</p>
            <div className="mt-5 space-y-4 text-xs leading-5 text-stone-500 dark:text-stone-400">
              <div className="flex items-start gap-2 text-stone-600 dark:text-stone-300"><BookOpen size={14} className="mt-0.5 shrink-0" />Source-led answers</div>
              <div className="flex items-start gap-2"><ShieldCheck size={14} className="mt-0.5 shrink-0" />General information</div>
              <div className="flex items-start gap-2"><RefreshCw size={14} className="mt-0.5 shrink-0" />Session preserved</div>
            </div>
            <div className="mt-auto pt-24 text-[11px] leading-5 text-stone-400 dark:text-stone-500">
              Responses are generated from the connected guideline service. Always assess information in context.
            </div>
          </aside>

          <section className="flex min-w-0 flex-1 flex-col">
            <div className="mx-auto w-full max-w-3xl flex-1">
              {hasMessages ? (
                <div className="pb-8" data-testid="message-list">
                  {messages.map((message) => <MessageItem key={message.id} message={message} />)}
                  {isLoading && <LoadingMessage />}
                  <div ref={scrollAnchor} />
                </div>
              ) : (
                <EmptyState onPrompt={usePrompt} />
              )}
            </div>
          </section>
        </div>
      </main>

      <footer className="sticky bottom-0 z-20 border-t border-stone-200 bg-[#f6f5f0] dark:border-[#304744] dark:bg-[#142220]">
        <div className="mx-auto w-full max-w-[1320px] px-3 py-3 sm:px-8 sm:py-4 lg:px-12">
          <form onSubmit={submitQuestion} className="mx-auto max-w-3xl">
            <div className={`flex items-end gap-2 border bg-[#fbfaf6] p-1.5 transition-colors dark:bg-[#1d2e2d] sm:gap-3 sm:p-2 ${hasConnectionError ? 'border-amber-300 dark:border-amber-400/60' : 'border-stone-300 focus-within:border-teal-700/70 dark:border-[#48615c] dark:focus-within:border-[#73c8c0]/75'}`}>
              <label htmlFor="question-composer" className="sr-only">Ask a clinical guidelines question</label>
              <textarea
                ref={composerRef}
                id="question-composer"
                value={draft}
                onChange={(event) => setDraft(event.target.value)}
                onKeyDown={handleComposerKeyDown}
                placeholder="Ask about a clinical guideline or source…"
                rows={1}
                className="max-h-36 min-h-11 flex-1 resize-none bg-transparent px-2.5 py-3 text-[15px] leading-5 text-stone-800 outline-none placeholder:text-stone-400 dark:text-stone-100 dark:placeholder:text-stone-500 sm:px-3"
                aria-describedby="composer-hint"
                data-testid="input-question"
              />
              <button
                type="submit"
                disabled={!canSend}
                className="flex size-11 shrink-0 items-center justify-center bg-teal-700 text-[#f6f5f0] transition-colors hover:bg-teal-800 dark:bg-[#278b84] dark:text-[#10201e] dark:hover:bg-[#3da59c] disabled:cursor-not-allowed disabled:bg-stone-300 disabled:text-stone-500 dark:disabled:bg-[#334845] dark:disabled:text-stone-500"
                aria-label={isLoading ? 'Sending question' : 'Send question'}
                data-testid="button-send"
              >
                {isLoading ? <LoaderCircle size={18} className="animate-spin" /> : <Send size={18} strokeWidth={1.8} />}
              </button>
            </div>
            <div className="flex flex-wrap items-center justify-between gap-x-3 gap-y-1 px-1 pt-2 text-[10px] text-stone-400 dark:text-stone-500">
              <span id="composer-hint">Enter to send · Shift + Enter for a new line</span>
              {hasMessages && (
                <button type="button" onClick={resetConversation} className="inline-flex items-center gap-1 text-stone-500 underline decoration-stone-300 underline-offset-2 hover:text-teal-800 dark:text-stone-400 dark:decoration-[#48615c] dark:hover:text-[#9bddd6]" data-testid="button-new-conversation">
                  <RefreshCw size={11} />
                  New conversation
                </button>
              )}
            </div>
          </form>
        </div>
      </footer>
    </div>
  );
}

function Router() {
  return (
    <ErrorBoundary resetKey={useLocation()[0]}>
      <Switch>
        <Route path="/" component={Home} />
        <Route component={NotFound} />
      </Switch>
    </ErrorBoundary>
  );
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <WouterRouter base={import.meta.env.BASE_URL.replace(/\/$/, '')}>
          <Router />
        </WouterRouter>
        <Toaster />
      </TooltipProvider>
    </QueryClientProvider>
  );
}

export default App;