import { t } from '../theme/tokens'
import { useEffect, useRef, useState } from 'react'
import { useAIStore } from '../store/useAIStore'
import type { AIMessage } from '../types/ai'

// ── Quick prompts ─────────────────────────────────────────────────────────────

const QUICK_PROMPTS = [
  { label: 'MNIST classifier', prompt: 'Build a 3-layer neural network for MNIST handwritten digit classification.' },
  { label: 'CIFAR-10 network', prompt: 'Build a dense network for CIFAR-10 image classification.' },
  { label: 'Tabular classifier', prompt: 'I have a tabular CSV dataset with 4 numeric features and 3 classes. Build an appropriate model.' },
  { label: 'Explain errors', prompt: 'My graph has validation errors. Can you explain what is wrong and how to fix it?' },
  { label: 'Improve accuracy', prompt: 'My model has low validation accuracy. What architecture changes would help?' },
]

// ── Main panel ────────────────────────────────────────────────────────────────

interface AIPanelProps {
  onClose: () => void
  mobile?: boolean
}

export default function AIPanel({ onClose, mobile }: AIPanelProps) {
  const messages = useAIStore((s) => s.messages)
  const isLoading = useAIStore((s) => s.isLoading)
  const apiKey = useAIStore((s) => s.apiKey)
  const model = useAIStore((s) => s.model)
  const setApiKey = useAIStore((s) => s.setApiKey)
  const setModel = useAIStore((s) => s.setModel)
  const sendMessage = useAIStore((s) => s.sendMessage)
  const clearMessages = useAIStore((s) => s.clearMessages)

  const [input, setInput] = useState('')
  const [showSettings, setShowSettings] = useState(!apiKey)
  const [keyDraft, setKeyDraft] = useState(apiKey)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)

  // Auto-scroll on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isLoading])

  function handleSend() {
    const text = input.trim()
    if (!text || isLoading) return
    setInput('')
    void sendMessage(text)
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  function handleQuickPrompt(prompt: string) {
    void sendMessage(prompt)
  }

  function saveKey() {
    setApiKey(keyDraft.trim())
    if (keyDraft.trim()) setShowSettings(false)
  }

  return (
    <div style={{
      position: 'absolute',
      top: 0, right: 0,
      width: mobile ? '100%' : 360,
      height: '100%',
      background: t.topbarBg,
      borderLeft: mobile ? 'none' : `1px solid ${t.borderSubtle}`,
      display: 'flex',
      flexDirection: 'column',
      zIndex: 20,
    }}>

      {/* Header */}
      <div style={{
        height: 44,
        display: 'flex',
        alignItems: 'center',
        padding: '0 14px',
        borderBottom: `1px solid ${t.borderSubtle}`,
        gap: 8,
        flexShrink: 0,
      }}>
        <SparkleIcon />
        <span style={{ fontSize: 13, fontWeight: 600, color: t.textPrimary }}>AI Assistant</span>
        <div style={{ flex: 1 }} />
        {messages.length > 0 && (
          <HeaderBtn onClick={clearMessages} title="Clear conversation">
            <TrashIcon />
          </HeaderBtn>
        )}
        <HeaderBtn onClick={() => setShowSettings((s) => !s)} title="Settings" active={showSettings}>
          <KeyIcon />
        </HeaderBtn>
        <HeaderBtn onClick={onClose} title="Close">×</HeaderBtn>
      </div>

      {/* Settings */}
      {showSettings && (
        <div style={{
          padding: '12px 14px',
          borderBottom: `1px solid ${t.borderSubtle}`,
          background: '#111116',
          flexShrink: 0,
        }}>
          <div style={{ fontSize: 10, fontWeight: 600, letterSpacing: '0.07em', textTransform: 'uppercase', color: t.textFaint, marginBottom: 6 }}>
            OpenAI API Key
          </div>
          <input
            type="password"
            value={keyDraft}
            onChange={(e) => setKeyDraft(e.target.value)}
            placeholder="sk-…"
            onKeyDown={(e) => { if (e.key === 'Enter') saveKey() }}
            style={{
              width: '100%', background: t.bgElevated, border: `1px solid ${t.borderDefault}`,
              borderRadius: 5, color: t.textPrimary, fontSize: 12,
              padding: '5px 8px', outline: 'none', boxSizing: 'border-box', marginBottom: 6,
            }}
          />
          <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
            <select
              value={model}
              onChange={(e) => setModel(e.target.value)}
              style={{
                flex: 1, background: t.bgElevated, border: `1px solid ${t.borderDefault}`,
                borderRadius: 5, color: t.textSecondary, fontSize: 11, padding: '4px 7px', outline: 'none',
              }}
            >
              <option value="gpt-4o" style={{ background: t.bgElevated }}>gpt-4o</option>
              <option value="gpt-4o-mini" style={{ background: t.bgElevated }}>gpt-4o-mini</option>
              <option value="gpt-4-turbo" style={{ background: t.bgElevated }}>gpt-4-turbo</option>
              <option value="gpt-3.5-turbo" style={{ background: t.bgElevated }}>gpt-3.5-turbo</option>
            </select>
            <button
              onClick={saveKey}
              style={{
                padding: '4px 12px', borderRadius: 5,
                border: `1px solid ${t.accent}`, background: 'rgba(124,58,237,0.12)',
                color: t.accentMuted, fontSize: 11, fontWeight: 600, cursor: 'pointer',
              }}
            >
              Save
            </button>
          </div>
          <p style={{ fontSize: 10, color: t.textDisabled, margin: '6px 0 0', lineHeight: 1.5 }}>
            Stored locally in your browser only. Never sent anywhere except OpenAI.
          </p>
        </div>
      )}

      {/* Messages */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '10px 0' }}>
        {messages.length === 0 ? (
          <EmptyState onQuickPrompt={handleQuickPrompt} />
        ) : (
          <>
            {messages.map((msg) => (
              <MessageBubble key={msg.id} message={msg} />
            ))}
            {isLoading && <ThinkingBubble />}
          </>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Quick prompts (shown when no messages) */}
      {messages.length === 0 && !showSettings && (
        <div style={{
          padding: '8px 12px',
          borderTop: `1px solid ${t.borderSubtle}`,
          display: 'flex',
          flexWrap: 'wrap',
          gap: 5,
          flexShrink: 0,
        }}>
          {QUICK_PROMPTS.map((q) => (
            <button
              key={q.label}
              onClick={() => handleQuickPrompt(q.prompt)}
              style={{
                padding: '3px 9px', borderRadius: 12,
                border: `1px solid ${t.borderDefault}`, background: 'transparent',
                color: t.textMuted, fontSize: 10, cursor: 'pointer',
                transition: 'border-color 0.1s, color 0.1s',
              }}
              onMouseEnter={(e) => { e.currentTarget.style.color = t.accentMuted; e.currentTarget.style.borderColor = t.accent }}
              onMouseLeave={(e) => { e.currentTarget.style.color = t.textMuted; e.currentTarget.style.borderColor = t.borderDefault }}
            >
              {q.label}
            </button>
          ))}
        </div>
      )}

      {/* Input */}
      <div style={{
        padding: '10px 12px',
        borderTop: `1px solid ${t.borderSubtle}`,
        display: 'flex',
        gap: 8,
        alignItems: 'flex-end',
        flexShrink: 0,
      }}>
        <textarea
          ref={inputRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask anything about your model…"
          rows={1}
          style={{
            flex: 1,
            background: t.bgElevated, border: `1px solid ${t.borderDefault}`,
            borderRadius: 8, color: t.textPrimary,
            fontSize: 12, padding: '7px 10px', outline: 'none',
            resize: 'none', lineHeight: 1.5,
            maxHeight: 100, overflowY: 'auto',
            fontFamily: 'inherit',
          }}
          onFocus={(e) => { e.currentTarget.style.borderColor = t.accent }}
          onBlur={(e) => { e.currentTarget.style.borderColor = t.borderDefault }}
        />
        <button
          onClick={handleSend}
          disabled={isLoading || !input.trim()}
          style={{
            width: 34, height: 34, borderRadius: 8,
            border: `1px solid ${t.accent}`,
            background: isLoading || !input.trim() ? '#1a1a2e' : 'rgba(124,58,237,0.15)',
            color: isLoading || !input.trim() ? t.textDisabled : t.accentMuted,
            cursor: isLoading || !input.trim() ? 'not-allowed' : 'pointer',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            flexShrink: 0,
            transition: 'background 0.1s',
          }}
          onMouseEnter={(e) => { if (!isLoading && input.trim()) e.currentTarget.style.background = 'rgba(124,58,237,0.28)' }}
          onMouseLeave={(e) => { if (!isLoading && input.trim()) e.currentTarget.style.background = 'rgba(124,58,237,0.15)' }}
        >
          <SendIcon />
        </button>
      </div>
    </div>
  )
}

// ── Message bubble ────────────────────────────────────────────────────────────

function MessageBubble({ message }: { message: AIMessage }) {
  const applyActions = useAIStore((s) => s.applyActions)
  const isUser = message.role === 'user'

  return (
    <div style={{
      padding: '4px 12px',
      display: 'flex',
      flexDirection: 'column',
      alignItems: isUser ? 'flex-end' : 'flex-start',
      gap: 4,
    }}>
      <div style={{
        maxWidth: '92%',
        padding: '8px 11px',
        borderRadius: isUser ? '10px 10px 3px 10px' : '10px 10px 10px 3px',
        background: isUser ? 'rgba(124,58,237,0.15)' : t.bgElevated,
        border: `1px solid ${isUser ? '#7c3aed30' : t.borderDefault}`,
        fontSize: 12,
        lineHeight: 1.6,
        color: t.textBody,
        wordBreak: 'break-word',
      }}>
        <MessageText content={message.content} />
      </div>

      {message.actions && (
        <div style={{
          maxWidth: '92%',
          background: '#111116',
          border: `1px solid ${t.borderDefault}`,
          borderRadius: 8,
          overflow: 'hidden',
        }}>
          <div style={{
            padding: '5px 10px',
            display: 'flex',
            alignItems: 'center',
            gap: 6,
            borderBottom: message.actionsApplied ? 'none' : `1px solid ${t.borderSubtle}`,
          }}>
            <span style={{ fontSize: 10, color: t.textFaint }}>
              {message.actions.length} graph {message.actions.length === 1 ? 'action' : 'actions'}
            </span>
            {message.actionsApplied ? (
              <span style={{ fontSize: 10, color: '#34d399', marginLeft: 'auto' }}>✓ Applied</span>
            ) : (
              <button
                onClick={() => applyActions(message.id)}
                style={{
                  marginLeft: 'auto',
                  padding: '2px 10px', borderRadius: 5,
                  border: `1px solid ${t.accent}`,
                  background: 'rgba(124,58,237,0.12)',
                  color: t.accentMuted, fontSize: 10, fontWeight: 600, cursor: 'pointer',
                }}
              >
                Apply to graph
              </button>
            )}
          </div>
          {!message.actionsApplied && (
            <div style={{ padding: '5px 10px', maxHeight: 80, overflowY: 'auto' }}>
              {message.actions.map((a, i) => (
                <div key={i} style={{ fontSize: 10, color: t.textFaint, fontFamily: 'monospace' }}>
                  {a.type === 'clear' && '⊘ clear graph'}
                  {a.type === 'addNode' && `+ ${a.nodeType} "${a.id}"`}
                  {a.type === 'connect' && `→ ${a.source} → ${a.target}`}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function ThinkingBubble() {
  return (
    <div style={{ padding: '4px 12px' }}>
      <div style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: 5,
        padding: '7px 11px',
        borderRadius: '10px 10px 10px 3px',
        background: t.bgElevated,
        border: `1px solid ${t.borderDefault}`,
      }}>
        {[0, 1, 2].map((i) => (
          <span
            key={i}
            style={{
              width: 5, height: 5, borderRadius: '50%',
              background: t.textFaint,
              animation: `pulse 1.2s ease-in-out ${i * 0.2}s infinite`,
            }}
          />
        ))}
      </div>
    </div>
  )
}

// ── Simple message text renderer ──────────────────────────────────────────────

function MessageText({ content }: { content: string }) {
  // Split into code blocks and prose
  const parts = content.split(/(```[\s\S]*?```)/g)
  return (
    <>
      {parts.map((part, i) => {
        if (part.startsWith('```') && part.endsWith('```')) {
          const lines = part.slice(3, -3).split('\n')
          const lang = lines[0].trim()
          const code = lines.slice(1).join('\n')
          return (
            <pre key={i} style={{
              background: t.topbarBg, border: `1px solid ${t.borderDefault}`,
              borderRadius: 5, padding: '6px 9px', margin: '4px 0',
              fontSize: 11, overflowX: 'auto', fontFamily: 'monospace',
              color: '#a5b4fc', whiteSpace: 'pre-wrap',
            }}>
              {lang && <div style={{ color: t.textFaint, marginBottom: 3, fontSize: 10 }}>{lang}</div>}
              {code}
            </pre>
          )
        }
        return (
          <span key={i} style={{ whiteSpace: 'pre-wrap' }}>
            {renderInline(part)}
          </span>
        )
      })}
    </>
  )
}

function renderInline(text: string): React.ReactNode[] {
  const parts = text.split(/(`[^`]+`)/g)
  return parts.map((p, i) => {
    if (p.startsWith('`') && p.endsWith('`')) {
      return (
        <code key={i} style={{
          background: t.topbarBg, borderRadius: 3, padding: '1px 4px',
          fontSize: 11, fontFamily: 'monospace', color: '#a5b4fc',
        }}>
          {p.slice(1, -1)}
        </code>
      )
    }
    return p
  })
}

// ── Empty state ───────────────────────────────────────────────────────────────

function EmptyState({ onQuickPrompt }: { onQuickPrompt: (p: string) => void }) {
  return (
    <div style={{
      flex: 1, padding: '24px 16px',
      display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 12,
    }}>
      <div style={{ width: 40, height: 40, borderRadius: '50%', background: 'rgba(124,58,237,0.12)', border: '1px solid #7c3aed30', display: 'flex', alignItems: 'center', justifyContent: 'center', color: t.accent }}>
        <SparkleIcon size={20} />
      </div>
      <div style={{ textAlign: 'center' }}>
        <p style={{ fontSize: 13, color: t.textSecondary, margin: '0 0 5px', fontWeight: 500 }}>
          AI Architecture Assistant
        </p>
        <p style={{ fontSize: 11, color: t.textFaint, margin: 0, lineHeight: 1.6 }}>
          Describe a model, ask for architecture advice,<br />or get help debugging your graph.
        </p>
      </div>
      <div style={{ width: '100%', borderTop: `1px solid ${t.borderSubtle}`, paddingTop: 12 }}>
        <p style={{ fontSize: 10, fontWeight: 600, letterSpacing: '0.07em', textTransform: 'uppercase', color: t.textDisabled, marginBottom: 8 }}>
          Quick start
        </p>
        {QUICK_PROMPTS.map((q) => (
          <button
            key={q.label}
            onClick={() => onQuickPrompt(q.prompt)}
            style={{
              width: '100%', textAlign: 'left',
              padding: '6px 10px', borderRadius: 6, marginBottom: 4,
              border: `1px solid ${t.borderSubtle}`, background: 'transparent',
              color: t.textMuted, fontSize: 11, cursor: 'pointer',
              transition: 'background 0.1s, color 0.1s',
            }}
            onMouseEnter={(e) => { e.currentTarget.style.background = t.bgElevated; e.currentTarget.style.color = t.textPrimary }}
            onMouseLeave={(e) => { e.currentTarget.style.background = 'transparent'; e.currentTarget.style.color = t.textMuted }}
          >
            {q.label}
          </button>
        ))}
      </div>
    </div>
  )
}

// ── Icons ─────────────────────────────────────────────────────────────────────

function SparkleIcon({ size = 15 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor">
      <path d="M12 2l2.4 7.4H22l-6.2 4.5 2.4 7.4L12 17 5.8 21.3l2.4-7.4L2 9.4h7.6z" />
    </svg>
  )
}

function KeyIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="7.5" cy="15.5" r="5.5" />
      <path d="M21 2l-9.6 9.6" />
      <path d="M15.5 7.5l3 3L22 7l-3-3" />
    </svg>
  )
}

function SendIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
      <line x1="22" y1="2" x2="11" y2="13" />
      <polygon points="22 2 15 22 11 13 2 9 22 2" />
    </svg>
  )
}

function TrashIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="3 6 5 6 21 6" />
      <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" />
    </svg>
  )
}

function HeaderBtn({ onClick, title, active, children }: { onClick: () => void; title: string; active?: boolean; children: React.ReactNode }) {
  return (
    <button
      onClick={onClick}
      title={title}
      style={{
        width: 28, height: 28,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        borderRadius: 6, border: 'none',
        background: active ? t.borderDefault : 'transparent',
        color: active ? t.textPrimary : t.textFaint,
        cursor: 'pointer', fontSize: 16, lineHeight: 1,
        transition: 'background 0.1s, color 0.1s',
      }}
      onMouseEnter={(e) => { e.currentTarget.style.background = t.borderDefault; e.currentTarget.style.color = t.textPrimary }}
      onMouseLeave={(e) => { e.currentTarget.style.background = active ? t.borderDefault : 'transparent'; e.currentTarget.style.color = active ? t.textPrimary : t.textFaint }}
    >
      {children}
    </button>
  )
}
