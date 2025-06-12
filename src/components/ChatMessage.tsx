import React from 'react'
import ReactMarkdown from 'react-markdown'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { atomDark } from 'react-syntax-highlighter/dist/esm/styles/prism'
import clsx from 'clsx'

type MessageProps = {
  message: {
    role: 'user' | 'assistant'
    content: string
  }
}

// Extend with HTML attributes to satisfy ReactMarkdown typings
type CodeProps = React.HTMLAttributes<HTMLElement> & {
  inline?: boolean
  className?: string
  children?: React.ReactNode
}

const CodeBlock: React.FC<CodeProps> = ({ inline, className, children, ...props }) => {
  const match = /language-(\w+)/.exec(className || '')
  if (!inline && match) {
    return (
      <SyntaxHighlighter
        language={match[1]}
        PreTag="div"
        style={atomDark as any} // cast to any to fix TS errors
        {...props}
      >
        {String(children).replace(/\n$/, '')}
      </SyntaxHighlighter>
    )
  }
  return (
    <code className={className} {...props}>
      {children}
    </code>
  )
}

const ChatMessage = ({ message }: MessageProps) => {
  const isUser = message.role === 'user'

  return (
    <div className={clsx('flex w-full', isUser ? 'justify-end' : 'justify-start')}>
      <div
        className={clsx(
          'max-w-[80%] rounded-lg p-4',
          isUser ? 'bg-blue-500 text-white' : 'bg-white shadow-md'
        )}
      >
        <ReactMarkdown components={{ code: CodeBlock }}>
          {message.content}
        </ReactMarkdown>
      </div>
    </div>
  )
}

export default ChatMessage
