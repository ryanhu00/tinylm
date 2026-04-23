import { useEffect, useRef } from 'react'

export default function StoryOutput({ text, isGenerating }) {
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
  }, [text])

  const isEmpty = !text && !isGenerating

  return (
    <section>
      <label className="block text-xs font-semibold uppercase tracking-widest text-gray-400 dark:text-gray-500 mb-2">
        Generated Story
      </label>
      <div
        className="output-scroll relative min-h-48 max-h-96 overflow-y-auto rounded-xl border
                   border-gray-200 dark:border-gray-700
                   bg-white dark:bg-black
                   px-5 py-4 shadow-sm
                   transition-colors duration-300"
      >
        {isEmpty ? (
          <p className="text-gray-300 dark:text-gray-600 select-none italic text-sm">
            Your story will appear here…
          </p>
        ) : (
          <p
            className={`whitespace-pre-wrap leading-relaxed text-gray-800 dark:text-gray-100 text-sm
                        ${isGenerating ? 'cursor-blink' : ''}`}
          >
            {text}
          </p>
        )}
        <div ref={bottomRef} />
      </div>
    </section>
  )
}
