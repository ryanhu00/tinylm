import { useState, useRef, useCallback } from 'react'
import Header from './components/Header'
import StoryOutput from './components/StoryOutput'
import GenerateForm from './components/GenerateForm'

const DEFAULT_BACKEND_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export default function App() {
  const [darkMode, setDarkMode] = useState(false)
  const [output, setOutput] = useState('')
  const [isGenerating, setIsGenerating] = useState(false)
  const [error, setError] = useState(null)

  const abortRef = useRef(null)
  const stopRef = useRef(false)

  const handleGenerate = useCallback(async ({ prompt, temperature, maxTokens, backendUrl }) => {
    stopRef.current = false
    setOutput(prompt)
    setError(null)
    setIsGenerating(true)

    const controller = new AbortController()
    abortRef.current = controller

    try {
      const response = await fetch(`${backendUrl}/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt,
          max_new_tokens: maxTokens,
          temperature,
        }),
        signal: controller.signal,
      })

      if (!response.ok) {
        const text = await response.text()
        let detail = text
        try { detail = JSON.parse(text).detail ?? text } catch {}
        throw new Error(`Server error ${response.status}: ${detail}`)
      }

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      let finished = false

      while (!finished && !stopRef.current) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop()

        for (const line of lines) {
          if (stopRef.current) { finished = true; break }
          if (!line.startsWith('data: ')) continue

          const data = line.slice(6).trim()
          if (data === '[DONE]') { finished = true; break }

          try {
            const { token } = JSON.parse(data)
            const cleanedToken = token.replace(/<\|endoftext\|>/g, '')
            if (cleanedToken) {
              setOutput(prev => prev + cleanedToken)
            }
          } catch {
          }
        }
      }

      reader.cancel().catch(() => {})
    } catch (err) {
      if (err.name !== 'AbortError') {
        setError(err.message)
      }
    } finally {
      setIsGenerating(false)
    }
  }, [])

  const handleStop = useCallback(() => {
    stopRef.current = true
    abortRef.current?.abort()
  }, [])

  return (
    <div className={darkMode ? 'dark' : ''}>
      <div className="min-h-screen bg-gray-50 dark:bg-black transition-colors duration-300">
        <div className="max-w-2xl mx-auto px-4 py-10">
          <Header darkMode={darkMode} onToggleDark={() => setDarkMode(d => !d)} />
          <main className="mt-8 flex flex-col gap-6">
            <StoryOutput text={output} isGenerating={isGenerating} />
            <GenerateForm
              onGenerate={handleGenerate}
              onStop={handleStop}
              isGenerating={isGenerating}
              error={error}
              defaultBackendUrl={DEFAULT_BACKEND_URL}
            />
          </main>
          <footer className="mt-10 text-center text-sm text-gray-500 dark:text-gray-400">
            <strong>© 2026 Ryan R. Hu.</strong>
          </footer>
        </div>
      </div>
    </div>
  )
}
