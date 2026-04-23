import { useState } from 'react'
import AdvancedSettings from './AdvancedSettings'

const DEFAULT_MAX_TOKENS = 250

export default function GenerateForm({
  onGenerate,
  onStop,
  isGenerating,
  error,
  defaultBackendUrl,
}) {
  const [prompt, setPrompt] = useState('Once upon a time')
  const [temperature, setTemperature] = useState(0.8)
  const [showAdvanced, setShowAdvanced] = useState(false)

  function handleSubmit(e) {
    e?.preventDefault()
    onGenerate({
      prompt: prompt.trim() || 'Once upon a time',
      maxTokens: DEFAULT_MAX_TOKENS,
      temperature,
      backendUrl: defaultBackendUrl.trim().replace(/\/$/, ''),
    })
  }

  return (
    <form onSubmit={handleSubmit} className="flex flex-col gap-5">
      <div>
        <label className="block text-xs font-semibold uppercase tracking-widest text-gray-400 dark:text-gray-500 mb-2">
          Prompt
        </label>
        <textarea
          value={prompt}
          onChange={e => setPrompt(e.target.value)}
          rows={3}
          placeholder="Once upon a time…"
          disabled={isGenerating}
          className="w-full rounded-xl border border-gray-200 dark:border-gray-700
                     bg-white dark:bg-black
                     px-4 py-3 text-sm text-gray-800 dark:text-gray-100
                     placeholder-gray-300 dark:placeholder-gray-600
                     shadow-sm outline-none resize-none
                     focus:border-violet-500 dark:focus:border-violet-300 focus:ring-2 focus:ring-violet-100 dark:focus:ring-violet-950
                     disabled:opacity-50 disabled:cursor-not-allowed
                     transition-colors duration-300"
        />
      </div>

      <AdvancedSettings
        open={showAdvanced}
        onToggle={() => setShowAdvanced(o => !o)}
        temperature={temperature}
        onTemperatureChange={setTemperature}
        disabled={isGenerating}
      />

      {error && (
        <div className="rounded-lg bg-red-50 dark:bg-red-950 border border-red-200 dark:border-red-800 px-4 py-3">
          <p className="text-xs text-red-600 dark:text-red-400 font-medium">{error}</p>
        </div>
      )}

      <button
        type="button"
        onClick={isGenerating ? onStop : handleSubmit}
        disabled={!isGenerating && !prompt.trim()}
        className={`w-full rounded-xl text-white font-semibold py-3 text-sm
                    shadow-sm transition-colors duration-150
                    ${isGenerating
                      ? 'bg-red-500 hover:bg-red-600 active:bg-red-700'
                      : 'bg-violet-700 hover:bg-violet-800 active:bg-violet-900 dark:bg-violet-400 dark:hover:bg-violet-300 dark:active:bg-violet-200 disabled:opacity-40 disabled:cursor-not-allowed'
                    }`}
      >
        {isGenerating ? 'Stop Generation' : 'Generate Story'}
      </button>
    </form>
  )
}
