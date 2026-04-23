export default function Header({ darkMode, onToggleDark }) {
  return (
    <header className="flex items-start justify-between gap-4">
      <div>
        <h1 className="text-4xl font-bold tracking-tight text-gray-900 dark:text-white">
          TinyLM
        </h1>
        <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
          A transformer language model coded from scratch, trained on TinyStories
        </p>
      </div>

      <button
        onClick={onToggleDark}
        aria-label="Toggle dark mode"
        className="mt-1 flex items-center gap-2 rounded-full border border-gray-200 dark:border-gray-700
                   bg-white dark:bg-black px-3 py-1.5 text-sm font-medium
                   text-gray-700 dark:text-gray-300 shadow-sm
                   hover:bg-gray-50 dark:hover:bg-gray-900 transition-colors"
      >
        {darkMode ? (
          <>
            <SunIcon />
            Light
          </>
        ) : (
          <>
            <MoonIcon />
            Dark
          </>
        )}
      </button>
    </header>
  )
}

function SunIcon() {
  return (
    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <circle cx="12" cy="12" r="5" />
      <path strokeLinecap="round" d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42" />
    </svg>
  )
}

function MoonIcon() {
  return (
    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M21 12.79A9 9 0 1111.21 3a7 7 0 109.79 9.79z" />
    </svg>
  )
}
