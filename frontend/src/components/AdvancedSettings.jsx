function SliderRow({ label, value, min, max, step, onChange, disabled, format }) {
  return (
    <div>
      <div className="flex justify-between items-center mb-1.5">
        <span className="text-xs font-medium text-gray-600 dark:text-gray-400">{label}</span>
        <span className="text-xs font-semibold text-violet-700 dark:text-violet-300 tabular-nums w-10 text-right">
          {format ? format(value) : value}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={e => onChange(Number(e.target.value))}
        disabled={disabled}
        className="w-full h-1.5 rounded-full appearance-none cursor-pointer
                   bg-gray-200 dark:bg-gray-700 accent-violet-700 dark:accent-violet-300
                   disabled:opacity-50 disabled:cursor-not-allowed"
      />
      <div className="flex justify-between mt-1">
        <span className="text-[10px] text-gray-400 dark:text-gray-600">{min}</span>
        <span className="text-[10px] text-gray-400 dark:text-gray-600">{max}</span>
      </div>
    </div>
  )
}

export default function AdvancedSettings({
  open,
  onToggle,
  temperature,
  onTemperatureChange,
  disabled,
}) {
  return (
    <div className="rounded-xl border border-gray-200 dark:border-gray-700 overflow-hidden transition-colors duration-300">
      <button
        type="button"
        onClick={onToggle}
        className="w-full flex items-center justify-between
                   px-4 py-3
                   bg-gray-50 dark:bg-gray-800/60
                   text-sm font-medium text-gray-700 dark:text-gray-300
                   hover:bg-gray-100 dark:hover:bg-gray-800
                   transition-colors"
      >
        <span className="flex items-center gap-2">
          <SettingsIcon />
          Advanced Settings
        </span>
        <ChevronIcon open={open} />
      </button>

      {open && (
        <div className="px-4 py-4 bg-white dark:bg-black flex flex-col gap-5 border-t border-gray-200 dark:border-gray-700">
          <SliderRow
            label="Temperature"
            value={temperature}
            min={0.1}
            max={2.0}
            step={0.05}
            onChange={onTemperatureChange}
            disabled={disabled}
            format={v => v.toFixed(2)}
          />
        </div>
      )}
    </div>
  )
}

function SettingsIcon() {
  return (
    <svg className="w-4 h-4 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round"
        d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
      <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
    </svg>
  )
}

function ChevronIcon({ open }) {
  return (
    <svg
      className={`w-4 h-4 text-gray-400 transition-transform duration-200 ${open ? 'rotate-180' : ''}`}
      fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}
    >
      <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
    </svg>
  )
}
