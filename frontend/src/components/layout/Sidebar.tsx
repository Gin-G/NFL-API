import { NavLink } from 'react-router-dom'
import { navItems } from './navItems'

export default function Sidebar() {
  return (
    <aside className="hidden md:flex flex-col w-56 shrink-0 bg-slate-800 border-r border-slate-700 min-h-screen">
      <div className="px-4 py-5 border-b border-slate-700">
        <span className="text-brand-green font-bold text-lg tracking-wide">NFL Analytics</span>
      </div>
      <nav className="flex-1 py-4 space-y-1 px-2">
        {navItems.map(({ to, label, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            className={({ isActive }) =>
              `flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                isActive
                  ? 'bg-brand-green text-white'
                  : 'text-slate-300 hover:bg-slate-700 hover:text-white'
              }`
            }
          >
            <Icon size={18} />
            {label}
          </NavLink>
        ))}
      </nav>
    </aside>
  )
}
