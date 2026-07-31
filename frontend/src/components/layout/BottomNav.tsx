import { NavLink } from 'react-router-dom'
import { navItems } from './navItems'

export default function BottomNav() {
  return (
    <nav className="md:hidden fixed bottom-0 left-0 right-0 bg-slate-800 border-t border-slate-700 flex overflow-x-auto z-50">
      {navItems.map(({ to, short, icon: Icon }) => (
        <NavLink
          key={to}
          to={to}
          end={to === '/'}
          className={({ isActive }) =>
            `flex-1 min-w-[3.5rem] flex flex-col items-center py-2 text-[10px] font-medium transition-colors ${
              isActive ? 'text-brand-green' : 'text-slate-400'
            }`
          }
        >
          <Icon size={18} />
          <span className="mt-0.5">{short}</span>
        </NavLink>
      ))}
    </nav>
  )
}
