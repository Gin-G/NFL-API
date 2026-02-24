import { NavLink } from 'react-router-dom'
import { LayoutDashboard, Users, Calendar, UserRound, Briefcase, MessageCircle } from 'lucide-react'

const navItems = [
  { to: '/', label: 'Overview', icon: LayoutDashboard },
  { to: '/teams', label: 'Teams', icon: Users },
  { to: '/schedule', label: 'Schedule', icon: Calendar },
  { to: '/players', label: 'Players', icon: UserRound },
  { to: '/coaches', label: 'Coaches', icon: Briefcase },
  { to: '/chat', label: 'Chat', icon: MessageCircle },
]

export default function BottomNav() {
  return (
    <nav className="md:hidden fixed bottom-0 left-0 right-0 bg-slate-800 border-t border-slate-700 flex z-50">
      {navItems.map(({ to, label, icon: Icon }) => (
        <NavLink
          key={to}
          to={to}
          end={to === '/'}
          className={({ isActive }) =>
            `flex-1 flex flex-col items-center py-2 text-xs font-medium transition-colors ${
              isActive ? 'text-brand-green' : 'text-slate-400'
            }`
          }
        >
          <Icon size={20} />
          <span className="mt-0.5">{label}</span>
        </NavLink>
      ))}
    </nav>
  )
}
