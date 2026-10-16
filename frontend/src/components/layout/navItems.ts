import {
  Award,
  Briefcase,
  Calendar,
  Gauge,
  LayoutDashboard,
  ListOrdered,
  MessageCircle,
  Radar,
  Shield,
  TrendingUp,
  UserRound,
  Users,
} from 'lucide-react'
import type { LucideIcon } from 'lucide-react'

export interface NavItem {
  to: string
  label: string
  /** Shorter label for the mobile bottom bar. */
  short: string
  icon: LucideIcon
}

/** Single source of nav entries, shared by the sidebar and the bottom bar. */
export const navItems: NavItem[] = [
  { to: '/', label: 'Overview', short: 'Home', icon: LayoutDashboard },
  { to: '/teams', label: 'Teams', short: 'Teams', icon: Users },
  { to: '/schedule', label: 'Schedule', short: 'Sched', icon: Calendar },
  { to: '/players', label: 'Players', short: 'Players', icon: UserRound },
  { to: '/ratings', label: 'Team Ratings', short: 'Ratings', icon: Gauge },
  { to: '/player-grades', label: 'Player Grades', short: 'Grades', icon: Award },
  { to: '/projections', label: 'Season Projections', short: 'Proj', icon: TrendingUp },
  { to: '/optimizer', label: 'Lineup Optimizer', short: 'Lineups', icon: ListOrdered },
  { to: '/trenches', label: 'Trench Report', short: 'Trench', icon: Shield },
  { to: '/coverage', label: 'Coverage Report', short: 'Coverage', icon: Radar },
  { to: '/coaches', label: 'Coaches', short: 'Coaches', icon: Briefcase },
  { to: '/chat', label: 'Chat', short: 'Chat', icon: MessageCircle },
]
