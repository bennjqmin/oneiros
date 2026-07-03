import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faMoon } from '@fortawesome/free-solid-svg-icons'
import { t } from '../theme/tokens'

interface AppLogoProps {
  size?: number
}

export function AppLogo({ size = 24 }: AppLogoProps) {
  return (
    <FontAwesomeIcon
      icon={faMoon}
      aria-hidden
      style={{
        fontSize: size,
        color: t.accentMuted,
        display: 'block',
        flexShrink: 0,
      }}
    />
  )
}
