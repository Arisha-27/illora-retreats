import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  base: 'https://storage.googleapis.com/ilora-frontend-ornate-veld-477511-n8/', // 👈 use your bucket URL
})