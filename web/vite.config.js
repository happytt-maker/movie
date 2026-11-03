import { defineConfig, loadEnv } from 'vite'
import vue from '@vitejs/plugin-vue'

// 前端只跟两种数据源打交道：
//   static —— 读 public/data/*.json（默认，接口没好也能演示）
//   api    —— 走 /api 反向代理到甲的 FastAPI（避免跨域，后端不用改 CORS）
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const target = env.VITE_PROXY_TARGET || 'http://127.0.0.1:8001'

  return {
    plugins: [vue()],
    server: {
      port: 5173,
      host: '127.0.0.1',
      proxy: {
        '/api': {
          target,
          changeOrigin: true,
          rewrite: (p) => p.replace(/^\/api/, ''),
        },
      },
    },
    build: {
      chunkSizeWarningLimit: 1500,
    },
  }
})
