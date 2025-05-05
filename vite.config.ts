import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { visualizer } from 'rollup-plugin-visualizer';

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [
    react(),
    visualizer({ // Optional: adds bundle visualization
      open: true,
      gzipSize: true,
    }),
  ],
  optimizeDeps: {
    exclude: ['lucide-react'],
  },
  build: {
    chunkSizeWarningLimit: 800, // Increase from default 500kb to 800kb
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom', 'react-router-dom'],
          ui: ['tailwindcss'],
          charts: ['recharts', 'd3'],
          // Add other logical groupings based on your dependencies
        }
      }
    }
  }
});