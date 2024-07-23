import { defineConfig } from 'astro/config';
import sitemap from '@astrojs/sitemap';
import UnoCSS from 'unocss/astro';
import react from "@astrojs/react";
import { nodePolyfills } from 'vite-plugin-node-polyfills';

// https://astro.build/config
export default defineConfig({
  site: 'https://Dennismain13kini.github.io',
  base: 'my_portfolio_site',
  // used to generate images
  // site: process.env.VERCEL_ENV === 'production' ? 'https://brutal.elian.codes/' : process.env.VERCEL_URL ? `https://${process.env.VERCEL_URL}/` : 'https://localhost:3000/',
  trailingSlash: 'ignore',
  integrations: [sitemap(), UnoCSS({
    injectReset: true
  }), react()],
  vite: {
    resolve: {
      alias: {
          process: "process/browser",
          buffer: "buffer",
          crypto: "crypto-browserify",
          stream: "stream-browserify",
          assert: "assert",
          http: "stream-http",
          https: "https-browserify",
          os: "os-browserify",
          path: "path-browserify",
          url: "url",
          module: "module",
          util: "util",
      }},
    optimizeDeps: {
      exclude: ['@resvg/resvg-js']
    },
    plugins: [
      nodePolyfills(), 
    ]
  }
});