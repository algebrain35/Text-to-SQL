import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// The API (schemagraph/api/main.py) allows http://localhost:5173 and http://127.0.0.1:5173 through CORS.
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, strictPort: true },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./src/test/setup.ts"],
    css: false,
  },
});
