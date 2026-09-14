/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./app/**/*.{js,ts,jsx,tsx}", "./components/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: {
          950: "#0a0e14",
          900: "#0f1419",
          800: "#151b24",
          700: "#1c2430",
          600: "#243041",
        },
        accent: {
          DEFAULT: "#3d8bfd",
          muted: "#1e3a5f",
        },
        long: "#2ecc71",
        short: "#e74c3c",
        neutral: "#95a5a6",
      },
      fontFamily: {
        sans: ["IBM Plex Sans", "Segoe UI", "sans-serif"],
        mono: ["IBM Plex Mono", "Consolas", "monospace"],
      },
    },
  },
  plugins: [],
};
