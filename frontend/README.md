# Sephira Frontend

Next.js 16 frontend application for Sephira sentiment data analysis platform.

## Design System

Built according to Sephira brand specifications:

- **Background**: `#0A0D1C` (primary), `#0F152F` (secondary)
- **Text**: `#FFFFFF` (primary), `rgba(255, 255, 255, 0.62)` (secondary)
- **Font**: Helvetica Neue / Helvetica
- **Cards**: Gradient from `#121834` to `#090D20` with 20px radius
- **Buttons**: 
  - Primary: White background, black text, 14px radius
  - Secondary: Transparent background, white text, rounded border

## Tech Stack

- **Next.js 16** - React framework
- **TypeScript** - Type safety
- **Tailwind CSS** - Styling
- **App Router** - Next.js app directory structure

## Setup

1. **Install dependencies**:
   ```bash
   cd frontend
   npm install
   ```

2. **Configure environment variables**:
   Create a `.env.local` file:
   ```
   NEXT_PUBLIC_API_URL=http://localhost:8000
   ```

3. **Run development server**:
   ```bash
   npm run dev
   ```

   The frontend will be available at `http://localhost:3000`

## Project Structure

```
frontend/
├── app/
│   ├── layout.tsx       # Root layout
│   ├── page.tsx         # Home page
│   └── globals.css      # Global styles
├── components/
│   ├── ChatInterface.tsx    # Main chat component
│   ├── MessageList.tsx      # Message display
│   ├── ChartDisplay.tsx     # Chart rendering
│   └── Button.tsx           # Button component
├── lib/
│   └── api.ts           # API client functions
└── tailwind.config.ts   # Tailwind configuration
```

## Features

- **Chat Interface**: Interactive chat with LLM backend
- **Chart Generation**: Automatic chart display when charts are requested
- **Session Management**: Maintains conversation context across messages
- **Error Handling**: Graceful error messages and loading states
- **Responsive Design**: Works on desktop and mobile devices

## API Integration

The frontend integrates with the backend API:

- `POST /api/chat` - Send chat messages
- `POST /api/generate-chart` - Generate charts
- `GET /api/analytics` - Get user analytics (future feature)

## Development

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run start` - Start production server
- `npm run lint` - Run ESLint

## Notes

- Ensure the backend API is running on the configured URL
- The frontend uses session IDs to maintain conversation context
- Charts are displayed automatically when requested in chat responses
