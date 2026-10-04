import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ReturnIQ · Return prevention workspace",
  description: "Understand return risk, approve preventive actions, and track delivery performance with ReturnIQ.",
  other: {
    "codex-preview": "development",
  },
  icons: {
    icon: "/returniq-logo.png",
    shortcut: "/returniq-logo.png",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  );
}
