"use client";

import * as React from "react";

export interface AudioPlayerProps {
  src: string;
  onReplaceUrl?: () => Promise<string | null>;
  className?: string;
  "aria-label"?: string;
}

export function AudioPlayer({
  src,
  onReplaceUrl,
  className,
  ...ariaProps
}: AudioPlayerProps) {
  const [currentSrc, setCurrentSrc] = React.useState(src);
  const audioRef = React.useRef<HTMLAudioElement | null>(null);
  const recoveringRef = React.useRef(false);

  React.useEffect(() => {
    setCurrentSrc(src);
  }, [src]);

  const handleError = async () => {
    if (!onReplaceUrl || recoveringRef.current) return;
    recoveringRef.current = true;
    try {
      const fresh = await onReplaceUrl();
      if (fresh) setCurrentSrc(fresh);
    } finally {
      recoveringRef.current = false;
    }
  };

  return (
    <audio
      ref={audioRef}
      src={currentSrc}
      controls
      preload="auto"
      onError={() => {
        void handleError();
      }}
      className={className}
      aria-label={ariaProps["aria-label"] ?? "Audio clip"}
    />
  );
}
