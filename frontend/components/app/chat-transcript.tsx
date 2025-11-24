'use client';

import { AnimatePresence, type HTMLMotionProps, motion } from 'motion/react';
import { type ReceivedChatMessage } from '@livekit/components-react';
import { ChatEntry } from '@/components/livekit/chat-entry';
import { useEffect, useRef } from 'react';

const MotionContainer = motion.create('div');
const MotionChatEntry = motion.create(ChatEntry);

interface ChatTranscriptProps {
  hidden?: boolean;
  messages?: ReceivedChatMessage[];
}

export function ChatTranscript({
  hidden = false,
  messages = [],
  ...props
}: ChatTranscriptProps & Omit<HTMLMotionProps<'div'>, 'ref'>) {
  const containerRef = useRef<HTMLDivElement | null>(null);

  // ✅ Auto-scroll whenever messages update
  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollTo({
        top: containerRef.current.scrollHeight,
        behavior: 'smooth', // change to 'auto' if you prefer instant jump
      });
    }
  }, [messages]);

  return (
    <AnimatePresence>
      {!hidden && (
        <MotionContainer
          ref={containerRef}
          style={{
            overflowY: 'auto',
            maxHeight: '100%',
            padding: '10px',
          }}
          initial={false}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          {...props}
        >
          {messages.map(
            ({
              id,
              timestamp,
              from,
              message,
              editTimestamp,
            }: ReceivedChatMessage) => {
              const locale = navigator?.language ?? 'en-US';
              const messageOrigin = from?.isLocal ? 'local' : 'remote';
              const hasBeenEdited = !!editTimestamp;

              return (
                <MotionChatEntry
                  key={id}
                  locale={locale}
                  timestamp={timestamp}
                  message={message}
                  messageOrigin={messageOrigin}
                  hasBeenEdited={hasBeenEdited}
                  initial={{ opacity: 0, translateY: 10 }}
                  animate={{ opacity: 1, translateY: 0 }}
                  transition={{ duration: 0.2 }}
                />
              );
            }
          )}
        </MotionContainer>
      )}
    </AnimatePresence>
  );
}
