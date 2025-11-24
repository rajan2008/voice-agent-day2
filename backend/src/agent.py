import logging

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    tokenize
)

from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")


class Assistant(Agent):
    def __init__(self):
        super().__init__(
            instructions="""
You are a Health & Wellness Voice Companion.

Conversation Rules:
- Ask ONLY ONE question at a time.
- Start by asking how they are feeling.
- After mood, ask about water.
- After water, ask about sleep.
- After sleep, ask about steps.
- Wait for the user's answer before moving on.
- Keep responses short, friendly, and simple.
- Track values during this session.
- If no number is given, ask a follow-up question.
- Never list multiple questions together.
"""
        )

        # ✅ Session memory
        self.state = {
            "mood": None,
            "water": 0,
            "sleep": None,
            "steps": 0
        }

        # ✅ Tracks which question we are on
        self.current_step = "mood"

    def extract_number(self, text):
        digits = ''.join(filter(str.isdigit, text))
        return int(digits) if digits else None

    def next_question(self):
        if self.current_step == "mood":
            return "How are you feeling today?"

        if self.current_step == "water":
            return "How much water have you had today?"

        if self.current_step == "sleep":
            return "How many hours did you sleep last night?"

        if self.current_step == "steps":
            return "How many steps have you walked today?"

        return "Great job! Let me know if you want to update anything."

    async def on_message(self, message, ctx):
        user_text = message.text.lower()

        # ✅ Step 1: Mood
        if self.current_step == "mood":
            if "feeling" in user_text or "mood" in user_text or user_text.strip() != "":
                mood = user_text.replace("feeling", "").strip()
                self.state["mood"] = mood
                self.current_step = "water"
                return await ctx.send_response(
                    f"Got it. You're feeling {mood}. {self.next_question()}"
                )
            return await ctx.send_response("How are you feeling today?")

        # ✅ Step 2: Water
        if self.current_step == "water":
            amount = self.extract_number(user_text)
            if amount is None:
                return await ctx.send_response("How much water did you drink?")
            self.state["water"] += amount
            self.current_step = "sleep"
            return await ctx.send_response(
                f"Water updated: {self.state['water']} ml. {self.next_question()}"
            )

        # ✅ Step 3: Sleep
        if self.current_step == "sleep":
            hours = self.extract_number(user_text)
            if hours is None:
                return await ctx.send_response("How many hours did you sleep?")
            self.state["sleep"] = hours
            self.current_step = "steps"
            return await ctx.send_response(
                f"Noted. You slept {hours} hours. {self.next_question()}"
            )

        # ✅ Step 4: Steps
        if self.current_step == "steps":
            steps = self.extract_number(user_text)
            if steps is None:
                return await ctx.send_response("How many steps did you walk?")
            self.state["steps"] += steps
            self.current_step = "done"
            return await ctx.send_response(
                f"Steps updated: {self.state['steps']}. Great job today!"
            )

        # ✅ After all steps complete
        return await ctx.send_response(
            "You're all set! Tell me if you want to update anything."
        )


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-matthew",
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        logger.info(f"Usage: {usage_collector.get_summary()}")

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
