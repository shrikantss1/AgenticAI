

from langchain.messages import HumanMessage
from langgraph.types import Command
from snackstack.logger import get_logger
from snackstack.voice.recorder import VoiceRecorder
from snackstack.voice.speaker import VoiceSpeaker
from snackstack.graph import snackstack_graph
import uuid
import argparse


logger = get_logger("main")

class SnackStackAssistant:

    def __init__(self, voice: str = "nova", enable_voice: bool = False):
        self.enable_voice = enable_voice
        self.thread_id = uuid.uuid4().hex
        if enable_voice:
            self.recorder = VoiceRecorder()
            self.speaker  = VoiceSpeaker(voice=voice, speed=1.1)
        else:
            self.recorder = None
            self.speaker  = None

    def ask(self, user_query: str):

        config = {"configurable": {"thread_id": self.thread_id}}
        result = snackstack_graph.invoke({
            "messages": [HumanMessage(content=user_query)],
            'user_query': user_query
        }, config)

        for task in snackstack_graph.get_state(config).tasks:
            if task.interrupts:
                
                for interrupt_info in task.interrupts:
                    logger.info(f"HITL interrupt {interrupt_info.value}")
                    if self.enable_voice:
                        self.speaker.speak(interrupt_info.value)
                        _, user_response = self.recorder.record_and_transcribe(duration=5)
                    else:
                        user_response = input(interrupt_info.value)
                    result = snackstack_graph.invoke(Command(resume=user_response), config=config)
        
        answer = result.get('final_answer', "")
        if not answer:
            answer = "Sorry, I wasn't able to process that. Could you try rephrasing?"
        return answer
    
    def run_text_loop(self):
        print("\n🛒  SnackStack Assistant  (type 'quit' to exit)\n")
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not user_input or user_input.lower() in ("quit", "exit", "bye"):
                print("Goodbye!")
                break
            answer = self.ask(user_input)
            print(f"\nAssistant: {answer}\n")
        
    def voice_loop(self, max_turns: int = 10):
        """Microphone-based conversation loop."""
        if not self.recorder or not self.speaker:
                logger.error("Voice components not initialised")
                return

        welcome = "Hello! I'm your SnackStack assistant. How can I help you today?"
        self.speaker.speak(welcome)

        for turn in range(1, max_turns + 1):
            logger.info("--- voice turn %d / %d ---", turn, max_turns)
            _, transcript = self.recorder.record_and_transcribe(duration=5)
            if not transcript:
                self.speaker.speak("I didn't catch that. Could you repeat?")
                continue
            if transcript.lower().strip() in ("goodbye", "bye", "quit", "exit", "thank you"):
                self.speaker.speak("Goodbye! Have a great day.")
                break
            print(f"\nYou: {transcript}")
            answer = self.ask(transcript)
            print(f"Assistant: {answer}\n")
            self.speaker.speak(answer)



def main() -> None:
    parser = argparse.ArgumentParser(description="SnackStack Multi-Agent Voice System")
    parser.add_argument("--voice", action="store_true", help="Use microphone input + TTS output")
    parser.add_argument("--query", type=str, help="Run a single text query and exit")
    args = parser.parse_args()
    assistant = SnackStackAssistant(enable_voice=args.voice)
    if args.query:
        print(assistant.ask(args.query))
    elif args.voice:
        assistant.voice_loop()
    else:
        assistant.run_text_loop()



if __name__ == "__main__":
    main()

    
        



