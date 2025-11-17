import os
from crewai_tools import BaseTool
from openai import OpenAI
from pydantic import BaseModel, Field

class TextToSpeechTool(BaseTool):
    name: str = "OpenAI_TTS_API"
    description: str = "A tool that converts long text into an audio file (.mp3) using the OpenAI Text-to-Speech API."

    def _run(self, text_description: str, output_path: str = "temp_audio.mp3") -> str:
        """Converts text to speech and returns the file path."""
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

        # Use a high-quality voice for natural narration
        response = client.audio.speech.create(
            model="tts-1",
            voice="onyx",  # A clear, professional voice
            input=text_description
        )

        # Save the audio file
        response.stream_to_file(output_path)
        return output_path

