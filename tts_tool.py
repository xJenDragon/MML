# tts_tool.py
from typing import Optional
from crewai.tools import BaseTool
from openai import OpenAI


class TextToSpeechTool(BaseTool):
    # REQUIRED TYPE ANNOTATIONS (due to Pydantic v2)
    name: str = "text_to_speech"
    description: str = "Converts text into an MP3 audio file using OpenAI TTS."

    def _run(self, text: str, output_path: Optional[str] = None) -> str:
        """Generate an MP3 file from text and return the absolute path."""

        client = OpenAI()   # uses OPENAI_API_KEY from env

        if output_path is None:
            output_path = "tts_output.mp3"

        # Call OpenAI TTS endpoint
        response = client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice="alloy",
            input=text
        )

        # Save MP3
        with open(output_path, "wb") as f:
            f.write(response.read())

        return output_path

