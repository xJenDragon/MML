# tts_tool.py
import os
from crewai_tools import BaseTool
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Type


class TextToSpeechInput(BaseModel):
    text_description: str = Field(description="The detailed text to be converted to speech.")
    output_path: str = Field(description="The local path where the MP3 file should be saved.")


class TextToSpeechTool(BaseTool):
    name: str = "OpenAI_TTS_API"
    description: "A tool that converts text into an audio file (.mp3) using the OpenAI TTS API."
    args_schema: Type[TextToSpeechInput] = TextToSpeechInput

    def _run(self, text_description: str, output_path: str = "output.mp3") -> str:
        """Converts text to speech and returns the file path."""
        # The client automatically picks up the API key from the environment
        client = OpenAI()

        response = client.audio.speech.create(
            model="tts-1",
            voice="onyx",
            input=text_description
        )

        # Save the audio file
        response.stream_to_file(output_path)
        return output_path
