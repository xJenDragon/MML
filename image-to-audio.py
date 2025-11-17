from crewai import Agent
from langchain_openai import ChatOpenAI
from crewai import Task, Crew, Process
from tts_tool import TextToSpeechTool

# Custom LLM instance supporting multimodal input
llm_multimodal = ChatOpenAI(model=OPENAI_MODEL)

# Visual Describer
visual_describer = Agent(
    role="WAI-Compliant Visual Describer",
    goal="Analyze images using W3C Web Accessibility Initiative (WAI) guidelines and generate detailed, multi-level textual descriptions.",
    backstory=(
        "You are an expert in accessibility, specifically trained on the COCO Captions and VizWiz datasets. "
        "Your goal is to provide accurate and meaningful descriptions for screen readers and audio narration, "
        "always outputting Brief, Standard, and Detailed levels."
    ),
    llm=llm_multimodal,
    multimodal=True,
    verbose=True
)

# Audio Producer
audio_producer = Agent(
    role="Natural Speech Producer",
    goal="Convert a detailed text description into a natural-sounding audio file using Text-to-Speech (TTS) technology.",
    backstory=(
        "You specialize in audio generation, ensuring the output is clear, well-paced, and has proper emphasis "
        "for an excellent listener experience. Your final output is a file path."
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm_multimodal
)


def define_image_tasks(image_path: str):
    # Image Analysis and Description Generation
    description_task = Task(
        description=(
            f"Analyze the image located at: {image_path}. "
            "Generate three distinct descriptions: **Brief** (3-5 words), **Standard** (1-2 sentences), "
            "and **Detailed** (3-4 sentences, covering context, color, emotion, and background). "
            "The final output MUST be a JSON object with keys: brief, standard, and detailed. "
            "The Detailed description is the main output for the next step."
        ),
        expected_output="A JSON object containing the three descriptions (brief, standard, detailed) generated from the image analysis.",
        agent=visual_describer,
        # The key to using GPT-4V is setting context with the image URL/path in the prompt or through the tool.
        # CrewAI's multimodal=True handles the image context injection automatically when an image is referenced in the task.
    )

    # Audio Production
    audio_task = Task(
        description=(
            "Take the 'detailed' description from the previous task's output. "
            "Convert this text into a natural-sounding MP3 audio file using the Text-to-Speech tool. "
            "Save the file with a unique name and return ONLY the full, absolute file path of the generated audio."
        ),
        expected_output="The absolute file path of the generated MP3 audio file (e.g., /path/to/output_123.mp3).",
        agent=audio_producer,
        tools=[TextToSpeechTool()],
        context=[description_task]
    )

    return description_task, audio_task


# --- Crew Orchestration ---

def create_image_to_audio_crew(image_path: str):
    description_task, audio_task = define_image_tasks(image_path)

    crew = Crew(
        agents=[visual_describer, audio_producer],
        tasks=[description_task, audio_task],
        process=Process.sequential,
        verbose=2
    )
    return crew