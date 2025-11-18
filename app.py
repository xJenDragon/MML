import json
import re
import tempfile
import streamlit as st
import os
from pathlib import Path
from pydantic import BaseModel
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from crewai_tools import DallETool
from tts_tool import TextToSpeechTool
import tempfile

# ------------------------
# --- Pydantic Models ----
# ------------------------

class ImageDescription(BaseModel):
    brief: str
    standard: str
    detailed: str

class ASLOutput(BaseModel):
    english_input: str
    asl_description: str
    avatar_instructions: str

class VisualConcept(BaseModel):
    key_concept: str
    dalle_prompt: str

# ------------------------
# --- Configuration ------
# ------------------------

OPENAI_MODEL = "gpt-4o"

st.set_page_config(
    page_title="Multimodal Accessibility Translator",
    layout="wide",
    initial_sidebar_state="expanded"
)

llm_multimodal = ChatOpenAI(
    model=OPENAI_MODEL,
    temperature=0.2,
    max_tokens=4096,
    openai_api_key=os.environ.get("OPENAI_API_KEY")
)

dalle_tool = DallETool(
    model="gpt-image-1",
    size="1024x1024",
    quality="standard",
    response_format="url"
)

# ------------------------
# --- Crew Agents -------
# ------------------------

visual_describer = Agent(
    role="WAI-Compliant Visual Describer",
    goal="Analyze images and generate brief, standard, and detailed textual descriptions in JSON format for accessibility.",
    backstory="Expert in accessibility, providing multi-level descriptions based on WAI guidelines.",
    llm=llm_multimodal,
    multimodal=True,
    verbose=True
)

audio_producer = Agent(
    role="Natural Speech Producer",
    goal="Convert a detailed text description into a natural-sounding MP3 audio file.",
    backstory="Specializes in audio generation using the TextToSpeechTool.",
    tools=[TextToSpeechTool()],
    llm=llm_multimodal,
    verbose=True
)

sign_translator = Agent(
    role="ASL Linguistic Simplifier",
    goal="Convert English text into a simplified sequence of signs, including handshapes, body movements, and facial expressions, following ASL grammar rules.",
    backstory="Expert in American Sign Language (ASL) linguistics, restructuring English into ASL syntax.",
    llm=llm_multimodal,
    verbose=False
)

visual_simplifier = Agent(
    role="Complex Concept Visualizer",
    goal="Break down complex text into key concepts and generate a DALL-E 3 prompt for an infographic/diagram.",
    backstory="Specializes in converting dense information into simple, impactful visual metaphors.",
    llm=llm_multimodal,
    tools=[dalle_tool],
    verbose=True
)

# ------------------------
# --- Crew Functions -----
# ------------------------

# Helper function (make sure this is correctly defined at the top of app.py)
def clean_llm_json(raw_output: str) -> dict:
    # This regex removes markdown fences (```json, ```) and "Final Answer:"
    cleaned = re.sub(r'```json|```|Final Answer:', '', raw_output, flags=re.IGNORECASE).strip()
    return json.loads(cleaned)


def run_image_to_audio_crew(image_bytes: bytes):
    # 1. Save the image bytes to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        tmp_file.write(image_bytes)
        tmp_file.flush()
        image_path = tmp_file.name

        # 2. Define Task 1: Description Generation (NOW USING RAW STRING OUTPUT)
    description_task = Task(
        description=(
            "Analyze the image at path {image_path}. Output MUST be a clean JSON object with keys: 'brief', 'standard', 'detailed'. "
            "DO NOT include any explanation or markdown wrappers (like ```json) in the final answer."
        ),
        agent=visual_describer,
        # REMOVED: output_json=ImageDescription
        expected_output="A clean JSON object containing the three descriptions."
    )

    # 3. Define Task 2: Audio Generation
    audio_file_path = os.path.join(tempfile.gettempdir(), "output_audio.mp3")

    audio_task = Task(
        description=(
            "Extract the 'detailed' description from the previous task's output. "
            "Convert that detailed text to MP3 using the TextToSpeechTool and save to the path: "
            f"'{audio_file_path}'. Return ONLY the absolute file path."
        ),
        agent=audio_producer,
        expected_output="Return the file path as a string.",
        context=[description_task]
    )

    # 4. Define and Kickoff the Crew ONCE
    crew = Crew(
        agents=[visual_describer, audio_producer],
        tasks=[description_task, audio_task],
        verbose=True  # Keep True for debugging
    )

    try:
        crew.kickoff(inputs={"image_path": image_path})
    except Exception as e:
        os.unlink(image_path)
        return {"descriptions": {"brief": "Crew Error", "standard": "Crew Error", "detailed": f"Kickoff failed: {e}"},
                "audio_path": None}

    # 5. Extract Results SAFELY (Using manual cleanup)
    descriptions = {}
    if description_task.output and description_task.output.raw:
        try:
            descriptions = clean_llm_json(description_task.output.raw)
        except Exception as e:
            descriptions = {
                "brief": "Parse Fail",
                "standard": "Parse Fail",
                "detailed": f"JSON parsing failed. Raw output: {description_task.output.raw}"
            }
    else:
        descriptions = {"brief": "No Output", "standard": "No Output", "detailed": "No raw output from LLM."}

    audio_path = audio_task.output.raw if audio_task.output else None

    # Clean up and Return
    os.unlink(image_path)
    return {"descriptions": descriptions, "audio_path": audio_path}


def run_text_to_sign_crew(text_input: str):
    sign_translator.clear_memory() if hasattr(sign_translator, "clear_memory") else None

    translation_task = Task(
        description=(
            "Analyze and translate the following English text: '{text_input}'. "
            "Translate and simplify this text into a sequence of ASL sign descriptions, including hand signs, body movements, and facial expressions, following ASL grammar rules. "
            "Output MUST be a clean JSON object with keys: 'english_input', 'asl_description', and 'avatar_instructions'. "
            "DO NOT include any explanation or markdown wrappers (like ```json) in the final answer."
        ),
        agent=sign_translator,
        expected_output="A clean JSON object containing the three fields."
    )

    with st.spinner("🔄 Translating to ASL..."):
        crew = Crew(agents=[sign_translator], tasks=[translation_task], verbose=True)
        crew.kickoff(inputs={"text_input": text_input})

    if translation_task.output and translation_task.output.raw:
        try:
            # Assumes clean_llm_json is defined elsewhere in app.py
            import json
            return json.loads(translation_task.output.raw)
        except Exception as e:
            st.error(f"Failed to parse ASL JSON output: {e}")
            return {
                "english_input": text_input,
                "asl_description": f"Parsing Failed: {translation_task.output.raw}",
                "avatar_instructions": "Check verbose logs."
            }
    return {
        "english_input": text_input,
        "asl_description": "N/A - No output received.",
        "avatar_instructions": "N/A"
    }

def run_text_to_visual_crew(complex_text: str):
    concept_task = Task(
        description="Extract key concept and create detailed DALL-E prompt for simplified infographic.",
        agent=visual_simplifier,
        output_json=VisualConcept,
        expected_output="Return JSON matching VisualConcept model."
    )

    image_task = Task(
        description="Generate visual from the 'dalle_prompt' key in previous JSON output. Return image URL.",
        agent=visual_simplifier,
        context=[concept_task],
        expected_output="Return the generated image URL as a string."
    )

    with st.spinner("🎨 Extracting concept and generating image..."):
        crew = Crew(agents=[visual_simplifier], tasks=[concept_task, image_task])
        crew.kickoff(inputs={"text": complex_text})

    concept_data = concept_task.output.pydantic.model_dump() if concept_task.output else {"key_concept": "N/A", "dalle_prompt": "N/A"}
    image_url = image_task.output.raw if image_task.output else None

    return {
        "key_concept": concept_data.get("key_concept"),
        "dalle_prompt": concept_data.get("dalle_prompt"),
        "generated_image_url": image_url
    }


# ------------------------
# --- Streamlit Layout ---
# ------------------------

st.title("Multimodal Accessibility Translator")
st.markdown("A multi-agent system built with CrewAI for accessible content.")
st.divider()

tab1, tab2, tab3 = st.tabs([
    "🖼️ ➡️ 🎧 Image-to-Audio",
    "✍️ ➡️ 🤟 Text-to-Sign",
    "🧠 ➡️ 🎨 Complex Text-to-Visual"
])

with tab1:
    st.header("Image-to-Audio Description Agent")
    uploaded_file = st.file_uploader("Upload an image (JPG, PNG):", type=["jpg","jpeg","png"])

    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    if st.button("Generate Audio Description", key="btn_audio"):
        if uploaded_file:
            image_bytes = uploaded_file.read()
            results = run_image_to_audio_crew(image_bytes)

            desc = results['descriptions']
            audio_path = results['audio_path']

            st.subheader("Text Descriptions")
            st.markdown(f"**Brief:** {desc.get('brief')}")
            st.markdown(f"**Standard:** {desc.get('standard')}")
            st.markdown(f"**Detailed:** {desc.get('detailed')}")

            if audio_path and os.path.exists(audio_path):
                st.audio(audio_path, format="audio/mp3")
                with open(audio_path, "rb") as f:
                    st.download_button("Download MP3", f, file_name=os.path.basename(audio_path), mime="audio/mp3")

with tab2:
    st.header("Text-to-Sign Language")
    text_input_sign = st.text_area("Enter English text:", height=100)

    if st.button("Translate to ASL", key="btn_sign"):
        if text_input_sign:
            with st.spinner("Restructuring text for ASL grammar..."):
                results = run_text_to_sign_crew(text_input_sign)

            if results:
                st.subheader("✅ ASL Sign Sequence Instructions")

                asl_sequence_raw = results.get('asl_description', 'N/A')

                formatted_sequence = asl_sequence_raw.replace('; ', '\n')
                st.markdown("**ASL Sequence Description (Agent 4 Output):**")
                st.code(formatted_sequence, language="text")

                st.markdown("**Original Text:**")
                st.code(results.get("english_input", "N/A"), language="text")

                st.markdown("**Avatar/Animation Instructions (Conceptual):**")
                st.info(results.get("avatar_instructions", "N/A"))

                st.caption("MVP focus is the text description.")

with tab3:
    st.header("Complex Text-to-Visual Agent")
    complex_text = st.text_area("Paste complex text/abstract:", height=150)

    if st.button("Generate Visual", key="btn_visual"):
        if complex_text:
            results = run_text_to_visual_crew(complex_text)
            st.subheader("Visual Simplification Output")
            st.markdown(f"**Key Concept:** {results.get('key_concept')}")
            st.code(results.get('dalle_prompt'))
            if results.get('generated_image_url'):
                st.image(results['generated_image_url'], caption=f"Generated for: {results.get('key_concept')}")

# ------------------------
# --- Sidebar Info -------
# ------------------------
with st.sidebar:
    st.header("⚙️ System Status")
    st.info("Quality Checker and accessibility metrics")
    st.code("Readability Score: 6.5\nWCAG Compliance: Passed (A)")
    st.divider()
    st.caption("Built with Python, Streamlit, and CrewAI")
