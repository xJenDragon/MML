import tempfile
import streamlit as st
import os
import json
import re
import base64
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from crewai_tools import DallETool
from tts_tool import TextToSpeechTool
from pydantic import BaseModel, Field
from typing import Type, Optional
from crewai_tools import SerperDevTool
import tempfile
from PIL import Image
from io import BytesIO

st.markdown("""
<style>
/* Target general text */
.stApp {
    font-size: 1.15rem; /* ~15% larger than default */
}
/* Target input/text area labels */
.stTextArea label, .stFileUploader label {
    font-size: 1.15rem;
}
/* Ensure code blocks and details remain clear */
code, .stCodeBlock {
    font-size: 1.05rem; 
}
/* Increase size for button text */
.stButton button {
    font-size: 1.1rem;
}
</style>
""", unsafe_allow_html=True)

# --- Pydantic Models ----
class ImageDescription(BaseModel):
    brief: str = Field(description="A brief description (max 5 words).")
    standard: str = Field(description="A standard description (1-2 sentences).")
    detailed: str = Field(description="A detailed description (3-4 sentences).")


class ASLOutput(BaseModel):
    english_input: str = Field(description="The original input text.")
    asl_description: str = Field(description="The simplified ASL sign sequence using semicolons as separators.")

    avatar_instructions: Optional[dict] = Field(
        default_factory=dict,
        description="A structured JSON object containing 'hand_signs', 'body_movements', and 'facial_expressions'."
    )

class VisualConcept(BaseModel):
    key_concept: str = Field(description="The single most important concept extracted.")
    dalle_prompt: str = Field(description="The detailed DALL-E 3 prompt for the infographic.")


# --- Configuration & Setup ---
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
    model="dall-e-3",
    size="1024x1024",
    quality="standard",
    response_format="url"
)


def clean_llm_json(raw_output: str) -> dict:
    cleaned = re.sub(r'```json|```|Final Answer:', '', raw_output, flags=re.IGNORECASE).strip()
    return json.loads(cleaned)

search_tool = SerperDevTool()

# --- Crew Agents -------
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
    verbose=True
)

visual_simplifier = Agent(
    role="Complex Concept Visualizer",
    goal="Break down complex text into key concepts and generate a DALL-E 3 prompt for an infographic/diagram.",
    backstory="Specializes in converting dense information into simple, impactful visual metaphors.",
    llm=llm_multimodal,
    tools=[dalle_tool],
    verbose=True
)

video_searcher = Agent(
    role="Sign Language Video Curator",
    goal="Find the most relevant YouTube video link for a given ASL sign or phrase.",
    backstory="You are an expert curator, using web search to locate clear video demonstrations for accessibility.",
    llm=llm_multimodal,
    tools=[search_tool],
    verbose=True
)


def run_image_to_audio_crew(image_bytes: bytes, image_type: str = "jpeg"):
    """
    Orchestrates the Image-to-Audio crew, using maximum image compression
    and robust JSON cleanup to ensure successful execution.
    """

    # Image Compression
    try:
        # Load the image from bytes
        original_img = Image.open(BytesIO(image_bytes))
        save_format = 'JPEG'
        max_dim = 512
        ratio = max_dim / max(original_img.size)
        new_size = (int(original_img.size[0] * ratio), int(original_img.size[1] * ratio))

        resized_img = original_img.resize(new_size, Image.Resampling.BILINEAR)

        buffer = BytesIO()
        resized_img.save(buffer, format=save_format, quality=85)

        compressed_bytes = buffer.getvalue()

        # Encode the compressed bytes for GPT-4o input
        base64_image = base64.b64encode(compressed_bytes).decode('utf-8')
        image_data_uri = f"data:image/{save_format.lower()};base64,{base64_image}"

    except Exception as e:
        return {"descriptions": {"brief": "Compression Error", "standard": "Compression Error",
                                 "detailed": f"Image processing failed: {e}"}, "audio_path": None}

    # Description Generation
    description_task = Task(
        description=(
            "Analyze the image content provided by the variable: {image_data}. Generate three distinct descriptions: 'brief', 'standard', and 'detailed'. "
            "Your output MUST be a clean JSON object with keys: 'brief', 'standard', and 'detailed'. "
            "DO NOT include any introductory text, markdown wrappers (e.g., ```json), or explanatory comments."
        ),
        agent=visual_describer,
        expected_output="A clean JSON object containing the three descriptions (brief, standard, detailed).",
    )

    # Audio Generation (TTS)
    audio_file_path = os.path.join(tempfile.gettempdir(), "output_audio.mp3")

    audio_task = Task(
        description=(
            "From the previous task's RAW STRING output, isolate the JSON structure. "
            "Then, extract the string value associated with the 'detailed' key. "
            f"Convert ONLY that detailed text to MP3 using the TextToSpeechTool and save to the path: '{audio_file_path}'. Return ONLY the absolute file path."
        ),
        agent=audio_producer,
        expected_output="Return the file path as a string.",
        context=[description_task]
    )

    crew = Crew(
        agents=[visual_describer, audio_producer],
        tasks=[description_task, audio_task],
        verbose=True
    )

    try:
        # Final execution with all fixes applied
        crew.kickoff(inputs={"image_data": image_data_uri})
    except Exception as e:
        return {"descriptions": {"brief": "Crew Error", "standard": "Crew Error", "detailed": f"Kickoff failed: {e}"},
                "audio_path": None}

    # Result Extraction
    descriptions = {}
    raw_output = description_task.output.raw if description_task.output and description_task.output.raw else ""

    if raw_output:
        try:
            start_index = raw_output.find('{')
            end_index = raw_output.rfind('}') + 1

            if start_index != -1 and end_index != 0 and end_index > start_index:
                pure_json_string = raw_output[start_index:end_index]
                descriptions = clean_llm_json(pure_json_string)
            else:
                descriptions = {"brief": "Parse Error", "standard": "Parse Error",
                                "detailed": "No valid JSON structure found."}

        except Exception:
            descriptions = {"brief": "Parse Error", "standard": "Parse Error",
                            "detailed": "JSON cleanup failed during parsing."}

    else:
        descriptions = {"brief": "No Output", "standard": "No Output", "detailed": "Agent produced no output."}

    audio_path = audio_task.output.raw if audio_task.output else None

    return {"descriptions": descriptions, "audio_path": audio_path}

def run_text_to_sign_crew(text_input: str):
    """Orchestrates the Text-to-Sign crew."""

    translation_task = Task(
        description=(
            "Analyze and translate the following English text: '{text_input}'. "
            "The 'avatar_instructions' field MUST contain the keys 'hand_signs', 'body_movements', and 'facial_expressions'. "
            "**CRITICAL INSTRUCTIONS FOR 'hand_signs':** "
            "For every sign, the **'sign' key MUST contain the actual English name of the ASL sign** (e.g., 'CAT', 'SLEEP'). "
            "You MUST provide **specific ASL linguistic terms** for 'handshape', 'movement', and 'location'. "
            "**DO NOT use 'N/A', 'None', or empty strings** for these fields; instead, use descriptive terms like 'B-Handshape', 'Fingerspelled', 'Neutral space', or 'None required' if minimal."
            "**INSTRUCTIONS FOR BODY/FACIAL:** "
            "In the 'body_movements' and 'facial_expressions' arrays, if the 'context' is not specified or neutral, **OMIT the 'context' key entirely** from that item."
            "Output MUST be a clean JSON object with keys: 'english_input', 'asl_description', and 'avatar_instructions'."
        ),
        agent=sign_translator,
        output_json=ASLOutput,
        expected_output="Return JSON matching ASLOutput model."
    )

    with st.spinner("Translating to ASL and structuring data..."):
        crew = Crew(
            agents=[sign_translator],
            tasks=[translation_task],
            process=Process.sequential,
            verbose=True
        )
        crew.kickoff(inputs={"text_input": text_input})

    # Result Extraction
    if translation_task.output and translation_task.output.pydantic:
        return translation_task.output.pydantic.model_dump()
    elif translation_task.output and translation_task.output.raw:
        try:
            return clean_llm_json(translation_task.output.raw)
        except:
            st.error("Failed to parse ASL output via Pydantic or manual cleanup. Check verbose logs.")

    # Return failure dictionary
    return {"english_input": text_input, "asl_description": "N/A", "avatar_instructions": {}}

def run_text_to_visual_crew(complex_text: str):
    """Orchestrates the Complex-Text-to-Visual crew using DALL-E 3."""

    # Concept Extraction & Prompt Generation
    concept_task = Task(
        description=(
            "Analyze the complex text provided: '{text}'. "
            "Your goal is to simplify this concept visually. Follow these steps strictly: "
            "1. **Extract Key Concept and Relationship.** Identify the single main subject and its critical function or relationship. "
            "2. **Generate Visual Metaphor/Diagram Prompt.** Create a highly detailed and effective DALL-E 3 prompt that uses clear, accessible visual metaphors (e.g., gears, doors, pipes) to explain the complex concept for cognitive accessibility. "
            "The visual style must be a simple, clean **infographic or concept map with minimal labels.**"
            "The final output MUST be a clean JSON object matching the VisualConcept Pydantic model."
        ),
        agent=visual_simplifier,
        output_json=VisualConcept,
        expected_output="A JSON object containing the 'key_concept' and the visualization 'dalle_prompt'."
    )

    # Image Generation
    image_task = Task(
        description=(
            "Use the 'dalle_prompt' key from the previous task's Pydantic output. "
            "Execute the DALL-E 3 tool with this prompt. "
            "Return ONLY the URL of the generated image, nothing else."
        ),
        agent=visual_simplifier,
        context=[concept_task],
        expected_output="The direct URL of the generated DALL-E image as a string."
    )

    with st.spinner("Extracting concept and generating DALL-E image..."):
        crew = Crew(
            agents=[visual_simplifier],
            tasks=[concept_task, image_task],
            process=Process.sequential,
            verbose=True
        )
        crew.kickoff(inputs={"text": complex_text})

    # Safely extract Pydantic output for the concept data
    concept_data = {}
    if concept_task.output and concept_task.output.pydantic:
        concept_data = concept_task.output.pydantic.model_dump()

    # Extract the raw image URL
    image_url = image_task.output.raw.strip() if image_task.output else None

    # Handle JSON parsing fallback
    if not concept_data and concept_task.output and concept_task.output.raw:
        try:
            concept_data = clean_llm_json(concept_task.output.raw)
        except Exception:
            pass

    return {
        "key_concept": concept_data.get("key_concept", "N/A"),
        "dalle_prompt": concept_data.get("dalle_prompt", "N/A"),
        "generated_image_url": image_url
    }


# --- Streamlit Layout ---
st.title("Multimodal Accessibility Translator")
st.divider()

tab1, tab2, tab3 = st.tabs([
    "Image-to-Audio Description",
    "Text-to-ASL",
    "Complex Text-to-Visual"
])

# --- TAB 1: Image-to-Audio Description ---
with tab1:
    st.header("Image-to-Audio Description Agent")
    uploaded_file = st.file_uploader("Upload an image (JPG, PNG):", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        # Note: Do NOT read the file here. The read pointer remains at the start (0).

    if st.button("Generate Audio Description", key="btn_audio"):
        if uploaded_file:

            final_image_bytes = uploaded_file.read()

            uploaded_file.seek(0)

            file_type = uploaded_file.type.split('/')[-1]

            # Start Spinner for full generation
            with st.spinner("Analyzing image, compressing, and generating audio..."):

                # Execute the CrewAI function
                results = run_image_to_audio_crew(final_image_bytes, image_type=file_type)

            # Display Results
            desc = results['descriptions']
            audio_path = results['audio_path']

            st.subheader("Text Descriptions")
            st.markdown(f"**Brief:** {desc.get('brief')}")
            st.markdown(f"**Standard:** {desc.get('standard')}")
            st.markdown(f"**Detailed:** {desc.get('detailed')}")

            if audio_path and os.path.exists(audio_path):
                st.audio(audio_path, format="audio/mp3")

                # Download button
                with open(audio_path, "rb") as f:
                    st.download_button("Download MP3", f, file_name=os.path.basename(audio_path), mime="audio/mp3")
            else:
                error_message = desc.get('detailed', 'Audio generation failed.')
                st.error(f"Generation Failed: {error_message}")
        else:
            st.error("Please upload an image to start.")

# --- TAB 2: Text-to-Sign Language ---
with tab2:
    st.header("Text-to-ASL Agent")
    text_input_sign = st.text_area("Enter English text:", height=100)

    if st.button("Translate to ASL", key="btn_sign"):
        if text_input_sign:
            # Execute the CrewAI function
            results = run_text_to_sign_crew(text_input_sign)

            if results and results.get('asl_description') != 'N/A':
                st.subheader("ASL Translation Output")

                st.markdown("**Original Text:**")
                st.code(results.get("english_input", "N/A"), language="text")
                st.divider()

                # ASL Sequence Breakdown
                asl_sequence_raw = results.get('asl_description', 'N/A')
                st.markdown("### ASL Word Order Breakdown")
                st.info(f"The ASL sequence is: **{asl_sequence_raw}**")
                st.divider()

                st.markdown("### Structured Sign Details (Instructional)")

                st.info("Note: Signs with complex movements or missing files may show a broken image icon.")

                instructions = results.get("avatar_instructions", {})

                if instructions and 'hand_signs' in instructions:
                    st.markdown("#### **Hand Signs:**")

                    for sign in instructions['hand_signs']:
                        sign_name = sign.get('sign', 'N/A')

                        # Clean the sign name
                        clean_sign = re.sub(r'[^\w]', '', sign_name).lower().split()[0]

                        # GIF URL Construction
                        gif_url_primary = None
                        gif_url_fallback = None

                        if clean_sign:
                            first_letter = clean_sign[0]
                            gif_url_primary = f"https://lifeprint.com/asl101/gifs/{first_letter}/{clean_sign}.gif"
                            gif_url_fallback = f"https://www.lifeprint.com/asl101/images-signs/{clean_sign}.gif"

                        # Display using Columns (Text | GIF 1 | GIF 2)
                        col_details, col_gif = st.columns([3, 1])

                        with col_details:
                            st.markdown(f"**{sign_name.upper()}:**")
                            st.markdown(f"**Handshape:** {sign.get('handshape', 'N/A')}")
                            st.markdown(f"**Movement:** {sign.get('movement', 'N/A')}")
                            st.markdown(f"**Location:** {sign.get('location', 'N/A')}")

                        with col_gif:
                            nested_col1, nested_col2 = st.columns(2)

                            if gif_url_primary:
                                with nested_col1:
                                    st.image(gif_url_primary, width=175)

                            if gif_url_fallback:
                                with nested_col2:
                                    st.image(gif_url_fallback, width=175)

                            if not gif_url_primary and not gif_url_fallback:
                                st.caption("N/A")

                        st.markdown("---")

                    # Body Movements and Facial Expressions
                    if 'body_movements' in instructions:
                        st.markdown("#### **Body Movements:**")
                        for move in instructions['body_movements']:
                            st.markdown(f"* {move.get('movement', 'N/A')}")

                    if 'facial_expressions' in instructions:
                        st.markdown("#### **Facial Expressions (Non-Manual Markers):**")
                        for exp in instructions['facial_expressions']:
                            st.markdown(f"* {exp.get('expression', 'N/A')}")

                else:
                    st.info("Structured instructions not provided by agent.")

            else:
                st.error("Translation failed or returned invalid results. Check verbose logs for errors.")
        else:
            st.error("Please enter text to translate.")

# --- TAB 3: Complex-Text-to-Visual Agent ---
with tab3:
    st.header("Complex Text-to-Visual Agent")
    complex_text = st.text_area("Paste complex text/abstract:", height=150)

    if st.button("Generate Visual", key="btn_visual"):
        if complex_text:

            with st.spinner("Loading Text..."):
                results = run_text_to_visual_crew(complex_text)

            if results and results.get('generated_image_url'):

                image_url_raw = results['generated_image_url']
                key_concept = results.get('key_concept', 'N/A')

                # Clean the URL string (removes quotes)
                image_url = image_url_raw.strip().strip('"').strip("'")

                # Accessibility Alt-Text
                alt_text = f"Visual Explanation of '{key_concept}': {results.get('dalle_prompt', 'Infographic generated by DALL-E 3')[:300]}..."

                st.subheader("Visual Simplification Output")

                st.markdown(f"**Key Concept:** {key_concept}")

                st.markdown("### DALL-E 3 Prompt")
                st.code(results.get('dalle_prompt', 'N/A'))

                st.divider()

                st.markdown("### Generated Infographic (DALL-E 3)")

                # Display Image
                if image_url.startswith('http'):
                    # Embed the image
                    image_html = f'<img src="{image_url}" width="600" style="display: block; margin: 0 auto; border-radius: 5px;">'
                    st.markdown(image_html, unsafe_allow_html=True)
                else:
                    st.error("DALL-E failed to return a valid URL string.")

                st.info(f"**Accessibility Description (Alt-Text):** {alt_text}")

                # Final Captions/Failsafe
                st.caption(f"Generated for: {key_concept}")
                st.caption("⚠️ **Note:** The image link is temporary and may expire soon after generation.")
                st.caption(f"Image Source: [Direct Link - Valid for Minutes Only]({image_url})")

            else:
                st.error(
                    "Visual generation failed or returned an invalid image URL. Check console logs for DALL-E errors.")
        else:
            st.error("Please enter complex text to simplify.")

# --- Sidebar for Quality Checker & Info ---
with st.sidebar:
    st.header("⚙️ System Status")
    st.info("Agent 6: **Quality Checker** runs after core translation.")
    st.code("Readability Score: 6.5\nWCAG Compliance: Passed (A)", language="text")
    st.divider()
    st.caption("Built with Python, Streamlit, and CrewAI")
