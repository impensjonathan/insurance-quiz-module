# insurance_quiz_app.py (Dedicated AI Quiz Module - NameError Corrected)

import streamlit as st
import os
import json
import re
import time
import random
import traceback
import io 
import google.generativeai as genai
import numpy as np
import faiss
import base64 # For displaying PDF if using bytes method

# --- Page Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(layout="centered", page_title="AI Quiz Tutor")



# --- Configuration ---
PREPROCESSED_DATA_DIR = "preprocessed_insurance_data"
ORIGINAL_DOC_FILENAME = "Intro into Insurance 2025 v04.docx" # !!! USER: Ensure this matches the copied filename in preprocessed_data
# --- Add Keyrus Logo ---
LOGO_PATH = "assets/logo-keyrus.svg" 
# --- End of Logo ---
THEME_OBJECTIVE_FILE = os.path.join(PREPROCESSED_DATA_DIR, "insurance_theme_objective.json")
DOC_CHUNK_DETAILS_FILE = os.path.join(PREPROCESSED_DATA_DIR, "insurance_doc_chunk_details.json")
FAISS_INDEX_FILE = os.path.join(PREPROCESSED_DATA_DIR, "insurance_faiss.index")
FAISS_CHUNKS_FILE = os.path.join(PREPROCESSED_DATA_DIR, "insurance_faiss_chunks.json")
PRE_GENERATED_QUESTIONS_FILE = os.path.join(PREPROCESSED_DATA_DIR, "insurance_pre_generated_questions.json")
ORIGINAL_DOCUMENT_PATH = os.path.join(PREPROCESSED_DATA_DIR, ORIGINAL_DOC_FILENAME)

DEFAULT_RUNTIME_FLASH_MODEL_ID = "gemini-1.5-flash"
NUM_CONTEXT_CHUNKS_FOR_DYNAMIC_QUESTIONS = 3
EMBEDDING_MODEL_ID = "models/text-embedding-004" 

# Add this new function definition to your insurance_quiz_app.py script

# This is the MODIFIED display_app_header() function
def display_app_header():
    # This function no longer displays the logo
    # It assumes the Welcome Page handles its own logo if needed globally at the top.

    st.title("Introduction to Insurance") 
 #  st.write("---") # Horizontal separator

# --- Helper Functions ---

@st.cache_resource 
def load_faiss_index(index_path):
    try:
        if os.path.exists(index_path):
            print(f"--- Loading FAISS index from: {index_path} ---")
            return faiss.read_index(index_path)
        else:
            st.error(f"FAISS index file not found at {index_path}. Please run the pre-processing script.")
            return None
    except Exception as e:
        st.error(f"Error loading FAISS index: {e}")
        traceback.print_exc()
        return None

@st.cache_data 
def load_json_data(file_path):
    try:
        if os.path.exists(file_path):
            print(f"--- Loading JSON data from: {file_path} ---")
            with open(file_path, "r") as f:
                return json.load(f)
        else:
            st.error(f"Data file not found: {file_path}. Please run the pre-processing script.")
            return None
    except Exception as e:
        st.error(f"Error loading JSON data from {file_path}: {e}")
        traceback.print_exc()
        return None

@st.cache_data
def load_document_bytes(file_path):
    try:
        if os.path.exists(file_path):
            print(f"--- Loading document bytes from: {file_path} ---")
            with open(file_path, "rb") as f:
                return f.read()
        else:
            st.error(f"Original document file not found: {file_path}. Please ensure it was copied by the pre-processing script.")
            return None
    except Exception as e:
        st.error(f"Error loading document bytes from {file_path}: {e}")
        traceback.print_exc()
        return None

def extract_with_pattern(key, pattern, text_to_search):
    flags = re.IGNORECASE
    match = re.search(pattern, text_to_search, flags)
    if match: 
        content = match.group(1).strip()
        content = re.sub(r'\.(?=[a-zA-Z0-9])', '. ', content) 
        content = re.sub(r'([a-zA-Z])([0-9])', r'\1 \2', content) 
        content = re.sub(r'([0-9])([a-zA-Z])', r'\1 \2', content)
        content = re.sub(r'([a-zA-Z0-9])(\()', r'\1 \2', content) 
        content = re.sub(r'(\))([a-zA-Z0-9])', r'\1 \2', content) 
        common_stuck_words = [
            'and', 'of', 'in', 'to', 'for', 'with', 'on', 'at', 'from', 'by', 'about', 'as', 'is', 'it', 'the', 'a',
            'million', 'thousand', 'hundred', 'billion', 'trillion', 'dollar', 'dollars', 'euro', 'euros', 'yen', 'pound', 'pounds',
            'usd', 'eur', 'gbp', 'jpy', 'january', 'february', 'march', 'april', 'may', 'june', 'july', 
            'august', 'september', 'october', 'november', 'december', 'monday', 'tuesday', 'wednesday', 
            'thursday', 'friday', 'saturday', 'sunday', 'policy', 'insurance', 'claim', 'risk', 'premium', 
            'treaty', 'actuary', 'cedent', 'annually', 'during', 'year', 'they', 'experience', 'losses', 
            'loss', 'totaling', 'aggregate', 'retention', 'how', 'much', 'if', 'any', 'will', 'reinsurer', 'pay',
            'what', 'main', 'purpose', 'company' 
        ]
        for word_to_space in common_stuck_words:
            pattern_stuck_after = rf'(\b{re.escape(word_to_space)}\b)([a-zA-Z0-9])'
            content = re.sub(pattern_stuck_after, rf'\1 \2', content, flags=re.IGNORECASE)
            pattern_stuck_before = rf'([a-zA-Z0-9])(\b{re.escape(word_to_space)}\b)'
            content = re.sub(pattern_stuck_before, rf'\1 \2', content, flags=re.IGNORECASE)
        content = re.sub(r'\s{2,}', ' ', content).strip() 
        return content
    print(f"--- Terminal Log: Parsing Warning: Could not find '{key}' in response using pattern: {pattern} ---")
    return None

def generate_quiz_question_runtime(subject, objective, all_doc_chunks_text, faiss_idx_loaded, runtime_llm, 
                                   difficulty="average", previous_question_text=None, focused_chunk_idx=None, 
                                   available_indices_list=None):
    # This function's content remains as you provided in message #121
    # For brevity, I'm not repeating its full internal logic here, but it's assumed to be complete.
    # Ensure it uses `extract_with_pattern` and `runtime_llm` correctly.
    # ... (Full content of generate_quiz_question_runtime as in your message #121) ...
    print(f"--- Runtime Question Gen: Mode: {'Focused' if focused_chunk_idx is not None else 'Normal'}. Difficulty: {difficulty}, Subject: '{subject}'. Chunks available: {len(all_doc_chunks_text) if all_doc_chunks_text else 'None'}. Focused Idx: {focused_chunk_idx}")
    if not runtime_llm:
        st.error("Runtime AI Model not configured for dynamic question generation.")
        return None, []
    if not all_doc_chunks_text or not faiss_idx_loaded:
        st.error("Document chunks or FAISS index not available for dynamic question generation.")
        return None, []

    context_text_list = []
    original_context_indices = []
    source_of_context = ""
    num_context_to_use_runtime = NUM_CONTEXT_CHUNKS_FOR_DYNAMIC_QUESTIONS

    if focused_chunk_idx is not None and (0 <= focused_chunk_idx < len(all_doc_chunks_text)):
        source_of_context = f"Focused on Chunk {focused_chunk_idx + 1}"
        query_text_for_focus = all_doc_chunks_text[focused_chunk_idx]
        try:
            query_embedding_response = genai.embed_content(model=EMBEDDING_MODEL_ID, content=query_text_for_focus, task_type="RETRIEVAL_QUERY")
            query_embedding = np.array(query_embedding_response['embedding']).astype('float32').reshape(1, -1)
            distances, faiss_indices_ret = faiss_idx_loaded.search(query_embedding, k=num_context_to_use_runtime)
            retrieved_indices = [int(i) for i in faiss_indices_ret[0] if 0 <= i < len(all_doc_chunks_text)]
            final_context_indices = []
            if focused_chunk_idx in retrieved_indices:
                final_context_indices.append(focused_chunk_idx)
            else:
                final_context_indices.append(focused_chunk_idx)
            for idx in retrieved_indices:
                if len(final_context_indices) >= num_context_to_use_runtime: break
                if idx != focused_chunk_idx and idx not in final_context_indices:
                    final_context_indices.append(idx)
            original_context_indices = sorted(list(set(final_context_indices)))
            context_text_list = [all_doc_chunks_text[i] for i in original_context_indices]
            source_of_context += f" + {len(context_text_list)-1 if len(context_text_list)>0 else 0} FAISS neighbors"
        except Exception as e_faiss_focus:
            print(f"FAISS query error (focused_chunk_idx mode runtime): {e_faiss_focus}")
            original_context_indices = [focused_chunk_idx]
            context_text_list = [all_doc_chunks_text[focused_chunk_idx]]
            source_of_context += " (FAISS failed, using only focused chunk)"
    elif previous_question_text and difficulty == "simpler":
        source_of_context = "Simpler (FAISS on Prev Q)"
        try:
            query_embedding_response = genai.embed_content(model=EMBEDDING_MODEL_ID, content=previous_question_text, task_type="RETRIEVAL_QUERY")
            query_embedding = np.array(query_embedding_response['embedding']).astype('float32').reshape(1, -1)
            distances, faiss_indices_ret = faiss_idx_loaded.search(query_embedding, k=num_context_to_use_runtime)
            original_context_indices = [int(i) for i in faiss_indices_ret[0] if 0 <= i < len(all_doc_chunks_text)]
            context_text_list = [all_doc_chunks_text[i] for i in original_context_indices]
        except Exception as e_faiss_simpler:
            print(f"FAISS query error (simpler, runtime): {e_faiss_simpler}")
            if available_indices_list and len(available_indices_list) >= num_context_to_use_runtime:
                original_context_indices = random.sample(available_indices_list, num_context_to_use_runtime)
                context_text_list = [all_doc_chunks_text[i] for i in original_context_indices]
                source_of_context = "Simpler (Random Fallback)"
    elif difficulty == "harder" and available_indices_list:
        source_of_context = "Harder (New Random Chunks)"
        if len(available_indices_list) >= num_context_to_use_runtime:
            original_context_indices = random.sample(available_indices_list, num_context_to_use_runtime)
            context_text_list = [all_doc_chunks_text[i] for i in original_context_indices]
        else:
            original_context_indices = available_indices_list[:]
            context_text_list = [all_doc_chunks_text[i] for i in original_context_indices]
    else: 
        source_of_context = "Dynamic - Random Chunks (Fallback)"
        num_to_sample = min(num_context_to_use_runtime, len(all_doc_chunks_text))
        if num_to_sample > 0:
            original_context_indices = random.sample(list(range(len(all_doc_chunks_text))), num_to_sample)
            context_text_list = [all_doc_chunks_text[i] for i in original_context_indices]
        else:
            st.error("No chunks available for random fallback context.")
            return None, []

    if not context_text_list:
        st.error("Failed to get any context for dynamic question generation.")
        return None, []

    print(f"--- Runtime Question Gen Context: {source_of_context}. Num: {len(context_text_list)}. Indices: {original_context_indices} ---")
    context_to_send = "\n\n---\n\n".join(context_text_list)
    max_context_chars = 8000 
    if len(context_to_send) > max_context_chars: context_to_send = context_to_send[:max_context_chars] + "..."

    difficulty_prompt_instruction = f"Generate a multiple-choice question of {difficulty} difficulty based on the 'Provided Text Context'. The document's primary objective is: '{objective}'."
    
    prompt = f"""
    You are an expert quiz generator. The subject of the document is '{subject}'.
    {difficulty_prompt_instruction}
    Guidelines:
    1. The question must test understanding of principles related to '{subject}' and the document's objective, directly covered in the 'Provided Text Context'.
    2. NO METADATA QUESTIONS. Focus strictly on the substance.
    3. Generate 4 plausible options (A, B, C, D).
    4. Ensure exactly ONE option is unambiguously correct according to the 'Provided Text Context'.
    5. Incorrect options must be relevant but clearly wrong based *only* on the 'Provided Text Context'.
    6. Output Format (EXACTLY as shown, using these precise labels and newlines, no extra markdown around labels):
    Question: [Your question here]
    A: [Option A text]
    B: [Option B text]
    C: [Option C text]
    D: [Option D text]
    Correct Answer: [Letter ONLY, e.g., C]
    Explanation: [Brief explanation from context.]

    Provided Text Context:
    ---
    {context_to_send}
    ---
    Generate the question now.
    """
    
    response_text = None; llm_response_obj = None
    max_retries = 3; retry_delay = 5
    try:
        for attempt in range(max_retries):
            try:
                print(f"--- Sending prompt to Runtime LLM (Flash) (Attempt {attempt + 1}/{max_retries}) ---")
                safety_settings = { 
                    genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                }
                llm_response_obj = runtime_llm.generate_content(prompt, safety_settings=safety_settings, request_options={'timeout': 60}) 
                if llm_response_obj and llm_response_obj.candidates and hasattr(llm_response_obj.candidates[0].content, 'parts') and llm_response_obj.candidates[0].content.parts:
                    response_text = llm_response_obj.candidates[0].content.parts[0].text.strip()
                    if response_text: 
                        print(f"--- Runtime LLM (Flash) responded (Attempt {attempt + 1}). ---")
                        break 
                reason = "Unknown reason or empty content"
                if llm_response_obj:
                    if not (llm_response_obj.candidates and hasattr(llm_response_obj.candidates[0].content, 'parts') and llm_response_obj.candidates[0].content.parts and response_text): 
                         reason = llm_response_obj.candidates[0].finish_reason.name if llm_response_obj.candidates and llm_response_obj.candidates[0].finish_reason else "Empty/invalid content part"
                    if llm_response_obj.prompt_feedback and llm_response_obj.prompt_feedback.block_reason:
                         reason = f"Blocked - {llm_response_obj.prompt_feedback.block_reason.name}"
                print(f"--- Runtime LLM Response Text Empty/Blocked (Attempt {attempt + 1}). Reason: {reason} ---")
                if attempt < max_retries - 1: time.sleep(retry_delay)
                else: 
                    st.error(f"Runtime LLM response issue after {max_retries} attempts. Reason: {reason}")
                    return None, []
            except Exception as e_api: 
                print(f"--- Runtime LLM API Error (Attempt {attempt + 1}/{max_retries}): {type(e_api).__name__}: {e_api} ---")
                if attempt < max_retries - 1: time.sleep(retry_delay)
                else:
                    st.error(f"Runtime LLM API Error after all retries: {e_api}") 
                    return None, [] 
        if not response_text: 
            st.error("Failed to get valid response text from Runtime LLM after retries.")
            return None, [] 
        
        patterns = {
            "question": r"^\**Question\**\s*[:\-]\s*(.+?)\s*(?=\n\s*\**\s*[A-Z]\s*[:\.\)]|\Z)",
            "A": r"\n\s*\**\s*[Aa]\s*[:\.\)]\s*\**(.+?)\**\s*(?=\n\s*\**\s*[Bb]\s*[:\.\)]|\Z)",
            "B": r"\n\s*\**\s*[Bb]\s*[:\.\)]\s*\**(.+?)\**\s*(?=\n\s*\**\s*[Cc]\s*[:\.\)]|\Z)",
            "C": r"\n\s*\**\s*[Cc]\s*[:\.\)]\s*\**(.+?)\**\s*(?=\n\s*\**\s*[Dd]\s*[:\.\)]|\Z)",
            "D": r"\n\s*\**\s*[Dd]\s*[:\.\)]\s*\**(.+?)\**\s*(?=\n\s*\**\s*Correct Answer\s*[:\.\)]|\Z)",
            "correct_answer": r"\n\s*\**\s*Correct Answer\s*[:\.\)]\s*\**\s*\[?([A-Da-d])\]?\s*\**",
            "explanation": r"\n\s*\**\s*Explanation\s*[:\.\)]\s*\**([\s\S]+?)\**\s*(\Z|\n\s*\**\s*(Question:|A:|B:|C:|D:|Correct Answer:))"
        }
        parsed_data = {}
        parsed_data["question"] = extract_with_pattern("Question", patterns["question"], response_text)
        options_dict = {}
        options_dict["A"] = extract_with_pattern("Option A", patterns["A"], response_text)
        options_dict["B"] = extract_with_pattern("Option B", patterns["B"], response_text)
        options_dict["C"] = extract_with_pattern("Option C", patterns["C"], response_text)
        options_dict["D"] = extract_with_pattern("Option D", patterns["D"], response_text)
        parsed_data["options"] = {k: v for k, v in options_dict.items() if v is not None}
        correct_ans_raw = extract_with_pattern("Correct Answer", patterns["correct_answer"], response_text)
        parsed_data["correct_answer"] = correct_ans_raw.upper() if correct_ans_raw else None
        parsed_data["explanation"] = extract_with_pattern("Explanation", patterns["explanation"], response_text)
        
        required_keys = ["question", "options", "correct_answer", "explanation"]
        if not all(parsed_data.get(k) for k in required_keys) or len(parsed_data.get("options", {})) != 4:
            print(f"--- ERROR: Parsing failed for Runtime LLM response. Data: {parsed_data}. Options count: {len(parsed_data.get('options', {}))}. Raw: {response_text[:200]} ---")
            st.error("AI response format issue during dynamic question generation.")
            return None, original_context_indices 
        if parsed_data["correct_answer"] not in ["A", "B", "C", "D"]:
            print(f"--- ERROR: Invalid correct answer from Runtime LLM: '{parsed_data['correct_answer']}'. Raw: {response_text[:200]} ---")
            st.error("AI produced an invalid correct answer letter.")
            return None, original_context_indices

        print(f"--- Successfully parsed question data from Runtime LLM. Indices: {original_context_indices}. ---")
        return parsed_data, original_context_indices
    except Exception as e_overall: 
        print(f"--- ERROR during Runtime LLM question generation: {type(e_overall).__name__}: {e_overall} ---")
        traceback.print_exc()
        st.error(f"An unexpected error occurred while generating the question: {e_overall}")
        return None, []

def display_heatmap_grid(): 
    st.subheader("📘 Document Coverage & Performance Heatmap") 
    st.caption("Click on a section's colored square to view its full text. Note: YOU'LL NEED TO SCROLL UP TO SEE!")
    # Corrected CSS selector for pop-up expander to be more specific
    st.markdown("""
    <style>
        button[aria-label^="heatmap_square_btn_"] { 
            width: 22px !important; min-width: 22px !important; height: 22px !important;
            padding: 0px !important; margin: 1px !important; border: none !important;
            background-color: transparent !important; box-shadow: none !important;
            font-size: 14px !important; line-height: 18px !important; 
            text-align: center !important; display: inline-flex !important;
            align-items: center !important; justify-content: center !important; overflow: hidden;
        }
        div[data-testid="stExpander"][id^="detail_expander_pop_up_"] div[data-testid="stVerticalBlock"] div[data-testid="stMarkdownContainer"] p,
        div[data-testid="stExpander"][id^="detail_expander_pop_up_"] div[data-testid="stVerticalBlock"] p { /* Added for direct p in expander */
            margin-top: 0.15rem !important; margin-bottom: 0.15rem !important; line-height: 1.3 !important;
        }
        div[data-testid="stExpander"][id^="detail_expander_pop_up_"] div[data-testid="stVerticalBlock"] hr {
            margin-top: 0.25rem !important; margin-bottom: 0.25rem !important; border-top: 1px solid #e0e0e0 !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    colors_map = {
        0: {"emoji": "🟦", "label": "Not Quizzed"}, 1: {"emoji": "🟩", "label": "Correct"},
        2: {"emoji": "🟨", "label": "Incorrect (1x)"}, 3: {"emoji": "🟥", "label": "Incorrect (2+x)"},
        4: {"emoji": "🟣", "label": "Reviewed"} 
    }
    doc_chunk_details_list = st.session_state.get('doc_chunk_details', [])
    hover_labels_list = st.session_state.get('chunk_hover_labels', [])
    statuses_list = st.session_state.get('chunk_review_status', [])

    if not doc_chunk_details_list or \
       not (len(doc_chunk_details_list) == len(hover_labels_list) == len(statuses_list)):
        st.warning("Heatmap data not fully initialized.")
        return
             
    legend_html_parts = [f'<span style="font-size:1.1em; margin-right:3px; vertical-align:middle;">{info["emoji"]}</span><span style="font-size:0.9em; margin-right:15px;">{info["label"]}</span>' for _, info in colors_map.items()]
    st.markdown("**Legend:** " + "".join(legend_html_parts), unsafe_allow_html=True)
    st.write("") 
            
    current_displayed_headings_path = [None] * 6 
    last_printed_heading_tuple = None
    cols_for_squares = None
    col_idx_for_squares = 0
    squares_per_row = 15 
    
    for chunk_idx, chunk_detail in enumerate(doc_chunk_details_list):
        chunk_full_headings = chunk_detail.get("full_headings_list", [])
        current_chunk_heading_tuple = tuple(chunk_full_headings)
        
        chunk_status_code = statuses_list[chunk_idx] if chunk_idx < len(statuses_list) else 0
        chunk_hover_text_for_tooltip = hover_labels_list[chunk_idx] if chunk_idx < len(hover_labels_list) else "Details"
        
        if current_chunk_heading_tuple != last_printed_heading_tuple:
            if cols_for_squares and col_idx_for_squares != 0: 
                for _ in range(col_idx_for_squares, squares_per_row): cols_for_squares[_].empty()
            
            for level, heading_text in enumerate(chunk_full_headings):
                if level >= len(current_displayed_headings_path) or current_displayed_headings_path[level] != heading_text:
                    for l_reset in range(level, len(current_displayed_headings_path)): 
                        current_displayed_headings_path[l_reset] = None
                    current_displayed_headings_path[level] = heading_text
                    if level == 0: st.markdown(f"<h5>{heading_text}</h5>", unsafe_allow_html=True) 
                    elif level == 1: st.markdown(f"<h6 style='padding-left: 20px;'>{heading_text}</h6>", unsafe_allow_html=True)
                    else: st.markdown(f"<p style='padding-left: {(level)*20}px; font-size:0.9em; font-weight:bold; margin-bottom:2px;'>{heading_text}</p>", unsafe_allow_html=True)
            
            last_printed_heading_tuple = current_chunk_heading_tuple
            cols_for_squares = st.columns(squares_per_row) 
            col_idx_for_squares = 0
        elif not chunk_full_headings and last_printed_heading_tuple != ("(General Content)",): # Handling chunks with no headings
            if cols_for_squares and col_idx_for_squares != 0:
                for _ in range(col_idx_for_squares, squares_per_row): cols_for_squares[_].empty()
            st.markdown(f"<h6><em>(Content without specific subsection heading)</em></h6>", unsafe_allow_html=True)
            last_printed_heading_tuple = ("(General Content)",)
            cols_for_squares = st.columns(squares_per_row)
            col_idx_for_squares = 0

        color_info = colors_map.get(chunk_status_code, colors_map[0])
        button_key = f"heatmap_square_btn_ded_{chunk_idx}" 

        def _create_show_detail_callback(idx_to_show):
            def callback_with_captured_idx(captured_idx=idx_to_show):
                if 'chunk_review_status' in st.session_state and \
                0 <= captured_idx < len(st.session_state.chunk_review_status) and \
                st.session_state.chunk_review_status[captured_idx] == 0: 
                    st.session_state.chunk_review_status[captured_idx] = 4 

                st.session_state.selected_heatmap_chunk_index = captured_idx
                st.session_state.quiz_mode = "view_chunk_detail"  # Navigate to new page mode
                # Ensure summary page scrolls to top when we eventually return to it
                st.session_state.scroll_to_summary_top = True 
                #st.rerun() 
            return callback_with_captured_idx

        if cols_for_squares is None: 
            cols_for_squares = st.columns(squares_per_row)
            col_idx_for_squares = 0
        with cols_for_squares[col_idx_for_squares]:
            st.button(label=f"{color_info['emoji']}", key=button_key, 
                        help=f"{chunk_hover_text_for_tooltip}", 
                        on_click=_create_show_detail_callback(chunk_idx),
                        use_container_width=False)
        col_idx_for_squares = (col_idx_for_squares + 1) % squares_per_row
        if col_idx_for_squares == 0 and chunk_idx < len(doc_chunk_details_list) -1 : 
            cols_for_squares = None 
    if cols_for_squares and col_idx_for_squares != 0:
        for _ in range(col_idx_for_squares, squares_per_row):
            cols_for_squares[_].empty()

# --- Load all data at the start ---
# This was already correctly placed at the beginning of the script in your provided code.
# The variables essential_data_loaded, doc_theme_objective, etc. are defined globally.

# CORRECTED PLACEMENT for the check on essential_data_loaded

if 'quiz_mode' not in st.session_state:
    st.session_state.quiz_mode = "welcome"

# --- LLM Configuration (for Runtime Dynamic Questions) ---
# This block should ideally be near the top, after imports and constants.
# It's generally fine where it is in your provided code as long as st.secrets is available
# when this part of the script is reached. For simplicity, keeping it as in your code.
st.session_state.setdefault('scroll_to_summary_top', False) 
st.session_state.setdefault('runtime_llm_configured', False)
# In your session state initialization / setdefault area
st.session_state.setdefault('scroll_to_view_chunk_detail_top', False)
st.session_state.setdefault('runtime_gemini_model', None)
if not st.session_state.get('runtime_llm_configured'):
    gemini_api_key_runtime = st.secrets.get("GEMINI_API_KEY")
    if gemini_api_key_runtime:
        try:
            print("--- Attempting to configure RUNTIME Gemini AI model... ---")
            runtime_model_id_to_use = st.secrets.get("RUNTIME_GEMINI_MODEL_ID", DEFAULT_RUNTIME_FLASH_MODEL_ID)
            genai.configure(api_key=gemini_api_key_runtime) 
            print(f"--- Initializing RUNTIME Gemini Model with ID: {runtime_model_id_to_use} ---")
            st.session_state.runtime_gemini_model = genai.GenerativeModel(runtime_model_id_to_use)
            st.session_state.runtime_llm_configured = True
            print(f"--- RUNTIME Gemini AI Model ({runtime_model_id_to_use}) Configured successfully. ---")
        except KeyError as ke:
            st.error(f"Runtime Gemini Config Error: Missing key in secrets - {ke}.")
        except Exception as e_gemini:
            st.error(f"Runtime AI Config Error: {e_gemini}")
    else:
        st.error("GEMINI_API_KEY not found in secrets for runtime model.")
if not st.session_state.get('runtime_gemini_model') and not st.session_state.pre_generated_questions:
    st.warning("Runtime LLM not configured and no pre-generated questions. Quiz functionality will be limited.")


# --- Load all data at the start ---
doc_theme_objective = load_json_data(THEME_OBJECTIVE_FILE)
doc_chunk_details = load_json_data(DOC_CHUNK_DETAILS_FILE)
faiss_index = load_faiss_index(FAISS_INDEX_FILE)
faiss_index_chunks = load_json_data(FAISS_CHUNKS_FILE)
pre_generated_questions = load_json_data(PRE_GENERATED_QUESTIONS_FILE)
original_document_bytes = load_document_bytes(ORIGINAL_DOCUMENT_PATH)

essential_data_loaded = (
    doc_theme_objective is not None and
    doc_chunk_details is not None and
    faiss_index is not None and
    faiss_index_chunks is not None and
    pre_generated_questions is not None
)

if not essential_data_loaded:
    st.error("Critical data files could not be loaded. Please run the pre-processing script and ensure the files are in the correct directory.")
    st.stop()

# Initialize quiz mode if not set
if 'quiz_mode' not in st.session_state:
    st.session_state.quiz_mode = "welcome"

# --- Initialize Session State ---
if 'app_initialized' not in st.session_state:
    print("--- Initializing app state for the first time ---")
    st.session_state.app_initialized = True
    
    # Initialize with loaded data
    st.session_state.doc_theme_objective = doc_theme_objective
    st.session_state.doc_chunk_details = doc_chunk_details
    st.session_state.faiss_index = faiss_index
    st.session_state.faiss_index_chunks = faiss_index_chunks
    st.session_state.pre_generated_questions = pre_generated_questions
    st.session_state.original_document_bytes = original_document_bytes
    
    # Initialize heatmap status arrays
    st.session_state.chunk_review_status = [0] * len(doc_chunk_details)
# Initialize chunk_hover_labels from loaded doc_chunk_details (first 50 words)
    # This replaces the placeholder line for st.session_state.chunk_hover_labels
    if st.session_state.doc_chunk_details: # Check if doc_chunk_details is loaded and not empty
        num_words_for_hover = 50
        hover_labels = []
        for item in st.session_state.doc_chunk_details: # Iterate over the loaded chunk details
            text_content = item.get('text', '') # Get the text from each chunk detail
            words = text_content.split()
            label = ' '.join(words[:num_words_for_hover])
            if len(words) > num_words_for_hover:
                label += "..."
            hover_labels.append(label)
        st.session_state.chunk_hover_labels = hover_labels

        # The following line for chunk_review_status is likely already correct in your code
        # but ensure it's based on len(st.session_state.doc_chunk_details)
        # If it's not already there or different, make sure it looks like this:
        if len(st.session_state.chunk_review_status) != len(st.session_state.doc_chunk_details):
            st.session_state.chunk_review_status = [0] * len(st.session_state.doc_chunk_details)
    else: 
        # If doc_chunk_details is empty or not loaded, set empty lists for safety
        st.session_state.chunk_hover_labels = []
        st.session_state.chunk_review_status = [] # Also ensure this is handled
    
    # Initialize quiz state
    st.session_state.available_pregen_q_indices = list(range(len(pre_generated_questions)))
    random.shuffle(st.session_state.available_pregen_q_indices)
    st.session_state.current_quiz_questions = []
    st.session_state.current_quiz_question_idx_ptr = 0
    st.session_state.question_number = 0
    st.session_state.quiz_started_at_least_once = False

if st.session_state.quiz_mode == "welcome":
    # Welcome Page specific header elements
    try:
        if 'LOGO_PATH' in globals() and os.path.exists(LOGO_PATH):
             st.image(LOGO_PATH, width=150) # Adjust width as needed
        # else: st.caption("Logo not found for welcome") # Optional: placeholder if needed
    except Exception as e:
        print(f"Error loading logo on welcome page: {e}")

    st.title("AI Quiz Tutor") 
    st.header("Introduction to Insurance") 

    # The st.write("---") below is fine, or the one in display_app_header handles it for other pages
    st.write("---") 

    col1_welcome, col2_welcome = st.columns(2)
#   with col1_welcome:
#       if st.button("Start Quiz", type="primary", use_container_width=True): # Emoji removed
#           st.session_state.quiz_mode = "quiz_init"
#           st.rerun()
#   with col2_welcome:
#       if st.button("📄 View Document", use_container_width=True):
#          st.session_state.quiz_mode = "view_document"
#          st.rerun()
    with col1_welcome:
        if st.button("Start Quiz", type="primary", use_container_width=True):
            st.session_state.quiz_mode = "quiz_init"
            st.rerun()
    with col2_welcome:
        if st.button("View Document", use_container_width=True):
            st.session_state.quiz_mode = "view_document"
            st.rerun()

elif st.session_state.quiz_mode == "view_document":
    st.subheader(f"Viewing: {ORIGINAL_DOC_FILENAME}") 
    if original_document_bytes:
        if ORIGINAL_DOC_FILENAME.lower().endswith(".pdf"):
            base64_pdf = base64.b64encode(original_document_bytes).decode('utf-8')
            pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600px" type="application/pdf"></iframe>'
            st.markdown(pdf_display, unsafe_allow_html=True)
        else:
            st.info(f"Document '{ORIGINAL_DOC_FILENAME}' can be downloaded for viewing.")
            st.download_button(
                label=f"Download {ORIGINAL_DOC_FILENAME}", data=original_document_bytes,
                file_name=ORIGINAL_DOC_FILENAME, mime="application/octet-stream" 
            )
    else:
        st.error("Document data could not be loaded for viewing.")
    if st.button("⬅️ Back to Welcome"):
        st.session_state.quiz_mode = "welcome"
        st.rerun()
    st.write("---")

elif st.session_state.quiz_mode == "quiz_init":
    print("--- Initializing new quiz ---")
    st.session_state.question_number = 0 
    st.session_state.current_question_data = None
    st.session_state.user_answer = None
    st.session_state.feedback_message = None
    st.session_state.show_explanation = False
    st.session_state.last_answer_correct = None
    st.session_state.quiz_started_at_least_once = True  
    st.session_state.incorrectly_answered_questions = []
    st.session_state.total_questions_answered_in_current_quiz = 0
    if st.session_state.doc_chunk_details:
        st.session_state.chunk_review_status = [0] * len(st.session_state.doc_chunk_details)
    else:
        st.session_state.chunk_review_status = []

    if st.session_state.pre_generated_questions:
        available_indices = list(range(len(st.session_state.pre_generated_questions)))
        random.shuffle(available_indices)
        st.session_state.current_quiz_questions = [st.session_state.pre_generated_questions[i] for i in available_indices]
        st.session_state.current_quiz_question_idx_ptr = 0
        if st.session_state.current_quiz_questions:
            st.session_state.current_question_data = st.session_state.current_quiz_questions[0]
            st.session_state.question_number = 1
            st.session_state.quiz_mode = "quiz"
        else:
            st.error("No pre-generated questions available to start the quiz.")
            st.session_state.quiz_mode = "welcome"
    else:
        st.error("Pre-generated question bank is empty. Cannot start quiz.")
        st.session_state.quiz_mode = "welcome"
    st.rerun()

elif st.session_state.quiz_mode == "quiz":
    display_app_header() # <--- THIS CALLS YOUR NEW HEADER

    # Optional: Expander for viewing document during quiz
    if original_document_bytes and ORIGINAL_DOC_FILENAME.lower().endswith(".pdf"):
        with st.expander("View Document Reference", expanded=False):
            base64_pdf = base64.b64encode(original_document_bytes).decode('utf-8')
            pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600px" type="application/pdf"></iframe>'
            st.markdown(pdf_display, unsafe_allow_html=True)
    # The st.write("---") is now part of display_app_header()

    q_data = st.session_state.current_question_data
    if q_data:
        quiz_container = st.container(border=True)
        with quiz_container:
            st.subheader(f"Question {st.session_state.question_number}")
            st.markdown(f"**{q_data.get('question', 'Error: Question text missing.')}**")
            options = q_data.get("options", {})
            options_list = []
            for opt_key in ["A", "B", "C", "D"]:
                opt_text = options.get(opt_key)
                if opt_text is not None: options_list.append(f"{opt_key}: {opt_text}")
                else: options_list.append(f"{opt_key}: [Option not available]")
            
            radio_key = f"quiz_q_{st.session_state.question_number}_{q_data.get('question','').replace(' ','_')[:20]}"
            current_radio_index = None
            if st.session_state.show_explanation and st.session_state.user_answer:
                user_answer_full_text_prefix = f"{st.session_state.user_answer}:"
                for i, opt_text_iter in enumerate(options_list):
                    if opt_text_iter.startswith(user_answer_full_text_prefix):
                        current_radio_index = i
                        break
            selected_opt_text = st.radio("Select your answer:", options_list, index=current_radio_index, key=radio_key, 
                                         disabled=st.session_state.show_explanation, label_visibility="collapsed")
            if not st.session_state.show_explanation:
                st.session_state.user_answer = selected_opt_text.split(":")[0] if selected_opt_text and ":" in selected_opt_text else None
            # --- New button logic in your 'quiz' mode ---
            st.write("---") # Separator before action buttons

            if not st.session_state.show_explanation:
                col1_action, col2_action = st.columns(2)
                with col1_action:
                    if st.button("Submit Answer", type="primary", key=f"submit_q_{st.session_state.question_number}", use_container_width=True):
                        if st.session_state.user_answer is None:
                            st.warning("Please select an answer.")
                        else:
                            # --- (Your existing answer processing logic for correct/incorrect) ---
                            st.session_state.total_questions_answered_in_current_quiz += 1
                            correct_answer = q_data.get("correct_answer")
                            context_indices_for_this_q_pregen = q_data.get("context_indices_used", [])

                            if st.session_state.user_answer == correct_answer:
                                st.session_state.feedback_message = "Correct!"
                                st.session_state.last_answer_correct = True
                                for idx in context_indices_for_this_q_pregen:
                                    if 0 <= idx < len(st.session_state.chunk_review_status) and st.session_state.chunk_review_status[idx] != 1:
                                        st.session_state.chunk_review_status[idx] = 1
                            else:
                                st.session_state.feedback_message = f"Incorrect. The correct answer was: **{correct_answer}**. {options.get(correct_answer, '')}"
                                st.session_state.last_answer_correct = False
                                incorrect_q_data = q_data.copy()
                                incorrect_q_data["question_number"] = f"Pre-gen Q (Attempt {st.session_state.question_number})"
                                incorrect_q_data["user_selected_answer_letter"] = st.session_state.user_answer
                                incorrect_q_data["user_selected_answer_text"] = options.get(st.session_state.user_answer, "N/A")
                                st.session_state.incorrectly_answered_questions.append(incorrect_q_data)
                                for idx in context_indices_for_this_q_pregen:
                                    if 0 <= idx < len(st.session_state.chunk_review_status):
                                        current_status = st.session_state.chunk_review_status[idx]
                                        if current_status in [0, 4]: st.session_state.chunk_review_status[idx] = 2
                                        elif current_status == 2: st.session_state.chunk_review_status[idx] = 3
                            # --- (End of answer processing logic) ---
                            st.session_state.show_explanation = True
                            st.rerun() # Keep rerun for immediate feedback display

                with col2_action:
                    if st.button("Stop Quiz & View Summary", key="stop_quiz_main_btn_ded_pre_submit", use_container_width=True): 
                        st.session_state.quiz_mode = "summary"
                        st.session_state.scroll_to_summary_top = True 
                        st.rerun() # Keep rerun for navigation

            # This 'if' block now handles the display AFTER an answer is submitted
            if st.session_state.show_explanation:
                if st.session_state.feedback_message:
                    if st.session_state.last_answer_correct:
                        st.success(st.session_state.feedback_message)
                    else:
                        st.error(st.session_state.feedback_message)
                st.caption(f"Explanation: {q_data.get('explanation', 'No explanation available.')}")

                st.write("---") # Separator before next action buttons
                col1_nav, col2_nav = st.columns(2)

                with col1_nav:
                    if st.session_state.current_quiz_question_idx_ptr < len(st.session_state.current_quiz_questions) - 1:
                        if st.button("Next Question", key=f"next_q_{st.session_state.question_number}", use_container_width=True, type="primary"):
                            st.session_state.current_quiz_question_idx_ptr += 1
                            st.session_state.question_number +=1
                            st.session_state.current_question_data = st.session_state.current_quiz_questions[st.session_state.current_quiz_question_idx_ptr]
                            st.session_state.user_answer = None
                            st.session_state.feedback_message = None
                            st.session_state.show_explanation = False
                            st.session_state.last_answer_correct = None
                            st.rerun() # Keep rerun for navigation
                    else: # End of quiz
                        st.info("You have completed all pre-generated questions for this quiz session!")
                        if st.button("View Summary", key="view_summary_from_quiz_end", type="primary", use_container_width=True):
                            st.session_state.quiz_mode = "summary"
                            st.session_state.scroll_to_summary_top = True
                            st.rerun() # Keep rerun for navigation
                with col2_nav:
                    # "Stop Quiz" button also available after feedback, next to "Next/View Summary"
                    if st.button("Stop Quiz & View Summary", key="stop_quiz_main_btn_ded_post_submit", use_container_width=True): 
                        st.session_state.quiz_mode = "summary"
                        st.session_state.scroll_to_summary_top = True 
                        st.rerun() # Keep rerun for navigation

            # The old st.divider() and single "Stop Quiz" button at the very end of quiz_container are removed.
# --- End of new button logic ---
    else:
        st.info("Loading question or quiz has ended.")
        if st.button("Back to Welcome", key="quiz_ended_back_to_welcome_btn_ded"): 
            st.session_state.quiz_mode = "welcome"
            st.session_state.scroll_to_summary_top = True # <-- SET FLAG
            st.rerun()

# This replaces your existing: elif st.session_state.quiz_mode == "summary": block

elif st.session_state.quiz_mode == "summary":
    # --- Scroll to Top of Summary Page Logic ---
    if st.session_state.get('scroll_to_summary_top', False):
        scroll_js_top = """ 
            <script>
                setTimeout(function() {
                    window.scrollTo({ top: 0, behavior: 'smooth' }); 
                }, 100); // Delay of 100 milliseconds
            </script>
        """
        st.markdown(scroll_js_top, unsafe_allow_html=True)
        st.session_state.scroll_to_summary_top = False 
    # --- End of Scroll to Top Block ---

    display_app_header() 

    _summary_scroll_anchor = st.empty() 

    # Display Document Subject/Filename Fallback
    if st.session_state.get('doc_theme_objective') and st.session_state.doc_theme_objective.get('core_subject'):
        # Line for "Document Subject" was intentionally removed as per your request.
        # You can add st.caption(f"Document Objective: {st.session_state.doc_theme_objective.get('primary_objective', 'N/A')}") here if desired.
        pass
    else:
        st.caption(f"Summary for document: {ORIGINAL_DOC_FILENAME}") 

    # Score and Stats Display
    total_answered = st.session_state.total_questions_answered_in_current_quiz
    incorrect_list = st.session_state.incorrectly_answered_questions
    num_incorrect = len(incorrect_list)
    num_correct = total_answered - num_incorrect
    col1_score, col2_score = st.columns([1, 3])
    with col1_score:
        st.metric(label="Score", value=f"{(num_correct / total_answered * 100):.1f}%" if total_answered > 0 else "N/A")
    with col2_score:
        st.write(f"**Questions Answered:** {total_answered}")
        st.write(f"**Correct:** {num_correct}, **Incorrect:** {num_incorrect}")
    st.divider()

    # Review Incorrect Answers Expander
    if not incorrect_list and total_answered > 0:
        st.success("Perfect score! All questions answered correctly.")
    elif incorrect_list:
        with st.expander("Review Your Incorrect Answers", expanded=False): # Default to collapsed
            for item_idx, item_data in enumerate(incorrect_list):
                question_text = item_data.get("question", "N/A")
                options_dict = item_data.get("options", {})
                user_answer_letter = item_data.get("user_selected_answer_letter", "?")
                user_answer_text = item_data.get("user_selected_answer_text", "Not recorded.")
                correct_answer_letter = item_data.get("correct_answer", "?")
                correct_answer_text_from_options = options_dict.get(correct_answer_letter, "N/A")
                explanation = item_data.get("explanation", "N/A")
                st.error(f"**Q ({item_data.get('question_number','Focused')}): {question_text}**")
                st.write(f"> Your Answer: **{user_answer_letter}**. {user_answer_text}")
                st.write(f"> Correct Answer: **{correct_answer_letter}**. {correct_answer_text_from_options}")
                st.caption(f"Explanation: {explanation}")
                if item_idx < len(incorrect_list) - 1: st.markdown("---")
    elif total_answered == 0 and not st.session_state.get("quiz_started_at_least_once", False):
        st.info("You haven't started the quiz yet for this session.")
    elif total_answered == 0:
        st.info("No questions were answered in this quiz attempt.")
    
    # --- MOVED BUTTONS ---
    st.divider()
    col_sum_btn1, col_sum_btn2 = st.columns(2)
    with col_sum_btn1:
        if st.button("Reset Score and Start New Quiz", key="start_new_quiz_summary_ded_btn", type="primary", use_container_width=True):
            st.session_state.quiz_mode = "quiz_init"
            st.session_state.scroll_to_summary_top = False # Reset this just in case
            st.session_state.scroll_to_heatmap_detail = False 
            st.rerun()
    with col_sum_btn2:
        if st.button("Back to Welcome Page", key="summary_to_welcome_ded_btn", use_container_width=True):
            st.session_state.quiz_mode = "welcome"
            # Full reset logic from your existing code
            st.session_state.question_number = 0
            st.session_state.current_question_data = None
            st.session_state.incorrectly_answered_questions = []
            st.session_state.total_questions_answered_in_current_quiz = 0
            if st.session_state.pre_generated_questions:
                st.session_state.available_pregen_q_indices = random.sample(
                    list(range(len(st.session_state.pre_generated_questions))),
                    len(st.session_state.pre_generated_questions)
                )
            else: st.session_state.available_pregen_q_indices = []
            st.session_state.current_quiz_questions = []
            st.session_state.current_quiz_question_idx_ptr = 0
            if st.session_state.doc_chunk_details:
                st.session_state.chunk_review_status = [0] * len(st.session_state.doc_chunk_details) 
            else: st.session_state.chunk_review_status = []
            st.session_state.scroll_to_heatmap_detail = False
            st.session_state.scroll_to_summary_top = False
            st.session_state.quiz_started_at_least_once = False
            st.rerun()
    st.divider()
    # --- END OF MOVED BUTTONS ---
    


    display_heatmap_grid()
    
# Replace your entire existing: elif st.session_state.quiz_mode == "view_chunk_detail": block with this:

elif st.session_state.quiz_mode == "view_chunk_detail":
    # --- Header for View Chunk Detail Page (Logo and "AI Quiz Tutor" title removed) ---
    st.header("Document Chunk Detail") # This is now the main title for this page
    st.write("---") # Separator after the header
    # --- End Header for View Chunk Detail Page ---

    selected_idx = st.session_state.get('selected_heatmap_chunk_index')

    if selected_idx is None or not (0 <= selected_idx < len(st.session_state.get('doc_chunk_details', []))):
        st.error("No chunk selected or invalid chunk index.")
        if st.button("Back to Summary Page", key="vcd_back_to_summary_error"):
            st.session_state.quiz_mode = "summary"
            st.session_state.scroll_to_summary_top = True 
           # st.rerun()
    else:
        chunk_info = st.session_state.doc_chunk_details[selected_idx]
        full_headings_str_for_title = " -> ".join(chunk_info.get("full_headings_list", ["Selected Chunk"]))
        
        st.subheader(f"Details for Chunk: {full_headings_str_for_title}")
        # The caption "(Based on Document Chunk Index: {selected_idx})" has been removed.
        # st.caption(f"(Based on Document Chunk Index: {selected_idx})") # <-- DELETED/COMMENTED OUT
        
        # To reduce spacing between the "Previous", "Selected", and "Next" chunk text blocks,
        # we will remove the st.markdown("---") that was previously between them.
        # The spacing within each paragraph is controlled by p_style.
        
        p_style = "margin-top: 2px; margin-bottom: 2px; line-height: 1.2;" # Reduced line-height slightly for tighter packing

        # Previous Chunk
        if selected_idx > 0:
            prev_chunk_idx = selected_idx - 1
            prev_chunk_info = st.session_state.doc_chunk_details[prev_chunk_idx]
            if chunk_info.get("full_headings_list", [None])[:-1] == prev_chunk_info.get("full_headings_list", [None])[:-1] and \
               len(chunk_info.get("full_headings_list", [])) > 0 :
                st.markdown(f"**Previous Chunk (Chunk {prev_chunk_idx} - for context):**")
                prev_text_content = prev_chunk_info.get('text', '').replace('\n', '<br>')
                st.markdown(f"<p style='{p_style}'>{prev_text_content}</p>", unsafe_allow_html=True)
                st.markdown("---") # Removed separator for tighter spacing
        
        # Selected Chunk
        st.markdown(f"**Selected Chunk Content (Chunk {selected_idx}):**")
        current_text_content = chunk_info.get('text', 'Error: Current text missing').replace('\n', '<br>')
        st.markdown(f"<p style='{p_style}'><b>{current_text_content}</b></p>", unsafe_allow_html=True)
        st.markdown("---")

        # Next Chunk
        if selected_idx < len(st.session_state.doc_chunk_details) - 1:
            next_chunk_idx = selected_idx + 1
            next_chunk_info = st.session_state.doc_chunk_details[next_chunk_idx]
            if chunk_info.get("full_headings_list", [None])[:-1] == next_chunk_info.get("full_headings_list", [None])[:-1] and \
               len(chunk_info.get("full_headings_list", [])) > 0 :
                # st.markdown("---") # Removed separator for tighter spacing
                st.markdown(f"**Next Chunk (Chunk {next_chunk_idx} - for context):**")
                next_text_content = next_chunk_info.get('text', '').replace('\n', '<br>')
                st.markdown(f"<p style='{p_style}'>{next_text_content}</p>", unsafe_allow_html=True)
        
        st.markdown("---") # Final separator before buttons
        
        # Buttons for this new page
        col1_vcd, col2_vcd = st.columns(2)
        with col1_vcd:
            if st.button("Quiz me on this Topic", key=f"quiz_me_btn_view_chunk_{selected_idx}", type="primary", use_container_width=True):
                st.session_state.heatmap_quiz_focal_chunk_idx = selected_idx 
                st.session_state.quiz_mode = "heatmap_quiz_init"
                st.rerun()
        with col2_vcd:
            if st.button("Back to Summary Page", key=f"view_chunk_to_summary_btn_{selected_idx}", use_container_width=True):
                st.session_state.quiz_mode = "summary"
                st.session_state.scroll_to_summary_top = True 
                st.rerun()
elif st.session_state.quiz_mode == "heatmap_quiz_init":
    st.title("Focused Quiz: Intro to Insurance")
    focal_chunk_idx = st.session_state.get('heatmap_quiz_focal_chunk_idx')
    if focal_chunk_idx is None or not (0 <= focal_chunk_idx < len(st.session_state.doc_chunk_details)):
        st.error("No valid chunk selected for focused quiz. Returning to summary.")
        st.session_state.quiz_mode = "summary"
        st.rerun()
    with st.spinner("Generating focused question..."):
        print(f"--- Heatmap Quiz Init: Preparing question for focal chunk {focal_chunk_idx} ---")
        st.session_state.current_question_data = None
        st.session_state.user_answer = None
        st.session_state.feedback_message = None
        st.session_state.show_explanation = False
        st.session_state.last_answer_correct = None
        st.session_state.current_focused_question_is_pregen = False
        found_pregen_q = False
        if st.session_state.pre_generated_questions:
            for i, q_data_item in enumerate(st.session_state.pre_generated_questions):
                if q_data_item.get("focal_chunk_index") == focal_chunk_idx:
                    st.session_state.current_question_data = q_data_item
                    st.session_state.current_focused_question_is_pregen = True
                    found_pregen_q = True
                    print(f"--- Heatmap Quiz Init: Found pre-generated question for focal chunk {focal_chunk_idx} ---")
                    break
        if not found_pregen_q:
            print(f"--- Heatmap Quiz Init: No suitable pre-generated question. Generating dynamically with Flash model. ---")
            if st.session_state.get('runtime_llm_configured') and st.session_state.get('runtime_gemini_model'):
                q_data_dynamic, context_indices = generate_quiz_question_runtime(
                    subject=st.session_state.doc_theme_objective.get('core_subject', 'Intro to Insurance'),
                    objective=st.session_state.doc_theme_objective.get('primary_objective', 'To provide an introduction to insurance'),
                    all_doc_chunks_text=st.session_state.faiss_index_chunks,
                    faiss_idx_loaded=st.session_state.faiss_index,
                    runtime_llm=st.session_state.runtime_gemini_model,
                    difficulty="average",
                    focused_chunk_idx=focal_chunk_idx
                )
                if q_data_dynamic:
                    st.session_state.current_question_data = q_data_dynamic
                    st.session_state.heatmap_quiz_current_context_indices = context_indices
                else: st.error("Failed to generate a dynamic question for this chunk.")
            else: st.error("Runtime LLM not configured, cannot generate dynamic question.")
        if st.session_state.current_question_data:
            st.session_state.quiz_mode = "heatmap_quiz"
        else: 
            # If still no question, go back to summary
            st.error("Could not prepare a focused question. Returning to summary.") # Added an error message for user
            st.session_state.quiz_mode = "summary"
            st.session_state.scroll_to_summary_top = True # <--- ADD THIS LINE
        st.rerun()

elif st.session_state.quiz_mode == "heatmap_quiz":
    display_app_header() # <--- THIS CALLS YOUR NEW HEADER
    # ... (rest of your heatmap_quiz logic, you can keep the st.caption for subject and st.info for focal chunk)
    focal_chunk_idx = st.session_state.get('heatmap_quiz_focal_chunk_idx')
    if st.session_state.get('doc_theme_objective'):
        st.caption(f"Document Subject: {st.session_state.doc_theme_objective.get('core_subject', 'N/A')}")
    if focal_chunk_idx is not None and 0 <= focal_chunk_idx < len(st.session_state.doc_chunk_details):
        focal_chunk_detail = st.session_state.doc_chunk_details[focal_chunk_idx]
        focal_chunk_headings = " -> ".join(focal_chunk_detail.get("full_headings_list", ["Topic"]))
        st.info(f"Focused question on topic from: **{focal_chunk_headings}** (around Chunk {focal_chunk_idx})")
    st.write("---")
    q_data = st.session_state.current_question_data
    if q_data:
        quiz_container = st.container(border=True)
        with quiz_container:
            st.markdown(f"**{q_data.get('question', 'Error: Question text missing.')}**")
            options = q_data.get("options", {})
            options_list = []
            for opt_key in ["A", "B", "C", "D"]:
                opt_text = options.get(opt_key)
                if opt_text is not None: options_list.append(f"{opt_key}: {opt_text}")
                else: options_list.append(f"{opt_key}: [Option not available]")
            radio_key_focused = f"focused_quiz_q_{focal_chunk_idx}_{q_data.get('question','').replace(' ','_')[:20]}"
            current_radio_index_focused = None
            if st.session_state.show_explanation and st.session_state.user_answer:
                user_answer_full_text_prefix = f"{st.session_state.user_answer}:"
                for i, opt_text_iter in enumerate(options_list):
                    if opt_text_iter.startswith(user_answer_full_text_prefix):
                        current_radio_index_focused = i
                        break
            selected_opt_text_focused = st.radio("Select your answer:", options_list, index=current_radio_index_focused,
                                                 key=radio_key_focused, disabled=st.session_state.show_explanation,
                                                 label_visibility="collapsed")
            if not st.session_state.show_explanation:
                st.session_state.user_answer = selected_opt_text_focused.split(":")[0] if selected_opt_text_focused and ":" in selected_opt_text_focused else None
            st.write("---")
            if not st.session_state.show_explanation:
                if st.button("Submit Answer", type="primary", key=f"submit_focused_q_{focal_chunk_idx}"):
                    if st.session_state.user_answer is None: st.warning("Please select an answer.")
                    else: # An answer was selected and submitted
                        st.session_state.total_questions_answered_in_current_quiz += 1 # <<< ADDED: Increment total
                        
                        correct_answer = q_data.get("correct_answer")
                        focal_chunk_idx = st.session_state.get('heatmap_quiz_focal_chunk_idx', 'N/A') # Get focal chunk for label

                        # Determine context indices used for this question for heatmap update
                        context_indices_for_this_q = []
                        if st.session_state.get("current_focused_question_is_pregen", False):
                             context_indices_for_this_q = q_data.get("context_indices_used", [focal_chunk_idx] if focal_chunk_idx != 'N/A' else [])
                        else: # Dynamically generated
                             context_indices_for_this_q = st.session_state.get("heatmap_quiz_current_context_indices", [focal_chunk_idx] if focal_chunk_idx != 'N/A' else [])
                        
                        if st.session_state.user_answer == correct_answer:
                            st.session_state.feedback_message = "Correct!"
                            st.session_state.last_answer_correct = True
                            # Update heatmap status for context chunks
                            for idx in context_indices_for_this_q:
                                if 0 <= idx < len(st.session_state.chunk_review_status):
                                    st.session_state.chunk_review_status[idx] = 1 # Mark as Correct
                        else:
                            st.session_state.feedback_message = f"Incorrect. Correct answer was: **{correct_answer}**. {options.get(correct_answer, '')}" # options should be defined from q_data
                            st.session_state.last_answer_correct = False
                            
                            # --- ADDED: Logic to add to main incorrectly_answered_questions list ---
                            incorrect_q_data = q_data.copy()
                            incorrect_q_data["question_number"] = f"Focused (Chunk {focal_chunk_idx})" # Identifier
                            incorrect_q_data["user_selected_answer_letter"] = st.session_state.user_answer
                            incorrect_q_data["user_selected_answer_text"] = options.get(st.session_state.user_answer, "N/A")
                            st.session_state.incorrectly_answered_questions.append(incorrect_q_data)
                            # --- END OF ADDED BLOCK ---

                            # Update heatmap status for context chunks
                            for idx in context_indices_for_this_q:
                                if 0 <= idx < len(st.session_state.chunk_review_status):
                                    current_status = st.session_state.chunk_review_status[idx]
                                    if current_status in [0, 1, 4]: st.session_state.chunk_review_status[idx] = 2
                                    elif current_status == 2: st.session_state.chunk_review_status[idx] = 3
                        
                        st.session_state.show_explanation = True
                        st.rerun()

            if st.session_state.show_explanation:
                if st.session_state.feedback_message:
                    if st.session_state.last_answer_correct: st.success(st.session_state.feedback_message)
                    else: st.error(st.session_state.feedback_message)
                st.caption(f"Explanation: {q_data.get('explanation', 'No explanation available.')}")
                col_fq1, col_fq2 = st.columns(2)
                with col_fq1:
                    if st.button("Try Another Question on This Topic", key="retry_focused_q_btn_ded", use_container_width=True): 
                        st.session_state.quiz_mode = "heatmap_quiz_init"
                        st.rerun()
                with col_fq2:
                    if st.button("Back to Quiz Summary", key="focused_q_to_summary_btn_ded", type="primary", use_container_width=True): 
                        st.session_state.quiz_mode = "summary"
                        st.session_state.current_question_data = None 
                        st.session_state.user_answer = None
                        st.session_state.feedback_message = None
                        st.session_state.show_explanation = False
                        st.session_state.last_answer_correct = None
                        #st.session_state.scroll_to_heatmap_detail = True 
                        st.session_state.scroll_to_summary_top = True    # <-- SET FLAG
                        st.rerun()
    else:
        st.error("Could not load a question for this focused quiz.")
        if st.button("Back to Quiz Summary", key="focused_q_fail_to_summary_btn_ded"): 
            st.session_state.quiz_mode = "summary"
            st.session_state.scroll_to_summary_top = True # <-- SET FLAG
            st.rerun()

else: 
    if st.session_state.get('quiz_mode') != "welcome":
        print(f"DEBUG: quiz_mode '{st.session_state.get('quiz_mode')}' is unexpected or None. Defaulting to 'welcome'.")
        st.session_state.quiz_mode = "welcome"
        st.rerun()