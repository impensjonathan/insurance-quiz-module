import os
import json
import re
import time
import random
import traceback
import io
import shutil # For copying the original PDF

import google.generativeai as genai
import numpy as np
import faiss
print("DEBUG: Script execution started, imports complete.")

# Docling and Transformers imports
try:
    from docling.document_converter import DocumentConverter
    from docling.datamodel.base_models import DocumentStream
    from docling.chunking import HybridChunker
    from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
    from transformers import AutoTokenizer
except ImportError as e_import:
    print(f"CRITICAL IMPORT ERROR: {e_import}")
    print("Ensure docling, docling-core, and transformers are installed in your Python 3.11 environment.")
    print("You may need to set up your 'py311_env' and install dependencies for this new project as well.")
    exit() # Exit script if crucial imports fail
except Exception as e_generic_import:
    print(f"UNEXPECTED ERROR during crucial imports: {e_generic_import}")
    exit()

# --- Configuration ---
# !!! USER: PLEASE UPDATE THE FOLLOWING TWO LINES !!!
PATH_TO_YOUR_INSURANCE_DOCUMENT = "/Users/jonathanimpens/Library/CloudStorage/OneDrive-Personal/1. Work/1 KR/LinkedIn/Intro in insurance/Intro into Insurance 2025 v04.docx"
GEMINI_API_KEY = "AIzaSyANoN0D34P_zbjM5j-jaTWiSy_rcpirxk4" 
# !!! END OF USER UPDATE SECTION !!!

OUTPUT_DATA_DIRECTORY = "preprocessed_insurance_data" # Folder to save output files (will be created inside this project)

# Model IDs
PREPROCESSING_PRO_MODEL_ID = "gemini-1.5-pro-latest"
EMBEDDING_MODEL_ID = "models/text-embedding-004"
CORE_SUBJECT = "Insurance Principles" 
#PREPROCESSING_PRO_MODEL_ID = "gemini-1.5-flash" # Use Flash for pre-processing for now

# Docling/Chunking Config
TOKENIZER_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MAX_TOKENS_PER_CHUNK_DOCLING = 150
MIN_WORDS_FOR_SUBSTANTIVE_CHUNK = 4

# FAISS/Question Generation Config
NUM_CONTEXT_CHUNKS_FOR_QUESTIONS = 3

# Pre-generated Questions Config
PREGEN_QS_MIN = 10
PREGEN_QS_MAX = 300
PREGEN_QS_PERCENT_OF_CHUNKS = 0.30

print(f"DEBUG: PATH_TO_YOUR_INSURANCE_DOCUMENT is set to: '{PATH_TO_YOUR_INSURANCE_DOCUMENT}'")
print(f"DEBUG: GEMINI_API_KEY ends with: '...{GEMINI_API_KEY[-4:] if GEMINI_API_KEY and len(GEMINI_API_KEY) >= 4 else 'NOT_SET_OR_TOO_SHORT'}'")
print("DEBUG: Checking placeholder values for critical constants...")

# Check if critical placeholder values have been updated
if PATH_TO_YOUR_INSURANCE_DOCUMENT.startswith("REPLACE_WITH_ACTUAL_PATH") or \
   GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE" or \
   not GEMINI_API_KEY:
    print("ERROR: Please update PATH_TO_YOUR_INSURANCE_DOCUMENT and GEMINI_API_KEY in the script before running.")
    exit()
print("DEBUG: Critical constants check passed.") # Add this line too


def process_document_with_docling(document_bytes, filename, tokenizer_model_id, max_tokens, min_words):
    print(f"--- Docling Processing: Starting for file: {filename} ---")
    # This list will store dictionaries with text, headings, and original index
    processed_chunks_intermediate = [] 
    
    start_time = time.time()
    try:
        buf = io.BytesIO(document_bytes)
        source = DocumentStream(name=filename, stream=buf) 
        print("--- Docling: Initializing DocumentConverter... ---")
        converter = DocumentConverter() 
        print("--- Docling: Converting document... ---")
        convert_result = converter.convert(source) 
        docling_doc_obj = convert_result.document
        
        if not docling_doc_obj:
            print("ERROR: Docling Processing: Failed to convert document.")
            return None, None # Return two Nones if conversion fails
        
        print(f"--- Docling: Document converted. Initial text elements found by converter: {len(docling_doc_obj.texts if hasattr(docling_doc_obj, 'texts') else 'N/A')} ---")
        print("--- Docling: Configuring Tokenizer for HybridChunker... ---")
        
        hf_tokenizer_instance = AutoTokenizer.from_pretrained(tokenizer_model_id)
        docling_tokenizer = HuggingFaceTokenizer(
            tokenizer=hf_tokenizer_instance,
            max_tokens=max_tokens
        )
        print(f"--- Docling: Initializing HybridChunker with max_tokens={max_tokens}, merge_peers=False ---")
        chunker = HybridChunker(tokenizer=docling_tokenizer, merge_peers=False)
        print("--- Docling: Starting HybridChunker process... ---")
        docling_chunk_iterator = chunker.chunk(docling_doc_obj)
        all_docling_chunks_from_hybridchunker = list(docling_chunk_iterator) 
        original_hybridchunker_count = len(all_docling_chunks_from_hybridchunker)
        print(f"--- Docling: HybridChunker produced {original_hybridchunker_count} initial chunks. Filtering... ---")
        
        for i, chunk_obj in enumerate(all_docling_chunks_from_hybridchunker):
            text = chunk_obj.text.strip() if hasattr(chunk_obj, 'text') else ""
            meta = chunk_obj.meta if hasattr(chunk_obj, 'meta') else None
            headings = meta.headings if meta and hasattr(meta, 'headings') and meta.headings else []
            words = text.split()
            num_words = len(words)
            # Filter for chunks that have headings and meet min word count
            if headings and num_words >= min_words: 
                processed_chunks_intermediate.append({
                    "text": text,
                    "headings": headings, # This is the full_headings_list
                    # "original_docling_chunk_index": i # We might not need this for the new app
                })
        
        final_substantive_chunk_count = len(processed_chunks_intermediate)
        processing_time = time.time() - start_time
        print(f"--- Docling Processing: Original HybridChunker chunks: {original_hybridchunker_count}. Final substantive chunks: {final_substantive_chunk_count}. Time: {processing_time:.2f}s. ---")
        
        if not processed_chunks_intermediate:
            print("WARNING: Docling processed the document, but no substantive chunks with headings were extracted after filtering.")
            return [], [] # Return empty lists

        # Prepare the two lists for output, as planned
        substantive_chunks_text_list = [item["text"] for item in processed_chunks_intermediate]
        # doc_chunk_details_list will contain both text and headings for later use (e.g., heatmap)
        doc_chunk_details_list = [{"text": item["text"], "full_headings_list": item["headings"]} for item in processed_chunks_intermediate]
        
        return substantive_chunks_text_list, doc_chunk_details_list

    except Exception as e:
        processing_time = time.time() - start_time
        print(f"ERROR in Docling Processing after {processing_time:.2f}s: {e}")
        print(f"Error type: {type(e).__name__}")
        traceback.print_exc()
        return None, None # Return two Nones on error
    
def determine_document_theme(sampled_chunks, pro_llm_instance): # Changed llm_model to pro_llm_instance
    if not sampled_chunks:
        print("--- Theme Determination: No chunks provided to determine theme. ---")
        # Using CORE_SUBJECT from global constants, ensure it's defined if you want a fallback here
        return CORE_SUBJECT, "To understand general concepts from the document." 
    print(f"--- Theme Determination: Analyzing {len(sampled_chunks)} sampled chunks. ---")
    combined_sample_text = ""
    char_limit_for_theme_prompt = 6000 
    for chunk in sampled_chunks:
        if len(combined_sample_text) + len(chunk) + 4 < char_limit_for_theme_prompt: 
            combined_sample_text += chunk + "\n---\n"
        else: break 
    if not combined_sample_text: 
        print("--- Theme Determination: Combined sample text is empty. Using fallback. ---")
        return CORE_SUBJECT, "To learn about the provided content."
    print(f"--- Theme Determination: Sending combined sample (approx {len(combined_sample_text)} chars) to LLM. ---")
    prompt = f"""
    Analyze the following text excerpts from a document. Your goal is to identify its main theme.
    1.  Identify the primary core subject of this document. Be concise and specific (e.g., "Principles of Insurance," "Risk Management in Software Projects," "Introduction to Astrophysics"). Aim for 3-7 words.
    2.  Identify the primary learning objective or purpose of this document from a reader's perspective (e.g., "To understand key components of reinsurance treaties," "To learn how to apply agile methodologies," "To explain the life cycle of stars"). Start with "To..."
    Text Excerpts:\n---\n{combined_sample_text}\n---\n
    Provide your answer in the following exact format, with each item on a new line:
    Core Subject: [Identified core subject here]
    Primary Objective: [Identified primary objective here]
    """
    try:
        response = pro_llm_instance.generate_content(prompt, request_options={'timeout': 90}) # Use pro_llm_instance
        if response and response.text:
            response_text = response.text.strip()
            print(f"--- Theme Determination LLM Raw Response: ---\n{response_text}\n----------------------------------------")
            core_subject_match = re.search(r"Core Subject:\s*(.+)", response_text, re.IGNORECASE)
            primary_objective_match = re.search(r"Primary Objective:\s*(To .+)", response_text, re.IGNORECASE) 
            determined_subject = core_subject_match.group(1).strip() if core_subject_match else None
            determined_objective = primary_objective_match.group(1).strip() if primary_objective_match else None
            if determined_subject and determined_objective:
                print(f"--- Theme Determined: Subject='{determined_subject}', Objective='{determined_objective}' ---")
                return determined_subject, determined_objective
            else:
                print(f"WARNING: Theme Determination: Could not parse subject/objective from LLM response. Core Subject Match: {core_subject_match}, Objective Match: {primary_objective_match} ---")
                subject_fallback = CORE_SUBJECT 
                objective_fallback = "To learn about the content of the uploaded document."
                if determined_subject: 
                    subject_fallback = determined_subject
                    objective_fallback = f"To understand key aspects of {determined_subject}."
                return subject_fallback, objective_fallback
        else:
            print("WARNING: Theme Determination: LLM response was empty or invalid. ---")
            return CORE_SUBJECT, "To learn about the content of the uploaded document."
    except Exception as e:
        print(f"ERROR during theme determination LLM call: {type(e).__name__}: {e} ---")
        traceback.print_exc()
        return CORE_SUBJECT, "To analyze the provided document."
    
def generate_embeddings_and_faiss_index(text_chunks_list, embedding_model_id):
    if not text_chunks_list:
        print("WARNING: Embedding Generation: No text chunks provided to generate embeddings.")
        return None
    
    print(f"--- Embedding Generation: Starting for {len(text_chunks_list)} chunks using model {embedding_model_id} ---")
    all_embeddings_list = []
    batch_size = 50 # You can adjust this batch size if needed
    num_batches = (len(text_chunks_list) + batch_size - 1) // batch_size
    
    print(f"--- Generating embeddings in {num_batches} batches of size {batch_size} ---")
    try:
        for i in range(num_batches):
            start_index = i * batch_size
            end_index = min((i + 1) * batch_size, len(text_chunks_list))
            batch_texts = text_chunks_list[start_index:end_index]
            
            if not batch_texts:
                continue
            
            print(f"Processing batch {i+1}/{num_batches}...")
            response = genai.embed_content(
                model=embedding_model_id,
                content=batch_texts,
                task_type="RETRIEVAL_DOCUMENT"
            )
            batch_embeddings = response['embedding']
            all_embeddings_list.extend(batch_embeddings)
            time.sleep(0.1) # Small delay to be kind to the API

        if not all_embeddings_list or len(all_embeddings_list) != len(text_chunks_list):
            print("ERROR: Embedding generation failed or produced incorrect number of embeddings.")
            return None
            
        embeddings_np = np.array(all_embeddings_list).astype('float32')
        dimension = embeddings_np.shape[1]
        print(f"--- Embeddings generated. Shape: {embeddings_np.shape}. ---")
        
        print("--- Building FAISS index (IndexFlatL2)... ---")
        faiss_index = faiss.IndexFlatL2(dimension)
        faiss_index.add(embeddings_np)
        print(f"--- FAISS index built. Total vectors in index: {faiss_index.ntotal}. ---")
        return faiss_index

    except Exception as e:
        print(f"ERROR during embedding generation or FAISS index creation: {e}")
        traceback.print_exc()
        return None
    
def extract_with_pattern(key, pattern, text_to_search):
    # This version is based on the refinements from message #107
    # to address issues like "25million" and "word1word2"
    flags = re.IGNORECASE

    match = re.search(pattern, text_to_search, flags)
    if match: 
        content = match.group(1).strip()
        
        # --- Apply formatting to ALL extracted textual content ---
        
        # 1. Add space after a period if followed by a letter/number
        content = re.sub(r'\.(?=[a-zA-Z0-9])', '. ', content) 
        
        # 2. Add space between a letter and a number (e.g., "Version3" -> "Version 3")
        content = re.sub(r'([a-zA-Z])([0-9])', r'\1 \2', content) 
        # 3. Add space between a number and a letter (e.g., "2million" -> "2 million")
        content = re.sub(r'([0-9])([a-zA-Z])', r'\1 \2', content)
        
        # 4. Add space before an opening parenthesis if preceded by alphanumeric
        content = re.sub(r'([a-zA-Z0-9])(\()', r'\1 \2', content) 
        # 5. Add space after a closing parenthesis if followed by alphanumeric
        content = re.sub(r'(\))([a-zA-Z0-9])', r'\1 \2', content) 
        
        common_stuck_words = [
            'and', 'of', 'in', 'to', 'for', 'with', 'on', 'at', 'from', 'by', 'about', 'as', 'is', 'it', 'the', 'a',
            'million', 'thousand', 'hundred', 'billion', 'trillion',
            'dollar', 'dollars', 'euro', 'euros', 'yen', 'pound', 'pounds',
            'usd', 'eur', 'gbp', 'jpy',
            'january', 'february', 'march', 'april', 'may', 'june', 'july', 
            'august', 'september', 'october', 'november', 'december',
            'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
            'policy', 'insurance', 'claim', 'risk', 'premium', 'treaty', 'actuary', 'cedent',
            'annually', 'during', 'year', 'they', 'experience', 'losses', 'loss', 'totaling', 
            'aggregate', 'retention', 'how', 'much', 'if', 'any', 'will', 'reinsurer', 'pay',
            'what', 'main', 'purpose', 'company' 
        ]
        
        for word_to_space in common_stuck_words:
            # Case 1: Common word stuck to a following alphanumeric character (e.g., "wordText" -> "word Text")
            pattern_stuck_after = rf'(\b{re.escape(word_to_space)}\b)([a-zA-Z0-9])'
            content = re.sub(pattern_stuck_after, rf'\1 \2', content, flags=re.IGNORECASE)

            # Case 2: Common word stuck to a preceding alphanumeric character (e.g., "textWord" -> "text Word")
            pattern_stuck_before = rf'([a-zA-Z0-9])(\b{re.escape(word_to_space)}\b)'
            content = re.sub(pattern_stuck_before, rf'\1 \2', content, flags=re.IGNORECASE)
        
        # Consolidate multiple spaces into a single space (run this last)
        content = re.sub(r'\s{2,}', ' ', content).strip() 
        
        return content
    
    print(f"--- Parsing Warning: Could not find '{key}' in response using pattern: {pattern} ---")
    return None

def generate_single_quiz_question_with_pro_model(pro_llm_instance, context_chunks_list, subject, objective):
    if not pro_llm_instance: 
        print("ERROR: Pro LLM instance not configured for question generation.")
        return None
    if not context_chunks_list:
        print("ERROR: No context chunks provided for question generation.")
        return None

    context_to_send = "\n\n---\n\n".join(context_chunks_list)
    max_context_chars = 8000 # Keep consistent with your existing app logic if desired
    if len(context_to_send) > max_context_chars:
        context_to_send = context_to_send[:max_context_chars] + "..."

    # This is inside your generate_single_quiz_question_with_pro_model function

    # This prompt goes into your generate_single_quiz_question_with_pro_model function
    # 'subject' and 'objective' come from the earlier theme determination step
    
    prompt = f"""
    You are an expert quiz generator. Your sole mission is to create ONE high-quality, self-contained multiple-choice question testing substantive understanding of **insurance principles, concepts, mechanisms, roles, or processes**. Base your question EXCLUSIVELY on the 'Provided Text Context' below.
    The overall subject of the source document is: '{subject}'.
    The primary learning objective is: '{objective}'.

    **ABSOLUTE CRITICAL INSTRUCTIONS - ADHERE STRICTLY OR STATE 'CANNOT_GENERATE':**

    1.  **TEST SUBSTANTIVE INSURANCE KNOWLEDGE:** The question MUST assess understanding of actual insurance subject matter (definitions, processes, relationships, calculations if present, policy elements, etc.) explicitly explained or directly implied *within the 'Provided Text Context'*.

    2.  **STRICTLY SELF-CONTAINED WITHIN THE PROVIDED TEXT:**
        * The question, ALL options, and the correct answer MUST be derivable *solely and entirely* from the information present in the 'Provided Text Context'. No outside knowledge is needed.

    3.  **DO NOT CREATE META-QUESTIONS ABOUT THE DOCUMENT ITSELF:**
        * **IGNORE & DO NOT USE any sentences in the 'Provided Text Context' that describe the document itself, its aims, its structure, its authorial choices (e.g., what is included/excluded for discussion *in the document as a whole*), or what the document states it will discuss later or has discussed previously.**
        * If a sentence says "This guide aims to..." or "Later, we will see..." or "This section excludes...", you MUST NOT form a question about that statement. Base questions only on the *insurance content being explained*.

    4.  **DO NOT CREATE QUESTIONS ABOUT FIGURES/TABLES REFERENCED BUT NOT SHOWN:**
        * If the 'Provided Text Context' *mentions or describes* a Figure or Table (e.g., "Figure 1 shows X," or "As seen in the table below...") but the Figure or Table content itself is NOT part of the 'Provided Text Context', you MUST NOT ask a question that relies on seeing, interpreting, or knowing the content of that Figure/Table.
        * Generate a question based *only* on other textual explanations in the snippet that are fully understandable without the visual aid.
        * If the *entire usable substance* of the snippet is only a caption or a description of an absent visual aid, you MUST output 'CANNOT_GENERATE'.

    5.  **AVOID TRIVIAL "WHAT IS MENTIONED/LISTED" QUESTIONS:**
        * Do not ask "Which of these is mentioned?" or "What entities are listed?" if the answer only requires spotting words from a list within the context without testing any deeper understanding of those terms, their roles, definitions, or relationships *as explained in the snippet*.
        * Prefer questions about the *function, definition, or characteristics* of items if they are actually detailed in the snippet.

    6.  **ENSURE FAIRNESS FOR THE SNIPPET'S SCOPE:**
        * If asking about a category or a set of items, the question must be answerable comprehensively and fairly *from the information within the snippet*. Do not ask questions that imply the snippet is exhaustive if it clearly isn't.

    7.  **APPLICATION & NUMERICAL QUESTIONS (HIGHLY PREFERRED IF CONTEXT ALLOWS):**
        * If the 'Provided Text Context' contains specific numerical data, formulas, scenarios (e.g., premium calculations, loss ratios with figures, pro-rata examples, financial limits), **STRONGLY PREFER** to formulate a question that requires applying this information, performing a calculation, or interpreting the scenario.

    8.  **PLAUSIBLE OPTIONS & SINGLE CORRECT ANSWER:** Generate 4 distinct options (A, B, C, D). Incorrect options must be plausible and related to insurance but definitively wrong according to the 'Provided Text Context'. There must be only ONE best correct answer.

    **Output Format (EXACTLY as specified below. Do not add any other text, apologies, or conversational fluff before or after this block):**
    Question: [Your question here]
    A: [Option A text]
    B: [Option B text]
    C: [Option C text]
    D: [Option D text]
    Correct Answer: [Letter ONLY, e.g., C]
    Explanation: [Brief explanation, strictly based on the 'Provided Text Context', detailing why the answer is correct and why others are not, according to the snippet. Do not refer to figures not present in the context.]
    RelevanceScore: [Integer 1-5. 5 = question is highly central to understanding key insurance concepts in the snippet. 1 = peripheral detail.]
    Difficulty: [Easy, Medium, or Hard, based on snippet complexity.]
    Keywords: [2-4 comma-separated keywords for this question's topic from the snippet.]

    Provided Text Context:
    ---
    {context_to_send}
    ---
    Generate the question and its associated metadata now. If, after carefully considering the 'Provided Text Context' and all **ABSOLUTE CRITICAL INSTRUCTIONS**, you determine that a high-quality, compliant question cannot be formed, output only the single word: CANNOT_GENERATE
    """

    response_text = None
    llm_response_obj = None # For potential safety feedback access
    max_retries = 2 # Fewer retries for pre-processing might be acceptable
    retry_delay = 3

    try:
        for attempt in range(max_retries):
            try:
                print(f"--- Sending prompt to Pro LLM for question generation (Attempt {attempt + 1}/{max_retries}) ---")
                # Define safety settings as in your main app
                safety_settings = { 
                    genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                }
                llm_response_obj = pro_llm_instance.generate_content(prompt, safety_settings=safety_settings, request_options={'timeout': 120}) # Longer timeout for Pro model
                
                if llm_response_obj and llm_response_obj.candidates and hasattr(llm_response_obj.candidates[0].content, 'parts') and llm_response_obj.candidates[0].content.parts:
                    response_text = llm_response_obj.candidates[0].content.parts[0].text.strip()
                    if response_text:
                        print(f"--- Pro LLM responded (Attempt {attempt + 1}). ---")
                        break
                
                # Handle cases where response might be empty or blocked
                reason = "Unknown reason"
                if llm_response_obj:
                    if not llm_response_obj.candidates or not hasattr(llm_response_obj.candidates[0].content, 'parts') or not llm_response_obj.candidates[0].content.parts:
                        reason = llm_response_obj.candidates[0].finish_reason.name if llm_response_obj.candidates and llm_response_obj.candidates[0].finish_reason else "Empty content part"
                    if llm_response_obj.prompt_feedback and llm_response_obj.prompt_feedback.block_reason:
                        reason = f"Blocked - {llm_response_obj.prompt_feedback.block_reason.name}"
                print(f"--- Pro LLM Response Text Empty/Blocked (Attempt {attempt + 1}). Reason: {reason} ---")
                if attempt < max_retries - 1:
                    print(f"--- Retrying Pro LLM call in {retry_delay}s... ---")
                    time.sleep(retry_delay)
                else:
                    print(f"ERROR: Pro LLM response issue after {max_retries} attempts. Reason: {reason}")
                    return None
            except Exception as e_api:
                print(f"--- Pro LLM API Error (Attempt {attempt + 1}/{max_retries}): {type(e_api).__name__}: {e_api} ---")
                if attempt < max_retries - 1:
                    print(f"--- Retrying Pro LLM API call in {retry_delay}s... ---")
                    time.sleep(retry_delay)
                else:
                    raise # Re-raise the last exception if all retries fail

        if not response_text:
            print("ERROR: Failed to get valid response text from Pro LLM after retries.")
            return None
            
        # Define patterns for parsing (same as in your app.py)
        patterns = {
            "question": r"^\**Question\**\s*[:\-]\s*(.+?)\s*(?=\n\s*\**\s*[A-Z]\s*[:\.\)]|\Z)",
            "A": r"\n\s*\**\s*[Aa]\s*[:\.\)]\s*\**(.+?)\**\s*(?=\n\s*\**\s*[Bb]\s*[:\.\)]|\Z)",
            "B": r"\n\s*\**\s*[Bb]\s*[:\.\)]\s*\**(.+?)\**\s*(?=\n\s*\**\s*[Cc]\s*[:\.\)]|\Z)",
            "C": r"\n\s*\**\s*[Cc]\s*[:\.\)]\s*\**(.+?)\**\s*(?=\n\s*\**\s*[Dd]\s*[:\.\)]|\Z)",
            "D": r"\n\s*\**\s*[Dd]\s*[:\.\)]\s*\**(.+?)\**\s*(?=\n\s*\**\s*Correct Answer\s*[:\.\)]|\Z)",
            "correct_answer": r"\n\s*\**\s*Correct Answer\s*[:\.\)]\s*\**\s*\[?([A-Da-d])\]?\s*\**",
            "explanation": r"\n\s*\**\s*Explanation\s*[:\.\)]\s*\**([\s\S]+?)\**\s*(\Z|\n\s*\**\s*(Question:|A:|B:|C:|D:|Correct Answer:))",
            "relevance_score": r"\n\s*\**\s*RelevanceScore\s*[:\-]\s*([1-5])\s*\**", # Expects a single digit 1-5
            "difficulty": r"\n\s*\**\s*Difficulty\s*[:\-]\s*(Easy|Medium|Hard)\s*\**", # Expects Easy, Medium, or Hard
            "keywords": r"\n\s*\**\s*Keywords\s*[:\-]\s*(.+?)\s*\**(?=\n\s*\**\s*(Question:|A:|B:|C:|D:|Correct Answer:|Explanation:|RelevanceScore:|Difficulty:)|(\Z))" # Adjusted lookahead
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
        
        relevance_score_raw = extract_with_pattern("RelevanceScore", patterns["relevance_score"], response_text)
        if relevance_score_raw and relevance_score_raw.isdigit() and 1 <= int(relevance_score_raw) <= 5:
            parsed_data["relevance_score"] = int(relevance_score_raw)
        else:
            parsed_data["relevance_score"] = None 
            if relevance_score_raw is not None: 
                print(f"--- Parsing Warning: Invalid RelevanceScore value '{relevance_score_raw}'. Expected 1-5. Defaulting to None. ---")    
        parsed_data["difficulty"] = extract_with_pattern("Difficulty", patterns["difficulty"], response_text)
    
        keywords_raw = extract_with_pattern("Keywords", patterns["keywords"], response_text)
        if keywords_raw:
            parsed_data["keywords"] = [kw.strip() for kw in keywords_raw.split(',') if kw.strip()]
        else:
            parsed_data["keywords"] = []

        required_keys = ["question", "options", "correct_answer", "explanation", "relevance_score", "difficulty", "keywords"]
        if not all(parsed_data.get(k) for k in required_keys) or len(parsed_data.get("options", {})) != 4:
            print(f"--- ERROR: Parsing failed for Pro LLM response. Data: {parsed_data}. Options count: {len(parsed_data.get('options', {}))}. Raw response: {response_text[:500]} ---")
            return None # Indicate failure
        if parsed_data["correct_answer"] not in ["A", "B", "C", "D"]:
            print(f"--- ERROR: Invalid correct answer from Pro LLM: '{parsed_data['correct_answer']}'. Raw response: {response_text[:500]} ---")
            return None

        print("--- Successfully parsed question data from Pro LLM. ---")
        return parsed_data

    except Exception as e_overall:
        print(f"--- ERROR during Pro LLM question generation: {type(e_overall).__name__}: {e_overall} ---")
        traceback.print_exc()
        return None
    
def main_preprocess():
    print("DEBUG: main_preprocess() function entered.") # ADD THIS
    print("--- Starting Document Pre-processing Script ---")

    # 0. Configure Gemini API
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        pro_model_instance = genai.GenerativeModel(PREPROCESSING_PRO_MODEL_ID)
        print(f"Successfully configured Gemini API and initialized Pro Model: {PREPROCESSING_PRO_MODEL_ID}")
    except Exception as e:
        print(f"ERROR: Failed to configure Gemini API or initialize Pro Model: {e}")
        traceback.print_exc()
        return

    # 1. Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DATA_DIRECTORY, exist_ok=True)
    print(f"Output directory '{OUTPUT_DATA_DIRECTORY}' ensured.")

    # 2. Load Document
    try:
        print(f"Loading document from: {PATH_TO_YOUR_INSURANCE_DOCUMENT}")
        with open(PATH_TO_YOUR_INSURANCE_DOCUMENT, "rb") as f:
            document_bytes = f.read()
        original_doc_filename = os.path.basename(PATH_TO_YOUR_INSURANCE_DOCUMENT)
    except FileNotFoundError:
        print(f"ERROR: Document not found at {PATH_TO_YOUR_INSURANCE_DOCUMENT}. Please check the path.")
        return
    except Exception as e:
        print(f"ERROR: Could not read document file: {e}")
        traceback.print_exc()
        return

    # Copy original document to output directory
    try:
        target_doc_path = os.path.join(OUTPUT_DATA_DIRECTORY, original_doc_filename)
        shutil.copy2(PATH_TO_YOUR_INSURANCE_DOCUMENT, target_doc_path)
        print(f"Copied original document to: {target_doc_path}")
    except Exception as e:
        print(f"ERROR: Could not copy original document to output directory: {e}")
        traceback.print_exc()
        # Continue processing if copy fails, but log warning
        print("WARNING: Original document could not be copied. Proceeding with processing...")


    # 3. Process with Docling
    print("\n--- Starting Document Processing with Docling ---")
    substantive_chunks_text_list, doc_chunk_details_list = process_document_with_docling(
        document_bytes,
        original_doc_filename,
        TOKENIZER_MODEL_ID,
        MAX_TOKENS_PER_CHUNK_DOCLING,
        MIN_WORDS_FOR_SUBSTANTIVE_CHUNK
    )
    if substantive_chunks_text_list is None or doc_chunk_details_list is None: # Check for None if function can return that on error
        print("ERROR: Docling processing failed to return valid chunk lists. Exiting.")
        return
    if not substantive_chunks_text_list:
        print("WARNING: No substantive text chunks extracted by Docling. Cannot proceed further.")
        return
    print(f"Docling processing yielded {len(substantive_chunks_text_list)} substantive chunks.")

    # 4. Determine Theme & Objective (using Pro model)
    print("\n--- Determining Document Theme & Objective (Pro Model) ---")
    sample_size_for_theme = min(8, len(substantive_chunks_text_list))
    sampled_chunks_for_theme = random.sample(substantive_chunks_text_list, sample_size_for_theme) if len(substantive_chunks_text_list) > sample_size_for_theme else substantive_chunks_text_list[:]
    
    core_subject, primary_objective = determine_document_theme(sampled_chunks_for_theme, pro_model_instance)
    document_metadata = {"core_subject": core_subject, "primary_objective": primary_objective}
    
    metadata_filepath = os.path.join(OUTPUT_DATA_DIRECTORY, "insurance_theme_objective.json")
    with open(metadata_filepath, "w") as f:
        json.dump(document_metadata, f, indent=4)
    print(f"Saved theme and objective to {metadata_filepath}")

    # 5. Generate Embeddings & Build FAISS Index
    print("\n--- Generating Embeddings & Building FAISS Index ---")
    faiss_index = generate_embeddings_and_faiss_index(substantive_chunks_text_list, EMBEDDING_MODEL_ID)
    if faiss_index is None:
        print("ERROR: FAISS index creation failed. Exiting.")
        return
        
    faiss_index_filepath = os.path.join(OUTPUT_DATA_DIRECTORY, "insurance_faiss.index")
    faiss.write_index(faiss_index, faiss_index_filepath)
    print(f"FAISS index saved to {faiss_index_filepath}")

    # Save the chunks that correspond to the FAISS index order (for easy lookup)
    faiss_chunks_filepath = os.path.join(OUTPUT_DATA_DIRECTORY, "insurance_faiss_chunks.json")
    with open(faiss_chunks_filepath, "w") as f:
        json.dump(substantive_chunks_text_list, f, indent=4)
    print(f"FAISS-indexed chunks saved to {faiss_chunks_filepath}")
    
    # Save doc_chunk_details (with text and headings) for heatmap use
    doc_details_filepath = os.path.join(OUTPUT_DATA_DIRECTORY, "insurance_doc_chunk_details.json")
    with open(doc_details_filepath, "w") as f:
        json.dump(doc_chunk_details_list, f, indent=4)
    print(f"Document chunk details (with text & headings) saved to {doc_details_filepath}")

    # 6. Pre-generate Quiz Questions (using Pro model)
    print("\n--- Pre-generating Quiz Questions (Pro Model) ---")
    total_chunks = len(substantive_chunks_text_list)
    num_questions_to_generate = int(total_chunks * PREGEN_QS_PERCENT_OF_CHUNKS)
    num_questions_to_generate = max(PREGEN_QS_MIN, min(num_questions_to_generate, PREGEN_QS_MAX))
    print(f"Calculated {num_questions_to_generate} questions to pre-generate (10% of {total_chunks} chunks, min {PREGEN_QS_MIN}, max {PREGEN_QS_MAX}).")

    pre_generated_questions = []
    if total_chunks == 0:
        print("WARNING: No chunks available to generate questions from.")
    else:
        # Create query for theme-relevant chunks
        theme_query_text = f"{core_subject}: {primary_objective}"
        print(f"Identifying theme-relevant chunks using query: '{theme_query_text}'")
        try:
            theme_query_embedding_response = genai.embed_content(model=EMBEDDING_MODEL_ID, content=theme_query_text, task_type="RETRIEVAL_QUERY")
            theme_query_embedding = np.array(theme_query_embedding_response['embedding']).astype('float32').reshape(1, -1)
            
            # Fetch more than needed initially to ensure we get enough unique ones if some are very similar
            num_to_fetch_for_theme = min(total_chunks, num_questions_to_generate * 2) # Fetch up to 2x needed
            distances, theme_relevant_indices_sorted = faiss_index.search(theme_query_embedding, k=num_to_fetch_for_theme)
            
            # Ensure unique indices, take top N
            focal_chunk_indices_for_questions = []
            seen_indices = set()
            for idx in theme_relevant_indices_sorted[0]:
                if idx not in seen_indices:
                    focal_chunk_indices_for_questions.append(idx)
                    seen_indices.add(idx)
                if len(focal_chunk_indices_for_questions) >= num_questions_to_generate:
                    break
            
            if not focal_chunk_indices_for_questions:
                 print("WARNING: Could not find any theme-relevant focal chunks using FAISS. Falling back to random selection.")
                 possible_focal_indices = list(range(total_chunks))
                 focal_chunk_indices_for_questions = random.sample(
                    possible_focal_indices, 
                    min(num_questions_to_generate, total_chunks)
                 )
            else:
                print(f"Selected {len(focal_chunk_indices_for_questions)} theme-relevant focal chunks for question generation.")

        except Exception as e_theme_search:
            print(f"ERROR during theme-based chunk selection: {e_theme_search}. Falling back to random selection.")
            traceback.print_exc()
            possible_focal_indices = list(range(total_chunks))
            focal_chunk_indices_for_questions = random.sample(
                possible_focal_indices, 
                min(num_questions_to_generate, total_chunks)
            )

        for i, focal_idx in enumerate(focal_chunk_indices_for_questions):
            print(f"Generating pre-generated question {i+1}/{len(focal_chunk_indices_for_questions)} based on focal chunk index {focal_idx}...")
            
            # Get context using FAISS around this focal chunk (Option C)
            try:
                # Reconstruct the embedding of the focal chunk to search for its neighbors
                # Ensure focal_idx is within bounds of the faiss_index
                if focal_idx < 0 or focal_idx >= faiss_index.ntotal:
                    print(f"WARNING: Focal index {focal_idx} out of bounds for FAISS index. Skipping question generation for this chunk.")
                    continue

                focal_chunk_embedding = faiss_index.reconstruct(int(focal_idx)).reshape(1, -1) # Ensure int for faiss
                distances, context_retrieved_indices_raw = faiss_index.search(focal_chunk_embedding, k=NUM_CONTEXT_CHUNKS_FOR_QUESTIONS)
                
                context_chunks_for_question = []
                actual_retrieved_indices_for_q = []
                for retrieved_idx in context_retrieved_indices_raw[0]:
                    # Ensure retrieved_idx is within the bounds of your substantive_chunks_text_list
                    if 0 <= retrieved_idx < len(substantive_chunks_text_list):
                        context_chunks_for_question.append(substantive_chunks_text_list[retrieved_idx])
                        actual_retrieved_indices_for_q.append(int(retrieved_idx))
                    if len(context_chunks_for_question) >= NUM_CONTEXT_CHUNKS_FOR_QUESTIONS:
                        break
                
                if not context_chunks_for_question:
                    print(f"WARNING: Could not retrieve sufficient context for focal chunk {focal_idx} using FAISS. Skipping.")
                    continue
            except Exception as e_context_faiss:
                print(f"ERROR retrieving context with FAISS for focal chunk {focal_idx}: {e_context_faiss}. Skipping.")
                traceback.print_exc()
                continue

            question_data = generate_single_quiz_question_with_pro_model(
                pro_model_instance, 
                context_chunks_for_question, 
                core_subject, 
                primary_objective
            )
            # ... (inside the for i, focal_idx ... loop, after question_data is assigned)

            # --- MODIFIED HANDLING OF question_data ---
            if question_data and isinstance(question_data, dict) and question_data.get("question") != "CANNOT_GENERATE":
                question_data["focal_chunk_index"] = int(focal_idx) 
                question_data["context_indices_used"] = actual_retrieved_indices_for_q # Ensure this var is defined from context gathering
                pre_generated_questions.append(question_data)
                print(f"Successfully generated and stored question {i+1} for focal chunk {focal_idx}.")
            elif question_data and isinstance(question_data, dict) and question_data.get("question") == "CANNOT_GENERATE":
                print(f"Skipping question for focal chunk {focal_idx}: LLM indicated valid question cannot be generated from this context.")
            else: # This means question_data itself was None (generation or parsing failed before returning "CANNOT_GENERATE")
                print(f"Failed entirely to generate or parse question {i+1} for focal chunk {focal_idx}. Output was: {question_data}")
            # --- END OF MODIFIED HANDLING ---

            # ... (your time.sleep() logic follows) ...
            # Respect API rate limits for the Pro model.
            # Adjust 'pro_model_wait_time' based on your model's observed or documented RPM.
            # Example: For 4 RPM (15s/request) or 2 RPM (30s/request)
            pro_model_wait_time = 15 # Start with 15 seconds, adjust if needed
            print(f"Question {i+1} attempt processed. Pausing for {pro_model_wait_time} seconds for Pro model rate limits...")
            time.sleep(pro_model_wait_time)

    questions_filepath = os.path.join(OUTPUT_DATA_DIRECTORY, "insurance_pre_generated_questions.json")
    with open(questions_filepath, "w") as f:
        json.dump(pre_generated_questions, f, indent=4)
    print(f"Saved {len(pre_generated_questions)} pre-generated questions to {questions_filepath}")

# This code is added at the end of the main_preprocess() function,
# just before the final "Pre-processing complete" print statement.

    # 7. Create a human-readable review file for all pre-generated questions
    if pre_generated_questions: # Check if any questions were actually generated
        review_filepath = os.path.join(OUTPUT_DATA_DIRECTORY, "insurance_questions_review.txt")
        print(f"\n--- Creating Human-Readable Review File: {review_filepath} ---")
        try:
            with open(review_filepath, "w", encoding="utf-8") as f_review:
                f_review.write(f"Review of Pre-Generated Questions for: {original_doc_filename}\n")
                f_review.write(f"Document Subject: {core_subject}\n")
                f_review.write(f"Document Objective: {primary_objective}\n")
                f_review.write(f"Total Questions Generated: {len(pre_generated_questions)}\n")
                f_review.write("=" * 50 + "\n\n")

                for i, q_data in enumerate(pre_generated_questions):
                    f_review.write(f"QUESTION #{i + 1}\n")
                    f_review.write("-" * 20 + "\n")
                    f_review.write(f"Focal Chunk Index: {q_data.get('focal_chunk_index', 'N/A')}\n")
                    # context_indices_used might be a list of integers
                    context_indices_str = ", ".join(map(str, q_data.get('context_indices_used', [])))
                    f_review.write(f"Context Indices Used: [{context_indices_str}]\n")
                    f_review.write(f"Relevance Score: {q_data.get('relevance_score', 'N/A')}\n")
                    f_review.write(f"Difficulty: {q_data.get('difficulty', 'N/A')}\n")
                    keywords_str = ", ".join(q_data.get('keywords', []))
                    f_review.write(f"Keywords: {keywords_str if keywords_str else 'N/A'}\n\n")

                    f_review.write(f"Question: {q_data.get('question', 'N/A')}\n")
                    options = q_data.get('options', {})
                    correct_answer_letter = q_data.get('correct_answer', '')

                    for opt_key in sorted(options.keys()): # A, B, C, D
                        indicator = " (Correct)" if opt_key == correct_answer_letter else ""
                        f_review.write(f"  {opt_key}: {options[opt_key]}{indicator}\n")

                    f_review.write(f"\nExplanation: {q_data.get('explanation', 'N/A')}\n")
                    f_review.write("=" * 50 + "\n\n")
            print(f"Successfully created review file: {review_filepath}")
        except Exception as e_review_file:
            print(f"ERROR: Could not write the review file: {e_review_file}")
            traceback.print_exc()
    else:
        print("--- No pre-generated questions to write to review file. ---")

# This should be the existing final print statement of main_preprocess()
print("\n--- Pre-processing complete! All data saved. ---")

if __name__ == "__main__":
    print("DEBUG: Script is being run directly, about to call main_preprocess().") # ADD THIS
    main_preprocess()
    
        