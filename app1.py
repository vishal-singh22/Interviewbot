import streamlit as st
import requests
from dotenv import load_dotenv
import os
import json

load_dotenv()

# API setup
HF_API_TOKEN = os.getenv("HF_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HF_API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"

hf_headers = {
    "Authorization": f"Bearer {HF_API_TOKEN}"
}


def generate_with_gemini(prompt):
    """Generate response using Gemini API"""
    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }]
    }

    response = requests.post(GEMINI_API_URL, json=payload)

    if response.status_code == 200:
        result = response.json()
        if 'candidates' in result and len(result['candidates']) > 0:
            return result['candidates'][0]['content']['parts'][0]['text']
        else:
            raise Exception("No response from Gemini API")
    else:
        raise Exception(f"Gemini API error: {response.status_code} - {response.text}")


def generate_with_huggingface(prompt):
    """Generate response using Hugging Face API"""
    payload = {"inputs": prompt, "options": {"wait_for_model": True}}
    response = requests.post(HF_API_URL, headers=hf_headers, json=payload)

    if response.status_code == 200:
        output = response.json()
        if isinstance(output, list):
            return output[0]['generated_text']
        elif isinstance(output, dict) and 'generated_text' in output:
            return output['generated_text']
        else:
            return str(output)
    else:
        raise Exception(f"Hugging Face API error: {response.status_code} - {response.text}")


# Query APIs with fallback mechanism
def generate_interview_test(jd, company_type, level, num_mcq, include_coding, num_short_ans):
    coding_section = ""
    if include_coding:
        coding_section = f"{1 if level == '0-1' else 2} Coding Question(s) (easy to hard) with problem description and expected output.\n"

    prompt = f"""
    Do not include job description or repeat inputs in output. response in form of Json.

You are an expert AI interviewer specializing in technical hiring for software and AI/ML roles.

Given the following inputs:

Job Description: {jd}  
Candidate Experience Level: {level}  
Industry Domain: {company_type}  
Number of MCQs: {num_mcq}  
Number of Short Answer Questions: {num_short_ans}  
Include Coding Section: {coding_section if coding_section else "No"}

Generate a comprehensive technical interview test as per the following structure:

1. **Multiple Choice Questions (MCQs)**  
   - Provide {num_mcq} MCQs relevant to the role.  
   - Each question should have **4 options** (A, B, C, D).  
   - **Clearly indicate the correct option** and provide a **1–2 line explanation**.

2. **Coding Challenge**  
   {coding_section if coding_section else "Skip this section if not required."}  
   - If included, make the problem aligned to the JD's core skills.
   - Provide sample input/output and a brief outline of the expected approach.

3. **Short Answer Questions**  
   - Provide {num_short_ans} questions.  
   - Each answer should be 3–5 lines long, with technically sound and concise explanations.

**Important Instructions:**  
- Do **not** include the job description again in the output.  
- Structure the output with **clear section headings** (MCQs, Coding, Short Answers).  
- The difficulty and focus should match the candidate's experience level and the domain.

Do not include job description or repeat inputs. Structure your response using markdown in JSON.

Return only the test.

"""

    # Try Hugging Face first, then fallback to Gemini
    try:
        if HF_API_TOKEN:
            return generate_with_huggingface(prompt)
        else:
            raise Exception("Hugging Face API token not found")
    except Exception as hf_error:
        try:
            if GEMINI_API_KEY:
                return generate_with_gemini(prompt)
            else:
                raise Exception("Gemini API key not found")
        except Exception as gemini_error:
            raise Exception(f"Both APIs failed. HF: {str(hf_error)}, Gemini: {str(gemini_error)}")


# Streamlit UI
st.set_page_config(page_title="AI Interview Test Generator", layout="centered")
st.title("🤖 AI-Powered Interview Test Generator")

# Inputs
company_type = st.selectbox("Select Company Type", [
    "Finance", "Healthcare", "Information Technology (IT) / Software", "Manufacturing", "Retail & E-commerce"
])
level = st.selectbox("Experience Level", ["0-1", "2-5", "More than 5"])

jd_input = st.text_area("Paste Job Description Here", height=300)

st.markdown("---")
st.subheader("Customize Your Test")

num_mcq = st.slider("Number of MCQs", 1, 10, 5)
include_coding = st.checkbox("Include Coding Questions?", value=True)
num_short_ans = st.slider("Number of Short Answer Questions", 1, 5, 2)

# Generate Test
if st.button("🚀 Generate Interview Test"):
    if jd_input.strip() == "":
        st.warning("Please enter a valid Job Description.")
    elif not HF_API_TOKEN and not GEMINI_API_KEY:
        st.error("❌ No API keys configured. Please set either HF_API_KEY or GEMINI_API_KEY in your environment.")
    else:
        with st.spinner("Generating test... Please wait..."):
            try:
                result = generate_interview_test(jd_input, company_type, level, num_mcq, include_coding, num_short_ans)
                st.success("✅ Interview test generated successfully!")
                st.markdown("### 📝 Test Output")
                st.code(result, language="markdown")
            except Exception as e:
                st.error(f"❌ Error: {e}")