import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline, RobertaForQuestionAnswering, RobertaTokenizer, Trainer, TrainingArguments
from datasets import Dataset

# ACM GIKI Chapter Data
data = ''' 
Association for Computing Machinery (ACM) GIKI Chapter
Website: www.acmgiki.com

About GIK Institute
The Ghulam Ishaq Khan Institute of Engineering Sciences and Technology is one of Pakistan's most prestigious engineering institutes. It is dedicated to promoting excellence in engineering, sciences, emerging technologies, and other disciplines, serving as a center of change in the country. GIKI has set the standard for brilliance in engineering by producing graduates working in some of the most sought-after national and international organizations.

About ACM GIKI Chapter
The Association for Computing Machinery (ACM) is a worldwide professional 
organization focused on advancing computer science theory and practice.

ACM GIKI Activities
Student workshops
Specialized courses
Introductory seminars
Software and computer game competitions

Message from the President
"Welcome to ACM GIKI! We are a community of passionate computer scientists and engineers working to build a better future. We offer events and activities to help you learn, grow, and connect with others. Make the most of your time at GIKI!"
- President ACM
Our Events
Softcom
ICPC Pakistan
Hackathons
Workshops
Outreach Programs

Softcom
An annual nationwide competition organized by GIKI since 2000. It includes:
Software competitions
Multimedia presentations
Quizzes
Speed programming

Event Highlights
Hackathon: Collaborative project introductions.
Speed Programming: Solve algorithmic problems efficiently.
Game Development: Hands-on sessions for beginners.

ICPC Pakistan
The ACM International Collegiate Programming Contest (ICPC) is the oldest and most prestigious programming competition in the world. It is often referred to as the "Olympics of Programming Competitions."

Outreach Program
Mission: To spread awareness, inspire youth, and empower underprivileged students through education and mentorship.

Key Initiatives
Career Counseling: Discussions on tech roles like AI and data science.
Motivational Speakers: Personal stories and encouragement to pursue education.

Sponsorship Packages
Available Packages
Platinum: PKR 600,000
Gold : PKR 400,000
Silver: PKR 200,000
Bronze: PKR 100,000

Contact Us
President
Name: Anas Raza Aslam
Phone: +92 336 7297360
Email : anasraza.me@gmail.com
Event Coordinator
Name: Ali Iftikhar
Phone: +92 311 1721609
Email : mianali5451@gmail.com
General Inquiries
Email : acm@giki.edu.pk
Instagram: @acm.giki
'''

# Initialize text splitter and embeddings
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_text(data)
pages = [Document(page_content=chunk) for chunk in chunks]

# Build FAISS index
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
faiss_index = FAISS.from_documents(pages, embedding_model)

# Use Hugging Face's question-answering pipeline
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

# Keywords to answer various types of questions based on the full data
keywords_to_answer = {
    "president": "Anas Raza Aslam is the President of ACM GIKI. He warmly welcomes everyone to ACM GIKI and encourages students to take full advantage of the opportunities provided by the chapter.",
    "event coordinator": "Ali Iftikhar is the Event Coordinator at ACM GIKI.",
    "contact": "You can reach us at acm@giki.edu.pk for general inquiries.",
    "sponsorship": "Our sponsorship packages are: Platinum: PKR 600,000, Gold: PKR 400,000, Silver: PKR 200,000, Bronze: PKR 100,000.",
    "softcom": "Softcom is an annual nationwide competition organized by GIKI since 2000. It includes Software competitions, Multimedia presentations, Quizzes, and Speed programming.",
    "icpc": "ICPC (International Collegiate Programming Contest) is the oldest and most prestigious programming competition, often referred to as the 'Olympics of Programming Competitions'.",
    "hackathon": "Hackathon: Collaborative project introductions, Speed Programming: Solve algorithmic problems efficiently, and Game Development: Hands-on sessions for beginners.",
    "outreach": "The Outreach Program at ACM GIKI aims to spread awareness, inspire youth, and empower underprivileged students through education and mentorship. Key initiatives include Career Counseling and Motivational Speakers to encourage students to pursue education.",
    "workshops": "ACM GIKI organizes student workshops, specialized courses, and introductory seminars.",
    "giki": "Ghulam Ishaq Khan Institute (GIKI) is one of Pakistan's most prestigious engineering institutes.",
    "website": "Visit our website at www.acmgiki.com for more details.",
    "president_message": "The President of ACM GIKI, Anas Raza Aslam, welcomes all students and members to join ACM GIKI and encourages them to participate in various activities for personal and professional growth. He emphasizes the importance of collaboration and knowledge-sharing in the tech community."
}

# Function to answer a user question
def answer_question(user_question):
    # Check if question contains a keyword
    for keyword, answer in keywords_to_answer.items():
        if keyword.lower() in user_question.lower():
            return answer
    
    # If no keyword found, perform similarity search to find the relevant documents
    docs = faiss_index.similarity_search(user_question, k=5)
    
    # Combine the text from the relevant documents
    context = "\n".join([doc.page_content for doc in docs])

    # Use the QA pipeline to extract the answer
    answer = qa_pipeline(question=user_question, context=context)
    
    return answer['answer']

# Collect user feedback
def collect_feedback():
    feedback = st.radio("Was this answer helpful?", ("Yes", "No"))
    if feedback == "No":
        user_feedback = st.text_area("Please provide more details about your question or the answer.")
        # You can store this feedback and use it for further model training or improvement
        st.write("Thanks for your feedback!")

# Fine-Tune the QA Model
def fine_tune_model():
    # Prepare dataset with question-answer pairs related to ACM GIKI
    data = [
        {"context": "Anas Raza Aslam is the President of ACM GIKI.", "question": "Who is the president of ACM GIKI?", "answer": "Anas Raza Aslam"},
        {"context": "Ali Iftikhar is the Event Coordinator at ACM GIKI.", "question": "Who is the Event Coordinator?", "answer": "Ali Iftikhar"},
        {"context": "Softcom is an annual nationwide competition organized by GIKI since 2000.", "question": "What is Softcom?", "answer": "Softcom is an annual nationwide competition organized by GIKI."},
        {"context": "The ACM International Collegiate Programming Contest (ICPC) is the oldest and most prestigious programming competition.", "question": "What is ICPC?", "answer": "ICPC is the oldest and most prestigious programming competition."},
        {"context": "Hackathon: Collaborative project introductions.", "question": "What is Hackathon?", "answer": "Hackathon is a collaborative project introduction event."},
        {"context": "Outreach Program: Spreading awareness and empowering underprivileged students.", "question": "What is the Outreach Program?", "answer": "The Outreach Program spreads awareness and empowers underprivileged students."},
        {"context": "Sponsorship Packages: Platinum, Gold, Silver, and Bronze.", "question": "What are the sponsorship packages?", "answer": "Sponsorship packages are Platinum, Gold, Silver, and Bronze."},
        {"context": "GIKI is one of Pakistan's most prestigious engineering institutes.", "question": "What is GIKI?", "answer": "GIKI is one of Pakistan's most prestigious engineering institutes."},
        {"context": "Anas Raza Aslam, the President of ACM GIKI, welcomes students and encourages them to take full advantage of ACM GIKI's offerings.", "question": "What is the message from the president?", "answer": "The President's message encourages students to engage with ACM GIKI and make the most of their time and opportunities."},
    ]
    
    # Convert to HuggingFace dataset
    dataset = Dataset.from_dict(data)
    
    # Tokenizer and model
    tokenizer = RobertaTokenizer.from_pretrained("deepset/roberta-base-squad2")
    model = RobertaForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")
    
    # Tokenize the dataset
    def preprocess_function(examples):
        return tokenizer(examples["question"], examples["context"], truncation=True, padding=True)
    
    tokenized_data = dataset.map(preprocess_function, batched=True)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy='epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
    )
    
    # Define Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data,
    )
    
    # Fine-tune the model
    trainer.train()

import streamlit as st
from streamlit.components.v1 import html
import time

# Set page config with logo
st.set_page_config(
    page_title="ACM GIKI Chapter Chatbot",
    page_icon="🤖",
    layout="centered"
)

# Custom CSS for professional styling
st.markdown("""
    <style>
        /* Main styling */
        .main {
            max-width: 800px;
            margin: auto;
        }
        
        /* Header styling */
        .header {
            display: flex;
            align-items: center;
            margin-bottom: 1.5rem;
        }
        
        /* Title styling */
        .title {
            color: #2c3e50;
            font-size: 2.2rem;
            font-weight: 700;
            margin-left: 1rem;
        }
        
        /* Input field */
        .stTextInput>div>div>input {
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            padding: 12px;
            font-size: 1rem;
        }
        
        /* Success message */
        .stSuccess {
            border-radius: 10px;
            padding: 1rem;
            background-color: #f8f9fa;
        }
        
        /* Feedback section */
        .feedback-title {
            font-size: 1.1rem;
            font-weight: 600;
            margin: 1.5rem 0 0.5rem 0;
            color: #2c3e50;
        }
        
        /* Spinner animation */
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .spinner {
            animation: spin 1s linear infinite;
            color: #0085ca;
        }
    </style>
""", unsafe_allow_html=True)

# Header with logo
st.markdown('<div class="header">', unsafe_allow_html=True)
st.image("logo.png", width=80)
st.markdown('<h1 class="title">ACM GIKI CHAPTER CHATBOT</h1>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("ASK ANYTHING ABOUT ACM GIKI!", unsafe_allow_html=True)

# User input
user_input = st.text_input(
    "YOUR QUESTION:", 
    placeholder="E.g., Who is the president of ACM?",
    key="user_input"
)

# Processing with animation
if user_input:
    with st.spinner('PROCESSING YOUR QUESTION...'):
        # Add processing animation
        html("""
        <div style="display: flex; justify-content: center; margin: 1rem 0;">
            <div class="spinner" style="font-size: 2rem;">⏳</div>
        </div>
        """)
        time.sleep(0.5)  # Simulate processing time
        response = answer_question(user_input)
    
    # Display response
    st.success(response.upper() if response else "NO RESPONSE GENERATED")

# Collect feedback (original function with styled container)
with st.container():
    st.markdown('<div class="feedback-title">PROVIDE FEEDBACK</div>', unsafe_allow_html=True)
    collect_feedback()  # Your original function remains unchanged

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #7f8c8d; font-size: 0.9rem; margin-top: 2rem;">
        <p>© 2025 ACM GIKI STUDENT CHAPTER | POWERED BY AI</p>
    </div>
""", unsafe_allow_html=True)