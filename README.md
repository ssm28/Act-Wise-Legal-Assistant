# Act Wise ⚖️

**Act Wise** is a Retrieval-Augmented Generation (RAG) based software application designed to provide precise and context-aware answers to legal queries regarding Indian statutes. 

This project was developed as a Minimum Viable Product (MVP) to bridge the gap between complex legal texts and accessible information, focusing specifically on modern Indian law.

## 🚀 Key Features

* **Contextual Legal Answers:** Leverages RAG architecture to fetch highly relevant legal clauses before generating responses, ensuring high accuracy.
* **Specialized Legal Domain:** Currently tailored to answer queries based on key Indian statutes, including:
  * The Consumer Protection Act, 2019
  * Bharatiya Nyaya Sanhita (BNS)
* **Intuitive User Interface:** Designed with a clean, user-friendly interface to make querying legal text straightforward for non-experts.

## 🛠️ Tech Stack

* **LLM:** Gemini API
* **Embedding Model:** all-MiniLM-L6-v2
* **Vector Database:** Chroma DB
* **Text Processing:** Sentence Transformers
* **Environment/Tools:** Python, VS Code, Git

## 💻 Local Setup & Installation

To run Act Wise on your local machine, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YourUsername/act-wise.git](https://github.com/YourUsername/act-wise.git)
   cd act-wise

2. **Create and activate a virtual environment:**
   # Windows (VS Code terminal)
    python -m venv venv
    .\venv\Scripts\activate

   # Mac/Linux
    python3 -m venv venv
    source venv/bin/activate

3. **Install dependencies:**
   pip install -r requirements.txt

4. **Set up environment variables:**
    Create a .env file in the root directory and add your API keys:
     GEMINI_API_KEY=your_google_gemini_api_key_here

5. **Run the application:**
   python app.py

🎓 About
This project was developed as a final-year Master of Computer Applications (MCA) academic project. It demonstrates the practical application of Large Language Models (LLMs) and Vector databases in specialized, high-stakes domains like the legal sector.

