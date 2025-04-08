## Problem Statement

Create a chatbot capable of diagnosing common medical conditions based on user-input symptoms. The chatbot should use machine learning models trained on medical data to provide accurate suggestions and recommendations for further action.

## Solution Overview
We've built a chatbot using **Large Language Models (LLMs)** and the **Langchain** framework. The chatbot takes symptoms as input from the user and provides a diagnosis based on this input. **It can ask follow-up questions** to gather more information if needed and offers recommendations for further action, such as seeing a doctor or getting tests done. The chatbot fetches answers from credible sources (books on Medicine, Pediatrics, and Dermatology) containing common diseases and their symptoms. The user interface is created using **Gradio** and the data is stored and retrieved using **Pinecone**. Answers are formulated using the **Gemini-API** due to its free and user-friendly nature.

## Project Setup

Follow these steps to set up the project locally:

1. **Clone the repository and navigate to the project directory:**
    ```bash
    git clone https://github.com/Sanchit-Motghare/Medical-Chatbot.git
    cd Medical-Chatbot
    ```
2. **Create a virtual environment using conda:**
    ```bash
    conda create -n chatbot python=3.10
    conda activate chatbot
    ```

3. **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Start the chatbot:**
    ```bash
    python app.py
    ```

5. **Access the chatbot:**
    Open your browser and navigate to `http://localhost:{port}/`, where `{port}` is the port number shown in the terminal (e.g., `http://127.0.0.1:7860`).

6. **Stop the server:**
    Press `Ctrl+C` in the terminal.

## Using the Chatbot

![Chatbot Interface](./media/UI.png)

1. **Enter symptoms:**
    Type the symptoms in the text box and click the `Submit` button.
2. **Receive diagnosis:**
    The chatbot will respond with a diagnosis based on the entered symptoms.
3. **Button Functions:**
    - `Clear`: Clears the chat history and starts a new conversation.

## Solution Workflow

![Solution Approach](./media/Approach.png)

## Running the Project on Your Own Data

1. **Add your data:**
    Place your data in the `data` folder.
2. **Create a Pinecone index:**
    Add your Pinecone API key to the `config/config.yaml` file and load your data to the index:
    ```bash
    python store_to_index.py
    ```
3. **Set up Gemini-API:**
    Create an account and add your API key to the `config/config.yaml` file.
4. **Start the chatbot:**
    ```bash
    python app.py
    ```
5. **Access the chatbot:**
    Open your browser and navigate to `http://localhost:{port}/`.

## Some Test Cases
**Disease: Common Cold**
- User message 1: I have a runny nose, headache, and sore throat.
- User message 2: I also have a mild fever and feel very tired.
- User message 3: I've noticed some congestion and occasional coughing.

Prevention, cautions, and medical advice:
- Prevention: Wash hands frequently, avoid close contact with sick individuals, and maintain a healthy immune system through diet and exercise.
- Cautions: Stay hydrated, rest as much as possible, and avoid sharing personal items.
- Medical advice: Over-the-counter medications can help relieve symptoms. If symptoms persist for more than a week or worsen, consult a healthcare provider.

---

**Disease: Influenza (Flu)**
- User message 1: I have a high fever, chills, and muscle aches.
- User message 2: I also have a cough, sore throat, and fatigue.
- User message 3: I've experienced some nausea and a runny nose.

Prevention, cautions, and medical advice:
- Prevention: Get the annual flu vaccine, practice good hygiene, and avoid close contact with infected individuals.
- Cautions: Stay home from work or school to prevent spreading the virus, and drink plenty of fluids.
- Medical advice: Antiviral medications may be prescribed by a doctor, and rest is crucial for recovery. Seek medical attention if symptoms become severe.



## References

- [Langchain Documentation](https://docs.langchain.com/)
- [Gradio Documentation](https://gradio.app/docs)
- [Pinecone Documentation](https://www.pinecone.io/docs/)
- [Gemini-API Documentation](https://geminiapi.com/docs/)
- [Hugging Face Transformers Documentation](https://huggingface.co/transformers/)

Thank you for checking out our project! If you have any questions or suggestions, feel free to open an issue or contact us.
