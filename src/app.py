import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

def create_the_quiz_prompt_template():
    template = """
You are an expert quiz maker for technical fields. Let's think step by step and
create a quiz with {num_questions} {quiz_type} questions about the following concept/content: {quiz_context}.

The format of the quiz could be one of the following:
- Multiple-choice: 
- Questions:
    <Question1>: 
        <a. Answer 1>
        <b. Answer 2>
        <c. Answer 3>
        <d. Answer 4>
    <Question2>: 
        <a. Answer 1>
        <b. Answer 2>
        <c. Answer 3>
        <d. Answer 4>
    ....
- Answers:
    <Answer1>: <a|b|c|d>
    <Answer2>: <a|b|c|d>
    ....
    Example:
    - Questions:
    - 1. What is the time complexity of a binary search tree?
        a. O(n)
        b. O(log n)
        c. O(n^2)
        d. O(1)
    - Answers: 
        1. b
- True-false:
    - Questions:
        <Question1>: 
            <a. True>
            <b. False>
        <Question2>: 
            <a. True>
            <b. False>
        .....
    - Answers:
        <Answer1>: <True|False>
        <Answer2>: <True|False>
        .....
    Example:
    - Questions:
        - 1. What is a binary search tree?
            a. True
            b. False
    - Answers:
        - 1. True
- Open-ended:
    - Questions:
        <Question1>: 
        <Question2>:
    - Answers:    
        <Answer1>: <Answer>
        <Answer2>: <Answer>
    Example:
    - Questions:
        - 1. What is a binary search tree? 
    - Answers: 
        - 1. A binary search tree is a data structure that is used to store data in a sorted manner.

"""
    prompt = PromptTemplate.from_template(template)
    prompt.format(num_questions=3, quiz_type="multiple-choice", quiz_context="Data Structures in Python Programming")
    return prompt

def create_quiz_chain(prompt_template, llm):
    return LLMChain(llm=llm, prompt=prompt_template)

def split_questions_answers(quiz_response):
    parts = quiz_response.split("Answers:")
    questions_raw = parts[0]  # 항상 첫 번째 부분은 질문
    answers = parts[1] if len(parts) > 1 else "No answers provided."  # "Answers:"가 있으면 두 번째 부분을 반환, 없으면 안내 메시지
    
    # 각 질문을 개별적으로 처리하고, 선택지 사이에 개행 추가하고 정렬
    questions = questions_raw.replace(" a.", "\n\t\t\ta.").replace(" b.", "\n\t\t\tb.").replace(" c.", "\n\t\t\tc.").replace(" d.", "\n\t\t\td.")
    return questions, answers


def get_text_from_url(url):
    loader = WebBaseLoader(url)
    document = loader.load()
    return document

def process_text_to_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(text)
    return document_chunks

def create_vector_store(chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(chunks, embeddings)
    return vector_store

def main():
    st.title("Quiz App")
    st.write("This app generates a quiz based on a given context or web document.")
    url = st.text_input("Enter the URL of a web document")
    base_context = st.text_area("Enter the concept/context for the quiz")
    
    num_questions = st.number_input("Enter the number of questions", min_value=1, max_value=10, value=3)
    quiz_type = st.selectbox("Select the quiz type", ["multiple-choice", "true-false", "open-ended"])

    prompt_template = create_the_quiz_prompt_template()
    llm = ChatOpenAI()

    if st.button("Generate Quiz"):
        if url:
            text = get_text_from_url(url)
            chunks = process_text_to_chunks(text)
            chunk_size = 5  # Define how many chunks you use to create one quiz context

            if len(chunks) < num_questions * chunk_size:
                st.write("Not enough content to generate the desired number of questions. Please provide a larger document or reduce the number of questions.")
            else:
                for i in range(num_questions):
                    context = " ".join([chunk.page_content for chunk in chunks[i*chunk_size:(i+1)*chunk_size]])
                    chain = create_quiz_chain(prompt_template, llm)
                    quiz_response = chain.invoke({"num_questions": 1, "quiz_type": quiz_type, "quiz_context": context})
                    if 'text' in quiz_response:
                        quiz_text = quiz_response['text']
                        questions, answers = split_questions_answers(quiz_text)
                        st.session_state[f'answers_{i}'] = answers
                        st.session_state[f'questions_{i}'] = questions
                        st.markdown(f"### Quiz Generated for Section {i+1}")
                        st.write(questions)
                    else:
                        st.write(f"No quiz generated for Section {i+1}. Please check the parameters or context.")
        elif base_context:
            chain = create_quiz_chain(prompt_template, llm)
            quiz_response = chain.invoke({"num_questions": num_questions, "quiz_type": quiz_type, "quiz_context": base_context})
            if 'text' in quiz_response:
                quiz_text = quiz_response['text']
                questions, answers = split_questions_answers(quiz_text)
                st.session_state['answers'] = answers
                st.session_state['questions'] = questions
                st.markdown("### Quiz Generated!")
                st.write(questions)
            else:
                st.write("No quiz generated. Please check the parameters or context.")

    if st.button("Show Answers"):
        if url:
            for i in range(num_questions):
                st.markdown(f"---\n### Answers for Section {i+1}")
                st.write(st.session_state[f'questions_{i}'])
                st.markdown("---")
                st.write(st.session_state[f'answers_{i}'])
        elif base_context:
            st.markdown("---")
            st.write(st.session_state['questions'])
            st.markdown("---")
            st.write(st.session_state['answers'])

if __name__ == "__main__":
    main()

