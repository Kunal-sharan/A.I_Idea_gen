from langchain import memory
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer
import streamlit as st
import json
from streamlit_sortables import sort_items
import os
from langchain.memory import ChatMessageHistory
try:
	from googlesearch import search
except ImportError:
	print("No module named 'google' found")

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
# to search
import ast
import mermaid as md
from mermaid.graph import Graph
import base64
import urllib.parse
from langchain_groq import ChatGroq
import base64
import urllib.parse
import re
from langchain.prompts import PromptTemplate
llm_idea_key=st.secrets["GROQ_KEY"]
llm_key=st.secrets["GEMINI_KEY"]
llm_idea = ChatGroq(
    model="mixtral-8x7b-32768",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=llm_idea_key
    # other params...
)
template = """You are a helpful AI assistant engaging in a conversation with a human. Be friendly, informative, and always try to provide accurate information.

Current conversation:
{history}
Human: {input}
AI Assistant:"""

prompt = PromptTemplate(
    input_variables=["history", "input"],
    template=template
)
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory()

# Initialize ConversationChain
conversation = ConversationChain(
    llm=llm_idea,
    memory=st.session_state.memory,
    prompt=prompt,
    verbose=True
)

def generate_vertical_flowchart(steps):
    # Start the Mermaid flowchart
    mermaid_code = "flowchart LR\n"

    # Add nodes and connections
    for i, step in enumerate(steps, start=1):
        # Escape any quotes in the step text
        step = step.replace('"', '\\"')

        # Use different shapes for start, end, and intermediate steps
        if i == 1:
            mermaid_code += f'    S{i}(["Start: {step}"])\n'
        elif i == len(steps):
            mermaid_code += f'    S{i}{{"End: {step}"}}\n'
        else:
            mermaid_code += f'    S{i}["{step}"]\n'

        if i < len(steps):
            mermaid_code += f"    S{i} --> S{i+1}\n"

    # Add a loop back arrow from the last step to an earlier step (e.g., step 2)
    if len(steps) > 2:
        mermaid_code += f"    S{len(steps)} -.-> S2\n"

    return mermaid_code

def flow(steps_string):
    # Convert the string representation of the array to an actual array
    steps = re.findall(r'"([^"]*)"', steps_string)

    # Generate the Mermaid code
    mermaid_code = generate_vertical_flowchart(steps)

    # Encode the Mermaid code
    encoded_code = base64.b64encode(mermaid_code.encode('utf-8')).decode('utf-8')

    # URL-encode the Base64 string
    url_encoded = urllib.parse.quote(encoded_code)

    # Construct the Mermaid.ink URL
    mermaid_url = f"https://mermaid.ink/img/{url_encoded}"

    return mermaid_url

# Set your Google API key

# Initialize the Gemini model
llm = ChatGoogleGenerativeAI(model="gemini-pro",google_api_key=llm_key)

# Initialize conversation memory
# memory = ConversationBufferWindowMemory(k=2)


# Streamlit app
st.set_page_config(layout="wide")
st.title("AI Idea generation app")
# use_data=""
# flow=None
@st.fragment
def drag_drop():
  title_arr=[]
  for i in all_data:
    title_arr.append(i["title"])
  original_items = [
              {'header': 'first container',  'items': title_arr },
              {'header': 'second container', 'items': []}
          ]

  sorted_items = sort_items(original_items, multi_containers=True)

  if st.button("Submit"):
    if len(sorted_items[1].get("items"))>1:
      st.warning("Cannot choose more than 1 link")
    else:
      # st.write(sorted_items[1].get("items"))
      for i in all_data:
        if i['title']==sorted_items[1].get("items")[0]:
          # st.markdown(i["title"])
          # st.markdown(i['url'])
          # st.markdown(i["llm_res"])
          get_idea(i["llm_res"])

flow_url=""

@st.fragment
def chatbot(idea_des):
        # st.header("Chatbot")
        # show_flow(flow_url)
        # Initialize chat history

        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Create a container for chat messages
        chat_container = st.container(height=450)

        # Create a container for the input box
        input_container = st.container()

        # Display chat messages from history on app rerun
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        # Create the input box at the bottom
        with input_container:
            prompt = st.chat_input("What is your message?")

        # React to user input
        if prompt:
            # Display user message in chat message container
            with chat_container:
                st.chat_message("user").markdown(prompt)
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Get response from ConversationChain
            prompt=f"based on this guide \n `{idea_des}` \n answer this `{prompt}`"
            try:
             response = conversation.predict(input=prompt)
            except:
              st.error("Refresh the page try again") 

            # Display assistant response in chat message container
            with chat_container:
                with st.chat_message("assistant"):
                    st.markdown(response)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})


@st.fragment
def show_flow(flow_url):
  if len(flow_url)>0:
    with st.spinner("Loading..."):
      st.image(flow_url)
      st.markdown("check out this [link](%s)" % flow_url)
@st.fragment
def get_idea(use_data):
  if st.button("Generate Idea"):
    idea_arr=llm_idea.invoke(f"""
Generate an innovative project idea for a student's portfolio using this `{use_data}` . The project should showcase creativity, technical skills, and problem-solving ability while being feasible for a student to complete.

Provide your response as an array of 5 strings:

[
"Project Title: [catchy, descriptive name]",
"Core Concept: [brief explanation of the project's main idea]",
"Unique Angle: [what makes this project stand out]",
"Key Technologies: [main tools or technologies used]",
"Portfolio Impact: [how this enhances the student's profile]",
"Problem Solved: [specific issue or challenge the project addresses]",
"Real-world Application: [how the project can be applied in practical scenarios]"
]

Guidelines
1. Emphasizes originality and innovation in the "Core Concept" and "Unique Angle" sections.
2. Focuses on student-level resources and skills in the "Key Technologies" point.
3. Highlights the practical application of academic knowledge in the "Practical Application" section.
4. Includes problem-solving and real-world issues in the "Problem Addressed" point.
5. Keeps each element concise yet informative for a quick portfolio overview.
6. Combines the previous "Portfolio Impact" with problem-solving showcase.
7. Maintains a structure that's easy to fill out and review quickly.

This format should help students create project descriptions that are innovative, achievable, practical, and impactful for their portfolios.

The idea should be explained clearly in these 5 elements, highlighting its uniqueness and value for a student's professional development..""")
    idea_arr=idea_arr.content
    # idea_arr
    idea=idea_arr[idea_arr.find("["):idea_arr.find("]")+1]
    # idea

    try:
      # arr=ast.literal_eval(idea)
      # st.write(arr)
      # st.write(flow(idea))
      # st.image(flow(arr))
      flow_url=flow(idea)
      # with st.sidebar:
      show_flow(flow_url)
      with st.container(border=True,height=500):
        st.header("Idea Guide")
        idea_des=(llm_idea.invoke(f"""Given the following array representing key elements of an AI project :

  {idea}
  Provide a detailed explanation of each element in the array. For each item:

  1. Describe what this element represents in the context of the AI project.
  2. Explain why this element is important for the project's success.
  3. Discuss potential challenges or considerations related to this element.
  4. Suggest practical approaches or tools that could be used to implement this element.

  Present your explanation in a clear, structured format, addressing each array element in order. Your response should offer insights into how these elements work together to create an innovative AI solution and keep it short""")).content
        st.markdown(idea_des)
      st.header("Chat Assistant")  
      with st.container(border=True,height=550):
        idea_des=llm.invoke(f"summarize this make sure to keep the summary short but precise without leaving any relavant points \n `{idea_des}`")
        chatbot(idea_des.content)
    except:
     st.error("Internal Server error Try again")
  # use_data





# Sidebar with consent form


links=[]
final_links=[]
all_data=[]
# Main content
    # Dropdown input
options = ["Option 2","education", "Option 3"]
selected_option = st.text_input("Enter the industry or field in one word or at max two ",placeholder="eg. sports, law, education, healthcare...")
if selected_option :
  query = f"issues that can be solved by A.I in {selected_option} industry"
  for j in search(query, tld="co.in", num=5, stop=5, pause=2):
    f=j.find('#')
    if f == -1:
      links.append(j)
  links=list(set(links))
  final_links=[]
  for i in links:
    final_links.append([i])


# Tabs
tab1 = st.container()

with tab1:
        # st.header("GET STARTED")
	
        if len(links) > 0 :
          all_docs=[]
          with st.spinner("Loading..."):
            for links in final_links:
              try:
                loader = AsyncHtmlLoader(links)
                docs = loader.load()
                html2text = Html2TextTransformer()
                docs_transformed = html2text.transform_documents(docs)
                all_docs.append(docs_transformed)
              except:
                continue
            final_links
            for docs_transformed in all_docs:
              for i in docs_transformed:
                try:
                  llm_response=res=llm.invoke(f"summarize in short paragraph all the necessary information from this chunk \n `{i.page_content}` \n  based on this title : {i.metadata['title']}")
                  llm_response=llm_response.content
                  # llm_response="fjdklsgfdks"
                  all_data.append({"url":i.metadata["source"],"title":i.metadata["title"],"content":i.page_content,"size":len(i.page_content),"llm_res":llm_response})
                except:			
                  st.error(i)
          if len(all_data)>0:
            for i in all_data:
                with st.expander(i["title"]):
                    st.markdown(i["url"])
                    st.markdown(i['llm_res'])
          if len(links)>0:
            drag_drop()
