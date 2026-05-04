# agent logic
import os
import re
#from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
import pandas as pd

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

import streamlit as st
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# Load environment variables
#load_dotenv()

# Initialize model
MODEL_LLM = "openai:gpt-4o-mini"
MODEL = init_chat_model(MODEL_LLM, temperature=0.8)

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# Load recipe dataset
df = pd.read_csv("tool_files/13k-recipes.csv")

SYSTEM_PROMPT = """
Your name is ChefMate. You are a smart recipe recommendation assistant.

Your job:
- Help users decide what to cook based on ingredients they already have
- Match user ingredients to real recipes from a dataset
- Rank recipes by best ingredient match
- Clearly show missing ingredients
- Provide simple cooking instructions

Rules:
- Be concise and helpful
- Prioritize recipes that use the most user ingredients
- Always return top 5 recipes
- Always explain why recipes were selected
- Assume users are on a budget
"""

@tool
def parse_ingredients(user_input: str) -> list:
    """
    Tool 1: Ingredient Parser
    Takes raw user input and returns a clean list of ingredients.
    Example: "I have chicken, rice and eggs" -> ["chicken", "rice", "eggs"]
    """
    print("The agent is using parse_ingredients tool")
    user_ingredients = [i.strip().lower() for i in re.split(r'[,]+|\band\b', user_input) if i.strip()]
    return user_ingredients


@tool
def match_recipes(user_input: str) -> str:
    """
    Tool 2: Recipe Matcher
    Takes raw user input, matches against CSV dataset,
    returns top 5 recipes ranked by ingredient match score.
    Use this tool when the user provides a list of ingredients and wants recipe suggestions.
    """
    print("The agent is using match_recipes tool")
    user_ingredients = parse_ingredients.run(user_input)

    results = []
    for _, row in df.iterrows():
        recipe_ingredients = str(row["Ingredients"]).lower()
        matched = [ing for ing in user_ingredients if ing in recipe_ingredients]
        match_score = len(matched)

        if match_score > 0:
            results.append({
                "title": row["Title"],
                "match_score": match_score,
                "matched_ingredients": matched,
                "ingredients": row["Ingredients"]
            })

    top_5 = sorted(results, key=lambda x: x["match_score"], reverse=True)[:5]

    # format as string for the agent
    output = "Top 5 matching recipes:\n"
    for r in top_5:
        output += f"\n- {r['title']} | Matched: {r['matched_ingredients']} | Score: {r['match_score']}"
    return output


@tool
def get_recipe_details(recipe_title: str) -> str:
    """
    Tool 3: Recipe Details Lookup
    ALWAYS use this tool when the user asks for full instructions,
    how to make a recipe, or asks for more details about a specific
    recipe by name. Do NOT answer from memory. You MUST call this
    tool to retrieve the real instructions from the dataset.
    Input should be the exact recipe title as a string.
    - When the user asks how to make a specific recipe or wants full
    instructions, you MUST use the get_recipe_details tool. NEVER
    answer cooking instructions from memory.
    - You do NOT have access to recipe instructions in your memory.
    - When a user asks how to make a recipe or wants full instructions,
      you MUST call the get_recipe_details tool. This is the ONLY way
      to retrieve instructions. Never generate cooking steps yourself.
    """
    print("The agent is using get_recipe_details tool")

    # case-insensitive search for the recipe title
    match = df[df["Title"].str.lower() == recipe_title.strip().lower()]

    if match.empty:
        # try partial match if exact match fails
        match = df[df["Title"].str.lower().str.contains(recipe_title.strip().lower())]

    if match.empty:
        return f"Sorry, could not find a recipe called '{recipe_title}'. Try using the exact title from the match_recipes results."

    row = match.iloc[0]
    output = (
        f"Recipe: {row['Title']}\n\n"
        f"Ingredients:\n{row['Ingredients']}\n\n"
        f"Instructions:\n{row['Instructions']}"
    )
    return output



agent = create_agent(
    model=MODEL,
    tools=[parse_ingredients, match_recipes, get_recipe_details],
    system_prompt=SYSTEM_PROMPT
)


def initialize_messages():
    """
    Creates a new conversation with the system prompt.
    """
    return [{"role": "system", "content": SYSTEM_PROMPT}]


def get_chefmate_response(messages, user_input):
    docs = vectorstore.similarity_search(user_input, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])
    augmented_prompt = f"Use this context to help answer:\n\n{context}\n\nQuestion: {user_input}"

    print(augmented_prompt)

    # Store the ORIGINAL message for display, augmented for the agent
    messages.append({"role": "user", "content": user_input, "display": user_input})

    results = agent.invoke({"messages": messages[:-1] + [{"role": "user", "content": augmented_prompt}]})
    assistant_message = results["messages"][-1].content
    messages.append({"role": "assistant", "content": assistant_message})
    return assistant_message, messages