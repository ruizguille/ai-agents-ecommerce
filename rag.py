import asyncio
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelResponse
from rich.console import Console
from kb import KB_INFORMATION
from dotenv import load_dotenv
load_dotenv()

console = Console()

class KnowledgeBase:
    def search(self, search_query: str) -> str:
        return KB_INFORMATION


SYSTEM_PROMPT = """\
You are a helpful and knowledgeable assistant for the luxury fashion store Maison Noir.
Your role is to provide detailed information and assistance about the store and its products.

Follow these guidelines:
- ALWAYS search the knowledge base using the search_knowledge_base tool to answer user questions.
- Provide accurate product and policy information based ONLY on the information retrieved from the knowledge base. Never make assumptions or provide information not present in the knowledge base.
- Structure your responses in a clear, concise and professional manner, maintaining our premium brand standards
- Highlight unique features, materials, and care instructions when relevant.
- If information is not found in the knowledge base, politely acknowledge this.
"""

rag_agent = Agent(
    model='openai:gpt-4o',
    deps_type=KnowledgeBase,
    system_prompt=SYSTEM_PROMPT
)

@rag_agent.tool
async def search_knowledge_base(ctx: RunContext[KnowledgeBase], search_query: str) -> str:
    """Search the knowledge base to retrieve information about Maison Noir, the store and its products"""
    return ctx.deps.search(search_query)

async def run_agent():
    kb = KnowledgeBase()
    messages = []
    console.print('Welcome to Maison Noir. How may I assist you today?', style='cyan', end = '\n\n')
    while True:
        user_message = input()
        console.print()
        result = await rag_agent.run(user_message, message_history=messages, deps=kb)
        messages += result.new_messages()
        console.print(result.data, style='cyan', end = '\n\n')

async def run_agent_streaming():
    kb = KnowledgeBase()
    messages = []
    console.print('Welcome to Maison Noir. How may I assist you today?', style='cyan', end = '\n\n')
    while True:
        result_content = ''
        user_message = input()
        print()
        async with rag_agent.run_stream(user_message, message_history=messages, deps=kb) as result:
            async for chunk in result.stream_text(delta=True):
                result_content += chunk
                console.print(chunk, style='cyan', end = '')
        
        console.print('\n')
        # The final result message will NOT be added to result messages
        # if you use stream_text(delta=True), so we need to add it manually
        messages += result.new_messages() + [
            ModelResponse.from_text(content=result_content, timestamp=result.timestamp())
        ]

if __name__ == '__main__':
    asyncio.run(run_agent_streaming())
