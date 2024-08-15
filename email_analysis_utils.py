from openai import OpenAI
import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

OPENAI_API_KEY = "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"
client = OpenAI(api_key=OPENAI_API_KEY)

def generate_llm_query_prompts(user_data):
    # Combine a sample of sent and received emails
    sample_sent = '\n'.join(user_data['sent'].head(3)['content'].tolist()) if not user_data['sent'].empty else ''
    sample_received = '\n'.join(user_data['received'].head(3)['content'].tolist()) if not user_data['received'].empty else ''
    
    prompt = f"""Based on the following user data and email samples, generate 10 highly specific and actionable prompts that this user might ask an AI language model to gain insights about their email communication patterns, productivity, or to get help with email-related tasks. The prompts should be questions or requests that the user could directly ask an AI to analyze their email data or assist with email management.

User Data:
- Most discussed topics: {', '.join(user_data['topics'][:5])}
- Sentiment: {user_data['sentiment']}
- Dominant emotion: {user_data['emotion']}
- Top mentioned people: {', '.join(user_data['mentioned_people'][:3])}
- Top mentioned organizations: {', '.join(user_data['mentioned_organizations'][:3])}
- Top mentioned locations: {', '.join(user_data['mentioned_locations'][:3])}
- Email categories: {user_data['email_categories']}

Sample of Sent Emails:
{sample_sent}

Sample of Received Emails:
{sample_received}

Please provide 10 LLM query prompts in the following format:
1. [LLM Query Prompt 1]
2. [LLM Query Prompt 2]
3. [LLM Query Prompt 3]
4. [LLM Query Prompt 4]
5. [LLM Query Prompt 5]
6. [LLM Query Prompt 6]
7. [LLM Query Prompt 7]
8. [LLM Query Prompt 8]
9. [LLM Query Prompt 9]
10. [LLM Query Prompt 10]

Each prompt MUST:
1. Be extremely specific, referencing actual names, topics, or events from the user's email data.
2. Focus on actionable insights or tasks related to the user's email content.
3. Address concrete aspects of email management, task tracking, or communication improvement.
4. Avoid vague or general questions about communication patterns or productivity.
5. Aim to extract or summarize key information from specific email threads or conversations.

Examples of the level of specificity required:
- "Summarize any action items I have from my [Specific Project] conversation with [Person's Name]."
- "What were the main points [Person's Name] wanted to discuss in our recent email exchange about [Specific Topic]?"
- "List the key items I need to be aware of from my latest discussion with [Person's Name] regarding [Specific Project/Topic]."

Ensure that each prompt is tailored to the user's actual email content, mentioned people, organizations, and topics."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an AI assistant that generates extremely specific and contextual LLM query prompts based on user email data. Your prompts must directly reference details from the user's emails and network position."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip().split("\n")
    except Exception as e:
        print(f"Error in OpenAI API call: {e}")
        return []

def process_user(user_data):
    return user_data.name, generate_llm_query_prompts(user_data)

def parallel_generate_llm_query_prompts(email_network_df, num_processes=None):
    if num_processes is None:
        num_processes = os.cpu_count()

    print(f"Generating LLM query prompts using {num_processes} processes...")
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        results = list(tqdm(
            executor.map(process_user, [row for _, row in email_network_df.iterrows()]),
            total=len(email_network_df),
            desc="Generating prompts"
        ))

    recommendations = dict(results)
    email_network_df['recommended_llm_queries'] = email_network_df.index.map(recommendations)

    return email_network_df