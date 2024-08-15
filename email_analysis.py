from openai import OpenAI
import time
import re
import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from openai import OpenAI
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import os
from sklearn.cluster import KMeans
import logging


OPENAI_API_KEY = "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"
client = OpenAI(api_key=OPENAI_API_KEY)

def parse_comprehensive_analysis(analysis):
    topics = []
    sentiment = "Unknown"
    emotion = "Unknown"
    people = []
    organizations = []
    locations = []
    categories = {}
    
    sections = analysis.split("\n\n")
    for section in sections:
        lines = section.split("\n")
        if lines and lines[0].startswith("1."):
            topics = [re.sub(r'^\d+\.\s*', '', topic.strip()) for topic in lines[1:] if topic.strip()]
        elif lines and lines[0].startswith("2."):
            sentiment = lines[-1].strip() if len(lines) > 1 else "Unknown"
        elif lines and lines[0].startswith("3."):
            emotion = lines[-1].strip() if len(lines) > 1 else "Unknown"
        elif lines and lines[0].startswith("4."):
            current_category = None
            for line in lines[1:]:
                if "People:" in line:
                    current_category = "people"
                elif "Organizations:" in line:
                    current_category = "organizations"
                elif "Locations:" in line:
                    current_category = "locations"
                elif line.strip() and current_category:
                    entity = re.sub(r'^\s*-\s*', '', line.strip())
                    if current_category == "people":
                        people.append(entity)
                    elif current_category == "organizations":
                        organizations.append(entity)
                    elif current_category == "locations":
                        locations.append(entity)
        elif lines and lines[0].startswith("5."):
            for line in lines[1:]:
                if ':' in line:
                    category, percentage = line.split(':')
                    categories[category.strip()] = percentage.strip()
    
    sentiment = sentiment.split(":")[-1].strip()
    emotion = emotion.split(":")[-1].strip()
    
    return topics, sentiment, emotion, people, organizations, locations, categories

def analyze_emails_with_openai(person_emails):
    person, emails = person_emails
    max_emails = 20
    max_tokens = 2000
    combined_text = " ".join(emails[:max_emails])[:max_tokens]
    
    prompt = f"""Analyze the following email content for user {person}. Provide your analysis in the following format:

1. Topics:
[List the top 10 main topics discussed, or fewer if there aren't enough relevant ones. One topic per line.]

2. Sentiment:
[State the overall sentiment (positive, negative, or neutral)]

3. Emotion:
[State the dominant emotion expressed]

4. Entities:
People:
[List the top 5 most frequently mentioned people, excluding the email sender/recipient. One per line.]
Organizations:
[List the top 5 most frequently mentioned organizations. One per line.]
Locations:
[List the top 5 most frequently mentioned locations. One per line.]

5. Categories:
[List each category followed by its percentage. One per line.]
Work-related: X%
Personal: X%
Urgent: X%
Informational: X%
Action Required: X%
Follow-up: X%
Other: X% (specify if significant)

Email content:
{combined_text}"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that analyzes email content. Provide your analysis strictly in the format specified in the prompt."},
                {"role": "user", "content": prompt}
            ]
        )
        analysis = response.choices[0].message.content.strip()
        time.sleep(1)  # To avoid rate limiting
        return person, parse_comprehensive_analysis(analysis)
    except Exception as e:
        print(f"Error in OpenAI API call for user {person}: {e}")
        return person, None