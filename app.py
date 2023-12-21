import os
import time
import torch
import pinecone
import requests
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
from youtube_transcript_api import YouTubeTranscriptApi
from flask import Flask, request, jsonify
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device != 'cuda':
  print(f"You are using {device}. This is much slower than using "
          "a CUDA-enabled GPU. If on Colab you can change this by "
          "clicking Runtime > Change runtime type > GPU.")

model = SentenceTransformer("multi-qa-mpnet-base-dot-v1", device=device)

index_name = 'topic-semantic-search'

pinecone.init(api_key=os.environ["PINECONE_API_KEY"], environment=os.environ["PINECONE_ENVIRONMENT"])

def url_to_video_id(url):
  try:
    url_without_https = url.removeprefix("https://")
    # we could be receiving links like follows:
    # https://www.youtube.com/watch?v=GyllRd2E6fg
    # https://youtu.be/GyllRd2E6fg?feature=shared
    if ".be" in url_without_https:
      url_without_query_params = url_without_https.split("?")[0]
      return url_without_query_params.split("/")[1]
    else:
      return url_without_https.split('?v=')[1]
  except Exception as e:
    print(f"An error occurred in url_to_video_id: {e}")


def get_youtube_transcript(video_id):
  try:
    transcript = YouTubeTranscriptApi.get_transcript(video_id)

    formatted_transcript = []
    for entry in transcript:
      formatted_transcript_entry = f"{entry['start']} - {entry['start'] + entry['duration']}: {entry['text']}"
      formatted_transcript.append(formatted_transcript_entry)
      print(formatted_transcript_entry)
    
    return formatted_transcript

  except Exception as e:
    print(f"An error occurred in get_youtube_transcript: {e}")

def upload_embedding_data(index, formatted_transcript):
  try:
    batch_size = 128
    vector_limit = 100000

    duration_list = []
    text_fragment_list = []
    for transcript_entry in formatted_transcript:
      duration = transcript_entry.split(':')[0]
      text_fragment = transcript_entry.split(':')[1]
      text_fragment_list.append(text_fragment)
      duration_list.append(duration)

    for i in tqdm(range(0, len(text_fragment_list), batch_size)):
      i_end = min(i+batch_size, len(text_fragment_list))
      ids = [str(x) for x in range(i, i_end)]
      metadatas = []
      for j in range(i, i_end):
        metadatas.append({'text': text_fragment_list[j], 'duration': duration_list[j]})
      xc = model.encode(text_fragment_list[i:i_end]).tolist()
      records = zip(ids, xc, metadatas)
      index.upsert(vectors=records)

    index.describe_index_stats()
  except Exception as e:
    print(f"An error occured in upload_embedding_data: {e}")

def run_query_vector(index, topic):
  xq = model.encode(topic).tolist()

  xc = index.query(xq, top_k=10, include_metadata=True)
  for result in xc['matches']:
    print(f"{round(result['score'], 2)}: {result['metadata']['topic_duration']} -{result['metadata']['text']}")
  
  return xc

def get_timestamp_from_query_vector(xc):
  # Here we're simply returning the closest vector embedding
  topic_duration_list = [match['metadata']['topic_duration'] for match in xc['matches']]
  scores_list = [round(match['score'], 2) for match in xc['matches']]
  start_topic_timestamps = [int(float(topic_duration.split(" - ")[0])) for topic_duration in topic_duration_list]

  return start_topic_timestamps[0], scores_list[0]

@app.get("/v1/topic_timestamp")
def get_video_timestamps_v1():
  if index_name not in pinecone.list_indexes():
    pinecone.create_index(
      name=index_name,
      dimension=model.get_sentence_embedding_dimension(),
      metric='cosine'
    )

  index = pinecone.Index(index_name)

  url = request.args.get('url')
  if url is None or len(url) == 0:
    return 'Missing url parameter', 400
  topic = request.args.get('topic')
  if topic is None or len(topic) == 0:
    return 'Missing topic parameter', 400

  video_id = url_to_video_id(url)
  if video_id is None:
    return 'Invalid video id', 400

  video_transcript = get_youtube_transcript(video_id)
  upload_embedding_data(index, video_transcript)
  xc = run_query_vector(index, topic)
  timestamp, average_score = get_timestamp_from_query_vector(xc)
  
  response = {"video_id": video_id, "timestamp": timestamp, "confidence": "high" if average_score >= 0.6 else "medium-low"}

  pinecone.delete_index(index_name)

  return response, 200

if __name__ == '__main__':
  app.run(debug=True, port=5000)