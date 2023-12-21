# youtube-topic-finder

Recommend running app in a `virtualenv`

First you can install dependencies as follows:

```
pip install -r requirements.txt
```

Before running app, you need to create a `.env` file at the root directory and provide the following env vars:

```
PINECONE_API_KEY=
PINECONE_ENVIRONMENT=
```

You can get those from your Pinecone account

Finally, you can run the app as follows:

```
python app.py
```

To send requests to your service, you simply need to send a GET request to `localhost:5000/v1/topic_timestamp` with query params `url` and `topic`.
