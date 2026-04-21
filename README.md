# car-dekho 
Access it here : [Application](https://car-dekho-eozn.onrender.com/) 

Another URL: [http://shaurya.dynv6.net:9006](http://shaurya.dynv6.net:9006) -  (personal laptop self hosted, might not work for some networks due to firewall issue, also change https to http if browser automatically adds https) 

- What did you build and why? What did you deliberately cut?
I built an AI-based car advisor that lets users just describe what they want in plain language, instead of forcing them to use filters. The system pulls out structured requirements from the query, filters cars based on those, and then reranks them using review data to find better matches.
The idea was to make the experience feel closer to how people actually think when buying a car.
I didn’t spend time on full product features like detailed car pages, comparison views, or a polished browsing UI. I kept the interface simple because I wanted to focus on whether the core idea is helping users reach a confident shortlist is actually working.

- What’s your tech stack and why did you pick it?
I used FastAPI for the backend because it’s lightweight and requires minimal setup, which helped me move quickly.
The rest of the stack is API-driven:
1- Supabase for the database
2- Cohere for embeddings
3- Pinecone for vector search
4- Groq for LLM inference
This was mainly because of deployment limitations. Free hosting doesn’t really support running heavy services locally, so using managed APIs made things simpler and more practical.

- What did you delegate to AI tools vs. do manually? Where did the tools help most?
I designed the overall flow and logic myself, how the query gets processed, how filtering and reranking work, and how everything connects.
AI tools helped mostly with writing code faster and handling repetitive parts. They were especially useful for building the UI quickly, I was able to get something functional without spending too much time on frontend work.

- Where did they get in the way?
Deployment was the most frustrating part. There were a lot of dependency conflicts and Python version issues. Different tools suggested different fixes, which made things more confusing than helpful at times.
I ended up switching between approaches like Poetry and different environments, which slowed things down quite a bit during deployment.

- If you had another 4 hours, what would you add?
I’d improve the reranking quality by using better and more detailed review data, since that’s a key part of the system and right now the dataset is pretty limited.
I’d also work on the UI to make it feel less like a chatbot and more like a proper decision-making tool. And on the backend, I’d clean up the structure a bit and experiment with better ways to chunk and use review data for the RAG pipeline.

# For local setup - 

---

### **Local Setup**

#### **1. Create and activate a virtual environment**

```bash
# create venv
python3 -m venv .venv

# activate (Linux / Mac)
source .venv/bin/activate

# activate (Windows)
.venv\Scripts\activate
```

---

#### **2. Install dependencies**

```bash
pip install -r requirements.txt
```

---

#### **3. Set up environment variables**

Create a `.env` file in the root directory:

```env
SUPABASE_URL=your_url
SUPABASE_SERVICE_KEY=your_key
COHERE_API_KEY=your_key
PINECONE_API_KEY=your_key
PINECONE_INDEX_NAME=your_index
GROQ_API_KEY=your_key
```

---

#### **4. Run the application**

```bash
uvicorn main:app --reload --port 8000
```

---

#### **5. Access the app**

* API: [http://localhost:8000](http://localhost:8000)
* Docs: [http://localhost:8000/docs](http://localhost:8000/docs)

---

This matches your actual backend flow (FastAPI + external services) from your code  and keeps setup straightforward for anyone trying it out.
