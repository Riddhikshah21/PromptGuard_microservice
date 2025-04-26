
# PromptGuard_microservice

A scalable microservice that evaluates text similarity between prompts and conditionally forwards one to an LLM, with comprehensive input/output sanitization and testing.

##  Features

1. **Input Sanitization**: Ensures responses are safe by detecting refusal patterns, stripping potentially harmful content, limiting query length, filtering out disallowed words/phrases, or rejecting the entire prompts.
2. **Text Similarity Analysis**: Multiple similarity algorithms:
   - Cosine similarity using sentence embeddings
   - Jaccard similarity using token comparison
3. **LLM Integration**: Forwards prompts to LLM API if similarity threshold is met.
4. **Output Sanitization**: Ensures responses are safe by detecting refusal patterns, data leaks, and inappropriate content.
5. **REST API**: Clean, well-documented API built with FastAPI.
6. **Comprehensive Testing**: Unit, integration, and load tests included.
7. **Containerization**: Ready for deployment with Docker.

---

##  Quick Start

### Prerequisites

- Python 3.10+

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Riddhikshah21/PromptGuard_microservice.git
   ```

2. Set up a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Set environment variables:

   ```bash
   export OPENAI_API_KEY=your_api_key_here
   export OPENAI_MODEL="gpt-3.5-turbo"
   export MAX_QUERY_LENGTH=512
   ```

---

##  Running the Service

Start the service locally:

```bash
uvicorn app.main:app --reload
```

The API will be available at: [http://localhost:8000](http://localhost:8000)  
API Docs: [http://localhost:8000/docs](http://localhost:8000/docs)

---

##  Using Docker

### 1. Build the Docker image

```bash
docker build -t prompt-similarity-service .
```

### 2. Run the container

If you do not have a openai api key, use the below command as it is else replace 'your_api_key' with your openai api key.

```bash
docker run -it -p 8000:8000 \
  -e OPENAI_API_KEY="your_api_key" \
  -e OPENAI_MODEL="gpt-3.5-turbo" \
  -e MAX_QUERY_LENGTH="512" \
  prompt-similarity-service
```

---

##  Text Similarity Methods

### 1. Cosine Similarity (Default)

- Computes the cosine similarity between two input text prompts using TF-IDF (Term Frequency–Inverse Document Frequency) vectorization.
- Evaluates how semantically similar two pieces of text are.

### 2. Jaccard Similarity

- Based on the intersection over union of tokens.
- Simpler calculation based on word overlap.
- Better for keyword/token matching.
- Less computationally intensive.

---

##  Testing

### 1. Run unit and integration tests:

```bash
pytest tests/
```

### 2. Run with coverage report:

```bash
coverage erase
coverage run -m pytest
coverage html
open htmlcov/index.html
```

### 3. Load Testing with Locust:

```bash
locust -f tests/load_test.py --host=http://localhost:8000
```

Navigate to [http://localhost:8089](http://localhost:8089) to configure and run the load test.

---

##  Scaling Considerations

### Horizontal Scaling

- The service is stateless, allowing for easy horizontal scaling.
- Deploy multiple instances behind a load balancer.
- Use container orchestration like Kubernetes for auto-scaling.

### Performance Optimizations

- Embedding model caching for frequently used prompts.
- Response caching for common queries.
- Batched processing for high volume requests.

### High Availability

- Deploy across multiple availability zones.
- Implement health checks and automatic recovery.
