# Chatbot_using_LLMs
We will create a basic chatbot, then add knowledge to the chatbot and finetune the model with a custom dataset

## Chatbot with Llama
Step-1. Sign up for an account on Huggingface.

Step-2. Request access to the Llama 3.1 models on Hugging Face.
Meta AI published several model sizes and formats in the Llama 3.1 series.
(You can also use Phi-3 or lighter models if there are any computation or resource constraints)

![Screenshot from 2024-12-03 12-14-11](https://github.com/user-attachments/assets/0d310c3d-98a3-4f7a-a310-9444eabf4ef9)

Step-3. Download the quantized model files.
As we discussed earlier, LLMs like Llama could require a lot of RAM to run. The original 8B model at FP16 precision will require 16GB RAM
Run the command below to download the the Q5_K_M version of the instruct-funed model.

curl -LO https://huggingface.co/second-state/Meta-Llama-3.1-8B-Instruct- 
GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf
![image](https://github.com/user-attachments/assets/355a5440-0ddf-438f-990d-09dee8ced4fc)

### Inference on command line
Step-1. Get the Wasm file 
Use this command to get the wasm file (it is used to get the inference of the model for tasks like text completion.
cd ~
curl -LO https://github.com/LlamaEdge/LlamaEdge/releases/latest/download/llama-simple.wasm

Step-2. Get the inference 
wasmedge --dir .:. \
  --nn-preload default:GGML:AUTO:Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf \
  llama-simple.wasm \
  --prompt "Robert Oppenheimer's most important achievement is "

![image](https://github.com/user-attachments/assets/2172f66f-43f1-4c0f-a441-130d0d669ff4)

### Use the Chat Template to Carry a Conversation
Our goal is to build a chat assistant using the Llama models. You will learn how to make the model follow conversations.

We put the <|start_header_id|>assistant<|end_header_id|>\n\n at the end to indicate that we expect the model to continue after that and fill in the words the “assistant” (i…e, the LLM itself) would say.

wasmedge --dir .:. \
       --nn-preload default:GGML:AUTO:Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf \
       llama-simple.wasm \
       --prompt "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n You are a history teacher. 
Answer briefly. <|eot_id|><|start_header_id|>user<|end_header_id|>\n\n What is Robert Oppenheimer's most important achievement? <|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
![image](https://github.com/user-attachments/assets/3db30d4e-0a2e-482d-8314-0e0f9b38c347)

### Create a Web Service API

Create an OpenAI-compatible API service using the Llama model. The API service will allow your private Llama models to work with a large ecosystem of tools.

Step-1. Get the Wasm file
cd ~
curl -LO https://github.com/LlamaEdge/LlamaEdge/releases/latest/download/llama-api-server.wasm

Step-2. Start the web server on port 8080. It provides and OpenAI-compatible API service.
nohup wasmedge --dir .:. \
  --nn-preload default:GGML:AUTO:Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf \
  llama-api-server.wasm \
  --prompt-template llama-3-chat \
  --socket-addr 0.0.0.0:8080 &
Send a prompt to the /chat/completions endpoint to start a conversation. Note that the prompt text now follows OpenAI’s JSON format. It is no longer the [INST] delimited prompt text required by plain Llama2 models.

curl -X POST http://localhost:8080/v1/chat/completions \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"system", "content": "You are a high school science teacher. Explain concepts in very simple English."}, {"role":"user", "content": "What is Mercury?"}]}'
![image](https://github.com/user-attachments/assets/8fbd2ca0-7308-4aec-88c6-4578b3e3a58d)

### Create a Chatbot

Create a web-based chatbot for users to converse with the Llama model you deployed. The chatbot UI allows a large number of users to access the private Llama model.

Step-1. Download the chatbot UI assets

curl -LO https://github.com/LlamaEdge/chatbot-ui/releases/latest/download/chatbot-ui.tar.gz
tar xzf chatbot-ui.tar.gz
rm chatbot-ui.tar.gz

Step-2. Start the API server again.
nohup wasmedge --dir .:. \
  --nn-preload default:GGML:AUTO:Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf \
  llama-api-server.wasm \
 --prompt-template llama-3-chat \
 --socket-addr 0.0.0.0:8080 &

Step-3. Start chatting with the LLM
http://localhost:8080/
![image](https://github.com/user-attachments/assets/dc1cc677-823c-4eae-a4ac-a2251e9981bf)

## Add Knowledge to the Chatbot
A commonly used technique to add knowledge and context to an LLM application is called Retrieval Augmented Generation (RAG). The first step is to prepare the external knowledge base by building an embedding vector store.

A specially trained embedding model “reads” and digests a body of text, and then turns each segment of knowledge into a vector (an embedding).
The embeddings are then stored in a searchable database.

Then, for each user question, the app does the following.

The user question itself is turned into a vector using the same embedding model.
The application searches the vector database for embeddings that are related to the question (Retrieval).
The app puts the original text associated with the search result into the prompt (Augmented).
The LLM answers the user question based on the prompt (Generation).

### Create a Vector Database for External Knowledge
Create a collection of vector embeddings from a body of knowledge to supplement the LLM. The generation and management of vector embeddings is the basis of the RAG approach for LLM applications. The LLM prompts are later supplemented by related source text retrieved from the vector database.

Step-1: Start the Qdrant vector database server.
docker pull qdrant/qdrant

Step-2: Create a ./qdrant_storage directory on your local computer for the vector database’s data storage. Create a ./qdrant_snapshots directory for the collection snapshots.

mkdir qdrant_storage
mkdir qdrant_snapshots

Then, start the Qdrant server.

docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    -v $(pwd)/qdrant_snapshots:/qdrant/snapshots:z \
    qdrant/qdrant
    
Step-3: Create a collection
curl -X PUT 'http://localhost:6333/collections/chemistry' \
  -H 'Content-Type: application/json' \
  --data-raw '{
    "vectors": {
      "size": 768,
      "distance": "Cosine",
      "on_disk": true
    }
  }'

Step-4: Create a text file for external chemistry knowledge.
A chemistry textbook is a great resource for relevant domain knowledge. 
curl -LO https://huggingface.co/datasets/gaianet/chemistry/resolve/main/chemistry.txt

Step-5: Download the embedding model. The embedding model is a special class of LLMs that are trained for generating embedding vectors, instead of sentence completions, from an input text. 
curl -LO https://huggingface.co/gaianet/Nomic-embed-text-v1.5-Embedding-GGUF/resolve/main/nomic-embed-text-v1.5.f16.gguf

Step-6: Create embeddings and save them to the vector database
Run the Wasm application using WasmEdge. It reads in the chemistry.txt file in chunks separated by empty lines, passes each chunk to the embedding LLM to generate an embedding voctor, and then stores the vector to the local Qdrant’s chemistry_book collection. The collection’s vector size is 768.
The default model is the nomic-embed-text-v1.5.f16.gguf file, which you downloaded previously. 

wasmedge --dir .:. --nn-preload embedding:GGML:AUTO:nomic-embed-text-v1.5.f16.gguf paragraph_embed.wasm embedding chemistry 768 chemistry.txt --ctx_size 8192

The following command generates a snapshot for the vector collection.

curl -X POST 'http://localhost:6333/collections/chemistry/snapshots'

![image](https://github.com/user-attachments/assets/d16e97c7-65bd-4e3c-bc4e-477b471bf9dd)

### Start a RAG API Server

Start an OpenAI-compatible API server that automatically searches the embeddings vector store and injects the related text into the system prompts. The API server allows ecosystem tools to work directly with the RAG-enhanced LLM. The ecosystem tools do not need to explicitly manage our vector database.

Step-1: Make sure that the Qdrant server is running with the chemistry collection.

curl 'http://localhost:6333/collections/chemistry'

Step-2: Download the cross-platform Wasm application for the server.

curl -LO https://github.com/LlamaEdge/rag-api-server/releases/latest/download/rag-api-server.wasm

Step-3: Start the server with the Qdrant database connection and collection name

wasmedge --dir .:. \
 --nn-preload default:GGML:AUTO:Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf \
 --nn-preload embedding:GGML:AUTO:nomic-embed-text-v1.5.f16.gguf \
 rag-api-server.wasm \
 --prompt-template llama-3-chat,embedding \
 --model-name llama31,nomic-embed \
 --ctx-size 16384,8192 \
  --batch-size 128,8192 \
  --rag-policy system-message \
 --rag-prompt "Use the following pieces of context to answer the user's question.\nIf you don't know the answer, just say that you don't know, don't try to make up an answer.\n----------------\n" \
 --qdrant-url http://127.0.0.1:6333 \
 --qdrant-collection-name "chemistry" \
 --qdrant-limit 1 \
 --qdrant-score-threshold 0.2 \
 --socket-addr 0.0.0.0:8080 \
  --log-prompts

Step-4: Test the API. From another terminal window, you can send an OpenAI-compatible chat request to 127.0.0.1:8080.

curl -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"system", "content": "You are a helpful, respectful and honest assistant. Always answer as short as possible, while being safe."}, {"role":"user", "content": "What is Mercury?"}]}'

### Create a Web-based Chatbot
Create a web UI to make the RAG chatbot service available to the public via the web.
A web-based chatbot UI is the easiest way for non-technical users to access the LLM application.

Step-1: Install a Gaia node.

curl -sSfL 'https://github.com/GaiaNet-AI/gaianet-node/releases/latest/download/install.sh' | bash

Step-2: Initialize the Gaia node with a selection of LLM models, vector snapshots, and prompts.

gaianet init --config https://raw.githubusercontent.com/GaiaNet-AI/node-configs/main/llama-3.1-8b-instruct_chemistry/config.json

Step-3: Start the Gaia node (gaianet start)
The output on the console is as follows.

    LlamaEdge API Server started with pid: 73855

    Verify the LlamaEdge-RAG API Server. Please wait seconds ...

    * LlamaEdge-RAG API Server is ready.

[+] Starting gaianet-domain ...

    gaianet-domain started with pid: 73875

    The GaiaNet node is started at: https://0x0e3229f7805d81d46cf0ca1ae89b524c2cd44c93.us.gaianet.network

>>> To stop the GaiaNet node, run the command: gaianet stop <<<

Note down the https://0x0e3229f7805d81d46cf0ca1ae89b524c2cd44c93.us.gaianet.network URL on your command line console.

Load the URL http://localhost:8080/chatbot-ui/index.html on your computer running the Gaia node. You can ask the chatbot questions about chemistry!

![image](https://github.com/user-attachments/assets/0798aa74-6981-4953-9789-30a4d0761fbd)

## Fine-Tune the Llama Model

We will discuss a very popular form of fine-tuning, supervised fine-tuning (SFT). This technique uses a set of question-and-answer pairs to train the model to answer those questions with the given answers. The LLMs can then generalize to answer similar questions with similar answers. The training result is a set of low-rank adaptation (LoRA) weights that can be applied to the original model. The SFT + LoRA approach is both effective and computationally efficient.

Furthermore, fine-tuning does not generally add new knowledge to the LLM; RAG is much better at adding knowledge. Instead, you should consider fine-tuning when you need to accomplish any of the following.

Bring focus to specific knowledge or facts when answering questions. That helps reducing hallucination.
Teach the model a specific speaking style, such as conversation-following or being either concise or verbose in its explanations.
Teach the model to reason. That is done by fine-tuning the model with carefully curated examples and instructions, such as the Orca and Dolphin datasets.
Align the model to avoid discussing certain topics. That’s what do in this liveProject!

A good case in point is the Llama models themselves. The base Llama models are not even capable of conversations. If you ask it "What is your first name?", it is likely to answer “What is your last name” – as those sentences are most likely to appear together in the pre-training corpus of data. The Llama “chat” models are already fine-tuned models. They are tuned by conversation data. The process is called Instruction Tuning.

### Create the Training Dataset

The examples in the fine-tuning dataset will be in JSON format, with fields to match the instruction, input and output in the Alpaca template. For instance, here is an example that teaches the LLM to recognize the user input as a valid chemistry question.

{
    "instruction": "Determine if the input is a question related to chemical science. If it is, return a JSON object with the 'valid' field set to true. If it is not, return a JSON object with the 'valid' field set to false, and the 'reason' field set to the type of question or statement it is.",
    "input": "Can elements be broken down into simpler substances?",
    "output": "{'valid': true}"
}

Here is another example that teaches the LLM to recognize that the user query is invalid because it discusses politics instead of chemistry.

{
    "instruction": "Determine if the input is a question related to chemical science. If it is, return a JSON object with the 'valid' field set to true. If it is not, return a JSON object with the 'valid' field set to false, and the 'reason' field set to the type of question or statement it is.",
    "input": "What motivated Dick Cheney to support Kamala Harris, given their differing political backgrounds?",
    "output": "{'valid': false, 'reason': 'politics'}"
}

Step-1: Download the open-source chemistry textbook
curl -LO https://huggingface.co/datasets/juntaoyuan/validate_chemistry_question/resolve/main/chemistry-by-chapter.txt

Step-2: Generate chemistry-related questions and the validator LLM’s desired responses in the Alpaca model’s JSON template format. We will use a simple Python script to do this.

curl -LO 
https://huggingface.co/datasets/juntaoyuan/validate_chemistry_question/resolve/main/generate_valid_questions.py

You can review the Python script. It takes text chapters from a the above chemistry textbook, and ask an LLM service to generate related questions based on those chapters. Run it.

python generate_valid_questions.py chemistry-by-chapter.txt valid_examples.json

Step-3: Create a text file filled with news headings and stories unrelated to chemistry. We will use it as the source to generate questions the validator model should reject. You can see an example here. Name the file politics.txt.

Step-4: Generate questions and validator LLM’s desired responses in Alpaca JSON template format. We will use a simple Python script to do this.

curl -LO https://huggingface.co/datasets/juntaoyuan/validate_chemistry_question/resolve/main/generate_invalid_questions.py

You can review the Python script. It takes news headlines from the politics.txt, and ask an LLM service to generate questions. Run it.

python generate_invalid_questions.py politics.txt invalid_examples.json

Combine the examples JSON files into a single JSON file for the dataset.

echo -n "[" > finetune.json
cat valid_examples.json >> finetune.json
sed '$s/,$//' invalid_examples.json | cat >> finetune.json
echo -n "]" >> finetune.json

Step-5: Create new Dataset in your Hugging Face account and upload this dataset

### Fine-tune the Model

Run the notebook section by section from the top. It will start by installing the Unsloth software on your Google Colab’s GPU VM. Our notebook uses the “Llama 3.1 8b instruct” LLM as the base model to fine-tune.

https://colab.research.google.com/drive/14cMzqpOiNVBa_MvDC9h5C98VB1YqRZlO?usp=sharing

Wait until the GGUF model files are all uploaded to Hugging Face (see the green bars in the screenshot below).

### Run an inference API server for the fine-tuned Model
As we have seen in previous liveProjects in this series, there are two easy ways to run an LLM locally on your own device. You can use LlamaEdge or Gaia.

Step-1: Install a Gaia node

curl -sSfL 'https://github.com/GaiaNet-AI/gaianet-node/releases/latest/download/install.sh' | bash

Step-2: Initialize the Gaia node with the fine-tuned validator LLM.
Edit the ~/gaianet/config.json file to point the chat model to one of the GGUF model files we uploaded in the previous milestone. You could also update the system-prompt to the fine-tuning instruction. Here is an example.

{
  "address": "",
  "chat": "https://huggingface.co/juntaoyuan/validate_chemistry_question/resolve/main/unsloth.Q5_K_M.gguf",
  "chat_batch_size": "128",
  "chat_ctx_size": "1024",
  "description": "Llama-3.1-8B-Instruct",
  "domain": "us.gaianet.network",
  "embedding": "https://huggingface.co/gaianet/Nomic-embed-text-v1.5-Embedding-GGUF/resolve/main/nomic-embed-text-v1.5.f16.gguf",
  "embedding_batch_size": "8192",
  "embedding_collection_name": "default",
  "embedding_ctx_size": "8192",
  "llamaedge_port": "8080",
  "prompt_template": "llama-3-chat",
  "qdrant_limit": "1",
  "qdrant_score_threshold": "0.2",
  "rag_policy": "system-message",
  "rag_prompt": "Answer the user request using the following context. \n----------------\n",
  "reverse_prompt": "",
  "snapshot": "",
  "system_prompt": "Determine if the input is a question related to chemical science. If it is, return a JSON object with the 'valid' field set to true. If it is not, return a JSON object with the 'valid' field set to false, and the 'reason' field set to the type of question or statement it is."
}

Run the init command to download the fine-tuned validator model.

gaianet init

Step-3: Start the Gaia node

Note down the https://0x0e3229f7805d81d46cf0ca1ae89b524c2cd44c93.us.gaianet.network URL on your command line console.

Chat with the validator LLM on your local machine.
Load the URL http://localhost:8080/chatbot-ui/index.html on your computer running the Gaia node. You can ask a politics question and see how it responds!

Access the validator LLM via an API. The AI chemistry teach chatbot app will access the validator LLM by its API to check if a user query is valid. You can try the API from the command line using curl. Make sure that you replace the NODE-ID with the URL printed from your Gaia start command.

curl -X POST https://NODE-ID.us.gaianet.network/v1/chat/completions \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"system", "content": "Determine if the input is a question related to chemical science. If it is, return a JSON object with the 'valid' field set to true. If it is not, return a JSON object with the 'valid' field set to false, and the 'reason' field set to the type of question or statement it is."}, {"role":"user", "content": "Was Donald Trump a good president?"}]}'


![image](https://github.com/user-attachments/assets/ef75433e-35bc-4f5a-9a76-cc5f8b6ebfac)

Publicly accessible url

![image](https://github.com/user-attachments/assets/3310a8ae-5bcc-42bd-ba11-5d93f9d97482)

















































