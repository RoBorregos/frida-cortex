⚙️ Configuring the Environment
🧩 Client Options
The ChromaDB client can be:

Ephemeral

Persistent

Hosted via Docker container

🔄 Resetting Collections
To reset the collections, run the appropriate command depending on your operating system:

🪟 Windows
bash
Copy
Edit
python .\command_interpreter\embeddings\chroma_adapter.py
🐧 Linux / macOS
bash
Copy
Edit
python3 ./command_interpreter/embeddings/chroma_adapter.py
⚠️ Note: Resetting will recreate the collections from scratch.

🧠 Embedding Model Configuration
By default, a local model is used for generating embeddings. However, ChromaDB supports integration with online providers via their APIs, such as:

🤗 Hugging Face

Each provider requires a different configuration approach.

📐 Distance Function in Vector Space
You can choose one of the following distance functions for your embedding space:

Function	Description	Use Case Example
l2	Squared L2 norm	Physical/spatial comparison
ip	Inner product	Ranking, recommender systems
cosine	Cosine similarity	Semantic similarity (✅ best fit)

You can configure this in the _get_or_create_collection method inside chroma_adapter.py.

⚠️ Important
To see changes in results, you must reset the collections after modifying the configuration