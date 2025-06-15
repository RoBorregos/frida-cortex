Configuring the enviroment:

-Client can be ephimeral, persistent or hosted via a docker container.

-You can reset the collections by running:

python .\command_interpreter\embeddings\chroma_adapter.py for Windows

python3 ./command_interpreter/embeddings/chroma_adapter.py for Linux

By default there is a local model for embeddings that is used, but it can support online providers of embeddings using their API, each provider has to be configured differently. 
For example, Huggin Face.https://docs.trychroma.com/integrations/embedding-models/hugging-face-server

The distance function of the embedding space can be one of three: squared ("l2"), inner product ("ip") or cosine similarity ("cosine"), each has a use case but in ours cosine similarity is the best fit.
To edit this configuration, you can go to _get_or_create_collection in chroma_adapter.py

BEWARE THAT TO SEE CHANGES IN THE RESULTS YOU NEED TO RESET COLLECTIONS