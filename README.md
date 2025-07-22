This Paper Presents a Novel approach for the
 AACUsers to communicate effectively by us
ing fine-tuned large language models. we
 started by integrating the EmpatheticDialogues
 dataset with a Retrival Augmentedd Genera
tion(RAG) Architecture, the system is capa
ble of generated contextually aware reponses
 which will helps the user to communicate. On
 top of this we introduced a personalization mod
ule that adapts responses to user-specific prefer
ences. The model is fine-tuned using LLaMA
 3.2, enabling to capture most of context. This
 allows the AAC system to operate efficiently
 on accessible software. The proposed archi
tecture significantly improves expressiveness
 and coherence of AAC outputs, enabling in
 more human way conversations. we evaluate
 the model using Perplexity, BLEU Scores and
 Human Feedback, demonstrating in real-world
 communication scenarios.


 <img width="181" height="47" alt="image" src="https://github.com/user-attachments/assets/5984265b-2655-4cf7-ab97-9ace25380306" />


 1. Extract text from his profile
 2. Split text into chunks and generate sentence
 embeddings.
 3. Index chunks using FAISS.
 4. Retrieve top-k similar chunks for a query.
 5. Combine retrieved context with query and
 pass to Mistral for response generation.

<img width="164" height="82" alt="image" src="https://github.com/user-attachments/assets/1aeb322a-9b79-4666-b540-2762f7990486" />


