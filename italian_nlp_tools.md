# **Architectural Optimization of Tokenizers and Embeddings for Multi-Platform Italian Social Media Analytics on Google Cloud Platform**

## **Temporal Analysis of Creator Content and Causal Community Inference**

Analyzing the historical evolution of a creator's community across Instagram, TikTok, Facebook, and YouTube demands an analytical architecture capable of processing highly heterogeneous data streams. The primary analytical dataset consists of multi-platform post metadata—such as publication timestamps, content types, caption text, and structural features—paired with large volumes of unstructured audience comments extracted from official platform application programming interfaces (APIs). To establish a baseline for identifying performance anomalies, the dataset also includes historical post and comment data from comparable YouTube channels.  
To track community evolution and establish causal relationships between creator behavior and audience engagement, the pipeline must analyze how specific content decisions affect community sentiment and semantic alignment over time. For instance, when the creator shifts from highly structured educational videos to informal, short-form video formats, this change represents a shift in content features.  
By analyzing the corresponding comment streams, the system can measure how these changes affect community topics. This analytical process requires extracting text features that capture subtle shifts in audience sentiment, slang usage, and semantic topics.  
By vectorizing these comments and comparing their spatial distributions to competitor channels over sequential time windows, the analyst can isolate broader platform-wide shifts from localized community reactions.

| Feature Type | Source Platform | Extracted Data Fields | Analytical Purpose | Causal Inference Mapping |
| :---- | :---- | :---- | :---- | :---- |
| **Creator Content Features** | YouTube, TikTok, Instagram, Facebook | Video length, caption text, platform source, posting frequency, hashtag density | Quantifying shifts in creator content strategy and format over time. | Direct correlation with shifts in comment sentiment and topic density. |
| **Audience Response Features** | All Platforms | Comment text, reply count, likes, publishing timestamp, emoji distribution | Extracting audience sentiment, sarcasm, and community topics. | Serving as the primary dependent variables to measure audience response. |
| **Comparable baseline Controls** | YouTube API | Peer channel post metrics, competitor comment streams, temporal engagement rates | Controlling for macro-level platform shifts and seasonal trends. | Distinguishing creator-driven community shifts from platform-wide algorithmic changes. |

To support this causal modeling, the underlying natural language processing pipeline must parse unstructured, informal Italian text with high semantic fidelity. Because social media text is highly informal, standard English-centric configurations often perform poorly, making language-specific optimization necessary.

## **Preprocessing Noisy Social Media Text and Resolving Italian Syntax**

The linguistic style of Italian social media is highly complex, featuring non-standard grammar, platform-specific abbreviations, and extensive orthographic variation. To prevent these factors from degrading downstream embedding quality, the text must pass through a specialized preprocessing pipeline before tokenization.  
A primary grammatical challenge in Italian is the use of clitic pronouns and articulated prepositions. Clitics are unstressed pronouns attached to the end of verbs (such as *dandolo*, meaning "giving it"), while articulated prepositions contract prepositions with articles into single words (such as *dalla*, meaning "from the").  
If left unaddressed, these contracted forms are tokenized as single, unique words, which artificially inflates vocabulary size and obscures grammatical relationships.  
The preprocessing workflow must split these compounds into their constituent components (for example, deconstructing *dalla* into *da* and *la*) to align with standard grammatical structures.  
Raw Multi-Platform Comment  
  │  
  ├──► Orthographic Normalization (Collapse letter duplications: "belllooo" ──► "bello")  
  │  
  ├──► Censored Bad-Word Restoration (Prefix/Suffix matching: "c\*zzo" ──► "cazzo")  
  │  
  ├──► Dynamic Hashtag Segmentation (Recursive dictionary boundary splitting)  
  │  
  ├──► Emoji Semantic Parsing (Map emojis to Italian descriptive words)  
  │  
  └──► Grammatical Decomposition (Decontract clitics & articulated prepositions)  
        │  
        └──► Cleaned Italian Text ready for Tokenizer & Embedding Model

Social media interactions are also characterized by orthographic noise, including vowel or consonant elongation used for emotional emphasis, such as *caaaaane* ("dog"). The preprocessing workflow uses regular expressions to collapse consecutive repeated letters to a maximum of two, preventing these elongations from creating out-of-vocabulary terms.  
Additionally, comments containing censored profanity must be restored. A Python module scans words containing non-alphabetic characters (such as asterisks or symbols used to bypass content filters) and uses boundary-matching algorithms to map the censored terms back to standard Italian terms, ensuring these emotional signals are preserved for analysis.  
Hashtags present a similar challenge because multiple words are concatenated without spaces. A recursive splitting algorithm matches these strings against an Italian dictionary, sorting candidate words by length to segment hashtags into readable phrases (such as splitting *\#creatorecontenuto* into *creatore contenuto*).  
Finally, because emojis carry significant emotional context on platforms like Instagram and TikTok, simply stripping them removes valuable sentiment data.  
Instead, the preprocessing pipeline translates emojis into their descriptive Italian text equivalents, preserving indicators of irony, enthusiasm, or frustration for subsequent sentiment and sarcasm classification.

| Raw Social Media Comment | Preprocessed Text Output | Applied Preprocessing Operations |
| :---- | :---- | :---- |
| "Bellllissimo video\!\!\! 😍 \#da\_non\_perdere" | "bellissimo video faccina con occhi a cuore da non perdere" | Consonant reduction, emoji translation, hashtag segmentation, lowercasing. |
| "Non l'ha fattto apposssito, regalaglielo..." | "non lo ha fatto apposito regala a lui lo" | Articulated contraction splitting, letter de-duplication, clitic decomposition. |
| "Questo creator è una m\*\*da\!\! 😡" | "questo creator è una merda faccina arrabbiata" | Censored profanity restoration, punctuation removal, emoji translation. |

## **Tokenizer Efficiency and Vocabulary Adaptation for Italian**

A central challenge when analyzing non-English social media text with large language models is the efficiency of the underlying tokenizer. Tokenizer performance is measured by its fertility, which represents the average number of sub-word tokens generated per word.  
Because many top-tier language models use English-centric tokenizers, they tend to over-segment Italian words.  
For instance, the contraction *l'intelligenza* ("the intelligence") is often split into three distinct tokens (l, ', and intelligenza), which wastes context window capacity and degrades model inference speeds.  
"l'intelligenza" (Italian Word)  
   │  
   ├──► English-Centric Tokenizer (High Fertility: 3 Tokens)  
   │     └───► \[ "l", "'", "intelligenza" \]  
   │  
   └──► Italian-Adapted Tokenizer (Low Fertility: 1 Token)  
         └───► \[ "l'intelligenza" \]

To address this issue, vocabulary adaptation techniques can be used to customize existing model tokenizers for the Italian language. These methods replace the default tokenizer and its corresponding embedding layer, adjusting the model parameters to better represent Italian text.  
The primary approaches for vocabulary adaptation include Fast Vocabulary Transfer (FVT), which initializes new target-language token embeddings by averaging the representations of their source-language sub-tokens.  
Alternatively, Semantic Alignment Vocabulary Adaptation (SAVA) uses an Italian-native helper model (such as Minerva-3B) to learn a neural mapping that aligns the embedding spaces, providing highly accurate token initializations.  
A third approach, Geometric Similarity Mapping (CLP), maps token embeddings by analyzing structural symmetries between the source and target token spaces.

| Adaptation Method | Underlying Tokenizer Source | Italian Vocabulary Size | Llama-3.1-8B Parameter Shift | Italian Token Fertility Reduction | Downstream Performance Impact |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **Standard Baseline** | Original English Llama-3.1 | 128,256 | Baseline (8.03 Billion) | 0% (High baseline fertility) | baseline reference performance. |
| **Fast Vocabulary Transfer (FVT)** | Minerva-3B Tokenizer | 32,768 | Reduced to 7.25 Billion (10% reduction) | **16% Reduction** | Rapid performance recovery within 400 batches. |
| **Semantic Alignment (SAVA)** | Minerva-3B Tokenizer | 32,768 | Reduced to 7.25 Billion (10% reduction) | **16% Reduction** | Strongest downstream scores, outperforming baseline on BOOLQ. |
| **Mistral-7B-v0.1 Adaptation** | Minerva-3B Tokenizer | 32,768 | Constant at 7.25 Billion | **25% Reduction** | Reaches baseline performance after 2 billion tokens of training. |

When adapting Llama-3.1-8B using the Minerva tokenizer—which was trained from scratch on balanced Italian and English web data—the vocabulary size is reduced by 75%, from 128k down to approximately 32k tokens. This vocabulary pruning removes nearly 1 billion parameters from the embedding layers, reducing the total model size by 10% (shrinking the parameter count to 7.25 billion).  
This parameter reduction decreases the model's memory footprint and processing requirements, while reducing Italian token fertility by 16% on Llama-3.1-8B and 25% on Mistral-7B-v0.1.  
Furthermore, models adapted using SAVA and FVT recover their baseline performance rapidly through language-adaptive pre-training (LAPT) on Italian corpora, requiring 80% less training time to converge compared to random initialization methods.

## **Comparative Analysis of Embedding Models on GCP**

Selecting an embedding model for a production pipeline on Google Cloud Platform requires balancing semantic accuracy, context window size, infrastructure requirements, and operational costs. The project can utilize either native Google APIs or open-weight models deployed on dedicated GPU instances via Vertex AI Model Garden.

### **Native Google Cloud Embedding Models**

Native models are managed, serverless options that are highly scalable and require zero infrastructure management.

* text-multilingual-embedding-002: Google's primary model for multilingual tasks, supporting over 100 languages with an output dimensionality of up to 768 dimensions and a 2,048-token context window.  
* gemini-embedding-001: A state-of-the-art embedding model that unifies English, code, and multilingual tasks. It supports a 2,048-token context window and adjustable output dimensions up to 3,072.

These native models natively support "Task Types," which allow developers to optimize the embedding space for specific downstream applications.  
For example, specifying the CLUSTERING task type optimizes the vectors to group semantically similar content, making it ideal for identifying emergent themes in comments.  
Conversely, the CLASSIFICATION task type organizes the embedding space to train small categorization models, which can be used to sort comments by sentiment or platform origin.

### **Deployed Open-Weight Models on GCP**

For large datasets, custom models, or strict data privacy requirements, open-weight models can be deployed on GCP GPU instances using Vertex AI Model Garden.

* jina-embeddings-v3: A 570M-parameter multilingual model designed for scalable search and content analysis. It supports an 8,192-token context window and uses Matryoshka Representation Learning (MRL), allowing output dimensions to be truncated from 1,024 down to 32 while preserving high retrieval accuracy. It also features task-specific Low-Rank Adaptation (LoRA) adapters—for clustering, classification, and retrieval—which can be selected via parameters at inference time.  
* bge-m3: A 568M-parameter multilingual model with an 8,192-token context window. It can output three representations in a single forward pass: standard dense vectors, sparse vectors (lexical token weights similar to BM25), and multi-vector representations (ColBERT-style late interaction), supporting advanced hybrid search pipelines.  
* qwen3-embedding:8b: An 8B-parameter open-weight model that ranks first on the MTEB multilingual leaderboard as of 2025\. It supports an 8,192-token context window and configurable output dimensions from 32 to 4,096, making it suitable for long-document analysis and complex semantic tasks.  
* embeddinggemma-300m: A lightweight, 300M-parameter model based on Gemma 3, optimized for resource-efficient deployment across 100+ languages.  
* nickprock/mmarco-bert-base-italian-uncased: A sentence-transformers model fine-tuned specifically for Italian semantic search. It achieves self-reported accuracy of 55.06% on the Italian MassiveIntentClassification benchmark and a Spearman correlation of 69.44% on the STS22 Italian benchmark, making it a reliable baseline for domain-specific tasks.

| Model Name | Parameters | Max Context | Default Dimensions | Task Optimization Support | GCP Deployment Architecture | API Pricing (per 1M input tokens) | Italian Semantic & MTEB Performance Notes |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| text-multilingual-embedding-002 | *Closed* | 2,048 Tokens | 768 | Native via task\_type parameter | Serverless API / BigQuery ML remote model | \~$0.025 / 1M tokens ($0.00002 per 1,000 chars) | Highly stable performance across multilingual benchmarks. |
| gemini-embedding-001 | *Closed* | 2,048 Tokens | 3,072 | Native via task\_type parameter | Serverless API / BigQuery ML remote model | $0.12 / 1M tokens | State-of-the-art multilingual and code retrieval. |
| jina-embeddings-v3 | 570 Million | 8,192 Tokens | 1,024 | 5 task LoRA adapters built-in | Deployed on NVIDIA L4 GPU via Model Garden | $0.02 / 1M tokens (via managed API) | Outstanding price-to-performance ratio; highly efficient. |
| bge-m3 | 568 Million | 8,192 Tokens | 1,024 | Dense, sparse, & multi-vector outputs | Deployed on NVIDIA L4 GPU via Model Garden | Free (Self-hosted GPU costs apply) | Excellent for hybrid search over noisy social comments. |
| qwen3-embedding:8b | 8 Billion | 8,192 Tokens | 4,096 | Native instruction-aware tuning | Deployed on NVIDIA L4 or A100 GPU | Free (Self-hosted GPU costs apply) | Ranks \#1 on MTEB Multilingual leaderboard. |
| embeddinggemma-300m | 300 Million | 2,048 Tokens | 768 | General search, retrieval & similarity | Serverless API or local CPU/GPU | Free (Self-hosted GPU costs apply) | Highly resource-efficient; ideal for low-latency tasks. |
| nickprock/mmarco-bert-base-italian-uncased | 110 Million | 512 Tokens | 768 | Basic pooling-based embeddings | Hugging Face container on Vertex AI | Free (Self-hosted GPU costs apply) | Italian-specialized; provides a strong local baseline. |

## **Spatial Optimization: Matryoshka Representation Learning and Dimension Truncation**

A key consideration when managing large-scale vector databases is the cost of storage and computation, both of which scale linearly with the dimensionality of the embeddings.  
For example, storing 10 million comments with 1,024-dimensional float32 vectors requires approximately 40 GB of vector database storage, which increases database latency and infrastructure costs.  
To address this challenge, modern models like jina-embeddings-v3 and gemini-embedding-001 incorporate Matryoshka Representation Learning (MRL). MRL is a training methodology that prioritizes information within the early dimensions of the vector.  
This allows developers to truncate the embeddings to smaller sizes (such as reducing vectors from 1,024 dimensions down to 256 or 128\) during database indexing.  
Matryoshka Vector (1024 Dimensions)  
 ├────── First 128 Dims ───► \[ High-density semantic core \]   ──► \~90% retrieval accuracy  
 ├────── First 256 Dims ───► \[ Additional nuance details \]    ──► \~92% retrieval accuracy  
 └────── Full 1024 Dims ───► \[ Maximum precision vector \]     ──► 100% baseline accuracy

Truncation significantly reduces storage requirements and speeds up vector search queries.  
For instance, truncating jina-embeddings-v3 from 1,024 down to 64 dimensions preserves approximately 92% of its retrieval performance while reducing the storage footprint and query computational load by over 90%.  
When configuring these dimensions, the database design should follow the data-to-dimensions ratio rule of thumb. This guidelines recommends matching the embedding dimensionality to the size of the document collection :

* For datasets with fewer than 10,000 documents, 384 dimensions are typically sufficient.  
* For datasets with 100,000 to 1,000,000 documents, 768 dimensions strike an optimal balance.  
* For datasets exceeding 1,000,000 documents, 1,024 dimensions are recommended to capture fine-grained semantic distinctions.

## **Cloud Architecture and Workflow Design on Google Cloud Platform**

To implement a scalable analytical pipeline on Google Cloud Platform, the system uses BigQuery for structured data storage, BigQuery ML (BQML) for batch embedding generation, and Cloud SQL for PostgreSQL with the pgvector extension for similarity search and spatial indexing.  
┌────────────────────────────────────────────────────────┐  
│                      Google Cloud                      │  
│                                                        │  
│  ┌─────────────────┐       BigQuery ML Connection      │  
│  │   BigQuery Table├──────────────────────────────┐    │  
│  │ (Social Comments│                              │    │  
│  └────────┬────────┘                              ▼    │  
│           │                          ┌────────────────┐│  
│           │ SQL Query                │Vertex AI API   ││  
│           │ ML.GENERATE\_EMBEDDING    │(multilingual-  ││  
│           ▼                          │ embedding-002) ││  
│  ┌─────────────────┐                 └────────┬───────┘│  
│  │Generated Vectors│                          │        │  
│  └────────┬────────┘                          │        │  
│           │ Export/Sync                       │ Vectors│  
│           ▼                                   │        │  
│  ┌────────────────────────────────────────────┼────────┘  
│  │ Cloud SQL (PostgreSQL \+ pgvector)          │  
│  │                                            │  
│  │  ┌──────────────────┐                      │  
│  │  │   HNSW Index     │◄─────────────────────┘  
│  │  │(vector\_cosine\_ops)                        
│  │  └──────────────────┘                        
│  └────────────────────────────────────────────┘

The pipeline begins by loading the extracted multi-platform comments into BigQuery.  
To use native embeddings, a remote connection is established between BigQuery and Vertex AI, and a remote model representation is initialized in BigQuery :  
SQL  
CREATE OR REPLACE MODEL \`project\_id.creator\_analysis.multilingual\_embedding\_model\`  
REMOTE WITH CONNECTION \`project\_id.us-central1.vertex\_ai\_connection\`  
OPTIONS (  
  ENDPOINT \= 'text-multilingual-embedding-002'  
);

Using this remote model, batch embedding generation is executed over the preprocessed social media comments table. This query applies the CLUSTERING task type, which optimizes the vector space for grouping semantically similar text :  
SQL  
CREATE OR REPLACE TABLE \`project\_id.creator\_analysis.generated\_comment\_embeddings\` AS  
SELECT   
  comment\_id,  
  platform,  
  creator\_name,  
  publish\_timestamp,  
  cleaned\_text,  
  ml\_generate\_embedding\_result AS embedding\_vector,  
  ml\_generate\_embedding\_status AS api\_status  
FROM   
  ML.GENERATE\_EMBEDDING(  
    MODEL \`project\_id.creator\_analysis.multilingual\_embedding\_model\`,  
    (  
      SELECT   
        comment\_id,  
        platform,  
        creator\_name,  
        publish\_timestamp,  
        cleaned\_text AS content  
      FROM   
        \`project\_id.creator\_analysis.preprocessed\_social\_comments\`  
      WHERE   
        cleaned\_text IS NOT NULL AND LENGTH(cleaned\_text) \> 0  
    ),  
    STRUCT(  
      TRUE AS flatten\_json\_output,  
      'CLUSTERING' AS task\_type  
    )  
  );

Once generated, these embeddings are synchronized to a Cloud SQL for PostgreSQL instance equipped with the google\_ml\_integration and pgvector extensions to support nearest-neighbor searches.  
To enable fast search queries across millions of rows, a Hierarchical Navigable Small World (HNSW) index is constructed using the cosine distance operator :  
SQL  
\-- Initialize pgvector and model integration extensions  
CREATE EXTENSION IF NOT EXISTS vector;  
CREATE EXTENSION IF NOT EXISTS google\_ml\_integration;

\-- Initialize vector storage table  
CREATE TABLE public.creator\_comments\_spatial\_index (  
    vector\_id SERIAL PRIMARY KEY,  
    comment\_id VARCHAR(255) NOT NULL,  
    platform VARCHAR(50),  
    publish\_timestamp TIMESTAMP,  
    cleaned\_content TEXT,  
    embedding vector(768)  
);

\-- Construct HNSW spatial index  
CREATE INDEX creator\_comments\_cosine\_hnsw\_idx   
ON public.creator\_comments\_spatial\_index   
USING hnsw (embedding vector\_cosine\_ops)  
WITH (m \= 16, ef\_construction \= 64);

With the HNSW index constructed, the project team can run similarity searches to identify comment clusters that align with specific thematic categories or creator initiatives :  
SQL  
SELECT   
    comment\_id,  
    platform,  
    publish\_timestamp,  
    cleaned\_content,  
    (embedding \<=\> public.embedding\_predict(  
        'text-multilingual-embedding-002',   
        'Questa nuova tipologia di video è fantastica',   
        'us-central1'  
    )::vector) AS cosine\_distance  
FROM   
    public.creator\_comments\_spatial\_index  
ORDER BY   
    cosine\_distance ASC  
LIMIT 10;

In this query, the \<=\> operator computes the cosine distance between the comment vectors and the search string.  
By tracking how the density of these highly similar vectors changes over time, the project team can quantitatively model the causal impact of content pivots on community engagement and alignment.

## **Conclusions**

Implementing a scalable analytical pipeline on Google Cloud Platform to analyze the evolution of a creator's community across Instagram, TikTok, Facebook, and YouTube requires optimizing every stage of the data lifecycle:  
                 ┌────────────────────────────────────────┐  
                  │ Preprocessing Social Media Comments    │  
                  │ (Regex, Normalization, Profanity Fix)  │  
                  └───────────────────┬────────────────────┘  
                                      │  
                                      ▼  
                  ┌────────────────────────────────────────┐  
                  │ Tokenization & Vocabulary Adaptation   │  
                  │ (SAVA/FVT to reduce Italian fertility)  │  
                  └───────────────────┬────────────────────┘  
                                      │  
                                      ▼  
                  ┌────────────────────────────────────────┐  
                  │   GCP Embedding Platform Allocation    │  
                  └───────────────────┬────────────────────┘  
                                      │  
            ┌─────────────────────────┴─────────────────────────┐  
            ▼                                                   ▼  
┌──────────────────────────────┐                    ┌──────────────────────────────┐  
│  Dataset \< 10 Million Rows   │                    │  Dataset \> 10 Million Rows   │  
├──────────────────────────────┤                    ├──────────────────────────────┤  
│ • Native Vertex AI APIs      │                    │ • Self-Host Open-Weight      │  
│ • text-multilingual-         │                    │ • jina-embeddings-v3 on L4   │  
│   embedding-002 via BQML     │                    │ • Matryoshka dimension       │  
│ • Zero management overhead   │                    │   reduction (e.g., 256 dims) │  
└───────────┬──────────────────┘                    └───────────┬──────────────────┘  
            │                                                   │  
            └─────────────────────────┬─────────────────────────┘  
                                      │  
                                      ▼  
                  ┌────────────────────────────────────────┐  
                  │ Cloud SQL PostgreSQL \+ pgvector        │  
                  │ (HNSW Indexing with Cosine Distance)   │  
                  └────────────────────────────────────────┘

The preprocessing pipeline must normalize non-standard Italian structures, such as clitics, articulated prepositions, elongated words, and censored terms. Translating emojis rather than stripping them is critical for preserving key emotional context.  
To optimize language model performance, vocabulary adaptation techniques can be applied. Substituting English-centric tokenizers with adapted alternatives (such as SAVA with the Minerva tokenizer) reduces Italian token fertility by 16% to 25%, while pruning model parameter sizes by up to 10%, reducing VRAM requirements and accelerating inference.  
The choice of embedding architecture on GCP should be guided by dataset scale:

* For datasets with fewer than 10 million comments, native Vertex AI APIs like text-multilingual-embedding-002 integrated via BigQuery ML provide a serverless, zero-maintenance solution. These APIs support task-type parameters to automatically optimize the embedding space for downstream tasks like clustering and classification.  
* For larger datasets, deploying open-weight models like jina-embeddings-v3 or bge-m3 on dedicated NVIDIA L4 GPUs via Vertex AI Model Garden provides flat-rate pricing, lower query latency, and strict data privacy.

To minimize vector database storage and compute overhead, models supporting Matryoshka Representation Learning (MRL) should be used. Truncating embeddings to 256 or 128 dimensions reduces storage footprints and query latency by over 75% while maintaining over 92% of baseline retrieval accuracy.  
Finally, synchronizing these vectors to Cloud SQL for PostgreSQL using HNSW indices configured with cosine distance supports scalable, near-instant semantic searches, enabling the project team to quantitatively analyze community evolution and identify causal drivers over time.  
