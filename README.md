# Enterprise GPU-Accelerated Visual Fault Diagnosis & Similarity Search System  
**Deep Learning + FAISS-GPU + LLM-RAG for Industrial Defect Analysis**



## Overview
A production-ready reference architecture for **enterprise manufacturing fault detection**, combining:

- TensorFlow-GPU VGG16 classifier  
-  FAISS-GPU similarity search index  
-  Streamlit visual QA UI  
-  LLM-based Fault Report Retrieval (RAG)  
-  Automated PDF fault report ingestion  
-  Docker + GPU compute support  

#### This repo demonstrates how enterprises can leverage **vision AI + vector search + knowledge-augmented LLMs** to automate:  

| Capability | Description |
|---|---|
Image defect classification | CNN (VGG16) trained on industrial fault dataset  
Visual similarity search | FAISS-GPU embedding index for finding similar failed parts  
LLM RAG fault assistance | LLM retrieves PDF maintenance reports for root-cause troubleshooting  
Interactive UI | Streamlit search + query panel  
Production-ready | GPU containers, config modules, modular code  

---

##  Architecture Overview

### System Components
| Component | Technology |
|---|---|
Image Preprocessing | TF Data / OpenCV  
Feature Extraction | VGG16 (TensorFlow-GPU)  
Vector Index | FAISS-GPU (IVF / Flat supported)  
Search API | Python + FAISS + Fast Retrieval Pipeline  
Knowledge Base | PDF fault reports → embeddings store  
RAG Engine | Sentence-Transformers + FAISS + LLM responses  
User Interface | Streamlit dashboard  
Compute | CUDA / NVIDIA container stack  

---

##  High-Level Pipeline

```python
flowchart LR
A[Upload Fault Image] --> B[GPU Preprocessing]
B --> C[VGG16 Embedding Extraction]
C --> D[FAISS-GPU Index Search]
D --> E[Retrieve Similar Fault Images + Labels]

E --> F[LLM PDF Knowledge Retrieval]
F --> G[Return Root Cause Analysis + Fix Steps]
```

## Repo structure
```
enterprise-image-fault-diagnosis/
│
├── data/
│   ├── raw/                   
│   ├── processed/             
│   ├── faiss_index/           
│   └── fault_reports/         
│       └── vector_store/      
│
├── src/
│   ├── config.py
│   ├── extract_embeddings.py
│   ├── train_classifier.py
│   ├── build_faiss_index.py
│   ├── query_similar_images.py
│   ├── pdf_ingest.py
│   ├── rag_query.py
│
├── app/
│   └── streamlit_visual_search.py
...

```

## Quickstart (GPU Required)

#### <ins>Install NVIDIA stack</ins>

```python
sudo apt install nvidia-driver-530 nvidia-cuda-toolkit
```

#### <ins>Install Python deps</ins>

```python
pip install -r requirements.txt
```

#### <ins>Preprocess + Train</ins>
```python
python src/train_classifier.py
python src/extract_embeddings.py
python src/build_faiss_index.py
```

#### Launch UI
```python
streamlit run app/streamlit_visual_search.py
```

---

## 
| Industry | AI Use Case |
|----------|-------------|
| Automotive | Visual defect detection of machined parts |
| Semiconductor | Wafer anomaly classification |
| Aerospace | Quality control and maintenance diagnostics |
| Energy | Turbine blade inspection & report lookup |
| Manufacturing | Line-side failure explanation and fix guidance |S



