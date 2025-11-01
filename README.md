# Enterprise GPU-Accelerated Visual Fault Diagnosis & Similarity Search System  <sub>(Deep Learning + FAISS-GPU + LLM-RAG for Image based Industrial Product Defect Analysis)</sub>




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

#### <ins>Launch UI</ins>
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

---

## Tech Highlights

| Stack | Tools |
|-------|-------|
| Deep Learning | TensorFlow-GPU, VGG16 |
| Vector Search | FAISS-GPU (Flat / IVF) |
| RAG | Sentence-Transformers + PDF loader |
| UI | Streamlit |
| Container | NVIDIA CUDA Docker |

---

## 

| Future Feature | Value |
|---------------|-------|
| YOLO segmentation | Bounding-box localized defect search |
| Realtime edge inference | Robotics + smart cameras |
| SAP / Maxivo connector | Automated maintenance logs |
| API Gateway + K8s | Production deployment pipeline |


---



## Summary

#### This repository demonstrates a real-world enterprise AI system combining:

- Computer vision defect detection

- GPU-accelerated similarity search

- PDF maintenance manual intelligence

- RAG + LLM technician assistant

<ins>Perfect for</ins>:

- Manufacturing AI innovation teams

- Maintenance automation & smart factory projects

- Industrial ML upskilling & PoC deployments

- Perfect for manufacturing AI teams, digital factories, MLOps training, and innovation pilots.

---

