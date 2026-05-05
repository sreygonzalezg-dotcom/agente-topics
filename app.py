from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import numpy as np
import uvicorn

app = FastAPI()

class TopicRequest(BaseModel):
    texts: List[str]
    num_topics: Optional[int] = 5

class TopicResponse(BaseModel):
    status: str
    topics: List[dict]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/")
async def extract_topics(req: TopicRequest):
    if not req.texts or len(req.texts) < 3:
        raise HTTPException(status_code=400, detail="Se necesitan al menos 3 textos para extraer tópicos")
    
    # Vectorizar textos
    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words=['que', 'de', 'la', 'el', 'en', 'y', 'a', 'los', 'las', 'un', 'una', 'por', 'con', 'para', 'como', 'mas', 'pero', 'sus', 'le', 'lo', 'del', 'esta', 'esta', 'este', 'esto', 'esta', 'año', 'día', 'mes'],
        ngram_range=(1, 2)
    )
    
    try:
        tfidf_matrix = vectorizer.fit_transform(req.texts)
        feature_names = vectorizer.get_feature_names_out()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al vectorizar: {str(e)}")
    
    # Reducir dimensionalidad con LSA (SVD)
    n_topics = min(req.num_topics, tfidf_matrix.shape[1], tfidf_matrix.shape[0] - 1)
    if n_topics < 1:
        n_topics = 1
    
    lsa = TruncatedSVD(n_components=n_topics, random_state=42)
    lsa.fit(tfidf_matrix)
    
    # Extraer palabras clave por tópico
    topics = []
    feature_array = np.array(feature_names)
    
    for topic_idx in range(n_topics):
        # Obtener las palabras más importantes para este tópico
        topic_vector = lsa.components_[topic_idx]
        top_indices = topic_vector.argsort()[-10:][::-1]
        top_words = feature_array[top_indices].tolist()
        
        # Calcular la importancia del tópico (varianza explicada)
        importance = lsa.explained_variance_ratio_[topic_idx] if topic_idx < len(lsa.explained_variance_ratio_) else 0.0
        
        # Crear nombre legible del tópico (usar las 2-3 palabras más importantes)
        topic_name = " + ".join(top_words[:3])
        
        topics.append({
            "name": topic_name,
            "keywords": top_words[:5],
            "importance": round(float(importance), 4)
        })
    
    # Ordenar por importancia (mayor primero)
    topics.sort(key=lambda x: x["importance"], reverse=True)
    
    return {
        "status": "ok",
        "topics": topics[:req.num_topics]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
