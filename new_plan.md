 Final target architecture                                                                                                      
                                                                                                                                
 Your current pipeline:                                                                                                         
                                                                                                                                
 ```text                                                                                                                        
   Candidates                                                                                                                   
     → honeypot filter                                                                                                          
     → coarse non-technical filter                                                                                              
     → rule scoring                                                                                                             
     → semantic scoring                                                                                                         
     → availability multiplier                                                                                                  
     → cached LLM rerank                                                                                                        
     → submission.csv                                                                                                           
 ```                                                                                                                            
                                                                                                                                
 Upgrade target:                                                                                                                
                                                                                                                                
 ```text                                                                                                                        
   100K candidates                                                                                                              
     ↓                                                                                                                          
   Stage 0: Data normalization + feature extraction                                                                             
     ↓                                                                                                                          
   Stage 1: Honeypot / impossibility detection                                                                                  
     ↓                                                                                                                          
   Stage 2: Multi-view rankers                                                                                                  
        ├── Existing structured rule ranker                                                                                     
        ├── Dense semantic ranker                                                                                               
        ├── Sparse lexical / BM25 ranker                                                                                        
        ├── Evidence Graph ranker                                                                                               
        ├── GNN candidate-job ranker                                                                                            
        ├── Learning-to-rank calibrator                                                                                         
        └── LLM recruiter-panel reranker                                                                                        
     ↓                                                                                                                          
   Stage 3: Rank fusion                                                                                                         
        ├── Weighted ensemble                                                                                                   
        ├── Reciprocal Rank Fusion                                                                                              
        └── learned fusion                                                                                                      
     ↓                                                                                                                          
   Stage 4: Robustness / anti-gaming audit                                                                                      
     ↓                                                                                                                          
   Stage 5: Final top-100 with explanations                                                                                     
 ```                                                                                                                            
                                                                                                                                
 Important principle:                                                                                                           
                                                                                                                                
 │ Every new model is a candidate ranker. It does not replace the current ranker until it proves itself.                        
                                                                                                                                
 ────────────────────────────────────────────────────────────────────────────────                                               
                                                                                                                                
 Papers / approaches to implement one by one                                                                                    
                                                                                                                                
 Approach 1: Hybrid retrieval + Reciprocal Rank Fusion                                                                          
                                                                                                                                
 Inspired by IR systems combining dense and sparse retrieval.                                                                   
                                                                                                                                
 Relevant idea:                                                                                                                 
                                                                                                                                
 │ Dense embeddings capture semantic similarity. Sparse/BM25 captures exact technical terms. RRF combines rankers without       
 │ needing calibration.                                                                                                         
                                                                                                                                
 Use this first because it is easy and low-risk.                                                                                
                                                                                                                                
 ### What to build                                                                                                              
                                                                                                                                
 Add:                                                                                                                           
                                                                                                                                
 ```text                                                                                                                        
   src/sparse_matcher.py                                                                                                        
   src/fusion.py                                                                                                                
   eval/ablation_fusion.py                                                                                                      
 ```                                                                                                                            
                                                                                                                                
 Rankers:                                                                                                                       
                                                                                                                                
 ```text                                                                                                                        
   1. rule_score ranking                                                                                                        
   2. semantic_score ranking                                                                                                    
   3. sparse/BM25 ranking                                                                                                       
   4. evidence_graph ranking later                                                                                              
   5. GNN ranking later                                                                                                         
   6. LLM ranking later                                                                                                         
 ```                                                                                                                            
                                                                                                                                
 RRF formula:                                                                                                                   
                                                                                                                                
 ```python                                                                                                                      
   rrf_score(candidate) = Σ 1 / (k + rank_i(candidate))                                                                         
 ```                                                                                                                            
                                                                                                                                
 Usually k=60.                                                                                                                  
                                                                                                                                
 ### Implementation details                                                                                                     
                                                                                                                                
 Use pure Python or lightweight dependency.                                                                                     
                                                                                                                                
 Since you want uv, create a project-managed environment.                                                                       
                                                                                                                                
 If no pyproject.toml exists, create one:                                                                                       
                                                                                                                                
 ```bash                                                                                                                        
   uv init --bare                                                                                                               
   uv add numpy                                                                                                                 
   uv add --dev pytest                                                                                                          
 ```                                                                                                                            
                                                                                                                                
 For BM25:                                                                                                                      
                                                                                                                                
 ```bash                                                                                                                        
   uv add rank-bm25                                                                                                             
 ```                                                                                                                            
                                                                                                                                
 Or implement BM25 yourself if runtime dependencies need to stay minimal.                                                       
                                                                                                                                
 Recommended:                                                                                                                   
                                                                                                                                
 - Precompute BM25 scores offline.                                                                                              
 - Save to cache/sparse_scores.json or .npz.                                                                                    
 - Runtime ranker only loads scores.                                                                                            
                                                                                                                                
 ### Success criteria                                                                                                           
                                                                                                                                
 Run ablation:                                                                                                                  
                                                                                                                                
 ```bash                                                                                                                        
   uv run python eval/evaluate.py                                                                                               
   uv run python eval/ablation_fusion.py                                                                                        
 ```                                                                                                                            
                                                                                                                                
 Compare:                                                                                                                       
                                                                                                                                
 ```text                                                                                                                        
   baseline current system                                                                                                      
   rule + semantic                                                                                                              
   rule + semantic + BM25                                                                                                       
   rule + semantic + BM25 via RRF                                                                                               
 ```                                                                                                                            
                                                                                                                                
 Accept if:                                                                                                                     
                                                                                                                                
 - NDCG@10 does not drop.                                                                                                       
 - Top 20 manual inspection improves or stays clean.                                                                            
 - No keyword stuffers enter top 100.                                                                                           
                                                                                                                                
 ### If it fails                                                                                                                
                                                                                                                                
 Possible failure:                                                                                                              
                                                                                                                                
 - BM25 brings keyword stuffers up.                                                                                             
                                                                                                                                
 Fix:                                                                                                                           
                                                                                                                                
 - Use BM25 only as a weak ranker in RRF.                                                                                       
 - Only apply BM25 after coarse filter.                                                                                         
 - Cap BM25 contribution.                                                                                                       
 - Penalize candidates where BM25 is high but title/career/evidence is low.                                                     
                                                                                                                                
 Example gate:                                                                                                                  
                                                                                                                                
 ```python                                                                                                                      
   if sparse_score > 0.9 and title_score < 0.4 and career_score < 0.4:                                                          
       sparse_score *= 0.2                                                                                                      
 ```                                                                                                                            
                                                                                                                                
 If still bad:                                                                                                                  
                                                                                                                                
 - Keep BM25 only in ablation/deck.                                                                                             
 - Do not use it in final.                                                                                                      
                                                                                                                                
 ────────────────────────────────────────────────────────────────────────────────                                               
                                                                                                                                
 Approach 2: Evidence Graph Ranker                                                                                              
                                                                                                                                
 This is the most important “recruiter-like” upgrade.                                                                           
                                                                                                                                
 Inspired by graph-based job-candidate matching and general evidence-based ranking.                                             
                                                                                                                                
 Instead of trusting fields independently, you reward corroboration:                                                            
                                                                                                                                
 ```text                                                                                                                        
   Skill says Qdrant                                                                                                            
   Career says built semantic search                                                                                            
   Assessment says strong vector search                                                                                         
   Role duration says 24 months                                                                                                 
   Company context says product                                                                                                 
   → strong evidence                                                                                                            
 ```                                                                                                                            
                                                                                                                                
 ### What to build                                                                                                              
                                                                                                                                
 Add:                                                                                                                           
                                                                                                                                
 ```text                                                                                                                        
   src/evidence_graph.py                                                                                                        
   eval/audit_evidence_graph.py                                                                                                 
 ```                                                                                                                            
                                                                                                                                
 Output per candidate:                                                                                                          
                                                                                                                                
 ```python                                                                                                                      
   {                                                                                                                            
       "score": 0.84,                                                                                                           
       "positive_evidence": [                                                                                                   
           "Retrieval/ranking appears in both skills and career descriptions",                                                  
           "Vector DB skill has 24+ months duration",                                                                           
           "Product-company ML role",                                                                                           
           "Assessment corroborates claimed Python/ML skill"                                                                    
       ],                                                                                                                       
       "negative_evidence": [                                                                                                   
           "Long notice period",                                                                                                
           "Skill claim unsupported by role text"                                                                               
       ]                                                                                                                        
   }                                                                                                                            
 ```                                                                                                                            
                                                                                                                                
 ### Evidence categories                                                                                                        
                                                                                                                                
 Use 6-8 categories.                                                                                                            
                                                                                                                                
 #### 1. Role-skill corroboration                                                                                               
                                                                                                                                
 High when title/career text and skills agree.                                                                                  
                                                                                                                                
 ```text                                                                                                                        
   Skill: FAISS / Qdrant / Vector Search                                                                                        
   Career: built semantic search / ranking / retrieval                                                                          
 ```                                                                                                                            
                                                                                                                                
 #### 2. Duration-supported expertise                                                                                           
                                                                                                                                
 High when important skill has non-trivial duration.                                                                            
                                                                                                                                
 ```text                                                                                                                        
   Embeddings for 24 months > Embeddings for 0 months                                                                           
 ```                                                                                                                            
                                                                                                                                
 #### 3. Assessment-supported expertise                                                                                         
                                                                                                                                
 High when platform assessments support claimed skills.                                                                         
                                                                                                                                
 #### 4. Product context                                                                                                        
                                                                                                                                
 High when candidate worked in product companies or product-like teams.                                                         
                                                                                                                                
 #### 5. Seniority consistency                                                                                                  
                                                                                                                                
 High when title, years, and responsibilities align.                                                                            
                                                                                                                                
 Bad:                                                                                                                           
                                                                                                                                
 ```text                                                                                                                        
   2 years experience + Staff AI Architect                                                                                      
 ```                                                                                                                            
                                                                                                                                
 #### 6. Ranking/retrieval maturity                                                                                             
                                                                                                                                
 Look for:                                                                                                                      
                                                                                                                                
 ```text                                                                                                                        
   NDCG                                                                                                                         
   MRR                                                                                                                          
   MAP                                                                                                                          
   A/B testing                                                                                                                  
   learning-to-rank                                                                                                             
   search relevance                                                                                                             
   recommendation systems                                                                                                       
   retrieval quality                                                                                                            
 ```                                                                                                                            
                                                                                                                                
 #### 7. Behavioral readiness                                                                                                   
                                                                                                                                
 Open to work, response rate, last active, notice period.                                                                       
                                                                                                                                
 #### 8. Anti-stuffing contradiction                                                                                            
                                                                                                                                
 High penalty when:                                                                                                             
                                                                                                                                
 ```text                                                                                                                        
   AI skills present but entire career is HR/Marketing/Sales                                                                    
   expert skills with zero duration                                                                                             
   no role text evidence                                                                                                        
 ```                                                                                                                            
                                                                                                                                
 ### Integrate score                                                                                                            
                                                                                                                                
 Add to config:                                                                                                                 
                                                                                                                                
 ```python                                                                                                                      
   FIT_WEIGHTS_HYBRID_GRAPH = {                                                                                                 
       "title": 0.23,                                                                                                           
       "skills": 0.19,                                                                                                          
       "career": 0.12,                                                                                                          
       "experience": 0.08,                                                                                                      
       "education": 0.04,                                                                                                       
       "location": 0.06,                                                                                                        
       "semantic": 0.13,                                                                                                        
       "evidence_graph": 0.15,                                                                                                  
   }                                                                                                                            
 ```                                                                                                                            
                                                                                                                                
 Or use it as a separate ranker in RRF.                                                                                         
                                                                                                                                
 I recommend initially:                                                                                                         
                                                                                                                                
 │ Use evidence graph as a ranker in RRF, not directly in weighted score.                                                       
                                                                                                                                
 Safer.                                                                                                                         
                                                                                                                                
 ### Success criteria                                                                                                           
                                                                                                                                
 Evidence graph should:                                                                                                         
                                                                                                                                
 - Push real search/retrieval/product ML candidates up.                                                                         
 - Push keyword stuffers down.                                                                                                  
 - Improve reasoning quality.                                                                                                   
 - Improve manual top-20 trust.                                                                                                 
                                                                                                                                
 ### If it fails                                                                                                                
                                                                                                                                
 Common failure:                                                                                                                
                                                                                                                                
 - Rules too strict and miss strong candidates with sparse descriptions.                                                        
                                                                                                                                
 Fix:                                                                                                                           
                                                                                                                                
 - Use soft scoring, not hard filtering.                                                                                        
 - Do not require exact phrase matches.                                                                                         
 - Add synonym groups.                                                                                                          
                                                                                                                                
 Example:                                                                                                                       
                                                                                                                                
 ```python                                                                                                                      
   RETRIEVAL_TERMS = {                                                                                                          
       "semantic search", "search relevance", "retrieval", "ranking",                                                           
       "recommendation", "recommender", "vector search", "ann",                                                                 
       "faiss", "qdrant", "pinecone", "weaviate", "milvus",                                                                     
       "elasticsearch", "opensearch"                                                                                            
   }                                                                                                                            
 ```                                                                                                                            
                                                                                                                                
 If evidence graph hurts NDCG:                                                                                                  
                                                                                                                                
 - Keep it for explanations only.                                                                                               
 - Use it only as tie-breaker.                                                                                                  
 - Use it only to penalize contradictions, not boost.                                                                           
                                                                                                                                
 ────────────────────────────────────────────────────────────────────────────────                                               
                                                                                                                                
 Approach 3: Anti-gaming / robustness audit                                                                                     
                                                                                                                                
 Inspired by position-robust and anti-gaming talent recommendation.                                                             
                                                                                                                                
 This is not necessarily a ranking module — it is a proof that your system is better than keyword matching.                     
                                                                                                                                
 This will be extremely useful in your deck.                                                                                    
                                                                                                                                
 ### What to build                                                                                                              
                                                                                                                                
 Add:                                                                                                                           
                                                                                                                                
 ```text                                                                                                                        
   eval/robustness_audit.py                                                                                                     
 ```                                                                                                                            
                                                                                                                                
 Test perturbations:                                                                                                            
                                                                                                                                
 #### Perturbation 1: keyword stuffing                                                                                          
                                                                                                                                
 Take weak/non-technical candidates and inject:                                                                                 
                                                                                                                                
 ```text                                                                                                                        
   RAG, FAISS, Pinecone, Qdrant, LLM, embeddings, semantic search, LangChain                                                    
 ```                                                                                                                            
                                                                                                                                
 Compare rank jump.                                                                                                             
                                                                                                                                
 Expected:                                                                                                                      
                                                                                                                                
 ```text                                                                                                                        
   keyword baseline: huge jump                                                                                                  
   your system: small jump                                                                                                      
 ```                                                                                                                            
                                                                                                                                
 #### Perturbation 2: remove skills                                                                                             
                                                                                                                                
 Take strong candidates and remove skill section.                                                                               
                                                                                                                                
 Expected:                                                                                                                      
                                                                                                                                
 ```text                                                                                                                        
   strong candidates should not collapse entirely if career evidence is strong                                                  
 ```                                                                                                                            
                                                                                                                                
 #### Perturbation 3: shuffle skill order                                                                                       
                                                                                                                                
 Ranking should not change.                                                                                                     
                                                                                                                                
 #### Perturbation 4: remove behavioral signals                                                                                 
                                                                                                                                
 Candidate may move slightly but should not become irrelevant.                                                                  
                                                                                                                                
 #### Perturbation 5: title inflation                                                                                           
                                                                                                                                
 Change weak candidate title to:                                                                                                
                                                                                                                                
 ```text                                                                                                                        
   Senior AI Engineer                                                                                                           
 ```                                                                                                                            
                                                                                                                                
 Expected:                                                                                                                      
                                                                                                                                
 ```text                                                                                                                        
   system should resist if career/skills do not corroborate                                                                     
 ```                                                                                                                            
                                                                                                                                
 ### Metrics                                                                                                                    
                                                                                                                                
 Report:                                                                                                                        
                                                                                                                                
 ```text                                                                                                                        
   Average rank change                                                                                                          
   Median rank change                                                                                                           
   Max rank jump                                                                                                                
   Number of weak candidates entering top 100                                                                                   
   Number entering top 500                                                                                                      
 ```                                                                                                                            
                                                                                                                                
 ### Success criteria                                                                                                           
                                                                                                                                
 You want a slide like:                                                                                                         
                                                                                                                                
 ```text                                                                                                                        
   Keyword stuffing test on 500 weak profiles                                                                                   
                                                                                                                                
   Keyword baseline:                                                                                                            
     71 profiles entered top 500                                                                                                
     12 entered top 100                                                                                                         
                                                                                                                                
   Our system:                                                                                                                  
     3 profiles entered top 500                                                                                                 
     0 entered top 100                                                                                                          
                                                                                                                                
   EvidenceGraph-Rank:                                                                                                          
     1 profile entered top 500                                                                                                  
     0 entered top 100                                                                                                          
 ```                                                                                                                            
                                                                                                                                
 ### If it fails                                                                                                                
                                                                                                                                
 If weak profiles enter top 100 after stuffing:                                                                                 
                                                                                                                                
 - Increase title/career/evidence weights.                                                                                      
 - Add contradiction penalty.                                                                                                   
 - Add “skills unsupported by career” penalty.                                                                                  
 - Reduce raw skill scorer power.                                                                                               
                                                                                                                                
 This audit directly guides tuning.                                                                                             
                                                                                                                                
 ────────────────────────────────────────────────────────────────────────────────                                               
                                                                                                                                
 Approach 4: Learning-to-Rank using pseudo labels                                                                               
                                                                                                                                
 Inspired by learning-to-retrieve / learning-to-rank job matching systems.                                                      
                                                                                                                                
 You already have LLM labels. Use them as weak supervision.                                                                     
                                                                                                                                
 ### What to build                                                                                                              
                                                                                                                                
 Add:                                                                                                                           
                                                                                                                                
 ```text                                                                                                                        
   precompute/train_ltr.py                                                                                                      
   src/ltr_model.py                                                                                                             
   cache/ltr_model.json                                                                                                         
   eval/ablation_ltr.py                                                                                                         
 ```                                                                                                                            
                                                                                                                                
 ### Feature vector                                                                                                             
                                                                                                                                
 For each candidate:                                                                                                            
                                                                                                                                
 ```python                                                                                                                      
   features = {                                                                                                                 
       "title": title_score,                                                                                                    
       "skills": skills_score,                                                                                                  
       "career": career_score,                                                                                                  
       "experience": experience_score,                                                                                          
       "education": education_score,                                                                                            
       "location": location_score,                                                                                              
       "semantic": semantic_score,                                                                                              
       "availability": availability_score,                                                                                      
       "evidence_graph": evidence_graph_score,                                                                                  
       "honeypot_flag_count": honeypot_flag_count,                                                                              
       "product_company_score": product_company_score,                                                                          
       "consulting_only_penalty": consulting_only_penalty,                                                                      
       "retrieval_term_count": retrieval_term_count,                                                                            
       "vector_db_skill_count": vector_db_skill_count,                                                                          
       "assessment_support_score": assessment_support_score,                                                                    
       "unsupported_skill_ratio": unsupported_skill_ratio,                                                                      
   }                                                                                                                            
 ```                                                                                                                            
                                                                                                                                
 ### Labels                                                                                                                     
                                                                                                                                
 Use:                                                                                                                           
                                                                                                                                
 ```text                                                                                                                        
   cache/llm_rerank.jsonl fit_score 0-10                                                                                        
   eval/golden_labels.csv grade 0-4                                                                                             
 ```                                                                                                                            
                                                                                                                                
 Possible combined target:                                                                                                      
                                                                                                                                
 ```python                                                                                                                      
   target = normalized_llm_fit_score                                                                                            
 ```                                                                                                                            
                                                                                                                                
 Or:                                                                                                                            
                                                                                                                                
 ```python                                                                                                                      
   target = integer grade                                                                                                       
 ```                                                                                                                            
                                                                                                                                
 ### Models                                                                                                                     
                                                                                                                                
 Start simple.                                                                                                                  
                                                                                                                                
 #### Option A: Linear model                                                                                                    
                                                                                                                                
 Most reproducible.                                                                                                             
                                                                                                                                
 ```bash                                                                                                                        
   uv add scikit-learn                                                                                                          
 ```                                                                                                                            
                                                                                                                                
 Train:                                                                                                                         
                                                                                                                                
 ```python                                                                                                                      
   Ridge()                                                                                                                      
   ElasticNet()                                                                                                                 
   LogisticRegression()                                                                                                         
 ```                                                                                                                            
                                                                                                                                
 Save coefficients to JSON.                                                                                                     
                                                                                                                                
 Runtime uses no sklearn if you manually apply coefficients.                                                                    
                                                                                                                                
 #### Option B: Gradient boosting                                                                                               
                                                                                                                                
 Better but slightly more dependency/runtime.                                                                                   
                                                                                                                                
 ```python                                                                                                                      
   HistGradientBoostingRegressor                                                                                                
   RandomForestRegressor                                                                                                        
   GradientBoostingRegressor                                                                                                    
 ```                                                                                                                            
                                                                                                                                
 Can save model with pickle/joblib.                                                                                             
                                                                                                                                
 #### Option C: LightGBM LambdaRank                                                                                             
                                                                                                                                
 Most “proper LTR,” but unnecessary unless you want max research flavor.                                                        
                                                                                                                                
 ```bash                                                                                                                        
   uv add lightgbm                                                                                                              
 ```                                                                                                                            
                                                                                                                                
 But this adds dependency friction.                                                                                             
                                                                                                                                
 My recommendation:                                                                                                             
                                                                                                                                
 │ Use sklearn HistGradientBoostingRegressor offline, and export predictions for all candidates to cache/ltr_scores.npz.        
                                                                                                                                
 Then runtime does not need sklearn.                                                                                            
                                                                                                                                
 ### Evaluation protocol                                                                                                        
                                                                                                                                
 Important: avoid self-delusion.                                                                                                
                                                                                                                                
 Use cross-validation on labels:                                                                                                
                                                                                                                                
 ```text                                                                                                                        
   train on 80% labels                                                                                                          
   test on 20%                                                                                                                  
   measure NDCG/MAP on heldout labeled candidates                                                                               
 ```                                                                                                                            
                                                                                                                                
 Also compare final top-100 manual quality.                                                                                     
                                                                                                                                
 ### Success criteria                                                                                                           
                                                                                                                                
 Accept LTR if:                                                                                                                 
                                                                                                                                
 - heldout NDCG improves over manual weighted score                                                                             
 - top 20 does not get worse                                                                                                    
 - no obvious keyword stuffers enter                                                                                            
                                                                                                                                
 ### If it fails                                                                                                                
                                                                                                                                
 Very possible because labels are small.                                                                                        
                                                                                                                                
 Failure modes:                                                                                                                 
                                                                                                                                
 #### Failure 1: overfits LLM labels                                                                                            
                                                                                                                                
 Symptoms:                                                                                                                      
                                                                                                                                
 - Great on labeled set, weird top 100.                                                                                         
                                                                                                                                
 Fix:                                                                                                                           
                                                                                                                                
 - Use simpler linear model.                                                                                                    
 - Reduce feature count.                                                                                                        
 - Use LTR only as a tie-breaker.                                                                                               
 - Train only on high-confidence LLM labels.                                                                                    
                                                                                                                                
 #### Failure 2: pushes availability too hard                                                                                   
                                                                                                                                
 Fix:                                                                                                                           
                                                                                                                                
 - Cap behavioral feature.                                                                                                      
 - Use fit-only target, availability applied separately.                                                                        
                                                                                                                                
 #### Failure 3: hurts top 10                                                                                                   
                                                                                                                                
 Fix:                                                                                                                           
                                                                                                                                
 - Use LTR only after top-300 shortlist.                                                                                        
 - Or blend:                                                                                                                    
                                                                                                                                
 ```python                                                                                                                      
   final = 0.75 * current_score + 0.25 * ltr_score                                                                              
 ```                                                                                                                            
                                                                                                                                
 #### Failure 4: no improvement                                                                                                 
                                                                                                                                
 Keep it as ablation:                                                                                                           
                                                                                                                                
 ```text                                                                                                                        
   LTR calibration was tested but final system retained evidence-fused ensemble because it was more stable on manual audit.     
 ```                                                                                                                            
                                                                                                                                
 That is still valuable.                                                                                                        
                                                                                                                                
 ────────────────────────────────────────────────────────────────────────────────                                               
                                                                                                                                
 Approach 5: GNN candidate-job matching                                                                                         
                                                                                                                                
 This is the most ambitious.                                                                                                    
                                                                                                                                
 Inspired by graph neural networks for candidate-job matching.                                                                  
                                                                                                                                
 You can do it, but you need to be careful because your label set is small. GNNs love data. If labels are weak/small, they can  
 easily become fancy noise.                                                                                                     
                                                                                                                                
 So the correct strategy:                                                                                                       
                                                                                                                                
 │ Build a heterogeneous graph and train a lightweight GNN using weak LLM labels plus pseudo-positive/pseudo-negative           
 │ candidates. Use it as a candidate ranker, not the only final ranker.                                                         
                                                                                                                                
 ────────────────────────────────────────────────────────────────────────────────                                               
                                                                                                                                
 GNN design                                                                                                                     
                                                                                                                                
 Graph type                                                                                                                     
                                                                                                                                
 Heterogeneous graph:                                                                                                           
                                                                                                                                
 ```text                                                                                                                        
   Candidate nodes                                                                                                              
   Job node                                                                                                                     
   Skill nodes                                                                                                                  
   Title nodes                                                                                                                  
   Company nodes                                                                                                                
   Education field nodes                                                                                                        
   Location nodes                                                                                                               
   Behavioral bucket nodes                                                                                                      
   Evidence concept nodes                                                                                                       
 ```                                                                                                                            
                                                                                                                                
 Edges:                                                                                                                         
                                                                                                                                
 ```text                                                                                                                        
   candidate -> skill                                                                                                           
   candidate -> title                                                                                                           
   candidate -> company                                                                                                         
   candidate -> location                                                                                                        
   candidate -> education                                                                                                       
   candidate -> behavior_bucket                                                                                                 
   candidate -> evidence_concept                                                                                                
   job -> required_skill                                                                                                        
   job -> target_title                                                                                                          
   job -> preferred_location                                                                                                    
   job -> evidence_concept                                                                                                      
 ```                                                                                                                            
                                                                                                                                
 Example:                                                                                                                       
                                                                                                                                
 ```text                                                                                                                        
   CAND_0077337 --HAS_SKILL--> Qdrant                                                                                           
   CAND_0077337 --HAS_TITLE--> Staff ML Engineer                                                                                
   CAND_0077337 --WORKED_AT--> Paytm                                                                                            
   CAND_0077337 --HAS_CONCEPT--> semantic_search                                                                                
   JOB --REQUIRES--> vector_database                                                                                            
   JOB --REQUIRES--> retrieval_ranking                                                                                          
 ```                                                                                                                            
                                                                                                                                
 Model objective                                                                                                                
                                                                                                                                
 You only have one job description. So train candidate-job compatibility as node classification/regression:                     
                                                                                                                                
 ```text                                                                                                                        
   Candidate node → predicted fit grade                                                                                         
 ```                                                                                                                            
                                                                                                                                
 Or link prediction:                                                                                                            
                                                                                                                                
 ```text                                                                                                                        
   (candidate, job) edge label = fit                                                                                            
 ```                                                                                                                            
                                                                                                                                
 Because there is one job, candidate node regression is simpler.                                                                
                                                                                                                                
 Labels                                                                                                                         
                                                                                                                                
 Use multiple weak label sources:                                                                                               
                                                                                                                                
 ```text                                                                                                                        
   Grade 4:                                                                                                                     
     LLM fit_score >= 8.5                                                                                                       
     top current rank candidates                                                                                                
     manually confirmed strong candidates                                                                                       
                                                                                                                                
   Grade 3:                                                                                                                     
     LLM fit_score 7-8.5                                                                                                        
                                                                                                                                
   Grade 2:                                                                                                                     
     plausible but weaker                                                                                                       
                                                                                                                                
   Grade 0/1:                                                                                                                   
     honeypots                                                                                                                  
     non-technical careers                                                                                                      
     keyword-stuffed profiles                                                                                                   
     consulting-only mismatch                                                                                                   
 ```                                                                                                                            
                                                                                                                                
 Make sure you include negatives. Otherwise GNN learns nothing.                                                                 
                                                                                                                                
 Features                                                                                                                       
                                                                                                                                
 Candidate initial feature vector:                                                                                              
                                                                                                                                
 ```python                                                                                                                      
   [                                                                                                                            
     title_score,                                                                                                               
     skills_score,                                                                                                              
     career_score,                                                                                                              
     experience_score,                                                                                                          
     education_score,                                                                                                           
     location_score,                                                                                                            
     semantic_score,                                                                                                            
     availability,                                                                                                              
     evidence_graph_score,                                                                                                      
     years_experience_normalized,                                                                                               
     unsupported_skill_ratio,                                                                                                   
     skill_count,                                                                                                               
     vector_db_skill_count,                                                                                                     
     retrieval_term_count,                                                                                                      
   ]                                                                                                                            
 ```                                                                                                                            
                                                                                                                                
 Skill/title/company nodes can have simple one-hot/embedding IDs.                                                               
                                                                                                                                
 Tools                                                                                                                          
                                                                                                                                
 Use PyTorch Geometric if setup is fine:                                                                                        
                                                                                                                                
 ```bash                                                                                                                        
   uv add torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121                                       
   uv add torch-geometric                                                                                                       
   uv add scikit-learn pandas tqdm                                                                                              
 ```                                                                                                                            
                                                                                                                                
 But PyG installation can be annoying depending on CUDA/Torch version.                                                          
                                                                                                                                
 Alternative simpler route:                                                                                                     
                                                                                                                                
 ```text                                                                                                                        
   Do not use PyG. Build candidate-feature graph embeddings manually:                                                           
   - candidate connected to concept nodes                                                                                       
   - run Node2Vec                                                                                                               
   - train regressor                                                                                                            
 ```                                                                                                                            
                                                                                                                                
 Dependencies:                                                                                                                  
                                                                                                                                
 ```bash                                                                                                                        
   uv add networkx node2vec scikit-learn                                                                                        
 ```                                                                                                                            
                                                                                                                                
 But if you want actual GNN, PyG is fine.                                                                                       
                                                                                                                                
 ────────────────────────────────────────────────────────────────────────────────                                               
                                                                                                                                
 Practical GNN implementation plan                                                                                              
                                                                                                                                
 Files                                                                                                                          
                                                                                                                                
 ```text                                                                                                                        
   precompute/build_graph.py                                                                                                    
   precompute/train_gnn.py                                                                                                      
   src/gnn_scores.py                                                                                                            
   cache/gnn_scores.npz                                                                                                         
   cache/gnn_model.pt                                                                                                           
   eval/ablation_gnn.py                                                                                                         
 ```                                                                                                                            
                                                                                                                                
 build_graph.py                                                                                                                 
                                                                                                                                
 Outputs:                                                                                                                       
                                                                                                                                
 ```text                                                                                                                        
   cache/graph_data.pt                                                                                                          
   cache/candidate_index.json                                                                                                   
 ```                                                                                                                            
                                                                                                                                
 Graph construction should use the full 100K, but keep features compact.                                                        
                                                                                                                                
 Avoid huge text nodes. Use normalized concepts.                                                                                
                                                                                                                                
 Concept nodes:                                                                                                                 
                                                                                                                                
 ```python                                                                                                                      
   CONCEPTS = [                                                                                                                 
       "semantic_search",                                                                                                       
       "information_retrieval",                                                                                                 
       "ranking",                                                                                                               
       "recommendation_systems",                                                                                                
       "vector_database",                                                                                                       
       "embeddings",                                                                                                            
       "llm_finetuning",                                                                                                        
       "evaluation_metrics",                                                                                                    
       "ab_testing",                                                                                                            
       "product_ml",                                                                                                            
       "consulting_only",                                                                                                       
       "keyword_stuffing_risk",                                                                                                 
   ]                                                                                                                            
 ```                                                                                                                            
                                                                                                                                
 train_gnn.py                                                                                                                   
                                                                                                                                
 Train on labeled candidates.                                                                                                   
                                                                                                                                
 Model:                                                                                                                         
                                                                                                                                
 ```python                                                                                                                      
   GraphSAGE                                                                                                                    
 ```                                                                                                                            
                                                                                                                                
 or                                                                                                                             
                                                                                                                                
 ```python                                                                                                                      
   HeteroConv with SAGEConv                                                                                                     
 ```                                                                                                                            
                                                                                                                                
 Keep it small:                                                                                                                 
                                                                                                                                
 ```text                                                                                                                        
   hidden_dim = 64                                                                                                              
   layers = 2                                                                                                                   
   dropout = 0.2                                                                                                                
   epochs = 100                                                                                                                 
   early stopping                                                                                                               
 ```                                                                                                                            
                                                                                                                                
 Output score for all candidate nodes.                                                                                          
                                                                                                                                
 Save:                                                                                                                          
                                                                                                                                
 ```text                                                                                                                        
   cache/gnn_scores.npz                                                                                                         
 ```                                                                                                                            
                                                                                                                                
 Runtime ranking just loads gnn_scores.npz.                                                                                     
                                                                                                                                
 Evaluation                                                                                                                     
                                                                                                                                
 Compare:                                                                                                                       
                                                                                                                                
 ```text                                                                                                                        
   baseline                                                                                                                     
   baseline + gnn_score weighted                                                                                                
   baseline + gnn as RRF ranker                                                                                                 
   gnn alone                                                                                                                    
 ```                                                                                                                            
                                                                                                                                
 ### Success criteria                                                                                                           
                                                                                                                                
 GNN is accepted only if:                                                                                                       
                                                                                                                                
 - It improves or does not hurt composite on labels.                                                                            
 - It improves manual top-50 quality.                                                                                           
 - It does not introduce weird candidates.                                                                                      
 - It is stable across random seeds.                                                                                            
                                                                                                                                
 Run 5 seeds:                                                                                                                   
                                                                                                                                
 ```bash                                                                                                                        
   uv run python precompute/train_gnn.py --seed 1                                                                               
   uv run python precompute/train_gnn.py --seed 2                                                                               
   ...                                                                                                                          
 ```                                                                                                                            
                                                                                                                                
 If rankings vary wildly, GNN is not stable.                                                                                    
                                                                                                                                
 ### If GNN fails                                                                                                               
                                                                                                                                
 Very possible. Handle professionally.                                                                                          
                                                                                                                                
 Possible failure modes:                                                                                                        
                                                                                                                                
 #### Failure 1: GNN overfits small labels                                                                                      
                                                                                                                                
 Symptoms:                                                                                                                      
                                                                                                                                
 - Good training performance, bad validation.                                                                                   
 - Weird candidates enter top 100.                                                                                              
                                                                                                                                
 Response:                                                                                                                      
                                                                                                                                
 - Do not use GNN in final.                                                                                                     
 - Use GNN score as weak RRF ranker only.                                                                                       
 - Or only use it for candidate clustering / explanation.                                                                       
                                                                                                                                
 #### Failure 2: graph too noisy                                                                                                
                                                                                                                                
 Symptoms:                                                                                                                      
                                                                                                                                
 - GNN ranks generic “AI keyword” candidates high.                                                                              
                                                                                                                                
 Response:                                                                                                                      
                                                                                                                                
 - Remove raw skill nodes for low-duration skills.                                                                              
 - Add unsupported-skill penalty feature.                                                                                       
 - Add edge weights based on duration/proficiency.                                                                              
 - Use evidence concept nodes instead of all raw skills.                                                                        
                                                                                                                                
 #### Failure 3: no improvement over evidence graph                                                                             
                                                                                                                                
 Response:                                                                                                                      
                                                                                                                                
 - That is acceptable.                                                                                                          
 - In deck say:                                                                                                                 
                                                                                                                                
 ```text                                                                                                                        
   We implemented a GNN candidate-job matcher but found the symbolic evidence graph was more stable under limited labels. We    
 therefore retained GNN as an analysis ranker and used evidence fusion in final.                                                
 ```                                                                                                                            
                                                                                                                                
 This actually looks mature.                                                                                                    
                                                                                                                                
 #### Failure 4: too slow / artifact too large                                                                                  
                                                                                                                                
 Response:                                                                                                                      
                                                                                                                                
 - Precompute GNN scores offline.                                                                                               
 - Runtime only loads candidate_id → score.                                                                                     
 - If cache too large, save top 5K only.                                                                                        
                                                                                                                                
 ────────────────────────────────────────────────────────────────────────────────                                               
                                                                                                                                
 Approach 6: Multi-agent LLM recruiter panel                                                                                    
                                                                                                                                
 Inspired by LLM-based explainable multi-agent resume screening.                                                                
                                                                                                                                
 You already have an LLM reranker. Upgrade it to a panel only if time allows.                                                   
                                                                                                                                
 ### Agents                                                                                                                     
                                                                                                                                
 ```text                                                                                                                        
   1. Hiring Manager                                                                                                            
      - production ML, seniority, ownership                                                                                     
                                                                                                                                
   2. Technical Interviewer                                                                                                     
      - retrieval, embeddings, vector DBs, evaluation metrics                                                                   
                                                                                                                                
   3. Recruiter                                                                                                                 
      - location, notice period, responsiveness, availability                                                                   
                                                                                                                                
   4. Skeptic / Anti-gaming Agent                                                                                               
      - keyword stuffing, impossible claims, consulting-only mismatch                                                           
 ```                                                                                                                            
                                                                                                                                
 ### Output                                                                                                                     
                                                                                                                                
 For each top-200 candidate:                                                                                                    
                                                                                                                                
 ```json                                                                                                                        
   {                                                                                                                            
     "candidate_id": "...",                                                                                                     
     "hiring_manager_score": 9.5,                                                                                               
     "technical_score": 9.2,                                                                                                    
     "recruiter_score": 8.0,                                                                                                    
     "skeptic_risk": 0.5,                                                                                                       
     "final_panel_score": 9.1,                                                                                                  
     "summary": "...",                                                                                                          
     "concerns": "..."                                                                                                          
   }                                                                                                                            
 ```                                                                                                                            
                                                                                                                                
 ### Use                                                                                                                        
                                                                                                                                
 Use only as cached offline rerank.                                                                                             
                                                                                                                                
 ### Success criteria                                                                                                           
                                                                                                                                
 - Better than current single LLM rerank.                                                                                       
 - Better explanations.                                                                                                         
 - No unstable weirdness.                                                                                                       
                                                                                                                                
 ### If it fails                                                                                                                
                                                                                                                                
 - Use single LLM rerank.                                                                                                       
 - Keep multi-agent panel as top-20 audit / demo.                                                                               
                                                                                                                                
 ────────────────────────────────────────────────────────────────────────────────                                               
                                                                                                                                
 uv migration plan                                                                                                              
                                                                                                                                
 Since you want uv moving forward, do this before adding more dependencies.                                                     
                                                                                                                                
 Step 1: Create pyproject.toml                                                                                                  
                                                                                                                                
 If not already present:                                                                                                        
                                                                                                                                
 ```bash                                                                                                                        
   uv init --bare                                                                                                               
 ```                                                                                                                            
                                                                                                                                
 Then add core runtime:                                                                                                         
                                                                                                                                
 ```bash                                                                                                                        
   uv add numpy                                                                                                                 
 ```                                                                                                                            
                                                                                                                                
 Add dev tools:                                                                                                                 
                                                                                                                                
 ```bash                                                                                                                        
   uv add --dev pytest ruff                                                                                                     
 ```                                                                                                                            
                                                                                                                                
 Add optional research dependencies as groups if you want clean separation.                                                     
                                                                                                                                
 uv supports dependency groups. Suggested structure:                                                                            
                                                                                                                                
 ```toml                                                                                                                        
   [project]                                                                                                                    
   name = "india-runs"                                                                                                          
   version = "0.1.0"                                                                                                            
   requires-python = ">=3.10"                                                                                                   
   dependencies = [                                                                                                             
       "numpy>=1.26",                                                                                                           
   ]                                                                                                                            
                                                                                                                                
   [dependency-groups]                                                                                                          
   dev = [                                                                                                                      
       "pytest>=8.0",                                                                                                           
       "ruff>=0.6",                                                                                                             
   ]                                                                                                                            
   embed = [                                                                                                                    
       "sentence-transformers",                                                                                                 
   ]                                                                                                                            
   ltr = [                                                                                                                      
       "scikit-learn",                                                                                                          
       "pandas",                                                                                                                
       "joblib",                                                                                                                
   ]                                                                                                                            
   graph = [                                                                                                                    
       "networkx",                                                                                                              
       "scikit-learn",                                                                                                          
   ]                                                                                                                            
   gnn = [                                                                                                                      
       "torch",                                                                                                                 
       "torch-geometric",                                                                                                       
   ]                                                                                                                            
   sandbox = [                                                                                                                  
       "streamlit",                                                                                                             
       "plotly",                                                                                                                
       "pandas",                                                                                                                
   ]                                                                                                                            
 ```                                                                                                                            
                                                                                                                                
 Depending on uv version, you may add groups with:                                                                              
                                                                                                                                
 ```bash                                                                                                                        
   uv add --dev pytest ruff                                                                                                     
   uv add --optional embed sentence-transformers                                                                                
   uv add --optional ltr scikit-learn pandas joblib                                                                             
   uv add --optional sandbox streamlit plotly pandas                                                                            
 ```                                                                                                                            
                                                                                                                                
 For PyTorch CUDA, use explicit command:                                                                                        
                                                                                                                                
 ```bash                                                                                                                        
   uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121                               
   uv pip install torch-geometric                                                                                               
 ```                                                                                                                            
                                                                                                                                
 If PyG installation is annoying, do not block the whole project. Use a separate .venv-gnn.                                     
                                                                                                                                
 ────────────────────────────────────────────────────────────────────────────────                                               
                                                                                                                                
 Branching strategy                                                                                                             
                                                                                                                                
 Use branches so you do not break the working submission.                                                                       
                                                                                                                                
 ```bash                                                                                                                        
   git checkout -b research/rrf-bm25                                                                                            
   git checkout -b research/evidence-graph                                                                                      
   git checkout -b research/ltr                                                                                                 
   git checkout -b research/gnn                                                                                                 
   git checkout -b research/robustness                                                                                          
 ```                                                                                                                            
                                                                                                                                
 After each module passes evaluation:                                                                                           
                                                                                                                                
 ```bash                                                                                                                        
   git checkout main                                                                                                            
   git merge research/evidence-graph                                                                                            
 ```                                                                                                                            
                                                                                                                                
 If a module fails:                                                                                                             
                                                                                                                                
 ```bash                                                                                                                        
   git keep branch for reference                                                                                                
   do not merge into final ranking                                                                                              
 ```                                                                                                                            
                                                                                                                                
 ────────────────────────────────────────────────────────────────────────────────                                               
                                                                                                                                
 Evaluation protocol for every new approach                                                                                     
                                                                                                                                
 Every approach must produce an ablation row.                                                                                   
                                                                                                                                
 Create:                                                                                                                        
                                                                                                                                
 ```text                                                                                                                        
   eval/ablation_report.py                                                                                                      
 ```                                                                                                                            
                                                                                                                                
 Output table:                                                                                                                  
                                                                                                                                
 ```text                                                                                                                        
   system                         ndcg@10  ndcg@50  map     p@10   composite  top100_honeypots                                  
   baseline_current               ...                                                                                           
   +bm25_rrf                      ...                                                                                           
   +evidence_graph                ...                                                                                           
   +evidence_graph_rrf            ...                                                                                           
   +ltr                           ...                                                                                           
   +gnn                           ...                                                                                           
   +llm_panel                     ...                                                                                           
   final                          ...                                                                                           
 ```                                                                                                                            
                                                                                                                                
 Also add manual audits:                                                                                                        
                                                                                                                                
 ```text                                                                                                                        
   top10_manual_notes.md                                                                                                        
   top100_distribution.json                                                                                                     
   robustness_audit.md                                                                                                          
 ```                                                                                                                            
                                                                                                                                
 Acceptance rule                                                                                                                
                                                                                                                                
 A method is final only if:                                                                                                     
                                                                                                                                
 ```text                                                                                                                        
   1. Composite improves OR stays within tiny margin.                                                                           
   2. Top 10 manual quality improves or stays same.                                                                             
   3. No honeypots/non-technical candidates enter top 100.                                                                      
   4. Runtime remains under 5 minutes CPU-only.                                                                                 
   5. Runtime uses no network.                                                                                                  
   6. Artifacts are documented.                                                                                                 
 ```                                                                                                                            
                                                                                                                                
 If not, it becomes:                                                                                                            
                                                                                                                                
 ```text                                                                                                                        
   - ablation result                                                                                                            
   - deck mention                                                                                                               
   - not used in final                                                                                                          
 ```                                                                                                                            
                                                                                                                                
 This is important. Judges like seeing that you tried strong methods and made evidence-based choices.                           
                                                                                                                                
 ────────────────────────────────────────────────────────────────────────────────                                               
                                                                                                                                
 Final runtime rule                                                                                                             
                                                                                                                                
 All heavy research models should be offline precompute only.                                                                   
                                                                                                                                
 At final ranking time:                                                                                                         
                                                                                                                                
 Allowed:                                                                                                                       
                                                                                                                                
 ```text                                                                                                                        
   load JSON/NPZ caches                                                                                                         
   numpy dot product                                                                                                            
   rule scoring                                                                                                                 
   RRF                                                                                                                          
   sort                                                                                                                         
   write CSV                                                                                                                    
 ```                                                                                                                            
                                                                                                                                
 Not allowed / risky:                                                                                                           
                                                                                                                                
 ```text                                                                                                                        
   training                                                                                                                     
   GPU dependency                                                                                                               
   network LLM calls                                                                                                            
   sentence-transformer encoding                                                                                                
   GNN inference if dependency-heavy                                                                                            
 ```                                                                                                                            
                                                                                                                                
 So final rank.py should remain lightweight.                                                                                    
                                                                                                                                
 For each research module:                                                                                                      
                                                                                                                                
 ```text                                                                                                                        
   precompute/build_x.py → writes cache/x_scores.npz                                                                            
   rank.py → loads cache/x_scores.npz                                                                                           
 ```                                                                                                                            
                                                                                                                                
 ────────────────────────────────────────────────────────────────────────────────                                               
                                                                                                                                
 Detailed 14-day execution plan                                                                                                 
                                                                                                                                
 Day 1: uv + baseline freeze                                                                                                    
                                                                                                                                
 Tasks:                                                                                                                         
                                                                                                                                
 ```bash                                                                                                                        
   uv init --bare                                                                                                               
   uv add numpy                                                                                                                 
   uv add --dev pytest ruff                                                                                                     
   uv run python rank.py --candidates data/candidates.jsonl --out output/baseline.csv                                           
   uv run python validate_submission.py output/baseline.csv                                                                     
   uv run python eval/evaluate.py                                                                                               
 ```                                                                                                                            
                                                                                                                                
 Deliverables:                                                                                                                  
                                                                                                                                
 ```text                                                                                                                        
   pyproject.toml                                                                                                               
   uv.lock                                                                                                                      
   output/baseline.csv                                                                                                          
   eval/baseline_metrics.txt                                                                                                    
 ```                                                                                                                            
                                                                                                                                
 Also tag current state:                                                                                                        
                                                                                                                                
 ```bash                                                                                                                        
   git tag baseline-working                                                                                                     
 ```                                                                                                                            
                                                                                                                                
 ────────────────────────────────────────────────────────────────────────────────                                               
                                                                                                                                
 Day 2: BM25 + RRF                                                                                                              
                                                                                                                                
 Implement:                                                                                                                     
                                                                                                                                
 ```text                                                                                                                        
   src/sparse_matcher.py                                                                                                        
   src/fusion.py                                                                                                                
   eval/ablation_fusion.py                                                                                                      
 ```                                                                                                                            
                                                                                                                                
 Run:                                                                                                                           
                                                                                                                                
 ```bash                                                                                                                        
   uv add rank-bm25                                                                                                             
   uv run python eval/ablation_fusion.py                                                                                        
 ```                                                                                                                            
                                                                                                                                
 Decision:                                                                                                                      
                                                                                                                                
 - If good, keep as optional final fusion.                                                                                      
 - If not, keep for ablation.                                                                                                   
                                                                                                                                
 ────────────────────────────────────────────────────────────────────────────────                                               
                                                                                                                                
 Days 3-4: Evidence Graph                                                                                                       
                                                                                                                                
 Implement:                                                                                                                     
                                                                                                                                
 ```text                                                                                                                        
   src/evidence_graph.py                                                                                                        
 ```                                                                                                                            
                                                                                                                                
 Integrate into:                                                                                                                
                                                                                                                                
 ```text                                                                                                                        
   src/ranker.py                                                                                                                
   src/reasoning.py                                                                                                             
   sandbox/app.py optional                                                                                                      
 ```                                                                                                                            
                                                                                                                                
 Run:                                                                                                                           
                                                                                                                                
 ```bash                                                                                                                        
   uv run python eval/evaluate.py                                                                                               
   uv run python eval/audit_evidence_graph.py                                                                                   
 ```                                                                                                                            
                                                                                                                                
 Manual inspect:                                                                                                                
                                                                                                                                
 ```bash                                                                                                                        
   head -30 output/submission.csv                                                                                               
 ```                                                                                                                            
                                                                                                                                
 Decision:                                                                                                                      
                                                                                                                                
 - If good, merge.                                                                                                              
 - If too strict, soften.                                                                                                       
 - If noisy, use only as explanation/penalty.                                                                                   
                                                                                                                                
 ────────────────────────────────────────────────────────────────────────────────                                               
                                                                                                                                
 Day 5: Robustness Audit                                                                                                        
                                                                                                                                
 Implement:                                                                                                                     
                                                                                                                                
 ```text                                                                                                                        
   eval/robustness_audit.py                                                                                                     
 ```                                                                                                                            
                                                                                                                                
 Test:                                                                                                                          
                                                                                                                                
 ```bash                                                                                                                        
   uv run python eval/robustness_audit.py --sample 500                                                                          
 ```                                                                                                                            
                                                                                                                                
 Generate:                                                                                                                      
                                                                                                                                
 ```text                                                                                                                        
   docs/robustness_audit.md                                                                                                     
 ```                                                                                                                            
                                                                                                                                
 This becomes a major PPT slide.                                                                                                
                                                                                                                                
 ────────────────────────────────────────────────────────────────────────────────                                               
                                                                                                                                
 Days 6-7: Learning-to-Rank                                                                                                     
                                                                                                                                
 Add dependencies:                                                                                                              
                                                                                                                                
 ```bash                                                                                                                        
   uv add --optional ltr scikit-learn pandas joblib                                                                             
 ```                                                                                                                            
                                                                                                                                
 Implement:                                                                                                                     
                                                                                                                                
 ```text                                                                                                                        
   precompute/train_ltr.py                                                                                                      
   src/ltr_model.py                                                                                                             
   eval/ablation_ltr.py                                                                                                         
 ```                                                                                                                            
                                                                                                                                
 Train:                                                                                                                         
                                                                                                                                
 ```bash                                                                                                                        
   uv run python precompute/train_ltr.py                                                                                        
   uv run python eval/ablation_ltr.py                                                                                           
 ```                                                                                                                            
                                                                                                                                
 Decision:                                                                                                                      
                                                                                                                                
 - If improves: use cache/ltr_scores.npz as ranker in RRF or score blend.                                                       
 - If not: include as ablation and keep out of final.                                                                           
                                                                                                                                
 ────────────────────────────────────────────────────────────────────────────────                                               
                                                                                                                                
 Days 8-10: GNN                                                                                                                 
                                                                                                                                
 Set up separate environment if needed:                                                                                         
                                                                                                                                
 ```bash                                                                                                                        
   uv venv --python 3.12 .venv-gnn                                                                                              
   uv pip install --python .venv-gnn torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121            
   uv pip install --python .venv-gnn torch-geometric scikit-learn pandas tqdm                                                   
 ```                                                                                                                            
                                                                                                                                
 Implement:                                                                                                                     
                                                                                                                                
 ```text                                                                                                                        
   precompute/build_graph.py                                                                                                    
   precompute/train_gnn.py                                                                                                      
   src/gnn_scores.py                                                                                                            
   eval/ablation_gnn.py                                                                                                         
 ```                                                                                                                            
                                                                                                                                
 Train:                                                                                                                         
                                                                                                                                
 ```bash                                                                                                                        
   .venv-gnn/bin/python precompute/build_graph.py                                                                               
   .venv-gnn/bin/python precompute/train_gnn.py --seed 1                                                                        
   .venv-gnn/bin/python precompute/train_gnn.py --seed 2                                                                        
   .venv-gnn/bin/python precompute/train_gnn.py --seed 3                                                                        
   uv run python eval/ablation_gnn.py                                                                                           
 ```                                                                                                                            
                                                                                                                                
 Decision:                                                                                                                      
                                                                                                                                
 - If stable and improves: use as weak RRF ranker.                                                                              
 - If unstable: do not use in final.                                                                                            
 - If very good: blend with current score but keep guardrails.                                                                  
                                                                                                                                
 ────────────────────────────────────────────────────────────────────────────────                                               
                                                                                                                                
 Day 11: Multi-agent LLM panel, optional                                                                                        
                                                                                                                                
 Only if everything else is stable.                                                                                             
                                                                                                                                
 Implement:                                                                                                                     
                                                                                                                                
 ```text                                                                                                                        
   precompute/llm_panel.py                                                                                                      
   cache/llm_panel.jsonl                                                                                                        
   src/llm_panel_reranker.py                                                                                                    
 ```                                                                                                                            
                                                                                                                                
 Run for top 100 or 200 only.                                                                                                   
                                                                                                                                
 Decision:                                                                                                                      
                                                                                                                                
 - If better than current LLM rerank, use.                                                                                      
 - Else keep existing.                                                                                                          
                                                                                                                                
 ────────────────────────────────────────────────────────────────────────────────                                               
                                                                                                                                
 Day 12: Final ablation and audit                                                                                               
                                                                                                                                
 Run all:                                                                                                                       
                                                                                                                                
 ```bash                                                                                                                        
   uv run python eval/ablation_report.py                                                                                        
   uv run python eval/robustness_audit.py                                                                                       
   uv run python rank.py --candidates data/candidates.jsonl --out output/submission.csv                                         
   uv run python validate_submission.py output/submission.csv                                                                   
 ```                                                                                                                            
                                                                                                                                
 Create:                                                                                                                        
                                                                                                                                
 ```text                                                                                                                        
   docs/final_ablation.md                                                                                                       
   docs/top100_audit.md                                                                                                         
   docs/robustness_audit.md                                                                                                     
 ```                                                                                                                            
                                                                                                                                
 ────────────────────────────────────────────────────────────────────────────────                                               
                                                                                                                                
 Day 13: Deck                                                                                                                   
                                                                                                                                
 Slides:                                                                                                                        
                                                                                                                                
 1. Problem                                                                                                                     
 2. Why keyword matching fails                                                                                                  
 3. System architecture                                                                                                         
 4. Evidence graph idea                                                                                                         
 5. Hybrid/RRF retrieval                                                                                                        
 6. GNN / LTR experiments                                                                                                       
 7. Anti-gaming robustness                                                                                                      
 8. Final ranking results                                                                                                       
 9. Runtime/reproducibility                                                                                                     
 10. Limitations/future work                                                                                                    
                                                                                                                                
 ────────────────────────────────────────────────────────────────────────────────                                               
                                                                                                                                
 Day 14: Freeze                                                                                                                 
                                                                                                                                
 Final commands:                                                                                                                
                                                                                                                                
 ```bash                                                                                                                        
   uv run python rank.py --candidates data/candidates.jsonl --out submission.csv                                                
   uv run python validate_submission.py submission.csv                                                                          
   uv run python eval/evaluate.py                                                                                               
   git status                                                                                                                   
 ```                                                                                                                            
                                                                                                                                
 No risky changes.                                                                                                              
                                                                                                                                
 ────────────────────────────────────────────────────────────────────────────────                                               
                                                                                                                                
 Recommended final method selection hierarchy                                                                                   
                                                                                                                                
 When deciding final ranking, use this priority:                                                                                
                                                                                                                                
 ```text                                                                                                                        
   1. Current working ranker as safety baseline                                                                                 
   2. Add evidence graph if it improves trust                                                                                   
   3. Add RRF if it improves robustness                                                                                         
   4. Add LTR if it improves heldout labels and manual audit                                                                    
   5. Add GNN only if stable across seeds                                                                                       
   6. Add multi-agent LLM only if clearly better than current cached LLM                                                        
 ```                                                                                                                            
                                                                                                                                
 The most likely best final system:                                                                                             
                                                                                                                                
 ```text                                                                                                                        
   Hard filters                                                                                                                 
   + rule/semantic/evidence graph rankers                                                                                       
   + RRF fusion                                                                                                                 
   + cached LLM rerank                                                                                                          
   + optional LTR tie-breaker                                                                                                   
 ```                                                                                                                            
                                                                                                                                
 The GNN may or may not beat this. That is okay.                                                                                
                                                                                                                                
 ────────────────────────────────────────────────────────────────────────────────                                               
                                                                                                                                
 How to talk about failed approaches                                                                                            
                                                                                                                                
 This is important. If a research approach fails, do not hide it. Present it as engineering maturity.                           
                                                                                                                                
 Example:                                                                                                                       
                                                                                                                                
 │ “We implemented a heterogeneous GNN over candidate, skill, company, title, and evidence-concept nodes. On our small labeled  
 │ set it improved recall but introduced instability in the top-20 across random seeds. Because recruiter trust is more         
 │ important than model novelty, we retained the graph as an evidence-based symbolic scorer and used the GNN only as an         
 │ ablation.”                                                                                                                   
                                                                                                                                
 That sounds better than blindly using a worse GNN.                                                                             
                                                                                                                                
 For LTR:                                                                                                                       
                                                                                                                                
 │ “LTR improved heldout labels but slightly over-weighted behavioral features, so we constrained it to top-window tie-breaking 
 │ rather than full-pool ranking.”                                                                                              
                                                                                                                                
 For BM25:                                                                                                                      
                                                                                                                                
 │ “BM25 improved exact matching for vector DB/evaluation terms but was vulnerable to keyword stuffing, so we fused it using    
 │ RRF after hard filters and contradiction penalties.”                                                                         
                                                                                                                                
 This is exactly how real ML systems are built.                                                                                 
                                                                                                                                
 ────────────────────────────────────────────────────────────────────────────────                                               
                                                                                                                                
 The core principle                                                                                                             
                                                                                                                                
 Do not let the project become:                                                                                                 
                                                                                                                                
 │ “I added every model.”                                                                                                       
                                                                                                                                
 Make it:                                                                                                                       
                                                                                                                                
 │ “I built a ranking laboratory, tested multiple research-inspired rankers, and selected the most recruiter-trustworthy        
 │ combination under real constraints.”  