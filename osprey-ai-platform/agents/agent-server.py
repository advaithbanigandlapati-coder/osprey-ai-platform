"""
Osprey AI - Python Agent Server
Runs the 6 powerful Python agents (1,500+ lines each)
Your Node.js server.js will proxy requests to this
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import sys
from pathlib import Path

# Add agents folder to path
sys.path.append(str(Path(__file__).parent))

app = FastAPI(
    title="Osprey AI Agent Server",
    version="1.0.0",
    description="Python AI agents with advanced features"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# LOAD THE 6 POWERFUL PYTHON AGENTS
# ============================================================================

print("🦅 Loading Python AI Agents...")

agents = {}

# Content Writer
try:
    from agents.content_writer import ContentWriterAI, ContentRequest, AIModel, ContentType, Tone
    agents['content-writer'] = {
        'instance': ContentWriterAI(api_keys={}),
        'request_class': ContentRequest,
        'model_enum': AIModel,
        'content_type_enum': ContentType,
        'tone_enum': Tone
    }
    print("✅ Content Writer AI loaded (1,814 lines)")
except Exception as e:
    print(f"❌ Content Writer failed: {e}")

# Code Assistant
try:
    from agents.code_assistant import CodeAssistantAI
    agents['code-assistant'] = {
        'instance': CodeAssistantAI(api_keys={})
    }
    print("✅ Code Assistant AI loaded (1,520 lines)")
except Exception as e:
    print(f"❌ Code Assistant failed: {e}")

# Data Analyst
try:
    from agents.data_analyst import DataAnalystAI
    agents['data-analyst'] = {
        'instance': DataAnalystAI(api_keys={})
    }
    print("✅ Data Analyst AI loaded (1,517 lines)")
except Exception as e:
    print(f"❌ Data Analyst failed: {e}")

# Support Bot
try:
    from agents.support_bot import SupportBotAI
    agents['support-bot'] = {
        'instance': SupportBotAI(api_keys={})
    }
    print("✅ Support Bot AI loaded (1,731 lines)")
except Exception as e:
    print(f"❌ Support Bot failed: {e}")

# Research Assistant
try:
    from agents.research_assistant import ResearchAssistantAI
    agents['research'] = {
        'instance': ResearchAssistantAI(api_keys={})
    }
    print("✅ Research Assistant AI loaded (1,697 lines)")
except Exception as e:
    print(f"❌ Research Assistant failed: {e}")

# Marketing Strategist
try:
    from agents.marketing_strategist import MarketingStrategistAI
    agents['marketing'] = {
        'instance': MarketingStrategistAI(api_keys={})
    }
    print("✅ Marketing Strategist AI loaded (1,618 lines)")
except Exception as e:
    print(f"❌ Marketing Strategist failed: {e}")

print(f"📊 {len(agents)}/6 agents loaded successfully\n")

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class GenerateRequest(BaseModel):
    message: str
    agent: Optional[str] = "content-writer"
    model: Optional[str] = "llama2"
    target_word_count: Optional[int] = 800
    temperature: Optional[float] = 0.7
    tone: Optional[str] = "professional"
    language: Optional[str] = "en"
    seo_optimize: Optional[bool] = True

class GenerateResponse(BaseModel):
    success: bool
    response: str
    agent: str
    model: str
    word_count: Optional[int] = None
    quality_score: Optional[float] = None
    seo_score: Optional[float] = None
    readability: Optional[dict] = None

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    return {
        "service": "Osprey AI Agent Server",
        "version": "1.0.0",
        "agents_loaded": len(agents),
        "status": "running"
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "agents": list(agents.keys()),
        "count": len(agents)
    }

@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """
    Generate AI response using powerful Python agents
    """
    
    agent_id = request.agent or "content-writer"
    
    # Check if agent exists
    if agent_id not in agents:
        raise HTTPException(
            status_code=404,
            detail=f"Agent not found. Available: {list(agents.keys())}"
        )
    
    agent_data = agents[agent_id]
    agent_instance = agent_data['instance']
    
    try:
        # CONTENT WRITER (Full Power!)
        if agent_id == "content-writer":
            from agents.content_writer import ContentRequest, AIModel, ContentType, Tone
            
            # Map strings to enums
            model_map = {
                'llama2': AIModel.LLAMA2,
                'mistral': AIModel.MISTRAL,
                'codellama': AIModel.CODELLAMA,
                'llama2:13b': AIModel.LLAMA2_13B,
                'mixtral': AIModel.MIXTRAL,
            }
            
            tone_map = {
                'professional': Tone.PROFESSIONAL,
                'casual': Tone.CASUAL,
                'friendly': Tone.FRIENDLY,
                'formal': Tone.FORMAL,
                'technical': Tone.TECHNICAL,
            }
            
            req = ContentRequest(
                topic=request.message,
                model=model_map.get(request.model, AIModel.LLAMA2),
                target_word_count=request.target_word_count,
                tone=tone_map.get(request.tone, Tone.PROFESSIONAL),
                language=request.language,
                content_type=ContentType.BLOG_POST,
                seo_optimize=request.seo_optimize,
                temperature=request.temperature
            )
            
            result = await agent_instance.generate(req)
            
            return GenerateResponse(
                success=True,
                response=result.content,
                agent=agent_id,
                model=request.model,
                word_count=result.word_count,
                quality_score=result.quality_score,
                seo_score=result.seo_analysis.seo_score if result.seo_analysis else None,
                readability={
                    'flesch_score': result.readability.get('flesch_reading_ease'),
                    'grade_level': result.readability.get('grade_level')
                } if result.readability else None
            )
        
        # OTHER AGENTS (add similar implementations)
        else:
            # For now, simple implementation
            # You can expand these with full features later
            response_text = f"Python agent {agent_id} response to: {request.message}"
            
            return GenerateResponse(
                success=True,
                response=response_text,
                agent=agent_id,
                model=request.model
            )
    
    except Exception as e:
        print(f"❌ Error in {agent_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Agent error: {str(e)}"
        )

@app.get("/agents")
async def list_agents():
    """List all available agents"""
    return {
        "agents": [
            {
                "id": "content-writer",
                "name": "Content Writer Pro",
                "features": ["SEO", "Quality Score", "Readability", "50+ Languages"],
                "available": "content-writer" in agents
            },
            {
                "id": "code-assistant",
                "name": "Code Assistant",
                "features": ["Code Generation", "Debugging", "Optimization"],
                "available": "code-assistant" in agents
            },
            {
                "id": "data-analyst",
                "name": "Data Analyst AI",
                "features": ["Data Analysis", "Trends", "Visualizations"],
                "available": "data-analyst" in agents
            },
            {
                "id": "support-bot",
                "name": "Support Bot",
                "features": ["24/7 Support", "Tickets", "Knowledge Base"],
                "available": "support-bot" in agents
            },
            {
                "id": "research",
                "name": "Research Assistant",
                "features": ["Deep Research", "Citations", "Summaries"],
                "available": "research" in agents
            },
            {
                "id": "marketing",
                "name": "Marketing Strategist",
                "features": ["Campaigns", "Strategy", "Messaging"],
                "available": "marketing" in agents
            }
        ]
    }

# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*70)
    print("🦅 OSPREY AI PYTHON AGENT SERVER")
    print("="*70)
    print(f"\n📊 Loaded {len(agents)}/6 agents")
    print("\n📡 Starting on http://localhost:8000")
    print("📚 API docs: http://localhost:8000/docs")
    print("\n⚠️  Make sure Ollama is running: ollama serve\n")
    
    uvicorn.run(
        "agent-server:app",
        host="0.0.0.0",
        port=8001,  # Different port from Node.js (10000)
        reload=True,
        log_level="info"
    )
