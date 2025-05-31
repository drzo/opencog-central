#!/usr/bin/env python3
"""
NanoCog Server

A FastAPI server that provides a chat interface to a CogPrime-trained nanoGPT model.
Features include:
- Standard chat completion endpoints
- Streaming response support
- Introspective diagnostics for CogPrime systems
- AtomSpace connectivity for live agent analysis
- Proper error handling and logging
"""

import os
import sys
import json
import logging
import argparse
from typing import List, Dict, Any, Optional, Union, AsyncGenerator
from datetime import datetime
from contextlib import asynccontextmanager, nullcontext

import torch
import tiktoken
import numpy as np
import requests
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request, status
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add the parent directory to sys.path to import nanoGPT modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import GPTConfig, GPT

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("nanocog")

# --- Model Configuration ---
class ModelConfig:
    """Configuration for the nanoGPT model."""
    def __init__(self, model_path: str, device: str = "cuda", max_tokens: int = 2048):
        self.model_path = model_path
        self.device = device
        self.max_tokens = max_tokens
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load the nanoGPT model from checkpoint."""
        try:
            logger.info(f"Loading model from {self.model_path}")
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Configure model from checkpoint
            gptconf = GPTConfig(**checkpoint['model_args'])
            self.model = GPT(gptconf)
            
            # Handle potential prefix in state dict keys
            state_dict = checkpoint['model']
            unwanted_prefix = '_orig_mod.'
            for k, v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            
            self.model.load_state_dict(state_dict)
            self.model.eval()
            self.model.to(self.device)
            
            # Set up tokenizer
            self.tokenizer = tiktoken.get_encoding("gpt2")
            
            logger.info("Model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise RuntimeError(f"Failed to load model: {str(e)}")

    def generate(self, prompt: str, max_new_tokens: int = 500, 
                 temperature: float = 0.7, top_k: int = 200,
                 stream: bool = False) -> Union[str, AsyncGenerator[str, None]]:
        """
        Generate text from the model.
        
        Args:
            prompt: The input prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Only sample from the top k most likely tokens
            stream: If True, stream the response token by token
            
        Returns:
            Generated text or an async generator for streaming
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Encode the prompt
        input_ids = self.tokenizer.encode(prompt, allowed_special={"<|endoftext|>"})
        
        # Truncate if needed to fit within context window
        if len(input_ids) > self.max_tokens - max_new_tokens:
            logger.warning(f"Prompt too long ({len(input_ids)} tokens), truncating")
            input_ids = input_ids[-(self.max_tokens - max_new_tokens):]
        
        # Convert to tensor
        x = torch.tensor(input_ids, dtype=torch.long, device=self.device)[None, ...]
        
        if stream:
            return self._stream_generate(x, max_new_tokens, temperature, top_k)
        else:
            return self._batch_generate(x, max_new_tokens, temperature, top_k)
    
    def _batch_generate(self, x, max_new_tokens, temperature, top_k):
        """Generate text in a single batch."""
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda' if 'cuda' in self.device else 'cpu'):
                y = self.model.generate(
                    x, max_new_tokens, temperature=temperature, top_k=top_k
                )
                generated_text = self.tokenizer.decode(y[0].tolist())
                # Remove the prompt from the generated text
                prompt_text = self.tokenizer.decode(x[0].tolist())
                if generated_text.startswith(prompt_text):
                    generated_text = generated_text[len(prompt_text):]
                return generated_text
    
    async def _stream_generate(self, x, max_new_tokens, temperature, top_k):
        """Stream tokens as they're generated."""
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda' if 'cuda' in self.device else 'cpu'):
                # Get the input sequence length to know where to start yielding new tokens
                input_len = x.shape[1]
                
                # Initialize past_key_values for more efficient generation
                past = None
                for token_index in range(max_new_tokens):
                    # Forward pass
                    if past is None:
                        # First forward pass with the full input
                        outputs = self.model(x, use_cache=True)
                        logits = outputs.logits
                        past = outputs.past_key_values
                    else:
                        # Subsequent passes with just the new token and cached past
                        outputs = self.model(
                            x[:, -1:], use_cache=True, past_key_values=past
                        )
                        logits = outputs.logits
                        past = outputs.past_key_values
                    
                    # Get logits for the next token and sample
                    next_token_logits = logits[:, -1, :]
                    
                    # Apply temperature
                    if temperature > 0:
                        next_token_logits = next_token_logits / temperature
                    
                    # Apply top-k filtering
                    if top_k > 0:
                        v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                        next_token_logits[next_token_logits < v[:, [-1]]] = float('-inf')
                    
                    # Sample from the filtered distribution
                    probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    
                    # Append the new token to the input
                    x = torch.cat((x, next_token), dim=1)
                    
                    # Decode and yield the new token
                    new_token_text = self.tokenizer.decode([next_token[0].item()])
                    yield new_token_text
                    
                    # Check if we've hit the end of text token
                    if next_token[0].item() == self.tokenizer.eot_token:
                        break

# --- AtomSpace Connection ---
class AtomSpaceClient:
    """Client for connecting to and querying an AtomSpace instance."""
    
    def __init__(self, endpoint: str):
        """
        Initialize the AtomSpace client.
        
        Args:
            endpoint: The URL of the AtomSpace REST API
        """
        self.endpoint = endpoint
        self.session = requests.Session()
    
    def test_connection(self) -> bool:
        """Test the connection to the AtomSpace."""
        try:
            response = self.session.get(f"{self.endpoint}/status", timeout=5)
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Failed to connect to AtomSpace at {self.endpoint}: {str(e)}")
            return False
    
    def get_atom_count(self) -> int:
        """Get the total number of atoms in the AtomSpace."""
        try:
            response = self.session.get(f"{self.endpoint}/atoms/count", timeout=5)
            response.raise_for_status()
            return response.json().get("count", 0)
        except Exception as e:
            logger.error(f"Failed to get atom count: {str(e)}")
            return 0
    
    def get_atoms_by_type(self, atom_type: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get atoms of a specific type."""
        try:
            response = self.session.get(
                f"{self.endpoint}/atoms/type/{atom_type}",
                params={"limit": limit},
                timeout=10
            )
            response.raise_for_status()
            return response.json().get("atoms", [])
        except Exception as e:
            logger.error(f"Failed to get atoms by type {atom_type}: {str(e)}")
            return []
    
    def get_high_sti_atoms(self, threshold: float = 0.5, limit: int = 100) -> List[Dict[str, Any]]:
        """Get atoms with STI above a threshold."""
        try:
            # This is a conceptual endpoint - actual implementation depends on the AtomSpace REST API
            response = self.session.get(
                f"{self.endpoint}/atoms/sti",
                params={"threshold": threshold, "limit": limit},
                timeout=10
            )
            response.raise_for_status()
            return response.json().get("atoms", [])
        except Exception as e:
            logger.error(f"Failed to get high STI atoms: {str(e)}")
            return []
    
    def get_active_goals(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get currently active goals in the system."""
        try:
            # This is a conceptual endpoint - actual implementation depends on the AtomSpace REST API
            response = self.session.get(
                f"{self.endpoint}/goals/active",
                params={"limit": limit},
                timeout=10
            )
            response.raise_for_status()
            return response.json().get("goals", [])
        except Exception as e:
            logger.error(f"Failed to get active goals: {str(e)}")
            return []
    
    def get_attention_allocation_summary(self) -> Dict[str, Any]:
        """Get a summary of the attention allocation in the system."""
        try:
            # This is a conceptual endpoint - actual implementation depends on the AtomSpace REST API
            response = self.session.get(
                f"{self.endpoint}/attention/summary",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get attention allocation summary: {str(e)}")
            return {}
    
    def get_agent_introspection_data(self) -> Dict[str, Any]:
        """
        Get comprehensive introspection data about the agent.
        This aggregates multiple queries to build a complete picture.
        """
        data = {
            "timestamp": datetime.now().isoformat(),
            "atom_count": self.get_atom_count(),
            "active_goals": self.get_active_goals(),
            "attention_summary": self.get_attention_allocation_summary(),
            "high_sti_atoms": self.get_high_sti_atoms(),
            # Add more data points as needed
        }
        return data

# --- API Models ---
class ChatMessage(BaseModel):
    """A chat message."""
    role: str = Field(..., description="The role of the message sender (user or assistant)")
    content: str = Field(..., description="The content of the message")

class ChatRequest(BaseModel):
    """Request model for chat completion."""
    messages: List[ChatMessage] = Field(..., description="The conversation history")
    max_tokens: int = Field(500, description="Maximum number of tokens to generate")
    temperature: float = Field(0.7, description="Sampling temperature")
    top_k: int = Field(200, description="Top-k sampling parameter")
    stream: bool = Field(False, description="Whether to stream the response")

class DiagnosticRequest(BaseModel):
    """Request model for introspective diagnostics."""
    atomspace_endpoint: str = Field(..., description="The AtomSpace REST API endpoint")
    focus_areas: List[str] = Field(default=["attention", "goals", "patterns"], 
                                  description="Areas to focus the diagnostic on")
    max_tokens: int = Field(1000, description="Maximum number of tokens to generate")
    temperature: float = Field(0.6, description="Sampling temperature")
    stream: bool = Field(False, description="Whether to stream the response")

class ChatResponse(BaseModel):
    """Response model for chat completion."""
    text: str = Field(..., description="The generated text")
    model: str = Field(..., description="The model used for generation")
    created_at: str = Field(..., description="Timestamp of the response")
    tokens_generated: int = Field(..., description="Number of tokens generated")

class DiagnosticResponse(BaseModel):
    """Response model for introspective diagnostics."""
    analysis: str = Field(..., description="The diagnostic analysis")
    raw_data: Dict[str, Any] = Field(..., description="The raw data used for the analysis")
    recommendations: List[str] = Field(..., description="List of recommendations")
    model: str = Field(..., description="The model used for generation")
    created_at: str = Field(..., description="Timestamp of the response")

# --- FastAPI App ---
# Use lifespan to load the model when the app starts
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model on startup
    try:
        app.state.model_config.load_model()
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        # We'll continue and let individual endpoints handle the error
    
    yield
    
    # Clean up on shutdown
    if hasattr(app.state, "model_config") and app.state.model_config.model:
        logger.info("Cleaning up model resources")
        # Any cleanup needed for the model

# Create the FastAPI app
app = FastAPI(
    title="NanoCog API",
    description="API for interacting with a CogPrime-trained nanoGPT model",
    version="0.1.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Dependency Injection ---
def get_model_config(request: Request) -> ModelConfig:
    """Dependency to get the model configuration."""
    if not hasattr(request.app.state, "model_config") or not request.app.state.model_config.model:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded or initialization failed"
        )
    return request.app.state.model_config

def get_atomspace_client(atomspace_endpoint: str) -> AtomSpaceClient:
    """Dependency to get an AtomSpace client."""
    client = AtomSpaceClient(atomspace_endpoint)
    if not client.test_connection():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to connect to AtomSpace at {atomspace_endpoint}"
        )
    return client

# --- API Routes ---
@app.get("/")
async def root():
    """Root endpoint with basic information."""
    return {
        "name": "NanoCog API",
        "description": "API for interacting with a CogPrime-trained nanoGPT model",
        "status": "operational",
        "version": "0.1.0",
    }

@app.get("/status")
async def status(request: Request):
    """Status endpoint with model information."""
    model_loaded = hasattr(request.app.state, "model_config") and request.app.state.model_config.model is not None
    
    return {
        "status": "operational" if model_loaded else "initializing",
        "model_loaded": model_loaded,
        "model_path": request.app.state.model_config.model_path if model_loaded else None,
        "device": request.app.state.model_config.device if model_loaded else None,
        "timestamp": datetime.now().isoformat(),
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    model_config: ModelConfig = Depends(get_model_config)
):
    """
    Generate a chat completion.
    
    This endpoint takes a conversation history and generates a response.
    """
    try:
        # Format the prompt from the conversation history
        prompt = format_chat_prompt(request.messages)
        
        # Generate text
        generated_text = model_config.generate(
            prompt=prompt,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            stream=False
        )
        
        # Count tokens in the generated text
        tokens_generated = len(model_config.tokenizer.encode(generated_text))
        
        return ChatResponse(
            text=generated_text,
            model=os.path.basename(model_config.model_path),
            created_at=datetime.now().isoformat(),
            tokens_generated=tokens_generated
        )
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating response: {str(e)}"
        )

@app.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    model_config: ModelConfig = Depends(get_model_config)
):
    """
    Stream a chat completion token by token.
    
    This endpoint takes a conversation history and streams the generated response.
    """
    try:
        # Format the prompt from the conversation history
        prompt = format_chat_prompt(request.messages)
        
        # Set stream to True to get a generator
        async def generate_stream():
            try:
                async for token in model_config.generate(
                    prompt=prompt,
                    max_new_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_k=request.top_k,
                    stream=True
                ):
                    # Yield each token as a JSON object
                    yield f"data: {json.dumps({'token': token})}\n\n"
            except Exception as e:
                logger.error(f"Error in streaming generation: {str(e)}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
            
            # End the stream
            yield f"data: [DONE]\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream"
        )
    except Exception as e:
        logger.error(f"Error in chat stream endpoint: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating streaming response: {str(e)}"
        )

@app.post("/diagnostics", response_model=DiagnosticResponse)
async def run_diagnostics(
    request: DiagnosticRequest,
    model_config: ModelConfig = Depends(get_model_config)
):
    """
    Run introspective diagnostics on a CogPrime agent.
    
    This endpoint connects to an AtomSpace, gathers data, and generates an analysis.
    """
    try:
        # Get AtomSpace client
        atomspace_client = get_atomspace_client(request.atomspace_endpoint)
        
        # Gather introspection data
        introspection_data = atomspace_client.get_agent_introspection_data()
        
        # Format a prompt for the model that includes the introspection data
        prompt = format_diagnostic_prompt(introspection_data, request.focus_areas)
        
        # Generate the diagnostic analysis
        analysis = model_config.generate(
            prompt=prompt,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_k=50,  # Lower top_k for more focused analysis
            stream=False
        )
        
        # Extract recommendations from the analysis
        recommendations = extract_recommendations(analysis)
        
        return DiagnosticResponse(
            analysis=analysis,
            raw_data=introspection_data,
            recommendations=recommendations,
            model=os.path.basename(model_config.model_path),
            created_at=datetime.now().isoformat()
        )
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error in diagnostics endpoint: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating diagnostic: {str(e)}"
        )

@app.post("/diagnostics/stream")
async def diagnostics_stream(
    request: DiagnosticRequest,
    model_config: ModelConfig = Depends(get_model_config)
):
    """
    Stream introspective diagnostics on a CogPrime agent.
    
    This endpoint connects to an AtomSpace, gathers data, and streams the analysis.
    """
    try:
        # Get AtomSpace client
        atomspace_client = get_atomspace_client(request.atomspace_endpoint)
        
        # Gather introspection data
        introspection_data = atomspace_client.get_agent_introspection_data()
        
        # Format a prompt for the model that includes the introspection data
        prompt = format_diagnostic_prompt(introspection_data, request.focus_areas)
        
        # Stream the diagnostic analysis
        async def generate_stream():
            # First yield the metadata
            yield f"data: {json.dumps({'type': 'metadata', 'raw_data': introspection_data})}\n\n"
            
            # Then stream the analysis
            try:
                async for token in model_config.generate(
                    prompt=prompt,
                    max_new_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_k=50,  # Lower top_k for more focused analysis
                    stream=True
                ):
                    yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"
            except Exception as e:
                logger.error(f"Error in streaming diagnostics: {str(e)}")
                yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
            
            # End the stream
            yield f"data: [DONE]\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream"
        )
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error in diagnostics stream endpoint: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error streaming diagnostic: {str(e)}"
        )

# --- Utility Functions ---
def format_chat_prompt(messages: List[ChatMessage]) -> str:
    """
    Format a conversation history into a prompt for the model.
    
    Args:
        messages: List of ChatMessage objects
        
    Returns:
        Formatted prompt string
    """
    prompt = ""
    for message in messages:
        if message.role.lower() == "user":
            prompt += f"\nUser: {message.content}\n"
        elif message.role.lower() == "assistant":
            prompt += f"\nNanoCog: {message.content}\n"
        elif message.role.lower() == "system":
            # System messages are prepended with a special format
            prompt += f"\n# System Instruction: {message.content}\n"
    
    # Add the final assistant prefix to prompt the model to respond
    prompt += "\nNanoCog: "
    return prompt

def format_diagnostic_prompt(introspection_data: Dict[str, Any], focus_areas: List[str]) -> str:
    """
    Format introspection data into a prompt for diagnostic analysis.
    
    Args:
        introspection_data: Dictionary of data from the AtomSpace
        focus_areas: List of areas to focus the diagnostic on
        
    Returns:
        Formatted prompt string
    """
    # Create a summary of the data for the prompt
    summary = []
    
    # Add atom count
    summary.append(f"Total atoms: {introspection_data.get('atom_count', 'unknown')}")
    
    # Add active goals if available
    if 'active_goals' in introspection_data and introspection_data['active_goals']:
        summary.append(f"Active goals: {len(introspection_data['active_goals'])}")
        for i, goal in enumerate(introspection_data['active_goals'][:5]):  # Show top 5
            goal_name = goal.get('name', 'Unnamed')
            goal_sti = goal.get('sti', 0.0)
            summary.append(f"  Goal {i+1}: {goal_name} (STI: {goal_sti:.2f})")
    
    # Add attention summary if available
    if 'attention_summary' in introspection_data:
        att_summary = introspection_data['attention_summary']
        if isinstance(att_summary, dict):
            summary.append("Attention allocation:")
            for key, value in att_summary.items():
                if isinstance(value, (int, float)):
                    summary.append(f"  {key}: {value}")
    
    # Add high STI atoms if available
    if 'high_sti_atoms' in introspection_data:
        high_sti = introspection_data['high_sti_atoms']
        summary.append(f"High STI atoms: {len(high_sti)}")
        atom_types = {}
        for atom in high_sti:
            atom_type = atom.get('type', 'unknown')
            atom_types[atom_type] = atom_types.get(atom_type, 0) + 1
        for atom_type, count in atom_types.items():
            summary.append(f"  {atom_type}: {count}")
    
    # Create the prompt
    prompt = f"""# System Instruction: You are NanoCog, an AI assistant specialized in CogPrime architecture and OpenCog systems. You're analyzing a live CogPrime agent's AtomSpace. Provide detailed introspective diagnostics based on the data below, focusing on: {', '.join(focus_areas)}.

## AtomSpace Data ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
{chr(10).join(summary)}

## Raw Data (JSON)
```json
{json.dumps(introspection_data, indent=2)}
```

## Task
Analyze the above AtomSpace data and provide:
1. A summary of the agent's current cognitive state
2. Identification of any bottlenecks or issues
3. Specific recommendations for optimization
4. Relevant CogPrime principles that apply

NanoCog (Diagnostic Analysis): 
"""
    return prompt

def extract_recommendations(analysis: str) -> List[str]:
    """
    Extract recommendations from the generated analysis.
    
    This is a simple implementation that looks for numbered lists or bullet points.
    A more sophisticated implementation could use regex or NLP techniques.
    
    Args:
        analysis: The generated diagnostic analysis
        
    Returns:
        List of extracted recommendations
    """
    recommendations = []
    
    # Look for lines that start with numbers or bullet points
    for line in analysis.split('\n'):
        line = line.strip()
        # Check for numbered recommendations (e.g., "1. Do this")
        if (line.startswith(("1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.", "0.")) and 
            len(line) > 3 and line[2] == ' '):
            recommendations.append(line[3:].strip())
        # Check for bullet points
        elif line.startswith(("- ", "* ", "• ")):
            recommendations.append(line[2:].strip())
        # Check for "Recommendation:" prefix
        elif line.lower().startswith("recommendation:"):
            recommendations.append(line[14:].strip())
    
    # If we didn't find any structured recommendations, try to extract sentences with suggestive words
    if not recommendations:
        suggestive_words = ["should", "recommend", "consider", "try", "increase", "decrease", "optimize"]
        for line in analysis.split('\n'):
            for word in suggestive_words:
                if word in line.lower():
                    recommendations.append(line.strip())
                    break
    
    return recommendations

# --- Error Handling ---
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "An unexpected error occurred"}
    )

# --- Main Function ---
def main():
    """Run the NanoCog server."""
    parser = argparse.ArgumentParser(description="NanoCog Server")
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to the model checkpoint")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run the model on (cuda, cpu, mps)")
    parser.add_argument("--max_tokens", type=int, default=2048,
                        help="Maximum number of tokens in context")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host to run the server on")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port to run the server on")
    
    args = parser.parse_args()
    
    # Create model config
    model_config = ModelConfig(
        model_path=args.model_path,
        device=args.device,
        max_tokens=args.max_tokens
    )
    
    # Store the model config in the app state
    app.state.model_config = model_config
    
    # Run the server
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
