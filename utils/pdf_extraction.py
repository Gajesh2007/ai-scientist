from __future__ import annotations

"""utils.pdf_extraction

Async PDF-to-markdown extraction using vision models.

Strategy
--------
1. Download PDF if needed (via arXiv ID, URL, or local path)
2. Convert each page to PNG using pdf2image
3. Send images to vision-capable LLMs (GPT-4V, Claude, Gemini)
4. Return structured markdown with per-page results

Dependencies:
- pdf2image (pip install pdf2image)
- poppler (system dependency: brew install poppler / apt install poppler-utils)
"""

import asyncio
import base64
import io
import logging
import os
import tempfile
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

from pdf2image import convert_from_path
from PIL import Image

from utils import arxiv as arxiv_api
from utils.llm import UnifiedLLM, Message, ModelType, LLMProvider

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class PdfDocument:
    """Extracted document with markdown content and metadata."""
    
    text: str  # Full markdown text
    pages: List[str]  # Per-page markdown content
    title: Optional[str] = None
    arxiv_id: Optional[str] = None
    local_path: Optional[str] = None
    page_count: int = 0
    model_used: Optional[str] = None
    total_tokens: dict = field(default_factory=lambda: {"input": 0, "output": 0})

    def __str__(self) -> str:
        name = self.arxiv_id or (Path(self.local_path).name if self.local_path else "<pdf>")
        return f"PdfDocument({name}, {self.page_count} pages)"


# ---------------------------------------------------------------------------
# Image encoding
# ---------------------------------------------------------------------------

def encode_pil_image(pil_image: Image.Image, format: str = "PNG") -> str:
    """Convert PIL image to base64 data URL."""
    buffer = io.BytesIO()
    pil_image.save(buffer, format=format)
    img_bytes = buffer.getvalue()
    b64 = base64.b64encode(img_bytes).decode('utf-8')
    return f"data:image/{format.lower()};base64,{b64}"


# ---------------------------------------------------------------------------
# Model detection
# ---------------------------------------------------------------------------

def get_vision_model(provider: Optional[LLMProvider] = None) -> tuple[LLMProvider, str]:
    """Determine the best available vision model based on environment."""
    # If provider specified, use appropriate vision model
    if provider == LLMProvider.OPENAI:
        return LLMProvider.OPENAI, ModelType.GPT_4O.value
    elif provider == LLMProvider.ANTHROPIC:
        return LLMProvider.ANTHROPIC, ModelType.CLAUDE_4_SONNET.value
    elif provider == LLMProvider.OPENROUTER:
        # Use Claude via OpenRouter for best quality
        return LLMProvider.OPENROUTER, "anthropic/claude-3-5-sonnet"
    
    # Auto-detect based on available keys
    if os.getenv("OPENAI_API_KEY"):
        return LLMProvider.OPENAI, ModelType.GPT_4O.value
    elif os.getenv("ANTHROPIC_API_KEY"):
        return LLMProvider.ANTHROPIC, ModelType.CLAUDE_4_SONNET.value
    elif os.getenv("OPENROUTER_API_KEY"):
        return LLMProvider.OPENROUTER, "anthropic/claude-3-5-sonnet"
    else:
        raise ValueError(
            "No vision-capable API keys found. Set OPENAI_API_KEY, ANTHROPIC_API_KEY, or OPENROUTER_API_KEY"
        )


# ---------------------------------------------------------------------------
# Vision LLM processing
# ---------------------------------------------------------------------------

async def process_page_async(
    image: Image.Image,
    page_num: int,
    llm: UnifiedLLM,
    provider: LLMProvider,
    model: str,
    maintain_format: bool = False,
    prior_page: Optional[str] = None,
) -> tuple[str, dict]:
    """Process a single page through vision LLM."""
    # Encode image
    image_url = encode_pil_image(image)
    
    # Build messages
    messages = []
    
    # System prompt
    system_prompt = (
        "Convert this research paper page to clean markdown. "
        "Preserve ALL content including mathematical notation (LaTeX), tables, and section structure. "
        "For figures/charts, provide a brief description in [Figure: description] format. "
        "Do not add any commentary or explanations - only convert what you see."
    )
    
    if maintain_format and prior_page:
        system_prompt += (
            f"\n\nMaintain consistent formatting with the previous page:\n"
            f"```markdown\n{prior_page[:500]}...\n```"
        )
    
    messages.append(Message(role="system", content=system_prompt))
    
    # Image message
    messages.append(Message(
        role="user",
        content=[
            {"type": "image_url", "image_url": {"url": image_url}}
        ]
    ))
    
    # Call LLM
    try:
        response = await asyncio.to_thread(
            llm.complete,
            messages,
            provider=provider,
            model=model
        )
        
        tokens = {
            "input": response.usage.get("prompt_tokens", 0) if response.usage else 0,
            "output": response.usage.get("completion_tokens", 0) if response.usage else 0,
        }
        
        return response.content.strip(), tokens
    except Exception as e:
        logger.error(f"Failed to process page {page_num}: {e}")
        return f"[Page {page_num} extraction failed]", {"input": 0, "output": 0}


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

async def extract_pdf_async(
    *,
    arxiv_id: Optional[str] = None,
    url: Optional[str] = None,
    local_path: Optional[str] = None,
    cache_dir: Union[str, Path] = "ideation/papers",
    llm: Optional[UnifiedLLM] = None,
    provider: LLMProvider,
    model: str,
    maintain_format: bool = True,
    concurrency: int = 5,
    select_pages: Optional[Union[int, List[int]]] = None,
    dpi: int = 200,
    window_size: int = 1,
) -> PdfDocument:
    """Extract text from PDF using vision models.

    Args:
        arxiv_id: ArXiv paper ID to download and process
        url: URL of PDF to download and process  
        local_path: Path to local PDF file
        cache_dir: Directory for caching downloaded PDFs
        llm: UnifiedLLM instance (created if not provided)
        provider: LLM provider to use (auto-detected if None)
        model: Specific model to use (auto-selected if None)
        maintain_format: Pass previous page context for consistent formatting
        concurrency: Max pages to process in parallel
        select_pages: Specific pages to extract (1-indexed), None for all
        dpi: Resolution for PDF to image conversion
        window_size: Size of window for batching pages

    Returns:
        PdfDocument with extracted markdown content
    """
    if sum(bool(x) for x in (arxiv_id, url, local_path)) != 1:
        raise ValueError("Provide exactly one of arxiv_id, url, or local_path")
    
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine file path
    if local_path:
        file_path = Path(local_path).resolve()
        title = file_path.stem
    elif arxiv_id:
        # Use arxiv helper to download
        pdf_name = f"{arxiv_id.replace('/', '_')}.pdf"
        file_path = cache_dir / pdf_name
        if not file_path.exists():
            arxiv_api.download_pdf(arxiv_id, str(cache_dir), pdf_name)
        title = arxiv_id
    else:  # url
        file_name = os.path.basename(url).split("?")[0] or "paper.pdf"
        file_path = cache_dir / file_name
        if not file_path.exists():
            logger.info(f"Downloading PDF from {url}")
            urllib.request.urlretrieve(url, file_path)
        title = file_path.stem
    
    # Convert PDF to images
    logger.info(f"Converting PDF to images at {dpi} DPI...")
    images = await asyncio.to_thread(
        convert_from_path,
        str(file_path),
        dpi=dpi,
        thread_count=os.cpu_count() or 4,
    )
    
    # Handle page selection
    if select_pages is not None:
        if isinstance(select_pages, int):
            select_pages = [select_pages]
        # Convert to 0-indexed
        selected_images = []
        for page_num in select_pages:
            if 1 <= page_num <= len(images):
                selected_images.append(images[page_num - 1])
            else:
                logger.warning(f"Page {page_num} out of range (1-{len(images)})")
        images = selected_images
    
    # Setup LLM
    if llm is None:
        llm = UnifiedLLM()
        # Try to add providers
        for prov in [LLMProvider.OPENAI, LLMProvider.ANTHROPIC, LLMProvider.OPENROUTER]:
            try:
                llm.add_provider(prov, default=(prov == LLMProvider.OPENAI))
            except ValueError:
                pass
    
    logger.info(f"Using {provider.value}/{model} for extraction")
    
    # Create windows
    if window_size < 1:
        window_size = 1
    windows = [images[i:i+window_size] for i in range(0, len(images), window_size)]

    page_markdowns: List[str] = [""] * len(images)
    total_tokens = {"input": 0, "output": 0}

    async def process_window(window_idx: int, window_imgs: List[Image.Image]):
        wnd_pages = list(range(window_idx * window_size + 1, window_idx * window_size + len(window_imgs) + 1))
        # Build combined image list
        content_blocks = []
        for pnum, img in zip(wnd_pages, window_imgs):
            content_blocks.append({"type": "text", "text": f"Page {pnum}"})
            content_blocks.append({"type": "image_url", "image_url": {"url": encode_pil_image(img)}})
        messages = [
            Message(role="system", content="Convert each page image to markdown in the same order, separated by `---`.") ,
            Message(role="user", content=content_blocks),
        ]
        resp = await asyncio.to_thread(
            llm.vision_complete,
            messages,
            provider=provider,
            model=model
        )
        parts = [s.strip() for s in resp.content.split("---") if s.strip()]
        for rel_idx, md in enumerate(parts):
            absolute_idx = window_idx * window_size + rel_idx
            if absolute_idx < len(page_markdowns):
                page_markdowns[absolute_idx] = md
        if resp.usage:
            total_tokens["input"] += resp.usage.get("prompt_tokens", 0)
            total_tokens["output"] += resp.usage.get("completion_tokens", 0)

    semaphore = asyncio.Semaphore(concurrency)

    async def sem_task(i, imgs):
        async with semaphore:
            await process_window(i, imgs)

    await asyncio.gather(*(sem_task(i, w) for i, w in enumerate(windows)))

    # Combine results
    full_text = "\n\n---\n\n".join(page_markdowns)
    
    return PdfDocument(
        text=full_text,
        pages=page_markdowns,
        title=title,
        arxiv_id=arxiv_id,
        local_path=str(file_path),
        page_count=len([md for md in page_markdowns if md]),
        model_used=f"{provider.value}/{model}",
        total_tokens=total_tokens,
    )


def extract_pdf(**kwargs) -> PdfDocument:
    """Synchronous wrapper for extract_pdf_async."""
    if "provider" not in kwargs or "model" not in kwargs:
        raise ValueError("'provider' and 'model' must be supplied explicitly (no auto-detect).")
    return asyncio.run(extract_pdf_async(**kwargs))
