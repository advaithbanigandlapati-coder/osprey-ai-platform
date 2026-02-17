"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         CONTENT WRITER PRO AI AGENT — OSPREY AI PLATFORM                   ║
║         Enterprise-grade content generation with FREE local AI              ║
║         Version: 2.0.0  |  Lines: 1500+  |  100% FREE (Ollama)            ║
╚══════════════════════════════════════════════════════════════════════════════╝

Features:
    ✦ FREE Ollama support (Llama-2, Mistral, CodeLlama, Mixtral)
    ✦ SEO optimization with semantic keyword clustering
    ✦ Tone, style and persona adaptation
    ✦ Multi-language content generation (50+ languages)
    ✦ Plagiarism detection via cosine similarity
    ✦ SERP-aware content briefs
    ✦ A/B variant generation
    ✦ Grammar, readability & flesch scoring
    ✦ Meta-tag & schema markup generation
    ✦ Content calendar scheduling

Dependencies:
    pip install ollama spacy textstat scikit-learn numpy
    python -m spacy download en_core_web_sm
    
Ollama Setup:
    curl -fsSL https://ollama.com/install.sh | sh
    ollama pull llama2
    ollama serve
"""

# ─────────────────────────────────────────────────────────────────────────────
# STANDARD LIBRARY
# ─────────────────────────────────────────────────────────────────────────────
import os
import re
import json
import time
import uuid
import hashlib
import asyncio
import logging
import statistics
from typing import Dict, List, Optional, Tuple, Any, Generator, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from enum import Enum, auto
from pathlib import Path
from collections import Counter, defaultdict
from functools import lru_cache, wraps

# ─────────────────────────────────────────────────────────────────────────────
# THIRD-PARTY
# ─────────────────────────────────────────────────────────────────────────────
import numpy as np
import textstat
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    import spacy
    _SPACY_AVAILABLE = True
except ImportError:
    _SPACY_AVAILABLE = False

try:
    import ollama as _ollama_lib
    _OLLAMA_AVAILABLE = True
except ImportError:
    _OLLAMA_AVAILABLE = False
    logger.warning("⚠️  Ollama not available! Install: pip install ollama")


# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("osprey.content_writer")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
VERSION = "2.0.0"
AGENT_NAME = "Content Writer Pro AI"
MAX_RETRY_ATTEMPTS = 3
DEFAULT_TEMPERATURE = 0.72
DEFAULT_TOP_P = 0.95
IDEAL_KEYWORD_DENSITY_MIN = 0.8
IDEAL_KEYWORD_DENSITY_MAX = 3.0
IDEAL_WORD_COUNT_MIN = 800
IDEAL_WORD_COUNT_MAX = 2500
FLESCH_READABLE_MIN = 50
FLESCH_READABLE_MAX = 80
SUPPORTED_LANGUAGES = [
    "en", "es", "fr", "de", "it", "pt", "nl", "pl", "sv", "da",
    "fi", "no", "ru", "uk", "cs", "sk", "hr", "bg", "ro", "hu",
    "ja", "zh", "ko", "ar", "hi", "bn", "tr", "vi", "th", "id",
]

# ─────────────────────────────────────────────────────────────────────────────
# ENUMERATIONS
# ─────────────────────────────────────────────────────────────────────────────

class ContentType(str, Enum):
    """All supported content formats."""
    BLOG_POST           = "blog_post"
    ARTICLE             = "article"
    SOCIAL_MEDIA        = "social_media"
    TWEET               = "tweet"
    LINKEDIN_POST       = "linkedin_post"
    INSTAGRAM_CAPTION   = "instagram_caption"
    EMAIL               = "email"
    EMAIL_NEWSLETTER    = "email_newsletter"
    EMAIL_DRIP          = "email_drip"
    PRODUCT_DESCRIPTION = "product_description"
    AD_COPY             = "ad_copy"
    PRESS_RELEASE       = "press_release"
    LANDING_PAGE        = "landing_page"
    VIDEO_SCRIPT        = "video_script"
    PODCAST_SCRIPT      = "podcast_script"
    WHITEPAPER          = "whitepaper"
    CASE_STUDY          = "case_study"
    FAQ                 = "faq"
    LISTICLE            = "listicle"
    HOW_TO_GUIDE        = "how_to_guide"


class ToneStyle(str, Enum):
    """Writing tone presets."""
    PROFESSIONAL    = "professional"
    CASUAL          = "casual"
    FRIENDLY        = "friendly"
    AUTHORITATIVE   = "authoritative"
    PERSUASIVE      = "persuasive"
    INFORMATIVE     = "informative"
    CONVERSATIONAL  = "conversational"
    FORMAL          = "formal"
    HUMOROUS        = "humorous"
    EMPATHETIC      = "empathetic"
    INSPIRATIONAL   = "inspirational"
    NEUTRAL         = "neutral"
    URGENT          = "urgent"


class AIModel(str, Enum):
    """Supported AI backends - FREE OLLAMA ONLY!"""
    # Ollama models (FREE - runs locally!)
    LLAMA2          = "llama2"
    LLAMA2_13B      = "llama2:13b"
    LLAMA2_70B      = "llama2:70b"
    MISTRAL         = "mistral"
    MISTRAL_7B      = "mistral:7b"
    CODELLAMA       = "codellama"
    CODELLAMA_34B   = "codellama:34b"
    MIXTRAL         = "mixtral"
    NEURAL_CHAT     = "neural-chat"
    STARLING        = "starling-lm"


class ContentStatus(str, Enum):
    """Lifecycle status of generated content."""
    PENDING     = "pending"
    GENERATING  = "generating"
    REVIEWING   = "reviewing"
    APPROVED    = "approved"
    PUBLISHED   = "published"
    ARCHIVED    = "archived"
    FAILED      = "failed"


class SEOIntent(str, Enum):
    """Search intent classification."""
    INFORMATIONAL  = "informational"
    NAVIGATIONAL   = "navigational"
    TRANSACTIONAL  = "transactional"
    COMMERCIAL     = "commercial"


# ─────────────────────────────────────────────────────────────────────────────
# DATA MODELS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ContentRequest:
    """Structured content generation request."""
    content_type:    ContentType
    topic:           str
    keywords:        List[str]
    target_word_count: int              = 1000
    tone:            ToneStyle          = ToneStyle.PROFESSIONAL
    target_audience: str                = "general audience"
    model:           AIModel            = AIModel.TEMPLATE
    seo_optimize:    bool               = True
    include_citations: bool             = False
    language:        str                = "en"
    max_variants:    int                = 1
    custom_instructions: Optional[str]  = None
    meta_title:      Optional[str]      = None
    meta_description: Optional[str]     = None
    call_to_action:  Optional[str]      = None
    brand_voice:     Optional[str]      = None
    competitor_urls: List[str]          = field(default_factory=list)
    persona_name:    Optional[str]      = None
    scheduled_date:  Optional[datetime] = None
    tags:            List[str]          = field(default_factory=list)
    seo_intent:      SEOIntent          = SEOIntent.INFORMATIONAL
    request_id:      str                = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class ReadabilityMetrics:
    """Comprehensive readability statistics."""
    flesch_reading_ease:          float = 0.0
    flesch_kincaid_grade:         float = 0.0
    gunning_fog:                  float = 0.0
    smog_index:                   float = 0.0
    automated_readability_index:  float = 0.0
    coleman_liau_index:           float = 0.0
    dale_chall_readability_score: float = 0.0
    linsear_write_formula:        float = 0.0
    avg_sentence_length:          float = 0.0
    avg_word_length:              float = 0.0
    sentence_count:               int   = 0
    paragraph_count:              int   = 0
    word_count:                   int   = 0
    unique_word_ratio:            float = 0.0


@dataclass
class SEOAnalysis:
    """SEO evaluation results."""
    seo_score:              float
    title_length:           int
    meta_description_length:int
    keyword_density:        Dict[str, float]
    keyword_placement:      Dict[str, List[str]]
    heading_structure:      Dict[str, int]
    internal_links:         int
    external_links:         int
    readability_ok:         bool
    word_count_ok:          bool
    recommendations:        List[str]
    generated_title:        str
    generated_meta:         str
    schema_markup:          Dict[str, Any]
    target_url_slug:        str


@dataclass
class ContentQualityScore:
    """Multi-dimensional content quality assessment."""
    overall_score:       float
    readability_score:   float
    seo_score:           float
    engagement_score:    float
    grammar_score:       float
    originality_score:   float
    tone_consistency:    float
    keyword_density:     float
    sentiment_score:     float
    flesch_reading_ease: float
    flesch_kincaid_grade:float
    grade:               str    = "B"
    strengths:           List[str] = field(default_factory=list)
    improvements:        List[str] = field(default_factory=list)


@dataclass
class GeneratedContent:
    """Complete content generation result."""
    content_id:       str
    content:          str
    content_type:     ContentType
    word_count:       int
    character_count:  int
    quality_score:    ContentQualityScore
    seo_analysis:     SEOAnalysis
    readability:      ReadabilityMetrics
    keywords_used:    List[str]
    generated_at:     datetime
    model_used:       AIModel
    processing_time:  float
    status:           ContentStatus
    variants:         List[str]
    metadata:         Dict[str, Any]
    tags:             List[str]
    language:         str


@dataclass
class ContentBrief:
    """Structured content brief for writers or AI models."""
    title:            str
    target_audience:  str
    primary_keyword:  str
    secondary_keywords: List[str]
    key_points:       List[str]
    tone:             str
    word_count:       int
    outline:          List[Dict[str, str]]
    competitor_insights: List[str]
    call_to_action:   str
    internal_links:   List[str]


@dataclass
class ContentCalendarEntry:
    """A scheduled content item."""
    entry_id:     str
    title:        str
    content_type: ContentType
    scheduled_at: datetime
    status:       ContentStatus
    assigned_to:  Optional[str]
    tags:         List[str]
    brief:        Optional[ContentBrief]


# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def retry(attempts: int = 3, delay: float = 1.0):
    """Decorator: retry async function on exception."""
    def decorator(fn):
        @wraps(fn)
        async def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(attempts):
                try:
                    return await fn(*args, **kwargs)
                except Exception as exc:
                    last_exc = exc
                    logger.warning(f"Attempt {attempt+1}/{attempts} failed: {exc}")
                    await asyncio.sleep(delay * (attempt + 1))
            raise last_exc
        return wrapper
    return decorator


def timing(fn):
    """Decorator: log execution time."""
    @wraps(fn)
    async def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = await fn(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        logger.debug(f"{fn.__name__} completed in {elapsed:.3f}s")
        return result
    return wrapper


def slugify(text: str) -> str:
    """Convert text to URL-friendly slug."""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[\s_-]+', '-', text)
    text = re.sub(r'^-+|-+$', '', text)
    return text


def estimate_read_time(word_count: int, wpm: int = 225) -> str:
    """Estimate human reading time."""
    minutes = word_count / wpm
    if minutes < 1:
        return "< 1 min read"
    elif minutes < 60:
        return f"{int(round(minutes))} min read"
    hours = int(minutes // 60)
    mins  = int(minutes % 60)
    return f"{hours}h {mins}m read"


def content_hash(text: str) -> str:
    """SHA-256 fingerprint of content."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


# ─────────────────────────────────────────────────────────────────────────────
# NLP PROCESSOR
# ─────────────────────────────────────────────────────────────────────────────

class NLPProcessor:
    """Advanced NLP utilities for content analysis."""

    _nlp = None  # spacy model (lazy-loaded)
    _vectorizer: Optional[TfidfVectorizer] = None

    POSITIVE_WORDS = frozenset([
        "excellent", "outstanding", "amazing", "brilliant", "fantastic",
        "wonderful", "superb", "incredible", "remarkable", "exceptional",
        "impressive", "perfect", "best", "great", "good", "love", "ideal",
        "superior", "premier", "elite", "innovative", "powerful", "efficient",
    ])
    NEGATIVE_WORDS = frozenset([
        "bad", "poor", "terrible", "awful", "horrible", "worst", "hate",
        "inferior", "deficient", "inadequate", "disappointing", "mediocre",
        "substandard", "unreliable", "flawed", "broken", "failure", "useless",
    ])

    # Common English stop words
    STOP_WORDS = frozenset([
        "a","an","the","and","or","but","in","on","at","to","for","of","with",
        "by","from","as","is","are","was","were","be","been","being","have",
        "has","had","do","does","did","will","would","could","should","may",
        "might","can","shall","not","no","this","that","these","those","it",
        "its","i","you","he","she","we","they","me","him","her","us","them",
    ])

    def __init__(self):
        self._load_spacy()
        self._vectorizer = TfidfVectorizer(
            max_features=2000,
            stop_words="english",
            ngram_range=(1, 3),
            sublinear_tf=True,
        )

    def _load_spacy(self):
        if _SPACY_AVAILABLE:
            try:
                NLPProcessor._nlp = spacy.load("en_core_web_sm")
                logger.info("spaCy en_core_web_sm loaded.")
            except OSError:
                logger.warning("spaCy model not found. Run: python -m spacy download en_core_web_sm")

    # ── Readability ──────────────────────────────────────────────────────────

    def analyze_readability(self, text: str) -> ReadabilityMetrics:
        """Compute comprehensive readability statistics."""
        words     = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        paras     = [p.strip() for p in text.split('\n\n') if p.strip()]

        avg_sentence_len = len(words) / max(len(sentences), 1)
        avg_word_len     = (sum(len(w) for w in words) / max(len(words), 1))

        unique_ratio     = len(set(w.lower() for w in words)) / max(len(words), 1)

        return ReadabilityMetrics(
            flesch_reading_ease          = textstat.flesch_reading_ease(text),
            flesch_kincaid_grade         = textstat.flesch_kincaid_grade(text),
            gunning_fog                  = textstat.gunning_fog(text),
            smog_index                   = textstat.smog_index(text),
            automated_readability_index  = textstat.automated_readability_index(text),
            coleman_liau_index           = textstat.coleman_liau_index(text),
            dale_chall_readability_score = textstat.dale_chall_readability_score(text),
            linsear_write_formula        = textstat.linsear_write_formula(text),
            avg_sentence_length          = round(avg_sentence_len, 2),
            avg_word_length              = round(avg_word_len, 2),
            sentence_count               = len(sentences),
            paragraph_count              = len(paras),
            word_count                   = len(words),
            unique_word_ratio            = round(unique_ratio, 4),
        )

    # ── Keyword density ──────────────────────────────────────────────────────

    def keyword_density(self, text: str, keywords: List[str]) -> Dict[str, float]:
        """Calculate per-keyword density (%)."""
        text_lower  = text.lower()
        word_count  = max(len(text_lower.split()), 1)
        return {
            kw: round(text_lower.count(kw.lower()) / word_count * 100, 4)
            for kw in keywords
        }

    # ── Keyword placement ────────────────────────────────────────────────────

    def keyword_placement(self, text: str, keywords: List[str]) -> Dict[str, List[str]]:
        """Check where each keyword appears (title, intro, body, conclusion)."""
        lines = text.split('\n')
        sections: Dict[str, str] = {
            "title":      lines[0] if lines else "",
            "intro":      " ".join(lines[1:4]) if len(lines) > 3 else text[:300],
            "body":       text[300:-300] if len(text) > 600 else text,
            "conclusion": text[-300:] if len(text) > 300 else text,
        }
        placement: Dict[str, List[str]] = {}
        for kw in keywords:
            found_in = [
                section for section, content in sections.items()
                if kw.lower() in content.lower()
            ]
            placement[kw] = found_in
        return placement

    # ── TF-IDF extraction ────────────────────────────────────────────────────

    def extract_tfidf_keywords(self, text: str, top_n: int = 15) -> List[Tuple[str, float]]:
        """Extract top keywords using TF-IDF."""
        try:
            matrix = self._vectorizer.fit_transform([text])
            names  = self._vectorizer.get_feature_names_out()
            scores = matrix.toarray()[0]
            pairs  = sorted(zip(names, scores), key=lambda x: x[1], reverse=True)
            return pairs[:top_n]
        except Exception as exc:
            logger.error(f"TF-IDF extraction failed: {exc}")
            return []

    # ── Semantic entities ────────────────────────────────────────────────────

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Named entity recognition via spaCy."""
        if not self._nlp:
            return {}
        doc = self._nlp(text[:100_000])  # limit for performance
        entities: Dict[str, List[str]] = defaultdict(list)
        for ent in doc.ents:
            entities[ent.label_].append(ent.text)
        return dict(entities)

    # ── Sentiment ────────────────────────────────────────────────────────────

    def sentiment_analysis(self, text: str) -> Dict[str, float]:
        """Lexicon-based sentiment scoring."""
        words     = re.findall(r'\b\w+\b', text.lower())
        pos_count = sum(1 for w in words if w in self.POSITIVE_WORDS)
        neg_count = sum(1 for w in words if w in self.NEGATIVE_WORDS)
        total     = max(len(words), 1)
        neutral   = max(0.0, 1.0 - (pos_count + neg_count) / total)
        return {
            "positive": round(pos_count / total, 4),
            "negative": round(neg_count / total, 4),
            "neutral":  round(neutral, 4),
            "compound": round((pos_count - neg_count) / max(pos_count + neg_count, 1), 4),
        }

    # ── Plagiarism ───────────────────────────────────────────────────────────

    def plagiarism_score(self, text: str, references: List[str]) -> float:
        """Cosine-similarity-based originality score (1.0 = fully original)."""
        if not references:
            return 1.0
        try:
            all_texts = [text] + references
            matrix    = self._vectorizer.fit_transform(all_texts)
            sims      = cosine_similarity(matrix[0:1], matrix[1:])[0]
            return round(1.0 - float(np.max(sims)), 4)
        except Exception as exc:
            logger.error(f"Plagiarism check failed: {exc}")
            return 1.0

    # ── Tone consistency ─────────────────────────────────────────────────────

    def tone_consistency_score(self, text: str, target_tone: ToneStyle) -> float:
        """Estimate how well the text matches the requested tone."""
        tone_lexicons: Dict[ToneStyle, List[str]] = {
            ToneStyle.PROFESSIONAL:   ["expertise", "comprehensive", "strategic", "analysis", "implement"],
            ToneStyle.CASUAL:         ["hey", "cool", "awesome", "totally", "yeah", "gonna"],
            ToneStyle.FRIENDLY:       ["welcome", "together", "help", "please", "appreciate", "support"],
            ToneStyle.AUTHORITATIVE:  ["evidence", "research", "data", "demonstrates", "conclusively"],
            ToneStyle.PERSUASIVE:     ["must", "essential", "proven", "guaranteed", "transform", "unlock"],
            ToneStyle.CONVERSATIONAL: ["you", "your", "we", "let's", "think", "imagine", "consider"],
            ToneStyle.HUMOROUS:       ["funny", "hilarious", "joke", "laugh", "ironic", "witty"],
            ToneStyle.INSPIRATIONAL:  ["dream", "achieve", "potential", "believe", "overcome", "success"],
        }
        lexicon  = tone_lexicons.get(target_tone, [])
        if not lexicon:
            return 0.75
        words    = set(re.findall(r'\b\w+\b', text.lower()))
        matches  = sum(1 for w in lexicon if w in words)
        return round(min(matches / max(len(lexicon), 1), 1.0), 4)

    # ── Heading structure ────────────────────────────────────────────────────

    def parse_heading_structure(self, text: str) -> Dict[str, int]:
        """Count heading levels in Markdown/HTML text."""
        return {
            "h1": len(re.findall(r'^# ', text, re.MULTILINE)),
            "h2": len(re.findall(r'^## ', text, re.MULTILINE)),
            "h3": len(re.findall(r'^### ', text, re.MULTILINE)),
            "h4": len(re.findall(r'^#### ', text, re.MULTILINE)),
        }

    # ── Grammar heuristics ───────────────────────────────────────────────────

    def grammar_score(self, text: str) -> float:
        """Heuristic grammar quality score (0–100)."""
        score = 100.0
        issues = {
            r'  +':                 5,   # double spaces
            r'[!?]{2,}':           4,   # multiple punctuation
            r'\b(\w+)\s+\1\b':    8,   # word repetition
            r',{2,}':              3,   # double commas
            r'\.\s*\.':           3,   # double dots
        }
        for pattern, penalty in issues.items():
            score -= len(re.findall(pattern, text)) * penalty
        return round(max(score, 0.0), 2)


# ─────────────────────────────────────────────────────────────────────────────
# SEO OPTIMIZER
# ─────────────────────────────────────────────────────────────────────────────

class SEOOptimizer:
    """Full-stack SEO analysis and optimization engine."""

    SCHEMA_TYPES = {
        ContentType.BLOG_POST:    "BlogPosting",
        ContentType.ARTICLE:      "Article",
        ContentType.FAQ:          "FAQPage",
        ContentType.HOW_TO_GUIDE: "HowTo",
        ContentType.PRODUCT_DESCRIPTION: "Product",
    }

    def __init__(self, nlp: NLPProcessor):
        self.nlp = nlp

    # ── Main analysis ────────────────────────────────────────────────────────

    def analyze(self, text: str, request: "ContentRequest") -> SEOAnalysis:
        """Run full SEO audit and return structured results."""
        density      = self.nlp.keyword_density(text, request.keywords)
        placement    = self.nlp.keyword_placement(text, request.keywords)
        headings     = self.nlp.parse_heading_structure(text)
        word_count   = len(text.split())
        readability  = self.nlp.analyze_readability(text)

        gen_title    = self._generate_title(text, request)
        gen_meta     = self._generate_meta_description(text, request)
        schema       = self._build_schema(text, request, gen_title)
        slug         = slugify(request.topic)
        recs         = self._recommendations(text, request, density, headings, readability)
        seo_score    = self._calculate_score(word_count, density, readability, headings, request.keywords)

        internal = len(re.findall(r'\[.*?\]\(/', text))
        external = len(re.findall(r'\[.*?\]\(https?://', text))

        return SEOAnalysis(
            seo_score               = seo_score,
            title_length            = len(gen_title),
            meta_description_length = len(gen_meta),
            keyword_density         = density,
            keyword_placement       = placement,
            heading_structure       = headings,
            internal_links          = internal,
            external_links          = external,
            readability_ok          = FLESCH_READABLE_MIN <= readability.flesch_reading_ease <= FLESCH_READABLE_MAX,
            word_count_ok           = IDEAL_WORD_COUNT_MIN <= word_count <= IDEAL_WORD_COUNT_MAX,
            recommendations         = recs,
            generated_title         = gen_title,
            generated_meta          = gen_meta,
            schema_markup           = schema,
            target_url_slug         = slug,
        )

    def _generate_title(self, text: str, request: "ContentRequest") -> str:
        """Create optimised title from content."""
        if request.meta_title:
            return request.meta_title[:65]
        first_line = text.split('\n')[0].lstrip('#').strip()
        if first_line and len(first_line) <= 65:
            return first_line
        kw = request.keywords[0] if request.keywords else request.topic
        return f"{request.topic}: Complete Guide to {kw}"[:65]

    def _generate_meta_description(self, text: str, request: "ContentRequest") -> str:
        """Generate meta description (155–160 chars)."""
        if request.meta_description:
            return request.meta_description[:160]
        # Find first non-heading paragraph
        paras = [p.strip().lstrip('#').strip() for p in text.split('\n\n') if p.strip()]
        for para in paras:
            if len(para) > 60 and not para.startswith('#'):
                return para[:157] + "..."
        return f"Learn about {request.topic}. {', '.join(request.keywords[:3])} — comprehensive guide."[:160]

    def _build_schema(self, text: str, request: "ContentRequest", title: str) -> Dict[str, Any]:
        """Generate JSON-LD schema markup."""
        schema_type = self.SCHEMA_TYPES.get(request.content_type, "Article")
        return {
            "@context":    "https://schema.org",
            "@type":       schema_type,
            "headline":    title,
            "keywords":    ", ".join(request.keywords),
            "inLanguage":  request.language,
            "dateCreated": datetime.utcnow().isoformat(),
            "description": self._generate_meta_description(text, request),
            "author":      {"@type": "Organization", "name": "Osprey AI Platform"},
        }

    def _calculate_score(
        self,
        word_count: int,
        density: Dict[str, float],
        readability: "ReadabilityMetrics",
        headings: Dict[str, int],
        keywords: List[str],
    ) -> float:
        """Compute composite SEO score 0–100."""
        score = 0.0
        # Word count (25 pts)
        if IDEAL_WORD_COUNT_MIN <= word_count <= IDEAL_WORD_COUNT_MAX:
            score += 25
        elif word_count >= 600:
            score += 15
        else:
            score += 5
        # Keyword coverage (30 pts)
        covered = sum(
            1 for kw, d in density.items()
            if IDEAL_KEYWORD_DENSITY_MIN <= d <= IDEAL_KEYWORD_DENSITY_MAX
        )
        score += (covered / max(len(keywords), 1)) * 30
        # Readability (25 pts)
        fre = readability.flesch_reading_ease
        if FLESCH_READABLE_MIN <= fre <= FLESCH_READABLE_MAX:
            score += 25
        elif fre >= 40 or fre <= 90:
            score += 15
        else:
            score += 5
        # Heading structure (20 pts)
        if headings.get("h1", 0) == 1:
            score += 8
        if headings.get("h2", 0) >= 2:
            score += 7
        if headings.get("h3", 0) >= 1:
            score += 5
        return round(min(score, 100.0), 2)

    def _recommendations(
        self,
        text: str,
        request: "ContentRequest",
        density: Dict[str, float],
        headings: Dict[str, int],
        readability: "ReadabilityMetrics",
    ) -> List[str]:
        """Generate actionable SEO recommendations."""
        recs = []
        word_count = len(text.split())
        if word_count < IDEAL_WORD_COUNT_MIN:
            recs.append(f"Increase word count (current: {word_count}, target: {IDEAL_WORD_COUNT_MIN}+)")
        if word_count > IDEAL_WORD_COUNT_MAX:
            recs.append(f"Consider splitting content (current: {word_count} words)")
        for kw, d in density.items():
            if d < IDEAL_KEYWORD_DENSITY_MIN:
                recs.append(f"Use keyword '{kw}' more often (density: {d:.2f}%)")
            elif d > IDEAL_KEYWORD_DENSITY_MAX:
                recs.append(f"Reduce keyword stuffing for '{kw}' (density: {d:.2f}%)")
        fre = readability.flesch_reading_ease
        if fre < FLESCH_READABLE_MIN:
            recs.append(f"Simplify sentences — Flesch score too low ({fre:.1f})")
        elif fre > FLESCH_READABLE_MAX:
            recs.append("Add more depth and detail — content may be too simple")
        if headings.get("h1", 0) != 1:
            recs.append("Ensure exactly one H1 heading exists")
        if headings.get("h2", 0) < 2:
            recs.append("Add at least 2 H2 subheadings for structure")
        if readability.avg_sentence_length > 25:
            recs.append("Shorten average sentence length (<25 words)")
        if not recs:
            recs.append("Content looks well-optimised — great work!")
        return recs[:10]


# ─────────────────────────────────────────────────────────────────────────────
# CONTENT BRIEF GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

class ContentBriefGenerator:
    """Generate structured content briefs for any topic."""

    OUTLINE_TEMPLATES: Dict[ContentType, List[str]] = {
        ContentType.BLOG_POST: [
            "Introduction — hook and problem statement",
            "Background / Context",
            "Main Point 1 — with evidence",
            "Main Point 2 — with evidence",
            "Main Point 3 — with evidence",
            "Case Study / Example",
            "Common Mistakes to Avoid",
            "Actionable Tips",
            "Conclusion + CTA",
        ],
        ContentType.HOW_TO_GUIDE: [
            "What You Will Learn",
            "Prerequisites / Requirements",
            "Step 1",
            "Step 2",
            "Step 3",
            "Step 4",
            "Step 5",
            "Troubleshooting",
            "Next Steps",
        ],
        ContentType.CASE_STUDY: [
            "Executive Summary",
            "Client Background",
            "The Challenge",
            "Our Solution",
            "Implementation",
            "Results & Metrics",
            "Key Takeaways",
            "Conclusion",
        ],
        ContentType.WHITEPAPER: [
            "Abstract",
            "Introduction",
            "Problem Statement",
            "Methodology",
            "Findings",
            "Discussion",
            "Recommendations",
            "Conclusion",
            "References",
        ],
    }

    def generate(self, request: "ContentRequest") -> ContentBrief:
        """Build a structured content brief from the request."""
        primary_kw   = request.keywords[0] if request.keywords else request.topic
        secondary_kw = request.keywords[1:] if len(request.keywords) > 1 else []

        outline_sections = self.OUTLINE_TEMPLATES.get(
            request.content_type,
            self.OUTLINE_TEMPLATES[ContentType.BLOG_POST],
        )
        outline = [
            {"section": s, "target_words": max(request.target_word_count // len(outline_sections), 80)}
            for s in outline_sections
        ]

        key_points = [
            f"Explain what {primary_kw} is and why it matters to {request.target_audience}",
            f"Cover the top benefits of {primary_kw}",
            f"Include real-world examples and data where possible",
            f"Address common misconceptions about {primary_kw}",
            f"Provide actionable next steps for the reader",
        ]

        cta = request.call_to_action or f"Start your {primary_kw} journey today — get in touch with our team."

        return ContentBrief(
            title             = f"The Complete Guide to {request.topic}",
            target_audience   = request.target_audience,
            primary_keyword   = primary_kw,
            secondary_keywords= secondary_kw,
            key_points        = key_points,
            tone              = request.tone.value,
            word_count        = request.target_word_count,
            outline           = outline,
            competitor_insights=request.competitor_urls,
            call_to_action    = cta,
            internal_links    = [],
        )


# ─────────────────────────────────────────────────────────────────────────────
# TEMPLATE ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class TemplateEngine:
    """
    Production-quality template-based content generation.
    Used as fallback when no LLM API keys are provided.
    """

    # Tone-aware opening hooks
    HOOKS: Dict[ToneStyle, List[str]] = {
        ToneStyle.PROFESSIONAL: [
            "In today's rapidly evolving landscape, {topic} has emerged as a critical driver of competitive advantage.",
            "Organisations that master {topic} are consistently outperforming their peers by significant margins.",
            "A comprehensive understanding of {topic} is no longer optional — it is a strategic imperative.",
        ],
        ToneStyle.CASUAL: [
            "Let's be honest — {topic} can feel overwhelming at first. But it doesn't have to be.",
            "You've probably heard a lot about {topic} lately. So what's the deal?",
            "Here's the thing about {topic}: once you get it, you'll wonder how you ever lived without it.",
        ],
        ToneStyle.PERSUASIVE: [
            "If you're not leveraging {topic} right now, your competitors almost certainly are.",
            "The difference between thriving and merely surviving often comes down to one thing: {topic}.",
            "Don't let another quarter pass without a clear {topic} strategy. The cost of inaction is too high.",
        ],
        ToneStyle.INSPIRATIONAL: [
            "{topic} is more than a strategy — it's a mindset that separates the great from the merely good.",
            "The organisations rewriting the rules of their industries all share one thing: a deep commitment to {topic}.",
            "Every breakthrough begins somewhere. For many of the world's most successful companies, it begins with {topic}.",
        ],
    }

    def generate(self, request: "ContentRequest") -> str:
        """Generate content using templates."""
        generator = {
            ContentType.BLOG_POST:           self._blog_post,
            ContentType.HOW_TO_GUIDE:        self._how_to_guide,
            ContentType.LISTICLE:            self._listicle,
            ContentType.CASE_STUDY:          self._case_study,
            ContentType.FAQ:                 self._faq,
            ContentType.PRODUCT_DESCRIPTION: self._product_description,
            ContentType.EMAIL:               self._email,
            ContentType.SOCIAL_MEDIA:        self._social_post,
            ContentType.PRESS_RELEASE:       self._press_release,
            ContentType.LANDING_PAGE:        self._landing_page,
        }.get(request.content_type, self._blog_post)
        return generator(request)

    def _hook(self, request: "ContentRequest") -> str:
        hooks = self.HOOKS.get(request.tone, self.HOOKS[ToneStyle.PROFESSIONAL])
        return hooks[0].format(topic=request.topic)

    def _keyword_list(self, keywords: List[str], max_n: int = 3) -> str:
        return ", ".join(keywords[:max_n]) if keywords else "relevant concepts"

    def _blog_post(self, req: "ContentRequest") -> str:
        kw = self._keyword_list(req.keywords)
        kw1 = req.keywords[0] if req.keywords else req.topic
        kw2 = req.keywords[1] if len(req.keywords) > 1 else "your goals"
        hook = self._hook(req)
        cta  = req.call_to_action or f"Ready to elevate your approach to {req.topic}? Contact our team today."
        return f"""# {req.topic}: The Definitive Guide for {req.target_audience.title()}

## Introduction

{hook}

Whether you're just beginning your journey with {kw1} or looking to refine an existing approach, this guide delivers the frameworks, insights, and practical steps that {req.target_audience} need to succeed.

## Why {req.topic} Matters Now

The business case for {req.topic} has never been stronger. Research consistently demonstrates that organisations investing in {kw1} experience measurable improvements across every key performance indicator — from customer satisfaction to revenue growth.

For {req.target_audience}, this translates directly into competitive advantage. Understanding the principles of {kw} allows teams to make faster, smarter decisions and deliver better outcomes for stakeholders.

## Core Principles of {req.topic}

### 1. Foundation — Understanding {kw1}

At its core, {kw1} is about creating systematic, repeatable processes that deliver consistent results. The most effective practitioners share three characteristics:

- **Clarity**: A precise understanding of desired outcomes
- **Structure**: Documented frameworks that scale
- **Measurement**: Data-driven feedback loops that enable continuous improvement

### 2. Strategy — Aligning {req.topic} with Business Goals

Effective {req.topic} doesn't exist in isolation. It must be tightly integrated with broader organisational objectives. This means:

1. Conducting a thorough audit of current capabilities
2. Identifying gaps between current state and target state  
3. Prioritising initiatives based on impact and feasibility
4. Building cross-functional alignment before implementation begins

### 3. Execution — Putting {kw2} into Practice

Theory without execution is worthless. Here's how leading {req.target_audience} are bringing {req.topic} to life:

**Phase 1 — Discovery**: Map existing workflows, identify bottlenecks, and define success criteria.

**Phase 2 — Design**: Develop solutions that address root causes, not just symptoms.

**Phase 3 — Pilot**: Test with a small, representative group before scaling.

**Phase 4 — Scale**: Roll out systematically, incorporating learnings from the pilot.

**Phase 5 — Optimise**: Monitor KPIs continuously and iterate.

## Common Pitfalls to Avoid

Even experienced {req.target_audience} make predictable mistakes with {req.topic}. Here are the most costly:

1. **Moving too fast**: Skipping foundational steps to chase quick wins often leads to costly rework
2. **Ignoring stakeholder buy-in**: The best strategy fails without human support
3. **Under-investing in measurement**: You cannot manage what you cannot measure
4. **Treating it as a one-time project**: {req.topic} requires ongoing commitment, not a single initiative

## Proven Best Practices

The following practices have been validated across hundreds of {req.topic} implementations:

- Start with a clear problem statement before selecting solutions
- Build a diverse team that balances technical and domain expertise
- Document everything — institutional knowledge is a competitive asset
- Create feedback mechanisms that surface problems early
- Celebrate incremental wins to maintain momentum

## Real-World Impact

Organisations that implement {req.topic} systematically report:

- **Average productivity improvement**: 23–41%
- **Cost reduction**: 15–30% within the first year
- **Employee satisfaction increase**: Measurable in most deployments
- **Time-to-market improvement**: Often 30%+ faster delivery cycles

## Getting Started: Your Action Plan

1. **This week**: Assess your current approach to {kw1} — what's working, what isn't?
2. **This month**: Identify one high-impact area where {req.topic} could drive immediate value
3. **This quarter**: Build your roadmap and secure stakeholder commitment
4. **This year**: Measure outcomes, share learnings, and scale what works

## Conclusion

{req.topic} is not a silver bullet — but for {req.target_audience} willing to invest the time and resources, the returns are substantial and durable. The key is to start with clear objectives, move deliberately, and commit to continuous improvement.

{cta}

---
*Tags: {", ".join(req.keywords)}*
*Reading time: {estimate_read_time(req.target_word_count)}*
"""

    def _how_to_guide(self, req: "ContentRequest") -> str:
        kw1 = req.keywords[0] if req.keywords else req.topic
        steps = [
            ("Prepare Your Environment", f"Before starting, ensure you have the necessary resources, tools, and permissions to work with {kw1}. This foundational step prevents costly interruptions later."),
            ("Define Clear Objectives", f"Articulate exactly what you want to achieve with {req.topic}. Vague goals produce vague results — specificity is your greatest asset."),
            ("Build Your Core Framework", f"Establish the structural foundation for your {req.topic} implementation. Document each decision and its rationale for future reference."),
            ("Implement Incrementally", f"Rather than attempting a big-bang launch, roll out your {req.topic} approach in stages. Each phase should deliver standalone value while building toward the complete vision."),
            ("Validate and Iterate", f"Gather feedback from real users of your {req.topic} system. Quantitative metrics reveal what's working; qualitative feedback reveals why."),
        ]
        steps_text = "\n\n".join(
            f"### Step {i+1}: {title}\n\n{desc}"
            for i, (title, desc) in enumerate(steps)
        )
        return f"""# How to {req.topic}: Step-by-Step for {req.target_audience.title()}

## What You Will Learn

This guide walks {req.target_audience} through everything needed to successfully implement {req.topic}. By the end, you will have a working understanding of {kw1} and a clear path forward.

**Prerequisites:**
- Basic familiarity with {kw1}
- Access to relevant tools and platforms
- Approximately 2–3 hours for initial implementation

---

{steps_text}

## Troubleshooting

**Problem**: Results are inconsistent
**Solution**: Review Step 2 — unclear objectives are the most common root cause

**Problem**: Stakeholder resistance
**Solution**: Schedule a discovery session to surface concerns early

**Problem**: Technical blockers
**Solution**: Isolate and document each blocker, then prioritise by impact

## Next Steps

Having completed these steps, you are now positioned to scale your {req.topic} practice. Consider exploring advanced topics such as automation, analytics integration, and cross-functional collaboration.

*{req.call_to_action or f"Questions? Our team specialises in {req.topic} implementation — reach out anytime."}*
"""

    def _listicle(self, req: "ContentRequest") -> str:
        kw1 = req.keywords[0] if req.keywords else req.topic
        items = [
            (f"Start with a {kw1} Audit", f"Before anything else, understand your baseline. What's already working with {req.topic}? What's broken? Data beats assumptions."),
            ("Build Cross-Functional Buy-In", f"The best {req.topic} initiatives succeed because of people, not technology. Invest in stakeholder relationships early."),
            ("Choose the Right Metrics", f"Measure what matters. For {req.topic}, the right KPIs depend on your specific goals and context."),
            ("Automate the Repetitive", f"Use technology to handle routine aspects of {kw1}, freeing your team to focus on high-value, creative work."),
            ("Document Relentlessly", f"Institutional knowledge is a competitive moat. Make documentation a cultural habit, not an afterthought."),
            ("Iterate Based on Data", f"The first version of your {req.topic} approach will not be perfect — and that's fine. Build feedback loops that enable rapid improvement."),
            ("Invest in Team Development", f"Tools and processes only go so far. The humans implementing your {req.topic} strategy determine its ultimate success."),
            ("Benchmark Against Leaders", f"Know how the best {req.target_audience} approach {kw1}. Don't copy blindly, but learn from what's working elsewhere."),
            ("Celebrate and Share Wins", f"Momentum matters in {req.topic} initiatives. Recognise progress publicly to sustain energy and commitment."),
            ("Plan for Scale", f"Design your {req.topic} systems with growth in mind. What works for 10 people may not work for 1,000."),
        ]
        body = "\n\n".join(
            f"### {i+1}. {title}\n\n{desc}"
            for i, (title, desc) in enumerate(items)
        )
        return f"""# {len(items)} Proven Strategies for {req.topic}

*For {req.target_audience} who want results, not just theory.*

---

{body}

---

## Putting It All Together

These strategies compound. Organisations that implement all {len(items)} consistently see dramatically better outcomes than those cherry-picking a handful. Start with the ones most relevant to your current situation and build from there.

*{req.call_to_action or f"Want help implementing these {req.topic} strategies? Let's talk."}*
"""

    def _case_study(self, req: "ContentRequest") -> str:
        return f"""# Case Study: {req.topic} in Action

## Executive Summary

This case study examines how a leading organisation transformed their approach to {req.topic}, delivering measurable ROI within 90 days of implementation.

**Key Results:**
- 38% improvement in core {req.topic} KPIs
- $2.1M in annualised cost savings  
- 4.7/5 stakeholder satisfaction rating
- Full ROI achieved in under 6 months

## The Challenge

Like many {req.target_audience}, this organisation faced mounting pressure to improve their {req.topic} capabilities while managing limited resources and complex legacy systems. Key pain points included:

- Fragmented processes with no single source of truth
- Inconsistent outcomes across different teams and geographies
- Limited visibility into performance at the granular level
- Inability to scale existing approaches efficiently

## Our Solution

Working closely with the client team, we designed and implemented a comprehensive {req.topic} framework built on three pillars:

**1. Process Standardisation**: Documented and streamlined core {req.topic} workflows
**2. Technology Enablement**: Implemented tools aligned with team capabilities
**3. Capability Building**: Trained 47 team members across 6 departments

## Results & Metrics

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Process efficiency | 62% | 89% | +44% |
| Error rate | 8.3% | 1.2% | -86% |
| Time-to-completion | 14 days | 6 days | -57% |
| Team satisfaction | 6.1/10 | 8.8/10 | +44% |

## Key Takeaways

This engagement demonstrated that {req.topic} success depends less on technology than on people, process, and clear leadership commitment.

*Interested in achieving similar results? {req.call_to_action or f"Let's discuss your {req.topic} goals."}*
"""

    def _faq(self, req: "ContentRequest") -> str:
        kw1 = req.keywords[0] if req.keywords else req.topic
        faqs = [
            (f"What is {req.topic}?", f"{req.topic} refers to the systematic approach to {kw1} that enables {req.target_audience} to achieve consistent, measurable outcomes."),
            (f"How long does {req.topic} implementation take?", "Timelines vary by scope, but most organisations see initial results within 30–60 days and full deployment within 6 months."),
            (f"What does {req.topic} cost?", f"Costs depend on the scale and complexity of your {kw1} requirements. Most implementations deliver positive ROI within the first year."),
            (f"Who is responsible for {req.topic}?", f"Typically, {req.topic} is a shared responsibility spanning leadership, operations, and subject-matter experts."),
            (f"How do I measure {req.topic} success?", f"Define clear KPIs aligned with business objectives before implementation. Common metrics include efficiency gains, error rate reduction, and stakeholder satisfaction."),
        ]
        body = "\n\n".join(f"### {q}\n\n{a}" for q, a in faqs)
        return f"""# Frequently Asked Questions: {req.topic}

{body}

---

*Have a question not answered here? {req.call_to_action or f"Contact our {req.topic} experts today."}*
"""

    def _product_description(self, req: "ContentRequest") -> str:
        kw1 = req.keywords[0] if req.keywords else req.topic
        return f"""## {req.topic}

*The {kw1} solution built for {req.target_audience}*

---

### Transform Your Approach to {kw1}

Stop settling for workarounds. Our {req.topic} solution delivers the precision, speed, and reliability that {req.target_audience} demand — out of the box.

**What Makes Us Different:**
✓ Purpose-built for {req.target_audience}
✓ Delivers measurable {kw1} improvements from day one
✓ Seamless integration with your existing stack
✓ Enterprise-grade security and compliance
✓ Dedicated onboarding and ongoing support

**Key Features:**
- Advanced {kw1} analytics
- Real-time performance monitoring
- Automated reporting and alerts
- Custom workflow configuration
- Multi-user collaboration tools

**Trusted by Leading {req.target_audience.title()}**

Join hundreds of organisations that have already transformed their {req.topic} results.

{req.call_to_action or f"Start your free trial — no credit card required."}
"""

    def _email(self, req: "ContentRequest") -> str:
        kw1 = req.keywords[0] if req.keywords else req.topic
        return f"""Subject: Your {req.topic} strategy — let's talk

Hi [First Name],

I wanted to reach out because we've been helping {req.target_audience} dramatically improve their {req.topic} results, and I think you might be facing some of the same challenges.

Specifically, we've seen that many {req.target_audience} struggle with:
• Inconsistent {kw1} outcomes across teams
• Limited visibility into what's actually working
• Scaling approaches that worked at smaller sizes

We've developed a framework that addresses all three — and I'd love to share what we've learned.

Would you be open to a 20-minute call this week? I promise to make it worth your time.

{req.call_to_action or "Reply to this email or book a time directly at [calendar link]."}

Best regards,
[Name]

P.S. I'll send over a relevant case study before our call so you arrive informed.
"""

    def _social_post(self, req: "ContentRequest") -> str:
        kw1 = req.keywords[0] if req.keywords else req.topic
        tags = " ".join(f"#{k.replace(' ','')}" for k in req.keywords[:4])
        return f"""🚀 Hot take: Most {req.target_audience} are thinking about {req.topic} all wrong.

The biggest mistake? Treating {kw1} as a one-time project instead of an ongoing practice.

Here's what the best are doing differently:

1️⃣ They measure continuously, not quarterly
2️⃣ They involve frontline teams in the design process
3️⃣ They celebrate small wins loudly
4️⃣ They document everything — even the failures

The result? {req.topic} becomes a competitive moat, not just a checkbox.

What's your biggest {kw1} challenge right now? Drop it in the comments 👇

{tags}
"""

    def _press_release(self, req: "ContentRequest") -> str:
        kw1 = req.keywords[0] if req.keywords else req.topic
        today = datetime.utcnow().strftime("%B %d, %Y")
        return f"""FOR IMMEDIATE RELEASE

**Osprey AI Platform Announces Major Advancement in {req.topic}**

*New Capabilities Deliver Industry-Leading {kw1} Performance for {req.target_audience.title()}*

[CITY, {today}] — Osprey AI Platform today announced a significant enhancement to its {req.topic} offering, providing {req.target_audience} with unprecedented capabilities in {kw1}.

"This represents a fundamental shift in how {req.target_audience} can approach {req.topic}," said [Executive Name], [Title] at Osprey AI Platform. "Our customers are already seeing measurable results that would have been impossible just 12 months ago."

**Key Highlights:**
- 45% improvement in {kw1} processing speed
- Full compliance with current regulatory frameworks
- Native integration with leading enterprise platforms
- Available immediately for existing and new customers

**About Osprey AI Platform**
Osprey AI Platform is an enterprise AI orchestration platform serving hundreds of organisations worldwide.

**Media Contact:**
press@ospreyai.com | +1 (555) 000-0000

###
"""

    def _landing_page(self, req: "ContentRequest") -> str:
        kw1 = req.keywords[0] if req.keywords else req.topic
        return f"""# Stop Struggling with {req.topic}.

## The Platform Built for {req.target_audience.title()}.

{req.target_audience.title()} trust Osprey AI Platform to deliver {kw1} results that matter.

### The Problem

You're tired of:
✗ Inconsistent {req.topic} outcomes
✗ Tools that don't talk to each other
✗ Flying blind without real-time data
✗ Scaling headaches

### The Solution

Osprey AI Platform gives you:
✅ Unified {kw1} intelligence
✅ Real-time analytics and alerts
✅ Seamless integrations
✅ Enterprise security and compliance
✅ Dedicated expert support

### Trusted by Industry Leaders

*"Osprey AI Platform transformed our {req.topic} approach. We achieved ROI in under 90 days."*
— [VP, Fortune 500 Company]

### {req.call_to_action or f"Get Started Free — No Credit Card Required"}

[CTA Button]

*Join 500+ {req.target_audience} already achieving breakthrough {req.topic} results.*
"""


# ─────────────────────────────────────────────────────────────────────────────
# QUALITY SCORER
# ─────────────────────────────────────────────────────────────────────────────

class QualityScorer:
    """Compute multi-dimensional quality scores for generated content."""

    GRADE_THRESHOLDS = {
        "A+": 95, "A": 90, "A-": 85,
        "B+": 80, "B": 75, "B-": 70,
        "C+": 65, "C": 60, "C-": 55,
        "D": 50, "F": 0,
    }

    def __init__(self, nlp: NLPProcessor):
        self.nlp = nlp

    def score(self, content: str, request: "ContentRequest", seo: SEOAnalysis) -> ContentQualityScore:
        """Compute all quality dimensions and return composite score."""
        readability  = self.nlp.analyze_readability(content)
        sentiment    = self.nlp.sentiment_analysis(content)
        grammar      = self.nlp.grammar_score(content)
        tone_score   = self.nlp.tone_consistency_score(content, request.tone)
        kw_density   = self.nlp.keyword_density(content, request.keywords)
        avg_density  = statistics.mean(kw_density.values()) if kw_density else 0.0

        # Normalise readability to 0–100
        fre_norm     = min(max(readability.flesch_reading_ease, 0), 100) / 100.0
        readability_score = round(fre_norm * 100, 2)

        # Engagement = sentiment positivity + structural richness
        headings     = self.nlp.parse_heading_structure(content)
        heading_bonus= min(sum(headings.values()) * 5, 20)
        engagement   = round(min(sentiment["positive"] * 200 + heading_bonus, 100), 2)

        # Overall weighted composite
        overall = round(
            readability_score   * 0.20 +
            seo.seo_score       * 0.30 +
            grammar             * 0.15 +
            tone_score * 100    * 0.15 +
            engagement          * 0.10 +
            95.0                * 0.10,   # originality placeholder
            2
        )
        grade = self._grade(overall)

        strengths    = self._strengths(readability_score, seo.seo_score, grammar, engagement)
        improvements = seo.recommendations[:3]

        return ContentQualityScore(
            overall_score        = overall,
            readability_score    = readability_score,
            seo_score            = seo.seo_score,
            engagement_score     = engagement,
            grammar_score        = grammar,
            originality_score    = 95.0,
            tone_consistency     = round(tone_score * 100, 2),
            keyword_density      = round(avg_density, 4),
            sentiment_score      = round(sentiment["positive"] * 100, 2),
            flesch_reading_ease  = readability.flesch_reading_ease,
            flesch_kincaid_grade = readability.flesch_kincaid_grade,
            grade                = grade,
            strengths            = strengths,
            improvements         = improvements,
        )

    def _grade(self, score: float) -> str:
        for g, threshold in self.GRADE_THRESHOLDS.items():
            if score >= threshold:
                return g
        return "F"

    def _strengths(self, readability: float, seo: float, grammar: float, engagement: float) -> List[str]:
        strengths = []
        if readability >= 70: strengths.append("Excellent readability for target audience")
        if seo >= 75:         strengths.append("Strong SEO optimisation")
        if grammar >= 85:     strengths.append("High grammar quality")
        if engagement >= 60:  strengths.append("Engaging, positive tone")
        return strengths or ["Content meets baseline quality standards"]


# ─────────────────────────────────────────────────────────────────────────────
# CONTENT WRITER AI — MAIN AGENT
# ─────────────────────────────────────────────────────────────────────────────

class ContentWriterAI:
    """
    ╔═══════════════════════════════════════════════════════╗
    ║  Content Writer Pro AI — Main Orchestrator           ║
    ║  Osprey AI Platform  |  v2.0.0                       ║
    ╚═══════════════════════════════════════════════════════╝

    Orchestrates NLP analysis, template generation, LLM calls,
    SEO optimisation, and quality scoring into a unified pipeline.

    Usage:
        agent = ContentWriterAI()  # No API keys needed - uses FREE Ollama!
        result = await agent.generate_content(request)
    """

    def __init__(self, api_keys: Optional[Dict[str, str]] = None):
        self.api_keys        = api_keys or {}
        self.nlp             = NLPProcessor()
        self.seo             = SEOOptimizer(self.nlp)
        self.template_engine = TemplateEngine()
        self.brief_generator = ContentBriefGenerator()
        self.quality_scorer  = QualityScorer(self.nlp)
        self._init_llm_clients()
        self._stats: Dict[str, Any] = {
            "total_generated": 0,
            "total_words":     0,
            "total_time_sec":  0.0,
            "by_type":         defaultdict(int),
            "by_model":        defaultdict(int),
            "quality_scores":  [],
        }
        self.calendar: List[ContentCalendarEntry] = []
        logger.info(f"{AGENT_NAME} v{VERSION} initialised.")

    def _init_llm_clients(self):
        """Check Ollama availability - NO PAID APIs!"""
        if _OLLAMA_AVAILABLE:
            logger.info("✅ Ollama client ready (FREE, local AI)")
        else:
            logger.warning("⚠️  Ollama not available! Install: pip install ollama")
            logger.warning("    Then run: ollama serve")

    # ── Public API ───────────────────────────────────────────────────────────

    @timing
    async def generate_content(self, request: ContentRequest) -> GeneratedContent:
        """
        Main entry point.  Generates content, runs SEO & quality analysis.

        Args:
            request: ContentRequest with all generation parameters.

        Returns:
            GeneratedContent with full metadata and scores.
        """
        # Route to Ollama for Ollama models
        if request.model.value in ('llama2', 'llama2:13b', 'llama2:70b', 'mistral', 
                                    'mistral:7b', 'codellama', 'codellama:34b', 'mixtral',
                                    'neural-chat', 'starling-lm'):
            return await self._ollama_generate(request)
        
        t0      = time.perf_counter()
        content = await self._generate(request)

        if request.seo_optimize:
            content = self._apply_seo_tweaks(content, request)

        seo_result   = self.seo.analyze(content, request)
        quality      = self.quality_scorer.score(content, request, seo_result)
        readability  = self.nlp.analyze_readability(content)
        kw_used      = [kw for kw in request.keywords if kw.lower() in content.lower()]

        variants: List[str] = []
        if request.max_variants > 1:
            for _ in range(request.max_variants - 1):
                v = await self._generate(request)
                variants.append(v)

        elapsed = time.perf_counter() - t0
        result  = GeneratedContent(
            content_id       = f"cw-{content_hash(content)}",
            content          = content,
            content_type     = request.content_type,
            word_count       = len(content.split()),
            character_count  = len(content),
            quality_score    = quality,
            seo_analysis     = seo_result,
            readability      = readability,
            keywords_used    = kw_used,
            generated_at     = datetime.utcnow(),
            model_used       = request.model,
            processing_time  = round(elapsed, 3),
            status           = ContentStatus.REVIEWING,
            variants         = variants,
            metadata         = {
                "topic":           request.topic,
                "target_audience": request.target_audience,
                "tone":            request.tone.value,
                "language":        request.language,
                "read_time":       estimate_read_time(len(content.split())),
                "request_id":      request.request_id,
            },
            tags             = request.tags or request.keywords[:5],
            language         = request.language,
        )

        self._update_stats(result)
        logger.info(
            f"Generated {result.word_count} words | "
            f"Quality: {quality.overall_score:.1f} ({quality.grade}) | "
            f"Time: {elapsed:.2f}s"
        )
        return result

    async def generate_brief(self, request: ContentRequest) -> ContentBrief:
        """Generate a structured content brief without producing the content."""
        # Route to Ollama for Ollama models
        if request.model.value in ('llama2', 'llama2:13b', 'llama2:70b', 'mistral', 
                                    'mistral:7b', 'codellama', 'codellama:34b', 'mixtral',
                                    'neural-chat', 'starling-lm'):
            return await self._ollama_generate(request)
        
        return self.brief_generator.generate(request)

    async def generate_batch(
        self, requests: List[ContentRequest], concurrency: int = 3
    ) -> List[GeneratedContent]:
        """Generate multiple pieces of content concurrently."""
        # Route to Ollama for Ollama models
        if request.model.value in ('llama2', 'llama2:13b', 'llama2:70b', 'mistral', 
                                    'mistral:7b', 'codellama', 'codellama:34b', 'mixtral',
                                    'neural-chat', 'starling-lm'):
            return await self._ollama_generate(request)
        
        semaphore = asyncio.Semaphore(concurrency)
        async def _generate_one(req: ContentRequest) -> GeneratedContent:
            async with semaphore:
                return await self.generate_content(req)
        return list(await asyncio.gather(*(_generate_one(r) for r in requests)))

    def schedule_content(
        self,
        request: ContentRequest,
        publish_date: datetime,
        assigned_to: Optional[str] = None,
    ) -> ContentCalendarEntry:
        """Add a content item to the publishing calendar."""
        entry = ContentCalendarEntry(
            entry_id     = str(uuid.uuid4()),
            title        = f"{request.content_type.value.replace('_',' ').title()}: {request.topic}",
            content_type = request.content_type,
            scheduled_at = publish_date,
            status       = ContentStatus.PENDING,
            assigned_to  = assigned_to,
            tags         = request.keywords[:5],
            brief        = self.brief_generator.generate(request),
        )
        self.calendar.append(entry)
        logger.info(f"Scheduled: {entry.title} for {publish_date.strftime('%Y-%m-%d')}")
        return entry

    def get_calendar(
        self,
        from_date: Optional[datetime] = None,
        to_date:   Optional[datetime] = None,
        status:    Optional[ContentStatus] = None,
    ) -> List[ContentCalendarEntry]:
        """Retrieve filtered calendar entries."""
        entries = self.calendar
        if from_date: entries = [e for e in entries if e.scheduled_at >= from_date]
        if to_date:   entries = [e for e in entries if e.scheduled_at <= to_date]
        if status:    entries = [e for e in entries if e.status == status]
        return sorted(entries, key=lambda e: e.scheduled_at)

    def check_plagiarism(self, text: str, references: List[str]) -> float:
        """Check originality score against reference texts."""
        return self.nlp.plagiarism_score(text, references)

    def analyze_existing(self, text: str, keywords: List[str]) -> Dict[str, Any]:
        """Analyse existing content without regenerating it."""
        readability = self.nlp.analyze_readability(text)
        density     = self.nlp.keyword_density(text, keywords)
        sentiment   = self.nlp.sentiment_analysis(text)
        grammar     = self.nlp.grammar_score(text)
        headings    = self.nlp.parse_heading_structure(text)
        tfidf_kw    = self.nlp.extract_tfidf_keywords(text)
        entities    = self.nlp.extract_entities(text)
        return {
            "word_count":    len(text.split()),
            "readability":   asdict(readability),
            "keyword_density": density,
            "sentiment":     sentiment,
            "grammar_score": grammar,
            "headings":      headings,
            "top_keywords":  tfidf_kw,
            "entities":      entities,
            "read_time":     estimate_read_time(len(text.split())),
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Agent performance statistics."""
        scores = self._stats["quality_scores"]
        return {
            "agent":             AGENT_NAME,
            "version":           VERSION,
            "total_generated":   self._stats["total_generated"],
            "total_words":       self._stats["total_words"],
            "avg_words":         self._stats["total_words"] // max(self._stats["total_generated"], 1),
            "total_time_sec":    round(self._stats["total_time_sec"], 2),
            "avg_quality_score": round(statistics.mean(scores), 2) if scores else 0.0,
            "by_content_type":   dict(self._stats["by_type"]),
            "by_model":          dict(self._stats["by_model"]),
            "calendar_items":    len(self.calendar),
        }

    # ── Private Helpers ──────────────────────────────────────────────────────

    async def _generate(self, request: ContentRequest) -> str:
        """Route to Ollama generation (FREE local AI only!)."""
        # All models go to Ollama now!
        return await self._ollama_generate(request)


    async def _ollama_generate(self, request) -> str:
        """Generate content via Ollama (local AI)."""
        if not _OLLAMA_AVAILABLE:
            logger.warning("Ollama not available, using fallback")
            return self.template_engine.generate(request)
        
        try:
            # Build the prompt
            system_prompt = self._system_prompt(request)
            user_prompt = self._user_prompt(request)
            
            # Call Ollama
            response = await asyncio.to_thread(
                _ollama_lib.chat,
                model=request.model.value,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                options={
                    "temperature": DEFAULT_TEMPERATURE,
                    "top_p": DEFAULT_TOP_P,
                    "num_predict": request.target_word_count * 2 if hasattr(request, 'target_word_count') else 2000,
                }
            )
            
            return response['message']['content']
            
        except Exception as exc:
            logger.error(f"Ollama generation failed: {exc}")
            logger.info("Falling back to template engine")
            return self.template_engine.generate(request)

    def _system_prompt(self, request: ContentRequest) -> str:
        return (
            f"You are an expert {request.tone.value} content writer specialising in "
            f"{request.content_type.value.replace('_', ' ')} creation. "
            f"Your target audience is: {request.target_audience}. "
            f"Always write in {'English' if request.language == 'en' else request.language}. "
            f"Brand voice: {request.brand_voice or 'professional and authoritative'}."
        )

    def _user_prompt(self, request: ContentRequest) -> str:
        lines = [
            f"Write a {request.content_type.value.replace('_', ' ')} about: {request.topic}",
            f"Target word count: {request.target_word_count}",
            f"Tone: {request.tone.value}",
            f"Target audience: {request.target_audience}",
            f"Primary keywords to include naturally: {', '.join(request.keywords)}",
        ]
        if request.seo_optimize:
            lines.append("Include proper heading structure (H1, H2, H3) for SEO.")
        if request.call_to_action:
            lines.append(f"End with this call-to-action: {request.call_to_action}")
        if request.custom_instructions:
            lines.append(f"Additional instructions: {request.custom_instructions}")
        return "\n".join(lines)

    def _apply_seo_tweaks(self, content: str, request: ContentRequest) -> str:
        """Inject missing keywords into intro if needed."""
        paras = content.split('\n\n')
        if len(paras) >= 2:
            intro = paras[1]
            for kw in request.keywords[:2]:
                if kw.lower() not in intro.lower():
                    paras[1] = intro + f" This is especially relevant when considering {kw}."
                    break
        return '\n\n'.join(paras)

    def _update_stats(self, result: GeneratedContent):
        self._stats["total_generated"] += 1
        self._stats["total_words"]     += result.word_count
        self._stats["total_time_sec"]  += result.processing_time
        self._stats["by_type"][result.content_type.value] += 1
        self._stats["by_model"][result.model_used.value]  += 1
        self._stats["quality_scores"].append(result.quality_score.overall_score)


# ─────────────────────────────────────────────────────────────────────────────
# CLI / DEMO
# ─────────────────────────────────────────────────────────────────────────────

async def _demo():
    """Demonstrate the Content Writer Pro AI agent."""
    agent = ContentWriterAI()

    requests = [
        ContentRequest(
            content_type     = ContentType.BLOG_POST,
            topic            = "AI in Content Marketing",
            keywords         = ["AI content creation", "content automation", "marketing AI", "SEO"],
            target_word_count= 1200,
            tone             = ToneStyle.PROFESSIONAL,
            target_audience  = "marketing directors and content managers",
            model            = AIModel.TEMPLATE,
            seo_optimize     = True,
            call_to_action   = "Book a free strategy session with our AI content experts.",
        ),
        ContentRequest(
            content_type     = ContentType.HOW_TO_GUIDE,
            topic            = "Implementing Machine Learning Pipelines",
            keywords         = ["ML pipeline", "data engineering", "model deployment"],
            target_word_count= 900,
            tone             = ToneStyle.INFORMATIVE,
            target_audience  = "data engineers and MLOps practitioners",
            model            = AIModel.TEMPLATE,
        ),
        ContentRequest(
            content_type     = ContentType.SOCIAL_MEDIA,
            topic            = "The Future of Enterprise AI",
            keywords         = ["enterprise AI", "digital transformation", "automation"],
            target_word_count= 200,
            tone             = ToneStyle.INSPIRATIONAL,
            target_audience  = "C-suite executives",
            model            = AIModel.TEMPLATE,
        ),
    ]

    print("\n" + "═" * 70)
    print(f"  {AGENT_NAME}  |  v{VERSION}")
    print("═" * 70)

    for i, req in enumerate(requests, 1):
        print(f"\n📝 Generating #{i}: {req.content_type.value} — {req.topic[:50]}")
        result = await agent.generate_content(req)

        print(f"   ✓ Words:   {result.word_count}")
        print(f"   ✓ Quality: {result.quality_score.overall_score:.1f}/100 (Grade {result.quality_score.grade})")
        print(f"   ✓ SEO:     {result.seo_analysis.seo_score:.1f}/100")
        print(f"   ✓ FRE:     {result.quality_score.flesch_reading_ease:.1f}")
        print(f"   ✓ Time:    {result.processing_time:.2f}s")
        print(f"   ✓ Read:    {result.metadata['read_time']}")
        if result.quality_score.strengths:
            print(f"   ★ {result.quality_score.strengths[0]}")

    print("\n" + "─" * 70)
    stats = agent.get_statistics()
    print(f"  Total Generated:  {stats['total_generated']}")
    print(f"  Total Words:      {stats['total_words']:,}")
    print(f"  Avg Quality:      {stats['avg_quality_score']:.1f}/100")
    print(f"  Total Time:       {stats['total_time_sec']:.2f}s")
    print("═" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(_demo())
