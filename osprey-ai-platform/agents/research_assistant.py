"""
EMAIL MARKETING AI AGENT
Enterprise-grade intelligent email marketing automation

Features:
- Campaign creation and management
- A/B testing and optimization
- Audience segmentation
- Personalization engine
- Send-time optimization
- Subject line optimization
- Content recommendation
- Deliverability analysis
- Performance analytics
- List hygiene and management
- Engagement prediction
- Automated workflows

Dependencies:
- pandas
- numpy
- sklearn
- textblob
- jinja2
"""

import os
import re
import json
import logging
from typing import Dict, List, Optional, Tuple, Any, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from enum import Enum
from collections import defaultdict, Counter
import hashlib
import random

# ML libraries
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CampaignType(Enum):
    """Email campaign types"""
    PROMOTIONAL = "promotional"
    TRANSACTIONAL = "transactional"
    NEWSLETTER = "newsletter"
    DRIP = "drip"
    WELCOME = "welcome"
    ABANDONED_CART = "abandoned_cart"
    RE_ENGAGEMENT = "re_engagement"
    PRODUCT_LAUNCH = "product_launch"
    EVENT = "event"


class CampaignStatus(Enum):
    """Campaign status"""
    DRAFT = "draft"
    SCHEDULED = "scheduled"
    SENDING = "sending"
    SENT = "sent"
    PAUSED = "paused"
    CANCELLED = "cancelled"


class SegmentType(Enum):
    """Audience segment types"""
    DEMOGRAPHIC = "demographic"
    BEHAVIORAL = "behavioral"
    ENGAGEMENT = "engagement"
    PURCHASE_HISTORY = "purchase_history"
    LIFECYCLE = "lifecycle"
    CUSTOM = "custom"


class EmailStatus(Enum):
    """Individual email status"""
    QUEUED = "queued"
    SENT = "sent"
    DELIVERED = "delivered"
    OPENED = "opened"
    CLICKED = "clicked"
    BOUNCED = "bounced"
    UNSUBSCRIBED = "unsubscribed"
    SPAM = "spam"


@dataclass
class Contact:
    """Email contact/subscriber"""
    contact_id: str
    email: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    company: Optional[str] = None
    location: Optional[str] = None
    timezone: str = "UTC"
    subscribed: bool = True
    subscribed_date: Optional[datetime] = None
    last_opened: Optional[datetime] = None
    last_clicked: Optional[datetime] = None
    total_opens: int = 0
    total_clicks: int = 0
    engagement_score: float = 0.0
    tags: List[str] = field(default_factory=list)
    custom_fields: Dict[str, Any] = field(default_factory=dict)
    segments: List[str] = field(default_factory=list)


@dataclass
class EmailTemplate:
    """Email template"""
    template_id: str
    name: str
    subject_line: str
    preview_text: str
    html_content: str
    text_content: str
    variables: List[str] = field(default_factory=list)
    category: str = "general"
    created_at: datetime = field(default_factory=datetime.now)
    performance_score: float = 0.0


@dataclass
class Campaign:
    """Email campaign"""
    campaign_id: str
    name: str
    campaign_type: CampaignType
    template_id: str
    subject_line: str
    from_name: str
    from_email: str
    reply_to: Optional[str] = None
    segments: List[str] = field(default_factory=list)
    send_time: Optional[datetime] = None
    status: CampaignStatus = CampaignStatus.DRAFT
    created_at: datetime = field(default_factory=datetime.now)
    sent_at: Optional[datetime] = None
    total_recipients: int = 0
    total_sent: int = 0
    total_delivered: int = 0
    total_opened: int = 0
    total_clicked: int = 0
    total_bounced: int = 0
    total_unsubscribed: int = 0
    ab_test: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Segment:
    """Audience segment"""
    segment_id: str
    name: str
    segment_type: SegmentType
    description: str
    criteria: Dict[str, Any]
    contact_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class ABTest:
    """A/B test configuration"""
    test_id: str
    name: str
    variable: str  # subject_line, from_name, send_time, content
    variant_a: Any
    variant_b: Any
    sample_size_percent: int = 10
    winner_criteria: str = "open_rate"  # open_rate, click_rate, conversion_rate
    results: Optional[Dict[str, Any]] = None


@dataclass
class CampaignAnalytics:
    """Campaign performance analytics"""
    campaign_id: str
    total_sent: int
    delivered: int
    opened: int
    clicked: int
    bounced: int
    unsubscribed: int
    spam_reports: int
    open_rate: float
    click_rate: float
    click_to_open_rate: float
    bounce_rate: float
    unsubscribe_rate: float
    engagement_score: float
    revenue: float = 0.0
    roi: float = 0.0
    top_links: List[Tuple[str, int]] = field(default_factory=list)
    device_breakdown: Dict[str, int] = field(default_factory=dict)
    location_breakdown: Dict[str, int] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class SubjectLineOptimizer:
    """AI-powered subject line optimization"""
    
    def __init__(self):
        self.high_performing_words = set([
            'exclusive', 'limited', 'new', 'free', 'save', 'discover',
            'announcing', 'introducing', 'unlock', 'secret', 'proven',
            'guaranteed', 'instantly', 'today', 'now', 'quick'
        ])
        
        self.spam_words = set([
            'free money', 'click here', 'act now', 'buy now', 'winner',
            'congratulations', 'urgent', 'cash', 'credit', 'million'
        ])
        
        self.power_words = set([
            'you', 'your', 'because', 'instantly', 'new', 'proven'
        ])
    
    def analyze(self, subject_line: str) -> Dict[str, Any]:
        """Analyze subject line effectiveness"""
        analysis = {
            'score': 0.0,
            'length': len(subject_line),
            'word_count': len(subject_line.split()),
            'has_emoji': bool(re.search(r'[\U0001F600-\U0001F64F]', subject_line)),
            'has_numbers': bool(re.search(r'\d', subject_line)),
            'has_personalization': bool(re.search(r'\{.*\}', subject_line)),
            'spam_risk': 0.0,
            'recommendations': []
        }
        
        subject_lower = subject_line.lower()
        words = subject_lower.split()
        
        # Score calculation
        score = 50  # Base score
        
        # Length optimization (40-60 chars ideal)
        if 40 <= analysis['length'] <= 60:
            score += 15
        elif 30 <= analysis['length'] <= 70:
            score += 10
        else:
            analysis['recommendations'].append("Aim for 40-60 characters in subject line")
        
        # Personalization
        if analysis['has_personalization']:
            score += 10
        else:
            analysis['recommendations'].append("Add personalization (e.g., {first_name})")
        
        # Power words
        power_word_count = sum(1 for word in words if word in self.power_words)
        if power_word_count > 0:
            score += min(power_word_count * 5, 15)
        
        # Numbers
        if analysis['has_numbers']:
            score += 5
        
        # Spam check
        spam_count = sum(1 for phrase in self.spam_words if phrase in subject_lower)
        if spam_count > 0:
            spam_risk = min(spam_count * 20, 80)
            analysis['spam_risk'] = spam_risk
            score -= spam_risk / 2
            analysis['recommendations'].append(f"Remove spam trigger words: {spam_count} found")
        
        # High performing words
        hp_count = sum(1 for word in words if word in self.high_performing_words)
        if hp_count > 0:
            score += min(hp_count * 3, 10)
        
        analysis['score'] = max(0, min(100, score))
        
        return analysis
    
    def generate_variants(self, base_subject: str, count: int = 5) -> List[str]:
        """Generate subject line variants"""
        variants = [base_subject]
        
        # Add personalization
        if '{first_name}' not in base_subject:
            variants.append(f"{{first_name}}, {base_subject}")
        
        # Add urgency
        urgent_prefixes = ["Don't miss:", "Last chance:", "Today only:", "Limited time:"]
        for prefix in random.sample(urgent_prefixes, min(2, len(urgent_prefixes))):
            variants.append(f"{prefix} {base_subject}")
        
        # Add emoji
        emojis = ['🎉', '🚀', '✨', '💡', '🔥']
        variants.append(f"{random.choice(emojis)} {base_subject}")
        
        # Add question
        if not base_subject.endswith('?'):
            variants.append(f"{base_subject}?")
        
        return variants[:count]


class SendTimeOptimizer:
    """Optimize email send times"""
    
    def __init__(self):
        # Historical engagement data by hour and day
        self.engagement_patterns: Dict[str, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
    
    def record_engagement(self, sent_time: datetime, opened: bool):
        """Record engagement for learning"""
        day_of_week = sent_time.strftime('%A')
        hour = sent_time.hour
        
        if opened:
            self.engagement_patterns[day_of_week][hour] += 1
    
    def predict_best_time(self, timezone: str = "UTC") -> datetime:
        """Predict best send time"""
        # Default best times if no data
        default_times = {
            'Monday': 10,
            'Tuesday': 10,
            'Wednesday': 14,
            'Thursday': 10,
            'Friday': 9,
            'Saturday': 11,
            'Sunday': 15
        }
        
        now = datetime.now()
        current_day = now.strftime('%A')
        
        if current_day in self.engagement_patterns and self.engagement_patterns[current_day]:
            # Use learned patterns
            best_hour = max(self.engagement_patterns[current_day].items(), 
                          key=lambda x: x[1])[0]
        else:
            # Use defaults
            best_hour = default_times.get(current_day, 10)
        
        # Create send time
        send_time = now.replace(hour=best_hour, minute=0, second=0, microsecond=0)
        
        # If time passed today, schedule for tomorrow
        if send_time < now:
            send_time += timedelta(days=1)
        
        return send_time


class PersonalizationEngine:
    """Personalize email content"""
    
    def __init__(self):
        self.default_values = {
            'first_name': 'there',
            'last_name': '',
            'company': 'your company',
            'location': 'your area'
        }
    
    def personalize(self, template: str, contact: Contact) -> str:
        """Personalize email template"""
        # Build replacement dict
        replacements = {
            'first_name': contact.first_name or self.default_values['first_name'],
            'last_name': contact.last_name or self.default_values['last_name'],
            'company': contact.company or self.default_values['company'],
            'location': contact.location or self.default_values['location'],
            'email': contact.email
        }
        
        # Add custom fields
        replacements.update(contact.custom_fields)
        
        # Replace variables
        personalized = template
        for key, value in replacements.items():
            personalized = personalized.replace(f'{{{key}}}', str(value))
        
        return personalized
    
    def extract_variables(self, template: str) -> List[str]:
        """Extract variables from template"""
        pattern = r'\{([^}]+)\}'
        return re.findall(pattern, template)


class SegmentationEngine:
    """Audience segmentation"""
    
    def __init__(self):
        self.segments: Dict[str, Segment] = {}
    
    def create_segment(self, name: str, segment_type: SegmentType,
                      description: str, criteria: Dict[str, Any]) -> Segment:
        """Create new segment"""
        segment_id = self._generate_segment_id()
        
        segment = Segment(
            segment_id=segment_id,
            name=name,
            segment_type=segment_type,
            description=description,
            criteria=criteria
        )
        
        self.segments[segment_id] = segment
        logger.info(f"Segment created: {name}")
        
        return segment
    
    def evaluate_contact(self, contact: Contact, segment: Segment) -> bool:
        """Check if contact matches segment criteria"""
        criteria = segment.criteria
        
        # Engagement-based
        if segment.segment_type == SegmentType.ENGAGEMENT:
            min_score = criteria.get('min_engagement_score', 0)
            if contact.engagement_score < min_score:
                return False
            
            min_opens = criteria.get('min_opens', 0)
            if contact.total_opens < min_opens:
                return False
        
        # Behavioral
        elif segment.segment_type == SegmentType.BEHAVIORAL:
            required_tags = criteria.get('tags', [])
            if not all(tag in contact.tags for tag in required_tags):
                return False
        
        # Demographic
        elif segment.segment_type == SegmentType.DEMOGRAPHIC:
            if 'location' in criteria and contact.location != criteria['location']:
                return False
        
        # Custom criteria
        elif segment.segment_type == SegmentType.CUSTOM:
            # Evaluate custom logic
            for field, value in criteria.items():
                contact_value = getattr(contact, field, None) or contact.custom_fields.get(field)
                if contact_value != value:
                    return False
        
        return True
    
    def get_segment_contacts(self, segment_id: str, 
                            all_contacts: List[Contact]) -> List[Contact]:
        """Get contacts matching segment"""
        if segment_id not in self.segments:
            return []
        
        segment = self.segments[segment_id]
        matching_contacts = [
            contact for contact in all_contacts
            if self.evaluate_contact(contact, segment)
        ]
        
        segment.contact_count = len(matching_contacts)
        segment.updated_at = datetime.now()
        
        return matching_contacts
    
    def _generate_segment_id(self) -> str:
        """Generate unique segment ID"""
        return f"SEG-{hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]}"


class DeliverabilityAnalyzer:
    """Email deliverability analysis"""
    
    def __init__(self):
        self.spam_triggers = [
            'free', 'click here', 'buy now', 'limited time', 'act now',
            'winner', 'congratulations', 'urgent', 'cash', 'credit'
        ]
    
    def analyze(self, subject: str, html_content: str, 
                from_email: str) -> Dict[str, Any]:
        """Analyze email deliverability"""
        analysis = {
            'deliverability_score': 100.0,
            'issues': [],
            'warnings': [],
            'recommendations': []
        }
        
        score = 100
        
        # Check subject line
        subject_lower = subject.lower()
        spam_words_found = [word for word in self.spam_triggers if word in subject_lower]
        if spam_words_found:
            score -= len(spam_words_found) * 5
            analysis['issues'].append(f"Spam trigger words in subject: {spam_words_found}")
        
        # Check for all caps
        if subject.isupper():
            score -= 10
            analysis['issues'].append("Subject line is all caps")
        
        # Check excessive punctuation
        if subject.count('!') > 1:
            score -= 5
            analysis['warnings'].append("Multiple exclamation marks in subject")
        
        # HTML content checks
        if html_content:
            # Image-to-text ratio
            img_count = html_content.count('<img')
            text_length = len(re.sub(r'<[^>]+>', '', html_content))
            
            if img_count > 5 and text_length < 500:
                score -= 15
                analysis['issues'].append("Poor image-to-text ratio")
                analysis['recommendations'].append("Add more text content")
            
            # Links check
            link_count = html_content.count('href=')
            if link_count > 20:
                score -= 10
                analysis['warnings'].append("High number of links may trigger spam filters")
        
        # From email check
        if 'noreply' in from_email.lower():
            score -= 10
            analysis['warnings'].append("'noreply' addresses may reduce deliverability")
        
        analysis['deliverability_score'] = max(0, score)
        
        return analysis


class ABTestManager:
    """Manage A/B tests"""
    
    def __init__(self):
        self.tests: Dict[str, ABTest] = {}
    
    def create_test(self, name: str, variable: str,
                   variant_a: Any, variant_b: Any,
                   sample_size_percent: int = 10) -> ABTest:
        """Create A/B test"""
        test_id = f"ABT-{hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]}"
        
        test = ABTest(
            test_id=test_id,
            name=name,
            variable=variable,
            variant_a=variant_a,
            variant_b=variant_b,
            sample_size_percent=sample_size_percent
        )
        
        self.tests[test_id] = test
        return test
    
    def evaluate_results(self, test_id: str,
                        variant_a_metrics: Dict[str, float],
                        variant_b_metrics: Dict[str, float]) -> str:
        """Evaluate A/B test results"""
        if test_id not in self.tests:
            return "unknown"
        
        test = self.tests[test_id]
        criteria = test.winner_criteria
        
        a_score = variant_a_metrics.get(criteria, 0)
        b_score = variant_b_metrics.get(criteria, 0)
        
        # Statistical significance check (simplified)
        min_difference = 0.05  # 5% minimum difference
        
        if abs(a_score - b_score) / max(a_score, b_score, 0.01) < min_difference:
            winner = "inconclusive"
        elif a_score > b_score:
            winner = "variant_a"
        else:
            winner = "variant_b"
        
        test.results = {
            'winner': winner,
            'variant_a_metrics': variant_a_metrics,
            'variant_b_metrics': variant_b_metrics,
            'evaluated_at': datetime.now().isoformat()
        }
        
        return winner


class EmailMarketingAI:
    """
    Main Email Marketing AI Agent
    Enterprise-grade intelligent email marketing automation
    """
    
    def __init__(self):
        """Initialize Email Marketing AI"""
        self.subject_line_optimizer = SubjectLineOptimizer()
        self.send_time_optimizer = SendTimeOptimizer()
        self.personalization_engine = PersonalizationEngine()
        self.segmentation_engine = SegmentationEngine()
        self.deliverability_analyzer = DeliverabilityAnalyzer()
        self.ab_test_manager = ABTestManager()
        
        # Storage
        self.contacts: Dict[str, Contact] = {}
        self.templates: Dict[str, EmailTemplate] = {}
        self.campaigns: Dict[str, Campaign] = {}
        
        # Analytics
        self.total_emails_sent = 0
        self.total_opens = 0
        self.total_clicks = 0
        
        logger.info("Email Marketing AI initialized successfully")
    
    def add_contact(self, contact: Contact) -> bool:
        """Add or update contact"""
        self.contacts[contact.contact_id] = contact
        logger.info(f"Contact added: {contact.email}")
        return True
    
    def import_contacts(self, contacts: List[Dict[str, Any]]) -> int:
        """Bulk import contacts"""
        imported = 0
        
        for contact_data in contacts:
            try:
                contact = Contact(
                    contact_id=contact_data.get('contact_id', self._generate_contact_id()),
                    email=contact_data['email'],
                    first_name=contact_data.get('first_name'),
                    last_name=contact_data.get('last_name'),
                    company=contact_data.get('company'),
                    location=contact_data.get('location'),
                    tags=contact_data.get('tags', []),
                    custom_fields=contact_data.get('custom_fields', {})
                )
                
                self.add_contact(contact)
                imported += 1
            
            except Exception as e:
                logger.error(f"Failed to import contact: {e}")
        
        logger.info(f"Imported {imported} contacts")
        return imported
    
    def create_template(self, name: str, subject_line: str,
                       preview_text: str, html_content: str,
                       text_content: str = "") -> EmailTemplate:
        """Create email template"""
        template_id = f"TPL-{hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]}"
        
        # Extract variables
        variables = self.personalization_engine.extract_variables(html_content)
        
        template = EmailTemplate(
            template_id=template_id,
            name=name,
            subject_line=subject_line,
            preview_text=preview_text,
            html_content=html_content,
            text_content=text_content or self._html_to_text(html_content),
            variables=variables
        )
        
        self.templates[template_id] = template
        logger.info(f"Template created: {name}")
        
        return template
    
    def create_campaign(self, name: str, campaign_type: CampaignType,
                       template_id: str, from_name: str, from_email: str,
                       segments: List[str] = None, 
                       send_time: datetime = None) -> Campaign:
        """Create email campaign"""
        if template_id not in self.templates:
            raise ValueError(f"Template not found: {template_id}")
        
        campaign_id = f"CMP-{hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]}"
        template = self.templates[template_id]
        
        campaign = Campaign(
            campaign_id=campaign_id,
            name=name,
            campaign_type=campaign_type,
            template_id=template_id,
            subject_line=template.subject_line,
            from_name=from_name,
            from_email=from_email,
            segments=segments or [],
            send_time=send_time
        )
        
        self.campaigns[campaign_id] = campaign
        logger.info(f"Campaign created: {name}")
        
        return campaign
    
    def optimize_campaign(self, campaign_id: str) -> Dict[str, Any]:
        """AI-powered campaign optimization"""
        if campaign_id not in self.campaigns:
            return {'error': 'Campaign not found'}
        
        campaign = self.campaigns[campaign_id]
        template = self.templates[campaign.template_id]
        
        optimizations = {
            'campaign_id': campaign_id,
            'recommendations': [],
            'predicted_performance': {}
        }
        
        # Optimize subject line
        subject_analysis = self.subject_line_optimizer.analyze(campaign.subject_line)
        optimizations['subject_line_score'] = subject_analysis['score']
        
        if subject_analysis['score'] < 70:
            variants = self.subject_line_optimizer.generate_variants(campaign.subject_line)
            optimizations['recommendations'].append({
                'type': 'subject_line',
                'message': 'Subject line score is low',
                'suggested_variants': variants[:3]
            })
        
        # Optimize send time
        best_time = self.send_time_optimizer.predict_best_time()
        if not campaign.send_time or campaign.send_time < datetime.now():
            optimizations['recommendations'].append({
                'type': 'send_time',
                'message': 'Optimize send time for better engagement',
                'suggested_time': best_time.isoformat()
            })
        
        # Check deliverability
        deliverability = self.deliverability_analyzer.analyze(
            campaign.subject_line,
            template.html_content,
            campaign.from_email
        )
        optimizations['deliverability_score'] = deliverability['deliverability_score']
        
        if deliverability['issues']:
            optimizations['recommendations'].append({
                'type': 'deliverability',
                'message': 'Deliverability issues detected',
                'issues': deliverability['issues']
            })
        
        # Predict performance
        predicted_open_rate = self._predict_open_rate(campaign, subject_analysis['score'])
        predicted_click_rate = self._predict_click_rate(campaign)
        
        optimizations['predicted_performance'] = {
            'open_rate': round(predicted_open_rate, 2),
            'click_rate': round(predicted_click_rate, 2),
            'deliverability': deliverability['deliverability_score']
        }
        
        return optimizations
    
    def send_campaign(self, campaign_id: str, 
                     test_mode: bool = False) -> Dict[str, Any]:
        """Send email campaign"""
        if campaign_id not in self.campaigns:
            return {'error': 'Campaign not found'}
        
        campaign = self.campaigns[campaign_id]
        template = self.templates[campaign.template_id]
        
        # Get recipients
        recipients = self._get_campaign_recipients(campaign)
        
        if not recipients:
            return {'error': 'No recipients found'}
        
        # Update campaign
        campaign.total_recipients = len(recipients)
        campaign.status = CampaignStatus.SENDING
        
        sent_count = 0
        
        # Send to each recipient
        for contact in recipients:
            if test_mode and sent_count >= 10:
                break
            
            # Personalize content
            personalized_html = self.personalization_engine.personalize(
                template.html_content, contact
            )
            personalized_subject = self.personalization_engine.personalize(
                campaign.subject_line, contact
            )
            
            # Simulate sending (in real app, integrate with email service)
            success = self._send_email(
                to_email=contact.email,
                from_email=campaign.from_email,
                from_name=campaign.from_name,
                subject=personalized_subject,
                html_content=personalized_html
            )
            
            if success:
                sent_count += 1
                self.total_emails_sent += 1
        
        # Update campaign
        campaign.total_sent = sent_count
        campaign.sent_at = datetime.now()
        campaign.status = CampaignStatus.SENT
        
        logger.info(f"Campaign sent: {campaign.name} ({sent_count} emails)")
        
        return {
            'campaign_id': campaign_id,
            'sent': sent_count,
            'total_recipients': campaign.total_recipients,
            'status': 'success'
        }
    
    def track_event(self, campaign_id: str, contact_id: str,
                   event_type: EmailStatus, metadata: Dict[str, Any] = None):
        """Track email events (open, click, etc.)"""
        if campaign_id not in self.campaigns:
            return
        
        campaign = self.campaigns[campaign_id]
        
        # Update campaign metrics
        if event_type == EmailStatus.DELIVERED:
            campaign.total_delivered += 1
        elif event_type == EmailStatus.OPENED:
            campaign.total_opened += 1
            self.total_opens += 1
        elif event_type == EmailStatus.CLICKED:
            campaign.total_clicked += 1
            self.total_clicks += 1
        elif event_type == EmailStatus.BOUNCED:
            campaign.total_bounced += 1
        elif event_type == EmailStatus.UNSUBSCRIBED:
            campaign.total_unsubscribed += 1
        
        # Update contact engagement
        if contact_id in self.contacts:
            contact = self.contacts[contact_id]
            
            if event_type == EmailStatus.OPENED:
                contact.total_opens += 1
                contact.last_opened = datetime.now()
            elif event_type == EmailStatus.CLICKED:
                contact.total_clicks += 1
                contact.last_clicked = datetime.now()
            elif event_type == EmailStatus.UNSUBSCRIBED:
                contact.subscribed = False
            
            # Update engagement score
            contact.engagement_score = self._calculate_engagement_score(contact)
        
        logger.debug(f"Event tracked: {event_type.value} for campaign {campaign_id}")
    
    def get_campaign_analytics(self, campaign_id: str) -> CampaignAnalytics:
        """Get campaign performance analytics"""
        if campaign_id not in self.campaigns:
            return None
        
        campaign = self.campaigns[campaign_id]
        
        # Calculate rates
        open_rate = (campaign.total_opened / max(campaign.total_delivered, 1)) * 100
        click_rate = (campaign.total_clicked / max(campaign.total_delivered, 1)) * 100
        cto_rate = (campaign.total_clicked / max(campaign.total_opened, 1)) * 100
        bounce_rate = (campaign.total_bounced / max(campaign.total_sent, 1)) * 100
        unsub_rate = (campaign.total_unsubscribed / max(campaign.total_delivered, 1)) * 100
        
        # Engagement score
        engagement_score = (open_rate * 0.4 + click_rate * 0.6)
        
        analytics = CampaignAnalytics(
            campaign_id=campaign_id,
            total_sent=campaign.total_sent,
            delivered=campaign.total_delivered,
            opened=campaign.total_opened,
            clicked=campaign.total_clicked,
            bounced=campaign.total_bounced,
            unsubscribed=campaign.total_unsubscribed,
            spam_reports=0,  # Would track separately
            open_rate=round(open_rate, 2),
            click_rate=round(click_rate, 2),
            click_to_open_rate=round(cto_rate, 2),
            bounce_rate=round(bounce_rate, 2),
            unsubscribe_rate=round(unsub_rate, 2),
            engagement_score=round(engagement_score, 2)
        )
        
        return analytics
    
    def _get_campaign_recipients(self, campaign: Campaign) -> List[Contact]:
        """Get campaign recipients based on segments"""
        if not campaign.segments:
            # All subscribed contacts
            return [c for c in self.contacts.values() if c.subscribed]
        
        recipients = set()
        
        for segment_id in campaign.segments:
            segment_contacts = self.segmentation_engine.get_segment_contacts(
                segment_id,
                list(self.contacts.values())
            )
            recipients.update(c.contact_id for c in segment_contacts)
        
        return [self.contacts[cid] for cid in recipients if self.contacts[cid].subscribed]
    
    def _send_email(self, to_email: str, from_email: str, from_name: str,
                   subject: str, html_content: str) -> bool:
        """Simulate sending email (integrate with ESP in production)"""
        # In production, integrate with services like:
        # - SendGrid
        # - Amazon SES
        # - Mailgun
        # - Postmark
        
        logger.debug(f"Sending email to {to_email}")
        return True  # Simulated success
    
    def _html_to_text(self, html: str) -> str:
        """Convert HTML to plain text"""
        # Simple HTML stripping
        text = re.sub(r'<[^>]+>', '', html)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _generate_contact_id(self) -> str:
        """Generate unique contact ID"""
        return f"CNT-{hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]}"
    
    def _calculate_engagement_score(self, contact: Contact) -> float:
        """Calculate contact engagement score (0-100)"""
        # Weighted scoring
        open_score = min(contact.total_opens * 2, 40)
        click_score = min(contact.total_clicks * 5, 50)
        recency_score = 10  # Default
        
        if contact.last_opened:
            days_since = (datetime.now() - contact.last_opened).days
            if days_since < 7:
                recency_score = 10
            elif days_since < 30:
                recency_score = 7
            elif days_since < 90:
                recency_score = 4
            else:
                recency_score = 1
        
        total_score = open_score + click_score + recency_score
        return min(total_score, 100)
    
    def _predict_open_rate(self, campaign: Campaign, subject_score: float) -> float:
        """Predict campaign open rate"""
        # Simplified prediction based on subject line quality
        base_rate = 20.0  # Industry average
        subject_factor = (subject_score / 100) * 10
        
        # Could factor in: send time, segment quality, sender reputation
        predicted = base_rate + subject_factor
        
        return min(predicted, 100)
    
    def _predict_click_rate(self, campaign: Campaign) -> float:
        """Predict campaign click rate"""
        # Simplified prediction
        base_rate = 3.0  # Industry average
        
        # Would factor in: content quality, CTA placement, personalization
        return base_rate
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get overall email marketing statistics"""
        total_contacts = len(self.contacts)
        subscribed = sum(1 for c in self.contacts.values() if c.subscribed)
        
        avg_open_rate = (self.total_opens / max(self.total_emails_sent, 1)) * 100
        avg_click_rate = (self.total_clicks / max(self.total_emails_sent, 1)) * 100
        
        return {
            'total_contacts': total_contacts,
            'subscribed_contacts': subscribed,
            'total_campaigns': len(self.campaigns),
            'total_templates': len(self.templates),
            'total_emails_sent': self.total_emails_sent,
            'average_open_rate': round(avg_open_rate, 2),
            'average_click_rate': round(avg_click_rate, 2),
            'total_segments': len(self.segmentation_engine.segments)
        }


# Example usage
if __name__ == "__main__":
    # Initialize AI
    ai = EmailMarketingAI()
    
    print("=" * 80)
    print("EMAIL MARKETING AI DEMO")
    print("=" * 80)
    
    # Import contacts
    sample_contacts = [
        {'email': 'john@example.com', 'first_name': 'John', 'last_name': 'Doe'},
        {'email': 'jane@example.com', 'first_name': 'Jane', 'last_name': 'Smith'},
        {'email': 'bob@example.com', 'first_name': 'Bob', 'last_name': 'Johnson'}
    ]
    
    imported = ai.import_contacts(sample_contacts)
    print(f"\nImported {imported} contacts")
    
    # Create template
    template = ai.create_template(
        name="Welcome Email",
        subject_line="Welcome to Osprey AI, {first_name}!",
        preview_text="Get started with your AI-powered platform",
        html_content="""
        <h1>Welcome {first_name}!</h1>
        <p>We're excited to have you on board at Osprey AI.</p>
        <p>Your company: {company}</p>
        <a href="https://example.com/get-started">Get Started</a>
        """
    )
    
    print(f"\nTemplate created: {template.name}")
    print(f"Variables found: {template.variables}")
    
    # Create campaign
    campaign = ai.create_campaign(
        name="Welcome Series - Week 1",
        campaign_type=CampaignType.WELCOME,
        template_id=template.template_id,
        from_name="Osprey AI Team",
        from_email="hello@ospreyai.com"
    )
    
    print(f"\nCampaign created: {campaign.name}")
    
    # Optimize campaign
    optimization = ai.optimize_campaign(campaign.campaign_id)
    print(f"\nCampaign Optimization:")
    print(f"Subject Score: {optimization['subject_line_score']}")
    print(f"Predicted Open Rate: {optimization['predicted_performance']['open_rate']}%")
    print(f"Deliverability Score: {optimization['deliverability_score']}")
    
    # Send campaign (test mode)
    result = ai.send_campaign(campaign.campaign_id, test_mode=True)
    print(f"\nCampaign sent: {result['sent']} emails")
    
    # Simulate some engagement
    for contact_id in list(ai.contacts.keys())[:2]:
        ai.track_event(campaign.campaign_id, contact_id, EmailStatus.DELIVERED)
        ai.track_event(campaign.campaign_id, contact_id, EmailStatus.OPENED)
    
    # Get analytics
    analytics = ai.get_campaign_analytics(campaign.campaign_id)
    print(f"\nCampaign Analytics:")
    print(f"Open Rate: {analytics.open_rate}%")
    print(f"Click Rate: {analytics.click_rate}%")
    print(f"Engagement Score: {analytics.engagement_score}")
    
    # Statistics
    stats = ai.get_statistics()
    print(f"\n{'=' * 80}")
    print("OVERALL STATISTICS")
    print("=" * 80)
    print(json.dumps(stats, indent=2))


# ===========================================================================
# ENTERPRISE EXTENSIONS — Research & Email Intelligence Layer
# ===========================================================================

import hashlib
import csv
import io
from collections import Counter
from typing import Iterator


# ---------------------------------------------------------------------------
# Email Deliverability Engine
# ---------------------------------------------------------------------------

class SpamSignalDetector:
    """Detect spam trigger words and patterns before sending."""

    SPAM_TRIGGERS = [
        "free money", "guaranteed", "no risk", "winner", "you've been selected",
        "click here now", "limited time offer", "act immediately", "urgent",
        "!!!",  "ALL CAPS SUBJECT", "dear friend", "this is not spam",
        "100% free", "cash bonus", "earn money", "extra income",
        "lose weight", "miracle", "amazing offer", "unsubscribe",
    ]

    CAPS_RATIO_THRESHOLD = 0.4

    def analyze(self, subject: str, body: str) -> dict:
        combined = f"{subject} {body}".lower()
        triggered = [t for t in self.SPAM_TRIGGERS if t in combined]
        caps_ratio = sum(1 for c in subject if c.isupper()) / max(len(subject), 1)
        link_count = combined.count("http")
        image_text_ratio = body.count("<img") / max(len(body.split()), 1)

        score = (
            len(triggered) * 10
            + (20 if caps_ratio > self.CAPS_RATIO_THRESHOLD else 0)
            + max(0, (link_count - 3) * 5)
            + (15 if image_text_ratio > 0.3 else 0)
        )
        return {
            "spam_score": min(score, 100),
            "triggered_words": triggered,
            "caps_ratio": round(caps_ratio, 3),
            "link_count": link_count,
            "deliverability_risk": "HIGH" if score >= 40 else "MEDIUM" if score >= 20 else "LOW",
            "recommendation": (
                "Revise subject and body — high spam probability."
                if score >= 40
                else "Minor adjustments recommended." if score >= 20
                else "Good to send."
            ),
        }


class BounceProcessor:
    """Process and categorize email bounces."""

    HARD_BOUNCE_CODES = {550, 551, 552, 553, 554}
    SOFT_BOUNCE_CODES = {421, 450, 451, 452}

    def __init__(self):
        self._bounces: list[dict] = []
        self._suppression_list: set[str] = set()

    def record_bounce(self, email: str, code: int, message: str) -> dict:
        bounce_type = (
            "hard" if code in self.HARD_BOUNCE_CODES
            else "soft" if code in self.SOFT_BOUNCE_CODES
            else "unknown"
        )
        if bounce_type == "hard":
            self._suppression_list.add(email.lower())
        entry = {
            "email": email,
            "code": code,
            "type": bounce_type,
            "message": message,
            "suppressed": bounce_type == "hard",
            "timestamp": datetime.now().isoformat() if "datetime" in dir() else "",
        }
        self._bounces.append(entry)
        return entry

    def is_suppressed(self, email: str) -> bool:
        return email.lower() in self._suppression_list

    def clean_list(self, emails: list[str]) -> tuple[list[str], list[str]]:
        """Return (clean, removed) lists."""
        clean = [e for e in emails if not self.is_suppressed(e)]
        removed = [e for e in emails if self.is_suppressed(e)]
        return clean, removed

    def summary(self) -> dict:
        hard = sum(1 for b in self._bounces if b["type"] == "hard")
        soft = sum(1 for b in self._bounces if b["type"] == "soft")
        return {
            "total_bounces": len(self._bounces),
            "hard_bounces": hard,
            "soft_bounces": soft,
            "suppressed_addresses": len(self._suppression_list),
        }


# ---------------------------------------------------------------------------
# Advanced Segmentation Engine
# ---------------------------------------------------------------------------

class BehavioralSegmenter:
    """Segment email lists by behavioral signals."""

    def __init__(self):
        self._segments: dict[str, list[str]] = {
            "champions": [],
            "loyal": [],
            "at_risk": [],
            "hibernating": [],
            "lost": [],
            "new": [],
        }

    def classify_subscriber(
        self,
        email: str,
        opens_last_30: int,
        clicks_last_30: int,
        days_since_last_open: int,
        total_purchases: int,
    ) -> str:
        if opens_last_30 >= 8 and clicks_last_30 >= 3 and total_purchases >= 3:
            segment = "champions"
        elif opens_last_30 >= 5 and total_purchases >= 1:
            segment = "loyal"
        elif days_since_last_open > 90:
            segment = "lost"
        elif days_since_last_open > 45:
            segment = "hibernating"
        elif days_since_last_open > 20:
            segment = "at_risk"
        else:
            segment = "new"
        self._segments[segment].append(email)
        return segment

    def get_segment(self, name: str) -> list[str]:
        return self._segments.get(name, [])

    def segment_sizes(self) -> dict[str, int]:
        return {k: len(v) for k, v in self._segments.items()}

    def re_engagement_candidates(self) -> list[str]:
        return self._segments["hibernating"] + self._segments["at_risk"]


class RFMAnalyzer:
    """Recency-Frequency-Monetary value analysis for email lists."""

    def score_subscriber(
        self, days_since_last_action: int, total_actions: int, total_value: float
    ) -> dict:
        r = max(1, 5 - (days_since_last_action // 30))
        f = min(5, total_actions // 5 + 1)
        m = min(5, int(total_value // 50) + 1)
        rfm_score = r + f + m
        segment = (
            "VIP" if rfm_score >= 13
            else "Loyal" if rfm_score >= 10
            else "Potential" if rfm_score >= 7
            else "At-Risk" if rfm_score >= 5
            else "Dormant"
        )
        return {"recency": r, "frequency": f, "monetary": m, "rfm_score": rfm_score, "segment": segment}


# ---------------------------------------------------------------------------
# Workflow Automation Engine
# ---------------------------------------------------------------------------

class TriggerCondition:
    def __init__(self, field: str, operator: str, value):
        self.field = field
        self.operator = operator
        self.value = value

    def evaluate(self, context: dict) -> bool:
        actual = context.get(self.field)
        if self.operator == "equals":
            return actual == self.value
        if self.operator == "greater_than":
            return isinstance(actual, (int, float)) and actual > self.value
        if self.operator == "less_than":
            return isinstance(actual, (int, float)) and actual < self.value
        if self.operator == "contains":
            return isinstance(actual, str) and self.value in actual
        if self.operator == "not_equals":
            return actual != self.value
        return False


class WorkflowStep:
    def __init__(self, step_id: str, action: str, params: dict, delay_hours: int = 0):
        self.step_id = step_id
        self.action = action
        self.params = params
        self.delay_hours = delay_hours

    def to_dict(self) -> dict:
        return {
            "step_id": self.step_id,
            "action": self.action,
            "params": self.params,
            "delay_hours": self.delay_hours,
        }


class EmailWorkflow:
    """Define multi-step automated email sequences."""

    def __init__(self, workflow_id: str, name: str, trigger: TriggerCondition):
        self.workflow_id = workflow_id
        self.name = name
        self.trigger = trigger
        self.steps: list[WorkflowStep] = []
        self._execution_log: list[dict] = []

    def add_step(self, step: WorkflowStep) -> "EmailWorkflow":
        self.steps.append(step)
        return self

    def evaluate_trigger(self, context: dict) -> bool:
        return self.trigger.evaluate(context)

    def simulate_execution(self, subscriber_email: str, context: dict) -> list[dict]:
        if not self.evaluate_trigger(context):
            return []
        log = []
        for step in self.steps:
            entry = {
                "workflow": self.workflow_id,
                "subscriber": subscriber_email,
                "step": step.step_id,
                "action": step.action,
                "scheduled_delay_h": step.delay_hours,
                "params": step.params,
                "status": "queued",
            }
            log.append(entry)
            self._execution_log.append(entry)
        return log

    def execution_summary(self) -> dict:
        total = len(self._execution_log)
        by_action = Counter(e["action"] for e in self._execution_log)
        return {"total_executions": total, "by_action": dict(by_action)}


class WorkflowLibrary:
    """Pre-built workflow templates for common email sequences."""

    @staticmethod
    def welcome_series(brand: str = "Our Team") -> EmailWorkflow:
        trigger = TriggerCondition("event", "equals", "signup")
        wf = EmailWorkflow("WF-WELCOME", f"Welcome Series — {brand}", trigger)
        wf.add_step(WorkflowStep("S1", "send_email", {
            "template": "welcome_immediate",
            "subject": f"Welcome to {brand}!",
        }, delay_hours=0))
        wf.add_step(WorkflowStep("S2", "send_email", {
            "template": "onboarding_tips",
            "subject": "3 tips to get started",
        }, delay_hours=24))
        wf.add_step(WorkflowStep("S3", "send_email", {
            "template": "social_proof",
            "subject": "See what others are achieving",
        }, delay_hours=72))
        wf.add_step(WorkflowStep("S4", "tag_subscriber", {
            "tag": "welcome_series_complete",
        }, delay_hours=96))
        return wf

    @staticmethod
    def re_engagement_sequence() -> EmailWorkflow:
        trigger = TriggerCondition("days_inactive", "greater_than", 45)
        wf = EmailWorkflow("WF-REENGAGE", "Re-engagement Sequence", trigger)
        wf.add_step(WorkflowStep("S1", "send_email", {
            "template": "we_miss_you",
            "subject": "We miss you! Here's 20% off",
        }, delay_hours=0))
        wf.add_step(WorkflowStep("S2", "wait_for_event", {
            "event": "email_open",
            "timeout_hours": 72,
        }, delay_hours=0))
        wf.add_step(WorkflowStep("S3", "send_email", {
            "template": "last_chance",
            "subject": "Last chance — offer expires soon",
        }, delay_hours=72))
        wf.add_step(WorkflowStep("S4", "update_segment", {
            "if_no_open": "suppress",
            "if_open": "re_engaged",
        }, delay_hours=144))
        return wf

    @staticmethod
    def cart_abandonment() -> EmailWorkflow:
        trigger = TriggerCondition("event", "equals", "cart_abandoned")
        wf = EmailWorkflow("WF-CART", "Cart Abandonment Recovery", trigger)
        wf.add_step(WorkflowStep("S1", "send_email", {
            "template": "cart_reminder",
            "subject": "You left something behind!",
        }, delay_hours=1))
        wf.add_step(WorkflowStep("S2", "send_email", {
            "template": "cart_incentive",
            "subject": "Complete your purchase — 10% off waiting",
        }, delay_hours=24))
        wf.add_step(WorkflowStep("S3", "send_email", {
            "template": "cart_urgency",
            "subject": "Almost gone — items in your cart are selling fast",
        }, delay_hours=48))
        return wf


# ---------------------------------------------------------------------------
# Reporting & Export Engine
# ---------------------------------------------------------------------------

class CampaignReportBuilder:
    """Build structured performance reports for email campaigns."""

    def __init__(self):
        self._reports: list[dict] = []

    def build_report(
        self,
        campaign_id: str,
        name: str,
        sent: int,
        delivered: int,
        opened: int,
        clicked: int,
        bounced: int,
        unsubscribed: int,
        revenue: float = 0.0,
    ) -> dict:
        open_rate = round(opened / max(delivered, 1) * 100, 2)
        click_rate = round(clicked / max(delivered, 1) * 100, 2)
        cto_rate = round(clicked / max(opened, 1) * 100, 2)
        bounce_rate = round(bounced / max(sent, 1) * 100, 2)
        unsub_rate = round(unsubscribed / max(delivered, 1) * 100, 2)
        rpu = round(revenue / max(sent, 1), 4) if revenue else 0
        report = {
            "campaign_id": campaign_id,
            "campaign_name": name,
            "metrics": {
                "sent": sent,
                "delivered": delivered,
                "opened": opened,
                "clicked": clicked,
                "bounced": bounced,
                "unsubscribed": unsubscribed,
            },
            "rates": {
                "open_rate": open_rate,
                "click_rate": click_rate,
                "click_to_open_rate": cto_rate,
                "bounce_rate": bounce_rate,
                "unsubscribe_rate": unsub_rate,
            },
            "revenue": {
                "total": revenue,
                "revenue_per_send": rpu,
            },
            "grade": self._grade(open_rate, click_rate, bounce_rate),
        }
        self._reports.append(report)
        return report

    def _grade(self, open_rate: float, click_rate: float, bounce_rate: float) -> str:
        score = 0
        score += 3 if open_rate >= 25 else 2 if open_rate >= 15 else 1
        score += 3 if click_rate >= 4 else 2 if click_rate >= 2 else 1
        score -= 2 if bounce_rate >= 5 else 1 if bounce_rate >= 2 else 0
        if score >= 5:
            return "A"
        if score >= 4:
            return "B"
        if score >= 3:
            return "C"
        return "D"

    def export_csv(self) -> str:
        if not self._reports:
            return "No reports available."
        buf = io.StringIO()
        fields = [
            "campaign_id", "campaign_name",
            "sent", "delivered", "opened", "clicked", "bounced",
            "open_rate", "click_rate", "bounce_rate", "grade", "revenue_total",
        ]
        writer = csv.DictWriter(buf, fieldnames=fields)
        writer.writeheader()
        for r in self._reports:
            writer.writerow({
                "campaign_id": r["campaign_id"],
                "campaign_name": r["campaign_name"],
                **r["metrics"],
                **r["rates"],
                "grade": r["grade"],
                "revenue_total": r["revenue"]["total"],
            })
        return buf.getvalue()

    def benchmark_comparison(self, industry: str = "SaaS") -> dict:
        BENCHMARKS = {
            "SaaS": {"open_rate": 21.5, "click_rate": 2.9, "bounce_rate": 0.9},
            "Ecommerce": {"open_rate": 15.7, "click_rate": 2.0, "bounce_rate": 0.3},
            "Media": {"open_rate": 22.1, "click_rate": 4.6, "bounce_rate": 0.4},
            "Finance": {"open_rate": 27.1, "click_rate": 2.4, "bounce_rate": 1.1},
        }
        bench = BENCHMARKS.get(industry, BENCHMARKS["SaaS"])
        if not self._reports:
            return {"benchmark": bench, "your_average": None}
        avg_open = sum(r["rates"]["open_rate"] for r in self._reports) / len(self._reports)
        avg_click = sum(r["rates"]["click_rate"] for r in self._reports) / len(self._reports)
        avg_bounce = sum(r["rates"]["bounce_rate"] for r in self._reports) / len(self._reports)
        return {
            "industry": industry,
            "benchmark": bench,
            "your_average": {
                "open_rate": round(avg_open, 2),
                "click_rate": round(avg_click, 2),
                "bounce_rate": round(avg_bounce, 2),
            },
            "vs_benchmark": {
                "open_rate": round(avg_open - bench["open_rate"], 2),
                "click_rate": round(avg_click - bench["click_rate"], 2),
                "bounce_rate": round(avg_bounce - bench["bounce_rate"], 2),
            },
        }


# ---------------------------------------------------------------------------
# Subject Line Intelligence
# ---------------------------------------------------------------------------

class SubjectLineOptimizer:
    """Score and suggest improvements for email subject lines."""

    POWER_WORDS = [
        "exclusive", "limited", "free", "new", "you", "instantly",
        "proven", "guaranteed", "results", "secret", "discover",
        "amazing", "simple", "fast", "effortless", "complete",
    ]
    CURIOSITY_HOOKS = ["?", "...", "here's why", "the truth about", "what nobody tells you"]
    PERSONALIZATION_TOKENS = ["{first_name}", "{company}", "{city}", "{{name}}"]

    def score(self, subject: str) -> dict:
        length = len(subject)
        word_count = len(subject.split())
        has_number = any(c.isdigit() for c in subject)
        has_emoji = any(ord(c) > 127 for c in subject)
        has_power_word = any(pw in subject.lower() for pw in self.POWER_WORDS)
        has_personalization = any(tok in subject for tok in self.PERSONALIZATION_TOKENS)
        has_curiosity = any(h in subject.lower() for h in self.CURIOSITY_HOOKS)

        score = 50  # base
        if 30 <= length <= 55:
            score += 15
        elif length < 20 or length > 70:
            score -= 10
        if has_number:
            score += 8
        if has_emoji:
            score += 5
        if has_power_word:
            score += 7
        if has_personalization:
            score += 10
        if has_curiosity:
            score += 8
        if subject.isupper():
            score -= 20
        if "!!" in subject or "???" in subject:
            score -= 10

        return {
            "subject": subject,
            "score": min(max(score, 0), 100),
            "length": length,
            "word_count": word_count,
            "has_number": has_number,
            "has_emoji": has_emoji,
            "has_personalization": has_personalization,
            "grade": "A" if score >= 80 else "B" if score >= 65 else "C" if score >= 50 else "D",
            "suggestions": self._suggestions(subject, score, length, has_personalization),
        }

    def _suggestions(self, subject: str, score: int, length: int, personalized: bool) -> list[str]:
        tips = []
        if length > 55:
            tips.append("Shorten to under 55 characters for mobile-friendly display.")
        if length < 25:
            tips.append("Consider adding more context — very short subjects can feel cryptic.")
        if not personalized:
            tips.append("Add a personalization token like {first_name} to boost open rates.")
        if not any(c.isdigit() for c in subject):
            tips.append("Include a number — e.g. '5 ways' or '3x faster' — to increase clicks.")
        if score < 60:
            tips.append("Try adding a curiosity hook like a question or 'here's why...'.")
        return tips

    def ab_variants(self, base_subject: str) -> list[dict]:
        variants = []
        words = base_subject.split()
        # Variant A: Add number
        variants.append(self.score(f"3 reasons why: {base_subject}"))
        # Variant B: Question form
        variants.append(self.score(f"Did you know? {base_subject}"))
        # Variant C: Personalized
        variants.append(self.score(f"{{first_name}}, {base_subject.lower()}"))
        # Variant D: Urgency
        variants.append(self.score(f"{base_subject} — limited time"))
        return sorted(variants, key=lambda x: x["score"], reverse=True)


# ---------------------------------------------------------------------------
# Enterprise demo extension
# ---------------------------------------------------------------------------

def _demo_research_extensions():
    print("\n" + "=" * 80)
    print("RESEARCH AGENT — ENTERPRISE EXTENSIONS DEMO")
    print("=" * 80)

    # Spam detection
    spam = SpamSignalDetector()
    result = spam.analyze(
        "FREE MONEY — GUARANTEED WINNER ACT NOW!!!",
        "Click here now to claim your cash bonus. This is not spam. 100% free.",
    )
    print(f"\n[SpamDetector] Score: {result['spam_score']} | Risk: {result['deliverability_risk']}")
    print(f"  Triggered: {result['triggered_words'][:3]}")

    # Behavioral segmentation
    seg = BehavioralSegmenter()
    profiles = [
        ("alice@co.com", 10, 4, 2, 5),
        ("bob@co.com", 3, 1, 15, 1),
        ("carol@co.com", 0, 0, 120, 0),
        ("dave@co.com", 6, 2, 5, 2),
    ]
    print("\n[BehavioralSegmenter]")
    for email, opens, clicks, days, purchases in profiles:
        segment = seg.classify_subscriber(email, opens, clicks, days, purchases)
        print(f"  {email} → {segment}")
    print(f"  Sizes: {seg.segment_sizes()}")

    # RFM
    rfm = RFMAnalyzer()
    print("\n[RFMAnalyzer]")
    for days, actions, value in [(5, 30, 500), (60, 5, 80), (180, 1, 10)]:
        score = rfm.score_subscriber(days, actions, value)
        print(f"  d={days}, a={actions}, v=${value} → {score['segment']} (RFM={score['rfm_score']})")

    # Workflow library
    welcome = WorkflowLibrary.welcome_series("Osprey AI")
    cart = WorkflowLibrary.cart_abandonment()
    context_signup = {"event": "signup"}
    context_cart = {"event": "cart_abandoned"}
    log = welcome.simulate_execution("test@example.com", context_signup)
    print(f"\n[WorkflowLibrary] Welcome series: {len(log)} steps queued")
    log2 = cart.simulate_execution("buyer@example.com", context_cart)
    print(f"  Cart abandonment: {len(log2)} steps queued")

    # Subject line optimizer
    opt = SubjectLineOptimizer()
    subjects = [
        "Newsletter #42",
        "Your exclusive 3-step guide to doubling revenue",
        "{first_name}, we have something special for you 🎁",
        "FREE FREE FREE — CLICK NOW — AMAZING OFFER!!!",
    ]
    print("\n[SubjectLineOptimizer]")
    for s in subjects:
        r = opt.score(s)
        print(f"  [{r['grade']}] {r['score']:3d}/100 — '{s[:60]}'")

    # Campaign report
    reporter = CampaignReportBuilder()
    rpt = reporter.build_report(
        "CAM-001", "Q1 Product Launch",
        sent=10000, delivered=9750, opened=2600, clicked=380,
        bounced=250, unsubscribed=45, revenue=12500.0,
    )
    print(f"\n[CampaignReport] Grade: {rpt['grade']} | "
          f"Open: {rpt['rates']['open_rate']}% | "
          f"Click: {rpt['rates']['click_rate']}% | "
          f"Revenue: ${rpt['revenue']['total']:,.2f}")
    bench = reporter.benchmark_comparison("SaaS")
    print(f"  vs SaaS benchmark: open {bench['vs_benchmark']['open_rate']:+.1f}% | "
          f"click {bench['vs_benchmark']['click_rate']:+.1f}%")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
    _demo_research_extensions()
