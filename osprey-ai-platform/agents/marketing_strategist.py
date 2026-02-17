"""
SALES INTELLIGENCE AI AGENT
Enterprise-grade intelligent sales automation and forecasting

Features:
- Lead scoring and qualification
- Sales forecasting and pipeline analysis
- CRM integration and management
- Deal intelligence and recommendations
- Competitor analysis
- Contact enrichment
- Activity tracking and analytics
- Win/loss analysis
- Territory management
- Sales coaching insights
- Churn prediction
- Revenue optimization

Dependencies:
- pandas
- numpy
- sklearn
- datetime
"""

import os
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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LeadSource(Enum):
    """Lead source channels"""
    WEBSITE = "website"
    REFERRAL = "referral"
    COLD_OUTREACH = "cold_outreach"
    INBOUND = "inbound"
    EVENT = "event"
    SOCIAL_MEDIA = "social_media"
    PARTNER = "partner"
    PAID_AD = "paid_ad"


class LeadStatus(Enum):
    """Lead qualification status"""
    NEW = "new"
    CONTACTED = "contacted"
    QUALIFIED = "qualified"
    UNQUALIFIED = "unqualified"
    NURTURING = "nurturing"
    CONVERTED = "converted"
    LOST = "lost"


class DealStage(Enum):
    """Sales pipeline stages"""
    PROSPECTING = "prospecting"
    QUALIFICATION = "qualification"
    NEEDS_ANALYSIS = "needs_analysis"
    PROPOSAL = "proposal"
    NEGOTIATION = "negotiation"
    CLOSED_WON = "closed_won"
    CLOSED_LOST = "closed_lost"


class CompanySize(Enum):
    """Company size segments"""
    STARTUP = "startup"  # 1-10
    SMALL = "small"  # 11-50
    MEDIUM = "medium"  # 51-200
    LARGE = "large"  # 201-1000
    ENTERPRISE = "enterprise"  # 1000+


class Industry(Enum):
    """Industry sectors"""
    TECHNOLOGY = "technology"
    HEALTHCARE = "healthcare"
    FINANCE = "finance"
    RETAIL = "retail"
    MANUFACTURING = "manufacturing"
    EDUCATION = "education"
    REAL_ESTATE = "real_estate"
    PROFESSIONAL_SERVICES = "professional_services"
    OTHER = "other"


@dataclass
class Contact:
    """Contact/Lead profile"""
    contact_id: str
    email: str
    first_name: str
    last_name: str
    title: Optional[str] = None
    company: Optional[str] = None
    phone: Optional[str] = None
    linkedin_url: Optional[str] = None
    location: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    last_contacted: Optional[datetime] = None
    engagement_score: float = 0.0
    tags: List[str] = field(default_factory=list)


@dataclass
class Lead:
    """Sales lead"""
    lead_id: str
    contact_id: str
    company_name: str
    source: LeadSource
    status: LeadStatus
    score: float
    created_at: datetime
    updated_at: datetime
    company_size: Optional[CompanySize] = None
    industry: Optional[Industry] = None
    estimated_revenue: float = 0.0
    assigned_to: Optional[str] = None
    next_action: Optional[str] = None
    notes: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Deal:
    """Sales opportunity/deal"""
    deal_id: str
    lead_id: str
    name: str
    stage: DealStage
    amount: float
    probability: float
    expected_close_date: datetime
    created_at: datetime
    updated_at: datetime
    closed_at: Optional[datetime] = None
    owner: str = "unassigned"
    products: List[str] = field(default_factory=list)
    competitors: List[str] = field(default_factory=list)
    close_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Activity:
    """Sales activity log"""
    activity_id: str
    contact_id: str
    activity_type: str  # call, email, meeting, demo
    subject: str
    description: str
    timestamp: datetime
    duration_minutes: Optional[int] = None
    outcome: Optional[str] = None
    created_by: str = "system"


@dataclass
class Forecast:
    """Sales forecast"""
    period: str  # e.g., "2025-Q1"
    total_pipeline: float
    weighted_pipeline: float
    expected_revenue: float
    best_case: float
    worst_case: float
    num_deals: int
    avg_deal_size: float
    win_rate: float
    confidence: float
    generated_at: datetime


@dataclass
class SalesMetrics:
    """Sales performance metrics"""
    period_start: datetime
    period_end: datetime
    total_leads: int
    qualified_leads: int
    converted_leads: int
    total_deals: int
    deals_won: int
    deals_lost: int
    total_revenue: float
    avg_deal_size: float
    avg_sales_cycle: float
    win_rate: float
    conversion_rate: float
    pipeline_value: float


class LeadScoringModel:
    """ML-based lead scoring"""
    
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Scoring weights for rule-based fallback
        self.weights = {
            'company_size': {
                CompanySize.ENTERPRISE: 25,
                CompanySize.LARGE: 20,
                CompanySize.MEDIUM: 15,
                CompanySize.SMALL: 10,
                CompanySize.STARTUP: 5
            },
            'source': {
                LeadSource.REFERRAL: 20,
                LeadSource.INBOUND: 15,
                LeadSource.EVENT: 12,
                LeadSource.WEBSITE: 10,
                LeadSource.SOCIAL_MEDIA: 8,
                LeadSource.PARTNER: 15,
                LeadSource.COLD_OUTREACH: 5,
                LeadSource.PAID_AD: 7
            },
            'engagement': {
                'high': 30,
                'medium': 15,
                'low': 5
            }
        }
    
    def train(self, training_data: List[Dict[str, Any]], labels: List[int]):
        """Train lead scoring model"""
        if len(training_data) < 10:
            logger.warning("Insufficient training data for ML model")
            return
        
        # Extract features
        X = self._extract_features(training_data)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, labels)
        self.is_trained = True
        
        logger.info(f"Lead scoring model trained on {len(training_data)} examples")
    
    def score_lead(self, lead: Lead, contact: Contact, 
                   engagement_data: Dict[str, Any] = None) -> float:
        """Score lead (0-100)"""
        if self.is_trained:
            return self._ml_score(lead, contact, engagement_data)
        else:
            return self._rule_based_score(lead, contact, engagement_data)
    
    def _ml_score(self, lead: Lead, contact: Contact,
                  engagement_data: Dict[str, Any]) -> float:
        """ML-based scoring"""
        features = self._extract_features([{
            'company_size': lead.company_size,
            'source': lead.source,
            'industry': lead.industry,
            'engagement_score': contact.engagement_score
        }])
        
        features_scaled = self.scaler.transform(features)
        probability = self.model.predict_proba(features_scaled)[0][1]
        
        return float(probability * 100)
    
    def _rule_based_score(self, lead: Lead, contact: Contact,
                         engagement_data: Dict[str, Any]) -> float:
        """Rule-based scoring fallback"""
        score = 0.0
        
        # Company size score
        if lead.company_size:
            score += self.weights['company_size'].get(lead.company_size, 5)
        
        # Source score
        score += self.weights['source'].get(lead.source, 5)
        
        # Engagement score
        if contact.engagement_score > 70:
            score += self.weights['engagement']['high']
        elif contact.engagement_score > 40:
            score += self.weights['engagement']['medium']
        else:
            score += self.weights['engagement']['low']
        
        # Title/seniority bonus
        if contact.title:
            title_lower = contact.title.lower()
            if any(word in title_lower for word in ['ceo', 'cto', 'founder', 'president', 'vp']):
                score += 15
            elif any(word in title_lower for word in ['director', 'head', 'manager']):
                score += 10
        
        # Revenue potential
        if lead.estimated_revenue > 100000:
            score += 10
        elif lead.estimated_revenue > 50000:
            score += 5
        
        return min(score, 100.0)
    
    def _extract_features(self, data: List[Dict[str, Any]]) -> np.ndarray:
        """Extract numerical features from lead data"""
        features = []
        
        for item in data:
            feature_vector = [
                # Company size (ordinal encoding)
                ['startup', 'small', 'medium', 'large', 'enterprise'].index(
                    item.get('company_size', CompanySize.SMALL).value if hasattr(item.get('company_size', CompanySize.SMALL), 'value') else 'small'
                ),
                # Source (ordinal encoding)
                list(LeadSource).index(item.get('source', LeadSource.WEBSITE)),
                # Engagement score
                item.get('engagement_score', 0.0)
            ]
            features.append(feature_vector)
        
        return np.array(features)


class DealIntelligence:
    """AI-powered deal insights and recommendations"""
    
    def __init__(self):
        self.win_probability_model = GradientBoostingRegressor(n_estimators=100)
        self.is_trained = False
    
    def analyze_deal(self, deal: Deal, activities: List[Activity],
                    historical_data: List[Deal] = None) -> Dict[str, Any]:
        """Comprehensive deal analysis"""
        analysis = {
            'deal_id': deal.deal_id,
            'health_score': 0.0,
            'win_probability': deal.probability,
            'risk_factors': [],
            'opportunities': [],
            'next_actions': [],
            'insights': []
        }
        
        # Calculate health score
        health_score = 50.0  # Base score
        
        # Stage progression
        stage_days = (datetime.now() - deal.updated_at).days
        expected_days = {
            DealStage.PROSPECTING: 7,
            DealStage.QUALIFICATION: 14,
            DealStage.NEEDS_ANALYSIS: 21,
            DealStage.PROPOSAL: 14,
            DealStage.NEGOTIATION: 21
        }
        
        if deal.stage in expected_days:
            if stage_days > expected_days[deal.stage]:
                health_score -= 15
                analysis['risk_factors'].append(f"Deal stalled in {deal.stage.value} stage")
            else:
                health_score += 10
        
        # Activity level
        recent_activities = [a for a in activities 
                           if a.timestamp > datetime.now() - timedelta(days=7)]
        
        if len(recent_activities) >= 3:
            health_score += 15
            analysis['insights'].append("High engagement - frequent touchpoints")
        elif len(recent_activities) == 0:
            health_score -= 20
            analysis['risk_factors'].append("No recent activity")
            analysis['next_actions'].append("Schedule follow-up call")
        
        # Close date proximity
        days_to_close = (deal.expected_close_date - datetime.now()).days
        if days_to_close < 0:
            health_score -= 25
            analysis['risk_factors'].append("Past expected close date")
        elif days_to_close < 7:
            analysis['opportunities'].append("Close imminent - push for commitment")
        
        # Competition
        if deal.competitors:
            health_score -= len(deal.competitors) * 5
            analysis['risk_factors'].append(f"Competing against {len(deal.competitors)} vendors")
            analysis['next_actions'].append("Prepare competitive battle card")
        
        # Deal size vs average
        # (Would compare with historical average)
        
        analysis['health_score'] = max(0, min(100, health_score))
        
        # Recommendations
        if analysis['health_score'] < 40:
            analysis['next_actions'].append("Escalate to sales manager")
        elif analysis['health_score'] > 70:
            analysis['insights'].append("Strong deal - maintain momentum")
        
        return analysis
    
    def predict_close_probability(self, deal: Deal, 
                                  activities: List[Activity]) -> float:
        """Predict deal close probability"""
        # Simplified prediction (would use ML model in production)
        base_probability = {
            DealStage.PROSPECTING: 10,
            DealStage.QUALIFICATION: 25,
            DealStage.NEEDS_ANALYSIS: 40,
            DealStage.PROPOSAL: 60,
            DealStage.NEGOTIATION: 75,
            DealStage.CLOSED_WON: 100,
            DealStage.CLOSED_LOST: 0
        }.get(deal.stage, 20)
        
        # Adjust based on activity
        activity_boost = min(len(activities) * 2, 15)
        
        # Adjust based on time in stage
        days_in_stage = (datetime.now() - deal.updated_at).days
        if days_in_stage > 30:
            time_penalty = 10
        else:
            time_penalty = 0
        
        probability = base_probability + activity_boost - time_penalty
        
        return max(0, min(100, probability)) / 100


class SalesForecasting:
    """Sales forecasting and pipeline analysis"""
    
    def __init__(self):
        self.historical_win_rates: Dict[DealStage, float] = {}
        self.seasonal_factors: Dict[int, float] = {}
    
    def generate_forecast(self, deals: List[Deal], 
                         period: str = "month") -> Forecast:
        """Generate sales forecast"""
        # Filter active deals
        active_deals = [d for d in deals if d.stage not in 
                       [DealStage.CLOSED_WON, DealStage.CLOSED_LOST]]
        
        if not active_deals:
            return Forecast(
                period=period,
                total_pipeline=0.0,
                weighted_pipeline=0.0,
                expected_revenue=0.0,
                best_case=0.0,
                worst_case=0.0,
                num_deals=0,
                avg_deal_size=0.0,
                win_rate=0.0,
                confidence=0.0,
                generated_at=datetime.now()
            )
        
        # Calculate pipeline metrics
        total_pipeline = sum(d.amount for d in active_deals)
        weighted_pipeline = sum(d.amount * d.probability for d in active_deals)
        
        # Stage-based weighted forecast
        stage_weights = {
            DealStage.PROSPECTING: 0.10,
            DealStage.QUALIFICATION: 0.25,
            DealStage.NEEDS_ANALYSIS: 0.40,
            DealStage.PROPOSAL: 0.60,
            DealStage.NEGOTIATION: 0.75
        }
        
        expected_revenue = sum(
            d.amount * stage_weights.get(d.stage, d.probability)
            for d in active_deals
        )
        
        # Best and worst case scenarios
        best_case = sum(
            d.amount for d in active_deals 
            if d.stage in [DealStage.PROPOSAL, DealStage.NEGOTIATION]
        )
        worst_case = sum(
            d.amount * 0.5 for d in active_deals
            if d.stage == DealStage.NEGOTIATION
        )
        
        # Win rate calculation (from historical data)
        win_rate = self._calculate_win_rate(deals)
        
        # Average deal size
        avg_deal_size = total_pipeline / len(active_deals) if active_deals else 0
        
        # Confidence score
        confidence = self._calculate_forecast_confidence(active_deals)
        
        return Forecast(
            period=period,
            total_pipeline=round(total_pipeline, 2),
            weighted_pipeline=round(weighted_pipeline, 2),
            expected_revenue=round(expected_revenue, 2),
            best_case=round(best_case, 2),
            worst_case=round(worst_case, 2),
            num_deals=len(active_deals),
            avg_deal_size=round(avg_deal_size, 2),
            win_rate=round(win_rate, 3),
            confidence=round(confidence, 2),
            generated_at=datetime.now()
        )
    
    def _calculate_win_rate(self, deals: List[Deal]) -> float:
        """Calculate historical win rate"""
        closed_deals = [d for d in deals if d.stage in 
                       [DealStage.CLOSED_WON, DealStage.CLOSED_LOST]]
        
        if not closed_deals:
            return 0.30  # Default assumption
        
        won = sum(1 for d in closed_deals if d.stage == DealStage.CLOSED_WON)
        
        return won / len(closed_deals)
    
    def _calculate_forecast_confidence(self, deals: List[Deal]) -> float:
        """Calculate forecast confidence score"""
        if not deals:
            return 0.0
        
        # More deals = higher confidence
        deal_count_factor = min(len(deals) / 20, 1.0) * 40
        
        # Stage distribution
        advanced_stage_deals = sum(
            1 for d in deals 
            if d.stage in [DealStage.PROPOSAL, DealStage.NEGOTIATION]
        )
        stage_factor = (advanced_stage_deals / len(deals)) * 30
        
        # Activity level (simplified)
        activity_factor = 30  # Would analyze actual activities
        
        confidence = deal_count_factor + stage_factor + activity_factor
        
        return min(confidence, 100)


class ContactEnrichment:
    """Enrich contact data from external sources"""
    
    def __init__(self):
        self.company_data_cache: Dict[str, Dict[str, Any]] = {}
    
    def enrich_contact(self, contact: Contact) -> Dict[str, Any]:
        """Enrich contact with additional data"""
        enrichment = {
            'contact_id': contact.contact_id,
            'enriched_fields': [],
            'company_data': {},
            'social_profiles': {}
        }
        
        # In production, integrate with:
        # - Clearbit
        # - ZoomInfo
        # - LinkedIn Sales Navigator
        # - Hunter.io
        
        # Simulate enrichment
        if contact.company and contact.company not in self.company_data_cache:
            enrichment['company_data'] = {
                'size': random.choice(list(CompanySize)).value,
                'industry': random.choice(list(Industry)).value,
                'employee_count': random.randint(10, 5000),
                'estimated_revenue': random.randint(1000000, 100000000),
                'technologies': ['Salesforce', 'HubSpot', 'Slack']
            }
            
            self.company_data_cache[contact.company] = enrichment['company_data']
            enrichment['enriched_fields'].append('company_data')
        
        # LinkedIn profile simulation
        if contact.linkedin_url:
            enrichment['social_profiles']['linkedin'] = {
                'connections': random.randint(100, 5000),
                'followers': random.randint(50, 2000)
            }
            enrichment['enriched_fields'].append('linkedin_data')
        
        return enrichment
    
    def validate_email(self, email: str) -> Dict[str, Any]:
        """Validate email deliverability"""
        # In production, use services like:
        # - NeverBounce
        # - ZeroBounce
        # - Kickbox
        
        # Simple validation
        import re
        email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        
        is_valid = bool(re.match(email_regex, email))
        
        return {
            'email': email,
            'valid': is_valid,
            'deliverable': is_valid,
            'role_account': any(prefix in email for prefix in ['info@', 'support@', 'sales@']),
            'disposable': False  # Would check against known disposable domains
        }


class SalesIntelligenceAI:
    """
    Main Sales Intelligence AI Agent
    Enterprise-grade intelligent sales automation and forecasting
    """
    
    def __init__(self):
        """Initialize Sales Intelligence AI"""
        self.lead_scoring = LeadScoringModel()
        self.deal_intelligence = DealIntelligence()
        self.forecasting = SalesForecasting()
        self.enrichment = ContactEnrichment()
        
        # Storage
        self.contacts: Dict[str, Contact] = {}
        self.leads: Dict[str, Lead] = {}
        self.deals: Dict[str, Deal] = {}
        self.activities: Dict[str, List[Activity]] = defaultdict(list)
        
        # Analytics
        self.total_leads_created = 0
        self.total_deals_created = 0
        self.total_revenue = 0.0
        
        logger.info("Sales Intelligence AI initialized successfully")
    
    def add_contact(self, contact: Contact) -> Contact:
        """Add or update contact"""
        self.contacts[contact.contact_id] = contact
        
        # Auto-enrich
        enrichment = self.enrichment.enrich_contact(contact)
        
        logger.info(f"Contact added: {contact.email}")
        return contact
    
    def create_lead(self, contact_id: str, company_name: str,
                   source: LeadSource, **kwargs) -> Lead:
        """Create new lead"""
        if contact_id not in self.contacts:
            raise ValueError(f"Contact not found: {contact_id}")
        
        contact = self.contacts[contact_id]
        lead_id = self._generate_lead_id()
        
        # Create lead
        lead = Lead(
            lead_id=lead_id,
            contact_id=contact_id,
            company_name=company_name,
            source=source,
            status=LeadStatus.NEW,
            score=0.0,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            **kwargs
        )
        
        # Score lead
        lead.score = self.lead_scoring.score_lead(lead, contact)
        
        # Auto-qualify if high score
        if lead.score >= 70:
            lead.status = LeadStatus.QUALIFIED
            lead.next_action = "Schedule discovery call"
        elif lead.score >= 40:
            lead.status = LeadStatus.CONTACTED
            lead.next_action = "Send introductory email"
        else:
            lead.status = LeadStatus.NURTURING
            lead.next_action = "Add to nurture campaign"
        
        self.leads[lead_id] = lead
        self.total_leads_created += 1
        
        logger.info(f"Lead created: {company_name} (Score: {lead.score})")
        
        return lead
    
    def create_deal(self, lead_id: str, name: str, amount: float,
                   expected_close_date: datetime, **kwargs) -> Deal:
        """Create sales opportunity"""
        if lead_id not in self.leads:
            raise ValueError(f"Lead not found: {lead_id}")
        
        deal_id = self._generate_deal_id()
        
        deal = Deal(
            deal_id=deal_id,
            lead_id=lead_id,
            name=name,
            stage=DealStage.PROSPECTING,
            amount=amount,
            probability=0.10,  # Initial probability
            expected_close_date=expected_close_date,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            **kwargs
        )
        
        self.deals[deal_id] = deal
        self.total_deals_created += 1
        
        # Update lead status
        self.leads[lead_id].status = LeadStatus.CONVERTED
        
        logger.info(f"Deal created: {name} (${amount:,.2f})")
        
        return deal
    
    def log_activity(self, contact_id: str, activity_type: str,
                    subject: str, description: str, **kwargs) -> Activity:
        """Log sales activity"""
        activity_id = f"ACT-{hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]}"
        
        activity = Activity(
            activity_id=activity_id,
            contact_id=contact_id,
            activity_type=activity_type,
            subject=subject,
            description=description,
            timestamp=datetime.now(),
            **kwargs
        )
        
        self.activities[contact_id].append(activity)
        
        # Update contact engagement
        if contact_id in self.contacts:
            contact = self.contacts[contact_id]
            contact.last_contacted = datetime.now()
            contact.engagement_score = self._calculate_engagement_score(
                self.activities[contact_id]
            )
        
        logger.debug(f"Activity logged: {activity_type} - {subject}")
        
        return activity
    
    def update_deal_stage(self, deal_id: str, new_stage: DealStage,
                         notes: str = "") -> Deal:
        """Update deal stage"""
        if deal_id not in self.deals:
            raise ValueError(f"Deal not found: {deal_id}")
        
        deal = self.deals[deal_id]
        old_stage = deal.stage
        
        deal.stage = new_stage
        deal.updated_at = datetime.now()
        
        # Update probability based on stage
        stage_probabilities = {
            DealStage.PROSPECTING: 0.10,
            DealStage.QUALIFICATION: 0.25,
            DealStage.NEEDS_ANALYSIS: 0.40,
            DealStage.PROPOSAL: 0.60,
            DealStage.NEGOTIATION: 0.75,
            DealStage.CLOSED_WON: 1.00,
            DealStage.CLOSED_LOST: 0.00
        }
        
        deal.probability = stage_probabilities.get(new_stage, deal.probability)
        
        # Handle closed deals
        if new_stage == DealStage.CLOSED_WON:
            deal.closed_at = datetime.now()
            self.total_revenue += deal.amount
            logger.info(f"Deal WON: {deal.name} (${deal.amount:,.2f})")
        
        elif new_stage == DealStage.CLOSED_LOST:
            deal.closed_at = datetime.now()
            logger.info(f"Deal LOST: {deal.name}")
        
        logger.info(f"Deal stage updated: {old_stage.value} → {new_stage.value}")
        
        return deal
    
    def analyze_deal(self, deal_id: str) -> Dict[str, Any]:
        """Get AI-powered deal analysis"""
        if deal_id not in self.deals:
            return {'error': 'Deal not found'}
        
        deal = self.deals[deal_id]
        lead = self.leads.get(deal.lead_id)
        
        if not lead:
            return {'error': 'Associated lead not found'}
        
        contact_activities = self.activities.get(lead.contact_id, [])
        
        # Get deal intelligence
        analysis = self.deal_intelligence.analyze_deal(deal, contact_activities)
        
        # Add win probability prediction
        win_prob = self.deal_intelligence.predict_close_probability(deal, contact_activities)
        analysis['predicted_win_probability'] = round(win_prob, 3)
        
        return analysis
    
    def get_pipeline_forecast(self, period: str = "month") -> Forecast:
        """Generate sales forecast"""
        all_deals = list(self.deals.values())
        return self.forecasting.generate_forecast(all_deals, period)
    
    def get_top_leads(self, limit: int = 10, 
                     min_score: float = 50.0) -> List[Lead]:
        """Get highest-scoring leads"""
        qualified_leads = [
            lead for lead in self.leads.values()
            if lead.status != LeadStatus.CONVERTED and lead.score >= min_score
        ]
        
        # Sort by score descending
        sorted_leads = sorted(qualified_leads, key=lambda l: l.score, reverse=True)
        
        return sorted_leads[:limit]
    
    def get_deals_at_risk(self) -> List[Tuple[Deal, Dict[str, Any]]]:
        """Identify deals at risk"""
        at_risk = []
        
        for deal in self.deals.values():
            if deal.stage in [DealStage.CLOSED_WON, DealStage.CLOSED_LOST]:
                continue
            
            analysis = self.analyze_deal(deal.deal_id)
            
            if analysis.get('health_score', 100) < 50:
                at_risk.append((deal, analysis))
        
        # Sort by health score ascending
        at_risk.sort(key=lambda x: x[1].get('health_score', 0))
        
        return at_risk
    
    def get_sales_metrics(self, start_date: datetime = None,
                         end_date: datetime = None) -> SalesMetrics:
        """Calculate sales performance metrics"""
        if not start_date:
            start_date = datetime.now() - timedelta(days=30)
        if not end_date:
            end_date = datetime.now()
        
        # Filter by date range
        period_leads = [
            l for l in self.leads.values()
            if start_date <= l.created_at <= end_date
        ]
        
        period_deals = [
            d for d in self.deals.values()
            if start_date <= d.created_at <= end_date
        ]
        
        # Calculate metrics
        total_leads = len(period_leads)
        qualified_leads = sum(1 for l in period_leads if l.status == LeadStatus.QUALIFIED)
        converted_leads = sum(1 for l in period_leads if l.status == LeadStatus.CONVERTED)
        
        total_deals = len(period_deals)
        deals_won = sum(1 for d in period_deals if d.stage == DealStage.CLOSED_WON)
        deals_lost = sum(1 for d in period_deals if d.stage == DealStage.CLOSED_LOST)
        
        won_deals = [d for d in period_deals if d.stage == DealStage.CLOSED_WON]
        total_revenue = sum(d.amount for d in won_deals)
        avg_deal_size = total_revenue / max(deals_won, 1)
        
        # Sales cycle calculation
        sales_cycles = []
        for deal in won_deals:
            if deal.closed_at:
                cycle_days = (deal.closed_at - deal.created_at).days
                sales_cycles.append(cycle_days)
        
        avg_sales_cycle = sum(sales_cycles) / len(sales_cycles) if sales_cycles else 0
        
        # Rates
        win_rate = deals_won / max(deals_won + deals_lost, 1)
        conversion_rate = converted_leads / max(total_leads, 1)
        
        # Pipeline value
        active_deals = [d for d in self.deals.values() 
                       if d.stage not in [DealStage.CLOSED_WON, DealStage.CLOSED_LOST]]
        pipeline_value = sum(d.amount for d in active_deals)
        
        return SalesMetrics(
            period_start=start_date,
            period_end=end_date,
            total_leads=total_leads,
            qualified_leads=qualified_leads,
            converted_leads=converted_leads,
            total_deals=total_deals,
            deals_won=deals_won,
            deals_lost=deals_lost,
            total_revenue=round(total_revenue, 2),
            avg_deal_size=round(avg_deal_size, 2),
            avg_sales_cycle=round(avg_sales_cycle, 1),
            win_rate=round(win_rate, 3),
            conversion_rate=round(conversion_rate, 3),
            pipeline_value=round(pipeline_value, 2)
        )
    
    def _calculate_engagement_score(self, activities: List[Activity]) -> float:
        """Calculate contact engagement score"""
        if not activities:
            return 0.0
        
        # Recent activity score
        recent = [a for a in activities if a.timestamp > datetime.now() - timedelta(days=30)]
        recency_score = min(len(recent) * 10, 40)
        
        # Activity type diversity
        types = set(a.activity_type for a in activities)
        diversity_score = min(len(types) * 15, 30)
        
        # Positive outcomes
        positive = sum(1 for a in activities if a.outcome and 'positive' in a.outcome.lower())
        outcome_score = min(positive * 10, 30)
        
        total = recency_score + diversity_score + outcome_score
        
        return min(total, 100.0)
    
    def _generate_lead_id(self) -> str:
        """Generate unique lead ID"""
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        suffix = hashlib.md5(str(datetime.now()).encode()).hexdigest()[:6]
        return f"LEAD-{timestamp}-{suffix}"
    
    def _generate_deal_id(self) -> str:
        """Generate unique deal ID"""
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        suffix = hashlib.md5(str(datetime.now()).encode()).hexdigest()[:6]
        return f"DEAL-{timestamp}-{suffix}"
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get overall sales statistics"""
        active_deals = [d for d in self.deals.values() 
                       if d.stage not in [DealStage.CLOSED_WON, DealStage.CLOSED_LOST]]
        
        return {
            'total_contacts': len(self.contacts),
            'total_leads': self.total_leads_created,
            'active_leads': len([l for l in self.leads.values() 
                               if l.status != LeadStatus.CONVERTED]),
            'total_deals': self.total_deals_created,
            'active_deals': len(active_deals),
            'total_revenue': round(self.total_revenue, 2),
            'pipeline_value': round(sum(d.amount for d in active_deals), 2),
            'weighted_pipeline': round(sum(d.amount * d.probability for d in active_deals), 2)
        }


# Example usage
if __name__ == "__main__":
    # Initialize AI
    ai = SalesIntelligenceAI()
    
    print("=" * 80)
    print("SALES INTELLIGENCE AI DEMO")
    print("=" * 80)
    
    # Add contacts
    contact1 = Contact(
        contact_id="CNT-001",
        email="john.doe@acmecorp.com",
        first_name="John",
        last_name="Doe",
        title="VP of Engineering",
        company="Acme Corp"
    )
    ai.add_contact(contact1)
    
    contact2 = Contact(
        contact_id="CNT-002",
        email="jane.smith@techstart.io",
        first_name="Jane",
        last_name="Smith",
        title="CTO",
        company="TechStart"
    )
    ai.add_contact(contact2)
    
    print(f"\nContacts added: {len(ai.contacts)}")
    
    # Create leads
    lead1 = ai.create_lead(
        contact_id="CNT-001",
        company_name="Acme Corp",
        source=LeadSource.REFERRAL,
        company_size=CompanySize.LARGE,
        industry=Industry.TECHNOLOGY,
        estimated_revenue=150000
    )
    
    print(f"\nLead created: {lead1.company_name}")
    print(f"Lead Score: {lead1.score}")
    print(f"Status: {lead1.status.value}")
    print(f"Next Action: {lead1.next_action}")
    
    # Log activities
    ai.log_activity(
        contact_id="CNT-001",
        activity_type="call",
        subject="Discovery Call",
        description="Discussed requirements and timeline",
        duration_minutes=45,
        outcome="Positive - scheduling demo"
    )
    
    ai.log_activity(
        contact_id="CNT-001",
        activity_type="demo",
        subject="Product Demo",
        description="Demonstrated platform features",
        duration_minutes=60,
        outcome="Very positive - ready for proposal"
    )
    
    # Create deal
    deal = ai.create_deal(
        lead_id=lead1.lead_id,
        name="Acme Corp - Enterprise License",
        amount=150000,
        expected_close_date=datetime.now() + timedelta(days=30),
        owner="rep_001",
        products=["Enterprise Plan", "Premium Support"]
    )
    
    print(f"\nDeal created: {deal.name}")
    print(f"Amount: ${deal.amount:,.2f}")
    print(f"Probability: {deal.probability * 100}%")
    
    # Analyze deal
    analysis = ai.analyze_deal(deal.deal_id)
    
    print(f"\nDeal Analysis:")
    print(f"Health Score: {analysis['health_score']}")
    print(f"Win Probability: {analysis['predicted_win_probability'] * 100:.1f}%")
    print(f"Insights: {analysis['insights']}")
    print(f"Next Actions: {analysis['next_actions']}")
    
    # Move deal forward
    ai.update_deal_stage(deal.deal_id, DealStage.PROPOSAL)
    ai.update_deal_stage(deal.deal_id, DealStage.NEGOTIATION)
    
    # Generate forecast
    forecast = ai.get_pipeline_forecast()
    
    print(f"\n{'=' * 80}")
    print("SALES FORECAST")
    print("=" * 80)
    print(f"Total Pipeline: ${forecast.total_pipeline:,.2f}")
    print(f"Weighted Pipeline: ${forecast.weighted_pipeline:,.2f}")
    print(f"Expected Revenue: ${forecast.expected_revenue:,.2f}")
    print(f"Best Case: ${forecast.best_case:,.2f}")
    print(f"Worst Case: ${forecast.worst_case:,.2f}")
    print(f"Win Rate: {forecast.win_rate * 100:.1f}%")
    print(f"Confidence: {forecast.confidence:.1f}%")
    
    # Sales metrics
    metrics = ai.get_sales_metrics()
    
    print(f"\n{'=' * 80}")
    print("SALES METRICS")
    print("=" * 80)
    print(f"Total Leads: {metrics.total_leads}")
    print(f"Conversion Rate: {metrics.conversion_rate * 100:.1f}%")
    print(f"Total Revenue: ${metrics.total_revenue:,.2f}")
    print(f"Win Rate: {metrics.win_rate * 100:.1f}%")
    print(f"Avg Deal Size: ${metrics.avg_deal_size:,.2f}")
    print(f"Avg Sales Cycle: {metrics.avg_sales_cycle:.0f} days")
    print(f"Pipeline Value: ${metrics.pipeline_value:,.2f}")
    
    # Statistics
    stats = ai.get_statistics()
    print(f"\n{'=' * 80}")
    print("OVERALL STATISTICS")
    print("=" * 80)
    print(json.dumps(stats, indent=2))


# ===========================================================================
# ENTERPRISE EXTENSIONS — Sales Intelligence & Revenue Operations
# ===========================================================================

import math
import csv
import io
from collections import defaultdict, Counter
from typing import Iterator


# ---------------------------------------------------------------------------
# Competitive Intelligence Engine
# ---------------------------------------------------------------------------

class CompetitorProfile:
    def __init__(self, name: str, market_share: float, strengths: List[str], weaknesses: List[str]):
        self.name = name
        self.market_share = market_share
        self.strengths = strengths
        self.weaknesses = weaknesses
        self.win_rate_against: float = 0.0
        self.loss_rate_against: float = 0.0
        self._encounters: List[dict] = []

    def record_encounter(self, deal_id: str, outcome: str, notes: str = "") -> None:
        self._encounters.append({
            "deal_id": deal_id,
            "outcome": outcome,
            "notes": notes,
            "timestamp": datetime.now().isoformat(),
        })
        wins = sum(1 for e in self._encounters if e["outcome"] == "win")
        total = len(self._encounters)
        self.win_rate_against = round(wins / total, 3) if total else 0.0
        self.loss_rate_against = round(1 - self.win_rate_against, 3)

    def battle_card(self) -> str:
        lines = [
            f"BATTLE CARD: {self.name}",
            f"Market Share: {self.market_share:.1f}%",
            f"Our Win Rate vs Them: {self.win_rate_against * 100:.1f}%",
            "Strengths to Counter:",
            *[f"  - {s}" for s in self.strengths],
            "Weaknesses to Exploit:",
            *[f"  - {w}" for w in self.weaknesses],
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "market_share": self.market_share,
            "win_rate_against": self.win_rate_against,
            "encounters": len(self._encounters),
        }


class CompetitorTracker:
    def __init__(self):
        self._competitors: Dict[str, CompetitorProfile] = {}

    def add_competitor(self, profile: CompetitorProfile) -> None:
        self._competitors[profile.name.lower()] = profile

    def get(self, name: str) -> Optional[CompetitorProfile]:
        return self._competitors.get(name.lower())

    def rank_by_threat(self) -> List[CompetitorProfile]:
        return sorted(
            self._competitors.values(),
            key=lambda c: c.market_share * c.loss_rate_against,
            reverse=True,
        )

    def competitive_landscape(self) -> List[dict]:
        return [c.to_dict() for c in self.rank_by_threat()]


# ---------------------------------------------------------------------------
# Territory Management
# ---------------------------------------------------------------------------

class Territory:
    def __init__(self, territory_id: str, name: str, region: str, quota: float):
        self.territory_id = territory_id
        self.name = name
        self.region = region
        self.quota = quota
        self.achieved: float = 0.0
        self.rep_ids: List[str] = []
        self._accounts: List[dict] = []

    def assign_rep(self, rep_id: str) -> None:
        if rep_id not in self.rep_ids:
            self.rep_ids.append(rep_id)

    def add_account(self, account_id: str, potential: float) -> None:
        self._accounts.append({"account_id": account_id, "potential": potential, "closed": 0.0})

    def record_close(self, account_id: str, amount: float) -> None:
        for acc in self._accounts:
            if acc["account_id"] == account_id:
                acc["closed"] += amount
                self.achieved += amount
                return

    @property
    def attainment_pct(self) -> float:
        return round(self.achieved / self.quota * 100, 1) if self.quota else 0.0

    def summary(self) -> dict:
        return {
            "territory_id": self.territory_id,
            "name": self.name,
            "region": self.region,
            "quota": self.quota,
            "achieved": self.achieved,
            "attainment_pct": self.attainment_pct,
            "reps": self.rep_ids,
            "total_accounts": len(self._accounts),
        }


class TerritoryManager:
    def __init__(self):
        self._territories: Dict[str, Territory] = {}

    def create(self, territory_id: str, name: str, region: str, quota: float) -> Territory:
        t = Territory(territory_id, name, region, quota)
        self._territories[territory_id] = t
        return t

    def get(self, territory_id: str) -> Optional[Territory]:
        return self._territories.get(territory_id)

    def region_rollup(self) -> Dict[str, dict]:
        rollup: Dict[str, dict] = defaultdict(lambda: {"quota": 0.0, "achieved": 0.0, "count": 0})
        for t in self._territories.values():
            r = rollup[t.region]
            r["quota"] += t.quota
            r["achieved"] += t.achieved
            r["count"] += 1
        return {
            region: {
                **data,
                "attainment_pct": round(data["achieved"] / data["quota"] * 100, 1)
                if data["quota"] else 0.0,
            }
            for region, data in rollup.items()
        }


# ---------------------------------------------------------------------------
# Sales Coaching Engine
# ---------------------------------------------------------------------------

class RepPerformanceProfile:
    def __init__(self, rep_id: str, name: str):
        self.rep_id = rep_id
        self.name = name
        self._activities: List[dict] = []
        self._deals: List[dict] = []

    def log_activity(self, activity_type: str, outcome: str, notes: str = "") -> None:
        self._activities.append({
            "type": activity_type,
            "outcome": outcome,
            "notes": notes,
            "ts": datetime.now().isoformat(),
        })

    def log_deal(self, deal_id: str, stage: str, value: float, won: Optional[bool] = None) -> None:
        self._deals.append({
            "deal_id": deal_id,
            "stage": stage,
            "value": value,
            "won": won,
        })

    def metrics(self) -> dict:
        closed = [d for d in self._deals if d["won"] is not None]
        wins = [d for d in closed if d["won"]]
        win_rate = round(len(wins) / len(closed), 3) if closed else 0.0
        avg_deal = round(sum(d["value"] for d in wins) / len(wins), 2) if wins else 0.0
        calls = [a for a in self._activities if a["type"] == "call"]
        emails = [a for a in self._activities if a["type"] == "email"]
        meetings = [a for a in self._activities if a["type"] == "meeting"]
        return {
            "rep_id": self.rep_id,
            "name": self.name,
            "total_deals": len(self._deals),
            "closed_deals": len(closed),
            "win_rate": win_rate,
            "avg_deal_size": avg_deal,
            "total_revenue": sum(d["value"] for d in wins),
            "activities": {
                "calls": len(calls),
                "emails": len(emails),
                "meetings": len(meetings),
            },
        }


class SalesCoach:
    """Generate personalized coaching recommendations for sales reps."""

    BENCHMARKS = {
        "win_rate": 0.25,
        "avg_deal_size": 5000.0,
        "calls_per_week": 20,
        "emails_per_week": 40,
        "meetings_per_week": 5,
    }

    def coach(self, profile: RepPerformanceProfile) -> List[str]:
        m = profile.metrics()
        recommendations: List[str] = []

        if m["win_rate"] < self.BENCHMARKS["win_rate"]:
            gap = round((self.BENCHMARKS["win_rate"] - m["win_rate"]) * 100, 1)
            recommendations.append(
                f"Win rate {m['win_rate']*100:.1f}% is {gap}pp below benchmark. "
                "Focus on qualification — try MEDDIC framework in discovery calls."
            )
        if m["avg_deal_size"] < self.BENCHMARKS["avg_deal_size"]:
            recommendations.append(
                f"Average deal size ${m['avg_deal_size']:,.0f} is below target. "
                "Practice multi-threading — engage champions AND economic buyers."
            )
        acts = m["activities"]
        if acts["calls"] < self.BENCHMARKS["calls_per_week"]:
            recommendations.append(
                f"Only {acts['calls']} calls logged. Increase outbound call cadence — "
                "aim for {self.BENCHMARKS['calls_per_week']}+ calls/week."
            )
        if acts["meetings"] < self.BENCHMARKS["meetings_per_week"]:
            recommendations.append(
                "Meeting count is low. Improve email-to-meeting conversion with "
                "personalized subject lines and value-first openers."
            )
        if not recommendations:
            recommendations.append(
                f"Excellent performance, {profile.name}! Maintain momentum and "
                "consider mentoring newer team members."
            )
        return recommendations


# ---------------------------------------------------------------------------
# Deal Risk Scorer
# ---------------------------------------------------------------------------

class DealRiskScorer:
    """Score deals on risk of slippage or loss."""

    def score(
        self,
        days_in_stage: int,
        avg_days_for_stage: int,
        last_activity_days_ago: int,
        stakeholder_count: int,
        has_legal_review: bool,
        champion_engaged: bool,
        budget_confirmed: bool,
        competition_present: bool,
    ) -> dict:
        risk = 0

        # Stage overdue
        overdue_ratio = days_in_stage / max(avg_days_for_stage, 1)
        if overdue_ratio > 2.0:
            risk += 30
        elif overdue_ratio > 1.5:
            risk += 15
        elif overdue_ratio > 1.2:
            risk += 8

        # Activity gap
        if last_activity_days_ago > 14:
            risk += 25
        elif last_activity_days_ago > 7:
            risk += 12

        # Stakeholder coverage
        if stakeholder_count < 2:
            risk += 20
        elif stakeholder_count < 3:
            risk += 10

        # Qualification flags
        if not champion_engaged:
            risk += 15
        if not budget_confirmed:
            risk += 12
        if competition_present:
            risk += 10
        if has_legal_review:
            risk -= 5  # positive signal — they're serious

        risk = min(max(risk, 0), 100)
        return {
            "risk_score": risk,
            "risk_level": "CRITICAL" if risk >= 70 else "HIGH" if risk >= 50 else "MEDIUM" if risk >= 30 else "LOW",
            "flags": {
                "stage_overdue": overdue_ratio > 1.2,
                "stale_activity": last_activity_days_ago > 7,
                "single_threaded": stakeholder_count < 2,
                "no_champion": not champion_engaged,
                "no_budget": not budget_confirmed,
                "competitive": competition_present,
            },
            "recommendations": self._recommend(risk, last_activity_days_ago, champion_engaged, budget_confirmed),
        }

    def _recommend(self, risk: int, last_act: int, champion: bool, budget: bool) -> List[str]:
        tips = []
        if last_act > 7:
            tips.append("Re-engage — send a value-add touchpoint or request a check-in call.")
        if not champion:
            tips.append("Identify and nurture an internal champion to drive the deal forward.")
        if not budget:
            tips.append("Confirm budget authority — ask for a direct conversation with the CFO/VP.")
        if risk >= 70:
            tips.append("CRITICAL: Escalate to manager for deal review and rescue strategy.")
        return tips


# ---------------------------------------------------------------------------
# Revenue Forecasting Engine
# ---------------------------------------------------------------------------

class RevenueForecastEngine:
    """Build weighted pipeline forecasts using multiple methods."""

    STAGE_WEIGHTS = {
        "prospecting": 0.05,
        "qualification": 0.10,
        "discovery": 0.20,
        "proposal": 0.40,
        "negotiation": 0.65,
        "contract_sent": 0.85,
        "closed_won": 1.00,
        "closed_lost": 0.00,
    }

    def __init__(self):
        self._pipeline: List[dict] = []

    def add_deal(self, deal_id: str, value: float, stage: str, close_date: str, rep_id: str = "") -> None:
        weight = self.STAGE_WEIGHTS.get(stage.lower(), 0.10)
        self._pipeline.append({
            "deal_id": deal_id,
            "value": value,
            "stage": stage,
            "weight": weight,
            "weighted_value": round(value * weight, 2),
            "close_date": close_date,
            "rep_id": rep_id,
        })

    def commit_forecast(self) -> dict:
        """Deals with weight >= 0.65 (negotiation+)."""
        committed = [d for d in self._pipeline if d["weight"] >= 0.65]
        return {
            "method": "commit",
            "deal_count": len(committed),
            "total": round(sum(d["value"] for d in committed), 2),
        }

    def best_case_forecast(self) -> dict:
        """Deals with weight >= 0.20."""
        best = [d for d in self._pipeline if d["weight"] >= 0.20]
        return {
            "method": "best_case",
            "deal_count": len(best),
            "total": round(sum(d["value"] for d in best), 2),
        }

    def weighted_forecast(self) -> dict:
        return {
            "method": "weighted",
            "deal_count": len(self._pipeline),
            "total": round(sum(d["weighted_value"] for d in self._pipeline), 2),
        }

    def by_rep(self) -> Dict[str, dict]:
        rep_map: Dict[str, dict] = defaultdict(lambda: {"deals": 0, "weighted": 0.0, "pipeline": 0.0})
        for d in self._pipeline:
            rep = d["rep_id"] or "unassigned"
            rep_map[rep]["deals"] += 1
            rep_map[rep]["weighted"] += d["weighted_value"]
            rep_map[rep]["pipeline"] += d["value"]
        return {
            rep: {
                **vals,
                "weighted": round(vals["weighted"], 2),
                "pipeline": round(vals["pipeline"], 2),
            }
            for rep, vals in rep_map.items()
        }

    def export_pipeline_csv(self) -> str:
        if not self._pipeline:
            return "No pipeline data."
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=["deal_id", "value", "stage", "weight", "weighted_value", "close_date", "rep_id"])
        writer.writeheader()
        writer.writerows(self._pipeline)
        return buf.getvalue()


# ---------------------------------------------------------------------------
# Enterprise demo extension
# ---------------------------------------------------------------------------

def _demo_sales_extensions():
    print("\n" + "=" * 80)
    print("MARKETING / SALES AGENT — ENTERPRISE EXTENSIONS DEMO")
    print("=" * 80)

    # Competitor tracking
    tracker = CompetitorTracker()
    comp1 = CompetitorProfile("SalesForce", 32.5, ["Brand recognition", "App ecosystem"], ["Price", "Complexity"])
    comp2 = CompetitorProfile("HubSpot", 18.2, ["Free tier", "Ease of use"], ["Enterprise scale", "Reporting"])
    comp1.record_encounter("D001", "win", "Sold on TCO")
    comp1.record_encounter("D002", "loss", "Lost on brand")
    comp1.record_encounter("D003", "win", "Sold on integration speed")
    comp2.record_encounter("D004", "win", "Sold on enterprise features")
    tracker.add_competitor(comp1)
    tracker.add_competitor(comp2)
    print("\n[CompetitorTracker] Ranked by threat:")
    for c in tracker.rank_by_threat():
        print(f"  {c['name']:15s} | Share: {c['market_share']}% | Win rate: {c['win_rate_against']*100:.0f}%")

    # Territory management
    tm = TerritoryManager()
    east = tm.create("T-EAST", "East Coast", "North America", 500000)
    west = tm.create("T-WEST", "West Coast", "North America", 750000)
    emea = tm.create("T-EMEA", "Europe", "EMEA", 600000)
    east.assign_rep("REP-001")
    east.assign_rep("REP-002")
    west.assign_rep("REP-003")
    emea.assign_rep("REP-004")
    east.record_close("ACC-001", 85000)
    west.record_close("ACC-002", 250000)
    emea.record_close("ACC-003", 180000)
    print("\n[TerritoryManager] Region rollup:")
    for region, data in tm.region_rollup().items():
        print(f"  {region}: ${data['achieved']:,.0f} / ${data['quota']:,.0f} ({data['attainment_pct']}%)")

    # Sales coaching
    rep = RepPerformanceProfile("REP-001", "Alex Johnson")
    rep.log_deal("D001", "closed_won", 15000, True)
    rep.log_deal("D002", "closed_lost", 8000, False)
    rep.log_deal("D003", "closed_lost", 12000, False)
    rep.log_deal("D004", "proposal", 20000)
    for _ in range(8):
        rep.log_activity("call", "voicemail")
    for _ in range(15):
        rep.log_activity("email", "opened")
    for _ in range(2):
        rep.log_activity("meeting", "completed")
    coach = SalesCoach()
    tips = coach.coach(rep)
    print(f"\n[SalesCoach] Coaching for {rep.name}:")
    for tip in tips:
        print(f"  • {tip[:90]}")

    # Deal risk
    risk_scorer = DealRiskScorer()
    deal_risk = risk_scorer.score(
        days_in_stage=45, avg_days_for_stage=20, last_activity_days_ago=12,
        stakeholder_count=1, has_legal_review=False, champion_engaged=False,
        budget_confirmed=False, competition_present=True,
    )
    print(f"\n[DealRiskScorer] Score: {deal_risk['risk_score']} | Level: {deal_risk['risk_level']}")
    for rec in deal_risk["recommendations"][:2]:
        print(f"  • {rec[:85]}")

    # Forecast
    forecast = RevenueForecastEngine()
    deals = [
        ("D001", 50000, "negotiation", "2026-02-28", "REP-001"),
        ("D002", 30000, "proposal", "2026-03-15", "REP-002"),
        ("D003", 120000, "contract_sent", "2026-02-20", "REP-001"),
        ("D004", 80000, "discovery", "2026-04-01", "REP-003"),
        ("D005", 15000, "prospecting", "2026-05-01", "REP-002"),
    ]
    for d in deals:
        forecast.add_deal(*d)
    print(f"\n[RevenueForecast]")
    print(f"  Commit:    ${forecast.commit_forecast()['total']:>12,.2f}")
    print(f"  Best Case: ${forecast.best_case_forecast()['total']:>12,.2f}")
    print(f"  Weighted:  ${forecast.weighted_forecast()['total']:>12,.2f}")
    print(f"  By Rep:")
    for rep_id, data in forecast.by_rep().items():
        print(f"    {rep_id}: weighted ${data['weighted']:,.0f} / pipeline ${data['pipeline']:,.0f}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
    _demo_sales_extensions()
