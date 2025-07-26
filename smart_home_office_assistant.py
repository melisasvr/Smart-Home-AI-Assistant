import datetime
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import pickle
import os

# Advanced ML imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

@dataclass
class UserAction:
    """Enhanced user action with more detailed context"""
    timestamp: datetime.datetime
    action_type: str
    context: Dict
    user_response: Optional[str] = None
    success: bool = True
    execution_time: float = 0.0
    user_satisfaction: Optional[int] = None  # 1-5 rating

@dataclass
class Suggestion:
    """Enhanced suggestion with ML confidence and reasoning"""
    timestamp: datetime.datetime
    suggestion_type: str
    message: str
    confidence: float
    context: Dict
    reasoning: str = ""
    user_response: Optional[str] = None
    response_time: Optional[float] = None

class AdvancedPatternLearner:
    """Advanced ML-based pattern learning system"""
    
    def __init__(self):
        self.action_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.time_predictor = GradientBoostingClassifier(n_estimators=50, random_state=42)
        self.user_clustering = KMeans(n_clusters=5, random_state=42)
        
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Feature storage
        self.feature_data = []
        self.target_data = []
        self.context_features = []
        
    def extract_features(self, action: UserAction) -> np.array:
        """Extract comprehensive features from user action"""
        features = []
        
        # Time features
        dt = action.timestamp
        features.extend([
            dt.hour,
            dt.minute,
            dt.weekday(),
            dt.day,
            dt.month,
            1 if dt.weekday() >= 5 else 0,  # is_weekend
        ])
        
        # Context features
        context = action.context
        features.extend([
            context.get('temperature', 72),
            1 if context.get('is_home', True) else 0,
            1 if context.get('is_working', False) else 0,
            context.get('stress_level', 3),  # 1-5 scale
            context.get('energy_level', 3),  # 1-5 scale
        ])
        
        # Device state features
        device_states = context.get('device_states', {})
        features.extend([
            1 if device_states.get('coffee_maker', False) else 0,
            1 if device_states.get('lights', {}).get('office', False) else 0,
            1 if device_states.get('music', False) else 0,
            device_states.get('temperature', 72),
        ])
        
        # Seasonal and weather features
        features.extend([
            1 if context.get('season') == 'winter' else 0,
            1 if context.get('season') == 'spring' else 0,
            1 if context.get('season') == 'summer' else 0,
            1 if context.get('season') == 'fall' else 0,
            context.get('weather_score', 5),  # 1-10 weather pleasantness
        ])
        
        return np.array(features)
    
    def learn_from_action(self, action: UserAction):
        """Learn from user action using advanced ML"""
        features = self.extract_features(action)
        
        self.feature_data.append(features)
        self.target_data.append(action.action_type)
        self.context_features.append(action.context)
        
        # Retrain model periodically
        if len(self.feature_data) > 10 and len(self.feature_data) % 5 == 0:
            self._train_models()
    
    def _train_models(self):
        """Train ML models on collected data"""
        if len(self.feature_data) < 5:
            return
        
        try:
            X = np.array(self.feature_data)
            y = self.target_data
            
            # Encode labels
            if 'action_type' not in self.label_encoders:
                self.label_encoders['action_type'] = LabelEncoder()
            
            y_encoded = self.label_encoders['action_type'].fit_transform(y)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train classifiers
            if len(set(y)) > 1:  # Need at least 2 classes
                self.action_classifier.fit(X_scaled, y_encoded)
                self.time_predictor.fit(X_scaled, y_encoded)
                self.is_trained = True
                
                # User behavior clustering
                if len(X_scaled) >= 5:  # Need minimum samples for clustering
                    self.user_clustering.fit(X_scaled)
                
        except Exception as e:
            print(f"Training error: {e}")
    
    def predict_next_actions(self, current_time: datetime.datetime, context: Dict) -> List[Tuple[str, float, str]]:
        """Predict actions using trained ML models"""
        if not self.is_trained:
            return self._fallback_predictions(current_time, context)
        
        # Create dummy action for feature extraction
        dummy_action = UserAction(current_time, "dummy", context)
        features = self.extract_features(dummy_action).reshape(1, -1)
        
        try:
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Get predictions
            action_probs = self.action_classifier.predict_proba(features_scaled)[0]
            time_probs = self.time_predictor.predict_proba(features_scaled)[0]
            
            # Combine predictions
            combined_probs = (action_probs + time_probs) / 2
            
            # Get class names
            class_names = self.label_encoders['action_type'].classes_
            
            # Create predictions with reasoning
            predictions = []
            for i, prob in enumerate(combined_probs):
                if prob > 0.3:  # Threshold
                    action_type = class_names[i]
                    reasoning = self._generate_reasoning(action_type, context, prob)
                    predictions.append((action_type, prob, reasoning))
            
            return sorted(predictions, key=lambda x: x[1], reverse=True)[:3]
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return self._fallback_predictions(current_time, context)
    
    def _generate_reasoning(self, action_type: str, context: Dict, confidence: float) -> str:
        """Generate human-readable reasoning for predictions"""
        reasons = []
        
        if confidence > 0.8:
            reasons.append("very strong pattern match")
        elif confidence > 0.6:
            reasons.append("strong pattern match")
        else:
            reasons.append("moderate pattern match")
        
        # Context-based reasoning
        hour = datetime.datetime.now().hour
        if action_type == 'coffee' and 7 <= hour <= 9:
            reasons.append("typical morning coffee time")
        elif action_type == 'meeting_prep' and context.get('is_working'):
            reasons.append("work hours detected")
        elif action_type == 'lights' and (hour < 7 or hour > 19):
            reasons.append("low light conditions")
        
        return ", ".join(reasons)
    
    def _fallback_predictions(self, current_time: datetime.datetime, context: Dict) -> List[Tuple[str, float, str]]:
        """Fallback predictions when ML models aren't ready"""
        hour = current_time.hour
        predictions = []
        
        if 7 <= hour <= 9:
            predictions.append(('coffee', 0.7, 'morning routine'))
        if context.get('is_working') and hour >= 9:
            predictions.append(('meeting_prep', 0.6, 'work hours'))
        if hour < 7 or hour > 19:
            predictions.append(('lights', 0.5, 'low light time'))
            
        return predictions
    
    def get_user_behavior_profile(self) -> Dict:
        """Analyze user behavior patterns using clustering"""
        if not self.is_trained or len(self.feature_data) < 5:
            return {"profile": "insufficient_data"}
        
        try:
            X = np.array(self.feature_data)
            X_scaled = self.scaler.transform(X)
            
            # Get cluster for latest behavior
            latest_cluster = self.user_clustering.predict(X_scaled[-1:])[0]
            
            # Analyze patterns
            action_frequency = defaultdict(int)
            for action in self.target_data:
                action_frequency[action] += 1
            
            # Calculate peak activity hours
            hour_activity = defaultdict(int)
            for features in self.feature_data:
                hour_activity[int(features[0])] += 1
            
            peak_hours = sorted(hour_activity.items(), key=lambda x: x[1], reverse=True)[:3]
            
            return {
                "profile": f"cluster_{latest_cluster}",
                "most_common_actions": dict(action_frequency),
                "peak_activity_hours": [h[0] for h in peak_hours],
                "total_actions": len(self.target_data),
                "behavior_diversity": len(set(self.target_data))
            }
            
        except Exception as e:
            return {"profile": "analysis_error", "error": str(e)}

class SmartHomeAI:
    """Enhanced Smart Home Assistant with Advanced AI"""
    
    def __init__(self, user_name: str = "User"):
        self.user_name = user_name
        self.pattern_learner = AdvancedPatternLearner()
        self.action_history = deque(maxlen=1000)
        self.suggestion_history = deque(maxlen=200)
        
        # Enhanced device states
        self.device_states = {
            'coffee_maker': {'status': False, 'last_used': None, 'strength': 'medium'},
            'presentation_mode': {'status': False, 'display': 'main', 'volume': 50},
            'lights': {
                'living_room': {'status': False, 'brightness': 100, 'color': 'warm'},
                'office': {'status': False, 'brightness': 80, 'color': 'cool'},
                'bedroom': {'status': False, 'brightness': 30, 'color': 'warm'}
            },
            'climate': {'temperature': 72, 'humidity': 45, 'mode': 'auto'},
            'music': {'status': False, 'volume': 30, 'playlist': 'focus'},
            'security': {'armed': False, 'cameras': True, 'motion_detected': False}
        }
        
        # Enhanced user preferences with learning weights
        self.user_preferences = {
            'coffee_preferences': {'times': [7, 8, 14], 'strength': 'medium', 'temp': 'hot'},
            'work_schedule': {'start': 9, 'end': 17, 'break_times': [12, 15]},
            'comfort_settings': {'temp_range': [70, 74], 'preferred_lighting': 'warm'},
            'proactive_level': 0.7,  # 0-1 scale for proactiveness
            'learning_rate': 0.8,    # How quickly to adapt
            'privacy_mode': False
        }
        
        # AI-specific features
        self.context_awareness = True
        self.personalization_level = 0.5
        self.predictive_accuracy = 0.0
        self.satisfaction_scores = deque(maxlen=50)
    
    def log_user_action(self, action_type: str, context: Dict = None, satisfaction: int = None):
        """Enhanced action logging with satisfaction tracking"""
        if context is None:
            context = self._get_enhanced_context()
        
        action = UserAction(
            timestamp=datetime.datetime.now(),
            action_type=action_type,
            context=context,
            user_satisfaction=satisfaction
        )
        
        self.action_history.append(action)
        self.pattern_learner.learn_from_action(action)
        
        if satisfaction:
            self.satisfaction_scores.append(satisfaction)
            self._update_personalization()
        
        print(f"🎯 Enhanced logging: {action_type} (satisfaction: {satisfaction or 'N/A'})")
    
    def _get_enhanced_context(self) -> Dict:
        """Get comprehensive context with AI features"""
        now = datetime.datetime.now()
        
        # Simulated environmental context
        context = {
            'timestamp': now.isoformat(),
            'hour': now.hour,
            'day_of_week': now.weekday(),
            'is_weekend': now.weekday() >= 5,
            'season': self._get_season(now),
            'is_home': True,  # Could be from location services
            'is_working': 9 <= now.hour <= 17 and now.weekday() < 5,
            'stress_level': np.random.randint(1, 6),  # Simulated biometric
            'energy_level': max(1, 6 - abs(now.hour - 14)),  # Peak at 2 PM
            'weather_score': np.random.randint(4, 9),  # Simulated weather API
            'device_states': self.device_states,
            'recent_activities': [a.action_type for a in list(self.action_history)[-5:]],
            'calendar_events': self._get_simulated_calendar(),
            'ambient_noise_level': np.random.randint(20, 60),  # dB
            'room_occupancy': self._detect_room_occupancy()
        }
        
        return context
    
    def _get_simulated_calendar(self) -> List[Dict]:
        """Simulate calendar events"""
        now = datetime.datetime.now()
        events = []
        
        # Simulate some meetings
        if 9 <= now.hour <= 17 and now.weekday() < 5:
            if np.random.random() > 0.7:
                events.append({
                    'title': 'Team Meeting',
                    'start_time': (now + datetime.timedelta(minutes=30)).isoformat(),
                    'duration': 60,
                    'type': 'meeting'
                })
        
        return events
    
    def _detect_room_occupancy(self) -> Dict:
        """Simulate room occupancy detection"""
        return {
            'living_room': np.random.random() > 0.6,
            'office': np.random.random() > 0.3,
            'bedroom': np.random.random() > 0.8,
            'kitchen': np.random.random() > 0.7
        }
    
    def make_ai_suggestions(self) -> List[Suggestion]:
        """Generate AI-powered proactive suggestions"""
        current_time = datetime.datetime.now()
        context = self._get_enhanced_context()
        
        # Get ML predictions
        predictions = self.pattern_learner.predict_next_actions(current_time, context)
        suggestions = []
        
        for action_type, confidence, reasoning in predictions:
            if confidence > (1 - self.user_preferences['proactive_level']):
                suggestion = self._create_ai_suggestion(
                    action_type, confidence, reasoning, current_time, context
                )
                if suggestion:
                    suggestions.append(suggestion)
                    self.suggestion_history.append(suggestion)
        
        # Add context-aware suggestions
        context_suggestions = self._get_context_aware_suggestions(current_time, context)
        suggestions.extend(context_suggestions)
        
        return suggestions
    
    def _create_ai_suggestion(self, action_type: str, confidence: float, reasoning: str,
                             current_time: datetime.datetime, context: Dict) -> Optional[Suggestion]:
        """Create AI-powered suggestion with reasoning"""
        
        # Personalized messages based on user behavior
        personal_touch = f"{self.user_name}, " if self.personalization_level > 0.5 else ""
        
        messages = {
            'coffee': f"{personal_touch}based on your routine, would you like me to start brewing coffee?",
            'meeting_prep': f"{personal_touch}I noticed you have a meeting soon. Shall I prepare your workspace?",
            'lights': f"{personal_touch}it looks like you could use better lighting. Should I adjust the lights?",
            'music': f"{personal_touch}some background music might help with focus. Start your playlist?",
            'climate': f"{personal_touch}the temperature seems off from your preferences. Adjust climate control?",
            'break': f"{personal_touch}you've been working for a while. Time for a break?"
        }
        
        if action_type in messages:
            return Suggestion(
                timestamp=current_time,
                suggestion_type=action_type,
                message=messages[action_type],
                confidence=confidence,
                context=context,
                reasoning=f"AI reasoning: {reasoning}"
            )
        return None
    
    def _get_context_aware_suggestions(self, current_time: datetime.datetime, context: Dict) -> List[Suggestion]:
        """Generate context-aware suggestions using environmental data"""
        suggestions = []
        
        # Calendar-based suggestions
        for event in context['calendar_events']:
            event_time = datetime.datetime.fromisoformat(event['start_time'])
            time_until = (event_time - current_time).total_seconds() / 60
            
            if 5 <= time_until <= 15:  # 5-15 minutes before
                suggestions.append(Suggestion(
                    timestamp=current_time,
                    suggestion_type='meeting_prep',
                    message=f"Your '{event['title']}' starts in {int(time_until)} minutes. Prepare now?",
                    confidence=0.9,
                    context=context,
                    reasoning="Calendar integration"
                ))
        
        # Stress-level based suggestions
        if context['stress_level'] > 4:
            suggestions.append(Suggestion(
                timestamp=current_time,
                suggestion_type='relaxation',
                message="You seem stressed. Would you like me to dim the lights and play calming music?",
                confidence=0.7,
                context=context,
                reasoning="High stress level detected"
            ))
        
        # Energy-level based suggestions
        if context['energy_level'] < 3 and 13 <= current_time.hour <= 15:
            suggestions.append(Suggestion(
                timestamp=current_time,
                suggestion_type='energy_boost',
                message="Afternoon energy dip detected. Coffee or a quick walk suggestion?",
                confidence=0.6,
                context=context,
                reasoning="Low energy during typical afternoon slump"
            ))
        
        return suggestions
    
    def _update_personalization(self):
        """Update personalization level based on satisfaction scores"""
        if len(self.satisfaction_scores) > 10:
            avg_satisfaction = np.mean(list(self.satisfaction_scores))
            self.personalization_level = min(1.0, max(0.1, avg_satisfaction / 5.0))
    
    def respond_to_suggestion(self, suggestion_id: int, response: str, satisfaction: int = None):
        """Enhanced response handling with learning"""
        if 0 <= suggestion_id < len(self.suggestion_history):
            suggestion = self.suggestion_history[suggestion_id]
            suggestion.user_response = response
            suggestion.response_time = (datetime.datetime.now() - suggestion.timestamp).total_seconds()
            
            # Learn from response
            if response == 'accepted':
                self.log_user_action(suggestion.suggestion_type, suggestion.context, satisfaction)
                self._execute_enhanced_action(suggestion.suggestion_type, suggestion.context)
                
                # Update predictive accuracy
                self._update_prediction_accuracy(True)
            else:
                self._update_prediction_accuracy(False)
            
            print(f"🧠 AI Learning: {response} response recorded with {satisfaction or 'no'} satisfaction rating")
    
    def _update_prediction_accuracy(self, correct: bool):
        """Update AI prediction accuracy metrics"""
        if hasattr(self, '_prediction_history'):
            self._prediction_history.append(correct)
        else:
            self._prediction_history = [correct]
        
        if len(self._prediction_history) > 20:
            self._prediction_history = self._prediction_history[-20:]
        
        self.predictive_accuracy = np.mean(self._prediction_history)
    
    def _execute_enhanced_action(self, action_type: str, context: Dict):
        """Execute actions with enhanced device control"""
        if action_type == 'coffee':
            strength = self.user_preferences['coffee_preferences']['strength']
            self.device_states['coffee_maker']['status'] = True
            self.device_states['coffee_maker']['last_used'] = datetime.datetime.now()
            print(f"☕ Coffee maker started - {strength} strength!")
            
        elif action_type == 'meeting_prep':
            self.device_states['presentation_mode']['status'] = True
            self.device_states['lights']['office']['status'] = True
            self.device_states['lights']['office']['brightness'] = 100
            print("🖥️ Meeting mode: Presentation ready, office lights optimized!")
            
        elif action_type == 'lights':
            occupied_rooms = [room for room, occupied in context.get('room_occupancy', {}).items() if occupied]
            for room in occupied_rooms:
                if room in self.device_states['lights']:
                    self.device_states['lights'][room]['status'] = True
            print(f"💡 Smart lighting: Activated in {occupied_rooms}!")
            
        elif action_type == 'relaxation':
            self.device_states['music']['status'] = True
            self.device_states['music']['playlist'] = 'relaxation'
            for room in self.device_states['lights']:
                self.device_states['lights'][room]['brightness'] = 30
            print("🧘 Relaxation mode: Dim lights, calming music activated!")
    
    def get_ai_analytics(self) -> Dict:
        """Get comprehensive AI analytics"""
        basic_stats = self.get_stats()
        
        # AI-specific metrics
        behavior_profile = self.pattern_learner.get_user_behavior_profile()
        
        ai_metrics = {
            'predictive_accuracy': round(self.predictive_accuracy * 100, 1),
            'personalization_level': round(self.personalization_level * 100, 1),
            'average_satisfaction': round(np.mean(list(self.satisfaction_scores)), 2) if self.satisfaction_scores else 0,
            'learning_effectiveness': round(len(self.pattern_learner.target_data) / max(1, len(self.action_history)) * 100, 1),
            'context_awareness_score': 85,  # Simulated based on features
            'behavior_profile': behavior_profile,
            'suggestion_response_time': round(np.mean([s.response_time for s in self.suggestion_history if s.response_time]), 2) if any(s.response_time for s in self.suggestion_history) else 0
        }
        
        return {**basic_stats, **ai_metrics}
    
    def get_stats(self) -> Dict:
        """Get basic statistics"""
        total_actions = len(self.action_history)
        total_suggestions = len(self.suggestion_history)
        
        accepted_suggestions = sum(1 for s in self.suggestion_history 
                                 if s.user_response == 'accepted')
        
        success_rate = (accepted_suggestions / total_suggestions * 100) if total_suggestions > 0 else 0
        
        action_types = defaultdict(int)
        for action in self.action_history:
            action_types[action.action_type] += 1
        
        return {
            'total_actions_logged': total_actions,
            'total_suggestions_made': total_suggestions,
            'suggestions_accepted': accepted_suggestions,
            'success_rate': round(success_rate, 1),
            'most_common_actions': dict(sorted(action_types.items(), 
                                             key=lambda x: x[1], reverse=True)[:5]),
            'device_states': self.device_states,
            'ml_model_trained': self.pattern_learner.is_trained
        }
    
    def _get_season(self, date: datetime.datetime) -> str:
        """Determine current season"""
        month = date.month
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        else:
            return 'fall'

# Enhanced simulation with AI features
def simulate_ai_assistant():
    """Simulate advanced AI assistant with machine learning"""
    print("🤖 Advanced AI Smart Home Assistant Starting...")
    print("="*60)
    
    assistant = SmartHomeAI("Alex")
    
    # Simulate extended learning period
    print("\n🧠 MACHINE LEARNING TRAINING PHASE")
    training_actions = [
        ('coffee', {'is_morning': True}, 5),
        ('coffee', {'is_morning': True}, 4),
        ('meeting_prep', {'is_working': True}, 5),
        ('lights', {'room': 'office'}, 4),
        ('music', {'activity': 'focus'}, 5),
        ('coffee', {'is_afternoon': True}, 3),
        ('lights', {'room': 'living_room'}, 4),
        ('climate', {'too_warm': True}, 4),
        ('break', {'long_work_session': True}, 5),
        ('relaxation', {'high_stress': True}, 5)
    ]
    
    for action, context, satisfaction in training_actions:
        assistant.log_user_action(action, context, satisfaction)
    
    print(f"✓ Trained on {len(training_actions)} actions")
    print(f"✓ ML Model Status: {'Trained' if assistant.pattern_learner.is_trained else 'Training'}")
    
    # AI-powered suggestions
    print("\n🎯 AI-POWERED PROACTIVE SUGGESTIONS")
    suggestions = assistant.make_ai_suggestions()
    
    for i, suggestion in enumerate(suggestions):
        print(f"{i+1}. {suggestion.message}")
        print(f"   Confidence: {suggestion.confidence:.2f} | {suggestion.reasoning}")
    
    # Simulate intelligent responses
    if suggestions:
        print("\n🤝 INTELLIGENT USER INTERACTION")
        assistant.respond_to_suggestion(0, 'accepted', 5)
        if len(suggestions) > 1:
            assistant.respond_to_suggestion(1, 'declined', 3)
    
    # Advanced analytics
    print("\n📊 ADVANCED AI ANALYTICS")
    analytics = assistant.get_ai_analytics()
    
    print(f"🎯 Predictive Accuracy: {analytics['predictive_accuracy']}%")
    print(f"🎨 Personalization Level: {analytics['personalization_level']}%")
    print(f"😊 Average Satisfaction: {analytics['average_satisfaction']}/5")
    print(f"🧠 Learning Effectiveness: {analytics['learning_effectiveness']}%")
    print(f"⚡ Context Awareness: {analytics['context_awareness_score']}%")
    
    behavior = analytics['behavior_profile']
    if behavior.get('profile') != 'insufficient_data':
        print(f"\n👤 USER BEHAVIOR PROFILE")
        print(f"Profile Type: {behavior['profile']}")
        print(f"Peak Hours: {behavior.get('peak_activity_hours', [])}")
        print(f"Behavior Diversity: {behavior.get('behavior_diversity', 0)} different action types")
    
    return assistant

if __name__ == "__main__":
    assistant = simulate_ai_assistant()
    
    print("\n" + "="*60)
    print("🚀 Advanced AI Assistant Ready!")
    print("Enhanced features:")
    print("• Machine Learning Pattern Recognition")
    print("• Context-Aware Suggestions")
    print("• Satisfaction-Based Learning")
    print("• Behavioral Profiling")
    print("• Predictive Analytics")