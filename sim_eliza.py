from typing import Dict, List, Tuple, Callable
import re
from dataclasses import dataclass
from datetime import datetime
import random

@dataclass
class Memory:
    key: str
    value: str
    timestamp: datetime

class BaseEliza:
    def __init__(self, name: str, specialty: str):
        self.name = name
        self.specialty = specialty
        self.memory: List[Memory] = []
        self.patterns: Dict[str, Callable] = {}
        
    def add_pattern(self, pattern: str, response_fn: Callable):
        self.patterns[pattern] = response_fn
        
    def remember(self, key: str, value: str):
        self.memory.append(Memory(key, value, datetime.now()))
        
    def analyze(self, input_text: str) -> str:
        for pattern, response_fn in self.patterns.items():
            match = re.match(pattern, input_text, re.I)
            if match:
                return response_fn(match.groups(), self)
        return self.default_response(input_text)
    
    def default_response(self, input_text: str) -> str:
        return f"Tell me more about that."

class TherapistEliza(BaseEliza):
    def __init__(self):
        super().__init__("Dr. ELIZA", "Therapy")
        self.setup_patterns()
        
    def setup_patterns(self):
        self.add_pattern(
            r"I am (.*)",
            lambda groups, self: f"How long have you been {groups[0]}?"
        )
        self.add_pattern(
            r"I feel (.*)",
            lambda groups, self: random.choice([
                f"Why do you think you feel {groups[0]}?",
                f"What triggered these feelings of {groups[0]}?",
                f"How do these feelings of {groups[0]} affect you?"
            ])
        )

class TeacherEliza(BaseEliza):
    def __init__(self):
        super().__init__("Professor ELIZA", "Education")
        self.setup_patterns()
        
    def setup_patterns(self):
        self.add_pattern(
            r"how (.*) work",
            lambda groups, self: f"Let's break down {groups[0]} step by step. What do you already know?"
        )
        self.add_pattern(
            r"explain (.*)",
            lambda groups, self: f"I'll help you understand {groups[0]}. Where should we start?"
        )

class PhilosopherEliza(BaseEliza):
    def __init__(self):
        super().__init__("Socrates ELIZA", "Philosophy")
        self.setup_patterns()
        
    def setup_patterns(self):
        self.add_pattern(
            r"what is (.*)",
            lambda groups, self: f"What do you think makes something {groups[0]}?"
        )
        self.add_pattern(
            r"why (.*)",
            lambda groups, self: f"What assumptions underlie your question about {groups[0]}?"
        )

class ElizaSimulation:
    def __init__(self):
        self.elizas = {
            "therapy": TherapistEliza(),
            "education": TeacherEliza(),
            "philosophy": PhilosopherEliza()
        }
        self.keywords = {
            "therapy": ["feel", "emotion", "anxiety", "stress"],
            "education": ["learn", "understand", "explain"],
            "philosophy": ["why", "meaning", "truth", "reality"]
        }
        
    def route_query(self, input_text: str) -> Tuple[str, str]:
        scores = self._calculate_scores(input_text)
        best_domain = max(scores.items(), key=lambda x: x[1])[0]
        response = self.elizas[best_domain].analyze(input_text)
        return f"{self.elizas[best_domain].name}: {response}"
    
    def _calculate_scores(self, input_text: str) -> Dict[str, float]:
        scores = {}
        input_text = input_text.lower()
        
        for domain, keywords in self.keywords.items():
            score = sum(2 for word in keywords if word in input_text)
            for pattern in self.elizas[domain].patterns.keys():
                if re.search(pattern, input_text, re.I):
                    score += 3
            scores[domain] = score
        return scores

# Simulation
def run_simulation():
    sim = ElizaSimulation()
    print("ELIZA Family Simulation (type 'exit' to end)")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'exit':
            break
        print(sim.route_query(user_input))

if __name__ == "__main__":
    run_simulation()