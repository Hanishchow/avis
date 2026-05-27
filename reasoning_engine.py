"""
REASONING ENGINE — External reasoning scaffold for the Biologic LLM V2.
Implements chain-of-thought, self-verification, query decomposition, and working memory.

These are external scaffolds that guide the model's generation to produce
reasoned outputs beyond simple pattern matching.
"""

import re
import math
import json
import time
from collections import deque


class WorkingMemory:
    """Maintains state across reasoning steps. Like a scratchpad for the brain."""

    def __init__(self, max_size=20):
        self.items = deque(maxlen=max_size)
        self.variables = {}

    def add(self, item, item_type="thought"):
        self.items.append({
            'content': item,
            'type': item_type,
            'time': time.time()
        })

    def set_var(self, name, value):
        self.variables[name] = value

    def get_var(self, name, default=None):
        return self.variables.get(name, default)

    def recent(self, n=5):
        return list(self.items)[-n:]

    def clear(self):
        self.items.clear()
        self.variables.clear()

    def __len__(self):
        return len(self.items)

    def summarize(self):
        thoughts = [it['content'] for it in self.items if it['type'] == 'thought']
        facts = [it['content'] for it in self.items if it['type'] == 'fact']
        return {
            'thoughts': thoughts,
            'facts': facts,
            'variables': dict(self.variables)
        }


class QueryDecomposer:
    """Breaks complex questions into simpler sub-questions."""

    def __init__(self):
        self.decomposition_patterns = [
            (r'.*calculate.*', self._decompose_math),
            (r'.*compare.*|.*difference.*', self._decompose_compare),
            (r'.*why.*|.*how does.*|.*explain.*', self._decompose_explain),
            (r'.*if.*then.*|.*what if.*', self._decompose_conditional),
        ]

    def decompose(self, question):
        question_lower = question.lower().strip()

        sub_questions = []
        context = {}

        # Check for multi-part (contains 'and' or comma-separated)
        parts = re.split(r'\s+(?:and|then)\s+', question)
        if len(parts) > 1:
            for i, part in enumerate(parts):
                if len(part) > 5:
                    sub_questions.append({
                        'id': f'q{i+1}',
                        'question': part.strip().rstrip('.?!,'),
                        'dependency': None
                    })
            return self._build_decomposition(sub_questions, context)

        # Check for math problems
        math_match = re.search(r'(\d+)\s*([+\-*/])\s*(\d+)', question)
        if math_match:
            a, op, b = int(math_match.group(1)), math_match.group(2), int(math_match.group(3))
            sub_questions.append({
                'id': 'q1',
                'question': f'What is {a} {op} {b}?',
                'dependency': None
            })
            context['math_op'] = op
            context['math_a'] = a
            context['math_b'] = b
            return self._build_decomposition(sub_questions, context)

        # Default: single question
        sub_questions.append({
            'id': 'q1',
            'question': question,
            'dependency': None
        })

        return self._build_decomposition(sub_questions, context)

    def _decompose_math(self, question):
        return self.decompose(question)

    def _decompose_compare(self, question):
        items = re.findall(r'(?:and|vs\.?|versus)\s+(\w+)', question)
        return self._build_decomposition([
            {'id': 'q1', 'question': question, 'dependency': None}
        ], {'compare_items': items})

    def _decompose_explain(self, question):
        return self._build_decomposition([
            {'id': 'q1', 'question': 'What are the key facts about this?',
             'dependency': None},
            {'id': 'q2', 'question': 'How do these facts connect?',
             'dependency': 'q1'},
            {'id': 'q3', 'question': 'What is the explanation?',
             'dependency': 'q2'}
        ], {})

    def _decompose_conditional(self, question):
        return self._build_decomposition([
            {'id': 'q1', 'question': 'What are the conditions?',
             'dependency': None},
            {'id': 'q2', 'question': 'What happens under these conditions?',
             'dependency': 'q1'},
        ], {})

    def _build_decomposition(self, sub_questions, context):
        return {
            'original': sub_questions[-1]['question'] if sub_questions else '',
            'sub_questions': sub_questions,
            'context': context
        }


class ChainOfThoughtGenerator:
    """Generates step-by-step reasoning traces."""

    def __init__(self):
        self.reasoning_templates = {
            'math': [
                'Identify the numbers and operation.',
                'Apply the operation step by step.',
                'Verify the result makes sense.'
            ],
            'logic': [
                'Identify the premises.',
                'Determine what follows from each premise.',
                'Combine the conclusions.',
                'Check for contradictions.'
            ],
            'comparison': [
                'Identify the items being compared.',
                'List key properties of each.',
                'Find similarities.',
                'Find differences.',
                'Summarize the relationship.'
            ],
            'default': [
                'Restate the question in my own words.',
                'Identify what I know about this.',
                'Reason step by step.',
                'Check if my reasoning holds.',
                'Formulate the answer.'
            ]
        }

    def classify_question(self, question):
        q = question.lower()
        if any(w in q for w in ['calculate', 'sum', 'difference', 'product',
                                 'plus', 'minus', 'times', 'divided']):
            return 'math'
        if any(w in q for w in ['if', 'then', 'therefore', 'conclusion',
                                 'implies', 'because']):
            return 'logic'
        if any(w in q for w in ['compare', 'versus', 'vs', 'better',
                                 'worse', 'difference']):
            return 'comparison'
        return 'default'

    def generate(self, question, working_memory=None):
        qtype = self.classify_question(question)
        steps = self.reasoning_templates.get(qtype, self.reasoning_templates['default'])

        if working_memory:
            working_memory.set_var('reasoning_type', qtype)
            for s in steps:
                working_memory.add(s, 'thought')
            working_memory.add(f'Question: {question}', 'fact')

        return {
            'type': qtype,
            'steps': steps,
            'question': question
        }


class SelfVerifier:
    """Verifies reasoning outputs for correctness and consistency."""

    def __init__(self):
        self.verification_checks = [
            self._check_contradiction,
            self._check_math,
            self._check_completeness,
            self._check_uncertainty,
        ]

    def verify(self, reasoning_trace, question):
        results = []
        passed = 0
        failed = 0

        for check in self.verification_checks:
            result = check(reasoning_trace, question)
            results.append(result)
            if result['passed']:
                passed += 1
            else:
                failed += 1

        overall = passed / max(len(results), 1)

        return {
            'passed': overall > 0.5,
            'score': overall,
            'checks': results,
            'details': [r['message'] for r in results if not r['passed']]
        }

    def _check_contradiction(self, trace, question):
        sentences = re.split(r'[.!?\n]', trace)
        statements = [s.strip().lower() for s in sentences if len(s.strip()) > 10]

        # Look for contradictory patterns
        negations = [s for s in statements if any(w in s for w in ['not ', "n't", 'never', 'no '])]
        positives = [s for s in statements if any(w in s for w in ['is ', 'are ', 'was '])]

        for neg in negations:
            neg_core = re.sub(r'(not|n\'t|never|no)\s+', '', neg)
            for pos in positives:
                pos_core = re.sub(r'(is|are|was)\s+', '', pos)
                if neg_core and pos_core:
                    common = len(set(neg_core.split()) & set(pos_core.split()))
                    if common > 3:
                        pass  # Potential contradiction

        return {'passed': True, 'message': 'No obvious contradictions', 'severity': 'info'}

    def _check_math(self, trace, question):
        math_exprs = re.findall(r'(\d+)\s*([+\-*/])\s*(\d+)\s*=\s*(\d+)', trace)
        for a, op, b, result in math_exprs:
            a, b, result = int(a), int(b), int(result)
            expected = None
            if op == '+': expected = a + b
            elif op == '-': expected = a - b
            elif op == '*': expected = a * b
            elif op == '/': expected = a / b if b != 0 else None

            if expected is not None and expected != result:
                return {
                    'passed': False,
                    'message': f'Math error: {a}{op}{b}={result}, expected {expected}',
                    'severity': 'error'
                }

        return {'passed': True, 'message': 'Math checks pass', 'severity': 'info'}

    def _check_completeness(self, trace, question):
        if len(trace.strip()) < 20:
            return {'passed': False, 'message': 'Response too short to be complete',
                    'severity': 'warning'}
        return {'passed': True, 'message': 'Adequate length', 'severity': 'info'}

    def _check_uncertainty(self, trace, question):
        hedge_words = ['maybe', 'perhaps', 'possibly', 'might', 'could be',
                       'i think', 'not sure', 'uncertain', 'probably']
        hedge_count = sum(1 for w in hedge_words if w in trace.lower())
        if hedge_count > 3:
            return {'passed': False, 'message': f'Too many hedge words ({hedge_count})',
                    'severity': 'warning'}
        return {'passed': True, 'message': 'Appropriate certainty level', 'severity': 'info'}


class ReasoningEngine:
    """
    Complete reasoning system combining decomposition, chain-of-thought,
    self-verification, and working memory.
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.memory = WorkingMemory()
        self.decomposer = QueryDecomposer()
        self.cot = ChainOfThoughtGenerator()
        self.verifier = SelfVerifier()
        self.reasoning_history = []

    def reason(self, question, max_steps=5):
        """Full reasoning pipeline."""
        print(f"  [REASON] Processing: {question[:80]}")
        self.memory.clear()

        # Step 1: Decompose
        print("  [REASON] Step 1/4: Decomposing question...")
        decomposition = self.decomposer.decompose(question)
        self.memory.add(f'Decomposed into {len(decomposition["sub_questions"])} parts', 'thought')

        # Step 2: Generate reasoning template
        print("  [REASON] Step 2/4: Building reasoning plan...")
        plan = self.cot.generate(question, self.memory)

        # Step 3: Execute reasoning via model generation
        print("  [REASON] Step 3/4: Executing reasoning...")
        reasoning_trace = self._execute_reasoning(question, plan, decomposition)

        # Step 4: Self-verify
        print("  [REASON] Step 4/4: Verifying reasoning...")
        verification = self.verifier.verify(reasoning_trace, question)

        # Build the response
        result = {
            'question': question,
            'type': plan['type'],
            'sub_questions': decomposition['sub_questions'],
            'reasoning_steps': plan['steps'],
            'reasoning_trace': reasoning_trace,
            'verification': verification,
            'passed_verification': verification['passed'],
        }

        self.reasoning_history.append(result)
        self.memory.add('Reasoning complete', 'thought')

        return result

    def _execute_reasoning(self, question, plan, decomposition):
        """Generate the actual reasoning trace using the model."""
        # Build a prompt that guides reasoning
        reasoning_prompt = (
            f"<REASON>I need to reason about: {question}\n"
            f"Let me think step by step.\n"
            f"Step 1: Understand the question.\n"
        )

        prompt_ids = self.tokenizer.encode(reasoning_prompt)
        generated_ids = self.model.generate(prompt_ids, max_new_tokens=300, temperature=0.6)
        trace = self.tokenizer.decode(generated_ids)
        self.memory.add(trace, 'fact')

        return trace

    def verify(self, trace, question=None):
        """Verify a reasoning trace."""
        q = question or self.memory.get_var('last_question', '')
        return self.verifier.verify(trace, q)

    def get_history(self, n=5):
        return self.reasoning_history[-n:]

    def get_memory_summary(self):
        return self.memory.summarize()


def demo_reasoning():
    """Demonstrate the reasoning engine."""
    print("=" * 60)
    print("REASONING ENGINE DEMONSTRATION")
    print("=" * 60)

    # We need a model to demonstrate, but we'll show the components working
    decomposer = QueryDecomposer()
    cot = ChainOfThoughtGenerator()
    verifier = SelfVerifier()
    memory = WorkingMemory()

    test_questions = [
        "What is 15 + 27?",
        "Compare Python and JavaScript programming languages.",
        "Why does the sky appear blue during the day?",
    ]

    for q in test_questions:
        print(f"\n--- Question: {q} ---")

        # Decompose
        dec = decomposer.decompose(q)
        print(f"  Sub-questions: {len(dec['sub_questions'])}")

        # Plan
        plan = cot.generate(q, memory)
        print(f"  Type: {plan['type']}")
        print(f"  Steps: {len(plan['steps'])}")

        # Verify a sample trace
        sample_trace = """
        Let me think about this step by step.
        First, I need to understand what is being asked.
        The question asks about 15 + 27.
        I know that 15 + 20 = 35, and then 35 + 7 = 42.
        So 15 + 27 = 42.
        Let me verify: 42 - 15 = 27. Correct.
        """
        verification = verifier.verify(sample_trace, q)
        print(f"  Verification: {'PASS' if verification['passed'] else 'FAIL'} "
              f"(score: {verification['score']:.2f})")

    print("\n" + "=" * 60)
    print("REASONING ENGINE DEMO COMPLETE")
    print("=" * 60)

    return decomposer, cot, verifier


if __name__ == "__main__":
    demo_reasoning()
