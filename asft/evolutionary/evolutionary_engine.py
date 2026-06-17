"""
Evolutionary Learning Engine — Gradient-free optimization through
population-based search, mutation, and selection.
Generates diverse candidates, benchmarks them, keeps winners.
"""
from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class Candidate:
    id: str
    content: str           # prompt, workflow, or strategy text
    generation: int = 0
    fitness: float = 0.0
    parent_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    evaluated: bool = False

    def __lt__(self, other: "Candidate") -> bool:
        return self.fitness < other.fitness


@dataclass
class EvolutionResult:
    best_candidate: Candidate
    final_population: List[Candidate]
    generations_run: int
    converged: bool
    fitness_history: List[float] = field(default_factory=list)
    duration_seconds: float = 0.0


class Mutator:
    """Applies mutation operators to candidate content."""

    def __init__(self, mutation_rate: float = 0.3):
        self._rate = mutation_rate

    def mutate(self, candidate: Candidate, generation: int) -> Candidate:
        if random.random() > self._rate:
            return candidate  # No mutation

        content = candidate.content
        op = random.choice(["append", "prepend", "replace_phrase", "rephrase"])

        if op == "append":
            additions = [
                "\nBe concise and precise.",
                "\nProvide concrete examples.",
                "\nExplain your reasoning step by step.",
                "\nDouble-check your answer for accuracy.",
            ]
            content = content + random.choice(additions)

        elif op == "prepend":
            prefixes = [
                "Think carefully. ",
                "As an expert, ",
                "Step by step: ",
                "Carefully and accurately: ",
            ]
            content = random.choice(prefixes) + content

        elif op == "replace_phrase":
            replacements = {
                "Provide": "Give",
                "Generate": "Create",
                "Explain": "Describe",
                "List": "Enumerate",
            }
            for old, new in replacements.items():
                if old in content and random.random() < 0.5:
                    content = content.replace(old, new, 1)
                    break

        elif op == "rephrase":
            # Simple structural rephrase: move sentences
            sentences = [s.strip() for s in content.split(". ") if s.strip()]
            if len(sentences) > 2:
                random.shuffle(sentences)
                content = ". ".join(sentences)

        import uuid
        return Candidate(
            id=str(uuid.uuid4())[:8],
            content=content,
            generation=generation,
            parent_ids=[candidate.id],
        )

    def crossover(self, a: Candidate, b: Candidate, generation: int) -> Candidate:
        """Combine two candidates (sentence-level mixing)."""
        a_parts = a.content.split(". ")
        b_parts = b.content.split(". ")

        combined = []
        for i in range(max(len(a_parts), len(b_parts))):
            if i < len(a_parts) and i < len(b_parts):
                combined.append(a_parts[i] if random.random() < 0.5 else b_parts[i])
            elif i < len(a_parts):
                combined.append(a_parts[i])
            else:
                combined.append(b_parts[i])

        import uuid
        return Candidate(
            id=str(uuid.uuid4())[:8],
            content=". ".join(combined),
            generation=generation,
            parent_ids=[a.id, b.id],
        )


class FitnessEvaluator:
    """Evaluates candidate fitness against a task benchmark."""

    def __init__(self, eval_fn: Callable[[str], float], n_samples: int = 10):
        self._eval_fn = eval_fn
        self._n_samples = n_samples
        self._cache: Dict[str, float] = {}

    def evaluate(self, candidate: Candidate) -> float:
        if candidate.content in self._cache:
            return self._cache[candidate.content]
        try:
            score = self._eval_fn(candidate.content)
        except Exception as e:
            logger.warning("Fitness eval failed for %s: %s", candidate.id, e)
            score = 0.0
        self._cache[candidate.content] = score
        candidate.fitness = score
        candidate.evaluated = True
        return score


class EvolutionaryEngine:
    """
    Main evolutionary loop:
      Generate → Evaluate → Select → Mutate → Repeat
    """

    def __init__(
        self,
        population_size: int = 20,
        elite_fraction: float = 0.2,
        mutation_rate: float = 0.3,
        max_generations: int = 50,
        convergence_threshold: float = 0.001,
    ):
        self._pop_size = population_size
        self._elite_n = max(1, int(population_size * elite_fraction))
        self._mutator = Mutator(mutation_rate=mutation_rate)
        self._max_generations = max_generations
        self._convergence_threshold = convergence_threshold

    def evolve(
        self,
        seed_candidates: List[str],
        fitness_fn: Callable[[str], float],
        eval_samples: int = 50,
    ) -> EvolutionResult:
        """
        Run the evolutionary optimization loop.

        Args:
            seed_candidates: initial set of candidate strings
            fitness_fn: callable(candidate_str) → float score 0–1
            eval_samples: number of eval samples per candidate
        """
        start_time = time.time()
        evaluator = FitnessEvaluator(fitness_fn, n_samples=eval_samples)

        # Initialize population
        import uuid
        population = [
            Candidate(id=str(uuid.uuid4())[:8], content=c, generation=0)
            for c in seed_candidates
        ]

        # Pad to pop_size with mutations of seeds
        while len(population) < self._pop_size:
            parent = random.choice(population)
            population.append(self._mutator.mutate(parent, generation=0))

        fitness_history: List[float] = []
        prev_best = -1.0
        converged = False

        for gen in range(self._max_generations):
            # Evaluate all
            for c in population:
                if not c.evaluated:
                    evaluator.evaluate(c)

            # Sort by fitness
            population.sort(key=lambda c: c.fitness, reverse=True)
            best_fitness = population[0].fitness
            fitness_history.append(best_fitness)

            logger.info(
                "Generation %d | best=%.4f | avg=%.4f",
                gen, best_fitness,
                sum(c.fitness for c in population) / len(population),
            )

            # Check convergence
            if gen > 5 and abs(best_fitness - prev_best) < self._convergence_threshold:
                logger.info("Converged at generation %d", gen)
                converged = True
                break
            prev_best = best_fitness

            # Selection: keep elites
            elites = population[:self._elite_n]

            # Generate offspring
            offspring = []
            while len(offspring) < self._pop_size - self._elite_n:
                if random.random() < 0.6 and len(elites) >= 2:
                    # Crossover
                    a, b = random.sample(elites, 2)
                    child = self._mutator.crossover(a, b, generation=gen + 1)
                else:
                    # Mutation
                    parent = random.choice(elites)
                    child = self._mutator.mutate(parent, generation=gen + 1)
                offspring.append(child)

            population = elites + offspring

        # Final evaluation
        for c in population:
            if not c.evaluated:
                evaluator.evaluate(c)

        population.sort(key=lambda c: c.fitness, reverse=True)

        return EvolutionResult(
            best_candidate=population[0],
            final_population=population,
            generations_run=gen + 1,
            converged=converged,
            fitness_history=fitness_history,
            duration_seconds=round(time.time() - start_time, 2),
        )
