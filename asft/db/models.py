from typing import Any

from sqlalchemy import JSON, Boolean, Column, Float, ForeignKey, Integer, String, Table
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


# Many-to-Many association table for Users and Roles
user_roles = Table(
    "user_roles",
    Base.metadata,
    Column("user_id", String, ForeignKey("users.id"), primary_key=True),
    Column("role_id", String, ForeignKey("roles.id"), primary_key=True),
)


class Role(Base):
    __tablename__ = "roles"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    description: Mapped[str] = mapped_column(String, nullable=True)


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    username: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String, nullable=False)
    email: Mapped[str] = mapped_column(String, unique=True, nullable=True)
    is_active: Mapped[bool] = mapped_column(default=True)

    roles: Mapped[list[Role]] = relationship(secondary=user_roles, lazy="joined")


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    job_type: Mapped[str] = mapped_column(String, nullable=False, index=True)
    status: Mapped[str] = mapped_column(String, nullable=False, default="queued", index=True)
    created_at: Mapped[float] = mapped_column(Float, nullable=False)
    updated_at: Mapped[float] = mapped_column(Float, nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    result: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=True)
    error: Mapped[str] = mapped_column(String, nullable=True)


class BenchmarkResult(Base):
    __tablename__ = "benchmark_results"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    claim_type: Mapped[str] = mapped_column(String, index=True)  # e.g., 'accuracy', 'resources'
    model_name: Mapped[str] = mapped_column(String)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON)
    timestamp: Mapped[float] = mapped_column(Float)


class RoutingHistory(Base):
    """
    Tracks OptimizerDecisionEngine routing history and outcomes for the Multi-Armed Bandit.
    """

    __tablename__ = "routing_history"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    task_hash: Mapped[str] = mapped_column(String, index=True)
    strategy_selected: Mapped[str] = mapped_column(
        String, index=True
    )  # e.g., 'lora', 'rag', 'skill'
    exploration_mode: Mapped[bool] = mapped_column(
        Boolean
    )  # True if random explore, False if exploit
    expected_cost: Mapped[float] = mapped_column(Float, nullable=True)
    expected_accuracy: Mapped[float] = mapped_column(Float, nullable=True)

    # Outcomes (populated after execution and benchmarking)
    actual_cost: Mapped[float] = mapped_column(Float, nullable=True)
    actual_accuracy: Mapped[float] = mapped_column(Float, nullable=True)
    actual_runtime: Mapped[float] = mapped_column(Float, nullable=True)
    success: Mapped[bool] = mapped_column(Boolean, nullable=True)
    reward_score: Mapped[float] = mapped_column(Float, nullable=True)

    timestamp: Mapped[float] = mapped_column(Float)


class StrategyOutcome(Base):
    """
    Aggregated historical outcomes for Bandit decision engine optimization.
    Prevents unbounded growth of RoutingHistory.
    """

    __tablename__ = "strategy_outcomes"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    task_hash: Mapped[str] = mapped_column(String, index=True)
    strategy: Mapped[str] = mapped_column(String, index=True)

    avg_accuracy: Mapped[float] = mapped_column(Float, nullable=True)
    avg_cost: Mapped[float] = mapped_column(Float, nullable=True)
    avg_runtime: Mapped[float] = mapped_column(Float, nullable=True)
    success_rate: Mapped[float] = mapped_column(Float, nullable=True)
    sample_count: Mapped[int] = mapped_column(Integer, default=0)

    last_updated: Mapped[float] = mapped_column(Float)
