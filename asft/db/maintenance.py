import time
import uuid
import logging
from sqlalchemy.orm import Session
from sqlalchemy import func

from asft.db.database import SessionLocal
from asft.db.models import RoutingHistory, StrategyOutcome

logger = logging.getLogger(__name__)

def prune_routing_history():
    """
    Maintains the RoutingHistory table to prevent unbounded growth.
    Retention Policy: Raw records: 30 days OR last 10,000 records (whichever is larger).
    Older data is aggregated into StrategyOutcome and then deleted.
    """
    with SessionLocal() as db:
        logger.info("Starting RoutingHistory pruning...")
        
        # Determine the cutoff for "30 days ago"
        cutoff_time = time.time() - (30 * 24 * 60 * 60)
        
        # Get total count
        total_records = db.query(RoutingHistory).count()
        if total_records <= 10000:
            logger.info(f"Total records ({total_records}) <= 10000. No pruning necessary.")
            return

        # Find the timestamp of the 10,000th newest record
        # Order by timestamp descending, offset 9999, get the timestamp
        record_10k = db.query(RoutingHistory.timestamp).order_by(RoutingHistory.timestamp.desc()).offset(9999).first()
        
        if not record_10k:
            return
            
        timestamp_10k = record_10k[0]
        
        # We want to keep at least 10k records, OR anything from the last 30 days
        # This means the threshold for deletion is the minimum of (cutoff_time, timestamp_10k)
        delete_threshold = min(cutoff_time, timestamp_10k)
        
        # Get all records older than the delete threshold
        old_records = db.query(RoutingHistory).filter(RoutingHistory.timestamp < delete_threshold).all()
        
        if not old_records:
            logger.info("No records match the pruning criteria.")
            return
            
        logger.info(f"Found {len(old_records)} old records to aggregate and prune.")
        
        # Aggregate by task_hash and strategy
        aggregates = {}
        for r in old_records:
            if not r.success:
                continue # We mostly care about success aggregates for utility
                
            key = (r.task_hash, r.strategy_selected)
            if key not in aggregates:
                aggregates[key] = {
                    "acc": [], "cost": [], "runtime": [], "successes": 0, "total": 0
                }
                
            aggregates[key]["total"] += 1
            if r.success:
                aggregates[key]["successes"] += 1
                
            if r.actual_accuracy is not None:
                aggregates[key]["acc"].append(r.actual_accuracy)
            if r.actual_cost is not None:
                aggregates[key]["cost"].append(r.actual_cost)
            if r.actual_runtime is not None:
                aggregates[key]["runtime"].append(r.actual_runtime)
                
        # Update or Insert into StrategyOutcome
        for (task_hash, strategy), stats in aggregates.items():
            outcome = db.query(StrategyOutcome).filter(
                StrategyOutcome.task_hash == task_hash,
                StrategyOutcome.strategy == strategy
            ).first()
            
            if not outcome:
                outcome = StrategyOutcome(
                    id=str(uuid.uuid4()),
                    task_hash=task_hash,
                    strategy=strategy,
                    avg_accuracy=0.0,
                    avg_cost=0.0,
                    avg_runtime=0.0,
                    success_rate=0.0,
                    sample_count=0,
                    last_updated=time.time()
                )
                db.add(outcome)
                
            # Rolling average update
            old_count = outcome.sample_count
            new_count = stats["total"]
            total_count = old_count + new_count
            
            def rolling_avg(old_val, old_c, new_vals):
                if not new_vals:
                    return old_val
                new_avg = sum(new_vals) / len(new_vals)
                new_c = len(new_vals)
                if old_val is None:
                    return new_avg
                return ((old_val * old_c) + (new_avg * new_c)) / (old_c + new_c)
                
            outcome.avg_accuracy = rolling_avg(outcome.avg_accuracy, old_count, stats["acc"])
            outcome.avg_cost = rolling_avg(outcome.avg_cost, old_count, stats["cost"])
            outcome.avg_runtime = rolling_avg(outcome.avg_runtime, old_count, stats["runtime"])
            
            # Success rate rolling average
            old_successes = (outcome.success_rate or 0.0) * old_count
            new_successes = stats["successes"]
            outcome.success_rate = (old_successes + new_successes) / max(1, total_count)
            
            outcome.sample_count = total_count
            outcome.last_updated = time.time()
            
        # Bulk delete the old records
        db.query(RoutingHistory).filter(RoutingHistory.timestamp < delete_threshold).delete()
        db.commit()
        
        logger.info(f"Successfully pruned {len(old_records)} records and updated StrategyOutcomes.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    prune_routing_history()
